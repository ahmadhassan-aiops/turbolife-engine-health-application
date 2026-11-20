"""The remaining-life model: prediction, an 80% range, a maintenance lane and a per-sensor explanation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from turbolife.features import CAP, FEATURES, SENSOR_CODES
from turbolife.schema import Policy

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
LANES = ("Ground", "Plan", "Fly")


@dataclass(frozen=True)
class Estimate:
    predicted: float   # remaining cycles (capped at CAP)
    low: float         # 10th percentile: the cycles the plan relies on ("safe life")
    high: float        # 90th percentile
    lane: str


class LifeModel:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model = joblib.load(model_dir / "turbolife_lgbm.joblib")
        self.meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.edges = np.array(self.meta["range"]["edges"], dtype=float)       # inner band edges of the prediction
        self.q = np.array(self.meta["range"]["quantiles"], dtype=float)       # per band: (10th, 90th) residual

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(X[FEATURES]), 0, CAP)

    def bounds(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """80% range from out-of-fold errors of engines with a similar prediction."""
        pred = np.asarray(pred, dtype=float)
        b = np.digitize(pred, self.edges)
        return np.clip(pred + self.q[b, 0], 0, None), np.clip(pred + self.q[b, 1], 0, None)

    @staticmethod
    def lanes(low: np.ndarray, policy: Policy) -> np.ndarray:
        low = np.asarray(low, dtype=float)
        return np.where(low < policy.ground, "Ground", np.where(low < policy.plan, "Plan", "Fly"))

    def estimate(self, X: pd.DataFrame, policy: Policy) -> pd.DataFrame:
        p = self.predict(X)
        lo, hi = self.bounds(p)
        return pd.DataFrame({"predicted": p, "low": lo, "high": hi, "lane": self.lanes(lo, policy)}, index=X.index)

    def explain(self, row: pd.DataFrame) -> tuple[float, pd.Series]:
        """Contributions in cycles for one row, summed per sensor (raw, means and slope together) plus cycles flown.
        Base + contributions = the uncapped prediction."""
        c = self.model.predict(row[FEATURES].iloc[[0]], pred_contrib=True)[0]
        base, parts = float(c[-1]), pd.Series(c[:-1], index=FEATURES)
        grouped = {"cycle": parts["cycle"]}
        for s in SENSOR_CODES:
            grouped[s] = parts[[f for f in FEATURES if f == s or f.startswith(f"{s}_")]].sum()
        return base, pd.Series(grouped)
