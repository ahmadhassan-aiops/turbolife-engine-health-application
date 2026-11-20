"""Rebuild model/ and data/oof.parquet from the C-MAPSS FD001 files in data/ (about 2 minutes).

Candidates are compared with 5-fold cross-validation grouped by engine (an engine's cycles never sit on both
sides), LightGBM is chosen on that score, refit on all 100 training engines and scored once on the 100 test engines.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from turbolife.features import CAP, FEATURES, SENSOR_CODES, make_features, read_cmapss, read_truth, remaining

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "model"
EDGES = [25, 50, 75, 100]          # bands of the prediction for the 80% range
POLICY = {"ground": 20, "plan": 50}


def candidates() -> dict:
    return {
        "Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.33, n_jobs=-1, random_state=42),
        "Histogram gradient boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=42),
    }


def nasa_score(d: np.ndarray) -> float:
    """PHM08 score: late predictions (d > 0) cost more than early ones."""
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    tc = np.minimum(truth, CAP)
    d = pred - tc
    return {"rmse": float(np.sqrt(np.mean(d ** 2))), "rmse_uncapped": float(np.sqrt(np.mean((pred - truth) ** 2))),
            "mae": float(np.mean(np.abs(d))), "r2": float(1 - np.sum(d ** 2) / np.sum((tc - tc.mean()) ** 2)),
            "nasa": nasa_score(d), "late_10": int((d > 10).sum()), "late_20": int((d > 20).sum()),
            "early_20": int((d < -20).sum())}


def main() -> None:
    train, test = read_cmapss(DATA / "train_FD001.txt"), read_cmapss(DATA / "test_FD001.txt")
    truth = read_truth(DATA / "RUL_FD001.txt")
    F = make_features(train)
    X, y, groups = F[FEATURES], remaining(train).clip(upper=CAP).to_numpy(), F["unit"].to_numpy()
    Xte = make_features(test).groupby("unit").tail(1)[FEATURES]
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))

    table, oof_best = [], None
    base_oof = np.zeros(len(y))
    for a, b in folds:
        base_oof[b] = y[a].mean()
    table.append({"model": "Baseline: average remaining life", "cv_rmse": float(np.sqrt(np.mean((base_oof - y) ** 2))),
                  **{f"test_{k}": v for k, v in scores(np.full(len(truth), y.mean()), truth).items()}})
    for name, est in candidates().items():
        oof = np.zeros(len(y))
        for a, b in folds:
            oof[b] = est.fit(X.iloc[a], y[a]).predict(X.iloc[b])
        oof = np.clip(oof, 0, CAP)
        est.fit(X, y)
        row = {"model": name, "cv_rmse": float(np.sqrt(np.mean((oof - y) ** 2))),
               **{f"test_{k}": v for k, v in scores(np.clip(est.predict(Xte), 0, CAP), truth).items()}}
        table.append(row)
        print(f"{name:28s} CV {row['cv_rmse']:.2f}  test {row['test_rmse']:.2f}")
        if name == "LightGBM":
            model, oof_best = est, oof
    best = min(table[1:], key=lambda r: r["cv_rmse"])["model"]
    assert best == "LightGBM", best

    # 80% range: 10th and 90th percentile of out-of-fold errors, per band of the prediction
    res = y - oof_best
    band = np.digitize(oof_best, EDGES)
    quant = [np.percentile(res[band == i], [10, 90]).round(2).tolist() for i in range(len(EDGES) + 1)]
    pred = np.clip(model.predict(Xte), 0, CAP)
    bt = np.digitize(pred, EDGES)
    lo = np.clip(pred + np.array([quant[i][0] for i in bt]), 0, None)
    hi = pred + np.array([quant[i][1] for i in bt])
    tc = np.minimum(truth, CAP)

    rul = remaining(train)
    gain = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
    by_sensor = {"cycle": float(gain["cycle"])} | {s: float(gain[[f for f in FEATURES if f == s or f.startswith(f"{s}_")]].sum())
                                                   for s in SENSOR_CODES}
    total = sum(by_sensor.values())
    meta = {
        "model": "LightGBM regressor (600 trees, learning rate 0.03, 31 leaves)",
        "cap": CAP, "features": FEATURES, "policy": POLICY,
        "train_engines": int(train["unit"].nunique()), "train_rows": int(len(train)),
        "test_engines": int(test["unit"].nunique()), "test_rows": int(len(test)),
        "comparison": table, "chosen": best,
        "test": scores(pred, truth) | {"coverage": float(((tc >= lo) & (tc <= hi)).mean())},
        "cv": {"rmse": float(np.sqrt(np.mean((oof_best - y) ** 2))),
               "coverage": float(((y >= np.clip(oof_best + np.array([quant[i][0] for i in band]), 0, None))
                                  & (y <= oof_best + np.array([quant[i][1] for i in band]))).mean())},
        "range": {"edges": EDGES, "quantiles": quant},
        "importance": {k: v / total for k, v in by_sensor.items()},
        "limits": {s: [float(train[s].min()), float(train[s].max())] for s in SENSOR_CODES},
        "bands": {s: {"healthy": np.percentile(train.loc[rul >= 150, s], [10, 90]).round(4).tolist(),
                      "worn": np.percentile(train.loc[rul < 20, s], [10, 90]).round(4).tolist()} for s in SENSOR_CODES},
        "life": train.groupby("unit")["cycle"].max().describe().round(1).to_dict(),
    }
    OUT.mkdir(exist_ok=True)
    joblib.dump(model, OUT / "turbolife_lgbm.joblib", compress=3)
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.DataFrame({"unit": groups.astype("int16"), "cycle": F["cycle"].astype("int16"), "rul": rul.astype("int16").to_numpy(),
                  "oof": oof_best.astype("float32")}).to_parquet(DATA / "oof.parquet", index=False)
    print("test", {k: round(v, 3) for k, v in meta["test"].items()})


if __name__ == "__main__":
    main()
