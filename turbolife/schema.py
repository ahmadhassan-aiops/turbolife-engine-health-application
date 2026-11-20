"""Validated inputs: the maintenance policy and an uploaded engine log."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from turbolife.features import RAW_COLUMNS, SENSOR_CODES, SENSORS


class Policy(BaseModel):
    """Lanes from the model's safe life (the low end of the 80% range): Ground below `ground` cycles, Plan below
    `plan`, otherwise Fly."""
    ground: int = Field(20, ge=1, le=100)
    plan: int = Field(50, ge=2, le=125)

    @model_validator(mode="after")
    def _order(self) -> "Policy":
        if self.plan <= self.ground:
            raise ValueError("The plan line must be above the ground line")
        return self


def read_log(df: pd.DataFrame, limits: dict[str, list[float]], max_cycles: int = 500) -> pd.DataFrame:
    """One engine's log: `cycle` plus the 14 sensors (names s2..s21 or T24, T30, ...), or the raw 26-column C-MAPSS
    layout without a header. Returns unit, cycle and the 14 sensors, or raises ValueError with a readable message."""
    df = df.dropna(axis=1, how="all")
    if df.shape[1] == len(RAW_COLUMNS) and all(isinstance(c, int) or str(c).strip().lstrip("-").replace(".", "").isdigit()
                                                for c in df.columns):
        df = pd.DataFrame(df.to_numpy(), columns=RAW_COLUMNS)
    rename = {v[0].lower(): k for k, v in SENSORS.items()} | {k: k for k in SENSOR_CODES} | {"cycle": "cycle", "unit": "unit"}
    df = df.rename(columns=lambda c: rename.get(str(c).strip().lower(), str(c).strip()))
    missing = [SENSORS[s][0] for s in SENSOR_CODES if s not in df.columns] + (["cycle"] if "cycle" not in df.columns else [])
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if "unit" in df.columns and df["unit"].nunique() > 1:
        raise ValueError("The log holds more than one engine; upload one engine at a time")
    out = df[["cycle"] + SENSOR_CODES].apply(pd.to_numeric, errors="coerce")
    if out.isna().any().any():
        raise ValueError("Every value must be a number")
    if not 1 <= len(out) <= max_cycles:
        raise ValueError(f"The log must hold between 1 and {max_cycles} cycles")
    out = out.sort_values("cycle")
    if out["cycle"].iloc[0] != 1 or not (out["cycle"].diff().dropna() == 1).all():
        raise ValueError("Cycles must run 1, 2, 3, ... without gaps (the features use the whole history)")
    for s in SENSOR_CODES:
        lo, hi = limits[s]
        pad = (hi - lo) * 0.5
        if out[s].lt(lo - pad).any() or out[s].gt(hi + pad).any():
            raise ValueError(f"{SENSORS[s][0]} ({SENSORS[s][1]}) is far outside the range seen in training "
                             f"({lo:g} to {hi:g})")
    out["cycle"] = out["cycle"].astype(int)
    out[SENSOR_CODES] = out[SENSOR_CODES].astype(float)
    out.insert(0, "unit", 0)
    return out.reset_index(drop=True)
