"""C-MAPSS FD001 data and the per-engine features the model reads.

Every feature at cycle t uses only that engine's cycles 1..t: the raw reading, its 10- and 30-cycle means and its
slope over the last 20 cycles, for the 14 sensors that change with wear; plus the cycles flown so far.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# The 14 sensors that move with wear in FD001 (the other seven are constant or nearly so).
SENSORS: dict[str, tuple[str, str, str]] = {
    "s2": ("T24", "LPC outlet temperature", "°R"),
    "s3": ("T30", "HPC outlet temperature", "°R"),
    "s4": ("T50", "LPT outlet temperature", "°R"),
    "s7": ("P30", "HPC outlet pressure", "psia"),
    "s8": ("Nf", "Fan speed", "rpm"),
    "s9": ("Nc", "Core speed", "rpm"),
    "s11": ("Ps30", "HPC outlet static pressure", "psia"),
    "s12": ("phi", "Fuel flow to Ps30 ratio", "pps/psi"),
    "s13": ("NRf", "Corrected fan speed", "rpm"),
    "s14": ("NRc", "Corrected core speed", "rpm"),
    "s15": ("BPR", "Bypass ratio", ""),
    "s17": ("htBleed", "Bleed enthalpy", ""),
    "s20": ("W31", "HPT coolant bleed", "lbm/s"),
    "s21": ("W32", "LPT coolant bleed", "lbm/s"),
}
SENSOR_CODES = list(SENSORS)
CAP = 125          # remaining life is capped: early in life the sensors cannot tell 200 cycles from 125
MEANS = (10, 30)
SLOPE_WINDOW = 20
FEATURES = ["cycle"] + [f for s in SENSOR_CODES for f in (s, f"{s}_m10", f"{s}_m30", f"{s}_sl")]


def read_cmapss(path: str | Path) -> pd.DataFrame:
    """A C-MAPSS text file (space separated, 26 columns, no header)."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS)


def read_truth(path: str | Path) -> np.ndarray:
    """RUL_FD001.txt: the true remaining cycles of each test engine after its last recorded cycle."""
    return pd.read_csv(path, sep=r"\s+", header=None)[0].to_numpy()


def _slope(x: pd.Series, k: int = SLOPE_WINDOW) -> pd.Series:
    """Least-squares slope per cycle over the last k cycles (from 3 cycles on), with rolling sums."""
    t = pd.Series(np.arange(len(x), dtype=float), index=x.index)
    mean = lambda s: s.rolling(k, min_periods=3).mean()  # noqa: E731
    cov = mean(t * x) - mean(t) * mean(x)
    var = mean(t * t) - mean(t) ** 2
    return (cov / var).fillna(0.0)


def make_features(history: pd.DataFrame) -> pd.DataFrame:
    """Features for every cycle of every engine. `history` needs `unit`, `cycle` and the 14 sensors, sorted by cycle
    within each unit. Returns `unit` plus FEATURES, same row order."""
    h = history.sort_values(["unit", "cycle"], kind="stable")
    g = h.groupby("unit", sort=False)
    out = {"unit": h["unit"], "cycle": h["cycle"].astype(float)}
    for s in SENSOR_CODES:
        out[s] = h[s].astype(float)
        for k in MEANS:
            out[f"{s}_m{k}"] = g[s].transform(lambda x, k=k: x.rolling(k, min_periods=1).mean())
        out[f"{s}_sl"] = g[s].transform(_slope)
    return pd.DataFrame(out)[["unit"] + FEATURES]


def remaining(train: pd.DataFrame) -> pd.Series:
    """True remaining cycles of each training row (every training engine runs to failure)."""
    return train.groupby("unit")["cycle"].transform("max") - train["cycle"]
