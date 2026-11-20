"""Cached loaders: the model, the test fleet with features for every cycle, and the out-of-fold training results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from turbolife.features import make_features, read_cmapss, read_truth
from turbolife.model import LifeModel
from turbolife.schema import Policy

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@st.cache_resource(show_spinner="Loading the model...")
def model() -> LifeModel:
    return LifeModel()


@st.cache_data
def test_log() -> pd.DataFrame:
    """The 100 test engines' sensor logs (each stops some time before failure)."""
    return read_cmapss(DATA / "test_FD001.txt")


@st.cache_data
def truth() -> pd.Series:
    """True remaining cycles after each test engine's last logged cycle, indexed by engine."""
    return pd.Series(read_truth(DATA / "RUL_FD001.txt"), index=pd.RangeIndex(1, 101, name="unit"))


@st.cache_data(show_spinner="Reading 13,096 cycles of sensor data...")
def test_scored() -> pd.DataFrame:
    """Every cycle of every test engine: features, prediction, 80% range and the true remaining life at that cycle."""
    m = model()
    F = make_features(test_log())
    F["predicted"] = m.predict(F)
    F["low"], F["high"] = m.bounds(F["predicted"].to_numpy())
    last = F.groupby("unit")["cycle"].transform("max")
    F["actual"] = F["unit"].map(truth()) + (last - F["cycle"])
    return F.reset_index(drop=True)


def fleet(policy_json: str) -> pd.DataFrame:
    """One row per test engine at its last logged cycle, with its lane under the policy."""
    policy = Policy.model_validate_json(policy_json)
    s = test_scored()
    f = s.groupby("unit").tail(1).set_index("unit")
    f["lane"] = LifeModel.lanes(f["low"].to_numpy(), policy)
    f["error"] = f["predicted"] - np.minimum(f["actual"], model().meta["cap"])
    return f


@st.cache_data
def oof() -> pd.DataFrame:
    """Training engines, every cycle: true remaining life and the out-of-fold prediction (engine held out)."""
    o = pd.read_parquet(DATA / "oof.parquet")
    lo, hi = model().bounds(o["oof"].to_numpy())
    return o.assign(low=lo, high=hi)


def engine_rows(unit: int) -> pd.DataFrame:
    s = test_scored()
    return s[s["unit"] == unit].reset_index(drop=True)
