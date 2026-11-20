"""The shipped model matches its evaluation, and the inputs are validated."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbolife.features import FEATURES, SENSOR_CODES, make_features, read_cmapss, read_truth
from turbolife.model import LifeModel
from turbolife.schema import Policy, read_log

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def model() -> LifeModel:
    return LifeModel()


@pytest.fixture(scope="module")
def test_last() -> tuple[pd.DataFrame, np.ndarray]:
    F = make_features(read_cmapss(ROOT / "data" / "test_FD001.txt"))
    return F.groupby("unit").tail(1), read_truth(ROOT / "data" / "RUL_FD001.txt")


def test_test_score_matches_metadata(model, test_last):
    X, truth = test_last
    p = model.predict(X)
    rmse = float(np.sqrt(np.mean((p - np.minimum(truth, 125)) ** 2)))
    assert rmse == pytest.approx(model.meta["test"]["rmse"], abs=1e-6)
    assert rmse < 15


def test_chosen_by_cross_validation(model):
    rows = model.meta["comparison"]
    best = min(rows, key=lambda r: r["cv_rmse"])
    assert best["model"] == model.meta["chosen"] == "LightGBM"
    assert rows[0]["cv_rmse"] > 3 * best["cv_rmse"]   # far better than the average-life baseline


def test_features_never_look_ahead():
    log = read_cmapss(ROOT / "data" / "test_FD001.txt")
    one = log[log["unit"] == 7]
    full = make_features(one)
    cut = make_features(one.head(40))
    pd.testing.assert_frame_equal(full.head(40).reset_index(drop=True), cut.reset_index(drop=True))


def test_features_stay_within_each_engine():
    log = read_cmapss(ROOT / "data" / "test_FD001.txt")
    both = make_features(log[log["unit"].isin([1, 2])])
    alone = make_features(log[log["unit"] == 2])
    pd.testing.assert_frame_equal(both[both["unit"] == 2].reset_index(drop=True), alone.reset_index(drop=True))


def test_explanation_adds_up(model, test_last):
    X, _ = test_last
    row = X.iloc[[3]]
    base, parts = model.explain(row)
    assert base + parts.sum() == pytest.approx(float(model.model.predict(row[FEATURES])[0]), abs=1e-6)
    assert set(parts.index) == {"cycle", *SENSOR_CODES}


def test_bounds_and_lanes(model):
    lo, hi = model.bounds(np.array([5.0, 60.0, 125.0]))
    assert (lo <= np.array([5, 60, 125])).all() and (hi >= np.array([5, 60, 125])).all() and (lo >= 0).all()
    lanes = model.lanes(np.array([3, 30, 90]), Policy(ground=20, plan=50))
    assert lanes.tolist() == ["Ground", "Plan", "Fly"]


def test_policy_order():
    with pytest.raises(ValueError):
        Policy(ground=50, plan=40)


def test_read_log_variants(model):
    raw = read_cmapss(ROOT / "data" / "test_FD001.txt")
    one = raw[raw["unit"] == 31]
    limits = model.meta["limits"]
    # named columns, tag names, and the raw 26-column layout all read the same
    a = read_log(one.drop(columns="unit"), limits)
    tags = one[["cycle"] + SENSOR_CODES].rename(columns={"s4": "T50", "s11": "Ps30"})
    b = read_log(tags, limits)
    c = read_log(pd.DataFrame(one.to_numpy()), limits)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)
    with pytest.raises(ValueError, match="Missing"):
        read_log(one.drop(columns=["s4"]), limits)
    with pytest.raises(ValueError, match="gaps"):
        read_log(one.drop(columns="unit").iloc[1:], limits)
    bad = one.drop(columns="unit").copy()
    bad.loc[bad.index[5], "s4"] = 9999
    with pytest.raises(ValueError, match="T50"):
        read_log(bad, limits)


def test_metadata_is_complete():
    meta = json.loads((ROOT / "model" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["features"] == FEATURES
    assert len(meta["range"]["quantiles"]) == len(meta["range"]["edges"]) + 1
