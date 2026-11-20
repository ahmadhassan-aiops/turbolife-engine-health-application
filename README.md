# TurboLife: turbofan engine remaining life

**Live app:** [turbolife-engine-health-application.streamlit.app](https://turbolife-engine-health-application.streamlit.app/) · **Notebook and project folder:** [AIOPS-Portfolio-Projects](https://github.com/ahmadhassan-aiops/AIOPS-Portfolio-Projects/tree/main/2.%20Machine%20Learning%20Projects/RLU%20jet%20prediction)

How many flight cycles does each engine have left, and which ones need the shop? TurboLife, by **Ahmad Hassan** (AI
Engineer), reads a turbofan engine's sensor log cycle by cycle and predicts its **remaining useful life**, with an 80%
range. The low end of that range, the **safe life**, sorts the engine into a lane: **Ground** (under 20 cycles),
**Plan** a shop visit (under 50) or **Fly**. Data: NASA C-MAPSS FD001, 100 engines run to failure for training and
100 test engines stopped before failure.

| Test engines (truth capped at 125) | |
|---|---|
| RMSE | 13.8 cycles (average-life baseline: 41.9) |
| MAE | 9.8 cycles |
| NASA score (PHM 2008) | 338 |
| 80% range holds | 77% of engines |
| Engines within 20 cycles of failure cleared to fly | 0 of 16 |

- **Fleet board**: the 100 test engines in Ground / Plan / Fly bays, what the lanes catch, the ranked list with the
  actual cycles left for checking; the policy lines can be moved.
- **Engine file**: one engine (or an uploaded log) on a remaining-life dial, the cycles each sensor adds or takes away
  (LightGBM's TreeSHAP contributions), the estimate at every cycle of its log against the truth, and each sensor
  against healthy and worn engines.
- **Safety check**: lanes against the truth on the test engines and on every held-out training cycle, every call,
  errors by life left, and the late calls.
- **Model report**: five candidates under cross-validation grouped by engine, test scores, what the model reads, the range.
- **Data & method**: data, target, features, checks, limits and the sensor reference.

## The model

LightGBM (600 trees) on 57 features built from each engine's own cycles up to the current one: for 14 sensors the
reading, its 10- and 30-cycle means and its 20-cycle slope, plus cycles flown. Remaining life is capped at 125,
because early in life the sensors cannot tell 200 cycles from 125.

| Model | CV RMSE (5 folds by engine) | Test RMSE |
|---|---|---|
| Baseline: average remaining life | 41.67 | 41.94 |
| Ridge regression | 15.46 | 15.71 |
| Random forest | 14.07 | 14.43 |
| Histogram gradient boosting | 13.78 | 14.30 |
| **LightGBM** | **13.58** | **13.79** |

The project notebook was audited and rebuilt before the app. The first version clipped every column at 125 (ten of
the fourteen sensors became constants), computed the test set's rolling features across engine boundaries, and
cross-validated on shuffled rows, which reports about half the real error; its test RMSE was 22.7. Details are in the
notebook.

## Why this stack

| Need | Choice |
|---|---|
| Accurate, small, explainable model | **LightGBM** (under 1 MB), with its built-in TreeSHAP contributions summed per sensor |
| Honest validation | **GroupKFold** by engine; the test engines scored once, after the model was chosen |
| A range a plan can rely on | 10th to 90th percentile of out-of-fold errors, per band of the prediction |
| Valid inputs | **pydantic** for the maintenance policy; uploaded logs checked for columns, gaps and readings far outside training |
| Trust that the shipped files are the evaluated model | **pytest** + GitHub Actions: the test score, the choice by CV, no look-ahead, no mixing of engines, the explanation adding up, every page |

`turbolife/` has no Streamlit dependency, so the same features and model can sit behind an API.

## Run it

```
pip install -r requirements.txt
python train.py        # optional: rebuilds model/ and data/oof.parquet (about 2 minutes)
streamlit run app.py
pip install -r requirements-dev.txt && pytest
```

Data: NASA C-MAPSS turbofan engine degradation simulation, subset FD001 (A. Saxena, K. Goebel, D. Simon and N. Eklund,
2008). TurboLife is a portfolio demonstration, not maintenance guidance.
