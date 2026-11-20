"""Engine file: one engine's remaining life on a dial, why the model says so, how the estimate moved as the log grew,
and its sensors against healthy and worn engines. A test engine, or an uploaded log."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from hangar import charts, data, ui
from hangar.ui import cycles
from turbolife.features import SENSORS, make_features
from turbolife.model import LifeModel
from turbolife.schema import Policy, read_log


def _uploaded(m: LifeModel) -> pd.DataFrame | None:
    a, b = st.columns([2, 1], gap="medium", vertical_alignment="bottom")
    up = a.file_uploader("One engine's log (CSV): cycle plus the 14 sensors, named s2 ... s21 or T24, T30, ...; or the raw "
                         "26-column C-MAPSS layout", type=["csv", "txt"], key="e_file")
    sample = data.test_log().query("unit == 31").drop(columns="unit")
    b.download_button("Download a sample log", sample.to_csv(index=False), "turbolife_sample_engine.csv", "text/csv",
                      icon=":material/download:", key="e_sample", width="stretch")
    if up is None:
        st.info("Upload a log to read it, or switch back to a test engine.", icon=":material/upload_file:")
        return None
    try:
        raw = pd.read_csv(up, sep=None, engine="python", header=None if up.name.endswith(".txt") else "infer")
        log = read_log(raw, m.meta["limits"])
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc), icon=":material/error:")
        return None
    rows = make_features(log).reset_index(drop=True)
    rows["predicted"] = m.predict(rows)
    rows["low"], rows["high"] = m.bounds(rows["predicted"].to_numpy())
    return rows


def render() -> None:
    m = data.model()
    meta, cap = m.meta, m.meta["cap"]
    policy = Policy(**meta["policy"])
    ui.page("TL-02", "ENGINE FILE", "How many cycles does this engine have left?",
            "The model reads the engine's whole log so far, cycle by cycle, and predicts the cycles left before failure. "
            "Pick one of the 100 test engines (their true remaining life is known) or upload a log.")

    with ui.panel("TL-02.1", "Engine", "precision_manufacturing"):
        src = st.segmented_control("Source", ["Test engine", "Upload a log"], default="Test engine", required=True, key="e_src")
        if src == "Test engine":
            fleet = data.fleet(policy.model_dump_json())
            st.session_state.setdefault("e_unit", int(fleet.sort_values("low").index[0]))
            unit = st.selectbox("Engine", list(range(1, 101)), key="e_unit",
                                format_func=lambda u: f"Engine {u}: {fleet.loc[u, 'lane']}, {int(fleet.loc[u, 'cycle'])} cycles flown")
            rows = data.engine_rows(unit)
        else:
            unit, rows = None, _uploaded(m)
    if rows is None:
        ui.footer()
        return

    last = rows.iloc[-1]
    lane = LifeModel.lanes([last["low"]], policy)[0]
    known = "actual" in rows
    ui.tiles([("Cycles flown", f"{int(last['cycle'])}", "logged so far"),
              ("Predicted left", cycles(last["predicted"]), "cycles before failure"),
              ("Safe life", cycles(last["low"]), f"low end of the 80% range ({cycles(last['low'])} to {cycles(last['high'])})"),
              ("Lane", ui.lane(lane), f"ground below {policy.ground}, plan below {policy.plan}"),
              ("Actual left", f"{int(last['actual'])}" if known else "unknown",
               (f"model {'late' if last['predicted'] > min(last['actual'], cap) else 'early'} by "
                f"{abs(last['predicted'] - min(last['actual'], cap)):.0f}") if known else "uploaded log")], highlight=1)

    a, b = st.columns([1, 1.25], gap="medium")
    with a:
        with ui.panel("TL-02.2", "Remaining life", "speed", main=True):
            st.html(ui.dial(last["predicted"], last["low"], last["high"], policy.ground, policy.plan, cap))
    with b:
        with ui.panel("TL-02.3", "Why this estimate", "balance", "cycles each sensor adds to or takes from the average engine's "
                      "estimate (its reading, means and trend together)"):
            base, parts = m.explain(rows.iloc[[-1]])
            order = parts.abs().sort_values(ascending=False).index
            top, rest = list(order[:7]), parts[order[7:]].sum()
            items = [("Cycles" if k == "cycle" else SENSORS[k][0], "cycles flown" if k == "cycle" else SENSORS[k][1], float(parts[k])) for k in top]
            items.append(("Other", f"{len(order) - 7} sensors", float(rest)))
            scale = max(abs(v) for *_, v in items) * 1.05
            st.html(f'<p class="sub" style="margin:0 0 .6rem">Average engine <b>{base:.0f}</b> cycles '
                    f'{"+" if parts.sum() >= 0 else "−"} this engine\'s readings <b>{abs(parts.sum()):.0f}</b> = <b>{base + parts.sum():.0f}</b>'
                    + (f" (shown as {cap}+)" if base + parts.sum() >= cap - 0.5 else "") + "</p>" + ui.contributions(items, scale))

    with ui.panel("TL-02.4", "The estimate as the log grew", "timeline", "the prediction the model would have made at every cycle "
                  "of this engine's log" + (", against the cycles truly left" if known else "")):
        charts.show(charts.life_history(rows, cap, known), f"hist_{unit}")
        notes = ["Early in life the sensors barely move, so the model holds near its ceiling of "
                 f"{cap} cycles: an engine that far from failure cannot be told apart from a newer one."]
        if known:
            gap = (rows["predicted"] - rows["actual"].clip(upper=cap)).abs()
            notes.append(f"Over the last 20 logged cycles the prediction was within {gap.tail(20).mean():.0f} cycles of the "
                         "truth on average.")
        ui.notes(*notes)

    with ui.panel("TL-02.5", "Sensors", "monitoring", "shaded: the range of the 100 training engines when healthy (green) and "
                  "in their last 20 cycles (red)"):
        imp = meta["importance"]
        codes = sorted(SENSORS, key=lambda s: -imp[s])
        code = st.selectbox("Sensor (most used by the model first)", codes, key="e_sensor",
                            format_func=lambda s: f"{SENSORS[s][0]} · {SENSORS[s][1]} ({imp[s]:.0%} of the model's gain)")
        charts.show(charts.sensor(rows, code, meta["bands"][code]), f"sens_{unit}_{code}")
    ui.footer()
