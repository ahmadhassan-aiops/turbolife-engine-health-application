"""Safety check: where the model is late (says more life than is left), and what the lanes catch, on the test engines
and on every cycle of the 100 training engines scored while held out."""

from __future__ import annotations

import numpy as np
import streamlit as st

from hangar import charts, data, ui
from turbolife.model import LifeModel
from turbolife.schema import Policy


def _matrix(actual: np.ndarray, lanes: np.ndarray, ground: int, plan: int) -> str:
    bands = [(0, ground, f"0 to {ground}"), (ground + 1, plan, f"{ground + 1} to {plan}"), (plan + 1, 10_000, f"over {plan}")]
    rows = ""
    for i, (lo, hi, label) in enumerate(bands):
        sel = (actual >= lo) & (actual <= hi)
        n = int(sel.sum())
        cells = ""
        for j, lane in enumerate(("Ground", "Plan", "Fly")):
            k = int((sel & (lanes == lane)).sum())
            bad = (i == 0 and lane == "Fly") or (i == 0 and lane == "Plan")
            good = i == j
            style = ' style="background:#e7f4ec;font-weight:700"' if good and k else (' class="n late"' if bad and k else "")
            cells += f'<td class="n"{style}>{k:,}<span style="color:#6f7b88;font-weight:400"> · {k / n:.0%}</span></td>' if n else '<td class="n">0</td>'
        rows += f"<tr><td>{label} cycles left</td><td class=\"n\">{n:,}</td>{cells}</tr>"
    return ('<table class="tbl"><thead><tr><th>Actually left</th><th class="n">Engines</th>' + "".join(
        f'<th class="n">{ui.lane(lane)}</th>' for lane in ("Ground", "Plan", "Fly")) + f"</tr></thead><tbody>{rows}</tbody></table>")


def render() -> None:
    m = data.model()
    cap, d = m.meta["cap"], m.meta["policy"]
    ui.page("TL-03", "SAFETY CHECK", "When is the model late, and what do the lanes catch?",
            "A prediction above the truth is <b>late</b>: it says an engine has more life than it does, the dangerous "
            "direction. Lanes use the low end of the 80% range, the <b>safe life</b>, to leave a margin for that.")
    a, b = st.columns(2, gap="medium")
    ground = a.slider("Ground below this safe life (cycles)", 5, 60, d["ground"], step=5, key="s_ground")
    plan = b.slider("Plan a shop visit below this safe life (cycles)", 10, 120, d["plan"], step=5, key="s_plan")
    if plan <= ground:
        st.error("The plan line must be above the ground line", icon=":material/error:")
        ui.footer()
        return
    policy = Policy(ground=ground, plan=plan)
    f = data.fleet(policy.model_dump_json())

    with ui.panel("TL-03.1", "What the lanes catch", "fact_check", "rows: the cycles truly left; columns: the lane the board "
                  "chose. Green: the right lane. Red: an engine near failure left in service", main=True):
        x, y = st.columns(2, gap="medium")
        with x:
            st.html(f'<p class="sub" style="margin:0 0 .5rem"><b>Test engines</b> at their last logged cycle (100)</p>'
                    + _matrix(f["actual"].to_numpy(), f["lane"].to_numpy(), ground, plan))
        with y:
            o = data.oof()
            lanes = LifeModel.lanes(o["low"].to_numpy(), policy)
            st.html(f'<p class="sub" style="margin:0 0 .5rem"><b>Training engines</b>, every cycle, each engine scored while held out '
                    f'({len(o):,} cycles)</p>' + _matrix(o["rul"].to_numpy(), lanes, ground, plan))
        near = o["rul"] <= ground
        missed = int((near & (lanes == "Fly")).sum())
        ui.notes(f"On the training engines, {missed:,} of {int(near.sum()):,} cycles within {ground} of failure "
                 f"({missed / near.sum():.1%}) were cleared to fly under this policy.",
                 "Raising the ground line catches more engines near failure and grounds more that still had life: the "
                 "price of safety is shop visits that come early.")

    c, e = st.columns([1.2, 1], gap="medium")
    with c:
        with ui.panel("TL-03.2", "Every call on the test engines", "scatter_plot", "predicted against actual cycles left; above the "
                      "dotted line the model is late"):
            charts.show(charts.calls(f, cap), f"calls_{ground}_{plan}")
    with e:
        with ui.panel("TL-03.3", "Error by life left", "bar_chart", "training engines, every cycle, held out"):
            charts.show(charts.error_by_band(o, "rul", "oof", cap), "bands")
            a_ = o["rul"].clip(upper=cap)
            d_ = o["oof"] - a_
            last25 = d_[a_ <= 25].abs().mean()
            mid = d_[(a_ > 25) & (a_ <= 100)].mean()
            far = d_[a_ > 100].mean()
            ui.notes(f"Sharpest near failure, where the sensors move fastest: off by {last25:.1f} cycles on average in "
                     "the last 25.",
                     f"With 26 to 100 cycles left it leans late, by {mid:.1f} cycles on average: wear has begun but the "
                     "sensors have barely moved. That is why lanes use the safe life, not the prediction.",
                     f"With over 100 left it leans early ({-far:.1f} cycles), held down by the ceiling.")

    with ui.panel("TL-03.4", "Late calls on the test engines", "flag", "engines the model gave more than 10 cycles too many"):
        late = f[f["error"] > 10].sort_values("error", ascending=False)
        rows = "".join(f'<tr><td class="n">{u}</td><td class="n">{int(r.cycle)}</td><td class="n">{r.predicted:.0f}</td>'
                       f'<td class="n">{r.low:.0f}</td><td class="n">{int(r.actual)}</td><td class="n late">+{r.error:.0f}</td>'
                       f"<td>{ui.lane(r.lane)}</td></tr>" for u, r in late.iterrows())
        st.html('<table class="tbl"><thead><tr><th class="n">Engine</th><th class="n">Cycles flown</th><th class="n">Predicted</th>'
                '<th class="n">Safe life</th><th class="n">Actual</th><th class="n">Late by</th><th>Lane</th></tr></thead>'
                f"<tbody>{rows}</tbody></table>")
        n_near = int((late["actual"] <= 50).sum())
        ui.notes(f"{len(late)} of 100 engines are late by more than 10 cycles; {n_near} of them had 50 cycles or fewer left.",
                 f"The other {len(late) - n_near} still had more than 50 cycles left at their last logged cycle.")
    ui.footer()
