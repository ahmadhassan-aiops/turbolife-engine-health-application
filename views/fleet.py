"""Fleet board: the 100 test engines sorted into Ground / Plan / Fly by the model's safe life, then the ranked list."""

from __future__ import annotations

import numpy as np
import streamlit as st
from pydantic import ValidationError

from hangar import data, ui
from hangar import theme as t
from hangar.ui import cycles
from turbolife.schema import Policy


def _policy() -> Policy | None:
    d = data.model().meta["policy"]
    with st.expander("Change the maintenance policy (the lines that sort engines into lanes)", icon=":material/tune:"):
        a, b = st.columns(2, gap="medium")
        ground = a.slider("Ground below this safe life (cycles)", 5, 60, d["ground"], step=5, key="f_ground")
        plan = b.slider("Plan a shop visit below this safe life (cycles)", 10, 120, d["plan"], step=5, key="f_plan")
        st.caption("Safe life is the low end of the model's 80% range: about 9 in 10 engines with a similar prediction "
                   "had at least that many cycles left.")
    try:
        return Policy(ground=ground, plan=plan)
    except ValidationError as exc:
        st.error(exc.errors()[0]["msg"].removeprefix("Value error, "), icon=":material/error:")
        return None


def _bays(f) -> str:
    out = ""
    for lane in ("Ground", "Plan", "Fly"):
        e = f[f["lane"] == lane].sort_values("low")
        chips = "".join(f'<i title="engine {u}: safe life {cycles(r.low)}, predicted {cycles(r.predicted)}">{u}</i>' for u, r in e.iterrows())
        out += (f'<div class="bay" style="--c:{t.LANE[lane]};--s:{t.LANE_SOFT[lane]}"><div class="bay-h">{ui.icon(t.LANE_ICON[lane])}'
                f'<b>{lane}</b><span class="ct">{len(e)}</span></div><p class="bay-n">{t.LANE_NOTE[lane]}; engines by safe life, lowest first</p>'
                f'<div class="eng">{chips or "<em>none</em>"}</div></div>')
    return f'<div class="bays">{out}</div>'


def render() -> None:
    m = data.model()
    cap = m.meta["cap"]
    ui.page("TL-01", "FLEET BOARD", "Which engines need the shop, and when?",
            "100 turbofan engines, each logged up to some cycle before it fails. For each one the model predicts the "
            "<b>cycles left</b>, with an 80% range; the low end of that range, the <b>safe life</b>, sorts it into a lane.")
    with ui.panel("TL-01.1", "Lanes", "flight", "every test engine at its last logged cycle; hover an engine number for its figures", main=True):
        policy = _policy()
        if policy is None:
            ui.footer()
            return
        f = data.fleet(policy.model_dump_json())
        st.html(_bays(f))
        near = f[f["actual"] <= policy.ground]
        flying = near[near["lane"] == "Fly"]
        caught = near[near["lane"] == "Ground"]
        if len(near):
            msg = (f"Of the <b>{len(near)}</b> engines truly within {policy.ground} cycles of failure, the board grounds "
                   f"<b>{len(caught)}</b>, books {len(near) - len(caught) - len(flying)} for the shop and clears "
                   f"<b>{len(flying)}</b> to fly.")
        else:
            msg = f"No engine in the test fleet is truly within {policy.ground} cycles of failure."
        early = f[(f["lane"] == "Ground") & (f["actual"] > policy.plan)]
        ui.takeaway(msg + f" {len(early)} grounded engine{'s' if len(early) != 1 else ''} had more than {policy.plan} cycles left.")

    ui.tiles([("Engines", "100", "C-MAPSS FD001 test set"),
              ("Typical error", f"{m.meta['test']['rmse']:.1f}", "cycles (RMSE, test engines)"),
              ("Late by over 20", f"{m.meta['test']['late_20']}", "engines the model gave 20+ cycles too many"),
              ("80% range holds", f"{m.meta['test']['coverage']:.0%}", "of test engines' true life")], highlight=1)

    with ui.panel("TL-01.2", "Engine list", "format_list_numbered", "sorted by safe life; actual cycles left is known for "
                  "these engines and shown for checking"):
        show = st.toggle("Show the actual cycles left", value=True, key="f_actual")
        rows = ""
        for u, r in f.sort_values(["low", "predicted"]).iterrows():
            late = r.predicted - min(r.actual, cap) > 20
            rows += (f'<tr><td class="n">{u}</td><td class="n">{int(r.cycle)}</td><td class="n"><b>{cycles(r.predicted)}</b></td>'
                     f'<td class="n">{cycles(r.low)} to {cycles(r.high)}</td><td>{ui.lane(r.lane)}</td>'
                     + (f'<td class="n{" late" if late else ""}">{int(r.actual)}{" ⚑" if late else ""}</td>' if show else "") + "</tr>")
        st.html('<div style="max-height:520px;overflow:auto"><table class="tbl"><thead><tr><th class="n">Engine</th>'
                '<th class="n">Cycles flown</th><th class="n">Predicted left</th><th class="n">80% range</th><th>Lane</th>'
                + ('<th class="n">Actual left</th>' if show else "") + f"</tr></thead><tbody>{rows}</tbody></table></div>")
        if show:
            ui.notes("⚑ marks an engine the model gave more than 20 cycles too many: the dangerous direction. "
                     f"{int((f['predicted'] - np.minimum(f['actual'], cap) > 20).sum())} of 100.")
        a, b = st.columns([2, 1], gap="small", vertical_alignment="bottom")
        unit = a.selectbox("Open an engine's file", f.sort_values("low").index.tolist(),
                           format_func=lambda u: f"Engine {u}: {f.loc[u, 'lane']}, predicted {cycles(f.loc[u, 'predicted'])} cycles left",
                           key="f_open")
        if b.button("Open engine file", icon=":material/open_in_new:", key="f_go", width="stretch"):
            st.session_state["e_unit"] = unit
            from nav import ENGINE  # noqa: PLC0415 (nav imports the views)
            st.switch_page(ENGINE)
    ui.footer()
