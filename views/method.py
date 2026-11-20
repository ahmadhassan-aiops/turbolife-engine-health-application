"""Data & method: the data, the target, the features, validation, the range, the lanes and the limits."""

from __future__ import annotations

import streamlit as st

from hangar import data, ui
from hangar.site import NOTEBOOK_URL
from turbolife.features import SENSORS


def render() -> None:
    meta = data.model().meta
    cap, life = meta["cap"], meta["life"]
    ui.page("TL-05", "DATA & METHOD", "How TurboLife is built",
            f'The data, the target, the features, how the model was chosen and checked, and its limits. The full analysis is '
            f'in the <a href="{NOTEBOOK_URL}" target="_blank">project notebook</a>.')
    a, b = st.columns(2, gap="medium")
    with a:
        with ui.panel("TL-05.1", "Data", "dataset", main=True):
            st.html(f"""<dl class="facts">
<dt>Source</dt><dd>NASA C-MAPSS turbofan degradation simulation, subset FD001: one operating condition, one fault mode (high-pressure compressor wear)</dd>
<dt>Training</dt><dd>{meta['train_engines']} engines run from a healthy start to failure: {meta['train_rows']:,} cycles; lives of {life['min']:.0f} to {life['max']:.0f} cycles (median {life['50%']:.0f})</dd>
<dt>Test</dt><dd>{meta['test_engines']} engines stopped some time before failure: {meta['test_rows']:,} cycles; NASA gives the true cycles left after each one's last cycle</dd>
<dt>Sensors</dt><dd>21 sensors and 3 settings per cycle; the settings and 7 sensors are constant or nearly so in FD001, so 14 sensors are used</dd>
</dl>""")
        with ui.panel("TL-05.2", "Target and features", "functions"):
            st.html(f"""<dl class="facts">
<dt>Target</dt><dd>Cycles left before failure, capped at {cap}: early in life the sensors cannot tell 200 cycles left from {cap}, so every healthy engine counts as {cap}</dd>
<dt>Features</dt><dd>For each of the 14 sensors: the reading, its 10- and 30-cycle means and its slope over the last 20 cycles; plus cycles flown. 57 in all</dd>
<dt>No look-ahead</dt><dd>Every feature at a cycle uses only that engine's cycles up to it, so the model sees what a live engine would show</dd>
</dl>""")
    with b:
        with ui.panel("TL-05.3", "Model and checks", "model_training"):
            st.html(f"""<dl class="facts">
<dt>Model</dt><dd>{meta['model']}</dd>
<dt>Selection</dt><dd>Five candidates (an average baseline, ridge regression, random forest, histogram gradient boosting, LightGBM) scored by 5-fold cross-validation grouped by engine: an engine's cycles are never on both sides of a fold</dd>
<dt>Test</dt><dd>The chosen model, refit on all 100 training engines, scored once on the test engines at their last logged cycle</dd>
<dt>Range</dt><dd>10th to 90th percentile of the cross-validation errors of cycles with a similar prediction (five bands)</dd>
<dt>Lanes</dt><dd>Safe life = the low end of the range. Ground below {meta['policy']['ground']} cycles, plan a shop visit below {meta['policy']['plan']}, otherwise fly. These lines are maintenance policy, set before scoring the test engines; both pages that use them let you move them</dd>
<dt>Explanation</dt><dd>LightGBM's own per-feature contributions (TreeSHAP), summed per sensor; they add up to the prediction before the cap</dd>
</dl>""")
        with ui.panel("TL-05.4", "Limits", "report"):
            ui.notes("FD001 is simulated, with one operating condition and one fault mode. Real engines fly mixed missions "
                     "and fail in more ways; the other C-MAPSS subsets (FD002 to FD004) would need their own model.",
                     f"The cap means the model never says more than {cap} cycles: it ranks engines near failure, not new ones.",
                     "Uploaded logs must look like FD001 data (same sensors and units, readings near the training range) "
                     "and start at cycle 1.")

    with ui.panel("TL-05.5", "Sensor reference", "table_rows", "the 14 sensors the model reads"):
        rows = "".join(f'<tr><td class="n">{c}</td><td><b>{tag}</b></td><td>{name}</td><td>{unit}</td><td class="n">{meta["importance"][c]:.1%}</td></tr>'
                       for c, (tag, name, unit) in SENSORS.items())
        st.html('<table class="tbl"><thead><tr><th class="n">Column</th><th>Tag</th><th>Measures</th><th>Unit</th><th class="n">Share of gain</th>'
                f"</tr></thead><tbody>{rows}</tbody></table>")
    ui.footer()
