"""Model report: the candidates under grouped cross-validation, the chosen model on the test engines, the range and
what the model reads."""

from __future__ import annotations

import streamlit as st

from hangar import charts, data, ui


def render() -> None:
    meta = data.model().meta
    te, cap = meta["test"], meta["cap"]
    ui.page("TL-04", "MODEL REPORT", "How accurate is the remaining-life model?",
            f"Five candidates scored by cross-validation on the 100 training engines, five folds by engine. The best, "
            f"<b>LightGBM</b>, refit on all of them and scored once on the 100 test engines. Errors are in cycles, with the "
            f"truth capped at {cap} like the model.")
    ui.tiles([("Test RMSE", f"{te['rmse']:.1f}", "cycles (truth capped at 125)"),
              ("Test MAE", f"{te['mae']:.1f}", "cycles, average miss"),
              ("R²", f"{te['r2']:.2f}", "share of the spread explained"),
              ("NASA score", f"{te['nasa']:.0f}", "lower is better; late errors cost more"),
              ("80% range holds", f"{te['coverage']:.0%}", f"of test engines (cross-validation {meta['cv']['coverage']:.0%})")], highlight=0)

    with ui.panel("TL-04.1", "Candidates", "leaderboard", "cross-validation RMSE picks the model; test columns are shown for "
                  "reference", main=True):
        rows = ""
        for r in meta["comparison"]:
            pick = r["model"] == meta["chosen"]
            rows += (f'<tr class="{"pick" if pick else ""}"><td>{r["model"]}</td><td class="n">{r["cv_rmse"]:.2f}</td>'
                     f'<td class="n">{r["test_rmse"]:.2f}</td><td class="n">{r["test_mae"]:.2f}</td><td class="n">{r["test_nasa"]:,.0f}</td>'
                     f'<td class="n">{r["test_late_20"]}</td><td class="n">{"chosen" if pick else ""}</td></tr>')
        st.html('<table class="tbl"><thead><tr><th>Model</th><th class="n">CV RMSE</th><th class="n">Test RMSE</th><th class="n">Test MAE</th>'
                f'<th class="n">NASA score</th><th class="n">Late by 20+</th><th class="n"></th></tr></thead><tbody>{rows}</tbody></table>')
        ui.notes("Every candidate reads the same features. Trees beat ridge regression because wear shows up late and "
                 "fast: the sensors bend rather than drift in a straight line.",
                 "The three tree models are within half a cycle of each other in cross-validation; LightGBM is best there "
                 "and on the test engines.",
                 "The NASA score (PHM 2008) grows exponentially with each error, faster for late predictions than early ones.")

    a, b = st.columns([1.1, 1], gap="medium")
    with a:
        with ui.panel("TL-04.2", "What the model reads", "sensors", "share of the trees' split gain, each sensor's reading, means "
                      "and trend together"):
            charts.show(charts.importance(meta["importance"]), "imp")
    with b:
        with ui.panel("TL-04.3", "The 80% range", "width", "from cross-validation errors of engines with a similar prediction"):
            edges, q = meta["range"]["edges"], meta["range"]["quantiles"]
            labels = [f"under {edges[0]}"] + [f"{edges[i]} to {edges[i + 1]}" for i in range(len(edges) - 1)] + [f"{edges[-1]} and over"]
            rows = "".join(f'<tr><td>{lab}</td><td class="n">{lo:+.1f}</td><td class="n">{hi:+.1f}</td></tr>' for lab, (lo, hi) in zip(labels, q))
            st.html('<table class="tbl"><thead><tr><th>Predicted cycles left</th><th class="n">Low end</th><th class="n">High end</th>'
                    f"</tr></thead><tbody>{rows}</tbody></table>")
            ui.notes("The range is narrowest near failure (about ±5 cycles) and widest in the middle of life.",
                     f"On the test engines it holds the truth {te['coverage']:.0%} of the time, a little under the 80% it holds "
                     "in cross-validation: the test engines stop at random points in their life.")
    ui.footer()
