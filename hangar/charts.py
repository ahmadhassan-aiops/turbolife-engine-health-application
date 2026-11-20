"""Plotly figures in the hangar style (system fonts inside charts: web fonts load late and clip labels)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hangar import theme as t
from turbolife.features import SENSORS


def _style(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=12, t=36 if legend else 12, b=8), paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family=t.CHART_FONT, size=12.5, color=t.INK), showlegend=legend, hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=12)),
                      hoverlabel=dict(font=dict(family=t.CHART_FONT, size=12.5), bgcolor="white", bordercolor=t.LINE))
    fig.update_xaxes(gridcolor=t.SOFT_LINE, linecolor=t.LINE, zeroline=False, ticks="outside", tickcolor=t.LINE)
    fig.update_yaxes(gridcolor=t.SOFT_LINE, linecolor=t.LINE, zeroline=False)
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def life_history(rows: pd.DataFrame, cap: int, show_actual: bool = True) -> go.Figure:
    """The prediction at every cycle so far, with its 80% range, against the true remaining life."""
    fig = go.Figure()
    x = rows["cycle"]
    fig.add_trace(go.Scatter(x=pd.concat([x, x[::-1]]), y=pd.concat([rows["high"], rows["low"][::-1]]), fill="toself", mode="lines",
                             fillcolor="rgba(67,56,202,.13)", line=dict(width=0), name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=rows["predicted"], name="Predicted cycles left", line=dict(color=t.ACCENT, width=2.4),
                             hovertemplate="cycle %{x}<br>predicted %{y:.0f} cycles left<extra></extra>"))
    if show_actual and "actual" in rows:
        fig.add_trace(go.Scatter(x=x, y=rows["actual"], name="Actual cycles left", line=dict(color=t.ACTUAL, width=1.8, dash="dash"),
                                 hovertemplate="cycle %{x}<br>actual %{y:.0f} cycles left<extra></extra>"))
    fig.add_hline(y=cap, line=dict(color=t.FAINT, width=1, dash="dot"), annotation_text=f"model ceiling {cap}",
                  annotation_position="top left", annotation_font=dict(size=11, color=t.MUTED))
    top = max(cap + 15, float(rows["actual"].max()) + 10 if show_actual and "actual" in rows else 0)
    fig.update_yaxes(title="cycles left", range=[0, top])
    fig.update_xaxes(title="cycles flown")
    return _style(fig, 330)


def sensor(rows: pd.DataFrame, code: str, bands: dict) -> go.Figure:
    """One sensor over the engine's history: raw readings, the 10-cycle mean, and the training fleet's healthy and
    near-failure ranges."""
    tag, name, unit = SENSORS[code]
    fig = go.Figure()
    h, w = bands["healthy"], bands["worn"]
    fig.add_hrect(y0=h[0], y1=h[1], fillcolor=t.LANE_SOFT["Fly"], line_width=0, layer="below")
    fig.add_hrect(y0=w[0], y1=w[1], fillcolor=t.LANE_SOFT["Ground"], line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=11, symbol="square", color=t.LANE_SOFT["Fly"],
                             line=dict(color=t.LANE["Fly"], width=1)), name="healthy engines (150+ cycles left)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=11, symbol="square", color=t.LANE_SOFT["Ground"],
                             line=dict(color=t.LANE["Ground"], width=1)), name="engines under 20 cycles from failure"))
    fig.add_trace(go.Scatter(x=rows["cycle"], y=rows[code], mode="lines", line=dict(color="#a7b0bb", width=1), name="reading",
                             hovertemplate="cycle %{x}<br>%{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=rows["cycle"], y=rows[f"{code}_m10"], mode="lines", line=dict(color=t.ACCENT, width=2.4),
                             name="10-cycle mean", hovertemplate="cycle %{x}<br>mean %{y:.2f}<extra></extra>"))
    fig.update_yaxes(title=f"{tag} {unit}".strip())
    fig.update_xaxes(title="cycles flown")
    return _style(fig, 300)


def calls(fleet: pd.DataFrame, cap: int) -> go.Figure:
    """Predicted against actual remaining life for the test engines, coloured by lane; the late zone shaded."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, cap, cap, 0], y=[10, cap + 10, cap + 40, 40], fill="toself", fillcolor="rgba(180,35,24,.07)", mode="lines",
                             line=dict(width=0), name="late by more than 10 cycles", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, cap], y=[0, cap], mode="lines", line=dict(color=t.FAINT, width=1, dash="dot"), name="exact",
                             hoverinfo="skip", showlegend=False))
    for lane in ("Ground", "Plan", "Fly"):
        f = fleet[fleet["lane"] == lane]
        fig.add_trace(go.Scatter(x=np.minimum(f["actual"], cap), y=f["predicted"], mode="markers", name=lane,
                                 marker=dict(size=10, color=t.LANE[lane], symbol={"Ground": "square", "Plan": "diamond", "Fly": "circle"}[lane],
                                             line=dict(color="white", width=1.5)),
                                 customdata=np.stack([f.index, f["actual"], f["low"]], axis=1),
                                 hovertemplate="engine %{customdata[0]}<br>actual %{customdata[1]} cycles left<br>"
                                               "predicted %{y:.0f}, safe life %{customdata[2]:.0f}<extra>" + lane + "</extra>"))
    fig.update_xaxes(title=f"actual cycles left (capped at {cap})", range=[-3, cap + 5])
    fig.update_yaxes(title="predicted cycles left", range=[-3, cap + 12])
    return _style(fig, 400)


def error_by_band(df: pd.DataFrame, actual: str, pred: str, cap: int) -> go.Figure:
    """Average error by how much life is truly left: late (prediction above the truth) and early, in cycles."""
    a = np.minimum(df[actual], cap)
    d = df[pred] - a
    band = pd.cut(a, [-1, 25, 50, 75, 100, cap], labels=["0-25", "26-50", "51-75", "76-100", f"101-{cap}"])
    g = pd.DataFrame({"band": band, "late": d.clip(lower=0), "early": (-d).clip(lower=0)}).groupby("band", observed=False).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g.index.astype(str), y=g["late"], name="late (says more life)", marker_color=t.LANE["Ground"],
                         hovertemplate="%{x} cycles left<br>late by %{y:.1f} on average<extra></extra>"))
    fig.add_trace(go.Bar(x=g.index.astype(str), y=g["early"], name="early (says less life)", marker_color=t.ACCENT,
                         hovertemplate="%{x} cycles left<br>early by %{y:.1f} on average<extra></extra>"))
    fig.update_layout(barmode="group", bargap=.35, bargroupgap=.08)
    fig.update_xaxes(title="actual cycles left")
    fig.update_yaxes(title="average error (cycles)")
    return _style(fig, 300)


def importance(imp: dict) -> go.Figure:
    s = pd.Series(imp).sort_values()
    labels = [("Cycles flown" if k == "cycle" else f"{SENSORS[k][0]} · {SENSORS[k][1]}") for k in s.index]
    fig = go.Figure(go.Bar(x=s.values * 100, y=labels, orientation="h", marker_color=[t.PLATE if k == "cycle" else t.ACCENT for k in s.index],
                           hovertemplate="%{y}<br>%{x:.1f}% of the model's gain<extra></extra>"))
    fig.update_xaxes(title="share of the model's split gain (%)")
    return _style(fig, 430, legend=False)
