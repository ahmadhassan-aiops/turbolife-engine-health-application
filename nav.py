"""Page registry (custom key row, native navigation hidden)."""

import streamlit as st

from views import engine, fleet, method, report, safety

FLEET = st.Page(fleet.render, title="Fleet", icon=":material/flight:", url_path="fleet", default=True)
ENGINE = st.Page(engine.render, title="Engine", icon=":material/precision_manufacturing:", url_path="engine")
SAFETY = st.Page(safety.render, title="Safety", icon=":material/health_and_safety:", url_path="safety")
REPORT = st.Page(report.render, title="Report", icon=":material/leaderboard:", url_path="report")
METHOD = st.Page(method.render, title="Method", icon=":material/description:", url_path="method")

PAGES = [FLEET, ENGINE, SAFETY, REPORT, METHOD]
