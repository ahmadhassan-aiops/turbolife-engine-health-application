"""
TurboLife: how many cycles each turbofan engine has left, and which lane it belongs in (ground, plan, fly). LightGBM on
per-engine sensor histories from NASA C-MAPSS FD001. Model in model/ (built by train.py), code in turbolife/.
"""

import streamlit as st

st.set_page_config(page_title="TurboLife | Turbofan remaining life | Ahmad Hassan", page_icon=":material/flight:",
                   layout="wide", initial_sidebar_state="collapsed")

from hangar.ui import icon, inject_css  # noqa: E402
from nav import PAGES  # noqa: E402

inject_css()
current = st.navigation(PAGES, position="hidden")
cols = st.columns([1.7, 1, 1, 1, 1, 1], vertical_alignment="center")
cols[0].html(f'<div class="brand"><span class="mk">{icon("flight")}</span><div><p class="name">TurboLife</p>'
             '<p class="tagline">TURBOFAN REMAINING LIFE</p></div></div>')
for col, p in zip(cols[1:], PAGES):
    col.page_link(p, label=p.title, icon=p.icon, width="stretch")
k = [p.title for p in PAGES].index(current.title) + 2
st.html(f'<style>[data-testid="stHorizontalBlock"]:has(.brand) > [data-testid="stColumn"]:nth-child({k}) a '
        '{ border-color: var(--accent) !important; box-shadow: 0 2px 0 var(--accent) !important; } '
        f'[data-testid="stHorizontalBlock"]:has(.brand) > [data-testid="stColumn"]:nth-child({k}) a::before '
        '{ background: var(--accent) !important; box-shadow: 0 0 0 3px rgba(67,56,202,.2), 0 0 8px rgba(67,56,202,.7); }</style>')
current.run()
