"""TurboLife components: a hangar maintenance board. Navigation is a row of panel keys with a status lamp; every block
is a riveted aluminium panel under a black placard; readouts are instrument tiles; the remaining life is a dial; notes
are a technician's log."""

from __future__ import annotations

import html
import math
from contextlib import contextmanager

import streamlit as st

from hangar import theme as t
from hangar.site import CODE_URL, DATA_CREDIT, LEARN_URL, NOTEBOOK_URL

_RIVET = "radial-gradient(circle at center, #b9c2cd 0 2.2px, #8f9aa6 2.4px 3px, transparent 3.2px)"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=Figtree:wght@400;500;600;700&family=Red+Hat+Mono:wght@400;500;600&display=swap');
:root {{ --floor:{t.FLOOR}; --panel:{t.PANEL}; --plate:{t.PLATE}; --ink:{t.INK}; --muted:{t.MUTED}; --faint:{t.FAINT}; --line:{t.LINE};
         --soft-line:{t.SOFT_LINE}; --accent:{t.ACCENT}; --accent-soft:{t.ACCENT_SOFT}; --ground:{t.LANE["Ground"]}; --plan:{t.LANE["Plan"]};
         --fly:{t.LANE["Fly"]}; --head:{t.HEAD}; --body:{t.BODY}; --mono:{t.MONO}; }}
html, body, [data-testid="stAppViewContainer"] {{ font-family: var(--body); color: var(--ink); }}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background-color: var(--floor);
    background-image: radial-gradient(#dfe4ea 1px, transparent 1.2px); background-size: 14px 14px; }}
.block-container {{ padding-top: 4.1rem; max-width: 1260px; }}
/* Streamlit Cloud's toolbar strip is 60px tall and takes clicks: the key row starts below it */
header[data-testid="stHeader"] {{ background: var(--floor); }}
.msr {{ font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1; display: inline-block;
        -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased; }}

/* Panels: riveted aluminium plates */
[data-testid="stVerticalBlock"][data-test-scroll-behavior] {{ border: 1px solid var(--line) !important; border-radius: 12px !important;
    background-color: var(--panel); box-shadow: 0 1px 0 #fff inset, 0 2px 6px rgba(23, 32, 43, .07);
    background-image: {_RIVET}, {_RIVET}, {_RIVET}, {_RIVET}; background-repeat: no-repeat; background-size: 14px 14px;
    background-position: 6px 6px, calc(100% - 6px) 6px, 6px calc(100% - 6px), calc(100% - 6px) calc(100% - 6px); padding: 1.1rem 1.25rem !important; }}
/* The page's main panel ("main_*") carries an indigo instrument edge */
[data-testid="stVerticalBlock"][data-test-scroll-behavior][class*="st-key-main_"] {{ border-top: 4px solid var(--accent) !important; box-shadow: 0 2px 10px rgba(67, 56, 202, .12) !important; }}

/* Key row (navigation) */
[data-testid="stHorizontalBlock"]:has(.brand) {{ gap: .45rem; align-items: stretch; margin-bottom: .2rem; }}
.brand {{ display: flex; align-items: center; gap: .7rem; padding: .2rem 0; }}
.brand .mk {{ width: 2.6rem; height: 2.6rem; border-radius: 50%; background: var(--plate); display: grid; place-items: center; color: #fff;
              box-shadow: 0 0 0 3px var(--panel), 0 0 0 4px var(--line); }}
.brand .mk .msr {{ font-size: 1.5rem; }}
.brand .name {{ font-family: var(--head); font-weight: 700; font-size: 1.5rem; letter-spacing: .03em; margin: 0; line-height: 1; }}
.brand .tagline {{ font-family: var(--mono); font-size: .68rem; color: var(--muted); margin: .25rem 0 0; letter-spacing: .06em; }}
[data-testid="stHorizontalBlock"]:has(.brand) [data-testid="stPageLink"] a {{ border-radius: 10px; padding: .55rem .6rem .55rem 1.5rem;
    justify-content: center; background: var(--panel); border: 1px solid var(--line); box-shadow: 0 2px 0 var(--line);
    position: relative; height: 100%; }}
[data-testid="stHorizontalBlock"]:has(.brand) [data-testid="stPageLink"] a::before {{ content: ""; position: absolute; left: .65rem; top: 50%;
    width: 8px; height: 8px; margin-top: -4px; border-radius: 50%; background: #c3cad3; }}
[data-testid="stHorizontalBlock"]:has(.brand) [data-testid="stPageLink"] a p {{ font-family: var(--head); font-weight: 600; font-size: .95rem;
    letter-spacing: .04em; text-transform: uppercase; color: var(--ink); white-space: nowrap; }}
[data-testid="stHorizontalBlock"]:has(.brand) [data-testid="stPageLink"] a span {{ color: var(--muted); }}
[data-testid="stHorizontalBlock"]:has(.brand) [data-testid="stPageLink"] a:hover {{ border-color: var(--accent); }}
[data-testid="stHorizontalBlock"]:has(.brand) [data-testid="stPageLink"] a:focus-visible {{ outline: 2px solid var(--accent) !important; outline-offset: 2px; }}

/* Page head */
.ata {{ display: inline-flex; gap: .5rem; align-items: center; margin: 1.3rem 0 .5rem; font-family: var(--mono); font-size: .74rem;
        letter-spacing: .06em; color: var(--muted); }}
.ata b {{ background: var(--plate); color: #fff; padding: .15rem .5rem; border-radius: 5px; font-weight: 600; }}
.title {{ font-family: var(--head); font-size: 2.45rem; font-weight: 700; margin: 0; line-height: 1.08; color: var(--ink); }}
.lead {{ color: var(--muted); font-size: 1.05rem; line-height: 1.6; max-width: 800px; margin: .5rem 0 1.1rem; }}
.lead b {{ color: var(--ink); }}

/* Placard (panel title) */
.plq {{ display: flex; align-items: center; gap: .7rem; margin: 0 0 .75rem; }}
.plq .p {{ display: inline-flex; align-items: center; gap: .55rem; background: var(--plate); color: #fff; border-radius: 6px; padding: .32rem .7rem .32rem .35rem; }}
.plq .no {{ font-family: var(--mono); font-size: .72rem; font-weight: 600; background: var(--accent); padding: .12rem .4rem; border-radius: 4px; }}
.plq .t {{ font-family: var(--head); font-weight: 600; font-size: 1rem; letter-spacing: .07em; text-transform: uppercase; margin: 0; }}
.plq .ic {{ margin-left: auto; color: var(--faint); }}
.plq .ic .msr {{ font-size: 1.35rem; }}
.sub {{ font-size: .9rem; color: var(--muted); margin: -.2rem 0 .85rem; line-height: 1.5; }}

/* Instrument tiles */
.tiles {{ display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); gap: .6rem; margin: 0 0 1rem; }}
@media (max-width: 860px) {{ .tiles {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.tile {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: .6rem .85rem .65rem; box-shadow: 0 2px 0 var(--soft-line); }}
.tile.hl {{ border-color: var(--accent); box-shadow: 0 2px 0 var(--accent-soft); }}
.tile-l {{ font-family: var(--mono); font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 0; }}
.tile-v {{ font-family: var(--head); font-size: 1.85rem; font-weight: 700; margin: .1rem 0; line-height: 1.1; }}
.tile-n {{ font-size: .8rem; color: var(--faint); margin: 0; line-height: 1.4; }}

/* Lane chips and bays */
.lane {{ display: inline-flex; align-items: center; gap: .3rem; font-family: var(--head); font-weight: 600; font-size: .82rem; letter-spacing: .05em;
         text-transform: uppercase; color: var(--c); background: var(--s); border: 1px solid var(--c); border-radius: 999px; padding: .08rem .55rem .08rem .4rem; }}
.lane .msr {{ font-size: 1rem; }}
.bays {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .8rem; }}
@media (max-width: 860px) {{ .bays {{ grid-template-columns: 1fr; }} }}
.bay {{ border: 1px solid var(--line); border-top: 5px solid var(--c); border-radius: 10px; background: var(--s); padding: .75rem .9rem .85rem; }}
.bay-h {{ display: flex; align-items: center; gap: .45rem; color: var(--c); }}
.bay-h .msr {{ font-size: 1.5rem; }}
.bay-h b {{ font-family: var(--head); font-size: 1.15rem; letter-spacing: .06em; text-transform: uppercase; }}
.bay-h .ct {{ margin-left: auto; font-family: var(--head); font-weight: 700; font-size: 2rem; color: var(--ink); line-height: 1; }}
.bay-n {{ font-size: .84rem; color: var(--muted); margin: .25rem 0 .6rem; }}
.eng {{ display: flex; flex-wrap: wrap; gap: .3rem; }}
.eng i {{ font-style: normal; font-family: var(--mono); font-size: .76rem; font-weight: 600; background: var(--panel); border: 1px solid var(--line);
          border-radius: 5px; padding: .05rem .35rem; color: var(--ink); }}
.eng i.x {{ border-color: var(--c); }}
.takeaway {{ display: flex; gap: .7rem; align-items: flex-start; margin: .9rem 0 0; padding: .7rem .9rem; background: var(--accent-soft);
             border-left: 4px solid var(--accent); border-radius: 0 8px 8px 0; font-size: .98rem; line-height: 1.55; }}
.takeaway .msr {{ color: var(--accent); font-size: 1.4rem; }}

/* The dial */
.dialwrap {{ display: flex; flex-direction: column; align-items: center; padding: .3rem 0 .2rem; }}
.dial {{ position: relative; width: 250px; height: 250px; border-radius: 50%; background: radial-gradient(circle, #fbfcfd 0 62%, #f1f4f7 63% 100%);
         box-shadow: 0 0 0 7px var(--plate), 0 0 0 9px var(--line), inset 0 2px 6px rgba(0,0,0,.08); margin: 14px 0 6px; }}
.dial::before {{ content: ""; position: absolute; inset: 10px; border-radius: 50%;
    background: conic-gradient(from 225deg, var(--ground) 0 var(--a), var(--plan) var(--a) var(--b), var(--fly) var(--b) 270deg, transparent 270deg);
    -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 14px), #000 calc(100% - 13px));
            mask: radial-gradient(farthest-side, transparent calc(100% - 14px), #000 calc(100% - 13px)); }}
.dial .rng {{ position: absolute; inset: 27px; border-radius: 50%;
    background: conic-gradient(from 225deg, transparent 0 var(--lo), rgba(67,56,202,.28) var(--lo) var(--hi), transparent var(--hi));
    -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 9px), #000 calc(100% - 8px));
            mask: radial-gradient(farthest-side, transparent calc(100% - 9px), #000 calc(100% - 8px)); }}
.dial .tk {{ position: absolute; transform: translate(-50%, -50%); font-family: var(--mono); font-size: .7rem; color: var(--muted); }}
.dial .ndl {{ position: absolute; left: 50%; top: 50%; width: 4px; height: 84px; margin-left: -2px; margin-top: -84px; background: var(--accent);
              border-radius: 2px; transform-origin: 50% 100%; transform: rotate(var(--r)); }}
.dial .hub {{ position: absolute; left: 50%; top: 50%; width: 18px; height: 18px; margin: -9px; border-radius: 50%; background: var(--plate);
              box-shadow: 0 0 0 3px #fff; }}
.dial .rd {{ position: absolute; left: 0; right: 0; bottom: 34px; text-align: center; }}
.dial .rd b {{ display: block; font-family: var(--head); font-size: 2.1rem; line-height: 1; }}
.dial .rd span {{ font-family: var(--mono); font-size: .68rem; letter-spacing: .06em; color: var(--muted); text-transform: uppercase; }}
.dial-k {{ display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center; font-size: .8rem; color: var(--muted); margin: .5rem 0 0; }}
.dial-k i {{ display: inline-block; width: 18px; height: 6px; border-radius: 3px; vertical-align: 2px; margin-right: .3rem; }}

/* Contribution bars (cycles of life added or taken by each sensor) */
.cb {{ display: grid; grid-template-columns: minmax(8.5rem, 34%) 1fr 4rem; gap: .2rem .6rem; align-items: center; font-size: .86rem; }}
.cb .n b {{ font-family: var(--mono); font-size: .78rem; }}
.cb .n span {{ color: var(--muted); font-size: .8rem; margin-left: .3rem; }}
.cb .tr {{ position: relative; height: 14px; }}
.cb .tr::before {{ content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: var(--line); }}
.cb .tr i {{ position: absolute; top: 1px; bottom: 1px; border-radius: 3px; }}
.cb .v {{ font-family: var(--mono); font-size: .8rem; text-align: right; }}
.cb-k {{ display: flex; justify-content: space-between; font-family: var(--mono); font-size: .7rem; color: var(--faint); margin: .5rem 0 0; }}

/* Tables */
.tbl {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: .88rem; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }}
.tbl th {{ position: sticky; top: 0; z-index: 1; font-family: var(--mono); font-size: .68rem; letter-spacing: .06em; text-transform: uppercase; color: #fff; font-weight: 600;
           text-align: left; padding: .5rem .6rem; background: var(--plate); }}
.tbl td {{ padding: .42rem .6rem; border-top: 1px solid var(--soft-line); }}
.tbl th.n, .tbl td.n {{ text-align: right; }}
.tbl td.n {{ font-family: var(--mono); }}
.tbl tr.pick td {{ background: var(--accent-soft); font-weight: 600; }}
.tbl tbody tr:nth-child(even) td {{ background: #f8fafb; }}
.tbl tbody tr.pick td {{ background: var(--accent-soft); }}
.late {{ color: var(--ground); font-weight: 600; }}

/* Technician's notes */
.notes {{ margin: .9rem 0 0; border: 1px solid var(--soft-line); border-left: 3px solid var(--plate); border-radius: 4px 8px 8px 4px;
          background: #fbfcfd; padding: .5rem .85rem .35rem; }}
.notes .h {{ font-family: var(--mono); font-size: .7rem; letter-spacing: .1em; color: var(--muted); margin: 0 0 .25rem; display: flex; gap: .35rem; align-items: center; }}
.notes .h .msr {{ font-size: 1rem; color: var(--accent); }}
.notes ul {{ margin: 0; padding-left: 1.1rem; }}
.notes li {{ font-size: .9rem; line-height: 1.55; margin: 0 0 .35rem; }}
.caution {{ display: flex; gap: .55rem; border: 1px solid #f0d9a8; padding: .55rem .8rem; margin: .8rem 0 0; font-size: .87rem; border-radius: 8px;
            line-height: 1.5; color: var(--ink); background: #fffaf0; }}
.caution .msr {{ color: var(--plan); font-size: 1.2rem; }}

/* Data & method */
.facts {{ display: grid; grid-template-columns: minmax(9rem, 30%) 1fr; margin: 0; padding: 0 !important; }}
.facts dt, .facts dd {{ padding: .48rem .3rem; border-bottom: 1px solid var(--soft-line); margin: 0; font-size: .9rem; line-height: 1.5; }}
.facts dt {{ font-family: var(--mono); font-size: .72rem; letter-spacing: .05em; text-transform: uppercase; color: var(--muted); }}

/* Footer */
.foot {{ margin: 2rem 0 .5rem; border-radius: 10px; background: var(--plate); color: #cfd6de; display: flex; flex-wrap: wrap; font-size: .84rem; overflow: hidden; }}
.foot > * {{ padding: .55rem .9rem; border-right: 1px solid #344152; }}
.foot b {{ color: #fff; }}
.foot a {{ color: #fff; font-weight: 600; text-decoration: none; }}
.foot a:hover {{ text-decoration: underline; }}
.foot .credit {{ flex-basis: 100%; border-top: 1px solid #344152; border-right: none; color: #a9b4c0; font-size: .78rem; }}
</style>
"""

CONTROLS = """<style>
/* Controls: one shape, one border weight and one focus ring for every button, segment and input */
:root { --k-a: #4338ca; --k-fill: #1d2633; --k-fill-2: #0f151d; --k-soft: #eef0fd; --k-ink: #17202b;
        --k-line: #aab4c0; --k-panel: #ffffff; --k-r: 8px; }
[data-testid="stMain"] :is([data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"],
  [data-testid="stBaseButton-secondaryFormSubmit"], [data-testid="stBaseButton-primaryFormSubmit"],
  [data-testid="stBaseButton-segmented_control"], [data-testid="stBaseButton-segmented_controlActive"],
  [data-testid="stBaseButton-pills"], [data-testid="stBaseButton-pillsActive"]) {
    border-radius: var(--k-r) !important; border: 1px solid var(--k-line) !important; background: var(--k-panel) !important;
    color: var(--k-ink) !important; box-shadow: 0 2px 0 #dfe4ea !important; min-height: 2.45rem; padding: .35rem .95rem !important;
    transition: border-color .12s ease, background-color .12s ease, color .12s ease; }
[data-testid="stMain"] :is([data-testid^="stBaseButton-secondary"], [data-testid^="stBaseButton-primary"],
  [data-testid^="stBaseButton-segmented_control"], [data-testid^="stBaseButton-pills"]) p {
    font-family: 'Chakra Petch', 'Segoe UI', sans-serif; font-size: .95rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; color: inherit; }
[data-testid="stMain"] :is([data-testid^="stBaseButton-secondary"], [data-testid="stBaseButton-segmented_control"],
  [data-testid="stBaseButton-pills"]):hover { border-color: var(--k-a) !important; color: var(--k-a) !important; background: var(--k-soft) !important; }
[data-testid="stMain"] :is([data-testid^="stBaseButton-primary"], [data-testid="stBaseButton-segmented_controlActive"],
  [data-testid="stBaseButton-pillsActive"]) { background: var(--k-fill) !important; border-color: var(--k-fill) !important; color: #fff !important; }
[data-testid="stMain"] :is([data-testid^="stBaseButton-primary"], [data-testid="stBaseButton-segmented_controlActive"],
  [data-testid="stBaseButton-pillsActive"]):hover { background: var(--k-fill-2) !important; border-color: var(--k-fill-2) !important; color: #fff !important; }
[data-testid="stMain"] :is([data-testid^="stBaseButton-secondary"], [data-testid^="stBaseButton-primary"]):active { transform: translateY(1px); }
[data-testid="stMain"] [data-testid^="stBaseButton-"]:disabled { opacity: .45; }

/* Segmented controls: one joined strip, the chosen segment filled */
[data-testid="stMain"] [data-baseweb="button-group"] { gap: 0 !important; width: fit-content; max-width: 100%; border-radius: var(--k-r); flex-wrap: wrap; }
[data-testid="stMain"] [data-baseweb="button-group"] > button { border-radius: 0 !important; margin: 0 0 0 -1px !important; position: relative;
    padding: .35rem .8rem !important; white-space: nowrap; }
[data-testid="stMain"] [data-baseweb="button-group"] > button:first-child { margin-left: 0 !important; border-radius: var(--k-r) 0 0 var(--k-r) !important; }
[data-testid="stMain"] [data-baseweb="button-group"] > button:last-child { border-radius: 0 var(--k-r) var(--k-r) 0 !important; }
[data-testid="stMain"] [data-baseweb="button-group"] > button:hover { z-index: 1; }
[data-testid="stMain"] [data-baseweb="button-group"] > [data-testid$="Active"] { z-index: 2; }

/* The focus ring: the same 2px ring, 2px off the edge, on every control (mouse click and keyboard alike) */
[data-testid="stMain"] [data-testid^="stBaseButton-"]:focus, [data-testid="stMain"] [data-testid^="stBaseButton-"]:focus-visible,
[data-testid="stMain"] [role="slider"]:focus-visible, [data-testid="stMain"] summary:focus-visible, [data-testid="stMain"] a:focus-visible {
    outline: 2px solid var(--k-a) !important; outline-offset: 2px; }
[data-testid="stMain"] [data-baseweb="button-group"] > button:focus, [data-testid="stMain"] [data-baseweb="button-group"] > button:focus-visible { outline: none !important; }
[data-testid="stMain"] [data-baseweb="button-group"]:focus-within { outline: 2px solid var(--k-a); outline-offset: 2px; }

/* Inputs */
[data-testid="stMain"] :is([data-testid="stNumberInputContainer"], [data-testid="stTextInput"] [data-baseweb="input"],
  [data-baseweb="select"] > div, [data-baseweb="textarea"]) {
    border-radius: var(--k-r) !important; border: 1px solid var(--k-line) !important; background: var(--k-panel) !important;
    min-height: 2.45rem; transition: border-color .12s ease; }
[data-testid="stMain"] :is([data-testid="stNumberInputContainer"], [data-baseweb="select"] > div):hover { border-color: var(--k-a) !important; }
[data-testid="stMain"] :is([data-testid="stNumberInputContainer"], [data-testid="stTextInput"] [data-baseweb="input"],
  [data-baseweb="select"] > div, [data-baseweb="textarea"]):focus-within {
    border-color: var(--k-a) !important; outline: 2px solid var(--k-a); outline-offset: 2px; box-shadow: none !important; }
[data-testid="stMain"] :is([data-baseweb="input"], [data-baseweb="base-input"], input, textarea) { background: transparent !important; }
[data-testid="stMain"] [data-testid="stNumberInputContainer"] [data-baseweb="input"] { border: none !important; }
[data-testid="stMain"] :is(input, textarea) { font-family: 'Red Hat Mono', Consolas, monospace; color: var(--k-ink); }
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul[role="listbox"] { border-radius: var(--k-r) !important; border: 1px solid var(--k-line); }
[data-baseweb="popover"] li[role="option"]:hover, [data-baseweb="popover"] li[aria-selected="true"] { background: var(--k-soft) !important; }
[data-testid="stMain"] [data-testid="stFileUploaderDropzone"] { border-radius: var(--k-r); border: 1px dashed var(--k-line); background: #fbfcfd; }
[data-testid="stMain"] [data-testid="stExpander"] details { border-radius: var(--k-r); border: 1px solid var(--k-line); background: var(--k-panel); }
[data-testid="stMain"] [data-testid="stExpander"] summary p { font-family: 'Chakra Petch', 'Segoe UI', sans-serif; font-weight: 600; letter-spacing: .03em; }

/* Labels and sliders */
[data-testid="stMain"] [data-testid="stWidgetLabel"]:not([data-testid="stCheckbox"] *) p { font-family: 'Red Hat Mono', Consolas, monospace; font-size: .74rem;
    font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: #4b5868; }
[data-testid="stMain"] [data-baseweb="slider"] [role="slider"] { background: var(--k-fill) !important; box-shadow: 0 0 0 3px var(--k-panel), 0 0 0 4px var(--k-line) !important; }
[data-testid="stMain"] [data-testid="stSliderThumbValue"] { font-family: 'Red Hat Mono', Consolas, monospace; font-weight: 600; color: var(--k-ink); }
[data-testid="stMain"] [data-testid="stAlert"] > div { border-radius: var(--k-r) !important; }
</style>"""


def inject_css() -> None:
    st.html(CSS + CONTROLS)


def icon(name: str) -> str:
    return f'<span class="msr" aria-hidden="true">{name}</span>'


def page(code: str, name: str, title: str, lead: str) -> None:
    st.html(f'<div class="ata"><b>{html.escape(code)}</b><span>{html.escape(name)}</span></div>'
            f'<h1 class="title">{html.escape(title)}</h1><p class="lead">{lead}</p>')


@contextmanager
def panel(no: str, title: str, pictogram: str, sub: str = "", main: bool = False):
    """A riveted panel under a black placard; the page's `main` panel gets the indigo instrument edge."""
    with st.container(border=True, key=f"main_{no.replace('.', '_').replace('-', '_')}" if main else None):
        st.html(f'<div class="plq"><span class="p"><span class="no">{html.escape(no)}</span><p class="t">{html.escape(title)}</p></span>'
                f'<span class="ic">{icon(pictogram)}</span></div>' + (f'<p class="sub">{sub}</p>' if sub else ""))
        yield


def tiles(items: list[tuple[str, str, str]], highlight: int | None = None) -> None:
    body = "".join(f'<div class="tile{" hl" if i == highlight else ""}"><p class="tile-l">{html.escape(l)}</p>'
                   f'<p class="tile-v">{v}</p><p class="tile-n">{n}</p></div>' for i, (l, v, n) in enumerate(items))
    st.html(f'<div class="tiles" style="--n:{len(items)}">{body}</div>')


def lane(name: str) -> str:
    return (f'<span class="lane" style="--c:{t.LANE[name]};--s:{t.LANE_SOFT[name]}">{icon(t.LANE_ICON[name])}{name}</span>')


def cycles(x: float, cap: int = 125) -> str:
    """Remaining cycles as shown everywhere: whole cycles, '125+' at the cap."""
    return f"{cap}+" if x >= cap - 0.5 else f"{x:.0f}"


def dial(pred: float, low: float, high: float, ground: int, plan: int, cap: int = 125) -> str:
    """A 270-degree instrument: lane zones (by the plan's safe-life lines), the 80% range as an indigo arc and the
    needle at the prediction."""
    deg = lambda v: 270 * min(max(v, 0), cap) / cap  # noqa: E731
    ticks = ""
    for v in (0, 25, 50, 75, 100, 125):
        a = math.radians(225 - deg(v))      # 0 at lower left, clockwise to lower right
        x, y = 50 + 31 * math.cos(a), 50 - 31 * math.sin(a)
        ticks += f'<span class="tk" style="left:{x:.1f}%;top:{y:.1f}%">{v}{"+" if v == cap else ""}</span>'
    return (f'<div class="dialwrap"><div class="dial" role="img" aria-label="predicted {cycles(pred)} cycles left, 80% range '
            f'{cycles(low)} to {cycles(high)}" style="--a:{deg(ground):.1f}deg;--b:{deg(plan):.1f}deg;--lo:{deg(low):.1f}deg;'
            f'--hi:{deg(high):.1f}deg;--r:{deg(pred) - 135:.1f}deg"><div class="rng"></div>{ticks}<div class="ndl"></div>'
            f'<div class="hub"></div><div class="rd"><b>{cycles(pred)}</b><span>cycles left</span></div></div>'
            f'<p class="dial-k"><span><i style="background:{t.ACCENT}"></i>prediction</span>'
            f'<span><i style="background:rgba(67,56,202,.35)"></i>80% range {cycles(low)} to {cycles(high)}</span>'
            f'<span>ring: lane lines at {ground} and {plan} cycles of safe life</span></p></div>')


def contributions(parts: list[tuple[str, str, float]], scale: float) -> str:
    """Rows of (code, name, cycles): bars right of centre add life, left take it away."""
    rows = ""
    for code, name, v in parts:
        w = min(abs(v) / scale, 1) * 50
        left = 50 if v >= 0 else 50 - w
        color = t.ACCENT if v >= 0 else t.LANE["Ground"]
        rows += (f'<span class="n"><b>{html.escape(code)}</b><span>{html.escape(name)}</span></span>'
                 f'<span class="tr"><i style="left:{left:.1f}%;width:{w:.1f}%;background:{color}"></i></span>'
                 f'<span class="v">{v:+.1f}</span>')
    return (f'<div class="cb">{rows}</div><p class="cb-k"><span>{icon("arrow_back")} less life (cycles)</span>'
            f'<span>more life (cycles) {icon("arrow_forward")}</span></p>')


def notes(*items: str) -> None:
    st.html(f'<div class="notes"><p class="h">{icon("edit_note")}TECH LOG</p><ul>' + "".join(f"<li>{i}</li>" for i in items) + "</ul></div>")


def takeaway(text: str) -> None:
    st.html(f'<div class="takeaway">{icon("lightbulb")}<div>{text}</div></div>')


def caution(text: str, pictogram: str = "warning") -> None:
    st.html(f'<div class="caution">{icon(pictogram)}<div>{text}</div></div>')


def footer() -> None:
    st.html(f'<div class="foot"><span>TurboLife · <b>Ahmad Hassan</b>, AI Engineer</span>'
            f'<a href="{NOTEBOOK_URL}" target="_blank">Notebook</a><a href="{CODE_URL}" target="_blank">App code</a>'
            f'<a href="{LEARN_URL}" target="_blank">Learn with Ahmad</a><span class="credit">{html.escape(DATA_CREDIT)}</span></div>')
