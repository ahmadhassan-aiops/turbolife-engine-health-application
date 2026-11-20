"""Every page renders without an exception, including the main interactions."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["views/fleet.py", "views/engine.py", "views/safety.py", "views/report.py", "views/method.py"]
ROOT = Path(__file__).resolve().parent.parent


def run(module: str) -> AppTest:
    script = f"""
import importlib
from hangar.ui import inject_css
inject_css()
importlib.import_module("{module.replace('/', '.').removesuffix('.py')}").render()
"""
    return AppTest.from_string(script, default_timeout=120).run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = run(page)
    assert not at.exception, at.exception


def test_fleet_policy_and_actuals():
    at = run("views/fleet.py")
    at.slider(key="f_ground").set_value(30).run()
    assert not at.exception
    assert any("engines truly within 30 cycles" in h.proto.body for h in at.get("html"))
    at.slider(key="f_plan").set_value(25).run()
    assert at.error
    at.slider(key="f_plan").set_value(60).run()
    at.toggle(key="f_actual").set_value(False).run()
    assert not at.exception


def test_engine_switch_and_sensor():
    at = run("views/engine.py")
    at.selectbox(key="e_unit").set_value(81).run()
    at.selectbox(key="e_sensor").set_value("s15").run()
    assert not at.exception
    assert any("cycles left" in h.proto.body for h in at.get("html"))


def test_engine_upload_mode_waits_for_a_file():
    at = run("views/engine.py")
    at.segmented_control(key="e_src").set_value("Upload a log").run()
    assert not at.exception
    assert at.info


def test_safety_lines():
    at = run("views/safety.py")
    at.slider(key="s_ground").set_value(40).run()
    assert not at.exception
    at.slider(key="s_plan").set_value(30).run()
    assert at.error
