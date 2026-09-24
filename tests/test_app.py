"""Smoke test: the Streamlit dashboard runs every tab without raising."""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_dashboard_renders_all_tabs_without_errors():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.tabs) == 8
    assert "PANDEMICA" in at.title[0].value


def test_dashboard_reacts_to_slider_changes():
    at = AppTest.from_file(APP, default_timeout=180).run()
    next(s for s in at.slider if s.label == "R0").set_value(0.8).run()
    assert not at.exception
    # with R0 < 1 the M1 metrics show almost nobody infected
    ever = next(m for m in at.metric if m.label == "Ever infected")
    assert ever.value.startswith("0.0")
