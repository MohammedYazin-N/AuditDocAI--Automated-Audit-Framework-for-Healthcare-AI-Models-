"""Drives the real Streamlit app headlessly (streamlit.testing AppTest). Run: python -m pytest -q tests/test_app_ui.py"""
import sys
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
T = 180


def fresh():
    at = AppTest.from_file(APP, default_timeout=T); at.run(); return at


def click(at, label):
    [b for b in at.button if b.label == label][0].click().run()


def no_exceptions(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not [e for e in at.error if "Something went wrong" in e.value], [e.value for e in at.error]


def load_demo(at, idx=0):
    at.selectbox(key="demo_pick").select_index(idx).run()
    click(at, "Load demo dataset + model")


def go(at, page):
    at.session_state["nav"] = page; at.run()


def test_home_renders_and_scenario_A_no_model():
    at = fresh(); no_exceptions(at)
    go(at, "Data Quality Audit"); assert "Upload a dataset" in " ".join(i.value for i in at.info)
    go(at, "Fairness Audit"); assert at.warning and "cannot run yet" in at.warning[0].value


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_full_flow_on_each_demo(idx):
    at = fresh(); load_demo(at, idx); no_exceptions(at)
    for pg in ["Data Quality Audit", "Model Upload"]:
        go(at, pg); no_exceptions(at)
    go(at, "Leakage & Performance"); no_exceptions(at)
    click(at, "Run leakage audit"); no_exceptions(at)
    click(at, "Run performance evaluation"); no_exceptions(at)
    assert at.session_state["res_perf"] is not None and at.session_state["res_leak"] is not None
    go(at, "Fairness Audit"); no_exceptions(at)
    if at.session_state["demo"]:
        click(at, "Run fairness audit"); no_exceptions(at); assert at.session_state["res_fair"] is not None
    else:
        assert "cannot run yet" in at.warning[0].value          # breast-cancer demo has no demographic column
    go(at, "Model Explainability"); no_exceptions(at)
    click(at, "Run explainability"); no_exceptions(at); assert at.session_state["res_expl"] is not None
    go(at, "Final Audit Report"); no_exceptions(at)
    click(at, "Generate report"); no_exceptions(at)
    assert at.session_state["pdf_bytes"][:4] == b"%PDF"


def test_incompatible_model_is_explained_not_crashed():
    at = fresh(); load_demo(at, 1)                      # heart data loaded
    from audit import model_io
    wrong, _ = model_io.load_model((ROOT / "demo_assets/synthetic_readmission_model.joblib").read_bytes(), "w.joblib")
    at.session_state["model"], at.session_state["mi"] = wrong, model_io.inspect_model(wrong)
    at.session_state["compat"] = None
    go(at, "Model Upload"); no_exceptions(at)
    assert any("missing" in e.value.lower() for e in at.error)
    go(at, "Leakage & Performance"); no_exceptions(at)
    assert at.session_state["res_perf"] is None
