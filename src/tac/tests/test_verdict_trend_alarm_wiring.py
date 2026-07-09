"""Wiring tests: the verdict-trend alarm downgrades the shadow controller's scalar
false-green, and surfaces a digest row (operator-catch 2026-07-09)."""
from __future__ import annotations

import json

from tac.witness_control import build_shadow_report
from tac.witness_control.shadow_controller import (
    VERDICT_RISING_DECOUPLING,
    RunInputs,
    _classify,
)

# Reuse the verbatim LIVE fixture from the sibling test module.
from tac.tests.test_verdict_trend_alarm import LIVE_UNIFY_TAU, MOD32CAP_CE


def _inputs(verdicts):
    return RunInputs(run_dir=__import__("pathlib").Path("."), verdicts=verdicts,
                     stage_rows={"transitions": [], "closed_loop": []}, flags={})


def test_classify_downgrades_converging_to_rising_decoupling_on_live():
    """The exact false-green fix: scalar says 'converging', overlay forbids it."""
    out = _classify(_inputs(LIVE_UNIFY_TAU))
    assert out is not None
    # the scalar monitor's raw verdict is preserved for transparency...
    assert out.get("scalar_classification") == "converging"
    # ...but the load-bearing classification is downgraded (CONVERGING forbidden).
    assert out["classification"] == VERDICT_RISING_DECOUPLING
    assert out["classification"] != "converging"
    assert "verdict_trend_alarm" in out
    assert out["verdict_trend_alarm"]["classification"] == "TRAIN_VERDICT_DECOUPLING"


def test_classify_leaves_mod32cap_descending_untouched():
    out = _classify(_inputs(MOD32CAP_CE))
    assert out is not None
    # descending baseline: overlay must NOT fire, scalar class stands (converging).
    assert out["classification"] != VERDICT_RISING_DECOUPLING
    assert out["verdict_trend_alarm"]["classification"] == "NO_ALARM"


def test_shadow_report_emits_investigate_recommendation_on_live():
    """A fired decoupling must produce a recommendation (not '(none identifiable)')."""
    report = build_shadow_report(_inputs(LIVE_UNIFY_TAU))
    assert report.classification["classification"] == VERDICT_RISING_DECOUPLING
    actions = [r["action"] for r in report.recommendations]
    assert "INVESTIGATE_VERDICT_RISING_DECOUPLING" in actions
    rec = next(r for r in report.recommendations
               if r["action"] == "INVESTIGATE_VERDICT_RISING_DECOUPLING")
    assert rec["predicted_dS"] == 0.0                # advisory investigate, never a fake ΔS
    assert "Lane" in rec["rationale"]                # names the worst-rising class


def test_digest_section_fires_on_live_silent_on_mod32cap(tmp_path):
    """tools/costate_digest.section_verdict_trend: line for the live run, None for mod32cap."""
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    if str(repo / "tools") not in sys.path:
        sys.path.insert(0, str(repo / "tools"))
    import costate_digest as cd

    def _run_dir(name, verdicts):
        d = tmp_path / name
        d.mkdir()
        (d / "run.log").write_text(
            "\n".join(json.dumps(v) for v in verdicts) + "\n", encoding="utf-8")
        return d

    live_line, live_data = cd.section_verdict_trend(_run_dir("live", LIVE_UNIFY_TAU))
    assert live_line is not None
    assert "DECOUPLING" in live_line
    assert live_data["stage"] == "confound_alarm"

    mod_line, mod_data = cd.section_verdict_trend(_run_dir("mod", MOD32CAP_CE))
    assert mod_line is None and mod_data is None      # silent -> omitted (fail-open)

    # None run_dir -> soft None (never crashes a session start).
    assert cd.section_verdict_trend(None) == (None, None)
