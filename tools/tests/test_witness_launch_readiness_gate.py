# SPDX-License-Identifier: MIT
"""Tests for the witness launch-readiness gate (pure decision surface)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_MOD = _REPO / "tools" / "witness_launch_readiness_gate.py"
_spec = importlib.util.spec_from_file_location("wlrg", _MOD)
wlrg = importlib.util.module_from_spec(_spec)
sys.modules["wlrg"] = wlrg  # dataclass field-type resolution needs the module registered
_spec.loader.exec_module(wlrg)

RungExpectation = wlrg.RungExpectation
decide_readiness = wlrg.decide_readiness


def _exp(name, rel, tokens=()):
    return RungExpectation(name=name, rel_sig_pct=rel, flag_tokens=tuple(tokens))


def test_high_ev_missing_refuses():
    exps = [_exp("HorizonWeightedMargin", 43.8, ("--horizon-weighted-margin",))]
    v = decide_readiness(exps, set(), "", {})
    assert not v.proceed
    assert any("HorizonWeightedMargin" in m for m in v.blocking_missing)


def test_high_ev_included_proceeds():
    exps = [_exp("HorizonWeightedMargin", 43.8, ("--horizon-weighted-margin",))]
    flags = {"--horizon-weighted-margin"}
    v = decide_readiness(exps, flags, "--horizon-weighted-margin 1.0", {})
    assert v.proceed
    assert "HorizonWeightedMargin" in v.included


def test_high_ev_deferred_with_reason_proceeds():
    exps = [_exp("StepNativeActivation", 31.6, ("--activation step_native",))]
    v = decide_readiness(exps, set(), "", {"StepNativeActivation": "measuring hosc arm first, own A/B"})
    assert v.proceed
    assert any("StepNativeActivation" in d for d in v.acknowledged_deferred)


def test_placeholder_defer_reason_still_refuses():
    exps = [_exp("StepNativeActivation", 31.6, ("--x",))]
    v = decide_readiness(exps, set(), "", {"StepNativeActivation": "TBD"})
    assert not v.proceed


def test_low_ev_missing_is_advisory_not_blocking():
    exps = [_exp("MarginBandSatisficing", 2.0, ("--margin-band-satisficing",))]
    v = decide_readiness(exps, set(), "", {})
    assert v.proceed  # below the 10% blocking bar
    assert any("MarginBandSatisficing" in m for m in v.advisory_missing)


def test_none_rel_sig_is_advisory():
    exps = [_exp("SomeRung", None, ("--some-rung",))]
    v = decide_readiness(exps, set(), "", {})
    assert v.proceed
    assert any("SomeRung" in m for m in v.advisory_missing)


def test_empty_expectations_fail_open_proceeds():
    v = decide_readiness([], set(), "", {})
    assert v.proceed
    assert v.notes


def test_fallback_kebab_probe_detects_flag():
    # no declared tokens -> name-derived --horizon-weighted-margin probe
    exps = [_exp("HorizonWeightedMargin", 43.8, ())]
    v = decide_readiness(exps, {"--horizon-weighted-margin-weight"}, "", {})
    assert v.proceed
    assert "HorizonWeightedMargin" in v.included


def test_fallback_kebab_probe_absent_refuses():
    exps = [_exp("HorizonWeightedMargin", 43.8, ())]
    v = decide_readiness(exps, {"--activation"}, "--activation hosc", {})
    assert not v.proceed


def test_mixed_included_and_missing():
    exps = [
        _exp("A", 50.0, ("--a",)),
        _exp("B", 40.0, ("--b",)),
    ]
    v = decide_readiness(exps, {"--a"}, "--a", {})
    assert not v.proceed
    assert "A" in v.included
    assert any("B" in m for m in v.blocking_missing)


def test_render_refuse_contains_fix_hint():
    exps = [_exp("HorizonWeightedMargin", 43.8, ("--x",))]
    v = decide_readiness(exps, set(), "", {})
    txt = v.render()
    assert "REFUSE" in txt
    assert "LAUNCH_READINESS_DEFER" in txt


def test_real_stopped_config_would_be_refused():
    """Regression: the exact 2026-07-10 naive-launch incident config."""
    launch = _REPO / "experiments/results/__v752_drystart_final__/dry_start_resume/launch.sh"
    table = _REPO / ".omx/research/default_off_decision_table_20260710.jsonl"
    if not (launch.is_file() and table.is_file()):
        return  # artifacts not present in this checkout; skip silently
    v = wlrg.assess_launch_readiness(launch, table)
    # The stopped config skipped HorizonWeightedMargin + StepNativeActivation.
    assert not v.proceed
    names = " ".join(v.blocking_missing)
    assert "HorizonWeightedMargin" in names
