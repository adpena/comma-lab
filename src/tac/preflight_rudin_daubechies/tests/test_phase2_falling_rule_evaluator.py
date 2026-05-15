# SPDX-License-Identifier: MIT
"""Tests for Phase 2 (falling-rule evaluator over preflight catalog gates)."""
from __future__ import annotations

import pytest

from tac.preflight_rudin_daubechies import (
    GateVerdictPanel,
    PreflightFallingRule,
    PreflightFallingRuleEvaluator,
)


def _make_rule(gate_number: int, fix: str = "use canonical helper") -> PreflightFallingRule:
    return PreflightFallingRule(
        gate_number=gate_number,
        gate_name=f"gate_{gate_number}",
        rationale_template="gate_{gate_number} fired due to violation pattern",
        recommended_fix=fix,
    )


def test_rule_evaluate_fires_on_violated():
    rule = _make_rule(1)
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED"})
    v = rule.evaluate(panel)
    assert v.rule_fired is True
    assert v.verdict == "VIOLATED"
    assert "gate_1" in v.rationale
    assert v.recommended_fix == "use canonical helper"


def test_rule_evaluate_does_not_fire_on_passed():
    rule = _make_rule(1)
    panel = GateVerdictPanel(verdicts={"1": "PASSED"})
    v = rule.evaluate(panel)
    assert v.rule_fired is False
    assert v.verdict == "PASSED"
    assert v.recommended_fix == ""


def test_rule_evaluate_does_not_fire_on_not_run():
    rule = _make_rule(1)
    panel = GateVerdictPanel()
    v = rule.evaluate(panel)
    assert v.rule_fired is False
    assert v.verdict == "NOT_RUN"


def test_rule_evaluate_does_not_fire_on_waived():
    rule = _make_rule(1)
    panel = GateVerdictPanel(verdicts={"1": "WAIVED"})
    v = rule.evaluate(panel)
    assert v.rule_fired is False
    assert v.verdict == "WAIVED"


def test_evaluator_first_match_wins():
    ev = PreflightFallingRuleEvaluator()
    ev.add_candidate_rule(_make_rule(1))
    ev.add_candidate_rule(_make_rule(5))
    ev.add_candidate_rule(_make_rule(7))
    panel = GateVerdictPanel(verdicts={"5": "VIOLATED", "7": "VIOLATED"})
    chain = ev.evaluate(panel)
    assert chain.first_fired_index == 1  # rule for gate 5 is at index 1


def test_evaluator_all_passed_no_first_match():
    ev = PreflightFallingRuleEvaluator()
    ev.add_candidate_rule(_make_rule(1))
    panel = GateVerdictPanel(verdicts={"1": "PASSED"})
    chain = ev.evaluate(panel)
    assert chain.first_fired_index is None
    assert chain.fired_rule_count() == 0


def test_evaluator_rejects_duplicate_gate():
    ev = PreflightFallingRuleEvaluator()
    ev.add_candidate_rule(_make_rule(1))
    with pytest.raises(ValueError):
        ev.add_candidate_rule(_make_rule(1))


def test_update_from_anchor_records_hits_and_resorts():
    ev = PreflightFallingRuleEvaluator()
    ev.add_candidate_rule(_make_rule(1))
    ev.add_candidate_rule(_make_rule(5))
    panel = GateVerdictPanel(verdicts={"5": "VIOLATED"})
    # 5 anchors, gate 5 always fires; gate 1 never fires.
    for _ in range(5):
        ev.update_from_anchor(panel)
    assert ev.n_anchors == 5
    # After re-sort: gate 5 should be at front (highest hit-rate 1.0).
    rules = ev.rules
    assert rules[0].gate_number == 5
    assert rules[0].empirical_hit_count == 5
    assert rules[1].gate_number == 1
    assert rules[1].empirical_hit_count == 0


def test_prune_ineffective_rule_drops_below_threshold():
    ev = PreflightFallingRuleEvaluator(prune_hit_rate_threshold=0.10)
    ev.add_candidate_rule(_make_rule(1))
    panel = GateVerdictPanel(verdicts={"1": "PASSED"})
    for _ in range(20):
        ev.update_from_anchor(panel)
    # gate 1 has hit_rate = 0/20 = 0.0 < 0.10
    assert ev.prune_ineffective_rule(1) is True
    assert all(r.gate_number != 1 for r in ev.rules)


def test_prune_ineffective_rule_keeps_above_threshold():
    ev = PreflightFallingRuleEvaluator(prune_hit_rate_threshold=0.10)
    ev.add_candidate_rule(_make_rule(1))
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED"})
    for _ in range(20):
        ev.update_from_anchor(panel)
    # gate 1 has hit_rate = 1.0 > 0.10
    assert ev.prune_ineffective_rule(1) is False


def test_prune_ineffective_rule_keeps_no_data():
    ev = PreflightFallingRuleEvaluator()
    ev.add_candidate_rule(_make_rule(1))
    # No data -> hit_rate is None -> keep
    assert ev.prune_ineffective_rule(1) is False


def test_chain_explain_includes_fired_marker():
    ev = PreflightFallingRuleEvaluator()
    ev.add_candidate_rule(_make_rule(1))
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED"})
    chain = ev.evaluate(panel)
    explanation = chain.explain()
    assert "first-match index=0" in explanation
    assert "FIRED" in explanation


def test_chain_explain_for_clean_panel():
    ev = PreflightFallingRuleEvaluator()
    ev.add_candidate_rule(_make_rule(1))
    panel = GateVerdictPanel()
    chain = ev.evaluate(panel)
    assert "no rule fired" in chain.explain()
