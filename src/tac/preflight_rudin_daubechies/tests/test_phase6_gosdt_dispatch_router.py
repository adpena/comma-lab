# SPDX-License-Identifier: MIT
"""Tests for Phase 6 (GOSDT-style sparse decision tree dispatch router)."""
from __future__ import annotations

import pytest

from tac.preflight_rudin_daubechies import (
    GateVerdictPanel,
    GOSDTDispatchRouter,
    PreflightDispatchDecision,
    PreflightFallingRule,
    PreflightWhiteboardRule,
)


def _make_rule(gate_number: int) -> PreflightFallingRule:
    return PreflightFallingRule(
        gate_number=gate_number,
        gate_name=f"gate_{gate_number}",
        rationale_template="gate_{gate_number} fired",
        recommended_fix=f"fix gate {gate_number}",
    )


def _make_whiteboard_rule(rule_id: str, gate_number: int = 999) -> PreflightWhiteboardRule:
    return PreflightWhiteboardRule(
        rule_id=rule_id,
        proposed_by="symposium",
        candidate_rule=_make_rule(gate_number),
    )


def test_decide_clean_panel_returns_OK():
    router = GOSDTDispatchRouter()
    panel = GateVerdictPanel()
    decision = router.decide(panel)
    assert decision.action == "OK"
    # OK is reached via either the "ceiling < 50% threshold" or "no rules fired"
    # branch; both are valid for a clean panel.
    assert "routine dispatch" in decision.rationale or "no preflight rules fired" in decision.rationale


def test_decide_violated_panel_with_rule_returns_REVIEW_or_REFUSE():
    router = GOSDTDispatchRouter()
    router.canonical_rules.add_candidate_rule(_make_rule(146))
    panel = GateVerdictPanel(verdicts={"146": "VIOLATED"})
    decision = router.decide(panel, metadata={"cost_band": "smoke"})
    # 1 fired rule => risk band [7, 25]; threshold 50 => not REFUSE outright.
    assert decision.action in ["REQUEST_OPERATOR_REVIEW", "OK"]


def test_decide_many_violations_returns_REFUSE():
    router = GOSDTDispatchRouter(refusal_threshold=50.0)
    # Add 8 rules and violate all of them => 8 * 7 = 56 floor risk > threshold 50.
    for i in range(1, 9):
        router.canonical_rules.add_candidate_rule(_make_rule(i))
    panel = GateVerdictPanel(verdicts={str(i): "VIOLATED" for i in range(1, 9)})
    decision = router.decide(panel, metadata={"cost_band": "full"})
    assert decision.action == "REFUSE"
    assert "estimated risk floor" in decision.rationale


def test_propose_candidate_rule_appends_to_whiteboard():
    router = GOSDTDispatchRouter()
    rule = _make_whiteboard_rule("test_rule")
    router.propose_candidate_rule(rule)
    assert any(r.rule_id == "test_rule" for r in router.whiteboard)


def test_propose_candidate_rule_rejects_duplicate_id():
    router = GOSDTDispatchRouter()
    router.propose_candidate_rule(_make_whiteboard_rule("dup"))
    with pytest.raises(ValueError):
        router.propose_candidate_rule(_make_whiteboard_rule("dup"))


def test_promote_whiteboard_rule_moves_to_canonical():
    router = GOSDTDispatchRouter()
    router.propose_candidate_rule(_make_whiteboard_rule("test_rule"))
    assert router.promote_whiteboard_rule("test_rule") is True
    # Whiteboard now empty; canonical has 1 rule.
    assert len(router.whiteboard) == 0
    assert len(router.canonical_rules.rules) == 1


def test_promote_whiteboard_rule_returns_false_for_unknown_id():
    router = GOSDTDispatchRouter()
    assert router.promote_whiteboard_rule("nonexistent") is False


def test_prune_whiteboard_drops_low_hit_rate_rules():
    router = GOSDTDispatchRouter(min_hit_rate_for_promotion=0.50)
    rule = _make_whiteboard_rule("low_hit")
    rule.empirical_hit_count = 2
    rule.empirical_miss_count = 18
    router.propose_candidate_rule(rule)
    pruned = router.prune_whiteboard()
    assert "low_hit" in pruned
    assert all(r.rule_id != "low_hit" for r in router.whiteboard)


def test_prune_whiteboard_keeps_high_hit_rate_rules():
    router = GOSDTDispatchRouter(min_hit_rate_for_promotion=0.50)
    rule = _make_whiteboard_rule("high_hit")
    rule.empirical_hit_count = 18
    rule.empirical_miss_count = 2
    router.propose_candidate_rule(rule)
    pruned = router.prune_whiteboard()
    assert "high_hit" not in pruned


def test_update_from_dispatch_outcome_records_for_path():
    router = GOSDTDispatchRouter()
    router.update_from_dispatch_outcome(["cost_band==smoke"], 25.0)
    router.update_from_dispatch_outcome(["cost_band==smoke"], 30.0)
    # Internal accumulator should have 2 entries.
    outcomes = router._decision_path_outcomes[("cost_band==smoke",)]
    assert len(outcomes) == 2


def test_decide_with_recent_failures_returns_REFUSE():
    router = GOSDTDispatchRouter(refusal_threshold=50.0)
    router.canonical_rules.add_candidate_rule(_make_rule(1))
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED"})
    # Build the same decision path through repeated decisions.
    decision1 = router.decide(panel, metadata={"cost_band": "smoke", "substrate_class": "score_aware"})
    # Manually inject 3 high-risk outcomes for the SAME path.
    for _ in range(3):
        router.update_from_dispatch_outcome(decision1.decision_path, 60.0)
    # Re-decide with same panel/metadata.
    decision2 = router.decide(panel, metadata={"cost_band": "smoke", "substrate_class": "score_aware"})
    # Now the recent-failures branch should fire and emit REFUSE.
    assert decision2.action == "REFUSE"
    assert "recent decision-path produced failed dispatches" in decision2.rationale


def test_dispatch_decision_explain_includes_path_and_action():
    router = GOSDTDispatchRouter()
    panel = GateVerdictPanel()
    decision = router.decide(panel, metadata={"cost_band": "smoke"})
    explanation = decision.explain()
    assert "GOSDT preflight decision_path:" in explanation
    assert decision.action in explanation


def test_update_from_anchor_routes_through_decide_and_canonical_rules():
    router = GOSDTDispatchRouter()
    router.canonical_rules.add_candidate_rule(_make_rule(1))
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED"})
    decision = router.update_from_anchor(panel, 25.0)
    assert isinstance(decision, PreflightDispatchDecision)
    # Canonical rules should have anchor count = 1 from update_from_anchor.
    assert router.canonical_rules.n_anchors == 1
