# SPDX-License-Identifier: MIT
"""Tests for GOSDT dispatcher + falling-rule whiteboard (Phase 6)."""
from __future__ import annotations

import pytest

from tac.autopilot_rudin_daubechies import (
    DispatchDecision,
    FallingRule,
    GOSDTDispatcher,
    PredicateRef,
    ProxyPanel,
    WhiteboardRule,
)


def test_default_construction():
    d = GOSDTDispatcher()
    assert d.max_depth == 4
    assert d.min_hit_rate_for_promotion == 0.50
    assert len(d.canonical_rules.rules) == 0
    assert len(d.whiteboard) == 0


def test_decide_default_when_no_rules():
    d = GOSDTDispatcher()
    decision = d.decide(ProxyPanel())
    # Default catch-all band [0.10, 0.30]; medal threshold 0.20; straddles.
    assert isinstance(decision, DispatchDecision)
    # 0.30 > 0.20 means "predicted band straddles medal threshold" -> review.
    assert decision.action == "REQUEST_OPERATOR_REVIEW"


def test_decide_dispatch_when_predicted_high_below_medal():
    d = GOSDTDispatcher()
    # Add a rule predicting band [0.10, 0.18] (well below 0.20 medal threshold).
    rule = FallingRule(
        name="dispatch_me",
        predicates=(PredicateRef.parse("seg_p0 < 0.10"),),
        predicted_score_low=0.10,
        predicted_score_high=0.18,
    )
    d.canonical_rules.add_candidate_rule(rule)
    decision = d.decide(ProxyPanel(seg_p0=0.005))
    assert decision.action == "DISPATCH"


def test_decide_refuse_when_predicted_low_above_ceiling():
    d = GOSDTDispatcher()
    rule = FallingRule(
        name="refuse_me",
        predicates=(PredicateRef.parse("seg_p0 < 0.10"),),
        predicted_score_low=0.60,
        predicted_score_high=0.80,
    )
    d.canonical_rules.add_candidate_rule(rule)
    decision = d.decide(ProxyPanel(seg_p0=0.005))
    assert decision.action == "REFUSE"


def test_decide_explain_is_readable():
    d = GOSDTDispatcher()
    decision = d.decide(ProxyPanel(seg_p0=0.005))
    expl = decision.explain()
    assert "GOSDT decision_path" in expl
    assert "predicted band" in expl


def test_decide_includes_metadata_in_path():
    d = GOSDTDispatcher()
    decision = d.decide(
        ProxyPanel(seg_p0=0.005),
        metadata={"cost_band": "smoke", "substrate_class": "score_aware"},
    )
    assert "cost_band==smoke" in decision.decision_path
    assert "substrate_class==score_aware" in decision.decision_path


# ── Whiteboard ────────────────────────────────────────────────────────────


def test_propose_candidate_rule_appends_to_whiteboard():
    d = GOSDTDispatcher()
    rule = FallingRule(
        name="new_idea",
        predicates=(),
        predicted_score_low=0.20,
        predicted_score_high=0.25,
        rule_id="r_new",
    )
    wb = WhiteboardRule(
        rule_id="wb1",
        proposed_by="wunderkind",
        candidate_rule=rule,
    )
    d.propose_candidate_rule(wb)
    assert len(d.whiteboard) == 1
    assert d.whiteboard[0].proposed_by == "wunderkind"


def test_propose_rejects_duplicate_id():
    d = GOSDTDispatcher()
    rule = FallingRule(
        name="r",
        predicates=(),
        predicted_score_low=0.20,
        predicted_score_high=0.25,
        rule_id="r",
    )
    wb = WhiteboardRule(rule_id="wb1", proposed_by="x", candidate_rule=rule)
    d.propose_candidate_rule(wb)
    with pytest.raises(ValueError, match="already exists"):
        d.propose_candidate_rule(wb)


def test_promote_whiteboard_rule_moves_to_canonical():
    d = GOSDTDispatcher()
    rule = FallingRule(
        name="promote_me",
        predicates=(PredicateRef.parse("seg_p0 < 0.01"),),
        predicted_score_low=0.18,
        predicted_score_high=0.22,
        rule_id="promote",
    )
    wb = WhiteboardRule(
        rule_id="wb1", proposed_by="wunderkind", candidate_rule=rule
    )
    d.propose_candidate_rule(wb)
    promoted = d.promote_whiteboard_rule("wb1")
    assert promoted is True
    assert len(d.whiteboard) == 0
    assert len(d.canonical_rules.rules) == 1


def test_promote_whiteboard_rule_unknown_id_returns_false():
    d = GOSDTDispatcher()
    assert d.promote_whiteboard_rule("nonexistent") is False


def test_prune_whiteboard_drops_low_hit_rate():
    d = GOSDTDispatcher()
    rule = FallingRule(
        name="bad",
        predicates=(),
        predicted_score_low=0.50,
        predicted_score_high=0.60,
        rule_id="bad",
    )
    wb = WhiteboardRule(
        rule_id="wb_bad",
        proposed_by="x",
        candidate_rule=rule,
        empirical_hit_count=1,
        empirical_miss_count=9,
    )
    d.propose_candidate_rule(wb)
    pruned = d.prune_whiteboard(hit_rate_min=0.5)
    assert "wb_bad" in pruned
    assert len(d.whiteboard) == 0


def test_prune_whiteboard_keeps_no_evidence():
    d = GOSDTDispatcher()
    rule = FallingRule(
        name="r",
        predicates=(),
        predicted_score_low=0.20,
        predicted_score_high=0.25,
        rule_id="r",
    )
    wb = WhiteboardRule(rule_id="wb_unknown", proposed_by="x", candidate_rule=rule)
    d.propose_candidate_rule(wb)
    pruned = d.prune_whiteboard()
    # No evidence -> keep (don't punish lack of data).
    assert pruned == []
    assert len(d.whiteboard) == 1


def test_update_whiteboard_outcome_records_hit():
    d = GOSDTDispatcher()
    rule = FallingRule(
        name="r",
        predicates=(),
        predicted_score_low=0.18,
        predicted_score_high=0.22,
        rule_id="r",
    )
    wb = WhiteboardRule(rule_id="wb1", proposed_by="x", candidate_rule=rule)
    d.propose_candidate_rule(wb)
    found = d.update_whiteboard_outcome("wb1", 0.20, ProxyPanel())
    assert found is True
    assert d.whiteboard[0].empirical_hit_count == 1
    assert d.whiteboard[0].empirical_miss_count == 0


def test_update_whiteboard_outcome_records_miss():
    d = GOSDTDispatcher()
    rule = FallingRule(
        name="r",
        predicates=(),
        predicted_score_low=0.18,
        predicted_score_high=0.22,
        rule_id="r",
    )
    wb = WhiteboardRule(rule_id="wb1", proposed_by="x", candidate_rule=rule)
    d.propose_candidate_rule(wb)
    d.update_whiteboard_outcome("wb1", 0.50, ProxyPanel())
    assert d.whiteboard[0].empirical_miss_count == 1


def test_update_whiteboard_outcome_unknown_id_returns_false():
    d = GOSDTDispatcher()
    assert d.update_whiteboard_outcome("nonexistent", 0.20, ProxyPanel()) is False


def test_update_from_dispatch_outcome_records_path():
    d = GOSDTDispatcher()
    d.update_from_dispatch_outcome(("substrate==z3", "cost==smoke"), 0.18)
    d.update_from_dispatch_outcome(("substrate==z3", "cost==smoke"), 0.40)
    d.update_from_dispatch_outcome(("substrate==z3", "cost==smoke"), 0.45)
    # 2 of 3 latest are >= 0.30 -> _has_recent_failures True
    assert d._has_recent_failures(("substrate==z3", "cost==smoke")) is True


def test_update_from_dispatch_outcome_no_failures_returns_false():
    d = GOSDTDispatcher()
    d.update_from_dispatch_outcome(("path",), 0.18)
    d.update_from_dispatch_outcome(("path",), 0.20)
    assert d._has_recent_failures(("path",)) is False


def test_decide_defers_after_recent_path_failures():
    d = GOSDTDispatcher()
    rule = FallingRule(
        name="r",
        predicates=(),
        predicted_score_low=0.20,
        predicted_score_high=0.30,
    )
    d.canonical_rules.add_candidate_rule(rule)
    # Build path failures.
    decision_first = d.decide(
        ProxyPanel(),
        metadata={"cost_band": "smoke", "substrate_class": "z3"},
    )
    # Manually register failures at the same path.
    for s in (0.40, 0.50, 0.45):
        d.update_from_dispatch_outcome(decision_first.decision_path, s)
    decision_second = d.decide(
        ProxyPanel(),
        metadata={"cost_band": "smoke", "substrate_class": "z3"},
    )
    assert decision_second.action == "DEFER"


def test_whiteboard_rule_hit_rate_property():
    rule = FallingRule(
        name="r",
        predicates=(),
        predicted_score_low=0.20,
        predicted_score_high=0.25,
        rule_id="r",
    )
    wb = WhiteboardRule(
        rule_id="wb",
        proposed_by="x",
        candidate_rule=rule,
        empirical_hit_count=3,
        empirical_miss_count=1,
    )
    assert wb.hit_rate == 0.75


def test_whiteboard_rule_hit_rate_none_when_no_data():
    rule = FallingRule(
        name="r",
        predicates=(),
        predicted_score_low=0.2,
        predicted_score_high=0.25,
        rule_id="r",
    )
    wb = WhiteboardRule(rule_id="wb", proposed_by="x", candidate_rule=rule)
    assert wb.hit_rate is None
