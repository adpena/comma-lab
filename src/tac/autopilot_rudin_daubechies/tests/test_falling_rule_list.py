# SPDX-License-Identifier: MIT
"""Tests for falling-rule-list ranking surface (Phase 2)."""
from __future__ import annotations

import pytest

from tac.autopilot_rudin_daubechies import (
    FallingRule,
    FallingRuleList,
    PredicateRef,
    ProxyPanel,
    RuleChain,
)


# ── PredicateRef ──────────────────────────────────────────────────────────


def test_predicate_ref_parse_basic():
    p = PredicateRef.parse("seg_p0 < 0.01")
    assert p.lhs == "seg_p0"
    assert p.op == "<"
    assert p.rhs == 0.01


def test_predicate_ref_parse_all_operators():
    for op in ("<", "<=", ">", ">=", "==", "!="):
        p = PredicateRef.parse(f"rate_p0 {op} 0.005")
        assert p.op == op


def test_predicate_ref_parse_negative_rhs():
    p = PredicateRef.parse("seg_p2 >= -0.001")
    assert p.rhs == -0.001


def test_predicate_ref_parse_scientific():
    p = PredicateRef.parse("pose_p0 < 1e-4")
    assert p.rhs == pytest.approx(1e-4)


def test_predicate_ref_parse_rejects_garbage():
    with pytest.raises(ValueError, match="unparseable"):
        PredicateRef.parse("not a predicate")


def test_predicate_ref_evaluate_against_proxy_panel():
    p = PredicateRef.parse("seg_p0 < 0.01")
    assert p.evaluate(ProxyPanel(seg_p0=0.005)) is True
    assert p.evaluate(ProxyPanel(seg_p0=0.02)) is False


def test_predicate_ref_evaluate_missing_proxy_returns_false():
    p = PredicateRef.parse("seg_p0 < 0.01")
    # Missing proxy resolves to None -> predicate cannot fire.
    assert p.evaluate(ProxyPanel()) is False


def test_predicate_ref_evaluate_against_metadata_dotted():
    p = PredicateRef.parse("substrate.class > 0.5")
    md = {"substrate": {"class": 0.7}}
    assert p.evaluate(ProxyPanel(), md) is True


def test_predicate_ref_evaluate_metadata_missing_returns_false():
    p = PredicateRef.parse("foo.bar < 1")
    assert p.evaluate(ProxyPanel(), None) is False
    assert p.evaluate(ProxyPanel(), {}) is False


# ── FallingRule ───────────────────────────────────────────────────────────


def test_falling_rule_rejects_inverted_band():
    with pytest.raises(ValueError, match="low.*>.*high"):
        FallingRule(
            name="bad",
            predicates=(PredicateRef.parse("seg_p0 < 0.01"),),
            predicted_score_low=0.5,
            predicted_score_high=0.1,
        )


def test_falling_rule_fires_on_conjunction():
    rule = FallingRule(
        name="medal_band_candidate",
        predicates=(
            PredicateRef.parse("seg_p0 < 0.01"),
            PredicateRef.parse("rate_p0 < 0.10"),
        ),
        predicted_score_low=0.18,
        predicted_score_high=0.22,
    )
    assert rule.fires_on(ProxyPanel(seg_p0=0.005, rate_p0=0.05)) is True
    assert rule.fires_on(ProxyPanel(seg_p0=0.005, rate_p0=0.50)) is False


def test_falling_rule_predicted_band_returns_tuple():
    rule = FallingRule(
        name="r1",
        predicates=(),
        predicted_score_low=0.1,
        predicted_score_high=0.2,
    )
    assert rule.predicted_band() == (0.1, 0.2)


# ── FallingRuleList evaluate ──────────────────────────────────────────────


def test_falling_rule_list_first_match_wins():
    rule_a = FallingRule(
        name="strict",
        predicates=(PredicateRef.parse("seg_p0 < 0.005"),),
        predicted_score_low=0.18,
        predicted_score_high=0.20,
        rule_id="r_strict",
    )
    rule_b = FallingRule(
        name="loose",
        predicates=(PredicateRef.parse("seg_p0 < 0.10"),),
        predicted_score_low=0.20,
        predicted_score_high=0.30,
        rule_id="r_loose",
    )
    fl = FallingRuleList(rules=[rule_a, rule_b])
    chain = fl.evaluate(ProxyPanel(seg_p0=0.003))
    assert chain.rule_name == "strict"
    assert chain.predicted_score_low == 0.18


def test_falling_rule_list_falls_through_to_default():
    fl = FallingRuleList(default_band_low=0.10, default_band_high=0.40)
    chain = fl.evaluate(ProxyPanel())
    assert chain.rule_id == "__default__"
    assert chain.predicted_band() == (0.10, 0.40)


def test_falling_rule_list_chain_explain_is_readable():
    rule = FallingRule(
        name="medal_band",
        predicates=(PredicateRef.parse("seg_p0 < 0.01"),),
        predicted_score_low=0.18,
        predicted_score_high=0.22,
        rule_id="r1",
    )
    fl = FallingRuleList(rules=[rule])
    chain = fl.evaluate(ProxyPanel(seg_p0=0.005))
    expl = chain.explain()
    assert "fired" in expl
    assert "seg_p0" in expl
    assert "[0.18, 0.22]" in expl


# ── FallingRuleList continual learning ────────────────────────────────────


def test_falling_rule_list_add_candidate_rule_appends():
    fl = FallingRuleList()
    rule = FallingRule(
        name="new_idea",
        predicates=(),
        predicted_score_low=0.20,
        predicted_score_high=0.25,
        rule_id="r_new",
    )
    fl.add_candidate_rule(rule)
    assert len(fl.rules) == 1


def test_falling_rule_list_add_candidate_rule_rejects_duplicate_id():
    fl = FallingRuleList()
    rule = FallingRule(
        name="r1",
        predicates=(),
        predicted_score_low=0.20,
        predicted_score_high=0.25,
        rule_id="dup",
    )
    fl.add_candidate_rule(rule)
    with pytest.raises(ValueError, match="already in list"):
        fl.add_candidate_rule(rule)


def test_falling_rule_list_update_records_hit_when_within_band():
    rule = FallingRule(
        name="medal",
        predicates=(PredicateRef.parse("seg_p0 < 0.01"),),
        predicted_score_low=0.18,
        predicted_score_high=0.22,
        rule_id="medal",
    )
    fl = FallingRuleList(rules=[rule])
    fl.update_from_anchor(0.20, ProxyPanel(seg_p0=0.005))
    assert fl.hit_rate("medal") == 1.0


def test_falling_rule_list_update_records_miss_when_outside_band():
    rule = FallingRule(
        name="medal",
        predicates=(PredicateRef.parse("seg_p0 < 0.01"),),
        predicted_score_low=0.18,
        predicted_score_high=0.22,
        rule_id="medal",
    )
    fl = FallingRuleList(rules=[rule])
    fl.update_from_anchor(0.30, ProxyPanel(seg_p0=0.005))
    assert fl.hit_rate("medal") == 0.0


def test_falling_rule_list_prune_drops_low_hitrate_rule():
    rule = FallingRule(
        name="bad_rule",
        predicates=(PredicateRef.parse("seg_p0 < 1.0"),),
        predicted_score_low=0.50,
        predicted_score_high=0.60,
        rule_id="bad",
    )
    fl = FallingRuleList(rules=[rule])
    # Several anchors all OUTSIDE the predicted band.
    for _ in range(5):
        fl.update_from_anchor(0.20, ProxyPanel(seg_p0=0.01))
    pruned = fl.prune_ineffective_rule("bad", hit_rate_min=0.5)
    assert pruned is True
    assert len(fl.rules) == 0


def test_falling_rule_list_prune_keeps_good_rule():
    rule = FallingRule(
        name="good_rule",
        predicates=(PredicateRef.parse("seg_p0 < 1.0"),),
        predicted_score_low=0.18,
        predicted_score_high=0.22,
        rule_id="good",
    )
    fl = FallingRuleList(rules=[rule])
    for _ in range(5):
        fl.update_from_anchor(0.20, ProxyPanel(seg_p0=0.01))
    pruned = fl.prune_ineffective_rule("good", hit_rate_min=0.5)
    assert pruned is False
    assert len(fl.rules) == 1


def test_falling_rule_list_prune_unknown_rule_returns_false():
    fl = FallingRuleList()
    assert fl.prune_ineffective_rule("nonexistent") is False


def test_falling_rule_list_prune_empirical_band_mismatch():
    """Rule predicts narrow band; empirical mean is far outside; should drop."""
    rule = FallingRule(
        name="r",
        predicates=(PredicateRef.parse("seg_p0 < 1.0"),),
        predicted_score_low=0.10,
        predicted_score_high=0.12,
        rule_id="r",
    )
    fl = FallingRuleList(rules=[rule])
    # Multiple anchors well outside the band.
    for _ in range(5):
        fl.update_from_anchor(0.50, ProxyPanel(seg_p0=0.01))
    # Empirical mean ~ 0.50; band is [0.10, 0.12]; tolerance 0.05 -> outside.
    pruned = fl.prune_ineffective_rule("r", hit_rate_min=0.0, empirical_band_tolerance=0.05)
    assert pruned is True


def test_falling_rule_list_empirical_mean_returns_none_when_no_data():
    fl = FallingRuleList()
    assert fl.empirical_mean("anything") is None


def test_falling_rule_list_hit_rate_returns_none_when_no_data():
    fl = FallingRuleList()
    assert fl.hit_rate("anything") is None
