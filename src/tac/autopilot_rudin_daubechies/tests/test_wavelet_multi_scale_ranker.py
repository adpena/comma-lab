# SPDX-License-Identifier: MIT
"""Tests for wavelet multi-scale falling-rule ranker (Phase 5)."""
from __future__ import annotations

import pytest

from tac.autopilot_rudin_daubechies import (
    FallingRule,
    PredicateRef,
    ProxyPanel,
    WAVELET_NUM_SCALES_DEFAULT,
    WaveletMultiScaleFallingRuleListRanker,
)


def test_default_num_scales_is_4():
    assert WAVELET_NUM_SCALES_DEFAULT == 4


def test_construction_default_4_scales():
    r = WaveletMultiScaleFallingRuleListRanker()
    assert r.num_scales == 4
    # Each scale has its own (initially empty) rule list.
    for s in range(4):
        assert len(r.rule_list_at_scale(s).rules) == 0


def test_rejects_zero_scales():
    with pytest.raises(ValueError, match=">= 1"):
        WaveletMultiScaleFallingRuleListRanker(num_scales=0)


def test_rule_list_at_scale_out_of_range():
    r = WaveletMultiScaleFallingRuleListRanker(num_scales=2)
    with pytest.raises(IndexError):
        r.rule_list_at_scale(2)
    with pytest.raises(IndexError):
        r.rule_list_at_scale(-1)


def test_default_band_when_no_rules():
    r = WaveletMultiScaleFallingRuleListRanker()
    result = r.evaluate(ProxyPanel())
    # No rules at any scale -> all chains hit the default catch-all.
    for chain in result.chain_per_scale:
        assert chain.rule_id == "__default__"


def test_coarsest_gate_skips_finer_when_predicted_too_high():
    """If coarsest rule emits a band whose LOWER bound > promotion threshold,
    finer scales are not consulted."""
    r = WaveletMultiScaleFallingRuleListRanker(
        num_scales=3, promotion_threshold_score=0.30
    )
    # Coarsest rule emits a band [0.50, 0.70] -> well above the threshold.
    high_band_rule = FallingRule(
        name="reject_too_high",
        predicates=(PredicateRef.parse("seg_p0 > -1"),),
        predicted_score_low=0.50,
        predicted_score_high=0.70,
    )
    r.rule_list_at_scale(0).rules.append(high_band_rule)
    # Add a fine-scale rule that WOULD fire but should be gated out.
    fine_rule = FallingRule(
        name="fine_rule",
        predicates=(PredicateRef.parse("seg_p0 > -1"),),
        predicted_score_low=0.18,
        predicted_score_high=0.22,
    )
    r.rule_list_at_scale(2).rules.append(fine_rule)
    result = r.evaluate(ProxyPanel(seg_p0=0.005))
    # Coarsest gate failed.
    assert result.coarsest_gate_passed is False
    # Predicted band reflects the coarsest rule.
    assert result.consensus_low == 0.50


def test_coarsest_gate_pass_evaluates_finer_scales():
    r = WaveletMultiScaleFallingRuleListRanker(num_scales=3)
    # Coarsest emits a band overlapping medal threshold.
    coarse = FallingRule(
        name="medal_ok",
        predicates=(PredicateRef.parse("seg_p0 < 0.10"),),
        predicted_score_low=0.15,
        predicted_score_high=0.30,
    )
    r.rule_list_at_scale(0).rules.append(coarse)
    fine = FallingRule(
        name="fine",
        predicates=(PredicateRef.parse("seg_p0 < 0.10"),),
        predicted_score_low=0.18,
        predicted_score_high=0.22,
    )
    r.rule_list_at_scale(2).rules.append(fine)
    result = r.evaluate(ProxyPanel(seg_p0=0.005))
    assert result.coarsest_gate_passed is True
    # Finer rule tightens the band.
    assert result.consensus_low == 0.18
    assert result.consensus_high == 0.22


def test_disjoint_scale_bands_keep_coarser():
    """When coarse and fine bands don't overlap, keep the coarser band per
    Daubechies' canonical 'coarsest dominates on disagreement' rule."""
    r = WaveletMultiScaleFallingRuleListRanker(num_scales=2)
    coarse = FallingRule(
        name="coarse",
        predicates=(PredicateRef.parse("seg_p0 > -1"),),
        predicted_score_low=0.10,
        predicted_score_high=0.15,
    )
    r.rule_list_at_scale(0).rules.append(coarse)
    # Fine rule emits a band totally disjoint from the coarse one.
    fine = FallingRule(
        name="fine",
        predicates=(PredicateRef.parse("seg_p0 > -1"),),
        predicted_score_low=0.30,
        predicted_score_high=0.40,
    )
    r.rule_list_at_scale(1).rules.append(fine)
    result = r.evaluate(ProxyPanel(seg_p0=0.01))
    # Disjoint -> keep coarser band (the one set first).
    assert result.consensus_low == 0.10
    assert result.consensus_high == 0.15


def test_explain_includes_all_scales():
    r = WaveletMultiScaleFallingRuleListRanker(num_scales=4)
    result = r.evaluate(ProxyPanel())
    expl = result.explain()
    # Each scale's chain appears.
    for s in range(4):
        assert f"scale[{s}]" in expl


# ── Continual learning ───────────────────────────────────────────────────


def test_update_at_scale_records_only_targeted_scale():
    r = WaveletMultiScaleFallingRuleListRanker(num_scales=3)
    rule_scale_1 = FallingRule(
        name="r_at_1",
        predicates=(PredicateRef.parse("seg_p0 > -1"),),
        predicted_score_low=0.18,
        predicted_score_high=0.22,
        rule_id="r1",
    )
    r.rule_list_at_scale(1).rules.append(rule_scale_1)
    chain = r.update_at_scale(1, 0.20, ProxyPanel(seg_p0=0.01))
    assert chain.rule_id == "r1"
    # Hit recorded at scale 1.
    assert r.rule_list_at_scale(1).hit_rate("r1") == 1.0
    # Scale 0 + 2 are untouched.
    assert r.rule_list_at_scale(0).hit_rate("r1") is None


def test_update_at_scale_invalid_index_raises():
    r = WaveletMultiScaleFallingRuleListRanker(num_scales=2)
    with pytest.raises(IndexError):
        r.update_at_scale(5, 0.20, ProxyPanel())


def test_update_all_scales_runs_each():
    r = WaveletMultiScaleFallingRuleListRanker(num_scales=3)
    chains = r.update_all_scales(0.20, ProxyPanel(seg_p0=0.01))
    assert len(chains) == 3
    # Each chain hits the default (no rules in any scale).
    for c in chains:
        assert c.rule_id == "__default__"


def test_construction_with_custom_promotion_threshold():
    r = WaveletMultiScaleFallingRuleListRanker(
        num_scales=2, promotion_threshold_score=0.40
    )
    assert r.promotion_threshold_score == 0.40
