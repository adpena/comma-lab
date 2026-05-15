# SPDX-License-Identifier: MIT
"""Tests for Phase 5 (wavelet multi-scale preflight ranker)."""
from __future__ import annotations

import pytest

from tac.preflight_rudin_daubechies import (
    PREFLIGHT_WAVELET_NUM_SCALES_DEFAULT,
    GateVerdictPanel,
    PreflightFallingRule,
    WaveletMultiScalePreflightRanker,
)


def _make_rule(gate_number: int) -> PreflightFallingRule:
    return PreflightFallingRule(
        gate_number=gate_number,
        gate_name=f"gate_{gate_number}",
        rationale_template="gate_{gate_number} fired",
        recommended_fix=f"fix gate {gate_number}",
    )


def test_default_num_scales_is_4():
    ranker = WaveletMultiScalePreflightRanker()
    assert ranker.num_scales == 4
    assert PREFLIGHT_WAVELET_NUM_SCALES_DEFAULT == 4


def test_classify_gate_to_scale_canonical_mappings():
    ranker = WaveletMultiScalePreflightRanker()
    # Scale 0 (file existence)
    assert ranker.classify_gate_to_scale(146) == 0
    assert ranker.classify_gate_to_scale(109) == 0
    # Scale 1 (integration contract)
    assert ranker.classify_gate_to_scale(117) == 1
    assert ranker.classify_gate_to_scale(125) == 1
    # Scale 2 (byte mutation)
    assert ranker.classify_gate_to_scale(1) == 2
    assert ranker.classify_gate_to_scale(5) == 2
    # Scale 3 (per-substrate feature)
    assert ranker.classify_gate_to_scale(164) == 3
    assert ranker.classify_gate_to_scale(220) == 3


def test_classify_gate_default_is_scale_2():
    ranker = WaveletMultiScalePreflightRanker()
    # An unmapped gate defaults to byte-mutation scale.
    assert ranker.classify_gate_to_scale(99999) == 2


def test_reclassify_gate_to_scale_changes_classification():
    ranker = WaveletMultiScalePreflightRanker()
    ranker.reclassify_gate_to_scale(99999, 0)
    assert ranker.classify_gate_to_scale(99999) == 0


def test_reclassify_rejects_out_of_range_scale():
    ranker = WaveletMultiScalePreflightRanker()
    with pytest.raises(IndexError):
        ranker.reclassify_gate_to_scale(99999, 99)
    with pytest.raises(IndexError):
        ranker.reclassify_gate_to_scale(99999, -1)


def test_evaluator_at_scale_returns_distinct_per_scale():
    ranker = WaveletMultiScalePreflightRanker()
    a = ranker.evaluator_at_scale(0)
    b = ranker.evaluator_at_scale(1)
    assert a is not b


def test_evaluator_at_scale_rejects_out_of_range():
    ranker = WaveletMultiScalePreflightRanker()
    with pytest.raises(IndexError):
        ranker.evaluator_at_scale(99)


def test_add_rule_at_scale_registers_in_classification():
    ranker = WaveletMultiScalePreflightRanker()
    ranker.add_rule_at_scale(0, _make_rule(99999))
    assert ranker.classify_gate_to_scale(99999) == 0


def test_evaluate_clean_panel_passes_all_scales():
    ranker = WaveletMultiScalePreflightRanker()
    ranker.add_rule_at_scale(0, _make_rule(146))
    ranker.add_rule_at_scale(1, _make_rule(117))
    panel = GateVerdictPanel()  # no violations
    result = ranker.evaluate(panel)
    assert result.coarsest_gate_passed
    assert result.short_circuited_at_scale is None


def test_evaluate_coarse_violation_short_circuits():
    ranker = WaveletMultiScalePreflightRanker()
    ranker.add_rule_at_scale(0, _make_rule(146))  # file-existence rule
    ranker.add_rule_at_scale(2, _make_rule(1))   # byte-mutation rule (would fire too)
    panel = GateVerdictPanel(verdicts={"146": "VIOLATED", "1": "VIOLATED"})
    result = ranker.evaluate(panel)
    assert not result.coarsest_gate_passed
    assert result.short_circuited_at_scale == 0
    # Finer scales should NOT have been evaluated separately — they're padded
    # with the coarsest chain per the wavelet hierarchical-gating discipline.


def test_evaluate_fine_only_violation_passes_coarse():
    ranker = WaveletMultiScalePreflightRanker()
    ranker.add_rule_at_scale(0, _make_rule(146))
    ranker.add_rule_at_scale(2, _make_rule(1))
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED"})
    result = ranker.evaluate(panel)
    assert result.coarsest_gate_passed
    # Fine-scale rule SHOULD have fired.
    fine_chain = result.chain_per_scale[2]
    assert fine_chain.first_fired_index == 0


def test_update_at_scale_only_updates_one_scale():
    ranker = WaveletMultiScalePreflightRanker()
    ranker.add_rule_at_scale(0, _make_rule(146))
    ranker.add_rule_at_scale(1, _make_rule(117))
    panel = GateVerdictPanel(verdicts={"146": "VIOLATED"})
    ranker.update_at_scale(0, panel)
    # Only scale-0 evaluator's anchor count should have incremented.
    assert ranker.evaluator_at_scale(0).n_anchors == 1
    assert ranker.evaluator_at_scale(1).n_anchors == 0


def test_update_all_scales_increments_every_scale():
    ranker = WaveletMultiScalePreflightRanker(num_scales=4)
    ranker.add_rule_at_scale(0, _make_rule(146))
    ranker.add_rule_at_scale(1, _make_rule(117))
    ranker.add_rule_at_scale(2, _make_rule(1))
    ranker.add_rule_at_scale(3, _make_rule(220))
    panel = GateVerdictPanel(verdicts={"146": "VIOLATED"})
    ranker.update_all_scales(panel)
    for s in range(4):
        assert ranker.evaluator_at_scale(s).n_anchors == 1


def test_classification_explain_includes_short_circuit_marker():
    ranker = WaveletMultiScalePreflightRanker()
    ranker.add_rule_at_scale(0, _make_rule(146))
    panel = GateVerdictPanel(verdicts={"146": "VIOLATED"})
    result = ranker.evaluate(panel)
    explanation = result.explain()
    assert "SHORT-CIRCUITED" in explanation
    assert "GATED-OUT" in explanation
