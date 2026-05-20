# SPDX-License-Identifier: MIT
"""Tests for cross_codec_orthogonality_predictor_consumer canonical contract.

Per Catalog #335 + #341 + WAVE-3-STRATEGIC-FINDINGS-CANONICAL-EXTENSION
task spec.
"""
from __future__ import annotations

import pytest

from tac.cathedral.consumer_contract import (
    HookNumber,
    validate_consumer_module,
)
from tac.cathedral_consumers import cross_codec_orthogonality_predictor_consumer as ccop


def test_consumer_module_satisfies_canonical_contract() -> None:
    """Catalog #335 STRICT contract satisfied."""
    registration = validate_consumer_module(ccop)
    assert registration.contract_compliant, (
        f"contract violations: {registration.validation_errors}"
    )


def test_consumer_declares_hook_numbers() -> None:
    """Catalog #125 hook declarations match."""
    assert ccop.CONSUMER_NAME == "cross_codec_orthogonality_predictor_consumer"
    assert ccop.CONSUMER_VERSION == "0.1.0"
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in ccop.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in ccop.CONSUMER_HOOK_NUMBERS


def test_consume_candidate_returns_canonical_routing_markers_per_catalog_341() -> None:
    """Catalog #341 canonical Tier A markers in every return."""
    result = ccop.consume_candidate({"substrate": "fec6_frontier_cuda_t4"})
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"


def test_consume_candidate_extracts_substrate_label() -> None:
    """Substrate label extraction works for known tokens."""
    result = ccop.consume_candidate({"lane_id": "lane_pr101_frame_exploit_fec6"})
    assert isinstance(result, dict)
    # Either pr101 or fec6 should be matched
    assert result.get("matched_substrate") in {"pr101_gold", "fec6_frontier_cuda_t4"}


def test_predict_top_k_leverage_cross_hardware_aware_empirical_values() -> None:
    """Equation 7 canonical empirical anchors return expected values at top-1%."""
    advisory = ccop.predict_top_k_leverage_cross_hardware_aware(
        1.0, measurement_axis="[macOS-CPU advisory]"
    )
    cuda_t4 = ccop.predict_top_k_leverage_cross_hardware_aware(
        1.0, measurement_axis="[contest-CUDA T4]"
    )
    assert abs(advisory - 0.0641) < 1e-6
    assert abs(cuda_t4 - 0.1111) < 1e-6
    # The 73% concentration delta is the canonical finding
    assert (cuda_t4 - advisory) / advisory > 0.50  # at least 50% delta


def test_predict_top_k_leverage_falls_back_to_uniform_for_other_k_values() -> None:
    """Other K values fall back to uniform Pareto baseline (sister of Equation 3)."""
    leverage_at_10pct = ccop.predict_top_k_leverage_cross_hardware_aware(
        10.0, measurement_axis="[contest-CUDA T4]"
    )
    # 10% uniform baseline = 0.10
    assert abs(leverage_at_10pct - 0.10) < 1e-6


def test_predict_top_k_leverage_rejects_out_of_range_k() -> None:
    """Catalog #287 validation: k_percent must be in [0, 100]."""
    with pytest.raises(ValueError):
        ccop.predict_top_k_leverage_cross_hardware_aware(-1.0)
    with pytest.raises(ValueError):
        ccop.predict_top_k_leverage_cross_hardware_aware(101.0)


def test_predict_hnerv_backbone_saturation_saturated_cluster() -> None:
    """Equation 8: medal cluster substrates classify as saturated."""
    result = ccop.predict_hnerv_backbone_saturation("pr101_gold")
    assert result["is_saturated_backbone"] is True
    assert result["predicted_per_axis_diff"] == 0.0
    assert result["medal_cluster_token_matched"] is True


def test_predict_hnerv_backbone_saturation_non_cluster_substrate() -> None:
    """Equation 8: non-cluster substrates classify as not saturated."""
    result = ccop.predict_hnerv_backbone_saturation("nscs06_strip_everything")
    assert result["is_saturated_backbone"] is False
    assert result["predicted_per_axis_diff"] is None


def test_predict_hnerv_backbone_saturation_out_of_range_size() -> None:
    """Equation 8: backbone size outside [170k, 185k] is not saturated."""
    result = ccop.predict_hnerv_backbone_saturation("pr101_gold", backbone_size_bytes=500_000)
    assert result["is_saturated_backbone"] is False
    assert result["backbone_size_in_canonical_range"] is False


def test_predict_cross_codec_super_additivity_empirical_anchor_match() -> None:
    """Equation 9: 4 canonical empirical anchors classify as SUPER_ADDITIVE."""
    for (a, b) in (
        ("hnerv_brotli", "format0d_score_table"),
        ("hnerv_brotli", "apogee_int4"),
        ("format0d_score_table", "fec6_huffman_k16"),
        ("apogee_int4", "fec6_huffman_k16"),
    ):
        result = ccop.predict_cross_codec_super_additivity(a, b)
        assert result["classification"] == "SUPER_ADDITIVE"
        assert result["anchor_match"] is True
        assert result["confidence"] > 0.5


def test_predict_cross_codec_super_additivity_same_codec_is_indeterminate() -> None:
    """Equation 9: same-codec pair returns INDETERMINATE."""
    result = ccop.predict_cross_codec_super_additivity("hnerv_brotli", "hnerv_brotli")
    assert result["classification"] == "INDETERMINATE"


def test_predict_cross_codec_super_additivity_observation_derived_super() -> None:
    """Equation 9: caller-supplied observations meeting criteria → SUPER_ADDITIVE."""
    result = ccop.predict_cross_codec_super_additivity(
        "novel_codec_a",
        "novel_codec_b",
        top_k_jaccard=0.02,
        per_axis_pearson_seg=-0.05,
    )
    assert result["classification"] == "SUPER_ADDITIVE"
    assert result["anchor_match"] is False


def test_predict_cross_codec_super_additivity_unknown_no_observations() -> None:
    """Equation 9: unknown codec pair without observations → UNKNOWN_CODEC_PAIR."""
    result = ccop.predict_cross_codec_super_additivity("novel_codec_a", "novel_codec_b")
    assert result["classification"] == "UNKNOWN_CODEC_PAIR"


def test_consume_candidate_returns_backbone_saturation_for_pr101() -> None:
    """Hook #4 surfaces Equation 8 prediction for medal-cluster candidates."""
    result = ccop.consume_candidate({"substrate": "pr101_gold"})
    sat = result.get("hnerv_backbone_saturation", {})
    assert sat.get("is_saturated_backbone") is True


def test_consume_candidate_handles_no_substrate_match() -> None:
    """Hook #4: unknown candidate does not crash; returns Tier A markers."""
    result = ccop.consume_candidate({"unrelated": "value"})
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"


def test_update_from_anchor_is_noop() -> None:
    """Hook #5: update_from_anchor is operator-triggered, NO-OP per Catalog #287."""
    # Should not raise
    ccop.update_from_anchor({"any": "anchor"})
    ccop.update_from_anchor(None)
