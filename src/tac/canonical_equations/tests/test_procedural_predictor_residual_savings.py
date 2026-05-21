# SPDX-License-Identifier: MIT
"""Tests for residual-hybrid procedural predictor + residual accounting."""
from __future__ import annotations

import math

import pytest

from tac.canonical_equations import DomainOfValidityViolation
from tac.canonical_equations.procedural_codebook_savings import (
    CANONICAL_RATE_DENOM_BYTES,
    CANONICAL_RATE_MULTIPLIER,
)
from tac.canonical_equations.procedural_predictor_residual_savings import (
    EQUATION_ID,
    build_procedural_predictor_plus_residual_correction_savings_v1,
    predict_procedural_predictor_plus_residual_correction_savings,
    validate_residual_hybrid_context,
)


def test_prediction_pair1_matches_observed_rate_regression() -> None:
    result = predict_procedural_predictor_plus_residual_correction_savings(
        original_payload_bytes=131_779,
        predictor_seed_or_code_bytes=96,
        residual_stream_bytes=186_958,
        context="magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands",
    )

    expected = CANONICAL_RATE_MULTIPLIER * 55_275 / CANONICAL_RATE_DENOM_BYTES
    assert result["equation_id"] == EQUATION_ID
    assert result["delta_bytes_replacement_minus_original"] == 55_275
    assert result["bytes_saved"] == -55_275
    assert result["verdict"] == "RATE_REGRESSION"
    assert math.isclose(result["predicted_delta_s_rate_only"], expected)
    assert math.isclose(result["predicted_delta_s_rate_only"], 0.036805353633828024)
    assert result["score_claim"] is False
    assert result["rank_or_kill_eligible"] is False


def test_prediction_pair2_matches_observed_rate_regression() -> None:
    result = predict_procedural_predictor_plus_residual_correction_savings(
        original_payload_bytes=16_292,
        predictor_seed_or_code_bytes=32,
        residual_stream_bytes=97_441,
        context="sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes",
    )

    expected = CANONICAL_RATE_MULTIPLIER * 81_181 / CANONICAL_RATE_DENOM_BYTES
    assert result["replacement_total_bytes"] == 97_473
    assert result["delta_bytes_replacement_minus_original"] == 81_181
    assert math.isclose(result["predicted_delta_s_rate_only"], expected)
    assert math.isclose(result["predicted_delta_s_rate_only"], 0.05405509567341099)


def test_prediction_detects_rate_win_and_container_overhead() -> None:
    result = predict_procedural_predictor_plus_residual_correction_savings(
        original_payload_bytes=1000,
        predictor_seed_or_code_bytes=32,
        residual_stream_bytes=400,
        container_overhead_bytes=8,
        context="_seed_plus_residual_future_candidate",
    )

    assert result["replacement_total_bytes"] == 440
    assert result["delta_bytes_replacement_minus_original"] == -560
    assert result["bytes_saved"] == 560
    assert result["verdict"] == "RATE_WIN"
    assert result["predicted_delta_s_rate_only"] < 0.0


def test_prediction_rejects_non_residual_context() -> None:
    with pytest.raises(DomainOfValidityViolation, match="not a residual-hybrid"):
        predict_procedural_predictor_plus_residual_correction_savings(
            original_payload_bytes=1000,
            predictor_seed_or_code_bytes=32,
            residual_stream_bytes=400,
            context="chroma_lut_replacement",
        )
    assert validate_residual_hybrid_context("chroma_lut_replacement", raise_on_invalid=False) is False


def test_prediction_rejects_negative_or_non_int_bytes() -> None:
    with pytest.raises(ValueError, match="original_payload_bytes"):
        predict_procedural_predictor_plus_residual_correction_savings(
            original_payload_bytes=-1,
            predictor_seed_or_code_bytes=32,
            residual_stream_bytes=400,
            context="_seed_plus_residual_future_candidate",
        )
    with pytest.raises(ValueError, match="residual_stream_bytes"):
        predict_procedural_predictor_plus_residual_correction_savings(
            original_payload_bytes=1000,
            predictor_seed_or_code_bytes=32,
            residual_stream_bytes=1.5,  # type: ignore[arg-type]
            context="_seed_plus_residual_future_candidate",
        )


def test_builder_registers_two_zero_residual_empirical_anchors() -> None:
    eq = build_procedural_predictor_plus_residual_correction_savings_v1()

    assert eq.equation_id == EQUATION_ID
    assert eq.python_callable_module_path.endswith(
        ":predict_procedural_predictor_plus_residual_correction_savings"
    )
    assert eq.domain_of_validity["requires_residual_hybrid_context"] is True
    assert eq.domain_of_validity["prediction_scope"] == "rate_axis_only_no_scorer_distortion_claim"
    assert len(eq.empirical_anchors) == 2
    assert all(anchor.residual == 0.0 for anchor in eq.empirical_anchors)
    assert {
        anchor.inputs["in_domain_context"] for anchor in eq.empirical_anchors
    } == {
        "magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands",
        "sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes",
    }
    assert eq.predicted_vs_empirical_residual == {
        "pair_1_dwt_detail_dense_streams_residual": 0.0,
        "pair_2_fec6_null_byte_srl1_residual": 0.0,
    }

