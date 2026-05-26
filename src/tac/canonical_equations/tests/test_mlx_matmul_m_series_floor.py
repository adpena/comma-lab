# SPDX-License-Identifier: MIT
"""Tests for the M-series MLX matmul drift canonical floor."""

from __future__ import annotations

import pytest

from tac.canonical_equations import (
    build_mlx_matmul_drift_m_series_canonical_floor_v1,
    classify_mlx_matmul_drift,
)
from tac.canonical_equations.mlx_matmul_m_series_floor import (
    CANONICAL_ABS_MAX_UPPER_BOUND,
    CANONICAL_RMS_UPPER_BOUND,
    EQUATION_ID,
)


def test_classifier_marks_research_signal_false_authority() -> None:
    verdict = classify_mlx_matmul_drift(
        measured_abs_max=4.6e-2,
        measured_rms=1.24e-2,
        measured_rel_median=7.75e-4,
        matmul_shape=(64, 256, 64),
    )

    assert verdict["equation_id"] == EQUATION_ID
    assert verdict["verdict"] == "WITHIN_CANONICAL_FLOOR"
    assert verdict["canonical_abs_max_upper_bound"] == CANONICAL_ABS_MAX_UPPER_BOUND
    assert verdict["canonical_rms_upper_bound"] == CANONICAL_RMS_UPPER_BOUND
    assert verdict["axis_tag"] == "[macOS-MLX research-signal]"
    assert verdict["score_claim"] is False
    assert verdict["promotion_eligible"] is False
    assert verdict["rank_or_kill_eligible"] is False
    assert verdict["ready_for_exact_eval_dispatch"] is False


def test_classifier_flags_above_floor_for_mitigation() -> None:
    verdict = classify_mlx_matmul_drift(
        measured_abs_max=CANONICAL_ABS_MAX_UPPER_BOUND * 1.1,
        measured_rms=CANONICAL_RMS_UPPER_BOUND * 1.1,
    )

    assert verdict["verdict"] == "ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION"


def test_classifier_rejects_bad_measurements() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        classify_mlx_matmul_drift(measured_abs_max=-1.0)

    with pytest.raises(ValueError, match="NaN"):
        classify_mlx_matmul_drift(measured_abs_max=float("nan"))


def test_build_equation_records_anchor_and_consumers() -> None:
    equation = build_mlx_matmul_drift_m_series_canonical_floor_v1()

    assert equation.equation_id == EQUATION_ID
    assert equation.empirical_anchors[0].anchor_id == (
        "r1_pp_k_independent_verification_5_substrate_dims_20260526"
    )
    assert equation.empirical_anchors[0].provenance.promotion_eligible is False
    assert equation.provenance.score_claim_valid is False
    assert (
        "tac.substrates.coin_pp_implicit_neural_representation.tests.test_basic"
        in equation.canonical_consumers
    )
    assert (
        "path_3_fix_wave_r1_prime_prime_k_independent_verification"
        in equation.canonical_producers
    )
