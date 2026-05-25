# SPDX-License-Identifier: MIT
"""Tests for MLX/PyTorch downstream drift canonical equations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations import (
    build_mlx_pytorch_drift_equation_from_result_json,
    build_mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1,
    mlx_pytorch_full_decoder_downstream_scorer_drift_bound,
)
from tac.canonical_equations.mlx_pytorch_drift import EQUATION_ID


def _payload() -> dict:
    return {
        "schema": "pr95_mlx_pytorch_full_decoder_downstream_scorer_drift_v1",
        "archive_zip_sha256": "a" * 64,
        "posenet_sha256": "b" * 64,
        "segnet_sha256": "c" * 64,
        "n_pairs_actual": 1,
        "covered_pair_window": [0, 1],
        "covered_pair_indices_sha256": "d" * 64,
        "checkpoint_mode": "trained_archive",
        "scorer_input_mode": "contest_uint8",
        "aggregate_verdict": "BELOW_SCORER_PRECISION",
        "aggregate_contest_score_drift_units": 7.411371190353894e-05,
        "selfcomp_mackay_theoretical_prediction_verified": True,
        "provenance": {
            "captured_at_utc": "2026-05-25T21:08:00+00:00",
            "hardware_substrate": "darwin_arm64_apple_silicon",
        },
        "stages": [
            {
                "stage_name": "stage_1_hnerv_decoder_forward",
                "metric": "per_pixel_rgb_float32_in_0_255_range",
                "max_abs": 3.05e-05,
                "mean_abs": 4.8e-06,
                "rms": 5.0e-06,
                "extra": {},
            },
            {
                "stage_name": "stage_2_uint8_quantization_at_inflate",
                "metric": "per_pixel_uint8_level_difference",
                "max_abs": 0.0,
                "mean_abs": 0.0,
                "rms": 0.0,
                "extra": {
                    "flipped_pixels": 0,
                    "total_pixels": 1179648,
                    "flipped_fraction": 0.0,
                },
            },
            {
                "stage_name": "stage_3_segnet_forward",
                "metric": "per_pixel_logit_max_abs_plus_argmax_flip_count",
                "max_abs": 1.0e-04,
                "mean_abs": 1.0e-05,
                "rms": 2.0e-05,
                "extra": {
                    "argmax_flip_pixels": 0,
                    "total_pixels": 196608,
                    "argmax_flip_fraction": 0.0,
                },
            },
            {
                "stage_name": "stage_5_contest_score_aggregation",
                "metric": "aggregate_contest_score_drift_due_to_mlx_vs_pytorch_framework",
                "max_abs": 7.411371190353894e-05,
                "mean_abs": 7.411371190353894e-05,
                "rms": 7.411371190353894e-05,
                "extra": {
                    "aggregate_contest_delta_units": 7.411371190353894e-05,
                    "verdict_per_stage": "BELOW_SCORER_PRECISION",
                    "precision_lo": 0.001,
                    "precision_hi": 0.01,
                },
            },
        ],
    }


def test_bound_false_authority_markers() -> None:
    result = mlx_pytorch_full_decoder_downstream_scorer_drift_bound(
        aggregate_contest_score_drift_units=7.4e-05,
    )
    assert result["equation_id"] == EQUATION_ID
    assert result["verdict"] == "BELOW_SCORER_PRECISION"
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert "requires_paired_contest_cpu_plus_cuda_for_score_claim" in result["blockers"]


def test_bound_classifies_boundary() -> None:
    result = mlx_pytorch_full_decoder_downstream_scorer_drift_bound(
        aggregate_contest_score_drift_units=0.003,
        precision_threshold=0.001,
    )
    assert result["verdict"] == "AT_SCORER_PRECISION_BOUNDARY"


def test_bound_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="precision_threshold"):
        mlx_pytorch_full_decoder_downstream_scorer_drift_bound(
            aggregate_contest_score_drift_units=0.0,
            precision_threshold=0.0,
        )


def test_build_equation_from_payload(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps(_payload()))
    eq = build_mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1(
        _payload(),
        source_artifact=str(path),
    )
    assert eq.equation_id == EQUATION_ID
    assert eq.canonical_producers == (
        "tools.measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift",
    )
    assert "tools.run_pr95_mlx_long_training" in eq.canonical_consumers
    assert eq.empirical_anchors[0].residual == pytest.approx(0.07411371190353894)
    assert eq.empirical_anchors[0].provenance.promotion_eligible is False
    assert eq.provenance.score_claim_valid is False


def test_build_equation_from_result_json(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps(_payload()))
    eq = build_mlx_pytorch_drift_equation_from_result_json(path)
    assert eq.equation_id == EQUATION_ID
    assert eq.empirical_anchors[0].source_artifact == str(path)
