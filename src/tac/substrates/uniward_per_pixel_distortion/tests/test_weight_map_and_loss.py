# SPDX-License-Identifier: MIT
"""MLX-local smoke + correctness tests for UNIWARD per-pixel distortion substrate.

Per CLAUDE.md "MLX portable-local-substrate authority": tagged
[macOS-MLX research-signal]; NO authority claim; NO promotion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tac.substrates.uniward_per_pixel_distortion import (
    SUBSTRATE_ID,
    SUBSTRATE_VERSION,
    compute_per_pixel_uniward_weight_map_numpy,
    compose_uniward_weighted_score_loss,
)
from tac.substrates.uniward_per_pixel_distortion.weight_map import (
    WEIGHT_MAP_EPS,
    decompose_per_axis_weights,
    histogram_weight_distribution,
    normalize_weight_map_to_unit_mean,
)
from tac.substrates.uniward_per_pixel_distortion.score_aware_loss import (
    CONTEST_SEG_WEIGHT,
    CONTEST_POSE_SQRT_WEIGHT,
    DEFAULT_UNIWARD_LAMBDA,
)


def test_substrate_id_and_version_canonical() -> None:
    assert SUBSTRATE_ID == "uniward_per_pixel_distortion"
    assert SUBSTRATE_VERSION == "v1_2026-05-26"


def test_weight_map_higher_for_lower_sensitivity_pixel() -> None:
    """Canonical Fridrich UNIWARD invariant: low gradient → high weight."""
    H, W = 4, 4
    # Low-sensitivity pixel at (0, 0), high-sensitivity at (3, 3)
    d_seg = np.zeros((H, W), dtype=np.float32)
    d_pose = np.zeros((H, W), dtype=np.float32)
    d_seg[3, 3] = 1.0
    d_pose[3, 3] = 1.0
    weight = compute_per_pixel_uniward_weight_map_numpy(d_seg, d_pose)
    # Low-sensitivity pixel (0,0) has weight = 1/eps; high-sensitivity (3,3) has weight = 1/(eps+2)
    assert weight[0, 0] > weight[3, 3], (
        f"low-sensitivity pixel weight {weight[0, 0]} should exceed high-sensitivity {weight[3, 3]}"
    )
    # Sanity: weight[0,0] ≈ 1/eps
    assert weight[0, 0] == pytest.approx(1.0 / WEIGHT_MAP_EPS, rel=1e-5)


def test_weight_map_shape_mismatch_raises() -> None:
    d_seg = np.zeros((4, 4), dtype=np.float32)
    d_pose = np.zeros((4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_per_pixel_uniward_weight_map_numpy(d_seg, d_pose)


def test_weight_map_negative_gradient_rejected() -> None:
    d_seg = -np.ones((4, 4), dtype=np.float32)
    d_pose = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="non-negative"):
        compute_per_pixel_uniward_weight_map_numpy(d_seg, d_pose)


def test_weight_map_unit_mean_normalization() -> None:
    np.random.seed(42)
    d_seg = np.abs(np.random.randn(8, 8).astype(np.float32))
    d_pose = np.abs(np.random.randn(8, 8).astype(np.float32))
    weight = compute_per_pixel_uniward_weight_map_numpy(d_seg, d_pose)
    norm = normalize_weight_map_to_unit_mean(weight)
    assert norm.mean() == pytest.approx(1.0, rel=1e-4)


def test_decompose_per_axis_weights_keys() -> None:
    """Observability per Catalog #305: per-axis decomposition available."""
    d_seg = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    d_pose = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    decomp = decompose_per_axis_weights(d_seg, d_pose)
    assert set(decomp.keys()) == {"weight_seg_only", "weight_pose_only", "weight_joint"}
    # seg-only: high at (0,0) (low d_seg) and (1,1) (low d_seg)
    assert decomp["weight_seg_only"][0, 0] > decomp["weight_seg_only"][1, 0]
    # pose-only: high at (0,1) (low d_pose) and (1,0)
    assert decomp["weight_pose_only"][0, 1] > decomp["weight_pose_only"][0, 0]
    # joint: low only where BOTH are high
    assert decomp["weight_joint"][1, 1] == pytest.approx(1.0 / (WEIGHT_MAP_EPS + 4.0))


def test_histogram_observability_surface() -> None:
    """Catalog #305 observability surface: histogram queryable post-hoc."""
    np.random.seed(7)
    d_seg = np.abs(np.random.randn(16, 16).astype(np.float32))
    d_pose = np.abs(np.random.randn(16, 16).astype(np.float32))
    weight = compute_per_pixel_uniward_weight_map_numpy(d_seg, d_pose)
    hist = histogram_weight_distribution(weight, bins=10)
    assert hist["histogram_counts"].shape == (10,)
    assert hist["histogram_edges"].shape == (11,)
    assert hist["weight_min"] <= hist["weight_median"] <= hist["weight_max"]
    assert hist["weight_counts" if False else "histogram_counts"].sum() == 16 * 16


def test_compose_uniward_weighted_score_loss_canonical_formula() -> None:
    """Test canonical contest formula composition with per-pixel UNIWARD term."""
    H, W = 4, 4
    scorer_components = {
        "seg_distortion": 0.01,  # canonical seg term: 100 * 0.01 = 1.0
        "pose_distortion": 0.10,  # canonical pose term: sqrt(10 * 0.10) = sqrt(1.0) = 1.0
    }
    perturbation = np.ones((H, W), dtype=np.float32) * 0.1
    weight_map = np.ones((H, W), dtype=np.float32) * 2.0
    result = compose_uniward_weighted_score_loss(
        scorer_loss_components=scorer_components,
        perturbation_per_pixel=perturbation,
        weight_map_per_pixel=weight_map,
        uniward_lambda=0.5,
    )
    # Score terms
    assert result["score_loss_seg"] == pytest.approx(1.0, rel=1e-5)
    assert result["score_loss_pose"] == pytest.approx(1.0, rel=1e-5)
    # UNIWARD cost: lambda * mean(perturbation * weight) = 0.5 * mean(0.1 * 2) = 0.1
    assert result["uniward_perturbation_cost"] == pytest.approx(0.1, rel=1e-5)
    # Total: 1.0 + 1.0 + 0.1 = 2.1
    assert result["total_loss"] == pytest.approx(2.1, rel=1e-5)


def test_compose_loss_provenance_canonical_markers() -> None:
    """Canonical Provenance per Catalog #323 + #341 routing markers."""
    scorer_components = {"seg_distortion": 0.0, "pose_distortion": 0.0}
    H, W = 2, 2
    result = compose_uniward_weighted_score_loss(
        scorer_loss_components=scorer_components,
        perturbation_per_pixel=np.zeros((H, W), dtype=np.float32),
        weight_map_per_pixel=np.ones((H, W), dtype=np.float32),
    )
    prov = result["provenance"]
    # Catalog #323 canonical Provenance + Catalog #341 non-promotable markers
    assert prov["substrate_id"] == "uniward_per_pixel_distortion"
    assert prov["evidence_grade"] == "macOS-MLX research-signal"
    assert prov["score_claim"] is False
    assert prov["promotable"] is False
    assert prov["axis_tag"] == "[predicted]"
    assert prov["entropy_position"] == "P2_loss_shape_TRAIN_phase"


def test_compose_loss_shape_mismatch_raises() -> None:
    scorer_components = {"seg_distortion": 0.0, "pose_distortion": 0.0}
    with pytest.raises(ValueError, match="shape mismatch"):
        compose_uniward_weighted_score_loss(
            scorer_loss_components=scorer_components,
            perturbation_per_pixel=np.zeros((4, 4), dtype=np.float32),
            weight_map_per_pixel=np.zeros((4, 5), dtype=np.float32),
        )


def test_compose_loss_missing_components_raises() -> None:
    with pytest.raises(ValueError, match="seg_distortion"):
        compose_uniward_weighted_score_loss(
            scorer_loss_components={"pose_distortion": 0.0},
            perturbation_per_pixel=np.zeros((2, 2), dtype=np.float32),
            weight_map_per_pixel=np.ones((2, 2), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="pose_distortion"):
        compose_uniward_weighted_score_loss(
            scorer_loss_components={"seg_distortion": 0.0},
            perturbation_per_pixel=np.zeros((2, 2), dtype=np.float32),
            weight_map_per_pixel=np.ones((2, 2), dtype=np.float32),
        )


def test_mlx_local_smoke_per_pixel_routing_demonstration() -> None:
    """Smoke: simulate per-pixel routing demonstrates Fridrich UNIWARD principle.

    Construct a synthetic gradient field where pixels (0, 0)-(0, 5) are
    LOW-sensitivity (smooth region per UNIWARD canonical) and pixels (5, 5)-(7, 7)
    are HIGH-sensitivity (textured boundary region). Verify the weight map
    correctly routes perturbation budget AWAY from high-sensitivity zones.
    """
    H, W = 8, 8
    # Construct synthetic gradient: low at (0,*) (smooth row), high at (5:, 5:)
    d_seg = np.zeros((H, W), dtype=np.float32)
    d_seg[5:, 5:] = 2.0  # high seg sensitivity at boundary
    d_pose = np.zeros((H, W), dtype=np.float32)
    d_pose[5:, 5:] = 2.0  # high pose sensitivity at same zone
    weight = compute_per_pixel_uniward_weight_map_numpy(d_seg, d_pose)
    # Routing test: simulate uniform perturbation; weight-mediated cost
    perturbation_uniform = np.ones((H, W), dtype=np.float32) * 0.1
    # Smooth zone (low sensitivity) gets high weight => high routing
    smooth_zone_weight_mean = weight[0:3, 0:3].mean()
    boundary_zone_weight_mean = weight[5:, 5:].mean()
    assert smooth_zone_weight_mean > boundary_zone_weight_mean, (
        f"Fridrich UNIWARD invariant violated: smooth zone weight "
        f"{smooth_zone_weight_mean} should exceed boundary zone "
        f"{boundary_zone_weight_mean}"
    )
    # Demonstrate weighted-loss respects routing
    result = compose_uniward_weighted_score_loss(
        scorer_loss_components={"seg_distortion": 0.001, "pose_distortion": 0.001},
        perturbation_per_pixel=perturbation_uniform,
        weight_map_per_pixel=normalize_weight_map_to_unit_mean(weight),
        uniward_lambda=DEFAULT_UNIWARD_LAMBDA,
    )
    # Loss is finite and positive
    assert math.isfinite(result["total_loss"])
    assert result["total_loss"] > 0.0
    # Score components agree with canonical formula
    assert result["score_loss_seg"] == pytest.approx(100.0 * 0.001, rel=1e-5)
    assert result["score_loss_pose"] == pytest.approx(
        CONTEST_POSE_SQRT_WEIGHT * math.sqrt(0.001), rel=1e-5
    )
