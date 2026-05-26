# SPDX-License-Identifier: MIT
"""Tests for UNIWARD-into-BoostNeRV integration module (6th-order recursive
doctrine continuation per N+1 verdict DIAGNOSED mechanism).

Sister of `test_weight_map_and_loss.py` (scaffold tests). THIS file covers
the integration module that layers UNIWARD per-pixel routing INTO the
BoostNeRV-PR110-residual capacity-constrained substrate's loss path.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.uniward_per_pixel_distortion.boostnerv_integration import (
    INTEGRATION_NAME,
    INTEGRATION_VERSION,
    DEFAULT_UNIWARD_LAMBDA_BOOSTNERV,
    CONTEST_SEG_WEIGHT,
    CONTEST_POSE_SQRT_INNER,
    CONTEST_RATE_DENOM_BYTES,
    apply_per_pixel_weight_to_residual_error,
    compose_uniward_into_boostnerv_loss,
    build_canonical_provenance_for_integration,
)


# ---- apply_per_pixel_weight_to_residual_error ----

def test_apply_2d_residual_error_shape_match():
    """2D (H,W) residual error multiplied element-wise by 2D weight map."""
    err = np.ones((4, 6), dtype=np.float32) * 2.0
    w = np.ones((4, 6), dtype=np.float32) * 3.0
    out = apply_per_pixel_weight_to_residual_error(
        residual_error_per_pixel=err, weight_map_per_pixel=w
    )
    assert out.shape == (4, 6)
    assert np.allclose(out, 6.0)


def test_apply_3d_residual_error_broadcasts_weight_map_over_batch():
    """3D (B,H,W) residual error broadcasts the 2D weight map over batch axis."""
    err = np.ones((5, 4, 6), dtype=np.float32) * 2.0
    w = np.arange(24, dtype=np.float32).reshape(4, 6)
    out = apply_per_pixel_weight_to_residual_error(
        residual_error_per_pixel=err, weight_map_per_pixel=w
    )
    assert out.shape == (5, 4, 6)
    # Each batch slice should equal 2.0 * w
    for b in range(5):
        assert np.allclose(out[b], 2.0 * w)


def test_apply_2d_shape_mismatch_raises():
    err = np.ones((4, 6), dtype=np.float32)
    w = np.ones((5, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_per_pixel_weight_to_residual_error(
            residual_error_per_pixel=err, weight_map_per_pixel=w
        )


def test_apply_3d_shape_mismatch_on_hw_raises():
    err = np.ones((5, 4, 6), dtype=np.float32)
    w = np.ones((4, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_per_pixel_weight_to_residual_error(
            residual_error_per_pixel=err, weight_map_per_pixel=w
        )


def test_apply_invalid_ndim_raises():
    err = np.ones((5,), dtype=np.float32)
    w = np.ones((5,), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 2D"):
        apply_per_pixel_weight_to_residual_error(
            residual_error_per_pixel=err, weight_map_per_pixel=w
        )


# ---- compose_uniward_into_boostnerv_loss ----

def test_compose_uniward_only_no_scorer_components():
    """UNIWARD-weighted-residual-only path (pure-distortion attack)."""
    err = np.ones((4, 6), dtype=np.float32) * 0.5
    w = np.ones((4, 6), dtype=np.float32) * 2.0
    result = compose_uniward_into_boostnerv_loss(
        boostnerv_residual_error_per_pixel=err,
        weight_map_per_pixel=w,
        uniward_lambda=0.01,
    )
    # weighted_residual = 0.5 * 2.0 = 1.0; mean = 1.0; uniward_term = 0.01
    assert result["uniward_weighted_residual_mean"] == pytest.approx(1.0)
    assert result["uniward_term"] == pytest.approx(0.01)
    assert result["total_loss"] == pytest.approx(0.01)
    assert "score_loss_seg" not in result
    assert "score_loss_pose" not in result


def test_compose_with_scorer_components_includes_canonical_seg_pose():
    """When scorer_loss_components provided, canonical SegNet + PoseNet terms included."""
    err = np.ones((4, 6), dtype=np.float32) * 0.1
    w = np.ones((4, 6), dtype=np.float32) * 1.0
    scorer = {"seg_distortion": 0.001, "pose_distortion": 1e-5}
    result = compose_uniward_into_boostnerv_loss(
        boostnerv_residual_error_per_pixel=err,
        weight_map_per_pixel=w,
        scorer_loss_components=scorer,
        uniward_lambda=0.01,
    )
    assert result["score_loss_seg"] == pytest.approx(CONTEST_SEG_WEIGHT * 0.001)
    expected_pose = (CONTEST_POSE_SQRT_INNER * 1e-5) ** 0.5
    assert result["score_loss_pose"] == pytest.approx(expected_pose)
    expected_uniward_term = 0.01 * 0.1
    expected_total = (
        CONTEST_SEG_WEIGHT * 0.001 + expected_pose + expected_uniward_term
    )
    assert result["total_loss"] == pytest.approx(expected_total)


def test_compose_safe_band_rejection_for_negative_lambda():
    err = np.ones((4, 6), dtype=np.float32)
    w = np.ones((4, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="SAFE band"):
        compose_uniward_into_boostnerv_loss(
            boostnerv_residual_error_per_pixel=err,
            weight_map_per_pixel=w,
            uniward_lambda=-0.01,
        )


def test_compose_safe_band_rejection_for_too_large_lambda():
    err = np.ones((4, 6), dtype=np.float32)
    w = np.ones((4, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="SAFE band"):
        compose_uniward_into_boostnerv_loss(
            boostnerv_residual_error_per_pixel=err,
            weight_map_per_pixel=w,
            uniward_lambda=0.5,
        )


def test_compose_scorer_components_missing_seg_raises():
    err = np.ones((4, 6), dtype=np.float32)
    w = np.ones((4, 6), dtype=np.float32)
    bad_scorer = {"pose_distortion": 1e-5}
    with pytest.raises(ValueError, match="seg_distortion"):
        compose_uniward_into_boostnerv_loss(
            boostnerv_residual_error_per_pixel=err,
            weight_map_per_pixel=w,
            scorer_loss_components=bad_scorer,
        )


def test_compose_scorer_components_missing_pose_raises():
    err = np.ones((4, 6), dtype=np.float32)
    w = np.ones((4, 6), dtype=np.float32)
    bad_scorer = {"seg_distortion": 0.001}
    with pytest.raises(ValueError, match="pose_distortion"):
        compose_uniward_into_boostnerv_loss(
            boostnerv_residual_error_per_pixel=err,
            weight_map_per_pixel=w,
            scorer_loss_components=bad_scorer,
        )


# ---- Canonical Provenance per Catalog #323 + #341 ----

def test_compose_returns_canonical_provenance_with_routing_markers():
    """Catalog #323 canonical Provenance + Catalog #341 non-promotable markers."""
    err = np.ones((4, 6), dtype=np.float32)
    w = np.ones((4, 6), dtype=np.float32) * 2.0
    result = compose_uniward_into_boostnerv_loss(
        boostnerv_residual_error_per_pixel=err,
        weight_map_per_pixel=w,
        uniward_lambda=0.01,
        pair_index=42,
    )
    prov = result["provenance"]
    # Catalog #341 routing markers (THE non-negotiable per Catalog #317)
    assert prov["score_claim"] is False
    assert prov["promotable"] is False
    assert prov["axis_tag"] == "[predicted]"
    assert prov["evidence_grade"] == "macOS-MLX research-signal"
    # Catalog #230 sister-disjoint discipline acknowledgment
    assert prov["consumed_substrate_scope"] == "read_only_consumer_import"
    assert prov["boostnerv_substrate_modification_scope"] == "none_read_only_consumer_import"
    # Per-pair routing
    assert prov["pair_index"] == 42
    # Integration identity
    assert prov["integration_id"] == INTEGRATION_NAME
    assert prov["integration_version"] == INTEGRATION_VERSION


def test_build_canonical_provenance_direct():
    """Direct provenance builder per Catalog #323."""
    prov = build_canonical_provenance_for_integration(
        uniward_lambda=0.01,
        weight_map_dynamic_range=19.7,
        pair_index=None,
    )
    # Canonical Provenance keys
    expected_keys = {
        "integration_id", "integration_version", "uniward_lambda",
        "weight_map_dynamic_range_ratio", "consumed_substrate_id",
        "consumed_substrate_scope", "pair_index", "evidence_grade",
        "score_claim", "promotable", "axis_tag", "hardware_substrate_recommendation",
        "measurement_axis", "hook_numbers_fired", "entropy_position",
        "boostnerv_substrate_modification_scope",
    }
    assert expected_keys.issubset(set(prov.keys()))
    assert prov["consumed_substrate_id"] == "boost_nerv_pr110_residual"
    # Per Catalog #341 routing markers
    assert prov["score_claim"] is False
    assert prov["promotable"] is False
    assert prov["axis_tag"] == "[predicted]"


def test_provenance_dynamic_range_propagates():
    """Weight map dynamic range surfaced in provenance for observability per Catalog #305."""
    # 19.7x dynamic range matching N+1 weight map
    w = np.ones((4, 6), dtype=np.float32)
    w[0, 0] = 0.052  # min
    w[3, 5] = 1.027  # max
    result = compose_uniward_into_boostnerv_loss(
        boostnerv_residual_error_per_pixel=np.ones((4, 6), dtype=np.float32),
        weight_map_per_pixel=w,
        uniward_lambda=0.01,
    )
    # max/min ratio = 1.027 / 0.052 ~= 19.75
    assert result["provenance"]["weight_map_dynamic_range_ratio"] == pytest.approx(1.027 / 0.052, abs=0.01)


# ---- Integration with BoostNeRV substrate (READ-ONLY consumer import) ----

def test_boostnerv_substrate_consumer_import_succeeds():
    """Sister-disjoint discipline per Catalog #230: BoostNeRV substrate READ-ONLY import."""
    # This import is the canonical sister-disjoint surface; if it fails the
    # integration module's consumer scope is broken.
    from tac.substrates.boost_nerv_pr110_residual import (
        ARCHIVE_MAGIC,
        ARCHIVE_VERSION,
        BPR1_HEADER_LEN,
        BoostNervPr110ResidualConfig,
        DEFAULT_NUM_BOOSTING_ROUNDS,
        DEFAULT_RESIDUAL_BUDGET_BYTES,
    )
    # Sanity-check canonical BoostNeRV constants
    assert ARCHIVE_MAGIC == b"BPR1\x00"
    assert ARCHIVE_VERSION == 1
    assert BPR1_HEADER_LEN == 29
    assert DEFAULT_NUM_BOOSTING_ROUNDS == 1
    assert DEFAULT_RESIDUAL_BUDGET_BYTES == 8192
    cfg = BoostNervPr110ResidualConfig()
    # BoostNeRV substrate's canonical capacity-constrained residual head
    assert cfg.residual_hidden_dim == 12
    assert cfg.residual_spatial_h == 96
    assert cfg.residual_spatial_w == 128
    assert cfg.boosting_gain_clamp == 0.05


def test_boostnerv_uniward_factory_uses_boostnerv_grid_shape():
    """End-to-end: integration factory works at BoostNeRV's canonical residual grid shape."""
    from tac.substrates.boost_nerv_pr110_residual import BoostNervPr110ResidualConfig

    cfg = BoostNervPr110ResidualConfig()
    # Residual error at BoostNeRV's canonical (residual_spatial_h, residual_spatial_w)
    err = np.ones((cfg.residual_spatial_h, cfg.residual_spatial_w), dtype=np.float32) * 0.05
    w = np.ones((cfg.residual_spatial_h, cfg.residual_spatial_w), dtype=np.float32) * 1.0
    result = compose_uniward_into_boostnerv_loss(
        boostnerv_residual_error_per_pixel=err,
        weight_map_per_pixel=w,
        uniward_lambda=DEFAULT_UNIWARD_LAMBDA_BOOSTNERV,
    )
    assert result["uniward_weighted_residual_mean"] == pytest.approx(0.05)
    assert result["uniward_term"] == pytest.approx(0.0005)
    assert result["total_loss"] == pytest.approx(0.0005)
    assert "provenance" in result
    assert result["provenance"]["consumed_substrate_id"] == "boost_nerv_pr110_residual"


# ---- Default lambda SAFE band per Catalog #303 ----

def test_default_uniward_lambda_boostnerv_in_safe_band():
    """Per Catalog #303 cargo-cult audit + CLAUDE.md "Forbidden empirical-claim-without-evidence-tag":
    default lambda is in SAFE band [0.0, 0.05] (bounded coefficient prevents PR105
    kitchen_sink anti-pattern where auxiliary terms overpower primary objective)."""
    assert 0.0 < DEFAULT_UNIWARD_LAMBDA_BOOSTNERV <= 0.05
