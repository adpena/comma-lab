"""Tests for F11/F12a/F12b: unified-action + codec-primitive xray classes."""

from __future__ import annotations

import math

import pytest
import torch

from tac.xray.base import XRayPrimitive
from tac.xray.foveation_ego_motion import (
    FoveationEgoMotionAnalyzer,
    FoveationReport,
)
from tac.xray.predictive_coding_hierarchy import (
    PredictiveCodingHierarchy,
    PredictiveCodingReport,
)
from tac.xray.unified_action_principle import (
    UnifiedActionPrinciple,
    UnifiedActionValue,
)


# ── F11 UnifiedActionPrinciple ──────────────────────────────────────────


def test_unified_protocol():
    assert isinstance(UnifiedActionPrinciple(), XRayPrimitive)


def test_unified_name():
    assert UnifiedActionPrinciple().name == "unified_action_principle"


def test_unified_hooks_four_canonical():
    """F11 engages 4 of 6 hooks (no continual_learning or probe_disambiguator)."""
    h = UnifiedActionPrinciple().wire_in_hooks
    assert "sensitivity_map" in h
    assert "pareto_constraint" in h
    assert "bit_allocator" in h
    assert "cathedral_autopilot" in h


def test_unified_compute_requires_wasserstein():
    with pytest.raises(ValueError, match="wasserstein_proxy is required"):
        UnifiedActionPrinciple().compute(target=None, archive_bytes=178262)


def test_unified_compute_basic():
    result = UnifiedActionPrinciple().compute(
        target=None,
        wasserstein_proxy=0.5,
        score_gradient_norm=0.3,
        archive_bytes=178_262,
    )
    v = result.primitive_value
    assert isinstance(v, UnifiedActionValue)
    assert v.wasserstein_term == 0.5
    assert v.fisher_term == 0.3


def test_unified_tropical_rate_term_matches_contest():
    """Tropical rate = 25 * B / N."""
    result = UnifiedActionPrinciple().compute(
        target=None,
        wasserstein_proxy=0.5,
        archive_bytes=178_262,
    )
    expected = 25.0 * 178_262 / 37_545_489
    assert result.primitive_value.tropical_rate_term == pytest.approx(
        expected, rel=1e-9
    )


def test_unified_product_policy():
    """Product policy = cube-root of W * g * T."""
    result = UnifiedActionPrinciple().compute(
        target=None,
        wasserstein_proxy=1.0,
        fisher_diagonal=8.0,
        archive_bytes=0,  # Tropical = 0; product collapses to 0.
        composition_policy="product",
    )
    assert result.primitive_value.s_total == 0.0


def test_unified_sum_policy():
    result = UnifiedActionPrinciple().compute(
        target=None,
        wasserstein_proxy=1.0,
        fisher_diagonal=2.0,
        archive_bytes=0,
        composition_policy="sum",
    )
    # 1.0 + 2.0 + 0.0 = 3.0.
    assert result.primitive_value.s_total == pytest.approx(3.0)


def test_unified_refuses_unknown_policy():
    with pytest.raises(ValueError, match="composition_policy"):
        UnifiedActionPrinciple().compute(
            target=None,
            wasserstein_proxy=1.0,
            archive_bytes=100,
            composition_policy="bogus",
        )


def test_unified_refuses_negative_wasserstein():
    with pytest.raises(ValueError, match="wasserstein_proxy must be non-negative"):
        UnifiedActionPrinciple().compute(
            target=None,
            wasserstein_proxy=-1.0,
            archive_bytes=100,
        )


def test_unified_refuses_negative_score_grad():
    with pytest.raises(ValueError, match="score_gradient_norm"):
        UnifiedActionPrinciple().compute(
            target=None,
            wasserstein_proxy=1.0,
            score_gradient_norm=-0.1,
            archive_bytes=100,
        )


def test_unified_uses_fisher_diagonal_tensor():
    """Fisher diagonal can be a 1-D tensor; sum of |values| is used."""
    diag = torch.tensor([1.0, 2.0, 3.0])
    result = UnifiedActionPrinciple().compute(
        target=None,
        wasserstein_proxy=0.5,
        fisher_diagonal=diag,
        archive_bytes=100,
    )
    assert result.primitive_value.fisher_term == 6.0


def test_unified_fisher_default_is_one():
    """When no Fisher or grad provided, defaults to 1.0."""
    result = UnifiedActionPrinciple().compute(
        target=None,
        wasserstein_proxy=0.5,
        archive_bytes=100,
    )
    assert result.primitive_value.fisher_term == 1.0


def test_unified_refuses_missing_archive_bytes():
    with pytest.raises(ValueError, match="archive_bytes or target"):
        UnifiedActionPrinciple().compute(
            target=None,
            wasserstein_proxy=0.5,
        )


def test_unified_refuses_negative_archive_bytes():
    with pytest.raises(ValueError, match="archive_bytes must be non-negative"):
        UnifiedActionPrinciple().compute(
            target=None,
            wasserstein_proxy=0.5,
            archive_bytes=-1,
        )


def test_unified_with_archive_path_records_sha(tmp_path):
    archive = tmp_path / "x.bin"
    archive.write_bytes(b"abc")
    result = UnifiedActionPrinciple().compute(
        target=archive,
        wasserstein_proxy=0.5,
    )
    assert result.archive_or_video_path == archive
    assert result.archive_sha256 is not None


def test_unified_gradient_step():
    primitive = UnifiedActionPrinciple()
    theta = torch.tensor([1.0, 2.0])
    grad = torch.tensor([0.1, 0.2])
    new_theta = primitive.gradient_descent_step(theta, lr=0.5, objective_grad=grad)
    assert torch.allclose(new_theta, torch.tensor([0.95, 1.9]))


def test_unified_gradient_step_refuses_shape_mismatch():
    primitive = UnifiedActionPrinciple()
    theta = torch.tensor([1.0, 2.0])
    grad = torch.tensor([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="shape"):
        primitive.gradient_descent_step(theta, lr=0.1, objective_grad=grad)


def test_unified_returns_envelope():
    result = UnifiedActionPrinciple().compute(
        target=None,
        wasserstein_proxy=0.5,
        archive_bytes=100,
    )
    assert result.primitive_name == "unified_action_principle"
    assert result.evidence_grade == "mathematical-derivation"


# ── F12a PredictiveCodingHierarchy ──────────────────────────────────────


def test_pc_protocol():
    assert isinstance(PredictiveCodingHierarchy(), XRayPrimitive)


def test_pc_name():
    assert PredictiveCodingHierarchy().name == "predictive_coding_hierarchy"


def test_pc_hooks():
    h = PredictiveCodingHierarchy().wire_in_hooks
    assert "bit_allocator" in h
    assert "sensitivity_map" in h


def test_pc_3d_input_accepted():
    x = torch.randn(3, 32, 32)
    result = PredictiveCodingHierarchy().compute(target=x, n_levels=2)
    assert result.primitive_value.n_frames == 1


def test_pc_4d_input_accepted():
    x = torch.randn(2, 3, 32, 32)
    result = PredictiveCodingHierarchy().compute(target=x, n_levels=2)
    assert result.primitive_value.n_frames == 2


def test_pc_refuses_wrong_dim():
    x = torch.randn(32, 32)
    with pytest.raises(ValueError, match="must be"):
        PredictiveCodingHierarchy().compute(target=x)


def test_pc_refuses_zero_levels():
    x = torch.randn(3, 32, 32)
    with pytest.raises(ValueError, match="n_levels must be positive"):
        PredictiveCodingHierarchy().compute(target=x, n_levels=0)


def test_pc_constant_frame_low_residual():
    """A constant frame has SMALL residual at every level."""
    x = torch.full((1, 3, 32, 32), 5.0)
    result = PredictiveCodingHierarchy().compute(target=x, n_levels=2)
    for norm in result.primitive_value.per_level_residual_norm:
        assert norm < 1e-4


def test_pc_random_frame_nonzero_residual():
    torch.manual_seed(0xDEAD)
    x = torch.randn(1, 3, 32, 32)
    result = PredictiveCodingHierarchy().compute(target=x, n_levels=2)
    # At least one level has nonzero residual.
    assert any(n > 0 for n in result.primitive_value.per_level_residual_norm)


def test_pc_compression_ratio_positive():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 32)
    result = PredictiveCodingHierarchy().compute(target=x, n_levels=2)
    assert result.primitive_value.compression_ratio_estimate > 0


def test_pc_returns_envelope():
    x = torch.randn(3, 16, 16)
    result = PredictiveCodingHierarchy().compute(target=x)
    assert result.primitive_name == "predictive_coding_hierarchy"
    assert result.evidence_grade == "council-deliberation"


# ── F12b FoveationEgoMotionAnalyzer ─────────────────────────────────────


def test_foveation_protocol():
    assert isinstance(FoveationEgoMotionAnalyzer(), XRayPrimitive)


def test_foveation_name():
    assert FoveationEgoMotionAnalyzer().name == "foveation_ego_motion"


def test_foveation_hooks():
    h = FoveationEgoMotionAnalyzer().wire_in_hooks
    assert "sensitivity_map" in h
    assert "bit_allocator" in h


def test_foveation_default_no_poses_center_foe():
    """When no poses provided, FOE defaults to image center (0.5, 0.5)."""
    result = FoveationEgoMotionAnalyzer().compute(target=None)
    r = result.primitive_value
    assert r.foe_x_normalized == 0.5
    assert r.foe_y_normalized == 0.5
    assert r.n_pairs == 0


def test_foveation_with_poses_computes_foe():
    """Lateral translation should shift FOE x off-center."""
    # Strong rightward translation.
    poses = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 0.3, 0.0],  # forward=1, lateral=0.3 -> FOE shifted right.
        [0.0, 0.0, 0.0, 1.0, 0.3, 0.0],
    ])
    result = FoveationEgoMotionAnalyzer().compute(ego_motion_poses=poses)
    r = result.primitive_value
    # FOE x_norm = 0.5 + 0.5 * 0.3 = 0.65.
    assert r.foe_x_normalized == pytest.approx(0.65)
    assert r.n_pairs == 2


def test_foveation_refuses_wrong_pose_shape():
    bad = torch.randn(5, 4)
    with pytest.raises(ValueError, match="ego_motion_poses must be"):
        FoveationEgoMotionAnalyzer().compute(ego_motion_poses=bad)


def test_foveation_refuses_zero_sigma():
    with pytest.raises(ValueError, match="foe_sigma_pixels"):
        FoveationEgoMotionAnalyzer().compute(
            target=None, foe_sigma_pixels=0.0
        )


def test_foveation_refuses_zero_floor():
    with pytest.raises(ValueError, match="foveation_floor"):
        FoveationEgoMotionAnalyzer().compute(
            target=None, foveation_floor=0.0
        )


def test_foveation_refuses_floor_above_one():
    with pytest.raises(ValueError, match="foveation_floor"):
        FoveationEgoMotionAnalyzer().compute(
            target=None, foveation_floor=1.5
        )


def test_foveation_refuses_zero_image_size():
    with pytest.raises(ValueError, match="image_size"):
        FoveationEgoMotionAnalyzer().compute(
            target=None, image_size=(0, 100)
        )


def test_foveation_budget_ratio_less_than_one():
    """Foveation always reduces budget vs uniform (assuming aggressive sigma)."""
    result = FoveationEgoMotionAnalyzer().compute(
        target=None,
        image_size=(384, 512),
        foe_sigma_pixels=64.0,  # tight FOE
        foveation_floor=0.1,
    )
    assert result.primitive_value.foveated_budget_ratio < 1.0


def test_foveation_central_25_above_uniform():
    """Central-25% region gets MORE than 25% of total weight under foveation."""
    result = FoveationEgoMotionAnalyzer().compute(
        target=None,
        image_size=(384, 512),
        foe_sigma_pixels=96.0,
    )
    # Uniform = 0.25; foveated should be > 0.25.
    assert result.primitive_value.central_25_percent_weight > 0.25


def test_foveation_returns_envelope():
    result = FoveationEgoMotionAnalyzer().compute(target=None)
    assert result.primitive_name == "foveation_ego_motion"
    assert result.evidence_grade == "council-deliberation"
    assert "predictive_coding_hierarchy" in result.composes_with


def test_foveation_records_image_size_in_metadata():
    result = FoveationEgoMotionAnalyzer().compute(
        target=None, image_size=(128, 256)
    )
    assert result.metadata["image_size_h"] == 128
    assert result.metadata["image_size_w"] == 256


# ── F12a + F12b composition ──────────────────────────────────────────────


def test_compose_predictive_coding_with_foveation():
    pc = PredictiveCodingHierarchy()
    fov = FoveationEgoMotionAnalyzer()
    composed = pc.compose_with(fov)
    assert "bit_allocator" in composed.wire_in_hooks
    assert "sensitivity_map" in composed.wire_in_hooks
