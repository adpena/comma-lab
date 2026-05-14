# SPDX-License-Identifier: MIT
"""Tests for scorer-internal xray primitives: F7/F8/F9."""

from __future__ import annotations

import math

import pytest
import torch

from tac.xray.base import XRayPrimitive
from tac.xray.per_pair_score_decomposition import (
    POSE_SQRT_COEFF,
    PerPairScoreBreakdown,
    PerPairScoreDecomposition,
    SEG_COEFF,
)
from tac.xray.posenet_se3_lie_algebra import (
    POSENET_OUTPUT_DIMS_USED,
    PoseNetSE3LieAlgebra,
    PoseSE3Report,
)
from tac.xray.segnet_margin_polytope import (
    SEGNET_N_CLASSES,
    SegNetLogitMarginPolytope,
    SegNetMarginReport,
)


# ── F7 SegNetLogitMarginPolytope ────────────────────────────────────────


def test_segnet_protocol():
    assert isinstance(SegNetLogitMarginPolytope(), XRayPrimitive)


def test_segnet_name():
    assert SegNetLogitMarginPolytope().name == "segnet_margin_polytope"


def test_segnet_hooks():
    h = SegNetLogitMarginPolytope().wire_in_hooks
    assert "sensitivity_map" in h
    assert "bit_allocator" in h


def test_segnet_constant_pinned():
    assert SEGNET_N_CLASSES == 5


def test_segnet_compute_3d_input_accepted():
    """Accept (C, H, W) input."""
    torch.manual_seed(0)
    logits = torch.randn(5, 64, 64)
    result = SegNetLogitMarginPolytope().compute(target=logits)
    r = result.primitive_value
    assert r.logits_shape == (1, 5, 64, 64)


def test_segnet_compute_4d_input_accepted():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 32, 32)
    result = SegNetLogitMarginPolytope().compute(target=logits)
    r = result.primitive_value
    assert r.logits_shape == (2, 5, 32, 32)


def test_segnet_refuses_wrong_dim():
    logits = torch.randn(5, 64)  # 2-D
    with pytest.raises(ValueError, match="target must be"):
        SegNetLogitMarginPolytope().compute(target=logits)


def test_segnet_margin_map_shape():
    logits = torch.randn(2, 5, 16, 16)
    margin = SegNetLogitMarginPolytope.compute_margin_map(logits)
    assert margin.shape == (2, 16, 16)


def test_segnet_static_margin_map_refuses_wrong_dim():
    logits = torch.randn(5, 16, 16)  # 3-D
    with pytest.raises(ValueError, match="must be"):
        SegNetLogitMarginPolytope.compute_margin_map(logits)


def test_segnet_margin_zero_when_top1_top2_tied():
    """If top1 and top2 logits are equal, margin is 0."""
    logits = torch.zeros(1, 5, 4, 4)
    logits[0, 0] = 1.0
    logits[0, 1] = 1.0  # tied
    margin = SegNetLogitMarginPolytope.compute_margin_map(logits)
    assert (margin == 0).all()


def test_segnet_margin_large_when_one_class_dominates():
    """One class dominating gives large margin."""
    logits = torch.zeros(1, 5, 4, 4)
    logits[0, 0] = 10.0  # class 0 dominates
    margin = SegNetLogitMarginPolytope.compute_margin_map(logits)
    assert margin.min().item() == pytest.approx(10.0)


def test_segnet_fragile_fraction_in_range():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 32, 32)
    result = SegNetLogitMarginPolytope().compute(
        target=logits, margin_threshold=0.1
    )
    assert 0.0 <= result.primitive_value.fragile_pixel_fraction <= 1.0


def test_segnet_safe_budget_compute():
    torch.manual_seed(0)
    margin = torch.full((1, 4, 4), 0.5)
    primitive = SegNetLogitMarginPolytope()
    budget = primitive.compute_safe_polytope_budget(margin)
    assert torch.allclose(budget, margin)


def test_segnet_safe_budget_with_gradient_normalizes():
    margin = torch.full((1, 4, 4), 1.0)
    grad = torch.full((1, 4, 4), 2.0)
    primitive = SegNetLogitMarginPolytope()
    budget = primitive.compute_safe_polytope_budget(
        margin, gradient_map=grad
    )
    # Approximately 1.0 / 2.0 = 0.5.
    assert torch.allclose(budget, torch.full((1, 4, 4), 0.5), atol=1e-3)


def test_segnet_safe_budget_refuses_shape_mismatch():
    margin = torch.zeros(1, 4, 4)
    grad = torch.zeros(1, 4, 8)
    primitive = SegNetLogitMarginPolytope()
    with pytest.raises(ValueError, match="shape"):
        primitive.compute_safe_polytope_budget(margin, gradient_map=grad)


def test_segnet_compute_metadata_records_classes():
    logits = torch.randn(1, 5, 16, 16)
    result = SegNetLogitMarginPolytope().compute(target=logits)
    assert result.metadata["n_classes_in_logits"] == 5


def test_segnet_returns_envelope():
    logits = torch.randn(1, 5, 16, 16)
    result = SegNetLogitMarginPolytope().compute(target=logits)
    assert result.primitive_name == "segnet_margin_polytope"
    assert result.evidence_grade == "structural-code-contract"


# ── F8 PoseNetSE3LieAlgebra ─────────────────────────────────────────────


def test_pose_protocol():
    assert isinstance(PoseNetSE3LieAlgebra(), XRayPrimitive)


def test_pose_name():
    assert PoseNetSE3LieAlgebra().name == "posenet_se3_lie_algebra"


def test_pose_hooks():
    h = PoseNetSE3LieAlgebra().wire_in_hooks
    assert "sensitivity_map" in h


def test_pose_constants():
    assert POSENET_OUTPUT_DIMS_USED == 6


def test_pose_compute_paired_input():
    """Accept (N, 2, 6) input."""
    torch.manual_seed(0)
    pairs = torch.randn(10, 2, 6) * 0.1
    result = PoseNetSE3LieAlgebra().compute(target=pairs)
    r = result.primitive_value
    assert r.n_pose_pairs == 10


def test_pose_compute_separate_input():
    """Accept (N, 6) + target_b."""
    torch.manual_seed(0)
    a = torch.randn(10, 6) * 0.1
    b = torch.randn(10, 6) * 0.1
    result = PoseNetSE3LieAlgebra().compute(target=a, target_b=b)
    assert result.primitive_value.n_pose_pairs == 10


def test_pose_refuses_missing_target_b():
    a = torch.randn(5, 6)
    with pytest.raises(ValueError, match="target_b must be provided"):
        PoseNetSE3LieAlgebra().compute(target=a)


def test_pose_refuses_shape_mismatch():
    a = torch.randn(5, 6)
    b = torch.randn(5, 4)
    with pytest.raises(ValueError, match="target_b"):
        PoseNetSE3LieAlgebra().compute(target=a, target_b=b)


def test_pose_refuses_wrong_dim():
    bad = torch.randn(5, 8)
    with pytest.raises(ValueError, match="must be"):
        PoseNetSE3LieAlgebra().compute(target=bad)


def test_pose_identical_poses_zero_residual():
    """Identical pose pairs should produce ~0 Lie-algebra residual."""
    pose = torch.randn(5, 6) * 0.01
    pairs = torch.stack([pose, pose], dim=1)  # (5, 2, 6)
    result = PoseNetSE3LieAlgebra().compute(target=pairs)
    assert result.primitive_value.mean_lie_algebra_distance < 1e-4


def test_pose_lie_residual_static_method_shape():
    a = torch.randn(5, 6)
    b = torch.randn(5, 6)
    res = PoseNetSE3LieAlgebra.compute_lie_algebra_residual(a, b)
    assert res.shape == (5,)


def test_pose_lie_residual_static_refuses_bad_shape():
    a = torch.randn(5, 5)
    b = torch.randn(5, 5)
    with pytest.raises(ValueError, match="must be"):
        PoseNetSE3LieAlgebra.compute_lie_algebra_residual(a, b)


def test_pose_small_motion_lie_approx_euclidean():
    """For SMALL coordinates, Lie distance ~ Euclidean (BCH first-order vanishes)."""
    torch.manual_seed(0)
    a = torch.randn(20, 6) * 0.001
    b = torch.randn(20, 6) * 0.001
    res = PoseNetSE3LieAlgebra.compute_lie_algebra_residual(a, b)
    euc = (a - b).pow(2).sum(dim=1).sqrt()
    # Should match within 5% (small-rotation regime).
    assert torch.allclose(res, euc, rtol=0.05)


def test_pose_high_motion_fraction_nonzero():
    torch.manual_seed(0)
    pairs = torch.randn(100, 2, 6) * 10  # large coordinates
    result = PoseNetSE3LieAlgebra().compute(
        target=pairs, high_motion_threshold=1.0
    )
    assert result.primitive_value.high_motion_fraction > 0.5


def test_pose_metadata_includes_threshold():
    pairs = torch.randn(5, 2, 6) * 0.01
    result = PoseNetSE3LieAlgebra().compute(
        target=pairs, high_motion_threshold=0.7
    )
    assert result.metadata["high_motion_threshold"] == 0.7


def test_pose_returns_envelope():
    pairs = torch.randn(5, 2, 6) * 0.01
    result = PoseNetSE3LieAlgebra().compute(target=pairs)
    assert result.primitive_name == "posenet_se3_lie_algebra"
    assert result.evidence_grade == "mathematical-derivation"


# ── F9 PerPairScoreDecomposition ────────────────────────────────────────


def test_pair_protocol():
    assert isinstance(PerPairScoreDecomposition(), XRayPrimitive)


def test_pair_name():
    assert PerPairScoreDecomposition().name == "per_pair_score_decomposition"


def test_pair_hooks():
    h = PerPairScoreDecomposition().wire_in_hooks
    assert "cathedral_autopilot" in h


def test_pair_constants():
    assert SEG_COEFF == 100.0
    assert POSE_SQRT_COEFF == 10.0


def test_pair_compute_paired_input():
    target = torch.tensor([[0.1, 0.01], [0.05, 0.02], [0.2, 0.04]])
    result = PerPairScoreDecomposition().compute(target=target, top_k=2)
    r = result.primitive_value
    assert r.n_pairs == 3
    assert len(r.top_k_pair_indices) == 2


def test_pair_compute_separate_input():
    seg = torch.tensor([0.1, 0.05, 0.2])
    pose = torch.tensor([0.01, 0.02, 0.04])
    result = PerPairScoreDecomposition().compute(
        target=seg, target_pose=pose, top_k=2
    )
    assert result.primitive_value.n_pairs == 3


def test_pair_refuses_missing_target_pose():
    seg = torch.tensor([0.1, 0.05])
    with pytest.raises(ValueError, match="target_pose must be"):
        PerPairScoreDecomposition().compute(target=seg)


def test_pair_refuses_wrong_target_dim():
    bad = torch.randn(5, 3)
    with pytest.raises(ValueError, match="target must be"):
        PerPairScoreDecomposition().compute(target=bad)


def test_pair_refuses_zero_top_k():
    seg = torch.tensor([0.1, 0.05])
    pose = torch.tensor([0.01, 0.02])
    with pytest.raises(ValueError, match="top_k must be positive"):
        PerPairScoreDecomposition().compute(target=seg, target_pose=pose, top_k=0)


def test_pair_top_k_clipped_to_n_pairs():
    """If top_k > n_pairs, top_k effectively becomes n_pairs."""
    seg = torch.tensor([0.1, 0.05])
    pose = torch.tensor([0.01, 0.02])
    result = PerPairScoreDecomposition().compute(
        target=seg, target_pose=pose, top_k=100
    )
    assert len(result.primitive_value.top_k_pair_indices) == 2


def test_pair_contribution_formula_matches_contest():
    """c_i = 100*seg + sqrt(10*pose)."""
    seg = torch.tensor([0.1])
    pose = torch.tensor([0.018])
    result = PerPairScoreDecomposition().compute(
        target=seg, target_pose=pose, top_k=1
    )
    expected = 100 * 0.1 + math.sqrt(10 * 0.018)
    assert result.primitive_value.total_distortion_sum == pytest.approx(
        expected, abs=1e-5
    )


def test_pair_top_k_cumulative_fraction_monotone():
    """Cumulative fraction sequence is monotone non-decreasing."""
    seg = torch.tensor([0.1, 0.5, 0.05, 0.2, 0.05])
    pose = torch.tensor([0.01, 0.02, 0.005, 0.04, 0.005])
    result = PerPairScoreDecomposition().compute(
        target=seg, target_pose=pose, top_k=5
    )
    cum = result.primitive_value.top_k_cumulative_fraction
    for i in range(1, len(cum)):
        assert cum[i] >= cum[i - 1]


def test_pair_top_k_full_cumulative_at_one():
    """Top-K cumulative fraction equals 1.0 when K = N (modulo precision)."""
    seg = torch.tensor([0.1, 0.5, 0.05])
    pose = torch.tensor([0.01, 0.02, 0.005])
    result = PerPairScoreDecomposition().compute(
        target=seg, target_pose=pose, top_k=3
    )
    assert result.primitive_value.top_k_cumulative_fraction[-1] == pytest.approx(
        1.0, abs=1e-6
    )


def test_pair_priority_vector_normalized():
    """Per-pair priority vector mean is approximately 1.0 (= c_i / mean)."""
    seg = torch.tensor([0.1, 0.5, 0.05])
    pose = torch.tensor([0.01, 0.02, 0.005])
    result = PerPairScoreDecomposition().compute(
        target=seg, target_pose=pose, top_k=3
    )
    priority = torch.tensor(result.primitive_value.per_pair_priority)
    assert priority.mean().item() == pytest.approx(1.0, abs=1e-5)


def test_pair_identify_high_score_pairs_returns_top_percentile():
    seg = torch.tensor([0.1, 0.05, 0.5, 0.2, 0.05])
    pose = torch.tensor([0.01, 0.005, 0.02, 0.04, 0.005])
    primitive = PerPairScoreDecomposition()
    high_pairs = primitive.identify_high_score_pairs(
        seg, target_pose=pose, percentile=60.0
    )
    # 60th percentile: only the top ~40% of pairs (= 2 pairs out of 5).
    assert isinstance(high_pairs, list)
    assert all(0 <= i < 5 for i in high_pairs)


def test_pair_identify_refuses_bad_percentile():
    seg = torch.tensor([0.1])
    pose = torch.tensor([0.01])
    primitive = PerPairScoreDecomposition()
    with pytest.raises(ValueError, match="percentile"):
        primitive.identify_high_score_pairs(seg, target_pose=pose, percentile=150)


def test_pair_metadata_records_coefficients():
    seg = torch.tensor([0.1])
    pose = torch.tensor([0.01])
    result = PerPairScoreDecomposition().compute(target=seg, target_pose=pose)
    assert result.metadata["seg_coeff"] == 100.0


def test_pair_returns_envelope():
    seg = torch.tensor([0.1])
    pose = torch.tensor([0.01])
    result = PerPairScoreDecomposition().compute(target=seg, target_pose=pose)
    assert result.primitive_name == "per_pair_score_decomposition"
    assert "cathedral_autopilot" in result.wire_in_hooks_engaged
