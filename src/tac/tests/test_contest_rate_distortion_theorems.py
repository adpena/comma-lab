"""Theorem-style identity tests for the contest rate-distortion system.

These tests verify the *mathematical properties* of the contest objective
(monotonicity, convexity, linearity, marginal identities, KKT activation
behavior) rather than just the numeric values. They are the formal
proof-of-correctness for the cathedral's math.

Theorems verified:

1. **Monotonicity in B**: ``∂S/∂B > 0`` always (more bytes ⇒ higher score).
2. **Linearity in d_seg**: ``∂²S/∂d_seg² = 0`` (S is affine in d_seg).
3. **Strict concavity in d_pose**: ``∂²S/∂d_pose² < 0`` (sqrt is concave;
   pose marginal decreases as d_pose increases).
4. **Importance-flip identity**: at ``d_pose = importance_flip_threshold()``,
   ``∂S/∂d_pose = ∂S/∂d_seg`` exactly.
5. **First-order Taylor consistency**: small-step ΔS ≈ ⟨∇S, Δx⟩.
6. **Decomposition reconstruction**: per-term decomposition sums to total.
7. **Zero-state identity**: S(0, 0, 0) = 0 (modulo 1e-30 clamp on pose).
8. **Pose-term concavity proof**: midpoint sqrt(10·avg(p1,p2)) ≥ avg(sqrt(10·p1), sqrt(10·p2)).
9. **Operating-point flip direction**: at PR106 frontier (d_pose=3.36e-5),
   pose marginal > seg marginal AND > rate marginal × CONTEST_RAW_VIDEO_BYTES.
"""

from __future__ import annotations

import math

import torch

from tac.contest_rate_distortion_system import (
    CONTEST_POSE_WEIGHT,
    CONTEST_RATE_WEIGHT,
    CONTEST_RAW_VIDEO_BYTES,
    CONTEST_SEG_WEIGHT,
    contest_score,
    contest_score_decomposition,
    contest_score_marginals,
    importance_flip_threshold,
)


# ---------------------------------------------------------------------------
# Theorem 1: Monotonicity in B
# ---------------------------------------------------------------------------

def test_score_strictly_monotone_in_archive_bytes() -> None:
    """For any (seg, pose), S(B + ΔB) > S(B) when ΔB > 0."""
    seg, pose = 6.7e-4, 3.36e-5
    s_lo = float(contest_score(seg_distortion=seg, pose_distortion=pose, archive_bytes=100_000))
    s_hi = float(contest_score(seg_distortion=seg, pose_distortion=pose, archive_bytes=200_000))
    assert s_hi > s_lo
    expected_delta = CONTEST_RATE_WEIGHT * 100_000 / CONTEST_RAW_VIDEO_BYTES
    assert abs((s_hi - s_lo) - expected_delta) < 1e-9


# ---------------------------------------------------------------------------
# Theorem 2: Linearity in d_seg (∂²S/∂d_seg² = 0)
# ---------------------------------------------------------------------------

def test_score_linear_in_seg_distortion() -> None:
    """For all (pose, B), S is affine in seg: S(αs1 + (1-α)s2) = αS(s1) + (1-α)S(s2)."""
    pose, B = 3.36e-5, 185_578
    s1, s2, alpha = 1e-3, 5e-3, 0.3
    s_mix_lhs = float(contest_score(
        seg_distortion=alpha * s1 + (1 - alpha) * s2,
        pose_distortion=pose, archive_bytes=B,
    ))
    s_mix_rhs = (
        alpha * float(contest_score(seg_distortion=s1, pose_distortion=pose, archive_bytes=B))
        + (1 - alpha) * float(contest_score(seg_distortion=s2, pose_distortion=pose, archive_bytes=B))
    )
    assert abs(s_mix_lhs - s_mix_rhs) < 1e-12


# ---------------------------------------------------------------------------
# Theorem 3: Strict concavity in d_pose (sqrt is concave)
# ---------------------------------------------------------------------------

def test_score_strictly_concave_in_pose_distortion() -> None:
    """For all p1 < p2, sqrt(10·(p1+p2)/2) > (sqrt(10·p1) + sqrt(10·p2))/2."""
    p1, p2 = 1e-5, 1e-3
    seg, B = 0.0, 0.0  # isolate pose contribution
    midpoint_score = float(contest_score(
        seg_distortion=seg, pose_distortion=(p1 + p2) / 2, archive_bytes=B,
    ))
    avg_endpoints = 0.5 * (
        float(contest_score(seg_distortion=seg, pose_distortion=p1, archive_bytes=B))
        + float(contest_score(seg_distortion=seg, pose_distortion=p2, archive_bytes=B))
    )
    # Strict concavity: midpoint of sqrt > average of sqrts (when p1 ≠ p2).
    assert midpoint_score > avg_endpoints


def test_pose_marginal_decreases_in_pose_distortion() -> None:
    """∂S/∂d_pose is monotonically decreasing in d_pose (concavity ⇒ decreasing marginal)."""
    pose_vals = [3.36e-5, 1e-4, 1e-3, 1e-2, 0.18]
    marginals = [
        contest_score_marginals(seg_distortion=0.001, pose_distortion=p, archive_bytes=200_000)["dS_dpose"]
        for p in pose_vals
    ]
    for i in range(len(marginals) - 1):
        assert marginals[i] > marginals[i + 1], (
            f"pose marginal not monotone decreasing: {marginals[i]} (at p={pose_vals[i]}) "
            f"<= {marginals[i+1]} (at p={pose_vals[i+1]})"
        )


# ---------------------------------------------------------------------------
# Theorem 4: Importance-flip threshold identity
# ---------------------------------------------------------------------------

def test_marginals_equal_exactly_at_flip_threshold() -> None:
    """At d_pose = importance_flip_threshold(), ∂S/∂d_pose = ∂S/∂d_seg = 100 exactly."""
    threshold = importance_flip_threshold()
    m = contest_score_marginals(seg_distortion=0.001, pose_distortion=threshold, archive_bytes=100_000)
    assert abs(m["dS_dpose"] - CONTEST_SEG_WEIGHT) < 1e-9
    assert abs(m["dS_dseg"] - CONTEST_SEG_WEIGHT) < 1e-12


def test_below_threshold_pose_dominates() -> None:
    threshold = importance_flip_threshold()
    m = contest_score_marginals(seg_distortion=0.001, pose_distortion=threshold * 0.1, archive_bytes=100_000)
    assert m["dS_dpose"] > m["dS_dseg"]


def test_above_threshold_seg_dominates() -> None:
    threshold = importance_flip_threshold()
    m = contest_score_marginals(seg_distortion=0.001, pose_distortion=threshold * 10, archive_bytes=100_000)
    assert m["dS_dpose"] < m["dS_dseg"]


# ---------------------------------------------------------------------------
# Theorem 5: First-order Taylor consistency
# ---------------------------------------------------------------------------

def test_first_order_taylor_agrees_with_finite_difference() -> None:
    """ΔS ≈ ⟨∇S, Δx⟩ at small step. Validates marginals against numerical diff."""
    seg0, pose0, B0 = 6.7e-4, 3.36e-5, 185_578
    s0 = float(contest_score(seg_distortion=seg0, pose_distortion=pose0, archive_bytes=B0))
    grad = contest_score_marginals(seg_distortion=seg0, pose_distortion=pose0, archive_bytes=B0)
    # Small step in each axis (relative to current value to keep step magnitude reasonable).
    d_seg, d_pose, d_B = 1e-7, 1e-9, 100.0
    s1 = float(contest_score(
        seg_distortion=seg0 + d_seg, pose_distortion=pose0 + d_pose, archive_bytes=B0 + d_B,
    ))
    predicted = grad["dS_dseg"] * d_seg + grad["dS_dpose"] * d_pose + grad["dS_dbytes"] * d_B
    actual = s1 - s0
    rel_err = abs(actual - predicted) / max(abs(actual), 1e-30)
    assert rel_err < 1e-3, f"Taylor first-order err {rel_err}; predicted={predicted}, actual={actual}"


# ---------------------------------------------------------------------------
# Theorem 6: Decomposition reconstruction
# ---------------------------------------------------------------------------

def test_decomposition_reconstructs_total_score() -> None:
    seg, pose, B = 6.7e-4, 3.36e-5, 185_578
    dec = contest_score_decomposition(seg_distortion=seg, pose_distortion=pose, archive_bytes=B)
    direct = float(contest_score(seg_distortion=seg, pose_distortion=pose, archive_bytes=B))
    assert abs(dec["total"] - direct) < 1e-9


def test_decomposition_shares_sum_to_one() -> None:
    seg, pose, B = 6.7e-4, 3.36e-5, 185_578
    dec = contest_score_decomposition(seg_distortion=seg, pose_distortion=pose, archive_bytes=B)
    assert abs(dec["seg_share"] + dec["pose_share"] + dec["rate_share"] - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Theorem 7: Zero-state identity (almost; modulo 1e-30 clamp)
# ---------------------------------------------------------------------------

def test_score_at_origin_is_essentially_zero() -> None:
    s = float(contest_score(seg_distortion=0.0, pose_distortion=0.0, archive_bytes=0))
    # sqrt(10·1e-30) ≈ 3.16e-15; the clamp produces a tiny positive value.
    assert 0.0 <= s < 1e-10


# ---------------------------------------------------------------------------
# Theorem 8: PR103-on-PR106 anchor reproduction (empirical validation)
# ---------------------------------------------------------------------------

def test_pr103_pr106_anchor_score_reproduces_to_seven_decimals() -> None:
    """The contest-CUDA T4 anchor: 0.20898105277982337 at 185,578 bytes,
    seg=0.00067082, pose=0.0000336.

    Tolerance 1e-7 because the anchor's seg/pose values are quoted to ~5
    significant figures; sqrt-non-linearity amplifies that uncertainty
    into the 8th decimal of the score.
    """
    s = float(contest_score(seg_distortion=0.00067082, pose_distortion=0.0000336, archive_bytes=185_578))
    assert abs(s - 0.20898105277982337) < 1e-7


# ---------------------------------------------------------------------------
# Theorem 9: PR106 operating-point flip (CLAUDE.md ground truth)
# ---------------------------------------------------------------------------

def test_pr106_frontier_pose_dominates_seg_marginal_by_2_71x() -> None:
    """At PR106 frontier (d_pose=3.36e-5), pose marginal is ~2.71× the seg marginal."""
    m = contest_score_marginals(seg_distortion=6.7e-4, pose_distortion=3.36e-5, archive_bytes=185_578)
    ratio = m["dS_dpose"] / m["dS_dseg"]
    # Pose ≈ sqrt(10) / (2·sqrt(3.36e-5)) = sqrt(10) / (2 · 5.797e-3) ≈ 272.78
    # Seg ≈ 100. Ratio ≈ 2.7278.
    assert 2.72 < ratio < 2.74


# ---------------------------------------------------------------------------
# Theorem 10: Autograd reproduces analytical marginals to numerical precision
# ---------------------------------------------------------------------------

def test_autograd_pose_marginal_matches_analytical_to_double_precision() -> None:
    pose_val = 3.36e-5
    pose = torch.tensor(pose_val, dtype=torch.float64, requires_grad=True)
    s = contest_score(seg_distortion=6.7e-4, pose_distortion=pose, archive_bytes=185_578)
    s.backward()
    expected = math.sqrt(CONTEST_POSE_WEIGHT) / (2.0 * math.sqrt(pose_val))
    assert abs(float(pose.grad) - expected) / expected < 1e-10


def test_autograd_seg_marginal_matches_analytical_exactly() -> None:
    seg = torch.tensor(6.7e-4, dtype=torch.float64, requires_grad=True)
    s = contest_score(seg_distortion=seg, pose_distortion=3.36e-5, archive_bytes=185_578)
    s.backward()
    assert abs(float(seg.grad) - CONTEST_SEG_WEIGHT) < 1e-12


def test_autograd_archive_bytes_marginal_matches_analytical_exactly() -> None:
    B = torch.tensor(185_578.0, dtype=torch.float64, requires_grad=True)
    s = contest_score(seg_distortion=6.7e-4, pose_distortion=3.36e-5, archive_bytes=B)
    s.backward()
    expected = CONTEST_RATE_WEIGHT / CONTEST_RAW_VIDEO_BYTES
    assert abs(float(B.grad) - expected) < 1e-15


# ---------------------------------------------------------------------------
# Theorem 11: Hessian sign tests via finite differences
# ---------------------------------------------------------------------------

def test_hessian_diagonal_d2S_d_pose2_is_negative_via_finite_diff() -> None:
    """Concavity in pose: ∂²S/∂d_pose² < 0. Verify via central difference on the marginal."""
    p0, h = 3.36e-5, 1e-7
    m_lo = contest_score_marginals(seg_distortion=0.001, pose_distortion=p0 - h, archive_bytes=200_000)
    m_hi = contest_score_marginals(seg_distortion=0.001, pose_distortion=p0 + h, archive_bytes=200_000)
    second_derivative = (m_hi["dS_dpose"] - m_lo["dS_dpose"]) / (2 * h)
    # Analytical: d²/dp² (sqrt(10p)) = -sqrt(10)/(4·p^(3/2))
    expected = -math.sqrt(CONTEST_POSE_WEIGHT) / (4.0 * p0**1.5)
    assert second_derivative < 0
    assert abs(second_derivative - expected) / abs(expected) < 1e-2


def test_hessian_diagonal_d2S_d_seg2_is_zero_via_finite_diff() -> None:
    """Linearity in seg: ∂²S/∂d_seg² = 0."""
    s0, h = 1e-3, 1e-6
    m_lo = contest_score_marginals(seg_distortion=s0 - h, pose_distortion=3.36e-5, archive_bytes=200_000)
    m_hi = contest_score_marginals(seg_distortion=s0 + h, pose_distortion=3.36e-5, archive_bytes=200_000)
    second_derivative = (m_hi["dS_dseg"] - m_lo["dS_dseg"]) / (2 * h)
    assert abs(second_derivative) < 1e-9
