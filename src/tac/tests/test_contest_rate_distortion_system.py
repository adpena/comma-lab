"""Tests for :mod:`tac.contest_rate_distortion_system`."""

from __future__ import annotations

import math

import pytest
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
# Frozen contest constants (must match upstream/evaluate.py)
# ---------------------------------------------------------------------------

def test_contest_constants_match_upstream() -> None:
    assert CONTEST_SEG_WEIGHT == 100.0
    assert CONTEST_POSE_WEIGHT == 10.0
    assert CONTEST_RATE_WEIGHT == 25.0
    assert CONTEST_RAW_VIDEO_BYTES == 37_545_489


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

def test_contest_score_returns_scalar_tensor() -> None:
    s = contest_score(seg_distortion=0.001, pose_distortion=0.0001, archive_bytes=200_000)
    assert isinstance(s, torch.Tensor)
    assert s.ndim == 0


def test_contest_score_pr103_pr106_rate_term_reproduces_exactly() -> None:
    """The rate term is purely arithmetic; reproducing it exactly verifies
    the constants. PR103-on-PR106 candidate: 185,578 bytes."""
    B = 185_578
    expected_rate_term = CONTEST_RATE_WEIGHT * B / CONTEST_RAW_VIDEO_BYTES
    s = float(contest_score(seg_distortion=0.0, pose_distortion=0.0, archive_bytes=B))
    # When seg=pose=0, score = rate_term + sqrt(0) (clamped to ~0 via 1e-30 floor).
    assert abs(s - expected_rate_term) < 1e-5


def test_contest_score_zero_inputs() -> None:
    s = float(contest_score(seg_distortion=0.0, pose_distortion=0.0, archive_bytes=0))
    # sqrt(10 * 1e-30) ≈ 3.16e-15; effectively 0.
    assert s < 1e-10


# ---------------------------------------------------------------------------
# Marginals (the operating-point-dependent importance flip)
# ---------------------------------------------------------------------------

def test_marginals_dS_dseg_is_constant() -> None:
    """∂S/∂d_seg = 100 regardless of operating point."""
    for seg, pose, B in [(0.001, 0.0001, 200_000), (0.5, 0.2, 1_000_000)]:
        m = contest_score_marginals(seg_distortion=seg, pose_distortion=pose, archive_bytes=B)
        assert m["dS_dseg"] == CONTEST_SEG_WEIGHT


def test_marginals_dS_dbytes_is_constant() -> None:
    """∂S/∂B = 25 / 37,545,489 ≈ 6.66e-7 regardless of operating point."""
    expected = CONTEST_RATE_WEIGHT / CONTEST_RAW_VIDEO_BYTES
    for seg, pose, B in [(0.001, 0.0001, 200_000), (0.5, 0.2, 1_000_000)]:
        m = contest_score_marginals(seg_distortion=seg, pose_distortion=pose, archive_bytes=B)
        assert abs(m["dS_dbytes"] - expected) < 1e-15


def test_marginals_dS_dpose_diverges_as_pose_drops() -> None:
    """∂S/∂d_pose = sqrt(10) / (2·sqrt(d_pose)) — diverges as d_pose → 0."""
    m_high = contest_score_marginals(seg_distortion=0.001, pose_distortion=0.18, archive_bytes=200_000)
    m_low = contest_score_marginals(seg_distortion=0.001, pose_distortion=3.4e-5, archive_bytes=200_000)
    assert m_low["dS_dpose"] > m_high["dS_dpose"]
    # PR106 frontier: pose marginal ≈ 271 (per CLAUDE.md)
    assert 270 < m_low["dS_dpose"] < 272


def test_marginals_pose_to_seg_ratio_matches_claudemd() -> None:
    """At PR106 frontier (d_pose=3.4e-5), pose marginal / seg marginal ≈ 2.71×."""
    m = contest_score_marginals(seg_distortion=0.001, pose_distortion=3.4e-5, archive_bytes=185_578)
    ratio = m["dS_dpose"] / m["dS_dseg"]
    assert 2.70 < ratio < 2.72, f"expected ~2.71x flip, got {ratio:.4f}x"


def test_importance_flip_threshold_is_2_5e_minus_4() -> None:
    """The d_pose value at which pose marginal == seg marginal."""
    assert abs(importance_flip_threshold() - 2.5e-4) < 1e-9


def test_importance_flip_marginals_equal_at_threshold() -> None:
    """Sanity: at d_pose = threshold, pose marginal exactly equals seg marginal."""
    threshold = importance_flip_threshold()
    m = contest_score_marginals(seg_distortion=0.001, pose_distortion=threshold, archive_bytes=200_000)
    assert abs(m["dS_dpose"] - m["dS_dseg"]) < 1e-9


# ---------------------------------------------------------------------------
# Decomposition (per-term forensic breakdown)
# ---------------------------------------------------------------------------

def test_decomposition_shares_sum_to_one() -> None:
    dec = contest_score_decomposition(seg_distortion=6.7e-4, pose_distortion=3.4e-5, archive_bytes=185_578)
    assert abs(dec["seg_share"] + dec["pose_share"] + dec["rate_share"] - 1.0) < 1e-9


def test_decomposition_total_matches_score() -> None:
    seg, pose, B = 0.001, 0.0001, 200_000
    dec = contest_score_decomposition(seg_distortion=seg, pose_distortion=pose, archive_bytes=B)
    s = float(contest_score(seg_distortion=seg, pose_distortion=pose, archive_bytes=B))
    assert abs(dec["total"] - s) < 1e-6


def test_decomposition_pr106_frontier_rate_dominated() -> None:
    """At the PR106 frontier (185,578 B, low distortion), rate dominates."""
    dec = contest_score_decomposition(seg_distortion=6.7e-4, pose_distortion=3.4e-5, archive_bytes=185_578)
    assert dec["rate_share"] > dec["seg_share"]
    assert dec["rate_share"] > dec["pose_share"]


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------

def test_contest_score_differentiable_in_archive_bytes() -> None:
    B = torch.tensor(185_578.0, requires_grad=True)
    s = contest_score(seg_distortion=0.001, pose_distortion=0.0001, archive_bytes=B)
    s.backward()
    expected_grad = CONTEST_RATE_WEIGHT / CONTEST_RAW_VIDEO_BYTES
    assert B.grad is not None
    assert abs(float(B.grad) - expected_grad) < 1e-12


def test_contest_score_differentiable_in_pose_distortion() -> None:
    """Verify autograd's ∂S/∂d_pose matches the analytical sqrt(10)/(2·sqrt(d_pose))."""
    pose_val = 3.4e-5
    pose = torch.tensor(pose_val, requires_grad=True)
    s = contest_score(seg_distortion=0.001, pose_distortion=pose, archive_bytes=200_000)
    s.backward()
    expected = math.sqrt(CONTEST_POSE_WEIGHT) / (2.0 * math.sqrt(pose_val))
    assert pose.grad is not None
    assert abs(float(pose.grad) - expected) / expected < 1e-5


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_non_scalar_tensor_rejected() -> None:
    with pytest.raises(ValueError, match="scalar"):
        contest_score(
            seg_distortion=torch.tensor([0.001, 0.002]),
            pose_distortion=0.0001,
            archive_bytes=200_000,
        )
