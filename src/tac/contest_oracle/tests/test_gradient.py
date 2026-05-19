# SPDX-License-Identifier: MIT
"""Tests for ``tac.contest_oracle.gradient`` -- closed-form score gradient."""
from __future__ import annotations

import math

import pytest

from tac.contest_oracle.gradient import (
    ContestGradientError,
    ContestScoreGradient,
    compute_score,
    compute_score_gradient,
)


def test_gradient_at_frontier_operating_point_matches_claude_md_anchor():
    """At PR106 frontier (pose_avg=3.4e-5), dS/d_pose ~= 271 per CLAUDE.md."""
    g = compute_score_gradient(d_seg=0.001, d_pose=3.4e-5, archive_bytes=500_000)
    # Per CLAUDE.md SegNet-vs-PoseNet operating-point-dependent rule:
    # dS/d_pose = 5/sqrt(10 * 3.4e-5) ~= 271
    assert 270 < g.dS_d_pose < 273
    # Seg marginal is constant 100
    assert g.dS_d_seg == 100.0
    # Per-byte marginal ~6.66e-7
    assert 6.0e-7 < g.dS_d_byte < 7.0e-7


def test_gradient_at_old_1x_point_seg_dominant():
    """At OLD 1.x point (pose_avg=0.18), dS/d_pose ~= 11.8 (seg is 77x more)."""
    g = compute_score_gradient(d_seg=0.1, d_pose=0.18, archive_bytes=500_000)
    # 5 / sqrt(10*0.18) = 5 / sqrt(1.8) ~= 3.726
    expected = 5.0 / math.sqrt(1.8)
    assert math.isclose(g.dS_d_pose, expected, rel_tol=1e-6)
    # seg/pose ratio ~ 100 / 3.7 ~= 27 (close to OLD 1.x ratio)
    assert g.dS_d_seg / g.dS_d_pose > 20


def test_gradient_dS_d_pose_diverges_at_zero():
    """dS/d_pose = 5/sqrt(10*d_pose) -> inf as d_pose -> 0."""
    g = compute_score_gradient(d_seg=0.001, d_pose=0.0, archive_bytes=500_000)
    assert g.dS_d_pose == math.inf
    # Other marginals still finite
    assert g.dS_d_seg == 100.0
    assert math.isfinite(g.dS_d_byte)


def test_compute_score_matches_closed_form():
    """S = 100*d_seg + sqrt(10*d_pose) + 25*R."""
    s = compute_score(0.01, 0.0001, 1_000_000)
    expected = 100 * 0.01 + math.sqrt(10 * 0.0001) + 25 * 1_000_000 / 37_545_489
    assert math.isclose(s, expected, rel_tol=1e-9)


def test_compute_score_zero_distortion_zero_bytes_zero():
    """Trivial sanity: zero everything -> zero score."""
    assert compute_score(0.0, 0.0, 0) == 0.0


def test_gradient_dataclass_round_trip():
    g = compute_score_gradient(d_seg=0.001, d_pose=0.001, archive_bytes=500_000)
    assert isinstance(g, ContestScoreGradient)
    assert g.d_seg_observed == 0.001
    assert g.d_pose_observed == 0.001
    assert g.archive_bytes_observed == 500_000


def test_gradient_rejects_negative_d_seg():
    with pytest.raises(ContestGradientError):
        compute_score_gradient(d_seg=-0.001, d_pose=0.001, archive_bytes=500_000)


def test_gradient_rejects_negative_d_pose():
    with pytest.raises(ContestGradientError):
        compute_score_gradient(d_seg=0.001, d_pose=-0.001, archive_bytes=500_000)


def test_gradient_rejects_negative_bytes():
    with pytest.raises(ContestGradientError):
        compute_score_gradient(d_seg=0.001, d_pose=0.001, archive_bytes=-1)


def test_compute_score_rejects_negative_inputs():
    with pytest.raises(ContestGradientError):
        compute_score(-0.001, 0.001, 500_000)
    with pytest.raises(ContestGradientError):
        compute_score(0.001, -0.001, 500_000)
    with pytest.raises(ContestGradientError):
        compute_score(0.001, 0.001, -500)


def test_gradient_score_value_matches_compute_score_helper():
    """ContestScoreGradient.score_value matches compute_score(...)."""
    g = compute_score_gradient(d_seg=0.005, d_pose=0.0002, archive_bytes=300_000)
    assert math.isclose(g.score_value, compute_score(0.005, 0.0002, 300_000), rel_tol=1e-12)
