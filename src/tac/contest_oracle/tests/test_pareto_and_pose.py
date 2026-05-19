# SPDX-License-Identifier: MIT
"""Tests for ``tac.contest_oracle.pareto_frontier`` + ``pose_axis_canonical``."""
from __future__ import annotations

import math

import pytest

from tac.contest_oracle.pareto_frontier import (
    AnalyticalParetoPoint,
    analytical_optimum,
    trace_pareto_frontier,
)
from tac.contest_oracle.pose_axis_canonical import (
    PoseAxisAnalysis,
    PoseAxisError,
    analyze_pose_axis,
    contest_curvature_pose_loss,
)


def test_analytical_optimum_returns_pareto_point():
    p = analytical_optimum(
        R_budget_bytes=500_000,
        architecture_d_seg_floor=0.001,
        architecture_d_pose_floor=1e-5,
    )
    assert isinstance(p, AnalyticalParetoPoint)
    assert p.R_budget_bytes == 500_000
    assert p.d_seg_analytical_optimum == 0.001
    assert p.d_pose_analytical_optimum == 1e-5


def test_analytical_optimum_evidence_grade_is_predicted():
    """Per Catalog #287/#323 -- predicted not contest-claimed."""
    p = analytical_optimum(R_budget_bytes=500_000)
    assert p.evidence_grade == "predicted_analytical"


def test_analytical_optimum_rejects_negative():
    with pytest.raises(ValueError):
        analytical_optimum(R_budget_bytes=-1)
    with pytest.raises(ValueError):
        analytical_optimum(R_budget_bytes=500_000, architecture_d_seg_floor=-0.001)
    with pytest.raises(ValueError):
        analytical_optimum(R_budget_bytes=500_000, architecture_d_pose_floor=-0.001)


def test_trace_pareto_frontier_returns_num_points():
    pts = trace_pareto_frontier(
        R_min_bytes=100_000,
        R_max_bytes=2_000_000,
        num_points=8,
        architecture_d_seg_floor=0.001,
        architecture_d_pose_floor=1e-5,
    )
    assert len(pts) == 8
    # First point at R_min, last at R_max
    assert pts[0].R_budget_bytes == 100_000
    assert pts[-1].R_budget_bytes == 2_000_000


def test_trace_pareto_frontier_rejects_bad_inputs():
    with pytest.raises(ValueError):
        trace_pareto_frontier(R_min_bytes=0, R_max_bytes=100_000)
    with pytest.raises(ValueError):
        trace_pareto_frontier(R_min_bytes=100_000, R_max_bytes=10_000)
    with pytest.raises(ValueError):
        trace_pareto_frontier(R_min_bytes=100_000, R_max_bytes=1_000_000, num_points=1)


def test_trace_pareto_frontier_log_spaced():
    """Log spacing: 4 points over [1e5, 1e8] -> {1e5, 1e6, 1e7, 1e8}."""
    pts = trace_pareto_frontier(
        R_min_bytes=100_000,
        R_max_bytes=100_000_000,
        num_points=4,
    )
    # log spacing in 10^x: 5.0, 6.0, 7.0, 8.0
    assert pts[0].R_budget_bytes == 100_000
    assert pts[1].R_budget_bytes == 1_000_000
    assert pts[2].R_budget_bytes == 10_000_000
    assert pts[3].R_budget_bytes == 100_000_000


def test_pareto_score_includes_rate_term():
    p = analytical_optimum(R_budget_bytes=1_000_000)
    expected_rate = 25.0 * 1_000_000 / 37_545_489
    assert math.isclose(p.score_at_analytical_optimum, expected_rate, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# pose_axis_canonical tests
# ---------------------------------------------------------------------------
def test_pose_curvature_loss_at_frontier_matches_contest():
    """sqrt(10*d_pose) at frontier d_pose=3.4e-5 = sqrt(3.4e-4) ~= 0.01844."""
    loss = contest_curvature_pose_loss(3.4e-5)
    assert math.isclose(loss, math.sqrt(10 * 3.4e-5), rel_tol=1e-9)
    assert 0.018 < loss < 0.019


def test_pose_curvature_loss_zero():
    assert contest_curvature_pose_loss(0.0) == 0.0


def test_pose_curvature_loss_rejects_negative():
    with pytest.raises(PoseAxisError):
        contest_curvature_pose_loss(-1.0)


def test_analyze_pose_axis_at_frontier():
    pa = analyze_pose_axis(d_pose=3.4e-5)
    assert isinstance(pa, PoseAxisAnalysis)
    assert pa.is_below_crossover is True
    assert pa.is_at_frontier is True
    # marginal at frontier: 5/sqrt(10*3.4e-5) ~= 271
    assert 270 < pa.pose_marginal_dS_d_dpose < 273


def test_analyze_pose_axis_at_old_1x_not_frontier():
    pa = analyze_pose_axis(d_pose=0.18)
    assert pa.is_below_crossover is False
    assert pa.is_at_frontier is False


def test_analyze_pose_axis_zero_pose_marginal_infinite():
    pa = analyze_pose_axis(d_pose=0.0)
    assert pa.pose_contribution == 0.0
    assert pa.pose_marginal_dS_d_dpose == math.inf


def test_analyze_pose_axis_rejects_negative():
    with pytest.raises(PoseAxisError):
        analyze_pose_axis(d_pose=-0.001)
