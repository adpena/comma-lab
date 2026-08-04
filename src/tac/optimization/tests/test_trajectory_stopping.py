# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.optimization.trajectory_stopping import (
    TrajectoryPoint,
    TrajectoryStopConfig,
    allocate_adaptive_depths,
    byte_score_units,
    evaluate_trajectory_stop,
    projection_interval,
    seg_flip_score_units,
)


_SQ1_PROXY_CURVE = (
    (0, 27_084),
    (5, 15_237),
    (10, 11_784),
    (15, 10_242),
    (20, 9_151),
    (25, 8_553),
    (30, 8_030),
    (35, 7_660),
    (40, 7_366),
    (45, 7_118),
    (50, 6_861),
)


def _sq1_config() -> TrajectoryStopConfig:
    return TrajectoryStopConfig(
        score_units_per_objective=seg_flip_score_units(),
        marginal_score_gain_per_compute=byte_score_units(),
    )


def _points_through(step: int) -> tuple[TrajectoryPoint, ...]:
    return tuple(
        TrajectoryPoint(compute=float(s), objective=float(v))
        for s, v in _SQ1_PROXY_CURVE
        if s <= step
    )


def test_sq1_25_and_50_step_positive_controls_report_caps_not_convergence() -> None:
    cfg = _sq1_config()
    for step in (25, 50):
        decision = evaluate_trajectory_stop(
            _points_through(step),
            cfg,
            safety_bound_compute=float(step),
        )
        assert decision.stop_reason == "safety_bound_REPORTED"
        assert decision.bound_reported is True
        assert decision.projected_remaining_score_gain is not None
        assert decision.projected_remaining_score_gain > decision.threshold_score_gain_per_compute
        assert decision.marginal_score_gain_per_compute > decision.threshold_score_gain_per_compute


def test_sq1_prefix_25_projects_the_measured_step_50_value() -> None:
    cfg = _sq1_config()
    interval = projection_interval(_points_through(25), cfg, target_compute=50.0)

    measured_step50_proxy_flips = 6_861
    assert interval.objective_low <= measured_step50_proxy_flips <= interval.objective_high
    assert set(interval.fits_used) == {"geometric", "power_law"}


def test_semantic_stop_requires_a_score_unit_tail_not_merely_a_cap() -> None:
    cfg = TrajectoryStopConfig(
        score_units_per_objective=1.0,
        marginal_score_gain_per_compute=0.25,
        min_fit_points=4,
    )
    decision = evaluate_trajectory_stop(
        (
            TrajectoryPoint(0.0, 100.0),
            TrajectoryPoint(1.0, 60.0),
            TrajectoryPoint(2.0, 59.96),
            TrajectoryPoint(3.0, 59.94),
            TrajectoryPoint(4.0, 59.93),
        ),
        cfg,
    )
    assert decision.should_stop is True
    assert decision.stop_reason in {"converged_projected", "marginal_below_bar"}


def test_adaptive_depth_waterfills_open_items_and_reports_caps() -> None:
    cfg = _sq1_config()
    open_decision = evaluate_trajectory_stop(
        _points_through(25),
        cfg,
        safety_bound_compute=25.0,
    )
    stopped_decision = evaluate_trajectory_stop(
        (
            TrajectoryPoint(0.0, 10.0),
            TrajectoryPoint(1.0, 9.0),
            TrajectoryPoint(2.0, 8.99),
            TrajectoryPoint(3.0, 8.985),
            TrajectoryPoint(4.0, 8.984),
        ),
        TrajectoryStopConfig(
            score_units_per_objective=1.0,
            marginal_score_gain_per_compute=0.25,
            min_fit_points=4,
        ),
    )
    assert stopped_decision.stop_reason in {"converged_projected", "marginal_below_bar"}

    allocations = {
        item.item_id: item
        for item in allocate_adaptive_depths(
            {"active": open_decision, "done": stopped_decision},
            total_extra_compute=5,
            safety_cap_per_item=4,
        )
    }
    assert allocations["active"].extra_compute == 4
    assert allocations["active"].safety_bound_reported is True
    assert allocations["done"].extra_compute == 0
    assert allocations["done"].projected_remaining_score_gain == 0.0


def test_malformed_thresholds_fail_closed() -> None:
    with pytest.raises(Exception, match="positive"):
        TrajectoryStopConfig(
            score_units_per_objective=seg_flip_score_units(),
            marginal_score_gain_per_compute=0.0,
        )
