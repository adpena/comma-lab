# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.optimization.trajectory_stopping import (
    StaircaseStopConfig,
    TrajectoryPoint,
    TrajectoryStopConfig,
    allocate_adaptive_depths,
    build_cap_stop_receipt,
    byte_score_units,
    evaluate_staircase_aware_stop,
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
    return tuple(TrajectoryPoint(compute=float(s), objective=float(v)) for s, v in _SQ1_PROXY_CURVE if s <= step)


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


def _m1_staircase_config() -> StaircaseStopConfig:
    # DERIVED test geometry: five 50-step eval intervals form the event-free horizon.
    return StaircaseStopConfig(
        min_eval_rows=5,
        window_rows=5,
        event_free_horizon_compute=250.0,
        event_score_delta=4.238552517361111e-6,
        creep_score_per_compute=2.0e-6,
        sustained_erosion_windows=3,
    )


def _m1_trajectory_config() -> TrajectoryStopConfig:
    return TrajectoryStopConfig(
        score_units_per_objective=1.0,
        marginal_score_gain_per_compute=8.477105034722223e-8,
        min_fit_points=5,
    )


def _m1_rows(objectives: list[float], losses: list[float]) -> list[dict[str, float | int]]:
    assert len(objectives) == len(losses)
    return [
        {
            "step": 50 * index,
            "objective_S": objective,
            "loss": loss,
            "weights_stepped": 50 * index,
            "accepted_batch_fraction": 1.0,
        }
        for index, (objective, loss) in enumerate(zip(objectives, losses, strict=True))
    ]


def test_staircase_plateau_then_drop_does_not_stop() -> None:
    # BAD-state positive control: a smooth-fit plateau immediately followed by an
    # event must stay open because its event-free horizon has restarted.
    rows = _m1_rows(
        [0.10, 0.10, 0.10, 0.10, 0.10, 0.09],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    decision = evaluate_staircase_aware_stop(
        rows,
        _m1_trajectory_config(),
        _m1_staircase_config(),
    )
    assert decision.action == "CONTINUE"
    assert "event_free_horizon_not_met" in decision.blockers


def test_staircase_flat_objective_with_falling_loss_does_not_stop() -> None:
    # BAD-state positive control: objective quantization cannot hide live loss descent.
    rows = _m1_rows(
        [0.10] * 6,
        [1.0, 0.95, 0.90, 0.85, 0.80, 0.75],
    )
    decision = evaluate_staircase_aware_stop(
        rows,
        _m1_trajectory_config(),
        _m1_staircase_config(),
    )
    assert decision.action == "CONTINUE"
    assert "loss_tail_not_flat" in decision.blockers


def test_staircase_flat_live_event_free_trace_can_stop() -> None:
    # GOOD-state silence control: the extra guards do not suppress a resolved,
    # event-free, live plateau.
    rows = _m1_rows([0.10] * 6, [1.0] * 6)
    decision = evaluate_staircase_aware_stop(
        rows,
        _m1_trajectory_config(),
        _m1_staircase_config(),
    )
    assert decision.action == "STOP_CONVERGED"
    assert decision.should_halt is True
    assert decision.blockers == ()


def test_staircase_frozen_weights_cannot_certify_convergence() -> None:
    rows = _m1_rows([0.10] * 6, [1.0] * 6)
    for row in rows:
        row["weights_stepped"] = 0
        row["accepted_batch_fraction"] = 0.0
    decision = evaluate_staircase_aware_stop(
        rows,
        _m1_trajectory_config(),
        _m1_staircase_config(),
    )
    assert decision.action == "CONTINUE"
    assert "weight_update_liveness_not_clear" in decision.blockers


def test_wall_clock_cap_receipt_fires_only_at_the_bound() -> None:
    receipt = build_cap_stop_receipt(
        stop_reason="cap_bound",
        steps_run=900,
        cap=None,
        still_descending=True,
        bound_kind="wall_clock_seconds",
        bound_value=28_800.0,
        observed_value=28_800.5,
    )
    assert receipt.to_payload()["bound_kind"] == "wall_clock_seconds"
    with pytest.raises(Exception, match="observed_value >= bound_value"):
        build_cap_stop_receipt(
            stop_reason="cap_bound",
            steps_run=899,
            cap=None,
            still_descending=True,
            bound_kind="wall_clock_seconds",
            bound_value=28_800.0,
            observed_value=28_799.9,
        )


# --- amendment-3 tail-slope adjudication (#874/#935 censored-cap genus) ---------------


def test_tail_slope_censored_still_descending() -> None:
    from tac.optimization.trajectory_stopping import adjudicate_tail_slope

    # the w2 shape: linear descent ~-3e-6/ep with tiny alternating noise -> censored
    steps = [949 + 5 * i for i in range(28)]
    values = [0.00415 - 3e-6 * (s - 949) + (2e-7 if i % 2 else -2e-7) for i, s in enumerate(steps)]
    v = adjudicate_tail_slope(steps, values)
    assert v.verdict == "censored_still_descending"
    assert v.endpoint_is_min
    assert any(f.slope < 0 and f.sigma >= v.sigma_threshold for f in v.fits)
    payload = v.to_payload()
    assert payload["verdict"] == "censored_still_descending"
    assert payload["fits"]


def test_tail_slope_converged_plateau() -> None:
    from tac.optimization.trajectory_stopping import adjudicate_tail_slope

    # flat with noise larger than any drift -> no fit clears the threshold
    steps = [float(5 * i) for i in range(30)]
    values = [0.004 + (3e-5 if i % 2 else -3e-5) for i in range(30)]
    v = adjudicate_tail_slope(steps, values)
    assert v.verdict == "converged_plateau"


def test_tail_slope_ascending_past_min() -> None:
    from tac.optimization.trajectory_stopping import adjudicate_tail_slope

    # the OFF-arm shape: descend to a minimum, then a clean ascent past it -> the
    # endpoint is NOT adoptable; the verdict must say so and record the minimum.
    steps = [float(5 * i) for i in range(30)]
    values = [0.004 - 4e-6 * s for s in steps[:15]]
    vmin = values[-1]
    values += [vmin + 4e-6 * (s - steps[14]) for s in steps[15:]]
    v = adjudicate_tail_slope(steps, values)
    assert v.verdict == "ascending_past_min"
    assert not v.endpoint_is_min
    assert v.min_value == pytest.approx(vmin)


def test_tail_slope_insufficient_points_raises() -> None:
    from tac.optimization.trajectory_stopping import (
        TrajectoryStoppingError,
        adjudicate_tail_slope,
    )

    with pytest.raises(TrajectoryStoppingError, match=">= 3"):
        adjudicate_tail_slope([0.0, 1.0], [1.0, 0.5])


def test_tail_slope_short_history_falls_back_to_full_fit() -> None:
    from tac.optimization.trajectory_stopping import adjudicate_tail_slope

    # 4 points spanning 300 steps: both default spans (40/20) hold <3 points each,
    # so the adjudicator must fall back to ONE full-trajectory fit, not error.
    steps = [0.0, 100.0, 200.0, 300.0]
    values = [0.01, 0.008, 0.006, 0.004]
    v = adjudicate_tail_slope(steps, values)
    assert v.verdict == "censored_still_descending"
    assert len(v.fits) == 1
