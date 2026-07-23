# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.canonical_equations.ddm_road_frame1_reach_curve_20260723 import (
    RoadReachPoint,
    certified_residual_interval,
    normalized_chord_knee,
    receiver_closed_reach_curve,
)


def _point(state: str, size: int, errors: int) -> RoadReachPoint:
    return RoadReachPoint(size, 100, errors, 1.0, state)


def test_curve_is_exact_byte_ordered_pareto_envelope() -> None:
    curve = receiver_closed_reach_curve(
        (_point("dominated", 11, 96), _point("base", 10, 100), _point("win", 12, 90))
    )
    assert [row.state_id for row in curve] == ["base", "dominated", "win"]
    assert receiver_closed_reach_curve((*curve, _point("worse", 13, 91))) == curve


def test_knee_uses_normalized_distance_to_chord() -> None:
    points = (_point("base", 10, 100), _point("knee", 12, 60), _point("end", 20, 50))
    assert normalized_chord_knee(points).state_id == "knee"


def test_nonexhaustive_residual_is_interval_not_point_claim() -> None:
    assert certified_residual_interval(
        control_errors=100,
        measured_reachable_errors_closed=30,
        exhaustive_reachable_set=False,
    ) == (0, 70)
    assert certified_residual_interval(
        control_errors=100,
        measured_reachable_errors_closed=30,
        exhaustive_reachable_set=True,
    ) == (70, 70)
