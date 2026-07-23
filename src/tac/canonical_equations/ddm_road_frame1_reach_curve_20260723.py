# SPDX-License-Identifier: MIT
"""Receiver-closed Road frame-1 reach-curve law.

The curve is defined only by exact replayed archive states.  Subset rows may
propose a state, but a full-n600 claim requires a separate full replay.  A
measured reachable state supplies a lower bound on reach; it does not certify
that the unclosed residual is infeasible.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoadReachPoint:
    archive_bytes: int
    control_errors: int
    candidate_errors: int
    joint_objective: float
    state_id: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.archive_bytes, bool)
            or not isinstance(self.archive_bytes, int)
            or self.archive_bytes <= 0
        ):
            raise ValueError("archive_bytes must be a positive integer")
        if (
            isinstance(self.control_errors, bool)
            or not isinstance(self.control_errors, int)
            or self.control_errors < 0
            or isinstance(self.candidate_errors, bool)
            or not isinstance(self.candidate_errors, int)
            or self.candidate_errors < 0
        ):
            raise ValueError("error counts must be nonnegative integers")
        if not math.isfinite(float(self.joint_objective)) or not self.state_id:
            raise ValueError("joint_objective and state_id must be valid")

    @property
    def net_errors_closed(self) -> int:
        return self.control_errors - self.candidate_errors


def receiver_closed_reach_curve(points: Iterable[RoadReachPoint]) -> tuple[RoadReachPoint, ...]:
    """Return the byte-ordered Pareto envelope of exact replayed states.

    A point enters the envelope only when it strictly improves the best Road
    error count seen at an equal or smaller exact archive size.  Joint
    objective remains attached so callers can apply the contest acceptance
    gate; Road reach alone never authorizes admission.
    """

    ordered = sorted(points, key=lambda row: (row.archive_bytes, row.candidate_errors, row.state_id))
    result: list[RoadReachPoint] = []
    best_errors: int | None = None
    for row in ordered:
        if best_errors is None or row.candidate_errors < best_errors:
            result.append(row)
            best_errors = row.candidate_errors
    return tuple(result)


def normalized_chord_knee(points: Iterable[RoadReachPoint]) -> RoadReachPoint | None:
    """Return the maximum normalized distance-to-chord knee.

    Two points do not define an interior knee.  Degenerate byte or reach spans
    return ``None`` rather than inventing a breakpoint.
    """

    curve = receiver_closed_reach_curve(points)
    if len(curve) < 3:
        return None
    x0, x1 = curve[0].archive_bytes, curve[-1].archive_bytes
    y0, y1 = curve[0].net_errors_closed, curve[-1].net_errors_closed
    if x1 == x0 or y1 == y0:
        return None
    best: tuple[float, RoadReachPoint] | None = None
    for row in curve[1:-1]:
        x = (row.archive_bytes - x0) / (x1 - x0)
        y = (row.net_errors_closed - y0) / (y1 - y0)
        distance = y - x
        key = (distance, row)
        if best is None or key[0] > best[0]:
            best = key
    return None if best is None else best[1]


def certified_residual_interval(
    *,
    control_errors: int,
    measured_reachable_errors_closed: int,
    exhaustive_reachable_set: bool,
) -> tuple[int, int]:
    """Bound the truly infeasible residual without converting search debt to proof."""

    if (
        isinstance(control_errors, bool)
        or not isinstance(control_errors, int)
        or control_errors < 0
        or isinstance(measured_reachable_errors_closed, bool)
        or not isinstance(measured_reachable_errors_closed, int)
        or not 0 <= measured_reachable_errors_closed <= control_errors
    ):
        raise ValueError("invalid residual-bound counts")
    remaining = control_errors - measured_reachable_errors_closed
    return (remaining, remaining) if exhaustive_reachable_set else (0, remaining)


__all__ = [
    "RoadReachPoint",
    "certified_residual_interval",
    "normalized_chord_knee",
    "receiver_closed_reach_curve",
]
