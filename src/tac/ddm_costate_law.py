# SPDX-License-Identifier: MIT
"""Dependency-light implementation of the canonical DDM costate laws."""

from __future__ import annotations

import math

EQUATION_ID = "ddm_joint_recursion_costate_d2_v1"
SCHEDULER_EQUATION_ID = "ddm_topological_gauss_southwell_validity_v1"
RATE_BREAK_EVEN_SCORE_PER_BYTE = 25.0 / 37_545_489.0


def _nonnegative_finite(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return out


def ddm_joint_costate(
    exact_gap: float,
    visibility: float,
    uint8_realizability: float,
    byte_price: float,
    dual_tolerance_d2: float,
) -> float:
    """Return ``gap * visibility * realizability * price * D2``."""

    factors = (
        _nonnegative_finite("exact_gap", exact_gap),
        _nonnegative_finite("visibility", visibility),
        _nonnegative_finite("uint8_realizability", uint8_realizability),
        _nonnegative_finite("byte_price", byte_price),
        _nonnegative_finite("dual_tolerance_d2", dual_tolerance_d2),
    )
    if factors[1] > 1.0 or factors[2] > 1.0 or factors[4] > 1.0:
        raise ValueError("visibility, uint8_realizability, and dual_tolerance_d2 must be <= 1")
    return math.prod(factors)


def gauss_southwell_validity_score(lambda_abs: float, validity_radius: float) -> float:
    """Return the within-frontier block priority ``|lambda| * validity_radius``."""

    return _nonnegative_finite("lambda_abs", lambda_abs) * _nonnegative_finite("validity_radius", validity_radius)


def realized_pair_distortion_delta(
    *,
    d_seg_before: float,
    d_seg_after: float,
    d_pose_before: float,
    d_pose_after: float,
) -> float:
    """Exact local Seg/Pose delta, excluding unallocated shared archive bytes."""

    values = tuple(float(v) for v in (d_seg_before, d_seg_after, d_pose_before, d_pose_after))
    if not all(math.isfinite(v) for v in values):
        raise ValueError("pair distortion values must be finite")
    if values[2] < 0.0 or values[3] < 0.0:
        raise ValueError("d_pose must be nonnegative")
    return 100.0 * (values[1] - values[0]) + (math.sqrt(10.0 * values[3]) - math.sqrt(10.0 * values[2]))
