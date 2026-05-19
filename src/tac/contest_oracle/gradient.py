# SPDX-License-Identifier: MIT
"""Impl 1 -- closed-form contest-score gradient oracle.

Thin wrapper that SURFACES the analytical contest-formula derivatives
(``dS/d(d_seg)``, ``dS/d(d_pose)``, ``dS/d(byte)``) as the canonical
ANALYTICAL anchor. Existing ``tac.master_gradient`` surfaces empirical
anchors; this module is the analytical sister.

Citations:
  - Boyd & Vandenberghe 2004 *Convex Optimization* Section 5.1 -- Lagrangian
    primal-dual: the contest scoring formula IS the dual; we work the primal.
  - ``upstream/evaluate.py`` (pinned snapshot) -- authoritative score formula.
  - ``tac.master_gradient.compute_marginal_coefficients`` -- sister helper that
    computes the same marginals from an ``OperatingPoint`` dataclass; this
    module exposes the bare-float interface for thin-wrapper composition.
  - Operator standing directive 2026-05-18 verbatim *"the contest information
    defines the problem and the solution... or the path to the solution"*.

Score formula (per upstream/evaluate.py):

    S(d_seg, d_pose, R) = 100 * d_seg + sqrt(10 * d_pose) + 25 * R

where ``R = archive_bytes / 37_545_489`` is the normalized rate term.

Analytical gradients:

    dS/d(d_seg)  = 100                                    (constant marginal)
    dS/d(d_pose) = 5 / sqrt(10 * d_pose)                  (hyperbolic, diverges -> 0)
    dS/d(R)      = 25                                     (constant)
    dS/d(bytes)  = 25 / 37_545_489 ~= 6.66e-7             (per-byte; constant)

The pose-axis marginal divergence is the canonical reason the PR106 frontier
operating point (pose_avg ~ 3.4e-5) flips the SegNet-vs-PoseNet importance
ratio from ``77x seg-dominant`` at the OLD 1.x point to ``2.71x pose-dominant``
at the frontier.

Catalog #125 hook 1 (sensitivity_map): ACTIVE -- every analytical marginal
contribution can be consumed by ``tac.sensitivity_map.*`` downstream
consumers as the canonical analytical anchor.
Catalog #125 hook 2 (pareto_constraint): ACTIVE -- the gradient defines
the canonical Pareto-front parameterization (sister Impl 3).
Catalog #305 observability surface: cite_able, decomposable_per_signal,
counterfactual_able.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import (
    CONTEST_POSE_SQRT_INNER,
    CONTEST_RATE_DENOM_BYTES,
    CONTEST_RATE_WEIGHT,
    CONTEST_SEG_WEIGHT,
)


class ContestGradientError(ValueError):
    """Raised when an invalid operating point is passed (e.g. ``d_pose < 0``)."""


@dataclass(frozen=True, slots=True)
class ContestScoreGradient:
    """Closed-form contest-score gradient at one operating point.

    All three marginals are scalar floats. ``dS_d_pose`` diverges as
    ``d_pose -> 0`` so the dataclass clamps to ``inf`` rather than
    blowing up; downstream consumers should check ``math.isfinite`` and
    apply their own regularization at boundary cases.
    """

    d_seg_observed: float
    """Current segmentation distortion."""

    d_pose_observed: float
    """Current pose distortion."""

    archive_bytes_observed: int
    """Current archive byte count."""

    dS_d_seg: float
    """Marginal ``dS/d(d_seg)`` (constant = 100)."""

    dS_d_pose: float
    """Marginal ``dS/d(d_pose) = 5 / sqrt(10 * d_pose)``; may be ``+inf``."""

    dS_d_byte: float
    """Marginal ``dS/d(archive_bytes) = 25 / 37_545_489``."""

    score_value: float
    """Current ``S = 100*d_seg + sqrt(10*d_pose) + 25*(bytes/DENOM)``."""


def compute_score_gradient(
    *,
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
) -> ContestScoreGradient:
    """Compute analytical contest-score gradient at the operating point.

    Args:
        d_seg: SegNet per-pixel distortion (mean argmax disagreement rate).
        d_pose: PoseNet per-pair distortion (MSE on first 6 pose dims).
        archive_bytes: Submission archive byte count.

    Returns:
        ``ContestScoreGradient`` with all three marginals + the current score.

    Raises:
        ContestGradientError: if any input is negative.
    """
    if d_seg < 0:
        raise ContestGradientError(f"d_seg must be >= 0 (got {d_seg})")
    if d_pose < 0:
        raise ContestGradientError(f"d_pose must be >= 0 (got {d_pose})")
    if archive_bytes < 0:
        raise ContestGradientError(
            f"archive_bytes must be >= 0 (got {archive_bytes})"
        )

    # Marginal 1: constant for the seg axis.
    dS_d_seg = CONTEST_SEG_WEIGHT

    # Marginal 2: pose; diverges as d_pose -> 0.
    if d_pose <= 0.0:
        dS_d_pose = math.inf
    else:
        # d/d_pose [ sqrt(10 * d_pose) ] = 5 / sqrt(10 * d_pose)
        dS_d_pose = 5.0 / math.sqrt(CONTEST_POSE_SQRT_INNER * d_pose)

    # Marginal 3: bytes (constant).
    dS_d_byte = CONTEST_RATE_WEIGHT / float(CONTEST_RATE_DENOM_BYTES)

    # Current score (closed-form contest formula).
    rate_term = CONTEST_RATE_WEIGHT * archive_bytes / float(CONTEST_RATE_DENOM_BYTES)
    pose_term = math.sqrt(CONTEST_POSE_SQRT_INNER * d_pose)
    seg_term = CONTEST_SEG_WEIGHT * d_seg
    score_value = seg_term + pose_term + rate_term

    return ContestScoreGradient(
        d_seg_observed=float(d_seg),
        d_pose_observed=float(d_pose),
        archive_bytes_observed=int(archive_bytes),
        dS_d_seg=dS_d_seg,
        dS_d_pose=dS_d_pose,
        dS_d_byte=dS_d_byte,
        score_value=score_value,
    )


def compute_score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    """Convenience: contest score at the operating point (closed-form).

    Sister of ``compute_score_gradient(...).score_value`` for callers that
    only want the scalar score.
    """
    if d_seg < 0 or d_pose < 0 or archive_bytes < 0:
        raise ContestGradientError(
            f"Negative input: d_seg={d_seg}, d_pose={d_pose}, "
            f"archive_bytes={archive_bytes}"
        )
    return (
        CONTEST_SEG_WEIGHT * d_seg
        + math.sqrt(CONTEST_POSE_SQRT_INNER * d_pose)
        + CONTEST_RATE_WEIGHT * archive_bytes / float(CONTEST_RATE_DENOM_BYTES)
    )


__all__ = [
    "ContestGradientError",
    "ContestScoreGradient",
    "compute_score",
    "compute_score_gradient",
]
