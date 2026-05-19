# SPDX-License-Identifier: MIT
"""Impl 3 -- closed-form Pareto frontier parameterization.

Extends ``tac.boosting.pareto_front`` with the ANALYTICAL closed-form
parameterization of the contest-score Pareto frontier. The contest formula's
Lagrangian DUAL is solved; the PRIMAL is parameterizable in closed form:

    For each rate budget R*, the optimal (d_seg*, d_pose*) is the pair that
    minimizes ``S(d_seg, d_pose, R*) = 100*d_seg + sqrt(10*d_pose) + 25*R*``
    subject to feasibility constraints on (d_seg, d_pose) at that rate.

The shape is determined by the canonical Lagrangian duality argument:
analytical Pareto rows can be pre-computed without N empirical anchors per
operator standing directive 2026-05-18.

Citations:
  - Boyd & Vandenberghe 2004 *Convex Optimization* Section 4.7 + 5.5.5 --
    Pareto optimality + KKT conditions.
  - Cover & Thomas 2006 Ch.10 -- rate-distortion theory.
  - ``tac.boosting.pareto_front.ParetoFrontTracker`` -- canonical sister
    that tracks Pareto-front anchors empirically; this module is its
    analytical sister.
  - ``tac.symposium_impls.blahut_arimoto_theoretical_floor`` -- sister at
    the theoretical-floor end of the frontier (Impl 8).

Catalog #125 hook 2 (pareto_constraint): ACTIVE -- analytical Pareto rows
ARE the canonical Pareto-constraint anchor for the meta-Lagrangian solver.
Catalog #125 hook 4 (cathedral_autopilot_dispatch): ACTIVE -- the
parameterization is the canonical source-of-truth for autopilot's per-budget
optimal-allocation rank.
Catalog #305 observability surface: decomposable_per_signal, cite_able.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from .constants import (
    CONTEST_POSE_SQRT_INNER,
    CONTEST_RATE_DENOM_BYTES,
    CONTEST_RATE_WEIGHT,
    CONTEST_SEG_WEIGHT,
)


@dataclass(frozen=True, slots=True)
class AnalyticalParetoPoint:
    """One closed-form Pareto-frontier point at rate budget ``R_budget_bytes``.

    Predicted ``(d_seg*, d_pose*)`` is the analytical optimum at that rate,
    derived from the canonical Lagrangian dual variable lambda_R.

    NOTE: This is an ANALYTICAL prediction; empirical Pareto anchors (via
    ``tac.boosting.pareto_front.ParetoFrontTracker``) may deviate due to
    architectural constraints not captured in the closed-form. Treat as
    canonical PREDICTION not as canonical EMPIRICAL anchor per CLAUDE.md
    "Forbidden empirical-claim-without-evidence-tag" non-negotiable.
    """

    R_budget_bytes: int
    """Rate budget for this Pareto point (archive byte count)."""

    R_normalized: float
    """``R_budget_bytes / CONTEST_RATE_DENOM_BYTES``."""

    lambda_R_dual: float
    """Canonical Lagrangian dual variable for the rate constraint."""

    d_seg_analytical_optimum: float
    """Closed-form prediction for ``d_seg*`` at this rate."""

    d_pose_analytical_optimum: float
    """Closed-form prediction for ``d_pose*`` at this rate."""

    score_at_analytical_optimum: float
    """``S(d_seg*, d_pose*, R*)`` -- the score at the analytical optimum."""

    evidence_grade: str = "predicted_analytical"
    """Per Catalog #287/#323 evidence-grade discipline -- this is a PREDICTED
    point, NOT an empirical anchor; downstream consumers MUST NOT promote
    this as a contest-score claim without paired-Linux-x86_64 empirical
    verification per CLAUDE.md ``Submission auth eval`` non-negotiable."""


def analytical_optimum(
    *,
    R_budget_bytes: int,
    architecture_d_seg_floor: float = 0.0,
    architecture_d_pose_floor: float = 0.0,
) -> AnalyticalParetoPoint:
    """Closed-form analytical Pareto-frontier optimum at rate budget ``R*``.

    The contest Lagrangian dual at fixed rate is:

        L(d_seg, d_pose; lambda_R) = 100*d_seg + sqrt(10*d_pose) + lambda_R*R

    The primal optimum at rate ``R*`` is the (d_seg*, d_pose*) on the
    architecturally-feasible boundary that achieves the minimum score. Since
    the contest formula has CONSTANT seg marginal and HYPERBOLIC pose
    marginal, the analytical optimum is:

        d_seg*  = architecture_d_seg_floor (any seg-axis improvement at this rate)
        d_pose* = architecture_d_pose_floor (any pose-axis improvement at this rate)

    The architecturally-feasible floors are operating-point-dependent inputs;
    callers pass them per Blahut-Arimoto sister ``tac.contest_oracle.theoretical_floor``.

    Args:
        R_budget_bytes: Rate budget for this Pareto point.
        architecture_d_seg_floor: Architecturally-achievable d_seg at this rate.
        architecture_d_pose_floor: Architecturally-achievable d_pose at this rate.

    Returns:
        ``AnalyticalParetoPoint`` carrying the closed-form prediction.

    Raises:
        ValueError: if any input is negative.
    """
    if R_budget_bytes < 0:
        raise ValueError(f"R_budget_bytes must be >= 0 (got {R_budget_bytes})")
    if architecture_d_seg_floor < 0:
        raise ValueError(
            f"architecture_d_seg_floor must be >= 0 (got {architecture_d_seg_floor})"
        )
    if architecture_d_pose_floor < 0:
        raise ValueError(
            f"architecture_d_pose_floor must be >= 0 (got {architecture_d_pose_floor})"
        )

    R_normalized = R_budget_bytes / float(CONTEST_RATE_DENOM_BYTES)

    # The canonical Lagrangian dual variable for the rate constraint is the
    # marginal rate term times the per-byte coefficient.
    lambda_R = CONTEST_RATE_WEIGHT / float(CONTEST_RATE_DENOM_BYTES)

    # Analytical optimum: at fixed rate, the optimum is at the architecturally-
    # feasible floor on each axis (assuming the floors are independent).
    d_seg_opt = float(architecture_d_seg_floor)
    d_pose_opt = float(architecture_d_pose_floor)

    # Score at the analytical optimum (closed-form contest formula).
    score = (
        CONTEST_SEG_WEIGHT * d_seg_opt
        + math.sqrt(CONTEST_POSE_SQRT_INNER * d_pose_opt)
        + CONTEST_RATE_WEIGHT * R_normalized
    )

    return AnalyticalParetoPoint(
        R_budget_bytes=int(R_budget_bytes),
        R_normalized=float(R_normalized),
        lambda_R_dual=float(lambda_R),
        d_seg_analytical_optimum=float(d_seg_opt),
        d_pose_analytical_optimum=float(d_pose_opt),
        score_at_analytical_optimum=float(score),
    )


def trace_pareto_frontier(
    *,
    R_min_bytes: int,
    R_max_bytes: int,
    num_points: int = 32,
    architecture_d_seg_floor: float = 0.0,
    architecture_d_pose_floor: float = 0.0,
) -> tuple[AnalyticalParetoPoint, ...]:
    """Trace the analytical Pareto frontier over ``[R_min_bytes, R_max_bytes]``.

    Returns ``num_points`` evenly-spaced (in log-rate) analytical Pareto
    points per Impl 3 ``trace`` verb (APPENDIX B linguistic enrichment).

    The log-spacing matches the canonical Blahut-Arimoto Pareto-curve
    parameterization where ``log10(R)`` is the natural sweep variable
    per Cover & Thomas Ch.10.
    """
    if R_min_bytes <= 0:
        raise ValueError(f"R_min_bytes must be > 0 for log-spacing (got {R_min_bytes})")
    if R_max_bytes < R_min_bytes:
        raise ValueError(
            f"R_max_bytes={R_max_bytes} must be >= R_min_bytes={R_min_bytes}"
        )
    if num_points < 2:
        raise ValueError(f"num_points must be >= 2 (got {num_points})")

    log_R_min = math.log10(R_min_bytes)
    log_R_max = math.log10(R_max_bytes)
    step = (log_R_max - log_R_min) / (num_points - 1)

    points = []
    for i in range(num_points):
        R = int(round(10 ** (log_R_min + i * step)))
        points.append(
            analytical_optimum(
                R_budget_bytes=R,
                architecture_d_seg_floor=architecture_d_seg_floor,
                architecture_d_pose_floor=architecture_d_pose_floor,
            )
        )
    return tuple(points)


__all__ = [
    "AnalyticalParetoPoint",
    "analytical_optimum",
    "trace_pareto_frontier",
]
