# SPDX-License-Identifier: MIT
"""Impl 6 + Impl 7 -- sqrt(10) pose curvature + frontier-free-real-estate theorem.

Surfaces the canonical pose-axis curvature ``sqrt(10*d_pose)`` as a named
helper that substrate trainers can consume drop-in. The existing canonical
sister ``tac.substrates.score_aware_common.CONTEST_POSE_SQRT_WEIGHT`` is the
constant; this module provides the FUNCTION that applies it.

Empirical anchor (Impl 7 theorem): at PR106 frontier pose_avg = 3.4e-5,
the marginal-pose-EV per byte EXCEEDS marginal-rate-EV per byte by
~2.71x (per CLAUDE.md "SegNet vs PoseNet importance" operating-point-
dependent rule). The implication is: ALL training compute at the
frontier should target pose-axis loss until pose_avg climbs back above
the 2.5e-4 crossover threshold.

Citations:
  - ``upstream/evaluate.py`` -- canonical sqrt(10*pose) formula.
  - ``tac.substrates.score_aware_common.CONTEST_POSE_SQRT_WEIGHT`` --
    canonical sister constant (= sqrt(10) ~= 3.1622776601683795).
  - CLAUDE.md "SegNet vs PoseNet importance" -- operating-point-dependent
    crossover at pose_avg ~= 2.5e-4.
  - PR106 frontier pose_avg = 3.4e-5 empirical anchor.

Catalog #125 hook 1 (sensitivity_map): ACTIVE -- the pose-axis canonical
curvature is the sensitivity-map's pose-axis kernel.
Catalog #305 observability surface: cite_able, decomposable_per_signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import CONTEST_POSE_SQRT_INNER, CONTEST_POSE_SQRT_WEIGHT
from .phase_classifier import CROSSOVER_POSE_AVG


class PoseAxisError(ValueError):
    """Raised when an invalid pose-axis input is passed."""


@dataclass(frozen=True, slots=True)
class PoseAxisAnalysis:
    """Pose-axis canonical analysis at one operating point."""

    d_pose_observed: float
    """Pose distortion at the operating point."""

    pose_contribution: float
    """``sqrt(10 * d_pose)`` -- the contribution to the contest score."""

    pose_marginal_dS_d_dpose: float
    """``5 / sqrt(10 * d_pose)`` -- the marginal sensitivity to d_pose."""

    is_below_crossover: bool
    """True if ``d_pose < CROSSOVER_POSE_AVG``; pose-axis becomes higher-EV."""

    is_at_frontier: bool
    """True if ``d_pose < 1e-4``; PR106-frontier operating regime."""


def contest_curvature_pose_loss(d_pose: float) -> float:
    """Canonical contest-curvature pose loss: ``sqrt(10 * d_pose)``.

    Drop-in replacement for ``MSE(pose)`` in substrate score-aware loss.
    The sqrt curvature MATCHES the contest formula; MSE is wrong-curvature
    at the frontier and produces gradient-direction mismatch per CLAUDE.md
    "HNeRV / leaderboard-implementation parity discipline" L6.

    Args:
        d_pose: Pose distortion >= 0.

    Returns:
        ``sqrt(10 * d_pose)`` scalar contribution to contest score.

    Raises:
        PoseAxisError: if d_pose < 0.
    """
    if d_pose < 0:
        raise PoseAxisError(f"d_pose must be >= 0 (got {d_pose})")
    return math.sqrt(CONTEST_POSE_SQRT_INNER * d_pose)


def analyze_pose_axis(*, d_pose: float) -> PoseAxisAnalysis:
    """Per-operating-point pose-axis canonical analysis.

    Reveals:
      - The contribution to the contest score (``sqrt(10*d_pose)``).
      - The marginal sensitivity (``5/sqrt(10*d_pose)``).
      - Whether we are below the marginal-flip crossover.
      - Whether we are at frontier (PR106 regime).
    """
    if d_pose < 0:
        raise PoseAxisError(f"d_pose must be >= 0 (got {d_pose})")
    return PoseAxisAnalysis(
        d_pose_observed=float(d_pose),
        pose_contribution=contest_curvature_pose_loss(d_pose),
        pose_marginal_dS_d_dpose=(
            math.inf if d_pose <= 0.0
            else 5.0 / math.sqrt(CONTEST_POSE_SQRT_INNER * d_pose)
        ),
        is_below_crossover=(d_pose < CROSSOVER_POSE_AVG),
        is_at_frontier=(d_pose < 1.0e-4),
    )


# Re-export the canonical constant so contest_oracle is the single-stop API
__all__ = [
    "CONTEST_POSE_SQRT_INNER",
    "CONTEST_POSE_SQRT_WEIGHT",
    "PoseAxisAnalysis",
    "PoseAxisError",
    "analyze_pose_axis",
    "contest_curvature_pose_loss",
]
