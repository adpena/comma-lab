# SPDX-License-Identifier: MIT
"""Impl 2 -- operating-point-dependent phase classifier (NEW; genuinely missing).

Classifies a contest archive into one of 4 canonical phases based on the
relative magnitudes of ``dS/d(d_seg)`` (constant 100) vs
``dS/d(d_pose) = 5/sqrt(10*d_pose)``, which crosses 100 at ``pose_avg = 2.5e-4``.

The 4 phases:
  - ``SEG_DOMINANT_OLD_1X``: pose_avg >= ~0.01 (OLD 1.x scores; seg ~77x more
    important per CLAUDE.md ``SegNet vs PoseNet importance`` heuristic).
  - ``MID_TRANSITION``: 2.5e-4 < pose_avg < 0.01.
  - ``CROSSOVER``: pose_avg ~= 2.5e-4 (marginal-equal point where ``dS/d_seg ==
    dS/d_pose == 100``).
  - ``POSE_DOMINANT_FRONTIER``: pose_avg < 2.5e-4 (PR106 frontier; pose ~2.71x
    more important; the marginal-value-per-byte should target pose-axis lanes
    first per CLAUDE.md operational rule).

Phase classification drives the optimal attack recommendation: at frontier
operating points pose-targeted lanes have higher marginal-value-per-byte and
should be prioritized; SegNet lanes become tertiary until pose is exhausted.
This extincts the historical "I 77x rule is universal" cargo-cult per
Catalog #303.

Citations:
  - CLAUDE.md "SegNet vs PoseNet importance -- operating-point dependent"
    -- the canonical operating-point-dependent rule + the empirical receipts
    (docs/pr97_anti_pattern_pose_vs_seg_marginal_20260504.md +
    docs/pr_family_evolution_timeline_20260504.md).
  - PR106 frontier empirical pose_avg = 3.4e-5 (7x below crossover threshold).
  - Boyd & Vandenberghe 2004 *Convex Optimization* -- gradient-direction
    arbitrates optimization priority at each operating point.

Catalog #125 hook 4 (cathedral_autopilot_dispatch): ACTIVE -- the phase
classifier returns the canonical phase that downstream autopilot rankers
condition on for per-archive optimal-attack recommendation.
Catalog #305 observability surface: inspectable_per_layer, cite_able,
counterfactual_able.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from .constants import CONTEST_POSE_SQRT_INNER, CONTEST_SEG_WEIGHT
from .gradient import compute_score_gradient

# Crossover threshold derived analytically:
#   dS/d(d_pose) = 5 / sqrt(10 * d_pose) == 100 (= dS/d_seg)
#   =>  sqrt(10 * d_pose) = 5 / 100 = 0.05
#   =>  10 * d_pose = 0.0025
#   =>  d_pose = 2.5e-4
CROSSOVER_POSE_AVG: Final[float] = 2.5e-4
"""Pose-vs-seg marginal-equal crossover at ``d_pose = 2.5e-4`` (closed-form)."""

# Boundaries: the operating-point-dependent rule from CLAUDE.md says the
# OLD 1.x heuristic (77x seg-dominant) holds at pose_avg >= 0.01-ish; below
# that the marginal-flip starts. We use 4 phases with closed-form boundaries.
# These match the empirical anchors documented in CLAUDE.md.
PHASE_BOUNDARY_OLD_1X_TO_MID: Final[float] = 1.0e-2
"""``pose_avg >= 1e-2`` -> ``SEG_DOMINANT_OLD_1X`` phase."""

PHASE_BOUNDARY_MID_TO_CROSSOVER: Final[float] = 3.0e-4
"""``pose_avg <= 3e-4`` -> close to crossover; ``MID_TRANSITION`` ends."""

PHASE_BOUNDARY_CROSSOVER_TO_FRONTIER: Final[float] = 2.0e-4
"""``pose_avg <= 2e-4`` -> ``POSE_DOMINANT_FRONTIER`` phase."""


class ContestPhase(StrEnum):
    """4 canonical contest-operating-point phases."""

    SEG_DOMINANT_OLD_1X = "seg_dominant_old_1x"
    """OLD 1.x scores; pose_avg ~ 0.01+; seg ~77x more important per marginal.

    Optimal attack: SegNet-targeted lanes (per-class FiLM, mask-codec, etc.)."""

    MID_TRANSITION = "mid_transition"
    """Pose marginal climbing; both axes contributing comparably.

    Optimal attack: BALANCED -- both seg and pose lanes; favor seg slightly."""

    CROSSOVER = "crossover"
    """Near marginal-equal point (pose_avg ~= 2.5e-4).

    Optimal attack: BALANCED -- pose lanes start to become higher-EV per byte."""

    POSE_DOMINANT_FRONTIER = "pose_dominant_frontier"
    """PR106 frontier; pose_avg < 2.5e-4; pose ~2.71x more important per marginal.

    Optimal attack: POSE-targeted lanes (latent sidecars, pixel translation
    sidechannels, multi-stage training)."""


class OptimalAttackRecommendation(StrEnum):
    """Canonical attack-prioritization recommendations per phase."""

    SEG_FIRST_TERTIARY_POSE = "seg_first_tertiary_pose"
    BALANCED_FAVOR_SEG = "balanced_favor_seg"
    BALANCED_PARITY = "balanced_parity"
    POSE_FIRST_TERTIARY_SEG = "pose_first_tertiary_seg"


_PHASE_TO_RECOMMENDATION: Final[dict[ContestPhase, OptimalAttackRecommendation]] = {
    ContestPhase.SEG_DOMINANT_OLD_1X: OptimalAttackRecommendation.SEG_FIRST_TERTIARY_POSE,
    ContestPhase.MID_TRANSITION: OptimalAttackRecommendation.BALANCED_FAVOR_SEG,
    ContestPhase.CROSSOVER: OptimalAttackRecommendation.BALANCED_PARITY,
    ContestPhase.POSE_DOMINANT_FRONTIER: OptimalAttackRecommendation.POSE_FIRST_TERTIARY_SEG,
}


@dataclass(frozen=True, slots=True)
class PhaseClassification:
    """Phase classification of one operating point."""

    d_pose_observed: float
    """Pose distortion driving the classification."""

    phase: ContestPhase
    """Classified phase."""

    optimal_attack: OptimalAttackRecommendation
    """Canonical attack-prioritization recommendation per phase."""

    pose_to_seg_marginal_ratio: float
    """``(dS/d_pose) / (dS/d_seg)``; > 1 means pose has higher marginal."""

    crossover_distance: float
    """Signed distance ``log10(d_pose / 2.5e-4)``; negative -> frontier side."""


def classify_phase(*, d_pose: float) -> PhaseClassification:
    """Classify an operating point into one of the 4 canonical phases.

    Args:
        d_pose: Pose distortion ``d_pose >= 0``.

    Returns:
        ``PhaseClassification`` with phase + recommendation + diagnostic ratios.

    Raises:
        ValueError: if ``d_pose < 0``.
    """
    if d_pose < 0:
        raise ValueError(f"d_pose must be >= 0 (got {d_pose})")

    # Closed-form boundary classification
    if d_pose >= PHASE_BOUNDARY_OLD_1X_TO_MID:
        phase = ContestPhase.SEG_DOMINANT_OLD_1X
    elif d_pose >= PHASE_BOUNDARY_MID_TO_CROSSOVER:
        phase = ContestPhase.MID_TRANSITION
    elif d_pose >= PHASE_BOUNDARY_CROSSOVER_TO_FRONTIER:
        phase = ContestPhase.CROSSOVER
    else:
        phase = ContestPhase.POSE_DOMINANT_FRONTIER

    # Diagnostic ratio: dS/d_pose vs dS/d_seg
    if d_pose <= 0.0:
        ratio = math.inf
    else:
        ds_dpose = 5.0 / math.sqrt(CONTEST_POSE_SQRT_INNER * d_pose)
        ratio = ds_dpose / CONTEST_SEG_WEIGHT

    # Signed log-distance from crossover (closed-form)
    if d_pose <= 0.0:
        crossover_distance = -math.inf
    else:
        crossover_distance = math.log10(d_pose / CROSSOVER_POSE_AVG)

    return PhaseClassification(
        d_pose_observed=float(d_pose),
        phase=phase,
        optimal_attack=_PHASE_TO_RECOMMENDATION[phase],
        pose_to_seg_marginal_ratio=float(ratio),
        crossover_distance=float(crossover_distance),
    )


def recommend_attack(*, d_seg: float, d_pose: float, archive_bytes: int) -> OptimalAttackRecommendation:
    """Convenience: full operating point -> canonical attack recommendation.

    Composes ``compute_score_gradient`` (Impl 1) with ``classify_phase``
    (this module) so callers can get the canonical attack recommendation
    in one line.
    """
    # Surface the gradient explicitly so the call site can inspect both
    # the classification AND the gradient that drove it (Catalog #305).
    _grad = compute_score_gradient(
        d_seg=d_seg, d_pose=d_pose, archive_bytes=archive_bytes
    )
    return classify_phase(d_pose=d_pose).optimal_attack


__all__ = [
    "CROSSOVER_POSE_AVG",
    "PHASE_BOUNDARY_OLD_1X_TO_MID",
    "PHASE_BOUNDARY_MID_TO_CROSSOVER",
    "PHASE_BOUNDARY_CROSSOVER_TO_FRONTIER",
    "ContestPhase",
    "OptimalAttackRecommendation",
    "PhaseClassification",
    "classify_phase",
    "recommend_attack",
]
