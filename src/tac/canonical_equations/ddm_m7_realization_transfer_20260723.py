# SPDX-License-Identifier: MIT
"""Pure score-transfer equations for the DDM M7 receiver-closed realization.

The transfer ratios in this module are instance-level diagnostics between two
explicitly named objects: the arithmetic counterfactual and the realized
177,169-byte receiver member.  They are not universal transfer coefficients
and must not be extrapolated to another archive, receiver, scorer, or hardware
axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DDM_M7_REALIZATION_TRANSFER_EQUATION_ID = (
    "ddm_m7_solve_to_realized_transfer_receiver_closed_v1"
)
DOMAIN_DECLARATION = {
    "scope": "instance_level_diagnostic",
    "objects": (
        "explicitly_named_arithmetic_counterfactual",
        "explicitly_named_receiver_closed_realized_archive",
    ),
    "not_a_universal_transfer_coefficient": True,
}


def _finite_nonnegative(value: float | int, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class ContestScoreTerms:
    """Additive terms of the contest objective."""

    seg: float
    pose: float
    rate: float
    total: float


@dataclass(frozen=True)
class RealizationTransferRatios:
    """Realized distortion divided by counterfactual distortion."""

    d_seg: float
    d_pose: float


@dataclass(frozen=True)
class ScoreGapDecomposition:
    """Realized-minus-counterfactual score gap by additive term."""

    seg: float
    pose: float
    rate: float
    total: float


def contest_score_terms(
    *,
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
    reference_bytes: int,
) -> ContestScoreTerms:
    """Return the Seg, Pose, rate, and total contest-score terms."""

    seg_distortion = _finite_nonnegative(d_seg, name="d_seg")
    pose_distortion = _finite_nonnegative(d_pose, name="d_pose")
    if type(archive_bytes) is not int or archive_bytes < 0:
        raise ValueError("archive_bytes must be a non-negative integer")
    if type(reference_bytes) is not int or reference_bytes <= 0:
        raise ValueError("reference_bytes must be a positive integer")

    seg = 100.0 * seg_distortion
    pose = math.sqrt(10.0 * pose_distortion)
    rate = 25.0 * archive_bytes / reference_bytes
    return ContestScoreTerms(seg=seg, pose=pose, rate=rate, total=seg + pose + rate)


def realization_transfer_ratios(
    *,
    counterfactual_d_seg: float,
    counterfactual_d_pose: float,
    realized_d_seg: float,
    realized_d_pose: float,
) -> RealizationTransferRatios:
    """Return instance-level realized/counterfactual distortion ratios."""

    counter_seg = _finite_nonnegative(
        counterfactual_d_seg, name="counterfactual_d_seg"
    )
    counter_pose = _finite_nonnegative(
        counterfactual_d_pose, name="counterfactual_d_pose"
    )
    realized_seg = _finite_nonnegative(realized_d_seg, name="realized_d_seg")
    realized_pose = _finite_nonnegative(realized_d_pose, name="realized_d_pose")
    if counter_seg == 0.0 or counter_pose == 0.0:
        raise ValueError("counterfactual distortions must be strictly positive")
    return RealizationTransferRatios(
        d_seg=realized_seg / counter_seg,
        d_pose=realized_pose / counter_pose,
    )


def score_gap_decomposition(
    *,
    counterfactual: ContestScoreTerms,
    realized: ContestScoreTerms,
) -> ScoreGapDecomposition:
    """Return the additive realized-minus-counterfactual score gap."""

    return ScoreGapDecomposition(
        seg=realized.seg - counterfactual.seg,
        pose=realized.pose - counterfactual.pose,
        rate=realized.rate - counterfactual.rate,
        total=realized.total - counterfactual.total,
    )


def score_gap_closes(
    gap: ScoreGapDecomposition,
    *,
    abs_tol: float = 1e-12,
) -> bool:
    """Whether the additive term gaps equal the independently computed total."""

    if not math.isfinite(abs_tol) or abs_tol < 0.0:
        raise ValueError("abs_tol must be finite and non-negative")
    return math.isclose(
        gap.seg + gap.pose + gap.rate,
        gap.total,
        rel_tol=0.0,
        abs_tol=abs_tol,
    )


def require_score_gap_closure(
    gap: ScoreGapDecomposition,
    *,
    abs_tol: float = 1e-12,
) -> None:
    """Raise when additive score-gap accounting does not close."""

    if not score_gap_closes(gap, abs_tol=abs_tol):
        raise ValueError("Seg/Pose/rate gaps do not sum to the total score gap")


__all__ = [
    "DDM_M7_REALIZATION_TRANSFER_EQUATION_ID",
    "DOMAIN_DECLARATION",
    "ContestScoreTerms",
    "RealizationTransferRatios",
    "ScoreGapDecomposition",
    "contest_score_terms",
    "realization_transfer_ratios",
    "require_score_gap_closure",
    "score_gap_closes",
    "score_gap_decomposition",
]
