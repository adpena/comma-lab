# SPDX-License-Identifier: MIT
"""Typed J5 warm-start slope falsifier; pure math, no launch surface."""

from __future__ import annotations

import math
from dataclasses import dataclass

_EPS = 1.0e-15


@dataclass(frozen=True, slots=True)
class ObjectiveTerms:
    """Seg and Pose terms in contest score units."""

    seg: float
    pose: float

    @classmethod
    def from_distortions(cls, *, d_seg: float, d_pose: float) -> ObjectiveTerms:
        values = (float(d_seg), float(d_pose))
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("distortions must be finite and non-negative")
        return cls(seg=100.0 * values[0], pose=math.sqrt(10.0 * values[1]))


@dataclass(frozen=True, slots=True)
class WarmStartGap:
    """Opening advantage/debt of W_seg relative to W_joint."""

    seg_advantage: float
    pose_debt: float
    critical_ratio: float


@dataclass(frozen=True, slots=True)
class SlopeVerdict:
    """Executable verdict for one bounded J5 smoke window."""

    decision: str
    reason: str
    seg_delta_per_step: float
    pose_progress_per_step: float
    seg_regression_per_step: float
    observed_ratio: float
    critical_ratio: float
    predicted_pose_repayment_steps: float
    predicted_seg_advantage_exhaustion_steps: float


def derive_warm_start_gap(
    *, wseg: ObjectiveTerms, wjoint: ObjectiveTerms
) -> WarmStartGap:
    """Derive R* = extra Pose debt / Seg advantage from measured endpoints."""

    seg_advantage = float(wjoint.seg - wseg.seg)
    pose_debt = float(wseg.pose - wjoint.pose)
    if seg_advantage <= 0.0:
        raise ValueError("W_seg must have a strict opening Seg-term advantage")
    if pose_debt <= 0.0:
        raise ValueError("W_seg must carry a strict opening Pose-term debt")
    return WarmStartGap(
        seg_advantage=seg_advantage,
        pose_debt=pose_debt,
        critical_ratio=pose_debt / seg_advantage,
    )


def critical_pose_to_seg_slope_ratio(
    *,
    wseg_d_seg: float,
    wseg_d_pose: float,
    wjoint_d_seg: float,
    wjoint_d_pose: float,
) -> float:
    """Callable scalar form of the preregistered critical-ratio equation."""

    gap = derive_warm_start_gap(
        wseg=ObjectiveTerms.from_distortions(
            d_seg=wseg_d_seg, d_pose=wseg_d_pose
        ),
        wjoint=ObjectiveTerms.from_distortions(
            d_seg=wjoint_d_seg, d_pose=wjoint_d_pose
        ),
    )
    return gap.critical_ratio


def evaluate_bounded_slope_window(
    *,
    gap: WarmStartGap,
    start: ObjectiveTerms,
    end: ObjectiveTerms,
    steps: int,
) -> SlopeVerdict:
    """Apply the preregistered fail-closed warm-start decision.

    Pose progress is positive when the Pose term falls.  Seg regression is
    positive when the Seg term rises.  The ratio test is the algebraic form of
    ``pose_debt / pose_progress <= seg_advantage / seg_regression``.
    A Pose stall or any Seg regression is independently terminal for W_seg, as
    required by the bounded smoke contract.
    """

    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    if gap.seg_advantage <= 0.0 or gap.pose_debt <= 0.0:
        raise ValueError("gap terms must be positive")
    step_count = int(steps)
    seg_delta = (float(end.seg) - float(start.seg)) / step_count
    pose_progress = (float(start.pose) - float(end.pose)) / step_count
    seg_regression = max(seg_delta, 0.0)
    if pose_progress <= 0.0:
        decision = "KEEP_WJOINT"
        reason = "POSE_STALL_OR_REGRESSION"
    elif seg_delta > 0.0:
        decision = "KEEP_WJOINT"
        reason = "SEG_REGRESSION"
    else:
        decision = "ADOPT_WSEG"
        reason = "POSE_DEBT_REPAID_BEFORE_SEG_ADVANTAGE_EXHAUSTION"
    observed = (
        math.inf
        if pose_progress > 0.0 and seg_regression <= _EPS
        else pose_progress / max(seg_regression, _EPS)
    )
    if decision == "ADOPT_WSEG" and observed < gap.critical_ratio:
        decision = "KEEP_WJOINT"
        reason = "SLOPE_RATIO_BELOW_CRITICAL"
    return SlopeVerdict(
        decision=decision,
        reason=reason,
        seg_delta_per_step=seg_delta,
        pose_progress_per_step=pose_progress,
        seg_regression_per_step=seg_regression,
        observed_ratio=observed,
        critical_ratio=gap.critical_ratio,
        predicted_pose_repayment_steps=(
            math.inf if pose_progress <= 0.0 else gap.pose_debt / pose_progress
        ),
        predicted_seg_advantage_exhaustion_steps=(
            math.inf
            if seg_regression <= _EPS
            else gap.seg_advantage / seg_regression
        ),
    )


__all__ = [
    "ObjectiveTerms",
    "SlopeVerdict",
    "WarmStartGap",
    "critical_pose_to_seg_slope_ratio",
    "derive_warm_start_gap",
    "evaluate_bounded_slope_window",
]
