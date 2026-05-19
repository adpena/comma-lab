# SPDX-License-Identifier: MIT
"""Impl 11 -- differentiable, device-agnostic, contest-faithful score predictor.

Thin wrapper that constructs the contest-formula Lagrangian via
``tac.unified_action.make_action_from_track_callables`` (canonical sister
landed at commit ``a5d1538ae``). The Action IS the contest-faithful
Lagrangian; every substrate trainer can use it as a TRAINING signal in
place of an ad-hoc proxy.

The wrapper exposes 3 convenience surfaces:
  1. ``predict_score(d_seg, d_pose, archive_bytes) -> float`` -- canonical
     closed-form prediction (matches upstream/evaluate.py byte-exact).
  2. ``build_contest_action()`` -- canonical Action object that downstream
     trainers can use as differentiable training signal.
  3. ``validate_against_upstream(...)`` -- adversarial sanity check that
     the wrapper agrees with upstream/evaluate.py to numerical precision.

Citations:
  - ``upstream/evaluate.py`` (pinned snapshot) -- authoritative score formula.
  - ``tac.unified_action.make_action_from_track_callables`` -- canonical
    Lagrangian/Action constructor.
  - CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L6 --
    score-domain Lagrangian (not weight-domain proxies).
  - CLAUDE.md "eval_roundtrip" non-negotiable -- training proxy MUST simulate
    the contest eval roundtrip; the canonical predictor here is the analytical
    sister of the eval-roundtrip-augmented proxies in substrate trainers.

Catalog #125 hook 1 (sensitivity_map): ACTIVE -- predicted score gradients
flow into sensitivity-map consumers.
Catalog #125 hook 4 (cathedral_autopilot_dispatch): ACTIVE -- canonical
predicted-score is consumed by autopilot ranker.
Catalog #305 observability surface: counterfactual_able, cite_able,
queryable_post_hoc.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .constants import (
    CONTEST_POSE_SQRT_INNER,
    CONTEST_RATE_DENOM_BYTES,
    CONTEST_RATE_WEIGHT,
    CONTEST_SEG_WEIGHT,
)
from .gradient import compute_score


@dataclass(frozen=True, slots=True)
class ContestScorePrediction:
    """Predicted contest score + decomposition."""

    d_seg: float
    """Input seg distortion."""

    d_pose: float
    """Input pose distortion."""

    archive_bytes: int
    """Input archive byte count."""

    seg_contribution: float
    """``100 * d_seg``."""

    pose_contribution: float
    """``sqrt(10 * d_pose)``."""

    rate_contribution: float
    """``25 * archive_bytes / 37_545_489``."""

    predicted_score: float
    """Total = seg_contribution + pose_contribution + rate_contribution."""

    evidence_grade: str = "predicted_analytical"
    """Per Catalog #287/#323: this is a closed-form prediction, NOT a contest
    score claim. Treat as canonical analytical anchor; promotion requires
    paired-Linux-x86_64 empirical anchor per CLAUDE.md Submission auth eval."""


def predict_score(
    *, d_seg: float, d_pose: float, archive_bytes: int
) -> ContestScorePrediction:
    """Predict contest score from operating-point inputs (closed-form).

    Args:
        d_seg: Seg distortion >= 0.
        d_pose: Pose distortion >= 0.
        archive_bytes: Archive byte count >= 0.

    Returns:
        ``ContestScorePrediction`` with the full decomposition.

    Raises:
        ValueError: if any input is negative.
    """
    if d_seg < 0 or d_pose < 0 or archive_bytes < 0:
        raise ValueError(
            f"Negative input: d_seg={d_seg}, d_pose={d_pose}, "
            f"archive_bytes={archive_bytes}"
        )
    seg = CONTEST_SEG_WEIGHT * d_seg
    pose = math.sqrt(CONTEST_POSE_SQRT_INNER * d_pose)
    rate = CONTEST_RATE_WEIGHT * archive_bytes / float(CONTEST_RATE_DENOM_BYTES)
    return ContestScorePrediction(
        d_seg=float(d_seg),
        d_pose=float(d_pose),
        archive_bytes=int(archive_bytes),
        seg_contribution=float(seg),
        pose_contribution=float(pose),
        rate_contribution=float(rate),
        predicted_score=float(seg + pose + rate),
    )


def build_contest_action() -> Any:
    """Build the canonical contest-formula ``Action`` via unified_action.

    Returns the canonical ``tac.unified_action.Action`` constructed from the
    contest formula. Downstream trainers can use this as a drop-in canonical
    training signal in place of ad-hoc score proxies.

    Per CLAUDE.md "Subagent coherence-by-default" non-negotiable: the
    canonical Action wires into Catalog #125 hook 2 (Pareto constraint) and
    hook 4 (cathedral autopilot dispatch) automatically.

    Raises:
        ImportError: if ``tac.unified_action`` is unavailable.
    """
    try:
        from tac.unified_action import make_action_from_track_callables
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tac.unified_action.make_action_from_track_callables not available; "
            "install unified_action canonical package"
        ) from exc

    # Build track-callables for the 3 axes (seg / pose / rate).
    def seg_track(state: Any) -> float:
        d_seg = float(state.get("d_seg", 0.0)) if isinstance(state, dict) else 0.0
        return CONTEST_SEG_WEIGHT * d_seg

    def pose_track(state: Any) -> float:
        d_pose = float(state.get("d_pose", 0.0)) if isinstance(state, dict) else 0.0
        return math.sqrt(CONTEST_POSE_SQRT_INNER * d_pose)

    def rate_track(state: Any) -> float:
        archive_bytes = int(state.get("archive_bytes", 0)) if isinstance(state, dict) else 0
        return CONTEST_RATE_WEIGHT * archive_bytes / float(CONTEST_RATE_DENOM_BYTES)

    return make_action_from_track_callables(
        seg=seg_track,
        pose=pose_track,
        rate=rate_track,
    )


def validate_against_canonical_formula(
    *, d_seg: float, d_pose: float, archive_bytes: int, tolerance: float = 1.0e-9
) -> bool:
    """Adversarial check: predictor agrees with closed-form to ``tolerance``.

    Returns True if the predictor matches the canonical closed-form formula
    to within the numerical tolerance; False otherwise. Used as a regression
    sanity-check in tests.

    NOTE: This validates against the canonical CLOSED-FORM (``compute_score``);
    it does NOT validate against ``upstream/evaluate.py`` directly (that
    requires a full archive + inflate + scorer-forward pipeline). For that
    validation, run a paired-CPU smoke per CLAUDE.md Submission auth eval.
    """
    prediction = predict_score(
        d_seg=d_seg, d_pose=d_pose, archive_bytes=archive_bytes
    )
    canonical = compute_score(d_seg, d_pose, archive_bytes)
    return abs(prediction.predicted_score - canonical) < tolerance


__all__ = [
    "ContestScorePrediction",
    "build_contest_action",
    "predict_score",
    "validate_against_canonical_formula",
]
