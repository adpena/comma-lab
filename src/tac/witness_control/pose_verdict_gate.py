# SPDX-License-Identifier: MIT
"""Fail-closed pose-blind compute gate for V9 progress verdicts.

The safe optimization is narrower than the retired banked-pose substitution:
while the two-phase trainer has not engaged its pose finish, a gated verdict may
omit PoseNet and report only live ``d_seg``.  Such a row carries ``d_pose=None``
and is ineligible for an implied-score claim.  Once pose finish engages, or when
the gate is off, PoseNet is computed live.

No historical or non-live numeric ``d_pose`` is admitted.  This distinction is
load-bearing: skipping an irrelevant forward is safe; substituting a scalar from
another payload/run is not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

POSE_VERDICT_GUARD_SCHEMA = "pose_verdict_live_or_progress_only.v2"

# Retained only so older checkpoints/config imports remain loadable.  This
# number cannot be returned by the live score path.
BANKED_R1_DPOSE_DEFAULT = 0.001610


class PoseVerdictGateError(ValueError):
    """A non-live pose scalar attempted to enter the score path."""


@dataclass(frozen=True)
class PoseGateDecision:
    compute_live: bool
    is_canary: bool
    reason: str
    d_pose_source: str


def check_pose_verdict_fallback_is_live_or_refused(
    *,
    gate_on: bool,
    configured_nonlive_dpose: float | None = None,
    legacy_read_only_waiver: bool = False,
) -> dict[str, Any]:
    """Return a guard receipt and refuse every numeric non-live substitution.

    ``legacy_read_only_waiver`` permits parsing an old *disabled* configuration for
    audit/replay.  It never permits a numeric score-path value.  ``gate_on`` now
    means d_seg-only progress while pose is blind, not banked-pose substitution.
    """

    if configured_nonlive_dpose is not None and (
        not legacy_read_only_waiver or gate_on
    ):
        raise PoseVerdictGateError(
            "numeric non-live d_pose is forbidden outside historical read-only parsing"
        )
    return {
        "schema": POSE_VERDICT_GUARD_SCHEMA,
        "gate_enabled": bool(gate_on),
        "score_path_d_pose_source": (
            "live_posenet_or_missing_progress_only" if gate_on else "live_posenet"
        ),
        "numeric_nonlive_dpose_admitted": False,
        "legacy_reference_present": configured_nonlive_dpose is not None,
        "legacy_read_only_waiver": bool(legacy_read_only_waiver),
        "reference_authority": (
            "full-n600 byte-closed macOS-CPU advisory; unselected"
            if configured_nonlive_dpose is not None
            else None
        ),
    }


def decide_pose_verdict(
    *,
    epoch: int,
    pose_engaged_epoch: int,
    verdict_index: int,
    gate_on: bool,
    canary_every: int,
    force_live: bool = False,
) -> PoseGateDecision:
    """Choose live PoseNet or an explicitly score-ineligible d_seg-only row."""

    del canary_every
    check_pose_verdict_fallback_is_live_or_refused(gate_on=gate_on)
    # The pre-loop baseline remains a full live verdict so the run has one
    # current-payload score anchor.  In-loop pose-blind rows skip PoseNet until
    # the existing pose-finish controller stamps an engagement epoch.
    pose_blind = (
        bool(gate_on)
        and not bool(force_live)
        and int(epoch) >= 0
        and int(pose_engaged_epoch) < 0
    )
    if pose_blind:
        return PoseGateDecision(
            compute_live=False,
            is_canary=False,
            reason="pose_blind_dseg_progress_only",
            d_pose_source="missing_progress_only",
        )
    return PoseGateDecision(
        compute_live=True,
        is_canary=False,
        reason=(
            "forced_live_current_payload_anchor"
            if force_live
            else (
                "pose_finish_engaged_live_posenet"
                if gate_on
                else "gate_off_live_posenet"
            )
        ),
        d_pose_source="live",
    )


def banked_pose_telemetry(banked_dpose: float, reason: str) -> dict[str, Any]:
    """Refuse the retired numeric substitution while preserving import ABI."""

    del banked_dpose, reason
    raise PoseVerdictGateError(
        "banked_pose_telemetry is retired: live PoseNet is required"
    )


def canary_drift(live_dpose: float, banked_dpose: float) -> dict[str, Any]:
    """Return explicitly non-authoritative historical drift diagnostics."""

    live = float(live_dpose)
    banked = float(banked_dpose)
    abs_drift = abs(live - banked)
    return {
        "schema": "pose_verdict_historical_drift_diagnostic.v1",
        "d_pose_live": live,
        "d_pose_historical_reference": banked,
        "abs_drift": abs_drift,
        "rel_drift": abs_drift / max(1e-12, abs(banked)),
        "score_claim": False,
        "selection_eligible": False,
    }


__all__ = [
    "BANKED_R1_DPOSE_DEFAULT",
    "POSE_VERDICT_GUARD_SCHEMA",
    "PoseGateDecision",
    "PoseVerdictGateError",
    "banked_pose_telemetry",
    "canary_drift",
    "check_pose_verdict_fallback_is_live_or_refused",
    "decide_pose_verdict",
]
