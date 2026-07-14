# SPDX-License-Identifier: MIT
"""Fail-closed guard for the retired banked-pose verdict substitution.

The R1 artifact is a real full-n600 byte-closed macOS-CPU advisory artifact, but
the live V9 program does not import its pose payload.  Substituting its scalar
``d_pose`` into a current-run verdict would therefore be a confounded score path.
Until a current-run, payload-bound cache with receiver custody exists, PoseNet is
always computed live.  No numeric non-live value can reach implied score,
checkpoint selection, or a controller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

POSE_VERDICT_GUARD_SCHEMA = "pose_verdict_live_or_refused.v1"

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
    """Return a guard receipt or refuse any enabled non-live score substitution.

    ``legacy_read_only_waiver`` permits parsing an old *disabled* configuration for
    audit/replay.  It never permits an enabled gate or a numeric score-path value.
    """

    if gate_on:
        raise PoseVerdictGateError(
            "--verdict-pose-gate REFUSE: current V9 has no payload-bound pose cache; "
            "live PoseNet is required"
        )
    if configured_nonlive_dpose is not None and not legacy_read_only_waiver:
        raise PoseVerdictGateError(
            "numeric non-live d_pose is forbidden outside historical read-only parsing"
        )
    return {
        "schema": POSE_VERDICT_GUARD_SCHEMA,
        "gate_enabled": False,
        "score_path_d_pose_source": "live_posenet",
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
) -> PoseGateDecision:
    """Select the only admitted current score path: a live PoseNet forward."""

    del epoch, pose_engaged_epoch, verdict_index, canary_every
    check_pose_verdict_fallback_is_live_or_refused(gate_on=gate_on)
    return PoseGateDecision(
        compute_live=True,
        is_canary=False,
        reason="live_posenet_required_no_payload_bound_fallback",
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
