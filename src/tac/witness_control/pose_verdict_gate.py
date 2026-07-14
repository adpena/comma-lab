# SPDX-License-Identifier: MIT
"""Pose-verdict gate — skip the CPU-torch PoseNet forward while pose is FROZEN.

The authority verdict computes d_seg (SegNet) + d_pose (PoseNet) every eval. MEASURED on
the n96 macOS-CPU Torch one-thread advisory timer
(``frozen_scorer_verdict_wallclock_n96_20260714``): d_pose = **22.6%** of 59.615 s,
or about 13.47 s. The corresponding n600 PoseNet cost is **DERIVED**, not measured:
about 84.2 s under linear projection. During the entire pre-pose-finish phase ``w_pose = 0``
— the pose carrier
(dxi) is FROZEN at the banked-R1 init, so pose is NOT descending and the live d_pose is
non-actionable telemetry. Recomputing PoseNet every verdict is dead work.

This gate SKIPS the live PoseNet forward while pose is frozen and ships the banked-R1 d_pose
**LABELLED ``banked``** (never passed off as a live measurement — a checkpoint number reported
as a live verdict is the surrogate-as-authority NO-FAKE trap). It composes with a cheap
**drift-canary**: every ``canary_every`` verdicts it forces a live compute so a real d_pose
drift (frame_1 changes as SEG trains, and PoseNet reads both frames) is caught for ~1/K the cost
instead of paying 23% every verdict.

Score-neutral: skipping a TELEMETRY forward changes no trained weight, no archive byte, no d_seg.
The costate/shadow controllers already tolerate ``d_pose=None``/banked (costate_estimator
line 141). Post-finish (pose engaged) the live forward returns — d_pose is actionable again.

Authority of the returned d_pose while gated: ``[banked-R1 pose-gated; NON-LIVE]``. NON-PROMOTABLE.
"""
from __future__ import annotations

from dataclasses import dataclass

# Banked R1 dxi d_pose (byte-close measured, memory L68 / r1_dxi_shippability): the frozen
# pre-finish pose value. Default only; the trainer passes the run's own banked value.
BANKED_R1_DPOSE_DEFAULT = 0.001610


@dataclass(frozen=True)
class PoseGateDecision:
    compute_live: bool          # True => run the real PoseNet forward this verdict
    is_canary: bool             # True => a forced live compute purely to measure drift
    reason: str                 # provenance for the telemetry row
    d_pose_source: str          # "live" | "banked_R1_pose_gated"


def decide_pose_verdict(
    *,
    epoch: int,
    pose_engaged_epoch: int,
    verdict_index: int,
    gate_on: bool,
    canary_every: int,
) -> PoseGateDecision:
    """Return whether to compute the live d_pose this verdict.

    ``pose_engaged_epoch < 0`` OR ``epoch < pose_engaged_epoch`` => pose is FROZEN (pre-finish).
    ``verdict_index`` is a monotonic per-run verdict counter (0-based); the canary fires when
    ``verdict_index % canary_every == 0`` (so index 0 always computes live => the first row is a
    real anchor, never a bare banked constant).
    """
    # Gate off, or pose has engaged => always live (the incumbent, byte-identical behaviour).
    engaged = pose_engaged_epoch >= 0 and epoch >= pose_engaged_epoch
    if not gate_on or engaged:
        return PoseGateDecision(True, False, "pose_engaged_or_gate_off", "live")
    # Pre-finish: canary cadence forces a live compute to measure drift; else ship banked.
    k = max(1, int(canary_every))
    if verdict_index % k == 0:
        return PoseGateDecision(True, True, f"canary(index={verdict_index},every={k})", "live")
    return PoseGateDecision(False, False, "pose_frozen_pre_finish", "banked_R1_pose_gated")


def banked_pose_telemetry(banked_dpose: float, reason: str) -> dict:
    """The LABELLED banked-d_pose telemetry fields (honest: never claims 'live')."""
    return {
        "d_pose": float(banked_dpose),
        "d_pose_source": "banked_R1_pose_gated",
        "d_pose_axis": "[banked-R1 pose-gated; NON-LIVE] NON-PROMOTABLE",
        "d_pose_live": False,
        "pose_gate_reason": reason,
    }


def canary_drift(live_dpose: float, banked_dpose: float) -> dict:
    """Drift row emitted whenever the canary fires: how far live has moved from banked-R1."""
    abs_drift = abs(float(live_dpose) - float(banked_dpose))
    rel = abs_drift / max(1e-12, abs(float(banked_dpose)))
    return {
        "stage": "pose_gate_canary",
        "d_pose_live": float(live_dpose),
        "d_pose_banked": float(banked_dpose),
        "abs_drift": abs_drift,
        "rel_drift": rel,
        "axis": "[macOS-CPU-torch 1-thread advisory] NON-PROMOTABLE",
        "note": "if abs_drift stays small the banked constant is trustworthy between canaries; "
                "a rising drift = SEG-training is moving PoseNet's input => widen the live cadence.",
    }


__all__ = [
    "BANKED_R1_DPOSE_DEFAULT",
    "PoseGateDecision",
    "banked_pose_telemetry",
    "canary_drift",
    "decide_pose_verdict",
]
