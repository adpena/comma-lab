# SPDX-License-Identifier: MIT
"""Canonical PoseNet amplitude-twin laws measured by DDM PA1 (2026-07-23).

The n600 E2 frame-0 scorer-stat arm reduced frozen-CPU advisory d_pose from
162.580958694146 to 147.491042043395 at zero incremental payload bytes while
leaving frame 1 byte-identical and d_seg exactly unchanged.  The same target
on both frames reduced d_pose only to 161.726887970608 and increased d_seg by
0.007106102837, so its joint score delta was +0.604562771375 (rejected).

The zero-byte classification is a FREE candidate under rule 118, not a
promotion claim: the target constants derive only from frozen scorer weights
and BN tables, and the affine derives from already-counted decoded E2 content.
DDM E3 subsequently closed governed receiver survival: all 38 source/corrected
camera batch hashes equal PA1, frame 1 is byte-identical to E2, and locked
upstream ``evaluate.sh`` passed on the exact 439,303-byte archive.

All constants are [macOS-CPU frozen-scorer advisory], score_claim=false,
pointer_moved=false.  Canonical receipt:
.omx/research/ddm_pa1_posenet_amplitude_twin_20260723T221923Z/
ddm_pa1_posenet_amplitude_twin_receipt.json
"""

from __future__ import annotations

import math
from typing import Literal

EQUATION_ID = "ddm_pa1_posenet_amplitude_twin_laws_v1"
RECEIVER_SURVIVAL_EQUATION_ID = "ddm_e3_pa1_receiver_survival_v1"
SOURCE_BYTES = 37_545_489

E2_DPOSE = 162.580958694146
E2_DSEG = 0.02861480712890625
FRAME0_SCORER_DPOSE = 147.49104204339514
FRAME0_SCORER_DSEG = E2_DSEG
JOINT_SCORER_DPOSE = 161.7268879706079
JOINT_SCORER_DSEG = 0.035720909966362846
FRAME0_SCORER_DELTA_BYTES = 0
FRAME0_GT_DELTA_BYTES = 24
E3_ARCHIVE_BYTES = 439_303
E3_ARCHIVE_SHA256 = "dd8fc5fed6ff11e532765dfe6104f02b3b97171b824123312a3ab469c1be6cbe"
E3_RAW_SHA256 = "4c553508b0bf92ccdc137e215799ae30a346b58e0617e5156441a7929302b4f1"
E3_FRAME1_SHA256 = "bfe8f686e5da8578a86029287b0a78430431cf612457ab84abc302cd8ac2bca1"


def joint_score_delta(
    baseline_dseg: float,
    baseline_dpose: float,
    candidate_dseg: float,
    candidate_dpose: float,
    delta_bytes: int,
) -> float:
    """Exact contest-action delta for an advisory candidate row."""
    for name, value in (
        ("baseline_dseg", baseline_dseg),
        ("baseline_dpose", baseline_dpose),
        ("candidate_dseg", candidate_dseg),
        ("candidate_dpose", candidate_dpose),
    ):
        if not isinstance(value, float) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite nonnegative float")
    if isinstance(delta_bytes, bool) or not isinstance(delta_bytes, int) or delta_bytes < 0:
        raise ValueError("delta_bytes must be a nonnegative integer")
    return (
        100.0 * (candidate_dseg - baseline_dseg)
        + math.sqrt(10.0 * candidate_dpose)
        - math.sqrt(10.0 * baseline_dpose)
        + 25.0 * delta_bytes / SOURCE_BYTES
    )


def amplitude_gap_is_small(
    standardized_mean_gap_rms: float,
    log_std_ratio_rms: float,
    equivalence_margin: float,
) -> bool:
    """Pre-registered amplitude falsifier used before opening the arm ladder."""
    for name, value in (
        ("standardized_mean_gap_rms", standardized_mean_gap_rms),
        ("log_std_ratio_rms", log_std_ratio_rms),
        ("equivalence_margin", equivalence_margin),
    ):
        if not isinstance(value, float) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite nonnegative float")
    if equivalence_margin == 0.0:
        raise ValueError("equivalence_margin must be positive")
    return standardized_mean_gap_rms <= equivalence_margin and log_std_ratio_rms <= equivalence_margin


def target_rate_partition(
    *,
    receiver_effective: bool,
    target_uses_video_derived_fact: bool,
) -> Literal["FREE", "NULL", "COUNTED"]:
    """FREE/NULL/COUNTED partition for an amplitude-target formulation."""
    if not isinstance(receiver_effective, bool) or not isinstance(target_uses_video_derived_fact, bool):
        raise ValueError("partition inputs must be bool")
    if not receiver_effective:
        return "NULL"
    if target_uses_video_derived_fact:
        return "COUNTED"
    return "FREE"


def receiver_survives(
    *,
    source_batch_sha256: list[str],
    packaged_batch_sha256: list[str],
    source_frame1_sha256: str,
    packaged_frame1_sha256: str,
    amplitude_payload_bytes: int,
) -> bool:
    """Apply the E3 receiver-survival law at exact batch/frame byte identity."""

    if (
        len(source_batch_sha256) != 38
        or len(packaged_batch_sha256) != 38
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in [
                *source_batch_sha256,
                *packaged_batch_sha256,
                source_frame1_sha256,
                packaged_frame1_sha256,
            ]
        )
    ):
        raise ValueError("receiver-survival custody must contain 38 lowercase SHA-256 pairs")
    if (
        isinstance(amplitude_payload_bytes, bool)
        or not isinstance(amplitude_payload_bytes, int)
        or amplitude_payload_bytes < 0
    ):
        raise ValueError("amplitude_payload_bytes must be a nonnegative integer")
    return (
        source_batch_sha256 == packaged_batch_sha256
        and source_frame1_sha256 == packaged_frame1_sha256
        and amplitude_payload_bytes == 0
    )
