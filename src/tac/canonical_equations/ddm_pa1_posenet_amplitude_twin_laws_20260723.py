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
Composition into the governed E2 inflate runtime remains owed.

All constants are [macOS-CPU frozen-scorer advisory], score_claim=false,
pointer_moved=false.  Canonical receipt:
.omx/research/ddm_pa1_posenet_amplitude_twin_20260723T221923Z/
ddm_pa1_posenet_amplitude_twin_receipt.json
"""

from __future__ import annotations

import math
from typing import Literal

EQUATION_ID = "ddm_pa1_posenet_amplitude_twin_laws_v1"
SOURCE_BYTES = 37_545_489

E2_DPOSE = 162.580958694146
E2_DSEG = 0.02861480712890625
FRAME0_SCORER_DPOSE = 147.49104204339514
FRAME0_SCORER_DSEG = E2_DSEG
JOINT_SCORER_DPOSE = 161.7268879706079
JOINT_SCORER_DSEG = 0.035720909966362846
FRAME0_SCORER_DELTA_BYTES = 0
FRAME0_GT_DELTA_BYTES = 24


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
