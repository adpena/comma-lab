# SPDX-License-Identifier: MIT
"""Canonical export/stage-attribution laws measured by ddm_e2 (2026-07-23).

Three measured facts, one evaluator surface:

1. Formulation-scoped zero-pose floor: a compact packet whose pose stream is
   ABSENT from the counted members (not counted-but-inert) has score floor
   seg_term + rate_term even at d_pose=0. For E2 at the C1 160,000 B ceiling
   the floor is 2.86148200 + 0.22869991 = 3.09018191 against a distortion
   budget of 0.08454539 under pointer 0.1910828242 — no nonnegative byte
   budget closes THIS formulation. verdict_scope: FORMULATION (E1/E2 compact
   packet boundary); the exact-lattice control keeps the family open.
2. Live-path stage-ledger closure: per stream x stage,
   errors_after = errors_before + introduced - corrected, and the final sum
   closes EXACTLY to the independent meter (3,375,540 / 117,964,800 =
   0.028614807129). PAINT introduces 3,349,482 of the 3,375,540 final errors
   (99.23%) — the residual is DESCRIPTION-limited (class i), not
   realization-limited (R-resample + uint8 net ~+26K, class ii small).
3. Semantic-paint de-duplication pays 19.09x the rate dual: +4,372 B bought
   -0.05557250 S (1.27110018e-5 S/byte vs 25/37,545,489 = 6.6586e-7).
   FIRST-RUNG advisory positive, score_claim=false.

Anchors: e2 receipt .omx/research/ddm_e2_pose_stream_and_doctrine_export_receipt.json
(merged main 6bbb4aecf4), v2 stage receipt binding
b4176ddc24ccfbd1c466aec1326572201b4dfdeb03f980cd6e91d4a7fb19d9c3.
All rows [macOS-CPU advisory]: score_claim=false, promotable=false.
"""

from __future__ import annotations

EQUATION_ID = "ddm_e2_export_stage_laws_v1"

# Measured constants (e2 receipts, SHA-pinned above).
E2_ARCHIVE_BYTES = 343_466
E2_DSEG_INDEPENDENT = 0.028614807129
E2_DPOSE = 162.580958694146
E2_ZERO_POSE_FLOOR_AT_C1_CEILING = 3.09018191
PAINT_INTRODUCED_ERRORS = 3_349_482  # chart 860,296 + semantic 2,489,186
FINAL_ERRORS = 3_375_540
FRAME1_SCORER_SITES = 117_964_800  # 600 pairs x 512 x 384
REALIZATION_GAP_DSEG = 0.001144523776  # official 0.02861482 - exact-solve 0.027470296224


def zero_pose_floor(seg_term: float, rate_term: float) -> float:
    """Formulation floor for a pose-absent packet: seg + rate at d_pose=0."""
    for name, value in (("seg_term", seg_term), ("rate_term", rate_term)):
        if not isinstance(value, float) or value < 0.0:
            raise ValueError(f"{name} must be a nonnegative float score term")
    return seg_term + rate_term


def stage_ledger_closes(errors_before: int, introduced: int, corrected: int, errors_after: int) -> bool:
    """Stage-ledger identity: after = before + introduced - corrected, all counts nonnegative."""
    for name, value in (
        ("errors_before", errors_before),
        ("introduced", introduced),
        ("corrected", corrected),
        ("errors_after", errors_after),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer site count")
    return errors_after == errors_before + introduced - corrected


def gain_per_byte_vs_rate_dual(delta_s: float, delta_bytes: int) -> float:
    """Ratio of a measured S-gain-per-byte to the contest rate dual 25/37,545,489.

    >1 means the bytes bought more S than the rate term charges for them
    (FIRST-RUNG positive); <=1 means dominated by simply shrinking the archive.
    """
    if not isinstance(delta_s, float) or delta_s <= 0.0:
        raise ValueError("delta_s must be a positive float S improvement")
    if isinstance(delta_bytes, bool) or not isinstance(delta_bytes, int) or delta_bytes <= 0:
        raise ValueError("delta_bytes must be a positive integer byte count")
    return (delta_s / delta_bytes) / (25 / 37_545_489)
