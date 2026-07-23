# SPDX-License-Identifier: MIT
"""Canonical error-source-tensor laws measured by ddm_sn1 (2026-07-23).

Three measured facts, one evaluator surface (n600, frozen SegNet/PoseNet,
[macOS-CPU advisory]; receipt SHA
ecf9f015fa6999b9bb7602c93027da713bb278389b92d5d1bf0b95f4ced19faa):

1. The exact v19c 2,265,811-error partition CLOSES:
   892,710 DESCRIBED_BUT_REALIZATION_LOST (39.3991%) + 738,090
   NEVER_DESCRIBED (32.5751%) + 635,011 STRUCTURALLY_HARD_IRREDUCIBLE
   (28.0258%). The third label is scoped to the current semantic program plus
   the SHA-pinned DV1 extension — NOT a family-impossibility claim.
   Realization-lost sub-partition: 587,913 COARSE_DESCRIPTION + 208,623
   PAINT_FUNCTION + 96,174 TEXTURE_PRIOR_REGION_ERF.
2. Sided SegNet decision distance is ASYMMETRIC (ordered pairs are not
   symmetric): Road->Lane q10 0.024886758 vs Lane->Road 0.019704731
   (ratio 1.26298); strongest reversal Undrivable<->Lane |dq10| 0.0323151
   (ratio 1.33648). MyCar<->Undrivable has NO measured boundary support —
   typed as no-support, never an invented zero.
3. Record-level constancy NEGATIVE: all 11 exact records (5 class cells +
   5 separatrices + pose pair-screw) have 600 unique states / 599 adjacent
   changes -> whole-record static coding is INADMISSIBLE for every current
   record; G4 pixel recurrence is only an UPPER BOUND on describable
   constancy (persistent-primitive + sparse-innovation coding remains open).

Anchors: findings memo
.omx/research/codex_findings_ddm_sn1_segnet_telemetry_asymmetry_20260723_codex.md
(merged main fd88afb6fd). score_claim=false, promotable=false.
"""

from __future__ import annotations

EQUATION_ID = "ddm_sn1_error_source_laws_v1"

# Measured constants (sn1 receipt, SHA-pinned above).
V19C_TOTAL_ERRORS = 2_265_811
REALIZATION_LOST_ERRORS = 892_710
NEVER_DESCRIBED_ERRORS = 738_090
SCOPED_HARD_ERRORS = 635_011
COARSE_DESCRIPTION_ERRORS = 587_913
PAINT_FUNCTION_ERRORS = 208_623
TEXTURE_PRIOR_ERRORS = 96_174
DV1_SHARED_PAYLOAD_BYTES = 1_610
SOLVE_MENU_ROWS = 2_649
RECORD_COUNT = 11
RECORD_UNIQUE_STATES = 600


def partition_closes(realization_lost: int, never_described: int, scoped_hard: int, total: int) -> bool:
    """Error-source partition closure: the three sources sum EXACTLY to total."""
    for name, value in (
        ("realization_lost", realization_lost),
        ("never_described", never_described),
        ("scoped_hard", scoped_hard),
        ("total", total),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer error count")
    return realization_lost + never_described + scoped_hard == total


def semantic_errors_per_shared_byte(errors_reached: int, shared_bytes: int) -> float:
    """Semantic reach per shared description byte (NOT receiver-survival or score value)."""
    for name, value in (("errors_reached", errors_reached), ("shared_bytes", shared_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return errors_reached / shared_bytes


def whole_record_static_coding_admissible(unique_states: int, timesteps: int) -> bool:
    """Whole-record static coding is admissible ONLY if the record repeats states.

    sn1 measured 600 unique states over 600 timesteps for all 11 records ->
    inadmissible everywhere; pixel recurrence never substitutes for this test.
    """
    for name, value in (("unique_states", unique_states), ("timesteps", timesteps)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if unique_states > timesteps:
        raise ValueError("unique_states cannot exceed timesteps")
    return unique_states < timesteps
