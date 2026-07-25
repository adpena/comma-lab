# SPDX-License-Identifier: MIT
"""Fail-closed PF3 materialization helpers for DDM dimension prices.

PF3 probes are physical ``V19C -> one RG3 coordinate`` edges.  RD1's 162-cell
cube, however, describes three different, already-settled hull edges.  This
module deliberately does not copy a PF3 secant into an RD1 cell unless the
caller supplies an exact application-edge identity.  A measured coordinate
secant is useful evidence, but it is not automatically an actionable RD1
price.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

SCORED_PIXELS: Final = 600 * 512 * 384
POSE_COORDINATES: Final = 600 * 6
RATE_DENOMINATOR_BYTES: Final = 37_545_489
RATE_BREAK_EVEN_SCORE_PER_BYTE: Final = 25.0 / RATE_DENOMINATOR_BYTES
RD1_CELL_COUNT: Final = 162
OCCUPIED_BUCKET_COUNT: Final = 37
RD1_EDGE_APPLICATION_BLOCKER: Final = (
    "PF3_COORDINATE_TO_RD1_EXISTING_DUAL_EDGE_APPLICATION_OPERATOR_ABSENT"
)


class PF3MaterializationError(ValueError):
    """PF3 input custody or a claimed materialization invariant differs."""


def joint_distortion_delta(
    *,
    base_errors: int,
    candidate_errors: int,
    base_d_pose: float,
    candidate_d_pose: float,
) -> dict[str, float]:
    """Return the exact master-action distortion delta, excluding rate."""

    if (
        isinstance(base_errors, bool)
        or isinstance(candidate_errors, bool)
        or base_errors < 0
        or candidate_errors < 0
    ):
        raise PF3MaterializationError("Seg error counts must be nonnegative integers")
    values = (base_d_pose, candidate_d_pose)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise PF3MaterializationError("Pose distortions must be finite nonnegative")
    seg_delta = 100.0 * (candidate_errors - base_errors) / SCORED_PIXELS
    pose_delta = math.sqrt(10.0 * candidate_d_pose) - math.sqrt(10.0 * base_d_pose)
    return {
        "delta_D_seg": seg_delta,
        "delta_D_pose": pose_delta,
        "delta_D_joint": seg_delta + pose_delta,
    }


def conditional_coordinate_price(
    *,
    delta_counted_bytes: int,
    delta_D_dimension: float,
    delta_D_joint: float,
) -> dict[str, Any]:
    """Price one physical coordinate without pretending it is an RD1 edge.

    The conditional secant is finite only when the intended semantic
    dimension and the complete joint objective both improve and the exact
    same-codec byte delta is positive.  Dominating nonpositive byte deltas are
    preserved as a separate status because downstream RD1 consumers require a
    strictly positive ``lambda_bytes_per_D_dimension``.
    """

    if isinstance(delta_counted_bytes, bool):
        raise PF3MaterializationError("counted-byte delta must be an integer")
    if any(not math.isfinite(value) for value in (delta_D_dimension, delta_D_joint)):
        raise PF3MaterializationError("distortion deltas must be finite")
    base = {
        "delta_counted_bytes_dimension": delta_counted_bytes,
        "delta_D_dimension": delta_D_dimension,
        "delta_D_joint_guard": delta_D_joint,
        "rate_break_even_score_per_byte": RATE_BREAK_EVEN_SCORE_PER_BYTE,
        "lambda_bytes_per_D_dimension": None,
        "distortion_reduction_per_byte": None,
        "passes_rate_break_even": False,
        "actionable_for_rd1": False,
    }
    if delta_D_dimension >= 0.0:
        return {
            **base,
            "status": "NONACTIONABLE_INTENDED_DIMENSION_NOT_IMPROVED",
        }
    if delta_D_joint >= 0.0:
        return {
            **base,
            "status": "NONACTIONABLE_JOINT_SPILL_ERASES_DIMENSION_GAIN",
        }
    if delta_counted_bytes <= 0:
        return {
            **base,
            "status": "DOMINATING_NONPOSITIVE_RATE_DELTA_REQUIRES_EDGE_REORDER",
        }
    reduction = -delta_D_joint / delta_counted_bytes
    return {
        **base,
        "lambda_bytes_per_D_dimension": delta_counted_bytes / -delta_D_dimension,
        "distortion_reduction_per_byte": reduction,
        "passes_rate_break_even": reduction > RATE_BREAK_EVEN_SCORE_PER_BYTE,
        "status": "FINITE_CONDITIONAL_COORDINATE_SECANT_NOT_RD1_EDGE",
    }


def materialized_bucket_report(
    occupied_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize five-edge PF3 coverage and exact per-bucket blockers."""

    if len(occupied_rows) != OCCUPIED_BUCKET_COUNT:
        raise PF3MaterializationError("PF3 requires exactly 37 occupied metric buckets")
    bucket_ids = [row.get("bucket_id") for row in occupied_rows]
    if any(not isinstance(value, str) for value in bucket_ids) or len(set(bucket_ids)) != len(bucket_ids):
        raise PF3MaterializationError("occupied bucket identities must be unique strings")
    candidates_by_bucket: dict[str, list[dict[str, Any]]] = {
        str(bucket_id): [] for bucket_id in bucket_ids
    }
    for candidate in candidate_rows:
        if candidate.get("five_edges_complete") is not True:
            continue
        candidate_id = candidate.get("candidate_id")
        measurements = candidate.get("bucket_measurements")
        if not isinstance(candidate_id, str) or not isinstance(measurements, list):
            raise PF3MaterializationError("candidate bucket evidence is malformed")
        for measurement in measurements:
            if not isinstance(measurement, Mapping):
                raise PF3MaterializationError("candidate bucket measurement is malformed")
            bucket_id = measurement.get("bucket_id")
            if bucket_id in candidates_by_bucket:
                candidates_by_bucket[str(bucket_id)].append(
                    {
                        "candidate_id": candidate_id,
                        "coordinate_id": candidate["coordinate_id"],
                        "direction_id": candidate["direction_id"],
                        "checkpoint_sha256": (
                            candidate["checkpoint_sha256"]
                            if "checkpoint_sha256" in candidate
                            else candidate["source_checkpoint_sha256"]
                        ),
                        "event_count": measurement["event_count"],
                        "event_delta_errors": measurement["event_delta_errors"],
                        "delta_D_joint": candidate["five_pf3_edges"][
                            "candidate_delta"
                        ]["delta_D_joint"],
                        "conditional_price_status": candidate[
                            "conditional_coordinate_price"
                        ]["status"],
                    }
                )
    rows = []
    for source in occupied_rows:
        bucket_id = str(source["bucket_id"])
        candidates = sorted(
            candidates_by_bucket[bucket_id],
            key=lambda row: (
                int(row["event_delta_errors"]),
                -int(row["event_count"]),
                str(row["candidate_id"]),
            ),
        )
        finite_candidates = [
            row
            for row in candidates
            if row["conditional_price_status"]
            == "FINITE_CONDITIONAL_COORDINATE_SECANT_NOT_RD1_EDGE"
        ]
        if not candidates:
            finite_price_blocker = "NO_EXISTING_RG3_COORDINATE_HIT_THIS_PF2_BUCKET"
        elif not finite_candidates and all(
            float(row["delta_D_joint"]) >= 0.0 for row in candidates
        ):
            finite_price_blocker = (
                "ALL_MEASURED_SAME_OBJECT_COORDINATE_EDGES_WORSEN_JOINT_D"
            )
        elif not finite_candidates:
            finite_price_blocker = (
                "NO_EVENT_CORRECTING_TARGET_STRATUM_WITH_POSITIVE_RATE_AND_JOINT_GAIN"
            )
        else:
            finite_price_blocker = None
        rows.append(
            {
                "bucket_id": bucket_id,
                "class_pair": source.get("class_pair"),
                "class_stratum": source.get("class_stratum"),
                "g4_temporal_class": source.get("g4_temporal_class"),
                "event_count": source.get("event_count"),
                "fully_materialized": bool(candidates),
                "five_materialization_fields": (
                    {
                        "receiver_object_builder": "MEASURED",
                        "realized_uint8_quantum": "MEASURED",
                        "candidate_delta": "MEASURED",
                        "dimension_rate_home": "MEASURED",
                        "coder_payload_owner": "MEASURED",
                    }
                    if candidates
                    else {
                        "receiver_object_builder": "NO_EXISTING_RG3_EVENT_HIT",
                        "realized_uint8_quantum": "NO_EXISTING_RG3_EVENT_HIT",
                        "candidate_delta": "NO_EXISTING_RG3_EVENT_HIT",
                        "dimension_rate_home": "NO_EXISTING_RG3_EVENT_HIT",
                        "coder_payload_owner": "NO_EXISTING_RG3_EVENT_HIT",
                    }
                ),
                "candidate_count": len(candidates),
                "candidate_evidence": candidates,
                "blocker": None if candidates else "NO_EXISTING_RG3_COORDINATE_HIT_THIS_PF2_BUCKET",
                "finite_conditional_price_candidate_count": len(finite_candidates),
                "finite_price_blocker": finite_price_blocker,
            }
        )
    materialized = sum(row["fully_materialized"] for row in rows)
    return {
        "schema": "ddm_pf3_materialized_bucket_report.v1",
        "occupied_bucket_count": len(rows),
        "fully_materialized_occupied_bucket_count": materialized,
        "still_unmaterialized_occupied_bucket_count": len(rows) - materialized,
        "rows": rows,
    }


def build_rd1_fail_closed_backfill(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    conditional_coordinate_rows: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Carry all 162 RD1 cells forward without cross-edge price fabrication."""

    if len(source_rows) != RD1_CELL_COUNT:
        raise PF3MaterializationError("RD1 backfill requires exactly 162 source cells")
    keys = {
        (
            row.get("dual_index"),
            row.get("stratum"),
            row.get("scorer_visibility"),
            row.get("g4_temporal_class"),
        )
        for row in source_rows
    }
    if len(keys) != RD1_CELL_COUNT:
        raise PF3MaterializationError("RD1 source cell identities are not unique")
    finite_conditional = [
        dict(row)
        for row in conditional_coordinate_rows
        if row.get("lambda_bytes_per_D_dimension") is not None
    ]
    cells = []
    for row in source_rows:
        copied = dict(row)
        copied.update(
            {
                "pf3_materialization_status": (
                    "CONDITIONAL_COORDINATE_EVIDENCE_AVAILABLE_BUT_EDGE_IDENTITY_ABSENT"
                    if any(
                        value.get("target_stratum") == row.get("stratum")
                        and value.get("g4_temporal_class") == row.get("g4_temporal_class")
                        for value in finite_conditional
                    )
                    else "NO_FINITE_CONDITIONAL_COORDINATE_EVIDENCE"
                ),
                "lambda_bytes_per_D_dimension": None,
                "actionable_for_train_decision": False,
                "lambda_measurement_status": RD1_EDGE_APPLICATION_BLOCKER,
                "score_claim": False,
            }
        )
        cells.append(copied)
    return {
        "schema": "ddm_pf3_rd1_dual_backfill.v1",
        "source": dict(source),
        "source_cell_count": len(source_rows),
        "conditional_coordinate_price_count": len(finite_conditional),
        "rung_measured_cell_count": 0,
        "lambda_measured_cell_count": 0,
        "still_null_lambda_cell_count": len(cells),
        "actionable_cell_count": 0,
        "blocker": RD1_EDGE_APPLICATION_BLOCKER,
        "blocker_explanation": (
            "PF3 measured V19C-to-coordinate edges; RD1 cells retain the identities "
            "of three different hull edges. No typed counted application operator "
            "maps either endpoint of those hull edges to the PF3 coordinate packet."
        ),
        "cells": cells,
        "score_claim": False,
        "main_landing_review_required": True,
        "verdict_scope": "PRECONDITION; null prices are not zero and not a family negative",
    }
