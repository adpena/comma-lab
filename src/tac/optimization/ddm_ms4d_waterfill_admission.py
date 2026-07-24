# SPDX-License-Identifier: MIT
"""Post-bundle admission gate for the DDM MS4D tolerance waterfill.

A complete scorer-metric bundle is necessary but not sufficient to construct a
receiver object.  This module keeps that boundary explicit.  It admits the
strict MS3 bundle, inventories the scorer-intrinsic rows, and refuses a
waterfill when the bundle has no realized uint8 coordinate, receiver-object
builder, candidate delta, or counted-byte rate home.

The refusal is deliberately downstream of ``require_complete=True``.  It does
not weaken the MS3 gate and it does not turn a Fisher/Hessian value into a fake
actuator, byte price, coder result, or parse-back object.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from tac.canonical_equations.ddm_ms2r_tolerance_capped_solve_20260724 import (
    tolerance_capped_rung_score,
)
from tac.optimization.ddm_lambda_continuation_frontier import publish_immutable_json
from tac.optimization.ddm_metric_custody_bundle import (
    DIRECT_METRIC_MODE,
    EVIDENCE_AXIS,
    SEG_DIRECT_DATA_SCHEMA,
    load_metric_custody_bundle,
)
from tac.repo_io import sha256_file

SCHEMA: Final = "ddm_ms4d_waterfill_post_admission.v1"
DUAL_BACKFILL_SCHEMA: Final = "ddm_ms4d_rd1_dual_backfill.v1"
RUN_ID: Final = "ddm_ms4d_direct_metric_completion_20260724T155932Z"
LANE_ID: Final = "lane_ddm_ms4d_direct_metric_completion_20260724"
POINTER: Final = "0.1910828242 [contest-CPU]"
SCORED_PIXELS: Final = 600 * 512 * 384
ALLOWED_ERRORS: Final = 136_839
EXPECTED_DUAL_CELLS: Final = 162
EXPECTED_DIRECT_BLOCKS: Final = 25
EXPECTED_OCCUPIED_BUCKETS: Final = 37
EXPECTED_EMPTY_BUCKETS: Final = 1_163
BLOCKER: Final = "PF3_RECEIVER_OBJECT_AND_TYPED_RATE_HOME_ABSENT"

CODER_RACE: Final = (
    "RAW_COMPACT",
    "ZLIB9",
    "RAW_LZMA1",
    "ORDER1_CONTEXT_ARITHMETIC",
    "E4_BROTLI_Q11",
    "G4_FREE_DECODER_DERIVED_SPATIAL_CONTEXT",
)
MATERIALIZATION_FIELDS: Final = (
    "receiver_object_builder",
    "realized_uint8_quantum",
    "candidate_delta",
    "dimension_rate_home",
    "coder_payload_owner",
)


class MS4DWaterfillAdmissionError(ValueError):
    """A post-admission source or custody edge differs from its sealed contract."""


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MS4DWaterfillAdmissionError(f"JSON source must be an object: {path}")
    return value


def _resolve_artifact(repository_root: Path, reference: Mapping[str, Any]) -> Path:
    raw = reference.get("path")
    expected = reference.get("sha256")
    if not isinstance(raw, str) or not isinstance(expected, str):
        raise MS4DWaterfillAdmissionError("artifact reference lacks path/SHA-256")
    path = Path(raw)
    resolved = (repository_root / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)
    if sha256_file(resolved) != expected:
        raise MS4DWaterfillAdmissionError(f"artifact SHA-256 differs: {resolved}")
    return resolved


def _artifact(path: Path, *, repository_root: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        display = str(resolved.relative_to(repository_root))
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "role": role,
    }


def candidate_materialization_gaps(seg_data: Mapping[str, Any]) -> dict[str, Any]:
    """Inventory the exact missing edges between direct metrics and real rungs."""

    if seg_data.get("schema") != SEG_DIRECT_DATA_SCHEMA:
        raise MS4DWaterfillAdmissionError("Seg component is not the strict direct schema")
    if seg_data.get("metric_mode") != DIRECT_METRIC_MODE:
        raise MS4DWaterfillAdmissionError("Seg component is not scorer-intrinsic direct mode")
    rows = seg_data.get("rows")
    blocks = seg_data.get("direct_blocks")
    if not isinstance(rows, list) or len(rows) != 1_200:
        raise MS4DWaterfillAdmissionError("direct Seg component must contain 1,200 rows")
    if not isinstance(blocks, list) or len(blocks) != EXPECTED_DIRECT_BLOCKS:
        raise MS4DWaterfillAdmissionError("direct Seg component must contain exactly 25 residual blocks")
    if any(
        row.get("actuation_status") != "UNREACHABLE_BY_COUNTED_COORDINATES"
        for row in blocks
        if isinstance(row, Mapping)
    ):
        raise MS4DWaterfillAdmissionError("direct residual actuation typing differs")
    occupied = sum(
        isinstance(row, Mapping) and isinstance(row.get("event_count"), int) and row["event_count"] > 0
        for row in rows
    )
    empty = sum(
        isinstance(row, Mapping) and row.get("event_count") == 0
        for row in rows
    )
    if (occupied, empty) != (EXPECTED_OCCUPIED_BUCKETS, EXPECTED_EMPTY_BUCKETS):
        raise MS4DWaterfillAdmissionError("direct occupied/empty PF2 population differs")

    field_counts = {
        field: sum(
            isinstance(row, Mapping) and row.get(field) is not None
            for row in rows
        )
        + sum(
            isinstance(row, Mapping) and row.get(field) is not None
            for row in blocks
        )
        for field in MATERIALIZATION_FIELDS
    }
    fully_materialized = sum(
        isinstance(row, Mapping)
        and row.get("event_count", 0) > 0
        and all(row.get(field) is not None for field in MATERIALIZATION_FIELDS)
        for row in rows
    )
    return {
        "metric_bucket_count": len(rows),
        "occupied_metric_bucket_count": occupied,
        "exact_empty_metric_bucket_count": empty,
        "direct_unreachable_pair_bucket_count": len(blocks),
        "direct_unreachable_included_in_metric_and_pricing": True,
        "direct_unreachable_excluded_from_proposal_allocation": True,
        "materialization_field_counts": field_counts,
        "fully_materialized_occupied_bucket_count": fully_materialized,
        "candidate_materialization_ready": fully_materialized > 0,
        "blocker": None if fully_materialized > 0 else BLOCKER,
    }


def _validate_rd1_rows(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    dimension = value.get("dimension_duals")
    if (
        value.get("schema") != "ddm_rd1_dimension_duals_effective_quantum.v1"
        or not isinstance(dimension, Mapping)
        or dimension.get("schema") != "ddm_rd1_typed_dimension_duals.v1"
    ):
        raise MS4DWaterfillAdmissionError("RD1 dual source schema differs")
    rows = dimension.get("bucket_rows")
    axes = dimension.get("axes")
    if not isinstance(rows, list) or len(rows) != EXPECTED_DUAL_CELLS or not isinstance(axes, Mapping):
        raise MS4DWaterfillAdmissionError("RD1 dual cube is not exactly 162 typed cells")
    keys = {
        (
            row.get("dual_index"),
            row.get("stratum"),
            row.get("scorer_visibility"),
            row.get("g4_temporal_class"),
        )
        for row in rows
        if isinstance(row, Mapping)
    }
    expected = set(
        itertools.product(
            (1, 2, 3),
            axes.get("stratum", ()),
            axes.get("scorer_visibility", ()),
            axes.get("g4_temporal_class", ()),
        )
    )
    if keys != expected:
        raise MS4DWaterfillAdmissionError("RD1 dual cube Cartesian identity differs")
    return [dict(row) for row in rows]


def build_still_null_backfill(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    source: Mapping[str, Any],
    complete_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve all train prices as NULL when no measured rung exists.

    The complete direct metric bundle is valid context for every RD1 cell, but
    it is not itself a measured candidate delta or byte dual.  The distinction
    is recorded cell by cell so bundle admission cannot masquerade as rung
    measurement.
    """

    if len(source_rows) != EXPECTED_DUAL_CELLS:
        raise MS4DWaterfillAdmissionError("RD1 backfill requires exactly 162 source cells")
    identity_fields = (
        "dual_index",
        "left_candidate_id",
        "right_candidate_id",
        "stratum",
        "scorer_visibility",
        "g4_temporal_class",
    )
    return {
        "schema": DUAL_BACKFILL_SCHEMA,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "source": dict(source),
        "complete_metric_bundle": dict(complete_bundle),
        "source_cell_count": len(source_rows),
        "metric_bundle_context_cell_count": len(source_rows),
        "rung_measured_cell_count": 0,
        "lambda_measured_cell_count": 0,
        "still_null_lambda_cell_count": len(source_rows),
        "actionable_cell_count": 0,
        "pooling": "FORBIDDEN_NON_ADDITIVE_POOLS",
        "blocker": BLOCKER,
        "cells": [
            {
                **{field: row[field] for field in identity_fields},
                "metric_bundle_status": "COMPLETE_DIRECT_SCORER_METRIC_CONTEXT",
                "rung_measurement_status": (
                    "STILL_NULL_NO_MATERIALIZED_SAME_OBJECT_RUNG"
                ),
                "source_metric_status": row.get("status"),
                "effective_quantum_D": row.get("effective_quantum_D"),
                "lambda_bytes_per_D_dimension": None,
                "lambda_measurement_status": (
                    "STILL_NULL_CANDIDATE_DELTA_X_DIMENSION_RATE_HOME_ABSENT"
                ),
                "actionable_for_train_decision": False,
                "score_claim": False,
            }
            for row in source_rows
        ],
        "verdict": (
            "162_OF_162_CELLS_HAVE_COMPLETE_METRIC_CONTEXT; "
            "0_OF_162_RUNG_DELTAS_MEASURED; 0_OF_162_BYTE_DUALS_MEASURED; "
            "162_OF_162_PRICES_STILL_NULL"
        ),
        "main_landing_review_required": True,
    }


def _unique_frontier_row(frontier: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    if frontier.get("schema") != "ddm_rd1_typed_rate_distortion_rows.v4":
        raise MS4DWaterfillAdmissionError("RD1 frontier schema differs")
    rows = [
        dict(row)
        for row in frontier.get("rows", ())
        if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id
    ]
    if not rows:
        raise MS4DWaterfillAdmissionError(f"RD1 frontier lacks {candidate_id}")
    fields = ("counted_bytes", "d_seg", "d_pose", "S_composed", "receiver_closure")
    if any(tuple(row[field] for field in fields) != tuple(rows[0][field] for field in fields) for row in rows[1:]):
        raise MS4DWaterfillAdmissionError(f"RD1 duplicate rows disagree for {candidate_id}")
    return rows[0]


def registered_callable_control(
    row: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    """Replay one settled same-object endpoint through the registered row law."""

    receiver_closure = row.get("receiver_closure")
    if receiver_closure not in {
        "archive_receiver_closed",
        "measurement_harness_receiver_closed",
    }:
        raise MS4DWaterfillAdmissionError(
            "settled control lacks an admitted exact receiver-closure status"
        )
    errors_float = float(row["d_seg"]) * SCORED_PIXELS
    errors = round(errors_float)
    if not math.isclose(errors_float, errors, rel_tol=0.0, abs_tol=1e-9):
        raise MS4DWaterfillAdmissionError("frontier d_seg does not carry an integer error count")
    result = tolerance_capped_rung_score(
        seg_errors=errors,
        scored_pixels=SCORED_PIXELS,
        d_pose=float(row["d_pose"]),
        raw_compact_bytes=int(row["counted_bytes"]),
        best_coded_bytes=int(row["counted_bytes"]),
        allowed_errors=ALLOWED_ERRORS,
        bundle_complete=True,
        parseback_exact=True,
        uint8_reverified=True,
    )
    if not math.isclose(result["joint_S"], float(row["S_composed"]), rel_tol=0.0, abs_tol=2e-12):
        raise MS4DWaterfillAdmissionError("registered callable differs from the settled frontier row")
    return {
        "role": role,
        "candidate_id": row["candidate_id"],
        "receiver_closure": receiver_closure,
        "seg_errors": errors,
        **result,
        "epistemic_status": "MEASURED_SETTLED_ENDPOINT_REPLAYED_NOT_A_NEW_RUNG",
        "coder_race_performed": False,
        "raw_compact_equals_existing_counted_object_control": True,
        "score_claim": False,
    }


def _seg_data_from_bundle(bundle_path: Path, repository_root: Path) -> tuple[dict[str, Any], Path]:
    manifest = _read_object(bundle_path)
    component = manifest.get("component_receipts", {}).get("SEG_METRIC")
    if not isinstance(component, Mapping):
        raise MS4DWaterfillAdmissionError("bundle lacks Seg component receipt")
    receipt_path = _resolve_artifact(repository_root, component)
    receipt = _read_object(receipt_path)
    data_reference = receipt.get("data_artifact")
    if not isinstance(data_reference, Mapping):
        raise MS4DWaterfillAdmissionError("Seg receipt lacks data artifact")
    data_path = _resolve_artifact(repository_root, data_reference)
    return _read_object(data_path), data_path


def build_post_admission_refusal(
    *,
    bundle_path: Path,
    rd1_duals_path: Path,
    rd1_frontier_path: Path,
    output_root: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], Path]:
    """Admit the complete metric bundle, then refuse absent rung materialization."""

    bundle = load_metric_custody_bundle(
        bundle_path,
        repository_root=repository_root,
        require_complete=True,
    )
    if not bundle.complete:
        raise MS4DWaterfillAdmissionError("strict loader returned a non-complete bundle")
    seg_data, seg_data_path = _seg_data_from_bundle(bundle_path, repository_root)
    population = candidate_materialization_gaps(seg_data)
    if population["candidate_materialization_ready"]:
        raise MS4DWaterfillAdmissionError(
            "candidate materialization is ready; refusal-only runner must hand off"
        )

    rd1_duals = _read_object(rd1_duals_path)
    source_rows = _validate_rd1_rows(rd1_duals)
    rd1_frontier = _read_object(rd1_frontier_path)
    exact = _unique_frontier_row(rd1_frontier, "c1_exact_solved_n600")
    proposal = _unique_frontier_row(
        rd1_frontier,
        "statistics_hard_analytic_composed_frame1",
    )

    output_root.mkdir(parents=True, exist_ok=True)
    bundle_artifact = _artifact(
        bundle_path,
        repository_root=repository_root,
        role="strict_ms3_complete_metric_bundle",
    )
    dual_source_artifact = _artifact(
        rd1_duals_path,
        repository_root=repository_root,
        role="rd1_162_cell_dual_source",
    )
    dual_payload = build_still_null_backfill(
        source_rows,
        source=dual_source_artifact,
        complete_bundle=bundle_artifact,
    )
    dual_path = output_root / "rd1_dual_backfill_post_admission.json"
    publish_immutable_json(dual_path, dual_payload)

    controls = [
        registered_callable_control(exact, role="inside_error_cap_exact_C1_control"),
        registered_callable_control(proposal, role="outside_error_cap_RD1_knee_control"),
    ]
    coder_rows = [
        {
            "coder": coder,
            "raw_compact_bytes": None,
            "coded_bytes": None,
            "parseback_exact": None,
            "status": "NOT_RUN_NO_MATERIALIZED_RUNG_OBJECT",
        }
        for coder in CODER_RACE
    ]
    receipt = {
        "schema": SCHEMA,
        "run_id": RUN_ID,
        "lane_id": LANE_ID,
        "authority": {
            "evidence_axis": EVIDENCE_AXIS,
            "research_only": True,
            "execution_allowed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "main_landing_review_required": True,
            "pointer": POINTER,
            "pointer_moved": False,
            "local_cost_usd": 0,
            "torch_threads_required": 4,
        },
        "input_custody": {
            "metric_bundle": bundle_artifact,
            "seg_metric_data": _artifact(
                seg_data_path,
                repository_root=repository_root,
                role="direct_scorer_intrinsic_seg_metric",
            ),
            "rd1_duals": dual_source_artifact,
            "rd1_frontier": _artifact(
                rd1_frontier_path,
                repository_root=repository_root,
                role="rd1_proposal_channel_frontier",
            ),
        },
        "metric_bundle_gate": {
            "loader": (
                "tac.optimization.ddm_metric_custody_bundle:"
                "load_metric_custody_bundle"
            ),
            "require_complete": True,
            "admitted": True,
            "status": "COMPLETE",
        },
        "candidate_materialization_gate": {
            **population,
            "required_fields": list(MATERIALIZATION_FIELDS),
            "status": f"REFUSED_{BLOCKER}",
            "verdict_scope": (
                "INSTANCE(current MS4D direct bundle) x "
                "FORMULATION(receiver-object tolerance-waterfill)"
            ),
            "forbidden_inference": (
                "Fisher/Hessian curvature is not a receiver actuator, "
                "realized uint8 quantum, candidate delta, or byte price."
            ),
        },
        "registered_callable_controls": {
            "equation_id": "ddm_tolerance_capped_min_score_waterfill_v1",
            "control_count": len(controls),
            "rows": controls,
            "status": "CALLABLE_REPLAYED_ON_SETTLED_ENDPOINTS_ONLY",
        },
        "homotopy": {
            "launched": False,
            "waterfilled_rung_count": 0,
            "uniform_control_rung_count": 0,
            "one_object_exact_parseback_rung_count": 0,
            "allowed_errors_global": ALLOWED_ERRORS,
            "waterfill_axes": [
                "stratum",
                "scorer_visibility",
                "g4_temporal_class",
            ],
            "unreachable_excluded_from_proposal_allocation": True,
            "unreachable_included_in_metric_and_pricing": True,
            "per_rung_required_columns": [
                "allowed_errors_global",
                "per_block_allocated_allowance",
                "per_block_realized_errors",
                "per_block_measured_dual_bytes_per_error",
                "d_seg",
                "d_pose_tube",
                "raw_compact_bytes",
                "best_coded_bytes",
                "winning_coder",
                "joint_S_same_parseback_object",
            ],
            "status": f"NOT_RUN_{BLOCKER}",
        },
        "visibility_partition": {
            "both_blind_gauge": {
                "camera_pixels_per_frame": 230_904,
                "counted_bytes": 0,
                "proposal_mass": None,
                "status": "STRUCTURAL_PARTITION_ONLY_NO_RUNG",
            },
            "seg_only_fine_chroma": {
                "proposal_mass": None,
                "counted_bytes": None,
                "status": "STILL_NULL_NO_RUNG",
            },
            "frame0_pose_only": {
                "proposal_mass": None,
                "counted_bytes": None,
                "status": "STILL_NULL_NO_RUNG",
            },
            "joint": {
                "proposal_mass": None,
                "counted_bytes": None,
                "status": "STILL_NULL_NO_RUNG",
            },
        },
        "uint8_null_revalidation_532": {
            "proposed_null_move_count": 0,
            "revalidated_move_count": 0,
            "admitted_zero_byte_move_count": 0,
            "status": "NOT_REACHED_NO_RECEIVER_COORDINATE_PROPOSALS",
        },
        "coder_context_race": {
            "required_coders": list(CODER_RACE),
            "rows": coder_rows,
            "winning_coder": None,
            "status": "NOT_RUN_NO_MATERIALIZED_RUNG_OBJECT",
        },
        "rd1_dual_backfill": {
            "artifact": _artifact(
                dual_path,
                repository_root=repository_root,
                role="rd1_post_admission_still_null_duals",
            ),
            "metric_bundle_context_cell_count": EXPECTED_DUAL_CELLS,
            "rung_measured_cell_count": 0,
            "lambda_measured_cell_count": 0,
            "still_null_lambda_cell_count": EXPECTED_DUAL_CELLS,
            "status": "METRIC_CONTEXT_COMPLETE_RUNG_DELTAS_AND_BYTE_PRICES_STILL_NULL",
        },
        "knee_comparison": {
            "ms4d_waterfilled_channel_knee": None,
            "rd1_proposal_channel_knee": controls[1],
            "channel_suboptimality_price": None,
            "status": "NULL_NO_MS4D_WATERFILLED_RUNG",
        },
        "actuation": {
            "torch_invoked": False,
            "receiver_invoked": False,
            "r_operator_invoked": False,
            "frozen_scorer_invoked": False,
            "real_coder_invoked": False,
            "training": False,
            "paid_dispatch": False,
            "exact_contest_eval": False,
            "frontier_mutation": False,
        },
        "storage": {
            "bulk_created": False,
            "selected_bulk_tier": "/Volumes/VertigoDataTier/pact",
            "auto_cleanup": "NO_RUNG_BULK_CREATED",
        },
        "verdict": f"BLOCKED_{BLOCKER}",
        "verdict_scope": (
            "INSTANCE(current MS4D direct bundle) x "
            "FORMULATION(receiver-object tolerance-waterfill); "
            "not a metric, coder, representation-family, or paradigm negative"
        ),
        "next_exact_measurement": (
            "Bind one scorer-recursive receiver coordinate to a deterministic "
            "object builder, realized uint8 quantum, candidate delta, and "
            "dimension rate home; then resume before the first coder race."
        ),
        "main_landing_review_required": True,
    }
    receipt_path = output_root / "waterfill_post_admission_receipt.json"
    publish_immutable_json(receipt_path, receipt)
    return receipt, receipt_path


__all__ = [
    "ALLOWED_ERRORS",
    "BLOCKER",
    "CODER_RACE",
    "DUAL_BACKFILL_SCHEMA",
    "MATERIALIZATION_FIELDS",
    "SCHEMA",
    "MS4DWaterfillAdmissionError",
    "build_post_admission_refusal",
    "build_still_null_backfill",
    "candidate_materialization_gaps",
    "registered_callable_control",
]
