#!/usr/bin/env python3
"""Fail-closed preflight for the DDM MS2R tolerance-capped solve.

This tool does not rebuild MS2 and does not estimate missing metric rows.  It
consumes the existing MS3 loader with ``require_complete=True``.  When the
bundle is partial, it writes a durable blocker receipt and a 162-cell RD1
supplement that preserves every train-decision lambda as NULL.

An admitted homotopy belongs to the existing scorer-metric solve apparatus and
must be measured through the real receiver/coder/scorers.  This preflight exits
2 before any such actuation when that scientific custody is unavailable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Final

REPO: Final = Path(__file__).resolve().parents[1]
SRC: Final = REPO / "src"
for local_path in (str(SRC), str(REPO)):
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

from tac.optimization.ddm_metric_custody_bundle import (  # noqa: E402
    MetricCustodyError,
    load_metric_custody_bundle,
)

SCHEMA: Final = "ddm_ms2r_tolerance_capped_solve_preflight.v1"
DUAL_SCHEMA: Final = "ddm_ms2r_rd1_dual_backfill.v1"
RUN_ID: Final = "ddm_ms2r_tolerance_capped_solve_20260724T152730Z"
LANE_ID: Final = "lane_ddm_ms2r_tolerance_capped_solve_20260724"
AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
POINTER: Final = "0.1910828242 [contest-CPU]"
SCORED_PIXELS: Final = 600 * 512 * 384
BOX_ALLOWED_ERRORS: Final = 136_839
EXPECTED_DUAL_CELLS: Final = 162
AUTHORITY_SHA256: Final = (
    "75cc043e842caf1206f9ea5a2b2d45d7fdf157596eb949367592674a9f8ada5d"
)
DEFAULT_OUTPUT_ROOT: Final = REPO / (
    ".omx/research/ddm_ms2r_tolerance_capped_solve_20260724T152730Z"
)
DEFAULT_BUNDLE: Final = REPO / (
    ".omx/research/ddm_ms4_metric_producers_and_measurement_20260724T042005Z/"
    "full_n600/BUNDLE-PARTIAL.json"
)
DEFAULT_RD1_DUALS: Final = REPO / (
    ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
    "typed_dimension_duals_effective_quantum.json"
)
DEFAULT_RD1_FRONTIER: Final = REPO / (
    ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
    "typed_R_D_frontier_rows_v5.json"
)
DEFAULT_RG3_SUMMARY: Final = REPO / (
    ".omx/research/ddm_rg3_residual_family_productions_20260724T110418Z/"
    "ddm_rg3_receiver_support_summary.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def _artifact(path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": _display_path(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
        "role": role,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_rd1_duals(value: dict[str, Any]) -> list[dict[str, Any]]:
    if value.get("schema") != "ddm_rd1_dimension_duals_effective_quantum.v1":
        raise ValueError("RD1 dual supplement schema drifted")
    duals = value.get("dimension_duals")
    if not isinstance(duals, dict) or duals.get("schema") != (
        "ddm_rd1_typed_dimension_duals.v1"
    ):
        raise ValueError("RD1 typed dual schema drifted")
    rows = duals.get("bucket_rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_DUAL_CELLS:
        raise ValueError("RD1 typed dual cube must contain exactly 162 rows")
    if any(row.get("lambda_bytes_per_D_dimension") is not None for row in rows):
        raise ValueError("RD1 source no longer contains 162 NULL train prices")
    if any(row.get("actionable_for_train_decision") is not False for row in rows):
        raise ValueError("RD1 source contains an actionable train-decision cell")
    axes = duals.get("axes")
    if not isinstance(axes, dict):
        raise ValueError("RD1 typed dual axes are missing")
    expected_keys = {
        (
            int(row["dual_index"]),
            str(row["stratum"]),
            str(row["scorer_visibility"]),
            str(row["g4_temporal_class"]),
        )
        for row in rows
    }
    cartesian = set(
        itertools.product(
            (1, 2, 3),
            axes["stratum"],
            axes["scorer_visibility"],
            axes["g4_temporal_class"],
        )
    )
    if expected_keys != cartesian:
        raise ValueError("RD1 typed dual cube does not cover its sealed Cartesian axes")
    return rows


def _unique_candidate(
    frontier: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    if frontier.get("schema") != "ddm_rd1_typed_rate_distortion_rows.v4":
        raise ValueError("RD1 typed frontier schema drifted")
    matches = [
        row for row in frontier.get("rows", []) if row.get("candidate_id") == candidate_id
    ]
    if not matches:
        raise ValueError(f"RD1 frontier is missing {candidate_id}")
    first = matches[0]
    custody = (
        first["counted_bytes"],
        first["d_seg"],
        first["d_pose"],
        first["S_composed"],
        first["receiver_closure"],
    )
    if any(
        (
            row["counted_bytes"],
            row["d_seg"],
            row["d_pose"],
            row["S_composed"],
            row["receiver_closure"],
        )
        != custody
        for row in matches
    ):
        raise ValueError(f"RD1 repeated rows disagree for {candidate_id}")
    return dict(first)


def _null_backfill(
    source_rows: list[dict[str, Any]],
    *,
    source: dict[str, Any],
    blocker: str,
) -> dict[str, Any]:
    keys = (
        "dual_index",
        "left_candidate_id",
        "right_candidate_id",
        "stratum",
        "scorer_visibility",
        "g4_temporal_class",
    )
    return {
        "schema": DUAL_SCHEMA,
        "evidence_axis": AXIS,
        "score_claim": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "source": source,
        "source_cell_count": len(source_rows),
        "measured_cell_count": 0,
        "still_null_cell_count": len(source_rows),
        "actionable_cell_count": 0,
        "pooling": "FORBIDDEN_NON_ADDITIVE_POOLS",
        "blocker": blocker,
        "cells": [
            {
                **{key: row[key] for key in keys},
                "lambda_bytes_per_D_dimension": None,
                "effective_quantum_D": row.get("effective_quantum_D"),
                "source_status": row["status"],
                "measurement_status": "STILL_NULL_MS3_BUNDLE_PARTIAL",
                "actionable_for_train_decision": False,
                "score_claim": False,
            }
            for row in source_rows
        ],
        "verdict": "NO_BACKFILL_PERFORMED; 162_OF_162_STILL_NULL",
        "main_landing_review_required": True,
    }


def build_fail_closed_receipt(
    *,
    bundle_path: Path,
    rd1_duals_path: Path,
    rd1_frontier_path: Path,
    rg3_summary_path: Path,
    output_root: Path,
    finished_at_utc: str,
) -> tuple[dict[str, Any], Path, Path]:
    """Rehash all inputs and emit a blocker only after the strict loader refuses."""

    bundle = load_metric_custody_bundle(
        bundle_path,
        repository_root=REPO,
        require_complete=False,
    )
    try:
        load_metric_custody_bundle(
            bundle_path,
            repository_root=REPO,
            require_complete=True,
        )
    except MetricCustodyError as exc:
        loader_refusal = str(exc)
    else:
        raise RuntimeError(
            "bundle is COMPLETE; this blocker-only preflight must hand off to "
            "the measured resumable MS2 homotopy driver"
        )

    rd1_duals = _read_json(rd1_duals_path)
    source_rows = _validate_rd1_duals(rd1_duals)
    rd1_frontier = _read_json(rd1_frontier_path)
    exact_row = _unique_candidate(rd1_frontier, "c1_exact_solved_n600")
    proposal_knee = _unique_candidate(
        rd1_frontier,
        "statistics_hard_analytic_composed_frame1",
    )
    exact_error_float = float(exact_row["d_seg"]) * SCORED_PIXELS
    exact_errors = round(exact_error_float)
    if not math.isclose(exact_error_float, exact_errors, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("C1 exact-row d_seg does not map to an integer error count")

    rg3_summary = _read_json(rg3_summary_path)
    coverage = rg3_summary.get("g3_top24_coverage", {})
    if (
        rg3_summary.get("producer_rerun_eligible") is not False
        or coverage.get("coverage_proven") is not False
    ):
        raise ValueError("RG3 producer eligibility no longer matches the blocker premise")
    missing_blocks = coverage.get("missing_blocks")
    if not isinstance(missing_blocks, list) or not missing_blocks:
        raise ValueError("RG3 blocker inventory is missing")

    output_root.mkdir(parents=True, exist_ok=True)
    dual_path = output_root / "01_rd1_dual_backfill.json"
    dual_payload = _null_backfill(
        source_rows,
        source=_artifact(rd1_duals_path, role="rd1_162_null_dual_source"),
        blocker=loader_refusal,
    )
    _atomic_write_json(dual_path, dual_payload)

    component_status = {
        component_id.value: {
            "status": receipt.status.value,
            "sample_count": receipt.sample_count,
            "scorer_batch_size": receipt.scorer_batch_size,
            "blockers": list(receipt.blockers),
        }
        for component_id, receipt in bundle.components.items()
    }
    receipt = {
        "schema": SCHEMA,
        "run_id": RUN_ID,
        "lane_id": LANE_ID,
        "finished_at_utc": finished_at_utc,
        "authority": {
            "authority_sha256": AUTHORITY_SHA256,
            "evidence_axis": AXIS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "main_landing_review_required": True,
            "pointer": POINTER,
            "pointer_moved": False,
            "local_cost_usd": 0,
            "torch_threads_required": 4,
        },
        "input_custody": {
            "metric_bundle": _artifact(bundle_path, role="ms3_loader_manifest"),
            "rd1_duals": _artifact(rd1_duals_path, role="rd1_162_null_dual_source"),
            "rd1_frontier": _artifact(
                rd1_frontier_path,
                role="rd1_proposal_channel_frontier",
            ),
            "rg3_summary": _artifact(
                rg3_summary_path,
                role="rg3_terminal_assignment_status",
            ),
        },
        "metric_bundle_gate": {
            "loader": (
                "tac.optimization.ddm_metric_custody_bundle:"
                "load_metric_custody_bundle"
            ),
            "require_complete": True,
            "status": bundle.status.value,
            "headline_flags": bundle.headline_flags(),
            "blockers": list(bundle.blockers),
            "component_status": component_status,
            "loader_refusal": loader_refusal,
            "admitted": False,
        },
        "rg3_assignment_gate": {
            "producer_rerun_eligible": False,
            "fully_joined_top24_pair_count": coverage["fully_joined_pair_count"],
            "missing_block_count": coverage["missing_block_count"],
            "missing_blocks": missing_blocks,
            "status": "PARTIAL_ASSIGNMENT_25_BLOCKS_REMAIN",
        },
        "exact_row_arithmetic": {
            "source_candidate_id": exact_row["candidate_id"],
            "scored_pixels": SCORED_PIXELS,
            "measured_d_seg": exact_row["d_seg"],
            "measured_error_count": exact_errors,
            "charter_rounded_error_count": 17_931,
            "rounding_difference_errors": 17_931 - exact_errors,
            "box_allowed_errors": BOX_ALLOWED_ERRORS,
            "allowance_over_measured_exact_errors": (
                BOX_ALLOWED_ERRORS / exact_errors
            ),
            "status": (
                "MEASURED_SOURCE_ROW_REDERIVATION; "
                "17,931_IS_ROUNDED_NOT_EXACT_CUSTODY"
            ),
        },
        "homotopy": {
            "launched": False,
            "rung_count": 0,
            "rungs": [],
            "uniform_homotopy_role": "LABELED_CONTROL_ONLY",
            "waterfill_required": True,
            "global_error_allowance": BOX_ALLOWED_ERRORS,
            "waterfill_axes": [
                "stratum",
                "scorer_visibility",
                "g4_temporal_class",
            ],
            "per_rung_required_columns": [
                "allowed_errors_global",
                "per_block_allocated_allowance",
                "per_block_realized_errors",
                "per_block_measured_dual_bytes_per_error",
                "achieved_d_seg",
                "d_pose_through_active_tube",
                "raw_compact_bytes",
                "best_coded_bytes",
                "winning_coder",
                "joint_S_same_parseback_object",
            ],
            "coder_race_required": [
                "RAW_COMPACT",
                "ZLIB9",
                "RAW_LZMA1",
                "ORDER1_CONTEXT_ARITHMETIC",
                "E4_BROTLI_Q11",
                "G4_FREE_DECODER_DERIVED_SPATIAL_CONTEXT",
            ],
            "non_additive_pool_rule": "SAME_POOL_TOLERANCES_COMPETE_NEVER_ADD",
            "status": "NOT_RUN_FAIL_CLOSED_AT_MS3_BUNDLE_GATE",
        },
        "visibility_partition_next_run_contract": {
            "both_blind_gauge": {
                "tolerance": 0,
                "counted_bytes": 0,
                "authority": "#580 exact resize null projector plus #401 blind coordinates",
                "required_check": "REVERIFY_EVERY_MOVE_THROUGH_UINT8_R_AND_PARSEBACK",
            },
            "seg_only": {
                "coordinate_family": "fine_scale_chroma_below_pose_2x2_box_support",
                "pose_tolerance": 0,
            },
            "pose_only": {
                "coordinate_family": "frame_0",
                "seg_tolerance": 0,
                "seg_description_bytes": 0,
            },
            "joint": {
                "coordinate_family": "shared_frame_1_or_pose_visible_coordinates",
                "metric": "rank4_margin_fisher_plus_pose6_plus_composite_R",
            },
            "generic_blind_fill": {
                "camera_pixels_per_frame": 230_904,
                "counted_bytes": 0,
            },
            "per_type_mass": None,
            "per_type_byte_spend": None,
            "status": "UNMEASURED_UNTIL_BUNDLE_COMPLETE_AND_UINT8_REVALIDATED",
        },
        "rd1_dual_backfill": {
            "artifact": _artifact(dual_path, role="ms2r_rd1_dual_backfill"),
            "source_cell_count": EXPECTED_DUAL_CELLS,
            "measured_cell_count": 0,
            "still_null_cell_count": EXPECTED_DUAL_CELLS,
            "status": "NO_BACKFILL_PERFORMED_BUNDLE_PARTIAL",
        },
        "knee_comparison": {
            "ms2r_waterfilled_channel_knee": None,
            "rd1_proposal_channel_knee": {
                "epistemic_status": (
                    "MEASURED_PROPOSAL_CHANNEL_UPPER_BOUND_QUOTED_NOT_RERUN"
                ),
                "candidate_id": proposal_knee["candidate_id"],
                "counted_bytes": proposal_knee["counted_bytes"],
                "d_seg": proposal_knee["d_seg"],
                "d_pose": proposal_knee["d_pose"],
                "joint_S": proposal_knee["S_composed"],
                "receiver_closure": proposal_knee["receiver_closure"],
            },
            "channel_suboptimality_price": None,
            "status": "NULL_NO_MS2R_RUNG",
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
            "selected_future_bulk_tier": (
                "/Volumes/VertigoDataTier/pact"
                if Path("/Volumes/VertigoDataTier/pact").is_dir()
                else None
            ),
            "cleanup_action": "NONE_NO_BULK_CREATED",
        },
        "verdict": "BLOCKED_MS3_BUNDLE_PARTIAL_PF2_BUCKET_INPUT_ASSIGNMENT_ABSENT",
        "verdict_scope": (
            "INSTANCE(current SHA-custodied MS4 full-n600 bundle) x "
            "FORMULATION(metric-active tolerance-waterfill preflight); "
            "no FAMILY or PARADIGM negative"
        ),
        "next_exact_action": (
            "Close the 25 RG3 pair/bucket assignment obligations, rerun MS4 to "
            "BUNDLE-COMPLETE, then resume at this require_complete loader gate."
        ),
        "main_landing_review_required": True,
    }
    receipt_path = output_root / "00_preflight_receipt.json"
    _atomic_write_json(receipt_path, receipt)
    return receipt, receipt_path, dual_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--rd1-duals", type=Path, default=DEFAULT_RD1_DUALS)
    parser.add_argument("--rd1-frontier", type=Path, default=DEFAULT_RD1_FRONTIER)
    parser.add_argument("--rg3-summary", type=Path, default=DEFAULT_RG3_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--finished-at-utc", default=None)
    parser.add_argument("--register-equation", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    finished = args.finished_at_utc or dt.datetime.now(dt.UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    receipt, receipt_path, dual_path = build_fail_closed_receipt(
        bundle_path=args.bundle,
        rd1_duals_path=args.rd1_duals,
        rd1_frontier_path=args.rd1_frontier,
        rg3_summary_path=args.rg3_summary,
        output_root=args.output_root,
        finished_at_utc=finished,
    )
    registered: list[str] = []
    if args.register_equation:
        from tac.canonical_equations.ddm_ms2r_tolerance_capped_solve_20260724 import (
            populate_ddm_ms2r_tolerance_capped_solve,
        )

        equation = populate_ddm_ms2r_tolerance_capped_solve(
            source_receipt=receipt_path,
            agent="codex",
            subagent_id="ddm_ms2r_tolerance_capped_solve_20260724T152730Z",
        )
        registered.append(equation.equation_id)
    print(
        json.dumps(
            {
                "receipt": _artifact(receipt_path, role="ms2r_preflight_receipt"),
                "dual_backfill": _artifact(
                    dual_path,
                    role="ms2r_rd1_dual_backfill",
                ),
                "verdict": receipt["verdict"],
                "registered_equation_ids": registered,
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
