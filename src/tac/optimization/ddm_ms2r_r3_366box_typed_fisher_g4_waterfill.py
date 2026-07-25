# SPDX-License-Identifier: MIT
"""Strict cross-chain admission for the Task #701 366-box waterfill.

The metric bundle can be complete while the requested optimization remains
non-executable.  A typed homotopy additionally needs an active solve-local Pose
tube, complete causal G3 coverage, same-object typed rate homes, and finite RD1
duals.  This module joins those independently sealed surfaces and refuses to
turn metric context or cross-object accounting into a measured rung.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Final

SCHEMA: Final = "ddm_ms2r_r3_366box_typed_fisher_g4_waterfill_receipt.v1"
PREFLIGHT_SCHEMA: Final = "ddm_ms2r_r3_366box_waterfill_preflight.v1"
TABLE_SCHEMA: Final = "ddm_ms2r_r3_366box_priced_rung_table.v1"
BACKFILL_SCHEMA: Final = "ddm_ms2r_r3_366box_rd1_dual_backfill.v1"
LANE_ID: Final = "ddm_ms2r_r3_366box_typed_fisher_g4_waterfill"
PAIR_COUNT: Final = 600
SCORED_PIXELS: Final = 117_964_800
ALLOWED_ERRORS: Final = 136_839
RD1_CELL_COUNT: Final = 162
R6_MAX_BYTES: Final = 154_600
FALSIFIER_MAX_BYTES: Final = 200_000
RATE_SCORE_PER_BYTE: Final = 25.0 / 37_545_489.0
DISPLAY_RATE_SCORE_PER_BYTE: Final = 6.66e-7
VERDICT: Final = (
    "BLOCKED_TYPED_HOMOTOPY_PRECONDITION; "
    "FORMULATION_FALSIFIER_NOT_REACHED"
)
VERDICT_SCOPE: Final = (
    "PRECONDITION/APPARATUS x current SHA-bound MS4D/PC1/RG3/EV2/R3 "
    "composition; not a negative on Fisher waterfilling, the representation "
    "family, the exact-solve pair, or the descent line"
)
BLOCKERS: Final = (
    "RG3_EXACT_TOP24_PAIR_BUCKET_CLOSURE_INCOMPLETE_25",
    "PC1_ACTIVE_POSE_TUBE_NOT_MEASURED_IN_SOLVE",
    "PF3_RECEIVER_OBJECT_AND_TYPED_RATE_HOME_ABSENT",
    "EV2_SAME_OBJECT_RATE_HOME_FORMULATION_MISPOSED_162_NULL",
    "TYPED_SOLVE_ALTERNATION_BLOCK_ATLAS_AND_QUANTA_INACTIVE",
)
REQUIRED_INPUTS: Final = frozenset(
    {
        "bundle",
        "ms4d_waterfill",
        "pose_metric",
        "pc1",
        "rg3",
        "ev2",
        "ev2_backfill",
        "prior_r3_admission",
        "r3",
        "r3_backfill",
        "r3_config",
        "ms2",
        "dm1",
        "dm2",
        "e4",
    }
)


class DDM366BoxAdmissionError(ValueError):
    """A sealed input drifted or an inadmissible measured claim was requested."""


@dataclass(frozen=True, slots=True)
class DDM366BoxArtifacts:
    """Pure artifacts emitted after all cross-chain invariants pass."""

    preflight: dict[str, Any]
    priced_rung_table: dict[str, Any]
    rd1_backfill: dict[str, Any]
    receipt: dict[str, Any]


def canonical_bytes(value: Any) -> bytes:
    """Return deterministic newline-terminated JSON."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DDM366BoxAdmissionError(f"{label} must be an object")
    return value


def _rows(value: Any, label: str, expected: int) -> Sequence[Mapping[str, Any]]:
    if (
        not isinstance(value, list)
        or len(value) != expected
        or not all(isinstance(row, Mapping) for row in value)
    ):
        raise DDM366BoxAdmissionError(
            f"{label} must contain exactly {expected} object rows"
        )
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise DDM366BoxAdmissionError(
            f"{label} differs: expected {expected!r}, observed {actual!r}"
        )


def _false_authority(value: Mapping[str, Any], label: str) -> None:
    for key in ("score_claim", "promotion_eligible", "pointer_moved"):
        if key in value and value.get(key) is not False:
            raise DDM366BoxAdmissionError(f"{label}.{key} must remain false")


def pose_score_derivative(d_pose: float) -> float:
    """Return d sqrt(10*d_pose) / d d_pose at a positive operating point."""

    if isinstance(d_pose, bool) or not isinstance(d_pose, (int, float)):
        raise DDM366BoxAdmissionError("d_pose must be numeric")
    result = float(d_pose)
    if not math.isfinite(result) or result <= 0.0:
        raise DDM366BoxAdmissionError("d_pose must be finite and positive")
    return 5.0 / math.sqrt(10.0 * result)


def _validate_sources(inputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(inputs) != REQUIRED_INPUTS:
        missing = sorted(REQUIRED_INPUTS - set(inputs))
        extra = sorted(set(inputs) - REQUIRED_INPUTS)
        raise DDM366BoxAdmissionError(
            f"input inventory differs; missing={missing}, extra={extra}"
        )

    bundle = _object(inputs["bundle"], "MS4D bundle")
    _require_equal(bundle.get("schema"), "ddm_metric_custody_bundle.v1", "bundle schema")
    _require_equal(bundle.get("status"), "COMPLETE", "bundle status")
    headline_admissibility = _object(
        bundle.get("headline_admissibility"),
        "bundle headline_admissibility",
    )
    for key in ("bundle_complete", "pose_tube_active", "scorer_metric_active"):
        _require_equal(headline_admissibility.get(key), True, f"bundle {key}")
    _false_authority(bundle, "bundle")

    waterfill = _object(inputs["ms4d_waterfill"], "MS4D waterfill")
    _require_equal(
        waterfill.get("schema"),
        "ddm_ms4d_waterfill_post_admission.v1",
        "MS4D waterfill schema",
    )
    _require_equal(
        waterfill.get("verdict"),
        "BLOCKED_PF3_RECEIVER_OBJECT_AND_TYPED_RATE_HOME_ABSENT",
        "MS4D waterfill verdict",
    )
    gate = _object(
        waterfill.get("candidate_materialization_gate"),
        "MS4D candidate materialization gate",
    )
    _require_equal(
        gate.get("fully_materialized_occupied_bucket_count"),
        0,
        "fully materialized MS4D bucket count",
    )
    materialization_counts = _object(
        gate.get("materialization_field_counts"),
        "MS4D materialization field counts",
    )
    if set(materialization_counts.values()) != {0}:
        raise DDM366BoxAdmissionError("MS4D materialization fields must all be zero")
    homotopy = _object(waterfill.get("homotopy"), "MS4D homotopy")
    _require_equal(homotopy.get("launched"), False, "MS4D homotopy launched")
    _require_equal(
        homotopy.get("waterfilled_rung_count"),
        0,
        "MS4D waterfilled rung count",
    )

    pose_metric = _object(inputs["pose_metric"], "Pose metric")
    _require_equal(
        pose_metric.get("schema"),
        "ddm_pose_metric_custody.v1",
        "Pose metric schema",
    )
    _require_equal(pose_metric.get("pair_count"), PAIR_COUNT, "Pose pair count")
    _require_equal(
        pose_metric.get("scorer_batch_size"),
        32,
        "Pose scorer batch size",
    )
    _require_equal(pose_metric.get("output_dimension"), 6, "Pose output dimension")
    _require_equal(
        pose_metric.get("metric_surface"),
        "EXACT_POSENET_OUTPUT_MSE_QUADRATIC",
        "Pose metric surface",
    )
    _rows(pose_metric.get("rows"), "Pose metric rows", PAIR_COUNT)
    _false_authority(pose_metric, "Pose metric")

    pc1 = _object(inputs["pc1"], "PC1")
    _require_equal(pc1.get("schema"), "ddm_pc1_pose_stream_admission.v1", "PC1 schema")
    pc1_admission = _object(pc1.get("admission"), "PC1 admission")
    _require_equal(pc1_admission.get("n600_batch32_measured"), True, "PC1 n600")
    _require_equal(pc1_admission.get("descent_was_run"), False, "PC1 descent")
    _require_equal(pc1_admission.get("tube_claim"), False, "PC1 tube claim")
    _false_authority(pc1, "PC1")

    rg3 = _object(inputs["rg3"], "RG3")
    _require_equal(
        rg3.get("schema"),
        "ddm_ms6_receiver_support_resume_summary.v1",
        "RG3 summary schema",
    )
    coverage = _object(rg3.get("g3_top24_coverage"), "RG3 coverage")
    _require_equal(coverage.get("coverage_proven"), False, "RG3 coverage")
    _require_equal(coverage.get("missing_block_count"), 25, "RG3 missing blocks")
    _rows(coverage.get("missing_blocks"), "RG3 missing block rows", 25)
    _require_equal(rg3.get("producer_rerun_eligible"), False, "RG3 producer eligibility")
    _false_authority(rg3, "RG3")

    ev2 = _object(inputs["ev2"], "EV2")
    _require_equal(
        ev2.get("schema"),
        "ddm_ev2_per_pair_allocation_receipt.v1",
        "EV2 schema",
    )
    mass = _object(ev2.get("mass_conservation"), "EV2 mass conservation")
    _require_equal(mass.get("conserved"), True, "EV2 mass conservation")
    _require_equal(mass.get("assigned_pair_cell_bytes"), 0, "EV2 assigned bytes")
    _require_equal(mass.get("unallocated_fraction"), 1.0, "EV2 unallocated fraction")
    falsifier = _object(ev2.get("falsifier"), "EV2 falsifier")
    _require_equal(falsifier.get("fired"), True, "EV2 falsifier")
    _require_equal(
        falsifier.get("verdict"),
        "FORMULATION_MISPOSED_FOR_CURRENT_C1_COMPOSITION",
        "EV2 verdict",
    )
    ev2_waterfill = _object(ev2.get("waterfill"), "EV2 waterfill")
    _require_equal(ev2_waterfill.get("full_solve_allowed"), False, "EV2 full solve")
    _false_authority(ev2, "EV2")

    ev2_backfill = _object(inputs["ev2_backfill"], "EV2 RD1 backfill")
    _require_equal(
        ev2_backfill.get("schema"),
        "ddm_ms4d_rd1_dual_backfill.v1",
        "EV2 RD1 schema",
    )
    _require_equal(
        ev2_backfill.get("source_cell_count"),
        RD1_CELL_COUNT,
        "EV2 RD1 source count",
    )
    _require_equal(
        ev2_backfill.get("rung_measured_cell_count"),
        0,
        "EV2 RD1 rung count",
    )
    _require_equal(
        ev2_backfill.get("still_null_lambda_cell_count"),
        RD1_CELL_COUNT,
        "EV2 RD1 null count",
    )
    ev2_cells = _rows(
        ev2_backfill.get("cells"),
        "EV2 RD1 cells",
        RD1_CELL_COUNT,
    )
    if any(
        row.get("lambda_bytes_per_D_dimension") is not None
        or row.get("actionable_for_train_decision") is not False
        for row in ev2_cells
    ):
        raise DDM366BoxAdmissionError("EV2 RD1 cells must remain null/non-actionable")

    prior_r3 = _object(inputs["prior_r3_admission"], "prior R3 admission")
    _require_equal(
        prior_r3.get("schema"),
        "ddm_ms2r_r3_typed_fisher_g4_waterfill_receipt.v1",
        "prior R3 admission schema",
    )
    _require_equal(
        prior_r3.get("verdict"),
        (
            "BLOCKED_NO_COMPOSABLE_TYPED_ACTUATOR_STREAM; "
            "R2_CONTROL_REMAINS_CHEAPEST_RECEIVER_CLOSED_BOX_MEMBER"
        ),
        "prior R3 admission verdict",
    )
    prior_pricing = _object(prior_r3.get("rd1_pricing"), "prior R3 pricing")
    _require_equal(prior_pricing.get("cell_count"), RD1_CELL_COUNT, "prior R3 cells")
    _require_equal(
        prior_pricing.get("finite_dimension_dual_count"),
        0,
        "prior R3 finite duals",
    )
    _require_equal(
        prior_pricing.get("actionable_for_train_decision_count"),
        0,
        "prior R3 actionable prices",
    )
    prior_headline = _object(prior_r3.get("headline"), "prior R3 headline")
    _require_equal(prior_headline.get("status"), "HEADLINE_BLOCKED", "prior headline")
    _require_equal(
        prior_headline.get("blockers"),
        [
            "TYPED_SUBPROBLEM_ALTERNATION_NOT_ACTIVE",
            "TYPED_BLOCK_ATLAS_NOT_ACTIVE",
            "PER_DIMENSION_EFFECTIVE_QUANTA_NOT_ACTIVE",
        ],
        "prior R3 headline blockers",
    )
    _false_authority(
        _object(prior_r3.get("authority"), "prior R3 authority"),
        "prior R3",
    )

    r3 = _object(inputs["r3"], "R3 box control")
    _require_equal(
        r3.get("schema"),
        "ddm_ms2r_r3_box_tolerance_solve_receipt.v1",
        "R3 schema",
    )
    _false_authority(_object(r3.get("authority"), "R3 authority"), "R3")
    solve = _object(r3.get("solve"), "R3 solve")
    _require_equal(solve.get("allowed_errors"), ALLOWED_ERRORS, "R3 allowed errors")
    _require_equal(solve.get("realized_errors"), ALLOWED_ERRORS, "R3 realized errors")
    candidate = _object(r3.get("candidate"), "R3 candidate")
    _require_equal(
        candidate.get("strict_production_parseback_exact"),
        True,
        "R3 parse-back",
    )
    _require_equal(
        candidate.get("canonical_archive_determinism_x2"),
        True,
        "R3 deterministic archive",
    )
    archive = _object(candidate.get("archive"), "R3 archive")
    _require_equal(archive.get("bytes"), 291_205_400, "R3 archive bytes")
    objective = _object(r3.get("objective_terms"), "R3 objective")
    _require_equal(objective.get("archive_bytes"), archive.get("bytes"), "R3 objective bytes")
    declarations = _object(
        _object(
            _object(
                r3.get("minimum_description_headline"),
                "R3 minimum-description headline",
            ).get("recursive_solve_typing"),
            "R3 recursive solve typing",
        ).get("declarations"),
        "R3 declarations",
    )
    expected_declarations = {
        "alternating_typed_subproblems": False,
        "per_dimension_quanta_active": False,
        "quotient_coordinates_only": True,
        "scorer_metric_active": True,
        "typed_blocks_active": False,
    }
    _require_equal(dict(declarations), expected_declarations, "R3 solve declarations")

    r3_backfill = _object(inputs["r3_backfill"], "R3 RD1 backfill")
    _require_equal(
        r3_backfill.get("schema"),
        "ddm_ms2r_r3_rd1_162_dual_backfill.v1",
        "R3 RD1 schema",
    )
    _require_equal(
        r3_backfill.get("finite_per_dimension_dual_count"),
        0,
        "R3 finite dual count",
    )
    _require_equal(
        r3_backfill.get("still_null_cell_count"),
        RD1_CELL_COUNT,
        "R3 null dual count",
    )

    r3_config = _object(inputs["r3_config"], "R3 config")
    _require_equal(
        r3_config.get("schema"),
        "DDMMS2RR3BoxToleranceSolveConfigV1",
        "R3 config schema",
    )
    _require_equal(r3_config.get("allowed_errors"), ALLOWED_ERRORS, "R3 config errors")
    _require_equal(r3_config.get("scorer_batch_size"), 32, "R3 config batch")

    ms2 = _object(inputs["ms2"], "MS2")
    _require_equal(
        ms2.get("schema"),
        "ddm_ms2_typed_quotient_solve_repo_receipt.v1",
        "MS2 schema",
    )
    _require_equal(
        ms2.get("verdict"),
        "BLOCKED_NO_ADMISSIBLE_METRIC_ACTIVE_N600_CANDIDATE",
        "MS2 verdict",
    )
    _false_authority(_object(ms2.get("authority"), "MS2 authority"), "MS2")

    dm1 = _object(inputs["dm1"], "DM1")
    _require_equal(dm1.get("schema"), "ddm_dm1_solved_value_pricing.v1", "DM1 schema")
    _require_equal(dm1.get("row_count"), 25, "DM1 row count")
    _require_equal(
        _object(dm1.get("joint_shared_context"), "DM1 context").get(
            "all_25_rows_parseback_exact"
        ),
        True,
        "DM1 semantic parse-back",
    )
    _false_authority(dm1, "DM1")

    dm2 = _object(inputs["dm2"], "DM2")
    _require_equal(dm2.get("schema"), "ddm_dm2_l3_realization_race.v1", "DM2 schema")
    _require_equal(dm2.get("row_count"), 25, "DM2 row count")
    realized_rgb = _object(
        _object(dm2.get("aggregate"), "DM2 aggregate").get("realized_rgb_joint"),
        "DM2 realized RGB",
    )
    _require_equal(realized_rgb.get("parseback_exact"), True, "DM2 parse-back")
    _false_authority(dm2, "DM2")

    e4 = _object(inputs["e4"], "E4")
    _require_equal(
        e4.get("schema"),
        "ddm_e4_brotli_rate_recovery_receipt.v1",
        "E4 schema",
    )
    _require_equal(
        e4.get("verdict"),
        "PASS_E4_BROTLI_RATE_RECOVERY_ADVISORY_ONLY",
        "E4 verdict",
    )
    _false_authority(e4, "E4")

    exact_controls = _object(
        waterfill.get("registered_callable_controls"),
        "MS4D registered controls",
    )
    control_rows = _rows(exact_controls.get("rows"), "MS4D control rows", 2)
    exact_control = next(
        (row for row in control_rows if row.get("candidate_id") == "c1_exact_solved_n600"),
        None,
    )
    if exact_control is None:
        raise DDM366BoxAdmissionError("exact C1 settled control is absent")

    return {
        "bundle": bundle,
        "waterfill": waterfill,
        "pc1": pc1,
        "rg3": rg3,
        "ev2": ev2,
        "ev2_backfill": ev2_backfill,
        "r3": r3,
        "r3_config": r3_config,
        "exact_control": exact_control,
    }


def _settled_control_rows(validated: Mapping[str, Any]) -> list[dict[str, Any]]:
    exact = validated["exact_control"]
    r3 = validated["r3"]
    r3_config = validated["r3_config"]
    solve = r3["solve"]
    candidate = r3["candidate"]
    archive = candidate["archive"]
    objective = r3["objective_terms"]
    predictor_bytes = int(candidate["predictor_payload_bytes"])
    archive_bytes = int(archive["bytes"])
    overhead = archive_bytes - predictor_bytes
    if overhead < 0:
        raise DDM366BoxAdmissionError("R3 predictor payload exceeds archive bytes")
    return [
        {
            "profile": "MS4D_SETTLED_EXACT_C1_CONTROL",
            "operating_point": "exact-solve endpoint replay",
            "base_state_sha256": str(r3_config["c1_archive_sha256"]),
            "scope": "SETTLED_CONTROL_NOT_A_TASK_701_HOMOTOPY_RUNG",
            "error_budget_used": int(exact["seg_errors"]),
            "d_seg": float(exact["d_seg"]),
            "d_pose": float(exact["d_pose"]),
            "dS_dd_pose": pose_score_derivative(float(exact["d_pose"])),
            "description_bytes_by_stream": {
                "complete_receiver_archive": int(exact["best_coded_bytes"])
            },
            "description_bytes_total": int(exact["best_coded_bytes"]),
            "coder": "SOURCE_ARCHIVE_CONTROL_NO_NEW_RACE",
            "receiver_parseback_byte_identity": True,
            "projected_S": float(exact["joint_S"]),
            "active_pose_tube_check": "NOT_ACTIVE_IN_TASK_701_SOLVE",
            "pose_stream_present": False,
            "epistemic_status": str(exact["epistemic_status"]),
            "eligible_as_task_rung": False,
            "score_claim": False,
        },
        {
            "profile": "R3_FINITE_Q4_Q8_BATCH32_CONTROL",
            "operating_point": "136839-error box boundary",
            "base_state_sha256": str(r3_config["c1_archive_sha256"]),
            "scope": "FINITE_Q4_Q8_CONTROL_NOT_TYPED_FISHER_G4_HOMOTOPY",
            "error_budget_used": int(solve["realized_errors"]),
            "d_seg": float(objective["d_seg"]),
            "d_pose": float(objective["d_pose"]),
            "dS_dd_pose": pose_score_derivative(float(objective["d_pose"])),
            "description_bytes_by_stream": {
                "predictor_payload": predictor_bytes,
                "container_receiver_overhead": overhead,
            },
            "description_bytes_total": archive_bytes,
            "coder": str(r3["coder_race"]["admitted_winner"]),
            "receiver_parseback_byte_identity": bool(
                candidate["strict_production_parseback_exact"]
            ),
            "projected_S": float(objective["objective"]),
            "active_pose_tube_check": "NOT_ACTIVE_IN_TASK_701_SOLVE",
            "pose_stream_present": False,
            "epistemic_status": (
                "MEASURED_BATCH32_RECEIVER_CLOSED_SETTLED_FINITE_FAMILY_CONTROL"
            ),
            "eligible_as_task_rung": False,
            "score_claim": False,
        },
    ]


def _preregistered_rungs(validated: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the full deterministic integer-lattice ladder, still unmeasured."""

    exact_errors = int(validated["exact_control"]["seg_errors"])
    if not 0 <= exact_errors < ALLOWED_ERRORS:
        raise DDM366BoxAdmissionError("exact control is outside the box error interval")
    headroom = ALLOWED_ERRORS - exact_errors
    targets = (
        (
            "r0_exact",
            "exact endpoint",
            exact_errors,
            "sealed exact-control error count",
        ),
        (
            "r1_near_exact",
            "near-exact first lattice slack",
            exact_errors + 1,
            "exact endpoint plus one argmax-error lattice unit",
        ),
        (
            "r2_early_slack",
            "early slack",
            exact_errors + headroom // 8,
            "one-eighth dyadic anchor over sealed integer headroom",
        ),
        (
            "r3_mid_slack",
            "mid slack",
            exact_errors + headroom // 2,
            "one-half dyadic anchor over sealed integer headroom",
        ),
        (
            "r4_near_boundary",
            "near-boundary slack",
            exact_errors + 7 * headroom // 8,
            "seven-eighths dyadic anchor over sealed integer headroom",
        ),
        (
            "r5_boundary",
            "box boundary",
            ALLOWED_ERRORS,
            "sealed box error cap",
        ),
    )
    base_sha = str(validated["r3_config"]["c1_archive_sha256"])
    return [
        {
            "rung_id": rung_id,
            "profile": "TYPED_FISHER_G4_RATE_INSIDE_OBJECTIVE",
            "operating_point": label,
            "base_state_sha256": base_sha,
            "scope": "PREREGISTERED_TASK_701_RUNG_NOT_MEASURED",
            "target_error_budget": target,
            "target_d_seg_ceiling": target / SCORED_PIXELS,
            "selection_rule": selection_rule,
            "active_pose_tube_required": True,
            "real_receiver_uint8_parseback_required": True,
            "real_coder_race_required": True,
            "full_ladder_execution_required": True,
            "execution_status": "BLOCKED_PRECONDITION_NOT_RUN",
            "epistemic_status": "DERIVED_PREREGISTRATION_NOT_MEASURED",
            "candidate": None,
            "score_claim": False,
        }
        for rung_id, label, target, selection_rule in targets
    ]


def _rd1_backfill(validated: Mapping[str, Any]) -> dict[str, Any]:
    source = validated["ev2_backfill"]
    cells = []
    for row in source["cells"]:
        copied = deepcopy(dict(row))
        copied["task_701_homotopy_exchange_rates"] = []
        copied["task_701_status"] = (
            "STILL_NULL_PRE_HOMOTOPY_SAME_OBJECT_RATE_HOME_ABSENT"
        )
        copied["lambda_bytes_per_D_dimension"] = None
        copied["actionable_for_train_decision"] = False
        copied["score_claim"] = False
        cells.append(copied)
    result = {
        "schema": BACKFILL_SCHEMA,
        "source_schema": source["schema"],
        "source_cell_count": RD1_CELL_COUNT,
        "metric_context_cell_count": RD1_CELL_COUNT,
        "rung_measured_cell_count": 0,
        "lambda_measured_cell_count": 0,
        "still_null_lambda_cell_count": RD1_CELL_COUNT,
        "actionable_cell_count": 0,
        "cells": cells,
        "blockers": list(BLOCKERS),
        "verdict": "NO_TASK_701_BACKFILL; 162_OF_162_PRICES_STILL_NULL",
        "verdict_scope": VERDICT_SCOPE,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "main_landing_review_required": True,
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def build_artifacts(
    inputs: Mapping[str, Mapping[str, Any]],
    *,
    input_custody: Mapping[str, Mapping[str, Any]],
    config_custody: Mapping[str, Any],
    available_memory_bytes: int,
) -> DDM366BoxArtifacts:
    """Validate sealed custody and emit a deterministic non-execution receipt."""

    if isinstance(available_memory_bytes, bool) or available_memory_bytes < 0:
        raise DDM366BoxAdmissionError("available memory must be nonnegative bytes")
    validated = _validate_sources(inputs)
    rg3_coverage = validated["rg3"]["g3_top24_coverage"]
    materialization = validated["waterfill"]["candidate_materialization_gate"]
    ev2 = validated["ev2"]
    r3 = validated["r3"]
    declarations = r3["minimum_description_headline"]["recursive_solve_typing"][
        "declarations"
    ]
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "REFUSED_BEFORE_HOMOTOPY",
        "verdict": VERDICT,
        "verdict_scope": VERDICT_SCOPE,
        "blockers": list(BLOCKERS),
        "custody_facts": {
            "metric_bundle_complete": True,
            "metric_bundle_pose_tube_capability_present": True,
            "prior_r3_admission_recalled": True,
            "prior_r3_finite_dimension_duals": 0,
            "solve_local_pose_tube_active": False,
            "pc1_descent_was_run": False,
            "pc1_tube_claim": False,
            "rg3_coverage_proven": False,
            "rg3_missing_exact_blocks": int(rg3_coverage["missing_block_count"]),
            "fully_materialized_occupied_metric_buckets": int(
                materialization["fully_materialized_occupied_bucket_count"]
            ),
            "ev2_assigned_pair_cell_bytes": int(
                ev2["mass_conservation"]["assigned_pair_cell_bytes"]
            ),
            "ev2_unallocated_fraction": float(
                ev2["mass_conservation"]["unallocated_fraction"]
            ),
            "rd1_finite_duals": 0,
            "active_declarations": dict(declarations),
        },
        "distinction": (
            "MS4D COMPLETE authenticates scorer-intrinsic metric context and a "
            "Pose quadratic validity tube. It does not activate that tube in a "
            "candidate solve or create receiver objects, same-object rate homes, "
            "candidate deltas, coder owners, or finite RD1 costates."
        ),
        "execution": {
            "homotopy_launched": False,
            "rung_count": 0,
            "receiver_invoked": False,
            "r_operator_invoked": False,
            "frozen_scorer_invoked": False,
            "coder_invoked": False,
            "training": False,
            "paid_dispatch": False,
            "frontier_mutation": False,
        },
        "memory_preflight": {
            "required_before_n600_scorer_pass_bytes": 20 * (1 << 30),
            "passes_threshold": available_memory_bytes >= 20 * (1 << 30),
            "observed_relation": (
                "AT_LEAST_REQUIRED"
                if available_memory_bytes >= 20 * (1 << 30)
                else "BELOW_REQUIRED"
            ),
            "exact_available_bytes_persisted": False,
            "determinism_reason": (
                "Exact free RAM is volatile; only the fail-closed threshold "
                "relation is part of the immutable resume checkpoint."
            ),
            "n600_scorer_pass_requested": False,
        },
        "input_custody": dict(input_custody),
        "config_custody": dict(config_custody),
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "main_landing_review_required": True,
    }
    preflight["content_sha256"] = canonical_sha256(preflight)

    controls = _settled_control_rows(validated)
    preregistered_rungs = _preregistered_rungs(validated)
    priced_table = {
        "schema": TABLE_SCHEMA,
        "status": "BLOCKED_PRE_HOMOTOPY_NO_MEASURED_TASK_701_RUNGS",
        "rate_score_per_byte_exact": RATE_SCORE_PER_BYTE,
        "rate_score_per_byte_card_display": DISPLAY_RATE_SCORE_PER_BYTE,
        "pose_exchange_law": "dS/dd_pose = 5/sqrt(10*d_pose)",
        "error_cap": ALLOWED_ERRORS,
        "d_seg_cap": ALLOWED_ERRORS / SCORED_PIXELS,
        "preregistered_rungs": preregistered_rungs,
        "preregistered_rung_status": (
            "NOT_EXECUTABLE_UNTIL_ALL_TYPED_PRECONDITIONS_CLOSE"
        ),
        "adaptive_refinement_rule": {
            "status": "PREREGISTERED_NOT_RUN",
            "eligible_after_base_ladder_complete": True,
            "selection": (
                "Bisect the integer-error interval with the largest absolute "
                "change in adjacent measured marginal score-per-error slopes."
            ),
            "tie_break": (
                "Lower-error interval first, then lexicographic endpoint rung ids."
            ),
            "invariants": [
                "Never skip a base rung because an earlier rung disappoints.",
                "Every refinement must remain an integer error budget.",
                "Every refinement requires the same active Pose tube, receiver, "
                "uint8 parse-back, frozen scorers, and real coder race.",
            ],
        },
        "measured_task_rungs": [],
        "settled_non_rung_controls": controls,
        "knee": None,
        "knee_status": "NULL_NO_TYPED_HOMOTOPY_CURVE",
        "r6_candidate_ready": False,
        "r6_gate": {
            "maximum_bytes": R6_MAX_BYTES,
            "maximum_d_seg": 0.00116,
            "pose_stream_required": True,
            "status": "NOT_EVALUATED_NO_TASK_RUNG",
        },
        "formulation_falsifier": {
            "approximately_maximum_bytes": FALSIFIER_MAX_BYTES,
            "evaluated": False,
            "status": "NOT_REACHED_PRECONDITION_BLOCKED",
            "finite_q4_q8_control_under_threshold": False,
            "reason": (
                "The 291205400-byte finite q4/q8 box control is measured, but "
                "cannot stand in for the unexecuted optimal typed formulation."
            ),
        },
        "blockers": list(BLOCKERS),
        "verdict_scope": VERDICT_SCOPE,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "main_landing_review_required": True,
    }
    priced_table["content_sha256"] = canonical_sha256(priced_table)

    backfill = _rd1_backfill(validated)
    receipt = {
        "schema": SCHEMA,
        "lane_id": LANE_ID,
        "verdict": VERDICT,
        "verdict_scope": VERDICT_SCOPE,
        "authority": {
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "research_only": True,
            "execution_allowed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": "0.1910828242 [contest-CPU]",
            "pointer_moved": False,
            "main_landing_review_required": True,
        },
        "preflight": {
            "status": preflight["status"],
            "content_sha256": preflight["content_sha256"],
            "blockers": list(BLOCKERS),
        },
        "priced_rung_table": {
            "status": priced_table["status"],
            "content_sha256": priced_table["content_sha256"],
            "measured_task_rung_count": 0,
            "settled_control_count": len(controls),
            "knee": None,
        },
        "rd1_backfill": {
            "content_sha256": backfill["content_sha256"],
            "source_cell_count": RD1_CELL_COUNT,
            "lambda_measured_cell_count": 0,
            "still_null_lambda_cell_count": RD1_CELL_COUNT,
        },
        "r6_candidate_ready": False,
        "formulation_falsifier_reached": False,
        "pointer_delta": "NONE",
        "actuation": preflight["execution"],
        "triality": {
            "dsl": config_custody.get("path"),
            "dag": (
                ".omx/research/"
                "ddm_ms2r_r3_366box_typed_fisher_g4_waterfill_"
                "20260725T162107Z/DAG_FEED.md"
            ),
            "equations": (
                ".omx/research/"
                "ddm_ms2r_r3_366box_typed_fisher_g4_waterfill_"
                "20260725T162107Z/EQUATIONS.md"
            ),
        },
        "exact_next_measurement": [
            "Close all 25 exact RG3 pair/bucket causal-support blocks.",
            "Run a candidate-local PC1/Pose6 descent and measure tube membership.",
            (
                "Construct parse-back-stable same-object typed byte homes with "
                "receiver builder, uint8 quantum, candidate delta, and coder owner."
            ),
            (
                "Populate finite RD1 cell costates, then run every preregistered "
                "homotopy rung through receiver/R/uint8/scorers and real coders."
            ),
        ],
        "stores_consulted": sorted(input_custody),
        "inbox_cursor": {
            "per_arm": "EMPTY",
            "broadcast": "2026-07-24T23:09:25Z",
        },
        "main_landing_review_required": True,
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    return DDM366BoxArtifacts(
        preflight=preflight,
        priced_rung_table=priced_table,
        rd1_backfill=backfill,
        receipt=receipt,
    )


__all__ = [
    "ALLOWED_ERRORS",
    "BACKFILL_SCHEMA",
    "BLOCKERS",
    "DISPLAY_RATE_SCORE_PER_BYTE",
    "FALSIFIER_MAX_BYTES",
    "LANE_ID",
    "PAIR_COUNT",
    "PREFLIGHT_SCHEMA",
    "R6_MAX_BYTES",
    "RATE_SCORE_PER_BYTE",
    "RD1_CELL_COUNT",
    "REQUIRED_INPUTS",
    "SCHEMA",
    "SCORED_PIXELS",
    "TABLE_SCHEMA",
    "VERDICT",
    "VERDICT_SCOPE",
    "DDM366BoxAdmissionError",
    "DDM366BoxArtifacts",
    "build_artifacts",
    "canonical_bytes",
    "canonical_sha256",
    "pose_score_derivative",
]
