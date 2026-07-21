# SPDX-License-Identifier: MIT
"""Unregistered Task #597 PREDICT-to-PROJECT equation candidates.

These are callable formalization surfaces, not canonical registry entries.
Registration remains blocked until a real measured B2 n600 anchor and every
required measured decoder gate exist.
Importing this module performs no I/O and cannot mutate the equation registry.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any, Final

from tac.optimization.predict_project_receiver import (
    CANONICAL_LAW_RESOLUTION_SHA256,
    GLOBAL_WATERFILL_LAMBDA_STAR,
    LOCAL_HARD_ORACLE_AXIS,
    PredictProjectReceiverError,
    hard_oracle_custody_sha256,
    validate_global_joint_waterfill_evidence,
    validate_hard_oracle_custody,
)
from tac.optimization.predict_project_schema import canonical_json_bytes

EQUATION_ID: Final = "predict_project_cell_tube_uint8_projection_v1"
REGISTRATION_POLICY: Final = (
    "UNREGISTERED_UNTIL_B2_DECODER_GATES_GLOBAL_WATERFILL_ATTRIBUTION_EDITS_AND_LEARNED_TAIL_RACES"
)
AUTO_REGISTER: Final = False
DECODER_GATE_REQUIREMENTS: Final = {
    "MS_native_rasterizer": ("MEASURED_MS_ARC_TO_CELL_RASTERIZATION", "full_n600"),
    "G1_pose_blind_constraint_tightening": (
        "MEASURED_REAL_UNIVERSAL_POSE_TIGHTENING",
        "real_full_n600",
    ),
    "G2_camera_resolution_inverse_r": ("MEASURED_FULL_N600_EXACT_PARSEBACK", "full_n600"),
    "G3_frame_asymmetry": ("MEASURED_EXACT_FRAME_ASYMMETRY", "full_n600"),
    "G4_cross_host_byte_identity": ("MEASURED_CPU_CUDA_BYTE_IDENTITY", "cpu_cuda_full_n600"),
    "G5_named_section_container": ("MEASURED_PARSEBACK_EXACT_CONSUMPTION", "full_n600"),
}


class PredictProjectEquationError(ValueError):
    """Invalid equation input or attempted authority promotion without B2."""


def predict_project_constraint_set(
    *,
    cell_constraints: int,
    pose_tube_constraints: int,
    resize_constraints: int,
    uint8_coordinates: int,
) -> dict[str, Any]:
    """Return the declared intersection cardinalities and equation text."""

    values = (cell_constraints, pose_tube_constraints, resize_constraints, uint8_coordinates)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise PredictProjectEquationError("constraint counts must be nonnegative exact integers")
    return {
        "equation_id": EQUATION_ID,
        "equation": (
            "C = C_MS_frame1(first-max) intersect P_pose_frame0_frame1(proved pixel polytope) "
            "intersect range(R_factor2_inverse_camera_uint8) intersect Z_uint8"
        ),
        "cell_constraints": cell_constraints,
        "pose_tube_constraints": pose_tube_constraints,
        "resize_constraints": resize_constraints,
        "uint8_coordinates": uint8_coordinates,
        "chart_schema": "morse_smale_graph_vineyard.v1",
        "native_rasterizer_status": "MS_SCHEMA_BUILT_NATIVE_RASTERIZER_BLOCKED",
        "native_rasterizer_blocker": "MS_ARC_TO_CELL_RASTERIZATION_SEMANTICS_UNMEASURED",
        "separatrix_arc_decoder_use": "causal_boundary_offsets_only",
        "frame_roles": {"frame0": "pose_only", "frame1": "seg_and_pose"},
        "pose_tightening_authority": "universal_hard_oracle_proof_required",
        "full_kernel_policy": "callable_nonserialized",
        "section_container": "predict_project_named_sections.v1",
        "registration_policy": REGISTRATION_POLICY,
        "score_claim": False,
        "promotion_eligible": False,
    }


def global_joint_waterfill_identity(
    *,
    delta_score: float,
    delta_bytes: int,
    interaction_delta: float,
) -> dict[str, Any]:
    """Formalize the measured global composition Lagrangian, not a verdict."""

    if (
        isinstance(delta_score, bool)
        or not isinstance(delta_score, (int, float))
        or not math.isfinite(float(delta_score))
        or isinstance(delta_bytes, bool)
        or not isinstance(delta_bytes, int)
        or isinstance(interaction_delta, bool)
        or not isinstance(interaction_delta, (int, float))
        or not math.isfinite(float(interaction_delta))
    ):
        raise PredictProjectEquationError("global-waterfill deltas must be finite and byte-exact")
    return {
        "equation": "Delta L_joint = Delta S_joint + lambda_star*Delta B; I_ij = Delta_joint - sum(Delta_singles)",
        "lambda_star": GLOBAL_WATERFILL_LAMBDA_STAR,
        "lambda_law_resolution_sha256": CANONICAL_LAW_RESOLUTION_SHA256,
        "delta_score": float(delta_score),
        "delta_bytes": delta_bytes,
        "interaction_delta": float(interaction_delta),
        "lagrangian_delta": float(delta_score) + GLOBAL_WATERFILL_LAMBDA_STAR * delta_bytes,
        "authority": "BUILT_INTERFACE_MEASUREMENT_BLOCKED",
        "independent_curve_authority": False,
        "registration_policy": REGISTRATION_POLICY,
    }


def violation_seed_identity(*, total_constraints: int, already_satisfied: int) -> dict[str, Any]:
    """Formalize ``|seed| = |constraints| - |already satisfied|``."""

    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (total_constraints, already_satisfied)
        )
        or already_satisfied > total_constraints
    ):
        raise PredictProjectEquationError("invalid seed-identity counts")
    emitted = total_constraints - already_satisfied
    return {
        "total_constraints": total_constraints,
        "already_satisfied": already_satisfied,
        "emitted_violations": emitted,
        "already_satisfied_fraction": already_satisfied / total_constraints if total_constraints else 0.0,
        "identity_exact": True,
        "authority": "DERIVED",
    }


def component_rate_identity(component_bytes: Mapping[str, int]) -> dict[str, Any]:
    """Formalize B5's component-complete raw-byte decomposition."""

    required = {"chart", "trajectory", "bulk", "jitter", "tracks", "events", "container_header"}
    if not isinstance(component_bytes, Mapping) or not required.issubset(component_bytes):
        raise PredictProjectEquationError("component rate identity is incomplete")
    values: dict[str, int] = {}
    for key, value in component_bytes.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PredictProjectEquationError("component bytes must be nonnegative exact integers")
        values[str(key)] = value
    return {
        "components": values,
        "raw_bytes": sum(values.values()),
        "equation": "B_total = B_chart + B_trajectory + B_bulk + B_jitter + B_tracks + B_events + B_container",
        "authority": "DERIVED_FROM_CANONICAL_BYTES",
        "archive_bytes_claim": False,
    }


def _nonzero_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value)
    )


def registration_gate(
    b2_anchor: Mapping[str, Any] | None,
    decoder_gates: Mapping[str, Any] | None = None,
    global_waterfill_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless B2 and every measured decoder gate are supplied.

    This function only reports eligibility.  It deliberately does not import or
    call the canonical registry mutation API; parent review owns registration.
    """

    blockers: list[str] = []
    global_waterfill_status = "INCONCLUSIVE_NO_MEASURED_GLOBAL_JOINT_SWEEP"
    per_flip_sellback_status = "INCONCLUSIVE_NO_MEASURED_PER_FLIP_SELLBACK"
    action_level_ladder_status = "INCONCLUSIVE_NO_MEASURED_ACTION_LEVEL_LADDER"
    attribution_edit_telemetry_status = "INCONCLUSIVE_NO_MEASURED_ATTRIBUTION_EDIT_TELEMETRY"
    learned_tail_race_status = "INCONCLUSIVE_NO_MEASURED_LEARNED_TAIL_THREE_WAY_RACE"
    pose_tube_knee_status = "INCONCLUSIVE_NO_MEASURED_POSE_TUBE_KNEE"
    declared_global_status: Any = None
    declared_per_flip_status: Any = None
    declared_action_ladder_status: Any = None
    declared_attribution_status: Any = None
    declared_learned_tail_status: Any = None
    declared_pose_knee_status: Any = None
    declared_global_evidence_sha256: Any = None
    b2_custody_hash: str | None = None
    if not isinstance(b2_anchor, Mapping):
        blockers.append("missing_b2_anchor")
    else:
        if b2_anchor.get("schema") != "predict_project_b2_hard_oracle.v0":
            blockers.append("wrong_anchor_schema")
        if b2_anchor.get("measurement_status") != "MEASURED":
            blockers.append("b2_not_measured")
        if (
            b2_anchor.get("scope") != "n600"
            or b2_anchor.get("pair_count") != 600
            or b2_anchor.get("n600_claim") is not True
        ):
            blockers.append("b2_not_real_n600")
        custody = b2_anchor.get("custody")
        try:
            validated_custody = validate_hard_oracle_custody(custody)
        except PredictProjectReceiverError:
            blockers.append("invalid_aggregate_hard_oracle_custody")
        else:
            b2_custody_hash = hard_oracle_custody_sha256(validated_custody)
            if b2_anchor.get("custody_sha256") != hard_oracle_custody_sha256(validated_custody):
                blockers.append("aggregate_custody_hash_mismatch")
            if b2_anchor.get("measurement_axis") != validated_custody["measurement_axis"]:
                blockers.append("aggregate_measurement_axis_mismatch")
            if validated_custody["measurement_axis"] != LOCAL_HARD_ORACLE_AXIS:
                blockers.append("unsupported_measurement_axis")
        if b2_anchor.get("custody_byte_identical_across_rows") is not True:
            blockers.append("mixed_or_unverified_row_custody")
        if b2_anchor.get("double_decode_equal") is not True:
            blockers.append("double_decode_not_equal")
        for key in ("cell_exact", "pose_within_tube", "uint8_factor2_exact"):
            if b2_anchor.get(key) is not True:
                blockers.append(f"{key}_not_exact")
        for key in ("d_seg", "d_pose"):
            value = b2_anchor.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                blockers.append(f"invalid_{key}")
        if (
            b2_anchor.get("score_claim") is not False
            or b2_anchor.get("promotion_eligible") is not False
            or b2_anchor.get("contest_authority") is not False
        ):
            blockers.append("authority_flags_not_fail_closed")
    if not isinstance(decoder_gates, Mapping):
        blockers.append("missing_decoder_gates_mapping")
    elif set(decoder_gates) != {"schema", "gates", "global_joint_waterfill"}:
        blockers.append("decoder_gates_mapping_fields_mismatch")
    else:
        if decoder_gates.get("schema") != "predict_project_decoder_registration_gates.v0":
            blockers.append("wrong_decoder_gates_schema")
        gates = decoder_gates.get("gates")
        if not isinstance(gates, Mapping) or set(gates) != set(DECODER_GATE_REQUIREMENTS):
            blockers.append("decoder_gate_set_mismatch")
        else:
            for gate_id, (expected_status, expected_scope) in DECODER_GATE_REQUIREMENTS.items():
                row = gates[gate_id]
                if not isinstance(row, Mapping) or set(row) != {
                    "status",
                    "measurement_scope",
                    "evidence_sha256",
                }:
                    blockers.append(f"{gate_id}_proof_fields_mismatch")
                    continue
                if row.get("status") != expected_status:
                    blockers.append(f"{gate_id}_not_measured")
                if row.get("measurement_scope") != expected_scope:
                    blockers.append(f"{gate_id}_scope_mismatch")
                if not _nonzero_sha256(row.get("evidence_sha256")):
                    blockers.append(f"{gate_id}_evidence_hash_invalid")
        global_status = decoder_gates.get("global_joint_waterfill")
        if not isinstance(global_status, Mapping) or set(global_status) != {
            "status",
            "required_for_projection_equation",
            "evidence_sha256",
            "per_flip_sellback_status",
            "action_level_ladder_status",
            "attribution_edit_telemetry_status",
            "learned_tail_race_status",
            "pose_tube_knee_status",
        }:
            blockers.append("global_joint_waterfill_status_missing")
        else:
            status = global_status.get("status")
            declared_global_status = status
            declared_per_flip_status = global_status.get("per_flip_sellback_status")
            declared_action_ladder_status = global_status.get("action_level_ladder_status")
            declared_attribution_status = global_status.get("attribution_edit_telemetry_status")
            declared_learned_tail_status = global_status.get("learned_tail_race_status")
            declared_pose_knee_status = global_status.get("pose_tube_knee_status")
            if status not in {
                "INCONCLUSIVE_NO_MEASURED_GLOBAL_JOINT_SWEEP",
                "MEASURED_GLOBAL_JOINT_SWEEP",
            }:
                blockers.append("global_joint_waterfill_status_invalid")
            if global_status.get("required_for_projection_equation") is not True:
                blockers.append("global_joint_waterfill_requirement_mismatch")
            evidence_sha256 = global_status.get("evidence_sha256")
            declared_global_evidence_sha256 = evidence_sha256
            if status == "MEASURED_GLOBAL_JOINT_SWEEP":
                if not _nonzero_sha256(evidence_sha256):
                    blockers.append("global_joint_waterfill_evidence_hash_invalid")
            elif evidence_sha256 is not None:
                blockers.append("inconclusive_global_joint_waterfill_has_evidence_hash")
    if not isinstance(global_waterfill_evidence, Mapping):
        blockers.append("missing_global_joint_waterfill_evidence")
    else:
        try:
            validated_global = validate_global_joint_waterfill_evidence(dict(global_waterfill_evidence))
        except PredictProjectReceiverError:
            blockers.append("invalid_global_joint_waterfill_evidence")
        else:
            global_waterfill_status = validated_global["measurement_status"]
            per_flip_sellback_status = validated_global["per_flip_sellback"]["measurement_status"]
            action_level_ladder_status = validated_global["action_level_ladder"]["measurement_status"]
            attribution_edit_telemetry_status = validated_global["attribution_edit_telemetry"]["measurement_status"]
            learned_tail_race_status = validated_global["learned_tail_race"]["measurement_status"]
            pose_tube_knee_status = validated_global["pose_tube_knee"]["measurement_status"]
            actual_evidence_sha256 = hashlib.sha256(canonical_json_bytes(global_waterfill_evidence)).hexdigest()
            if declared_global_status != global_waterfill_status:
                blockers.append("global_joint_waterfill_declared_status_mismatch")
            if declared_per_flip_status != per_flip_sellback_status:
                blockers.append("per_flip_sellback_declared_status_mismatch")
            if declared_action_ladder_status != action_level_ladder_status:
                blockers.append("action_level_ladder_declared_status_mismatch")
            if declared_attribution_status != attribution_edit_telemetry_status:
                blockers.append("attribution_edit_telemetry_declared_status_mismatch")
            if declared_learned_tail_status != learned_tail_race_status:
                blockers.append("learned_tail_race_declared_status_mismatch")
            if declared_pose_knee_status != pose_tube_knee_status:
                blockers.append("pose_tube_knee_declared_status_mismatch")
            if declared_global_evidence_sha256 != actual_evidence_sha256:
                blockers.append("global_joint_waterfill_evidence_hash_mismatch")
            if b2_custody_hash is None or hard_oracle_custody_sha256(validated_global["custody"]) != b2_custody_hash:
                blockers.append("global_joint_waterfill_b2_custody_mismatch")
    if global_waterfill_status != "MEASURED_GLOBAL_JOINT_SWEEP":
        blockers.append("global_joint_waterfill_not_measured")
    if per_flip_sellback_status != "MEASURED_ITERATIVE_RECODE_FIXED_POINT":
        blockers.append("per_flip_sellback_not_measured")
    if action_level_ladder_status != "MEASURED_SAME_JOINT_DECODE_ACTION_LEVEL_LADDER":
        blockers.append("action_level_ladder_not_measured")
    if attribution_edit_telemetry_status != "MEASURED_EXACT_ATTRIBUTION_AND_LADDER_EDITS":
        blockers.append("attribution_edit_telemetry_not_measured")
    if learned_tail_race_status != "MEASURED_EQUAL_FIDELITY_THREE_WAY_RACE":
        blockers.append("learned_tail_race_not_measured")
    if pose_tube_knee_status != "MEASURED_SAME_JOINT_DECODE_POSE_TUBE_SWEEP":
        blockers.append("pose_tube_knee_not_measured")
    return {
        "equation_id": EQUATION_ID,
        "registration_allowed": not blockers,
        "blockers": blockers,
        "policy": REGISTRATION_POLICY,
        "global_joint_waterfill_status": global_waterfill_status,
        "per_flip_sellback_status": per_flip_sellback_status,
        "action_level_ladder_status": action_level_ladder_status,
        "attribution_edit_telemetry_status": attribution_edit_telemetry_status,
        "learned_tail_race_status": learned_tail_race_status,
        "pose_tube_knee_status": pose_tube_knee_status,
        "registry_mutated": False,
    }


__all__ = [
    "AUTO_REGISTER",
    "DECODER_GATE_REQUIREMENTS",
    "EQUATION_ID",
    "REGISTRATION_POLICY",
    "PredictProjectEquationError",
    "component_rate_identity",
    "global_joint_waterfill_identity",
    "predict_project_constraint_set",
    "registration_gate",
    "violation_seed_identity",
]
