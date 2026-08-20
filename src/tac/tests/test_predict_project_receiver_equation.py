# SPDX-License-Identifier: MIT
"""Focused tests for Task #597's unregistered equation surface."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from tac.canonical_equations.day_consolidation_laws_20260720 import breakeven_bytes
from tac.canonical_equations.evaluators import (
    _EVALUATORS,
    LAWREF_BUILTIN_EVALUATORS,
    get_evaluator,
    populate_lawref_evaluators,
    register_evaluator,
)
from tac.canonical_equations.partition_temporal_transport_amortization_20260715 import (
    EQUATION_ID as TEMPORAL_JITTER_EQUATION_ID,
)
from tac.canonical_equations.partition_temporal_transport_amortization_20260715 import (
    amortization_ratio,
    build_partition_temporal_transport_amortization_v1,
)
from tac.canonical_equations.predict_project_receiver_20260721 import (
    AUTO_REGISTER,
    component_rate_identity,
    global_joint_waterfill_identity,
    predict_project_constraint_set,
    registration_gate,
    violation_seed_identity,
)
from tac.canonical_equations.predict_project_receiver_20260721 import (
    GLOBAL_WATERFILL_LAMBDA_STAR as EQUATION_GLOBAL_WATERFILL_LAMBDA_STAR,
)
from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
    EQUATION_ID as SEGNET_HEAD_RANK_EQUATION_ID,
)
from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
    build_segnet_head_rank4_linear_flipdist_v1,
    head_difference_rank,
)
from tac.optimization.predict_project_receiver import (
    CANONICAL_LAW_RESOLUTION_CUSTODY,
    CANONICAL_LAW_RESOLUTION_SHA256,
    GLOBAL_WATERFILL_LAMBDA_STAR,
    REALIZATION_BREAKEVEN_EQUATION_ID,
    SEGNET_CENTERED_HEAD_RANK,
    TEMPORAL_JITTER_AMORTIZATION_RATIO,
    _register_lawref_adapter,
)
from tac.optimization.predict_project_schema import canonical_json_bytes


def _decoder_gates() -> dict:
    def digest(value: str) -> str:
        return hashlib.sha256(value.encode("ascii")).hexdigest()

    requirements = {
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
    return {
        "schema": "predict_project_decoder_registration_gates.v0",
        "gates": {
            gate_id: {
                "status": status,
                "measurement_scope": scope,
                "evidence_sha256": digest(gate_id),
            }
            for gate_id, (status, scope) in requirements.items()
        },
        "global_joint_waterfill": {
            "status": "INCONCLUSIVE_NO_MEASURED_GLOBAL_JOINT_SWEEP",
            "required_for_projection_equation": True,
            "evidence_sha256": None,
            "per_flip_sellback_status": "INCONCLUSIVE_NO_MEASURED_PER_FLIP_SELLBACK",
            "action_level_ladder_status": "INCONCLUSIVE_NO_MEASURED_ACTION_LEVEL_LADDER",
            "attribution_edit_telemetry_status": "INCONCLUSIVE_NO_MEASURED_ATTRIBUTION_EDIT_TELEMETRY",
            "learned_tail_race_status": "INCONCLUSIVE_NO_MEASURED_LEARNED_TAIL_THREE_WAY_RACE",
            "pose_tube_knee_status": "INCONCLUSIVE_NO_MEASURED_POSE_TUBE_KNEE",
        },
    }


def test_equation_is_unregistered_without_real_b2_anchor():
    assert AUTO_REGISTER is False
    result = registration_gate(None)
    assert result["registration_allowed"] is False
    assert result["registry_mutated"] is False
    assert {
        "missing_b2_anchor",
        "missing_decoder_gates_mapping",
        "missing_global_joint_waterfill_evidence",
        "global_joint_waterfill_not_measured",
        "per_flip_sellback_not_measured",
        "action_level_ladder_not_measured",
        "attribution_edit_telemetry_not_measured",
        "learned_tail_race_not_measured",
        "pose_tube_knee_not_measured",
    }.issubset(result["blockers"])
    assert result["per_flip_sellback_status"] == "INCONCLUSIVE_NO_MEASURED_PER_FLIP_SELLBACK"
    assert result["action_level_ladder_status"] == "INCONCLUSIVE_NO_MEASURED_ACTION_LEVEL_LADDER"
    assert result["attribution_edit_telemetry_status"] == "INCONCLUSIVE_NO_MEASURED_ATTRIBUTION_EDIT_TELEMETRY"
    assert result["learned_tail_race_status"] == "INCONCLUSIVE_NO_MEASURED_LEARNED_TAIL_THREE_WAY_RACE"
    assert result["pose_tube_knee_status"] == "INCONCLUSIVE_NO_MEASURED_POSE_TUBE_KNEE"


def test_equation_gate_requires_exact_n600_hard_oracle_custody():
    result = registration_gate(
        {
            "schema": "predict_project_b2_hard_oracle.v0",
            "measurement_status": "MEASURED",
            "scope": "prefix_or_slice_n2",
            "pair_count": 2,
            "custody": None,
            "custody_sha256": None,
            "measurement_axis": None,
            "custody_byte_identical_across_rows": False,
            "n600_claim": False,
            "double_decode_equal": True,
            "cell_exact": True,
            "pose_within_tube": True,
            "uint8_factor2_exact": True,
            "d_seg": 0.0,
            "d_pose": 0.0,
            "score_claim": False,
            "promotion_eligible": False,
            "contest_authority": False,
        }
    )
    assert result["registration_allowed"] is False
    assert "b2_not_real_n600" in result["blockers"]
    assert "invalid_aggregate_hard_oracle_custody" in result["blockers"]


def test_equation_gate_verifies_exact_structured_custody_hash():
    def digest(value: str) -> str:
        return hashlib.sha256(value.encode("ascii")).hexdigest()

    custody = {
        "schema": "predict_project_hard_oracle_custody.v0",
        "seed": 1234,
        "batch_size": 16,
        "measurement_axis": "[macOS-CPU advisory]",
        "scorer": {
            "implementation_id": "test.scorer",
            "version": "v1",
            "source_sha256": digest("source"),
            "segnet_weights_sha256": digest("seg"),
            "posenet_weights_sha256": digest("pose"),
        },
        "inputs": {
            "source_sha256": digest("input-source"),
            "cache_sha256": digest("cache"),
            "evaluated_input_sha256": digest("evaluated"),
        },
        "adapter": {"identity": "test:adapter", "source_sha256": digest("adapter")},
    }
    from tac.optimization.predict_project_receiver import hard_oracle_custody_sha256

    anchor = {
        "schema": "predict_project_b2_hard_oracle.v0",
        "measurement_status": "MEASURED",
        "scope": "n600",
        "pair_count": 600,
        "custody": custody,
        "custody_sha256": hard_oracle_custody_sha256(custody),
        "measurement_axis": "[macOS-CPU advisory]",
        "custody_byte_identical_across_rows": True,
        "n600_claim": True,
        "double_decode_equal": True,
        "cell_exact": True,
        "pose_within_tube": True,
        "uint8_factor2_exact": True,
        "d_seg": 0.0,
        "d_pose": 0.0,
        "score_claim": False,
        "promotion_eligible": False,
        "contest_authority": False,
    }
    b2_only = registration_gate(anchor)
    assert b2_only["registration_allowed"] is False
    assert "missing_decoder_gates_mapping" in b2_only["blockers"]
    allocation_absent = registration_gate(anchor, _decoder_gates())
    assert allocation_absent["registration_allowed"] is False
    assert "missing_global_joint_waterfill_evidence" in allocation_absent["blockers"]
    assert "per_flip_sellback_not_measured" in allocation_absent["blockers"]
    assert "action_level_ladder_not_measured" in allocation_absent["blockers"]
    assert "attribution_edit_telemetry_not_measured" in allocation_absent["blockers"]
    assert "learned_tail_race_not_measured" in allocation_absent["blockers"]
    assert "pose_tube_knee_not_measured" in allocation_absent["blockers"]
    anchor["custody_sha256"] = digest("forged")
    rejected = registration_gate(anchor, _decoder_gates())
    assert rejected["registration_allowed"] is False
    assert "aggregate_custody_hash_mismatch" in rejected["blockers"]


def test_equation_identities_are_exact_and_non_archive_authority():
    constraint_set = predict_project_constraint_set(
        cell_constraints=1,
        pose_tube_constraints=1,
        resize_constraints=1,
        uint8_coordinates=1,
    )
    assert constraint_set["native_rasterizer_status"] == "MS_SCHEMA_BUILT_NATIVE_RASTERIZER_BLOCKED"
    assert constraint_set["native_rasterizer_blocker"] == "MS_ARC_TO_CELL_RASTERIZATION_SEMANTICS_UNMEASURED"
    assert constraint_set["separatrix_arc_decoder_use"] == "causal_boundary_offsets_only"
    assert violation_seed_identity(total_constraints=10, already_satisfied=7)["emitted_violations"] == 3
    rate = component_rate_identity(
        {"chart": 1, "trajectory": 2, "bulk": 3, "jitter": 4, "tracks": 5, "events": 6, "container_header": 7}
    )
    assert rate["raw_bytes"] == 28
    assert rate["archive_bytes_claim"] is False
    joint = global_joint_waterfill_identity(delta_score=-0.001, delta_bytes=100, interaction_delta=0.0002)
    assert joint["lambda_star"] == GLOBAL_WATERFILL_LAMBDA_STAR
    assert joint["lambda_law_resolution_sha256"] == CANONICAL_LAW_RESOLUTION_SHA256
    assert joint["independent_curve_authority"] is False
    assert joint["authority"] == "BUILT_INTERFACE_MEASUREMENT_BLOCKED"


def test_numeric_laws_are_canonical_lawref_resolutions_without_timestamp_or_fallback():
    custody = CANONICAL_LAW_RESOLUTION_CUSTODY
    numeric = custody["numeric_laws"]

    lambda_custody = numeric["global_waterfill_lambda"]
    assert lambda_custody["equation_id"] == REALIZATION_BREAKEVEN_EQUATION_ID
    assert 1.0 / breakeven_bytes(1.0) == GLOBAL_WATERFILL_LAMBDA_STAR
    assert lambda_custody["resolved_value"] == GLOBAL_WATERFILL_LAMBDA_STAR

    temporal_equation = build_partition_temporal_transport_amortization_v1()
    temporal_anchor = temporal_equation.empirical_anchors[0]
    temporal_rate = temporal_anchor.empirical_output["rate_zlib9_proxy"]
    expected_temporal_ratio = amortization_ratio(
        temporal_rate["naive_bytes_total_600_frames"],
        temporal_rate["trajectory_bytes_total"]["screw"],
    )
    assert temporal_equation.equation_id == TEMPORAL_JITTER_EQUATION_ID
    assert numeric["temporal_jitter_amortization_ratio"]["equation_id"] == TEMPORAL_JITTER_EQUATION_ID
    assert expected_temporal_ratio == TEMPORAL_JITTER_AMORTIZATION_RATIO

    rank_equation = build_segnet_head_rank4_linear_flipdist_v1()
    rank_anchor = rank_equation.empirical_anchors[0]
    expected_rank = head_difference_rank(rank_anchor.empirical_output["singvals"])
    assert rank_equation.equation_id == SEGNET_HEAD_RANK_EQUATION_ID
    assert numeric["segnet_centered_head_rank"]["equation_id"] == SEGNET_HEAD_RANK_EQUATION_ID
    assert expected_rank == SEGNET_CENTERED_HEAD_RANK

    assert all(not law["fallback_used"] for law in numeric.values())
    assert custody["contains_timestamp"] is False
    assert custody["persistent_registry_mutated"] is False
    serialized = json.dumps(custody, sort_keys=True)
    assert "resolved_at" not in serialized
    assert hashlib.sha256(canonical_json_bytes(custody)).hexdigest() == CANONICAL_LAW_RESOLUTION_SHA256
    assert EQUATION_GLOBAL_WATERFILL_LAMBDA_STAR == GLOBAL_WATERFILL_LAMBDA_STAR


def test_module_imports_when_a_local_adapter_id_graduates_into_the_builtin_registry() -> None:
    """Regression: a graduated builtin must not make this module unimportable.

    `realization_breakeven_bytes_v1` graduated into `LAWREF_BUILTIN_EVALUATORS`
    in 81337cd93c.  `_register_lawref_adapter` then saw the builtin, found it was
    not its own callable, and raised at module scope (the laws resolve during
    import), so every `import tac.optimization.predict_project_receiver` — and
    the whole predictor chain and its test modules — failed collection.
    """

    populate_lawref_evaluators()
    # Positive control: if this id ever stops being builtin-owned the deferral
    # branch below is never exercised and this test would pass vacuously.
    assert REALIZATION_BREAKEVEN_EQUATION_ID in LAWREF_BUILTIN_EVALUATORS
    builtin = LAWREF_BUILTIN_EVALUATORS[REALIZATION_BREAKEVEN_EQUATION_ID]
    assert get_evaluator(REALIZATION_BREAKEVEN_EQUATION_ID) is builtin

    # Both implementations must agree numerically, else deferring changes values.
    for recovery in (0.0, 1.0, 0.15, 1234.5):
        assert builtin({"realized_recovery_s": recovery}) == breakeven_bytes(recovery)

    # A fresh interpreter must import the module cleanly.
    proc = subprocess.run(
        [sys.executable, "-c", "import tac.optimization.predict_project_receiver"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    # A genuine conflict — two non-builtin adapters for one id — still refuses.
    def _first(_inputs: object) -> float:
        return 0.0

    def _second(_inputs: object) -> float:
        return 1.0

    unique_id = "ddm_ad1_conflict_probe_v1"
    register_evaluator(unique_id, _first)
    try:
        with pytest.raises(RuntimeError, match="conflicting in-process LawRef evaluator"):
            _register_lawref_adapter(unique_id, _second)
        # Re-registering the SAME callable stays a no-op.
        _register_lawref_adapter(unique_id, _first)
    finally:
        _EVALUATORS.pop(unique_id, None)
