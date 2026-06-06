# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.analysis.nerv_pair_local_distortion_servo import (
    PAIR_LOCAL_DISTORTION_SERVO_RECEIPT_SCHEMA,
)
from tac.analysis.nerv_witness_readiness_dag import (
    NERV_WITNESS_GATE_STATUS_SCHEMA,
    NERV_WITNESS_READINESS_DAG_SCHEMA,
    build_nerv_witness_readiness_dag,
    check_witness_gate_status,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_witness_dag_blocks_hinerv_long_run_on_spilling_hard_birth_smoke(
    tmp_path: Path,
) -> None:
    report = _write_hinerv_smoke(
        tmp_path,
        accepted=0,
        worst_reduction=0.037,
        total_reduction=0.0,
        min_ratio_increase=0.0,
        total_spill=3.8,
    )

    payload = build_nerv_witness_readiness_dag(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "dag_out",
        hinerv_smoke_report=report,
        partner_source_refs=[Path("/Users/adpena/Downloads/pact_first_principles_trace.zip")],
    )

    assert payload["schema"] == NERV_WITNESS_READINESS_DAG_SCHEMA
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["long_training_approved"] is False
    assert payload["hinerv_long_training_approved"] is False
    assert payload["objective"]["segnet_geometry"] == "last_frame_rgb_argmax_pixels"
    assert payload["objective"]["posenet_geometry"] == "two_frame_official_yuv6_pair"
    assert payload["parseback_contract_evidence"]["implemented_contract_present"] is True
    nodes = {row["node_id"]: row for row in payload["gate_nodes"]}
    assert nodes["shared.parseback_selection_contract"]["status"] == "succeeded"
    servo = nodes["shared.pair_local_distortion_servo_contract"]
    assert servo["status"] == "succeeded"
    assert servo["evidence"]["implemented_contract_present"] is True
    assert "archive_parseback_survival" in servo["evidence"]["contract"]["survival_gates"]
    assert "shared.pair_local_distortion_servo_contract" in nodes["shared.joint_seg_pose_trust_region"]["dependencies"]
    assert nodes["hinerv.short_receiver_surface_smoke"]["status"] == "succeeded"
    birth = nodes["shared.distortion_birth_before_rate_pressure"]
    assert birth["status"] == "blocked"
    assert "receiver_visible_hard_birth_update_not_accepted" in birth["blockers"]
    assert "accepted_update_receiver_uint8_movement_missing" in birth["blockers"]
    localized = nodes["hinerv.localized_target_region_projection_actuator"]
    assert localized["status"] == "blocked"
    assert "hinerv_localized_projection_trust_region_missing" in localized["blockers"]
    assert "hinerv_target_min_ratio_not_lifted" in localized["blockers"]
    assert any(
        blocker.endswith("hinerv_localized_projection_trust_region_missing")
        for blocker in payload["actionable_blockers"]
    )
    assert payload["status_map"]["hinerv__localized_target_region_projection_actuator.gate"] == "blocked"
    assert payload["dag"]["schema"] == "staircase_dag.v1"


def test_witness_dag_consumes_clean_source_boundary_audit_report(
    tmp_path: Path,
) -> None:
    source_report = _write_source_boundary_report(tmp_path, clean=True)

    payload = build_nerv_witness_readiness_dag(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "dag_out",
        source_boundary_audit_report=source_report,
    )

    nodes = {row["node_id"]: row for row in payload["gate_nodes"]}
    source = nodes["shared.source_boundary_compliance_audit"]
    assert source["status"] == "succeeded"
    assert source["evidence"]["report_loaded"] is True
    assert source["evidence"]["source_boundary_clean"] is True
    assert payload["status_map"]["shared__source_boundary_compliance_audit.gate"] == ("succeeded")
    assert not any(
        "nerv_witness_source_boundary_audit_missing" in blocker for blocker in payload["actionable_blockers"]
    )


def test_witness_dag_blocks_unclean_source_boundary_audit_report(
    tmp_path: Path,
) -> None:
    source_report = _write_source_boundary_report(
        tmp_path,
        clean=False,
        blockers=["external_artifact_reference_in_eval_source:/tmp/leaked.npy"],
    )

    status = check_witness_gate_status(
        node_id="shared.source_boundary_compliance_audit",
        source_boundary_audit_report=source_report,
        repo_root=REPO_ROOT,
    )

    assert status["schema"] == NERV_WITNESS_GATE_STATUS_SCHEMA
    assert status["satisfied"] is False
    assert "external_artifact_reference_in_eval_source:/tmp/leaked.npy" in status["blockers"]


def test_witness_dag_keeps_snerv_source_forward_gate_blocked_without_report(
    tmp_path: Path,
) -> None:
    payload = build_nerv_witness_readiness_dag(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "dag_out",
        hinerv_smoke_report=_write_hinerv_smoke(
            tmp_path,
            accepted=1,
            worst_reduction=0.02,
            total_reduction=0.01,
            min_ratio_increase=0.05,
            total_spill=0.0,
        ),
    )

    nodes = {row["node_id"]: row for row in payload["gate_nodes"]}
    assert nodes["snerv.official_mfu_hfr_tub_source_forward"]["status"] == "blocked"
    assert (
        "snerv_official_mfu_hfr_tub_authority_gate_missing"
        in nodes["snerv.official_mfu_hfr_tub_source_forward"]["blockers"]
    )
    assert payload["snerv_long_training_approved"] is False


def test_check_witness_gate_status_reports_selected_node_blockers(
    tmp_path: Path,
) -> None:
    report = _write_hinerv_smoke(
        tmp_path,
        accepted=0,
        worst_reduction=0.01,
        total_reduction=0.0,
        min_ratio_increase=0.0,
        total_spill=1.0,
    )

    status = check_witness_gate_status(
        node_id="hinerv.localized_target_region_projection_actuator",
        hinerv_smoke_report=report,
        repo_root=REPO_ROOT,
    )

    assert status["schema"] == NERV_WITNESS_GATE_STATUS_SCHEMA
    assert status["score_claim"] is False
    assert status["satisfied"] is False
    assert "hinerv_localized_projection_trust_region_missing" in status["blockers"]


def test_distortion_birth_gate_requires_receiver_uint8_movement(
    tmp_path: Path,
) -> None:
    report = _write_hinerv_smoke(
        tmp_path,
        accepted=1,
        worst_reduction=0.02,
        total_reduction=0.01,
        min_ratio_increase=0.05,
        total_spill=0.0,
        accepted_uint8_changed=0,
        accepted_uint8_delta=0,
    )

    status = check_witness_gate_status(
        node_id="shared.distortion_birth_before_rate_pressure",
        hinerv_smoke_report=report,
        repo_root=REPO_ROOT,
    )

    assert status["satisfied"] is False
    assert "accepted_update_receiver_uint8_movement_missing" in status["blockers"]
    assert "accepted_update_receiver_uint8_delta_missing" in status["blockers"]


def test_distortion_birth_gate_accepts_receiver_visible_no_spill_smoke(
    tmp_path: Path,
) -> None:
    report = _write_hinerv_smoke(
        tmp_path,
        accepted=1,
        worst_reduction=0.02,
        total_reduction=0.01,
        min_ratio_increase=0.05,
        total_spill=0.0,
        accepted_uint8_changed=9,
        accepted_uint8_delta=2,
    )

    status = check_witness_gate_status(
        node_id="shared.distortion_birth_before_rate_pressure",
        hinerv_smoke_report=report,
        repo_root=REPO_ROOT,
    )

    assert status["satisfied"] is True
    assert status["blockers"] == []
    assert status["evidence"]["rate_pressure_controls_blocked_until_satisfied"] == [
        "coder_qat",
        "section_byte_duals",
        "c1a_entropy_pressure",
        "byte_compiler_selection",
        "muon_late_polish",
    ]


def test_distortion_birth_gate_consumes_hard_birth_actuator_min_ratio_source(
    tmp_path: Path,
) -> None:
    report = _write_hinerv_smoke(
        tmp_path,
        accepted=1,
        worst_reduction=0.02,
        total_reduction=0.01,
        min_ratio_increase=0.0,
        min_ratio_increase_by_source={
            "bootstrap": 0.0,
            "hard_birth_actuator": 0.18,
        },
        target_support_by_source={
            "hard_birth_actuator": {
                "target_hard_won_count": 7.0,
                "target_hard_lost_count": 2.0,
                "net_target_support_delta": 5.0,
            }
        },
        total_spill=0.0,
        accepted_uint8_changed=9,
        accepted_uint8_delta=2,
    )

    status = check_witness_gate_status(
        node_id="shared.distortion_birth_before_rate_pressure",
        hinerv_smoke_report=report,
        repo_root=REPO_ROOT,
    )

    assert status["satisfied"] is True
    metrics = status["evidence"]["metrics"]
    assert metrics["max_candidate_segnet_min_ratio_increase"] == pytest.approx(0.0)
    assert metrics["min_ratio_increase_by_source"] == {
        "bootstrap": pytest.approx(0.0),
        "hard_birth_actuator": pytest.approx(0.18),
    }
    assert metrics["min_ratio_increase_authority_source"] == "hard_birth_actuator"
    assert metrics["max_candidate_segnet_target_min_ratio_increase_authoritative"] == pytest.approx(0.18)
    assert metrics["target_support_by_source"] == {
        "hard_birth_actuator": {
            "target_hard_won_count": pytest.approx(7.0),
            "target_hard_lost_count": pytest.approx(2.0),
            "net_target_support_delta": pytest.approx(5.0),
        }
    }
    assert metrics["hard_birth_target_hard_won_count"] == pytest.approx(7.0)
    assert metrics["hard_birth_net_target_support_delta"] == pytest.approx(5.0)

    payload = build_nerv_witness_readiness_dag(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "dag_out",
        hinerv_smoke_report=report,
    )
    nodes = {row["node_id"]: row for row in payload["gate_nodes"]}
    localized = nodes["hinerv.localized_target_region_projection_actuator"]
    assert localized["status"] == "succeeded"
    assert "hinerv_target_min_ratio_not_lifted" not in localized["blockers"]


def test_hinerv_short_receiver_surface_smoke_command_keeps_live_score_controls(
    tmp_path: Path,
) -> None:
    payload = build_nerv_witness_readiness_dag(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "dag_out",
    )

    nodes = {row["node_id"]: row for row in payload["gate_nodes"]}
    command = nodes["hinerv.short_receiver_surface_smoke"]["command"]

    assert "--segnet-direct-live-distillation-weight" in command
    seg_weight_index = command.index("--segnet-direct-live-distillation-weight") + 1
    assert float(command[seg_weight_index]) > 0.0
    assert "--segnet-direct-live-target-min-ratio-floor-weight" in command
    min_ratio_index = command.index("--segnet-direct-live-target-min-ratio-floor-weight") + 1
    assert float(command[min_ratio_index]) > 0.0
    assert "--pose-direct-live-distillation-weight" in command
    pose_weight_index = command.index("--pose-direct-live-distillation-weight") + 1
    assert float(command[pose_weight_index]) > 0.0
    assert "--pose-trust-required" in command


def test_distortion_birth_gate_rejects_bootstrap_only_min_ratio_lift(
    tmp_path: Path,
) -> None:
    report = _write_hinerv_smoke(
        tmp_path,
        accepted=1,
        worst_reduction=0.02,
        total_reduction=0.01,
        min_ratio_increase=0.18,
        min_ratio_increase_by_source={"bootstrap": 0.18},
        total_spill=0.0,
        accepted_uint8_changed=9,
        accepted_uint8_delta=2,
    )

    status = check_witness_gate_status(
        node_id="shared.distortion_birth_before_rate_pressure",
        hinerv_smoke_report=report,
        repo_root=REPO_ROOT,
    )

    assert status["satisfied"] is False
    assert "target_region_min_ratio_lift_missing" in status["blockers"]
    metrics = status["evidence"]["metrics"]
    assert metrics["min_ratio_increase_by_source"] == {"bootstrap": pytest.approx(0.18)}
    assert metrics["min_ratio_increase_authority_source"] is None


def test_joint_seg_pose_trust_region_accepts_pr95_grade_pair_servo_receipt(
    tmp_path: Path,
) -> None:
    receipt = _write_pair_local_servo_receipt(tmp_path, good=True)

    status = check_witness_gate_status(
        node_id="shared.joint_seg_pose_trust_region",
        pair_local_servo_receipt=receipt,
        repo_root=REPO_ROOT,
    )

    assert status["schema"] == NERV_WITNESS_GATE_STATUS_SCHEMA
    assert status["satisfied"] is True
    assert status["blockers"] == []
    assert status["evidence"]["long_run_admission_ready"] is True
    assert status["evidence"]["source"] == "receipt_generated_report"
    assert status["evidence"]["report_summary"]["exact_score_delta"] < 0.0
    assert "--pair-local-servo-receipt" in _joint_trust_region_command(
        tmp_path,
        receipt,
    )


def test_joint_seg_pose_trust_region_blocks_live_only_pair_servo_receipt(
    tmp_path: Path,
) -> None:
    receipt = _write_pair_local_servo_receipt(tmp_path, good=False)

    status = check_witness_gate_status(
        node_id="shared.joint_seg_pose_trust_region",
        pair_local_servo_receipt=receipt,
        repo_root=REPO_ROOT,
    )

    assert status["satisfied"] is False
    assert "pair_local_servo_archive_parseback_authority_missing" in status["blockers"]
    assert "pair_local_servo_value_per_byte_not_priced" in status["blockers"]
    assert status["evidence"]["report_summary"]["long_run_admission_ready"] is False


def test_joint_seg_pose_trust_region_command_preserves_report_path(
    tmp_path: Path,
) -> None:
    receipt = _write_pair_local_servo_receipt(tmp_path, good=True)
    payload = build_nerv_witness_readiness_dag(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "dag_out",
        pair_local_servo_receipt=receipt,
    )

    nodes = {row["node_id"]: row for row in payload["gate_nodes"]}
    joint = nodes["shared.joint_seg_pose_trust_region"]

    assert joint["status"] == "succeeded"
    assert joint["blockers"] == []
    assert "--pair-local-servo-receipt" in joint["command"]
    assert receipt.as_posix() in joint["command"]


def _write_hinerv_smoke(
    tmp_path: Path,
    *,
    accepted: int,
    worst_reduction: float,
    total_reduction: float,
    min_ratio_increase: float,
    min_ratio_increase_by_source: dict[str, float] | None = None,
    target_support_by_source: dict[str, dict[str, float]] | None = None,
    total_spill: float,
    accepted_uint8_changed: int | None = None,
    accepted_uint8_delta: int | None = None,
) -> Path:
    run_dir = tmp_path / "hinerv_smoke"
    training_dir = run_dir / "hi_nerv_mlx_training"
    training_dir.mkdir(parents=True)
    (training_dir / "nerv_crux_trace_rows.json").write_text("[]", encoding="utf-8")
    (training_dir / "hi_nerv_short_scorer_smoke_readiness.json").write_text(
        json.dumps({"schema": "hi_nerv_short_scorer_smoke_readiness.v1"}),
        encoding="utf-8",
    )
    report = {
        "schema": "compact_renderer_mlx_spine_runner_report.v1",
        "direct_smoke_rerun_argv": ["uv", "run", "python", "tools/run_compact_renderer_mlx_spine_runner.py"],
        "accepted_step_count": accepted,
        "hard_birth_argmax_progress_accepted_step_count": accepted,
        "hard_birth_argmax_progress_rejected_step_count": 2,
        "receiver_quantum_attempt_count": 4,
        "receiver_quantum_shrink_attempt_count": 1,
        "receiver_quantum_growth_attempt_count": 2,
        "max_candidate_segnet_worst_debt_reduction": worst_reduction,
        "max_candidate_segnet_total_debt_reduction": total_reduction,
        "max_candidate_segnet_min_ratio_increase": min_ratio_increase,
        "min_ratio_increase_by_source": (
            {
                str(source): float(value)
                for source, value in (
                    min_ratio_increase_by_source
                    if min_ratio_increase_by_source is not None
                    else {"hard_birth_actuator": min_ratio_increase}
                ).items()
            }
        ),
        "target_min_region_ratio_delta_by_source": (
            {
                str(source): float(value)
                for source, value in (
                    min_ratio_increase_by_source
                    if min_ratio_increase_by_source is not None
                    else {"hard_birth_actuator": min_ratio_increase}
                ).items()
            }
        ),
        "max_candidate_segnet_total_debt_spill_given_worst_improvement": total_spill,
        "max_accepted_frame1_receiver_uint8_changed_count": (
            accepted if accepted_uint8_changed is None else accepted_uint8_changed
        ),
        "max_accepted_frame1_receiver_uint8_delta_abs": (
            accepted if accepted_uint8_delta is None else accepted_uint8_delta
        ),
        "max_candidate_frame1_receiver_uint8_changed_count": (
            accepted if accepted_uint8_changed is None else accepted_uint8_changed
        ),
        "max_candidate_frame1_receiver_uint8_delta_abs": (
            accepted if accepted_uint8_delta is None else accepted_uint8_delta
        ),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }
    if target_support_by_source:
        report["target_region_birth_actuator"] = {
            "schema": "hi_nerv_target_region_birth.v1",
            "actuator_id": "fit_target_region_birth_from_segnet",
            **{
                str(key): float(value)
                for key, value in target_support_by_source.get(
                    "hard_birth_actuator",
                    {},
                ).items()
            },
        }
    path = run_dir / "compact_renderer_mlx_spine_runner_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _write_pair_local_servo_receipt(tmp_path: Path, *, good: bool) -> Path:
    receipt = {
        "schema": PAIR_LOCAL_DISTORTION_SERVO_RECEIPT_SCHEMA,
        "family": "hinerv",
        "pair_ids": [7],
        "pair_id": 7,
        "pair_index": 7,
        "stage": "round_ste_eval_surface",
        "authority": "parseback_mlx",
        "old_d_seg": 0.010,
        "new_d_seg": 0.009,
        "old_d_pose": 0.0001,
        "new_d_pose": 0.0001,
        "old_archive_bytes": 1000,
        "new_archive_bytes": 1000,
        "value_per_byte": 0.01,
        "archive_sha256": "a" * 64,
        "frame_scope": "frame1_seg_pose_joint",
        "actuator_id": "hinerv_latents_fine_target_birth",
        "trained_param_groups": ["latents_fine", "output_head"],
        "worst_scorer_debt": {
            "target_id": "pair7_class1_region3",
            "score_debt_before": 0.22,
            "score_debt_after": 0.18,
        },
        "frame_incidence": {
            "frame0_pose_only": True,
            "frame0_segnet_incidence": False,
            "frame1_segnet_incidence": True,
            "frame1_posenet_incidence": True,
            "frame0_frame1_control_split": True,
        },
        "stage_manifest": {
            "completed_stage_ids": [
                "ce_birth",
                "tau_softplus_margin",
                "smooth_disagreement",
                "round_ste_eval_surface",
                "fakequant_qat",
                "hard_pixel_c1a_entropy",
                "lambda_sigma_trust_region",
                "final_optimizer_polish",
            ],
            "stage_order_respected": True,
            "byte_pressure_after_birth": True,
            "qat_after_round_ste": True,
            "final_optimizer_after_survival": True,
        },
        "actuation": {
            "actuator_id": "hinerv_latents_fine_target_birth",
            "pair_local": True,
            "trained_param_groups": ["latents_fine", "output_head"],
            "grad_norm_by_group": {"latents_fine": 0.2, "output_head": 0.1},
            "update_norm_by_group": {"latents_fine": 0.02, "output_head": 0.01},
        },
        "grad_norm_by_group": {"latents_fine": 0.2, "output_head": 0.1},
        "update_norm_by_group": {"latents_fine": 0.02, "output_head": 0.01},
        "action_algebra_trace": {
            "selected_action_id": "target_region_birth_delta",
            "frame_scope": "frame1_seg_pose_joint",
            "effect_delta_seg": -0.001,
            "effect_delta_pose": 0.0,
            "effect_delta_bytes": 0.0,
            "runtime_delta_ms": 0.1,
            "action_payload_bits": 0.0,
            "noncommutative_interactions_checked": True,
        },
        "hardware_margin": {
            "target_authority": "parseback_mlx",
            "cpu_cuda_margin_checked": True,
            "hardware_drift_risk": "bounded",
            "segnet_margin_min": 0.1,
            "pose_margin_radius": 0.01,
        },
        "float_rgb_delta_linf": 4.0,
        "uint8_changed_pixels": 33,
        "uint8_delta_abs_max": 3.0,
        "segnet_input_delta_linf": 0.04,
        "segnet_margin_delta": 0.12,
        "segnet_argmax_flipped_pixels": 18,
        "fakequant_segnet_margin_delta": 0.08,
        "fakequant_argmax_flipped_pixels": 12,
        "parseback_segnet_margin_delta": 0.07,
        "parseback_argmax_flipped_pixels": 11,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }
    if not good:
        receipt["authority"] = "live_mlx"
        receipt["archive_sha256"] = None
        receipt["new_archive_bytes"] = 1100
        receipt["value_per_byte"] = None
    path = tmp_path / ("good_pair_servo_receipt.json" if good else "bad_pair_servo_receipt.json")
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def _joint_trust_region_command(tmp_path: Path, receipt: Path) -> list[str]:
    payload = build_nerv_witness_readiness_dag(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "dag_out",
        pair_local_servo_receipt=receipt,
    )
    nodes = {row["node_id"]: row for row in payload["gate_nodes"]}
    return list(nodes["shared.joint_seg_pose_trust_region"]["command"])


def _write_source_boundary_report(
    tmp_path: Path,
    *,
    clean: bool,
    blockers: list[str] | None = None,
) -> Path:
    report = {
        "schema": "nerv_source_boundary_audit.v1",
        "mode": "conservative",
        "source_boundary_clean": clean,
        "ready_for_witness_compile": clean,
        "long_training_gate_satisfied": clean,
        "source_reports": [{"path": "inflate.py", "issues": []}],
        "issues": [],
        "blockers": blockers or [],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }
    path = tmp_path / "source_boundary_audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path
