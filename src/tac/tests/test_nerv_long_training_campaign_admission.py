# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.experiment_queue import normalize_queue_definition
from tac.analysis.nerv_long_training_campaign_admission import (
    ADMISSION_SCHEMA,
    DEFAULT_LOCAL_MLX_LONG_TRAINING_TIMEOUT_SECONDS,
    build_nerv_long_training_campaign_execution_admission,
)
from tac.analysis.nerv_long_training_campaign_plan import (
    build_nerv_long_training_campaign_plan,
)
from tac.analysis.nerv_pair_local_distortion_servo import (
    PAIR_LOCAL_DISTORTION_SERVO_RECEIPT_SCHEMA,
    PR95_SERVO_CURRICULUM_STAGES,
)
from tac.analysis.pr95_distortion_practices_guard import (
    AXIS_TRACE_CONTRACT_SCHEMA,
    POSE_MARGINAL_TELEMETRY_CONTRACT_SCHEMA,
    PRACTICE_DAG_SCHEMA,
    SCORER_ATOM_ACTUATOR_CONTRACT_SCHEMA,
    STAGE_DAG_SCHEMA,
)
from tac.cathedral_consumers.nerv_long_training_campaign_consumer import consume_candidate

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_nerv_long_training_campaign_admission_builds_storage_gated_queue(
    tmp_path: Path,
) -> None:
    verdict = _runnable_verdict(tmp_path / "ssd")
    claims = _claims_file(
        tmp_path,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
    )

    admission = build_nerv_long_training_campaign_execution_admission(
        verdict,
        repo_root=tmp_path,
        active_claims_path=claims,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
        limit=1,
        storage_expected_bytes_per_row=1024,
        storage_reserve_free_gb=0.0,
        allowed_output_roots=(tmp_path / "ssd",),
        now_utc="2026-06-02T18:40:00Z",
    )

    assert admission["schema"] == ADMISSION_SCHEMA
    assert admission["experiment_queue_ready"] is True
    assert admission["local_mlx_execution_ready"] is True
    assert admission["admitted_experiment_count"] == 1
    assert admission["score_claim"] is False
    assert admission["ready_for_exact_eval_dispatch"] is False
    assert admission["blockers"] == []
    queue = normalize_queue_definition(admission["experiment_queue"])
    assert queue["queue_id"] == "nerv_manifest_pinned_long_training_local_mlx_admission.v1"
    assert queue["experiments"][0]["id"] == "nerv_campaign_storage_preflight"
    selected = queue["experiments"][1]
    assert selected["steps"][0]["requires"] == [
        "nerv_campaign_storage_preflight.proactive_cleanup"
    ]
    assert selected["steps"][0]["resources"]["kind"] == "local_mlx"
    assert selected["steps"][0]["timeout_seconds"] == (
        DEFAULT_LOCAL_MLX_LONG_TRAINING_TIMEOUT_SECONDS
    )
    command = selected["steps"][0]["command"]
    output_dir = Path(command[command.index("--output-dir") + 1])
    artifact_paths = set(selected["steps"][0]["telemetry"]["artifact_paths"])
    assert (
        output_dir / "compact_renderer_mlx_spine_runner_report.json"
    ).as_posix() in artifact_paths
    assert (
        output_dir / "compact_renderer_mlx_spine_runner_startup.json"
    ).as_posix() in artifact_paths
    assert (
        output_dir / "hi_nerv_mlx_training" / "telemetry.jsonl"
    ).as_posix() in artifact_paths
    assert (
        output_dir / "hi_nerv_mlx_training" / "local_mlx_prefilter_progress.jsonl"
    ).as_posix() in artifact_paths
    assert (
        output_dir / "hi_nerv_mlx_training" / "nerv_crux_trace_rows.json"
    ).as_posix() in artifact_paths
    json_postcondition_paths = {
        condition["path"]
        for condition in selected["steps"][0]["postconditions"]
        if condition["type"].startswith("json_")
    }
    assert json_postcondition_paths == {
        (output_dir / "compact_renderer_mlx_spine_runner_report.json").as_posix()
    }
    assert output_dir.as_posix() not in json_postcondition_paths
    assert selected["metadata"]["human_visual_fidelity_relevance"] == (
        "irrelevant_unless_scorer_causal"
    )
    source_row = selected["metadata"]["source_selected_row"]
    axis_contract = _source_row_contract(
        source_row,
        "pr95_distortion_axis_trace_contract",
    )
    assert axis_contract["schema"] == AXIS_TRACE_CONTRACT_SCHEMA
    assert axis_contract["required_axes"] == [
        "live_forward",
        "fakequant_forward",
        "archive_parseback",
        "inflate_replay",
        "official_evaluate_py",
    ]
    pose_contract = _source_row_contract(
        source_row,
        "pr95_posenet_marginal_telemetry_contract",
    )
    assert pose_contract["schema"] == POSE_MARGINAL_TELEMETRY_CONTRACT_SCHEMA
    assert pose_contract["pose_marginal_formula"] == "5/sqrt(10*d_pose)"
    actuator_contract = _source_row_contract(
        source_row,
        "pr95_scorer_atom_actuator_contract",
    )
    assert actuator_contract["schema"] == SCORER_ATOM_ACTUATOR_CONTRACT_SCHEMA
    assert "pair_local_film_or_latent_adapter" in actuator_contract[
        "family_actuators"
    ]
    row_guard = admission["selected_rows"][0]["pr95_distortion_practices_guard"]
    assert row_guard["schema"] == "pr95_distortion_practices_guard.v1"
    assert row_guard["launch_allowed"] is True
    assert row_guard["blockers"] == []
    practice_rows = {row["practice_id"]: row for row in row_guard["practice_rows"]}
    assert practice_rows["archive_parseback_distortion_axis_trace"]["observed"] is True
    assert row_guard["practice_dag"]["schema"] == PRACTICE_DAG_SCHEMA
    assert row_guard["practice_dag"]["all_nodes_green"] is True
    assert row_guard["optimization_stage_dag"]["schema"] == STAGE_DAG_SCHEMA
    assert row_guard["optimization_stage_dag"][
        "all_required_stage_signals_observed"
    ] is True
    assert admission["pr95_distortion_source_inventory"]["source_ready"] is True


def test_nerv_long_training_campaign_admission_blocks_without_active_claim(
    tmp_path: Path,
) -> None:
    verdict = _runnable_verdict(tmp_path / "ssd")
    claims = tmp_path / "claims.md"
    claims.write_text("# empty\n", encoding="utf-8")

    admission = build_nerv_long_training_campaign_execution_admission(
        verdict,
        repo_root=tmp_path,
        active_claims_path=claims,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
        limit=1,
        storage_expected_bytes_per_row=1024,
        storage_reserve_free_gb=0.0,
        allowed_output_roots=(tmp_path / "ssd",),
        now_utc="2026-06-02T18:40:00Z",
    )

    assert admission["experiment_queue_ready"] is False
    assert admission["experiment_queue"] is None
    assert admission["admitted_experiment_count"] == 0
    assert "active_lane_claim_missing_or_terminal" in admission["blockers"]
    assert admission["score_claim"] is False


def test_nerv_long_training_campaign_admission_blocks_non_ssd_output(
    tmp_path: Path,
) -> None:
    verdict = _runnable_verdict(tmp_path / "local_disk")
    claims = _claims_file(
        tmp_path,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
    )

    admission = build_nerv_long_training_campaign_execution_admission(
        verdict,
        repo_root=tmp_path,
        active_claims_path=claims,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
        limit=1,
        storage_expected_bytes_per_row=1024,
        storage_reserve_free_gb=0.0,
        allowed_output_roots=(tmp_path / "ssd",),
        now_utc="2026-06-02T18:40:00Z",
    )

    assert admission["experiment_queue_ready"] is False
    assert "selected_row_output_dir_not_on_allowed_ssd_tier" in admission["blockers"]


def test_nerv_long_training_campaign_admission_blocks_existing_output_artifacts(
    tmp_path: Path,
) -> None:
    verdict = _runnable_verdict(tmp_path / "ssd")
    selected = verdict["selected_local_mlx_experiments"][0]
    command = selected["command"]
    out_dir = Path(command[command.index("--output-dir") + 1])
    telemetry = out_dir / "hi_nerv_mlx_training" / "telemetry.jsonl"
    telemetry.parent.mkdir(parents=True)
    telemetry.write_text('{"epoch": 1}\n', encoding="utf-8")
    claims = _claims_file(
        tmp_path,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
    )

    admission = build_nerv_long_training_campaign_execution_admission(
        verdict,
        repo_root=tmp_path,
        active_claims_path=claims,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
        limit=1,
        storage_expected_bytes_per_row=1024,
        storage_reserve_free_gb=0.0,
        allowed_output_roots=(tmp_path / "ssd",),
        now_utc="2026-06-02T18:40:00Z",
    )

    assert admission["experiment_queue_ready"] is False
    assert admission["admitted_experiment_count"] == 0
    assert "selected_row_output_dir_contains_prior_training_artifacts" in admission[
        "blockers"
    ]
    row = admission["selected_rows"][0]
    assert telemetry.as_posix() in row["existing_output_artifact_paths"]
    assert row["admitted"] is False


def test_nerv_long_training_campaign_admission_blocks_active_local_mlx_process(
    tmp_path: Path,
) -> None:
    verdict = _runnable_verdict(tmp_path / "ssd")
    claims = _claims_file(
        tmp_path,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
    )

    admission = build_nerv_long_training_campaign_execution_admission(
        verdict,
        repo_root=tmp_path,
        active_claims_path=claims,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
        limit=1,
        storage_expected_bytes_per_row=1024,
        storage_reserve_free_gb=0.0,
        allowed_output_roots=(tmp_path / "ssd",),
        active_local_mlx_processes=(
            {
                "pid": 12345,
                "ppid": 1,
                "stat": "R",
                "etime": "00:12",
                "command": (
                    "python tools/run_compact_renderer_mlx_spine_runner.py "
                    "--execute-family hi_nerv --output-dir /Volumes/VertigoDataTier/pact/live"
                ),
            },
        ),
        now_utc="2026-06-02T18:40:00Z",
    )

    assert admission["experiment_queue_ready"] is False
    assert admission["local_mlx_execution_ready"] is False
    assert admission["admitted_experiment_count"] == 0
    assert "active_local_mlx_training_process_present" in admission["blockers"]
    assert admission["active_local_mlx_process_count"] == 1
    assert admission["active_local_mlx_processes"][0]["pid"] == 12345
    assert admission["score_claim"] is False
    assert admission["ready_for_exact_eval_dispatch"] is False


def test_nerv_long_training_campaign_admission_blocks_missing_pr95_distortion_practice(
    tmp_path: Path,
) -> None:
    verdict = _runnable_verdict(tmp_path / "ssd")
    selected = verdict["selected_local_mlx_experiments"][0]
    command = selected["command"]
    command[command.index("--pose-distillation-weight") + 1] = "0"
    claims = _claims_file(
        tmp_path,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
    )

    admission = build_nerv_long_training_campaign_execution_admission(
        verdict,
        repo_root=tmp_path,
        active_claims_path=claims,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
        limit=1,
        storage_expected_bytes_per_row=1024,
        storage_reserve_free_gb=0.0,
        allowed_output_roots=(tmp_path / "ssd",),
        now_utc="2026-06-02T18:40:00Z",
    )

    assert admission["experiment_queue_ready"] is False
    assert admission["admitted_experiment_count"] == 0
    assert (
        "hi_nerv_pr95_distortion_scorer_preprocess_eval_roundtrip_yuv6_missing"
        in admission["blockers"]
    )
    row_guard = admission["selected_rows"][0]["pr95_distortion_practices_guard"]
    assert row_guard["launch_allowed"] is False
    assert "hi_nerv_pr95_distortion_dual_component_real_scorer_pressure_missing" in row_guard[
        "blockers"
    ]


def test_nerv_long_training_campaign_admission_cli_writes_artifacts(
    tmp_path: Path,
) -> None:
    verdict_path = tmp_path / "verdict.json"
    out_json = tmp_path / "admission.json"
    out_md = tmp_path / "admission.md"
    out_queue = tmp_path / "queue.json"
    verdict_path.write_text(
        json.dumps(_runnable_verdict(tmp_path / "ssd")),
        encoding="utf-8",
    )
    claims = _claims_file(
        tmp_path,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/build_nerv_long_training_campaign_execution_admission.py"),
            "--consumer-verdict",
            str(verdict_path),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-queue",
            str(out_queue),
            "--lane-id",
            "lane_nerv_local_mlx",
            "--instance-job-id",
            "job_first",
            "--active-claims-path",
            str(claims),
            "--storage-expected-bytes-per-row",
            "1024",
            "--storage-reserve-free-gb",
            "0",
            "--local-mlx-timeout-seconds",
            "777",
            "--allowed-output-root",
            str(tmp_path / "ssd"),
            "--skip-active-local-mlx-process-scan",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    assert stdout["experiment_queue_ready"] is True
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema"] == ADMISSION_SCHEMA
    assert payload["score_claim"] is False
    assert out_md.read_text(encoding="utf-8").startswith(
        "# NeRV Long-Training Campaign Execution Admission"
    )
    queue = json.loads(out_queue.read_text(encoding="utf-8"))
    assert queue["schema"] == "experiment_queue.v1"
    selected = queue["experiments"][1]
    assert selected["steps"][0]["timeout_seconds"] == 777


def _campaign_plan(output_root: Path) -> dict:
    return build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget={
            "schema": "nerv_modelsize_budget.v1",
            "selected_candidates": [
                {
                    "schema": "hinerv_modelsize_candidate.v1",
                    "family": "hi_nerv",
                    "candidate_id": "hinerv_tiny",
                    "num_pairs": 600,
                    "hard_byte_ceiling": 178_000,
                    "decoder_codec": "int4_mixed",
                    "nominal_total_payload_bytes": 120_000,
                    "nominal_under_ceiling": True,
                    "use_hierarchical_feature_grid": True,
                    "use_convnext_blocks": True,
                    "pr95_scorer_atom_actuator_execution_evidence": (
                        _hinerv_actuator_execution_evidence()
                    ),
                    "pr95_distortion_axis_trace_measurements": (
                        _pr95_axis_trace_measurements()
                    ),
                }
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        snerv_modelsize_budget={
            "schema": "snerv_modelsize_budget.v1",
            "selected_candidates": [
                {
                    "schema": "snerv_modelsize_candidate.v1",
                    "family": "snerv",
                    "candidate_id": "snerv_tiny",
                    "num_pairs": 600,
                    "hard_byte_ceiling": 178_000,
                    "decoder_payload_codec": "int4_symmetric",
                    "nominal_total_payload_bytes": 160_000,
                    "nominal_under_ceiling": True,
                    "pr95_scorer_atom_actuator_execution_evidence": (
                        _snerv_actuator_execution_evidence()
                    ),
                    "pr95_distortion_axis_trace_measurements": (
                        _pr95_axis_trace_measurements()
                    ),
                }
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        optimizer_kinds=("adamw",),
        epochs=16,
        batch_pairs=4,
        learning_rate=3.0e-4,
        output_root=output_root,
        max_candidates_per_family=1,
        hinerv_distortion_birth_evidence_sources=(
            _hinerv_distortion_birth_evidence(),
        ),
    )

def _hinerv_actuator_execution_evidence() -> dict:
    return {
        "schema": "pr95_scorer_atom_actuator_execution_evidence.v1",
        "family": "hi_nerv",
        "pair_local_smoke_schema": "hinerv_pair_local_actuator_smoke.v1",
        "pair_local_smoke_artifact_schema": (
            "hinerv_pair_local_actuator_smoke_artifact.v1"
        ),
        "pair_local_smoke_artifact_path": (
            "/Volumes/VertigoDataTier/pact/test_admission/"
            "hi_nerv_pair_local_actuator_smoke/"
            "hinerv_pair_local_actuator_smoke_pair000000_aaaaaaaaaaaa.json"
        ),
        "pair_local_smoke_artifact_sha256": "d" * 64,
        "pair_local_smoke_artifact_bytes": 2048,
        "actuator_kind": "pair_local_latent_row",
        "actuator_tensor_name": "latents_fine",
        "updated_tensor_names": ["latents_fine"],
        "state_mutation_scope": "latents_fine_row_only",
        "runtime_sidecar_bytes": 0,
        "pair_local_adapter_bytes": 128,
        "pair_local_adapter_sha256": "a" * 64,
        "pair_local_grad_norm": 0.25,
        "pair_local_grad_norm_by_group": {"latents_fine": 0.25},
        "pair_local_output_delta_l2": 0.031,
        "pair_local_output_delta_max_abs": 0.004,
        "pair_local_output_delta_max_abs_uint8": 1.02,
        "receiver_uint8_half_step_normalized": 0.5 / 255.0,
        "receiver_uint8_crossing_potential": True,
        "receiver_uint8_changed": True,
        "receiver_uint8_changed_count": 12,
        "receiver_uint8_changed_fraction": 0.001,
        "receiver_uint8_delta_abs_max": 2,
        "non_target_pair_receiver_uint8_changed_count": 0,
        "non_target_pair_receiver_uint8_delta_abs_max": 0,
        "pair_locality_verified": True,
        "non_target_pair_output_delta_l2_max": 0.0,
        "state_restored_after_smoke": True,
        "pair_local_latents_fine_original_row_sha256": "c" * 64,
        "pair_local_latents_fine_restored_row_sha256": "c" * 64,
        "section_output_delta_per_byte_rows": [
            {
                "section": "pair_local_latents_fine",
                "output_delta_l2_per_byte": 0.004,
                "value_semantics": "receiver_output_l2_per_byte_not_score_value",
                "score_value_per_byte_measured": False,
            }
        ],
        "section_value_per_byte_rows": [],
        "pair_local_distortion_servo_receipt": _pair_local_distortion_servo_receipt(
            "hi_nerv"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }


def _snerv_actuator_execution_evidence() -> dict:
    return {
        "schema": "pr95_scorer_atom_actuator_execution_evidence.v1",
        "family": "snerv",
        "state_artifact_schema": "snerv_official_source_forward_state_artifact.v1",
        "official_state_dict_value_artifact_bytes": 512,
        "official_state_dict_value_artifact_sha256": "b" * 64,
        "checkpoint_export_lineage_bound": True,
        "mfu_hfr_tub_source_forward_parity_proven": True,
        "tub_output2_source_forward_parity_proven": True,
        "source_forward_replay_proof": _snerv_source_forward_numerical_proof(),
        "pair_local_distortion_servo_receipt": _pair_local_distortion_servo_receipt(
            "snerv"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }


def _pair_local_distortion_servo_receipt(family: str) -> dict:
    if family == "snerv":
        actuation = {
            "pair_local": True,
            "trained_param_groups": ["tub.output_2", "mfu.adapter"],
            "source_forward_replay_bound": True,
            "mfu_hfr_tub_source_forward_parity_proven": True,
        }
        trained_groups = ["tub.output_2", "mfu.adapter"]
        grad = {"tub.output_2": 0.4, "mfu.adapter": 0.2}
        update = {"tub.output_2": 0.05, "mfu.adapter": 0.02}
        actuator_id = "snerv_tub_output2_pair_birth"
        actuator_kind = "pair_conditioned_mfu_hfr_tub_adapter"
    else:
        actuation = {
            "pair_local": True,
            "trained_param_groups": ["latents_fine", "output_head.rgb_1"],
        }
        trained_groups = ["latents_fine", "output_head.rgb_1"]
        grad = {"latents_fine": 0.25, "output_head.rgb_1": 0.13}
        update = {"latents_fine": 0.04, "output_head.rgb_1": 0.02}
        actuator_id = "hinerv_target_region_birth_pair17"
        actuator_kind = "pair_local_latent_row"
    return {
        "schema": PAIR_LOCAL_DISTORTION_SERVO_RECEIPT_SCHEMA,
        "family": family,
        "pair_ids": [17],
        "authority": "parseback_mlx",
        "old_d_seg": 0.020,
        "new_d_seg": 0.019,
        "old_d_pose": 0.0020,
        "new_d_pose": 0.00195,
        "old_archive_bytes": 178_000,
        "new_archive_bytes": 178_128,
        "value_per_byte": 0.0007,
        "float_rgb_delta_linf": 0.007,
        "uint8_changed_pixels": 12,
        "segnet_input_delta_linf": 0.002,
        "posenet_input_delta_linf": 0.001,
        "segnet_margin_delta": 0.32,
        "segnet_argmax_flipped_pixels": 7,
        "pose_output_delta_l2": 0.015,
        "fakequant_segnet_margin_delta": 0.24,
        "fakequant_argmax_flipped_pixels": 5,
        "parseback_segnet_margin_delta": 0.21,
        "parseback_argmax_flipped_pixels": 4,
        "frame_scope": "frame1_seg_pose_joint",
        "actuator_id": actuator_id,
        "actuator_kind": actuator_kind,
        "worst_scorer_debt": {
            "target_id": "pair17_class4_region2",
            "score_debt_before": 2.4,
            "score_debt_after": 1.1,
        },
        "frame_incidence": {
            "frame0_pose_only": True,
            "frame0_posenet_incidence": True,
            "frame0_segnet_incidence": False,
            "frame1_segnet_incidence": True,
            "frame1_posenet_incidence": True,
            "frame0_frame1_control_split": True,
            "separate_frame_heads": True,
        },
        "stage_manifest": {
            "completed_stage_ids": list(PR95_SERVO_CURRICULUM_STAGES),
            "stage_order_respected": True,
            "byte_pressure_after_birth": True,
            "qat_after_round_ste": True,
            "final_optimizer_after_survival": True,
        },
        "actuation": actuation,
        "trained_param_groups": trained_groups,
        "grad_norm_by_group": grad,
        "update_norm_by_group": update,
        "action_algebra_trace": {
            "selected_action_id": "target_region_rgb_bias_then_pair_adapter",
            "frame_scope": "frame1_seg_pose_joint",
            "effect_delta_seg": -0.001,
            "effect_delta_pose": -0.00005,
            "effect_delta_bytes": 128,
            "runtime_delta_ms": 0.4,
            "selector_bits": 12,
            "noncommutative_interactions_checked": True,
        },
        "hardware_margin_trace": {
            "target_authority": "parseback_mlx",
            "target_authority_margin_checked": True,
            "hardware_drift_risk": "bounded",
            "segnet_margin_min": 0.03,
            "pose_error_slack": 0.0002,
        },
    }


def _pr95_axis_trace_measurements() -> list[dict[str, float | int | str | bool]]:
    return [
        {
            "axis": "live_forward",
            "measured": True,
            "score_delta": -0.010,
            "d_seg": 0.020,
            "d_pose": 0.0020,
        },
        {
            "axis": "fakequant_forward",
            "measured": True,
            "score_delta": -0.009,
            "d_seg": 0.021,
            "d_pose": 0.0021,
        },
        {
            "axis": "archive_parseback",
            "measured": True,
            "score_delta": -0.008,
            "d_seg": 0.022,
            "d_pose": 0.0022,
        },
        {
            "axis": "inflate_replay",
            "measured": True,
            "score_delta": -0.007,
            "d_seg": 0.023,
            "d_pose": 0.0023,
        },
        {
            "axis": "official_evaluate_py",
            "measured": True,
            "score": 0.2,
            "archive_bytes": 178_000,
        },
    ]


def _snerv_source_forward_numerical_proof() -> dict:
    return {
        "official_torch_frame_hash": "1" * 64,
        "mlx_frame_hash": "2" * 64,
        "numpy_receiver_frame_hash": "3" * 64,
        "parseback_frame_hash": "4" * 64,
        "tub_output_2_hash": "5" * 64,
        "max_abs_frame_delta_official_mlx": 0.0,
        "max_abs_yuv6_delta_official_numpy": 0.0,
        "seg_logit_linf_official_parseback": 0.0,
        "pose_linf_official_parseback": 0.0,
        "mfu_tensor_hashes": {"mfu.upsample_mid.weight": "6" * 64},
        "hfr_tensor_hashes": {"hfr.lh.conv1.weight": "7" * 64},
    }


def _hinerv_distortion_birth_evidence(
    *,
    candidate_id: str = "hinerv_tiny",
) -> dict:
    return {
        "schema": "compact_renderer_mlx_spine_runner.v1",
        "family": "hi_nerv",
        "candidate_id": candidate_id,
        "receiver_quantum_attempt_count": 3.0,
        "hard_birth_argmax_progress_accepted_step_count": 1.0,
        "max_candidate_segnet_worst_debt_reduction": 0.125,
        "max_candidate_segnet_min_ratio_increase": 0.0625,
        "max_candidate_segnet_total_debt_spill_given_worst_improvement": 0.0,
        "max_accepted_frame1_receiver_uint8_changed_count": 512.0,
        "max_accepted_frame1_receiver_uint8_delta_abs": 2048.0,
        "max_candidate_pose_exact_delta": 0.0,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _source_row_contract(source_row: dict, key: str) -> dict:
    direct = source_row.get(key)
    if isinstance(direct, dict):
        return direct
    launch = source_row.get("launch_authority_contract")
    if isinstance(launch, dict) and isinstance(launch.get(key), dict):
        return launch[key]
    metadata = source_row.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get(key), dict):
        return metadata[key]
    raise KeyError(key)


def _runnable_verdict(output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    queue = json.loads(json.dumps(_campaign_plan(output_root)["experiment_queue"]))
    hi = next(row for row in queue["experiments"] if row["family"] == "hi_nerv")
    hi["status"] = "queued"
    hi["blocked"] = False
    contract = hi["launch_authority_contract"]
    contract["queue_status_is_local_mlx_plan"] = True
    contract["queue_status_is_runnable_plan"] = True
    contract["queue_launch_blockers"] = []
    contract["queue_status_is_receiver_proof"] = False
    contract["queue_status_is_cpu_replay_proof"] = False
    contract["queue_status_is_exact_eval_authority"] = False
    gate = hi["score_lowering_gate"]
    gate["local_mlx_executable"] = True
    gate["prelaunch_allowed"] = True
    gate["cpu_replay_ready"] = False
    gate["exact_gate_ready"] = False
    return dict(consume_candidate(queue))


def _claims_file(tmp_path: Path, *, lane_id: str, instance_job_id: str) -> Path:
    claims = tmp_path / "claims.md"
    claims.write_text(
        "\n".join(
            [
                "# Active lane dispatch claims",
                "",
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                (
                    f"| 2026-06-02T18:34:58Z | codex:gpt-5 | {lane_id} | "
                    f"local_mlx | {instance_job_id} | 2026-06-03T00:34:58Z | "
                    "active_local_mlx_queue_first_row | test claim |"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return claims
