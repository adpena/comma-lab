# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from comma_lab.scheduler.experiment_queue import load_queue_definition
from tac.analysis.nerv_long_training_campaign_plan import (
    HINERV_POSE_INSTABILITY_LOW_LR_FLOOR,
    NervLongTrainingCampaignPlanError,
    build_nerv_long_training_campaign_plan,
    render_nerv_long_training_campaign_plan_markdown,
)
from tools import build_nerv_long_training_campaign_plan as cli


def test_long_training_campaign_plan_builds_optimizer_matrix() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion", "adafactor"),
        epochs=29_650,
        batch_pairs=8,
        learning_rate=3.0e-4,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    assert report["schema"] == "nerv_long_training_campaign_plan.v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["campaign_row_count"] == 3
    assert report["experiment_queue"]["schema"] == "experiment_queue.v1"
    assert report["experiment_queue_experiment_count"] == 3
    assert report["launchable_local_row_count"] == 3
    assert report["family_counts"] == {"hi_nerv": 2, "snerv": 1}

    hi_rows = [row for row in report["campaign_rows"] if row["family"] == "hi_nerv"]
    assert {row["optimizer_kind"] for row in hi_rows} == {"lion", "adafactor"}
    assert all("--optimizer-kind" in row["command_argv"] for row in hi_rows)
    assert all("--coder-aware-qat" in row["command_argv"] for row in hi_rows)
    assert all(row["local_mlx_launch_command_ready"] is True for row in hi_rows)
    assert all(row["local_mlx_executable"] is True for row in hi_rows)
    assert all("--auto-joint-recon-pixel-weight" in row["command_argv"] for row in hi_rows)
    assert all("--recon-pixel-weight-path" not in row["command_argv"] for row in hi_rows)
    assert all(row["cpu_replay_ready"] is False for row in hi_rows)
    assert all(row["exact_gate_ready"] is False for row in hi_rows)
    assert all(
        row["score_lowering_gate"]["schema"]
        == "nerv_long_training_score_lowering_gate.v1"
        for row in hi_rows
    )
    assert all(
        {
            "archive_in_loop_byte_oracle",
            "byte_closed_archive_export",
            "receiver_proof",
            "full_video_local_prefilter",
            "local_cpu_replay_gate",
        }.issubset(
            set(row["score_lowering_gate"]["post_run_missing_requirement_ids"])
        )
        for row in hi_rows
    )
    assert all("hi_nerv_receiver_proof_missing" in row["blockers"] for row in hi_rows)
    assert all(
        "hi_nerv_byte_closed_archive_export_missing" in row["promotion_blockers"]
        for row in hi_rows
    )
    assert all(row["experiment_queue_entry"]["status"] == "queued" for row in hi_rows)
    assert all(
        row["experiment_queue_entry"]["cpu_replay_ready"] is False
        for row in hi_rows
    )
    assert all(
        row["experiment_queue_entry"]["exact_gate_ready"] is False
        for row in hi_rows
    )
    hi_step = hi_rows[0]["experiment_queue_entry"]["steps"][0]
    assert hi_step["command"] == hi_rows[0]["command_argv"]
    assert hi_step["resources"]["kind"] == "local_mlx"
    assert {
        (condition["key"], condition.get("equals"))
        for condition in hi_step["postconditions"]
        if condition["type"] == "json_equals"
    } >= {
        ("schema", "compact_renderer_mlx_spine_runner.v1"),
        ("execute_family", "hi_nerv"),
        ("training_executed", True),
        ("score_claim", False),
        ("promotion_eligible", False),
        ("ready_for_exact_eval_dispatch", False),
    }

    snerv_row = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert snerv_row["local_mlx_launch_command_ready"] is True
    assert snerv_row["score_lowering_gate"]["local_mlx_executable"] is True
    assert snerv_row["cpu_replay_ready"] is False
    assert snerv_row["exact_gate_ready"] is False
    assert snerv_row["experiment_queue_entry"]["status"] == "queued"
    assert snerv_row["experiment_queue_entry"]["blocked"] is False
    assert (
        "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only"
        not in snerv_row["blockers"]
    )
    assert snerv_row["execution_epochs"] == 29_650
    assert snerv_row["current_command_is_bounded_proof_not_long_training"] is False
    assert "--snerv-scorer-loop-qat" in snerv_row["command_argv"]
    assert snerv_row["command_argv"][
        snerv_row["command_argv"].index("--epochs") + 1
    ] == "29650"
    snerv_step = snerv_row["experiment_queue_entry"]["steps"][0]
    assert {
        condition["type"] for condition in snerv_step["postconditions"]
    } >= {"json_equals"}
    snerv_blocker_postconditions = [
        condition
        for condition in snerv_step["postconditions"]
        if condition["type"] == "json_array_contains"
    ]
    assert not snerv_blocker_postconditions

    markdown = render_nerv_long_training_campaign_plan_markdown(report)
    assert "NeRV Long-Training Campaign Plan" in markdown
    assert "hi_nerv::hinerv_tiny::lion" in markdown


def test_long_training_campaign_plan_pins_verified_joint_recon_weight(
    tmp_path: Path,
) -> None:
    manifest = _joint_recon_weight_manifest(tmp_path, num_pairs=600)

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        joint_recon_weight_manifest_paths=(manifest,),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    argv = hi["command_argv"]
    assert "--auto-joint-recon-pixel-weight" not in argv
    assert "--recon-pixel-weight-path" in argv
    weight_arg = argv[argv.index("--recon-pixel-weight-path") + 1]
    assert Path(weight_arg).is_file()
    assert "requires_verified_joint_p18_p19_recon_pixel_weight_artifact" not in hi[
        "blockers"
    ]
    artifact = hi["joint_recon_pixel_weight_artifact"]
    assert artifact["num_pairs"] == 600
    assert artifact["manifest_path"] == manifest.as_posix()
    assert artifact["score_claim"] is False
    assert report["joint_recon_weight_artifact_count"] == 1


def test_long_training_campaign_plan_attaches_hinerv_decoder_weight_waterfill(
    tmp_path: Path,
) -> None:
    waterfill_path = tmp_path / "decoder_weight_waterfill.json"
    waterfill = _decoder_weight_waterfill_plan(candidate_id="hinerv_tiny")
    waterfill_path.write_text(json.dumps(waterfill, sort_keys=True), encoding="utf-8")
    waterfill["_decoder_weight_waterfill_plan_path"] = waterfill_path.as_posix()
    waterfill["_decoder_weight_waterfill_plan_sha256"] = _sha256(waterfill_path)
    waterfill["_decoder_weight_waterfill_source_path"] = waterfill_path.as_posix()

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        decoder_weight_waterfill_sources=(waterfill,),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    argv = hi["command_argv"]
    assert "--decoder-weight-waterfill-plan-json" in argv
    assert argv[argv.index("--decoder-weight-waterfill-plan-json") + 1] == (
        waterfill_path.as_posix()
    )
    attachment = hi["decoder_weight_waterfill_plan"]
    assert attachment["attached"] is True
    assert attachment["sha256"] == _sha256(waterfill_path)
    assert attachment["candidate_keys"] == ["hinerv_tiny"]
    assert "hinerv_decoder_weight_waterfill_plan_missing" not in hi["blockers"]
    assert report["decoder_weight_waterfill_source_count"] == 1
    assert report["decoder_weight_waterfill_attached_row_count"] == 1
    assert hi["score_claim"] is False


def test_long_training_campaign_plan_keeps_snerv_bounded_proof_explicit() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_bounded_proof_only=True,
        snerv_bounded_proof_epochs=5,
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert report["snerv_bounded_proof_only"] is True
    assert snerv["execution_epochs"] == 5
    assert snerv["current_command_is_bounded_proof_not_long_training"] is True
    assert snerv["implementation_status"] == "bounded_native_export_scorer_loop_stage_ready"
    assert snerv["command_argv"][snerv["command_argv"].index("--epochs") + 1] == "5"
    assert (
        "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only"
        in snerv["blockers"]
    )
    assert snerv["curriculum_plan"]["training_plan"][
        "native_mlx_long_training_bound"
    ] is False


def test_long_training_campaign_plan_consumes_candidate_feedback_sources() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "scope_matches_candidate": True,
                "receiver_proof_attached": True,
                "full_video_local_prefilter_attached": True,
                "local_cpu_replay_gate_attached": True,
                "measured_archive_bytes": 111_000,
                "measured_num_pairs": 600,
            },
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "family": "snerv",
                "candidate_id": "snerv_tiny",
                "scope_matches_candidate": True,
                "receiver_proof_attached": True,
                "full_video_local_prefilter_attached": True,
                "local_cpu_replay_gate_attached": True,
                "native_mlx_receiver_proof_passed": True,
                "native_mlx_full600_campaign_ready": True,
                "native_mlx_scorer_loop_qat_receiver_contract_satisfied": True,
                "native_mlx_scorer_loop_qat_ready_for_pose_guard_gate": True,
                "native_mlx_scorer_loop_qat_accepted_improvement": True,
                "native_mlx_scorer_loop_qat_best_materialized": True,
                "measured_payload_bytes": 175_000,
                "measured_archive_bytes": 176_000,
                "measured_num_pairs": 600,
            },
        ),
    )

    assert report["candidate_feedback_source_count"] == 2
    assert report["candidate_feedback_row_count"] == 2
    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    hi_curriculum = hi["curriculum_plan"]
    assert hi_curriculum["byte_oracle_logging"]["feedback_ready"] is True
    assert hi_curriculum["byte_oracle_logging"]["measured_archive_bytes"] == 111_000
    assert "hinerv_trained_archive_byte_oracle_feedback_missing" not in hi[
        "blockers"
    ]
    assert "hi_nerv_receiver_proof_missing" not in hi["blockers"]
    assert "hi_nerv_full_video_local_prefilter_missing" not in hi["blockers"]
    assert "hi_nerv_local_cpu_replay_gate_missing" not in hi["blockers"]
    assert hi["candidate_feedback"]["measured_num_pairs"] == 600

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    snerv_curriculum = snerv["curriculum_plan"]
    assert snerv_curriculum["byte_oracle_logging"]["feedback_ready"] is True
    assert snerv_curriculum["byte_oracle_logging"]["measured_payload_bytes"] == 175_000
    assert "snerv_snar1_byte_feedback_missing" not in snerv["blockers"]
    assert "snerv_receiver_proof_missing" not in snerv["blockers"]
    assert "snerv_full_video_local_prefilter_missing" not in snerv["blockers"]
    assert "snerv_local_cpu_replay_gate_missing" not in snerv["blockers"]
    assert "snerv_scorer_loop_qat_receiver_contract_failed" not in snerv["blockers"]
    assert "snerv_scorer_loop_qat_no_accepted_improvement" not in snerv["blockers"]
    assert snerv["candidate_feedback"]["measured_archive_bytes"] == 176_000
    assert (
        "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only"
        not in snerv["blockers"]
    )
    assert snerv["execution_epochs"] == 29_650


def test_long_training_campaign_plan_applies_hinerv_pose_instability_feedback(
) -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=1.0e-3,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "pose_instability_detected": True,
                "observed_learning_rate": 1.0e-3,
                "recommended_learning_rate": 3.0e-4,
                "recommended_launch_mutations": [
                    "lower_learning_rate_from_pose_instability_telemetry"
                ],
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    argv = hi["command_argv"]
    lr = argv[argv.index("--learning-rate") + 1]
    assert lr == "0.0003"
    output_dir = argv[argv.index("--output-dir") + 1]
    output_name = Path(output_dir).name
    assert output_name.startswith("hi_nerv_hinerv_tiny_adamw_feedback")
    assert "pose_instability" in output_name
    assert "lr0.0003" in output_name
    assert "lower_learning_rate_from_pose_instability" in output_name
    assert hi["output_dir_basename"] == output_name
    assert hi["output_dir_reuse_policy"] == "fresh_feedback_mutation_path"
    adjustment = hi["feedback_launch_adjustment"]
    assert adjustment["schema"] == "hinerv_feedback_launch_adjustment.v1"
    assert adjustment["applied"] is True
    assert adjustment["requested_learning_rate"] == 1.0e-3
    assert adjustment["learning_rate"] == 3.0e-4
    assert "lower_learning_rate_from_pose_instability_telemetry" in adjustment[
        "launch_mutations"
    ]
    assert "hinerv_pose_instability_feedback_unapplied" not in hi["blockers"]
    assert hi["candidate_feedback"]["feedback_kind"] == "training_telemetry"
    assert hi["score_claim"] is False


def test_long_training_campaign_plan_applies_hinerv_lr9e5_recovery_feedback(
) -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=9.0e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "pose_instability_detected": True,
                "observed_learning_rate": 9.0e-5,
                "recommended_learning_rate": 2.7e-5,
                "recommended_launch_mutations": [
                    "lower_learning_rate_from_pose_instability_telemetry"
                ],
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    adjustment = hi["feedback_launch_adjustment"]
    assert HINERV_POSE_INSTABILITY_LOW_LR_FLOOR == 3.0e-5
    assert adjustment["applied"] is True
    assert adjustment["lower_learning_rate_applied"] is True
    assert adjustment["pose_protected_pathway_applied"] is False
    assert adjustment["repeated_low_lr_pose_instability"] is False
    assert adjustment["learning_rate"] == 2.7e-5
    assert adjustment["pose_distillation_loss"] == "mse"
    assert adjustment["pose_distillation_huber_delta"] is None
    assert adjustment["reason"] == (
        "pose_instability_recommended_lower_learning_rate"
    )
    assert "above low_learning_rate_floor applies" in adjustment["policy_logic"]
    assert "hinerv_pose_instability_feedback_unapplied" not in hi["blockers"]
    assert (
        "hinerv_repeated_low_lr_pose_instability_requires_pose_protected_pathway"
        not in hi["blockers"]
    )
    assert hi["command_argv"][hi["command_argv"].index("--learning-rate") + 1] == (
        "2.7e-05"
    )
    assert "--pose-distillation-loss" not in hi["command_argv"]
    assert "--pose-distillation-huber-delta" not in hi["command_argv"]


def test_long_training_campaign_plan_blocks_repeated_low_lr_pose_instability(
) -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "pose_instability_detected": True,
                "observed_learning_rate": 2.7e-5,
                "recommended_learning_rate": 8.1e-6,
                "recommended_launch_mutations": [
                    "lower_learning_rate_from_pose_instability_telemetry"
                ],
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    adjustment = hi["feedback_launch_adjustment"]
    assert adjustment["applied"] is True
    assert adjustment["pose_protected_pathway_applied"] is True
    assert adjustment["repeated_low_lr_pose_instability"] is True
    assert adjustment["learning_rate"] == 2.7e-5
    assert adjustment["low_learning_rate_floor"] == 3.0e-5
    assert adjustment["reason"] == (
        "repeated_pose_instability_at_low_lr_pose_protected_pathway"
    )
    assert "switches to pose_distillation_loss=huber" in adjustment["policy_logic"]
    assert "hinerv_pose_instability_feedback_unapplied" not in hi["blockers"]
    assert (
        "hinerv_repeated_low_lr_pose_instability_requires_pose_protected_pathway"
        not in hi["blockers"]
    )
    assert "--learning-rate" in hi["command_argv"]
    assert hi["command_argv"][hi["command_argv"].index("--learning-rate") + 1] == (
        "2.7e-05"
    )
    assert hi["command_argv"][
        hi["command_argv"].index("--pose-distillation-loss") + 1
    ] == "huber"
    assert hi["command_argv"][
        hi["command_argv"].index("--pose-distillation-huber-delta") + 1
    ] == "1"


def test_long_training_campaign_plan_consumes_partial_snerv_runner_feedback() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(_snerv_partial_compact_runner_report(),),
    )

    assert report["candidate_feedback_source_count"] == 1
    assert report["candidate_feedback_row_count"] == 1
    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    feedback = snerv["candidate_feedback"]
    assert feedback["schema"] == "nerv_candidate_feedback_row.v1"
    assert feedback["candidate_id"] == "snerv_tiny"
    assert feedback["measured_num_pairs"] == 2
    assert feedback["scope_matches_candidate"] is False
    assert "partial_pair_byte_feedback_only" in snerv["blockers"]
    assert "snerv_archive_in_loop_byte_oracle_missing" in snerv["blockers"]
    assert "snerv_native_scorer_loop_best_packet_not_materialized" not in snerv[
        "blockers"
    ]
    assert "snerv_scorer_loop_qat_receiver_contract_failed" not in snerv[
        "blockers"
    ]
    assert "snerv_scorer_loop_qat_pose_guard_not_ready" not in snerv["blockers"]
    assert "snerv_scorer_loop_qat_no_accepted_improvement" not in snerv["blockers"]
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" not in snerv[
        "blockers"
    ]
    assert "snerv_mlx_native_full600_campaign_not_ready" in snerv["blockers"]
    assert snerv["score_claim"] is False
    assert snerv["ready_for_exact_eval_dispatch"] is False


def test_long_training_campaign_plan_rejects_unknown_optimizer() -> None:
    with pytest.raises(NervLongTrainingCampaignPlanError, match="unsupported"):
        build_nerv_long_training_campaign_plan(
            hinerv_modelsize_budget=_hinerv_budget(),
            snerv_modelsize_budget=_snerv_budget(),
            optimizer_kinds=("muon",),
        )


def test_build_long_training_campaign_plan_cli_writes_outputs(tmp_path: Path) -> None:
    hinerv = tmp_path / "hinerv_budget.json"
    snerv = tmp_path / "snerv_budget.json"
    out_json = tmp_path / "campaign.json"
    out_md = tmp_path / "campaign.md"
    out_queue = tmp_path / "campaign_queue.json"
    feedback_jsonl = tmp_path / "feedback.jsonl"
    waterfill_bundle = tmp_path / "hinerv_archive_ladder_waterfill.json"
    hinerv.write_text(json.dumps(_hinerv_budget()), encoding="utf-8")
    snerv.write_text(json.dumps(_snerv_budget()), encoding="utf-8")
    feedback_jsonl.write_text(
        json.dumps(_snerv_partial_compact_runner_report(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    waterfill_bundle.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_ladder_waterfill.v1",
                "rows": [
                    {
                        "row_id": "hinerv_tiny",
                        "waterfill_plan": _decoder_weight_waterfill_plan(
                            candidate_id="source_prefix:hinerv_tiny"
                        ),
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "--hinerv-modelsize-budget",
            str(hinerv),
            "--snerv-modelsize-budget",
            str(snerv),
            "--joint-recon-weight-manifest",
            str(_joint_recon_weight_manifest(tmp_path, num_pairs=600)),
            "--optimizer-kind",
            "lion",
            "--candidate-feedback-source",
            str(feedback_jsonl),
            "--decoder-weight-waterfill-source",
            str(waterfill_bundle),
            "--epochs",
            "16",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-queue",
            str(out_queue),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["campaign_row_count"] == 2
    assert payload["candidate_feedback_row_count"] == 1
    assert payload["decoder_weight_waterfill_attached_row_count"] == 1
    hi = next(row for row in payload["campaign_rows"] if row["family"] == "hi_nerv")
    assert "--recon-pixel-weight-path" in hi["command_argv"]
    assert "--decoder-weight-waterfill-plan-json" in hi["command_argv"]
    waterfill_sidecar = Path(
        hi["command_argv"][
            hi["command_argv"].index("--decoder-weight-waterfill-plan-json") + 1
        ]
    )
    assert waterfill_sidecar.is_file()
    assert waterfill_sidecar.parent.name == "decoder_weight_waterfill_sidecars"
    assert hi["decoder_weight_waterfill_plan"]["source_path"] == (
        waterfill_bundle.resolve(strict=False).as_posix()
    )
    snerv_row = next(
        row for row in payload["campaign_rows"] if row["family"] == "snerv"
    )
    assert snerv_row["candidate_feedback"]["candidate_id"] == "snerv_tiny"
    assert "partial_pair_byte_feedback_only" in snerv_row["blockers"]
    assert payload["experiment_queue"]["schema"] == "experiment_queue.v1"
    queue = json.loads(out_queue.read_text(encoding="utf-8"))
    assert queue == payload["experiment_queue"]
    assert queue["experiments"][0]["steps"][0]["postconditions"]
    loaded_queue = load_queue_definition(out_queue)
    assert loaded_queue["schema"] == "experiment_queue.v1"
    assert loaded_queue["experiments"][0]["status"] == "queued"
    assert loaded_queue["experiments"][0]["steps"][0]["resources"]["kind"] == "local_mlx"
    assert out_md.read_text(encoding="utf-8").startswith(
        "# NeRV Long-Training Campaign Plan"
    )

    rc = cli.main(
        [
            "--hinerv-modelsize-budget",
            str(hinerv),
            "--snerv-modelsize-budget",
            str(snerv),
            "--optimizer-kind",
            "lion",
            "--epochs",
            "16",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-queue",
            str(out_queue),
            "--expected-output-json-sha256",
            _sha256(out_json),
            "--expected-output-md-sha256",
            _sha256(out_md),
            "--expected-output-queue-sha256",
            _sha256(out_queue),
        ]
    )

    assert rc == 0


def _hinerv_budget() -> dict:
    return {
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
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _snerv_budget() -> dict:
    return {
        "schema": "snerv_modelsize_budget.v1",
        "selected_candidates": [
            {
                "schema": "snerv_modelsize_candidate.v1",
                "family": "snerv",
                "candidate_id": "snerv_tiny",
                "num_pairs": 600,
                "hard_byte_ceiling": 178_000,
                "decoder_payload_codec": "int4_symmetric",
                "nominal_total_payload_bytes": 190_000,
                "nominal_under_ceiling": False,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _decoder_weight_waterfill_plan(*, candidate_id: str) -> dict:
    return {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": candidate_id,
        "group_count": 2,
        "full_video_coverage": True,
        "receiver_proof_status": "missing",
        "rows": [
            {
                "group_name": "blocks.0.conv.weight",
                "selected_bits": 4,
                "selected_action": "int4",
                "blockers": ["receiver_proof_not_satisfied"],
            },
            {
                "group_name": "head_rgb_0.bias",
                "selected_bits": 2,
                "selected_action": "int2",
                "blockers": ["receiver_proof_not_satisfied"],
            },
        ],
        "blockers": ["receiver_proof_not_satisfied"],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _joint_recon_weight_manifest(root: Path, *, num_pairs: int) -> Path:
    out = root / f"joint_weight_{num_pairs}"
    out.mkdir(parents=True, exist_ok=True)
    weight = out / "joint_p18_p19_recon_pixel_weight.npz"
    weight.write_bytes(b"unit-weight-bytes")
    manifest = out / "joint_p18_p19_recon_pixel_weight_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "joint_p18_p19_recon_pixel_weight_manifest.v1",
                "weight_path": weight.as_posix(),
                "weight_sha256": _sha256(weight),
                "config": {"num_pairs": int(num_pairs)},
                "metadata": {
                    "gradient_health": {
                        "status": "pass_finite",
                        "nonfinite_count": 0,
                    },
                    "training_consumption_recommended": True,
                    "blockers": [],
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def _decoder_weight_waterfill_plan(*, candidate_id: str) -> dict:
    return {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": candidate_id,
        "group_count": 1,
        "rows": [
            {
                "schema": "nerv_decoder_weight_waterfill_row.v1",
                "group_name": "decoder.weight",
                "selected_bits": 4,
                "action": "quantize_int4",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _snerv_partial_compact_runner_report() -> dict:
    candidate = dict(_snerv_budget()["selected_candidates"][0])
    return {
        "schema": "compact_renderer_mlx_spine_runner.v1",
        "execute_family": "snerv",
        "mode": "executed_snerv_archive_bound_advisory_and_exported",
        "num_pairs": 2,
        "archive_bytes": 57_892,
        "archive_sha256": "f" * 64,
        "modelsize_candidate_selection": {"candidate": candidate},
        "candidate_curriculum_plan": {
            "schema": "nerv_candidate_curriculum_plan.v1",
            "family": "snerv",
            "candidate_id": "snerv_tiny",
            "candidate_conditioned": True,
            "byte_oracle_logging": {
                "schema": "nerv_candidate_byte_feedback.v1",
                "candidate_id": "snerv_tiny",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 2,
                "feedback_scope": "partial_pair_advisory",
                "scope_matches_candidate": False,
                "feedback_ready": False,
                "hard_byte_ceiling": 178_000,
                "nominal_total_payload_bytes": 190_000,
                "measured_payload_bytes": 10_441,
                "measured_archive_bytes": 57_892,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        },
        "snerv_mlx_native_export": {
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "native_mlx_full600_campaign_ready": False,
            "scorer_loop_qat_receiver_contract_satisfied": True,
            "scorer_loop_qat_ready_for_pose_guard_gate": True,
            "scorer_loop_qat_accepted_improvement": True,
            "scorer_loop_qat_best_materialized": True,
        },
        "snerv_binary_profile": {
            "profile_written": True,
            "verdict": "snerv_payload_lf_dominant_but_archive_under_frontier",
            "charged_archive_bytes": 57_892,
            "snar1_packet_bytes": 10_824,
            "lf_payload_bytes": 6_156,
            "lf_payload_fraction_of_packet": 0.5687,
            "blockers": [],
        },
        "blockers": ["snerv_mlx_native_full600_campaign_not_ready"],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
