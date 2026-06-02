# SPDX-License-Identifier: MIT
"""Tests for harvestable NeRV candidate byte-feedback rows."""

from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.nerv_candidate_feedback import (
    build_nerv_candidate_feedback_row,
    build_nerv_training_telemetry_feedback_row,
    refresh_nerv_candidate_feedback_report,
    write_nerv_candidate_feedback_files,
    write_nerv_training_telemetry_feedback_files,
    write_refreshed_nerv_candidate_feedback_files,
)


def _runner_report(tmp_path: Path) -> dict[str, object]:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"candidate")
    proof = tmp_path / "receiver_proof.json"
    proof.write_text('{"receiver_contract_satisfied": true}\n', encoding="utf-8")
    return {
        "mode": "executed_hi_nerv_mlx_scoreaware_and_exported",
        "execute_family": "hi_nerv",
        "modelsize_candidate_selection": {
            "candidate": {
                "candidate_id": "hinerv_np600_ld12_ed24_dc32_int4",
            },
            "pr95_stack_binding": {
                "schema": "pr95_stack_binding_requirements.v1",
                "satisfied_count": 5,
                "missing_count": 12,
                "complete": False,
                "blockers": [
                    "hi_nerv_real_posenet_teacher_missing",
                    "hi_nerv_archive_in_loop_byte_oracle_missing",
                ],
            },
            "long_campaign_prelaunch_gate": {
                "schema": "pr95_stack_binding_long_campaign_prelaunch_gate.v1",
                "launch_allowed": False,
                "blockers": [
                    "hi_nerv_real_posenet_teacher_missing",
                ],
            },
        },
        "candidate_curriculum_plan": {
            "candidate_id": "hinerv_np600_ld12_ed24_dc32_int4",
            "candidate_conditioned": True,
            "byte_oracle_logging": {
                "candidate_num_pairs": 600,
                "measured_num_pairs": 32,
                "feedback_scope": "partial_pair_advisory",
                "scope_matches_candidate": False,
                "feedback_ready": False,
                "hard_byte_ceiling": 178_000,
                "nominal_total_payload_bytes": 160_000,
                "measured_payload_bytes": None,
                "measured_archive_bytes": 42_000,
                "measured_minus_nominal_bytes": -118_000,
            },
        },
        "archive_path": archive.as_posix(),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": "archive-sha",
        "snerv_binary_profile": {
            "profile_written": True,
            "profile_path": (tmp_path / "snerv_binary_profile.json").as_posix(),
            "verdict": "current_snerv_artifact_rate_blocked_by_explicit_lf_payload",
            "charged_archive_bytes": 10_057_021,
            "snar1_packet_bytes": 10_030_524,
            "lf_payload_bytes": 9_996_235,
            "lf_payload_fraction_of_packet": 0.9965,
            "lf_payload_bytes_per_coeff": 0.1729,
            "blockers": ["snerv_lf_payload_dominates_packet"],
        },
        "receiver_proof_report_paths": [proof.as_posix()],
        "local_cpu_replay_gate": {
            "schema": "compact_runner_local_cpu_replay_gate.v1",
            "requested": None,
            "default_enabled_for_full_coverage": False,
            "has_full_video_mlx_prefilter": True,
            "local_replay_mlx_prefilter_passed": False,
            "coverage_valid_for_replay": True,
            "executed": False,
        },
        "mlx_prefilter_coverage": {
            "schema": "hprc_mlx_prefilter_coverage.v1",
            "profile_count": 1,
            "has_full_video_mlx_prefilter": True,
            "local_replay_mlx_prefilter_passed": False,
            "best_full_video_mlx_score": 91.0,
            "full_video_profile_paths": [
                (tmp_path / "batched_mlx_profile.json").as_posix()
            ],
            "local_replay_profile_paths": [],
            "blockers": ["mlx_profile_batch_pairs_not_singleton"],
        },
        "blockers": ["partial_pair_byte_feedback_only"],
    }


def test_build_nerv_candidate_feedback_row_preserves_scope_and_false_authority(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runner_report.json"
    source.write_text('{"schema": "runner"}\n', encoding="utf-8")

    row = build_nerv_candidate_feedback_row(
        runner_report=_runner_report(tmp_path),
        source_report_path=source,
    )

    assert row["schema"] == "nerv_candidate_feedback_row.v1"
    assert row["family"] == "hi_nerv"
    assert row["candidate_id"] == "hinerv_np600_ld12_ed24_dc32_int4"
    assert row["candidate_num_pairs"] == 600
    assert row["measured_num_pairs"] == 32
    assert row["feedback_scope"] == "partial_pair_advisory"
    assert row["scope_matches_candidate"] is False
    assert row["feedback_ready"] is False
    assert row["measured_archive_bytes"] == 42_000
    assert row["archive_sha256"] == "archive-sha"
    assert row["snerv_binary_profile_written"] is True
    assert (
        row["snerv_binary_profile_verdict"]
        == "current_snerv_artifact_rate_blocked_by_explicit_lf_payload"
    )
    assert row["snerv_binary_profile_lf_payload_bytes"] == 9_996_235
    assert row["snerv_binary_profile_lf_payload_fraction_of_packet"] == 0.9965
    assert row["snerv_binary_profile_blockers"] == [
        "snerv_lf_payload_dominates_packet"
    ]
    assert row["pr95_stack_binding_schema"] == (
        "pr95_stack_binding_requirements.v1"
    )
    assert row["pr95_stack_binding_satisfied_count"] == 5
    assert row["pr95_stack_binding_missing_count"] == 12
    assert row["pr95_stack_binding_complete"] is False
    assert row["pr95_stack_binding_blockers"] == [
        "hi_nerv_real_posenet_teacher_missing",
        "hi_nerv_archive_in_loop_byte_oracle_missing",
    ]
    assert row["long_campaign_prelaunch_gate_schema"] == (
        "pr95_stack_binding_long_campaign_prelaunch_gate.v1"
    )
    assert row["long_campaign_prelaunch_launch_allowed"] is False
    assert row["long_campaign_prelaunch_blockers"] == [
        "hi_nerv_real_posenet_teacher_missing"
    ]
    assert row["local_cpu_replay_gate_requested"] is None
    assert (
        row["local_cpu_replay_gate_default_enabled_for_full_coverage"]
        is False
    )
    assert row["local_cpu_replay_gate_has_full_video_mlx_prefilter"] is True
    assert (
        row["local_cpu_replay_gate_local_replay_mlx_prefilter_passed"]
        is False
    )
    assert row["local_cpu_replay_gate_coverage_valid_for_replay"] is True
    assert row["local_cpu_replay_gate_executed"] is False
    assert row["mlx_prefilter_profile_count"] == 1
    assert row["mlx_prefilter_has_full_video"] is True
    assert row["mlx_prefilter_local_replay_passed"] is False
    assert row["mlx_prefilter_best_full_video_mlx_score"] == 91.0
    assert len(row["mlx_prefilter_full_video_profile_paths"]) == 1
    assert row["mlx_prefilter_local_replay_profile_paths"] == []
    assert row["mlx_prefilter_blockers"] == [
        "mlx_profile_batch_pairs_not_singleton"
    ]
    assert row["source_report_sha256"] is not None
    assert row["blockers"] == ["partial_pair_byte_feedback_only"]
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_write_nerv_candidate_feedback_files_writes_json_and_append_ledger(
    tmp_path: Path,
) -> None:
    output = write_nerv_candidate_feedback_files(
        runner_report=_runner_report(tmp_path),
        output_dir=tmp_path / "feedback",
    )

    row_path = Path(output["row_path"])
    ledger_path = Path(output["ledger_path"])
    assert row_path.is_file()
    assert ledger_path.is_file()

    row = json.loads(row_path.read_text())
    ledger_rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert output["schema"] == "nerv_candidate_byte_feedback_ledger.v1"
    assert output["append_only"] is True
    assert row["feedback_ready"] is False
    assert ledger_rows == [row]
    assert output["score_claim"] is False
    assert output["promotion_eligible"] is False


def test_training_telemetry_feedback_detects_pose_instability_and_lr_replan(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    rows = []
    for epoch in range(40):
        pose_loss = 100.0 if epoch < 12 else 1_500.0
        pose_axis = 50.0 if epoch < 12 else 1_250.0
        rows.append(
            {
                "epoch": epoch,
                "learning_rate": 1.0e-3,
                "loss_components": {"loss_part_pose_distill": pose_loss},
                "per_axis_decomposition": {"pose": pose_axis, "seg": 5.0},
            }
        )
    telemetry.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    row = build_nerv_training_telemetry_feedback_row(
        telemetry_path=telemetry,
        family="hinerv",
        candidate_id="hinerv_np600_ld4_ed12_dc12_int4_mixed_ceil36000",
        candidate_num_pairs=600,
    )

    assert row["schema"] == "nerv_candidate_feedback_row.v1"
    assert row["telemetry_feedback_schema"] == "nerv_training_telemetry_feedback.v1"
    assert row["family"] == "hi_nerv"
    assert row["feedback_kind"] == "training_telemetry"
    assert row["measured_num_pairs"] == 600
    assert row["scope_matches_candidate"] is True
    assert row["feedback_ready"] is False
    assert row["pose_instability_detected"] is True
    assert row["recommended_learning_rate"] == 3.0e-4
    assert "lower_learning_rate_from_pose_instability_telemetry" in row[
        "recommended_launch_mutations"
    ]
    assert "hi_nerv_pose_instability_telemetry_feedback" in row["blockers"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_write_training_telemetry_feedback_files_writes_manifest_and_ledger(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 1,
                "learning_rate": 1.0e-3,
                "loss_components": {"loss_part_pose_distill": 10.0},
                "per_axis_decomposition": {"pose": 10.0, "seg": 1.0},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    output = write_nerv_training_telemetry_feedback_files(
        telemetry_path=telemetry,
        output_dir=tmp_path / "feedback",
        family="hi_nerv",
        candidate_id="hinerv_tiny",
        candidate_num_pairs=600,
    )

    row_path = Path(output["row_path"])
    ledger_path = Path(output["ledger_path"])
    manifest_path = Path(output["manifest_path"])
    assert row_path.is_file()
    assert ledger_path.is_file()
    assert manifest_path.is_file()
    row = json.loads(row_path.read_text(encoding="utf-8"))
    ledger_rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert output["schema"] == "nerv_training_telemetry_feedback.v1"
    assert ledger_rows == [row]
    assert row["pose_instability_detected"] is False
    assert output["score_claim"] is False


def test_refresh_nerv_candidate_feedback_report_repairs_batched_mlx_signal(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "batched_mlx_profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "max_pairs": 600,
                "num_pairs": 600,
                "n_samples": 600,
                "scorer_batch_pairs": 8,
                "scope_status": {"full_video": "executed"},
                "score_components": {"canonical_score": 91.0},
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report = _runner_report(tmp_path)
    report["num_pairs"] = 600
    report["mlx_profile_paths"] = [profile.as_posix()]
    report["local_cpu_replay_gate"] = {
        "schema": "compact_runner_local_cpu_replay_gate.v1",
        "requested": None,
        "default_enabled_for_full_coverage": False,
        "has_full_video_mlx_prefilter": False,
        "local_replay_mlx_prefilter_passed": False,
        "coverage_valid_for_replay": True,
        "executed": False,
    }
    report["blockers"] = [
        "contest_cpu_cuda_exact_eval_not_executed",
        "full_video_mlx_scorer_replay_not_attached",
        "local_cpu_replay_waiting_for_full_video_mlx_prefilter",
        "hi_nerv_full_video_local_prefilter_missing",
    ]
    report["candidate_curriculum_plan"]["pr95_stack_binding"] = {
        "schema": "pr95_stack_binding_requirements.v1",
        "satisfied_count": 15,
        "missing_count": 2,
        "complete": False,
        "blockers": [
            "hi_nerv_full_video_local_prefilter_missing",
            "hi_nerv_local_cpu_replay_gate_missing",
        ],
    }

    refreshed, refresh = refresh_nerv_candidate_feedback_report(
        runner_report=report,
        repo_root=tmp_path,
    )

    assert refresh["has_full_video_mlx_prefilter"] is True
    assert refresh["local_replay_mlx_prefilter_passed"] is False
    assert refresh["removed_stale_blockers"] == [
        "full_video_mlx_scorer_replay_not_attached",
        "local_cpu_replay_waiting_for_full_video_mlx_prefilter",
        "hi_nerv_full_video_local_prefilter_missing",
    ]
    assert refresh["removed_nested_pr95_stack_binding_blockers"] == [
        "hi_nerv_full_video_local_prefilter_missing"
    ]
    assert refreshed["local_cpu_replay_gate"][
        "has_full_video_mlx_prefilter"
    ] is True
    assert refreshed["local_cpu_replay_gate"][
        "local_replay_mlx_prefilter_passed"
    ] is False
    assert "full_video_mlx_scorer_replay_not_attached" not in refreshed["blockers"]
    assert "local_cpu_replay_waiting_for_full_video_mlx_prefilter" not in refreshed[
        "blockers"
    ]
    assert "mlx_profile_batch_pairs_not_singleton" in refreshed["blockers"]
    assert "local_cpu_replay_blocked_by_mlx_prefilter_score" in refreshed[
        "blockers"
    ]
    pr95_binding = refreshed["candidate_curriculum_plan"]["pr95_stack_binding"]
    assert pr95_binding["blockers"] == ["hi_nerv_local_cpu_replay_gate_missing"]
    assert pr95_binding["missing_count"] == 1
    assert pr95_binding["satisfied_count"] == 16
    assert pr95_binding["complete"] is False


def test_write_refreshed_nerv_candidate_feedback_files_writes_manifest(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "batched_mlx_profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "max_pairs": 600,
                "num_pairs": 600,
                "n_samples": 600,
                "scorer_batch_pairs": 8,
                "scope_status": {"full_video": "executed"},
                "score_components": {"canonical_score": 91.0},
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report = _runner_report(tmp_path)
    report["num_pairs"] = 600
    source = tmp_path / "runner_report.json"
    source.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    output = write_refreshed_nerv_candidate_feedback_files(
        runner_report=report,
        output_dir=tmp_path / "feedback_refresh",
        repo_root=tmp_path,
        source_report_path=source,
        mlx_profile_paths=(profile,),
    )

    refresh_path = Path(output["refresh_path"])
    refreshed_report_path = Path(output["refreshed_runner_report_path"])
    row_path = Path(output["candidate_feedback"]["row_path"])
    assert refresh_path.is_file()
    assert refreshed_report_path.is_file()
    assert row_path.is_file()
    row = json.loads(row_path.read_text(encoding="utf-8"))
    refresh = json.loads(refresh_path.read_text(encoding="utf-8"))
    assert row["mlx_prefilter_has_full_video"] is True
    assert row["mlx_prefilter_blockers"] == ["mlx_profile_batch_pairs_not_singleton"]
    assert refresh["candidate_feedback_row_path"] == row_path.as_posix()
    assert output["score_claim"] is False
