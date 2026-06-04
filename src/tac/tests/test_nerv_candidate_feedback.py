# SPDX-License-Identifier: MIT
"""Tests for harvestable NeRV candidate byte-feedback rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tac.analysis.nerv_candidate_feedback import (
    FULL_VIDEO_MLX_SCORER_FEEDBACK_SCHEMA,
    HINERV_ARCHIVE_LADDER_FEEDBACK_SCHEMA,
    build_hinerv_archive_ladder_feedback_report,
    build_nerv_candidate_feedback_row,
    build_nerv_full_video_mlx_scorer_feedback_row,
    build_nerv_training_telemetry_feedback_row,
    refresh_nerv_candidate_feedback_report,
    write_nerv_candidate_feedback_files,
    write_nerv_full_video_mlx_scorer_feedback_files,
    write_nerv_training_telemetry_feedback_files,
    write_refreshed_nerv_candidate_feedback_files,
)
from tac.repo_io import ArtifactWriteError
from tools import harvest_hinerv_archive_ladder_feedback as harvest_ladder_feedback_cli
from tools import harvest_nerv_full_video_mlx_feedback as harvest_full_video_mlx_cli
from tools.harvest_nerv_training_telemetry_feedback import (
    _effective_stop_reason,
)
from tools.harvest_nerv_training_telemetry_feedback import (
    main as harvest_training_feedback_main,
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


def test_hinerv_archive_ladder_feedback_preserves_rate_proof_without_score_authority(
    tmp_path: Path,
) -> None:
    proof = tmp_path / "receiver_proof.json"
    proof.write_text('{"runtime_consumption_proof_ready": true}\n', encoding="utf-8")
    ladder_path = tmp_path / "hinerv_ladder.json"
    ladder = {
        "schema": "hinerv_archive_size_ladder.v1",
        "num_pairs": 600,
        "archive_rows": [
            {
                "row_id": "hinerv_np600_ld4_ed32_dc4_cnx_int2_mixed_ceil36000_tgtmp0p02",
                "archive_bytes": 45_834,
                "archive_path": "/Volumes/VertigoDataTier/pact/candidate/archive.zip",
                "archive_sha256": "0" * 64,
                "runtime_consumption_proof_ready": True,
                "receiver_proof_path": proof.as_posix(),
                "modelsize_candidate": {"hard_byte_ceiling": 36_000},
                "blockers": [
                    "hinerv_archive_size_row_has_no_nonrate_score",
                    "contest_cpu_cuda_exact_eval_not_executed",
                ],
            }
        ],
    }
    ladder_path.write_text(json.dumps(ladder), encoding="utf-8")

    report = build_hinerv_archive_ladder_feedback_report(
        archive_ladder_report=ladder,
        source_report_path=ladder_path,
    )

    assert report["schema"] == HINERV_ARCHIVE_LADDER_FEEDBACK_SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["row_count"] == 1
    row = report["rows"][0]
    assert row["schema"] == "nerv_candidate_feedback_row.v1"
    assert row["family"] == "hi_nerv"
    assert row["feedback_ready"] is True
    assert row["receiver_proof_attached"] is True
    assert row["receiver_proof_sha256"] == hashlib.sha256(proof.read_bytes()).hexdigest()
    assert row["measured_archive_bytes"] == 45_834
    assert row["measured_num_pairs"] == 600
    assert row["full_video_local_prefilter_attached"] is False
    assert row["local_cpu_replay_gate_attached"] is False
    assert "hinerv_archive_ladder_nonrate_score_missing" in row["sample_generalization_blockers"]
    assert row["promotion_eligible"] is False


def test_hinerv_archive_ladder_feedback_cli_writes_json_and_jsonl(tmp_path: Path) -> None:
    proof = tmp_path / "receiver_proof.json"
    proof.write_text("{}\n", encoding="utf-8")
    ladder = {
        "schema": "hinerv_archive_size_ladder.v1",
        "num_pairs": 600,
        "archive_rows": [
            {
                "row_id": "hinerv_candidate",
                "archive_bytes": 50_000,
                "archive_path": "/Volumes/VertigoDataTier/pact/candidate/archive.zip",
                "archive_sha256": "1" * 64,
                "runtime_consumption_proof_ready": True,
                "receiver_proof_path": proof.as_posix(),
            }
        ],
    }
    ladder_path = tmp_path / "ladder.json"
    output_json = tmp_path / "feedback.json"
    output_jsonl = tmp_path / "feedback.jsonl"
    ladder_path.write_text(json.dumps(ladder), encoding="utf-8")

    assert (
        harvest_ladder_feedback_cli.main(
            [
                "--ladder-json",
                ladder_path.as_posix(),
                "--output-json",
                output_json.as_posix(),
                "--output-jsonl",
                output_jsonl.as_posix(),
            ]
        )
        == 0
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert payload["schema"] == HINERV_ARCHIVE_LADDER_FEEDBACK_SCHEMA
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "hinerv_candidate"
    assert rows[0]["score_claim"] is False


def _mlx_response(
    *,
    family: str = "hi_nerv",
    archive_sha256: str = "5" * 64,
    archive_size_bytes: int = 122_074,
    score: float = 91.571,
    avg_segnet_dist: float = 0.55,
    avg_posenet_dist: float = 132.0,
    n_samples: int = 600,
) -> dict[str, object]:
    return {
        "schema": "mlx_scorer_response.v1",
        "response_family": family,
        "n_samples": n_samples,
        "max_pairs": n_samples,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "score_recomputed_from_components": score,
        "avg_segnet_dist": avg_segnet_dist,
        "avg_posenet_dist": avg_posenet_dist,
        "score_rate_contribution": 0.081,
        "score_axis": "[macOS-MLX research-signal]",
        "candidate_generation_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cache_identity": {
            "candidate": {
                "archive_sha256": archive_sha256,
                "pair_count": n_samples,
            }
        },
    }


def _checkpoint_export(
    tmp_path: Path,
    *,
    family: str = "hi_nerv",
    candidate_id: str = "hinerv_np600_ld16_ed8_dc16_int7_mixed_ceil178000",
    archive_sha256: str = "5" * 64,
    archive_bytes: int = 122_074,
    hard_byte_ceiling: int = 178_000,
) -> dict[str, object]:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    proof = tmp_path / "receiver_proof.json"
    proof.write_text("{}\n", encoding="utf-8")
    payload: dict[str, object] = {
        "schema": f"{family.replace('_', '')}_checkpoint_archive_export.v1",
        "family": family,
        "candidate_id": candidate_id,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "archive_path": archive.as_posix(),
        "receiver_proof_ready": True,
        "receiver_proof_path": proof.as_posix(),
        "receiver_proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
        "modelsize_candidate": {
            "candidate_id": candidate_id,
            "family": family,
            "num_pairs": 600,
            "hard_byte_ceiling": hard_byte_ceiling,
            "nominal_total_payload_bytes": 160_000,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if family == "snerv":
        packet = tmp_path / "packet.snar"
        packet.write_bytes(b"packet")
        payload.update(
            {
                "schema": "snerv_checkpoint_archive_export.v1",
                "packet_bytes": 2_347_396,
                "packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
                "packet_path": packet.as_posix(),
                "receiver_proof_passed": True,
                "receiver_contract_satisfied": True,
            }
        )
    return payload


def test_full_video_mlx_response_feedback_binds_archive_export_and_false_authority(
    tmp_path: Path,
) -> None:
    response_path = tmp_path / "mlx_response.json"
    export_path = tmp_path / "export.json"
    response = _mlx_response()
    export = _checkpoint_export(tmp_path)
    response_path.write_text(json.dumps(response), encoding="utf-8")
    export_path.write_text(json.dumps(export), encoding="utf-8")

    row = build_nerv_full_video_mlx_scorer_feedback_row(
        mlx_response=response,
        archive_export_report=export,
        mlx_response_path=response_path,
        archive_export_report_path=export_path,
        current_segnet_distillation_weight=2.0,
    )

    assert row["schema"] == "nerv_candidate_feedback_row.v1"
    assert row["full_video_mlx_feedback_schema"] == FULL_VIDEO_MLX_SCORER_FEEDBACK_SCHEMA
    assert row["feedback_kind"] == "full_video_mlx_scorer_response"
    assert row["family"] == "hi_nerv"
    assert row["candidate_id"] == "hinerv_np600_ld16_ed8_dc16_int7_mixed_ceil178000"
    assert row["measured_num_pairs"] == 600
    assert row["scope_matches_candidate"] is True
    assert row["feedback_ready"] is True
    assert row["receiver_proof_attached"] is True
    assert row["full_video_local_prefilter_attached"] is True
    assert row["local_cpu_replay_gate_local_replay_mlx_prefilter_passed"] is False
    assert row["full_video_mlx_scorer_response"]["seg_score_term"] == pytest.approx(55.0)
    assert row["full_video_mlx_scorer_response"]["pose_score_term"] == pytest.approx(
        (10.0 * 132.0) ** 0.5
    )
    assert row["full_video_mlx_scorer_response"]["archive_under_hard_byte_ceiling"] is True
    control = row["full_video_mlx_response_control"]
    assert control["action"] == "checkpoint_then_supersede_with_full_video_fit_mutation"
    assert control["segnet_fit_failure_detected"] is True
    assert control["pose_fit_failure_detected"] is True
    assert row["pose_tail_burst_detected"] is True
    assert control["recommended_segnet_distillation_weight"] == 4.0
    assert "increase_segnet_distillation_weight_from_full_video_mlx_response" in row[
        "recommended_launch_mutations"
    ]
    assert row["direct_feedback_blockers"] == []
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_full_video_mlx_response_feedback_infers_segnet_weight_from_export(
    tmp_path: Path,
) -> None:
    export = _checkpoint_export(tmp_path)
    export["command_args"] = {"segnet_distillation_weight": 16.0}

    row = build_nerv_full_video_mlx_scorer_feedback_row(
        mlx_response=_mlx_response(),
        archive_export_report=export,
    )

    assert row["observed_segnet_distillation_weight"] == 16.0
    assert row["segnet_distillation_weight_source"] == (
        "command_args.segnet_distillation_weight"
    )
    assert row["recommended_segnet_distillation_weight"] is None
    assert (
        "increase_segnet_distillation_weight_from_full_video_mlx_response"
        not in row["recommended_launch_mutations"]
    )
    control = row["full_video_mlx_response_control"]
    assert control["observed_segnet_distillation_weight"] == 16.0
    assert control["recommended_segnet_distillation_weight"] is None


def test_full_video_mlx_feedback_writer_loads_export_startup_json_weight(
    tmp_path: Path,
) -> None:
    startup = tmp_path / "compact_renderer_mlx_spine_runner_startup.json"
    startup.write_text(
        json.dumps(
            {
                "campaign_identity": {
                    "argv": {"segnet_distillation_weight": 16.0}
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    export = _checkpoint_export(tmp_path)
    export["startup_json_path"] = startup.as_posix()

    manifest = write_nerv_full_video_mlx_scorer_feedback_files(
        mlx_response=_mlx_response(),
        archive_export_report=export,
        output_dir=tmp_path / "feedback",
    )

    row = manifest["row"]
    assert row["observed_segnet_distillation_weight"] == 16.0
    assert row["segnet_distillation_weight_source"] == (
        "runner_startup_json.campaign_identity.argv.segnet_distillation_weight"
    )
    assert row["recommended_segnet_distillation_weight"] is None


def test_full_video_mlx_response_feedback_fails_closed_on_archive_sha_mismatch(
    tmp_path: Path,
) -> None:
    row = build_nerv_full_video_mlx_scorer_feedback_row(
        mlx_response=_mlx_response(archive_sha256="6" * 64),
        archive_export_report=_checkpoint_export(tmp_path, archive_sha256="5" * 64),
    )

    assert row["feedback_ready"] is False
    assert "full_video_mlx_response_archive_sha256_mismatch" in row[
        "direct_feedback_blockers"
    ]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_full_video_mlx_response_feedback_marks_snerv_rate_over_cap(
    tmp_path: Path,
) -> None:
    candidate_id = "snerv_np600_haar_lv5_fc36_int8_ceil178000"
    row = build_nerv_full_video_mlx_scorer_feedback_row(
        mlx_response=_mlx_response(
            family="snerv",
            archive_size_bytes=444_036,
            archive_sha256="7" * 64,
        ),
        archive_export_report=_checkpoint_export(
            tmp_path,
            family="snerv",
            candidate_id=candidate_id,
            archive_sha256="7" * 64,
            archive_bytes=444_036,
            hard_byte_ceiling=178_000,
        ),
    )

    assert row["family"] == "snerv"
    assert row["candidate_id"] == candidate_id
    assert row["feedback_ready"] is False
    assert row["full_video_mlx_scorer_response"]["archive_under_hard_byte_ceiling"] is False
    assert "snerv_full_video_mlx_response_archive_over_hard_byte_ceiling" in row[
        "direct_feedback_blockers"
    ]
    control = row["full_video_mlx_response_control"]
    assert control["action"] == "checkpoint_then_stop_same_representation_rate_over_cap"
    assert "switch_snerv_representation_before_more_same_modelsize_training" in control[
        "recommended_launch_mutations"
    ]
    assert row["score_claim"] is False


def test_write_and_cli_full_video_mlx_feedback_emit_candidate_feedback_row(
    tmp_path: Path,
) -> None:
    response_path = tmp_path / "mlx_response.json"
    export_path = tmp_path / "export.json"
    response_path.write_text(json.dumps(_mlx_response()), encoding="utf-8")
    export_path.write_text(json.dumps(_checkpoint_export(tmp_path)), encoding="utf-8")
    output = write_nerv_full_video_mlx_scorer_feedback_files(
        mlx_response=json.loads(response_path.read_text(encoding="utf-8")),
        archive_export_report=json.loads(export_path.read_text(encoding="utf-8")),
        mlx_response_path=response_path,
        archive_export_report_path=export_path,
        output_dir=tmp_path / "feedback",
    )

    row = json.loads(Path(output["row_path"]).read_text(encoding="utf-8"))
    assert output["schema"] == FULL_VIDEO_MLX_SCORER_FEEDBACK_SCHEMA
    assert row["schema"] == "nerv_candidate_feedback_row.v1"
    assert row["full_video_mlx_response_attached"] is True

    cli_out = tmp_path / "cli_feedback"
    assert (
        harvest_full_video_mlx_cli.main(
            [
                "--mlx-response",
                response_path.as_posix(),
                "--archive-export-json",
                export_path.as_posix(),
                "--output-dir",
                cli_out.as_posix(),
            ]
        )
        == 0
    )
    cli_row = json.loads(
        (cli_out / "nerv_full_video_mlx_scorer_feedback_row.json").read_text(
            encoding="utf-8"
        )
    )
    assert cli_row["candidate_id"] == row["candidate_id"]
    assert cli_row["score_claim"] is False


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


def test_candidate_feedback_flags_small_pair_distortion_as_smoke_only(
    tmp_path: Path,
) -> None:
    report = _runner_report(tmp_path)
    byte_feedback = report["candidate_curriculum_plan"]["byte_oracle_logging"]
    byte_feedback["measured_num_pairs"] = 4
    byte_feedback["measured_archive_bytes"] = 12_345
    report["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] = False
    report["local_cpu_replay_gate"]["coverage_valid_for_replay"] = False
    report["mlx_prefilter_coverage"] = {
        "schema": "hprc_mlx_prefilter_coverage.v1",
        "profile_count": 0,
        "has_full_video_mlx_prefilter": False,
        "local_replay_mlx_prefilter_passed": False,
        "blockers": ["full_video_mlx_scorer_replay_not_attached"],
    }

    row = build_nerv_candidate_feedback_row(runner_report=report)

    gate = row["sample_generalization_gate"]
    assert gate["schema"] == "nerv_sample_generalization_gate.v1"
    assert gate["candidate_num_pairs"] == 600
    assert gate["measured_num_pairs"] == 4
    assert gate["small_pair_smoke_only"] is True
    assert gate["representative_distortion_evidence"] is False
    assert "small_pair_distortion_smoke_only_not_representative" in row[
        "sample_generalization_blockers"
    ]
    assert "full600_or_hardpair_distortion_replay_required" in row["blockers"]
    assert "segnet_scores_only_last_frame_so_frame1_boundary_coverage_matters" in gate[
        "why_small_pair_can_look_good"
    ]
    assert gate["contest_score_geometry"]["segnet_domain"] == (
        "last_frame_of_each_pair_only_at_scorer_resize"
    )
    assert gate["contest_score_geometry"]["posenet_domain"] == (
        "both_frames_of_each_pair_at_scorer_resize"
    )
    assert gate["contest_score_geometry"]["rate_domain"] == (
        "single_receiver_archive_zip_bytes"
    )
    assert "score_axis_distillation_on_segnet_frame1_and_posenet_pair" in gate[
        "pr95_distortion_controls_to_bind"
    ]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_candidate_feedback_keeps_four_pair_rows_profile_only_until_single_archive(
    tmp_path: Path,
) -> None:
    report = _runner_report(tmp_path)
    report["four_pair_byte_rows"] = [
        {
            "schema": "nerv_candidate_feedback_row.v1",
            "candidate_num_pairs": 600,
            "measured_num_pairs": 4,
            "measured_archive_bytes": 1_024,
        },
        {
            "schema": "nerv_candidate_feedback_row.v1",
            "candidate_num_pairs": 600,
            "measured_num_pairs": 4,
            "measured_archive_bytes": 2_048,
        },
    ]

    row = build_nerv_candidate_feedback_row(runner_report=report)

    gate = row["sample_generalization_gate"]
    assert gate["representative_distortion_evidence"] is True
    assert gate["chunked_micro_rows_profile_only"] is True
    assert gate["chunked_micro_row_evidence"]["row_count"] == 2
    assert gate["chunked_micro_row_evidence"]["measured_pair_sum"] == 8
    assert gate["chunked_micro_row_evidence"]["archive_byte_sum"] == 3_072
    assert "four_pair_chunk_rows_profile_only_no_rate_arbitrage" in row[
        "sample_generalization_blockers"
    ]
    assert "rate_term_is_total_archive_zip_bytes_not_per_chunk_best_row_bytes" in gate[
        "why_four_pair_rows_do_not_trick_the_scorer"
    ]


def test_candidate_feedback_does_not_count_candidate_pairs_as_replay_coverage(
    tmp_path: Path,
) -> None:
    report = _runner_report(tmp_path)
    report["local_cpu_replay_summary"] = {
        "schema": "compact_base_local_replay_summary.v1",
        "candidate_num_pairs": 600,
        "measured_num_pairs": 4,
        "score_claim": False,
    }
    report["mlx_prefilter_coverage"] = {
        "schema": "hprc_mlx_prefilter_coverage.v1",
        "profile_count": 0,
        "has_full_video_mlx_prefilter": False,
        "local_replay_mlx_prefilter_passed": False,
        "blockers": [],
    }
    byte_feedback = report["candidate_curriculum_plan"]["byte_oracle_logging"]
    byte_feedback["measured_num_pairs"] = 4

    row = build_nerv_candidate_feedback_row(runner_report=report)

    gate = row["sample_generalization_gate"]
    assert gate["local_replay_full_video"] is False
    assert gate["local_replay_num_pairs"] == 4
    assert gate["small_pair_smoke_only"] is True
    assert gate["representative_distortion_evidence"] is False
    assert "full600_or_hardpair_distortion_replay_required" in row["blockers"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_candidate_feedback_accepts_hard_pair_coverage_as_distortion_gate(
    tmp_path: Path,
) -> None:
    report = _runner_report(tmp_path)
    byte_feedback = report["candidate_curriculum_plan"]["byte_oracle_logging"]
    byte_feedback["measured_num_pairs"] = 4
    report["mlx_prefilter_coverage"] = {
        "schema": "hprc_mlx_prefilter_coverage.v1",
        "profile_count": 0,
        "has_full_video_mlx_prefilter": False,
        "local_replay_mlx_prefilter_passed": False,
        "blockers": [],
    }
    report["xray_hardpair_coverage"] = {
        "schema": "xray_hardpair_coverage.v1",
        "hard_pair_count": 48,
        "score_axis_hard_pair_coverage": True,
        "verdict": "covers_segnet_frame1_and_posenet_pair_tail",
    }

    row = build_nerv_candidate_feedback_row(runner_report=report)

    gate = row["sample_generalization_gate"]
    assert gate["hard_pair_distortion_coverage"] is True
    assert gate["representative_distortion_evidence"] is True
    assert gate["small_pair_smoke_only"] is False
    assert "small_pair_distortion_smoke_only_not_representative" not in row[
        "sample_generalization_blockers"
    ]


def test_candidate_feedback_hard_pair_coverage_parse_fail_closes(
    tmp_path: Path,
) -> None:
    report = _runner_report(tmp_path)
    report["xray_hardpair_coverage"] = {
        "schema": "xray_hardpair_coverage.v1",
        "hard_pair_count": 2,
        "score_axis_hard_pair_coverage": True,
        "prioritized_pair_indices": [17, 1.9],
    }

    row = build_nerv_candidate_feedback_row(runner_report=report)

    coverage = row["sample_generalization_gate"]["hard_pair_coverage"]
    assert coverage["prioritized_pair_indices"] == []
    assert coverage["blockers"] == ["hard_pair_indices_parse_failed"]
    assert "hard_pair_indices_parse_failed" in row["blockers"]
    assert "hard_pair_indices_parse_failed" in row[
        "sample_generalization_blockers"
    ]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def _snerv_native_runner_report(
    tmp_path: Path,
    *,
    required_pair_proof: bool,
    native_num_pairs: int = 600,
    loss_worsened_training: bool = False,
) -> dict[str, object]:
    packet = tmp_path / "packet.snar"
    packet.write_bytes(b"snerv-native-packet")
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"snerv-native-archive")
    packet_sha = hashlib.sha256(packet.read_bytes()).hexdigest()
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    native_export = {
        "executed": True,
        "candidate_id": "snerv-native-full600",
        "num_pairs": native_num_pairs,
        "packet_path": packet.as_posix(),
        "packet_bytes": packet.stat().st_size,
        "packet_sha256": packet_sha,
        "archive_path": archive.as_posix(),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
        "receiver_proof_passed": True,
        "receiver_contract_satisfied": True,
        "blockers": [],
    }
    if loss_worsened_training:
        native_export.update(
            {
                "native_mlx_training_executed": False,
                "native_mlx_hf_decoder_training": {
                    "schema": "snerv_native_mlx_hf_decoder_training.v1",
                    "attempted": True,
                    "requested_steps": 2,
                    "executed": False,
                    "accepted": False,
                    "any_loss_worsened": True,
                    "all_final_losses_finite": True,
                    "blockers": ["snerv_native_mlx_decoder_loss_worsened"],
                },
            }
        )
    return {
        "mode": "executed_snerv_native_mlx_and_exported",
        "execute_family": "snerv",
        "modelsize_candidate_selection": {
            "candidate": {
                "candidate_id": "snerv-native-full600",
                "num_pairs": 600,
            }
        },
        "candidate_curriculum_plan": {
            "candidate_id": "snerv-native-full600",
            "candidate_conditioned": True,
            "byte_oracle_logging": {
                "candidate_num_pairs": 600,
                "measured_num_pairs": 32,
                "feedback_scope": "partial_pair_advisory",
                "scope_matches_candidate": False,
                "feedback_ready": False,
                "hard_byte_ceiling": 178_000,
                "nominal_total_payload_bytes": 150_000,
                "measured_payload_bytes": None,
                "measured_archive_bytes": None,
                "measured_minus_nominal_bytes": None,
            },
        },
        "snerv_mlx_native_export": native_export,
        "snerv_mlx_native_file_backed_export_evidence": {
            "file_backed_export_proof_passed": True,
            "required_pair_file_backed_export_proof_passed": required_pair_proof,
            "num_pairs": native_num_pairs,
            "blockers": [] if required_pair_proof else ["partial_pair_file_backed_export"],
        },
        "blockers": [],
    }


def test_snerv_native_file_backed_full600_bytes_become_feedback(
    tmp_path: Path,
) -> None:
    row = build_nerv_candidate_feedback_row(
        runner_report=_snerv_native_runner_report(
            tmp_path,
            required_pair_proof=True,
            native_num_pairs=600,
        )
    )

    assert row["schema"] == "nerv_candidate_feedback_row.v1"
    assert row["family"] == "snerv"
    assert row["candidate_id"] == "snerv-native-full600"
    assert row["byte_feedback_source"] == "snerv_mlx_native_file_backed_export"
    assert row["feedback_scope"] == "full600_native_file_backed_snar1_export"
    assert row["candidate_num_pairs"] == 600
    assert row["measured_num_pairs"] == 600
    assert row["scope_matches_candidate"] is True
    assert row["feedback_ready"] is True
    assert row["measured_payload_bytes"] == len(b"snerv-native-packet")
    assert row["measured_archive_bytes"] == len(b"snerv-native-archive")
    assert row["measured_minus_nominal_bytes"] == (
        len(b"snerv-native-packet") - 150_000
    )
    native_feedback = row["snerv_mlx_native_file_backed_byte_feedback"]
    assert native_feedback["packet_sha256"] == hashlib.sha256(
        b"snerv-native-packet"
    ).hexdigest()
    assert native_feedback["archive_sha256"] == hashlib.sha256(
        b"snerv-native-archive"
    ).hexdigest()
    assert native_feedback["score_claim"] is False
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_snerv_native_file_backed_bytes_require_matching_file_hashes(
    tmp_path: Path,
) -> None:
    report = _snerv_native_runner_report(
        tmp_path,
        required_pair_proof=True,
        native_num_pairs=600,
    )
    report["snerv_mlx_native_export"]["packet_sha256"] = "0" * 64

    row = build_nerv_candidate_feedback_row(runner_report=report)

    assert row["byte_feedback_source"] is None
    assert row["feedback_ready"] is False
    assert row["snerv_mlx_native_file_backed_byte_feedback"] is None


def test_snerv_native_loss_worsened_training_blocks_byte_feedback(
    tmp_path: Path,
) -> None:
    row = build_nerv_candidate_feedback_row(
        runner_report=_snerv_native_runner_report(
            tmp_path,
            required_pair_proof=True,
            native_num_pairs=600,
            loss_worsened_training=True,
        )
    )

    assert row["byte_feedback_source"] is None
    assert row["feedback_ready"] is False
    assert row["snerv_mlx_native_file_backed_byte_feedback"] is None
    assert row["snerv_mlx_native_training_export_guard_passed"] is False
    assert "snerv_native_mlx_decoder_loss_worsened_export_blocked" in row[
        "snerv_mlx_native_training_export_guard_blockers"
    ]
    assert "snerv_native_mlx_decoder_loss_worsened_export_blocked" in row["blockers"]
    assert row["score_claim"] is False


def test_snerv_native_partial_file_backed_bytes_do_not_unblock_feedback(
    tmp_path: Path,
) -> None:
    row = build_nerv_candidate_feedback_row(
        runner_report=_snerv_native_runner_report(
            tmp_path,
            required_pair_proof=False,
            native_num_pairs=32,
        )
    )

    assert row["byte_feedback_source"] is None
    assert row["feedback_scope"] == "partial_pair_advisory"
    assert row["measured_num_pairs"] == 32
    assert row["feedback_ready"] is False
    assert row["measured_payload_bytes"] is None
    assert row["measured_archive_bytes"] is None
    assert row["snerv_mlx_native_file_backed_byte_feedback"] is None
    assert row["snerv_mlx_native_required_pair_file_backed_export_proof_passed"] is False
    assert row["score_claim"] is False


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
    assert row["training_row_count"] == 40
    assert row["training_first_epoch"] == 0
    assert row["training_last_epoch"] == 39
    assert row["training_median_pose_axis"] == 1_250.0
    assert row["training_median_seg_axis"] == 5.0
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


def test_training_telemetry_feedback_detects_partial_window_pose_instability(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "partial_window_telemetry.jsonl"
    rows = [
        {
            "epoch": epoch,
            "learning_rate": 2.7e-5,
            "loss_components": {"loss_part_pose_distill": 1_200_000.0},
            "per_axis_decomposition": {"pose": 900_000.0, "seg": 6.0},
        }
        for epoch in range(27)
    ]
    telemetry.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    row = build_nerv_training_telemetry_feedback_row(
        telemetry_path=telemetry,
        family="hi_nerv",
        candidate_id="hinerv_np600_ld4_ed12_dc12_int4_mixed_ceil36000",
        candidate_num_pairs=600,
    )

    assert row["pose_instability_detected"] is True
    assert row["pose_instability_partial_window_detected"] is True
    assert row["pose_instability_last_window_bad_fraction"] == 1.0
    assert abs(float(row["recommended_learning_rate"]) - 8.1e-6) < 1.0e-12
    assert "lower_learning_rate_from_pose_instability_telemetry" in row[
        "recommended_launch_mutations"
    ]


def test_training_telemetry_feedback_does_not_replan_recovered_midrun_instability(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "recovered_midrun_telemetry.jsonl"
    rows = []
    for epoch in range(96):
        pose_bad = epoch < 24
        rows.append(
            {
                "epoch": epoch,
                "learning_rate": 2.7e-5,
                "loss_components": {
                    "loss_part_pose_distill": 1_200_000.0 if pose_bad else 2.0,
                },
                "per_axis_decomposition": {
                    "pose": 900_000.0 if pose_bad else 2.0,
                    "seg": 6.0,
                },
            }
        )
    telemetry.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    row = build_nerv_training_telemetry_feedback_row(
        telemetry_path=telemetry,
        family="hi_nerv",
        candidate_id="hinerv_np600_ld4_ed12_dc8_portfolio_auto_ceil36000",
        candidate_num_pairs=600,
        stop_reason="midrun_feedback_snapshot_do_not_stop_training",
    )

    assert row["training_stopped"] is False
    assert row["pose_instability_ever_detected"] is True
    assert row["pose_instability_recovered"] is True
    assert row["pose_instability_active_latest_window"] is False
    assert row["pose_instability_detected"] is False
    assert row["recommended_learning_rate"] is None
    assert row["recommended_launch_mutations"] == []
    assert "hi_nerv_pose_instability_telemetry_feedback" not in row["blockers"]


def test_training_telemetry_feedback_detects_pose_tail_burst_without_explosion(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "pose_tail_burst_telemetry.jsonl"
    rows = []
    for epoch in range(96):
        recent_tail_spike = epoch >= 32 and (epoch - 32) in {4, 19, 37, 55}
        pose_axis = 20.0 if recent_tail_spike else 2.0
        rows.append(
            {
                "epoch": epoch,
                "learning_rate": 2.7e-5,
                "loss_components": {"loss_part_pose_distill": pose_axis},
                "per_axis_decomposition": {"pose": pose_axis, "seg": 6.0},
            }
        )
    telemetry.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    row = build_nerv_training_telemetry_feedback_row(
        telemetry_path=telemetry,
        family="hi_nerv",
        candidate_id="hinerv_np600_ld28_ed12_dc32_hfg_cnx_int4_mixed_ceil285000",
        candidate_num_pairs=600,
        stop_reason="training_running_midrun_feedback_snapshot",
    )

    assert row["training_stopped"] is False
    assert row["pose_instability_detected"] is False
    assert row["pose_tail_burst_detected"] is True
    assert row["pose_tail_burst_threshold"] == 8.0
    assert row["pose_tail_burst_recent_bad_fraction"] == 4 / 64
    assert row["pose_tail_burst_recent_max"] == 20.0
    assert row["training_control_action"] == (
        "continue_running_queue_hardpair_prioritized_successor"
    )
    assert row["training_control_should_stop_current_run"] is False
    assert row["training_control_successor_required"] is True
    assert "hi_nerv_pose_tail_burst_telemetry_feedback" in row["blockers"]
    assert (
        "launch_hard_pair_prioritized_sampler_successor"
        in row["recommended_launch_mutations"]
    )


def test_training_telemetry_feedback_detects_segnet_stagnation(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "segnet_stagnation_telemetry.jsonl"
    rows = [
        {
            "epoch": epoch,
            "learning_rate": 2.7e-5,
            "loss_components": {
                "loss_part_pose_distill": 1.0,
                "loss_part_distill": 6.3,
                "loss_part_weighted_distill": 12.6,
            },
            "per_axis_decomposition": {
                "pose": 2.0,
                "seg": 6.3 - min(epoch, 127) * 0.0005,
            },
        }
        for epoch in range(160)
    ]
    telemetry.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    row = build_nerv_training_telemetry_feedback_row(
        telemetry_path=telemetry,
        family="hi_nerv",
        candidate_id="hinerv_np600_ld4_ed12_dc8_portfolio_auto_ceil36000",
        candidate_num_pairs=600,
    )

    assert row["pose_instability_detected"] is False
    assert row["seg_stagnation_detected"] is True
    assert row["training_last_epoch"] == 159
    assert row["training_median_seg_axis"] is not None
    assert row["training_median_pose_axis"] == 2.0
    assert row["observed_segnet_distillation_weight"] == 2.0
    assert row["recommended_segnet_distillation_weight"] == 4.0
    assert row["seg_stagnation_relative_improvement"] < 0.05
    assert row["training_control_action"] == "terminal_feedback_no_live_training_action"
    assert row["training_control_should_stop_current_run"] is False
    assert "increase_segnet_distillation_weight_from_stagnation_telemetry" in row[
        "recommended_launch_mutations"
    ]
    assert "hi_nerv_segnet_stagnation_telemetry_feedback" in row["blockers"]
    assert row["score_claim"] is False


def test_training_telemetry_feedback_uses_family_specific_blockers(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "snerv_segnet_stagnation_telemetry.jsonl"
    rows = [
        {
            "epoch": epoch,
            "learning_rate": 2.7e-5,
            "loss_components": {
                "loss_part_pose_distill": 1.0,
                "loss_part_distill": 6.3,
                "loss_part_weighted_distill": 12.6,
            },
            "per_axis_decomposition": {
                "pose": 2.0,
                "seg": 6.3 - min(epoch, 127) * 0.0005,
            },
        }
        for epoch in range(160)
    ]
    telemetry.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    row = build_nerv_training_telemetry_feedback_row(
        telemetry_path=telemetry,
        family="snerv",
        candidate_id="snerv_np600_haar_lv5_fc36e0_native",
        candidate_num_pairs=600,
    )

    assert row["family"] == "snerv"
    assert "snerv_trained_archive_byte_oracle_feedback_missing" in row["blockers"]
    assert "snerv_byte_closed_archive_export_missing" in row["blockers"]
    assert "snerv_receiver_proof_missing" in row["blockers"]
    assert "snerv_full_video_local_prefilter_missing" in row["blockers"]
    assert "snerv_local_cpu_replay_gate_missing" in row["blockers"]
    assert "snerv_segnet_stagnation_telemetry_feedback" in row["blockers"]
    assert "hi_nerv_segnet_stagnation_telemetry_feedback" not in row["blockers"]
    assert (
        "treat_previous_snerv_run_as_segnet_fit_failure_not_rate_negative"
        in row["recommended_launch_mutations"]
    )
    assert all(
        "treat_previous_hi_nerv_run_as" not in mutation
        for mutation in row["recommended_launch_mutations"]
    )
    assert (
        "treat_previous_snerv_run_as_segnet_fit_failure_not_rate_negative"
        in row["training_telemetry"]["recommended_launch_mutations"]
    )
    assert all(
        "treat_previous_hi_nerv_run_as" not in mutation
        for mutation in row["training_telemetry"]["recommended_launch_mutations"]
    )


def test_training_telemetry_feedback_uses_current_segnet_pressure_for_pr95_stage(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "snerv_pr95_stage_seg_stagnation.jsonl"
    rows = [
        {
            "epoch": epoch,
            "learning_rate": 2.7e-5,
            "loss_components": {
                "loss_part_pr95_stage_pose_surrogate": 3.0,
                "loss_part_pr95_stage_seg_surrogate": 1.5,
                "loss_part_weighted_pr95_stage_pose_surrogate": 3.0,
                "loss_part_weighted_pr95_stage_seg_surrogate": 150.0,
                "pr95_stage_index": 1.0,
                "pr95_stage_uses_muon": 0.0,
            },
            "per_axis_decomposition": {
                "pose": 3.0,
                "seg": 6.4 - min(epoch, 127) * 0.0005,
            },
        }
        for epoch in range(192)
    ]
    telemetry.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    row = build_nerv_training_telemetry_feedback_row(
        telemetry_path=telemetry,
        family="snerv",
        candidate_id="snerv_np600_pr95_stage_current_seg_w2",
        candidate_num_pairs=600,
        stop_reason="training_running_midrun_feedback_snapshot",
        current_segnet_distillation_weight=2.0,
    )

    assert row["seg_stagnation_detected"] is True
    assert row["observed_segnet_distillation_weight"] == 2.0
    assert row["segnet_distillation_weight_source"] == (
        "harvest_current_segnet_distillation_weight"
    )
    assert row["recommended_segnet_distillation_weight"] == 4.0
    assert row["training_control_action"] == (
        "checkpoint_then_supersede_with_higher_segnet_weight"
    )
    assert row["training_control_should_stop_current_run"] is True


def test_training_telemetry_feedback_does_not_lower_above_cap_segnet_pressure(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "snerv_pr95_stage_seg_stagnation_above_cap.jsonl"
    rows = [
        {
            "epoch": epoch,
            "learning_rate": 2.7e-5,
            "loss_components": {
                "loss_part_pr95_stage_pose_surrogate": 1.0,
                "loss_part_pr95_stage_seg_surrogate": 1.4,
                "loss_part_weighted_pr95_stage_pose_surrogate": 1.0,
                "loss_part_weighted_pr95_stage_seg_surrogate": 2240.0,
                "pr95_stage_index": 1.0,
                "pr95_stage_uses_muon": 1.0,
            },
            "per_axis_decomposition": {
                "pose": 1.0,
                "seg": 8.4 - min(epoch, 127) * 0.0001,
            },
        }
        for epoch in range(192)
    ]
    telemetry.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    row = build_nerv_training_telemetry_feedback_row(
        telemetry_path=telemetry,
        family="hi_nerv",
        candidate_id="hinerv_np600_current_seg_w16",
        candidate_num_pairs=600,
        stop_reason="training_running_midrun_feedback_snapshot",
        current_segnet_distillation_weight=16.0,
    )

    assert row["seg_stagnation_detected"] is True
    assert row["observed_segnet_distillation_weight"] == 16.0
    assert row["recommended_segnet_distillation_weight"] is None
    assert row["training_control_action"] == "continue_running"
    assert row["training_control_should_stop_current_run"] is False


def test_running_training_telemetry_feedback_recommends_checkpoint_supersede_for_flat_segnet(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "running_segnet_flat_telemetry.jsonl"
    rows = [
        {
            "epoch": epoch,
            "learning_rate": 2.7e-5,
            "loss_components": {
                "loss_part_pose_distill": 1.0,
                "loss_part_distill": 6.1,
                "loss_part_weighted_distill": 24.4,
            },
            "per_axis_decomposition": {
                "pose": 1.5,
                "seg": 6.2 - min(epoch, 127) * 0.0001,
            },
        }
        for epoch in range(192)
    ]
    telemetry.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    row = build_nerv_training_telemetry_feedback_row(
        telemetry_path=telemetry,
        family="hi_nerv",
        candidate_id="hinerv_np600_ld28_ed12_dc32_hfg_cnx_int4_mixed_ceil285000",
        candidate_num_pairs=600,
        stop_reason="training_running_midrun_feedback_snapshot",
    )

    assert row["training_stopped"] is False
    assert row["seg_stagnation_detected"] is True
    assert row["seg_recent_relative_improvement"] < 0.01
    assert (
        row["training_control_action"]
        == "checkpoint_then_supersede_with_higher_segnet_weight"
    )
    assert row["training_control_should_stop_current_run"] is True
    assert row["training_control_successor_required"] is True
    assert (
        "increase_segnet_distillation_weight_from_stagnation_telemetry"
        in row["training_control"]["recommended_successor_mutations"]
    )


def test_training_telemetry_feedback_treats_pr95_pre_final_no_muon_as_expected(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "pr95_stage5_telemetry.jsonl"
    rows = [
        {
            "epoch": epoch,
            "learning_rate": 2.7e-5,
            "loss_components": {
                "loss_part_pose_distill": 1.0,
                "loss_part_distill": 5.8,
                "pr95_stage_index": 5.0,
                "pr95_stage_uses_muon": 0.0,
            },
            "per_axis_decomposition": {"pose": 2.0, "seg": 5.8},
        }
        for epoch in range(18_160, 18_166)
    ]
    telemetry.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    row = build_nerv_training_telemetry_feedback_row(
        telemetry_path=telemetry,
        family="hi_nerv",
        candidate_id="hinerv_np600_ld28_ed12_dc32_hfg_cnx_int4_mixed_ceil285000",
        candidate_num_pairs=600,
        stop_reason="training_running_midrun_feedback_snapshot",
    )

    assert row["training_stopped"] is False
    assert row["pr95_curriculum_observed"] is True
    assert row["pr95_current_stage_index"] == 5
    assert row["pr95_canonical_expected_stage_index"] == 5
    assert row["pr95_authoritative_stage_index"] == 5
    assert row["pr95_stage_mismatch_detected"] is False
    assert row["pr95_stage_uses_muon_current"] is False
    assert row["pr95_final_stage_reached"] is False
    assert row["pr95_final_stage_muon_expected_currently"] is False
    assert row["pr95_final_stage_muon_missing"] is False
    assert (
        row["optimizer_stage_assessment"]
        == "pr95_curriculum_pre_final_muon_not_expected"
    )
    assert row["pr95_stage_status"][
        "canonical_final_muon_stage_start_epoch"
    ] == 24_650
    assert row["pr95_stage_status"][
        "observed_stage_matches_canonical_epoch"
    ] is True
    assert "hi_nerv_pr95_final_stage_muon_missing_telemetry" not in row["blockers"]


def test_training_telemetry_feedback_blocks_stale_pr95_stage_before_final_muon(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "pr95_stale_stage5_at_final_epoch_telemetry.jsonl"
    rows = [
        {
            "epoch": epoch,
            "learning_rate": 2.7e-5,
            "loss_components": {
                "loss_part_pose_distill": 1.0,
                "loss_part_distill": 5.8,
                "pr95_stage_index": 5.0,
                "pr95_stage_uses_muon": 0.0,
            },
            "per_axis_decomposition": {"pose": 2.0, "seg": 5.8},
        }
        for epoch in range(24_650, 24_656)
    ]
    telemetry.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    row = build_nerv_training_telemetry_feedback_row(
        telemetry_path=telemetry,
        family="hi_nerv",
        candidate_id="hinerv_np600_ld28_ed12_dc32_hfg_cnx_int4_mixed_ceil285000",
        candidate_num_pairs=600,
        stop_reason="training_running_midrun_feedback_snapshot",
    )

    assert row["pr95_current_stage_index"] == 5
    assert row["pr95_canonical_expected_stage_index"] == 8
    assert row["pr95_authoritative_stage_index"] == 8
    assert row["pr95_stage_mismatch_detected"] is True
    assert row["pr95_final_stage_reached"] is True
    assert row["pr95_final_stage_muon_expected_currently"] is True
    assert row["pr95_final_stage_muon_missing"] is True
    assert row["pr95_stage_status"][
        "observed_stage_matches_canonical_epoch"
    ] is False
    assert row["optimizer_stage_assessment"] == "pr95_final_stage_muon_missing"
    assert "hi_nerv_pr95_stage_index_mismatch_telemetry" in row["blockers"]
    assert "hi_nerv_pr95_final_stage_muon_missing_telemetry" in row["blockers"]
    assert "fix_pr95_stage_telemetry_or_curriculum_epoch_routing" in row[
        "recommended_launch_mutations"
    ]
    assert "fix_pr95_final_stage_muon_optimizer_routing" in row[
        "recommended_launch_mutations"
    ]


def test_training_telemetry_feedback_flags_pr95_final_stage_without_muon(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "pr95_stage8_missing_muon_telemetry.jsonl"
    rows = [
        {
            "epoch": epoch,
            "learning_rate": 2.7e-5,
            "loss_components": {
                "loss_part_pose_distill": 1.0,
                "loss_part_distill": 5.8,
                "pr95_stage_index": 8.0,
                "pr95_stage_uses_muon": 0.0,
            },
            "per_axis_decomposition": {"pose": 2.0, "seg": 5.8},
        }
        for epoch in range(24_650, 24_656)
    ]
    telemetry.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    row = build_nerv_training_telemetry_feedback_row(
        telemetry_path=telemetry,
        family="hi_nerv",
        candidate_id="hinerv_np600_ld28_ed12_dc32_hfg_cnx_int4_mixed_ceil285000",
        candidate_num_pairs=600,
        stop_reason="training_running_midrun_feedback_snapshot",
    )

    assert row["pr95_final_stage_reached"] is True
    assert row["pr95_canonical_expected_stage_index"] == 8
    assert row["pr95_authoritative_stage_index"] == 8
    assert row["pr95_stage_mismatch_detected"] is False
    assert row["pr95_final_stage_muon_expected_currently"] is True
    assert row["pr95_final_stage_muon_missing"] is True
    assert row["optimizer_stage_assessment"] == "pr95_final_stage_muon_missing"
    assert row["pr95_stage_status"][
        "observed_stage_matches_canonical_epoch"
    ] is True
    assert "hi_nerv_pr95_final_stage_muon_missing_telemetry" in row["blockers"]
    assert "fix_pr95_final_stage_muon_optimizer_routing" in row[
        "recommended_launch_mutations"
    ]


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


def test_training_telemetry_harvester_marks_running_snapshot_explicitly() -> None:
    assert (
        _effective_stop_reason(stop_reason=None, training_running=True)
        == "training_running_midrun_feedback_snapshot"
    )
    assert (
        _effective_stop_reason(stop_reason="completed", training_running=True)
        == "completed"
    )
    assert _effective_stop_reason(stop_reason=None, training_running=False) is None


def test_harvest_training_telemetry_feedback_tool_output_json_is_guarded(
    tmp_path: Path,
    capsys,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 1,
                "learning_rate": 2.7e-5,
                "loss_components": {"loss_part_pose_distill": 1.0},
                "per_axis_decomposition": {"pose": 1.0, "seg": 6.0},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "harvest_manifest.json"
    argv = [
        "--telemetry",
        str(telemetry),
        "--family",
        "hi_nerv",
        "--candidate-id",
        "hinerv_np600_guarded_output",
        "--candidate-num-pairs",
        "600",
        "--current-segnet-distillation-weight",
        "2.0",
        "--training-running",
        "--output-dir",
        str(tmp_path / "feedback"),
        "--output-json",
        str(output_json),
    ]

    assert harvest_training_feedback_main(argv) == 0
    capsys.readouterr()
    assert output_json.is_file()
    manifest = json.loads(output_json.read_text(encoding="utf-8"))
    assert manifest["row"]["observed_segnet_distillation_weight"] == 2.0
    assert manifest["row"]["segnet_distillation_weight_source"] == (
        "harvest_current_segnet_distillation_weight"
    )
    with pytest.raises(ArtifactWriteError, match="refusing to overwrite"):
        harvest_training_feedback_main(argv)


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
