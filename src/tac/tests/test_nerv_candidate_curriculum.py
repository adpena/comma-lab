# SPDX-License-Identifier: MIT
"""Tests for candidate-conditioned NeRV curriculum planning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tac.analysis.nerv_candidate_curriculum import (
    build_hinerv_candidate_curriculum_plan,
    build_snerv_candidate_curriculum_plan,
    strip_candidate_curriculum_authority_fields,
)
from tac.analysis.nerv_modelsize_budget import (
    analyze_hinerv_modelsize_candidate,
    analyze_snerv_modelsize_candidate,
)


def _snerv_native_artifact(tmp_path: Path, *, num_pairs: int = 600) -> dict[str, object]:
    report = tmp_path / "snerv_native_report.json"
    packet = tmp_path / "snerv_native_packet.snar"
    archive = tmp_path / "snerv_native_archive.zip"
    proof = tmp_path / "snerv_native_receiver_proof.json"
    report.write_text(
        '{"schema":"snerv_mlx_native_train_export.v1"}\n',
        encoding="utf-8",
    )
    packet.write_bytes(b"snerv packet")
    archive.write_bytes(b"snerv archive")
    proof.write_text(
        json.dumps(
            {
                "schema": "snerv_receiver_proof.v1",
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "num_pairs": int(num_pairs),
        "executed": True,
        "artifact_report_path": report.as_posix(),
        "packet_path": packet.as_posix(),
        "packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
        "archive_path": archive.as_posix(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "receiver_proof_path": proof.as_posix(),
        "receiver_proof_passed": True,
        "receiver_contract_satisfied": True,
    }


def test_hinerv_candidate_curriculum_enables_lowbit_qat_and_blocks_missing_scorers() -> None:
    candidate = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        latent_dim=12,
        embed_dim=24,
        decoder_channel=32,
        decoder_codec="int4_mixed",
    ).as_dict()

    plan = build_hinerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=4,
        num_pairs=32,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        coder_aware_qat=False,
        coder_qat_quant_bits=8,
        recon_pixel_weight_attached=False,
    )

    assert plan["schema"] == "nerv_candidate_curriculum_plan.v1"
    assert plan["family"] == "hi_nerv"
    assert plan["candidate_conditioned"] is True
    assert plan["pr95_stage_plan"]["enabled"] is False
    assert plan["coder_pressure"] == {
        "enabled": True,
        "regularizer_enabled": True,
        "fake_quant_forward_enabled": True,
        "quant_bits": 4,
        "candidate_decoder_codec": "int4_mixed",
        "candidate_decoder_codec_bits": 4,
        "source": "modelsize_candidate",
        "implementation_status": (
            "decoder_weight_fake_quant_forward_plus_quant_residual_regularizer"
        ),
    }
    assert "enabled_decoder_coder_regularizer_from_modelsize_candidate" in plan[
        "launch_mutations"
    ]
    assert "aligned_coder_qat_quant_bits_to_candidate_codec" in plan[
        "launch_mutations"
    ]
    assert "hinerv_candidate_curriculum_requires_min_8_epochs" in plan["blockers"]
    assert "hinerv_candidate_curriculum_requires_real_segnet_teacher" in plan[
        "blockers"
    ]
    assert "hinerv_candidate_curriculum_requires_real_posenet_teacher" in plan[
        "blockers"
    ]
    assert "hinerv_candidate_curriculum_recon_pixel_weight_missing" in plan[
        "blockers"
    ]
    assert "hinerv_candidate_curriculum_full600_required_for_promotion" in plan[
        "blockers"
    ]
    assert "hinerv_trained_archive_byte_oracle_feedback_missing" in plan["blockers"]
    assert plan["pr95_stack_binding"]["schema"] == (
        "pr95_stack_binding_requirements.v1"
    )
    assert plan["pr95_stack_binding"]["complete"] is False
    assert plan["long_campaign_prelaunch_gate"]["schema"] == (
        "pr95_stack_binding_long_campaign_prelaunch_gate.v1"
    )
    assert plan["long_campaign_prelaunch_gate"]["launch_allowed"] is False
    assert "hi_nerv_real_segnet_teacher_missing" in plan["blockers"]
    assert "hi_nerv_eval_roundtrip_ste_missing" in plan["blockers"]
    assert "hi_nerv_differentiable_pose_preprocess_missing" in plan["blockers"]
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False


def test_hinerv_candidate_curriculum_records_measured_archive_byte_feedback() -> None:
    candidate = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        latent_dim=12,
        embed_dim=24,
        decoder_channel=32,
        decoder_codec="int4_mixed",
    ).as_dict()
    measured = int(candidate["nominal_total_payload_bytes"]) + 123

    plan = build_hinerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=8,
        num_pairs=600,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        recon_pixel_weight_attached=True,
        eval_roundtrip_ste_attached=True,
        scorer_input_distribution_guard_attached=True,
        differentiable_pose_preprocess_attached=True,
        ema_archive_selection_attached=True,
        measured_archive_bytes=measured,
    )

    feedback = plan["byte_oracle_logging"]
    assert feedback["candidate_num_pairs"] == 600
    assert feedback["measured_num_pairs"] == 600
    assert feedback["feedback_scope"] == "candidate_full_scope"
    assert feedback["scope_matches_candidate"] is True
    assert feedback["feedback_ready"] is True
    assert feedback["measured_archive_bytes"] == measured
    assert feedback["measured_minus_nominal_bytes"] == 123
    assert "hinerv_trained_archive_byte_oracle_feedback_missing" not in plan[
        "blockers"
    ]
    assert "hinerv_candidate_curriculum_requires_min_8_epochs" not in plan[
        "blockers"
    ]
    assert plan["pr95_stack_binding"]["rows"][1]["requirement_id"] == (
        "modelsize_archive_budget"
    )
    assert plan["pr95_stack_binding"]["rows"][1]["satisfied"] is True
    assert "hi_nerv_archive_in_loop_byte_oracle_missing" not in plan["blockers"]
    assert "hi_nerv_receiver_proof_missing" in plan["blockers"]
    assert plan["long_campaign_prelaunch_gate"]["launch_allowed"] is True
    assert "hi_nerv_eval_roundtrip_ste_missing" not in plan[
        "long_campaign_prelaunch_gate"
    ]["blockers"]
    assert "hi_nerv_differentiable_pose_preprocess_missing" not in plan[
        "long_campaign_prelaunch_gate"
    ]["blockers"]
    assert plan["scorer_pressure"]["eval_roundtrip_ste_attached"] is True
    assert plan["scorer_pressure"][
        "differentiable_pose_preprocess_attached"
    ] is True
    assert plan["scorer_pressure"]["ema_archive_selection_attached"] is True
    assert "hi_nerv_ema_archive_selection_missing" not in plan[
        "long_campaign_prelaunch_gate"
    ]["blockers"]
    assert "hi_nerv_receiver_proof_missing" not in plan[
        "long_campaign_prelaunch_gate"
    ]["blockers"]
    assert "hi_nerv_receiver_proof_missing" in plan["blockers"]

    receiver_proven = build_hinerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=8,
        num_pairs=600,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        recon_pixel_weight_attached=True,
        eval_roundtrip_ste_attached=True,
        scorer_input_distribution_guard_attached=True,
        differentiable_pose_preprocess_attached=True,
        ema_archive_selection_attached=True,
        receiver_proof_attached=True,
        measured_archive_bytes=measured,
    )
    assert "hi_nerv_receiver_proof_missing" not in receiver_proven["blockers"]
    assert receiver_proven["scorer_pressure"]["receiver_proof_attached"] is True


def test_hinerv_candidate_curriculum_uses_explicit_optimizer_binding_evidence() -> None:
    candidate = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        latent_dim=12,
        embed_dim=24,
        decoder_channel=32,
        decoder_codec="int4_mixed",
    ).as_dict()
    measured = int(candidate["nominal_total_payload_bytes"])

    native_optimizer_plan = build_hinerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=29_650,
        num_pairs=600,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        recon_pixel_weight_attached=True,
        eval_roundtrip_ste_attached=True,
        scorer_input_distribution_guard_attached=True,
        differentiable_pose_preprocess_attached=True,
        ema_archive_selection_attached=True,
        pr95_staged_curriculum_bound=False,
        muon_adamw_partition_bound=True,
        measured_archive_bytes=measured,
    )

    assert native_optimizer_plan["pr95_stage_plan"]["requested_epochs"] == 29_650
    assert native_optimizer_plan["pr95_stage_plan"]["enabled"] is False
    assert native_optimizer_plan["pr95_stage_plan"]["evidence_source"] == (
        "explicit_runner_optimizer_policy"
    )
    assert native_optimizer_plan["pr95_stage_plan"][
        "muon_adamw_partition_bound"
    ] is True
    assert "hi_nerv_pr95_staged_curriculum_missing" in native_optimizer_plan[
        "blockers"
    ]
    assert "hi_nerv_muon_adamw_partition_missing" not in native_optimizer_plan[
        "blockers"
    ]
    assert native_optimizer_plan["long_campaign_prelaunch_gate"][
        "launch_allowed"
    ] is False

    pr95_bound_plan = build_hinerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=29_650,
        num_pairs=600,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        recon_pixel_weight_attached=True,
        eval_roundtrip_ste_attached=True,
        scorer_input_distribution_guard_attached=True,
        differentiable_pose_preprocess_attached=True,
        ema_archive_selection_attached=True,
        pr95_staged_curriculum_bound=True,
        muon_adamw_partition_bound=True,
        measured_archive_bytes=measured,
    )

    assert pr95_bound_plan["pr95_stage_plan"]["enabled"] is True
    assert "hi_nerv_pr95_staged_curriculum_missing" not in pr95_bound_plan[
        "blockers"
    ]
    assert "hi_nerv_muon_adamw_partition_missing" not in pr95_bound_plan[
        "blockers"
    ]
    assert pr95_bound_plan["long_campaign_prelaunch_gate"][
        "launch_allowed"
    ] is True


def test_hinerv_candidate_curriculum_harvests_partial_bytes_without_readiness() -> None:
    candidate = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        latent_dim=12,
        embed_dim=24,
        decoder_channel=32,
        decoder_codec="int4_mixed",
    ).as_dict()
    measured = int(candidate["nominal_total_payload_bytes"]) - 512

    plan = build_hinerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=8,
        num_pairs=32,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        recon_pixel_weight_attached=True,
        measured_archive_bytes=measured,
    )

    feedback = plan["byte_oracle_logging"]
    assert feedback["candidate_num_pairs"] == 600
    assert feedback["measured_num_pairs"] == 32
    assert feedback["feedback_scope"] == "partial_pair_advisory"
    assert feedback["scope_matches_candidate"] is False
    assert feedback["feedback_ready"] is False
    assert feedback["measured_archive_bytes"] == measured
    assert feedback["measured_minus_nominal_bytes"] == -512
    assert "partial_pair_byte_feedback_only" in plan["blockers"]
    assert "hinerv_trained_archive_byte_oracle_feedback_missing" not in plan[
        "blockers"
    ]


def test_hinerv_candidate_curriculum_keeps_partial_feedback_scope_on_full600_plan() -> None:
    candidate = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        latent_dim=12,
        embed_dim=24,
        decoder_channel=32,
        decoder_codec="int4_mixed",
    ).as_dict()

    plan = build_hinerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=8,
        num_pairs=600,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        recon_pixel_weight_attached=True,
        measured_archive_bytes=57_892,
        measured_num_pairs=2,
    )

    feedback = plan["byte_oracle_logging"]
    assert feedback["candidate_num_pairs"] == 600
    assert feedback["measured_num_pairs"] == 2
    assert feedback["feedback_scope"] == "partial_pair_advisory"
    assert feedback["scope_matches_candidate"] is False
    assert feedback["feedback_ready"] is False
    assert "partial_pair_byte_feedback_only" in plan["blockers"]
    assert "hinerv_trained_archive_byte_oracle_feedback_missing" not in plan[
        "blockers"
    ]
    assert "hi_nerv_archive_in_loop_byte_oracle_missing" in plan["blockers"]


def test_snerv_candidate_curriculum_records_snar1_byte_feedback() -> None:
    candidate = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=216_000,
        num_pairs=600,
        levels=5,
        bits_per_coeff=1.5,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
    ).as_dict()

    plan = build_snerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=0,
        num_pairs=600,
        step_map_coder_mode="waterfill",
        measured_packet_bytes=190_000,
        measured_archive_bytes=191_000,
    )

    assert plan["family"] == "snerv"
    assert plan["receiver_grammar_controls"] == {
        "levels": 5,
        "lf_bits_per_coeff": 1.5,
        "step_map_bits_per_coeff": 0.5,
        "step_map_coder_mode": "waterfill",
        "decoder_payload_codec": "int8_symmetric",
    }
    assert plan["byte_oracle_logging"]["feedback_ready"] is True
    assert plan["byte_oracle_logging"]["candidate_num_pairs"] == 600
    assert plan["byte_oracle_logging"]["measured_num_pairs"] == 600
    assert plan["byte_oracle_logging"]["feedback_scope"] == "candidate_full_scope"
    assert plan["byte_oracle_logging"]["scope_matches_candidate"] is True
    assert plan["byte_oracle_logging"]["measured_payload_bytes"] == 190_000
    assert plan["byte_oracle_logging"]["measured_archive_bytes"] == 191_000
    assert "snerv_snar1_byte_feedback_missing" not in plan["blockers"]
    assert "snerv_candidate_curriculum_requires_waterfill_step_maps" not in plan[
        "blockers"
    ]
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in plan[
        "blockers"
    ]
    assert plan["training_plan"]["attachment_authority_contract"] == {
        "schema": "snerv_training_attachment_authority_contract.v1",
        "planned_surfaces_are_not_receiver_or_score_authority": True,
        "queue_ready_is_not_receiver_or_exact_authority": True,
        "receiver_authority_requires_file_backed_export_and_replay": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    assert plan["training_plan"]["native_mlx_adapter_surfaces_ready"] is True
    assert plan["training_plan"]["native_mlx_adapter_full600_campaign_ready"] is False
    assert plan["training_plan"]["current_execution_path"] == (
        "cpu_advisory_receiver_bound_packet"
    )
    assert plan["training_plan"]["next_required_adapter"] == (
        "snerv_learned_scoreaware_mlx_training_loop_bound_to_native_export"
    )
    assert plan["training_plan"]["native_mlx_train_export_planned"] is False
    assert plan["training_plan"]["native_mlx_train_export_verified"] is False
    assert (
        plan["training_plan"]["native_mlx_export_attachment_is_learned_training"]
        is False
    )
    assert (
        plan["training_plan"]["learned_scoreaware_mlx_training_required_next"]
        is True
    )
    assert plan["training_plan"]["native_mlx_scorer_loop_qat_planned"] is False
    assert plan["training_plan"]["native_mlx_scorer_loop_qat_verified"] is False
    assert plan["pr95_stack_binding"]["family"] == "snerv"
    assert plan["pr95_stack_binding"]["complete"] is False
    assert plan["long_campaign_prelaunch_gate"]["launch_allowed"] is False
    assert "snerv_archive_in_loop_byte_oracle_missing" not in plan["blockers"]
    assert "snerv_real_posenet_teacher_missing" in plan["blockers"]
    assert "snerv_receiver_proof_missing" in plan["blockers"]
    assert "snerv_scorer_loop_qat_not_attached" in plan["blockers"]
    assert plan["score_claim"] is False


def test_snerv_candidate_curriculum_consumes_scorer_loop_qat_evidence() -> None:
    candidate = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=216_000,
        num_pairs=600,
        levels=5,
        bits_per_coeff=1.5,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
    ).as_dict()

    plan = build_snerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=3,
        num_pairs=600,
        step_map_coder_mode="waterfill",
        measured_packet_bytes=190_000,
        measured_archive_bytes=191_000,
        scorer_loop_qat_attached=True,
        scorer_loop_qat_receiver_contract_satisfied=True,
        scorer_loop_qat_ready_for_pose_guard_gate=True,
        scorer_loop_qat_accepted_improvement=True,
        receiver_proof_attached=True,
        full_video_local_prefilter_attached=True,
        local_cpu_replay_gate_attached=True,
    )

    assert plan["training_plan"]["scorer_loop_qat_attached"] is True
    assert plan["training_plan"]["scorer_loop_qat_receiver_contract_satisfied"] is True
    assert plan["training_plan"]["scorer_loop_qat_ready_for_pose_guard_gate"] is True
    assert plan["training_plan"]["scorer_loop_qat_accepted_improvement"] is True
    assert plan["training_plan"]["standalone_scorer_loop_qat_verified"] is False
    assert (
        plan["training_plan"][
            "standalone_scorer_loop_qat_requires_receiver_packet_materialization"
        ]
        is True
    )
    assert plan["training_plan"]["native_mlx_scorer_loop_qat_planned"] is False
    assert plan["training_plan"]["native_mlx_scorer_loop_qat_verified"] is False
    assert plan["training_plan"]["receiver_proof_attached"] is True
    assert plan["training_plan"]["full_video_local_prefilter_attached"] is True
    assert plan["training_plan"]["local_cpu_replay_gate_attached"] is True
    assert "snerv_scorer_loop_qat_not_attached" not in plan["blockers"]
    assert "snerv_scorer_loop_qat_receiver_contract_failed" not in plan["blockers"]
    assert "snerv_scorer_loop_qat_no_accepted_improvement" not in plan["blockers"]
    assert "snerv_real_segnet_teacher_missing" in plan["blockers"]
    assert "snerv_real_posenet_teacher_missing" in plan["blockers"]
    assert "snerv_qat_forward_missing" in plan["blockers"]
    assert "snerv_coder_aware_regularizer_missing" in plan["blockers"]
    assert plan["pr95_stack_binding"]["complete"] is False
    assert "snerv_receiver_proof_missing" not in plan["blockers"]
    assert "snerv_full_video_local_prefilter_missing" not in plan["blockers"]
    assert "snerv_local_cpu_replay_gate_missing" not in plan["blockers"]
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in plan[
        "blockers"
    ]
    assert plan["training_plan"]["native_mlx_adapter_surfaces_ready"] is True
    assert (
        "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only"
        in plan["blockers"]
    )


def test_snerv_candidate_curriculum_consumes_native_mlx_export_evidence(
    tmp_path: Path,
) -> None:
    candidate = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=216_000,
        num_pairs=600,
        levels=5,
        bits_per_coeff=1.5,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
    ).as_dict()

    report = tmp_path / "report.json"
    packet = tmp_path / "candidate.snar"
    archive = tmp_path / "archive.zip"
    proof = tmp_path / "receiver_proof.json"
    report.write_text('{"schema":"unit_report"}\n', encoding="utf-8")
    packet.write_bytes(b"packet")
    archive.write_bytes(b"archive")
    proof.write_text(
        json.dumps(
            {
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
            }
        ),
        encoding="utf-8",
    )

    plan = build_snerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=3,
        num_pairs=600,
        step_map_coder_mode="waterfill",
        measured_packet_bytes=190_000,
        measured_archive_bytes=191_000,
        native_mlx_train_export_attached=True,
        native_mlx_receiver_proof_passed=True,
        native_mlx_full600_campaign_ready=True,
        native_mlx_artifact_evidence={
            "num_pairs": 600,
            "artifact_report_path": report.as_posix(),
            "packet_path": packet.as_posix(),
            "packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
            "archive_path": archive.as_posix(),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "receiver_proof_path": proof.as_posix(),
        },
    )

    training_plan = plan["training_plan"]
    authority_contract = training_plan["attachment_authority_contract"]
    assert authority_contract["schema"] == (
        "snerv_training_attachment_authority_contract.v1"
    )
    assert authority_contract["planned_surfaces_are_not_receiver_or_score_authority"] is True
    assert authority_contract["queue_ready_is_not_receiver_or_exact_authority"] is True
    assert training_plan["native_mlx_train_export_attached"] is True
    assert training_plan["native_mlx_train_export_planned"] is True
    assert training_plan["current_execution_path"] == (
        "cpu_advisory_plus_mlx_native_export_attachment"
    )
    assert training_plan["next_required_adapter"] == (
        "snerv_learned_scoreaware_mlx_training_loop_bound_to_native_export"
    )
    assert training_plan["native_mlx_train_export_verified"] is True
    assert training_plan["native_mlx_export_attachment_is_learned_training"] is False
    assert training_plan["learned_scoreaware_mlx_training_required_next"] is True
    assert training_plan["native_mlx_receiver_proof_passed"] is True
    assert training_plan["native_mlx_file_backed_export_proof_passed"] is True
    assert training_plan["native_mlx_export_verified"] is True
    assert training_plan["native_mlx_export_full600_campaign_ready"] is True
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" not in plan[
        "blockers"
    ]
    assert "snerv_mlx_native_full600_campaign_not_ready" not in plan["blockers"]
    assert (
        "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only"
        in plan["blockers"]
    )
    assert "snerv_scorer_loop_qat_not_attached" in plan["blockers"]


def test_snerv_candidate_curriculum_consumes_native_mlx_scorer_loop_without_overclaim() -> None:
    candidate = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=216_000,
        num_pairs=600,
        levels=5,
        bits_per_coeff=1.5,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
    ).as_dict()

    plan = build_snerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=3,
        num_pairs=600,
        step_map_coder_mode="waterfill",
        measured_packet_bytes=190_000,
        measured_archive_bytes=191_000,
        native_mlx_train_export_attached=True,
        native_mlx_receiver_proof_passed=True,
        native_mlx_full600_campaign_ready=True,
        native_mlx_scorer_loop_qat_attached=True,
        native_mlx_scorer_loop_qat_receiver_contract_satisfied=True,
        native_mlx_scorer_loop_qat_ready_for_pose_guard_gate=True,
        native_mlx_scorer_loop_qat_accepted_improvement=True,
        native_mlx_scorer_loop_qat_best_materialized=False,
    )

    training_plan = plan["training_plan"]
    assert training_plan["scorer_loop_qat_attached"] is True
    assert training_plan["standalone_scorer_loop_qat_attached"] is False
    assert training_plan["native_mlx_scorer_loop_qat_attached"] is True
    assert training_plan["native_mlx_scorer_loop_qat_planned"] is True
    assert training_plan["native_mlx_scorer_loop_qat_verified"] is False
    assert (
        training_plan["native_mlx_scorer_loop_qat_receiver_contract_satisfied"]
        is True
    )
    assert training_plan["scorer_loop_qat_receiver_contract_satisfied"] is True
    assert training_plan["scorer_loop_qat_ready_for_pose_guard_gate"] is True
    assert training_plan["scorer_loop_qat_accepted_improvement"] is True
    assert "snerv_scorer_loop_qat_not_attached" not in plan["blockers"]
    assert "snerv_scorer_loop_qat_receiver_contract_failed" not in plan["blockers"]
    assert "snerv_scorer_loop_qat_no_accepted_improvement" not in plan["blockers"]
    assert "snerv_real_segnet_teacher_missing" in plan["blockers"]
    assert "snerv_real_posenet_teacher_missing" in plan["blockers"]
    assert "snerv_qat_forward_missing" in plan["blockers"]
    assert "snerv_coder_aware_regularizer_missing" in plan["blockers"]
    assert "snerv_native_scorer_loop_best_packet_not_materialized" in plan[
        "blockers"
    ]
    assert plan["pr95_stack_binding"]["complete"] is False
    assert (
        "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only"
        in plan["blockers"]
    )
    assert plan["score_claim"] is False


def test_snerv_candidate_curriculum_promotes_file_backed_scorer_loop_to_long_training_ready(
    tmp_path: Path,
) -> None:
    candidate = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=216_000,
        num_pairs=600,
        levels=5,
        bits_per_coeff=1.5,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
    ).as_dict()

    report = tmp_path / "native_export.json"
    packet = tmp_path / "native_packet.snar"
    archive = tmp_path / "archive.zip"
    proof = tmp_path / "receiver_proof.json"
    report.write_text('{"schema":"snerv_native_export_unit"}\n', encoding="utf-8")
    packet.write_bytes(b"packet")
    archive.write_bytes(b"archive")
    proof.write_text(
        json.dumps(
            {
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    plan = build_snerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=29_650,
        num_pairs=600,
        step_map_coder_mode="waterfill",
        measured_packet_bytes=190_000,
        measured_archive_bytes=191_000,
        native_mlx_train_export_attached=True,
        native_mlx_receiver_proof_passed=True,
        native_mlx_full600_campaign_ready=True,
        native_mlx_scorer_loop_qat_attached=True,
        native_mlx_scorer_loop_qat_receiver_contract_satisfied=True,
        native_mlx_scorer_loop_qat_ready_for_pose_guard_gate=True,
        native_mlx_scorer_loop_qat_accepted_improvement=True,
        native_mlx_scorer_loop_qat_best_materialized=True,
        native_mlx_artifact_evidence={
            "num_pairs": 600,
            "artifact_report_path": report.as_posix(),
            "packet_path": packet.as_posix(),
            "packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
            "archive_path": archive.as_posix(),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "receiver_proof_path": proof.as_posix(),
            "scorer_loop_qat": {
                "executed": True,
                "receiver_contract_satisfied": True,
                "ready_for_pose_guard_gate": True,
                "accepted_improvement": True,
                "emitted_packet_uses_scorer_loop_best_decoder": True,
            },
        },
    )

    training_plan = plan["training_plan"]
    assert training_plan["native_mlx_train_export_verified"] is True
    assert training_plan["native_mlx_scorer_loop_qat_verified"] is True
    assert (
        training_plan["native_mlx_scorer_loop_file_backed_long_training_ready"]
        is True
    )
    assert training_plan["learned_scoreaware_mlx_training_required_next"] is False
    assert (
        "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only"
        not in plan["blockers"]
    )
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False


def test_snerv_candidate_curriculum_blocks_receiver_proven_over_ceiling_lf_grammar() -> None:
    candidate = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=216_000,
        num_pairs=600,
        levels=5,
        bits_per_coeff=1.5,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
    ).as_dict()

    plan = build_snerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=29_650,
        num_pairs=600,
        step_map_coder_mode="waterfill",
        measured_packet_bytes=2_347_476,
        measured_archive_bytes=444_828,
        native_mlx_train_export_attached=True,
        native_mlx_long_training_bound=True,
        native_mlx_receiver_proof_passed=True,
        native_mlx_full600_campaign_ready=True,
        native_mlx_scorer_loop_qat_attached=True,
        native_mlx_scorer_loop_qat_receiver_contract_satisfied=True,
        native_mlx_scorer_loop_qat_ready_for_pose_guard_gate=True,
        native_mlx_scorer_loop_qat_accepted_improvement=True,
        native_mlx_scorer_loop_qat_best_materialized=True,
        native_mlx_real_segnet_teacher_bound=True,
        native_mlx_real_posenet_teacher_bound=True,
        native_mlx_pr95_curriculum_bound=True,
        native_mlx_eval_roundtrip_ste_bound=True,
        native_mlx_differentiable_pose_preprocess_bound=True,
        native_mlx_coder_qat_bound=True,
        native_mlx_muon_adamw_partition_bound=True,
        receiver_proof_attached=True,
        full_video_local_prefilter_attached=True,
        local_cpu_replay_gate_attached=True,
    )

    feedback = plan["byte_oracle_logging"]
    assert feedback["feedback_ready"] is True
    assert feedback["archive_under_hard_byte_ceiling"] is False
    assert feedback["archive_over_hard_byte_ceiling_bytes"] == 228_828
    assert feedback["rate_axis_feedback_verdict"] == (
        "receiver_proven_archive_over_hard_byte_ceiling"
    )
    assert "snerv_receiver_proven_archive_over_hard_byte_ceiling" in plan[
        "blockers"
    ]
    assert (
        "snerv_over_ceiling_local_lf_grammar_reroute_to_official_packet_or_lf_recode"
        in plan["blockers"]
    )


def test_snerv_candidate_curriculum_harvests_partial_bytes_without_readiness() -> None:
    candidate = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=216_000,
        num_pairs=600,
        levels=5,
        bits_per_coeff=1.5,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
    ).as_dict()

    plan = build_snerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=0,
        num_pairs=128,
        step_map_coder_mode="waterfill",
        measured_packet_bytes=12_000,
        measured_archive_bytes=13_000,
    )

    feedback = plan["byte_oracle_logging"]
    assert feedback["candidate_num_pairs"] == 600
    assert feedback["measured_num_pairs"] == 128
    assert feedback["feedback_scope"] == "partial_pair_advisory"
    assert feedback["scope_matches_candidate"] is False
    assert feedback["feedback_ready"] is False
    assert feedback["measured_payload_bytes"] == 12_000
    assert feedback["measured_archive_bytes"] == 13_000
    assert "partial_pair_byte_feedback_only" in plan["blockers"]
    assert "snerv_snar1_byte_feedback_missing" not in plan["blockers"]


def test_snerv_candidate_curriculum_keeps_partial_feedback_scope_on_full600_plan() -> None:
    candidate = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=216_000,
        num_pairs=600,
        levels=5,
        bits_per_coeff=1.5,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
    ).as_dict()

    plan = build_snerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=3,
        num_pairs=600,
        step_map_coder_mode="waterfill",
        measured_packet_bytes=10_441,
        measured_archive_bytes=57_892,
        measured_num_pairs=2,
        native_mlx_train_export_attached=True,
        native_mlx_receiver_proof_passed=True,
        native_mlx_scorer_loop_qat_attached=True,
        native_mlx_scorer_loop_qat_receiver_contract_satisfied=True,
        native_mlx_scorer_loop_qat_ready_for_pose_guard_gate=True,
        native_mlx_scorer_loop_qat_accepted_improvement=True,
        native_mlx_scorer_loop_qat_best_materialized=True,
    )

    feedback = plan["byte_oracle_logging"]
    assert feedback["candidate_num_pairs"] == 600
    assert feedback["measured_num_pairs"] == 2
    assert feedback["feedback_scope"] == "partial_pair_advisory"
    assert feedback["scope_matches_candidate"] is False
    assert feedback["feedback_ready"] is False
    assert "partial_pair_byte_feedback_only" in plan["blockers"]
    assert "snerv_snar1_byte_feedback_missing" not in plan["blockers"]
    assert "snerv_archive_in_loop_byte_oracle_missing" in plan["blockers"]
    assert plan["training_plan"]["native_mlx_train_export_planned"] is True
    assert plan["training_plan"]["native_mlx_train_export_verified"] is False
    assert plan["training_plan"]["native_mlx_scorer_loop_qat_planned"] is True
    assert plan["training_plan"]["native_mlx_scorer_loop_qat_verified"] is True
    assert "snerv_native_scorer_loop_best_packet_not_materialized" not in plan[
        "blockers"
    ]


def test_snerv_candidate_curriculum_blocks_non_waterfill_and_partial_coverage() -> None:
    plan = build_snerv_candidate_curriculum_plan(
        candidate=None,
        requested_epochs=0,
        num_pairs=32,
        step_map_coder_mode="uniform",
    )

    assert "snerv_modelsize_candidate_not_selected_manual_probe" in plan["blockers"]
    assert "snerv_candidate_curriculum_full600_required_for_promotion" in plan[
        "blockers"
    ]
    assert "snerv_candidate_curriculum_requires_waterfill_step_maps" in plan[
        "blockers"
    ]
    assert plan["byte_oracle_logging"]["feedback_ready"] is False


def test_candidate_curriculum_metadata_sanitizer_removes_authority_fields() -> None:
    plan = build_snerv_candidate_curriculum_plan(
        candidate=None,
        requested_epochs=0,
        num_pairs=1,
        step_map_coder_mode="uniform",
    )

    sanitized = strip_candidate_curriculum_authority_fields(plan)

    assert "score_claim" not in sanitized
    assert "promotion_eligible" not in sanitized
    assert "ready_for_exact_eval_dispatch" not in sanitized
    assert "score_claim" not in sanitized["byte_oracle_logging"]
    assert "score_claim" not in sanitized["pr95_stack_binding"]
    assert "promotion_eligible" not in sanitized["pr95_stack_binding"]
    assert "score_claim" not in sanitized["long_campaign_prelaunch_gate"]
    assert sanitized["schema"] == "nerv_candidate_curriculum_plan.v1"
    assert sanitized["byte_oracle_logging"]["schema"] == (
        "nerv_candidate_byte_feedback.v1"
    )
