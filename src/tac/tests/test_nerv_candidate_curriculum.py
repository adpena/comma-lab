# SPDX-License-Identifier: MIT
"""Tests for candidate-conditioned NeRV curriculum planning."""

from __future__ import annotations

from tac.analysis.nerv_candidate_curriculum import (
    build_hinerv_candidate_curriculum_plan,
    build_snerv_candidate_curriculum_plan,
    strip_candidate_curriculum_authority_fields,
)
from tac.analysis.nerv_modelsize_budget import (
    analyze_hinerv_modelsize_candidate,
    analyze_snerv_modelsize_candidate,
)


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
        differentiable_pose_preprocess_attached=True,
        ema_archive_selection_attached=True,
        receiver_proof_attached=True,
        measured_archive_bytes=measured,
    )
    assert "hi_nerv_receiver_proof_missing" not in receiver_proven["blockers"]
    assert receiver_proven["scorer_pressure"]["receiver_proof_attached"] is True


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
    assert "snerv_mlx_native_train_export_adapter_missing" in plan["blockers"]
    assert plan["pr95_stack_binding"]["family"] == "snerv"
    assert plan["pr95_stack_binding"]["complete"] is False
    assert plan["long_campaign_prelaunch_gate"]["launch_allowed"] is False
    assert "snerv_archive_in_loop_byte_oracle_missing" not in plan["blockers"]
    assert "snerv_real_posenet_teacher_missing" in plan["blockers"]
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
    )

    assert plan["training_plan"]["scorer_loop_qat_attached"] is True
    assert plan["training_plan"]["scorer_loop_qat_receiver_contract_satisfied"] is True
    assert plan["training_plan"]["scorer_loop_qat_ready_for_pose_guard_gate"] is True
    assert plan["training_plan"]["scorer_loop_qat_accepted_improvement"] is True
    assert "snerv_scorer_loop_qat_not_attached" not in plan["blockers"]
    assert "snerv_scorer_loop_qat_receiver_contract_failed" not in plan["blockers"]
    assert "snerv_scorer_loop_qat_no_accepted_improvement" not in plan["blockers"]
    assert "snerv_real_segnet_teacher_missing" not in plan["blockers"]
    assert "snerv_real_posenet_teacher_missing" not in plan["blockers"]
    assert "snerv_qat_forward_missing" not in plan["blockers"]
    assert "snerv_coder_aware_regularizer_missing" not in plan["blockers"]
    assert "snerv_mlx_native_train_export_adapter_missing" in plan["blockers"]
    assert "snerv_score_aware_curriculum_not_native_mlx_yet" in plan["blockers"]


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
