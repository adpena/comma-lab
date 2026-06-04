# SPDX-License-Identifier: MIT
"""Regression tests for NeRV candidate curriculum planning.

These tests pin the PR95 binding semantics that gate long HiNeRV/SNeRV
campaigns. HiNeRV now has a real decoder-weight fake-quant forward path, but
that does not grant launch authority until the remaining PR95 scorer/eval/EMA
bindings are present.
"""

from __future__ import annotations

from typing import Any

from tac.analysis.nerv_candidate_curriculum import (
    build_hinerv_candidate_curriculum_plan,
    build_snerv_candidate_curriculum_plan,
)


def _candidate() -> dict[str, Any]:
    return {
        "candidate_id": "hi_nerv_full600_ceil180k_int4_smoke",
        "num_pairs": 600,
        "hard_byte_ceiling": 180_000,
        "nominal_total_payload_bytes": 150_000,
        "latent_dim": 8,
        "embed_dim": 8,
        "decoder_channel": 8,
        "decoder_codec": "int4",
    }


def _snerv_candidate(*, official: bool = False) -> dict[str, Any]:
    row = {
        "candidate_id": "snerv_full600_ceil180k",
        "num_pairs": 600,
        "hard_byte_ceiling": 180_000,
        "nominal_total_payload_bytes": 160_000,
        "levels": 3,
        "bits_per_coeff": 2.5,
        "step_map_bits_per_coeff": 4.0,
        "decoder_payload_codec": "snar1",
    }
    if official:
        row["snerv_model_size_adapter"] = (
            "snerv_official_mfu_hfr_tub_numeric_primitives_v1"
        )
    return row


def _requirement(plan: dict[str, Any], requirement_id: str) -> dict[str, Any]:
    rows = plan["pr95_stack_binding"]["rows"]
    for row in rows:
        if row["requirement_id"] == requirement_id:
            return row
    raise AssertionError(f"missing requirement row {requirement_id!r}")


def test_hinerv_modelsize_candidate_enables_decoder_fake_quant_forward_qat() -> None:
    plan = build_hinerv_candidate_curriculum_plan(
        candidate=_candidate(),
        requested_epochs=8,
        num_pairs=600,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        coder_aware_qat=False,
        coder_qat_quant_bits=8,
        recon_pixel_weight_attached=True,
    )

    assert "enabled_decoder_coder_regularizer_from_modelsize_candidate" in plan[
        "launch_mutations"
    ]
    assert plan["coder_pressure"]["regularizer_enabled"] is True
    assert plan["coder_pressure"]["fake_quant_forward_enabled"] is True
    assert plan["coder_pressure"]["quant_bits"] == 4

    assert _requirement(plan, "coder_aware_regularizer")["satisfied"] is True
    assert _requirement(plan, "qat_forward")["satisfied"] is True
    assert "hi_nerv_qat_forward_missing" not in plan[
        "long_campaign_prelaunch_gate"
    ]["blockers"]


def test_hinerv_prelaunch_gate_keeps_true_unimplemented_pr95_pieces_blocking() -> None:
    plan = build_hinerv_candidate_curriculum_plan(
        candidate=_candidate(),
        requested_epochs=8,
        num_pairs=600,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        recon_pixel_weight_attached=True,
    )

    gate = plan["long_campaign_prelaunch_gate"]
    assert gate["launch_allowed"] is False
    assert set(gate["blocking_requirement_ids"]) >= {
        "differentiable_pose_preprocess",
        "eval_roundtrip_ste",
        "ema_archive_selection",
    }
    assert _requirement(plan, "real_segnet_teacher")["satisfied"] is True
    assert _requirement(plan, "real_posenet_teacher")["satisfied"] is True
    assert _requirement(plan, "muon_adamw_partition")["satisfied"] is True


def test_hinerv_prelaunch_gate_consumes_eval_roundtrip_ste_evidence() -> None:
    plan = build_hinerv_candidate_curriculum_plan(
        candidate=_candidate(),
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
    )

    gate = plan["long_campaign_prelaunch_gate"]
    assert _requirement(plan, "eval_roundtrip_ste")["satisfied"] is True
    assert _requirement(plan, "differentiable_pose_preprocess")["satisfied"] is True
    assert "hi_nerv_eval_roundtrip_ste_missing" not in gate["blockers"]
    assert "hi_nerv_eval_roundtrip_ste_missing" not in plan["blockers"]
    assert "hi_nerv_differentiable_pose_preprocess_missing" not in gate["blockers"]
    assert "hi_nerv_differentiable_pose_preprocess_missing" not in plan["blockers"]
    assert _requirement(plan, "ema_archive_selection")["satisfied"] is True
    assert "hi_nerv_ema_archive_selection_missing" not in gate["blockers"]
    assert gate["launch_allowed"] is True


def test_hinerv_partial_pair_byte_feedback_remains_advisory_only() -> None:
    plan = build_hinerv_candidate_curriculum_plan(
        candidate=_candidate(),
        requested_epochs=8,
        num_pairs=32,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        recon_pixel_weight_attached=True,
        measured_archive_bytes=120_000,
    )

    feedback = plan["byte_oracle_logging"]
    assert feedback["feedback_scope"] == "partial_pair_advisory"
    assert feedback["feedback_ready"] is False
    assert "partial_pair_byte_feedback_only" in plan["blockers"]
    assert _requirement(plan, "archive_in_loop_byte_oracle")["satisfied"] is False


def test_snerv_prelaunch_stays_blocked_until_native_mlx_training_exists() -> None:
    plan = build_snerv_candidate_curriculum_plan(
        candidate=_snerv_candidate(),
        requested_epochs=8,
        num_pairs=600,
        step_map_coder_mode="waterfill",
    )

    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in plan[
        "blockers"
    ]
    assert plan["training_plan"]["native_mlx_adapter_surfaces_ready"] is True
    assert (
        "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only"
        in plan["blockers"]
    )
    assert plan["long_campaign_prelaunch_gate"]["launch_allowed"] is False
    assert _requirement(plan, "real_segnet_teacher")["satisfied"] is False
    assert _requirement(plan, "real_posenet_teacher")["satisfied"] is False


def test_snerv_prelaunch_consumes_native_scorer_loop_but_keeps_materialization_blocker() -> None:
    plan = build_snerv_candidate_curriculum_plan(
        candidate=_snerv_candidate(),
        requested_epochs=8,
        num_pairs=600,
        step_map_coder_mode="waterfill",
        measured_packet_bytes=160_000,
        measured_archive_bytes=161_000,
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
    assert "snerv_scorer_loop_qat_not_attached" not in plan["blockers"]
    assert "snerv_real_segnet_teacher_missing" in plan["blockers"]
    assert "snerv_real_posenet_teacher_missing" in plan["blockers"]
    assert "snerv_native_scorer_loop_best_packet_not_materialized" in plan[
        "blockers"
    ]
    assert plan["score_claim"] is False


def test_snerv_official_receiver_payload_does_not_close_source_forward_authority() -> None:
    plan = build_snerv_candidate_curriculum_plan(
        candidate=_snerv_candidate(official=True),
        requested_epochs=8,
        num_pairs=600,
        step_map_coder_mode="waterfill",
        measured_packet_bytes=160_000,
        measured_archive_bytes=161_000,
        native_mlx_train_export_attached=True,
        native_mlx_receiver_proof_passed=True,
        native_mlx_full600_campaign_ready=True,
        native_mlx_long_training_bound=True,
        native_mlx_artifact_evidence={
            "schema": "snerv_mlx_native_train_export.v1",
            "snerv_official_mfu_hfr_tub_numeric_primitives_requested": True,
            "snerv_official_mfu_hfr_tub_export_bound": True,
            "snerv_official_mfu_hfr_tub_export_bound_semantics": (
                "receiver_payload_bound_not_source_forward_parity"
            ),
            "snerv_official_mfu_hfr_tub_receiver_payload_bound": True,
            "snerv_official_mfu_hfr_tub_frame_producing_export": True,
            "snerv_official_mfu_hfr_tub_source_forward_replay_bound": False,
            "snerv_official_mfu_hfr_tub_source_forward_replay_authority": False,
            "source_faithful_stack": False,
            "score_aware_long_training": {
                "official_mfu_hfr_tub_source_forward_replay": {
                    "blockers": [
                        "snerv_official_mfu_hfr_tub_trained_weight_mapping_to_long_training_missing"
                    ]
                }
            },
        },
    )

    split = plan["official_source_forward_authority_split"]
    assert split["receiver_payload_bound"] is True
    assert split["receiver_bound_training_evidence_usable"] is True
    assert split["full_source_forward_authority_proven"] is False
    assert split["launch_semantics"] == (
        "receiver_bound_training_allowed_but_official_source_authority_false"
    )
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        in plan["blockers"]
    )
    assert (
        "snerv_official_mfu_hfr_tub_trained_weight_mapping_to_long_training_missing"
        in plan["blockers"]
    )


def test_snerv_official_source_forward_authority_can_close_independently() -> None:
    plan = build_snerv_candidate_curriculum_plan(
        candidate=_snerv_candidate(official=True),
        requested_epochs=8,
        num_pairs=600,
        step_map_coder_mode="waterfill",
        measured_packet_bytes=160_000,
        measured_archive_bytes=161_000,
        native_mlx_train_export_attached=True,
        native_mlx_receiver_proof_passed=True,
        native_mlx_full600_campaign_ready=True,
        native_mlx_long_training_bound=True,
        native_mlx_artifact_evidence={
            "schema": "snerv_mlx_native_train_export.v1",
            "snerv_official_mfu_hfr_tub_numeric_primitives_requested": True,
            "snerv_official_mfu_hfr_tub_export_bound": True,
            "snerv_official_mfu_hfr_tub_receiver_payload_bound": True,
            "snerv_official_mfu_hfr_tub_frame_producing_export": True,
            "snerv_official_mfu_hfr_tub_source_forward_replay_bound": True,
            "source_forward_replay_verified": True,
            "snerv_official_mfu_hfr_tub_source_forward_replay_authority": True,
            "source_faithful_stack": True,
        },
    )

    split = plan["official_source_forward_authority_split"]
    assert split["full_source_forward_authority_proven"] is True
    assert split["launch_semantics"] == (
        "official_source_forward_parity_available_false_authority_until_score_gate"
    )
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        not in plan["blockers"]
    )
