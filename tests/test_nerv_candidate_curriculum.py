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
        differentiable_pose_preprocess_attached=True,
    )

    gate = plan["long_campaign_prelaunch_gate"]
    assert _requirement(plan, "eval_roundtrip_ste")["satisfied"] is True
    assert _requirement(plan, "differentiable_pose_preprocess")["satisfied"] is True
    assert "hi_nerv_eval_roundtrip_ste_missing" not in gate["blockers"]
    assert "hi_nerv_eval_roundtrip_ste_missing" not in plan["blockers"]
    assert "hi_nerv_differentiable_pose_preprocess_missing" not in gate["blockers"]
    assert "hi_nerv_differentiable_pose_preprocess_missing" not in plan["blockers"]
    assert set(gate["blocking_requirement_ids"]) >= {"ema_archive_selection"}


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
        candidate={
            "candidate_id": "snerv_full600_ceil180k",
            "num_pairs": 600,
            "hard_byte_ceiling": 180_000,
            "nominal_total_payload_bytes": 160_000,
            "levels": 3,
            "bits_per_coeff": 2.5,
            "step_map_bits_per_coeff": 4.0,
            "decoder_payload_codec": "snar1",
        },
        requested_epochs=8,
        num_pairs=600,
        step_map_coder_mode="waterfill",
    )

    assert "snerv_mlx_native_train_export_adapter_missing" in plan["blockers"]
    assert "snerv_score_aware_curriculum_not_native_mlx_yet" in plan["blockers"]
    assert plan["long_campaign_prelaunch_gate"]["launch_allowed"] is False
    assert _requirement(plan, "real_segnet_teacher")["satisfied"] is False
    assert _requirement(plan, "real_posenet_teacher")["satisfied"] is False
