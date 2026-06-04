# SPDX-License-Identifier: MIT
"""Tests for compact-carrier score-aware training plan routing."""
from __future__ import annotations

import pytest

from tac.substrates._shared.mlx_score_aware.carrier_training_plan import (
    CANONICAL_SCORE_AWARE_DECODER_TRAINING_STACK,
    CarrierTrainingPlanError,
    build_score_aware_carrier_training_plan,
)


def _all_stack_ready() -> dict[str, bool]:
    return {
        "real_segnet_teacher_ready": True,
        "real_posenet_teacher_ready": True,
        "eval_roundtrip_ready": True,
        "ema_ready": True,
        "pr95_curriculum_ready": True,
        "adamw_ready": True,
        "muon_ready": True,
        "coder_aware_regularization_ready": True,
        "sigma_noise_qat_ready": True,
        "quant_noise_qat_ready": True,
        "nvrc_learned_quant_ready": True,
        "byte_closed_archive_export_ready": True,
    }


def _receiver_closed_modelsize_rows() -> list[dict[str, object]]:
    return [
        {
            "row_id": "tiny",
            "archive_bytes": 20_000,
            "archive_sha256": "a" * 64,
            "nonrate_score": 0.240,
            "modelsize_mparams": 0.02,
            "fc_dim": 8,
            "receiver_closed": True,
            "receiver_proof_passed": True,
            "receiver_proof_path": "proofs/tiny.json",
            "receiver_proof_sha256": "b" * 64,
            "receiver_proof_axis_tag": "[macOS-CPU advisory]",
            "num_pairs": 600,
        },
        {
            "row_id": "small",
            "archive_bytes": 40_000,
            "archive_sha256": "c" * 64,
            "nonrate_score": 0.205,
            "modelsize_mparams": 0.04,
            "fc_dim": 16,
            "receiver_closed": True,
            "receiver_proof_passed": True,
            "receiver_proof_path": "proofs/small.json",
            "receiver_proof_sha256": "d" * 64,
            "receiver_proof_axis_tag": "[macOS-CPU advisory]",
            "num_pairs": 600,
        },
        {
            "row_id": "medium",
            "archive_bytes": 80_000,
            "archive_sha256": "e" * 64,
            "nonrate_score": 0.200,
            "modelsize_mparams": 0.08,
            "fc_dim": 32,
            "receiver_closed": True,
            "receiver_proof_passed": True,
            "receiver_proof_path": "proofs/medium.json",
            "receiver_proof_sha256": "f" * 64,
            "receiver_proof_axis_tag": "[macOS-CPU advisory]",
            "num_pairs": 600,
        },
    ]


def _advisory_modelsize_rows() -> list[dict[str, object]]:
    return [
        {
            "row_id": "projected_tiny",
            "projected_archive_bytes_600pair": 20_000,
            "nonrate_score": 0.240,
            "lower_bound_only": True,
        },
        {
            "row_id": "zip_without_receiver",
            "archive_bytes": 40_000,
            "nonrate_score": 0.205,
        },
    ]


def test_cheap_but_unfit_hinerv_routes_to_decoder_weight_training() -> None:
    row = build_score_aware_carrier_training_plan(
        {
            "archive_bytes": 40_491,
            "projected_archive_bytes_600pair": 36_000,
            "d_seg": 0.508,
            "advisory_score": 92.84,
            "g3_adjoint_exact": True,
            "latent_jvp_norm_max": 1.0e-4,
            "modelsize_knob_present": True,
            "modelsize_budget_rows": _receiver_closed_modelsize_rows(),
            "linf_delta_vs_l2": 0.31,
            **_all_stack_ready(),
        },
        carrier_id="hi_nerv",
    )

    assert row["planner_action"] == "run_score_aware_decoder_weight_training_full_main"
    assert row["carrier_fit_status"] == "unusable"
    assert row["rate_knob_status"] == "structural_rate_knob_present"
    assert row["modelsize_budget_receiver_closed_ready"] is True
    assert row["allocator_target_surface"] == "decoder_weights"
    assert row["linf_latent_posthoc_status"] == "demoted"
    assert "carrier_fit_unusable_d_seg" in row["dispatch_blockers"]
    assert "latent_posthoc_allocator_demoted_low_leverage" in row["dispatch_blockers"]
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["rank_or_kill_eligible"] is False


def test_missing_real_scorer_teachers_blocks_training_readiness() -> None:
    evidence = {
        "d_seg": 0.03,
        "advisory_score": 0.31,
        "g3_adjoint_exact": True,
        "latent_jvp_norm_max": 3.0e-3,
        "modelsize_knob_present": True,
        "modelsize_budget_rows": _receiver_closed_modelsize_rows(),
        **_all_stack_ready(),
    }
    evidence["real_segnet_teacher_ready"] = False
    evidence["real_posenet_teacher_ready"] = False

    row = build_score_aware_carrier_training_plan(evidence, carrier_id="snerv")

    assert row["score_aware_training_ready"] is False
    assert row["modelsize_budget_receiver_closed_ready"] is True
    assert "missing_training_stack:real_segnet_teacher" in row["dispatch_blockers"]
    assert "missing_training_stack:real_posenet_teacher" in row["dispatch_blockers"]
    assert row["planner_action"] == "run_score_aware_decoder_weight_training_full_main"


def test_plausible_fit_with_full_stack_routes_to_local_replay_not_auth() -> None:
    row = build_score_aware_carrier_training_plan(
        {
            "d_seg": 0.025,
            "advisory_score": 0.19,
            "g3_adjoint_exact": True,
            "latent_jvp_norm_max": 2.0e-3,
            "modelsize_knob_present": True,
            "modelsize_budget_rows": _receiver_closed_modelsize_rows(),
            **_all_stack_ready(),
        },
        carrier_id="hi_nerv",
    )

    assert row["planner_action"] == "run_byte_closed_local_replay_gate_before_exact_auth"
    assert row["carrier_fit_status"] == "locally_plausible"
    assert row["score_aware_training_ready"] is True
    assert row["modelsize_budget_receiver_closed_ready"] is True
    assert row["allocator_target_surface"] == "decoder_weights_plus_latents_pending_leverage_probe"
    assert "requires_contest_cpu_then_cuda_auth_only_after_local_win" in row["dispatch_blockers"]
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["exact_cuda_auth_eval"] is False


def test_truthy_authority_fields_are_rejected() -> None:
    with pytest.raises(CarrierTrainingPlanError, match="forbidden truthy authority fields"):
        build_score_aware_carrier_training_plan(
            {
                "d_seg": 0.03,
                "advisory_score": 0.2,
                "score_claim": True,
                **_all_stack_ready(),
            },
            carrier_id="hi_nerv",
        )


def test_required_stack_mentions_optimizer_qat_rate_levers() -> None:
    row = build_score_aware_carrier_training_plan(
        {
            "d_seg": 0.508,
            "advisory_score": 92.84,
            "g3_adjoint_exact": True,
            "latent_leverage_status": "near_zero",
            "modelsize_knob_present": True,
            "modelsize_budget_rows": _receiver_closed_modelsize_rows(),
            **_all_stack_ready(),
        },
        carrier_id="hi_nerv",
    )

    assert tuple(row["required_training_stack"]) == CANONICAL_SCORE_AWARE_DECODER_TRAINING_STACK
    assert {
        "coder_aware_regularization_c1a",
        "sigma_noise_qat",
        "quant_noise_qat",
        "nvrc_learned_quantization",
    }.issubset(set(row["q_at_rate_levers"]))


def test_training_plan_consumes_measured_modelsize_budget_ladder() -> None:
    row = build_score_aware_carrier_training_plan(
        {
            "d_seg": 0.025,
            "advisory_score": 0.19,
            "g3_adjoint_exact": True,
            "latent_jvp_norm_max": 2.0e-3,
            "modelsize_knob_present": True,
            "modelsize_budget_rows": _receiver_closed_modelsize_rows(),
            **_all_stack_ready(),
        },
        carrier_id="hi_nerv",
    )

    assert (
        row["modelsize_budget_plan_status"]
        == "receiver_closed_modelsize_budget_selected"
    )
    assert row["modelsize_budget_plan"]["selected_point"]["row_id"] == "small"
    assert row["evidence_summary"]["selected_modelsize_archive_bytes"] == 40_000
    assert (
        row["evidence_summary"]["receiver_closed_selected_modelsize_archive_bytes"]
        == 40_000
    )
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_training_plan_passes_hard_byte_ceiling_to_modelsize_budget() -> None:
    row = build_score_aware_carrier_training_plan(
        {
            "d_seg": 0.025,
            "advisory_score": 0.19,
            "g3_adjoint_exact": True,
            "latent_jvp_norm_max": 2.0e-3,
            "modelsize_knob_present": True,
            "hard_byte_ceiling": 50_000,
            "modelsize_budget_rows": _receiver_closed_modelsize_rows(),
            **_all_stack_ready(),
        },
        carrier_id="snerv",
    )

    plan = row["modelsize_budget_plan"]
    assert plan["hard_byte_ceiling"] == 50_000
    assert plan["selected_point"]["row_id"] == "small"
    assert plan["selected_archive_bytes"] == 40_000
    assert plan["selected_under_hard_byte_ceiling"] is True
    assert row["modelsize_budget_receiver_closed_ready"] is True
    assert row["score_claim"] is False


def test_advisory_modelsize_budget_blocks_training_ready() -> None:
    row = build_score_aware_carrier_training_plan(
        {
            "d_seg": 0.025,
            "advisory_score": 0.19,
            "g3_adjoint_exact": True,
            "latent_jvp_norm_max": 2.0e-3,
            "modelsize_knob_present": True,
            "modelsize_budget_rows": _advisory_modelsize_rows(),
            **_all_stack_ready(),
        },
        carrier_id="snerv",
    )

    assert (
        row["planner_action"]
        == "run_receiver_closed_modelsize_ladder_before_score_aware_training"
    )
    assert row["score_aware_training_ready"] is False
    assert row["modelsize_budget_receiver_closed_ready"] is False
    assert row["modelsize_budget_plan_status"] == (
        "advisory_or_projected_modelsize_budget_selected"
    )
    assert row["evidence_summary"]["selected_modelsize_archive_bytes"] is not None
    assert (
        row["evidence_summary"]["receiver_closed_selected_modelsize_archive_bytes"]
        is None
    )
    assert "receiver_closed_modelsize_budget_ladder_missing" in row["dispatch_blockers"]
    assert (
        "modelsize_budget:modelsize_budget_selection_is_advisory_or_projected"
        in row["dispatch_blockers"]
    )
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
