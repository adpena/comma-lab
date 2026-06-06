# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

from tac.analysis.action_effect import ACTION_EFFECT_SCHEMA
from tac.analysis.nerv_pair_local_distortion_servo import (
    PAIR_LOCAL_DISTORTION_SERVO_ADMISSION_SCHEMA,
    PAIR_LOCAL_DISTORTION_SERVO_RECEIPT_SCHEMA,
    PAIR_LOCAL_DISTORTION_SERVO_REPORT_SCHEMA,
    PairLocalScoreState,
    PairLocalSurfaceTrace,
    admit_pair_local_distortion_action,
    build_pr95_grade_pair_local_servo_report,
    byte_cost_score_units,
    exact_pair_local_score_delta,
    pair_local_servo_static_contract,
    seg_argmax_pixel_debt_score_units,
    select_worst_scorer_debt_target,
)


def test_admits_frame1_pair_local_action_only_after_parseback_survival() -> None:
    before = PairLocalScoreState(d_seg=0.010, d_pose=0.0001, archive_bytes=1000)
    after = PairLocalScoreState(d_seg=0.009, d_pose=0.0001, archive_bytes=1000)
    trace = PairLocalSurfaceTrace(
        family="hinerv",
        frame_scope="frame1_seg_pose_joint",
        actuator_id="hinerv_target_region_birth",
        pair_index=7,
        float_rgb_delta_linf=4.0,
        uint8_changed_pixels=33,
        uint8_delta_abs_max=3.0,
        segnet_input_delta_linf=0.04,
        segnet_margin_delta=0.12,
        segnet_argmax_flipped_pixels=18,
        fakequant_segnet_margin_delta=0.08,
        fakequant_argmax_flipped_pixels=12,
        parseback_segnet_margin_delta=0.07,
        parseback_argmax_flipped_pixels=11,
    )

    receipt = admit_pair_local_distortion_action(
        before=before,
        after=after,
        trace=trace,
    )

    assert receipt.schema == PAIR_LOCAL_DISTORTION_SERVO_ADMISSION_SCHEMA
    assert receipt.admitted is True
    assert receipt.blockers == ()
    assert receipt.exact_score_delta < 0
    assert receipt.surfaces["uint8_motion"] is True
    assert receipt.surfaces["fakequant_survival"] is True
    assert receipt.surfaces["parseback_survival"] is True
    assert receipt.score_claim is False


def test_pair_local_servo_uses_target_support_not_generic_argmax_churn() -> None:
    before = PairLocalScoreState(d_seg=0.010, d_pose=0.0001, archive_bytes=1000)
    after = PairLocalScoreState(d_seg=0.009, d_pose=0.0001, archive_bytes=1000)
    trace = PairLocalSurfaceTrace(
        family="hinerv",
        frame_scope="frame1_seg_pose_joint",
        actuator_id="hinerv_target_region_birth",
        pair_index=7,
        uint8_changed_pixels=33,
        uint8_delta_abs_max=3.0,
        segnet_input_delta_linf=0.04,
        target_hard_won_count=8,
        target_hard_lost_count=0,
        net_target_support_delta=8,
        wrong_to_target_count=8,
        argmax_changed_count_region=40,
        fakequant_target_hard_won_count=7,
        fakequant_net_target_support_delta=7,
        parseback_target_hard_won_count=6,
        parseback_net_target_support_delta=6,
    )

    receipt = admit_pair_local_distortion_action(
        before=before,
        after=after,
        trace=trace,
    )

    assert receipt.admitted is True
    assert receipt.surfaces["seg_movement"] is True
    assert receipt.surfaces["target_support_birth"] is True
    assert receipt.surfaces["fakequant_target_support_survival"] is True
    assert receipt.surfaces["parseback_target_support_survival"] is True


def test_rejects_subquantum_float_update_even_when_score_numbers_improve() -> None:
    before = PairLocalScoreState(d_seg=0.010, d_pose=0.0001, archive_bytes=1000)
    after = PairLocalScoreState(d_seg=0.009, d_pose=0.0001, archive_bytes=1000)
    trace = PairLocalSurfaceTrace(
        family="hinerv",
        frame_scope="frame1_seg_pose_joint",
        actuator_id="hinerv_target_region_birth",
        float_rgb_delta_linf=0.1,
        uint8_changed_pixels=0,
        segnet_input_delta_linf=0.0,
        segnet_margin_delta=0.2,
        segnet_argmax_flipped_pixels=4,
        fakequant_argmax_flipped_pixels=4,
        parseback_argmax_flipped_pixels=4,
    )

    receipt = admit_pair_local_distortion_action(
        before=before,
        after=after,
        trace=trace,
    )

    assert receipt.admitted is False
    assert "pair_local_servo_subquantum_float_update_no_uint8_motion" in (receipt.blockers)
    assert "pair_local_servo_receiver_uint8_motion_missing" in receipt.blockers
    assert "pair_local_servo_scorer_preprocess_motion_missing" in receipt.blockers


def test_rejects_live_argmax_motion_lost_by_fakequant_or_parseback() -> None:
    before = PairLocalScoreState(d_seg=0.010, d_pose=0.0001, archive_bytes=1000)
    after = PairLocalScoreState(d_seg=0.009, d_pose=0.0001, archive_bytes=1000)
    trace = PairLocalSurfaceTrace(
        family="hinerv",
        frame_scope="frame1_seg_pose_joint",
        actuator_id="hinerv_target_region_birth",
        uint8_changed_pixels=64,
        uint8_delta_abs_max=2.0,
        segnet_input_delta_linf=0.03,
        segnet_argmax_flipped_pixels=9,
        fakequant_argmax_flipped_pixels=0,
        parseback_argmax_flipped_pixels=0,
    )

    receipt = admit_pair_local_distortion_action(
        before=before,
        after=after,
        trace=trace,
    )

    assert receipt.admitted is False
    assert "pair_local_servo_fakequant_survival_missing" in receipt.blockers
    assert "pair_local_servo_archive_parseback_survival_missing" in receipt.blockers


def test_rejects_when_exact_nonlinear_score_worsens_despite_seg_improvement() -> None:
    before = PairLocalScoreState(d_seg=0.010, d_pose=0.000001, archive_bytes=1000)
    after = PairLocalScoreState(d_seg=0.00999, d_pose=0.000004, archive_bytes=1000)
    trace = PairLocalSurfaceTrace(
        family="snerv",
        frame_scope="frame1_seg_pose_joint",
        actuator_id="snerv_hf_boundary_gate",
        uint8_changed_pixels=20,
        uint8_delta_abs_max=1.0,
        segnet_input_delta_linf=0.02,
        segnet_margin_delta=0.03,
        fakequant_segnet_margin_delta=0.03,
        parseback_segnet_margin_delta=0.02,
    )

    assert exact_pair_local_score_delta(before, after) > 0
    receipt = admit_pair_local_distortion_action(
        before=before,
        after=after,
        trace=trace,
    )

    assert receipt.admitted is False
    assert "pair_local_servo_exact_nonlinear_score_not_improved" in receipt.blockers


def test_frame0_pose_only_cannot_claim_segnet_mutation() -> None:
    before = PairLocalScoreState(d_seg=0.010, d_pose=0.0004, archive_bytes=1000)
    after = PairLocalScoreState(d_seg=0.009, d_pose=0.0003, archive_bytes=1000)
    trace = PairLocalSurfaceTrace(
        family="snerv",
        frame_scope="frame0_pose_only",
        actuator_id="snerv_lf_pose_gate",
        uint8_changed_pixels=30,
        uint8_delta_abs_max=2.0,
        posenet_input_delta_linf=0.04,
        pose_output_delta_l2=0.01,
        fakequant_pose_output_delta_l2=0.01,
        parseback_pose_output_delta_l2=0.01,
    )

    receipt = admit_pair_local_distortion_action(
        before=before,
        after=after,
        trace=trace,
    )

    assert receipt.admitted is False
    assert "pair_local_servo_frame0_pose_only_changed_segnet_distortion" in (receipt.blockers)


def test_pr95_grade_report_accepts_parseback_priced_hinerv_receipt() -> None:
    receipt = _good_receipt()

    report = build_pr95_grade_pair_local_servo_report(receipt)

    assert report["schema"] == PAIR_LOCAL_DISTORTION_SERVO_REPORT_SCHEMA
    assert report["long_run_admission_ready"] is True
    assert report["blockers"] == []
    assert report["authority"] == "parseback_mlx"
    assert report["value_per_byte"] > report["byte_price"]
    assert report["action_effect"]["schema"] == ACTION_EFFECT_SCHEMA
    assert report["action_effect"]["action_effect_admitted"] is True
    assert report["action_effect"]["delta_score_total"] == report["exact_score_delta"]
    assert report["action_effect"]["receiver_visible"] is True
    assert report["action_effect"]["state_custody"]["archive_sha256"] == "a" * 64
    assert report["score_claim"] is False


def test_pr95_grade_report_rejects_live_only_and_unpriced_byte_growth() -> None:
    receipt = {
        **_good_receipt(),
        "authority": "live_mlx",
        "archive_sha256": None,
        "new_archive_bytes": 1100,
        "value_per_byte": None,
    }

    report = build_pr95_grade_pair_local_servo_report(receipt)

    assert report["long_run_admission_ready"] is False
    assert "pair_local_servo_archive_parseback_authority_missing" in report["blockers"]
    assert "pair_local_servo_value_per_byte_not_priced" in report["blockers"]


def test_debt_selection_and_score_unit_helpers_are_evaluator_priced() -> None:
    target = select_worst_scorer_debt_target(
        [
            {"target_id": "seg_region_1", "score_units": 0.2, "axis": "seg"},
            {"target_id": "pose_pair_9", "score_units": 0.31, "axis": "pose"},
        ]
    )

    assert target.target_id == "pose_pair_9"
    assert math.isclose(
        seg_argmax_pixel_debt_score_units(wrong_pixels=5, total_scored_pixels=1000),
        0.5,
    )
    assert math.isclose(byte_cost_score_units(100), 25.0 * 100 / 37_545_489)
    contract = pair_local_servo_static_contract()
    assert contract["families"] == ["hinerv", "snerv"]
    assert "archive_parseback_survival" in contract["survival_gates"]


def _good_receipt() -> dict[str, object]:
    return {
        "schema": PAIR_LOCAL_DISTORTION_SERVO_RECEIPT_SCHEMA,
        "family": "hinerv",
        "pair_ids": [7],
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
    }
