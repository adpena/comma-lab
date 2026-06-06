# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.substrates.hi_nerv.short_scorer_readiness import (
    HI_NERV_SHORT_SCORER_SMOKE_AXIS_TAG,
    build_hinerv_short_scorer_smoke_readiness_report,
    hinerv_short_scorer_smoke_long_run_admission,
)


def _controls(**overrides: float) -> dict[str, float]:
    values = {
        "segnet_direct_live_distillation_weight": 0.4,
        "segnet_direct_live_class_histogram_weight": 0.0,
        "segnet_direct_live_class_balanced_hinge_weight": 0.0,
        "segnet_direct_live_class_balanced_ce_weight": 0.0,
        "segnet_direct_live_class_balanced_squared_hinge_weight": 0.0,
        "segnet_direct_live_class_region_recon_weight": 0.0,
        "segnet_direct_live_rare_class_logit_weight": 0.0,
        "segnet_direct_live_target_mass_floor_weight": 0.0,
        "segnet_direct_live_target_min_ratio_floor_weight": 0.0,
        "pose_direct_live_distillation_weight": 0.0,
        "scorer_input_contrast_floor_weight": 0.5,
        "scorer_input_contrast_floor_segnet_min_std_ratio": 0.6,
        "scorer_input_contrast_floor_posenet_yuv6_min_std_ratio": 0.6,
        "scorer_input_shape_tether_weight": 0.25,
        "posenet_temporal_signal_floor_weight": 0.25,
    }
    values.update(overrides)
    return values


def _receiver_quality() -> dict[str, object]:
    return {
        "quality_gate_passed": True,
        "quality_gate": {"verdict": "PASS", "stats": {}},
        "scorer_input_distribution_gate": {
            "fit_gate_passed": True,
            "blockers": [],
        },
        "segnet_argmax_probe": {
            "fit_gate_passed": True,
            "candidate_occupied_class_fraction": 0.8,
            "candidate_target_class_coverage_fraction": 0.8,
            "candidate_target_class_min_ratio": 0.25,
            "candidate_target_material_class_covered_count": 4.0,
            "target_material_class_count": 5.0,
            "reference_occupied_class_fraction": 0.9,
            "segnet_argmax_disagreement_rate": 0.02,
            "blockers": [],
        },
        "mlx_scorer_response_probe_required": True,
        "mlx_scorer_response_probe": {
            "fit_gate_passed": True,
            "avg_posenet_dist": 0.002,
            "avg_segnet_dist": 0.02,
            "blockers": [],
        },
        "blockers": [],
    }


def _base_metrics() -> dict[str, float]:
    return {
        "loss_part_segnet_direct_live_distill": 0.12,
        "loss_part_segnet_direct_live_argmax_disagreement": 0.03,
        "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
        "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
        "loss_part_segnet_direct_live_candidate_target_class_missing_fraction": 0.0,
        "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.25,
        "loss_part_scorer_input_contrast_floor": 0.01,
        "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 0.75,
        "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 0.8,
        "loss_part_scorer_input_shape_tether": 0.02,
        "loss_part_scorer_input_shape_tether_segnet_last_rgb": 0.003,
        "loss_part_scorer_input_shape_tether_posenet_yuv6_pair": 0.004,
        "loss_part_scorer_input_shape_tether_posenet_yuv6_temporal_delta": 0.005,
        "loss_part_posenet_temporal_signal_floor": 0.03,
        "loss_part_posenet_temporal_signal_floor_mean_std_ratio": 0.7,
        "loss_part_posenet_temporal_signal_floor_mean_abs_ratio": 0.72,
    }


def _pose_direct_live_metrics() -> dict[str, float]:
    return {
        "loss_part_pose_direct_live_score_term": 0.14,
        "loss_part_pose_direct_live_raw_mse": 0.00196,
        "loss_part_pose_direct_live_yuv6_pair_std": 0.22,
        "loss_part_pose_direct_live_yuv6_pair_temporal_delta_std": 0.08,
    }


def _dual_metrics(
    *constraint_keys: str,
    weight_applied: float = 1.0,
    effective_loss_weight: float = 0.5,
    violation: float = 0.1,
    lambda_value: float = 0.04,
) -> dict[str, float]:
    expanded_constraint_keys: list[str] = []
    for constraint_key in constraint_keys:
        if constraint_key not in expanded_constraint_keys:
            expanded_constraint_keys.append(constraint_key)
        implied: tuple[str, ...] = ()
        if constraint_key == "hi_nerv_segnet_direct_live_distill":
            implied = ("hi_nerv_segnet_direct_live_argmax_disagreement",)
        elif constraint_key == "hi_nerv_segnet_direct_live_class_histogram":
            implied = (
                "hi_nerv_segnet_direct_live_target_missing_fraction_histogram",
            )
        elif constraint_key == "hi_nerv_segnet_direct_live_class_balanced_ce":
            implied = ("hi_nerv_segnet_direct_live_target_missing_fraction_ce",)
        elif constraint_key == "hi_nerv_segnet_direct_live_class_region_recon":
            implied = (
                "hi_nerv_segnet_direct_live_target_min_ratio_region_recon",
            )
        elif constraint_key == "hi_nerv_segnet_direct_live_rare_class_logit":
            implied = (
                "hi_nerv_segnet_direct_live_target_min_ratio_rare_class_logit",
            )
        elif constraint_key == "hi_nerv_segnet_direct_live_target_mass_floor":
            implied = (
                "hi_nerv_segnet_direct_live_target_min_ratio_mass_floor",
            )
        elif constraint_key == "hi_nerv_segnet_direct_live_target_min_ratio_floor":
            implied = (
                "hi_nerv_segnet_direct_live_target_min_ratio_floor_gate",
            )
        for implied_key in implied:
            if implied_key not in expanded_constraint_keys:
                expanded_constraint_keys.append(implied_key)
    metrics = {
        "dual_ascent_active": 1.0,
        "dual_ascent_constraint_count": float(len(expanded_constraint_keys)),
    }
    for constraint_key in expanded_constraint_keys:
        metrics.update(
            {
                f"dual_ascent_metric__{constraint_key}": 0.12,
                f"dual_ascent_missing_metric__{constraint_key}": 0.0,
                f"dual_ascent_lambda__{constraint_key}": lambda_value,
                f"dual_ascent_update_count__{constraint_key}": 1.0,
                f"dual_ascent_weight_applied__{constraint_key}": weight_applied,
                f"dual_ascent_effective_loss_weight__{constraint_key}": (
                    effective_loss_weight
                ),
                f"dual_ascent_violation__{constraint_key}": violation,
            }
        )
    return metrics


def test_direct_live_pose_only_accepts_observed_posenet_and_dual_telemetry() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(pose_direct_live_distillation_weight=0.6),
        final_loss_components={
            **_base_metrics(),
            **_pose_direct_live_metrics(),
            **_dual_metrics(
                "hi_nerv_segnet_direct_live_distill",
                "hi_nerv_posenet_yuv6_pair_distill",
            ),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=0.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["axis_tag"] == HI_NERV_SHORT_SCORER_SMOKE_AXIS_TAG
    assert report["ready_for_long_run"] is True
    assert report["score_claim"] is False
    assert report["teacher_gate"]["direct_live_posenet_only"] is True
    assert report["direct_live_posenet_gate"]["metrics"][
        "loss_part_pose_direct_live_score_term"
    ] == pytest.approx(0.14)
    assert report["direct_live_dual_ascent_gate"]["required"] is True
    assert "hi_nerv_short_smoke_real_posenet_teacher_not_requested" not in report[
        "actionable_blockers"
    ]
    admission = hinerv_short_scorer_smoke_long_run_admission(report)
    assert admission["long_run_admission_passed"] is True
    assert admission["short_scorer_teacher_smoke_passed"] is True
    assert admission["admission_blockers"] == []


def test_direct_live_segnet_requires_dual_even_with_generic_teacher() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components=_base_metrics(),
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is False
    assert report["direct_live_dual_ascent_gate"]["required"] is True
    assert "hi_nerv_short_smoke_missing_direct_live_dual_ascent_telemetry" in report[
        "actionable_blockers"
    ]
    admission = hinerv_short_scorer_smoke_long_run_admission(report)
    assert admission["long_run_admission_passed"] is False
    assert "hi_nerv_short_scorer_smoke_not_ready_for_long_run" in admission[
        "admission_blockers"
    ]
    assert "hi_nerv_short_smoke_missing_direct_live_dual_ascent_telemetry" in admission[
        "admission_blockers"
    ]


def test_decoder_weight_waterfill_requires_live_gradient_actuation() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components={
            **_base_metrics(),
            **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        decoder_weight_waterfill_plan_metadata={
            "attached": True,
            "row_count": 1,
            "train_time_fake_quant_bound": True,
            "fake_quant_forward": {
                "configured": True,
                "targeted_tensor_count": 1,
            },
        },
    )

    assert report["ready_for_long_run"] is False
    assert report["decoder_weight_waterfill_actuation_gate"]["required"] is True
    assert (
        "hi_nerv_short_smoke_decoder_waterfill_gradient_multiplier_not_observed"
        in report["actionable_blockers"]
    )


def test_decoder_weight_waterfill_blocks_zero_requested_controls_for_nonempty_plan() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components={
            **_base_metrics(),
            **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
            "gradient_multiplier_requested_control_count": 0.0,
            "gradient_multiplier_applied_leaf_count": 0.0,
            "gradient_multiplier_requested_but_unapplied": 0.0,
            "gradient_multiplier_missing_exact_name_count": 0.0,
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        decoder_weight_waterfill_plan_metadata={
            "attached": True,
            "row_count": 1,
            "train_time_fake_quant_bound": True,
            "fake_quant_forward": {
                "configured": True,
                "targeted_tensor_count": 1,
            },
        },
    )

    gate = report["decoder_weight_waterfill_actuation_gate"]
    assert gate["gradient_multiplier_metrics_present"] is True
    assert (
        gate["gradient_multiplier_requested_control_absent_for_nonempty_plan"]
        is True
    )
    assert "hi_nerv_short_smoke_decoder_waterfill_gradient_multiplier_unapplied" in (
        report["actionable_blockers"]
    )


def test_decoder_weight_waterfill_accepts_live_gradient_actuation() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components={
            **_base_metrics(),
            **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
            "gradient_multiplier_requested_control_count": 1.0,
            "gradient_multiplier_applied_leaf_count": 1.0,
            "gradient_multiplier_requested_but_unapplied": 0.0,
            "gradient_multiplier_missing_exact_name_count": 0.0,
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        decoder_weight_waterfill_plan_metadata={
            "attached": True,
            "row_count": 1,
            "train_time_fake_quant_bound": True,
            "fake_quant_forward": {
                "configured": True,
                "targeted_tensor_count": 1,
            },
        },
    )

    assert report["ready_for_long_run"] is True
    gate = report["decoder_weight_waterfill_actuation_gate"]
    assert gate["gradient_multiplier_metrics_present"] is True
    assert gate["gradient_multiplier_applied_leaf_count"] == pytest.approx(1.0)
    assert report["actionable_blockers"] == []


def test_direct_live_rare_class_logit_requires_loss_and_dual_telemetry() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(segnet_direct_live_rare_class_logit_weight=0.7),
        final_loss_components={
            **_base_metrics(),
            **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert "loss_part_segnet_direct_live_rare_class_logit_loss" in report[
        "direct_live_segnet_gate"
    ]["active_subcontrol_metric_keys"]
    assert "hi_nerv_short_smoke_missing_direct_live_segnet_telemetry" in report[
        "actionable_blockers"
    ]
    assert "hi_nerv_short_smoke_missing_direct_live_segnet_subcontrol_telemetry" in (
        report["actionable_blockers"]
    )
    assert "hi_nerv_short_smoke_missing_direct_live_dual_ascent_telemetry" in (
        report["actionable_blockers"]
    )


def test_direct_live_rare_class_logit_clears_with_loss_and_dual_telemetry() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(segnet_direct_live_rare_class_logit_weight=0.7),
        final_loss_components={
            **_base_metrics(),
            "loss_part_segnet_direct_live_rare_class_logit_loss": 0.05,
            **_dual_metrics(
                "hi_nerv_segnet_direct_live_distill",
                "hi_nerv_segnet_direct_live_rare_class_logit",
            ),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is True
    assert report["actionable_blockers"] == []


def test_direct_live_target_mass_floor_requires_loss_and_dual_telemetry() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(segnet_direct_live_target_mass_floor_weight=0.7),
        final_loss_components={
            **_base_metrics(),
            **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert "loss_part_segnet_direct_live_target_mass_floor_loss" in report[
        "direct_live_segnet_gate"
    ]["active_subcontrol_metric_keys"]
    assert "hi_nerv_short_smoke_missing_direct_live_segnet_telemetry" in report[
        "actionable_blockers"
    ]
    assert "hi_nerv_short_smoke_missing_direct_live_segnet_subcontrol_telemetry" in (
        report["actionable_blockers"]
    )
    assert "hi_nerv_short_smoke_missing_direct_live_dual_ascent_telemetry" in (
        report["actionable_blockers"]
    )


def test_direct_live_target_mass_floor_clears_with_loss_and_dual_telemetry() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(segnet_direct_live_target_mass_floor_weight=0.7),
        final_loss_components={
            **_base_metrics(),
            "loss_part_segnet_direct_live_target_mass_floor_loss": 0.05,
            **_dual_metrics(
                "hi_nerv_segnet_direct_live_distill",
                "hi_nerv_segnet_direct_live_target_mass_floor",
            ),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is True
    assert report["actionable_blockers"] == []


def test_direct_live_target_min_ratio_floor_requires_loss_and_dual_telemetry() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(
            segnet_direct_live_target_min_ratio_floor_weight=0.7
        ),
        final_loss_components={
            **_base_metrics(),
            **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert "loss_part_segnet_direct_live_target_min_ratio_floor_loss" in report[
        "direct_live_segnet_gate"
    ]["active_subcontrol_metric_keys"]
    assert "hi_nerv_short_smoke_missing_direct_live_segnet_telemetry" in report[
        "actionable_blockers"
    ]
    assert "hi_nerv_short_smoke_missing_direct_live_segnet_subcontrol_telemetry" in (
        report["actionable_blockers"]
    )
    assert "hi_nerv_short_smoke_missing_direct_live_dual_ascent_telemetry" in (
        report["actionable_blockers"]
    )


def test_direct_live_target_min_ratio_floor_clears_with_loss_and_dual_telemetry() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(
            segnet_direct_live_target_min_ratio_floor_weight=0.7
        ),
        final_loss_components={
            **_base_metrics(),
            "loss_part_segnet_direct_live_target_min_ratio_floor_loss": 0.05,
            **_dual_metrics(
                "hi_nerv_segnet_direct_live_distill",
                "hi_nerv_segnet_direct_live_target_min_ratio_floor",
            ),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is True
    assert report["actionable_blockers"] == []


def test_required_pose_direct_live_blocks_generic_posenet_only() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components={
            **_base_metrics(),
            **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        require_pose_direct_live_distillation=True,
    )

    assert report["direct_live_posenet_gate"]["required"] is True
    assert "hi_nerv_short_smoke_direct_live_posenet_distillation_required" in (
        report["actionable_blockers"]
    )


def test_output_head_target_contrast_init_is_required_when_metadata_is_bound() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components={
            **_base_metrics(),
            **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        output_head_target_bias_init_metadata={
            "schema": "hi_nerv_output_head_target_bias_init.v1",
            "enabled": True,
            "contrast_init": {
                "schema": "hi_nerv_output_head_target_contrast_init.v1",
                "enabled": False,
            },
        },
    )

    gate = report["output_head_target_init_gate"]
    assert gate["required"] is True
    assert gate["bias_init_enabled"] is True
    assert gate["contrast_init_enabled"] is False
    assert "hi_nerv_short_smoke_output_head_target_contrast_init_not_enabled" in (
        report["actionable_blockers"]
    )


def test_direct_live_segnet_blocks_train_target_class_coverage_collapse() -> None:
    metrics = {
        **_base_metrics(),
        "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
        "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.6,
        **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
    }

    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components=metrics,
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is False
    assert (
        report["direct_live_segnet_gate"][
            "min_candidate_target_class_coverage_fraction_for_fit_gate"
        ]
        == pytest.approx(0.8)
    )
    assert (
        "hi_nerv_short_smoke_direct_live_target_class_coverage_collapsed"
        in report["actionable_blockers"]
    )


def test_direct_live_segnet_blocks_train_target_class_mass_collapse() -> None:
    metrics = {
        **_base_metrics(),
        "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
        "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
        "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.05,
        **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
    }

    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components=metrics,
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is False
    assert (
        report["direct_live_segnet_gate"][
            "min_candidate_target_class_min_ratio_for_fit_gate"
        ]
        == pytest.approx(0.2)
    )
    assert "hi_nerv_short_smoke_direct_live_target_class_mass_collapsed" in report[
        "actionable_blockers"
    ]


def test_direct_live_segnet_blocks_receiver_target_class_coverage_collapse() -> None:
    receiver = _receiver_quality()
    receiver["segnet_argmax_probe"] = {
        **receiver["segnet_argmax_probe"],
        "candidate_occupied_class_fraction": 0.8,
        "candidate_target_class_coverage_fraction": 0.6,
    }

    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components={
            **_base_metrics(),
            **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
        },
        post_export_quality=receiver,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is False
    assert report["receiver_cache_quality"][
        "candidate_argmax_target_class_coverage_fraction"
    ] == pytest.approx(0.6)
    assert (
        "hi_nerv_short_smoke_receiver_cache_segnet_target_class_coverage_collapsed"
        in report["actionable_blockers"]
    )


def test_direct_live_segnet_blocks_receiver_target_class_mass_collapse() -> None:
    receiver = _receiver_quality()
    receiver["segnet_argmax_probe"] = {
        **receiver["segnet_argmax_probe"],
        "candidate_occupied_class_fraction": 0.8,
        "candidate_target_class_coverage_fraction": 0.8,
        "candidate_target_class_min_ratio": 0.05,
    }

    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components={
            **_base_metrics(),
            **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
        },
        post_export_quality=receiver,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is False
    assert (
        "hi_nerv_short_smoke_receiver_cache_segnet_target_class_mass_collapsed"
        in report["actionable_blockers"]
    )


def test_direct_live_pose_enabled_requires_live_yuv6_score_metrics() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(pose_direct_live_distillation_weight=0.6),
        final_loss_components={
            **_base_metrics(),
            **_dual_metrics("hi_nerv_posenet_yuv6_pair_distill"),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=0.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is False
    assert "hi_nerv_short_smoke_missing_direct_live_posenet_telemetry" in report[
        "actionable_blockers"
    ]


def test_direct_live_only_segnet_requires_dual_update_telemetry() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components=_base_metrics(),
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=0.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is False
    assert report["teacher_gate"]["direct_live_segnet_only"] is True
    assert "hi_nerv_short_smoke_missing_direct_live_dual_ascent_telemetry" in report[
        "actionable_blockers"
    ]


def test_direct_live_only_segnet_blocks_fake_dual_weight_not_applied() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components={
            **_base_metrics(),
            **_dual_metrics(
                "hi_nerv_segnet_direct_live_distill",
                weight_applied=0.0,
                effective_loss_weight=0.0,
            ),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=0.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is False
    assert "hi_nerv_short_smoke_direct_live_dual_ascent_weight_not_applied" in report[
        "actionable_blockers"
    ]


def test_section_byte_metrics_require_dual_actuation_before_long_run() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components={
            **_base_metrics(),
            "train_time_archive_rate_score": 0.22,
            "train_time_section_rate_score__decoder_state": 0.18,
            **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is False
    gate = report["section_byte_dual_ascent_gate"]
    assert gate["required"] is True
    assert gate["archive_metric_present"] is True
    assert gate["section_constraint_count"] == 1
    assert set(gate["missing_constraint_telemetry"]) == {
        "hi_nerv_archive_total_bytes",
        "hi_nerv_decoder_state_section_bytes",
    }
    assert "hi_nerv_short_smoke_missing_section_byte_dual_ascent_telemetry" in report[
        "actionable_blockers"
    ]


def test_section_byte_duals_can_clear_long_run_readiness() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components={
            **_base_metrics(),
            "train_time_archive_rate_score": 0.22,
            "train_time_section_rate_score__decoder_state": 0.18,
            **_dual_metrics(
                "hi_nerv_segnet_direct_live_distill",
                "hi_nerv_archive_total_bytes",
                "hi_nerv_decoder_state_section_bytes",
            ),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is True
    gate = report["section_byte_dual_ascent_gate"]
    assert gate["required"] is True
    assert gate["section_or_archive_metric_present"] is True
    assert gate["missing_constraint_telemetry"] == []
    assert gate["constraints_without_updates"] == []
    assert gate["constraints_without_applied_weight"] == []


def test_section_byte_duals_accept_slack_kkt_zero_pressure() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components={
            **_base_metrics(),
            "train_time_archive_rate_score": 0.08,
            "train_time_section_rate_score__decoder_state": 0.05,
            **_dual_metrics(
                "hi_nerv_segnet_direct_live_distill",
                weight_applied=1.0,
                effective_loss_weight=0.5,
            ),
            **_dual_metrics(
                "hi_nerv_archive_total_bytes",
                "hi_nerv_decoder_state_section_bytes",
                weight_applied=0.0,
                effective_loss_weight=0.0,
                lambda_value=0.0,
                violation=-0.02,
            ),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is True
    gate = report["section_byte_dual_ascent_gate"]
    assert gate["constraints_without_applied_weight"] == []
    assert set(gate["slack_constraints_without_applied_weight"]) == {
        "hi_nerv_archive_total_bytes",
        "hi_nerv_decoder_state_section_bytes",
    }


def test_section_byte_duals_still_block_positive_violation_without_pressure() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components={
            **_base_metrics(),
            "train_time_archive_rate_score": 0.22,
            "train_time_section_rate_score__decoder_state": 0.18,
            **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
            **_dual_metrics(
                "hi_nerv_archive_total_bytes",
                "hi_nerv_decoder_state_section_bytes",
                weight_applied=0.0,
                effective_loss_weight=0.0,
                lambda_value=0.0,
                violation=0.02,
            ),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is False
    gate = report["section_byte_dual_ascent_gate"]
    assert set(gate["constraints_without_applied_weight"]) == {
        "hi_nerv_archive_total_bytes",
        "hi_nerv_decoder_state_section_bytes",
    }
    assert "hi_nerv_short_smoke_section_byte_dual_ascent_weight_not_applied" in report[
        "actionable_blockers"
    ]


def test_section_byte_duals_accept_active_coder_qat_pressure() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components={
            **_base_metrics(),
            "train_time_archive_rate_score": 0.22,
            "train_time_section_rate_score__decoder_state": 0.18,
            "active_loss_weight__coder_qat_c1a_entropy": 0.0003,
            "active_loss_weight_positive__coder_qat_c1a_entropy": 1.0,
            "loss_part_coder_qat_c1a_entropy": 6.0,
            "loss_part_weighted_coder_qat_c1a_entropy": 0.0018,
            **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
            **_dual_metrics(
                "hi_nerv_archive_total_bytes",
                "hi_nerv_decoder_state_section_bytes",
                weight_applied=0.0,
                effective_loss_weight=0.0,
                lambda_value=0.04,
                violation=0.02,
            ),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is True
    gate = report["section_byte_dual_ascent_gate"]
    assert gate["constraints_without_applied_weight"] == []
    assert set(gate["constraints_with_active_loss_pressure"]) == {
        "hi_nerv_archive_total_bytes",
        "hi_nerv_decoder_state_section_bytes",
    }


def test_section_byte_gate_tracks_priced_only_packet_sections_without_blocking() -> None:
    report = build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_controls(),
        final_loss_components={
            **_base_metrics(),
            "train_time_archive_rate_score": 0.08,
            "train_time_section_rate_score__decoder_state": 0.05,
            "train_time_section_rate_score__hiv1_header": 0.00002,
            "train_time_section_rate_score__meta_json": 0.0004,
            **_dual_metrics("hi_nerv_segnet_direct_live_distill"),
            **_dual_metrics(
                "hi_nerv_archive_total_bytes",
                "hi_nerv_decoder_state_section_bytes",
                weight_applied=0.0,
                effective_loss_weight=0.0,
                lambda_value=0.0,
                violation=-0.02,
            ),
        },
        post_export_quality=_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
    )

    assert report["ready_for_long_run"] is True
    gate = report["section_byte_dual_ascent_gate"]
    assert set(gate["priced_only_constraints"]) == {
        "hi_nerv_hiv1_header_section_bytes",
        "hi_nerv_meta_json_section_bytes",
    }
    assert "hi_nerv_short_smoke_missing_section_byte_dual_ascent_telemetry" not in report[
        "actionable_blockers"
    ]
