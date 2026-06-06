# SPDX-License-Identifier: MIT
"""Tests for train-time dual-ascent score/rate controls."""

from __future__ import annotations

import pytest

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES
from tac.substrates._shared.mlx_score_aware.dual_ascent import (
    CONTEST_RATE_SCORE_PER_BYTE,
    TRAIN_TIME_DUAL_ASCENT_SCHEMA,
    TrainTimeDualAscentController,
    TrainTimeDualAscentError,
    build_default_nerv_train_time_dual_ascent_config,
)


def test_train_time_byte_price_uses_auth_eval_schema_denominator() -> None:
    assert pytest.approx(CONTEST_RATE_SCORE_PER_BYTE) == (
        25.0 / float(ORIGINAL_VIDEO_BYTES)
    )


def test_dual_ascent_bootstraps_relative_target_and_updates_weights() -> None:
    controller = TrainTimeDualAscentController.from_config(
        {
            "schema": TRAIN_TIME_DUAL_ASCENT_SCHEMA,
            "enabled": True,
            "constraints": [
                {
                    "constraint_id": "seg_distill",
                    "metric_name": "loss_part_distill",
                    "loss_weight_key": "distill",
                    "target_fraction_of_initial": 0.5,
                    "dual_lr": 2.0,
                    "max_lambda": 10.0,
                }
            ],
        }
    )

    assert controller.effective_loss_weights({"distill": 1.0}) == {"distill": 1.0}
    first = controller.observe({"loss_part_distill": 4.0})
    assert first["dual_ascent_target__seg_distill"] == pytest.approx(2.0)
    assert first["dual_ascent_lambda__seg_distill"] == pytest.approx(0.0)

    second = controller.observe({"loss_part_distill": 3.0})
    assert second["dual_ascent_violation__seg_distill"] == pytest.approx(1.0)
    assert second["dual_ascent_lambda__seg_distill"] == pytest.approx(2.0)
    assert controller.effective_loss_weights({"distill": 1.0})["distill"] == pytest.approx(3.0)


def test_dual_ascent_ratcheting_tightens_after_satisfied_constraint() -> None:
    controller = TrainTimeDualAscentController.from_config(
        {
            "enabled": True,
            "constraints": [
                {
                    "constraint_id": "pose",
                    "metric_name": "loss_part_pose_distill",
                    "loss_weight_key": "pose_distill",
                    "target": 10.0,
                    "target_ratchet_fraction": 0.9,
                    "dual_lr": 1.0,
                    "initial_lambda": 2.0,
                    "max_lambda": 4.0,
                }
            ],
        }
    )

    metrics = controller.observe({"loss_part_pose_distill": 8.0})
    assert metrics["dual_ascent_target__pose"] == pytest.approx(10.0)
    assert metrics["dual_ascent_next_target__pose"] == pytest.approx(7.2)
    assert metrics["dual_ascent_violation__pose"] == pytest.approx(-2.0)
    assert metrics["dual_ascent_lambda__pose"] == pytest.approx(0.0)
    assert controller.as_metadata()["state"]["pose"]["target"] == pytest.approx(7.2)


def test_dual_ascent_projects_lambda_down_when_constraint_satisfied() -> None:
    controller = TrainTimeDualAscentController.from_config(
        {
            "enabled": True,
            "constraints": [
                {
                    "constraint_id": "rate_proxy",
                    "metric_name": "loss_part_coder_qat_c1a_entropy",
                    "loss_weight_key": "coder_qat_c1a_entropy",
                    "target": 1.0,
                    "dual_lr": 0.5,
                    "initial_lambda": 2.0,
                    "weight_scale": 0.001,
                }
            ],
        }
    )

    metrics = controller.observe({"loss_part_coder_qat_c1a_entropy": 0.25})
    assert metrics["dual_ascent_lambda__rate_proxy"] == pytest.approx(1.625)
    assert controller.effective_loss_weights({})["coder_qat_c1a_entropy"] == pytest.approx(
        0.001625
    )


def test_dual_ascent_respects_explicit_zero_curriculum_stage_masks() -> None:
    controller = TrainTimeDualAscentController.from_config(
        {
            "enabled": True,
            "constraints": [
                {
                    "constraint_id": "pose",
                    "metric_name": "loss_part_pose_score_term",
                    "loss_weight_key": "pose_distill",
                    "target": 0.0,
                    "dual_lr": 1.0,
                    "initial_lambda": 3.0,
                }
            ],
        }
    )

    assert controller.effective_loss_weights({"pose_distill": 0.0})[
        "pose_distill"
    ] == pytest.approx(0.0)
    assert controller.effective_loss_weights({"pose_distill": 0.25})[
        "pose_distill"
    ] == pytest.approx(3.25)
    assert controller.effective_loss_weights({})["pose_distill"] == pytest.approx(3.0)


def test_dual_ascent_byte_constraints_can_activate_zero_base_qat_weight() -> None:
    controller = TrainTimeDualAscentController.from_config(
        {
            "enabled": True,
            "constraints": [
                {
                    "constraint_id": "archive_total_bytes",
                    "metric_name": "train_time_archive_rate_score",
                    "loss_weight_key": "coder_qat_quant_residual",
                    "target": 0.0,
                    "initial_lambda": 2.0,
                    "weight_scale": 0.25,
                    "activate_when_base_weight_zero": True,
                }
            ],
        }
    )

    weights = controller.effective_loss_weights({"coder_qat_quant_residual": 0.0})
    assert weights["coder_qat_quant_residual"] == pytest.approx(0.5)

    metrics = controller.observe({"train_time_archive_rate_score": 1.0})
    assert metrics["dual_ascent_weight_contribution__archive_total_bytes"] == (
        pytest.approx(0.5)
    )
    assert metrics["dual_ascent_effective_loss_weight__archive_total_bytes"] == (
        pytest.approx(0.5)
    )
    assert metrics["dual_ascent_weight_applied__archive_total_bytes"] == pytest.approx(
        1.0
    )
    assert metrics["dual_ascent_zero_base_masked__archive_total_bytes"] == (
        pytest.approx(0.0)
    )


def test_dual_ascent_rejects_missing_target() -> None:
    with pytest.raises(TrainTimeDualAscentError, match="needs target"):
        TrainTimeDualAscentController.from_config(
            {
                "enabled": True,
                "constraints": [
                    {
                        "constraint_id": "bad",
                        "metric_name": "loss_part_distill",
                        "loss_weight_key": "distill",
                    }
                ],
            }
        )


def test_dual_ascent_metadata_strips_nested_authority_keys() -> None:
    controller = TrainTimeDualAscentController.from_config(
        {
            "schema": TRAIN_TIME_DUAL_ASCENT_SCHEMA,
            "enabled": True,
            "score_claim": False,
            "promotion_eligible": False,
            "constraints": [
                {
                    "constraint_id": "seg_distill",
                    "metric_name": "loss_part_distill",
                    "loss_weight_key": "distill",
                    "target": 1.0,
                }
            ],
            "contest_grounding": {
                "score_claim": False,
                "nested": {
                    "ready_for_exact_eval_dispatch": False,
                    "safe_note": "metadata survives",
                },
            },
        }
    )

    metadata = controller.as_metadata()["metadata"]
    assert "score_claim" not in metadata
    assert "promotion_eligible" not in metadata
    assert "score_claim" not in metadata["contest_grounding"]
    assert "ready_for_exact_eval_dispatch" not in metadata["contest_grounding"]["nested"]
    assert metadata["contest_grounding"]["nested"]["safe_note"] == "metadata survives"


def test_default_nerv_dual_ascent_config_prices_active_scorer_and_coder_terms() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="hi_nerv",
        segnet_distillation_weight=8.0,
        pose_distillation_weight=1.0,
        coder_qat_loss_weight_map={
            "coder_qat_quant_residual": 0.001,
            "coder_qat_magnitude": 0.0,
        },
    )

    assert config["enabled"] is True
    constraints = {row["loss_weight_key"]: row for row in config["constraints"]}
    assert set(constraints) == {
        "distill",
        "pose_distill",
        "coder_qat_quant_residual",
    }
    assert constraints["distill"]["metric_name"] == "loss_part_distill"
    assert constraints["pose_distill"]["metric_name"] == "loss_part_pose_score_term"
    assert constraints["coder_qat_quant_residual"]["metric_name"] == (
        "loss_part_coder_qat_quant_residual"
    )
    assert constraints["distill"]["weight_scale"] == pytest.approx(8.0)
    assert constraints["coder_qat_quant_residual"]["weight_scale"] == pytest.approx(
        0.001
    )
    assert config["score_claim"] is False
    assert config["promotion_eligible"] is False


def test_default_nerv_scorer_duals_update_on_bootstrap_without_rate_guess() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="hi_nerv",
        segnet_distillation_weight=1.0,
        segnet_direct_live_distillation_weight=1.0,
        segnet_direct_live_class_balanced_hinge_weight=1.0,
        pose_distillation_weight=1.0,
        pose_direct_live_distillation_weight=1.0,
        scorer_input_distribution_guard_weight=1.0,
        coder_qat_loss_weight_map={"coder_qat_quant_residual": 1.0},
    )
    constraints = {row["constraint_id"]: row for row in config["constraints"]}

    scorer_ids = {
        "hi_nerv_segnet_last_frame_distill",
        "hi_nerv_segnet_direct_live_distill",
        "hi_nerv_segnet_direct_live_class_balanced_hinge",
        "hi_nerv_posenet_yuv6_pair_distill",
        "hi_nerv_posenet_yuv6_pair_direct_live_distill",
        "hi_nerv_scorer_input_distribution_guard",
    }
    for constraint_id in scorer_ids:
        assert constraints[constraint_id]["bootstrap_update"] is True
    assert constraints["hi_nerv_coder_qat_quant_residual"]["bootstrap_update"] is False

    controller = TrainTimeDualAscentController.from_config(config)
    metrics = controller.observe(
        {
            "loss_part_distill": 10.0,
            "loss_part_segnet_direct_live_distill": 20.0,
            "loss_part_segnet_direct_live_class_balanced_hinge_loss": 30.0,
            "loss_part_pose_score_term": 40.0,
            "loss_part_pose_direct_live_score_term": 50.0,
            "loss_part_scorer_input_distribution_guard": 60.0,
            "loss_part_coder_qat_quant_residual": 70.0,
        }
    )

    for constraint_id in scorer_ids:
        key = constraint_id
        assert metrics[f"dual_ascent_update_count__{key}"] == pytest.approx(1.0)
        assert metrics[f"dual_ascent_lambda__{key}"] > 0.0
    assert (
        metrics["dual_ascent_update_count__hi_nerv_coder_qat_quant_residual"]
        == pytest.approx(0.0)
    )
    assert (
        metrics["dual_ascent_lambda__hi_nerv_coder_qat_quant_residual"]
        == pytest.approx(0.0)
    )


def test_default_nerv_dual_ascent_config_prices_direct_live_segnet_term() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="hi_nerv",
        segnet_distillation_weight=0.0,
        segnet_direct_live_distillation_weight=3.0,
        segnet_direct_live_class_histogram_weight=2.0,
        segnet_direct_live_class_balanced_hinge_weight=5.0,
        segnet_direct_live_class_balanced_ce_weight=7.0,
        segnet_direct_live_class_balanced_squared_hinge_weight=11.0,
        segnet_direct_live_class_region_recon_weight=13.0,
        segnet_direct_live_rare_class_logit_weight=17.0,
        segnet_direct_live_target_mass_floor_weight=19.0,
        segnet_direct_live_target_min_ratio_floor_weight=23.0,
    )

    assert config["enabled"] is True
    constraints = {row["constraint_id"]: row for row in config["constraints"]}
    direct = constraints["hi_nerv_segnet_direct_live_distill"]
    assert direct["metric_name"] == "loss_part_segnet_direct_live_distill"
    assert direct["loss_weight_key"] == "segnet_direct_live_distill"
    assert direct["weight_scale"] == pytest.approx(3.0)
    hist = constraints["hi_nerv_segnet_direct_live_class_histogram"]
    assert hist["metric_name"] == (
        "loss_part_segnet_direct_live_class_histogram_loss"
    )
    assert hist["loss_weight_key"] == "segnet_direct_live_class_histogram"
    assert hist["weight_scale"] == pytest.approx(2.0)
    balanced = constraints["hi_nerv_segnet_direct_live_class_balanced_hinge"]
    assert balanced["metric_name"] == (
        "loss_part_segnet_direct_live_class_balanced_hinge_loss"
    )
    assert balanced["loss_weight_key"] == "segnet_direct_live_class_balanced_hinge"
    assert balanced["weight_scale"] == pytest.approx(5.0)
    balanced_ce = constraints["hi_nerv_segnet_direct_live_class_balanced_ce"]
    assert balanced_ce["metric_name"] == (
        "loss_part_segnet_direct_live_class_balanced_ce_loss"
    )
    assert balanced_ce["loss_weight_key"] == "segnet_direct_live_class_balanced_ce"
    assert balanced_ce["weight_scale"] == pytest.approx(7.0)
    squared_hinge = constraints[
        "hi_nerv_segnet_direct_live_class_balanced_squared_hinge"
    ]
    assert squared_hinge["metric_name"] == (
        "loss_part_segnet_direct_live_class_balanced_squared_hinge_loss"
    )
    assert squared_hinge["loss_weight_key"] == (
        "segnet_direct_live_class_balanced_squared_hinge"
    )
    assert squared_hinge["weight_scale"] == pytest.approx(11.0)
    region_recon = constraints["hi_nerv_segnet_direct_live_class_region_recon"]
    assert region_recon["metric_name"] == (
        "loss_part_segnet_direct_live_class_region_recon_loss"
    )
    assert region_recon["loss_weight_key"] == (
        "segnet_direct_live_class_region_recon"
    )
    assert region_recon["weight_scale"] == pytest.approx(13.0)
    rare_class = constraints["hi_nerv_segnet_direct_live_rare_class_logit"]
    assert rare_class["metric_name"] == (
        "loss_part_segnet_direct_live_rare_class_logit_loss"
    )
    assert rare_class["loss_weight_key"] == "segnet_direct_live_rare_class_logit"
    assert rare_class["weight_scale"] == pytest.approx(17.0)
    target_mass = constraints["hi_nerv_segnet_direct_live_target_mass_floor"]
    assert target_mass["metric_name"] == (
        "loss_part_segnet_direct_live_target_mass_floor_loss"
    )
    assert target_mass["loss_weight_key"] == "segnet_direct_live_target_mass_floor"
    assert target_mass["weight_scale"] == pytest.approx(19.0)
    target_min = constraints["hi_nerv_segnet_direct_live_target_min_ratio_floor"]
    assert target_min["metric_name"] == (
        "loss_part_segnet_direct_live_target_min_ratio_floor_loss"
    )
    assert target_min["loss_weight_key"] == (
        "segnet_direct_live_target_min_ratio_floor"
    )
    assert target_min["weight_scale"] == pytest.approx(23.0)
    target_min_gate = constraints[
        "hi_nerv_segnet_direct_live_target_min_ratio_floor_gate"
    ]
    assert target_min_gate["metric_name"] == (
        "loss_part_segnet_direct_live_candidate_target_class_min_ratio"
    )
    assert target_min_gate["loss_weight_key"] == (
        "segnet_direct_live_target_min_ratio_floor"
    )
    assert target_min_gate["direction"] == "lower_bound"
    assert target_min_gate["target"] == pytest.approx(0.35)
    argmax = constraints["hi_nerv_segnet_direct_live_argmax_disagreement"]
    assert argmax["metric_name"] == (
        "loss_part_segnet_direct_live_argmax_disagreement"
    )
    assert argmax["loss_weight_key"] == "segnet_direct_live_distill"
    assert argmax["target"] == pytest.approx(0.15)
    assert argmax["direction"] == "upper_bound"
    hist_missing = constraints[
        "hi_nerv_segnet_direct_live_target_missing_fraction_histogram"
    ]
    assert hist_missing["metric_name"] == (
        "loss_part_segnet_direct_live_candidate_target_class_missing_fraction"
    )
    assert hist_missing["loss_weight_key"] == "segnet_direct_live_class_histogram"
    assert hist_missing["target"] == pytest.approx(0.0)
    assert hist_missing["direction"] == "upper_bound"
    ce_missing = constraints[
        "hi_nerv_segnet_direct_live_target_missing_fraction_ce"
    ]
    assert ce_missing["metric_name"] == (
        "loss_part_segnet_direct_live_candidate_target_class_missing_fraction"
    )
    assert ce_missing["loss_weight_key"] == "segnet_direct_live_class_balanced_ce"
    assert ce_missing["target"] == pytest.approx(0.0)
    region_min_ratio = constraints[
        "hi_nerv_segnet_direct_live_target_min_ratio_region_recon"
    ]
    assert region_min_ratio["metric_name"] == (
        "loss_part_segnet_direct_live_candidate_target_class_min_ratio"
    )
    assert region_min_ratio["loss_weight_key"] == (
        "segnet_direct_live_class_region_recon"
    )
    assert region_min_ratio["direction"] == "lower_bound"
    assert region_min_ratio["target"] == pytest.approx(0.35)
    rare_min_ratio = constraints[
        "hi_nerv_segnet_direct_live_target_min_ratio_rare_class_logit"
    ]
    assert rare_min_ratio["metric_name"] == (
        "loss_part_segnet_direct_live_candidate_target_class_min_ratio"
    )
    assert rare_min_ratio["loss_weight_key"] == "segnet_direct_live_rare_class_logit"
    assert rare_min_ratio["direction"] == "lower_bound"
    assert rare_min_ratio["target"] == pytest.approx(0.35)
    target_mass_min_ratio = constraints[
        "hi_nerv_segnet_direct_live_target_min_ratio_mass_floor"
    ]
    assert target_mass_min_ratio["metric_name"] == (
        "loss_part_segnet_direct_live_candidate_target_class_min_ratio"
    )
    assert target_mass_min_ratio["loss_weight_key"] == (
        "segnet_direct_live_target_mass_floor"
    )
    assert target_mass_min_ratio["direction"] == "lower_bound"
    assert target_mass_min_ratio["target"] == pytest.approx(0.35)


def test_direct_live_dual_ascent_prices_bounded_segnet_geometry() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="hi_nerv",
        segnet_direct_live_distillation_weight=1.0,
        segnet_direct_live_class_histogram_weight=1.0,
        segnet_direct_live_class_balanced_ce_weight=1.0,
        segnet_direct_live_class_region_recon_weight=1.0,
        segnet_direct_live_rare_class_logit_weight=1.0,
        segnet_direct_live_target_mass_floor_weight=1.0,
        segnet_direct_live_target_min_ratio_floor_weight=1.0,
    )
    controller = TrainTimeDualAscentController.from_config(config)

    metrics = controller.observe(
        {
            "loss_part_segnet_direct_live_distill": 10.0,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.52,
            "loss_part_segnet_direct_live_class_histogram_loss": 2.0,
            "loss_part_segnet_direct_live_class_balanced_ce_loss": 3.0,
            "loss_part_segnet_direct_live_class_region_recon_loss": 0.5,
            "loss_part_segnet_direct_live_rare_class_logit_loss": 100.0,
            "loss_part_segnet_direct_live_target_mass_floor_loss": 12.0,
            "loss_part_segnet_direct_live_target_min_ratio_floor_loss": 13.0,
            "loss_part_segnet_direct_live_candidate_target_class_missing_fraction": (
                0.20
            ),
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.0,
        }
    )
    weights = controller.effective_loss_weights(
        {
            "segnet_direct_live_distill": 1.0,
            "segnet_direct_live_class_histogram": 1.0,
            "segnet_direct_live_class_balanced_ce": 1.0,
            "segnet_direct_live_class_region_recon": 1.0,
            "segnet_direct_live_rare_class_logit": 1.0,
            "segnet_direct_live_target_mass_floor": 1.0,
            "segnet_direct_live_target_min_ratio_floor": 1.0,
        }
    )

    assert metrics[
        "dual_ascent_violation__hi_nerv_segnet_direct_live_argmax_disagreement"
    ] == pytest.approx(0.37)
    assert metrics[
        "dual_ascent_lambda__hi_nerv_segnet_direct_live_argmax_disagreement"
    ] > 0.0
    assert metrics[
        "dual_ascent_violation__hi_nerv_segnet_direct_live_target_min_ratio_rare_class_logit"
    ] == pytest.approx(0.35)
    assert metrics[
        "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_min_ratio_rare_class_logit"
    ] > 0.0
    assert metrics[
        "dual_ascent_violation__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor"
    ] == pytest.approx(0.35)
    assert metrics[
        "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor"
    ] > 0.0
    assert metrics[
        "dual_ascent_violation__hi_nerv_segnet_direct_live_target_min_ratio_floor"
    ] > 0.0
    assert metrics[
        "dual_ascent_violation__hi_nerv_segnet_direct_live_target_min_ratio_floor_gate"
    ] == pytest.approx(0.35)
    assert weights["segnet_direct_live_distill"] > 1.0
    assert weights["segnet_direct_live_class_histogram"] > 1.0
    assert weights["segnet_direct_live_class_balanced_ce"] > 1.0
    assert weights["segnet_direct_live_class_region_recon"] > 1.0
    assert weights["segnet_direct_live_rare_class_logit"] > 1.0
    assert weights["segnet_direct_live_target_mass_floor"] > 1.0
    assert weights["segnet_direct_live_target_min_ratio_floor"] > 1.0


def test_direct_live_dual_ascent_survives_generic_distill_zero() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="hi_nerv",
        segnet_direct_live_distillation_weight=1.0,
    )
    controller = TrainTimeDualAscentController.from_config(config)

    controller.observe({"loss_part_segnet_direct_live_distill": 4.0})
    controller.observe({"loss_part_segnet_direct_live_distill": 5.0})
    weights = controller.effective_loss_weights({"distill": 0.0})

    assert weights["distill"] == pytest.approx(0.0)
    assert weights["segnet_direct_live_distill"] > 0.0


def test_default_nerv_dual_ascent_config_prices_direct_live_pose_term() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="snerv",
        pose_distillation_weight=0.0,
        pose_direct_live_distillation_weight=2.5,
    )

    constraints = {row["constraint_id"]: row for row in config["constraints"]}
    pose = constraints["snerv_posenet_yuv6_pair_distill"]
    assert pose["metric_name"] == "loss_part_pose_direct_live_score_term"
    assert pose["loss_weight_key"] == "pose_direct_live_distill"
    assert pose["weight_scale"] == pytest.approx(2.5)

    controller = TrainTimeDualAscentController.from_config(config)
    controller.observe({"loss_part_pose_direct_live_score_term": 4.0})
    metrics = controller.observe({"loss_part_pose_direct_live_score_term": 5.0})
    weights = controller.effective_loss_weights({"pose_distill": 0.0})

    assert (
        metrics["dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill"]
        == pytest.approx(0.0)
    )
    assert (
        metrics["dual_ascent_metric__snerv_posenet_yuv6_pair_distill"]
        == pytest.approx(5.0)
    )
    assert weights["pose_distill"] == pytest.approx(0.0)
    assert weights["pose_direct_live_distill"] > 0.0


def test_default_nerv_dual_ascent_config_prices_contrast_floor_guard() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="snerv",
        scorer_input_contrast_floor_weight=0.25,
    )

    assert config["enabled"] is True
    constraints = {row["constraint_id"]: row for row in config["constraints"]}
    floor = constraints["snerv_scorer_input_contrast_floor"]
    assert floor["metric_name"] == "loss_part_scorer_input_contrast_floor"
    assert floor["loss_weight_key"] == "scorer_input_contrast_floor"
    assert floor["target"] == pytest.approx(0.0)
    assert floor["target_fraction_of_initial"] is None
    assert floor["weight_scale"] == pytest.approx(0.25)
    assert "evaluate.py" in floor["rationale"]


def test_default_nerv_dual_ascent_config_prices_distribution_guard() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="hi_nerv",
        scorer_input_distribution_guard_weight=0.75,
    )

    assert config["enabled"] is True
    constraints = {row["constraint_id"]: row for row in config["constraints"]}
    guard = constraints["hi_nerv_scorer_input_distribution_guard"]
    assert guard["metric_name"] == "loss_part_scorer_input_distribution_guard"
    assert guard["loss_weight_key"] == "scorer_input_guard"
    assert guard["target_fraction_of_initial"] == pytest.approx(0.97)
    assert guard["weight_scale"] == pytest.approx(0.75)
    assert "PoseNet YUV6 pair/temporal-delta" in guard["rationale"]
    assert "class-collapsed scorer basin" in guard["rationale"]


def test_default_nerv_dual_ascent_config_prices_shape_tether_guard() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="hi_nerv",
        scorer_input_shape_tether_weight=1.5,
    )

    assert config["enabled"] is True
    constraints = {row["constraint_id"]: row for row in config["constraints"]}
    tether = constraints["hi_nerv_scorer_input_shape_tether"]
    assert tether["metric_name"] == "loss_part_scorer_input_shape_tether"
    assert tether["loss_weight_key"] == "scorer_input_shape_tether"
    assert tether["target_fraction_of_initial"] == pytest.approx(0.97)
    assert tether["weight_scale"] == pytest.approx(1.5)
    assert "PoseNet YUV6" in tether["rationale"]


def test_default_nerv_dual_ascent_config_prices_posenet_temporal_signal_floor() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="hi_nerv",
        posenet_temporal_signal_floor_weight=2.25,
    )

    assert config["enabled"] is True
    constraints = {row["constraint_id"]: row for row in config["constraints"]}
    floor = constraints["hi_nerv_posenet_temporal_signal_floor"]
    assert floor["metric_name"] == "loss_part_posenet_temporal_signal_floor"
    assert floor["loss_weight_key"] == "posenet_temporal_signal_floor"
    assert floor["target"] == pytest.approx(0.0)
    assert floor["target_fraction_of_initial"] is None
    assert floor["weight_scale"] == pytest.approx(2.25)
    assert "frame_1-frame_0" in floor["rationale"]
    assert "YUV6 temporal delta" in floor["rationale"]


def test_default_nerv_dual_ascent_config_prices_posenet_yuv6_geometry_tether() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="hi_nerv",
        posenet_yuv6_geometry_tether_weight=1.75,
    )

    assert config["enabled"] is True
    constraints = {row["constraint_id"]: row for row in config["constraints"]}
    tether = constraints["hi_nerv_posenet_yuv6_geometry_tether"]
    assert tether["metric_name"] == "loss_part_posenet_yuv6_geometry_tether"
    assert tether["loss_weight_key"] == "posenet_yuv6_geometry_tether"
    assert tether["target_fraction_of_initial"] == pytest.approx(0.97)
    assert tether["weight_scale"] == pytest.approx(1.75)
    assert "spatial-gradient" in tether["rationale"]
    assert "temporally nonflat" in tether["rationale"]


def test_default_nerv_dual_ascent_config_prices_section_byte_budgets() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="snerv",
        archive_byte_budget=2_500,
        archive_byte_loss_weight_key="coder_qat_delta",
        archive_byte_loss_weight_scale=0.5,
        section_byte_budgets={"decoder_payload": 1_000, "lf payload": 2_000},
        section_byte_loss_weight_key_map={"decoder_payload": "coder_qat_delta"},
        section_byte_loss_weight_scale_map={"decoder_payload": 0.25},
    )

    assert config["enabled"] is True
    constraints = {row["constraint_id"]: row for row in config["constraints"]}
    archive = constraints["snerv_archive_total_bytes"]
    decoder = constraints["snerv_decoder_payload_section_bytes"]
    lf = constraints["snerv_lf_payload_section_bytes"]
    byte_price = config["contest_grounding"][
        "archive_byte_price_score_per_byte"
    ]
    assert archive["metric_name"] == "train_time_archive_rate_score"
    assert archive["loss_weight_key"] == "coder_qat_delta"
    assert archive["target"] == pytest.approx(2_500 * byte_price)
    assert archive["weight_scale"] == pytest.approx(0.5)
    assert archive["activate_when_base_weight_zero"] is True
    assert decoder["metric_name"] == (
        "train_time_section_rate_score__decoder_payload"
    )
    assert decoder["loss_weight_key"] == "coder_qat_delta"
    assert decoder["target"] == pytest.approx(1_000 * byte_price)
    assert decoder["weight_scale"] == pytest.approx(0.25)
    assert decoder["activate_when_base_weight_zero"] is True
    assert lf["metric_name"] == "train_time_section_rate_score__lf_payload"
    assert lf["loss_weight_key"] == "coder_qat_c1a_entropy"
    assert lf["target"] == pytest.approx(2_000 * byte_price)
    assert lf["activate_when_base_weight_zero"] is True


def test_default_nerv_dual_ascent_archive_budget_uses_active_section_qat_key() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="hi_nerv",
        coder_qat_loss_weight_map={"coder_qat_quant_residual": 0.25},
        archive_byte_budget=10_000,
        section_byte_budgets={"decoder_state": 4_000},
        section_byte_loss_weight_key_map={
            "decoder_state": "coder_qat_quant_residual"
        },
    )

    constraints = {row["constraint_id"]: row for row in config["constraints"]}
    archive = constraints["hi_nerv_archive_total_bytes"]
    assert archive["loss_weight_key"] == "coder_qat_quant_residual"


def test_default_nerv_dual_ascent_section_budget_prefers_active_qat_key() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="snerv",
        coder_qat_loss_weight_map={
            "coder_qat_quant_residual": 1.0e-3,
            "coder_qat_c1a_entropy": 0.0,
        },
        archive_byte_budget=10_000,
        section_byte_budgets={"lf_payload": 4_000},
    )

    constraints = {row["constraint_id"]: row for row in config["constraints"]}
    archive = constraints["snerv_archive_total_bytes"]
    lf = constraints["snerv_lf_payload_section_bytes"]
    assert archive["loss_weight_key"] == "coder_qat_quant_residual"
    assert lf["loss_weight_key"] == "coder_qat_quant_residual"


def test_default_nerv_dual_ascent_config_disables_without_active_terms() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="snerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        coder_qat_loss_weight_map={"coder_qat_delta": 0.0},
    )

    assert config["enabled"] is False
    assert config["constraint_count"] == 0
