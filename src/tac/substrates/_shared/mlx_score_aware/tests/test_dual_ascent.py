# SPDX-License-Identifier: MIT
"""Tests for train-time dual-ascent score/rate controls."""

from __future__ import annotations

import pytest

from tac.substrates._shared.mlx_score_aware.dual_ascent import (
    TRAIN_TIME_DUAL_ASCENT_SCHEMA,
    TrainTimeDualAscentController,
    TrainTimeDualAscentError,
    build_default_nerv_train_time_dual_ascent_config,
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
    assert metrics["dual_ascent_target__pose"] == pytest.approx(7.2)
    assert metrics["dual_ascent_lambda__pose"] == pytest.approx(2.8)


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
    assert constraints["pose_distill"]["metric_name"] == "loss_part_pose_distill"
    assert constraints["coder_qat_quant_residual"]["metric_name"] == (
        "loss_part_coder_qat_quant_residual"
    )
    assert constraints["distill"]["weight_scale"] == pytest.approx(8.0)
    assert constraints["coder_qat_quant_residual"]["weight_scale"] == pytest.approx(
        0.001
    )
    assert config["score_claim"] is False
    assert config["promotion_eligible"] is False


def test_default_nerv_dual_ascent_config_disables_without_active_terms() -> None:
    config = build_default_nerv_train_time_dual_ascent_config(
        family="snerv",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        coder_qat_loss_weight_map={"coder_qat_delta": 0.0},
    )

    assert config["enabled"] is False
    assert config["constraint_count"] == 0
