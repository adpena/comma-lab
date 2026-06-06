# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

from tac.analysis.action_effect import ACTION_EFFECT_SCHEMA
from tac.substrates._shared.mlx_score_aware import (
    PARSEBACK_SERVO_LIFT_SCHEMA,
    servo_lift,
)


def test_servo_lift_accepts_parseback_surviving_receiver_action() -> None:
    lifted = servo_lift(
        {
            "action_id": "hinerv_pair7_birth_step12",
            "authority": "parseback_mlx",
            "producer": "hinerv_target_region_birth",
            "affected_pairs": [7],
            "affected_regions": ["pair7_c1_region3"],
            "payload_sections": ["head_rgb_1", "latents_fine"],
            "state_custody": {"archive_sha256": "a" * 64},
            "trace_new": {
                "d_seg": 0.009,
                "d_pose": 0.0001,
                "archive_bytes": 1000,
            },
            "receiver_surface": {
                "uint8_changed_pixels": 41,
                "segnet_input_delta_linf": 0.05,
                "argmax_flipped_pixels": 19,
                "fakequant_survival": True,
                "parseback_survival": True,
                "inflate_survival": True,
            },
        },
        {"d_seg": 0.010, "d_pose": 0.0001, "archive_bytes": 1000},
        family="hi_nerv",
        stage="birth_contact",
    )

    effect = lifted["action_effect"]
    assert lifted["schema"] == PARSEBACK_SERVO_LIFT_SCHEMA
    assert lifted["servo_lift_accepted"] is True
    assert lifted["uint8_receiver_contact"] is True
    assert lifted["scorer_surface_motion"] is True
    assert lifted["inflate_survived"] is True
    assert effect["schema"] == ACTION_EFFECT_SCHEMA
    assert effect["family"] == "hinerv"
    assert effect["action_effect_admitted"] is True
    assert effect["receiver_visible"] is True
    assert effect["state_custody"]["archive_sha256"] == "a" * 64
    assert math.isclose(effect["delta_score_total"], -0.1)


def test_servo_lift_rejects_subquantum_continuous_proposal() -> None:
    lifted = servo_lift(
        {
            "action_id": "snerv_lf_subquantum",
            "authority": "live_mlx",
            "producer": "snerv_lf_hf_gate",
            "affected_pairs": [12],
            "state_custody": {"payload_sha256": "b" * 64},
            "old_d_seg": 0.010,
            "new_d_seg": 0.009,
            "old_d_pose": 0.0001,
            "new_d_pose": 0.0001,
            "old_bytes": 1000,
            "new_bytes": 1000,
            "receiver_surface": {
                "float_rgb_delta_linf": 0.01,
                "uint8_changed_pixels": 0,
                "fakequant_survival": False,
                "parseback_survival": False,
            },
        },
        family="snerv",
        stage="birth_contact",
    )

    assert lifted["servo_lift_accepted"] is False
    assert "action_effect_receiver_surface_motion_missing" in lifted["blockers"]
    assert "action_effect_fakequant_survival_missing" in lifted["blockers"]
    assert "action_effect_parseback_survival_missing" in lifted["blockers"]
    assert "servo_lift_uint8_receiver_contact_missing" in lifted["blockers"]
    assert "servo_lift_scorer_surface_motion_missing" in lifted["blockers"]
    assert "servo_lift_inflate_survival_missing" in lifted["blockers"]


def test_servo_lift_rejects_parseback_motion_without_inflate_survival() -> None:
    lifted = servo_lift(
        {
            "action_id": "hinerv_parseback_only",
            "authority": "parseback_mlx",
            "producer": "hinerv_target_region_birth",
            "affected_pairs": [8],
            "state_custody": {"archive_sha256": "c" * 64},
            "old_d_seg": 0.010,
            "new_d_seg": 0.009,
            "old_d_pose": 0.0001,
            "new_d_pose": 0.0001,
            "old_bytes": 1000,
            "new_bytes": 1000,
            "receiver_surface": {
                "uint8_changed_pixels": 5,
                "segnet_input_delta_linf": 0.01,
                "argmax_flipped_pixels": 3,
                "fakequant_survival": True,
                "parseback_survival": True,
            },
        },
        family="hi_nerv",
        stage="parseback_selection",
    )

    assert lifted["action_effect"]["action_effect_admitted"] is True
    assert lifted["servo_lift_accepted"] is False
    assert "servo_lift_inflate_survival_missing" in lifted["blockers"]


def test_servo_lift_imports_from_score_aware_package() -> None:
    assert PARSEBACK_SERVO_LIFT_SCHEMA == "mlx_score_aware_parseback_servo_lift.v1"
    assert callable(servo_lift)
