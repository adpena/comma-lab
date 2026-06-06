# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

from tac.analysis.action_effect import (
    ACTION_COMMUTATOR_ROW_SCHEMA,
    ACTION_EFFECT_LEDGER_SCHEMA,
    ACTION_EFFECT_SCHEMA,
    build_action_commutator_row,
    build_action_effect,
    build_action_effect_ledger,
)


def test_action_effect_prices_exact_nonlinear_score_and_receiver_survival() -> None:
    effect = build_action_effect(
        {
            "action_id": "hinerv_region_birth_pair7",
            "family": "hi_nerv",
            "authority": "parseback_mlx",
            "producer": "hinerv_target_region_birth",
            "consumer": "nerv_long_training_campaign_admission",
            "affected_pairs": [7],
            "affected_regions": ["pair7_class1_region3"],
            "payload_sections": ["decoder_head"],
            "state_custody": {"archive_sha256": "a" * 64},
            "old_d_seg": 0.010,
            "new_d_seg": 0.009,
            "old_d_pose": 0.0001,
            "new_d_pose": 0.0001,
            "old_bytes": 1000,
            "new_bytes": 1000,
            "receiver_surface": {
                "uint8_changed_pixels": 33,
                "segnet_input_delta_linf": 0.04,
                "argmax_flipped_pixels": 18,
            },
            "fakequant_survived": True,
            "parseback_survived": True,
        }
    )

    assert effect["schema"] == ACTION_EFFECT_SCHEMA
    assert effect["family"] == "hinerv"
    assert effect["action_effect_admitted"] is True
    assert effect["receiver_visible"] is True
    assert effect["state_custody"]["archive_sha256"] == "a" * 64
    assert effect["delta_score_total"] == effect["delta_score_nonrate"]
    assert math.isclose(effect["delta_score_total"], -0.1)
    assert effect["score_claim"] is False


def test_action_effect_rejects_subquantum_parseback_lost_byte_growth() -> None:
    effect = build_action_effect(
        {
            "action_id": "bad_live_delta",
            "family": "snerv",
            "authority": "live_mlx",
            "producer": "snerv_lf_hf_gate",
            "old_d_seg": 0.010,
            "new_d_seg": 0.009,
            "old_d_pose": 0.0001,
            "new_d_pose": 0.0001,
            "old_bytes": 1000,
            "new_bytes": 1200,
            "receiver_surface": {"uint8_changed_pixels": 0},
            "fakequant_survived": False,
            "parseback_survived": False,
        }
    )

    assert effect["action_effect_admitted"] is False
    assert "action_effect_receiver_surface_motion_missing" in effect["blockers"]
    assert "action_effect_consumer_missing" in effect["blockers"]
    assert "action_effect_state_custody_hash_missing" in effect["blockers"]
    assert "action_effect_fakequant_survival_missing" in effect["blockers"]
    assert "action_effect_parseback_survival_missing" in effect["blockers"]
    assert "action_effect_byte_delta_not_priced" in effect["blockers"]


def test_action_effect_rejects_metadata_only_missing_score_or_byte_state() -> None:
    effect = build_action_effect(
        {
            "action_id": "metadata_only_surface",
            "family": "hinerv",
            "authority": "parseback_mlx",
            "producer": "unit_test",
            "new_d_seg": 0.009,
            "old_d_pose": 0.0001,
            "new_d_pose": 0.0001,
            "old_bytes": 1000,
            "receiver_surface": {
                "uint8_changed_pixels": 11,
                "argmax_flipped_pixels": 7,
            },
            "fakequant_survived": True,
            "parseback_survived": True,
        }
    )

    assert effect["receiver_visible"] is True
    assert effect["score_admissible"] is False
    assert effect["byte_priced"] is False
    assert effect["action_effect_admitted"] is False
    assert "action_effect_score_state_invalid" in effect["blockers"]
    assert "action_effect_archive_byte_state_invalid" in effect["blockers"]
    assert "action_effect_exact_score_delta_not_admissible" in effect["blockers"]
    assert "action_effect_byte_delta_not_priced" in effect["blockers"]


def test_action_commutator_ledger_promotes_superadditive_macro_action() -> None:
    first = build_action_effect(_effect_payload("frame0_pose", -0.01))
    second = build_action_effect(_effect_payload("frame1_seg", -0.02))
    composed = build_action_effect(_effect_payload("pose_then_seg_macro", -0.05))

    row = build_action_commutator_row(
        first=first,
        second=second,
        composed=composed,
    )
    ledger = build_action_effect_ledger(
        [first, second, composed],
        commutators=[row],
    )

    assert row["schema"] == ACTION_COMMUTATOR_ROW_SCHEMA
    assert row["macro_action_recommended"] is True
    assert math.isclose(row["commutator_delta_score_total"], -0.02)
    assert ledger["schema"] == ACTION_EFFECT_LEDGER_SCHEMA
    assert ledger["effect_count"] == 3
    assert ledger["admitted_effect_count"] == 3
    assert ledger["score_claim"] is False


def _effect_payload(action_id: str, delta_score: float) -> dict[str, object]:
    old_d_seg = 0.01
    new_d_seg = old_d_seg + delta_score / 100.0
    return {
        "action_id": action_id,
        "family": "selector",
        "authority": "parseback_mlx",
        "producer": "unit_test",
        "consumer": "action_commutator_ledger",
        "state_custody": {"payload_sha256": "b" * 64},
        "old_d_seg": old_d_seg,
        "new_d_seg": new_d_seg,
        "old_d_pose": 0.0001,
        "new_d_pose": 0.0001,
        "old_bytes": 1000,
        "new_bytes": 1000,
        "receiver_surface": {"uint8_changed_pixels": 1},
        "fakequant_survived": True,
        "parseback_survived": True,
    }
