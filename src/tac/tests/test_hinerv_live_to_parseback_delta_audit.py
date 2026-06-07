# SPDX-License-Identifier: MIT
"""Tests for HiNeRV live-to-parseback scorer-effect delta audits."""

from __future__ import annotations

from tac.analysis.hinerv_live_to_parseback_delta_audit import (
    HI_NERV_LIVE_TO_PARSEBACK_DELTA_AUDIT_SCHEMA,
    build_hinerv_live_to_parseback_scorer_effect_delta_audit,
)


def test_delta_audit_blocks_payload_survival_without_scorer_effect_retention() -> None:
    row = build_hinerv_live_to_parseback_scorer_effect_delta_audit(
        fakequant_survival={
            "schema": "hi_nerv_target_region_birth_survival.v1",
            "action_id": "a" * 64,
            "survived": True,
            "live_wrong_to_target_count": 13488,
            "fakequant_wrong_to_target_count": 12183,
            "wrong_to_target_count": 12183,
            "target_to_wrong_count": 0,
            "total_scored_pixels": 196608,
            "region_margin_min": -0.5,
            "region_margin_mean": 0.2,
            "scorer_effect_retention_floor": 0.5,
            "blockers": [],
        },
        selected_birth_parseback_survival={
            "schema": "hi_nerv_target_region_birth_survival.v1",
            "action_id": "a" * 64,
            "selected_archive_sha256": "b" * 64,
            "selected_archive_bytes": 100,
            "parseback_payload_survived": True,
            "parseback_scorer_effect_survived": False,
            "parseback_wrong_to_target_count": 2,
            "wrong_to_target_count": 2,
            "target_to_wrong_count": 0,
            "total_scored_pixels": 196608,
            "region_margin_min": -0.01,
            "region_margin_mean": 1.0,
            "blockers": ["hinerv_birth_parseback_scorer_effect_collapse"],
        },
        target_region_action_parseback_survival={
            "schema": "hi_nerv_target_region_action_parseback_survival.v1",
            "action_id": "a" * 64,
            "archive_sha256": "b" * 64,
            "archive_bytes": 100,
            "support_sha256": "c" * 64,
            "expected_support_sha256": "c" * 64,
            "decoded_action_sha256": "d" * 64,
            "parseback_payload_survived": True,
            "parseback_program_survived": True,
            "parseback_survived": True,
            "exact_uint8_action_pixels_applied": 2286,
            "receiver_changed_action_pixels": 2286,
            "max_abs_action_rgb_error": 0.0,
            "blockers": [],
            "score_claim": True,
            "promotion_eligible": True,
            "ready_for_exact_eval_dispatch": True,
        },
        target_region_action_export_selection={
            "schema": "hi_nerv_target_region_action_program_export_selection.v1",
            "selected_for_export": True,
            "action_id": "a" * 64,
            "target_region_action_support_sha256": "c" * 64,
            "target_region_action_section_telemetry": {
                "decoded_action_sha256": "d" * 64,
                "support_cardinality": 2286,
            },
        },
        live_receiver_export_parity={
            "schema": "hi_nerv_mlx_live_receiver_export_parity_proof.v1",
            "passed": False,
            "receiver_decode_passed": True,
            "max_abs_delta": 0.875,
            "mean_abs_delta": 0.0015,
            "changed_element_count": 1179625,
            "live_tensor_sha256": "e" * 64,
            "receiver_tensor_sha256": "f" * 64,
            "blockers": ["sampled_live_receiver_export_parity_not_full_video"],
        },
    )

    assert row["schema"] == HI_NERV_LIVE_TO_PARSEBACK_DELTA_AUDIT_SCHEMA
    assert row["first_divergence"] == "quantization_mismatch"
    assert row["retention"]["fakequant_wrong_to_target_retention_ratio"] == (
        12183 / 13488
    )
    assert row["retention"]["parseback_wrong_to_target_retention_ratio"] == (
        2 / 13488
    )
    assert row["parseback_payload_survived"] is True
    assert row["parseback_program_survived"] is True
    assert row["parseback_scorer_effect_survived"] is False
    assert "hinerv_birth_parseback_scorer_effect_collapse" in row["blockers"]
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["surfaces"][3]["rgb_uint8_delta_on_support"]["max_abs_delta"] == 0.875


def test_delta_audit_catches_archive_selection_action_swap_first() -> None:
    row = build_hinerv_live_to_parseback_scorer_effect_delta_audit(
        fakequant_survival={
            "action_id": "a" * 64,
            "live_wrong_to_target_count": 10,
            "fakequant_wrong_to_target_count": 9,
        },
        selected_birth_parseback_survival={
            "action_id": "b" * 64,
            "parseback_payload_survived": True,
            "parseback_scorer_effect_survived": False,
        },
        target_region_action_parseback_survival={
            "action_id": "a" * 64,
            "support_sha256": "c" * 64,
            "parseback_survived": True,
        },
    )

    assert row["first_divergence"] == "archive_selection_swapped_candidate"
    assert "live_fakequant_parseback_action_id_mismatch" in row["first_divergence_reasons"]
    assert row["score_claim"] is False


def test_delta_audit_catches_support_identity_drift_before_scorer_labels() -> None:
    row = build_hinerv_live_to_parseback_scorer_effect_delta_audit(
        fakequant_survival={
            "action_id": "a" * 64,
            "live_wrong_to_target_count": 10,
            "fakequant_wrong_to_target_count": 9,
        },
        selected_birth_parseback_survival={
            "action_id": "a" * 64,
            "selected_archive_sha256": "b" * 64,
            "parseback_payload_survived": True,
            "parseback_scorer_effect_survived": False,
        },
        target_region_action_parseback_survival={
            "action_id": "a" * 64,
            "archive_sha256": "b" * 64,
            "support_sha256": "c" * 64,
            "expected_support_sha256": "d" * 64,
            "parseback_survived": True,
        },
    )

    assert row["first_divergence"] == "support_identity_drift"
    assert "target_region_action_support_hash_mismatch" in row["first_divergence_reasons"]
