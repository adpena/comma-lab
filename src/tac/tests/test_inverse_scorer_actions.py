# SPDX-License-Identifier: MIT
"""Tests for measured inverse-scorer ActionEffect materialization.

Fixtures here are synthetic validation inputs only.  They exercise the real
``ActionEffect`` score math, generator, commutator, and CLI paths, but carry no
empirical or contest authority.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.analysis.action_commutator import build_commutator_ledger
from tac.analysis.action_effect import ActionEffect, append_action_effect, read_action_effects
from tac.analysis.evaluator_action_lowering_race import LOWERING_RACE_SCHEMA
from tac.analysis.inverse_scorer_actions import (
    BACKEND_REALIZATION_FAILED,
    BLOCKER_ARCHIVE_CLOSED_BIRTH_REQUIRES_EXECUTABLE_SUPPORT,
    BLOCKER_NO_COMPOSITE,
    BLOCKER_REGION_SUPPORT_IDENTITY_MISSING,
    BLOCKER_REGION_SUPPORT_RESEARCH_ONLY,
    BLOCKER_SCORE_DELTA_MISSING,
    BLOCKER_SCORE_PROGRAM_ARCHIVE_HASH_MISSING,
    BLOCKER_SCORE_PROGRAM_INFLATE_MISSING,
    BLOCKER_SCORE_PROGRAM_PARSEBACK_MISSING,
    BLOCKER_UINT8_MOTION_WITHOUT_SEGNET_WALL_CROSSING,
    BLOCKER_WALL_NORMAL_BACKEND_FIT_MISSING,
    BLOCKER_WALL_NORMAL_BACKEND_NOT_REALIZED,
    BLOCKER_WALL_NORMAL_DIRECT_TEACHER_EXACT_SCORE_NOT_ACCEPTED,
    BLOCKER_WALL_NORMAL_DIRECT_TEACHER_MISSING,
    BLOCKER_WALL_NORMAL_DIRECT_TEACHER_NOT_CROSSED,
    BLOCKER_WALL_NORMAL_DIRECT_TEACHER_NOT_TRUE_WALL_NORMAL,
    BLOCKER_WALL_NORMAL_SIDECAR_ARCHIVE_UNCLOSED,
    BLOCKER_WALL_NORMAL_SIDECAR_DISTORTION_ENDPOINTS_MISSING,
    DIRECT_SEG_WALL_ORACLE_SCHEMA,
    DIRECT_TEACHER_NO_WALL_CROSS,
    SCORE_PROGRAM_WORD_SCHEMA,
    SIDECAR_FALLBACK_ACCEPTED,
    TARGET_REGION_WALL_NORMAL_LIFT_SCHEMA,
    WALL_NORMAL_BRANCH_RECEIPT_SCHEMA,
    WALL_NORMAL_FIRST_FAILING_SURFACES,
    build_direct_seg_wall_oracle_receipt,
    build_score_program_word,
    build_target_region_wall_normal_lift_receipt,
    build_wall_normal_branch_action_effects,
    build_wall_normal_branch_receipt,
    generate_inverse_scorer_candidates,
)
from tac.analysis.pr110_baseline_reproduction import (
    BLOCKER_GLOBAL_K,
    BLOCKER_SELECTOR_BITS,
    BLOCKER_SELECTOR_PAIR_COUNT,
    build_pr110_k16_baseline_reproduction_from_action_effects,
    validate_pr110_k16_baseline_reproduction,
)
from tools import generate_inverse_evaluate_actions as inverse_materializer


def _frame0_pose_effect() -> ActionEffect:
    return ActionEffect.build(
        action_id="real_frame0_pose_source",
        family="hinerv",
        action_kind="frame0_pose_target_only",
        authority="batch_local_live_mlx",
        producer="fixture",
        consumer="fixture",
        pair_ids=[0],
        trained_groups=["compensation_head_rgb_0"],
        old_d_seg=0.50,
        new_d_seg=0.50,
        old_d_pose=0.20,
        new_d_pose=0.15,
        receiver_surface={
            "posenet_input_delta_linf": 1.0,
            "pose_output_l2_delta": 1.25,
        },
        posenet_input_delta_linf_pair=1.0,
        pose_output_l2_delta=1.25,
        exact_score_decision="accept",
    )


def _frame1_birth_effect() -> ActionEffect:
    return ActionEffect.build(
        action_id="real_frame1_birth_source",
        family="hinerv",
        action_kind="target_region_birth",
        authority="batch_local_live_mlx",
        producer="fixture",
        consumer="fixture",
        pair_ids=[0],
        class_ids=[4],
        region_ids=["b0/c4/r1"],
        trained_groups=["high_grid", "output_head_rgb_1"],
        old_d_seg=0.50,
        new_d_seg=0.48,
        old_d_pose=0.20,
        new_d_pose=0.20,
        receiver_surface={
            "uint8_changed_pixels": 32,
            "seg_input_delta_linf": 1.0,
            "seg_argmax_changed_pixels": 7,
            "seg_wrong_to_target_count": 5,
        },
        uint8_changed_count_region=32,
        seg_input_delta_linf_region=1.0,
        argmax_changed_count_region=7,
        wrong_to_target=5,
        segnet_margin_delta=-0.125,
        fakequant_segnet_margin_delta=-0.100,
        parseback_segnet_margin_delta=-0.075,
        support_source="explicit_payload_coordinates",
        support_cardinality=32,
        support_sha256="a" * 64,
        support_encoding="explicit_yx_u16_coordinates",
        support_encoded_bytes=64,
        support_research_only=False,
        exact_score_decision="accept",
    )


def _composite_effect() -> ActionEffect:
    return ActionEffect.build(
        action_id="real_birth_plus_pose_composite_source",
        family="hinerv",
        action_kind="independent_birth_plus_frame0_pose",
        authority="batch_local_live_mlx",
        producer="fixture",
        consumer="fixture",
        pair_ids=[0],
        class_ids=[4],
        region_ids=["b0/c4/r1"],
        trained_groups=["high_grid", "output_head_rgb_1", "compensation_head_rgb_0"],
        old_d_seg=0.50,
        new_d_seg=0.46,
        old_d_pose=0.20,
        new_d_pose=0.14,
        receiver_surface={
            "uint8_changed_pixels": 48,
            "seg_input_delta_linf": 1.0,
            "posenet_input_delta_linf": 1.0,
            "seg_argmax_changed_pixels": 11,
            "seg_wrong_to_target_count": 8,
            "pose_output_l2_delta": 1.5,
        },
        uint8_changed_count_region=48,
        seg_input_delta_linf_region=1.0,
        posenet_input_delta_linf_pair=1.0,
        argmax_changed_count_region=11,
        wrong_to_target=8,
        pose_output_l2_delta=1.5,
        interaction_or_commutator=-0.25,
        segnet_margin_delta=-0.25,
        support_source="explicit_payload_coordinates",
        support_cardinality=48,
        support_sha256="a" * 64,
        support_encoding="explicit_yx_u16_coordinates",
        support_encoded_bytes=96,
        support_research_only=False,
        exact_score_decision="accept",
    )


def _reverse_composite_effect() -> ActionEffect:
    return ActionEffect.build(
        action_id="real_pose_then_birth_composite_source",
        family="hinerv",
        action_kind="frame0_pose_then_birth_composite",
        authority="batch_local_live_mlx",
        producer="fixture",
        consumer="fixture",
        pair_ids=[0],
        class_ids=[4],
        region_ids=["b0/c4/r1"],
        trained_groups=["compensation_head_rgb_0", "high_grid", "output_head_rgb_1"],
        old_d_seg=0.50,
        new_d_seg=0.465,
        old_d_pose=0.20,
        new_d_pose=0.145,
        receiver_surface={
            "uint8_changed_pixels": 44,
            "seg_input_delta_linf": 1.0,
            "posenet_input_delta_linf": 1.0,
            "seg_argmax_changed_pixels": 10,
            "seg_wrong_to_target_count": 7,
            "pose_output_l2_delta": 1.4,
        },
        uint8_changed_count_region=44,
        seg_input_delta_linf_region=1.0,
        posenet_input_delta_linf_pair=1.0,
        argmax_changed_count_region=10,
        wrong_to_target=7,
        pose_output_l2_delta=1.4,
        interaction_or_commutator=-0.2,
        support_source="explicit_payload_coordinates",
        support_cardinality=44,
        support_sha256="a" * 64,
        support_encoding="explicit_yx_u16_coordinates",
        support_encoded_bytes=88,
        support_research_only=False,
        exact_score_decision="accept",
    )


def _with_base_state(effect: ActionEffect, base_state_sha256: str) -> ActionEffect:
    payload = effect.as_dict()
    payload["base_state_sha256"] = base_state_sha256
    return ActionEffect.from_dict(payload)


def test_direct_seg_wall_oracle_accepts_measured_wrong_to_target_crossing() -> None:
    support = np.zeros((1, 4, 4), dtype=bool)
    support[0, 1:3, 1:3] = True
    before_argmax = np.zeros((1, 4, 4), dtype=np.int64)
    after_argmax = before_argmax.copy()
    after_argmax[support] = 2
    before_u8 = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    after_u8 = before_u8.copy()
    after_u8[support] = np.array([255, 0, 0], dtype=np.uint8)

    receipt = build_direct_seg_wall_oracle_receipt(
        action_id="direct-wall-positive",
        authority="batch_local_live_mlx",
        pair_id=0,
        target_class=2,
        support_mask=support,
        before_argmax=before_argmax,
        after_argmax=after_argmax,
        before_uint8=before_u8,
        after_uint8=after_u8,
        old_d_seg=0.50,
        new_d_seg=0.49,
        old_d_pose=0.20,
        new_d_pose=0.20,
        region_id="b0/c2/r1",
        support_encoded_bytes=16,
    )

    assert receipt["schema"] == DIRECT_SEG_WALL_ORACLE_SCHEMA
    assert receipt["crossed_target_wall"] is True
    assert receipt["archive_executable"] is True
    assert receipt["support_cardinality"] == 4
    assert len(receipt["support_sha256"]) == 64
    assert receipt["blockers"] == []
    effect = ActionEffect.from_dict(receipt["action_effect"])
    assert effect.exact_score_decision == "accept"
    assert effect.wrong_to_target == 4
    assert effect.delta_score_nonrate < 0.0


def test_direct_seg_wall_oracle_labels_uint8_motion_without_wall_crossing() -> None:
    support = np.ones((1, 2, 3), dtype=bool)
    before_argmax = np.zeros((1, 2, 3), dtype=np.int64)
    after_argmax = np.ones((1, 2, 3), dtype=np.int64)
    before_u8 = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    after_u8 = np.full((1, 2, 3, 3), 127, dtype=np.uint8)

    receipt = build_direct_seg_wall_oracle_receipt(
        action_id="direct-wall-v31-negative",
        authority="batch_local_live_mlx",
        pair_id=0,
        target_class=2,
        support_mask=support,
        before_argmax=before_argmax,
        after_argmax=after_argmax,
        before_uint8=before_u8,
        after_uint8=after_u8,
        old_d_seg=0.50,
        new_d_seg=0.51,
        old_d_pose=0.20,
        new_d_pose=0.19,
    )

    assert receipt["crossed_target_wall"] is False
    assert receipt["uint8_changed_pixels"] == 6
    assert BLOCKER_UINT8_MOTION_WITHOUT_SEGNET_WALL_CROSSING in receipt["blockers"]
    effect = ActionEffect.from_dict(receipt["action_effect"])
    assert effect.exact_score_decision == "reject"
    assert effect.wrong_to_target == 0


def _direct_wall_candidate(
    *,
    wrong_to_target: int = 4,
    inverse_source: str = "segnet_margin_vjp",
    archive_executable_support_sha256: str | None = None,
) -> dict:
    support = np.zeros((1, 4, 4), dtype=bool)
    support[0, 1:3, 1:3] = True
    before_argmax = np.zeros((1, 4, 4), dtype=np.int64)
    after_argmax = before_argmax.copy()
    if wrong_to_target > 0:
        after_argmax[support] = 2
    before_u8 = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    after_u8 = before_u8.copy()
    after_u8[support] = np.array([255, 0, 0], dtype=np.uint8)
    receipt = build_direct_seg_wall_oracle_receipt(
        action_id="wall-normal-direct",
        authority="batch_local_live_mlx",
        pair_id=0,
        target_class=2,
        support_mask=support,
        before_argmax=before_argmax,
        after_argmax=after_argmax,
        before_uint8=before_u8,
        after_uint8=after_u8,
        old_d_seg=0.50,
        new_d_seg=0.49 if wrong_to_target > 0 else 0.50,
        old_d_pose=0.20,
        new_d_pose=0.20,
        region_id="b0/c2/r1",
        support_encoded_bytes=16,
        archive_executable_support_sha256=archive_executable_support_sha256,
        archive_executable_support_encoding="target_region_action_coordinates_v1",
        archive_executable_support_cardinality=4,
        archive_executable_support_encoded_bytes=16,
        inverse_source=inverse_source,
        inverse_basis=(
            "support_projected_segnet_margin_vjp"
            if inverse_source == "segnet_margin_vjp"
            else "receiver_surface_masked_rgb_residual_on_support"
        ),
        uses_official_seg_preprocess=True,
        uses_target_class_margin=inverse_source == "segnet_margin_vjp",
        margin_convention=(
            "target_minus_max_wrong"
            if inverse_source == "segnet_margin_vjp"
            else "measured_argmax_transition_after_candidate"
        ),
        frontier_pixel_policy="fixture_support",
    )
    return {
        "schema": "hi_nerv_target_region_masked_residual_oracle_candidate.v1",
        "oracle_kind": "scorer_wall_normal_basis",
        "direct_seg_wall_oracle": receipt,
        "exact_delta_score_nonrate": -1.0 if wrong_to_target > 0 else 0.0,
        "target_region_action_payload_bytes": 23,
        "target_region_action_value_per_payload_byte_nonrate": 1.0 / 23.0,
        "target_region_action_section_telemetry": {
            "support_sha256": receipt["support_sha256"],
            "support_encoded_bytes": 16,
        },
        "charged_byte_sections_missing": [
            "target_region_action_archive_meta_not_materialized",
            "target_region_action_parseback_survival_missing",
            "target_region_action_inflate_survival_missing",
        ],
    }


def _backend_birth_receipt(*, wrong_to_target: int, accepted: bool = True) -> dict:
    return {
        "schema": "hi_nerv_target_region_birth_receipt.v1",
        "action_id": "wall-normal-action",
        "surface": "live_mlx",
        "worst_region": {
            "schema": "hi_nerv_target_region_debt.v1",
            "batch_index": 0,
            "class_index": 2,
            "region_label": 1,
            "region_pixel_count": 4,
            "region_unsolved_pixel_count": 4,
            "score_debt_units": 25.0,
        },
        "accepted_step_count": 1 if accepted else 0,
        "updated_parameter_names": ["head_rgb_1.weight"] if accepted else [],
        "trained_groups": ["head_rgb_1"] if accepted else [],
        "exact_nonrate": {
            "old_d_seg_batch": 0.50,
            "new_d_seg_batch": 0.49 if wrong_to_target > 0 else 0.50,
            "old_d_pose_batch": 0.20,
            "new_d_pose_batch": 0.20,
            "delta_score_nonrate": -1.0 if wrong_to_target > 0 else 0.0,
            "pose_term_available": True,
        },
        "admission_decision": {
            "exact_score_decision": "accepted" if wrong_to_target > 0 else "rejected",
            "catastrophic_guard_decision": "satisfied",
            "raw_cap_decision": "satisfied",
        },
        "argmax_transitions": {
            "wrong_to_target_count": int(wrong_to_target),
            "target_to_wrong_count": 0,
            "wrong_to_wrong_count": 0,
            "net_target_support_delta": int(wrong_to_target),
        },
        "receiver_surface_uint8_changed_pixels": 4 if wrong_to_target > 0 else 0,
        "receiver_surface_uint8_delta_abs_max": 1.0 if wrong_to_target > 0 else 0.0,
        "receiver_surface_float_rgb_delta_linf": 1.0 / 255.0 if wrong_to_target > 0 else 0.0,
        "runtime_sidecar_bytes": 0,
        "blockers": [] if accepted else ["hinerv_target_region_birth_no_accepted_step"],
    }


def test_target_region_wall_normal_lift_selects_backend_when_realized() -> None:
    direct = _direct_wall_candidate()
    receipt = build_target_region_wall_normal_lift_receipt(
        action_id="wall-normal-action",
        pair_id=0,
        target_class=2,
        region_id="b0/c2/r1",
        direct_teacher_candidate=direct,
        backend_birth_receipt=_backend_birth_receipt(wrong_to_target=4),
        sidecar_candidate=direct,
    )

    assert receipt["schema"] == TARGET_REGION_WALL_NORMAL_LIFT_SCHEMA
    assert receipt["operator"] == "TargetRegionWallNormalLift"
    assert receipt["stage_order"][1:4] == [
        "SegNetWallNormalLift",
        "PoseYUV6TrustProjection",
        "BackendRealization",
    ]
    assert receipt["direct_teacher"]["crossed_target_wall"] is True
    assert receipt["direct_teacher"]["teacher_is_true_wall_normal"] is True
    assert receipt["direct_teacher"]["decision_state"] == "DIRECT_TEACHER_ACCEPTED"
    assert receipt["decision_state"] == "BACKEND_REALIZATION_ACCEPTED"
    assert receipt["backend_fit"]["realized_target_wall"] is True
    assert receipt["backend_fit"]["decision_state"] == "BACKEND_REALIZATION_ACCEPTED"
    assert receipt["backend_fit"]["realization_gap_wrong_to_target_count"] == 0
    assert receipt["selected_next_operator"] == "backend_fit_live"
    assert receipt["next_required_surface"] == "fakequant_archive_parseback_survival"
    assert receipt["backend_realization_required_before_long_run"] is False
    assert receipt["promotion_eligible"] is False


def test_target_region_wall_normal_lift_names_backend_gap_and_sidecar_fallback() -> None:
    direct = _direct_wall_candidate()
    receipt = build_target_region_wall_normal_lift_receipt(
        action_id="wall-normal-action",
        pair_id=0,
        target_class=2,
        region_id="b0/c2/r1",
        direct_teacher_candidate=direct,
        backend_birth_receipt=_backend_birth_receipt(
            wrong_to_target=0,
            accepted=False,
        ),
        sidecar_candidate=direct,
    )

    assert receipt["direct_teacher"]["crossed_target_wall"] is True
    assert receipt["backend_fit"]["realized_target_wall"] is False
    assert receipt["backend_fit"]["decision_state"] == "BACKEND_REALIZATION_FAILED"
    assert receipt["backend_fit"]["realization_gap_wrong_to_target_count"] == 4
    assert receipt["sidecar_fallback"]["available"] is True
    assert receipt["sidecar_fallback"]["decision_state"] == "SUPPORT_NOT_ARCHIVE_EXECUTABLE"
    assert receipt["sidecar_fallback"]["archive_executable"] is False
    assert receipt["sidecar_fallback"]["payload_bytes"] == 23
    assert receipt["selected_next_operator"] == "byte_priced_action_fallback"
    assert receipt["next_required_surface"] == "archive_materialize_parseback_inflate"
    assert BLOCKER_WALL_NORMAL_BACKEND_NOT_REALIZED in receipt["blockers"]
    assert BLOCKER_WALL_NORMAL_SIDECAR_ARCHIVE_UNCLOSED in receipt["blockers"]
    assert receipt["decision_state"] == "BACKEND_REALIZATION_FAILED"
    assert receipt["first_failing_surface"] == BACKEND_REALIZATION_FAILED
    assert receipt["score_claim"] is False


def test_wall_normal_lift_receipt_materializes_same_action_branch_effects() -> None:
    direct = _direct_wall_candidate()
    receipt = build_target_region_wall_normal_lift_receipt(
        action_id="wall-normal-action",
        pair_id=0,
        target_class=2,
        region_id="b0/c2/r1",
        direct_teacher_candidate=direct,
        backend_birth_receipt=_backend_birth_receipt(
            wrong_to_target=0,
            accepted=False,
        ),
        sidecar_candidate=direct,
    )

    effects = build_wall_normal_branch_action_effects(
        receipt,
        artifact_ref="/tmp/training_artifact.json",
    )

    assert [effect.action_kind for effect in effects] == [
        "wall_normal_direct_teacher",
        "wall_normal_backend_fit",
        "wall_normal_sidecar_fallback",
    ]
    assert {effect.action_id for effect in effects} == {"wall-normal-action"}
    assert {effect.support_sha256 for effect in effects} == {
        receipt["direct_teacher"]["support_sha256"]
    }
    assert {effect.support_source for effect in effects} == {
        receipt["direct_teacher"]["support_source"]
    }
    direct_effect, backend_effect, sidecar_effect = effects
    assert direct_effect.exact_score_decision == "accept"
    assert direct_effect.wrong_to_target == 4
    assert direct_effect.delta_score_nonrate == pytest.approx(-1.0)
    assert backend_effect.exact_score_decision == "reject"
    assert backend_effect.wrong_to_target == 0
    assert BLOCKER_WALL_NORMAL_BACKEND_NOT_REALIZED in backend_effect.blockers
    assert sidecar_effect.exact_score_decision == "reject"
    assert sidecar_effect.delta_score_nonrate is None
    assert BLOCKER_WALL_NORMAL_SIDECAR_ARCHIVE_UNCLOSED in sidecar_effect.blockers
    assert (
        BLOCKER_WALL_NORMAL_SIDECAR_DISTORTION_ENDPOINTS_MISSING
        in sidecar_effect.blockers
    )

    result = generate_inverse_scorer_candidates(list(effects))
    rows = result["candidate_queue"]
    assert {row["action_kind"] for row in rows} == {
        "wall_normal_direct_teacher",
        "wall_normal_backend_fit",
        "wall_normal_sidecar_fallback",
    }
    assert {row["action_id"] for row in rows} == {"wall-normal-action"}
    sidecar_row = next(row for row in rows if row["action_kind"] == "wall_normal_sidecar_fallback")
    assert BLOCKER_SCORE_DELTA_MISSING in sidecar_row["blockers"]


def test_wall_normal_lift_prefers_archive_executable_support_for_branch_identity() -> None:
    direct = _direct_wall_candidate(
        archive_executable_support_sha256="b" * 64,
    )
    receipt = build_target_region_wall_normal_lift_receipt(
        action_id="wall-normal-action",
        pair_id=0,
        target_class=2,
        region_id="b0/c2/r1",
        direct_teacher_candidate=direct,
        backend_birth_receipt=_backend_birth_receipt(
            wrong_to_target=0,
            accepted=False,
        ),
        sidecar_candidate=direct,
    )

    assert receipt["direct_teacher"]["support_sha256"] != (
        receipt["direct_teacher"]["archive_executable_support_sha256"]
    )
    assert receipt["direct_teacher"]["archive_executable_support_sha256"] == "b" * 64
    assert receipt["direct_teacher"]["archive_executable_support_encoding"] == (
        "target_region_action_coordinates_v1"
    )
    assert receipt["support_identity_source"] == "archive_executable_support"
    assert receipt["support_sha256"] == "b" * 64
    assert receipt["support_source"] == "archive_executable_target_region_action_support"
    assert receipt["support_encoding"] == "target_region_action_coordinates_v1"
    assert receipt["support_cardinality"] == receipt["direct_teacher"][
        "archive_executable_support_cardinality"
    ]
    assert receipt["support_encoded_bytes"] == receipt["direct_teacher"][
        "archive_executable_support_encoded_bytes"
    ]
    assert receipt["support_research_only"] is False

    effects = build_wall_normal_branch_action_effects(receipt)
    assert {effect.action_id for effect in effects} == {"wall-normal-action"}
    assert {effect.support_sha256 for effect in effects} == {"b" * 64}
    assert {effect.support_source for effect in effects} == {
        "archive_executable_target_region_action_support"
    }
    branch_receipt = build_wall_normal_branch_receipt(effects)
    assert branch_receipt["same_action_id"] is True
    assert branch_receipt["same_support_sha256"] is True
    assert branch_receipt["support_sha256s"] == ["b" * 64]
    assert "wall_normal_branch_action_id_mismatch" not in branch_receipt["blockers"]


def test_wall_normal_branch_receipt_promotes_survived_sidecar_support_divergence() -> None:
    direct = _direct_wall_candidate(
        archive_executable_support_sha256="a" * 64,
    )
    sidecar = _direct_wall_candidate(
        archive_executable_support_sha256="a" * 64,
    )
    sidecar["target_region_action_section_telemetry"]["support_sha256"] = "b" * 64
    receipt = build_target_region_wall_normal_lift_receipt(
        action_id="wall-normal-action",
        pair_id=0,
        target_class=2,
        region_id="b0/c2/r1",
        direct_teacher_candidate=direct,
        backend_birth_receipt=_backend_birth_receipt(
            wrong_to_target=0,
            accepted=False,
        ),
        sidecar_candidate=sidecar,
    )

    effects = build_wall_normal_branch_action_effects(receipt)
    sidecar_effect = next(
        effect
        for effect in effects
        if effect.action_kind == "wall_normal_sidecar_fallback"
    )
    branch_receipt = build_wall_normal_branch_receipt(effects)

    assert (
        "direct_teacher_and_survived_sidecar_support_hashes_diverge"
        in sidecar_effect.blockers
    )
    assert (
        "direct_teacher_and_survived_sidecar_support_hashes_diverge"
        in branch_receipt["blockers"]
    )
    assert branch_receipt["same_support_sha256"] is True
    assert branch_receipt["support_executable_count"] == 3


def test_target_region_wall_normal_lift_blocks_when_direct_teacher_missing() -> None:
    receipt = build_target_region_wall_normal_lift_receipt(
        action_id="wall-normal-action",
        pair_id=0,
        target_class=2,
        region_id="b0/c2/r1",
        direct_teacher_candidate=None,
        backend_birth_receipt=None,
    )

    assert receipt["direct_teacher"]["available"] is False
    assert receipt["direct_teacher"]["decision_state"] == "DIRECT_TEACHER_NO_WALL_CROSS"
    assert receipt["backend_fit"]["attempted"] is False
    assert receipt["backend_fit"]["decision_state"] == "SKIPPED_DIRECT_TEACHER_FAILED"
    assert receipt["selected_next_operator"] == "direct_wall_teacher_gap"
    assert receipt["next_required_surface"] == "inverse_scorer_candidate_generation"
    assert BLOCKER_WALL_NORMAL_DIRECT_TEACHER_MISSING in receipt["blockers"]
    assert BLOCKER_WALL_NORMAL_BACKEND_FIT_MISSING in receipt["blockers"]
    assert receipt["decision_state"] == "DIRECT_TEACHER_NO_WALL_CROSS"
    assert receipt["backend_realization_required_before_long_run"] is True
    assert receipt["promotion_eligible"] is False


def test_target_region_wall_normal_lift_keeps_non_crossing_teacher_out_of_backend_gap() -> None:
    receipt = build_target_region_wall_normal_lift_receipt(
        action_id="wall-normal-action",
        pair_id=0,
        target_class=2,
        region_id="b0/c2/r1",
        direct_teacher_candidate=_direct_wall_candidate(wrong_to_target=0),
        backend_birth_receipt=_backend_birth_receipt(wrong_to_target=0),
        sidecar_candidate=_direct_wall_candidate(wrong_to_target=0),
    )

    assert receipt["direct_teacher"]["available"] is True
    assert receipt["direct_teacher"]["crossed_target_wall"] is False
    assert receipt["direct_teacher"]["decision_state"] == "DIRECT_TEACHER_NO_WALL_CROSS"
    assert receipt["backend_fit"]["realized_target_wall"] is False
    assert receipt["backend_fit"]["decision_state"] == "SKIPPED_DIRECT_TEACHER_FAILED"
    assert receipt["sidecar_fallback"]["available"] is False
    assert receipt["sidecar_fallback"]["decision_state"] == "SKIPPED_DIRECT_TEACHER_FAILED"
    assert receipt["selected_next_operator"] == "direct_wall_teacher_gap"
    assert receipt["decision_state"] == "DIRECT_TEACHER_NO_WALL_CROSS"
    assert BLOCKER_WALL_NORMAL_DIRECT_TEACHER_NOT_CROSSED in receipt["blockers"]
    assert BLOCKER_WALL_NORMAL_BACKEND_NOT_REALIZED not in receipt["blockers"]
    assert BLOCKER_WALL_NORMAL_SIDECAR_ARCHIVE_UNCLOSED not in receipt["blockers"]


def test_target_region_wall_normal_lift_rejects_crossing_teacher_without_exact_score() -> None:
    direct = _direct_wall_candidate(wrong_to_target=4)
    direct_wall = direct["direct_seg_wall_oracle"]
    direct_wall["blockers"] = [
        *direct_wall["blockers"],
        "direct_seg_wall_oracle_exact_score_not_improved",
    ]
    direct_wall["action_effect"]["exact_score_decision"] = "reject"
    direct_wall["action_effect"]["blockers"] = [
        *direct_wall["action_effect"]["blockers"],
        "direct_seg_wall_oracle_exact_score_not_improved",
    ]
    receipt = build_target_region_wall_normal_lift_receipt(
        action_id="wall-normal-action",
        pair_id=0,
        target_class=2,
        region_id="b0/c2/r1",
        direct_teacher_candidate=direct,
        backend_birth_receipt=_backend_birth_receipt(
            wrong_to_target=0,
            accepted=False,
        ),
        sidecar_candidate=direct,
    )

    assert receipt["direct_teacher"]["crossed_target_wall"] is True
    assert receipt["direct_teacher"]["exact_score_decision"] == "reject"
    assert receipt["direct_teacher"]["decision_state"] == "DIRECT_TEACHER_EXACT_REJECTED"
    assert receipt["sidecar_fallback"]["available"] is False
    assert receipt["sidecar_fallback"]["decision_state"] == "SKIPPED_DIRECT_TEACHER_FAILED"
    assert receipt["selected_next_operator"] == "direct_wall_teacher_gap"
    assert receipt["decision_state"] == "DIRECT_TEACHER_EXACT_REJECTED"
    assert BLOCKER_WALL_NORMAL_DIRECT_TEACHER_EXACT_SCORE_NOT_ACCEPTED in (
        receipt["blockers"]
    )
    assert BLOCKER_WALL_NORMAL_SIDECAR_ARCHIVE_UNCLOSED not in receipt["blockers"]


def test_target_region_wall_normal_lift_rejects_masked_residual_as_true_wall_normal() -> None:
    direct = _direct_wall_candidate(inverse_source="masked_residual")
    receipt = build_target_region_wall_normal_lift_receipt(
        action_id="wall-normal-action",
        pair_id=0,
        target_class=2,
        region_id="b0/c2/r1",
        direct_teacher_candidate=direct,
        backend_birth_receipt=_backend_birth_receipt(wrong_to_target=4),
        sidecar_candidate=direct,
    )

    assert receipt["direct_teacher"]["crossed_target_wall"] is True
    assert receipt["direct_teacher"]["inverse_source"] == "masked_residual"
    assert receipt["direct_teacher"]["teacher_is_true_wall_normal"] is False
    assert receipt["direct_teacher"]["qualified_crossed_target_wall"] is False
    assert receipt["direct_teacher"]["decision_state"] == "DIRECT_TEACHER_NO_WALL_CROSS"
    assert receipt["selected_next_operator"] == "direct_wall_teacher_gap"
    assert receipt["decision_state"] == "DIRECT_TEACHER_NO_WALL_CROSS"
    assert receipt["first_failing_surface"] == DIRECT_TEACHER_NO_WALL_CROSS
    assert receipt["sidecar_fallback"]["available"] is False
    assert BLOCKER_WALL_NORMAL_DIRECT_TEACHER_NOT_TRUE_WALL_NORMAL in receipt[
        "blockers"
    ]


def test_wall_normal_branch_receipt_names_one_strict_first_failure() -> None:
    direct = _direct_wall_candidate()
    receipt = build_target_region_wall_normal_lift_receipt(
        action_id="wall-normal-action",
        pair_id=0,
        target_class=2,
        region_id="b0/c2/r1",
        direct_teacher_candidate=direct,
        backend_birth_receipt=_backend_birth_receipt(
            wrong_to_target=0,
            accepted=False,
        ),
        sidecar_candidate=direct,
    )
    effects = build_wall_normal_branch_action_effects(
        receipt,
        artifact_ref="/tmp/training_artifact.json",
    )

    branch_receipt = build_wall_normal_branch_receipt(
        effects,
        source_artifact_paths=["/tmp/training_artifact.json"],
    )

    assert branch_receipt["schema"] == WALL_NORMAL_BRANCH_RECEIPT_SCHEMA
    assert branch_receipt["branch_count"] == 3
    assert branch_receipt["same_action_id"] is True
    assert branch_receipt["same_support_sha256"] is True
    assert branch_receipt["support_executable_count"] == 3
    assert branch_receipt["first_failing_surface"] == BACKEND_REALIZATION_FAILED
    assert branch_receipt["first_failing_surface"] in WALL_NORMAL_FIRST_FAILING_SURFACES
    backend = next(
        row
        for row in branch_receipt["branches"]
        if row["action_kind"] == "wall_normal_backend_fit"
    )
    assert backend["first_failing_surface"] == BACKEND_REALIZATION_FAILED


def test_wall_normal_branch_receipt_keeps_divergence_separate_from_executable_support() -> None:
    sidecar = ActionEffect.build(
        action_id="same-action",
        family="hinerv",
        action_kind="sidecar_grammar",
        authority="inflate_raw",
        producer="fixture",
        consumer="fixture",
        pair_ids=[0],
        class_ids=[4],
        region_ids=["b0/c4/r1"],
        payload_sections=["support_codec=explicit_yx_u16_coordinates"],
        old_d_seg=0.49,
        new_d_seg=0.25,
        old_d_pose=194.0,
        new_d_pose=190.0,
        old_bytes=1,
        new_bytes=2,
        receiver_surface={
            "uint8_changed_pixels": 16,
            "seg_argmax_changed_pixels": 8,
            "seg_wrong_to_target_count": 8,
        },
        exact_score_decision="reject",
        fakequant_survived=True,
        parseback_survived=True,
        inflate_survived=True,
        wrong_to_target=8,
        support_source="survived_target_region_action_sidecar",
        support_cardinality=8,
        support_sha256="b" * 64,
        support_encoding="explicit_yx_u16_coordinates",
        support_encoded_bytes=16,
        support_research_only=False,
        blockers=["direct_teacher_and_survived_sidecar_support_hashes_diverge"],
    )

    receipt = build_wall_normal_branch_receipt([sidecar])

    assert receipt["first_failing_surface"] == SIDECAR_FALLBACK_ACCEPTED
    assert receipt["support_executable_count"] == 1
    assert receipt["same_support_sha256"] is True
    assert (
        "direct_teacher_and_survived_sidecar_support_hashes_diverge"
        in receipt["blockers"]
    )
    assert receipt["branches"][0]["support_executable"] is True
    assert receipt["branches"][0]["first_failing_surface"] == SIDECAR_FALLBACK_ACCEPTED


def test_wall_normal_branch_receipt_accepts_survived_sidecar_fallback() -> None:
    sidecar = ActionEffect.build(
        action_id="same-action",
        family="hinerv",
        action_kind="sidecar_grammar",
        authority="inflate_raw",
        producer="fixture",
        consumer="fixture",
        pair_ids=[0],
        class_ids=[4],
        region_ids=["b0/c4/r1"],
        payload_sections=["support_codec=explicit_yx_u16_coordinates"],
        old_d_seg=0.49,
        new_d_seg=0.25,
        old_d_pose=194.0,
        new_d_pose=190.0,
        old_bytes=1,
        new_bytes=2,
        receiver_surface={
            "uint8_changed_pixels": 16,
            "seg_argmax_changed_pixels": 8,
            "seg_wrong_to_target_count": 8,
        },
        exact_score_decision="accept",
        fakequant_survived=True,
        parseback_survived=True,
        inflate_survived=True,
        wrong_to_target=8,
        support_source="survived_target_region_action_sidecar",
        support_cardinality=8,
        support_sha256="b" * 64,
        support_encoding="explicit_yx_u16_coordinates",
        support_encoded_bytes=16,
        support_research_only=False,
    )

    receipt = build_wall_normal_branch_receipt([sidecar])

    assert receipt["first_failing_surface"] == SIDECAR_FALLBACK_ACCEPTED
    assert receipt["support_executable_count"] == 1
    assert receipt["blockers"] == []


def _pr110_k1_replay_effect() -> ActionEffect:
    return ActionEffect.build(
        action_id="lfv1v2_k01_replay",
        family="pr110",
        action_kind="selector_replay",
        authority="[macOS-CPU advisory] pr110_selector_replay",
        normalization_scope="full_video_equiv_estimate",
        producer="fixture",
        consumer="fixture",
        pair_ids=[43],
        payload_sections=["lfv1v2", "k01"],
        old_d_seg=0.00056039,
        new_d_seg=0.00056039,
        old_d_pose=2.943e-05,
        new_d_pose=2.943e-05,
        old_bytes=178_517,
        new_bytes=178_674,
        restore_state_pass=True,
        inflate_survived=True,
    )


def _four_arm_training_artifact() -> dict:
    base = {
        "schema": "hi_nerv_target_region_birth.v1",
        "action_id": "artifact_four_arm_action",
        "parameter_group_sha256_before": {"head_rgb_0": "a" * 64, "head_rgb_1": "b" * 64},
        "four_arm_ablation": {
            "schema": "hi_nerv_target_region_birth_four_arm_ablation.v1",
            "action_id": "artifact_four_arm_action",
            "authority": "batch_local_live_mlx",
            "normalization_scope": "batch_local",
            "parameter_group_sha256_before": {"head_rgb_0": "a" * 64, "head_rgb_1": "b" * 64},
            "arms": [
                _four_arm_row("A", "birth_only", -0.2, wrong_to_target=4, uint8_changed=16),
                _four_arm_row("B", "frame0_pose_target_only", -0.3, pose_delta=1.0),
                _four_arm_row(
                    "C",
                    "independent_birth_plus_frame0_pose",
                    -0.8,
                    wrong_to_target=5,
                    uint8_changed=24,
                    pose_delta=1.5,
                    comm=-0.3,
                ),
                _four_arm_row(
                    "E",
                    "frame0_pose_then_birth_composite",
                    -0.7,
                    wrong_to_target=5,
                    uint8_changed=24,
                    pose_delta=1.4,
                    comm=-0.2,
                ),
            ],
        },
    }
    return {"schema": "synthetic_training_artifact.v1", "target_region_birth_actuator": base}


def _four_arm_row(
    arm: str,
    action_kind: str,
    delta: float,
    *,
    wrong_to_target: int = 0,
    uint8_changed: int = 0,
    pose_delta: float | None = None,
    comm: float | None = None,
) -> dict:
    return {
        "schema": "hi_nerv_target_region_birth_four_arm.v1",
        "action_id": "artifact_four_arm_action",
        "authority": "batch_local_live_mlx",
        "normalization_scope": "batch_local",
        "action_kind": action_kind,
        "arm": arm,
        "decision": "measured",
        "accepted": True,
        "blockers": [],
        "pair_index": 0,
        "worst_region": {
            "schema": "hi_nerv_target_region_debt.v1",
            "batch_index": 0,
            "class_index": 4,
            "region_label": 1,
            "region_pixel_count": 12,
            "region_unsolved_pixel_count": 12,
            "score_debt_units": 1.0,
        },
        "updated_parameter_names": ["head_rgb_0"] if "frame0" in action_kind else ["head_rgb_1"],
        "trained_groups": ["compensation_head_rgb_0"] if "frame0" in action_kind else ["head_rgb_1"],
        "exact_nonrate": {
            "old_d_seg_batch": 0.5,
            "new_d_seg_batch": 0.5 + delta / 100.0,
            "old_d_pose_batch": 0.2,
            "new_d_pose_batch": 0.2,
            "delta_score_nonrate": delta,
            "pose_term_available": True,
        },
        "admission_decision": {
            "exact_score_decision": "accepted",
            "catastrophic_guard_decision": "satisfied",
            "raw_cap_decision": "satisfied",
            "pose_output_l2_delta": pose_delta,
        },
        "argmax_transitions": {
            "wrong_to_target_count": wrong_to_target,
            "target_to_wrong_count": 0,
            "wrong_to_wrong_count": 0,
            "net_target_support_delta": wrong_to_target,
        },
        "receiver_uint8_changed_pixels_region": uint8_changed,
        "receiver_uint8_delta_abs_max": 1.0 if uint8_changed else 0.0,
        "receiver_float_rgb_delta_linf": 1.0 / 255.0 if uint8_changed else 0.0,
        "pose_output_l2_delta": pose_delta,
        "interaction_or_commutator": comm,
        "restore_state_pass": True,
        "promotion_eligible": False,
    }


def test_inverse_scorer_generator_reemits_measured_candidates_and_ordered_composite() -> None:
    result = generate_inverse_scorer_candidates(
        [_frame0_pose_effect(), _frame1_birth_effect(), _composite_effect()]
    )

    assert result["passed"] is True
    assert result["candidate_count"] == 3
    effects = result["action_effects"]
    assert {effect.candidate_status for effect in effects} == {"measured"}
    assert all(effect.promotion_eligible is False for effect in effects)

    singles = [effect for effect in effects if effect.frame_index != "both"]
    composites = [effect for effect in effects if effect.frame_index == "both"]
    assert len(singles) == 2
    assert len(composites) == 1
    composite_id = composites[0].action_id
    assert singles[0].action_id in composite_id or singles[1].action_id in composite_id

    ledger = build_commutator_ledger(singles, composites)
    assert ledger["measured_commutator_count"] == 1
    assert ledger["needs_measurement_count"] == 1
    assert ledger["rows"][0]["authority"] == "batch_local_live_mlx"

    queue = result["candidate_queue"]
    assert len(queue) == 3
    assert all(row["menu_ilp_allowed"] is False for row in queue)
    assert all("pr110_k16_baseline_reproduction_missing" in row["menu_ilp_blockers"] for row in queue)
    assert [row["score_program_opcode"] for row in queue] == [
        "APPLY_FRAME0_POSE_ACTION",
        "APPLY_FRAME1_SEG_ACTION",
        "APPLY_BOTH_FRAME_COMPOSITE",
    ]
    assert [row["evaluator_action_basis"] for row in queue] == [
        "B0_frame0_pose_only",
        "B1_frame1_seg_wall_cross",
        "B3_both_frame_composite",
    ]
    assert {row["backend"] for row in queue} == {"hinerv_grid_adapter"}
    assert all(row["receiver_visible"] is True for row in queue)
    assert all(row["score_program_operation"]["score_claim"] is False for row in queue)
    assert all(
        BLOCKER_SCORE_PROGRAM_PARSEBACK_MISSING in row["promotion_blockers"]
        for row in queue
    )

    word = result["score_program_word"]
    assert word["schema"] == SCORE_PROGRAM_WORD_SCHEMA
    assert word["interpreter"] == "inflate_action_word_v1"
    assert word["operation_count"] == 3
    assert [op["opcode"] for op in word["operations"]] == [
        "APPLY_FRAME0_POSE_ACTION",
        "APPLY_FRAME1_SEG_ACTION",
        "APPLY_BOTH_FRAME_COMPOSITE",
    ]
    assert word["score_claim"] is False
    assert word["promotion_eligible"] is False
    assert word["ready_for_exact_eval_dispatch"] is False
    assert "pr110_k16_baseline_reproduction_missing" in word["blockers"]
    assert BLOCKER_SCORE_PROGRAM_ARCHIVE_HASH_MISSING in word["promotion_blockers"]
    assert BLOCKER_SCORE_PROGRAM_PARSEBACK_MISSING in word["promotion_blockers"]
    assert BLOCKER_SCORE_PROGRAM_INFLATE_MISSING in word["promotion_blockers"]


def test_inverse_scorer_generator_keeps_both_ordered_composites() -> None:
    result = generate_inverse_scorer_candidates(
        [
            _frame0_pose_effect(),
            _frame1_birth_effect(),
            _composite_effect(),
            _reverse_composite_effect(),
        ]
    )

    assert result["passed"] is True
    assert result["candidate_count"] == 4
    effects = result["action_effects"]
    singles = [effect for effect in effects if effect.frame_index != "both"]
    composites = [effect for effect in effects if effect.frame_index == "both"]
    assert len(singles) == 2
    assert len(composites) == 2
    assert {composite.action_id for composite in composites} == {
        f"{singles[1].action_id}__then__{singles[0].action_id}__inverse_composite",
        f"{singles[0].action_id}__then__{singles[1].action_id}__inverse_composite",
    }
    ledger = build_commutator_ledger(singles, composites)
    assert ledger["measured_commutator_count"] == 2
    assert ledger["needs_measurement_count"] == 0


def test_inverse_scorer_marks_missing_region_support_research_only() -> None:
    payload = _frame1_birth_effect().as_dict()
    for key in (
        "support_source",
        "support_cardinality",
        "support_sha256",
        "support_encoding",
        "support_encoded_bytes",
    ):
        payload[key] = None
    payload["support_research_only"] = None
    unsupported_birth = ActionEffect.from_dict(payload)

    result = generate_inverse_scorer_candidates(
        [_frame0_pose_effect(), unsupported_birth, _composite_effect()]
    )

    frame1 = next(row for row in result["candidate_queue"] if row["frame_index"] == 1)
    assert frame1["support_research_only"] is True
    assert frame1["support_source"] == "action_effect_region_id_only_no_pixel_support"
    assert BLOCKER_REGION_SUPPORT_RESEARCH_ONLY in frame1["blockers"]
    assert BLOCKER_REGION_SUPPORT_IDENTITY_MISSING in frame1["blockers"]
    assert BLOCKER_ARCHIVE_CLOSED_BIRTH_REQUIRES_EXECUTABLE_SUPPORT in frame1["blockers"]


def test_candidate_queue_carries_support_hash_continuity() -> None:
    result = generate_inverse_scorer_candidates(
        [_frame0_pose_effect(), _frame1_birth_effect(), _composite_effect()]
    )

    frame1 = next(row for row in result["candidate_queue"] if row["frame_index"] == 1)
    assert frame1["support_source"] == "explicit_payload_coordinates"
    assert frame1["support_cardinality"] == 32
    assert frame1["support_sha256"] == "a" * 64
    assert frame1["support_encoding"] == "explicit_yx_u16_coordinates"
    assert frame1["support_encoded_bytes"] == 64
    assert frame1["support_research_only"] is False
    assert frame1["segnet_margin_delta"] == pytest.approx(-0.125)
    assert frame1["fakequant_segnet_margin_delta"] == pytest.approx(-0.100)
    assert frame1["parseback_segnet_margin_delta"] == pytest.approx(-0.075)
    assert frame1["score_program_operation"]["segnet_margin_delta"] == pytest.approx(-0.125)
    assert frame1["score_program_operation"]["fakequant_segnet_margin_delta"] == pytest.approx(-0.100)
    assert frame1["score_program_operation"]["parseback_segnet_margin_delta"] == pytest.approx(-0.075)
    assert BLOCKER_REGION_SUPPORT_RESEARCH_ONLY not in frame1["blockers"]


def test_score_program_word_keeps_blocked_alternative_candidate_local() -> None:
    result = generate_inverse_scorer_candidates(
        [
            _frame0_pose_effect(),
            _frame1_birth_effect(),
            _composite_effect(),
            _reverse_composite_effect(),
        ]
    )
    rows = [dict(row) for row in result["candidate_queue"]]
    for row in rows:
        row["menu_ilp_allowed"] = True
        row["menu_ilp_blockers"] = []
    blocked = dict(rows[-1])
    blocked["action_id"] = f"{blocked['action_id']}__blocked_alternative"
    blocked["blockers"] = ["blocked_alternative_receiver_motion_missing"]
    blocked["score_program_operation"] = dict(blocked["score_program_operation"])
    blocked["score_program_operation"]["action_id"] = blocked["action_id"]
    blocked["score_program_operation"]["blockers"] = list(blocked["blockers"])
    rows.append(blocked)

    word = build_score_program_word(rows)

    assert "blocked_alternative_receiver_motion_missing" not in word["blockers"]
    assert "blocked_alternative_receiver_motion_missing" in word["candidate_blockers"]
    assert word["executable_operation_count"] == len(rows) - 1
    assert "B3_both_frame_composite" in word["basis_with_clean_candidate"]


def test_inverse_materializer_preserves_distinct_same_action_id_rows() -> None:
    clean = _composite_effect()
    payload = clean.as_dict()
    payload["blockers"] = ["same_action_id_distinct_attempt_blocked"]
    blocked = ActionEffect.from_dict(payload)

    rows = inverse_materializer._unique_effects([clean, blocked, clean])

    assert [row.blockers for row in rows] == [
        (),
        ("same_action_id_distinct_attempt_blocked",),
    ]


def test_inverse_scorer_generator_names_missing_composite_without_inventing_row() -> None:
    result = generate_inverse_scorer_candidates([_frame0_pose_effect(), _frame1_birth_effect()])

    assert result["passed"] is False
    assert result["candidate_count"] == 2
    assert BLOCKER_NO_COMPOSITE in result["blockers"]


def test_pr110_k16_reproduction_from_sparse_k1_row_is_precise_blocker() -> None:
    proof = build_pr110_k16_baseline_reproduction_from_action_effects([_pr110_k1_replay_effect()])

    assert proof["passed"] is False
    assert proof["global_k"] == 1
    assert proof["pair_count"] == 1
    assert BLOCKER_GLOBAL_K in proof["blockers"]
    assert BLOCKER_SELECTOR_PAIR_COUNT in proof["blockers"]
    assert BLOCKER_SELECTOR_BITS in proof["blockers"]

    validation = validate_pr110_k16_baseline_reproduction(proof)
    assert validation["passed"] is False
    assert BLOCKER_SELECTOR_PAIR_COUNT in validation["blockers"]
    assert BLOCKER_SELECTOR_BITS in validation["blockers"]


def test_generate_inverse_evaluate_actions_cli_writes_artifacts(tmp_path: Path) -> None:
    seed_ledger = tmp_path / "seed_action_effects.jsonl"
    pr110_ledger = tmp_path / "pr110_action_effects.jsonl"
    out_dir = tmp_path / "out"
    for effect in (_frame0_pose_effect(), _frame1_birth_effect(), _composite_effect()):
        append_action_effect(effect, seed_ledger)
    append_action_effect(_pr110_k1_replay_effect(), pr110_ledger)

    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "generate_inverse_evaluate_actions.py"),
            "--seed-action-effects",
            str(seed_ledger),
            "--pr110-action-effects",
            str(pr110_ledger),
            "--output-dir",
            str(out_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["schema"] == "tac.inverse_evaluate_action_materialization.v1"
    assert summary["inverse_candidate_count"] == 3
    assert summary["pr110_replay_row_count"] == 1
    assert summary["menu_ilp_allowed"] is False
    assert BLOCKER_GLOBAL_K in summary["pr110_k16_blockers"]

    rows = read_action_effects(out_dir / "action_effect_rows.jsonl")
    assert len(rows) == 4
    assert any(row.family == "pr110" for row in rows)
    assert (out_dir / "inverse_candidate_queue.jsonl").is_file()
    assert (out_dir / "score_program_word.json").is_file()
    assert (out_dir / "commutator_summary.json").is_file()
    assert (out_dir / "wall_normal_branch_receipt.json").is_file()
    assert (out_dir / "wall_normal_branch_action_effect_rows.jsonl").is_file()
    assert (out_dir / "wall_normal_branch_lowering_race.json").is_file()
    assert (out_dir / "next_blocker.md").is_file()
    note = (out_dir / "next_blocker.md").read_text(encoding="utf-8")
    assert "wall_normal_branch_lowering_race.json" in note
    score_program_word = json.loads((out_dir / "score_program_word.json").read_text(encoding="utf-8"))
    assert score_program_word["schema"] == SCORE_PROGRAM_WORD_SCHEMA
    assert score_program_word["operation_count"] == 3
    assert [row["opcode"] for row in score_program_word["operations"]] == [
        "APPLY_FRAME0_POSE_ACTION",
        "APPLY_FRAME1_SEG_ACTION",
        "APPLY_BOTH_FRAME_COMPOSITE",
    ]
    assert score_program_word["score_claim"] is False

    commutator = json.loads((out_dir / "commutator_summary.json").read_text(encoding="utf-8"))
    assert commutator["measured_commutator_count"] == 1
    queued = commutator["measurement_queue"][0]
    assert queued["measurement_command_available"] is False
    assert queued["first_measurement_command"] is None
    assert "inverse_scorer_reverse_order_composite_producer_missing" in queued["measurement_command_blockers"]
    assert "inverse_scorer_composite_base_identity_producer_missing" in queued["measurement_command_blockers"]


def test_generate_inverse_evaluate_actions_cli_filters_non_pr110_rows_from_pr110_input(
    tmp_path: Path,
) -> None:
    seed_ledger = tmp_path / "seed_action_effects.jsonl"
    pr110_ledger = tmp_path / "mixed_pr110_action_effects.jsonl"
    out_dir = tmp_path / "out"
    for effect in (_frame0_pose_effect(), _frame1_birth_effect(), _composite_effect()):
        append_action_effect(effect, seed_ledger)
    append_action_effect(_pr110_k1_replay_effect(), pr110_ledger)
    append_action_effect(_frame1_birth_effect(), pr110_ledger)

    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "generate_inverse_evaluate_actions.py"),
            "--seed-action-effects",
            str(seed_ledger),
            "--pr110-action-effects",
            str(pr110_ledger),
            "--output-dir",
            str(out_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["pr110_input_row_count"] == 2
    assert summary["pr110_replay_row_count"] == 1
    assert summary["pr110_filtered_non_pr110_count"] == 1
    assert summary["pr110_filtered_non_pr110_action_ids"] == [
        _frame1_birth_effect().action_id
    ]

    rows = read_action_effects(out_dir / "action_effect_rows.jsonl")
    assert len([row for row in rows if row.family == "pr110"]) == 1
    assert len(rows) == 4
    assert _frame1_birth_effect().action_id not in {row.action_id for row in rows}
    assert any(row.action_id.endswith("__inverse_frame1_seg") for row in rows)


def test_generate_inverse_evaluate_actions_cli_reads_training_artifact_base_identity(tmp_path: Path) -> None:
    training_artifact = tmp_path / "training_artifact.json"
    pr110_ledger = tmp_path / "pr110_action_effects.jsonl"
    out_dir = tmp_path / "out"
    training_artifact.write_text(json.dumps(_four_arm_training_artifact()), encoding="utf-8")
    append_action_effect(_pr110_k1_replay_effect(), pr110_ledger)

    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "generate_inverse_evaluate_actions.py"),
            "--seed-training-artifact",
            str(training_artifact),
            "--pr110-action-effects",
            str(pr110_ledger),
            "--output-dir",
            str(out_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    rows = read_action_effects(out_dir / "action_effect_rows.jsonl")
    hinerv_rows = [row for row in rows if row.family == "hinerv"]
    assert len(hinerv_rows) == 4
    assert all(row.base_state_sha256 for row in hinerv_rows)

    commutator = json.loads((out_dir / "commutator_summary.json").read_text(encoding="utf-8"))
    assert commutator["measured_commutator_count"] == 2
    assert commutator["needs_measurement_count"] == 0
    assert commutator["measurement_queue"] == []


def test_generate_inverse_evaluate_actions_cli_reads_wall_normal_lift_branches(
    tmp_path: Path,
) -> None:
    direct = _direct_wall_candidate()
    receipt = build_target_region_wall_normal_lift_receipt(
        action_id="wall-normal-action",
        pair_id=0,
        target_class=2,
        region_id="b0/c2/r1",
        direct_teacher_candidate=direct,
        backend_birth_receipt=_backend_birth_receipt(
            wrong_to_target=0,
            accepted=False,
        ),
        sidecar_candidate=direct,
    )
    training_artifact = tmp_path / "training_artifact.json"
    pr110_ledger = tmp_path / "pr110_action_effects.jsonl"
    out_dir = tmp_path / "out"
    training_artifact.write_text(
        json.dumps({"target_region_wall_normal_lift": receipt}),
        encoding="utf-8",
    )
    append_action_effect(_pr110_k1_replay_effect(), pr110_ledger)

    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "generate_inverse_evaluate_actions.py"),
            "--seed-training-artifact",
            str(training_artifact),
            "--pr110-action-effects",
            str(pr110_ledger),
            "--output-dir",
            str(out_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    rows = read_action_effects(out_dir / "action_effect_rows.jsonl")
    hinerv_rows = [row for row in rows if row.family == "hinerv"]
    assert [row.action_kind for row in hinerv_rows] == [
        "wall_normal_direct_teacher",
        "wall_normal_backend_fit",
        "wall_normal_sidecar_fallback",
    ]
    assert {row.action_id for row in hinerv_rows} == {"wall-normal-action"}
    assert {row.support_sha256 for row in hinerv_rows} == {
        receipt["direct_teacher"]["support_sha256"]
    }

    queue_rows = [
        json.loads(line)
        for line in (out_dir / "inverse_candidate_queue.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hinerv_queue = [row for row in queue_rows if row["family"] == "hinerv"]
    assert {row["action_kind"] for row in hinerv_queue} == {
        "wall_normal_direct_teacher",
        "wall_normal_backend_fit",
        "wall_normal_sidecar_fallback",
    }
    sidecar_row = next(row for row in hinerv_queue if row["action_kind"] == "wall_normal_sidecar_fallback")
    assert BLOCKER_SCORE_DELTA_MISSING in sidecar_row["blockers"]
    assert BLOCKER_WALL_NORMAL_SIDECAR_ARCHIVE_UNCLOSED in sidecar_row["blockers"]

    branch_receipt = json.loads(
        (out_dir / "wall_normal_branch_receipt.json").read_text(encoding="utf-8")
    )
    branch_rows = read_action_effects(out_dir / "wall_normal_branch_action_effect_rows.jsonl")
    lowering_race = json.loads(
        (out_dir / "wall_normal_branch_lowering_race.json").read_text(encoding="utf-8")
    )
    assert branch_receipt["schema"] == WALL_NORMAL_BRANCH_RECEIPT_SCHEMA
    assert lowering_race["schema"] == LOWERING_RACE_SCHEMA
    assert branch_receipt["branch_count"] == 3
    assert branch_receipt["same_action_id"] is True
    assert branch_receipt["first_failing_surface"] == BACKEND_REALIZATION_FAILED
    assert summary["wall_normal_branch_action_effect_row_count"] == 3
    assert summary["wall_normal_branch_action_effect_rows_path"] == (
        out_dir / "wall_normal_branch_action_effect_rows.jsonl"
    ).as_posix()
    assert summary["wall_normal_branch_lowering_race_path"] == (
        out_dir / "wall_normal_branch_lowering_race.json"
    ).as_posix()
    assert summary["wall_normal_branch_lowering_candidate_count"] == 3
    assert len(lowering_race["lowering_candidates"]) == 3
    assert lowering_race["support_identity"]["all_candidates_same_support"] is True
    assert [row.action_kind for row in branch_rows] == [
        "wall_normal_direct_teacher",
        "wall_normal_backend_fit",
        "wall_normal_sidecar_fallback",
    ]
    assert {row.action_id for row in branch_rows} == {"wall-normal-action"}


def test_generate_inverse_evaluate_actions_cli_scopes_wall_normal_receipt_to_fixed_action(
    tmp_path: Path,
) -> None:
    direct = _direct_wall_candidate()
    receipt_a = build_target_region_wall_normal_lift_receipt(
        action_id="wall-normal-action-a",
        pair_id=0,
        target_class=2,
        region_id="b0/c2/r1",
        direct_teacher_candidate=direct,
        backend_birth_receipt=_backend_birth_receipt(
            wrong_to_target=0,
            accepted=False,
        ),
        sidecar_candidate=direct,
    )
    receipt_b = build_target_region_wall_normal_lift_receipt(
        action_id="wall-normal-action-b",
        pair_id=0,
        target_class=2,
        region_id="b0/c2/r1",
        direct_teacher_candidate=direct,
        backend_birth_receipt=_backend_birth_receipt(
            wrong_to_target=0,
            accepted=False,
        ),
        sidecar_candidate=direct,
    )
    support_sha = receipt_a["direct_teacher"]["support_sha256"]
    training_artifact = tmp_path / "training_artifact.json"
    pr110_ledger = tmp_path / "pr110_action_effects.jsonl"
    out_dir = tmp_path / "out"
    training_artifact.write_text(
        json.dumps({"wall_normal_lifts": [receipt_a, receipt_b]}),
        encoding="utf-8",
    )
    append_action_effect(_pr110_k1_replay_effect(), pr110_ledger)

    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "generate_inverse_evaluate_actions.py"),
            "--seed-training-artifact",
            str(training_artifact),
            "--pr110-action-effects",
            str(pr110_ledger),
            "--wall-normal-action-id",
            "wall-normal-action-a",
            "--wall-normal-support-sha256",
            support_sha,
            "--output-dir",
            str(out_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    branch_receipt = json.loads(
        (out_dir / "wall_normal_branch_receipt.json").read_text(encoding="utf-8")
    )

    assert summary["wall_normal_receipt_filter"]["action_id"] == "wall-normal-action-a"
    assert summary["wall_normal_receipt_filter"]["support_sha256"] == support_sha
    assert summary["wall_normal_receipt_filter"]["filtered_count"] == 3
    assert summary["wall_normal_receipt_filter"]["dropped_count"] > 0
    assert branch_receipt["branch_count"] == 3
    assert branch_receipt["same_action_id"] is True
    assert branch_receipt["action_ids"] == ["wall-normal-action-a"]
    assert branch_receipt["same_support_sha256"] is True
    assert branch_receipt["support_sha256s"] == [support_sha]
    assert branch_receipt["blockers"] == []
    assert branch_receipt["first_failing_surface"] == BACKEND_REALIZATION_FAILED
    assert summary["wall_normal_branch_action_effect_row_count"] == 3
    assert summary["wall_normal_branch_action_effect_rows_path"] == (
        out_dir / "wall_normal_branch_action_effect_rows.jsonl"
    ).as_posix()
    assert summary["wall_normal_branch_lowering_race_path"] == (
        out_dir / "wall_normal_branch_lowering_race.json"
    ).as_posix()
    assert summary["wall_normal_branch_lowering_candidate_count"] == 3
    branch_rows = read_action_effects(out_dir / "wall_normal_branch_action_effect_rows.jsonl")
    lowering_race = json.loads(
        (out_dir / "wall_normal_branch_lowering_race.json").read_text(encoding="utf-8")
    )
    assert len(branch_rows) == 3
    assert {row.action_id for row in branch_rows} == {"wall-normal-action-a"}
    assert {row.support_sha256 for row in branch_rows} == {support_sha}
    assert lowering_race["schema"] == LOWERING_RACE_SCHEMA
    assert len(lowering_race["lowering_candidates"]) == 3
    assert lowering_race["support_identity"]["support_sha256s"] == [support_sha]


def test_wall_normal_fixed_scope_keeps_selected_receiver_bound_sidecar() -> None:
    support_sha = "a" * 64
    action_id = "wall-normal-action"
    direct = ActionEffect.build(
        action_id=action_id,
        family="hinerv",
        action_kind="wall_normal_direct_teacher",
        authority="batch_local_live_mlx",
        producer="fixture",
        consumer="fixture",
        pair_ids=[0],
        class_ids=[4],
        support_source="archive_executable_target_region_action_support",
        support_sha256=support_sha,
        support_cardinality=9,
        support_encoding="target_region_action_coordinates_v1",
        support_encoded_bytes=36,
        support_research_only=False,
        wrong_to_target=5,
        target_to_wrong=0,
        exact_score_decision="accept",
    )
    backend = ActionEffect.build(
        action_id=action_id,
        family="hinerv",
        action_kind="wall_normal_backend_fit",
        authority="batch_local_live_mlx",
        producer="fixture",
        consumer="fixture",
        pair_ids=[0],
        class_ids=[4],
        support_source="archive_executable_target_region_action_support",
        support_sha256=support_sha,
        support_cardinality=9,
        support_encoding="target_region_action_coordinates_v1",
        support_encoded_bytes=36,
        support_research_only=False,
        wrong_to_target=0,
        target_to_wrong=0,
        exact_score_decision="reject",
        blockers=[BLOCKER_WALL_NORMAL_BACKEND_NOT_REALIZED],
    )
    unbound_sidecar = ActionEffect.build(
        action_id=action_id,
        family="hinerv",
        action_kind="sidecar_grammar_candidate",
        authority="analysis_payload_model",
        producer="fixture",
        consumer="fixture",
        pair_ids=[0],
        class_ids=[4],
        payload_sections=["support_codec=rle_u32_start_len"],
        support_source="survived_target_region_action_sidecar",
        support_sha256=support_sha,
        support_cardinality=9,
        support_encoding="rle_u32_start_len",
        support_encoded_bytes=12,
        support_research_only=True,
        exact_score_decision="reject",
        blockers=["target_region_action_runtime_decoder_not_bound"],
    )
    receiver_sidecar = ActionEffect.build(
        action_id=action_id,
        family="hinerv",
        action_kind="sidecar_grammar",
        authority="inflate_raw",
        producer="fixture",
        consumer="fixture",
        pair_ids=[0],
        class_ids=[4],
        payload_sections=["support_codec=brotli_tile_bitmap_little_endian"],
        support_source="survived_target_region_action_sidecar",
        support_sha256=support_sha,
        support_cardinality=9,
        support_encoding="brotli_tile_bitmap_little_endian",
        support_encoded_bytes=8,
        support_research_only=False,
        wrong_to_target=5,
        target_to_wrong=0,
        exact_score_decision="accept",
        parseback_survived=True,
        inflate_survived=True,
    )

    kept, selection = inverse_materializer._wall_normal_receipt_effects(
        [direct, backend, unbound_sidecar, receiver_sidecar],
        action_id=action_id,
        support_sha256=support_sha,
        receiver_bound_only=False,
    )

    assert [row.action_kind for row in kept] == [
        "wall_normal_direct_teacher",
        "wall_normal_backend_fit",
        "sidecar_grammar",
    ]
    assert selection["branch_selection"]["selected_branch_kinds"] == [
        "direct_teacher",
        "backend_fit",
        "sidecar_fallback",
    ]
    assert any(
        item["action_kind"] == "sidecar_grammar_candidate"
        and item["reasons"] == ["dominated_fixed_wall_normal_sidecar_fallback_row"]
        for item in selection["branch_selection"]["dropped"]
    )
    receipt = build_wall_normal_branch_receipt(kept)
    assert receipt["branch_count"] == 3
    assert receipt["same_support_sha256"] is True
    assert receipt["blockers"] == []
    assert receipt["first_failing_surface"] == BACKEND_REALIZATION_FAILED


def test_generate_inverse_evaluate_actions_cli_does_not_call_blocked_reverse_producer_missing(
    tmp_path: Path,
) -> None:
    seed_ledger = tmp_path / "seed_action_effects.jsonl"
    pr110_ledger = tmp_path / "pr110_action_effects.jsonl"
    out_dir = tmp_path / "out"
    base_a = "1" * 64
    base_b = "2" * 64
    for effect in (
        _with_base_state(_frame0_pose_effect(), base_a),
        _with_base_state(_frame1_birth_effect(), base_a),
        _with_base_state(_composite_effect(), base_a),
        _with_base_state(_reverse_composite_effect(), base_b),
    ):
        append_action_effect(effect, seed_ledger)
    append_action_effect(_pr110_k1_replay_effect(), pr110_ledger)

    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "generate_inverse_evaluate_actions.py"),
            "--seed-action-effects",
            str(seed_ledger),
            "--pr110-action-effects",
            str(pr110_ledger),
            "--output-dir",
            str(out_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert "inverse_scorer_reverse_order_composite_producer_missing" not in (
        summary["commutator_measurement_blockers"]
    )
    assert any(
        str(blocker).startswith("measured_composite_incompatible:base_state_sha256 mismatch")
        for blocker in summary["commutator_measurement_blockers"]
    )


def test_convert_real_pr110_k16_packet_to_action_effect_clears_reproduction_gate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    packet_manifest = (
        repo_root
        / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
        / "packet_manifest.json"
    )
    archive_manifest = (
        repo_root
        / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
        / "archive_manifest.json"
    )
    ledger = tmp_path / "pr110_k16_action_effects.jsonl"

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "convert_pr110_k16_packet_to_action_effect.py"),
            "--packet-manifest",
            str(packet_manifest),
            "--archive-manifest",
            str(archive_manifest),
            "--output-jsonl",
            str(ledger),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    conversion = json.loads(proc.stdout)
    assert conversion["pair_count"] == 600
    assert conversion["selector_bits"] == 1944
    assert conversion["validation"]["passed"] is True

    rows = read_action_effects(ledger)
    assert len(rows) == 1
    proof = build_pr110_k16_baseline_reproduction_from_action_effects(rows)
    assert proof["passed"] is True
    assert proof["blockers"] == []
    assert proof["global_k"] == 16
    assert proof["pair_count"] == 600
    assert proof["selector_bits"] == 1944
