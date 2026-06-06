# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.analysis.action_effect import (
    ACTION_COMMUTATOR_ROW_SCHEMA,
    ACTION_EFFECT_LEDGER_SCHEMA,
    ACTION_EFFECT_PLANNING_AUTHORITY,
    ACTION_EFFECT_SCHEMA,
    ACTION_EFFECT_V1_SCHEMA,
    ActionEffect,
    DeltaScores,
    append_action_effect,
    build_action_commutator_row,
    build_action_effect,
    build_action_effect_ledger,
    compute_delta_scores,
    exact_delta_score,
    read_action_effects,
    validate_action_effect_payload,
    value_per_byte,
)
from tac.analysis.nerv_pair_local_distortion_servo import (
    PairLocalScoreState,
    PairLocalSurfaceTrace,
    admit_pair_local_distortion_action,
)
from tac.score_geometry import CONTEST_REFERENCE_BYTES, contest_score
from tac.substrates.hi_nerv.target_region_birth import (
    birth_action_id,
    build_target_region_birth_receipt,
    find_target_region_debts,
    select_worst_target_region,
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
    assert math.isclose(effect["value_per_byte"], 0.1 / 200.0)


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


def test_action_effect_serializes_pr110_selector_replay_row() -> None:
    effect = build_action_effect(
        {
            "action_id": "pr110_pair0042_frame0_mode07",
            "family": "selector",
            "authority": "parseback_pr110_selector_replay",
            "producer": "pr110_selector_sweep",
            "consumer": "selector_menu_ilp",
            "affected_pairs": [42],
            "affected_regions": [],
            "payload_sections": ["selector_stream", "mode_table"],
            "state_custody": {
                "source_archive_sha256": "c" * 64,
                "payload_sha256": "d" * 64,
            },
            "old_d_seg": 0.010,
            "new_d_seg": 0.010,
            "old_d_pose": 0.0004,
            "new_d_pose": 0.000324,
            "old_bytes": 1_000,
            "new_bytes": 1_016,
            "receiver_surface": {
                "posenet_input_delta_linf": 0.02,
                "pose_output_delta_l2": 0.006,
            },
            "fakequant_survived": True,
            "parseback_survived": True,
            "inflate_survived": True,
        }
    )

    assert effect["schema"] == ACTION_EFFECT_SCHEMA
    assert effect["family"] == "selector"
    assert effect["authority"] == "parseback_pr110_selector_replay"
    assert effect["receiver_visible"] is True
    assert effect["byte_priced"] is True
    assert effect["value_per_byte"] is not None
    assert effect["reported_value_per_byte"] is None
    assert effect["payload_sections"] == ["selector_stream", "mode_table"]
    assert effect["action_effect_admitted"] is True


def test_pair_local_action_effect_preserves_receipt_pair_ids() -> None:
    effect = ActionEffect.from_pair_local_admission(
        {
            "schema": "nerv_pair_local_distortion_servo_receipt.v1",
            "family": "hi_nerv",
            "authority": "parseback_mlx",
            "actuator_id": "hinerv_pair_17",
            "pair_ids": [17],
            "old_d_seg": 0.020,
            "new_d_seg": 0.019,
            "old_d_pose": 0.0020,
            "new_d_pose": 0.00195,
            "old_archive_bytes": 178_000,
            "new_archive_bytes": 178_128,
        }
    )

    assert effect.schema == ACTION_EFFECT_V1_SCHEMA
    assert effect.family == "hinerv"
    assert effect.action_id == "hinerv_pair_17"
    assert effect.pair_ids == (17,)
    assert effect.delta_score_total is not None


def test_action_effect_rejects_malformed_custody_hashes() -> None:
    effect = build_action_effect(
        {
            "action_id": "bad_hash_custody",
            "family": "hinerv",
            "authority": "parseback_mlx",
            "producer": "unit_test",
            "consumer": "nerv_long_training_campaign_admission",
            "state_custody": {"archive_sha256": "not-a-sha"},
            "old_d_seg": 0.010,
            "new_d_seg": 0.009,
            "old_d_pose": 0.0001,
            "new_d_pose": 0.0001,
            "old_bytes": 1000,
            "new_bytes": 1000,
            "receiver_surface": {
                "uint8_changed_pixels": 1,
                "argmax_flipped_pixels": 1,
            },
            "fakequant_survived": True,
            "parseback_survived": True,
        }
    )

    assert effect["state_custody"] == {}
    assert "action_effect_state_custody_hash_missing" in effect["blockers"]
    assert effect["action_effect_admitted"] is False


def test_action_effect_keeps_reported_value_per_byte_separate_for_zero_byte_delta() -> None:
    effect = build_action_effect(
        {
            "action_id": "zero_byte_reported_value",
            "family": "hinerv",
            "authority": "parseback_mlx",
            "producer": "unit_test",
            "consumer": "nerv_long_training_campaign_admission",
            "archive_sha256": "a" * 64,
            "old_d_seg": 0.010,
            "new_d_seg": 0.009,
            "old_d_pose": 0.0001,
            "new_d_pose": 0.0001,
            "old_bytes": 1000,
            "new_bytes": 1000,
            "receiver_surface": {
                "uint8_changed_pixels": 1,
                "argmax_flipped_pixels": 1,
            },
            "fakequant_survived": True,
            "parseback_survived": True,
            "value_per_byte": 123.0,
        }
    )

    assert effect["action_effect_admitted"] is True
    assert effect["value_per_byte"] is None
    assert effect["reported_value_per_byte"] == 123.0


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
        "receiver_surface": {"uint8_changed_pixels": 1, "argmax_flipped_pixels": 1},
        "fakequant_survived": True,
        "parseback_survived": True,
    }


# ════════════════════════════════════════════════════════════════════════
# Partner amendment #5 — thin ``tac.action_effect.v1`` currency tests.
#
# These assert BEHAVIOR: the exact nonlinear score deltas are hand-checked
# against tac.score_geometry.contest_score so a marker-stub that returned
# canonical metadata without doing the scoring math would FAIL.  Real landed
# schemas are exercised end-to-end (hi_nerv_target_region_birth_receipt.v1,
# nerv_pair_local_distortion_servo_admission.v1, servo receipt, PR110 rows).
# ════════════════════════════════════════════════════════════════════════


def _real_birth_receipt(
    *,
    surface: str = "parseback_mlx",
    old_d_seg: float = 0.0010,
    new_d_seg: float = 0.0008,
    old_d_pose: float = 1.0e-4,
    new_d_pose: float = 1.0e-4,
    with_exact_nonrate: bool = True,
    old_archive_bytes: int | None = None,
    new_archive_bytes: int | None = None,
    runtime_sidecar_bytes: int = 128,
) -> tuple[dict, str]:
    """Build a real ``hi_nerv_target_region_birth_receipt.v1`` + its action_id.

    The (1,6,8) label grid is a deterministic UNIT FIXTURE for the region-debt
    math; it makes no empirical/score-authority claim.
    """

    target = np.zeros((1, 6, 8), dtype=np.int64)
    target[0, 1:4, 1:5] = 2
    candidate = target.copy()
    candidate[0, 1:4, 1:5] = 0  # whole region wrong → priced debt
    worst = select_worst_target_region(find_target_region_debts(target, candidate))
    action_id = birth_action_id(
        debt=worst,
        initial_group_sha256={"latents_fine": "a" * 64},
        trained_groups=["latents_fine"],
    )
    exact_nonrate = (
        {
            "old_d_seg": old_d_seg,
            "new_d_seg": new_d_seg,
            "old_d_pose": old_d_pose,
            "new_d_pose": new_d_pose,
        }
        if with_exact_nonrate
        else None
    )
    receipt = build_target_region_birth_receipt(
        debt=worst,
        before_margin_stats={"margin_p50": 0.5, "region_hard_ratio": 0.0, "region_pixel_count": 12.0},
        after_margin_stats={"margin_p50": -0.2, "region_hard_ratio": 0.9, "region_pixel_count": 12.0},
        receiver_uint8_changed_pixels_region=12,
        receiver_uint8_delta_abs_max=7.0,
        receiver_float_rgb_delta_linf=0.03,
        argmax_flipped_pixels_region=10,
        accepted_step_count=5,
        rejected_step_count=1,
        blockers=[],
        grad_norm_by_group={"latents_fine": 0.1},
        update_norm_by_group={"latents_fine": 0.01},
        updated_parameter_names=["latents_fine"],
        pose_guard={"ok": True},
        runtime_sidecar_bytes=runtime_sidecar_bytes,
        argmax_transitions={
            "wrong_to_target_count": 10,
            "target_to_wrong_count": 0,
            "net_target_support_delta": 10,
        },
        exact_nonrate=exact_nonrate,
        action_id=action_id,
        surface=surface,
    )
    if old_archive_bytes is not None:
        receipt["old_archive_bytes"] = old_archive_bytes
    if new_archive_bytes is not None:
        receipt["new_archive_bytes"] = new_archive_bytes
    return receipt, action_id


def _real_servo_admission_dict() -> dict:
    """Build a real ``nerv_pair_local_distortion_servo_admission.v1`` dict."""

    before = PairLocalScoreState(d_seg=0.0010, d_pose=1.0e-4, archive_bytes=178258)
    after = PairLocalScoreState(d_seg=0.0008, d_pose=1.0e-4, archive_bytes=178258)
    trace = PairLocalSurfaceTrace(
        family="hinerv",
        frame_scope="frame1_seg_pose_joint",
        actuator_id="hinerv_target_region_birth",
        pair_index=43,
        float_rgb_delta_linf=0.02,
        uint8_changed_pixels=12,
        segnet_input_delta_linf=0.01,
        segnet_margin_delta=-0.5,
        target_hard_won_count=8,
        net_target_support_delta=8,
        fakequant_segnet_margin_delta=-0.4,
        fakequant_target_hard_won_count=7,
        parseback_segnet_margin_delta=-0.3,
        parseback_target_hard_won_count=6,
    )
    return admit_pair_local_distortion_action(before=before, after=after, trace=trace).as_dict()


def test_hinerv_four_arm_ablation_imports_receipt_shaped_rows() -> None:
    receipt, action_id = _real_birth_receipt(surface="live_mlx")
    arm_a = {
        **receipt,
        "schema": "hi_nerv_target_region_birth_four_arm.v1",
        "arm": "A",
        "action_kind": "birth_only",
        "action_id": action_id,
        "blockers": [],
        "interaction_or_commutator": 0.0,
    }
    arm_b = {
        **receipt,
        "schema": "hi_nerv_target_region_birth_four_arm.v1",
        "arm": "B",
        "action_kind": "frame0_pose_target_only",
        "action_id": action_id,
        "accepted": False,
        "blockers": ["hinerv_four_arm_frame0_pose_snapshot_missing"],
        "updated_parameter_names": ["head_rgb_0.weight"],
        "payload_sections": ["head_rgb_0.weight"],
    }

    effects = ActionEffect.from_hinerv_four_arm_ablation(
        {
            "schema": "hi_nerv_target_region_birth_four_arm_ablation.v1",
            "action_id": action_id,
            "arms": [arm_a, arm_b],
        }
    )

    assert len(effects) == 2
    rows = [effect.as_dict() for effect in effects]
    assert rows[0]["action_id"] == action_id
    assert rows[0]["interaction_or_commutator"] == pytest.approx(0.0)
    assert rows[0]["blockers"] == []
    assert rows[1]["payload_sections"] == ["head_rgb_0.weight"]
    assert rows[1]["blockers"] == ["hinerv_four_arm_frame0_pose_snapshot_missing"]
    assert rows[1]["rejection_source"] is None


def _servo_receipt_with_absolute_endpoints() -> dict:
    return {
        "schema": "nerv_pair_local_distortion_servo_receipt.v1",
        "actuator_id": "hinerv_target_region_birth",
        "family": "hinerv",
        "pair_index": 43,
        "authority": "parseback_mlx",
        "old_d_seg": 0.0010,
        "new_d_seg": 0.0008,
        "old_d_pose": 1.0e-4,
        "new_d_pose": 1.0e-4,
        "old_archive_bytes": 178258,
        "new_archive_bytes": 178258,
        "surfaces": {"parseback_survival": True, "inflate_survival": False},
        "affected_regions": ["b0/c2/r1"],
        "payload_sections": ["latents_fine"],
    }


def _pr110_lattice_atom() -> dict:
    return {
        "atom_id": "pair:2:mode:none",
        "scope": {"pair_index": 2},
        "scope_kind": "pair",
        "score": {
            "seg_dist": 0.0005544026498682797,
            "pose_dist": 3.321919939480722e-05,
            "score_delta": None,
        },
        "budget": {"archive_delta_bytes": 0},
        "metadata": {
            "family": "identity",
            "mode_id": "none",
            "frame_indices": [4, 5],
            "source_row": {"pair": 2, "segnet_dist": 0.0005544, "posenet_dist": 3.3e-05},
        },
    }


def _pr110_selector_candidate() -> dict:
    return {
        "candidate_id": "lfv1v2_k01_a0p00001_r0p45_p0p8_oy0p38_76f61ee22b55",
        "kind": "hfv1_sidecar_candidate_v1",
        "selected_pairs": [43],
        "selected_frames": [86, 87],
        "archive": {
            "bytes": 178674,
            "delta_bytes_vs_source_archive": 157,
            "members": [
                {"name": "lapose_foveation_tuples.lfv1", "bytes": 25},
                {"name": "x", "bytes": 178417},
            ],
        },
    }


# ── compute_delta_scores: hand-checked exact nonlinear math ────────────────


def test_v1_compute_delta_scores_matches_direct_contest_score_exactly() -> None:
    old = (0.0012, 2.0e-4, 178258)
    new = (0.0009, 1.0e-4, 178358)
    ds = compute_delta_scores(old[0], new[0], old[1], new[1], old[2], new[2])
    direct_total = contest_score(*new) - contest_score(*old)
    assert ds.delta_score_total == pytest.approx(direct_total, abs=1e-15)
    rate = 25.0 * (new[2] - old[2]) / CONTEST_REFERENCE_BYTES
    assert ds.delta_score_nonrate == pytest.approx(direct_total - rate, abs=1e-15)


def test_v1_value_per_byte_hand_checked_number() -> None:
    n = CONTEST_REFERENCE_BYTES
    expect_nonrate = 100.0 * (0.0008 - 0.0010)  # = -0.02
    expect_total = expect_nonrate + 25.0 * 100 / n
    expect_vpb = -expect_total / 100.0
    ds = compute_delta_scores(0.0010, 0.0008, 1.0e-4, 1.0e-4, 178258, 178358)
    assert ds.delta_bytes == 100
    assert ds.delta_score_nonrate == pytest.approx(expect_nonrate, abs=1e-12)
    assert ds.delta_score_total == pytest.approx(expect_total, abs=1e-12)
    assert ds.value_per_byte == pytest.approx(expect_vpb, abs=1e-15)
    assert ds.value_per_byte > 0.0


def test_v1_public_exact_delta_and_value_helpers_are_single_source() -> None:
    old = (0.0012, 2.0e-4, 178258)
    new = (0.0012, 2.0e-4, 178221)
    delta = exact_delta_score(old[0], new[0], old[1], new[1], old[2], new[2])
    assert delta == pytest.approx(25.0 * (new[2] - old[2]) / CONTEST_REFERENCE_BYTES, abs=1e-15)
    assert value_per_byte(delta, new[2] - old[2]) == pytest.approx(25.0 / CONTEST_REFERENCE_BYTES, abs=1e-18)


def test_v1_value_per_byte_none_when_bytes_unchanged() -> None:
    ds = compute_delta_scores(0.0010, 0.0008, 1.0e-4, 1.0e-4, 178258, 178258)
    assert ds.delta_bytes == 0
    assert ds.value_per_byte is None
    assert ds.delta_score_total == pytest.approx(ds.delta_score_nonrate, abs=1e-15)


def test_v1_nonrate_none_when_endpoint_missing() -> None:
    ds = compute_delta_scores(None, 0.0008, 1.0e-4, 1.0e-4, None, None)
    assert ds.delta_score_nonrate is None
    assert ds.delta_score_total is None
    assert ds.value_per_byte is None


def test_v1_total_none_when_bytes_unknown() -> None:
    ds = compute_delta_scores(0.0010, 0.0008, 1.0e-4, 1.0e-4, None, None)
    assert ds.delta_score_nonrate is not None
    assert ds.delta_score_total is None
    assert isinstance(ds, DeltaScores)


def test_v1_compute_delta_scores_rejects_negative_distortion() -> None:
    with pytest.raises(ValueError):
        compute_delta_scores(-0.1, 0.0, 0.0, 0.0, None, None)


# ── ActionEffect validation ────────────────────────────────────────────────


def test_v1_authority_required_empty_raises() -> None:
    with pytest.raises(ValueError, match="authority"):
        ActionEffect.build(action_id="a1", family="hinerv", authority="   ", producer="p")


def test_v1_promotion_eligible_true_raises() -> None:
    base = ActionEffect.build(action_id="a1", family="hinerv", authority="parseback_mlx", producer="p")
    payload = base.as_dict()
    payload.pop("old_archive_bytes")
    payload.pop("new_archive_bytes")
    with pytest.raises(ValueError, match="promotion_eligible"):
        ActionEffect(**{**payload, "promotion_eligible": True})


def test_v1_action_id_required_nonempty() -> None:
    with pytest.raises(ValueError, match="action_id"):
        ActionEffect.build(action_id="  ", family="hinerv", authority="x", producer="p")


def test_v1_built_effect_schema_and_planning_marker() -> None:
    eff = ActionEffect.build(
        action_id="a1",
        family="pr110",
        authority=ACTION_EFFECT_PLANNING_AUTHORITY,
        producer="p",
    )
    assert eff.schema == ACTION_EFFECT_V1_SCHEMA
    assert eff.promotion_eligible is False
    assert eff.authority == ACTION_EFFECT_PLANNING_AUTHORITY


def test_v1_carries_no_canonical_false_authority_keys() -> None:
    # single-custody-surface rule: must NOT spread canonical promotion keys.
    eff = ActionEffect.build(action_id="a1", family="hinerv", authority="parseback_mlx", producer="p")
    payload = eff.as_dict()
    for forbidden in (
        "score_claim",
        "score_claim_valid",
        "rank_or_kill_eligible",
        "promotable",
        "ready_for_exact_eval_dispatch",
    ):
        assert forbidden not in payload, f"{forbidden} must not appear on a thin ActionEffect row"
    assert payload["promotion_eligible"] is False


# ── constructor: HiNeRV birth receipt (real schema) ────────────────────────


def test_v1_from_hinerv_birth_receipt_real_schema_roundtrips() -> None:
    receipt, action_id = _real_birth_receipt(surface="parseback_mlx")
    receipt["receiver_surface"] = {"segnet_input_delta_linf": 0.03125}
    receipt["pose_guard"]["posenet_input_delta_linf_pair"] = 0.015625
    eff = ActionEffect.from_hinerv_birth_receipt(receipt)
    assert eff.schema == ACTION_EFFECT_V1_SCHEMA
    assert eff.action_id == action_id  # CARRIED, not recomputed
    assert eff.family == "hinerv"
    assert eff.authority == "parseback_mlx"
    assert eff.region_ids == ("b0/c2/r1",)
    assert eff.payload_sections == ("latents_fine",)
    assert eff.old_d_seg == pytest.approx(0.0010)
    assert eff.new_d_seg == pytest.approx(0.0008)
    assert eff.delta_score_nonrate == pytest.approx(100.0 * (0.0008 - 0.0010), abs=1e-12)
    assert eff.parseback_survived is True
    assert eff.hard_won_count == 10
    assert eff.wrong_to_target == 10
    assert eff.target_to_wrong == 0
    assert eff.net_target_support_delta == 10
    assert eff.uint8_changed_count_region == 12
    assert eff.seg_input_delta_linf_region == pytest.approx(0.03125)
    assert eff.posenet_input_delta_linf_pair == pytest.approx(0.015625)
    roundtrip = ActionEffect.from_dict(eff.as_dict())
    assert roundtrip.hard_won_count == 10
    assert roundtrip.uint8_changed_count_region == 12


def test_v1_from_hinerv_birth_receipt_no_pose_teacher_leaves_distortion_none() -> None:
    receipt, _ = _real_birth_receipt(with_exact_nonrate=False)
    eff = ActionEffect.from_hinerv_birth_receipt(receipt)
    assert eff.old_d_seg is None
    assert eff.new_d_seg is None
    assert eff.delta_score_nonrate is None  # never fabricated


def test_v1_from_hinerv_birth_receipt_carries_v6_pose_trusted_admission_fields() -> None:
    receipt, action_id = _real_birth_receipt(
        old_d_seg=0.49237060546875,
        new_d_seg=0.4360555013020833,
        old_d_pose=194.20106506347656,
        new_d_pose=196.9571990966797,
    )
    receipt["action_kind"] = "four_arm_composite_ablation"
    receipt["candidate_frontier_telemetry"] = {
        "attempts": [
            {
                "arm": "A",
                "exact_score_decision": "rejected",
                "raw_cap_decision": "violated_counterfactual_only",
                "catastrophic_guard_decision": "satisfied",
                "would_accept_exact_score_if_raw_cap_disabled": False,
                "would_accept_without_catastrophic_guard": False,
                "rejected_by_raw_pose_cap": False,
                "rejected_by_exact_delta_score": True,
                "rejected_by_catastrophic_pose_guard": False,
                "pose_output_l2_delta": 0.37355837225914,
                "seg_score_delta": 0.0,
                "pose_score_delta": 0.0,
                "rejection_source": "rejected_by_exact_delta_score",
            },
            {
                "arm": "C",
                "decision": "backtrack_no_unsolved_tail_progress",
                "exact_score_decision": "accepted",
                "raw_cap_decision": "violated_counterfactual_only",
                "catastrophic_guard_decision": "satisfied",
                "would_accept_exact_score_if_raw_cap_disabled": True,
                "would_accept_without_catastrophic_guard": True,
                "rejected_by_raw_pose_cap": False,
                "rejected_by_exact_delta_score": False,
                "rejected_by_catastrophic_pose_guard": False,
                "pose_output_l2_delta": 1.1900334358215332,
                "seg_score_delta": 1.384989420572913,
                "pose_score_delta": -2.9173961550826245,
                "rejection_source": None,
            },
            {
                "arm": "D",
                "decision": "accepted",
                "exact_score_decision": "accepted",
                "raw_cap_decision": "violated_counterfactual_only",
                "catastrophic_guard_decision": "satisfied",
                "would_accept_exact_score_if_raw_cap_disabled": True,
                "would_accept_without_catastrophic_guard": True,
                "rejected_by_raw_pose_cap": False,
                "rejected_by_exact_delta_score": False,
                "rejected_by_catastrophic_pose_guard": False,
                "pose_output_l2_delta": 0.37355837225914,
                "seg_score_delta": -5.631510416666668,
                "pose_score_delta": 0.3116102797153206,
                "rejection_source": None,
            },
        ],
    }

    eff = ActionEffect.from_hinerv_birth_receipt(receipt)

    assert eff.action_id == action_id
    assert eff.action_kind == "four_arm_composite_ablation"
    assert eff.arm == "D"
    assert eff.exact_score_decision == "accept"
    assert eff.raw_cap_decision == "violated_counterfactual_only"
    assert eff.catastrophic_guard_decision == "satisfied"
    assert eff.would_accept_exact_score_if_raw_cap_disabled is True
    assert eff.would_accept_without_catastrophic_guard is True
    assert eff.rejected_by_raw_cap is False
    assert eff.rejected_by_exact_score is False
    assert eff.rejected_by_catastrophic_guard is False
    assert eff.pose_output_l2_delta == pytest.approx(0.37355837225914)
    assert eff.seg_score_delta == pytest.approx(-5.631510416666668)
    assert eff.pose_score_delta == pytest.approx(0.3116102797153206)
    assert eff.rejection_source is None
    roundtrip = ActionEffect.from_dict(eff.as_dict())
    assert roundtrip.arm == "D"
    assert roundtrip.raw_cap_decision == "violated_counterfactual_only"
    assert roundtrip.would_accept_exact_score_if_raw_cap_disabled is True
    assert roundtrip.rejected_by_exact_score is False
    assert roundtrip.pose_output_l2_delta == pytest.approx(0.37355837225914)


def test_v1_from_hinerv_four_arm_ablation_expands_all_real_arm_rows() -> None:
    receipt, action_id = _real_birth_receipt(
        old_d_seg=0.50,
        new_d_seg=0.47,
        old_d_pose=194.0,
        new_d_pose=194.2,
    )

    def arm(arm_id: str, action_kind: str, *, new_d_seg: float, blockers: list[str] | None = None) -> dict:
        old_d_seg = 0.50
        old_d_pose = 194.0
        new_d_pose = 194.2
        return {
            "schema": "hi_nerv_target_region_birth_four_arm.v1",
            "action_id": action_id,
            "surface": "live_mlx",
            "authority": "batch_local_live_mlx",
            "normalization_scope": "batch_local",
            "action_kind": action_kind,
            "arm": arm_id,
            "pair_index": 0,
            "worst_region": receipt["worst_region"],
            "updated_parameter_names": ["head_rgb_0.weight"] if arm_id == "B" else ["head_rgb_1.weight"],
            "exact_nonrate": {
                "old_d_seg_batch": old_d_seg,
                "new_d_seg_batch": new_d_seg,
                "old_d_pose_batch": old_d_pose,
                "new_d_pose_batch": new_d_pose,
                "pose_term_available": True,
            },
            "argmax_transitions": {
                "wrong_to_target_count": 3 if arm_id != "B" else 0,
                "target_to_wrong_count": 0,
                "wrong_to_wrong_count": 1,
                "net_target_support_delta": 3 if arm_id != "B" else 0,
            },
            "receiver_surface_uint8_changed_pixels": 4 if arm_id != "B" else 0,
            "receiver_uint8_delta_abs_max": 3.0,
            "receiver_float_rgb_delta_linf": 0.02,
            "argmax_changed_count_region": 4 if arm_id != "B" else 0,
            "pose_output_l2_delta": 0.2,
            "interaction_or_commutator": -0.125 if arm_id in {"C", "D"} else None,
            "admission_decision": {
                "arm": arm_id,
                "exact_score_decision": "accepted" if not blockers else "rejected",
                "raw_cap_decision": "violated_counterfactual_only",
                "catastrophic_guard_decision": "satisfied",
                "would_accept_exact_score_if_raw_cap_disabled": not blockers,
                "would_accept_without_catastrophic_guard": not blockers,
                "rejected_by_raw_pose_cap": False,
                "rejected_by_exact_delta_score": bool(blockers),
                "rejected_by_catastrophic_pose_guard": False,
                "pose_output_l2_delta": 0.2,
                "seg_score_delta": 100.0 * (new_d_seg - old_d_seg),
                "pose_score_delta": math.sqrt(10.0 * new_d_pose) - math.sqrt(10.0 * old_d_pose),
                "rejection_source": None if not blockers else "blocked_by_fixture",
            },
            "blockers": blockers or [],
        }

    effects = ActionEffect.from_hinerv_four_arm_ablation(
        {
            "action_id": action_id,
            "four_arm_ablation": {
                "schema": "hi_nerv_target_region_birth_four_arm_ablation.v1",
                "action_id": action_id,
                "arms": [
                    arm("A", "birth_only", new_d_seg=0.47),
                    arm("B", "frame0_pose_target_only", new_d_seg=0.50, blockers=["b_has_no_seg_birth"]),
                    arm("C", "independent_birth_plus_frame0_pose", new_d_seg=0.46),
                    arm("D", "joint_line_search_composite", new_d_seg=0.45),
                ],
            },
        }
    )

    assert [effect.arm for effect in effects] == ["A", "B", "C", "D"]
    assert [effect.action_kind for effect in effects] == [
        "birth_only",
        "frame0_pose_target_only",
        "independent_birth_plus_frame0_pose",
        "joint_line_search_composite",
    ]
    assert effects[0].exact_score_decision == "accept"
    assert effects[1].exact_score_decision == "reject"
    assert effects[1].rejected_by_exact_score is True
    assert effects[1].blockers == ("b_has_no_seg_birth",)
    assert effects[2].interaction_or_commutator == pytest.approx(-0.125)
    assert effects[3].wrong_to_target == 3
    roundtrip = [ActionEffect.from_dict(effect.as_dict()) for effect in effects]
    assert roundtrip[1].blockers == ("b_has_no_seg_birth",)
    assert roundtrip[1].rejected_by_exact_score is True
    assert roundtrip[3].arm == "D"


# ── constructor: pair-local servo (real schema) ────────────────────────────


def test_v1_from_pair_local_admission_real_admission_roundtrips_structurally() -> None:
    adm = _real_servo_admission_dict()
    assert adm["schema"] == "nerv_pair_local_distortion_servo_admission.v1"
    eff = ActionEffect.from_pair_local_admission(adm)
    assert eff.schema == ACTION_EFFECT_V1_SCHEMA
    assert eff.action_id == "hinerv_target_region_birth"
    assert eff.family == "hinerv"
    assert eff.pair_ids == (43,)
    assert eff.parseback_survived is True
    assert eff.hard_won_count == 8
    assert eff.wrong_to_target == 8
    assert eff.net_target_support_delta == 8
    assert eff.uint8_changed_count_region == 12
    assert eff.seg_input_delta_linf_region == pytest.approx(0.01)
    # deltas-only admission ⇒ no absolute endpoints ⇒ no fabricated nonrate.
    assert eff.old_d_seg is None
    assert eff.delta_score_nonrate is None


def test_v1_from_pair_local_admission_receipt_with_absolute_endpoints() -> None:
    eff = ActionEffect.from_pair_local_admission(_servo_receipt_with_absolute_endpoints())
    assert eff.old_d_seg == pytest.approx(0.0010)
    assert eff.new_d_seg == pytest.approx(0.0008)
    assert eff.old_bytes == 178258
    assert eff.new_bytes == 178258
    assert eff.delta_score_nonrate == pytest.approx(100.0 * (0.0008 - 0.0010), abs=1e-12)


def test_v1_both_paths_compute_delta_via_the_same_function() -> None:
    # Load-bearing partner invariant: identical d-inputs on the HiNeRV birth
    # path and the pair-local servo path yield BYTE-IDENTICAL score deltas
    # because both route through compute_delta_scores().
    receipt, _ = _real_birth_receipt(old_archive_bytes=178258, new_archive_bytes=178258)
    birth_eff = ActionEffect.from_hinerv_birth_receipt(receipt)
    servo_eff = ActionEffect.from_pair_local_admission(_servo_receipt_with_absolute_endpoints())
    assert birth_eff.delta_score_nonrate == servo_eff.delta_score_nonrate
    assert birth_eff.delta_score_total == servo_eff.delta_score_total
    direct = contest_score(0.0008, 1.0e-4, 178258) - contest_score(0.0010, 1.0e-4, 178258)
    assert birth_eff.delta_score_total == pytest.approx(direct, abs=1e-15)


# ── constructor: PR110 selector / lattice atom (real shapes) ───────────────


def test_v1_from_pr110_lattice_atom() -> None:
    eff = ActionEffect.from_pr110_selector_row(_pr110_lattice_atom())
    assert eff.action_id == "pair:2:mode:none"
    assert eff.pair_ids == (2,)
    assert eff.new_d_seg == pytest.approx(0.0005544026498682797)
    assert eff.new_d_pose == pytest.approx(3.321919939480722e-05)
    assert eff.old_d_seg is None and eff.old_d_pose is None
    assert eff.payload_sections == ("none",)
    assert eff.promotion_eligible is False


def test_v1_from_pr110_selector_candidate() -> None:
    eff = ActionEffect.from_pr110_selector_row(_pr110_selector_candidate())
    assert eff.action_id == "lfv1v2_k01_a0p00001_r0p45_p0p8_oy0p38_76f61ee22b55"
    assert eff.pair_ids == (43,)
    assert eff.new_d_seg is None and eff.new_d_pose is None
    assert eff.new_bytes == 178674
    assert eff.old_bytes == 178674 - 157
    assert eff.payload_sections == ("lapose_foveation_tuples.lfv1", "x")
    assert eff.delta_score_total == pytest.approx(25.0 * 157 / CONTEST_REFERENCE_BYTES, abs=1e-15)


def test_v1_from_pr110_exact_replay_carries_cpu_authority_and_exact_delta() -> None:
    row = _pr110_selector_candidate()
    row.update(
        {
            "action_kind": "selector_mode",
            "authority": "contest_cpu",
            "old_d_seg": 0.00056039,
            "new_d_seg": 0.00055979,
            "old_d_pose": 2.943e-05,
            "new_d_pose": 2.943e-05,
            "old_archive_bytes": 178517,
            "new_archive_bytes": 178493,
            "official_inflate_control_passed": True,
            "archive_sha256": "b" * 64,
        }
    )
    eff = ActionEffect.from_pr110_selector_row(row)
    assert eff.authority == "contest_cpu"
    assert eff.normalization_scope == "full_video_exact"
    assert eff.old_bytes == 178517
    assert eff.new_bytes == 178493
    assert eff.delta_score_total == pytest.approx(
        exact_delta_score(0.00056039, 0.00055979, 2.943e-05, 2.943e-05, 178517, 178493),
        abs=1e-15,
    )
    assert eff.value_per_byte is not None
    assert eff.promotion_eligible is False
    assert "score_claim" not in eff.as_dict()


def test_v1_from_frontier_rate_materializer_prices_current_final_rate_saving() -> None:
    manifest = {
        "schema": "fp11_source_brotli_recode_manifest.v1",
        "operation_family": "fp11_source_brotli_recode_v1",
        "source_archive": {"bytes": 178530, "sha256": "1" * 64},
        "candidate_archive": {
            "bytes": 178493,
            "sha256": "b7106c9bdbb8a2df18af622636ca79a11fa0c771a09c75219474d980b8997c8c",
        },
        "selected_member_names": ["x"],
        "receiver_contract_satisfied": True,
        "receiver_proof_ready": True,
        "score_claim": False,
    }
    auth_eval = {
        "lane_tag": "[contest-CPU]",
        "score_axis": "contest_cpu",
        "archive_size_bytes": 178493,
        "avg_segnet_dist": 0.00055979,
        "avg_posenet_dist": 2.943e-05,
    }
    eff = ActionEffect.from_frontier_rate_materializer(manifest, auth_eval=auth_eval)
    assert eff.family == "frontier_rate_attack"
    assert eff.authority == "[contest-CPU] frontier_final_rate_attack"
    assert eff.normalization_scope == "full_video_exact"
    assert eff.payload_sections == ("x",)
    assert eff.old_bytes == 178530
    assert eff.new_bytes == 178493
    assert eff.delta_score_total == pytest.approx(25.0 * -37 / CONTEST_REFERENCE_BYTES, abs=1e-15)
    assert eff.value_per_byte == pytest.approx(25.0 / CONTEST_REFERENCE_BYTES, abs=1e-18)
    assert eff.parseback_survived is True
    assert eff.inflate_survived is True
    assert eff.archive_sha256 == "b7106c9bdbb8a2df18af622636ca79a11fa0c771a09c75219474d980b8997c8c"


def test_frontier_rate_materializer_cli_emits_valid_action_effect(tmp_path: Path) -> None:
    proof_path = tmp_path / "runtime_proof.json"
    proof_payload = {
        "schema": "fp11_source_brotli_recode_runtime_consumption_proof.v1",
        "runtime_consumption_proof_passed": True,
        "passed": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    proof_path.write_text(json.dumps(proof_payload), encoding="utf-8")
    proof_sha = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "fp11_source_brotli_recode_manifest.json"
    manifest = {
        "schema": "fp11_source_brotli_recode_manifest.v1",
        "target_kind": "fp11_source_brotli_recode_v1",
        "operation_family": "source_brotli_recode",
        "materializer_id": "fp11_source_brotli_recode_adapter",
        "source_archive": {"bytes": 178530, "sha256": "1" * 64},
        "candidate_archive": {
            "bytes": 178493,
            "sha256": "b7106c9bdbb8a2df18af622636ca79a11fa0c771a09c75219474d980b8997c8c",
        },
        "selected_member_names": ["x"],
        "receiver_contract_satisfied": True,
        "receiver_proof_ready": True,
        "runtime_consumption_proof_path": proof_path.as_posix(),
        "runtime_consumption_proof_sha256": proof_sha,
        "score_claim": False,
        "promotion_eligible": False,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out_jsonl = tmp_path / "action_effect_rows.jsonl"

    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "convert_frontier_rate_materializer_to_action_effect.py"),
            "--manifest",
            str(manifest_path),
            "--output-jsonl",
            str(out_jsonl),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["validation"]["passed"] is True
    rows = [json.loads(line) for line in out_jsonl.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["schema"] == ACTION_EFFECT_V1_SCHEMA
    assert row["family"] == "frontier_rate_attack"
    assert row["authority"] == "receiver_closed_frontier_rate_attack"
    assert row["old_archive_bytes"] == 178530
    assert row["new_archive_bytes"] == 178493
    assert row["delta_score_total"] == pytest.approx(25.0 * -37 / CONTEST_REFERENCE_BYTES, abs=1e-15)
    assert row["value_per_byte"] == pytest.approx(25.0 / CONTEST_REFERENCE_BYTES, abs=1e-18)
    assert row["artifact_ref"] == manifest_path.resolve(strict=False).as_posix()
    assert row["inflate_survived"] is True
    assert row["restore_state_pass"] is True
    assert "score_claim" not in row


def test_v1_from_pr110_tolerant_on_unknown_shape() -> None:
    eff = ActionEffect.from_pr110_selector_row({"unrecognized": True})
    assert eff.action_id == "pr110_selector_action"
    assert eff.pair_ids == ()
    assert eff.new_d_seg is None
    assert eff.old_bytes is None and eff.new_bytes is None


# ── JSONL ledger (fcntl-locked append + read) ──────────────────────────────


def test_v1_ledger_append_and_read_roundtrip(tmp_path) -> None:
    ledger = tmp_path / "action_effect_ledger.jsonl"
    e1 = ActionEffect.build(
        action_id="a1",
        family="hinerv",
        authority="parseback_mlx",
        producer="hinerv_target_region_birth",
        pair_ids=[1, 2],
        region_ids=["b0/c2/r1"],
        old_d_seg=0.0010,
        new_d_seg=0.0008,
        old_d_pose=1.0e-4,
        new_d_pose=1.0e-4,
        old_bytes=178258,
        new_bytes=178358,
    )
    e2 = ActionEffect.build(
        action_id="a2",
        family="pr110",
        authority=ACTION_EFFECT_PLANNING_AUTHORITY,
        producer="pr110_frame_exploit_selector",
    )
    written = append_action_effect(e1, ledger)
    assert written["written_at_utc"]
    append_action_effect(e2, ledger)

    rows = read_action_effects(ledger)
    assert len(rows) == 2
    assert [r.action_id for r in rows] == ["a1", "a2"]
    assert rows[0].pair_ids == (1, 2)
    assert rows[0].region_ids == ("b0/c2/r1",)
    assert rows[0].value_per_byte == pytest.approx(e1.value_per_byte, abs=1e-18)
    assert rows[0].delta_score_total == pytest.approx(e1.delta_score_total, abs=1e-15)
    assert all(r.promotion_eligible is False for r in rows)


def test_v1_ledger_read_filter_by_action_id(tmp_path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    for aid in ("x", "y", "x"):
        append_action_effect(
            ActionEffect.build(action_id=aid, family="hinerv", authority="parseback_mlx", producer="p"),
            ledger,
        )
    only_x = read_action_effects(ledger, action_id="x")
    assert len(only_x) == 2
    assert all(r.action_id == "x" for r in only_x)


def test_v1_ledger_read_missing_file_returns_empty(tmp_path) -> None:
    assert read_action_effects(tmp_path / "does_not_exist.jsonl") == []


def test_v1_ledger_read_skips_malformed_lines(tmp_path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    append_action_effect(
        ActionEffect.build(action_id="ok", family="hinerv", authority="parseback_mlx", producer="p"),
        ledger,
    )
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
        fh.write("\n")
    rows = read_action_effects(ledger)
    assert [r.action_id for r in rows] == ["ok"]


def test_v1_validation_rejects_nested_score_authority_and_survival_mismatch() -> None:
    eff = ActionEffect.build(
        action_id="same-action",
        family="hinerv",
        authority="batch_local_live_mlx",
        producer="fixture",
        receiver_surface={"uint8_changed_pixels": 1},
    )
    payload = eff.as_dict()
    payload["receiver_surface"]["official_score"] = 0.1
    status = validate_action_effect_payload(payload)
    assert status["passed"] is False
    assert any(str(item).startswith("action_effect_forbidden_score_authority") for item in status["blockers"])

    mismatch = eff.as_dict()
    mismatch["parseback_survival"] = {"action_id": "different-action"}
    status = validate_action_effect_payload(mismatch)
    assert status["passed"] is False
    assert "action_id_survival_mismatch" in status["blockers"]


def test_validate_action_effect_rows_cli_accepts_good_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "rows.jsonl"
    append_action_effect(
        ActionEffect.build(
            action_id="ok",
            family="hinerv",
            authority="batch_local_live_mlx",
            producer="fixture",
            old_bytes=100,
            new_bytes=90,
        ),
        ledger,
    )
    append_action_effect(
        ActionEffect.build(
            action_id="ok2",
            family="pr110",
            authority="contest_cpu",
            producer="fixture",
            old_bytes=100,
            new_bytes=90,
        ),
        ledger,
    )
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, str(repo_root / "tools" / "validate_action_effect_rows.py"), str(ledger)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["row_count"] == 2
    assert summary["failed_count"] == 0


def test_validate_action_effect_rows_cli_rejects_bad_ledger(tmp_path: Path) -> None:
    rows = tmp_path / "bad.jsonl"
    payload = ActionEffect.build(
        action_id="bad",
        family="hinerv",
        authority="batch_local_live_mlx",
        producer="fixture",
    ).as_dict()
    payload["score_claim_valid"] = True
    rows.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, str(repo_root / "tools" / "validate_action_effect_rows.py"), str(rows)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 1
    summary = json.loads(proc.stdout)
    assert summary["failed_count"] == 1
    assert any(
        str(item).startswith("action_effect_forbidden_score_authority")
        for item in summary["rows"][0]["blockers"]
    )


def test_hinerv_scorer_bootstrap_converter_records_real_blockers(tmp_path: Path) -> None:
    artifact = tmp_path / "training_artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "archive_sha256": "e" * 64,
                "substrate_artifact_metadata": {
                    "substrate_supplied_score_aware_training": {
                        "scorer_domain_bootstrap": {
                            "schema": "hi_nerv_scorer_domain_bootstrap.v1",
                            "authority": "macos_mlx_research_signal_false_authority",
                            "accepted_step_count": 2,
                            "archive_charged_decoder_tensors": [
                                "latents_fine",
                                "feature_grids.*",
                                "head_rgb_1.*",
                            ],
                            "metrics_before": {
                                "segnet_margin_bootstrap_argmax_disagreement": 0.50,
                                "segnet_margin_bootstrap_score_weighted_total_unsolved_argmax_mass": 50.0,
                                "segnet_margin_bootstrap_worst_class_index": 4.0,
                            },
                            "metrics_after": {
                                "segnet_margin_bootstrap_argmax_disagreement": 0.49,
                                "segnet_margin_bootstrap_score_weighted_total_unsolved_argmax_mass": 49.0,
                                "segnet_margin_bootstrap_worst_class_index": 4.0,
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "action_effect_rows.jsonl"
    repo_root = Path(__file__).resolve().parents[3]

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "convert_hinerv_scorer_bootstrap_to_action_effect.py"),
            "--training-artifact",
            str(artifact),
            "--output-jsonl",
            str(ledger),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["schema"] == "tac.hinerv_scorer_bootstrap_action_effect_conversion.v1"
    assert summary["authority"] == "batch_local_live_mlx"
    assert "hinerv_scorer_bootstrap_pose_endpoint_missing" in summary["blockers"]
    assert "hinerv_scorer_bootstrap_exact_nonrate_delta_missing" in summary["blockers"]

    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["schema"] == ACTION_EFFECT_V1_SCHEMA
    assert row["promotion_eligible"] is False
    assert row["old_d_seg"] == pytest.approx(0.50)
    assert row["new_d_seg"] == pytest.approx(0.49)
    assert row["delta_score_nonrate"] is None
    assert row["seg_score_delta"] == pytest.approx(-1.0)
    assert row["payload_sections"] == ["latents_fine", "feature_grids.*", "head_rgb_1.*"]
    assert row["region_ids"] == ["class:4"]
    assert validate_action_effect_payload(row)["passed"] is True
    for forbidden in ("score_claim", "score_claim_valid", "official_score"):
        assert forbidden not in row
