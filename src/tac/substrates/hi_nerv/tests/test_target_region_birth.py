# SPDX-License-Identifier: MIT
"""Behavioral tests for HiNeRV worst-target-region hard class birth.

NO-FAKE discipline: every MLX test asserts observable behavior (receiver
uint8 motion, argmax flips inside the selected region, bit-identical frozen
tensors, full restore on rejection).  If ``fit_target_region_birth_from_segnet``
were replaced by a canonical-marker stub, every MLX test here fails.

The numpy fixtures are deterministic synthetic label grids used to verify the
region-selection math itself; they are unit fixtures, not empirical anchors
(``research_only`` semantics — no score claim is derived from them).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from tac.substrates.hi_nerv.target_region_birth import (
    TARGET_REGION_BIRTH_RECEIPT_SCHEMA,
    TargetRegionDebt,
    allowed_birth_update_name,
    allowed_pose_compensation_update_name,
    birth_action_id,
    build_target_region_birth_receipt,
    find_target_region_debts,
    region_argmax_transition_counts,
    region_margin_stats,
    select_worst_target_region,
    select_worst_target_region_with_mask,
)


def test_region_argmax_transition_counts_disambiguate_churn_from_birth() -> None:
    region = np.ones((1, 2, 3), dtype=np.float32)
    before = np.array([[[0, 1, 2], [0, 0, 1]]], dtype=np.int64)
    after = np.array([[[1, 0, 2], [2, 0, 1]]], dtype=np.int64)
    # target class = 1: (0,0) wrong->target; (0,1) target->wrong;
    # (1,0) wrong->wrong churn; (0,2)/(1,1)/(1,2) unchanged.
    counts = region_argmax_transition_counts(before, after, region, 1)
    assert counts["argmax_changed_count_region"] == 3
    assert counts["wrong_to_target_count"] == 1
    assert counts["target_to_wrong_count"] == 1
    assert counts["wrong_to_wrong_count"] == 1
    assert counts["target_hard_won_count"] == 1
    assert counts["target_hard_lost_count"] == 1
    assert counts["net_target_support_delta"] == 0
    # Churn-only motion must NOT read as birth: 3 flips, zero net support.
    masked = region_argmax_transition_counts(before, after, np.zeros_like(region), 1)
    assert masked["argmax_changed_count_region"] == 0


skip_no_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="MLX is not installed",
)


def test_allowed_birth_update_name_scopes_late_tensors_only() -> None:
    assert allowed_birth_update_name("head_rgb_1.weight")
    assert allowed_birth_update_name("head_rgb_1.bias")
    assert allowed_birth_update_name("latents_fine")
    assert allowed_birth_update_name("feature_grids.0.grids.1")
    assert allowed_birth_update_name("fine_injector.proj.weight")
    assert allowed_birth_update_name(("head_rgb_1", "weight"))
    assert not allowed_birth_update_name("head_rgb_0.weight")
    assert allowed_pose_compensation_update_name("head_rgb_0.weight")
    assert allowed_pose_compensation_update_name(("head_rgb_0", "bias"))
    assert not allowed_pose_compensation_update_name("head_rgb_1.weight")
    assert not allowed_pose_compensation_update_name("latents_fine")
    assert not allowed_birth_update_name("latents_coarse")
    assert not allowed_birth_update_name("latents_mid")
    assert not allowed_birth_update_name("latent_embed.weight")
    assert not allowed_birth_update_name("blocks.0.conv.weight")
    assert not allowed_birth_update_name("mid_injector.proj.weight")


def _two_region_labels() -> tuple[np.ndarray, np.ndarray]:
    """Item 0 holds two 4-connected class-1 regions; item 1 is all class 0."""

    labels = np.zeros((2, 8, 10), dtype=np.int64)
    labels[0, 1:3, 1:3] = 1  # 4-pixel region, label order first
    labels[0, 5:8, 6:10] = 1  # 12-pixel region, the worst when unsolved
    candidate = np.zeros_like(labels)  # argmax says class 0 everywhere
    return labels, candidate


def test_find_target_region_debts_prices_components_in_score_units() -> None:
    labels, candidate = _two_region_labels()
    rows = find_target_region_debts(labels, candidate)
    total_scored = labels.size
    class1 = [r for r in rows if r.class_index == 1]
    assert {r.region_pixel_count for r in class1} == {4, 12}
    for row in class1:
        # Candidate never wins class 1, so every region pixel is unsolved.
        assert row.region_unsolved_pixel_count == row.region_pixel_count
        assert row.region_hard_ratio == 0.0
        assert row.total_scored_pixels == total_scored
        assert row.score_debt_units == pytest.approx(100.0 * row.region_pixel_count / total_scored)
    # Class 0 is fully solved everywhere it appears: zero debt rows allowed,
    # but they must carry zero unsolved pixels.
    for row in rows:
        if row.class_index == 0:
            assert row.region_unsolved_pixel_count == 0
            assert row.score_debt_units == 0.0


def test_select_worst_target_region_is_deterministic() -> None:
    labels, candidate = _two_region_labels()
    rows = find_target_region_debts(labels, candidate)
    worst = select_worst_target_region(rows)
    assert worst.class_index == 1
    assert worst.region_pixel_count == 12
    # Tie-break is (batch, class, region label) once debt ties.
    tie_a = TargetRegionDebt(
        batch_index=1,
        class_index=2,
        region_label=3,
        region_pixel_count=5,
        region_unsolved_pixel_count=5,
        region_hard_ratio=0.0,
        frame_pixel_count=80,
        total_scored_pixels=160,
        score_debt_units=3.125,
        frame_fraction=5 / 80,
        bbox_y0=0,
        bbox_y1=1,
        bbox_x0=0,
        bbox_x1=5,
    )
    tie_b = TargetRegionDebt(
        batch_index=0,
        class_index=2,
        region_label=1,
        region_pixel_count=5,
        region_unsolved_pixel_count=5,
        region_hard_ratio=0.0,
        frame_pixel_count=80,
        total_scored_pixels=160,
        score_debt_units=3.125,
        frame_fraction=5 / 80,
        bbox_y0=0,
        bbox_y1=1,
        bbox_x0=0,
        bbox_x1=5,
    )
    assert select_worst_target_region([tie_a, tie_b]) is tie_b


def test_select_worst_target_region_with_mask_matches_row() -> None:
    labels, candidate = _two_region_labels()
    worst, mask = select_worst_target_region_with_mask(labels, candidate)
    assert mask.shape == labels.shape
    assert mask.dtype == np.float32
    assert int(mask.sum()) == worst.region_pixel_count == 12
    assert mask[1].sum() == 0.0  # the other batch item carries no mask
    ys, xs = np.nonzero(mask[0])
    assert ys.min() == worst.bbox_y0 and ys.max() + 1 == worst.bbox_y1
    assert xs.min() == worst.bbox_x0 and xs.max() + 1 == worst.bbox_x1


def test_find_target_region_debts_separates_diagonal_components() -> None:
    # Two diagonal pixels are 8-connected but NOT 4-connected: the pricing
    # must treat them as separate regions.
    labels = np.zeros((1, 4, 4), dtype=np.int64)
    labels[0, 0, 0] = 1
    labels[0, 1, 1] = 1
    rows = [r for r in find_target_region_debts(labels, np.zeros_like(labels)) if r.class_index == 1]
    assert len(rows) == 2
    assert all(r.region_pixel_count == 1 for r in rows)


def test_region_margin_stats_reports_raw_frontier_margin() -> None:
    logits = np.zeros((1, 2, 2, 3), dtype=np.float32)
    # Pixel (0,0): class 1 wins by 0.5 -> margin -0.5.
    logits[0, 0, 0] = [0.0, 1.0, 0.5]
    # Pixel (0,1): class 0 wins; margin for class 1 is +0.7.
    logits[0, 0, 1] = [1.2, 0.5, 0.1]
    mask = np.zeros((1, 2, 2), dtype=np.float32)
    mask[0, 0, 0] = 1.0
    mask[0, 0, 1] = 1.0
    stats = region_margin_stats(logits, mask, 1)
    assert stats["region_pixel_count"] == 2.0
    assert stats["region_hard_won_pixels"] == 1.0
    assert stats["region_hard_ratio"] == pytest.approx(0.5)
    assert stats["margin_min"] == pytest.approx(-0.5)
    assert stats["margin_p50"] == pytest.approx((0.7 - 0.5) / 2)


def test_region_margin_stats_rejects_empty_mask() -> None:
    logits = np.zeros((1, 2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="zero pixels"):
        region_margin_stats(logits, np.zeros((1, 2, 2), dtype=np.float32), 1)


def test_receipt_refuses_out_of_scope_parameter_names() -> None:
    labels, candidate = _two_region_labels()
    worst, _mask = select_worst_target_region_with_mask(labels, candidate)
    stats = {
        "region_pixel_count": 12.0,
        "region_hard_ratio": 0.0,
        "region_hard_won_pixels": 0.0,
        "margin_min": 0.4,
        "margin_p50": 0.6,
        "margin_mean": 0.6,
    }
    with pytest.raises(ValueError, match="escape the birth scope"):
        build_target_region_birth_receipt(
            debt=worst,
            before_margin_stats=stats,
            after_margin_stats=stats,
            receiver_uint8_changed_pixels_region=1,
            receiver_uint8_delta_abs_max=1.0,
            receiver_float_rgb_delta_linf=0.01,
            argmax_flipped_pixels_region=0,
            accepted_step_count=1,
            rejected_step_count=0,
            blockers=[],
            grad_norm_by_group={},
            update_norm_by_group={},
            updated_parameter_names=["latents_coarse"],
            pose_guard={"available": False},
        )


def test_receipt_emits_crux_trace_compatible_keys() -> None:
    labels, candidate = _two_region_labels()
    worst, _mask = select_worst_target_region_with_mask(labels, candidate)
    before = {
        "region_pixel_count": 12.0,
        "region_hard_ratio": 0.0,
        "region_hard_won_pixels": 0.0,
        "margin_min": 0.4,
        "margin_p50": 0.6,
        "margin_mean": 0.6,
    }
    after = dict(before, margin_p50=0.2, region_hard_ratio=0.25, region_hard_won_pixels=3.0)
    receipt = build_target_region_birth_receipt(
        debt=worst,
        before_margin_stats=before,
        after_margin_stats=after,
        receiver_uint8_changed_pixels_region=9,
        receiver_uint8_delta_abs_max=4.0,
        receiver_float_rgb_delta_linf=0.02,
        argmax_flipped_pixels_region=3,
        accepted_step_count=2,
        rejected_step_count=1,
        blockers=["x"],
        grad_norm_by_group={"head_rgb_1": 0.5},
        update_norm_by_group={"head_rgb_1": 0.001},
        updated_parameter_names=["head_rgb_1.bias"],
        pose_guard={"available": True},
        candidate_frontier_telemetry={
            "schema": "hi_nerv_target_region_birth_candidate_frontier_telemetry.v1",
            "candidate_attempt_count": 3,
            "max_candidate_margin_mean_improvement": 0.125,
            "min_pose_rejected_pose_output_delta_l2": 0.08,
        },
    )
    assert receipt["schema"] == TARGET_REGION_BIRTH_RECEIPT_SCHEMA
    assert receipt["receiver_surface_uint8_changed_pixels"] == 9
    assert receipt["receiver_surface_argmax_flipped_pixels"] == 3
    assert receipt["receiver_surface_worst_region_margin_p50_delta"] == pytest.approx(-0.4)
    assert receipt["receiver_surface_float_rgb_delta_linf"] == pytest.approx(0.02)
    assert receipt["frame_scope"] == "frame1_seg_pose_joint"
    assert receipt["candidate_frontier_telemetry"] == {
        "schema": "hi_nerv_target_region_birth_candidate_frontier_telemetry.v1",
        "candidate_attempt_count": 3,
        "max_candidate_margin_mean_improvement": 0.125,
        "min_pose_rejected_pose_output_delta_l2": 0.08,
    }
    # Custody contract: receipts travel inside substrate_artifact_metadata,
    # whose harness validator REFUSES nested authority/readiness keys (single
    # custody surface). The receipt must carry the non-authority marker and
    # must NOT carry any forbidden key — even as a false-valued copy.
    assert receipt["authority"] == "planning_control_false_authority"
    assert receipt["pose_compensation"] is None
    forbidden = {
        "score_claim",
        "score_claim_valid",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "rank_or_kill_eligible",
        "promotable",
    }
    assert not (forbidden & receipt.keys())
    assert not (forbidden & receipt["worst_region"].keys())


def test_receipt_carries_frame0_compensation_without_relaxing_birth_scope() -> None:
    labels, candidate = _two_region_labels()
    worst, _mask = select_worst_target_region_with_mask(labels, candidate)
    before = {
        "region_pixel_count": 12.0,
        "region_hard_ratio": 0.0,
        "region_hard_won_pixels": 0.0,
        "margin_min": 0.4,
        "margin_p50": 0.6,
        "margin_mean": 0.6,
    }
    after = dict(before, margin_p50=0.2, region_hard_ratio=0.25, region_hard_won_pixels=3.0)
    pose_compensation = {
        "authority": "batch_local_live_mlx",
        "normalization_scope": "batch_local",
        "pose_compensation_attempted": True,
        "pose_compensation_frame": 0,
        "composite_accepted": True,
        "composite_attempt_count": 1,
        "composite_accepted_count": 1,
        "composite_delta_score_nonrate": -0.125,
        "compensation_updated_parameter_names": ["head_rgb_0.bias"],
        "compensation_scope": "head_rgb_0",
        "frame1_receiver_uint8_unchanged_by_compensation": True,
        "human_visual_fidelity_objective": False,
    }

    receipt = build_target_region_birth_receipt(
        debt=worst,
        before_margin_stats=before,
        after_margin_stats=after,
        receiver_uint8_changed_pixels_region=9,
        receiver_uint8_delta_abs_max=4.0,
        receiver_float_rgb_delta_linf=0.02,
        argmax_flipped_pixels_region=3,
        accepted_step_count=2,
        rejected_step_count=1,
        blockers=[],
        grad_norm_by_group={"head_rgb_1": 0.5, "compensation_head_rgb_0": 0.25},
        update_norm_by_group={"head_rgb_1": 0.001, "compensation_head_rgb_0": 0.0005},
        updated_parameter_names=["head_rgb_1.bias"],
        pose_guard={"available": True},
        pose_compensation=pose_compensation,
    )

    assert receipt["updated_parameter_names"] == ["head_rgb_1.bias"]
    assert receipt["pose_compensation"] == pose_compensation
    assert receipt["pose_compensation"]["compensation_updated_parameter_names"] == ["head_rgb_0.bias"]

    with pytest.raises(ValueError, match="frame0 compensation scope"):
        build_target_region_birth_receipt(
            debt=worst,
            before_margin_stats=before,
            after_margin_stats=after,
            receiver_uint8_changed_pixels_region=9,
            receiver_uint8_delta_abs_max=4.0,
            receiver_float_rgb_delta_linf=0.02,
            argmax_flipped_pixels_region=3,
            accepted_step_count=2,
            rejected_step_count=1,
            blockers=[],
            grad_norm_by_group={},
            update_norm_by_group={},
            updated_parameter_names=["head_rgb_1.bias"],
            pose_guard={"available": True},
            pose_compensation={
                **pose_compensation,
                "compensation_updated_parameter_names": ["head_rgb_1.bias"],
            },
        )


def test_birth_action_id_is_stable_and_group_sensitive() -> None:
    labels, candidate = _two_region_labels()
    worst, _mask = select_worst_target_region_with_mask(labels, candidate)
    group_hashes = {
        "head_rgb_1": "a" * 64,
        "latents_fine": "b" * 64,
    }

    first = birth_action_id(
        debt=worst,
        initial_group_sha256=group_hashes,
        trained_groups=["latents_fine", "head_rgb_1"],
    )
    second = birth_action_id(
        debt=worst,
        initial_group_sha256=dict(reversed(list(group_hashes.items()))),
        trained_groups=["head_rgb_1", "latents_fine"],
    )
    changed_groups = birth_action_id(
        debt=worst,
        initial_group_sha256=group_hashes,
        trained_groups=["head_rgb_1"],
    )
    compensated_groups = birth_action_id(
        debt=worst,
        initial_group_sha256=group_hashes,
        trained_groups=["head_rgb_1", "latents_fine", "compensation_head_rgb_0"],
    )

    assert first == second
    assert first != changed_groups
    assert first != compensated_groups
    assert len(first) == 64


# ---------------------------------------------------------------------------
# MLX behavioral tests
# ---------------------------------------------------------------------------


def _smoke_cfg():
    from tac.substrates.hi_nerv.architecture import HinervConfig

    return HinervConfig(
        latent_dim_coarse=4,
        latent_dim_mid=6,
        latent_dim_fine=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def _green_dominant_targets(cfg, mx):
    ramp = mx.reshape(
        mx.linspace(0.05, 0.95, cfg.output_height * cfg.output_width),
        (1, cfg.output_height, cfg.output_width, 1),
    )
    target0 = mx.tile(
        mx.concatenate([0.15 + 0.05 * ramp, 0.55 + 0.1 * ramp, 0.1 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )
    target1 = mx.tile(
        mx.concatenate([0.18 + 0.05 * ramp, 0.6 + 0.05 * ramp, 0.05 + 0.1 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )
    return target0, target1


def _block_labels(cfg, np_module):
    labels = np_module.zeros(
        (cfg.num_pairs, cfg.output_height, cfg.output_width),
        dtype=np_module.int32,
    )
    labels[0, 4:10, 6:14] = 1  # one 48-pixel class-1 region in item 0
    return labels


class _BehavioralSegNetTeacher:
    """Real-VJP behavioral stand-in: logits derived from frame RGB."""

    num_classes = 2

    def __init__(self, mx, labels):
        self._mx = mx
        self._labels = labels

    def teacher_argmax_for_indices(self, indices):
        return self._mx.take(self._labels, indices, axis=0)

    def teacher_logits_for_frames_nhwc01(self, frames):
        red = frames[..., 0]
        green = frames[..., 1]
        class0 = green - red
        class1 = red - green
        return self._mx.stack([class0, class1], axis=-1)


class _SubquantumSegNetTeacher(_BehavioralSegNetTeacher):
    """Gradient exists but is so small no uint8 quantum can be crossed."""

    def teacher_logits_for_frames_nhwc01(self, frames):
        red = frames[..., 0]
        green = frames[..., 1]
        scale = 1.0e-12
        class0 = scale * (green - red)
        class1 = scale * (red - green)
        return self._mx.stack([class0, class1], axis=-1)


class _MeanTrackingPoseTeacher:
    """Pose output amplifies frame-1 mean so any visible edit breaches the cap."""

    def __init__(self, mx):
        self._mx = mx

    def teacher_pose_for_yuv6_pair_nhwc(self, yuv6_pair):
        mean = self._mx.mean(yuv6_pair, axis=(1, 2, 3))
        return 1000.0 * self._mx.stack([mean] * 6, axis=-1)


def _frozen_tensor_snapshot(model, np_module):
    from mlx.utils import tree_flatten

    frozen = {}
    for raw_name, leaf in tree_flatten(model.parameters()):
        name = ".".join(str(p) for p in raw_name) if isinstance(raw_name, (tuple, list)) else str(raw_name)
        if leaf is None:
            continue
        if not allowed_birth_update_name(name):
            frozen[name] = np_module.array(leaf, copy=True)
    return frozen


@skip_no_mlx
def test_target_region_birth_lifts_frontier_margin_under_scoped_param_update() -> None:
    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(7)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    target0, target1 = _green_dominant_targets(cfg, mx)
    labels_np = _block_labels(cfg, np)
    teacher = _BehavioralSegNetTeacher(mx, mx.array(labels_np))
    model.initialize_output_head_bias_from_targets(target0, target1)
    frozen_before = _frozen_tensor_snapshot(model, np)

    payload = model.fit_target_region_birth_from_segnet(
        scorer_teacher=teacher,
        target_rgb_0=target0,
        target_rgb_1=target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        target_segnet_argmax_1=mx.array(labels_np),
        max_steps=24,
        learning_rate=2.0e-3,
        target_min_region_ratio=0.02,
    )

    assert payload["schema"] == "hi_nerv_target_region_birth.v1"
    assert payload["enabled"] is True
    # The worst region must be the synthetic class-1 block, fully unsolved.
    assert payload["birth_class_index"] == 1
    assert payload["worst_region"]["region_pixel_count"] == 48
    assert payload["before_region_hard_ratio"] == 0.0
    # Behavioral acceptance: at least one receiver-visible accepted step that
    # moved the frontier (median margin down) or lifted the hard ratio.
    assert payload["accepted"] is True
    assert payload["accepted_step_count"] >= 1
    margin_delta = (
        payload["after_region_margin_stats"]["margin_p50"] - payload["before_region_margin_stats"]["margin_p50"]
    )
    assert margin_delta < 0.0 or payload["after_region_hard_ratio"] > 0.0
    receipt = payload["receipt"]
    assert receipt["receiver_surface_uint8_changed_pixels"] > 0
    assert receipt["receiver_surface_float_rgb_delta_linf"] > 0.0
    # Scope proof: every updated tensor is birth-scoped, and at least one
    # update actually landed.
    assert payload["updated_parameter_names"]
    assert all(allowed_birth_update_name(name) for name in payload["updated_parameter_names"])
    assert payload["grad_norm_by_group"]
    assert payload["runtime_sidecar_bytes"] == 0
    # Pose telemetry was unavailable -> the payload must say so, loudly.
    assert "hinerv_target_region_birth_pose_trust_telemetry_missing" in payload["blockers"]
    # Transition disambiguation: the accepted birth must show net target
    # support, and total churn must bound the won count from above.
    transitions = payload["argmax_transitions"]
    assert transitions["target_hard_won_count"] > 0
    assert transitions["net_target_support_delta"] > 0
    assert transitions["argmax_changed_count_region"] >= transitions["target_hard_won_count"]
    # No pose teacher -> exact joint term explicitly unavailable, not faked.
    assert payload["exact_nonrate"]["pose_term_available"] is False
    assert payload["exact_nonrate"]["delta_score_nonrate"] is None
    # Frozen-tensor proof: nothing outside the birth scope moved a single bit.
    frozen_after = _frozen_tensor_snapshot(model, np)
    assert frozen_after.keys() == frozen_before.keys()
    for name, before_value in frozen_before.items():
        assert np.array_equal(before_value, frozen_after[name]), name
    # Receipt-level proof of the same: out-of-scope group hash unchanged while
    # at least one scoped group hash moved.
    assert payload["out_of_scope_bit_frozen_verified"] is True
    before_hashes = payload["parameter_group_sha256_before"]
    after_hashes = payload["parameter_group_sha256_after"]
    assert before_hashes["out_of_scope"] == after_hashes["out_of_scope"]
    assert any(before_hashes[group] != after_hashes[group] for group in before_hashes if group != "out_of_scope")


@skip_no_mlx
def test_target_region_birth_rejects_subquantum_updates_and_restores() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(7)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    target0, target1 = _green_dominant_targets(cfg, mx)
    labels_np = _block_labels(cfg, np)
    teacher = _SubquantumSegNetTeacher(mx, mx.array(labels_np))
    model.initialize_output_head_bias_from_targets(target0, target1)
    all_before = {
        (".".join(str(p) for p in raw) if isinstance(raw, (tuple, list)) else str(raw)): np.array(leaf, copy=True)
        for raw, leaf in tree_flatten(model.parameters())
        if leaf is not None
    }

    payload = model.fit_target_region_birth_from_segnet(
        scorer_teacher=teacher,
        target_rgb_0=target0,
        target_rgb_1=target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        target_segnet_argmax_1=mx.array(labels_np),
        max_steps=4,
        learning_rate=5.0e-4,
    )

    # Float gradients existed, but no step crossed a receiver uint8 quantum:
    # the actuator must reject every step and restore the initial state.
    assert payload["accepted"] is False
    assert payload["accepted_step_count"] == 0
    assert payload["subquantum_rejected_step_count"] >= 1
    assert payload["receiver_quantum_growth_attempt_count"] >= 1
    assert "hinerv_target_region_birth_no_accepted_step" in payload["blockers"]
    assert payload["updated_parameter_names"] == []
    all_after = {
        (".".join(str(p) for p in raw) if isinstance(raw, (tuple, list)) else str(raw)): np.array(leaf, copy=True)
        for raw, leaf in tree_flatten(model.parameters())
        if leaf is not None
    }
    assert all_after.keys() == all_before.keys()
    for name, before_value in all_before.items():
        assert np.array_equal(before_value, all_after[name]), name


@skip_no_mlx
def test_birth_rejection_restores_full_state_and_replays_identically() -> None:
    """Restore must cover ALL state, not just tensors.

    The actuator is optimizer-stateless by construction: raw scoped SGD from
    explicit snapshots — no momentum/Adam slots, no EMA update, no dual
    variables, and no RNG draws inside the loop.  This test proves the
    construction: after a fully-rejected fit, a SECOND fit from the restored
    state must observe an identical starting world (same before-stats, same
    worst region, same group hashes) and reach identical rejection counters.
    Any hidden state advanced by the first run would break the replay.
    """

    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(7)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    target0, target1 = _green_dominant_targets(cfg, mx)
    labels_np = _block_labels(cfg, np)
    teacher = _SubquantumSegNetTeacher(mx, mx.array(labels_np))
    model.initialize_output_head_bias_from_targets(target0, target1)

    def _run():
        return model.fit_target_region_birth_from_segnet(
            scorer_teacher=teacher,
            target_rgb_0=target0,
            target_rgb_1=target1,
            pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
            target_segnet_argmax_1=mx.array(labels_np),
            max_steps=3,
            learning_rate=5.0e-4,
        )

    first = _run()
    second = _run()
    assert first["accepted"] is False and second["accepted"] is False
    # The second run starts from a world bit-identical to the first run's
    # start: same group hashes, same worst region, same margin stats.
    assert (
        first["parameter_group_sha256_before"]
        == first["parameter_group_sha256_after"]
        == second["parameter_group_sha256_before"]
        == second["parameter_group_sha256_after"]
    )
    assert first["worst_region"] == second["worst_region"]
    assert first["before_region_margin_stats"] == second["before_region_margin_stats"]
    assert first["out_of_scope_bit_frozen_verified"] is True
    assert second["out_of_scope_bit_frozen_verified"] is True
    # Identical rejection trajectory == no hidden state advanced.
    for key in (
        "accepted_step_count",
        "rejected_step_count",
        "subquantum_rejected_step_count",
        "receiver_quantum_growth_attempt_count",
        "blockers",
    ):
        assert first[key] == second[key], key


@skip_no_mlx
def test_target_region_birth_pose_guard_blocks_visible_pose_harm() -> None:
    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(7)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    target0, target1 = _green_dominant_targets(cfg, mx)
    labels_np = _block_labels(cfg, np)
    teacher = _BehavioralSegNetTeacher(mx, mx.array(labels_np))
    model.initialize_output_head_bias_from_targets(target0, target1)

    payload = model.fit_target_region_birth_from_segnet(
        scorer_teacher=teacher,
        target_rgb_0=target0,
        target_rgb_1=target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        target_segnet_argmax_1=mx.array(labels_np),
        pose_teacher=_MeanTrackingPoseTeacher(mx),
        max_steps=4,
        learning_rate=2.0e-3,
        max_pose_output_delta_l2=1.0e-6,
    )

    # Any uint8-visible step breaches the (deliberately impossible) pose cap,
    # so nothing may be accepted and pose telemetry must be present.
    assert payload["accepted"] is False
    assert payload["pose_guard"]["available"] is True
    assert payload["pose_guard_rejected_step_count"] >= 1 or payload["subquantum_rejected_step_count"] >= 1
    assert "hinerv_target_region_birth_pose_trust_telemetry_missing" not in payload["blockers"]
    assert "hinerv_target_region_birth_no_accepted_step" in payload["blockers"]


@skip_no_mlx
def test_require_pose_trust_fails_closed_without_teacher() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(7)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    target0, target1 = _green_dominant_targets(cfg, mx)
    labels_np = _block_labels(cfg, np)
    teacher = _BehavioralSegNetTeacher(mx, mx.array(labels_np))
    model.initialize_output_head_bias_from_targets(target0, target1)
    before = {str(raw): np.array(leaf, copy=True) for raw, leaf in tree_flatten(model.parameters()) if leaf is not None}
    payload = model.fit_target_region_birth_from_segnet(
        scorer_teacher=teacher,
        target_rgb_0=target0,
        target_rgb_1=target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        target_segnet_argmax_1=mx.array(labels_np),
        require_pose_trust=True,
        max_steps=4,
    )
    assert payload["accepted"] is False
    assert payload["reason"] == "pose_trust_required_but_teacher_missing"
    assert "hinerv_target_region_birth_pose_trust_required_but_teacher_missing" in payload["blockers"]
    assert payload["accepted_step_count"] == 0
    after = {str(raw): np.array(leaf, copy=True) for raw, leaf in tree_flatten(model.parameters()) if leaf is not None}
    for name, value in before.items():
        assert np.array_equal(value, after[name]), name


@skip_no_mlx
def test_accepted_birth_with_pose_teacher_requires_joint_score_improvement() -> None:
    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(7)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    target0, target1 = _green_dominant_targets(cfg, mx)
    labels_np = _block_labels(cfg, np)
    teacher = _BehavioralSegNetTeacher(mx, mx.array(labels_np))
    model.initialize_output_head_bias_from_targets(target0, target1)
    payload = model.fit_target_region_birth_from_segnet(
        scorer_teacher=teacher,
        target_rgb_0=target0,
        target_rgb_1=target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        target_segnet_argmax_1=mx.array(labels_np),
        pose_teacher=_MeanTrackingPoseTeacher(mx),
        # Raw cap deliberately huge: only the EXACT joint score may arbitrate.
        max_pose_output_delta_l2=1.0e9,
        max_steps=12,
        learning_rate=2.0e-3,
    )
    nonrate = payload["exact_nonrate"]
    assert nonrate["pose_term_available"] is True
    assert nonrate["old_nonrate_score"] is not None
    # The binding invariant: ACCEPTED implies the exact nonlinear joint score
    # (100*d_seg + sqrt(10*d_pose), batch-local) improved; otherwise every
    # candidate must have been rejected through the joint gate.
    if payload["accepted"]:
        assert nonrate["delta_score_nonrate"] is not None
        assert nonrate["delta_score_nonrate"] < 0.0
    else:
        assert payload["joint_score_rejected_step_count"] >= 1 or payload["subquantum_rejected_step_count"] >= 1
    assert "hinerv_target_region_birth_pose_trust_telemetry_missing" not in (payload["blockers"])


@skip_no_mlx
def test_target_region_birth_returns_disabled_when_no_debt() -> None:
    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(7)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    target0, target1 = _green_dominant_targets(cfg, mx)
    # Labels say class 0 everywhere; the green-dominant init wins class 0
    # everywhere, so there is no unsolved region and nothing to birth.
    labels_np = np.zeros((cfg.num_pairs, cfg.output_height, cfg.output_width), dtype=np.int32)
    teacher = _BehavioralSegNetTeacher(mx, mx.array(labels_np))
    model.initialize_output_head_bias_from_targets(target0, target1)

    payload = model.fit_target_region_birth_from_segnet(
        scorer_teacher=teacher,
        target_rgb_0=target0,
        target_rgb_1=target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        target_segnet_argmax_1=mx.array(labels_np),
        max_steps=4,
    )

    assert payload["enabled"] is False
    assert payload["reason"] == "no_unsolved_target_region"
    assert payload["blockers"] == []


# ---------------------------------------------------------------------------
# Frame0 pose-compensation composite operator behavioral tests
# ---------------------------------------------------------------------------


class _FrameDiffPoseTeacher:
    """Pose = k * (mean(frame1 YUV6) - mean(frame0 YUV6)).

    The frame1 birth step (region brightening) raises pose; a frame0
    brightening lowers it by the SAME mechanism, so a frame1-driven pose
    excursion is EXACTLY compensable by a frame0 update.  YUV6 luma tracks
    brightness, so the per-frame mean over all 6 YUV6 channels is a faithful
    differentiable proxy for "frame brightness".  No ``teacher_pose_for_indices``
    => the actuator derives the target pose from the target frames through this
    same surface (keeping candidate and target pose comparable).
    """

    def __init__(self, mx, k):
        self._mx = mx
        self._k = float(k)

    def teacher_pose_for_yuv6_pair_nhwc(self, yuv6_pair):
        frame0 = self._mx.mean(yuv6_pair[..., :6], axis=(1, 2, 3))
        frame1 = self._mx.mean(yuv6_pair[..., 6:], axis=(1, 2, 3))
        diff = self._k * (frame1 - frame0)
        return self._mx.stack([diff] * 6, axis=-1)


class _Frame1OnlyPoseTeacher:
    """Pose = k * mean(frame1 YUV6) only — structurally uncompensable by frame0.

    A frame1 birth step that moves pose cannot be undone by a frame0 update
    because pose has no frame0 dependence at all.  Used to prove the composite
    gate REJECTS (and fully restores) when compensation cannot recover.
    """

    def __init__(self, mx, k):
        self._mx = mx
        self._k = float(k)

    def teacher_pose_for_yuv6_pair_nhwc(self, yuv6_pair):
        frame1 = self._mx.mean(yuv6_pair[..., 6:], axis=(1, 2, 3))
        return self._k * self._mx.stack([frame1] * 6, axis=-1)


def _all_tensor_snapshot(model, np_module):
    from mlx.utils import tree_flatten

    out = {}
    for raw_name, leaf in tree_flatten(model.parameters()):
        name = ".".join(str(p) for p in raw_name) if isinstance(raw_name, (tuple, list)) else str(raw_name)
        if leaf is None:
            continue
        out[name] = np_module.array(leaf, copy=True)
    return out


def _near_boundary_all_class1_targets(cfg, mx):
    """Targets where frame1 sits just below the SegNet class-1 boundary.

    ``_BehavioralSegNetTeacher`` decides class1 iff ``red > green``.  Frame1 is
    near-uniform with ``red`` just *below* ``green`` (so every pixel starts
    wrong for an all-class-1 label), and a single red-brightening birth step
    flips the WHOLE frame to class1 at once — a clean, large, deterministic
    seg gain with no non-region pixels left to corrupt.  Frame0 is mid-gray so
    a frame0 brightening has clear headroom to absorb a pose excursion.
    """

    n, h, w = cfg.num_pairs, cfg.output_height, cfg.output_width
    target0 = mx.concatenate(
        [mx.full((n, h, w, 1), 0.45), mx.full((n, h, w, 1), 0.50), mx.full((n, h, w, 1), 0.30)],
        axis=-1,
    )
    target1 = mx.concatenate(
        [mx.full((n, h, w, 1), 0.49), mx.full((n, h, w, 1), 0.50), mx.full((n, h, w, 1), 0.30)],
        axis=-1,
    )
    return target0, target1


@skip_no_mlx
def test_frame0_compensation_admits_composite_when_frame1_pose_harm_is_compensable() -> None:
    """A compensable pose teacher => frame0 compensation lands a net-improving composite.

    The frame1 birth step lowers d_seg but pushes the (frame1-mean-driven)
    pose past its cap; a frame0 brightening exactly undoes that excursion, so
    the COMPOSITE (frame1 birth + frame0 compensation) strictly improves the
    exact nonrate score while keeping pose under cap.  This is the headline
    behavior: it FAILS if the operator is a marker stub (no composite landed),
    if frame0 compensation moved frame1, or if head_rgb_0 leaks into the birth
    scope.
    """

    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(7)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    target0, target1 = _near_boundary_all_class1_targets(cfg, mx)
    # Every pixel is class 1, sitting just below the SegNet boundary: a single
    # red-brightening birth step flips the whole frame (huge, clean seg gain)
    # with no non-region pixels to corrupt.
    labels_np = np.ones((cfg.num_pairs, cfg.output_height, cfg.output_width), dtype=np.int32)
    teacher = _BehavioralSegNetTeacher(mx, mx.array(labels_np))
    model.initialize_output_head_bias_from_targets(target0, target1)

    # Bit-frozen baseline EXCLUDING the frame0 compensation head: head_rgb_0 is
    # allowed to move (compensation scope), but the true out-of-scope tensors
    # (latents_coarse/mid, latent_embed, blocks, mid_injector, ...) must not.
    def _out_of_scope_snapshot():
        from mlx.utils import tree_flatten

        frozen = {}
        for raw_name, leaf in tree_flatten(model.parameters()):
            name = ".".join(str(p) for p in raw_name) if isinstance(raw_name, (tuple, list)) else str(raw_name)
            if leaf is None:
                continue
            if allowed_birth_update_name(name) or allowed_pose_compensation_update_name(name):
                continue
            frozen[name] = np.array(leaf, copy=True)
        return frozen

    out_of_scope_before = _out_of_scope_snapshot()

    payload = model.fit_target_region_birth_from_segnet(
        scorer_teacher=teacher,
        target_rgb_0=target0,
        target_rgb_1=target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        target_segnet_argmax_1=mx.array(labels_np),
        # k moderate: the birth step's frame1 brightening moves pose past the
        # cap below, but frame0 can undo it through the same per-frame-mean term.
        pose_teacher=_FrameDiffPoseTeacher(mx, k=0.5),
        # Pose cap deliberately tight enough that the raw frame1 birth step
        # breaches it (triggering compensation), yet loose enough that the
        # frame0-compensated composite returns under it.
        max_pose_output_delta_l2=0.05,
        max_steps=24,
        learning_rate=2.0e-3,
    )

    assert payload["enabled"] is True
    assert payload["birth_class_index"] == 1
    # Compensation was attempted AND accepted at least one composite.
    assert payload["pose_compensation_attempted"] is True
    assert payload["pose_compensation_frame"] == 0
    assert payload["composite_accepted"] is True
    assert payload["composite_accepted_count"] >= 1
    assert payload["accepted"] is True
    # The composite strictly improved the exact nonrate score (batch-local).
    assert payload["composite_delta_score_nonrate"] is not None
    assert payload["composite_delta_score_nonrate"] < 0.0
    # head_rgb_0 is the ONLY compensated scope, and it is recorded SEPARATELY
    # from the birth updated_parameter_names (never folded into the birth scope).
    comp_names = payload["compensation_updated_parameter_names"]
    assert comp_names
    assert all(allowed_pose_compensation_update_name(name) for name in comp_names)
    assert all(name.startswith("head_rgb_0") for name in comp_names)
    assert not any(allowed_birth_update_name(name) for name in comp_names)
    # Birth scope must NOT contain head_rgb_0 even though a composite landed.
    assert all(allowed_birth_update_name(name) for name in payload["updated_parameter_names"])
    assert not any(name.startswith("head_rgb_0") for name in payload["updated_parameter_names"])
    # SegNet reads frame1 only: frame0 compensation must not move frame1's
    # receiver uint8. The actuator asserts this internally; the payload mirrors
    # it for every attempt.
    pc = payload["pose_compensation"]
    assert pc is not None
    assert pc["pose_compensation_attempted"] is True
    assert pc["pose_compensation_frame"] == 0
    assert pc["composite_accepted"] is True
    assert pc["frame1_receiver_uint8_unchanged_by_compensation"] is True
    assert pc["compensation_scope"] == "head_rgb_0"
    assert all(rec["frame1_receiver_uint8_unchanged"] for rec in pc["attempts"])
    # The accepted composite satisfied the pose cap.
    accepted_attempts = [rec for rec in pc["attempts"] if rec["accepted"]]
    assert accepted_attempts
    assert all(rec["composite_pose_cap_satisfied"] for rec in accepted_attempts)
    # The receipt carries the same composite record and still refuses the birth
    # scope leak (built via the scope-validated receipt builder).
    receipt = payload["receipt"]
    assert receipt["pose_compensation"] is not None
    assert receipt["pose_compensation"]["composite_accepted"] is True
    assert not any(name.startswith("head_rgb_0") for name in receipt["updated_parameter_names"])
    # Out-of-scope tensors (minus head_rgb_0) stayed bit-identical.
    out_of_scope_after = _out_of_scope_snapshot()
    assert out_of_scope_after.keys() == out_of_scope_before.keys()
    for name, before_value in out_of_scope_before.items():
        assert np.array_equal(before_value, out_of_scope_after[name]), name
    # The frozen-bit receipt proof still holds: the canonical out_of_scope group
    # hash never moved (head_rgb_0 lives in its own compensation group).
    assert payload["out_of_scope_bit_frozen_verified"] is True
    before_hashes = payload["parameter_group_sha256_before"]
    after_hashes = payload["parameter_group_sha256_after"]
    assert before_hashes["out_of_scope"] == after_hashes["out_of_scope"]
    # The compensation group hash DID move (frame0 head was updated).
    assert before_hashes.get("compensation_head_rgb_0") != after_hashes.get("compensation_head_rgb_0")


@skip_no_mlx
def test_frame0_compensation_rejects_and_restores_when_pose_harm_is_uncompensable() -> None:
    """A frame1-only pose teacher => compensation attempted, no composite admitted, full restore.

    Pose has zero frame0 dependence, so a frame0 update cannot recover the
    frame1-driven pose excursion.  The composite gate must reject every
    attempt and restore the model exactly to the frame1 birth state (then the
    caller restores to pre-step), leaving head_rgb_0 bit-identical.
    """

    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(7)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    target0, target1 = _green_dominant_targets(cfg, mx)
    labels_np = _block_labels(cfg, np)
    teacher = _BehavioralSegNetTeacher(mx, mx.array(labels_np))
    model.initialize_output_head_bias_from_targets(target0, target1)
    all_before = _all_tensor_snapshot(model, np)

    payload = model.fit_target_region_birth_from_segnet(
        scorer_teacher=teacher,
        target_rgb_0=target0,
        target_rgb_1=target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        target_segnet_argmax_1=mx.array(labels_np),
        # frame1-only pose => any visible frame1 birth step moves pose and frame0
        # cannot undo it. k large + cap tiny guarantees the cap rejects.
        pose_teacher=_Frame1OnlyPoseTeacher(mx, k=50.0),
        max_pose_output_delta_l2=1.0e-4,
        max_steps=6,
        learning_rate=2.0e-3,
    )

    # Compensation was attempted (region progress + pose teacher present) but no
    # composite was admissible, so nothing accepted and the state is restored.
    assert payload["pose_compensation_attempted"] is True
    assert payload["composite_attempt_count"] >= 1
    assert payload["composite_accepted"] is False
    assert payload["composite_accepted_count"] == 0
    assert payload["accepted"] is False
    assert payload["composite_delta_score_nonrate"] is None
    assert "hinerv_target_region_birth_no_accepted_step" in payload["blockers"]
    # No compensation names survived (every composite was restored).
    assert payload["compensation_updated_parameter_names"] == []
    # Every attempt proves frame0 never moved frame1, even on reject.
    pc = payload["pose_compensation"]
    assert pc is not None
    assert pc["composite_accepted"] is False
    assert all(rec["frame1_receiver_uint8_unchanged"] for rec in pc["attempts"])
    assert all(rec["accepted"] is False for rec in pc["attempts"])
    # Full restore: ALL tensors (including head_rgb_0) are bit-identical.
    all_after = _all_tensor_snapshot(model, np)
    assert all_after.keys() == all_before.keys()
    for name, before_value in all_before.items():
        assert np.array_equal(before_value, all_after[name]), name


@skip_no_mlx
def test_frame0_compensation_restores_head0_before_frame1_violation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structural frame1-safety violation must not leave frame0 edits behind."""

    import mlx.core as mx
    from mlx.utils import tree_flatten

    import tac.substrates.hi_nerv.mlx_renderer as renderer_mod
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(7)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    target0, target1 = _near_boundary_all_class1_targets(cfg, mx)
    labels_np = np.ones(
        (cfg.num_pairs, cfg.output_height, cfg.output_width),
        dtype=np.int32,
    )
    teacher = _BehavioralSegNetTeacher(mx, mx.array(labels_np))
    model.initialize_output_head_bias_from_targets(target0, target1)

    def _head0_snapshot() -> dict[str, np.ndarray]:
        return {
            (
                ".".join(str(p) for p in raw)
                if isinstance(raw, (tuple, list))
                else str(raw)
            ): np.array(leaf, copy=True)
            for raw, leaf in tree_flatten(model.parameters())
            if leaf is not None
            and (
                ".".join(str(p) for p in raw)
                if isinstance(raw, (tuple, list))
                else str(raw)
            ).startswith("head_rgb_0")
        }

    head0_before = _head0_snapshot()
    assert head0_before
    original_array_equal = renderer_mod.np.array_equal
    monkeypatch.setattr(renderer_mod.np, "array_equal", lambda _left, _right: False)

    with pytest.raises(RuntimeError, match="frame0 pose compensation moved"):
        model.fit_target_region_birth_from_segnet(
            scorer_teacher=teacher,
            target_rgb_0=target0,
            target_rgb_1=target1,
            pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
            target_segnet_argmax_1=mx.array(labels_np),
            pose_teacher=_FrameDiffPoseTeacher(mx, k=0.5),
            max_pose_output_delta_l2=0.05,
            max_steps=24,
            learning_rate=2.0e-3,
        )

    head0_after = _head0_snapshot()
    assert head0_after.keys() == head0_before.keys()
    for name, before_value in head0_before.items():
        assert original_array_equal(before_value, head0_after[name]), name


@skip_no_mlx
def test_no_pose_teacher_path_never_attempts_compensation() -> None:
    """Without a pose teacher the composite operator is a no-op (byte-identical path).

    The compensation operator only exists when a pose teacher is available;
    with none, the payload must report compensation was never attempted and the
    frame0 head must be bit-identical to its pre-fit value.
    """

    import mlx.core as mx
    from mlx.utils import tree_flatten

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(7)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    target0, target1 = _green_dominant_targets(cfg, mx)
    labels_np = _block_labels(cfg, np)
    teacher = _BehavioralSegNetTeacher(mx, mx.array(labels_np))
    model.initialize_output_head_bias_from_targets(target0, target1)
    head0_before = {
        (".".join(str(p) for p in raw) if isinstance(raw, (tuple, list)) else str(raw)): np.array(leaf, copy=True)
        for raw, leaf in tree_flatten(model.parameters())
        if leaf is not None
        and (".".join(str(p) for p in raw) if isinstance(raw, (tuple, list)) else str(raw)).startswith("head_rgb_0")
    }
    assert head0_before  # the frame0 head exists

    payload = model.fit_target_region_birth_from_segnet(
        scorer_teacher=teacher,
        target_rgb_0=target0,
        target_rgb_1=target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        target_segnet_argmax_1=mx.array(labels_np),
        max_steps=24,
        learning_rate=2.0e-3,
    )

    assert payload["pose_compensation_attempted"] is False
    assert payload["pose_compensation_frame"] is None
    assert payload["composite_attempt_count"] == 0
    assert payload["composite_accepted_count"] == 0
    assert payload["composite_accepted"] is False
    assert payload["composite_delta_score_nonrate"] is None
    assert payload["compensation_updated_parameter_names"] == []
    assert payload["pose_compensation"] is None
    assert payload["receipt"]["pose_compensation"] is None
    # The frame0 head never moved on the no-pose path.
    head0_after = {
        (".".join(str(p) for p in raw) if isinstance(raw, (tuple, list)) else str(raw)): np.array(leaf, copy=True)
        for raw, leaf in tree_flatten(model.parameters())
        if leaf is not None
        and (".".join(str(p) for p in raw) if isinstance(raw, (tuple, list)) else str(raw)).startswith("head_rgb_0")
    }
    assert head0_after.keys() == head0_before.keys()
    for name, before_value in head0_before.items():
        assert np.array_equal(before_value, head0_after[name]), name


@skip_no_mlx
def test_compensation_never_relaxes_birth_allow_list_invariant() -> None:
    """Structural invariant: head_rgb_0 is never birth-scoped, on any outcome.

    Whether or not a composite is admitted, the frame0 compensation head must
    never appear in the birth allow-list nor in the birth ``updated_parameter_names``;
    it lives only in the separate compensation record.  This is the seg-safety
    contract (SegNet reads frame1 only) made a structural test.
    """

    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    # head_rgb_0 is categorically excluded from the birth allow-list.
    assert not allowed_birth_update_name("head_rgb_0.weight")
    assert not allowed_birth_update_name("head_rgb_0.bias")
    assert allowed_pose_compensation_update_name("head_rgb_0.weight")
    # ... and the two scopes are disjoint: nothing is both birth- and
    # compensation-scoped.
    for name in ("head_rgb_0.weight", "head_rgb_1.weight", "latents_fine", "fine_injector.proj.weight"):
        assert not (allowed_birth_update_name(name) and allowed_pose_compensation_update_name(name))

    mx.random.seed(7)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    target0, target1 = _near_boundary_all_class1_targets(cfg, mx)
    labels_np = np.ones((cfg.num_pairs, cfg.output_height, cfg.output_width), dtype=np.int32)
    teacher = _BehavioralSegNetTeacher(mx, mx.array(labels_np))
    model.initialize_output_head_bias_from_targets(target0, target1)

    payload = model.fit_target_region_birth_from_segnet(
        scorer_teacher=teacher,
        target_rgb_0=target0,
        target_rgb_1=target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        target_segnet_argmax_1=mx.array(labels_np),
        pose_teacher=_FrameDiffPoseTeacher(mx, k=0.5),
        max_pose_output_delta_l2=0.05,
        max_steps=24,
        learning_rate=2.0e-3,
    )

    # A composite actually landed in this config, exercising the populated path.
    assert payload["composite_accepted"] is True
    assert payload["compensation_updated_parameter_names"]
    # The birth scope (allowed prefixes + actual birth updates) excludes frame0.
    assert "head_rgb_0.*" not in payload["receipt"]["allowed_update_prefixes"]
    assert all(not name.startswith("head_rgb_0") for name in payload["updated_parameter_names"])
    assert all(allowed_birth_update_name(name) for name in payload["updated_parameter_names"])
    # If any compensation landed, its names are compensation-scoped ONLY and the
    # receipt builder (which fail-closes on a birth-scope collision) accepted it.
    for name in payload["compensation_updated_parameter_names"]:
        assert allowed_pose_compensation_update_name(name)
        assert not allowed_birth_update_name(name)
    assert payload["receipt"]["schema"] == TARGET_REGION_BIRTH_RECEIPT_SCHEMA
