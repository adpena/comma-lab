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
    build_target_region_birth_receipt,
    find_target_region_debts,
    region_margin_stats,
    select_worst_target_region,
    select_worst_target_region_with_mask,
)

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
        assert row.score_debt_units == pytest.approx(
            100.0 * row.region_pixel_count / total_scored
        )
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
    )
    assert receipt["schema"] == TARGET_REGION_BIRTH_RECEIPT_SCHEMA
    assert receipt["receiver_surface_uint8_changed_pixels"] == 9
    assert receipt["receiver_surface_argmax_flipped_pixels"] == 3
    assert receipt["receiver_surface_worst_region_margin_p50_delta"] == pytest.approx(
        -0.4
    )
    assert receipt["receiver_surface_float_rgb_delta_linf"] == pytest.approx(0.02)
    assert receipt["frame_scope"] == "frame1_seg_pose_joint"
    # False-authority contract: receipts can never promote or claim score.
    assert receipt["score_claim"] is False
    assert receipt["promotion_eligible"] is False


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
        name = (
            ".".join(str(p) for p in raw_name)
            if isinstance(raw_name, (tuple, list))
            else str(raw_name)
        )
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
        payload["after_region_margin_stats"]["margin_p50"]
        - payload["before_region_margin_stats"]["margin_p50"]
    )
    assert margin_delta < 0.0 or payload["after_region_hard_ratio"] > 0.0
    receipt = payload["receipt"]
    assert receipt["receiver_surface_uint8_changed_pixels"] > 0
    assert receipt["receiver_surface_float_rgb_delta_linf"] > 0.0
    # Scope proof: every updated tensor is birth-scoped, and at least one
    # update actually landed.
    assert payload["updated_parameter_names"]
    assert all(
        allowed_birth_update_name(name) for name in payload["updated_parameter_names"]
    )
    assert payload["grad_norm_by_group"]
    assert payload["runtime_sidecar_bytes"] == 0
    # Pose telemetry was unavailable -> the payload must say so, loudly.
    assert (
        "hinerv_target_region_birth_pose_trust_telemetry_missing"
        in payload["blockers"]
    )
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
    assert any(
        before_hashes[group] != after_hashes[group]
        for group in before_hashes
        if group != "out_of_scope"
    )


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
        (
            ".".join(str(p) for p in raw) if isinstance(raw, (tuple, list)) else str(raw)
        ): np.array(leaf, copy=True)
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
        (
            ".".join(str(p) for p in raw) if isinstance(raw, (tuple, list)) else str(raw)
        ): np.array(leaf, copy=True)
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
    assert (
        payload["pose_guard_rejected_step_count"] >= 1
        or payload["subquantum_rejected_step_count"] >= 1
    )
    assert (
        "hinerv_target_region_birth_pose_trust_telemetry_missing"
        not in payload["blockers"]
    )
    assert "hinerv_target_region_birth_no_accepted_step" in payload["blockers"]


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
    labels_np = np.zeros(
        (cfg.num_pairs, cfg.output_height, cfg.output_width), dtype=np.int32
    )
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
