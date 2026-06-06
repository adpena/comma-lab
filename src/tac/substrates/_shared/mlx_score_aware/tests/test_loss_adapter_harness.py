# SPDX-License-Identifier: MIT
"""MLX-bound tests: score-aware loss + adapter + end-to-end harness run.

MLX-bound tests skip cleanly on non-Apple-Silicon CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.substrates._shared.mlx_score_aware import (
    MlxScoreAwareAdapter,
    RendererBundle,
    component_loss_weight,
    decode_frames_nhwc01,
    pose_student_inputs_nhwc,
    run_mlx_score_aware_full_main,
    score_aware_loss,
)
from tac.substrates._shared.mlx_score_aware import adapter as score_adapter

try:
    import mlx.core as _mx  # noqa: F401

    _MLX = True
except ImportError:
    _MLX = False

mlx_only = pytest.mark.skipif(not _MLX, reason="MLX required (Apple Silicon)")

_LANE = "lane_mlx_score_aware_harness_refactor_plus_4_unlock_20260527"


def test_direct_live_trace_components_include_class_escape_actuators() -> None:
    assert (
        "segnet_direct_live_class_region_recon"
        in score_adapter._DIRECT_LIVE_TRACE_COMPONENTS
    )
    assert (
        "segnet_direct_live_rare_class_logit"
        in score_adapter._DIRECT_LIVE_TRACE_COMPONENTS
    )
    assert (
        "segnet_direct_live_target_mass_floor"
        in score_adapter._DIRECT_LIVE_TRACE_COMPONENTS
    )
    assert (
        "segnet_direct_live_target_min_ratio_floor"
        in score_adapter._DIRECT_LIVE_TRACE_COMPONENTS
    )


def test_scorer_support_ladder_stage_order_damps_base_loss() -> None:
    weights, metrics = score_adapter._apply_scorer_support_ladder_loss_weights(
        {"segnet_direct_live_base_loss": 1.0},
        stage=2,
        activation_weights=(
            score_adapter._normalized_scorer_support_ladder_activation_weights(None)
        ),
        growth_factor=2.0,
        max_multiplier=16.0,
        base_loss_max_when_active=0.25,
    )

    assert weights["segnet_direct_live_target_mass_floor"] == pytest.approx(2.0)
    assert weights["segnet_direct_live_rare_class_logit"] == pytest.approx(1.0)
    assert weights["segnet_direct_live_target_min_ratio_floor"] == pytest.approx(1.0)
    assert weights["segnet_direct_live_rare_class_logit_config_floor"] == pytest.approx(
        1.0
    )
    assert weights[
        "segnet_direct_live_target_min_ratio_floor_config_floor"
    ] == pytest.approx(1.0)
    assert "segnet_direct_live_class_balanced_hinge" not in weights
    assert "segnet_direct_live_class_balanced_hinge_config_floor" not in weights
    weights_stage3, _metrics_stage3 = (
        score_adapter._apply_scorer_support_ladder_loss_weights(
            {"segnet_direct_live_base_loss": 1.0},
            stage=3,
            activation_weights=(
                score_adapter._normalized_scorer_support_ladder_activation_weights(None)
            ),
            growth_factor=2.0,
            max_multiplier=16.0,
            base_loss_max_when_active=0.25,
        )
    )
    assert weights_stage3[
        "segnet_direct_live_class_balanced_hinge_config_floor"
    ] == pytest.approx(1.0)
    assert weights_stage3["segnet_direct_live_class_balanced_hinge"] == pytest.approx(
        1.0
    )
    assert weights[
        "segnet_direct_live_target_min_ratio_floor_config_floor"
    ] == pytest.approx(1.0)
    assert "segnet_direct_live_class_region_recon" not in weights
    assert "segnet_direct_live_class_region_recon_config_floor" not in weights
    assert weights["segnet_direct_live_base_loss"] == pytest.approx(0.25)
    assert metrics["scorer_support_ladder_active"] == pytest.approx(1.0)
    assert metrics["scorer_support_ladder_stage"] == pytest.approx(2.0)
    assert metrics["scorer_support_ladder_base_loss_damped"] == pytest.approx(1.0)
    assert metrics[
        "scorer_support_ladder_component_config_floor__segnet_direct_live_rare_class_logit"
    ] == pytest.approx(1.0)


def test_scorer_support_ladder_observation_escalates_only_after_stall() -> None:
    adapter = object.__new__(MlxScoreAwareAdapter)
    adapter._scorer_support_ladder_enabled = True
    adapter._scorer_support_ladder_target_coverage_floor = 1.0
    adapter._scorer_support_ladder_target_min_ratio_floor = 0.2
    adapter._scorer_support_ladder_patience_steps = 1
    adapter._scorer_support_ladder_stage = 0
    adapter._scorer_support_ladder_stale_steps = 0
    adapter._scorer_support_ladder_last_coverage = None
    adapter._scorer_support_ladder_last_min_ratio = None
    adapter._scorer_support_ladder_last_observation = {}

    first = adapter._observe_scorer_support_ladder(
        {
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.4,
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.0,
            "scorer_space_step_guard_rejected": 0.0,
        }
    )
    second = adapter._observe_scorer_support_ladder(
        {
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.4,
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.0,
            "scorer_space_step_guard_rejected": 0.0,
        }
    )
    progress = adapter._observe_scorer_support_ladder(
        {
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.6,
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.05,
            "scorer_space_step_guard_rejected": 0.0,
        }
    )

    assert first["scorer_support_ladder_next_stage"] == pytest.approx(1.0)
    assert second["scorer_support_ladder_next_stage"] == pytest.approx(2.0)
    assert second["scorer_support_ladder_previous_stage"] == pytest.approx(1.0)
    assert progress["scorer_support_ladder_next_stage"] == pytest.approx(2.0)
    assert progress["scorer_support_ladder_progress"] == pytest.approx(1.0)
    assert progress["scorer_support_ladder_next_stale_steps"] == pytest.approx(0.0)


@mlx_only
def test_adapter_marks_temporal_yuv6_floor_as_active_loss_part() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    bundle.posenet_temporal_signal_floor_weight = 0.5
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")

    metrics = adapter._score_aware_loss_part_metrics(
        mx.array([0, 1], dtype=mx.int32),
    )

    assert "loss_part_posenet_temporal_signal_floor" in metrics
    assert metrics["score_aware_loss_parts_active"] == pytest.approx(1.0)


def _tiny_dreamer_bundle(num_pairs: int = 4, distill: float = 0.5) -> RendererBundle:
    import mlx.core as mx

    from tac.substrates.dreamer_v3_rssm.module import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
    )

    cfg = DreamerV3RSSMConfig(
        num_pairs=num_pairs,
        num_groups=2,
        num_categories=4,
        decoder_latent_dim=8,
        base_channels=4,
        eval_size=(384, 512),
    )
    model = DreamerV3RSSMSubstrateMLX(cfg)
    t0 = mx.zeros((num_pairs, 384, 512, 3))
    t1 = mx.zeros((num_pairs, 384, 512, 3))
    # When a distillation term is active these helper bundles use the
    # scorer-BLIND mock (no real SegNet staged in the unit-test fast path), so
    # they must EXPLICITLY opt in via allow_mock_scorer_teacher per the C6 IBPS
    # fail-closed invariant. The real-scorer-bound path is exercised in
    # test_scorer_binding.py with a real SegNet teacher.
    return RendererBundle(
        model=model,
        target_rgb_0=t0,
        target_rgb_1=t1,
        num_pairs=num_pairs,
        forward_convention="call_b2chw_255",
        distillation_weight=distill,
        allow_mock_scorer_teacher=distill > 0.0,
    )


# --------------------------------------------------------------------------- #
# loss module
# --------------------------------------------------------------------------- #


@mlx_only
def test_decode_frames_nhwc01_shapes() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle()
    idx = mx.array([0, 1], dtype=mx.int32)
    rgb_0, rgb_1 = decode_frames_nhwc01(bundle, idx)
    assert rgb_0.shape == (2, 384, 512, 3)
    assert rgb_1.shape == (2, 384, 512, 3)


@mlx_only
def test_score_aware_loss_is_finite_and_decomposed() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(distill=0.5)
    idx = mx.array([0, 1], dtype=mx.int32)
    total, parts = score_aware_loss(bundle, idx)
    mx.eval(total)
    assert float(total.item()) == float(total.item())  # not NaN
    assert {"recon", "distill", "total"} <= set(parts)


@mlx_only
def test_score_aware_loss_applies_core_stage_weights() -> None:
    import mlx.core as mx

    assert component_loss_weight({"segnet": 0.25}, "distill") == 0.25
    assert component_loss_weight({"posenet_distill": 0.0}, "pose_distill") == 0.0
    assert component_loss_weight({}, "recon") == 1.0

    bundle = _tiny_dreamer_bundle(distill=0.5)
    idx = mx.array([0, 1], dtype=mx.int32)
    total_default, parts_default = score_aware_loss(bundle, idx)
    total_zero_seg, parts_zero_seg = score_aware_loss(
        bundle,
        idx,
        loss_weights={"distill": 0.0},
    )
    mx.eval(total_default, total_zero_seg, parts_default["recon"])

    assert "distill" in parts_default
    assert "distill" not in parts_zero_seg
    assert float(total_zero_seg.item()) == pytest.approx(
        float(parts_default["recon"].item())
    )
    assert float(total_default.item()) >= float(total_zero_seg.item())


@mlx_only
def test_score_aware_loss_no_distill_when_weight_zero() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(distill=0.0)
    idx = mx.array([0, 1], dtype=mx.int32)
    _total, parts = score_aware_loss(bundle, idx)
    assert "distill" not in parts
    assert "recon" in parts


@mlx_only
def test_scorer_input_distribution_guard_includes_dynamic_range_term() -> None:
    import mlx.core as mx

    base = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        scorer_input_distribution_guard_weight=1.0,
    )
    idx = mx.array([0, 1], dtype=mx.int32)

    _total, parts = score_aware_loss(bundle, idx)
    mx.eval(
        parts["scorer_input_distribution_guard"],
        parts["scorer_input_distribution_guard_dynamic_range"],
        parts["scorer_input_distribution_guard_spatial_gradient"],
        parts["scorer_input_distribution_guard_segnet_frame1_spatial_gradient"],
        parts["scorer_input_distribution_guard_segnet_frame1_mse"],
        parts["scorer_input_distribution_guard_segnet_frame1_mae"],
        parts["scorer_input_distribution_guard_yuv6_pair_dynamic_range"],
        parts["scorer_input_distribution_guard_yuv6_pair_spatial_gradient"],
        parts["scorer_input_distribution_guard_yuv6_pair_mse"],
        parts["scorer_input_distribution_guard_yuv6_pair_mae"],
        parts["scorer_input_distribution_guard_yuv6_temporal_delta"],
        parts["scorer_input_distribution_guard_yuv6_temporal_delta_mse"],
        parts["scorer_input_distribution_guard_yuv6_temporal_delta_mae"],
    )

    assert "scorer_input_distribution_guard_dynamic_range" in parts
    assert "scorer_input_distribution_guard_spatial_gradient" in parts
    assert "scorer_input_distribution_guard_segnet_frame1_spatial_gradient" in parts
    assert "scorer_input_distribution_guard_segnet_frame1_mse" in parts
    assert "scorer_input_distribution_guard_segnet_frame1_mae" in parts
    assert "scorer_input_distribution_guard_yuv6_pair" in parts
    assert "scorer_input_distribution_guard_yuv6_pair_dynamic_range" in parts
    assert "scorer_input_distribution_guard_yuv6_pair_spatial_gradient" in parts
    assert "scorer_input_distribution_guard_yuv6_pair_mse" in parts
    assert "scorer_input_distribution_guard_yuv6_pair_mae" in parts
    assert "scorer_input_distribution_guard_yuv6_temporal_delta" in parts
    assert "scorer_input_distribution_guard_yuv6_temporal_delta_mse" in parts
    assert "scorer_input_distribution_guard_yuv6_temporal_delta_mae" in parts
    assert float(parts["scorer_input_distribution_guard_dynamic_range"].item()) >= 0.0
    assert (
        float(parts["scorer_input_distribution_guard_spatial_gradient"].item())
        >= 0.0
    )
    assert (
        float(
            parts[
                "scorer_input_distribution_guard_segnet_frame1_spatial_gradient"
            ].item()
        )
        >= 0.0
    )
    assert (
        float(parts["scorer_input_distribution_guard_segnet_frame1_mse"].item())
        >= 0.0
    )
    assert (
        float(parts["scorer_input_distribution_guard_segnet_frame1_mae"].item())
        >= 0.0
    )
    assert (
        float(
            parts[
                "scorer_input_distribution_guard_yuv6_pair_dynamic_range"
            ].item()
        )
        >= 0.0
    )
    assert (
        float(
            parts[
                "scorer_input_distribution_guard_yuv6_pair_spatial_gradient"
            ].item()
        )
        >= 0.0
    )
    assert (
        float(parts["scorer_input_distribution_guard_yuv6_temporal_delta"].item())
        >= 0.0
    )
    assert (
        float(parts["scorer_input_distribution_guard_yuv6_pair_mse"].item()) >= 0.0
    )
    assert (
        float(parts["scorer_input_distribution_guard_yuv6_pair_mae"].item()) >= 0.0
    )
    assert (
        float(
            parts["scorer_input_distribution_guard_yuv6_temporal_delta_mse"].item()
        )
        >= 0.0
    )
    assert (
        float(
            parts["scorer_input_distribution_guard_yuv6_temporal_delta_mae"].item()
        )
        >= 0.0
    )


@mlx_only
def test_score_aware_loss_applies_pr95_eval_roundtrip_before_recon() -> None:
    import mlx.core as mx

    class _ConstantPair:
        def __call__(self, idx):
            batch = int(idx.shape[0])
            return mx.full((batch, 2, 3, 4, 5), 127.4)

    targets = mx.zeros((2, 4, 5, 3))
    common = {
        "model": _ConstantPair(),
        "target_rgb_0": targets,
        "target_rgb_1": targets,
        "num_pairs": 2,
        "forward_convention": "call_b2chw_255",
    }
    idx = mx.array([0, 1], dtype=mx.int32)
    disabled = RendererBundle(**common)
    enabled = RendererBundle(
        **common,
        eval_roundtrip_ste_enabled=True,
        eval_roundtrip_camera_hw=(8, 10),
    )

    _disabled_total, disabled_parts = score_aware_loss(disabled, idx)
    _enabled_total, enabled_parts = score_aware_loss(enabled, idx)
    mx.eval(disabled_parts["recon"], enabled_parts["recon"])

    assert float(enabled_parts["recon"].item()) < float(
        disabled_parts["recon"].item()
    )
    expected = 2.0 * (127.0 / 255.0) ** 2
    assert abs(float(enabled_parts["recon"].item()) - expected) < 1e-5


@mlx_only
def test_pose_student_inputs_can_use_pr95_yuv6_preprocess() -> None:
    import mlx.core as mx

    bundle = RendererBundle(
        model=object(),
        target_rgb_0=None,
        target_rgb_1=None,
        num_pairs=1,
        pose_student_input_preprocess="pr95_yuv6",
    )
    rgb_0 = mx.ones((1, 8, 8, 3)) * 0.5
    rgb_1 = mx.zeros((1, 8, 8, 3))

    pose_0, pose_1 = pose_student_inputs_nhwc(bundle, rgb_0, rgb_1)
    mx.eval(pose_0, pose_1)

    assert pose_0.shape == (1, 4, 4, 6)
    assert pose_1.shape == (1, 4, 4, 6)
    assert float(mx.max(mx.abs(pose_0)).item()) > 0.0


@mlx_only
def test_decode_mlx_targets_honors_explicit_source_pair_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.data as data
    from tac.substrates._shared.mlx_score_aware.targets import decode_mlx_targets

    class _Frame:
        def __init__(self, value: int) -> None:
            self._value = int(value)

        def numpy(self) -> np.ndarray:
            return np.full((2, 3, 3), self._value, dtype=np.uint8)

    captured: dict[str, int] = {}

    def _fake_decode_video(*_args, **kwargs):
        captured["max_frames"] = int(kwargs["max_frames"])
        return [_Frame(i) for i in range(captured["max_frames"])]

    monkeypatch.setattr(data, "decode_video", _fake_decode_video)

    target0, target1 = decode_mlx_targets(
        "unit.mkv",
        num_pairs=2,
        output_height=2,
        output_width=3,
        pair_indices=(3, 1, 3),
    )

    assert captured["max_frames"] == 8
    assert tuple(target0.shape) == (2, 2, 3, 3)
    np.testing.assert_allclose(
        np.asarray(target0)[:, 0, 0, 0],
        np.array([6.0 / 255.0, 2.0 / 255.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(target1)[:, 0, 0, 0],
        np.array([7.0 / 255.0, 3.0 / 255.0], dtype=np.float32),
    )


@mlx_only
def test_decode_mlx_targets_uses_official_scorer_surface_at_eval_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.data as data
    from tac.substrates._shared.mlx_score_aware.targets import decode_mlx_targets

    class _Frame:
        def __init__(self, value: int, h: int, w: int) -> None:
            self._value = int(value)
            self._h = int(h)
            self._w = int(w)

        def numpy(self) -> np.ndarray:
            return np.full((self._h, self._w, 3), self._value, dtype=np.uint8)

    seen: dict[str, int] = {}

    def _fake_decode_video(*_args, **kwargs):
        seen["target_h"] = int(kwargs["target_h"])
        seen["target_w"] = int(kwargs["target_w"])
        seen["max_frames"] = int(kwargs["max_frames"])
        return [
            _Frame(i, seen["target_h"], seen["target_w"])
            for i in range(seen["max_frames"])
        ]

    monkeypatch.setattr(data, "decode_video", _fake_decode_video)

    target0, target1 = decode_mlx_targets(
        "unit.mkv",
        num_pairs=1,
        output_height=384,
        output_width=512,
    )

    assert seen == {"target_h": 874, "target_w": 1164, "max_frames": 2}
    assert tuple(target0.shape) == (1, 384, 512, 3)
    assert tuple(target1.shape) == (1, 384, 512, 3)
    np.testing.assert_allclose(np.asarray(target0)[0, 0, 0, 0], 0.0)
    np.testing.assert_allclose(np.asarray(target1)[0, 0, 0, 0], 1.0 / 255.0)


@mlx_only
def test_decode_mlx_targets_chunked_official_surface_decodes_only_selected_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tac.substrates._shared.mlx_score_aware import targets as targets_mod

    captured: dict[str, object] = {}

    def _fail_decode_video(*_args, **_kwargs):
        raise AssertionError("chunked official scorer hydration must not use prefix decode")

    def _fake_decode_selected_full_camera_frames(_video_path, *, frame_indices):
        captured["frame_indices"] = tuple(int(idx) for idx in frame_indices)
        return [
            np.full((874, 1164, 3), int(idx), dtype=np.uint8)
            for idx in frame_indices
        ]

    monkeypatch.setattr(
        targets_mod,
        "decode_video",
        _fail_decode_video,
        raising=False,
    )
    monkeypatch.setattr(
        targets_mod,
        "_decode_selected_full_camera_frames",
        _fake_decode_selected_full_camera_frames,
    )

    target0, target1 = targets_mod.decode_mlx_targets(
        "unit.mkv",
        num_pairs=2,
        output_height=384,
        output_width=512,
        pair_indices=(3, 1),
        target_hydration_strategy="chunked",
        official_scorer_surface_chunk_frames=1,
    )

    assert captured["frame_indices"] == (6, 7, 2, 3)
    assert tuple(target0.shape) == (2, 384, 512, 3)
    assert tuple(target1.shape) == (2, 384, 512, 3)
    np.testing.assert_allclose(
        np.asarray(target0)[:, 0, 0, 0],
        np.array([6.0 / 255.0, 2.0 / 255.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(target1)[:, 0, 0, 0],
        np.array([7.0 / 255.0, 3.0 / 255.0], dtype=np.float32),
    )


@mlx_only
def test_score_aware_loss_uses_source_pairs_for_model_and_local_rows_for_targets() -> None:
    import mlx.core as mx

    class _SourceIndexedPair:
        def __init__(self) -> None:
            self.calls: list[list[int]] = []

        def __call__(self, idx):
            self.calls.append([int(value) for value in np.asarray(idx).tolist()])
            src = idx.astype(mx.float32)
            base = mx.reshape(src, (-1, 1, 1, 1))
            rgb_0 = mx.broadcast_to(base, (int(idx.shape[0]), 3, 1, 1))
            rgb_1 = mx.broadcast_to(base + 10.0, (int(idx.shape[0]), 3, 1, 1))
            return mx.stack([rgb_0, rgb_1], axis=1)

    model = _SourceIndexedPair()
    target_rgb_0 = mx.array(
        [
            [[[2.0 / 255.0, 2.0 / 255.0, 2.0 / 255.0]]],
            [[[0.0, 0.0, 0.0]]],
        ],
        dtype=mx.float32,
    )
    target_rgb_1 = mx.array(
        [
            [[[12.0 / 255.0, 12.0 / 255.0, 12.0 / 255.0]]],
            [[[10.0 / 255.0, 10.0 / 255.0, 10.0 / 255.0]]],
        ],
        dtype=mx.float32,
    )
    bundle = RendererBundle(
        model=model,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        num_pairs=2,
        forward_convention="call_b2chw_255",
        source_pair_indices=(2, 0),
    )

    total, parts = score_aware_loss(bundle, mx.array([0, 1], dtype=mx.int32))
    mx.eval(total, parts["recon"])

    assert model.calls[-1] == [2, 0]
    assert float(parts["recon"].item()) == pytest.approx(0.0)


@mlx_only
def test_adapter_priority_sampling_maps_source_pairs_to_local_rows() -> None:
    import mlx.core as mx

    bundle = RendererBundle(
        model=object(),
        target_rgb_0=mx.zeros((2, 1, 1, 3), dtype=mx.float32),
        target_rgb_1=mx.zeros((2, 1, 1, 3), dtype=mx.float32),
        num_pairs=2,
        source_pair_indices=(417, 22),
    )
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="hi_nerv",
        prioritized_pair_indices=(417, 22, 999),
    )

    sampled = adapter.sample_batch(batch_size=2, seed=0)
    observability = adapter.batch_observability(sampled)

    assert np.asarray(sampled).tolist() == [0, 1]
    assert observability is not None
    assert observability["requested_priority_pair_indices"] == [417, 22, 999]
    assert observability["priority_local_pair_indices_in_batch"] == [0, 1]
    assert observability["priority_source_pair_indices_in_batch"] == [417, 22]
    assert observability["unresolved_priority_pair_indices"] == [999]
    assert observability["source_pair_indices"] == [417, 22]
    assert observability["priority_pair_alignment_mode"] == (
        "source_priority_pairs_to_local_rows"
    )
    assert observability["pair_index_alignment_mode"] == (
        "local_target_rows_to_source_pair_indices"
    )
    assert observability["score_claim"] is False


@mlx_only
def test_adapter_scorer_error_sampling_maps_source_weights_to_local_rows() -> None:
    import mlx.core as mx

    bundle = RendererBundle(
        model=object(),
        target_rgb_0=mx.zeros((3, 1, 1, 3), dtype=mx.float32),
        target_rgb_1=mx.zeros((3, 1, 1, 3), dtype=mx.float32),
        num_pairs=3,
        forward_convention="call_b2chw_255",
        source_pair_indices=(10, 20, 30),
    )
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="hi_nerv",
        pair_sampling_weights={20: 1.0},
        pair_sampling_default_weight=0.0,
    )

    sampled = adapter.sample_batch(batch_size=1, seed=7)
    observability = adapter.batch_observability(sampled)

    assert np.asarray(sampled).tolist() == [1]
    assert observability is not None
    assert observability["sampling_policy"] == "scorer_error_weighted_random"
    assert observability["source_pair_indices"] == [20]
    scorer_sampling = observability["scorer_error_pair_sampling"]
    assert scorer_sampling["enabled"] is True
    assert scorer_sampling["pair_weight_alignment_mode"] == (
        "source_weight_pairs_to_local_rows"
    )
    assert scorer_sampling["sampled_pair_weights"] == [1.0]
    assert scorer_sampling["score_claim"] is False


@mlx_only
def test_adapter_scorer_error_sampling_refuses_zero_mass_curriculum() -> None:
    import mlx.core as mx

    bundle = RendererBundle(
        model=object(),
        target_rgb_0=mx.zeros((2, 1, 1, 3), dtype=mx.float32),
        target_rgb_1=mx.zeros((2, 1, 1, 3), dtype=mx.float32),
        num_pairs=2,
        forward_convention="call_b2chw_255",
        source_pair_indices=(10, 20),
    )
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="hi_nerv",
        pair_sampling_weights={999: 1.0},
        pair_sampling_default_weight=0.0,
    )

    with pytest.raises(ValueError, match="zero sampling mass"):
        adapter.sample_batch(batch_size=1, seed=0)


@mlx_only
def test_score_aware_loss_extra_term_weighted() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(distill=0.0)
    bundle.extra_loss_terms = lambda _m, _i: {"commit": mx.array(2.0)}
    bundle.extra_loss_weights = {"commit": 0.5}
    idx = mx.array([0], dtype=mx.int32)
    _total, parts = score_aware_loss(bundle, idx)
    assert "commit" in parts
    mx.eval(parts["commit"])
    assert abs(float(parts["commit"].item()) - 2.0) < 1e-5


def test_component_loss_weight_supports_core_aliases() -> None:
    weights = {
        "reconstruction": 0.25,
        "segnet": 0.5,
        "posenet_distill": 0.125,
        "contrast_floor": 0.75,
        "shape_tether": 1.25,
        "direct_live_segnet": 1.5,
        "segnet_direct_live_balanced_ce": 0.0,
    }

    assert component_loss_weight(weights, "recon") == pytest.approx(0.25)
    assert component_loss_weight(weights, "distill") == pytest.approx(0.5)
    assert component_loss_weight(weights, "pose_distill") == pytest.approx(0.125)
    assert component_loss_weight(weights, "scorer_input_contrast_floor") == pytest.approx(0.75)
    assert component_loss_weight(weights, "scorer_input_shape_tether") == pytest.approx(1.25)
    assert component_loss_weight(weights, "segnet_direct_live_distill") == pytest.approx(1.5)
    assert component_loss_weight(weights, "segnet_direct_live_class_balanced_ce") == 0.0
    assert component_loss_weight(weights, "other", default=2.0) == pytest.approx(2.0)


@mlx_only
def test_score_aware_loss_core_stage_weights_gate_terms() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(distill=0.5)
    idx = mx.array([0, 1], dtype=mx.int32)
    total_default, parts_default = score_aware_loss(bundle, idx)
    total_gated, parts_gated = score_aware_loss(
        bundle,
        idx,
        loss_weights={"recon": 0.25, "distill": 0.0},
    )
    mx.eval(total_default, total_gated, parts_default["recon"], parts_default["distill"])

    expected_gated = 0.25 * float(parts_default["recon"].item())
    assert "distill" in parts_default
    assert "distill" not in parts_gated
    assert float(total_gated.item()) == pytest.approx(expected_gated, abs=1e-6)
    assert float(total_default.item()) > float(total_gated.item())


# --------------------------------------------------------------------------- #
# adapter module
# --------------------------------------------------------------------------- #


@mlx_only
def test_adapter_train_step_reduces_loss_over_steps() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(distill=0.0)
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    batch = mx.array([0, 1, 2, 3], dtype=mx.int32)
    losses = [
        adapter.train_step(batch, learning_rate=1e-2, loss_weights={})["total"]
        for _ in range(20)
    ]
    assert losses[-1] < losses[0]


@mlx_only
def test_adapter_runs_substrate_post_train_step_update_hook() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(distill=0.0)
    calls: list[tuple[int, ...]] = []

    def _post_train_step_update(batch):
        calls.append(tuple(int(x) for x in mx.array(batch).tolist()))
        return {
            "eval_targets": [],
            "metrics": {"custom_post_update_called": len(calls)},
        }

    bundle.model.post_train_step_update = _post_train_step_update
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    batch = mx.array([0, 1, 2, 3], dtype=mx.int32)
    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    assert calls == [(0, 1, 2, 3)]
    assert metrics["custom_post_update_called"] == 1.0


@mlx_only
def test_adapter_trains_pose_head_jointly() -> None:
    import mlx.core as mx

    from tac.substrates.hinton_distilled_scorer_surrogate import (
        RealPoseNetTeacherCache,
        build_learnable_pose_student_head,
    )

    base = _tiny_dreamer_bundle(num_pairs=4, distill=0.0)
    pose_teacher = RealPoseNetTeacherCache(
        teacher_pose_np=mx.ones((4, 6)),
        num_pairs=4,
        pose_dims=6,
    )
    pose_head = build_learnable_pose_student_head(seed=11, input_channels=6)
    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        pose_distillation_weight=0.5,
        pose_scorer_teacher=pose_teacher,
        learnable_pose_student_head=pose_head,
        pose_student_input_preprocess="pr95_yuv6",
    )
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    batch = mx.array([0, 1, 2, 3], dtype=mx.int32)
    w0 = mx.array(pose_head.weight)
    adapter.train_step(batch, learning_rate=1e-2, loss_weights={})
    moved = float(mx.max(mx.abs(pose_head.weight - w0)).item())
    assert moved > 0.0, "pose head params must train jointly (sibling step)"


@mlx_only
def test_adapter_train_step_emits_active_score_loss_parts() -> None:
    import mlx.core as mx

    from tac.substrates.hinton_distilled_scorer_surrogate import (
        RealPoseNetTeacherCache,
        build_learnable_pose_student_head,
    )

    base = _tiny_dreamer_bundle(num_pairs=4, distill=0.5)
    pose_teacher = RealPoseNetTeacherCache(
        teacher_pose_np=mx.ones((4, 6)),
        num_pairs=4,
        pose_dims=6,
    )
    pose_head = build_learnable_pose_student_head(seed=23, input_channels=6)
    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        distillation_weight=0.5,
        allow_mock_scorer_teacher=True,
        pose_distillation_weight=0.5,
        pose_scorer_teacher=pose_teacher,
        learnable_pose_student_head=pose_head,
        pose_student_input_preprocess="pr95_yuv6",
        scorer_input_distribution_guard_weight=2.0,
    )
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    batch = mx.array([0, 1, 2, 3], dtype=mx.int32)

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    assert metrics["score_aware_loss_parts_active"] == 1.0
    assert "loss_part_distill" in metrics
    assert "loss_part_weighted_distill" in metrics
    assert "loss_part_pose_distill" in metrics
    assert "loss_part_pose_distill_raw_mse" in metrics
    assert "loss_part_pose_distill_train_loss" in metrics
    assert "loss_part_pose_score_term" in metrics
    assert "loss_part_weighted_pose_raw_mse" in metrics
    assert "loss_part_weighted_pose_distill_train_loss" in metrics
    assert "loss_part_weighted_pose_score_term" in metrics
    assert "loss_part_weighted_pose_distill" in metrics
    assert "loss_part_scorer_input_distribution_guard" in metrics
    assert "loss_part_scorer_input_distribution_guard_dynamic_range" in metrics
    assert "loss_part_scorer_input_distribution_guard_spatial_gradient" in metrics
    assert (
        "loss_part_scorer_input_distribution_guard_segnet_frame1_spatial_gradient"
        in metrics
    )
    assert "loss_part_scorer_input_distribution_guard_segnet_frame1_mse" in metrics
    assert "loss_part_scorer_input_distribution_guard_segnet_frame1_mae" in metrics
    assert "loss_part_scorer_input_distribution_guard_yuv6_pair" in metrics
    assert (
        "loss_part_scorer_input_distribution_guard_yuv6_pair_spatial_gradient"
        in metrics
    )
    assert "loss_part_scorer_input_distribution_guard_yuv6_pair_mse" in metrics
    assert "loss_part_scorer_input_distribution_guard_yuv6_pair_mae" in metrics
    assert "loss_part_scorer_input_distribution_guard_yuv6_temporal_delta" in metrics
    assert (
        "loss_part_scorer_input_distribution_guard_yuv6_temporal_delta_mse"
        in metrics
    )
    assert (
        "loss_part_scorer_input_distribution_guard_yuv6_temporal_delta_mae"
        in metrics
    )
    assert "loss_part_weighted_scorer_input_distribution_guard" in metrics
    assert metrics["loss_part_config_weight_scorer_input_distribution_guard"] == pytest.approx(2.0)


@mlx_only
def test_adapter_train_step_emits_weighted_direct_live_segnet_part() -> None:
    import mlx.core as mx

    class _LiveSegTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            return mx.ones((int(idx.shape[0]), 384, 512, self.num_classes))

        def teacher_logits_for_frames_nhwc01(self, frames):
            b, h, w, _c = frames.shape
            return mx.zeros((int(b), int(h), int(w), self.num_classes))

    base = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        scorer_teacher=_LiveSegTeacher(),
        segnet_direct_live_distillation_weight=0.25,
    )
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={"distill": 0.5},
    )

    raw = metrics["loss_part_segnet_direct_live_distill"]
    assert metrics["score_aware_loss_parts_active"] == pytest.approx(1.0)
    assert metrics["loss_part_stage_weight_segnet_direct_live_distill"] == pytest.approx(
        0.5
    )
    assert metrics[
        "loss_part_weighted_segnet_direct_live_distill"
    ] == pytest.approx(raw * 0.25 * 0.5)
    assert metrics[
        "loss_part_segnet_direct_live_argmax_disagreement"
    ] == pytest.approx(0.0)
    assert metrics[
        "loss_part_segnet_direct_live_candidate_class_0_fraction"
    ] == pytest.approx(1.0)
    assert metrics[
        "loss_part_segnet_direct_live_candidate_occupied_class_fraction"
    ] == pytest.approx(0.2)


@mlx_only
def test_pr95_stage_train_step_emits_direct_live_posenet_loss_parts() -> None:
    import mlx.core as mx

    from tac.substrates.hinton_distilled_scorer_surrogate import (
        RealPoseNetTeacherCache,
    )

    class _FakeSegTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, indices):
            batch = int(indices.shape[0])
            return mx.zeros((batch, 384, 512, self.num_classes), dtype=mx.float32)

        def teacher_logits_for_frames_nhwc01(self, frames):
            batch, height, width, _channels = frames.shape
            return mx.zeros(
                (batch, height, width, self.num_classes),
                dtype=mx.float32,
            )

    class _TinyLivePoseAdapter:
        def __call__(self, yuv6_pair_nhwc):
            mean = mx.mean(yuv6_pair_nhwc, axis=(1, 2, 3))
            pose = mx.stack(
                [
                    mean,
                    mean * 0.5,
                    mean * 0.25,
                    mean * 0.125,
                    mean * 0.0625,
                    mean * 0.03125,
                ],
                axis=-1,
            )
            return {"pose": pose.astype(mx.float32)}

    base = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    pose_teacher = RealPoseNetTeacherCache(
        teacher_pose_np=mx.zeros((2, 6), dtype=mx.float32),
        num_pairs=2,
        pose_dims=6,
        per_dim_scale=mx.ones((6,), dtype=mx.float32),
        live_posenet_adapter=_TinyLivePoseAdapter(),
    )
    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        scorer_teacher=_FakeSegTeacher(),
        segnet_direct_live_distillation_weight=0.25,
        pose_direct_live_distillation_weight=0.75,
        pose_scorer_teacher=pose_teacher,
    )
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=8,
    )

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-3,
        loss_weights={"pose_direct_live_distill": 1.0},
    )

    assert metrics["pr95_stage_loss_parts_active"] == pytest.approx(1.0)
    assert "loss_part_pr95_stage_pose_direct_live_score_term" in metrics
    assert "loss_part_pose_direct_live_score_term" in metrics
    assert "loss_part_pose_direct_live_raw_mse" in metrics
    assert "loss_part_weighted_pose_direct_live_score_term" in metrics
    archive_health = adapter.archive_selection_health(
        adapter.model,
        mx.array([0, 1], dtype=mx.int32),
    )
    assert archive_health is not None
    assert "segnet_direct_live_candidate_target_class_coverage_fraction" in archive_health
    assert "segnet_direct_live_candidate_target_class_min_ratio" in archive_health
    assert "pose_direct_live_score_term" in archive_health
    assert "pose_direct_live_raw_mse" in archive_health
    assert "pose_direct_live_yuv6_pair_std" in archive_health
    assert "pose_direct_live_yuv6_pair_temporal_delta_std" in archive_health
    assert metrics["loss_part_stage_weight_pose_direct_live_distill"] == pytest.approx(
        1.0
    )
    assert metrics["loss_part_config_weight_pose_direct_live_distill"] == pytest.approx(
        0.75
    )
    assert metrics["loss_part_weighted_pose_direct_live_score_term"] > 0.0


@mlx_only
def test_pr95_stage_direct_live_region_recon_gets_target_frame() -> None:
    import mlx.core as mx

    class _LiveSegTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, indices):
            batch = int(indices.shape[0])
            zeros = mx.zeros((batch, 384, 512, self.num_classes), dtype=mx.float32)
            class_one = mx.ones((batch, 384, 512, 1), dtype=mx.float32) * 4.0
            return mx.concatenate(
                [zeros[..., :1], class_one, zeros[..., 2:]],
                axis=-1,
            )

        def teacher_logits_for_frames_nhwc01(self, frames):
            batch, height, width, _channels = frames.shape
            mean = mx.mean(frames, axis=-1, keepdims=True)
            return mx.concatenate(
                [
                    mean,
                    1.0 - mean,
                    mean * 0.5,
                    mean * 0.25,
                    mean * 0.125,
                ],
                axis=-1,
            ).reshape((int(batch), int(height), int(width), self.num_classes))

    base = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        scorer_teacher=_LiveSegTeacher(),
        segnet_direct_live_distillation_weight=0.0,
        segnet_direct_live_class_region_recon_weight=0.75,
    )
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=8,
    )

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-3,
        loss_weights={
            "segnet_direct_live_distill": 1.0,
            "segnet_direct_live_class_region_recon": 1.0,
        },
    )

    assert metrics["pr95_stage_loss_parts_active"] == pytest.approx(1.0)
    assert "loss_part_pr95_stage_segnet_direct_live_class_region_recon_loss" in metrics
    assert "loss_part_segnet_direct_live_class_region_recon_loss" in metrics
    assert (
        metrics["loss_part_pr95_stage_segnet_direct_live_class_region_recon_loss"]
        == pytest.approx(
            metrics["loss_part_segnet_direct_live_class_region_recon_loss"]
        )
    )
    assert metrics[
        "loss_part_pr95_stage_segnet_direct_live_class_region_recon_weight"
    ] == pytest.approx(0.75)


@mlx_only
def test_pr95_faithful_train_step_emits_scorer_space_guard_metrics() -> None:
    import mlx.core as mx

    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        build_learnable_student_head,
    )

    class _LiveSegTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            b = int(idx.shape[0])
            zeros = mx.zeros((b, 384, 512, self.num_classes), dtype=mx.float32)
            class_two = mx.ones((b, 384, 512, 1), dtype=mx.float32) * 3.0
            return mx.concatenate(
                [zeros[..., :2], class_two, zeros[..., 3:]],
                axis=-1,
            )

        def teacher_logits_for_frames_nhwc01(self, frames):
            mean = mx.mean(frames, axis=-1, keepdims=True)
            return mx.concatenate(
                [mean * float(i + 1) for i in range(self.num_classes)],
                axis=-1,
            )

    base = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        distillation_weight=0.5,
        scorer_teacher=_LiveSegTeacher(),
        learnable_student_head=build_learnable_student_head(num_classes=5, seed=37),
        allow_segnet_only_research=True,
    )
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=8,
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.1,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.1,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=0.1,
        scorer_space_step_guard_max_post_segnet_contrast_ratio=2.0,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.6,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.6,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.2,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 1.0,
            "loss_part_pose_score_term": 0.1,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-3,
        loss_weights={},
    )

    assert call_count >= 3
    assert metrics["pr95_stage_loss_parts_active"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_enabled"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_eligible"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction"
    ] == pytest.approx(0.1)
    assert (
        metrics["scorer_space_step_guard_pr95_optimizer_lr_scale_applied"]
        == pytest.approx(1.0)
    )
    assert metrics["scorer_space_step_guard_effective_optimizer_learning_rate"] > 0.0
    assert (
        "dynamics_pre_update_loss_part_segnet_direct_live_argmax_disagreement"
        in metrics
    )
    assert "dynamics_gradient_all_l2" in metrics
    assert "dynamics_param_delta_all_l2" in metrics


@mlx_only
def test_adapter_train_step_updates_dual_ascent_once_and_records_metadata() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)

    def _post_train_step_update(_batch):
        return {
            "eval_targets": [],
            "metrics": {"unit_rate_proxy": 2.0},
        }

    bundle.model.post_train_step_update = _post_train_step_update
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        train_time_dual_ascent_config={
            "enabled": True,
            "constraints": [
                {
                    "constraint_id": "unit_rate",
                    "metric_name": "unit_rate_proxy",
                    "loss_weight_key": "recon",
                    "target": 1.0,
                    "dual_lr": 0.5,
                    "max_lambda": 4.0,
                }
            ],
        },
    )
    batch = mx.array([0, 1], dtype=mx.int32)

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    assert metrics["dual_ascent_active"] == pytest.approx(1.0)
    assert metrics["dual_ascent_step"] == pytest.approx(1.0)
    assert metrics["dual_ascent_metric__unit_rate"] == pytest.approx(2.0)
    assert metrics["dual_ascent_lambda__unit_rate"] == pytest.approx(0.5)
    metadata = adapter.artifact_metadata()["score_aware_training"][
        "train_time_dual_ascent"
    ]
    assert metadata["enabled"] is True
    assert metadata["step_count"] == 1
    assert metadata["state"]["unit_rate"]["lambda"] == pytest.approx(0.5)


@mlx_only
def test_adapter_train_step_feeds_section_bytes_into_dual_ascent() -> None:
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.dual_ascent import (
        CONTEST_RATE_SCORE_PER_BYTE,
    )

    base = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    callback_calls: list[dict[str, object]] = []

    def _section_byte_metrics(model, batch, loss_weights):
        callback_calls.append(
            {
                "model_is_bundle_model": model is base.model,
                "batch_shape": tuple(batch.shape),
                "loss_weight_keys": sorted(loss_weights),
            }
        )
        return {
            "archive_bytes": 5_000,
            "section_bytes": {"decoder_payload": 2_000},
            "section_rate_scores": {
                "decoder_payload": 2_000 * CONTEST_RATE_SCORE_PER_BYTE,
            },
            "lf_payload": 1_500,
            "aux_section": 100,
            "rate_score_per_byte": 2.0 * CONTEST_RATE_SCORE_PER_BYTE,
            "train_time_section_rate_score__lf_payload": (
                1_500 * CONTEST_RATE_SCORE_PER_BYTE
            ),
            "metadata": {"refreshed_this_step": True},
            "schema": "unit_test_section_metrics.v1",
        }

    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        train_time_section_byte_metrics=_section_byte_metrics,
    )
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        train_time_dual_ascent_config={
            "enabled": True,
            "constraints": [
                {
                    "constraint_id": "archive_total_bytes",
                    "metric_name": "train_time_archive_rate_score",
                    "loss_weight_key": "coder_qat_c1a_entropy",
                    "target": 2_000 * CONTEST_RATE_SCORE_PER_BYTE,
                    "dual_lr": 1.0,
                    "max_lambda": 2.0,
                },
                {
                    "constraint_id": "decoder_payload_bytes",
                    "metric_name": (
                        "train_time_section_rate_score__decoder_payload"
                    ),
                    "loss_weight_key": "coder_qat_c1a_entropy",
                    "target": 1_000 * CONTEST_RATE_SCORE_PER_BYTE,
                    "dual_lr": 1.0,
                    "max_lambda": 2.0,
                }
            ],
        },
    )
    batch = mx.array([0, 1], dtype=mx.int32)

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    assert callback_calls == [
        {
            "model_is_bundle_model": True,
            "batch_shape": (2,),
            "loss_weight_keys": [],
        }
    ]
    assert metrics["train_time_archive_bytes"] == pytest.approx(5_000)
    assert metrics["train_time_archive_rate_score"] == pytest.approx(
        10_000 * CONTEST_RATE_SCORE_PER_BYTE
    )
    assert metrics["train_time_section_bytes__aux_section"] == pytest.approx(100)
    assert metrics["train_time_section_rate_score__aux_section"] == pytest.approx(
        200 * CONTEST_RATE_SCORE_PER_BYTE
    )
    assert metrics["train_time_section_bytes__decoder_payload"] == pytest.approx(
        2_000
    )
    assert metrics["train_time_section_bytes__lf_payload"] == pytest.approx(1_500)
    assert metrics[
        "train_time_section_rate_score__decoder_payload"
    ] == pytest.approx(2_000 * CONTEST_RATE_SCORE_PER_BYTE)
    assert metrics["train_time_section_rate_score__lf_payload"] == pytest.approx(
        1_500 * CONTEST_RATE_SCORE_PER_BYTE
    )
    assert "train_time_section_bytes__rate_score_per_byte" not in metrics
    assert (
        "train_time_section_rate_score__train_time_section_rate_score__lf_payload"
        not in metrics
    )
    assert "train_time_section_bytes__metadata" not in metrics
    assert metrics["dual_ascent_metric__decoder_payload_bytes"] == pytest.approx(
        2_000 * CONTEST_RATE_SCORE_PER_BYTE
    )
    assert metrics["dual_ascent_metric__archive_total_bytes"] == pytest.approx(
        10_000 * CONTEST_RATE_SCORE_PER_BYTE
    )
    assert metrics["dual_ascent_lambda__archive_total_bytes"] == pytest.approx(
        8_000 * CONTEST_RATE_SCORE_PER_BYTE
    )
    assert metrics["dual_ascent_lambda__decoder_payload_bytes"] == pytest.approx(
        1_000 * CONTEST_RATE_SCORE_PER_BYTE
    )
    second_metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})
    assert callback_calls[-1] == {
        "model_is_bundle_model": True,
        "batch_shape": (2,),
        "loss_weight_keys": ["coder_qat_c1a_entropy"],
    }
    assert second_metrics["active_loss_weight__coder_qat_c1a_entropy"] > 0.0
    assert second_metrics["active_loss_weight_positive__coder_qat_c1a_entropy"] == (
        pytest.approx(1.0)
    )
    assert second_metrics[
        "dual_ascent_weight_applied__archive_total_bytes"
    ] == pytest.approx(1.0)
    assert second_metrics[
        "dual_ascent_effective_loss_weight__archive_total_bytes"
    ] > 0.0
    assert second_metrics[
        "dual_ascent_weight_applied__decoder_payload_bytes"
    ] == pytest.approx(1.0)
    metadata = adapter.artifact_metadata()["score_aware_training"][
        "train_time_section_byte_metrics"
    ]
    assert metadata["enabled"] is True
    assert metadata["source"] == "renderer_bundle_callback"
    assert metadata["last_metrics"]["train_time_section_bytes__decoder_payload"] == (
        pytest.approx(2_000)
    )


@mlx_only
def test_adapter_gradient_multiplier_reports_real_exact_leaf_actuation() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        gradient_multiplier_by_name={"rgb_1.bias": 0.25},
    )
    batch = mx.array([0, 1], dtype=mx.int32)

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})
    controls = adapter.artifact_metadata()["score_aware_training"][
        "gradient_multiplier_controls"
    ]

    assert metrics["gradient_multiplier_requested_control_count"] == pytest.approx(1)
    assert metrics["gradient_multiplier_applied_leaf_count"] >= 1
    assert metrics["gradient_multiplier_missing_exact_name_count"] == pytest.approx(0)
    assert metrics["gradient_multiplier_requested_but_unapplied"] == pytest.approx(0)
    assert controls["enabled"] is True
    assert controls["exact_active_name_count"] == 1
    assert controls["exact_active_names"] == ["rgb_1.bias"]
    leaf_inventory = controls["leaf_inventory"]
    assert leaf_inventory["schema"] == (
        "mlx_score_aware_gradient_multiplier_leaf_inventory.v1"
    )
    assert "rgb_1.bias" in leaf_inventory["all_leaf_names"]["names"]
    assert "rgb_1.bias" in leaf_inventory["bias_leaf_names"]["names"]
    assert "rgb_1.bias" in leaf_inventory["output_head_bias_candidate_names"]["names"]
    assert leaf_inventory["all_leaf_names"]["count"] >= len(
        leaf_inventory["all_leaf_names"]["names"]
    )
    assert len(leaf_inventory["all_leaf_names"]["names_sha256"]) == 64


@mlx_only
def test_adapter_gradient_multiplier_reports_stale_exact_leaf_noop() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        gradient_multiplier_by_name={"missing.decoder.weight": 0.0},
    )
    batch = mx.array([0, 1], dtype=mx.int32)

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})
    controls = adapter.artifact_metadata()["score_aware_training"][
        "gradient_multiplier_controls"
    ]

    assert metrics["gradient_multiplier_requested_control_count"] == pytest.approx(1)
    assert metrics["gradient_multiplier_applied_leaf_count"] == pytest.approx(0)
    assert metrics["gradient_multiplier_missing_exact_name_count"] == pytest.approx(1)
    assert metrics["gradient_multiplier_missing_requested_count"] == pytest.approx(1)
    assert metrics["gradient_multiplier_requested_but_unapplied"] == pytest.approx(1)
    assert controls["exact_active_names"] == ["missing.decoder.weight"]
    assert "missing.decoder.weight" not in controls["leaf_inventory"]["all_leaf_names"][
        "names"
    ]
    assert controls["leaf_inventory"]["all_leaf_names"]["count"] > 0


@mlx_only
def test_adapter_scorer_space_step_guard_restores_collapsing_step() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_max_post_segnet_contrast_ratio=2.0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.53,
                "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 1.0,
            }
        if call_count == 2:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.2,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.51,
                "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 4.5,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.53,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 1.0,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]
    batch = mx.array([0, 1], dtype=mx.int32)
    before = {
        str(name): mx.array(value)
        for name, value in tree_flatten(adapter.model.parameters())
    }

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    after = {str(name): value for name, value in tree_flatten(adapter.model.parameters())}
    max_delta = 0.0
    for name, before_value in before.items():
        delta = mx.max(mx.abs(after[name] - before_value))
        mx.eval(delta)
        max_delta = max(max_delta, float(delta.item()))

    assert metrics["scorer_space_step_guard_enabled"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_eligible"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_parameters_restored"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_optimizer_state_restored"] == pytest.approx(0.0)
    assert metrics["scorer_space_step_guard_optimizer_state_advanced"] == pytest.approx(1.0)
    assert (
        metrics[
            "scorer_space_step_guard_reject_reason_post_segnet_occupied_class_fraction_below_floor"
        ]
        == pytest.approx(1.0)
    )
    assert (
        metrics[
            "scorer_space_step_guard_reject_reason_post_segnet_contrast_ratio_above_ceiling"
        ]
        == pytest.approx(1.0)
    )
    assert metrics["scorer_space_step_guard_student_heads_skipped"] == pytest.approx(0.0)
    assert (
        metrics["scorer_space_step_guard_student_heads_trained_after_reject"]
        == pytest.approx(0.0)
    )
    assert metrics["scorer_space_step_guard_persistent_lr_feedback_scale"] == (
        pytest.approx(1.0)
    )
    assert (
        metrics["scorer_space_step_guard_persistent_lr_feedback_source_step_scale"]
        == pytest.approx(1.0)
    )
    assert metrics["dynamics_param_delta_all_l2"] == pytest.approx(0.0, abs=1e-7)
    assert max_delta == pytest.approx(0.0, abs=1e-7)

    guard = adapter.artifact_metadata()["score_aware_training"][
        "scorer_space_step_guard"
    ]
    assert guard["enabled"] is True
    assert guard["rollback_contract"]["parameters_restored_on_reject"] is True
    assert guard["rollback_contract"]["optimizer_state_restored_on_reject"] is False


@mlx_only
def test_adapter_scorer_space_step_guard_accepts_damped_backtracking_step() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_max_post_segnet_contrast_ratio=2.0,
        scorer_space_step_guard_backtracking_steps=3,
        scorer_space_step_guard_backtracking_shrink=0.5,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
                "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 1.0,
            }
        if call_count == 2:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.2,
                "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 4.5,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 1.5,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]
    batch = mx.array([0, 1], dtype=mx.int32)
    before = {
        str(name): mx.array(value)
        for name, value in tree_flatten(adapter.model.parameters())
    }

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    after = {str(name): value for name, value in tree_flatten(adapter.model.parameters())}
    delta_sq = mx.array(0.0, dtype=mx.float32)
    for name, before_value in before.items():
        diff = after[name] - before_value
        delta_sq = delta_sq + mx.sum(diff * diff)
    delta = mx.sqrt(delta_sq)
    mx.eval(delta)

    assert metrics["scorer_space_step_guard_enabled"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_intervened"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(0.0)
    assert metrics["scorer_space_step_guard_backtracking_accepted"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_parameters_damped"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_parameters_restored"] == pytest.approx(0.0)
    assert metrics["scorer_space_step_guard_backtracking_attempt_count"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_accepted_step_scale"] == pytest.approx(0.5)
    assert metrics["scorer_space_step_guard_accepted_step_scale"] == pytest.approx(0.5)
    assert metrics["scorer_space_step_guard_persistent_lr_feedback_scale"] == (
        pytest.approx(0.5**0.5)
    )
    assert (
        metrics["scorer_space_step_guard_persistent_lr_feedback_source_step_scale"]
        == pytest.approx(0.5)
    )
    assert metrics[
        "scorer_space_step_guard_persistent_lr_feedback_recovery_floor"
    ] == pytest.approx(0.0)
    assert metrics["scorer_space_step_guard_learning_rate_scale_after"] == (
        pytest.approx(0.5**0.5)
    )
    assert (
        metrics[
            "scorer_space_step_guard_learning_rate_scale_update_reason_backtracking_accept"
        ]
        == pytest.approx(1.0)
    )
    assert adapter._effective_wave_n11_learning_rate(1e-2) == pytest.approx(
        (0.5**0.5) * 1e-2
    )
    assert metrics["scorer_space_step_guard_student_heads_skipped"] == pytest.approx(0.0)
    assert float(delta.item()) > 0.0


@mlx_only
def test_adapter_scorer_space_step_guard_reject_uses_gentle_persistent_damping() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_max_post_segnet_contrast_ratio=2.0,
        scorer_space_step_guard_backtracking_steps=3,
        scorer_space_step_guard_backtracking_shrink=0.5,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
                "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 1.0,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.2,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 4.5,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]
    batch = mx.array([0, 1], dtype=mx.int32)

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_parameters_restored"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_backtracking_attempt_count"] == pytest.approx(
        3.0
    )
    assert (
        metrics["scorer_space_step_guard_persistent_lr_feedback_source_step_scale"]
        == pytest.approx(0.125)
    )
    assert metrics["scorer_space_step_guard_persistent_lr_feedback_scale"] == (
        pytest.approx(0.5)
    )
    assert metrics["scorer_space_step_guard_learning_rate_scale_after"] == pytest.approx(
        0.5
    )


@mlx_only
def test_adapter_scorer_space_step_guard_blocks_low_occupancy_contrast_crossing() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_max_post_segnet_contrast_ratio=2.0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.2,
                "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 1.2,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.2,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 4.5,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]
    batch = mx.array([0, 1], dtype=mx.int32)
    before = {
        str(name): mx.array(value)
        for name, value in tree_flatten(adapter.model.parameters())
    }

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    after = {str(name): value for name, value in tree_flatten(adapter.model.parameters())}
    max_delta = 0.0
    for name, before_value in before.items():
        delta = mx.max(mx.abs(after[name] - before_value))
        mx.eval(delta)
        max_delta = max(max_delta, float(delta.item()))

    assert metrics["scorer_space_step_guard_eligible"] == pytest.approx(1.0)
    assert (
        metrics["scorer_space_step_guard_eligible_pre_noncollapsed"]
        == pytest.approx(0.0)
    )
    assert (
        metrics["scorer_space_step_guard_eligible_contrast_crossed_ceiling"]
        == pytest.approx(1.0)
    )
    assert (
        metrics["scorer_space_step_guard_eligible_low_occupancy_non_improving"]
        == pytest.approx(1.0)
    )
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_parameters_restored"] == pytest.approx(1.0)
    assert max_delta == pytest.approx(0.0, abs=1e-7)


@mlx_only
def test_adapter_scorer_space_step_guard_rejects_target_class_loss() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=0.8,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.6,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.45,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.6,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.6,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.44,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]
    batch = mx.array([0, 1], dtype=mx.int32)
    before = {
        str(name): mx.array(value)
        for name, value in tree_flatten(adapter.model.parameters())
    }

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    after = {str(name): value for name, value in tree_flatten(adapter.model.parameters())}
    max_delta = 0.0
    for name, before_value in before.items():
        delta = mx.max(mx.abs(after[name] - before_value))
        mx.eval(delta)
        max_delta = max(max_delta, float(delta.item()))

    assert metrics["scorer_space_step_guard_eligible"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_pre_segnet_occupied_class_fraction"] == (
        pytest.approx(0.6)
    )
    assert (
        metrics["scorer_space_step_guard_pre_segnet_target_class_coverage_fraction"]
        == pytest.approx(0.8)
    )
    assert (
        metrics[
            "scorer_space_step_guard_post_segnet_target_class_coverage_fraction_before_restore"
        ]
        == pytest.approx(0.6)
    )
    assert (
        metrics[
            "scorer_space_step_guard_reject_reason_post_segnet_target_class_coverage_fraction_below_floor"
        ]
        == pytest.approx(1.0)
    )
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_parameters_restored"] == pytest.approx(1.0)
    assert max_delta == pytest.approx(0.0, abs=1e-7)


@mlx_only
def test_adapter_scorer_space_step_guard_calibrates_head_after_reject() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        build_learnable_student_head,
    )

    class _LiveSegTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            b = int(idx.shape[0])
            zeros = mx.zeros((b, 384, 512, self.num_classes), dtype=mx.float32)
            target = mx.zeros((b, 384, 512, 1), dtype=mx.float32) + 3.0
            return zeros + mx.concatenate(
                [
                    mx.zeros((b, 384, 512, 2), dtype=mx.float32),
                    target,
                    mx.zeros((b, 384, 512, 2), dtype=mx.float32),
                ],
                axis=-1,
            )

        def teacher_logits_for_frames_nhwc01(self, frames):
            mean = mx.mean(frames, axis=-1, keepdims=True)
            return mx.concatenate(
                [mean * float(i + 1) for i in range(self.num_classes)],
                axis=-1,
            )

    base = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    head = build_learnable_student_head(num_classes=5, seed=7)
    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        distillation_weight=0.5,
        scorer_teacher=_LiveSegTeacher(),
        learnable_student_head=head,
        segnet_student_live_calibration_weight=1.0,
        allow_segnet_only_research=True,
    )
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_max_post_segnet_contrast_ratio=2.0,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.53,
                "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 1.0,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.2,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.51,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 4.5,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]
    batch = mx.array([0, 1], dtype=mx.int32)
    renderer_before = {
        str(name): mx.array(value)
        for name, value in tree_flatten(adapter.model.parameters())
    }
    head_before = mx.array(head.weight)

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    renderer_after = {
        str(name): value for name, value in tree_flatten(adapter.model.parameters())
    }
    renderer_max_delta = 0.0
    for name, before_value in renderer_before.items():
        delta = mx.max(mx.abs(renderer_after[name] - before_value))
        mx.eval(delta)
        renderer_max_delta = max(renderer_max_delta, float(delta.item()))
    head_delta = mx.max(mx.abs(head.weight - head_before))
    mx.eval(head_delta)

    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_parameters_restored"] == pytest.approx(1.0)
    assert renderer_max_delta == pytest.approx(0.0, abs=1e-7)
    assert metrics["scorer_space_step_guard_student_heads_skipped"] == pytest.approx(0.0)
    assert (
        metrics["scorer_space_step_guard_student_heads_trained_after_reject"]
        == pytest.approx(1.0)
    )
    assert metrics["segnet_student_head_update_active"] == pytest.approx(1.0)
    assert metrics["segnet_student_live_calibration_active"] == pytest.approx(1.0)
    assert metrics["loss_part_segnet_student_live_calibration"] >= 0.0
    assert float(head_delta.item()) > 0.0


@mlx_only
def test_adapter_scorer_space_step_guard_accepts_bootstrap_escape_step() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.400001,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.400001,
        scorer_space_step_guard_max_post_segnet_contrast_ratio=2.0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.60,
                "loss_part_segnet_direct_live_escape_selection": 6.60,
                "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 1.0,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.54,
            "loss_part_segnet_direct_live_escape_selection": 6.54,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 1.2,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]
    batch = mx.array([0, 1], dtype=mx.int32)
    before = {
        str(name): mx.array(value)
        for name, value in tree_flatten(adapter.model.parameters())
    }

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    after = {str(name): value for name, value in tree_flatten(adapter.model.parameters())}
    delta_sq = mx.array(0.0, dtype=mx.float32)
    for name, before_value in before.items():
        diff = after[name] - before_value
        delta_sq = delta_sq + mx.sum(diff * diff)
    delta = mx.sqrt(delta_sq)
    mx.eval(delta)

    assert metrics["scorer_space_step_guard_eligible"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_bootstrap_escape_allowed"] == pytest.approx(
        1.0
    )
    assert metrics["scorer_space_step_guard_bootstrap_escape_improved"] == pytest.approx(
        1.0
    )
    assert (
        metrics["scorer_space_step_guard_bootstrap_argmax_improved_meaningfully"]
        == pytest.approx(1.0)
    )
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(0.0)
    assert (
        metrics[
            "scorer_space_step_guard_reject_reason_post_segnet_occupied_class_fraction_below_floor"
        ]
        == pytest.approx(0.0)
    )
    assert (
        metrics[
            "scorer_space_step_guard_reject_reason_post_segnet_occupied_class_fraction_below_floor_suppressed_by_bootstrap_escape"
        ]
        == pytest.approx(1.0)
    )
    assert metrics["scorer_space_step_guard_segnet_escape_delta"] == pytest.approx(
        -0.06
    )
    assert float(delta.item()) > 0.0


@mlx_only
def test_adapter_scorer_space_step_guard_rejects_tiny_argmax_twitch() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.400001,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.400001,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.506866,
                "loss_part_segnet_direct_live_escape_selection": 6.506866,
                "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 1.0,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.506861,
            "loss_part_segnet_direct_live_escape_selection": 6.506861,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 1.0,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]
    batch = mx.array([0, 1], dtype=mx.int32)
    before = {
        str(name): mx.array(value)
        for name, value in tree_flatten(adapter.model.parameters())
    }

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    after = {str(name): value for name, value in tree_flatten(adapter.model.parameters())}
    max_delta = 0.0
    for name, before_value in before.items():
        delta = mx.max(mx.abs(after[name] - before_value))
        mx.eval(delta)
        max_delta = max(max_delta, float(delta.item()))

    assert metrics["scorer_space_step_guard_eligible"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_bootstrap_argmax_improved"] == pytest.approx(
        1.0
    )
    assert (
        metrics["scorer_space_step_guard_bootstrap_argmax_improved_meaningfully"]
        == pytest.approx(0.0)
    )
    assert (
        metrics["scorer_space_step_guard_bootstrap_escape_improved_meaningfully"]
        == pytest.approx(0.0)
    )
    assert metrics["scorer_space_step_guard_bootstrap_escape_allowed"] == pytest.approx(
        0.0
    )
    assert (
        metrics["scorer_space_step_guard_ceiling_recovery_argmax_allowed"]
        == pytest.approx(0.0)
    )
    assert metrics["scorer_space_step_guard_argmax_recovery"] == pytest.approx(5e-6)
    assert metrics["scorer_space_step_guard_min_argmax_recovery"] > 1e-4
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(1.0)
    assert max_delta == pytest.approx(0.0, abs=1e-7)


@mlx_only
def test_adapter_scorer_space_step_guard_rejects_worse_bootstrap_escape_step() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.400001,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.400001,
        scorer_space_step_guard_max_post_segnet_contrast_ratio=2.0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.54,
                "loss_part_segnet_direct_live_escape_selection": 6.54,
                "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 1.0,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.60,
            "loss_part_segnet_direct_live_escape_selection": 6.60,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 1.2,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]
    batch = mx.array([0, 1], dtype=mx.int32)
    before = {
        str(name): mx.array(value)
        for name, value in tree_flatten(adapter.model.parameters())
    }

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    after = {str(name): value for name, value in tree_flatten(adapter.model.parameters())}
    max_delta = 0.0
    for name, before_value in before.items():
        delta = mx.max(mx.abs(after[name] - before_value))
        mx.eval(delta)
        max_delta = max(max_delta, float(delta.item()))

    assert metrics["scorer_space_step_guard_eligible"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_bootstrap_escape_allowed"] == pytest.approx(
        0.0
    )
    assert metrics["scorer_space_step_guard_bootstrap_escape_improved"] == pytest.approx(
        0.0
    )
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(1.0)
    assert (
        metrics[
            "scorer_space_step_guard_reject_reason_post_segnet_occupied_class_fraction_below_floor"
        ]
        == pytest.approx(1.0)
    )
    assert max_delta == pytest.approx(0.0, abs=1e-7)


@mlx_only
def test_adapter_scorer_space_step_guard_accepts_argmax_ceiling_recovery() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.6,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.63,
                "loss_part_segnet_direct_live_escape_selection": 4.63,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.6,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.52,
            "loss_part_segnet_direct_live_escape_selection": 4.52,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]
    batch = mx.array([0, 1], dtype=mx.int32)
    before = {
        str(name): mx.array(value)
        for name, value in tree_flatten(adapter.model.parameters())
    }

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    after = {str(name): value for name, value in tree_flatten(adapter.model.parameters())}
    delta_sq = mx.array(0.0, dtype=mx.float32)
    for name, before_value in before.items():
        diff = after[name] - before_value
        delta_sq = delta_sq + mx.sum(diff * diff)
    delta = mx.sqrt(delta_sq)
    mx.eval(delta)

    assert metrics["scorer_space_step_guard_eligible"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(0.0)
    assert metrics["scorer_space_step_guard_reject_reason_count"] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_ceiling_recovery_argmax_allowed"
    ] == pytest.approx(1.0)
    assert (
        metrics[
            "scorer_space_step_guard_reject_reason_post_segnet_argmax_disagreement_above_ceiling"
        ]
        == pytest.approx(0.0)
    )
    assert float(delta.item()) > 0.0


@mlx_only
def test_adapter_scorer_space_step_guard_rejects_argmax_and_pose_ceiling_crossing() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
        scorer_space_step_guard_max_post_pose_score_term=4.0,
        scorer_space_step_guard_max_post_pose_direct_live_score_term=3.0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.6,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.2,
                "loss_part_pose_score_term": 1.0,
                "loss_part_pose_direct_live_score_term": 1.0,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.6,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.9,
            "loss_part_pose_score_term": 9.0,
            "loss_part_pose_direct_live_score_term": 8.0,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]
    batch = mx.array([0, 1], dtype=mx.int32)
    before = {
        str(name): mx.array(value)
        for name, value in tree_flatten(adapter.model.parameters())
    }

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    after = {str(name): value for name, value in tree_flatten(adapter.model.parameters())}
    max_delta = 0.0
    for name, before_value in before.items():
        delta = mx.max(mx.abs(after[name] - before_value))
        mx.eval(delta)
        max_delta = max(max_delta, float(delta.item()))

    assert metrics["scorer_space_step_guard_eligible"] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_eligible_argmax_crossed_ceiling"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_eligible_pose_crossed_ceiling"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_eligible_pose_direct_live_crossed_ceiling"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_segnet_argmax_disagreement_above_ceiling"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_pose_score_term_above_ceiling"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_pose_direct_live_score_term_above_ceiling"
    ] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_parameters_restored"] == pytest.approx(1.0)
    assert max_delta == pytest.approx(0.0, abs=1e-7)


@mlx_only
def test_adapter_scorer_space_step_guard_rejects_pose_worsening_without_ceiling() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening=0.0,
        scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening=0.0,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.2,
                "loss_part_pose_direct_live_score_term": 1.0,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.2,
            "loss_part_pose_direct_live_score_term": 1.25,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]
    batch = mx.array([0, 1], dtype=mx.int32)
    before = {
        str(name): mx.array(value)
        for name, value in tree_flatten(adapter.model.parameters())
    }

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    after = {str(name): value for name, value in tree_flatten(adapter.model.parameters())}
    max_delta = 0.0
    for name, before_value in before.items():
        delta = mx.max(mx.abs(after[name] - before_value))
        mx.eval(delta)
        max_delta = max(max_delta, float(delta.item()))

    assert metrics["scorer_space_step_guard_eligible"] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_eligible_pose_direct_live_worsened"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_pose_direct_live_score_term_worsened"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_direct_nonrate_score_worsened"
    ] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_direct_nonrate_score_worsening_delta"] == (
        pytest.approx(0.25)
    )
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_parameters_restored"] == pytest.approx(1.0)
    assert max_delta == pytest.approx(0.0, abs=1e-7)


@mlx_only
def test_adapter_scorer_space_step_guard_rejects_distribution_escape_blowup() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.8,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=0.8,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.2,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.2,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.50,
                "loss_part_segnet_direct_live_escape_selection": 8.50,
                "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.22,
                "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.17,
                "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 2.9,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.45,
            "loss_part_segnet_direct_live_escape_selection": 2.45,
            "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.33,
            "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.23,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 3.8,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics["scorer_space_step_guard_eligible"] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_segnet_distribution_mae_above_ceiling"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_posenet_yuv6_distribution_mae_above_ceiling"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_posenet_yuv6_contrast_ratio_above_ceiling"
    ] == pytest.approx(1.0)


@mlx_only
def test_adapter_scorer_space_step_guard_accepts_bounded_missing_class_bootstrap() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=0.8,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.001,
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=5.0,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.6,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.6,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.52,
                "loss_part_pose_direct_live_score_term": 44.0,
                "loss_part_pose_score_term": 100.0,
                "loss_part_segnet_direct_live_rare_class_logit_loss": 2000.0,
                "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.11,
                "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.08,
                "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.4,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.7,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.7,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.56,
                "loss_part_pose_direct_live_score_term": 43.8,
                "loss_part_pose_score_term": 103.0,
                "loss_part_segnet_direct_live_rare_class_logit_loss": 1990.0,
                "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.12,
                "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.09,
                "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.5,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics["scorer_space_step_guard_eligible"] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_bootstrap_direct_nonrate_worsening_allowed"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_target_class_structural_recovery"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_rare_class_logit_recovered_meaningfully"
    ] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_segnet_target_class_coverage_fraction_below_floor"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_segnet_argmax_disagreement_above_ceiling"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_direct_nonrate_score_worsened_blocking"
    ] == pytest.approx(0.0)


@mlx_only
def test_adapter_scorer_space_step_guard_accepts_priced_target_coverage_breakthrough() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=1.0,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.001,
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=15.0,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.3715,
                "loss_part_pose_direct_live_score_term": 44.588,
                "loss_part_pose_score_term": 185.44,
                "loss_part_segnet_direct_live_rare_class_logit_loss": 4003.0,
                "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.105,
                "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.075,
                "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.18,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 1.0,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 1.0,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.499,
            "loss_part_pose_direct_live_score_term": 44.489,
            "loss_part_pose_score_term": 188.85,
            "loss_part_segnet_direct_live_rare_class_logit_loss": 996.0,
            "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.111,
            "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.080,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.27,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics[
        "scorer_space_step_guard_target_class_coverage_breakthrough"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_target_class_argmax_trade_priced"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_bootstrap_direct_nonrate_worsening_allowed"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_target_class_bootstrap_escape_allowed"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_target_class_structural_recovery"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_segnet_argmax_disagreement_above_ceiling"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_direct_nonrate_score_worsened"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_pose_score_term_worsened_blocking"
    ] == pytest.approx(0.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(0.0)


@mlx_only
def test_adapter_scorer_space_step_guard_rejects_unpriced_target_coverage_breakthrough() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=1.0,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.001,
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=15.0,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.3715,
                "loss_part_pose_direct_live_score_term": 44.588,
                "loss_part_pose_score_term": 185.44,
                "loss_part_segnet_direct_live_rare_class_logit_loss": 4003.0,
                "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.105,
                "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.075,
                "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.18,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 1.0,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 1.0,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.5056,
            "loss_part_pose_direct_live_score_term": 44.489,
            "loss_part_pose_score_term": 188.85,
            "loss_part_segnet_direct_live_rare_class_logit_loss": 996.0,
            "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.111,
            "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.080,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.27,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics[
        "scorer_space_step_guard_target_class_coverage_breakthrough"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_target_class_argmax_trade_priced"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_bootstrap_direct_nonrate_worsening_allowed"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_segnet_argmax_disagreement_above_ceiling"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_direct_nonrate_score_worsened"
    ] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(1.0)


@mlx_only
def test_adapter_scorer_space_step_guard_accepts_bounded_target_class_escape_argmax_ceiling() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.2,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=1.0,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.001,
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=5.0,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.2,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.2,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.50692,
                "loss_part_segnet_direct_live_rare_class_logit_loss": 36248.0,
                "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
                "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
                "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.4,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.50756,
            "loss_part_segnet_direct_live_rare_class_logit_loss": 18422.0,
            "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
            "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics[
        "scorer_space_step_guard_target_class_bootstrap_escape_allowed"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_target_class_argmax_ceiling_suppression_allowed"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_segnet_argmax_disagreement_above_ceiling"
    ] == pytest.approx(0.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(0.0)


@mlx_only
def test_adapter_scorer_space_step_guard_accepts_priced_class_birth_without_min_ratio_move() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.2,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=1.0,
        scorer_space_step_guard_min_post_segnet_target_class_min_ratio=0.2,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.001,
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=5.0,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.2,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.2,
                "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.0,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.5069,
                "loss_part_pose_direct_live_score_term": 41.8,
                "loss_part_pose_score_term": 194.0,
                "loss_part_segnet_direct_live_rare_class_logit_loss": 44185.0,
                "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
                "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
                "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.4,
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.0,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.5090,
            "loss_part_pose_direct_live_score_term": 45.2,
            "loss_part_pose_score_term": 314.0,
            "loss_part_segnet_direct_live_rare_class_logit_loss": 84170.0,
            "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
            "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics[
        "scorer_space_step_guard_target_class_structural_recovery"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_target_class_min_ratio_birth_allowed"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_bootstrap_direct_nonrate_worsening_allowed"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_segnet_target_class_min_ratio_below_floor"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_pose_direct_live_score_term_worsened"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_direct_nonrate_score_worsened_blocking"
    ] == pytest.approx(0.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(0.0)


@mlx_only
def test_adapter_scorer_space_step_guard_accepts_priced_min_ratio_recovery() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.2,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=0.8,
        scorer_space_step_guard_min_post_segnet_target_class_min_ratio=0.2,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.6,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.001,
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=5.0,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        common = {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "loss_part_segnet_direct_live_target_class_0_fraction": 0.2,
            "loss_part_segnet_direct_live_target_class_1_fraction": 0.2,
            "loss_part_segnet_direct_live_target_class_2_fraction": 0.2,
            "loss_part_segnet_direct_live_target_class_3_fraction": 0.2,
            "loss_part_segnet_direct_live_target_class_4_fraction": 0.2,
            "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
            "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
        }
        if call_count == 1:
            return {
                **common,
                "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.0,
                "loss_part_segnet_direct_live_candidate_target_class_0_ratio": 1.0,
                "loss_part_segnet_direct_live_candidate_target_class_1_ratio": 0.0,
                "loss_part_segnet_direct_live_candidate_target_class_2_ratio": 0.0,
                "loss_part_segnet_direct_live_candidate_target_class_3_ratio": 0.0,
                "loss_part_segnet_direct_live_candidate_target_class_4_ratio": 0.0,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.50,
                "loss_part_pose_direct_live_score_term": 40.0,
                "loss_part_pose_score_term": 190.0,
            }
        return {
            **common,
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.05,
            "loss_part_segnet_direct_live_candidate_target_class_0_ratio": 1.0,
            "loss_part_segnet_direct_live_candidate_target_class_1_ratio": 0.05,
            "loss_part_segnet_direct_live_candidate_target_class_2_ratio": 0.0,
            "loss_part_segnet_direct_live_candidate_target_class_3_ratio": 0.0,
            "loss_part_segnet_direct_live_candidate_target_class_4_ratio": 0.0,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.51,
            "loss_part_pose_direct_live_score_term": 41.0,
            "loss_part_pose_score_term": 191.0,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics[
        "scorer_space_step_guard_target_class_min_ratio_recovery"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_target_class_min_ratio_support_credit_eligible"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_target_class_min_ratio_support_nonrate_credit"
    ] == pytest.approx(5.0)
    assert metrics[
        "scorer_space_step_guard_bootstrap_direct_nonrate_worsening_allowed"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_segnet_target_class_min_ratio_below_floor"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_direct_nonrate_score_worsened"
    ] == pytest.approx(0.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(0.0)


@mlx_only
def test_adapter_scorer_space_step_guard_accepts_scorer_priced_class_support_credit() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.2,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=1.0,
        scorer_space_step_guard_min_post_segnet_target_class_min_ratio=0.2,
        scorer_space_step_guard_max_post_segnet_target_class_ratio_drop=0.05,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.001,
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=5.0,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        common = {
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.0,
            "loss_part_segnet_direct_live_target_class_0_fraction": 0.2,
            "loss_part_segnet_direct_live_target_class_1_fraction": 0.2,
            "loss_part_segnet_direct_live_target_class_2_fraction": 0.2,
            "loss_part_segnet_direct_live_target_class_3_fraction": 0.2,
            "loss_part_segnet_direct_live_target_class_4_fraction": 0.2,
            "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
            "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
        }
        if call_count == 1:
            return {
                **common,
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.2,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.2,
                "loss_part_segnet_direct_live_candidate_target_class_0_ratio": 1.0,
                "loss_part_segnet_direct_live_candidate_target_class_1_ratio": 0.0,
                "loss_part_segnet_direct_live_candidate_target_class_2_ratio": 0.0,
                "loss_part_segnet_direct_live_candidate_target_class_3_ratio": 0.0,
                "loss_part_segnet_direct_live_candidate_target_class_4_ratio": 0.0,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.5069,
                "loss_part_pose_direct_live_score_term": 41.8,
                "loss_part_pose_score_term": 194.0,
            }
        return {
            **common,
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_0_ratio": 1.0,
            "loss_part_segnet_direct_live_candidate_target_class_1_ratio": 0.0,
            "loss_part_segnet_direct_live_candidate_target_class_2_ratio": 0.0,
            "loss_part_segnet_direct_live_candidate_target_class_3_ratio": 0.0,
            "loss_part_segnet_direct_live_candidate_target_class_4_ratio": 0.0,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.745,
            "loss_part_pose_direct_live_score_term": 45.2,
            "loss_part_pose_score_term": 314.0,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics[
        "scorer_space_step_guard_target_class_support_credit_eligible"
    ] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_target_class_coverage_delta"] == (
        pytest.approx(0.6)
    )
    assert metrics[
        "scorer_space_step_guard_target_class_support_nonrate_credit"
    ] == pytest.approx(60.0)
    assert metrics[
        "scorer_space_step_guard_bootstrap_direct_nonrate_worsening_budget"
    ] == pytest.approx(65.0)
    assert metrics[
        "scorer_space_step_guard_bootstrap_direct_nonrate_worsening_allowed"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_segnet_argmax_disagreement_above_ceiling"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_direct_nonrate_score_worsened"
    ] == pytest.approx(0.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(0.0)


@mlx_only
def test_adapter_scorer_space_step_guard_backtracking_accepts_bounded_target_class_escape() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.2,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=1.0,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.001,
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=5.0,
        scorer_space_step_guard_backtracking_steps=2,
        scorer_space_step_guard_backtracking_shrink=0.5,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.2,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.2,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.50692,
                "loss_part_segnet_direct_live_rare_class_logit_loss": 36248.0,
                "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
                "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
                "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
            }
        if call_count == 2:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.4,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.52,
                "loss_part_segnet_direct_live_rare_class_logit_loss": 18422.0,
                "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
                "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
                "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.4,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.50756,
            "loss_part_segnet_direct_live_rare_class_logit_loss": 18422.0,
            "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
            "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics["scorer_space_step_guard_backtracking_accepted"] == pytest.approx(
        1.0
    )
    assert metrics[
        "scorer_space_step_guard_backtracking_last_target_class_argmax_ceiling_suppression_allowed"
    ] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_accepted_step_scale"] == pytest.approx(0.5)
    assert metrics[
        "scorer_space_step_guard_persistent_lr_feedback_source_step_scale"
    ] == pytest.approx(0.5)
    assert metrics[
        "scorer_space_step_guard_persistent_lr_feedback_recovery_floor"
    ] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_persistent_lr_feedback_scale"] == (
        pytest.approx(1.0)
    )
    assert metrics["scorer_space_step_guard_learning_rate_scale_after"] == (
        pytest.approx(1.0)
    )
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(0.0)


@mlx_only
def test_adapter_target_class_soft_escape_keeps_persistent_lr_damped_until_hard_coverage_moves() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.2,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=1.0,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.8,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.001,
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=5.0,
        scorer_space_step_guard_backtracking_steps=2,
        scorer_space_step_guard_backtracking_shrink=0.5,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.4,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.61,
                "loss_part_segnet_direct_live_rare_class_logit_loss": 54281.0,
                "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
                "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
                "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
            }
        if call_count == 2:
            return {
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.2,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.2,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.51,
                "loss_part_segnet_direct_live_rare_class_logit_loss": 23892.0,
                "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
                "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
                "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
            }
        return {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.4,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.405,
            "loss_part_segnet_direct_live_rare_class_logit_loss": 21393.0,
            "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
            "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics["scorer_space_step_guard_backtracking_accepted"] == pytest.approx(
        1.0
    )
    assert metrics[
        "scorer_space_step_guard_backtracking_last_target_class_bootstrap_escape_allowed"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_backtracking_last_target_class_coverage_improved"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_backtracking_last_rare_class_logit_recovered_meaningfully"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_persistent_lr_feedback_recovery_floor"
    ] == pytest.approx(0.0)
    assert metrics["scorer_space_step_guard_persistent_lr_feedback_scale"] == (
        pytest.approx(0.5**0.5)
    )
    assert metrics["scorer_space_step_guard_learning_rate_scale_after"] == (
        pytest.approx(0.5**0.5)
    )
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(0.0)


@mlx_only
def test_adapter_scorer_space_step_guard_rejects_unbounded_argmax_ceiling_during_target_class_bootstrap() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.2,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=1.0,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.001,
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=5.0,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        base = {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.2,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.2,
            "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
            "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
        }
        if call_count == 1:
            return {
                **base,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.5075,
                "loss_part_segnet_direct_live_rare_class_logit_loss": 36248.0,
            }
        return {
            **base,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.52,
            "loss_part_segnet_direct_live_rare_class_logit_loss": 18422.0,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics[
        "scorer_space_step_guard_target_class_bootstrap_escape_allowed"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_target_class_structural_recovery"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_bootstrap_direct_nonrate_worsening_allowed"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_target_class_argmax_ceiling_suppression_allowed"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_segnet_argmax_disagreement_above_ceiling"
    ] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(1.0)


@mlx_only
def test_adapter_scorer_space_step_guard_accepts_within_ceiling_rare_recovery_trade() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=1.0,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.001,
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=5.0,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        common = {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "loss_part_pose_direct_live_score_term": 40.0,
            "loss_part_pose_score_term": 55.0,
            "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
            "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
        }
        if call_count == 1:
            return {
                **common,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.406,
                "loss_part_segnet_direct_live_rare_class_logit_loss": 55104.0,
            }
        return {
            **common,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.422,
            "loss_part_segnet_direct_live_rare_class_logit_loss": 51304.0,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics[
        "scorer_space_step_guard_target_class_argmax_within_ceiling"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_target_class_argmax_trade_priced"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_bootstrap_direct_nonrate_worsening_allowed"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_direct_nonrate_score_worsened"
    ] == pytest.approx(0.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(0.0)


@mlx_only
def test_adapter_scorer_space_step_guard_accepts_preserved_rare_class_min_ratio_bootstrap() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=1.0,
        scorer_space_step_guard_min_post_segnet_target_class_min_ratio=0.2,
        scorer_space_step_guard_max_post_segnet_target_class_ratio_drop=0.05,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.001,
        scorer_space_step_guard_backtracking_steps=1,
        scorer_space_step_guard_backtracking_shrink=0.5,
    )
    call_count = 0

    stable_support = {
        "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
        "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
        "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.0,
        "loss_part_segnet_direct_live_candidate_target_class_0_ratio": 0.5,
        "loss_part_segnet_direct_live_candidate_target_class_1_ratio": 0.0,
        "loss_part_segnet_direct_live_candidate_target_class_2_ratio": 1.0,
        "loss_part_segnet_direct_live_candidate_target_class_3_ratio": 0.25,
        "loss_part_segnet_direct_live_candidate_target_class_4_ratio": 0.25,
        "loss_part_pose_direct_live_score_term": 40.0,
        "loss_part_pose_score_term": 55.0,
        "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
        "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
        "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
    }

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                **stable_support,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.468,
            }
        if call_count == 2:
            return {
                **stable_support,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.6,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.61,
            }
        return {
            **stable_support,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.398,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics["scorer_space_step_guard_backtracking_accepted"] == pytest.approx(
        1.0
    )
    assert metrics[
        "scorer_space_step_guard_backtracking_last_target_class_min_ratio_bootstrap_hold_allowed"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_backtracking_last_target_class_bootstrap_escape_allowed"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_backtracking_last_reject_reason_post_segnet_target_class_coverage_fraction_below_floor"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_backtracking_last_reject_reason_post_segnet_target_class_min_ratio_below_floor"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_backtracking_last_direct_nonrate_improved"
    ] == pytest.approx(1.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(0.0)


@mlx_only
def test_adapter_scorer_space_step_guard_rejects_target_class_ratio_theft() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=1.0,
        scorer_space_step_guard_min_post_segnet_target_class_min_ratio=0.2,
        scorer_space_step_guard_max_post_segnet_target_class_ratio_drop=0.05,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.001,
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=5.0,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        common = {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_pose_direct_live_score_term": 40.0,
            "loss_part_pose_score_term": 55.0,
            "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.09,
            "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.07,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.1,
            "loss_part_segnet_direct_live_target_class_0_fraction": 0.2,
            "loss_part_segnet_direct_live_target_class_1_fraction": 0.2,
            "loss_part_segnet_direct_live_target_class_2_fraction": 0.6,
        }
        if call_count == 1:
            return {
                **common,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.6,
                "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.0,
                "loss_part_segnet_direct_live_candidate_target_class_0_ratio": 0.8,
                "loss_part_segnet_direct_live_candidate_target_class_1_ratio": 0.0,
                "loss_part_segnet_direct_live_candidate_target_class_2_ratio": 1.0,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.406,
                "loss_part_segnet_direct_live_rare_class_logit_loss": 55104.0,
            }
        return {
            **common,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.0,
            "loss_part_segnet_direct_live_candidate_target_class_0_ratio": 0.2,
            "loss_part_segnet_direct_live_candidate_target_class_1_ratio": 0.0,
            "loss_part_segnet_direct_live_candidate_target_class_2_ratio": 1.0,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.422,
            "loss_part_segnet_direct_live_rare_class_logit_loss": 51304.0,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics["scorer_space_step_guard_segnet_target_class_ratio_drop"] == (
        pytest.approx(0.6)
    )
    assert metrics[
        "scorer_space_step_guard_target_class_ratio_drop_within_limit"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_segnet_target_class_ratio_drop_above_ceiling"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_bootstrap_direct_nonrate_worsening_allowed"
    ] == pytest.approx(0.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(1.0)


@mlx_only
def test_adapter_scorer_space_step_guard_accepts_low_occupancy_rare_recovery() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=1.0,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.001,
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=5.0,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        common = {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.2,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.2,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.51,
            "loss_part_pose_direct_live_score_term": 44.0,
            "loss_part_pose_score_term": 100.0,
            "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.11,
            "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.08,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.4,
        }
        if call_count == 1:
            return {
                **common,
                "loss_part_segnet_direct_live_rare_class_logit_loss": 2000.0,
            }
        return {
            **common,
            "loss_part_segnet_direct_live_rare_class_logit_loss": 1900.0,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics["scorer_space_step_guard_eligible"] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_rare_class_logit_recovered_meaningfully"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_bootstrap_direct_nonrate_worsening_allowed"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_target_class_bootstrap_escape_allowed"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_segnet_occupied_class_fraction_below_floor_suppressed_by_bootstrap_escape"
    ] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_segnet_occupied_class_fraction_below_floor"
    ] == pytest.approx(0.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(0.0)


@mlx_only
def test_adapter_scorer_space_step_guard_rejects_missing_class_worsening_without_rare_recovery() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        scorer_space_step_guard_enabled=True,
        scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_occupied_class_fraction=0.4,
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=0.8,
        scorer_space_step_guard_max_post_segnet_argmax_disagreement=0.5,
        scorer_space_step_guard_max_post_segnet_distribution_mae=0.30,
        scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=0.20,
        scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=3.5,
        scorer_space_step_guard_max_direct_nonrate_score_worsening=0.001,
        scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=5.0,
        scorer_space_step_guard_backtracking_steps=0,
    )
    call_count = 0

    def _fake_loss_part_metrics(_batch, *, loss_weights=None):
        nonlocal call_count
        call_count += 1
        common = {
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.6,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.6,
            "loss_part_pose_direct_live_score_term": 44.0,
            "loss_part_pose_score_term": 100.0,
            "loss_part_segnet_direct_live_rare_class_logit_loss": 2000.0,
            "loss_part_scorer_input_distribution_guard_segnet_frame1_mae": 0.11,
            "loss_part_scorer_input_distribution_guard_yuv6_pair_mae": 0.08,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 1.4,
        }
        if call_count == 1:
            return {
                **common,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.52,
            }
        return {
            **common,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.56,
        }

    adapter._score_aware_loss_part_metrics = _fake_loss_part_metrics  # type: ignore[method-assign]

    metrics = adapter.train_step(
        mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={},
    )

    assert metrics[
        "scorer_space_step_guard_rare_class_logit_recovered_meaningfully"
    ] == pytest.approx(0.0)
    assert metrics[
        "scorer_space_step_guard_bootstrap_direct_nonrate_worsening_allowed"
    ] == pytest.approx(0.0)
    assert metrics["scorer_space_step_guard_rejected"] == pytest.approx(1.0)
    assert metrics[
        "scorer_space_step_guard_reject_reason_post_direct_nonrate_score_worsened"
    ] == pytest.approx(1.0)


@mlx_only
def test_pact_muon_adamw_train_step_passes_clipped_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    import tac.local_acceleration.pr95_hnerv_mlx as pr95_mlx

    bundle = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    captured: dict[str, float | None] = {}

    def _fake_apply_pr95_step(_model, grads, _state, _config):
        leaves = [
            grad.astype(mx.float32)
            for _name, grad in tree_flatten(grads)
            if hasattr(grad, "shape")
        ]
        total_sq = mx.array(0.0, dtype=mx.float32)
        for grad in leaves:
            total_sq = total_sq + mx.sum(grad * grad)
        total_norm = mx.sqrt(total_sq)
        mx.eval(total_norm)
        captured["grad_norm"] = float(total_norm.item())
        captured["helper_grad_clip"] = _config.grad_clip
        captured["helper_grad_clip_muon"] = _config.grad_clip_muon
        return {
            "use_muon": True,
            "muon_tensor_count": 0,
            "adamw_tensor_count": len(leaves),
        }

    monkeypatch.setattr(
        pr95_mlx,
        "apply_pr95_mlx_optimizer_step",
        _fake_apply_pr95_step,
    )
    max_norm = 1.0e-7
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
        optimizer_kind="pact_muon_adamw",
        grad_clip_max_norm=max_norm,
    )
    batch = mx.array([0, 1], dtype=mx.int32)

    metrics = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})

    assert metrics["pact_optimizer_uses_muon"] == pytest.approx(1.0)
    assert "grad_norm" in captured
    assert captured["grad_norm"] <= max_norm * 1.0001 + 1.0e-10
    assert captured["helper_grad_clip"] is None
    assert captured["helper_grad_clip_muon"] is None


@mlx_only
def test_adapter_stage_weights_skip_student_head_updates_when_gated() -> None:
    import mlx.core as mx

    from tac.substrates.hinton_distilled_scorer_surrogate import (
        RealPoseNetTeacherCache,
        build_learnable_pose_student_head,
    )

    base = _tiny_dreamer_bundle(num_pairs=4, distill=0.5)
    pose_teacher = RealPoseNetTeacherCache(
        teacher_pose_np=mx.ones((4, 6)),
        num_pairs=4,
        pose_dims=6,
    )
    pose_head = build_learnable_pose_student_head(seed=31, input_channels=6)
    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        distillation_weight=0.5,
        allow_mock_scorer_teacher=True,
        pose_distillation_weight=0.5,
        pose_scorer_teacher=pose_teacher,
        learnable_pose_student_head=pose_head,
        pose_student_input_preprocess="pr95_yuv6",
    )
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    batch = mx.array([0, 1, 2, 3], dtype=mx.int32)
    w0 = mx.array(pose_head.weight)

    metrics = adapter.train_step(
        batch,
        learning_rate=1e-2,
        loss_weights={"recon": 1.0, "distill": 0.0, "pose_distill": 0.0},
    )

    moved = float(mx.max(mx.abs(pose_head.weight - w0)).item())
    assert moved == pytest.approx(0.0)
    assert "loss_part_distill" not in metrics
    assert "loss_part_pose_distill" not in metrics
    assert metrics["loss_part_stage_weight_recon"] == pytest.approx(1.0)


@mlx_only
def test_adapter_satisfies_protocol() -> None:
    from tac.training.long_training_canonical import validate_substrate_adapter

    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(), substrate_id="dreamer_v3_rssm"
    )
    validate_substrate_adapter(adapter)


@mlx_only
def test_adapter_artifact_metadata_records_score_aware_objective() -> None:
    bundle = _tiny_dreamer_bundle(distill=0.5)
    bundle.segnet_distillation_objective = "boundary_argmax_hinge"
    bundle.segnet_tau_boundary = 2.0
    bundle.segnet_hinge_margin = 0.5
    bundle.eval_roundtrip_ste_enabled = True
    bundle.eval_roundtrip_camera_hw = (874, 1164)
    bundle.pose_distillation_loss = "huber"
    bundle.pose_distillation_huber_delta = 2.5
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")

    metadata = adapter.artifact_metadata()
    objective = metadata["score_aware_training"]
    assert objective["segnet_distillation_objective"] == "boundary_argmax_hinge"
    assert objective["segnet_tau_boundary"] == 2.0
    assert objective["segnet_hinge_margin"] == 0.5
    assert objective["allow_mock_scorer_teacher"] is True
    assert objective["pose_distillation_loss"] == "huber"
    assert objective["pose_distillation_huber_delta"] == 2.5
    assert objective["eval_roundtrip_ste"]["enabled"] is True
    assert objective["eval_roundtrip_ste"]["camera_hw"] == [874, 1164]
    assert objective["pose_student_input_preprocess"]["mode"] == "rgb"
    assert metadata["decoder_weight_gradient_saliency"]["row_count"] == 0
    assert "score_claim" not in json.dumps(
        metadata["decoder_weight_gradient_saliency"], sort_keys=True
    )


@mlx_only
def test_adapter_metadata_marks_direct_live_active_without_student_distill() -> None:
    import mlx.core as mx

    class _LiveSegTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            return mx.ones((int(idx.shape[0]), 384, 512, self.num_classes))

        def teacher_logits_for_frames_nhwc01(self, frames):
            b, h, w, _c = frames.shape
            return mx.zeros((int(b), int(h), int(w), self.num_classes))

    base = _tiny_dreamer_bundle(distill=0.0)
    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        scorer_teacher=_LiveSegTeacher(),
        distillation_weight=0.0,
        segnet_direct_live_distillation_weight=0.75,
        segnet_distillation_objective="boundary_argmax_hinge",
    )
    metadata = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
    ).artifact_metadata()

    direct = metadata["score_aware_training"]["segnet_direct_live_distillation"]
    assert direct["enabled"] is True
    assert direct["weight"] == pytest.approx(0.75)
    assert direct["class_histogram_weight"] == pytest.approx(0.0)
    assert direct["class_balanced_hinge_weight"] == pytest.approx(0.0)
    assert direct["target_mass_floor_weight"] == pytest.approx(0.0)


@mlx_only
def test_adapter_metadata_marks_target_support_floor_active_without_base_direct_live() -> None:
    import mlx.core as mx

    class _LiveSegTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            return mx.ones((int(idx.shape[0]), 384, 512, self.num_classes))

        def teacher_logits_for_frames_nhwc01(self, frames):
            b, h, w, _c = frames.shape
            return mx.zeros((int(b), int(h), int(w), self.num_classes))

    base = _tiny_dreamer_bundle(distill=0.0)
    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        scorer_teacher=_LiveSegTeacher(),
        distillation_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        segnet_direct_live_target_mass_floor_weight=0.75,
        segnet_direct_live_target_min_ratio_floor_weight=0.5,
        segnet_distillation_objective="boundary_argmax_hinge",
    )
    metadata = MlxScoreAwareAdapter(
        bundle,
        substrate_id="dreamer_v3_rssm",
    ).artifact_metadata()

    direct = metadata["score_aware_training"]["segnet_direct_live_distillation"]
    assert direct["enabled"] is True
    assert direct["weight"] == pytest.approx(0.0)
    assert direct["target_mass_floor_weight"] == pytest.approx(0.75)
    assert direct["target_min_ratio_floor_weight"] == pytest.approx(0.5)
    assert direct["objective"] == "boundary_argmax_hinge"


@mlx_only
def test_adapter_accumulates_real_decoder_weight_gradient_saliency() -> None:
    import mlx.core as mx

    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(distill=0.0), substrate_id="dreamer_v3_rssm"
    )
    adapter._accumulate_decoder_weight_gradient_saliency(
        {
            "decoder": {
                "weight": mx.ones((2, 3), dtype=mx.float32),
                "bias": mx.array([2.0], dtype=mx.float32),
            },
            "latents": {"weight": mx.ones((1,), dtype=mx.float32)},
            "student": {"head": mx.ones((1,), dtype=mx.float32)},
            "codebook": {"values": mx.ones((1,), dtype=mx.float32)},
        }
    )
    adapter._accumulate_decoder_weight_gradient_saliency(
        {"decoder": {"weight": mx.full((2, 3), 2.0, dtype=mx.float32)}}
    )

    summary = adapter.decoder_weight_gradient_saliency_summary()
    rows = {row["group_name"]: row for row in summary["rows"]}
    assert summary["schema"] == "mlx_decoder_weight_gradient_saliency.v1"
    assert summary["row_count"] == 2
    assert set(rows) == {"decoder.bias", "decoder.weight"}
    assert rows["decoder.weight"]["sum_grad_sq"] == pytest.approx(30.0)
    assert rows["decoder.weight"]["sample_count"] == 2
    assert rows["decoder.weight"]["numel"] == 6
    assert rows["decoder.weight"]["saliency"] == pytest.approx(2.5)
    assert rows["decoder.weight"]["max_abs_grad"] == pytest.approx(2.0)
    assert rows["decoder.bias"]["sum_grad_sq"] == pytest.approx(4.0)
    assert rows["decoder.bias"]["saliency"] == pytest.approx(4.0)

    metadata = adapter.artifact_metadata()
    saliency = metadata["decoder_weight_gradient_saliency"]
    assert saliency["row_count"] == 2
    assert saliency["saliency_by_name"]["decoder.weight"] == pytest.approx(2.5)


@mlx_only
def test_adapter_optimizer_step_raises_style_a_stub() -> None:
    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(), substrate_id="dreamer_v3_rssm"
    )
    with pytest.raises(NotImplementedError, match="Style B train_step"):
        adapter.optimizer_step(adapter.model, None, 1e-3)


@mlx_only
def test_adapter_export_state_dict_writes_portable_npsd(tmp_path: Path) -> None:
    from tac.substrates._shared.numpy_portable_inflate import (
        unpack_state_dict_numpy,
    )

    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(), substrate_id="dreamer_v3_rssm"
    )
    target = tmp_path / "ckpt.state"
    adapter.export_state_dict(adapter.model, target)
    blob = target.with_suffix(target.suffix + ".npsd")
    assert blob.is_file()
    restored = unpack_state_dict_numpy(blob.read_bytes())
    assert len(restored) > 0


@mlx_only
def test_adapter_import_state_dict_restores_portable_npsd(tmp_path: Path) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(), substrate_id="dreamer_v3_rssm"
    )
    target = tmp_path / "ckpt.state"
    adapter.export_state_dict(adapter.model, target)
    before = {
        name: np.array(value)
        for name, value in tree_flatten(adapter.model.parameters())
    }

    mutated = [
        (name, mx.ones_like(value))
        for name, value in tree_flatten(adapter.model.parameters())
    ]
    adapter.model.update(tree_unflatten(mutated))
    mx.eval(*[value for _name, value in tree_flatten(adapter.model.parameters())])

    adapter.import_state_dict(adapter.model, target)
    restored = {
        name: np.array(value)
        for name, value in tree_flatten(adapter.model.parameters())
    }

    assert set(restored) == set(before)
    for name, expected in before.items():
        np.testing.assert_allclose(restored[name], expected, rtol=0.0, atol=0.0)


@mlx_only
def test_adapter_score_aware_components_pure_recon_returns_none() -> None:
    """Pure-reconstruction mode preserves legacy None contract.

    PER_AXIS_DECOMPOSITION GAP FIX 2026-05-28: sister-adapter parity preserved
    when neither scorer surrogate is active. The fix MUST NOT synthesize
    scorer-unbound per-axis rows.
    """
    import mlx.core as mx

    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(distill=0.0), substrate_id="dreamer_v3_rssm"
    )
    assert adapter.score_aware_components(adapter.model, mx.array([0])) is None


@mlx_only
def test_adapter_score_aware_components_seg_bound_populates_per_axis() -> None:
    """Hinton-distilled scorer-bound surrogate populates per-axis per Catalog #356.

    PER_AXIS_DECOMPOSITION GAP FIX 2026-05-28 per Z6-v2 + Hinton + 600-pair
    Contrarian VETO op-routable #4: when ``distillation_weight > 0`` (SegNet
    teacher wired via mock or real) the per-axis decomposition MUST emit
    seg+recon_aux+archive_bytes rows so the canonical
    ``AxisDecomposition.from_dict`` round-trip works at the downstream
    cathedral ranker boundary.
    """
    import mlx.core as mx

    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(distill=0.5), substrate_id="dreamer_v3_rssm"
    )
    out = adapter.score_aware_components(adapter.model, mx.array([0, 1], dtype=mx.int32))
    assert out is not None
    assert "seg" in out
    assert "pose" in out  # 0.0 (no pose teacher in mock fixture) per fail-closed
    assert "recon_aux" in out
    assert "archive_bytes" in out
    # All values finite per AxisDecomposition __post_init__ invariant.
    for key, value in out.items():
        assert isinstance(value, float), f"{key}={value!r} must be float"
        assert value == value, f"{key}={value!r} must not be NaN"
    # seg > 0 because Hinton-KL is non-negative + the bundle has distill=0.5.
    assert out["seg"] >= 0.0
    # pose = 0.0 because the mock fixture does not wire a pose teacher
    # (no pose_distill component emitted by score_aware_loss).
    assert out["pose"] == 0.0
    # archive_bytes = 0.0 per AxisDecomposition NaN-safe rule (per-step
    # delta undefined at MLX L2; archive built post-training).
    assert out["archive_bytes"] == 0.0


@mlx_only
def test_adapter_score_aware_components_both_teachers_populates_seg_and_pose() -> None:
    """Both SegNet + PoseNet teachers wired → per-axis seg AND pose populated.

    PER_AXIS_DECOMPOSITION GAP FIX 2026-05-28 cross-family sister: the
    canonical scorer-bound BOTH-TEACHER-WIRED contract (Catalog #164) IS
    the surface where the GAP closed. Cross-family seg/pose attribution
    becomes possible only when BOTH axes are populated empirically.
    """
    import mlx.core as mx

    from tac.substrates.hinton_distilled_scorer_surrogate import (
        RealPoseNetTeacherCache,
        build_learnable_pose_student_head,
    )

    base = _tiny_dreamer_bundle(num_pairs=4, distill=0.5)
    pose_teacher = RealPoseNetTeacherCache(
        teacher_pose_np=mx.ones((4, 6)),
        num_pairs=4,
        pose_dims=6,
    )
    pose_head = build_learnable_pose_student_head(seed=17, input_channels=6)
    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        distillation_weight=0.5,
        allow_mock_scorer_teacher=True,  # seg side via mock (no real SegNet here)
        pose_distillation_weight=0.5,
        pose_scorer_teacher=pose_teacher,
        learnable_pose_student_head=pose_head,
        pose_student_input_preprocess="pr95_yuv6",
    )
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    out = adapter.score_aware_components(
        adapter.model, mx.array([0, 1, 2, 3], dtype=mx.int32)
    )
    assert out is not None
    assert out["seg"] >= 0.0
    assert out["pose"] >= 0.0  # pose-MSE is non-negative
    assert out["recon_aux"] >= 0.0
    assert out["archive_bytes"] == 0.0


@mlx_only
def test_adapter_score_aware_components_direct_live_pose_populates_pose_axis() -> None:
    import mlx.core as mx

    from tac.substrates.hinton_distilled_scorer_surrogate import (
        RealPoseNetTeacherCache,
    )

    class _TinyLivePoseAdapter:
        def __call__(self, yuv6_pair_nhwc):
            mean = mx.mean(yuv6_pair_nhwc, axis=(1, 2, 3))
            pose = mx.stack(
                [
                    mean,
                    mean * 0.5,
                    mean * 0.25,
                    mean * 0.125,
                    mean * 0.0625,
                    mean * 0.03125,
                ],
                axis=-1,
            )
            return {"pose": pose.astype(mx.float32)}

    base = _tiny_dreamer_bundle(num_pairs=2, distill=0.0)
    pose_teacher = RealPoseNetTeacherCache(
        teacher_pose_np=mx.zeros((2, 6), dtype=mx.float32),
        num_pairs=2,
        pose_dims=6,
        per_dim_scale=mx.ones((6,), dtype=mx.float32),
        live_posenet_adapter=_TinyLivePoseAdapter(),
    )
    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        pose_direct_live_distillation_weight=0.5,
        pose_scorer_teacher=pose_teacher,
    )
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")

    out = adapter.score_aware_components(
        adapter.model,
        mx.array([0, 1], dtype=mx.int32),
    )

    assert out is not None
    assert out["seg"] == 0.0
    assert out["pose"] >= 0.0
    assert out["recon_aux"] >= 0.0
    assert out["archive_bytes"] == 0.0

    metrics = adapter._score_aware_loss_part_metrics(
        mx.array([0, 1], dtype=mx.int32),
    )
    assert "loss_part_pose_direct_live_score_term" in metrics
    assert metrics["score_aware_loss_parts_active"] == pytest.approx(1.0)


@mlx_only
def test_adapter_score_aware_components_compatible_with_axis_decomposition() -> None:
    """Per-axis dict round-trips through canonical AxisDecomposition contract.

    Catalog #356 STRICT preflight gate contract: the per-axis surface MUST
    map directly into ``AxisDecomposition.from_dict``-like consumption. The
    canonical mapping is: seg → predicted_d_seg_delta; pose →
    predicted_d_pose_delta; archive_bytes → predicted_archive_bytes_delta.
    """
    import mlx.core as mx

    from tac.cathedral.consumer_contract import AxisDecomposition
    from tac.provenance import (
        build_provenance_for_predicted,
        provenance_to_dict,
    )

    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(distill=0.5), substrate_id="dreamer_v3_rssm"
    )
    out = adapter.score_aware_components(
        adapter.model, mx.array([0, 1], dtype=mx.int32)
    )
    assert out is not None
    # Build a canonical AxisDecomposition from the per-axis dict + canonical
    # Provenance per Catalog #323. This verifies the GAP-fix output integrates
    # with the downstream cathedral ranker boundary contract.
    prov = build_provenance_for_predicted(
        model_id="mlx_score_aware_per_axis_decomposition_v1",
        inputs_sha256="a" * 64,
        measurement_axis="[macOS-MLX research-signal]",
        hardware_substrate="macos_arm64",
    )
    decomp = AxisDecomposition(
        predicted_d_seg_delta=out["seg"],
        predicted_d_pose_delta=out["pose"],
        predicted_archive_bytes_delta=int(out["archive_bytes"]),
        axis_tag="[predicted]",
        canonical_provenance=provenance_to_dict(prov),
    )
    # Round-trip stable (no NaN, no infinite, no type rejection).
    d = decomp.as_dict()
    assert d["predicted_d_seg_delta"] == out["seg"]
    assert d["predicted_d_pose_delta"] == out["pose"]
    assert d["predicted_archive_bytes_delta"] == 0
    assert d["axis_tag"] == "[predicted]"
    assert d["canonical_provenance"]["measurement_axis"] == "[macOS-MLX research-signal]"


# --------------------------------------------------------------------------- #
# harness orchestrator (end-to-end through canonical run_long_training)
# --------------------------------------------------------------------------- #


@mlx_only
def test_run_mlx_score_aware_full_main_end_to_end(tmp_path: Path) -> None:
    bundle = _tiny_dreamer_bundle(num_pairs=4, distill=0.5)
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="dreamer_v3_rssm",
        lane_id=_LANE,
        output_dir=tmp_path / "run",
        epochs=3,
        batch_pair_indices_per_step=2,
        learning_rate=1e-3,
        seed=0,
        notes="harness refactor end-to-end: dreamer renderer + zero targets",
    )
    assert artifact.total_epochs_completed == 3
    assert artifact.promotable is False
    d = artifact.as_dict()
    assert d.get("score_claim") is False
    assert d.get("promotion_eligible") is False


@mlx_only
def test_run_mlx_score_aware_full_main_forwards_telemetry_flush_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.training.long_training_canonical as canonical

    captured = {}

    def _capture_run_long_training(adapter, config, *, on_epoch_end=None):
        captured["adapter"] = adapter
        captured["config"] = config
        captured["on_epoch_end"] = on_epoch_end
        return config

    monkeypatch.setattr(canonical, "run_long_training", _capture_run_long_training)
    bundle = _tiny_dreamer_bundle(num_pairs=4, distill=0.0)

    result = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="dreamer_v3_rssm",
        lane_id=_LANE,
        output_dir=tmp_path / "run",
        epochs=3,
        batch_pair_indices_per_step=2,
        telemetry_flush_interval_epochs=1,
        checkpoint_dir=tmp_path / "external_checkpoints",
        resume_from_checkpoint=tmp_path / "external_checkpoints/epoch000001.meta.json",
        checkpoint_selection_metric_key=(
            "loss_part_segnet_direct_live_escape_selection"
        ),
        checkpoint_selection_metric_required=True,
        checkpoint_selection_tie_break_metric_key=(
            "loss_part_segnet_direct_live_argmax_disagreement"
        ),
        checkpoint_selection_tie_break_metric_required=True,
        notes="telemetry flush pass-through unit test",
    )

    assert result is captured["config"]
    assert captured["config"].telemetry_flush_interval_epochs == 1
    assert captured["config"].checkpoint_dir == tmp_path / "external_checkpoints"
    assert captured["config"].resume_from_checkpoint == (
        tmp_path / "external_checkpoints/epoch000001.meta.json"
    )
    assert captured["config"].checkpoint_selection_metric_key == (
        "loss_part_segnet_direct_live_escape_selection"
    )
    assert captured["config"].checkpoint_selection_metric_required is True
    assert captured["config"].checkpoint_selection_tie_break_metric_key == (
        "loss_part_segnet_direct_live_argmax_disagreement"
    )
    assert captured["config"].checkpoint_selection_tie_break_metric_required is True


@mlx_only
def test_run_verifies_inflate_portability_fails_closed(tmp_path: Path) -> None:
    from tac.substrates._shared.mlx_score_aware import MlxScoreAwareHarnessError

    bad_inflate = tmp_path / "inflate.py"
    bad_inflate.write_text("import torch\n", encoding="utf-8")
    bundle = _tiny_dreamer_bundle(num_pairs=4, distill=0.0)
    with pytest.raises(MlxScoreAwareHarnessError, match="forbidden non-portable"):
        run_mlx_score_aware_full_main(
            bundle=bundle,
            substrate_id="dreamer_v3_rssm",
            lane_id=_LANE,
            output_dir=tmp_path / "run",
            epochs=1,
            batch_pair_indices_per_step=2,
            inflate_py_path=bad_inflate,
            notes="inflate portability fail-closed before training",
        )
