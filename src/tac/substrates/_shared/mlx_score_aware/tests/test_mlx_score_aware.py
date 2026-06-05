# SPDX-License-Identifier: MIT
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tac.substrates._shared.mlx_score_aware import (
    MlxScoreAwareHarnessError,
    RendererBundle,
    assert_numpy_portable_inflate,
    build_mlx_posenet_pair_teacher,
    decode_frames_nhwc01,
    is_mlx_available,
    require_mlx_for_harness,
    score_aware_loss,
)

mx = pytest.importorskip("mlx.core")


class ReconstructPairModel:
    def __init__(self, target_rgb_0, target_rgb_1) -> None:
        self.target_rgb_0 = target_rgb_0
        self.target_rgb_1 = target_rgb_1

    def parameters(self):
        return {}

    def reconstruct_pair(self, idx):
        rgb_0 = mx.transpose(self.target_rgb_0[idx], (0, 3, 1, 2))
        rgb_1 = mx.transpose(self.target_rgb_1[idx], (0, 3, 1, 2))
        return rgb_0, rgb_1


class CallPairModel:
    def __init__(self, target_rgb_0, target_rgb_1) -> None:
        pair = mx.stack(
            [
                mx.transpose(target_rgb_0, (0, 3, 1, 2)),
                mx.transpose(target_rgb_1, (0, 3, 1, 2)),
            ],
            axis=1,
        )
        self.pair_255 = pair * 255.0

    def parameters(self):
        return {}

    def __call__(self, idx):
        return self.pair_255[idx]


def _targets():
    target_0 = mx.array(
        np.linspace(0.0, 1.0, num=2 * 4 * 4 * 3, dtype=np.float32).reshape(
            2, 4, 4, 3
        )
    )
    target_1 = 1.0 - target_0
    return target_0, target_1


def _scalar(value) -> float:
    return float(np.array(value))


def test_device_gate_reports_mlx_available_on_local_host() -> None:
    assert is_mlx_available()
    assert require_mlx_for_harness().__name__ == "mlx.core"


def test_output_head_bias_gradient_multiplier_zeroes_exact_bias_updates() -> None:
    import mlx.nn as nn

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    class HeadBiasRenderer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head_rgb_0 = nn.Linear(1, 3)
            self.head_rgb_1 = nn.Linear(1, 3)

        def reconstruct_pair(self, idx):
            batch_size = int(idx.shape[0])
            z = mx.ones((batch_size, 2, 2, 1), dtype=mx.float32)
            rgb0 = mx.sigmoid(self.head_rgb_0(z))
            rgb1 = mx.sigmoid(self.head_rgb_1(z))
            return (
                mx.transpose(rgb0, (0, 3, 1, 2)),
                mx.transpose(rgb1, (0, 3, 1, 2)),
            )

    model = HeadBiasRenderer()
    bundle = RendererBundle(
        model=model,
        target_rgb_0=mx.zeros((2, 2, 2, 3), dtype=mx.float32),
        target_rgb_1=mx.ones((2, 2, 2, 3), dtype=mx.float32),
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
    )
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="head_bias_multiplier_test",
        optimizer_kind="adamw",
        weight_decay=0.0,
        output_head_bias_gradient_multiplier=0.0,
    )
    b0_before = mx.array(model.head_rgb_0.bias)
    b1_before = mx.array(model.head_rgb_1.bias)

    metrics = adapter.train_step(
        batch=mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={"recon": 1.0},
    )
    mx.eval(model.parameters())

    assert metrics["gradient_multiplier_active"] == pytest.approx(1.0)
    assert metrics["gradient_multiplier_zeroed_leaf_count"] == pytest.approx(2.0)
    assert metrics["gradient_multiplier_output_head_bias"] == pytest.approx(0.0)
    assert float(
        mx.max(mx.abs(model.head_rgb_0.bias - b0_before)).item()
    ) == pytest.approx(0.0)
    assert float(
        mx.max(mx.abs(model.head_rgb_1.bias - b1_before)).item()
    ) == pytest.approx(0.0)


def test_bias_gradient_multiplier_zeroes_non_head_bias_but_keeps_weights_live() -> None:
    import mlx.nn as nn

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    class DecoderBiasRenderer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = nn.Linear(1, 3)

        def reconstruct_pair(self, idx):
            batch_size = int(idx.shape[0])
            z = mx.ones((batch_size, 2, 2, 1), dtype=mx.float32)
            rgb = mx.sigmoid(self.decoder(z))
            nchw = mx.transpose(rgb, (0, 3, 1, 2))
            return nchw, nchw

    model = DecoderBiasRenderer()
    bundle = RendererBundle(
        model=model,
        target_rgb_0=mx.zeros((2, 2, 2, 3), dtype=mx.float32),
        target_rgb_1=mx.zeros((2, 2, 2, 3), dtype=mx.float32),
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
    )
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="decoder_bias_multiplier_test",
        optimizer_kind="adamw",
        weight_decay=0.0,
        bias_gradient_multiplier=0.0,
    )
    bias_before = mx.array(model.decoder.bias)
    weight_before = mx.array(model.decoder.weight)

    metrics = adapter.train_step(
        batch=mx.array([0, 1], dtype=mx.int32),
        learning_rate=5e-2,
        loss_weights={"recon": 1.0},
    )
    mx.eval(model.parameters())

    assert metrics["gradient_multiplier_active"] == pytest.approx(1.0)
    assert metrics["gradient_multiplier_zeroed_leaf_count"] >= 1.0
    assert metrics["gradient_multiplier_bias"] == pytest.approx(0.0)
    assert float(mx.max(mx.abs(model.decoder.bias - bias_before)).item()) == (
        pytest.approx(0.0)
    )
    assert float(mx.max(mx.abs(model.decoder.weight - weight_before)).item()) > 0.0


def test_native_optimizer_reports_single_global_clip_application() -> None:
    import mlx.nn as nn

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    class TinyRenderer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = nn.Linear(1, 3)

        def reconstruct_pair(self, idx):
            batch_size = int(idx.shape[0])
            z = mx.ones((batch_size, 2, 2, 1), dtype=mx.float32)
            rgb = mx.sigmoid(self.decoder(z))
            nchw = mx.transpose(rgb, (0, 3, 1, 2))
            return nchw, nchw

    model = TinyRenderer()
    bundle = RendererBundle(
        model=model,
        target_rgb_0=mx.zeros((2, 2, 2, 3), dtype=mx.float32),
        target_rgb_1=mx.ones((2, 2, 2, 3), dtype=mx.float32),
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
    )
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="native_clip_telemetry_test",
        optimizer_kind="adamw",
        weight_decay=0.0,
        grad_clip_max_norm=1.0e-6,
    )

    metrics = adapter.train_step(
        batch=mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={"recon": 1.0},
    )

    assert metrics["gradient_clip_enabled"] == pytest.approx(1.0)
    assert metrics["gradient_clip_actual_application_count"] == pytest.approx(1.0)
    assert metrics["gradient_clip_delegated_to_pr95_partition_helper"] == pytest.approx(
        0.0
    )
    assert metrics["gradient_global_norm_pre_clip"] > 0.0
    assert metrics["gradient_clip_would_clip"] == pytest.approx(1.0)
    assert 0.0 < metrics["gradient_clip_scale"] < 1.0


def test_pact_muon_preclips_and_keeps_partition_clip_cap() -> None:
    import mlx.nn as nn

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    class TinyRenderer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = nn.Linear(1, 3)

        def reconstruct_pair(self, idx):
            batch_size = int(idx.shape[0])
            z = mx.ones((batch_size, 2, 2, 1), dtype=mx.float32)
            rgb = mx.sigmoid(self.decoder(z))
            nchw = mx.transpose(rgb, (0, 3, 1, 2))
            return nchw, nchw

    model = TinyRenderer()
    bundle = RendererBundle(
        model=model,
        target_rgb_0=mx.zeros((2, 2, 2, 3), dtype=mx.float32),
        target_rgb_1=mx.ones((2, 2, 2, 3), dtype=mx.float32),
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
    )
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="pact_muon_clip_telemetry_test",
        optimizer_kind="pact_muon_adamw",
        weight_decay=0.0,
        grad_clip_max_norm=1.0e-6,
    )

    metrics = adapter.train_step(
        batch=mx.array([0, 1], dtype=mx.int32),
        learning_rate=1e-2,
        loss_weights={"recon": 1.0},
    )

    assert metrics["gradient_clip_enabled"] == pytest.approx(1.0)
    assert metrics["gradient_clip_actual_application_count"] == pytest.approx(1.0)
    assert metrics["gradient_clip_delegated_to_pr95_partition_helper"] == pytest.approx(
        1.0
    )
    assert metrics["gradient_global_norm_pre_clip"] > 0.0
    assert metrics["gradient_clip_would_clip"] == pytest.approx(1.0)
    assert 0.0 < metrics["gradient_clip_scale"] < 1.0
    assert metrics["pact_optimizer_uses_muon"] == pytest.approx(1.0)


def test_renderer_bundle_validation_fail_closed() -> None:
    target_0, target_1 = _targets()
    with pytest.raises(MlxScoreAwareHarnessError, match="forward_convention"):
        RendererBundle(
            model=object(),
            target_rgb_0=target_0,
            target_rgb_1=target_1,
            num_pairs=2,
            forward_convention="unknown",
        )
    with pytest.raises(MlxScoreAwareHarnessError, match="num_pairs"):
        RendererBundle(
            model=object(),
            target_rgb_0=target_0,
            target_rgb_1=target_1,
            num_pairs=0,
        )
    with pytest.raises(MlxScoreAwareHarnessError, match="distillation_weight"):
        RendererBundle(
            model=object(),
            target_rgb_0=target_0,
            target_rgb_1=target_1,
            num_pairs=2,
            distillation_weight=-1.0,
        )
    with pytest.raises(
        MlxScoreAwareHarnessError,
        match="scorer_input_distribution_guard_weight",
    ):
        RendererBundle(
            model=object(),
            target_rgb_0=target_0,
            target_rgb_1=target_1,
            num_pairs=2,
            scorer_input_distribution_guard_weight=-1.0,
        )
    with pytest.raises(
        MlxScoreAwareHarnessError,
        match="scorer_input_distribution_guard_temperature",
    ):
        RendererBundle(
            model=object(),
            target_rgb_0=target_0,
            target_rgb_1=target_1,
            num_pairs=2,
            scorer_input_distribution_guard_temperature=0.0,
        )
    with pytest.raises(
        MlxScoreAwareHarnessError,
        match="scorer_input_contrast_floor_weight",
    ):
        RendererBundle(
            model=object(),
            target_rgb_0=target_0,
            target_rgb_1=target_1,
            num_pairs=2,
            scorer_input_contrast_floor_weight=-1.0,
        )
    with pytest.raises(
        MlxScoreAwareHarnessError,
        match="scorer_input_contrast_floor_posenet_yuv6_min_std_ratio",
    ):
        RendererBundle(
            model=object(),
            target_rgb_0=target_0,
            target_rgb_1=target_1,
            num_pairs=2,
            scorer_input_contrast_floor_posenet_yuv6_min_std_ratio=0.0,
        )


def test_decode_frames_supports_reconstruct_pair_nchw01() -> None:
    target_0, target_1 = _targets()
    bundle = RendererBundle(
        model=ReconstructPairModel(target_0, target_1),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
    )
    rgb_0, rgb_1 = decode_frames_nhwc01(bundle, mx.array([0, 1]))
    np.testing.assert_allclose(np.array(rgb_0), np.array(target_0), atol=1e-7)
    np.testing.assert_allclose(np.array(rgb_1), np.array(target_1), atol=1e-7)


def test_decode_frames_supports_call_b2chw_255() -> None:
    target_0, target_1 = _targets()
    bundle = RendererBundle(
        model=CallPairModel(target_0, target_1),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        forward_convention="call_b2chw_255",
    )
    rgb_0, rgb_1 = decode_frames_nhwc01(bundle, mx.array([0, 1]))
    np.testing.assert_allclose(np.array(rgb_0), np.array(target_0), atol=1e-6)
    np.testing.assert_allclose(np.array(rgb_1), np.array(target_1), atol=1e-6)


def test_source_pair_priorities_sample_local_hydrated_rows() -> None:
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    target_0, target_1 = _targets()
    bundle = RendererBundle(
        model=object(),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        source_pair_indices=(7, 2),
    )
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="source_pair_priority_test",
        prioritized_pair_indices=(7,),
    )

    batch = adapter.sample_batch(batch_size=1, seed=0)
    observed = adapter.batch_observability(batch)

    assert np.array(batch).tolist() == [0]
    assert observed is not None
    assert observed["requested_priority_pair_indices"] == [7]
    assert observed["priority_local_pair_indices_in_batch"] == [0]
    assert observed["priority_source_pair_indices_in_batch"] == [7]
    assert observed["source_pair_indices"] == [7]
    assert observed["priority_pair_alignment_mode"] == (
        "source_priority_pairs_to_local_rows"
    )
    assert observed["pair_index_alignment_mode"] == (
        "local_target_rows_to_source_pair_indices"
    )


def test_score_aware_loss_recon_distill_and_extra_terms_are_composed() -> None:
    target_0, target_1 = _targets()

    def extra_loss(_model, _idx):
        return {"regularizer": mx.array(2.0)}

    bundle = RendererBundle(
        model=ReconstructPairModel(target_0, target_1),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
        extra_loss_terms=extra_loss,
        extra_loss_weights={"regularizer": 0.25},
        distillation_weight=1.0,
        allow_mock_scorer_teacher=True,
    )
    total, parts = score_aware_loss(bundle, mx.array([0, 1]))
    assert _scalar(parts["recon"]) < 1e-10
    assert _scalar(parts["distill"]) < 1e-8
    assert _scalar(parts["regularizer"]) == pytest.approx(2.0)
    assert _scalar(total) == pytest.approx(0.5, abs=1e-6)
    assert _scalar(parts["total"]) == pytest.approx(_scalar(total), abs=1e-7)


def test_score_aware_loss_applies_scorer_input_distribution_guard() -> None:
    target_0, target_1 = _targets()
    zeros = mx.zeros_like(target_0)
    bundle = RendererBundle(
        model=ReconstructPairModel(zeros, zeros),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
        scorer_input_distribution_guard_weight=3.0,
    )
    idx = mx.array([0, 1], dtype=mx.int32)

    total_guard, parts_guard = score_aware_loss(bundle, idx)
    total_disabled, parts_disabled = score_aware_loss(
        bundle,
        idx,
        loss_weights={"scorer_input_guard": 0.0},
    )
    mx.eval(total_guard, total_disabled)

    assert "scorer_input_distribution_guard" in parts_guard
    assert "scorer_input_distribution_guard" not in parts_disabled
    assert _scalar(parts_guard["scorer_input_distribution_guard"]) > 0.0
    assert _scalar(total_guard) > _scalar(total_disabled)


def test_score_aware_loss_applies_scorer_input_contrast_floor() -> None:
    target_0, target_1 = _targets()
    flat = mx.ones_like(target_0) * 0.5
    bundle = RendererBundle(
        model=ReconstructPairModel(flat, flat),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
        scorer_input_distribution_guard_weight=0.0,
        scorer_input_contrast_floor_weight=2.0,
        scorer_input_contrast_floor_segnet_min_std_ratio=0.7,
        scorer_input_contrast_floor_posenet_yuv6_min_std_ratio=0.6,
    )
    idx = mx.array([0, 1], dtype=mx.int32)

    total_guard, parts_guard = score_aware_loss(bundle, idx)
    total_disabled, parts_disabled = score_aware_loss(
        bundle,
        idx,
        loss_weights={"scorer_input_guard": 0.0},
    )
    mx.eval(total_guard, total_disabled)

    assert "scorer_input_contrast_floor" in parts_guard
    assert "scorer_input_contrast_floor" not in parts_disabled
    assert _scalar(parts_guard["scorer_input_contrast_floor"]) > 0.0
    assert _scalar(
        parts_guard["scorer_input_contrast_floor_segnet_last_rgb_min_std_ratio"]
    ) < 0.7
    assert _scalar(
        parts_guard["scorer_input_contrast_floor_posenet_yuv6_pair_min_std_ratio"]
    ) < 0.6
    assert _scalar(total_guard) > _scalar(total_disabled)


def test_score_aware_loss_applies_scorer_input_shape_tether() -> None:
    target_0, target_1 = _targets()
    flat = mx.ones_like(target_0) * 0.5
    bundle = RendererBundle(
        model=ReconstructPairModel(flat, flat),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
        scorer_input_shape_tether_weight=2.0,
    )
    idx = mx.array([0, 1], dtype=mx.int32)

    total_guard, parts_guard = score_aware_loss(bundle, idx)
    total_disabled, parts_disabled = score_aware_loss(
        bundle,
        idx,
        loss_weights={"scorer_input_guard": 0.0},
    )
    mx.eval(total_guard, total_disabled)

    assert "scorer_input_shape_tether" in parts_guard
    assert "scorer_input_shape_tether" not in parts_disabled
    assert _scalar(parts_guard["scorer_input_shape_tether"]) > 0.0
    assert _scalar(
        parts_guard[
            "scorer_input_shape_tether_segnet_last_rgb_candidate_centered_std"
        ]
    ) < _scalar(
        parts_guard[
            "scorer_input_shape_tether_segnet_last_rgb_reference_centered_std"
        ]
    )
    assert _scalar(total_guard) > _scalar(total_disabled)


def test_score_aware_loss_rejects_negative_scorer_input_shape_tether_weight() -> None:
    target_0, target_1 = _targets()
    with pytest.raises(
        MlxScoreAwareHarnessError,
        match="scorer_input_shape_tether_weight must be >= 0",
    ):
        RendererBundle(
            model=ReconstructPairModel(target_0, target_1),
            target_rgb_0=target_0,
            target_rgb_1=target_1,
            num_pairs=2,
            forward_convention="reconstruct_pair_nchw01",
            scorer_input_shape_tether_weight=-0.125,
        )


def test_real_scorer_distill_selects_contest_segnet_frame_by_default() -> None:
    target_0 = mx.zeros((2, 4, 4, 3))
    target_1 = mx.ones((2, 4, 4, 3))

    class _Teacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            return mx.zeros((idx.shape[0], 4, 4, self.num_classes))

    class _RecordingHead:
        def __init__(self) -> None:
            self.last_mean = None

        def __call__(self, frames):
            self.last_mean = _scalar(mx.mean(frames))
            b, h, w, _c = frames.shape
            return mx.zeros((b, h, w, 5))

    head = _RecordingHead()
    bundle = RendererBundle(
        model=ReconstructPairModel(target_0, target_1),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
        distillation_weight=1.0,
        scorer_teacher=_Teacher(),
        learnable_student_head=head,
        allow_segnet_only_research=True,
    )
    score_aware_loss(bundle, mx.array([0, 1]))
    assert head.last_mean == pytest.approx(1.0)

    frame_0_head = _RecordingHead()
    frame_0_bundle = RendererBundle(
        model=ReconstructPairModel(target_0, target_1),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
        distillation_weight=1.0,
        scorer_teacher=_Teacher(),
        learnable_student_head=frame_0_head,
        segnet_teacher_frame_index=0,
        allow_segnet_only_research=True,
    )
    score_aware_loss(frame_0_bundle, mx.array([0, 1]))
    assert frame_0_head.last_mean == pytest.approx(0.0)


def test_score_aware_loss_routes_configured_segnet_objective() -> None:
    target_0 = mx.zeros((2, 4, 4, 3))
    target_1 = mx.ones((2, 4, 4, 3))

    class _Teacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            arr = np.zeros((idx.shape[0], 4, 4, self.num_classes), dtype=np.float32)
            arr[..., 0] = 3.0
            arr[..., 2] = 2.9
            return mx.array(arr)

    class _Head:
        def __call__(self, frames):
            b, h, w, _c = frames.shape
            return mx.zeros((b, h, w, 5))

    common = {
        "model": ReconstructPairModel(target_0, target_1),
        "target_rgb_0": target_0,
        "target_rgb_1": target_1,
        "num_pairs": 2,
        "forward_convention": "reconstruct_pair_nchw01",
        "distillation_weight": 1.0,
        "scorer_teacher": _Teacher(),
        "learnable_student_head": _Head(),
        "allow_segnet_only_research": True,
    }
    kl_bundle = RendererBundle(**common, segnet_distillation_objective="kl_t2")
    hinge_bundle = RendererBundle(
        **common,
        segnet_distillation_objective="boundary_argmax_hinge",
        segnet_tau_boundary=0.75,
        segnet_hinge_margin=0.5,
    )

    _kl_total, kl_parts = score_aware_loss(kl_bundle, mx.array([0, 1]))
    _hinge_total, hinge_parts = score_aware_loss(hinge_bundle, mx.array([0, 1]))

    assert _scalar(kl_parts["distill"]) != pytest.approx(
        _scalar(hinge_parts["distill"])
    )


def test_direct_live_segnet_routes_argmax_hinge_objective() -> None:
    target_0 = mx.zeros((2, 4, 4, 3))
    target_1 = mx.ones((2, 4, 4, 3))

    class _LiveTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            arr = np.zeros((idx.shape[0], 4, 4, self.num_classes), dtype=np.float32)
            arr[..., 0] = 4.0
            arr[..., 1] = 3.5
            return mx.array(arr)

        def teacher_logits_for_frames_nhwc01(self, frames):
            arr = np.zeros(
                (frames.shape[0], frames.shape[1], frames.shape[2], self.num_classes),
                dtype=np.float32,
            )
            arr[..., 1] = 4.0
            arr[..., 0] = 3.0
            return mx.array(arr)

    common = {
        "model": ReconstructPairModel(target_0, target_1),
        "target_rgb_0": target_0,
        "target_rgb_1": target_1,
        "num_pairs": 2,
        "forward_convention": "reconstruct_pair_nchw01",
        "scorer_teacher": _LiveTeacher(),
        "segnet_direct_live_distillation_weight": 1.0,
        "segnet_direct_live_class_histogram_weight": 0.0,
        "allow_segnet_only_research": True,
    }
    mse_bundle = RendererBundle(**common, segnet_distillation_objective="kl_t2")
    hinge_bundle = RendererBundle(
        **common,
        segnet_distillation_objective="boundary_argmax_hinge",
        segnet_tau_boundary=0.75,
        segnet_hinge_margin=0.5,
    )

    _mse_total, mse_parts = score_aware_loss(mse_bundle, mx.array([0, 1]))
    _hinge_total, hinge_parts = score_aware_loss(hinge_bundle, mx.array([0, 1]))

    assert _scalar(mse_parts["segnet_direct_live_distill"]) != pytest.approx(
        _scalar(hinge_parts["segnet_direct_live_distill"])
    )
    assert _scalar(hinge_parts["segnet_direct_live_distill"]) > 0.0
    assert _scalar(
        hinge_parts["segnet_direct_live_argmax_disagreement"]
    ) == pytest.approx(1.0)
    assert _scalar(
        hinge_parts["segnet_direct_live_candidate_class_1_fraction"]
    ) == pytest.approx(1.0)
    assert _scalar(
        hinge_parts["segnet_direct_live_target_class_0_fraction"]
    ) == pytest.approx(1.0)


def test_direct_live_segnet_routes_all_pixel_argmax_hinge_objective() -> None:
    target_0 = mx.zeros((2, 4, 4, 3))
    target_1 = mx.ones((2, 4, 4, 3))

    class _LiveTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            arr = np.zeros((idx.shape[0], 4, 4, self.num_classes), dtype=np.float32)
            arr[..., 0] = 4.0
            arr[..., 1] = 3.5
            return mx.array(arr)

        def teacher_logits_for_frames_nhwc01(self, frames):
            arr = np.zeros(
                (frames.shape[0], frames.shape[1], frames.shape[2], self.num_classes),
                dtype=np.float32,
            )
            arr[..., 1] = 4.0
            arr[..., 0] = 3.0
            return mx.array(arr)

    bundle = RendererBundle(
        model=ReconstructPairModel(target_0, target_1),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
        scorer_teacher=_LiveTeacher(),
        segnet_direct_live_distillation_weight=1.0,
        segnet_direct_live_class_histogram_weight=0.0,
        allow_segnet_only_research=True,
        segnet_distillation_objective="argmax_hinge",
        segnet_hinge_margin=0.5,
    )

    _total, parts = score_aware_loss(bundle, mx.array([0, 1]))

    assert _scalar(parts["segnet_direct_live_distill"]) == pytest.approx(1.5)
    assert _scalar(
        parts["segnet_direct_live_argmax_disagreement"]
    ) == pytest.approx(1.0)


def test_direct_live_segnet_material_occupancy_ignores_one_pixel_crumb() -> None:
    target_0 = mx.zeros((1, 4, 4, 3))
    target_1 = mx.ones((1, 4, 4, 3))

    class _LiveTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            arr = np.zeros((idx.shape[0], 4, 4, self.num_classes), dtype=np.float32)
            arr[..., 0] = 4.0
            return mx.array(arr)

        def teacher_logits_for_frames_nhwc01(self, frames):
            arr = np.zeros(
                (frames.shape[0], frames.shape[1], frames.shape[2], self.num_classes),
                dtype=np.float32,
            )
            arr[..., 0] = 4.0
            arr[:, 0, 0, 1] = 5.0
            return mx.array(arr)

    bundle = RendererBundle(
        model=ReconstructPairModel(target_0, target_1),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=1,
        forward_convention="reconstruct_pair_nchw01",
        scorer_teacher=_LiveTeacher(),
        segnet_direct_live_distillation_weight=1.0,
        allow_segnet_only_research=True,
    )

    _total, parts = score_aware_loss(bundle, mx.array([0]))

    assert _scalar(
        parts["segnet_direct_live_candidate_any_occupied_class_fraction"]
    ) == pytest.approx(0.4)
    assert _scalar(
        parts["segnet_direct_live_candidate_occupied_class_fraction"]
    ) == pytest.approx(0.2)
    assert _scalar(
        parts["segnet_direct_live_occupancy_min_class_pixel_count"]
    ) == pytest.approx(2.0)


def test_direct_live_segnet_class_histogram_tether_penalizes_collapse() -> None:
    target_0 = mx.zeros((2, 4, 4, 3))
    target_1 = mx.ones((2, 4, 4, 3))

    class _LiveTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            arr = np.zeros((idx.shape[0], 4, 4, self.num_classes), dtype=np.float32)
            arr[..., 0] = 4.0
            arr[..., 1] = 3.5
            return mx.array(arr)

        def teacher_logits_for_frames_nhwc01(self, frames):
            arr = np.zeros(
                (frames.shape[0], frames.shape[1], frames.shape[2], self.num_classes),
                dtype=np.float32,
            )
            arr[..., 1] = 4.0
            arr[..., 0] = 3.0
            return mx.array(arr)

    common = {
        "model": ReconstructPairModel(target_0, target_1),
        "target_rgb_0": target_0,
        "target_rgb_1": target_1,
        "num_pairs": 2,
        "forward_convention": "reconstruct_pair_nchw01",
        "scorer_teacher": _LiveTeacher(),
        "segnet_direct_live_distillation_weight": 1.0,
        "allow_segnet_only_research": True,
        "segnet_distillation_objective": "argmax_hinge",
        "segnet_hinge_margin": 0.5,
    }
    raw_bundle = RendererBundle(
        **common,
        segnet_direct_live_class_histogram_weight=0.0,
    )
    tethered_bundle = RendererBundle(
        **common,
        segnet_direct_live_class_histogram_weight=1.0,
    )

    _raw_total, raw_parts = score_aware_loss(raw_bundle, mx.array([0, 1]))
    _tethered_total, tethered_parts = score_aware_loss(
        tethered_bundle,
        mx.array([0, 1]),
    )

    assert _scalar(tethered_parts["segnet_direct_live_class_histogram_loss"]) > 0.0
    assert _scalar(
        tethered_parts["segnet_direct_live_class_histogram_cross_entropy"]
    ) > 0.0
    assert _scalar(
        tethered_parts["segnet_direct_live_class_histogram_l1"]
    ) > 0.0
    assert _scalar(
        tethered_parts["segnet_direct_live_class_histogram_loss"]
    ) == pytest.approx(
        _scalar(tethered_parts["segnet_direct_live_class_histogram_cross_entropy"])
        + _scalar(tethered_parts["segnet_direct_live_class_histogram_l1"])
    )
    assert _scalar(
        tethered_parts["segnet_direct_live_distill"]
    ) > _scalar(raw_parts["segnet_direct_live_distill"])
    assert _scalar(
        tethered_parts["segnet_direct_live_target_hist_class_0_fraction"]
    ) == pytest.approx(1.0)
    assert _scalar(
        tethered_parts["segnet_direct_live_candidate_soft_class_1_fraction"]
    ) > _scalar(
        tethered_parts["segnet_direct_live_candidate_soft_class_0_fraction"]
    )


def test_direct_live_segnet_class_balanced_hinge_is_train_time_loss() -> None:
    target_0 = mx.zeros((2, 4, 4, 3))
    target_1 = mx.ones((2, 4, 4, 3))

    class _LiveTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            arr = np.zeros((idx.shape[0], 4, 4, self.num_classes), dtype=np.float32)
            arr[..., 0] = 4.0
            arr[..., 1] = 3.5
            return mx.array(arr)

        def teacher_logits_for_frames_nhwc01(self, frames):
            arr = np.zeros(
                (frames.shape[0], frames.shape[1], frames.shape[2], self.num_classes),
                dtype=np.float32,
            )
            arr[..., 1] = 4.0
            arr[..., 0] = 3.0
            return mx.array(arr)

    common = {
        "model": ReconstructPairModel(target_0, target_1),
        "target_rgb_0": target_0,
        "target_rgb_1": target_1,
        "num_pairs": 2,
        "forward_convention": "reconstruct_pair_nchw01",
        "scorer_teacher": _LiveTeacher(),
        "segnet_direct_live_distillation_weight": 1.0,
        "allow_segnet_only_research": True,
        "segnet_distillation_objective": "argmax_hinge",
        "segnet_hinge_margin": 0.5,
    }
    raw_bundle = RendererBundle(
        **common,
        segnet_direct_live_class_balanced_hinge_weight=0.0,
    )
    balanced_bundle = RendererBundle(
        **common,
        segnet_direct_live_class_balanced_hinge_weight=1.0,
    )

    _raw_total, raw_parts = score_aware_loss(raw_bundle, mx.array([0, 1]))
    _balanced_total, balanced_parts = score_aware_loss(
        balanced_bundle,
        mx.array([0, 1]),
    )

    assert _scalar(
        balanced_parts["segnet_direct_live_class_balanced_hinge_loss"]
    ) == pytest.approx(1.5)
    assert _scalar(
        balanced_parts["segnet_direct_live_class_balanced_hinge_class_0"]
    ) == pytest.approx(1.5)
    assert _scalar(
        balanced_parts["segnet_direct_live_target_occupied_class_fraction"]
    ) == pytest.approx(0.2)
    assert _scalar(
        balanced_parts["segnet_direct_live_distill"]
    ) > _scalar(raw_parts["segnet_direct_live_distill"])


def test_direct_live_segnet_class_balanced_ce_is_sharp_collapse_escape_loss() -> None:
    target_0 = mx.zeros((2, 4, 4, 3))
    target_1 = mx.ones((2, 4, 4, 3))

    class _LiveTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            arr = np.zeros((idx.shape[0], 4, 4, self.num_classes), dtype=np.float32)
            arr[..., 0] = 4.0
            arr[..., 1] = 3.5
            return mx.array(arr)

        def teacher_logits_for_frames_nhwc01(self, frames):
            arr = np.zeros(
                (frames.shape[0], frames.shape[1], frames.shape[2], self.num_classes),
                dtype=np.float32,
            )
            arr[..., 2] = 8.0
            arr[..., 0] = -4.0
            return mx.array(arr)

    common = {
        "model": ReconstructPairModel(target_0, target_1),
        "target_rgb_0": target_0,
        "target_rgb_1": target_1,
        "num_pairs": 2,
        "forward_convention": "reconstruct_pair_nchw01",
        "scorer_teacher": _LiveTeacher(),
        "segnet_direct_live_distillation_weight": 1.0,
        "allow_segnet_only_research": True,
        "segnet_distillation_objective": "argmax_hinge",
        "segnet_hinge_margin": 0.5,
    }
    raw_bundle = RendererBundle(
        **common,
        segnet_direct_live_class_balanced_ce_weight=0.0,
    )
    ce_bundle = RendererBundle(
        **common,
        segnet_direct_live_class_balanced_ce_weight=1.0,
    )

    _raw_total, raw_parts = score_aware_loss(raw_bundle, mx.array([0, 1]))
    _ce_total, ce_parts = score_aware_loss(ce_bundle, mx.array([0, 1]))

    assert _scalar(ce_parts["segnet_direct_live_class_balanced_ce_loss"]) > 10.0
    assert _scalar(
        ce_parts["segnet_direct_live_class_balanced_ce_class_0"]
    ) > 10.0
    assert _scalar(
        ce_parts["segnet_direct_live_class_balanced_ce_target_occupied_class_fraction"]
    ) == pytest.approx(0.2)
    assert _scalar(
        ce_parts["segnet_direct_live_distill"]
    ) > _scalar(raw_parts["segnet_direct_live_distill"])


def test_direct_live_segnet_squared_hinge_prices_far_margin_collapse() -> None:
    target_0 = mx.zeros((2, 4, 4, 3))
    target_1 = mx.ones((2, 4, 4, 3))

    class _LiveTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            arr = np.zeros((idx.shape[0], 4, 4, self.num_classes), dtype=np.float32)
            arr[..., 0] = 4.0
            arr[..., 1] = 3.5
            return mx.array(arr)

        def teacher_logits_for_frames_nhwc01(self, frames):
            arr = np.zeros(
                (frames.shape[0], frames.shape[1], frames.shape[2], self.num_classes),
                dtype=np.float32,
            )
            arr[..., 2] = 8.0
            arr[..., 0] = -4.0
            return mx.array(arr)

    common = {
        "model": ReconstructPairModel(target_0, target_1),
        "target_rgb_0": target_0,
        "target_rgb_1": target_1,
        "num_pairs": 2,
        "forward_convention": "reconstruct_pair_nchw01",
        "scorer_teacher": _LiveTeacher(),
        "segnet_direct_live_distillation_weight": 1.0,
        "segnet_direct_live_base_loss_weight": 0.0,
        "allow_segnet_only_research": True,
        "segnet_distillation_objective": "argmax_hinge",
        "segnet_hinge_margin": 0.5,
    }
    linear_bundle = RendererBundle(
        **common,
        segnet_direct_live_class_balanced_hinge_weight=1.0,
    )
    squared_bundle = RendererBundle(
        **common,
        segnet_direct_live_class_balanced_squared_hinge_weight=1.0,
    )

    _linear_total, linear_parts = score_aware_loss(linear_bundle, mx.array([0, 1]))
    _squared_total, squared_parts = score_aware_loss(
        squared_bundle,
        mx.array([0, 1]),
    )

    linear = _scalar(linear_parts["segnet_direct_live_class_balanced_hinge_loss"])
    squared = _scalar(
        squared_parts["segnet_direct_live_class_balanced_squared_hinge_loss"]
    )
    assert linear == pytest.approx(12.5)
    assert squared == pytest.approx(156.25)
    assert squared > linear * 10.0
    assert _scalar(
        squared_parts["segnet_direct_live_class_balanced_squared_hinge_class_0"]
    ) == pytest.approx(156.25)
    assert _scalar(
        squared_parts[
            "segnet_direct_live_class_balanced_squared_hinge_target_occupied_class_fraction"
        ]
    ) == pytest.approx(0.2)
    assert _scalar(
        squared_parts["segnet_direct_live_distill"]
    ) > _scalar(linear_parts["segnet_direct_live_distill"])


def test_direct_live_segnet_base_loss_weight_zero_keeps_ce_escape_active() -> None:
    target_0 = mx.zeros((2, 4, 4, 3))
    target_1 = mx.ones((2, 4, 4, 3))

    class _LiveTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            arr = np.zeros((idx.shape[0], 4, 4, self.num_classes), dtype=np.float32)
            arr[..., 0] = 4.0
            return mx.array(arr)

        def teacher_logits_for_frames_nhwc01(self, frames):
            arr = np.zeros(
                (frames.shape[0], frames.shape[1], frames.shape[2], self.num_classes),
                dtype=np.float32,
            )
            arr[..., 2] = 8.0
            arr[..., 0] = -4.0
            return mx.array(arr)

    common = {
        "model": ReconstructPairModel(target_0, target_1),
        "target_rgb_0": target_0,
        "target_rgb_1": target_1,
        "num_pairs": 2,
        "forward_convention": "reconstruct_pair_nchw01",
        "scorer_teacher": _LiveTeacher(),
        "segnet_direct_live_distillation_weight": 1.0,
        "segnet_direct_live_class_balanced_ce_weight": 1.0,
        "allow_segnet_only_research": True,
        "segnet_distillation_objective": "argmax_hinge",
        "segnet_hinge_margin": 0.5,
    }
    full_bundle = RendererBundle(**common, segnet_direct_live_base_loss_weight=1.0)
    escape_only_bundle = RendererBundle(
        **common,
        segnet_direct_live_base_loss_weight=0.0,
    )

    _full_total, full_parts = score_aware_loss(full_bundle, mx.array([0, 1]))
    _escape_total, escape_parts = score_aware_loss(
        escape_only_bundle,
        mx.array([0, 1]),
    )

    assert _scalar(full_parts["segnet_direct_live_base_loss_weight"]) == pytest.approx(
        1.0
    )
    assert _scalar(
        escape_parts["segnet_direct_live_base_loss_weight"]
    ) == pytest.approx(0.0)
    assert _scalar(
        escape_parts["segnet_direct_live_class_balanced_ce_loss"]
    ) == pytest.approx(
        _scalar(full_parts["segnet_direct_live_class_balanced_ce_loss"])
    )
    assert _scalar(escape_parts["segnet_direct_live_distill"]) == pytest.approx(
        _scalar(full_parts["segnet_direct_live_distill"])
        - _scalar(full_parts["segnet_direct_live_base_loss"])
    )


def test_direct_live_segnet_subterm_weights_are_curriculum_stageable() -> None:
    target_0, target_1 = _targets()

    class _LiveTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            mx = pytest.importorskip("mlx.core")
            n = int(idx.shape[0])
            arr = np.zeros((n, target_0.shape[1], target_0.shape[2], 5), dtype=np.float32)
            arr[..., 1] = 8.0
            return mx.array(arr)

        def teacher_logits_for_frames_nhwc01(self, frames):
            mx = pytest.importorskip("mlx.core")
            shape = (*tuple(frames.shape[:-1]), 5)
            arr = np.zeros(shape, dtype=np.float32)
            arr[..., 2] = 8.0
            arr[..., 0] = -4.0
            return mx.array(arr)

    bundle = RendererBundle(
        model=ReconstructPairModel(target_0, target_1),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
        scorer_teacher=_LiveTeacher(),
        segnet_direct_live_distillation_weight=1.0,
        segnet_direct_live_base_loss_weight=1.0,
        segnet_direct_live_class_balanced_ce_weight=1.0,
        allow_segnet_only_research=True,
        segnet_distillation_objective="argmax_hinge",
        segnet_hinge_margin=0.5,
    )

    idx = pytest.importorskip("mlx.core").array([0, 1])
    _full_total, full_parts = score_aware_loss(bundle, idx)
    _escape_total, escape_parts = score_aware_loss(
        bundle,
        idx,
        loss_weights={
            "segnet_direct_live_base_loss": 0.0,
            "segnet_direct_live_class_balanced_ce": 1.0,
        },
    )
    _fit_total, fit_parts = score_aware_loss(
        bundle,
        idx,
        loss_weights={
            "segnet_direct_live_base_loss": 1.0,
            "segnet_direct_live_class_balanced_ce": 0.0,
        },
    )

    assert _scalar(
        escape_parts["segnet_direct_live_base_loss_stage_weight"]
    ) == pytest.approx(0.0)
    assert _scalar(
        escape_parts["segnet_direct_live_class_balanced_ce_stage_weight"]
    ) == pytest.approx(1.0)
    assert _scalar(
        fit_parts["segnet_direct_live_base_loss_stage_weight"]
    ) == pytest.approx(1.0)
    assert _scalar(
        fit_parts["segnet_direct_live_class_balanced_ce_stage_weight"]
    ) == pytest.approx(0.0)
    assert _scalar(escape_parts["segnet_direct_live_distill"]) == pytest.approx(
        _scalar(full_parts["segnet_direct_live_distill"])
        - _scalar(full_parts["segnet_direct_live_base_loss"])
    )
    assert _scalar(fit_parts["segnet_direct_live_distill"]) == pytest.approx(
        _scalar(full_parts["segnet_direct_live_distill"])
        - _scalar(full_parts["segnet_direct_live_class_balanced_ce_loss"])
    )


def test_pose_distill_composes_real_pose_teacher_and_head() -> None:
    target_0, target_1 = _targets()

    class _PoseTeacher:
        pose_dims = 6

        def teacher_pose_for_indices(self, idx):
            return mx.zeros((idx.shape[0], self.pose_dims))

    class _PoseHead:
        def __init__(self) -> None:
            self.seen_means = None

        def __call__(self, rgb_0, rgb_1):
            self.seen_means = (_scalar(mx.mean(rgb_0)), _scalar(mx.mean(rgb_1)))
            return mx.ones((rgb_0.shape[0], 6))

    head = _PoseHead()
    bundle = RendererBundle(
        model=ReconstructPairModel(target_0, target_1),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
        pose_distillation_weight=2.0,
        pose_scorer_teacher=_PoseTeacher(),
        learnable_pose_student_head=head,
    )
    total, parts = score_aware_loss(bundle, mx.array([0, 1]))
    assert _scalar(parts["recon"]) < 1e-10
    assert _scalar(parts["pose_distill"]) == pytest.approx(1.0)
    assert _scalar(parts["pose_score_term"]) == pytest.approx(10.0**0.5)
    assert _scalar(total) == pytest.approx(2.0 * (10.0**0.5))
    assert head.seen_means == pytest.approx(
        (_scalar(mx.mean(target_0)), _scalar(mx.mean(target_1)))
    )


def test_pose_distill_huber_keeps_raw_mse_telemetry() -> None:
    target_0, target_1 = _targets()

    class _PoseTeacher:
        pose_dims = 6

        def teacher_pose_for_indices(self, idx):
            return mx.zeros((idx.shape[0], self.pose_dims))

    class _PoseHead:
        def __call__(self, rgb_0, rgb_1):
            return mx.full((rgb_0.shape[0], 6), 10.0)

    bundle = RendererBundle(
        model=ReconstructPairModel(target_0, target_1),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
        pose_distillation_weight=1.0,
        pose_scorer_teacher=_PoseTeacher(),
        learnable_pose_student_head=_PoseHead(),
        pose_distillation_loss="huber",
        pose_distillation_huber_delta=1.0,
    )
    total, parts = score_aware_loss(bundle, mx.array([0, 1]))

    assert _scalar(parts["recon"]) < 1e-10
    assert _scalar(parts["pose_distill_raw_mse"]) == pytest.approx(100.0)
    assert _scalar(parts["pose_distill"]) == pytest.approx(19.0)
    assert _scalar(parts["pose_score_term"]) == pytest.approx((10.0 * 19.0) ** 0.5)
    assert _scalar(total) == pytest.approx((10.0 * 19.0) ** 0.5)


def test_build_mlx_posenet_pair_teacher_uses_upstream_pair_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    observed: dict[str, float] = {}

    class _FakePoseNet:
        def eval(self) -> None:
            observed["eval_called"] = 1.0

        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            observed["input_max"] = float(x.max().item())
            observed["shape_t"] = float(x.shape[1])
            return x

        def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            b = x.shape[0]
            return {"pose": torch.arange(12, dtype=torch.float32).repeat(b, 1)}

    def _fake_load_default_scorers(_upstream_dir: str, *, device: str):
        observed["device_is_cpu"] = 1.0 if device == "cpu" else 0.0
        return _FakePoseNet(), object()

    import tac.scorer

    monkeypatch.setattr(tac.scorer, "load_default_scorers", _fake_load_default_scorers)
    target_0 = mx.ones((1, 384, 512, 3))
    target_1 = mx.zeros((1, 384, 512, 3))
    bundle = RendererBundle(
        model=object(),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=1,
        pose_dims=6,
    )

    cache = build_mlx_posenet_pair_teacher(bundle, upstream_dir="upstream", device="cpu")

    assert observed["eval_called"] == 1.0
    assert observed["device_is_cpu"] == 1.0
    assert observed["input_max"] == 255.0
    assert observed["shape_t"] == 2.0
    assert cache.num_pairs == 1
    assert cache.pose_dims == 6
    assert tuple(cache.per_dim_scale.shape) == (6,)
    np.testing.assert_allclose(np.array(cache.per_dim_scale), np.full((6,), 1e-3))
    np.testing.assert_allclose(
        np.array(cache.teacher_pose_for_indices(mx.array([0]))),
        np.arange(6, dtype=np.float32).reshape(1, 6),
    )


def test_numpy_portable_inflate_gate_uses_fail_closed_error_type() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        good = root / "inflate_good.py"
        bad = root / "inflate_bad.py"
        good.write_text("import numpy as np\nx = np.array([1])\n", encoding="utf-8")
        bad.write_text("import torch\n", encoding="utf-8")
        result = assert_numpy_portable_inflate(good)
        assert result["numpy_portable"] is True
        assert "numpy" in result["import_roots"]
        with pytest.raises(MlxScoreAwareHarnessError, match="forbidden"):
            assert_numpy_portable_inflate(bad)
