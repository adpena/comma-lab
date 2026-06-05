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
