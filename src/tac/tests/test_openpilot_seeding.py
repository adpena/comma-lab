# SPDX-License-Identifier: MIT
"""Tests for tac.openpilot_seeding (Lane OS-A core module).

These tests pin the contracts of the supercombo loader, the YUV preprocessing,
the affine calibration map, and the masks-only fallback. The supercombo ONNX
model is ~30 MB — tests that need a real session monkeypatch a fake one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from tac.lane_mark_pose import POSENET_DIM0_MEAN
from tac.openpilot_seeding import (
    OPENPILOT_SUPERCOMBO_DEFAULT_PATH,
    OPENPILOT_SUPERCOMBO_URL,
    SUPERCOMBO_INPUT_NAME,
    SUPERCOMBO_INPUT_SHAPE,
    SUPERCOMBO_POSE_HEAD_END,
    SUPERCOMBO_POSE_HEAD_START,
    SupercomboUnavailable,
    _build_supercombo_extra_inputs,
    _frames_to_supercombo_yuv,
    fallback_seed_from_masks,
    infer_pose_from_video,
    load_supercombo_model,
    seed_pose_tto,
)


# ── Constants and metadata ────────────────────────────────────────────


def test_supercombo_url_is_github_raw() -> None:
    """The download URL must point at openpilot's master branch."""
    assert OPENPILOT_SUPERCOMBO_URL.startswith("https://"), (
        "URL must be https for integrity"
    )
    assert "commaai/openpilot" in OPENPILOT_SUPERCOMBO_URL, (
        "URL must point to the commaai/openpilot repo"
    )
    assert OPENPILOT_SUPERCOMBO_URL.endswith("supercombo.onnx"), (
        "URL must end at the .onnx model file"
    )


def test_default_path_is_workspace_relative() -> None:
    """Default path must match the remote bootstrap convention."""
    assert OPENPILOT_SUPERCOMBO_DEFAULT_PATH == (
        "/workspace/openpilot/models/supercombo.onnx"
    ), "default path must match the Vast.ai workspace convention"


def test_pose_head_slice_is_6d() -> None:
    """The pose head slice must yield exactly 6 floats."""
    assert SUPERCOMBO_POSE_HEAD_END - SUPERCOMBO_POSE_HEAD_START == 6, (
        "pose head must be 6-dim (3 trans + 3 rot)"
    )


def test_supercombo_input_shape() -> None:
    """The input shape contract is (1, 12, 128, 256)."""
    assert SUPERCOMBO_INPUT_SHAPE == (1, 12, 128, 256)
    assert SUPERCOMBO_INPUT_NAME == "input_imgs"


# ── load_supercombo_model failure modes ───────────────────────────────


def test_load_supercombo_missing_file_raises_unavailable(tmp_path: Path) -> None:
    """Missing .onnx file → SupercomboUnavailable with a download hint."""
    with pytest.raises(SupercomboUnavailable, match="not found"):
        load_supercombo_model(
            tmp_path / "does_not_exist.onnx", torch.device("cpu"),
        )


def test_load_supercombo_missing_onnxruntime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If onnxruntime isn't importable → SupercomboUnavailable."""
    fake_model = tmp_path / "fake.onnx"
    fake_model.write_bytes(b"not a real onnx file")

    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "onnxruntime":
            raise ImportError("onnxruntime not installed (test)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SupercomboUnavailable, match="onnxruntime"):
        load_supercombo_model(fake_model, torch.device("cpu"))


# ── _frames_to_supercombo_yuv ─────────────────────────────────────────


def test_yuv_preprocess_shape_and_range() -> None:
    """YUV output must be (1, 12, 128, 256) float in [0, 1]."""
    h, w = 384, 512
    f0 = torch.zeros(h, w, 3, dtype=torch.uint8)
    f1 = torch.full((h, w, 3), 200, dtype=torch.uint8)
    out = _frames_to_supercombo_yuv(f1, f0)
    assert out.shape == (1, 12, 128, 256), (
        f"expected (1, 12, 128, 256), got {tuple(out.shape)}"
    )
    assert out.dtype == torch.float32
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0 + 1e-4


def test_yuv_preprocess_dtype_guard() -> None:
    """Non-uint8 input must raise."""
    f = torch.zeros(384, 512, 3, dtype=torch.float32)
    with pytest.raises(TypeError, match="uint8"):
        _frames_to_supercombo_yuv(f, f)


def test_yuv_preprocess_shape_mismatch_guard() -> None:
    f0 = torch.zeros(384, 512, 3, dtype=torch.uint8)
    f1 = torch.zeros(192, 256, 3, dtype=torch.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        _frames_to_supercombo_yuv(f1, f0)


# ── seed_pose_tto: affine calibration ────────────────────────────────


def test_seed_pose_tto_shape_validation() -> None:
    """Wrong shape → ValueError."""
    with pytest.raises(ValueError, match=r"\(N, 6\)"):
        seed_pose_tto(torch.zeros(10, 5))


def test_seed_pose_tto_no_baseline_uses_lane_mark_constants() -> None:
    """Without baseline AND with explicit calibration mode, dim 0 lands near
    POSENET_DIM0_MEAN and dims 1-5 are zero.

    V2 changed the default from ``scale_to_match=True`` (which fell through
    to lane_mark constants when no baseline was given) to ``mode='none'``
    (passthrough — let TTO learn the scale). To exercise the lane-mark
    constants code path we must now opt in with ``scale_to_match=True``
    (V1 alias for ``mode='linear'``).
    """
    raw = torch.randn(100, 6) * 0.5  # arbitrary supercombo-like output
    out = seed_pose_tto(raw, baseline_poses=None, scale_to_match=True)
    assert out.shape == (100, 6)
    # Dim 0 should be centered on POSENET_DIM0_MEAN.
    assert abs(out[:, 0].mean().item() - POSENET_DIM0_MEAN) < 1.0, (
        f"dim 0 mean {out[:, 0].mean().item():.4f} should be near "
        f"POSENET_DIM0_MEAN={POSENET_DIM0_MEAN}"
    )
    # Dims 1-5 must be exactly zero.
    for d in range(1, 6):
        assert (out[:, d] == 0.0).all(), f"dim {d} should be zero"


def test_seed_pose_tto_no_baseline_no_scale_returns_raw() -> None:
    """V2 default (mode='none', no baseline) returns raw poses unchanged."""
    raw = torch.randn(100, 6) * 0.5
    out = seed_pose_tto(raw, baseline_poses=None)  # V2 default: mode='none'
    assert torch.allclose(out, raw)


def test_seed_pose_tto_baseline_calibration_matches_distribution() -> None:
    """With baseline + scale_to_match=True, output dim stats match baseline."""
    torch.manual_seed(0)
    # Raw supercombo: Gaussian centered at (1.5, 0.1, 0.0, 0.0, 0.0, 0.0).
    raw = torch.randn(600, 6) * 0.3
    raw[:, 0] += 1.5
    raw[:, 1] += 0.1
    # Baseline: PoseNet-scale, dim 0 mean ~31.295.
    baseline = torch.randn(600, 6) * 1.265
    baseline[:, 0] += 31.295
    baseline[:, 1] += 0.0  # dims 1-5 are near-zero per Yousfi geometric analysis

    out = seed_pose_tto(raw, baseline_poses=baseline, scale_to_match=True)
    assert out.shape == (600, 6)
    # Per-dim stats must match baseline (up to numerical tolerance).
    for d in range(6):
        out_mean = out[:, d].mean().item()
        out_std = out[:, d].std().item()
        base_mean = baseline[:, d].mean().item()
        base_std = baseline[:, d].std().item()
        assert abs(out_mean - base_mean) < 0.01, (
            f"dim {d}: out mean {out_mean:.4f} != baseline mean {base_mean:.4f}"
        )
        assert abs(out_std - base_std) < 0.01, (
            f"dim {d}: out std {out_std:.4f} != baseline std {base_std:.4f}"
        )


def test_seed_pose_tto_no_scale_returns_raw() -> None:
    """scale_to_match=False with baseline → return raw poses unchanged."""
    raw = torch.randn(50, 6)
    baseline = torch.randn(50, 6) + 30.0
    out = seed_pose_tto(raw, baseline_poses=baseline, scale_to_match=False)
    assert torch.allclose(out, raw)


def test_seed_pose_tto_baseline_shape_validation() -> None:
    """Wrong baseline shape → ValueError."""
    raw = torch.randn(50, 6)
    bad = torch.randn(50, 5)
    with pytest.raises(ValueError, match=r"baseline_poses"):
        seed_pose_tto(raw, baseline_poses=bad, scale_to_match=True)


def test_seed_pose_tto_degenerate_dim_uses_baseline_mean() -> None:
    """If a raw dim has zero variance, the calibrated dim is the baseline mean."""
    raw = torch.zeros(50, 6)  # all dims constant
    baseline = torch.randn(50, 6) + 5.0
    out = seed_pose_tto(raw, baseline_poses=baseline, scale_to_match=True)
    for d in range(6):
        assert torch.allclose(
            out[:, d], torch.full((50,), baseline[:, d].mean().item()),
            atol=1e-5,
        )


# ── fallback_seed_from_masks ─────────────────────────────────────────


def test_fallback_seed_from_masks_shape() -> None:
    """The fallback delegates to lane_mark_pose and returns (N//2, 6)."""
    n = 12
    h, w = 384, 512
    masks = torch.zeros(n, h, w, dtype=torch.long)
    out = fallback_seed_from_masks(masks)
    assert out.shape == (n // 2, 6)
    assert out.dtype == torch.float32


# ── infer_pose_from_video integration with a fake supercombo ──────────


class _FakeSupercombo:
    """A fake onnxruntime InferenceSession for unit testing.

    Returns a 6504-element output where the pose head slice
    [SUPERCOMBO_POSE_HEAD_START:SUPERCOMBO_POSE_HEAD_END] is filled with
    a deterministic per-call value so we can verify the slice extraction.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def get_inputs(self) -> list[Any]:
        class _Spec:
            def __init__(self, name: str, shape: tuple[int, ...]) -> None:
                self.name = name
                self.shape = shape

        return [
            _Spec(SUPERCOMBO_INPUT_NAME, SUPERCOMBO_INPUT_SHAPE),
            _Spec("desire", (1, 100, 8)),
            _Spec("traffic_convention", (1, 2)),
        ]

    def run(self, _output_names: list[str] | None, _feed: dict[str, Any]) -> list[Any]:
        import numpy as np

        out = np.zeros((1, 6504), dtype=np.float32)
        # Fill the pose head with [0, 1, 2, 3, 4, 5] so we can verify slicing.
        out[0, SUPERCOMBO_POSE_HEAD_START:SUPERCOMBO_POSE_HEAD_END] = (
            np.arange(6, dtype=np.float32) + self.call_count * 0.1
        )
        self.call_count += 1
        return [out]


def test_infer_pose_from_video_with_fake_session(tmp_path: Path) -> None:
    """End-to-end test of pose extraction with a fake supercombo + fake video."""
    av = pytest.importorskip("av")

    # Build a tiny 4-frame video.
    video_path = tmp_path / "tiny.mkv"
    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("h264", rate=20)
    stream.width = 64
    stream.height = 48
    stream.pix_fmt = "yuv420p"
    import numpy as np

    for i in range(4):
        arr = np.full((48, 64, 3), i * 30, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()

    fake = _FakeSupercombo()
    poses = infer_pose_from_video(fake, video_path, n_frames=4, device=torch.device("cpu"))
    assert poses.shape == (2, 6), (
        f"expected (2, 6) for 4 frames, got {tuple(poses.shape)}"
    )
    # First call's pose head: [0, 1, 2, 3, 4, 5]; second: [0.1, 1.1, ...].
    assert poses[0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert poses[1, 0] == pytest.approx(0.1)
    assert poses[1, 5] == pytest.approx(5.1)


def test_infer_pose_from_video_missing_video(tmp_path: Path) -> None:
    """Missing video → SupercomboUnavailable."""
    fake = _FakeSupercombo()
    with pytest.raises(SupercomboUnavailable, match="video not found"):
        infer_pose_from_video(fake, tmp_path / "nope.mkv", n_frames=4)


# ── _build_supercombo_extra_inputs ───────────────────────────────────


def test_build_supercombo_extra_inputs_excludes_main_input() -> None:
    fake = _FakeSupercombo()
    extras = _build_supercombo_extra_inputs(fake)
    assert SUPERCOMBO_INPUT_NAME not in extras, (
        "main input must be omitted (caller fills it separately)"
    )
    assert "desire" in extras
    assert "traffic_convention" in extras


def test_build_supercombo_extra_inputs_zero_filled() -> None:
    fake = _FakeSupercombo()
    extras = _build_supercombo_extra_inputs(fake)
    for name, arr in extras.items():
        assert (arr == 0).all(), f"extra {name} must be zero-filled"
