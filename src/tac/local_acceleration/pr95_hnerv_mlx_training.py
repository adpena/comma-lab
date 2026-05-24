# SPDX-License-Identifier: MIT
"""Source-faithful PR95/HNeRV training preprocessing in native MLX.

The decoder/archive MLX port is intentionally kept in
``tac.local_acceleration.pr95_hnerv_mlx``.  This module owns the differentiable
training-side preprocessing that PR95 made score-relevant: bicubic camera
roundtrip, uint8 STE, bilinear return to scorer resolution, and RGB->YUV6 with
gradient reachability.

These helpers are local training signal only.  They do not claim score,
promotion, rank/kill, dispatch, or exact-eval authority.
"""

from __future__ import annotations

import hashlib
import importlib.util
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tac.local_acceleration.pr95_hnerv_mlx import (
    FALSE_AUTHORITY,
    Pr95HNeRVMlxError,
    mx,
    require_mlx,
)

CAMERA_HW: tuple[int, int] = (874, 1164)
SCORER_HW: tuple[int, int] = (384, 512)
SOURCE_FAITHFUL_PREPROCESS_SCHEMA = "pr95_hnerv_mlx_source_faithful_preprocess_smoke_v1"
SOURCE_VIDEO_PREPROCESS_SCHEMA = "pr95_hnerv_mlx_source_video_preprocess_smoke_v1"
GRAD_PROBE_SCHEMA = "pr95_hnerv_mlx_preprocess_gradient_probe_v1"
FrameReader = Callable[[Sequence[int]], np.ndarray]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bicubic_resize_to_camera_nhwc(
    rgb_nhwc: Any,
    *,
    camera_hw: tuple[int, int] = CAMERA_HW,
) -> Any:
    """Resize NHWC RGB to camera resolution with PyTorch-style bicubic semantics."""

    return resize_nhwc_align_corners_false(rgb_nhwc, size=camera_hw, mode="bicubic")


def bilinear_eval_roundtrip_downsample_nhwc(
    rgb_nhwc: Any,
    *,
    output_hw: tuple[int, int] = SCORER_HW,
) -> Any:
    """Resize NHWC RGB to scorer resolution with align_corners=False bilinear."""

    return resize_nhwc_align_corners_false(rgb_nhwc, size=output_hw, mode="bilinear")


def apply_eval_roundtrip_nhwc(
    rgb_nhwc: Any,
    *,
    camera_hw: tuple[int, int] = CAMERA_HW,
    output_hw: tuple[int, int] | None = None,
    simulate_resize: bool = True,
    simulate_uint8: bool = True,
    ste_round: bool = True,
) -> Any:
    """Apply PR95's train-time eval roundtrip to ``(..., H, W, 3)`` MLX RGB.

    The sequence matches the PyTorch oracle in
    ``tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training``:
    bicubic up to camera resolution, bilinear down to scorer resolution, then
    uint8 clamp/round through an STE.  Leading dimensions are preserved.
    """

    require_mlx()
    flat, original_shape = _flatten_rgb_nhwc(rgb_nhwc)
    target_hw = output_hw or (int(original_shape[-3]), int(original_shape[-2]))
    out = flat
    if simulate_resize:
        out = resize_nhwc_align_corners_false(out, size=camera_hw, mode="bicubic")
        out = resize_nhwc_align_corners_false(out, size=target_hw, mode="bilinear")
    if simulate_uint8:
        out = _uint8_ste_mlx(out) if ste_round else mx.clip(out, 0.0, 255.0)  # type: ignore[union-attr]
    return _restore_rgb_nhwc(out, original_shape, target_hw)


def rgb_to_yuv6_mlx(rgb_nhwc: Any) -> Any:
    """Autograd-preserving BT.601 RGB->YUV6 for ``(..., H, W, 3)`` MLX tensors."""

    require_mlx()
    _validate_rgb_nhwc(rgb_nhwc)
    height = int(rgb_nhwc.shape[-3])
    width = int(rgb_nhwc.shape[-2])
    h2 = height // 2
    w2 = width // 2
    if h2 < 1 or w2 < 1:
        raise Pr95HNeRVMlxError(
            f"rgb_to_yuv6_mlx requires spatial dims at least 2x2; got {(height, width)}"
        )
    rgb = rgb_nhwc[..., : 2 * h2, : 2 * w2, :]
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]

    y = mx.clip(red * 0.299 + green * 0.587 + blue * 0.114, 0.0, 255.0)  # type: ignore[union-attr]
    u = mx.clip((blue - y) / 1.772 + 128.0, 0.0, 255.0)  # type: ignore[union-attr]
    v = mx.clip((red - y) / 1.402 + 128.0, 0.0, 255.0)  # type: ignore[union-attr]

    u_sub = (
        u[..., 0::2, 0::2]
        + u[..., 1::2, 0::2]
        + u[..., 0::2, 1::2]
        + u[..., 1::2, 1::2]
    ) * 0.25
    v_sub = (
        v[..., 0::2, 0::2]
        + v[..., 1::2, 0::2]
        + v[..., 0::2, 1::2]
        + v[..., 1::2, 1::2]
    ) * 0.25

    return mx.stack(  # type: ignore[union-attr]
        [
            y[..., 0::2, 0::2],
            y[..., 1::2, 0::2],
            y[..., 0::2, 1::2],
            y[..., 1::2, 1::2],
            u_sub,
            v_sub,
        ],
        axis=-1,
    )


def pr95_pair_frame_indices(pair_indices: Sequence[int]) -> list[int]:
    """Return ordered frame indices for PR95 two-frame pair indices."""

    pairs = [int(index) for index in pair_indices]
    if not pairs:
        raise Pr95HNeRVMlxError("at least one PR95 pair index is required")
    if any(index < 0 for index in pairs):
        raise Pr95HNeRVMlxError(f"PR95 pair indices must be non-negative: {pairs}")
    frame_indices: list[int] = []
    seen: set[int] = set()
    for pair_index in pairs:
        for frame_index in (2 * pair_index, 2 * pair_index + 1):
            if frame_index not in seen:
                frame_indices.append(frame_index)
                seen.add(frame_index)
    return frame_indices


def load_upstream_video_frames_nhwc(
    video_path: str | Path,
    frame_indices: Sequence[int],
    *,
    upstream_dir: str | Path,
) -> np.ndarray:
    """Decode selected camera-resolution RGB frames via upstream CPU semantics."""

    requested = [int(index) for index in frame_indices]
    if not requested:
        raise Pr95HNeRVMlxError("at least one frame index is required")
    if any(index < 0 for index in requested):
        raise Pr95HNeRVMlxError(f"frame indices must be non-negative: {requested}")
    requested_set = set(requested)
    video = Path(video_path)
    if not video.is_file():
        raise Pr95HNeRVMlxError(f"source video not found: {video}")
    upstream = Path(upstream_dir)
    if not upstream.is_dir():
        raise Pr95HNeRVMlxError(f"upstream dir not found: {upstream}")

    frame_utils_path = upstream / "frame_utils.py"
    if not frame_utils_path.is_file():
        raise Pr95HNeRVMlxError(
            f"upstream frame_utils.py not found: {frame_utils_path}"
        )
    try:
        import av
    except Exception as exc:  # pragma: no cover - dependency guard.
        raise Pr95HNeRVMlxError("PyAV is required to decode PR95 source video") from exc
    module_name = (
        "pr95_upstream_frame_utils_"
        + hashlib.sha256(str(frame_utils_path.resolve()).encode("utf-8")).hexdigest()[
            :12
        ]
    )
    spec = importlib.util.spec_from_file_location(module_name, frame_utils_path)
    if spec is None or spec.loader is None:
        raise Pr95HNeRVMlxError(
            f"unable to load upstream frame_utils.py: {frame_utils_path}"
        )
    frame_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frame_utils)
    try:
        width, height = (int(dim) for dim in frame_utils.camera_size)
        decoded: dict[int, np.ndarray] = {}
        container = av.open(str(video))
        try:
            stream = container.streams.video[0]
            max_requested = max(requested_set)
            for frame_number, frame in enumerate(container.decode(stream)):
                if frame_number not in requested_set:
                    if frame_number > max_requested and len(decoded) == len(
                        requested_set
                    ):
                        break
                    continue
                rgb = frame_utils.yuv420_to_rgb(frame).numpy()
                if rgb.shape != (height, width, 3):
                    raise Pr95HNeRVMlxError(
                        f"decoded frame {frame_number} shape {rgb.shape}; "
                        f"expected {(height, width, 3)}"
                    )
                decoded[frame_number] = rgb.astype(np.float32, copy=False)
                if len(decoded) == len(requested_set):
                    break
        finally:
            container.close()
    except Pr95HNeRVMlxError:
        raise
    except Exception as exc:
        raise Pr95HNeRVMlxError(
            f"failed to decode PR95 source video frames from {video}"
        ) from exc
    missing = [index for index in requested if index not in decoded]
    if missing:
        raise Pr95HNeRVMlxError(f"source video missing requested frame(s): {missing}")
    return np.stack([decoded[index] for index in requested], axis=0)


def load_pr95_source_pairs_nhwc(
    video_path: str | Path,
    *,
    pair_indices: Sequence[int],
    upstream_dir: str | Path,
    frame_reader: FrameReader | None = None,
) -> np.ndarray:
    """Return source contest pairs as ``(pairs, 2, H, W, 3)`` float32 NHWC."""

    pairs = [int(index) for index in pair_indices]
    frame_indices = pr95_pair_frame_indices(pairs)
    frames = (
        np.asarray(frame_reader(frame_indices), dtype=np.float32)
        if frame_reader is not None
        else load_upstream_video_frames_nhwc(
            video_path,
            frame_indices,
            upstream_dir=upstream_dir,
        )
    )
    if frames.shape[0] != len(frame_indices) or frames.ndim != 4 or frames.shape[-1] != 3:
        raise Pr95HNeRVMlxError(
            "source frame reader must return (n_frames, H, W, 3); "
            f"got {frames.shape}, expected n_frames={len(frame_indices)}"
        )
    by_index = {frame_index: frames[offset] for offset, frame_index in enumerate(frame_indices)}
    return np.stack(
        [
            np.stack([by_index[2 * pair_index], by_index[2 * pair_index + 1]], axis=0)
            for pair_index in pairs
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def pr95_source_pairs_to_scorer_targets_mlx(
    source_pairs_nhwc: Any,
    *,
    output_hw: tuple[int, int] = SCORER_HW,
) -> tuple[Any, Any]:
    """Downsample source camera pairs to scorer resolution and YUV6 targets."""

    require_mlx()
    _validate_rgb_nhwc(source_pairs_nhwc)
    shape = tuple(int(dim) for dim in source_pairs_nhwc.shape)
    flat = mx.reshape(source_pairs_nhwc, (-1, shape[-3], shape[-2], shape[-1]))  # type: ignore[union-attr]
    scorer_rgb = bilinear_eval_roundtrip_downsample_nhwc(flat, output_hw=output_hw)
    scorer_rgb = mx.reshape(  # type: ignore[union-attr]
        scorer_rgb,
        (*shape[:-3], int(output_hw[0]), int(output_hw[1]), 3),
    )
    return scorer_rgb, rgb_to_yuv6_mlx(scorer_rgb)


def resize_nhwc_align_corners_false(
    x: Any,
    *,
    size: tuple[int, int],
    mode: str,
) -> Any:
    """Resize 4D NHWC tensors with align_corners=False coordinate semantics."""

    require_mlx()
    if len(x.shape) != 4:
        raise Pr95HNeRVMlxError(f"expected 4D NHWC tensor, got shape {x.shape}")
    out_h, out_w = _positive_hw(size)
    mode = mode.strip().lower()
    if mode not in {"bilinear", "bicubic"}:
        raise Pr95HNeRVMlxError(f"unsupported resize mode: {mode}")
    y = _resize_axis_nhwc(x, axis=1, out_size=out_h, mode=mode)
    return _resize_axis_nhwc(y, axis=2, out_size=out_w, mode=mode)


def pr95_mlx_preprocess_grad_probe(
    *,
    input_shape: Sequence[int] = (1, 2, 16, 20, 3),
    camera_hw: tuple[int, int] = (23, 29),
    seed: int = 0,
    min_abs_gradient: float = 1.0e-8,
) -> dict[str, Any]:
    """Return a small MLX gradient-reachability proof for roundtrip+YUV6."""

    require_mlx()
    shape = tuple(int(dim) for dim in input_shape)
    if len(shape) < 4 or shape[-1] != 3:
        raise Pr95HNeRVMlxError(f"input_shape must end in H,W,3; got {shape}")
    rng = np.random.default_rng(int(seed))
    sample = rng.uniform(24.0, 232.0, size=shape).astype(np.float32)
    x = mx.array(sample)  # type: ignore[union-attr]

    def loss_fn(value: Any) -> Any:
        rounded = apply_eval_roundtrip_nhwc(value, camera_hw=camera_hw)
        yuv6 = rgb_to_yuv6_mlx(rounded)
        return mx.mean(yuv6)  # type: ignore[union-attr]

    started = time.perf_counter()
    grad = mx.grad(loss_fn)(x)  # type: ignore[union-attr]
    mx.eval(grad)  # type: ignore[union-attr]
    elapsed = time.perf_counter() - started
    grad_np = np.asarray(grad)
    max_abs = float(np.max(np.abs(grad_np))) if grad_np.size else 0.0
    nonzero = int(np.count_nonzero(np.abs(grad_np) > float(min_abs_gradient)))
    return {
        "schema": GRAD_PROBE_SCHEMA,
        "input_shape": list(shape),
        "camera_hw": [int(camera_hw[0]), int(camera_hw[1])],
        "max_abs_gradient": max_abs,
        "nonzero_gradient_count": nonzero,
        "min_abs_gradient": float(min_abs_gradient),
        "gradient_reachable": max_abs > float(min_abs_gradient) and nonzero > 0,
        "elapsed_seconds": elapsed,
        **FALSE_AUTHORITY,
    }


def run_pr95_mlx_source_faithful_smoke(
    *,
    input_shape: Sequence[int] = (1, 2, 384, 512, 3),
    camera_hw: tuple[int, int] = CAMERA_HW,
    seed: int = 0,
    include_gradient_probe: bool = True,
    gradient_probe_shape: Sequence[int] = (1, 2, 16, 20, 3),
) -> dict[str, Any]:
    """Run a local MLX preprocessing smoke and return a false-authority manifest."""

    require_mlx()
    shape = tuple(int(dim) for dim in input_shape)
    if len(shape) < 4 or shape[-1] != 3:
        raise Pr95HNeRVMlxError(f"input_shape must end in H,W,3; got {shape}")
    rng = np.random.default_rng(int(seed))
    rgb = mx.array(rng.uniform(0.0, 255.0, size=shape).astype(np.float32))  # type: ignore[union-attr]
    started = time.perf_counter()
    rounded = apply_eval_roundtrip_nhwc(rgb, camera_hw=camera_hw)
    yuv6 = rgb_to_yuv6_mlx(rounded)
    mx.eval(rounded, yuv6)  # type: ignore[union-attr]
    elapsed = time.perf_counter() - started
    grad_probe = (
        pr95_mlx_preprocess_grad_probe(
            input_shape=gradient_probe_shape,
            camera_hw=(
                min(max(int(camera_hw[0]), 3), 37),
                min(max(int(camera_hw[1]), 3), 43),
            ),
            seed=seed,
        )
        if include_gradient_probe
        else None
    )
    gradient_reachable = (
        grad_probe is not None and grad_probe.get("gradient_reachable") is True
    )
    blockers = [
        "pr95_source_video_loader_not_wired_to_mlx_preprocess_smoke",
        "pr95_scorer_loss_not_wired_to_mlx_preprocess_smoke",
        "pr95_training_loop_not_yet_source_faithful",
        "requires_pytorch_export_forward_parity_on_source_checkpoint",
        "requires_byte_closed_contest_archive_export",
        "requires_exact_cpu_cuda_auth_eval_before_score_claim",
    ]
    if not gradient_reachable:
        blockers.append("pr95_mlx_preprocess_gradient_not_reachable")
    return {
        "schema": SOURCE_FAITHFUL_PREPROCESS_SCHEMA,
        "lane_id": "lane_pr95_hnerv_mlx_reproduction",
        "input_shape": list(shape),
        "camera_hw": [int(camera_hw[0]), int(camera_hw[1])],
        "roundtrip_output_shape": [int(dim) for dim in rounded.shape],
        "yuv6_output_shape": [int(dim) for dim in yuv6.shape],
        "elapsed_seconds": elapsed,
        "source_faithful_preprocess_ready": gradient_reachable,
        "gradient_probe": grad_probe,
        "exact_readiness_refusal": {
            "ready": False,
            "blockers": blockers,
        },
        **FALSE_AUTHORITY,
    }


def run_pr95_mlx_source_video_preprocess_smoke(
    *,
    video_path: str | Path,
    upstream_dir: str | Path,
    pair_indices: Sequence[int] = (0,),
    output_hw: tuple[int, int] = SCORER_HW,
    include_gradient_probe: bool = True,
    gradient_probe_shape: Sequence[int] = (1, 2, 16, 20, 3),
    frame_reader: FrameReader | None = None,
) -> dict[str, Any]:
    """Decode real PR95 source pairs and build scorer-resolution MLX targets."""

    require_mlx()
    started = time.perf_counter()
    pairs = [int(index) for index in pair_indices]
    frame_indices = pr95_pair_frame_indices(pairs)
    source_pairs = load_pr95_source_pairs_nhwc(
        video_path,
        pair_indices=pairs,
        upstream_dir=upstream_dir,
        frame_reader=frame_reader,
    )
    source_pairs_mlx = mx.array(source_pairs)  # type: ignore[union-attr]
    scorer_rgb, yuv6 = pr95_source_pairs_to_scorer_targets_mlx(
        source_pairs_mlx,
        output_hw=output_hw,
    )
    mx.eval(scorer_rgb, yuv6)  # type: ignore[union-attr]
    elapsed = time.perf_counter() - started
    grad_probe = (
        pr95_mlx_preprocess_grad_probe(
            input_shape=gradient_probe_shape,
            camera_hw=(
                min(max(int(source_pairs.shape[-3]), 3), 37),
                min(max(int(source_pairs.shape[-2]), 3), 43),
            ),
            seed=sum(pairs) if pairs else 0,
        )
        if include_gradient_probe
        else None
    )
    gradient_reachable = (
        grad_probe is not None and grad_probe.get("gradient_reachable") is True
    )
    video = Path(video_path)
    source_hash = _sha256_file(video) if video.is_file() and frame_reader is None else None
    blockers = [
        "pr95_decoder_training_loop_not_wired_to_source_video_preprocess",
        "pr95_scorer_loss_not_wired_to_mlx_source_video_preprocess",
        "pr95_training_loop_not_yet_source_faithful",
        "requires_pytorch_export_forward_parity_on_source_checkpoint",
        "requires_byte_closed_contest_archive_export",
        "requires_exact_cpu_cuda_auth_eval_before_score_claim",
    ]
    if not gradient_reachable:
        blockers.append("pr95_mlx_preprocess_gradient_not_reachable")
    return {
        "schema": SOURCE_VIDEO_PREPROCESS_SCHEMA,
        "lane_id": "lane_pr95_hnerv_mlx_reproduction",
        "video_path": str(video),
        "video_sha256": source_hash,
        "upstream_dir": str(upstream_dir),
        "pair_indices": pairs,
        "frame_indices": frame_indices,
        "source_frame_pair_shape": [int(dim) for dim in source_pairs.shape],
        "scorer_rgb_shape": [int(dim) for dim in scorer_rgb.shape],
        "yuv6_output_shape": [int(dim) for dim in yuv6.shape],
        "source_video_loader_ready": True,
        "source_video_preprocess_ready": gradient_reachable,
        "frame_reader_kind": "injected" if frame_reader is not None else "upstream_pyav_cpu",
        "elapsed_seconds": elapsed,
        "gradient_probe": grad_probe,
        "exact_readiness_refusal": {
            "ready": False,
            "blockers": blockers,
        },
        **FALSE_AUTHORITY,
    }


def _resize_axis_nhwc(x: Any, *, axis: int, out_size: int, mode: str) -> Any:
    in_size = int(x.shape[axis])
    if in_size == out_size:
        return x
    indices, weights = _resize_indices_weights(
        in_size=in_size,
        out_size=out_size,
        mode=mode,
    )
    taps = int(indices.shape[1])
    gathered = mx.take(x, mx.reshape(indices, (-1,)), axis=axis)  # type: ignore[union-attr]
    if axis == 1:
        gathered = mx.reshape(  # type: ignore[union-attr]
            gathered,
            (int(x.shape[0]), out_size, taps, int(x.shape[2]), int(x.shape[3])),
        )
        return mx.sum(gathered * mx.reshape(weights, (1, out_size, taps, 1, 1)), axis=2)  # type: ignore[union-attr]
    if axis == 2:
        gathered = mx.reshape(  # type: ignore[union-attr]
            gathered,
            (int(x.shape[0]), int(x.shape[1]), out_size, taps, int(x.shape[3])),
        )
        return mx.sum(gathered * mx.reshape(weights, (1, 1, out_size, taps, 1)), axis=3)  # type: ignore[union-attr]
    raise Pr95HNeRVMlxError(f"unsupported NHWC resize axis: {axis}")


def _resize_indices_weights(*, in_size: int, out_size: int, mode: str) -> tuple[Any, Any]:
    scale = float(in_size) / float(out_size)
    out = mx.arange(out_size, dtype=mx.float32)  # type: ignore[union-attr]
    real = (out + 0.5) * scale - 0.5
    base = mx.floor(real).astype(mx.int32)  # type: ignore[union-attr]
    if mode == "bilinear":
        indices = mx.stack([base, base + 1], axis=1)  # type: ignore[union-attr]
        right_weight = real - base.astype(mx.float32)  # type: ignore[union-attr]
        weights = mx.stack([1.0 - right_weight, right_weight], axis=1)  # type: ignore[union-attr]
    elif mode == "bicubic":
        offsets = mx.array([-1, 0, 1, 2], dtype=mx.int32)  # type: ignore[union-attr]
        indices = base[:, None] + offsets[None, :]
        weights = _cubic_convolution_weight(real[:, None] - indices.astype(mx.float32))
    else:  # pragma: no cover - caller validates.
        raise Pr95HNeRVMlxError(f"unsupported resize mode: {mode}")
    indices = mx.clip(indices, 0, in_size - 1).astype(mx.int32)  # type: ignore[union-attr]
    return indices, weights.astype(mx.float32)


def _cubic_convolution_weight(distance: Any) -> Any:
    a = -0.75
    x = mx.abs(distance)  # type: ignore[union-attr]
    x2 = x * x
    x3 = x2 * x
    inner = (a + 2.0) * x3 - (a + 3.0) * x2 + 1.0
    outer = a * x3 - 5.0 * a * x2 + 8.0 * a * x - 4.0 * a
    return mx.where(x <= 1.0, inner, mx.where(x < 2.0, outer, 0.0))  # type: ignore[union-attr]


def _uint8_ste_mlx(x: Any) -> Any:
    clipped = mx.clip(x, 0.0, 255.0)  # type: ignore[union-attr]
    rounded = mx.round(clipped)  # type: ignore[union-attr]
    return clipped + mx.stop_gradient(rounded - clipped)  # type: ignore[union-attr]


def _flatten_rgb_nhwc(value: Any) -> tuple[Any, tuple[int, ...]]:
    _validate_rgb_nhwc(value)
    shape = tuple(int(dim) for dim in value.shape)
    flat = mx.reshape(value, (-1, shape[-3], shape[-2], shape[-1]))  # type: ignore[union-attr]
    return flat, shape


def _restore_rgb_nhwc(flat: Any, original_shape: tuple[int, ...], hw: tuple[int, int]) -> Any:
    return mx.reshape(flat, (*original_shape[:-3], int(hw[0]), int(hw[1]), 3))  # type: ignore[union-attr]


def _validate_rgb_nhwc(value: Any) -> None:
    if not hasattr(value, "shape"):
        raise Pr95HNeRVMlxError("expected an MLX array with shape")
    shape = tuple(int(dim) for dim in value.shape)
    if len(shape) < 4 or shape[-1] != 3:
        raise Pr95HNeRVMlxError(f"expected (..., H, W, 3) RGB tensor, got {shape}")


def _positive_hw(size: tuple[int, int]) -> tuple[int, int]:
    h, w = int(size[0]), int(size[1])
    if h < 1 or w < 1:
        raise Pr95HNeRVMlxError(f"resize dimensions must be positive, got {(h, w)}")
    return h, w


__all__ = [
    "CAMERA_HW",
    "GRAD_PROBE_SCHEMA",
    "SCORER_HW",
    "SOURCE_FAITHFUL_PREPROCESS_SCHEMA",
    "SOURCE_VIDEO_PREPROCESS_SCHEMA",
    "apply_eval_roundtrip_nhwc",
    "bicubic_resize_to_camera_nhwc",
    "bilinear_eval_roundtrip_downsample_nhwc",
    "load_pr95_source_pairs_nhwc",
    "load_upstream_video_frames_nhwc",
    "pr95_mlx_preprocess_grad_probe",
    "pr95_pair_frame_indices",
    "pr95_source_pairs_to_scorer_targets_mlx",
    "resize_nhwc_align_corners_false",
    "rgb_to_yuv6_mlx",
    "run_pr95_mlx_source_faithful_smoke",
    "run_pr95_mlx_source_video_preprocess_smoke",
]
