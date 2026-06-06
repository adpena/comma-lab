# SPDX-License-Identifier: MIT
"""Real-video target decode for the MLX score-aware harness.

Separation of concerns: this module owns ONLY the "decode the real contest
video into per-pair MLX target buffers" contract per Catalog #114 (real contest
video, NEVER ``make_synthetic_*`` outside ``--smoke``). It delegates the decode
to the canonical ``tac.data.decode_video`` and reshapes into the canonical MLX
NHWC ``[0, 1]`` target layout the loss module consumes.

[verified-against: tac.data.decode_video canonical real-video decoder]
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from tac.adaptation.hard_pair_indices import normalize_pair_indices
from tac.substrates._shared.mlx_score_aware.device_gate import (
    MlxScoreAwareHarnessError,
    require_mlx_for_harness,
)

#: Full contest pair count (1200 frames / 2 per pair).
N_PAIRS_FULL: int = 600
_SCORER_HW: tuple[int, int] = (384, 512)
_CAMERA_HW: tuple[int, int] = (874, 1164)
_TARGET_HYDRATION_STRATEGIES: frozenset[str] = frozenset(
    ("auto", "materialized", "chunked")
)
_DEFAULT_CHUNKED_FULL_CAMERA_FRAMES: int = 32
_DEFAULT_MAX_MATERIALIZED_FULL_CAMERA_FRAMES: int = 1200


def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise MlxScoreAwareHarnessError(
            f"{name} must be a positive integer; got {raw!r}"
        ) from exc
    if value < 1:
        raise MlxScoreAwareHarnessError(
            f"{name} must be a positive integer; got {raw!r}"
        )
    return value


def _resolve_target_hydration_strategy(
    *,
    requested: str,
    official_scorer_surface: bool,
    target_is_scorer_hw: bool,
    needed_frames: int,
) -> str:
    strategy = os.environ.get("PACT_MLX_TARGET_HYDRATION_STRATEGY", requested)
    strategy = str(strategy or "auto").strip().lower().replace("-", "_")
    if strategy == "full_camera_materialized":
        strategy = "materialized"
    if strategy not in _TARGET_HYDRATION_STRATEGIES:
        allowed = ", ".join(sorted(_TARGET_HYDRATION_STRATEGIES))
        raise MlxScoreAwareHarnessError(
            f"target_hydration_strategy must be one of {allowed}; got {strategy!r}"
        )
    if not official_scorer_surface or not target_is_scorer_hw:
        return "materialized"
    if strategy != "auto":
        return strategy
    max_materialized = _positive_int_from_env(
        "PACT_MLX_TARGET_HYDRATION_MAX_MATERIALIZED_FULL_CAMERA_FRAMES",
        _DEFAULT_MAX_MATERIALIZED_FULL_CAMERA_FRAMES,
    )
    return "materialized" if int(needed_frames) <= max_materialized else "chunked"


def _decode_selected_full_camera_frames(
    video_path: Any,
    *,
    frame_indices: Sequence[int],
) -> list[Any]:
    """Decode only selected full-camera source frames in scorer order."""

    if not frame_indices:
        return []
    import av

    from tac.data import yuv420_to_rgb

    wanted = tuple(int(idx) for idx in frame_indices)
    wanted_set = set(wanted)
    max_needed = max(wanted_set)
    decoded: dict[int, Any] = {}
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx > max_needed:
                break
            if frame_idx in wanted_set:
                decoded[frame_idx] = yuv420_to_rgb(frame).numpy()
                if len(decoded) == len(wanted_set):
                    break
    finally:
        container.close()
    missing = [idx for idx in wanted if idx not in decoded]
    if missing:
        raise MlxScoreAwareHarnessError(
            f"decoded selected frames from {video_path!s} but missing frame "
            f"indices {missing[:8]}{'...' if len(missing) > 8 else ''}"
        )
    return [decoded[idx] for idx in wanted]


def _resize_full_camera_frames_to_scorer_hw(
    frames: Sequence[Any],
    *,
    chunk_frames: int,
) -> Any:
    """Apply the upstream full-camera -> SegNet/PoseNet bilinear receiver resize."""

    import numpy as np
    import torch
    import torch.nn.functional as F

    chunk = max(1, int(chunk_frames))
    resized_chunks = []
    for start in range(0, len(frames), chunk):
        arr = np.stack(frames[start : start + chunk], axis=0)
        x = torch.from_numpy(np.ascontiguousarray(arr)).float()
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=_SCORER_HW, mode="bilinear")
        resized_chunks.append(
            x.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)
        )
    return np.concatenate(resized_chunks, axis=0)


def _decode_video_frames(video_path: Any, **kwargs: Any) -> list[Any]:
    """Import the heavy video stack only at the real decode boundary."""

    from tac.data import decode_video

    return decode_video(video_path, **kwargs)


def decode_mlx_targets(
    video_path: Any,
    *,
    num_pairs: int,
    output_height: int,
    output_width: int,
    pair_indices: Sequence[Any] | str | None = None,
    official_scorer_surface: bool = True,
    target_hydration_strategy: str = "auto",
    official_scorer_surface_chunk_frames: int | None = None,
) -> tuple[Any, Any]:
    """Decode real contest video into per-pair MLX target frame buffers.

    Per Catalog #114 the targets come from the actual contest video, NEVER
    ``make_synthetic_*`` outside ``--smoke``. By default this decodes the prefix
    ``2 * num_pairs`` frames, reshapes to ``(num_pairs, 2, H, W, 3)`` and splits
    into two NHWC MLX float32 buffers normalized to ``[0, 1]`` (the canonical
    MLX target layout consumed by the loss module). When ``pair_indices`` is
    supplied, those source pair ids are decoded and returned in normalized
    first-seen order instead of silently falling back to prefix pairs.

    Args:
        video_path: path to the contest video (e.g. ``upstream/videos/0.mkv``).
        num_pairs: number of adjacent-frame pairs to decode (full = 600).
        output_height / output_width: target spatial size (the substrate
            renderer's output resolution). When this is the canonical scorer
            size ``384x512``, the default path decodes full camera frames and
            applies the same bilinear scorer resize as upstream ``evaluate.py``.
        official_scorer_surface: for canonical ``384x512`` targets, hydrate
            from the official full-camera -> bilinear receiver surface. Set
            ``False`` only for deliberate legacy/direct-resize tests.
        target_hydration_strategy: ``auto`` uses full materialization on this
            workstation when it fits, and falls back to ``chunked`` above the
            configurable frame threshold. ``chunked`` decodes only the selected
            full-camera frames and applies the same official bilinear receiver
            resize in bounded chunks. ``materialized`` preserves the historical
            prefix-materialized path. Environment
            ``PACT_MLX_TARGET_HYDRATION_STRATEGY`` can override the default for
            production hardware.
        official_scorer_surface_chunk_frames: chunk size for the official
            full-camera -> scorer resize. Defaults to
            ``PACT_MLX_TARGET_HYDRATION_CHUNK_FRAMES`` or 32.
        pair_indices: optional arbitrary source pair ids, matching hard-pair
            CPU advisory/QAT behavior.

    Returns:
        ``(target_rgb_0, target_rgb_1)`` each MLX float32 ``(num_pairs, H, W,
        3)`` in ``[0, 1]``.

    Raises:
        MlxScoreAwareHarnessError: fewer than ``2 * num_pairs`` frames decoded.
    """
    import numpy as np

    mx = require_mlx_for_harness()

    if pair_indices is None:
        if int(num_pairs) < 1:
            raise MlxScoreAwareHarnessError(f"num_pairs must be >= 1; got {num_pairs}")
        source_pair_indices = tuple(range(int(num_pairs)))
    else:
        source_pair_indices = normalize_pair_indices(
            pair_indices,
            field="pair_indices",
        )
        if not source_pair_indices:
            raise MlxScoreAwareHarnessError("pair_indices must contain at least one pair")
        if len(source_pair_indices) != int(num_pairs):
            raise MlxScoreAwareHarnessError(
                "pair_indices normalized length "
                f"{len(source_pair_indices)} does not match num_pairs {int(num_pairs)}"
            )

    needed_frames = (
        2 * int(num_pairs)
        if pair_indices is None
        else max(2 * int(pair_idx) + 2 for pair_idx in source_pair_indices)
    )
    target_is_scorer_hw = (int(output_height), int(output_width)) == _SCORER_HW
    strategy = _resolve_target_hydration_strategy(
        requested=target_hydration_strategy,
        official_scorer_surface=official_scorer_surface,
        target_is_scorer_hw=target_is_scorer_hw,
        needed_frames=needed_frames,
    )
    selected_frame_indices: list[int] = []
    for pair_idx in source_pair_indices:
        selected_frame_indices.extend([2 * int(pair_idx), 2 * int(pair_idx) + 1])
    chunk_frames = (
        int(official_scorer_surface_chunk_frames)
        if official_scorer_surface_chunk_frames is not None
        else _positive_int_from_env(
            "PACT_MLX_TARGET_HYDRATION_CHUNK_FRAMES",
            _DEFAULT_CHUNKED_FULL_CAMERA_FRAMES,
        )
    )
    if chunk_frames < 1:
        raise MlxScoreAwareHarnessError(
            "official_scorer_surface_chunk_frames must be >= 1; "
            f"got {official_scorer_surface_chunk_frames}"
        )
    if official_scorer_surface and target_is_scorer_hw and strategy == "chunked":
        selected_frames = _decode_selected_full_camera_frames(
            video_path,
            frame_indices=selected_frame_indices,
        )
        gt_arr = _resize_full_camera_frames_to_scorer_hw(
            selected_frames,
            chunk_frames=chunk_frames,
        )
        gt_pairs = gt_arr.reshape(len(source_pair_indices), 2, 384, 512, 3)
    else:
        decode_height = (
            _CAMERA_HW[0]
            if official_scorer_surface and target_is_scorer_hw
            else output_height
        )
        decode_width = (
            _CAMERA_HW[1]
            if official_scorer_surface and target_is_scorer_hw
            else output_width
        )
        frames = _decode_video_frames(
            video_path,
            target_h=decode_height,
            target_w=decode_width,
            max_frames=needed_frames,
        )
        if len(frames) < needed_frames:
            need_message = (
                f"{2 * int(num_pairs)} for {num_pairs} pairs"
                if pair_indices is None
                else f"frame {needed_frames - 1} for pair_indices {list(source_pair_indices)}"
            )
            raise MlxScoreAwareHarnessError(
                f"decoded {len(frames)} frames from {video_path!s}; need {need_message} at "
                f"({output_height}, {output_width})"
            )
        selected_frames = [frames[idx].numpy() for idx in selected_frame_indices]
        if official_scorer_surface and target_is_scorer_hw:
            gt_arr = _resize_full_camera_frames_to_scorer_hw(
                selected_frames,
                chunk_frames=len(selected_frames),
            )
            gt_pairs = gt_arr.reshape(len(source_pair_indices), 2, 384, 512, 3)
        else:
            gt_arr = np.stack(selected_frames, axis=0)
            gt_pairs = gt_arr.reshape(
                len(source_pair_indices),
                2,
                decode_height,
                decode_width,
                3,
            )
    target_rgb_0 = mx.array((gt_pairs[:, 0] / 255.0).astype(np.float32))
    target_rgb_1 = mx.array((gt_pairs[:, 1] / 255.0).astype(np.float32))
    return target_rgb_0, target_rgb_1


__all__ = [
    "N_PAIRS_FULL",
    "decode_mlx_targets",
]
