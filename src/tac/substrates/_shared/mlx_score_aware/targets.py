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

from collections.abc import Sequence
from typing import Any

from tac.adaptation.hard_pair_indices import normalize_pair_indices
from tac.substrates._shared.mlx_score_aware.device_gate import (
    MlxScoreAwareHarnessError,
    require_mlx_for_harness,
)

#: Full contest pair count (1200 frames / 2 per pair).
N_PAIRS_FULL: int = 600


def decode_mlx_targets(
    video_path: Any,
    *,
    num_pairs: int,
    output_height: int,
    output_width: int,
    pair_indices: Sequence[Any] | str | None = None,
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
            renderer's output resolution).
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

    from tac.data import decode_video

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
    frames = decode_video(
        video_path,
        target_h=output_height,
        target_w=output_width,
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
    selected_frames = []
    for pair_idx in source_pair_indices:
        selected_frames.append(frames[2 * int(pair_idx)])
        selected_frames.append(frames[2 * int(pair_idx) + 1])
    gt_arr = np.stack([f.numpy() for f in selected_frames], axis=0)
    gt_pairs = gt_arr.reshape(
        len(source_pair_indices),
        2,
        output_height,
        output_width,
        3,
    )
    target_rgb_0 = mx.array((gt_pairs[:, 0] / 255.0).astype(np.float32))
    target_rgb_1 = mx.array((gt_pairs[:, 1] / 255.0).astype(np.float32))
    return target_rgb_0, target_rgb_1


__all__ = [
    "N_PAIRS_FULL",
    "decode_mlx_targets",
]
