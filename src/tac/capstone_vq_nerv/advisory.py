# SPDX-License-Identifier: MIT
"""Reloaded-int8-archive advisory scoring (audit [A2]) — close the decoupling.

The capstone's live advisory measures d_seg/d_pose on the LIVE fp32 MLX render, but
the archive that ships is int8-quantized (per-tensor symmetric int8 decoder/codebook
+ fp16 pose). The quantization loss is NEVER scored by the live number — so a
sub-0.15 live advisory can hide an int8-quant regression (PR95 re-parses the
RELOADED decoder + evaluates THAT; we did not). This module reloads the byte-closed
int8 archive through the SAME pure-numpy contest inflate decode (``decode_archive`` +
``numpy_decode_pair``) and re-scores the reloaded frames through the frozen scorer
bridge — the honest predictor of ``inflate.sh -> evaluate.py``.

This is the contest decode path WITHOUT the camera upsample (the bridge scores at the
384x512 native render resolution and applies its own eval_roundtrip, exactly as the
live advisory does — so the live-vs-reloaded gap isolates the int8/fp16 archive
quantization, not a resize mismatch). NO MLX, NO scorer here beyond the bridge the
caller already built; pure numpy decode of the REAL archive bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tac.capstone_vq_nerv.inflate import decode_archive
from tac.capstone_vq_nerv.numpy_reference import numpy_decode_pair


@dataclass(frozen=True)
class ReloadedInt8Advisory:
    """d_seg/d_pose on the reloaded int8 archive (the honest contest predictor)."""

    d_seg: float
    d_pose: float
    num_pairs: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "reloaded_int8_d_seg": self.d_seg,
            "reloaded_int8_d_pose": self.d_pose,
            "reloaded_int8_num_pairs": self.num_pairs,
        }


def score_reloaded_int8_archive(
    archive_bytes: bytes,
    config: dict[str, Any],
    bridge: Any,
    *,
    batch_size: int = 8,
) -> ReloadedInt8Advisory:
    """Decode the int8 archive (contest path) and re-score it through ``bridge``.

    Args:
        archive_bytes: the byte-closed archive payload (member ``x`` bytes — the
            ``parse_capstone_archive_bytes`` 4-section blob, NOT the ZIP wrapper).
        config: the ``capstone_config_v1`` sidecar dict (base_channels / num_pairs /
            codebook_size / decoder_dtype / pose_mean,std / film_enabled ...).
        bridge: a :class:`tac.mlx_pr95_port.score_bridge.TorchScorerBridge` holding
            the frozen scorer + GT seg/pose targets — the SAME bridge the live
            advisory used (so the only difference is fp32-live vs int8-reloaded).
        batch_size: pairs per scorer forward.

    Returns:
        :class:`ReloadedInt8Advisory` with mean d_seg + d_pose over all pairs,
        measured on the REAL reloaded int8 frames.
    """
    decoded = decode_archive(archive_bytes, config)
    weights = decoded["weights"]
    codebook = decoded["codebook"]
    vq_indices = decoded["vq_indices"]
    pose = decoded["pose"]
    cfg = decoded["cfg"]
    num_pairs = int(decoded["num_pairs"])

    pose_enabled = getattr(bridge, "pose_enabled", False) and bridge.pose_targets is not None
    d_seg_total = 0.0
    d_pose_total = 0.0
    n = 0
    for start in range(0, num_pairs, batch_size):
        end = min(start + batch_size, num_pairs)
        idx_np = np.arange(start, end)
        z_q = codebook[vq_indices[start:end]]
        # numpy decode of the RELOADED int8 weights -> (b, 2, 3, 384, 512) float32
        render = numpy_decode_pair(z_q, pose[start:end], weights, cfg)
        idx_t = torch.from_numpy(idx_np.astype(np.int64))
        d_seg_total += bridge.exact_d_seg(render, idx_t) * len(idx_np)
        if pose_enabled:
            d_pose_total += bridge.exact_d_pose(render, idx_t) * len(idx_np)
        n += len(idx_np)

    denom = max(n, 1)
    return ReloadedInt8Advisory(
        d_seg=d_seg_total / denom,
        d_pose=(d_pose_total / denom) if pose_enabled else 0.0,
        num_pairs=num_pairs,
    )


__all__ = ["ReloadedInt8Advisory", "score_reloaded_int8_archive"]
