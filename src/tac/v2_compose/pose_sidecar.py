# SPDX-License-Identifier: MIT
"""v2_compose.pose_sidecar — the STORE dual-use pose sidecar (Step 5; thin reuse).

The stored 6-PoseNet-scalars/pair sidecar (FACT 3, the WEAK dual-use): consumed TWICE —
(1) at inflate the render is supervised to hit these targets -> d_pose ~ 3.4e-5 directly;
(2) the SAME stored pose drives the Step-3 per-class stratified warp for the d_seg bulk at
~0 extra bytes. One store, two free read-outs (calibrated independently; the lossless dual-use
grok was REFUTED — pose and d_seg want OPPOSITE warp scales, FACT 3).

This is a THIN wrapper over the BUILT ``tac.scorer_targets`` (PNTG format: magic + fp16 + zlib).
The $0 path: the GT cache's ``gt_poses`` (n,6) ARE the frozen-PoseNet outputs on the GT pairs (the
exact d_pose targets) — so the sidecar builds directly from the cache, no PoseNet re-run. The full
path (``tac.scorer_targets.extract_and_save``) re-runs PoseNet on the GT video when no cache exists.

Optional further compression: the low-rank pose codec (``tac.qp1_pose_codec`` / ``tac.pose_from_embedding``,
task #140, rank-2 ~2 KB) — available but NOT default (the PNTG fp16+zlib ~875 B is already tiny).

NO-FAKE: makes no score claim. The sidecar BYTES are COUNTED (video-derived pose scalars). The
d_pose it enables is realized through the frozen PoseNet at byte-close, never asserted here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tac.scorer_targets import load_posenet_targets, save_posenet_targets

__all__ = [
    "build_pose_sidecar_from_cache_poses",
    "build_pose_sidecar_from_gt_video",
    "pose_sidecar_byte_cost",
]


def build_pose_sidecar_from_cache_poses(
    gt_poses: np.ndarray,
    out_path: str | Path,
) -> int:
    """Build the PNTG pose sidecar directly from the GT cache ``gt_poses`` (the $0 path).

    ``gt_poses`` (n_pairs, 6) ARE the frozen-PoseNet outputs on the GT pairs (the exact d_pose
    targets the scorer compares against). Stores them in the canonical PNTG format (fp16 + zlib).

    Args:
        gt_poses: (n_pairs, 6) array of the 6 PoseNet scalars per pair (the cache ``gt_poses`` key).
        out_path: where to write ``posenet_targets.bin``.

    Returns:
        The sidecar file size in bytes (the COUNTED contribution).
    """
    import torch

    poses = np.asarray(gt_poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise ValueError(f"gt_poses must be (n_pairs, 6); got {poses.shape}")
    n_pairs = int(poses.shape[0])
    targets_dict = {
        "targets": torch.from_numpy(poses),  # (n_pairs, 6) float32
        "n_pairs": n_pairs,
        "n_frames": 2 * n_pairs,
    }
    return save_posenet_targets(targets_dict, out_path)


def build_pose_sidecar_from_gt_video(
    gt_video_path: str | Path,
    posenet_path: str | Path,
    out_path: str | Path,
    *,
    upstream_dir: str | Path | None = None,
    device: str = "cpu",
) -> int:
    """Full path: decode the GT video, run the frozen PoseNet, store targets (reuses the BUILT
    ``tac.scorer_targets.extract_and_save``). Use only when the GT cache poses are unavailable.

    NEVER MPS for the PoseNet forward (CLAUDE.md MPS-noise non-negotiable); device defaults cpu.
    """
    from tac.scorer_targets import extract_and_save

    if device == "mps":
        raise ValueError("pose_sidecar: MPS is FORBIDDEN for the frozen PoseNet (corrupts pose)")
    return extract_and_save(
        gt_video_path=gt_video_path,
        posenet_path=posenet_path,
        output_path=out_path,
        upstream_dir=upstream_dir,
        device=device,
    )


def pose_sidecar_byte_cost(path: str | Path) -> dict[str, Any]:
    """Read back a PNTG sidecar and report its COUNTED byte cost + the rate contribution."""
    from tac.contest_score import rate_term

    p = Path(path)
    size = int(p.stat().st_size)
    loaded = load_posenet_targets(p)
    n_pairs = int(loaded["n_pairs"]) if loaded else 0
    return {
        "pose_sidecar_bytes": size,
        "n_pairs": n_pairs,
        "rate_contribution": rate_term(size),
        "format": "PNTG (fp16 + zlib)",
        "note": "COUNTED (video-derived pose scalars); dual-use: d_pose direct + drives per-class warp.",
    }
