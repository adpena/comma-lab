# SPDX-License-Identifier: MIT
"""Build the frozen upstream scorer + GT-argmax/pose targets (PR95 ``data.py`` port).

The GT targets are the contest scorer's OWN outputs on the GT frames — literally
the d_seg/d_pose reference the evaluator charges. GT decode is ONLY via
``upstream/frame_utils.yuv420_to_rgb`` (CLAUDE.md non-negotiable; PyAV rgb24
manufactures phantom pose). The upstream ``rgb_to_yuv6`` is patched to be
gradient-reachable via the canonical token so a downstream trainer's pose
gradient is not severed (CLAUDE.md "eval_roundtrip").

This is the reusable targets surface that the working loop (and #74/#63/#65)
consume against the real video.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def _ensure_upstream_on_path(upstream_dir: str = "upstream") -> Path:
    root = Path(upstream_dir).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_frozen_distortion_net(
    *, upstream_dir: str = "upstream", device: str = "cpu"
) -> torch.nn.Module:
    """Return the upstream ``DistortionNet`` with frozen SegNet + PoseNet weights.

    All parameters are frozen (``requires_grad=False``) — the gradient must only
    update the carrier, never the scorer (CLAUDE.md "Strict scorer rule").
    """
    _ensure_upstream_on_path(upstream_dir)
    # Patch the @torch.no_grad / in-place upstream rgb_to_yuv6 so a pose gradient
    # reaches the carrier (CLAUDE.md "eval_roundtrip" non-negotiable).
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally

    patch_upstream_yuv6_globally()
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in net.parameters():
        p.requires_grad = False
    return net


def build_gt_targets(
    distortion_net: torch.nn.Module,
    *,
    video_path: str = "upstream/videos/0.mkv",
    max_pairs: int | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Stream-decode the video and cache GT SegNet-argmax + PoseNet targets.

    Pairs are non-overlapping consecutive frames (``seq_len=2``), matching
    ``upstream/evaluate.py``. GT decode is via ``frame_utils.yuv420_to_rgb`` only.

    Args:
        distortion_net: frozen upstream scorer (from :func:`load_frozen_distortion_net`).
        video_path: the contest video.
        max_pairs: optional cap (for cheap $0 proofs); ``None`` = all pairs.
        device: torch device string.

    Returns:
        ``(seg_targets_hard, pose_targets, n_pairs)``:
            - ``seg_targets_hard``: ``(n_pairs, 384, 512)`` int64 SegNet argmax.
            - ``pose_targets``: ``(n_pairs, 6)`` float32 PoseNet output.
            - ``n_pairs``: number of pairs cached.
    """
    _ensure_upstream_on_path()
    import av
    from frame_utils import yuv420_to_rgb

    container = av.open(str(video_path))
    seg_targets: list[torch.Tensor] = []
    pose_targets: list[torch.Tensor] = []
    prev = None
    with torch.inference_mode():
        for frame in container.decode(container.streams.video[0]):
            f = yuv420_to_rgb(frame)
            if prev is None:
                prev = f
                continue
            f0, f1 = prev, f
            prev = None
            pair = torch.stack([f0, f1]).unsqueeze(0).to(device)
            po, so = distortion_net(pair)
            seg_targets.append(so.argmax(dim=1).squeeze(0).clone())
            pose_targets.append(po["pose"][:, :6].float().squeeze(0).clone())
            if max_pairs is not None and len(seg_targets) >= max_pairs:
                break
    container.close()

    seg_t = torch.stack(seg_targets)
    pose_t = torch.stack(pose_targets)
    return seg_t, pose_t, seg_t.shape[0]
