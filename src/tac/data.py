"""Data loading utilities for task-aware codec training.

Handles video decoding, frame pair construction, and saliency weight loading
with BT.601 limited-range YUV420->RGB matching the scorer's frame_utils.py.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

# PoseNet uses seq_len=2 frame pairs
SEQ_LEN = 2


def yuv420_to_rgb(frame) -> torch.Tensor:
    """BT.601 limited-range YUV420->RGB. Returns (H, W, 3) uint8 tensor.

    Matches the scorer's frame_utils.py conversion exactly.
    """
    H, W = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(H, frame.planes[0].line_size)[:, :W]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(H // 2, frame.planes[1].line_size)[:, : W // 2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(H // 2, frame.planes[2].line_size)[:, : W // 2]

    y_t = torch.from_numpy(y.copy()).float()
    u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)
    u_up = F.interpolate(u_t, size=(H, W), mode="bilinear", align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(H, W), mode="bilinear", align_corners=False).squeeze()

    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)

    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)


def decode_video(
    path: str | Path,
    target_h: int = 874,
    target_w: int = 1164,
) -> list[torch.Tensor]:
    """Decode video to list of (H, W, 3) uint8 tensors.

    Upscales to target resolution via bicubic if needed.
    """
    container = av.open(str(path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        t = yuv420_to_rgb(frame)
        H, W, _ = t.shape
        if target_h != H or target_w != W:
            x = t.permute(2, 0, 1).unsqueeze(0).float()
            x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
            t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
        frames.append(t)
    container.close()
    return frames


def decode_archive(archive_path: str | Path) -> list[torch.Tensor]:
    """Extract .mkv from archive.zip and decode to frames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(str(archive_path)) as zf:
            zf.extractall(tmpdir)
        mkv = next(iter(Path(tmpdir).glob("*.mkv")))
        return decode_video(str(mkv))


def build_pairs(frames: list[torch.Tensor]) -> list[torch.Tensor]:
    """Build consecutive frame pairs as (1, 2, H, W, 3) uint8 tensors."""
    pairs = []
    for i in range(0, len(frames) - 1, SEQ_LEN):
        if i + SEQ_LEN > len(frames):
            break
        pair = torch.stack(frames[i : i + SEQ_LEN]).unsqueeze(0)
        pairs.append(pair)
    return pairs


def pair_from_frames(frames: list[torch.Tensor], start_idx: int) -> torch.Tensor:
    """Build a single pair on-the-fly from a frame list. Returns (1, 2, H, W, 3).

    This avoids pre-building all 600 pairs in memory, which is the key
    pattern for surviving MPS memory pressure on long training runs.
    """
    return torch.stack(frames[start_idx : start_idx + SEQ_LEN]).unsqueeze(0)


def pair_start_indices(frame_count: int) -> list[int]:
    """Get valid pair start indices for a given frame count."""
    return list(range(0, frame_count - 1, SEQ_LEN))


def saliency_for_pair(
    base_saliency: torch.Tensor,
    start_idx: int,
    alpha: float,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build saliency weights for a single pair on-the-fly.

    Returns (2, 1, H, W) weight tensor. Avoids pre-building all saliency
    pairs in memory.
    """
    slices = []
    last = base_saliency[-1]
    for offset in range(SEQ_LEN):
        frame_idx = start_idx + offset
        sal = base_saliency[frame_idx] if frame_idx < base_saliency.shape[0] else last
        slices.append((1.0 + alpha * sal).unsqueeze(0))
    return torch.stack(slices, dim=0).to(device)


def load_raw_saliency(saliency_path: str | Path) -> torch.Tensor:
    """Load raw saliency map (N, H, W) without applying alpha weighting."""
    return torch.from_numpy(np.load(str(saliency_path))).float()


def load_saliency_weights(
    saliency_path: str | Path,
    alpha: float,
    n_frames: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load PoseNet saliency map and build per-pixel weight tensor.

    Returns (n_frames, 1, H, W) weight tensor.
    weight = 1.0 + alpha * saliency
    """
    sal = np.load(str(saliency_path))
    sal_t = torch.from_numpy(sal).float()
    if sal_t.shape[0] < n_frames:
        pad = sal_t[-1:].expand(n_frames - sal_t.shape[0], -1, -1)
        sal_t = torch.cat([sal_t, pad], dim=0)
    sal_t = sal_t[:n_frames]
    weights = (1.0 + alpha * sal_t).unsqueeze(1).to(device)
    return weights


def save_precomputed(
    comp_frames: list[torch.Tensor],
    gt_frames: list[torch.Tensor],
    output_dir: str | Path,
) -> None:
    """Save decoded frames as tensors for fast cloud loading.

    Skips the 5-10 min video decode on subsequent runs.
    ~3.7GB per tensor (1200 × 874 × 1164 × 3 uint8).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(torch.stack(comp_frames), out / "comp_frames.pt")
    torch.save(torch.stack(gt_frames), out / "gt_frames.pt")


def load_precomputed(precomputed_dir: str | Path) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Load precomputed frame tensors (instant, no video decode).

    Returns (comp_frames, gt_frames) as lists of (H, W, 3) uint8 tensors.
    """
    d = Path(precomputed_dir)
    comp = torch.load(d / "comp_frames.pt", map_location="cpu", weights_only=True)
    gt = torch.load(d / "gt_frames.pt", map_location="cpu", weights_only=True)
    return list(comp), list(gt)


def load_frames(
    archive_path: str | Path | None = None,
    gt_video_path: str | Path | None = None,
    precomputed_dir: str | Path | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Load frames from precomputed tensors or decode from video.

    Precomputed path is checked first (instant). Falls back to video decode.
    This is the recommended entry point for all training scripts.
    """
    if precomputed_dir and Path(precomputed_dir).exists():
        comp_pt = Path(precomputed_dir) / "comp_frames.pt"
        gt_pt = Path(precomputed_dir) / "gt_frames.pt"
        if comp_pt.exists() and gt_pt.exists():
            print("[data] Loading precomputed frames (fast path)")
            return load_precomputed(precomputed_dir)

    if archive_path is None or gt_video_path is None:
        raise ValueError("Must provide archive_path + gt_video_path, or a valid precomputed_dir")

    print("[data] Decoding from video (slow path)")
    comp_frames = decode_archive(str(archive_path))
    gt_frames = decode_video(str(gt_video_path))
    return comp_frames, gt_frames
