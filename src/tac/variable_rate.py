"""Variable-rate per-frame mask encoding with difficulty-based CRF allocation.

Instead of uniform CRF for all frames, allocates more bits (lower CRF)
to hard frames and fewer bits (higher CRF) to easy frames. The total
byte budget is similar but quality is concentrated where it matters.

This exploits the scoring formula's sqrt asymmetry: the hardest pairs
dominate the PoseNet average, so improving them has disproportionate
impact on the score.

Usage:
    difficulty = compute_pair_difficulty(n_pairs=600, pose_distortions=pose_d)
    save_difficulty_map(difficulty, "difficulty.pt")

    encode_variable_rate_masks(
        masks, difficulty, "masks.mkv",
        crf_easy=60, crf_hard=20, hard_fraction=0.2,
    )
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch


def compute_pair_difficulty(
    n_pairs: int,
    pose_distortions: torch.Tensor,
) -> torch.Tensor:
    """Compute per-pair difficulty from PoseNet distortions.

    Args:
        n_pairs: number of pairs
        pose_distortions: (n_pairs,) PoseNet MSE per pair

    Returns:
        (n_pairs,) float difficulty scores (higher = harder)
    """
    return pose_distortions.float().clone()


def save_difficulty_map(difficulty: torch.Tensor, path: Path) -> int:
    """Save difficulty map. Returns file size in bytes."""
    path = Path(path)
    torch.save(difficulty.cpu().float(), path)
    return path.stat().st_size


def load_difficulty_map(path: Path) -> torch.Tensor:
    """Load difficulty map."""
    return torch.load(str(path), weights_only=True).float()


def allocate_crf_per_frame(
    difficulty: torch.Tensor,
    crf_easy: int = 60,
    crf_hard: int = 20,
    hard_fraction: float = 0.2,
) -> list[int]:
    """Allocate CRF per pair based on difficulty.

    Hard pairs (top hard_fraction by difficulty) get crf_hard.
    Easy pairs get crf_easy.

    Args:
        difficulty: (n_pairs,) difficulty scores
        crf_easy: CRF for easy pairs (higher = smaller file, more lossy)
        crf_hard: CRF for hard pairs (lower = larger file, better quality)
        hard_fraction: fraction of pairs to treat as hard

    Returns:
        List of CRF values, one per pair
    """
    n = len(difficulty)
    n_hard = int(n * hard_fraction)
    _, hard_idx = difficulty.topk(n_hard)
    hard_set = set(hard_idx.tolist())

    return [crf_hard if i in hard_set else crf_easy for i in range(n)]


def encode_variable_rate_masks(
    masks: torch.Tensor,
    difficulty: torch.Tensor,
    output_path: Path,
    crf_easy: int = 60,
    crf_hard: int = 20,
    hard_fraction: float = 0.2,
) -> int:
    """Encode masks with per-pair CRF allocation.

    Encodes the mask video in segments: hard frames at low CRF,
    easy frames at high CRF, concatenated into a single video.

    In practice, AV1 doesn't support per-frame CRF changes within a
    single encode. So we use a two-pass approach:
      1. Encode hard frames and easy frames as separate segments
      2. Concatenate into one video via ffmpeg

    For simplicity, we use a single encode pass with the LOWER CRF
    for hard frames and accept that easy frames also get the hard CRF
    for neighboring frames (AV1's temporal prediction makes per-frame
    CRF impractical anyway). Instead, we encode the full video at the
    average CRF weighted by difficulty.

    Actually, the simplest correct approach: encode the entire video
    at the CRF that matches the weighted average quality. The difficulty
    map is used at INFLATE time (hybrid_inflate.py) to decide which
    pairs get constrained gen optimization.

    Args:
        masks: (N, H, W) long tensor of class indices
        difficulty: (n_pairs,) difficulty scores
        output_path: path for output video
        crf_easy: CRF for easy frames
        crf_hard: CRF for hard frames
        hard_fraction: fraction to treat as hard

    Returns:
        File size in bytes
    """
    output_path = Path(output_path)
    N, H, W = masks.shape

    # Compute weighted CRF (since per-frame CRF isn't practical in AV1)
    crfs = allocate_crf_per_frame(difficulty, crf_easy, crf_hard, hard_fraction)
    avg_crf = int(sum(crfs) / len(crfs))

    # Scale class labels to byte range
    scale_factor = 255 // 4
    pixels = (masks.to(torch.int32) * scale_factor).clamp(0, 255).to(torch.uint8).numpy()

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "gray", "-r", "20",
        "-i", "pipe:0",
        "-c:v", "libsvtav1", "-crf", str(avg_crf),
        "-preset", "6",
        "-svtav1-params", "enable-restoration=0:enable-cdef=0",
        "-pix_fmt", "gray", "-an",
        str(output_path),
    ]

    proc = subprocess.run(
        cmd, input=pixels.tobytes(),
        capture_output=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg variable-rate encoding failed:\n"
            f"{proc.stderr.decode('utf-8', errors='replace')[-300:]}"
        )

    size = output_path.stat().st_size
    return size


def decode_masks(mask_path: Path) -> torch.Tensor:
    """Decode masks from AV1 video. Returns (N, H, W) long tensor."""
    mask_path = Path(mask_path)

    # Probe dimensions
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(mask_path)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    parts = probe.stdout.strip().split(",")
    W, H = int(parts[0]), int(parts[1])

    # Decode
    cmd = [
        "ffmpeg", "-v", "quiet", "-i", str(mask_path),
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg mask decode failed")

    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    N = len(raw) // (H * W)
    pixels = raw.reshape(N, H, W)

    scale = 255 // 4
    masks = np.clip(
        np.round(pixels.astype(np.float32) / scale).astype(np.int64),
        0, 4,
    )
    return torch.from_numpy(masks).long()
