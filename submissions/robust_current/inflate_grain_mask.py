#!/usr/bin/env python
"""Saliency-masked grain synthesis inflate path.

Decode with BT.601 + torch bicubic (canonical), then add synthetic grain
ONLY on the pixels PoseNet cares about (from saliency map).

Rationale:
- fg=0 gives best SegNet (0.00548) and best rate (717KB)
- fg=22 is essential for PoseNet (fg=0 scores 2.94 because PoseNet loses temporal cues)
- PoseNet saliency is extremely sparse: only 7% of pixels > 0.05
- By adding grain selectively, we get: small archive (fg=0) + good SegNet + PoseNet texture cues

The grain model is deterministic seeded noise matching AV1's grain synthesis characteristics.
"""
import sys
import os
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import av


def yuv420_to_rgb(frame) -> torch.Tensor:
    """BT.601 limited range, matching frame_utils.py exactly."""
    H, W = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(H, frame.planes[0].line_size)[:, :W]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(H // 2, frame.planes[1].line_size)[:, :W // 2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(H // 2, frame.planes[2].line_size)[:, :W // 2]

    y_t = torch.from_numpy(y.copy()).float()
    u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)

    u_up = F.interpolate(u_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()

    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)

    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)


def generate_grain(h: int, w: int, strength: float, seed: int) -> np.ndarray:
    """Generate film-grain-like noise matching AV1 grain synthesis characteristics.

    AV1 grain is modeled as correlated Gaussian noise with per-luma-level scaling.
    We approximate this with simple Gaussian noise scaled by strength.
    """
    rng = np.random.RandomState(seed)
    noise = rng.randn(h, w, 3).astype(np.float32) * strength
    return noise


def inflate_with_grain_mask(
    video_path: str,
    dst: str,
    saliency_path: str,
    target_w: int = 1164,
    target_h: int = 874,
    grain_strength: float = 8.0,
    saliency_threshold: float = 0.02,
) -> int:
    """Decode, upscale, add grain only where PoseNet cares."""
    # Load saliency map
    saliency = np.load(saliency_path)  # (N_pairs, H, W)
    print(f"  Saliency: shape={saliency.shape}, mean={saliency.mean():.4f}", file=sys.stderr)

    container = av.open(video_path)
    stream = container.streams.video[0]
    n = 0
    with open(dst, 'wb') as f:
        for frame in container.decode(stream):
            t = yuv420_to_rgb(frame)  # (H, W, 3) uint8
            H, W, _ = t.shape
            if H != target_h or W != target_w:
                x = t.permute(2, 0, 1).unsqueeze(0).float()
                x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
                t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)

            # Get saliency mask for this frame (interpolate from keyframes)
            if saliency.shape[0] > 0:
                ratio = n / max(1199, 1)
                sal_idx = ratio * (saliency.shape[0] - 1)
                lo = int(sal_idx)
                hi = min(lo + 1, saliency.shape[0] - 1)
                interp = sal_idx - lo
                sal_frame = saliency[lo] * (1.0 - interp) + saliency[hi] * interp
            else:
                sal_frame = np.ones((target_h, target_w), dtype=np.float32)

            # Generate grain for this frame
            grain = generate_grain(target_h, target_w, grain_strength, seed=n * 7 + 42)

            # Apply grain only where saliency exceeds threshold
            mask = (sal_frame > saliency_threshold).astype(np.float32)
            # Smooth the mask edges slightly
            mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            mask_smooth = F.avg_pool2d(mask_t, kernel_size=5, stride=1, padding=2).squeeze().numpy()
            mask_smooth = np.clip(mask_smooth * 2, 0, 1)  # sharpen the smooth mask

            # Apply masked grain
            frame_np = t.numpy().astype(np.float32)
            frame_np += grain * mask_smooth[:, :, np.newaxis]
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

            f.write(frame_np.tobytes())
            n += 1
            if n % 300 == 0:
                print(f"  Processed {n} frames ...", file=sys.stderr, flush=True)

    container.close()
    print(f"Inflated {n} frames with saliency-masked grain -> {dst}", file=sys.stderr)
    return n


if __name__ == "__main__":
    archive_dir = sys.argv[1]
    inflated_dir = sys.argv[2]
    video_names_file = sys.argv[3]
    saliency_path = sys.argv[4] if len(sys.argv) > 4 else "experiments/masks/posenet_saliency.npy"
    grain_strength = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0

    inflated_dir = Path(inflated_dir)
    inflated_dir.mkdir(parents=True, exist_ok=True)

    for line in Path(video_names_file).read_text().splitlines():
        rel = line.strip()
        if not rel:
            continue
        stem = rel.rsplit(".", 1)[0]
        mkv_path = Path(archive_dir) / f"{stem}.mkv"
        out_path = inflated_dir / f"{stem}.raw"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Inflating {mkv_path} -> {out_path} (saliency-masked grain)", file=sys.stderr)
        inflate_with_grain_mask(str(mkv_path), str(out_path), saliency_path, grain_strength=grain_strength)
