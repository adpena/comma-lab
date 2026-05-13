#!/usr/bin/env python
"""Generate visual comparison frames for the writeup.

Creates side-by-side PNG images showing:
1. Original GT frame
2. AV1 decoded (no filter)
3. Our filter applied
4. Pixel difference heatmap (magnified 10x)
5. Saliency overlay (PoseNet gradient map)
6. SegNet class boundary overlay

Output: reports/graphs/media/visual_comparison_*.png

Usage::

    cd /tmp/pact-mine
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy --with pillow \\
        python -u experiments/generate_visual_comparison.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tac.data import decode_archive, decode_video
from tac.scorer import detect_device, load_scorers
from tac.proxy_eval import _default_paths
from tac.quantization import load_postfilter_int8

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent.parent  # src/tac/research -> project root

_PROJECT, _UPSTREAM, VIDEOS_DIR, _LIVE_ARCHIVE, ARCHIVE_ZIP = _default_paths()
DEVICE = detect_device()

OUTPUT_DIR = PROJECT / "reports" / "graphs" / "media"


def save_image(tensor_hwc: torch.Tensor, path: str):
    """Save a (H, W, 3) uint8 tensor as PNG."""
    from PIL import Image
    arr = tensor_hwc.cpu().numpy().astype(np.uint8)
    Image.fromarray(arr).save(path)
    print(f"  Saved {path} ({os.path.getsize(path) / 1024:.1f} KB)")


def make_heatmap(diff_hw: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    """Convert a single-channel diff to a red-blue heatmap (H, W, 3) uint8."""
    # Positive diff → red, negative → blue, zero → black
    d = diff_hw.float() * scale
    r = d.clamp(0, 255)
    b = (-d).clamp(0, 255)
    g = torch.zeros_like(r)
    return torch.stack([r, g, b], dim=-1).clamp(0, 255).to(torch.uint8)


def make_saliency_overlay(frame_hwc: torch.Tensor, saliency_hw: torch.Tensor) -> torch.Tensor:
    """Overlay saliency as green-channel boost on the frame."""
    frame = frame_hwc.float().clone()
    sal_norm = saliency_hw.float()
    sal_norm = sal_norm / sal_norm.max().clamp(min=1e-6)
    # Boost green channel by saliency
    frame[:, :, 1] = (frame[:, :, 1] + sal_norm * 200).clamp(0, 255)
    return frame.to(torch.uint8)


def main():
    print(f"[visual] device={DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load filter
    weights_path = PROJECT / "submissions" / "robust_current" / "postfilter_int8.pt"
    print(f"[visual] Loading filter from {weights_path}")
    model = load_postfilter_int8(str(weights_path), device=str(DEVICE))

    # Load saliency
    sal_path = HERE / "masks" / "posenet_saliency.npy"
    saliency = None
    if sal_path.exists():
        saliency = torch.from_numpy(np.load(str(sal_path))).float()
        print(f"[visual] Saliency loaded: {saliency.shape}")

    # Decode frames
    print("[visual] Decoding...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))

    # Load SegNet for boundary overlay
    print("[visual] Loading SegNet...")
    _, segnet = load_scorers(DEVICE)

    # Select representative frames (beginning, middle, end)
    frame_indices = [0, 15, 30, 45, 59]  # from the 30 saliency frames
    # Map to actual frame indices (each pair has 2 frames, we use frame 1)
    pair_indices = [i * 2 + 1 for i in frame_indices if i * 2 + 1 < n]

    for i, fidx in enumerate(pair_indices):
        print(f"\n[visual] Processing frame {fidx}/{n}...")
        gt = gt_frames[fidx]         # (H, W, 3) uint8
        comp = comp_frames[fidx]     # (H, W, 3) uint8

        # Apply filter
        with torch.no_grad():
            x = comp.float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            y = model(x)
            filtered = y.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu()

        # Diff heatmap (filtered - decoded, per-channel luma)
        diff = filtered.float() - comp.float()
        diff_luma = 0.299 * diff[:, :, 0] + 0.587 * diff[:, :, 1] + 0.114 * diff[:, :, 2]
        heatmap = make_heatmap(diff_luma, scale=15.0)

        # Saliency overlay
        if saliency is not None and fidx // 2 < saliency.shape[0]:
            sal_frame = saliency[fidx // 2]
            sal_overlay = make_saliency_overlay(comp, sal_frame)
        else:
            sal_overlay = comp

        # SegNet boundary overlay
        with torch.no_grad():
            seg_in = filtered.float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(DEVICE)
            seg_in = seg_in[:, 0]  # (1, 3, H, W)
            seg_in = F.interpolate(seg_in, size=(384, 512), mode='bilinear', align_corners=False)
            seg_out = segnet(seg_in)
            seg_labels = seg_out.argmax(dim=1).float().unsqueeze(1)  # (1, 1, H_seg, W_seg)
            # Boundary detection
            max_p = F.max_pool2d(seg_labels, 3, 1, 1)
            min_p = -F.max_pool2d(-seg_labels, 3, 1, 1)
            boundary = (max_p != min_p).float()
            # Upscale boundary to full res
            boundary_full = F.interpolate(boundary, size=(874, 1164), mode='nearest').squeeze().cpu()

        seg_overlay = comp.float().clone()
        seg_overlay[:, :, 0] = (seg_overlay[:, :, 0] + boundary_full * 255).clamp(0, 255)
        seg_overlay = seg_overlay.to(torch.uint8)

        # Save individual images
        prefix = f"visual_frame{fidx:04d}"
        save_image(gt, str(OUTPUT_DIR / f"{prefix}_1_gt.png"))
        save_image(comp, str(OUTPUT_DIR / f"{prefix}_2_decoded.png"))
        save_image(filtered, str(OUTPUT_DIR / f"{prefix}_3_filtered.png"))
        save_image(heatmap, str(OUTPUT_DIR / f"{prefix}_4_diff_heatmap.png"))
        save_image(sal_overlay, str(OUTPUT_DIR / f"{prefix}_5_saliency.png"))
        save_image(seg_overlay, str(OUTPUT_DIR / f"{prefix}_6_segnet_boundary.png"))

    print(f"\n[visual] Done! {len(pair_indices)} frames × 6 views = {len(pair_indices) * 6} images")
    print(f"[visual] Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
