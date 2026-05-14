#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a 3-panel YUV Y00 channel comparison GIF showing what PoseNet sees.

PoseNet processes YUV 4:2:0, not RGB. The upstream frame_utils.py has
rgb_to_yuv6() which converts RGB to 6 channels: y00, y10, y01, y11, U_sub,
V_sub. The y00 channel (luma at half-res) is where compression artifacts are
most visible.

Three panels side by side:
  Left:   Original (GT) Y00 channel
  Center: Baseline (compressed, no filter) Y00 channel
  Right:  Ours (post-filtered) Y00 channel

Specs: 512x384 total, dark_background, monospace labels, 150 frames @ 100ms.

Usage:
    tac viz-yuv-gif \\
        --upstream workspace/upstream/comma_video_compression_challenge \\
        --archive submissions/robust_current/archive.zip \\
        --checkpoint submissions/robust_current/postfilter_int8.pt \\
        --output reports/graphs/site/yuv_y00_comparison.gif \\
        --variant standard --hidden 64
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch

from tac.versioned_output import versioned_write

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate 3-panel YUV Y00 channel comparison GIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--upstream", type=str,
        default="workspace/upstream/comma_video_compression_challenge",
        help="Path to upstream repo (contains videos/, models/, frame_utils.py)",
    )
    p.add_argument(
        "--archive", type=str,
        default="submissions/robust_current/archive.zip",
        help="Path to compressed archive.zip",
    )
    p.add_argument(
        "--checkpoint", type=str,
        default="submissions/robust_current/postfilter_int8.pt",
        help="Path to int8 post-filter checkpoint (.pt)",
    )
    p.add_argument(
        "--output", type=str,
        default="reports/graphs/site/yuv_y00_comparison.gif",
        help="Output GIF path",
    )
    p.add_argument("--variant", type=str, default="standard", help="Post-filter architecture variant")
    p.add_argument("--hidden", type=int, default=64, help="Post-filter hidden channel count")
    p.add_argument("--kernel", type=int, default=3, help="Post-filter kernel size")
    p.add_argument("--device", type=str, default=None, help="Compute device (auto-detected if omitted)")
    p.add_argument("--num-frames", type=int, default=150, help="Number of GIF frames to produce")
    p.add_argument("--frame-duration", type=int, default=100, help="GIF frame duration in ms")
    p.add_argument("--total-video-frames", type=int, default=1200, help="Total frames in video")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# 512x384 px at 100 dpi
FIG_W_IN = 5.12
FIG_H_IN = 3.84
DPI = 100


def _render_frame(
    *,
    gt_y00: np.ndarray,
    baseline_y00: np.ndarray,
    ours_y00: np.ndarray,
) -> np.ndarray:
    """Render one 512x384 frame with 3 horizontal Y00 panels.

    Returns (384, 512, 3) uint8 numpy array.
    """
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 3, figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)

        panels = [
            (axes[0], gt_y00, "original (y00)"),
            (axes[1], baseline_y00, "compressed (y00)"),
            (axes[2], ours_y00, "ours (y00)"),
        ]

        for ax, img, label in panels:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(label, fontsize=7, fontfamily="monospace",
                         color="#cccccc", pad=2)
            ax.axis("off")

        fig.tight_layout(pad=0.4)

        buf = io.BytesIO()
        fig.savefig(buf, format="raw", dpi=DPI, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)

        w = int(fig.get_size_inches()[0] * DPI)
        h = int(fig.get_size_inches()[1] * DPI)
        raw = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(h, w, 4)
        return raw[:, :, :3]  # drop alpha


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    upstream_dir = Path(args.upstream)
    upstream_str = str(upstream_dir)
    if upstream_str not in sys.path:
        sys.path.insert(0, upstream_str)

    from PIL import Image

    from frame_utils import rgb_to_yuv6
    from tac.architectures import build_postfilter
    from tac.data import decode_archive, decode_video
    from tac.quantization import load_int8
    from tac.scorer import detect_device

    device = args.device or str(detect_device())
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load post-filter model
    # ------------------------------------------------------------------
    checkpoint_path = Path(args.checkpoint)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    print(f"Loading post-filter: {args.variant} h={args.hidden} k={args.kernel}")
    model = build_postfilter(args.variant, hidden=args.hidden, kernel=args.kernel)
    load_int8(str(checkpoint_path), model, device=device)
    model = model.eval().to(device)

    # ------------------------------------------------------------------
    # Decode frames
    # ------------------------------------------------------------------
    archive_path = Path(args.archive)
    assert archive_path.exists(), f"Archive not found: {archive_path}"
    gt_video_path = upstream_dir / "videos" / "0.mkv"
    assert gt_video_path.exists(), f"GT video not found: {gt_video_path}"

    print(f"Decoding archive: {archive_path}")
    comp_frames = decode_archive(str(archive_path))
    print(f"Decoding GT video: {gt_video_path}")
    gt_frames = decode_video(str(gt_video_path))

    assert len(comp_frames) == len(gt_frames), (
        f"Frame count mismatch: {len(comp_frames)} vs {len(gt_frames)}"
    )
    total_frames = len(comp_frames)
    print(f"Total video frames: {total_frames}")

    # Sample every 8th frame, take up to num_frames
    stride = 8
    sample_indices = list(range(0, total_frames, stride))[: args.num_frames]
    print(f"Sampling {len(sample_indices)} frames (stride={stride})")

    # ------------------------------------------------------------------
    # Generate GIF frames
    # ------------------------------------------------------------------
    gif_images: list[Image.Image] = []

    print("Generating GIF frames...")
    with torch.no_grad():
        for fi, idx in enumerate(sample_indices):
            comp = comp_frames[idx].to(device)  # (H, W, 3) uint8
            gt = gt_frames[idx].to(device)

            # CHW float for models and yuv conversion
            baseline_chw = comp.float().permute(2, 0, 1).unsqueeze(0).contiguous()
            gt_chw = gt.float().permute(2, 0, 1).unsqueeze(0).contiguous()

            # Post-filter
            filtered_chw = model(baseline_chw).round().clamp(0, 255)

            # Extract Y00 channel from each: rgb_to_yuv6 expects (..., 3, H, W)
            # returns (..., 6, H/2, W/2); channel 0 is y00
            gt_yuv6 = rgb_to_yuv6(gt_chw)           # (1, 6, H/2, W/2)
            baseline_yuv6 = rgb_to_yuv6(baseline_chw)
            ours_yuv6 = rgb_to_yuv6(filtered_chw)

            gt_y00 = gt_yuv6[0, 0].cpu().numpy()           # (H/2, W/2)
            baseline_y00 = baseline_yuv6[0, 0].cpu().numpy()
            ours_y00 = ours_yuv6[0, 0].cpu().numpy()

            frame_rgb = _render_frame(
                gt_y00=gt_y00,
                baseline_y00=baseline_y00,
                ours_y00=ours_y00,
            )

            gif_images.append(Image.fromarray(frame_rgb))

            if (fi + 1) % 25 == 0 or fi == 0:
                print(f"  [{fi + 1}/{len(sample_indices)}] video frame {idx}")

    # ------------------------------------------------------------------
    # Save GIF
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gif_buf = io.BytesIO()
    gif_images[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=gif_images[1:],
        duration=args.frame_duration,
        loop=0,
        optimize=True,
    )
    config_tag = f"{args.variant}_h{args.hidden}"
    versioned_path = versioned_write(out_path, gif_buf.getvalue(), config_tag=config_tag)
    size_mb = versioned_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote GIF: {versioned_path} ({size_mb:.1f} MB)")
    print(f"  {len(gif_images)} frames, {args.frame_duration}ms/frame, 512x384 px")


if __name__ == "__main__":
    main()
