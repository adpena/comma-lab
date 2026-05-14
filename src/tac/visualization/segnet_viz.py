#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate SegNet visualization data for the segnet-comparison.html page.

Produces a JSON file with per-frame SegNet class predictions for:
  - Baseline (compressed, no post-filter)
  - Filtered (compressed + post-filter)
  - Ground truth

Usage:
    tac viz-segnet \\
        --checkpoint weights/best_int8.pt \\
        --archive submissions/robust_current/archive.zip \\
        --upstream /path/to/upstream \\
        --output reports/graphs/site/segnet-viz-data.json \\
        --stride 50 \\
        --variant standard \\
        --hidden 64

Output JSON schema:
    {
        "width": 1164,
        "height": 874,
        "stride": 50,
        "num_source_frames": 1200,
        "checkpoint": "...",
        "frames": [
            {
                "frame_idx": 0,
                "baseline_disagreements": 1234,
                "filtered_disagreements": 567,
                "baseline_classes": [...],   // flat H*W array of class indices (0-4)
                "filtered_classes": [...],
                "gt_classes": [...]
            },
            ...
        ]
    }

Requires: torch, tac library (src/tac), upstream modules on sys.path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from tac.versioned_output import versioned_write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SegNet class-comparison data for the visualization page.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to int8 post-filter checkpoint (.pt)",
    )
    parser.add_argument(
        "--archive",
        type=str,
        required=True,
        help="Path to compressed archive.zip",
    )
    parser.add_argument(
        "--upstream",
        type=str,
        required=True,
        help="Path to upstream repo (contains videos/, models/, modules.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/graphs/site/segnet-viz-data.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=50,
        help="Sample every Nth frame (e.g., 50 means frames 0, 50, 100, ...)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="standard",
        help="Post-filter architecture variant",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=64,
        help="Post-filter hidden channel count",
    )
    parser.add_argument(
        "--kernel",
        type=int,
        default=3,
        help="Post-filter kernel size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (auto-detected if omitted)",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=4,
        help="Downscale class maps by this factor to reduce JSON size (1 = full res)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve paths
    checkpoint_path = Path(args.checkpoint)
    archive_path = Path(args.archive)
    upstream_dir = Path(args.upstream)
    output_path = Path(args.output)

    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    assert archive_path.exists(), f"Archive not found: {archive_path}"
    assert upstream_dir.exists(), f"Upstream dir not found: {upstream_dir}"

    # Add upstream to sys.path for modules.py
    upstream_str = str(upstream_dir)
    if upstream_str not in sys.path:
        sys.path.insert(0, upstream_str)

    # Lazy imports after path setup
    from tac.architectures import build_postfilter
    from tac.data import decode_archive, decode_video
    from tac.quantization import load_int8
    from tac.scorer import detect_device, load_scorers

    device = args.device or str(detect_device())
    print(f"Device: {device}")

    # Load post-filter
    print(f"Loading post-filter: {args.variant} h={args.hidden} k={args.kernel}")
    model = build_postfilter(args.variant, hidden=args.hidden, kernel=args.kernel)
    load_int8(str(checkpoint_path), model, device=device)
    model = model.eval().to(device)

    # Load SegNet
    print("Loading SegNet scorer...")
    _, segnet = load_scorers(
        upstream_dir / "models" / "posenet.safetensors",
        upstream_dir / "models" / "segnet.safetensors",
        device=device,
        upstream_dir=upstream_str,
    )

    # Decode frames
    print(f"Decoding archive: {archive_path}")
    comp_frames = decode_archive(str(archive_path))
    print(f"Decoding GT video: {upstream_dir / 'videos' / '0.mkv'}")
    gt_frames = decode_video(str(upstream_dir / "videos" / "0.mkv"))

    assert len(comp_frames) == len(gt_frames), (
        f"Frame count mismatch: {len(comp_frames)} vs {len(gt_frames)}"
    )
    num_frames = len(comp_frames)
    H, W = comp_frames[0].shape[:2]
    print(f"Total frames: {num_frames}, resolution: {W}x{H}")

    # Select frames to sample
    sample_indices = list(range(0, num_frames, args.stride))
    print(f"Sampling {len(sample_indices)} frames (stride={args.stride})")

    downscale = max(1, args.downscale)
    out_h = H // downscale
    out_w = W // downscale

    results = []
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            comp = comp_frames[idx].to(device)  # (H, W, 3) uint8
            gt = gt_frames[idx].to(device)

            # Apply post-filter to compressed frame
            comp_chw = comp.float().permute(2, 0, 1).unsqueeze(0).contiguous()  # (1, 3, H, W)
            filtered_chw = model(comp_chw)
            filtered_chw = filtered_chw.round().clamp(0, 255).to(torch.uint8).float()

            # SegNet expects (B, T, C, H, W) with T=2, but for single-frame class maps
            # we duplicate the frame to form a pair (scorer requires pairs).
            # We only care about the class map, so duplication is fine.
            def get_segnet_classes(frame_chw: torch.Tensor) -> torch.Tensor:
                """Get per-pixel argmax class from SegNet for a single frame.

                Args:
                    frame_chw: (1, 3, H, W) float tensor

                Returns:
                    (H, W) int tensor of class indices
                """
                # SegNet.preprocess_input expects (B, T, C, H, W)
                pair = frame_chw.unsqueeze(1).expand(-1, 2, -1, -1, -1)
                seg_input = segnet.preprocess_input(pair)
                seg_output = segnet(seg_input)
                # seg_output shape: (B*T, num_classes, H', W')
                # Take first frame from pair
                classes = seg_output[0].argmax(dim=0)  # (H', W')
                return classes

            baseline_cls = get_segnet_classes(comp_chw)
            filtered_cls = get_segnet_classes(filtered_chw)
            gt_cls = get_segnet_classes(gt.float().permute(2, 0, 1).unsqueeze(0).contiguous())

            # Compute full-res disagreements before downscaling
            baseline_disagree = int((baseline_cls != gt_cls).sum().item())
            filtered_disagree = int((filtered_cls != gt_cls).sum().item())

            # Downscale class maps for JSON size
            if downscale > 1:
                # Nearest-neighbor downscale
                baseline_cls = baseline_cls[::downscale, ::downscale]
                filtered_cls = filtered_cls[::downscale, ::downscale]
                gt_cls = gt_cls[::downscale, ::downscale]

            frame_data = {
                "frame_idx": idx,
                "baseline_disagreements": baseline_disagree,
                "filtered_disagreements": filtered_disagree,
                "baseline_classes": baseline_cls.cpu().tolist(),
                "filtered_classes": filtered_cls.cpu().tolist(),
                "gt_classes": gt_cls.cpu().tolist(),
            }
            results.append(frame_data)

            if (i + 1) % 5 == 0 or i == 0:
                pct = (i + 1) / len(sample_indices) * 100
                print(
                    f"  [{pct:5.1f}%] frame {idx}: "
                    f"baseline={baseline_disagree:,} filtered={filtered_disagree:,} "
                    f"(reduction: {(1 - filtered_disagree / max(baseline_disagree, 1)) * 100:.1f}%)"
                )

    # Flatten 2D class arrays to 1D for compact JSON
    for frame_data in results:
        for key in ("baseline_classes", "filtered_classes", "gt_classes"):
            arr = frame_data[key]
            if arr and isinstance(arr[0], list):
                frame_data[key] = [v for row in arr for v in row]

    output = {
        "width": out_w,
        "height": out_h,
        "stride": args.stride,
        "downscale": downscale,
        "full_width": W,
        "full_height": H,
        "num_source_frames": num_frames,
        "checkpoint": str(checkpoint_path),
        "variant": args.variant,
        "hidden": args.hidden,
        "frames": results,
    }

    config_tag = f"{args.variant}_h{args.hidden}"
    versioned_write(
        output_path,
        json.dumps(output, separators=(",", ":")),
        config_tag=config_tag,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {output_path} ({size_mb:.1f} MB)")
    print(f"  {len(results)} frames, {out_w}x{out_h} class maps (downscale={downscale})")

    # Summary
    total_baseline = sum(f["baseline_disagreements"] for f in results)
    total_filtered = sum(f["filtered_disagreements"] for f in results)
    if total_baseline > 0:
        reduction = (1 - total_filtered / total_baseline) * 100
        print(f"  Total baseline disagreements: {total_baseline:,}")
        print(f"  Total filtered disagreements: {total_filtered:,}")
        print(f"  Overall reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()
