#!/usr/bin/env python3
"""CLI wrapper for 6-panel analysis visualization.

Loads TTO frames from a ``.pt`` file and generates analysis panels
comparing reconstruction quality against the ground truth video.

Usage::

    PYTHONPATH=src:upstream python scripts/generate_analysis_viz.py \\
        --frames /tmp/tto_v5a_frames.pt \\
        --upstream upstream/ \\
        --output ~/Downloads/ \\
        --auth-matched

    # Quick test at model resolution (faster, no upscaling):
    PYTHONPATH=src:upstream python scripts/generate_analysis_viz.py \\
        --frames /tmp/tto_frames.pt \\
        --upstream upstream/ \\
        --output /tmp/viz/ \\
        --clip-start 10 --clip-duration 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate 6-panel analysis visualization from TTO frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--frames",
        type=str,
        required=True,
        help="Path to TTO frames tensor (.pt file, shape (N, 384, 512, 3) uint8)",
    )
    p.add_argument(
        "--upstream",
        type=str,
        default="upstream",
        help="Path to upstream repo (contains videos/, models/, modules.py)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory for GIF and MP4",
    )
    p.add_argument(
        "--auth-matched",
        action="store_true",
        help="Upscale to camera resolution (874x1164) to match auth scorer",
    )
    p.add_argument(
        "--clip-start",
        type=float,
        default=12.0,
        help="Clip start time in seconds",
    )
    p.add_argument(
        "--clip-duration",
        type=float,
        default=6.0,
        help="Clip duration in seconds",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video frame rate",
    )
    p.add_argument(
        "--gif-scale",
        type=float,
        default=0.5,
        help="GIF downscale factor (0.5 = half resolution)",
    )
    p.add_argument(
        "--gif-fps",
        type=int,
        default=15,
        help="GIF playback frame rate",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (auto-detected if omitted)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    frames_path = Path(args.frames)
    upstream_dir = Path(args.upstream)
    output_dir = Path(args.output)

    if not frames_path.exists():
        print(f"ERROR: Frames file not found: {frames_path}", file=sys.stderr)
        sys.exit(1)

    gt_video_path = upstream_dir / "videos" / "0.mkv"
    if not gt_video_path.exists():
        print(f"ERROR: GT video not found: {gt_video_path}", file=sys.stderr)
        sys.exit(1)

    # Add upstream to sys.path for modules.py (SegNet/PoseNet definitions).
    upstream_str = str(upstream_dir.resolve())
    if upstream_str not in sys.path:
        sys.path.insert(0, upstream_str)

    import torch

    from tac.scorer import detect_device, load_scorers
    from tac.viz.analysis_panels import generate_analysis_panels

    device = args.device or str(detect_device())
    print(f"Device: {device}")

    # Load TTO frames.
    print(f"Loading TTO frames: {frames_path}")
    tto_frames = torch.load(str(frames_path), map_location="cpu", weights_only=True)
    print(f"  Shape: {tto_frames.shape}, dtype: {tto_frames.dtype}")

    if tto_frames.ndim != 4 or tto_frames.dtype != torch.uint8:
        print(
            f"ERROR: Expected (N, H, W, 3) uint8 tensor, "
            f"got shape={tto_frames.shape} dtype={tto_frames.dtype}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load SegNet (only need SegNet, not PoseNet, for this visualization).
    print("Loading SegNet scorer...")
    _, segnet = load_scorers(
        upstream_dir / "models" / "posenet.safetensors",
        upstream_dir / "models" / "segnet.safetensors",
        device=device,
        upstream_dir=upstream_str,
    )

    result = generate_analysis_panels(
        tto_frames=tto_frames,
        gt_video_path=gt_video_path,
        segnet=segnet,
        output_dir=output_dir,
        clip_start_sec=args.clip_start,
        clip_duration_sec=args.clip_duration,
        fps=args.fps,
        auth_matched=args.auth_matched,
        gif_scale=args.gif_scale,
        gif_fps=args.gif_fps,
    )

    print(f"\nResults:")
    print(f"  GIF:                {result['gif_path']}")
    print(f"  MP4:                {result['mp4_path'] or 'N/A (ffmpeg not found)'}")
    print(f"  SegNet disagree:    {result['seg_disagree_mean']:.4f}")
    print(f"  Pixel error (mean): {result['pixel_error_mean']:.2f}")
    print(f"  Frames rendered:    {result['n_frames_rendered']}")
    print(f"  Elapsed:            {result['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
