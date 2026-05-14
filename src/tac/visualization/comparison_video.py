#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a side-by-side SegNet overlay comparison video.

Produces a two-panel video matching the comma challenge README GIF format:
  Left  = Baseline (compressed, no post-filter) + SegNet class overlay
  Right = Ours (compressed + post-filter) + SegNet class overlay

Each panel shows the original frame with a semi-transparent SegNet class mask.
Labels at top show method name and score. A bottom strip shows the frame number
and a running count of pixel disagreements vs ground truth.

Output: MP4 at 4x speed (120 fps playback of 30 fps content), animated GIF,
and optionally individual frame PNGs.

Usage:
    tac viz-comparison \\
        --upstream workspace/upstream/comma_video_compression_challenge \\
        --archive submissions/robust_current/archive.zip \\
        --checkpoint submissions/robust_current/postfilter_int8.pt \\
        --output-dir reports/graphs/site/comparison \\
        --variant standard --hidden 64

Requires: torch, av (PyAV), PIL/Pillow, numpy, tac library.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch

from tac.versioned_output import versioned_write

# SegNet class colors (5 classes)
SEGNET_COLORS = np.array(
    [
        [64, 64, 64],  # 0: road (dark gray)
        [255, 255, 0],  # 1: lane marking (yellow)
        [0, 0, 255],  # 2: vehicle (blue)
        [255, 0, 0],  # 3: pedestrian (red)
        [0, 255, 0],  # 4: other (green)
    ],
    dtype=np.uint8,
)

OVERLAY_ALPHA = 0.4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate side-by-side SegNet overlay comparison video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--upstream",
        type=str,
        default="workspace/upstream/comma_video_compression_challenge",
        help="Path to upstream repo (contains videos/, models/, modules.py)",
    )
    p.add_argument(
        "--archive",
        type=str,
        default="submissions/robust_current/archive.zip",
        help="Path to compressed archive.zip",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="submissions/robust_current/postfilter_int8.pt",
        help="Path to int8 post-filter checkpoint (.pt)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="reports/graphs/site/comparison",
        help="Output directory for video, GIF, and frame PNGs",
    )
    p.add_argument("--variant", type=str, default="standard", help="Post-filter architecture variant")
    p.add_argument("--hidden", type=int, default=64, help="Post-filter hidden channel count")
    p.add_argument("--kernel", type=int, default=3, help="Post-filter kernel size")
    p.add_argument("--device", type=str, default=None, help="Compute device (auto-detected if omitted)")
    p.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    p.add_argument("--fps-playback", type=int, default=120, help="Playback FPS (4x speed = 120)")
    p.add_argument("--gif-stride", type=int, default=4, help="Only include every Nth frame in GIF (size control)")
    p.add_argument("--gif-fps", type=int, default=30, help="GIF playback FPS")
    p.add_argument("--save-pngs", action="store_true", help="Save individual frame PNGs")
    p.add_argument("--png-stride", type=int, default=50, help="Save every Nth frame as PNG")
    p.add_argument(
        "--baseline-score", type=float, default=None, help="Override baseline score label (computed if omitted)"
    )
    p.add_argument("--ours-score", type=float, default=None, help="Override our score label (computed if omitted)")
    p.add_argument("--label-height", type=int, default=48, help="Height of top label strip in pixels")
    p.add_argument("--bar-height", type=int, default=36, help="Height of bottom info strip in pixels")
    return p.parse_args()


def blend_overlay(frame_rgb: np.ndarray, class_map: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """Blend SegNet class colors onto a frame.

    Args:
        frame_rgb: (H, W, 3) uint8 RGB frame
        class_map: (H, W) int class indices (0-4)
        alpha: overlay opacity

    Returns:
        (H, W, 3) uint8 blended frame
    """
    color_mask = SEGNET_COLORS[class_map]  # (H, W, 3)
    blended = (1.0 - alpha) * frame_rgb.astype(np.float32) + alpha * color_mask.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_text_bar(
    width: int,
    height: int,
    texts: list[str],
    bg_color: tuple[int, int, int] = (12, 16, 22),
    text_color: tuple[int, int, int] = (220, 225, 232),
    font_size: int = 18,
) -> np.ndarray:
    """Draw a horizontal bar with centered text labels using PIL.

    Args:
        width: total bar width
        height: bar height
        texts: list of text strings, evenly spaced across width
        bg_color: background RGB
        text_color: text RGB
        font_size: approximate font size

    Returns:
        (height, width, 3) uint8 RGB array
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Try to load a clean font; fall back to default
    font = None
    for candidate in [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]:
        if os.path.exists(candidate):
            try:
                font = ImageFont.truetype(candidate, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    section_width = width // len(texts)
    for i, text in enumerate(texts):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = i * section_width + (section_width - tw) // 2
        y = (height - th) // 2
        draw.text((x, y), text, fill=text_color, font=font)

    return np.array(img)


def get_segnet_classes(frame_chw: torch.Tensor, segnet) -> torch.Tensor:
    """Get per-pixel argmax class from SegNet for a single frame.

    Args:
        frame_chw: (1, 3, H, W) float tensor
        segnet: loaded SegNet model

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


def main() -> None:
    args = parse_args()

    upstream_dir = Path(args.upstream)
    upstream_str = str(upstream_dir)
    if upstream_str not in sys.path:
        sys.path.insert(0, upstream_str)

    # Lazy imports after path setup
    import av
    from PIL import Image

    from tac.architectures import build_postfilter
    from tac.data import decode_archive, decode_video
    from tac.quantization import load_int8
    from tac.scorer import detect_device, load_scorers

    device = args.device or str(detect_device())
    print(f"Device: {device}")

    # Load post-filter
    checkpoint_path = Path(args.checkpoint)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
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
    num_frames = len(comp_frames)
    if args.max_frames > 0:
        num_frames = min(num_frames, args.max_frames)
    H, W = comp_frames[0].shape[:2]
    print(f"Processing {num_frames} of {len(comp_frames)} frames, resolution: {W}x{H}")

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_pngs:
        (out_dir / "frames").mkdir(exist_ok=True)

    # Composite dimensions
    panel_w = W
    total_w = panel_w * 2
    label_h = args.label_height
    bar_h = args.bar_height
    total_h = label_h + H + bar_h

    # Set up MP4 output -- write to a temp name, then version it
    from datetime import datetime as _dt
    _config_tag = f"{args.variant}_h{args.hidden}"
    _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    mp4_versioned = out_dir / f"comparison_{_ts}_{_config_tag}.mp4"
    mp4_path = out_dir / "comparison.mp4"
    out_container = av.open(str(mp4_versioned), mode="w")
    stream = out_container.add_stream("libx264", rate=args.fps_playback)
    stream.width = total_w
    stream.height = total_h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}

    # For GIF: collect PIL frames
    gif_frames: list[Image.Image] = []

    # Running disagreement counters
    total_baseline_disagree = 0
    total_filtered_disagree = 0

    print("Generating comparison frames...")
    with torch.no_grad():
        for idx in range(num_frames):
            comp = comp_frames[idx].to(device)  # (H, W, 3) uint8
            gt = gt_frames[idx].to(device)

            # Baseline: compressed frame as-is (already upscaled by decode_video)
            baseline_chw = comp.float().permute(2, 0, 1).unsqueeze(0).contiguous()

            # Filtered: apply post-filter
            filtered_chw = model(baseline_chw)
            filtered_chw = filtered_chw.round().clamp(0, 255)

            # GT for SegNet reference
            gt_chw = gt.float().permute(2, 0, 1).unsqueeze(0).contiguous()

            # Get SegNet class maps
            baseline_cls = get_segnet_classes(baseline_chw, segnet)
            filtered_cls = get_segnet_classes(filtered_chw, segnet)
            gt_cls = get_segnet_classes(gt_chw, segnet)

            # Disagreements vs GT
            baseline_disagree = int((baseline_cls != gt_cls).sum().item())
            filtered_disagree = int((filtered_cls != gt_cls).sum().item())
            total_baseline_disagree += baseline_disagree
            total_filtered_disagree += filtered_disagree

            # Convert to numpy for compositing
            baseline_np = comp.cpu().numpy()  # (H, W, 3) uint8
            filtered_np = filtered_chw[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            baseline_cls_np = baseline_cls.cpu().numpy()
            filtered_cls_np = filtered_cls.cpu().numpy()

            # Resize class maps if they differ from frame size
            if baseline_cls_np.shape != (H, W):
                from PIL import Image as _Img

                baseline_cls_np = np.array(
                    _Img.fromarray(baseline_cls_np.astype(np.uint8)).resize((W, H), _Img.NEAREST)
                )
                filtered_cls_np = np.array(
                    _Img.fromarray(filtered_cls_np.astype(np.uint8)).resize((W, H), _Img.NEAREST)
                )

            # Blend overlays
            left_panel = blend_overlay(baseline_np, baseline_cls_np)
            right_panel = blend_overlay(filtered_np, filtered_cls_np)

            # Build top label bar
            baseline_label = "Baseline (no filter)"
            ours_label = "Ours (post-filter)"
            if args.baseline_score is not None:
                baseline_label = f"Baseline (score: {args.baseline_score:.2f})"
            if args.ours_score is not None:
                ours_label = f"Ours (score: {args.ours_score:.2f})"
            top_bar = draw_text_bar(total_w, label_h, [baseline_label, ours_label], font_size=20)

            # Build bottom info bar
            reduction_pct = 0.0
            if total_baseline_disagree > 0:
                reduction_pct = (1.0 - total_filtered_disagree / total_baseline_disagree) * 100
            bottom_texts = [
                f"Frame {idx + 1}/{num_frames}",
                f"Baseline disagree: {baseline_disagree:,}",
                f"Filtered disagree: {filtered_disagree:,}",
                f"Cumulative reduction: {reduction_pct:.1f}%",
            ]
            bottom_bar = draw_text_bar(total_w, bar_h, bottom_texts, font_size=14)

            # Assemble composite frame
            side_by_side = np.concatenate([left_panel, right_panel], axis=1)  # (H, 2W, 3)
            composite = np.concatenate([top_bar, side_by_side, bottom_bar], axis=0)  # (total_h, total_w, 3)

            # Write to MP4
            av_frame = av.VideoFrame.from_ndarray(composite, format="rgb24")
            for packet in stream.encode(av_frame):
                out_container.mux(packet)

            # Collect for GIF (with stride)
            if idx % args.gif_stride == 0:
                gif_frames.append(Image.fromarray(composite))

            # Save PNG
            if args.save_pngs and idx % args.png_stride == 0:
                png_path = out_dir / "frames" / f"frame_{idx:04d}.png"
                Image.fromarray(composite).save(str(png_path))

            if (idx + 1) % 100 == 0 or idx == 0:
                print(
                    f"  [{(idx + 1) / num_frames * 100:5.1f}%] frame {idx}: "
                    f"baseline={baseline_disagree:,} filtered={filtered_disagree:,}"
                )

    # Flush MP4
    for packet in stream.encode():
        out_container.mux(packet)
    out_container.close()
    # Create latest symlink for MP4
    from tac.versioned_output import _update_latest_link
    _update_latest_link(mp4_path, mp4_versioned)
    print(f"Wrote MP4: {mp4_versioned} ({mp4_versioned.stat().st_size / (1024 * 1024):.1f} MB)")

    # Write GIF
    if gif_frames:
        gif_path = out_dir / "comparison.gif"
        # Downscale GIF frames to keep file size manageable
        gif_w = total_w // 2
        gif_h = total_h // 2
        resized = [f.resize((gif_w, gif_h), Image.LANCZOS) for f in gif_frames]
        duration_ms = int(1000 / args.gif_fps)
        gif_buf = io.BytesIO()
        resized[0].save(
            gif_buf,
            format="GIF",
            save_all=True,
            append_images=resized[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        gif_versioned = versioned_write(gif_path, gif_buf.getvalue(), config_tag=_config_tag)
        print(f"Wrote GIF: {gif_versioned} ({gif_versioned.stat().st_size / (1024 * 1024):.1f} MB)")

    # Summary
    if total_baseline_disagree > 0:
        reduction = (1 - total_filtered_disagree / total_baseline_disagree) * 100
        print(f"\nTotal baseline disagreements: {total_baseline_disagree:,}")
        print(f"Total filtered disagreements: {total_filtered_disagree:,}")
        print(f"Overall SegNet disagreement reduction: {reduction:.1f}%")

    print(f"\nOutputs in: {out_dir}")


if __name__ == "__main__":
    main()
