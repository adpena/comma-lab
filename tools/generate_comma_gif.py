#!/usr/bin/env python3
"""Generate a 512x384 animated GIF replicating the comma.ai challenge README format.

Produces a pixel-perfect 2x2 grid matching the original comma challenge
GIF specs:

  512x384 px, 150 frames, 100ms/frame (10fps), black background.

  Top-left:     video frame (original or baseline)
  Top-right:    video frame (compressed or ours)
  Bottom-left:  segnet errors (binary white-on-black, upscaled nearest)
  Bottom-right: posenet errors (progressive line chart, minimal axes)

Each panel has a small white monospace label overlaid at the top-left corner
(not matplotlib titles -- raw PIL text on the composited image).

The comma README GIF uses edge-to-edge panels with no matplotlib chrome:
no tick marks, no axis labels, no spines on image panels, and only minimal
spines/ticks on the PoseNet chart.

Three modes:
  --mode baseline   Replicate the original comma GIF (original vs compressed).
  --mode ours       Replace compressed with post-filtered output.
  --mode comparison Dual traces on PoseNet chart; top panels show baseline vs ours.

Usage:
    python tools/generate_comma_gif.py \\
        --upstream workspace/upstream/comma_video_compression_challenge \\
        --archive submissions/robust_current/archive.zip \\
        --checkpoint submissions/robust_current/postfilter_int8.pt \\
        --output reports/graphs/site/comma_readme.gif \\
        --mode ours --variant standard --hidden 64
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate comma-challenge-style 512x384 animated GIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--upstream", type=str,
        default="workspace/upstream/comma_video_compression_challenge",
        help="Path to upstream repo (contains videos/, models/, modules.py)",
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
        default="reports/graphs/site/comma_readme.gif",
        help="Output GIF path",
    )
    p.add_argument(
        "--mode", type=str, default="baseline",
        choices=["baseline", "ours", "comparison"],
        help="GIF mode: baseline (comma replica), ours (post-filter), comparison (both)",
    )
    p.add_argument("--variant", type=str, default="standard", help="Post-filter architecture variant")
    p.add_argument("--hidden", type=int, default=64, help="Post-filter hidden channel count")
    p.add_argument("--kernel", type=int, default=3, help="Post-filter kernel size")
    p.add_argument("--device", type=str, default=None, help="Compute device (auto-detected if omitted)")
    p.add_argument("--num-frames", type=int, default=150, help="Number of GIF frames to produce")
    p.add_argument("--frame-duration", type=int, default=100, help="GIF frame duration in ms")
    p.add_argument("--total-video-frames", type=int, default=1200, help="Total frames in video (for stride calc)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model helpers (same API as generate_comma_format_video.py)
# ---------------------------------------------------------------------------

def get_segnet_classes(frame_chw: torch.Tensor, segnet) -> np.ndarray:
    """Run SegNet on a (1,3,H,W) float tensor, return (Hs,Ws) int argmax."""
    pair = frame_chw.unsqueeze(1).expand(-1, 2, -1, -1, -1)
    seg_input = segnet.preprocess_input(pair)
    seg_output = segnet(seg_input)
    return seg_output[0].argmax(dim=0).cpu().numpy()


def get_posenet_mse(
    frame_a: torch.Tensor, frame_b: torch.Tensor,
    gt_a: torch.Tensor, gt_b: torch.Tensor,
    posenet,
) -> float:
    """PoseNet distortion (MSE) between consecutive-frame pairs."""
    comp_pair = torch.cat([frame_a, frame_b], dim=0).unsqueeze(0)
    gt_pair = torch.cat([gt_a, gt_b], dim=0).unsqueeze(0)
    comp_out = posenet(posenet.preprocess_input(comp_pair))
    gt_out = posenet(posenet.preprocess_input(gt_pair))
    return posenet.compute_distortion(comp_out, gt_out).item()


# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------

_MONO_FONT_CACHE: dict[int, "ImageFont.FreeTypeFont | ImageFont.ImageFont"] = {}


def _get_mono_font(size: int = 11):
    """Load a monospace font for PIL text drawing, with caching."""
    if size in _MONO_FONT_CACHE:
        return _MONO_FONT_CACHE[size]

    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    font = None
    for path in candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    _MONO_FONT_CACHE[size] = font
    return font


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

# Output canvas: 512 x 384 pixels. Each quadrant is 256 x 192.
CANVAS_W = 512
CANVAS_H = 384
PANEL_W = CANVAS_W // 2   # 256
PANEL_H = CANVAS_H // 2   # 192

# PoseNet chart: rendered via matplotlib at exact panel pixel size.
CHART_DPI = 100
CHART_W_IN = PANEL_W / CHART_DPI   # 2.56 inches
CHART_H_IN = PANEL_H / CHART_DPI   # 1.92 inches

# Label styling
LABEL_FONT_SIZE = 10
LABEL_COLOR = (200, 200, 200)
LABEL_SHADOW_COLOR = (0, 0, 0)
LABEL_PAD_X = 4
LABEL_PAD_Y = 3


def _resize_frame_to_panel(img: np.ndarray) -> np.ndarray:
    """Resize an (H, W, 3) uint8 image to (PANEL_H, PANEL_W, 3) using Lanczos."""
    from PIL import Image
    pil = Image.fromarray(img).resize((PANEL_W, PANEL_H), Image.LANCZOS)
    return np.array(pil)


def _upscale_segnet_mask(mask: np.ndarray) -> np.ndarray:
    """Upscale a (Hs, Ws) binary mask to (PANEL_H, PANEL_W, 3) with nearest neighbor.

    White pixels where argmax differs; black background. Nearest-neighbor
    preserves the hard binary look of the comma GIF.
    """
    from PIL import Image
    # mask is (Hs, Ws) with values 0 or 255
    pil = Image.fromarray(mask, mode="L").resize((PANEL_W, PANEL_H), Image.NEAREST)
    mono = np.array(pil)
    return np.stack([mono, mono, mono], axis=-1)


def _upscale_segnet_diff(
    ours_mask: np.ndarray,
    baseline_mask: np.ndarray,
) -> np.ndarray:
    """Color-coded SegNet error diff between ours and baseline.

    Returns (PANEL_H, PANEL_W, 3) uint8 with:
      Red:   error in baseline only (we FIXED it)
      Green: error in ours only (we INTRODUCED it — regression)
      White: error in both (shared, irreducible)
      Black: no error in either
    """
    from PIL import Image
    ours_up = np.array(Image.fromarray(ours_mask, mode="L").resize(
        (PANEL_W, PANEL_H), Image.NEAREST)) > 127
    base_up = np.array(Image.fromarray(baseline_mask, mode="L").resize(
        (PANEL_W, PANEL_H), Image.NEAREST)) > 127

    canvas = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

    # White: both have error (shared)
    both = ours_up & base_up
    canvas[both] = [255, 255, 255]

    # Red: baseline only (we fixed)
    fixed = base_up & ~ours_up
    canvas[fixed] = [255, 60, 60]

    # Green: ours only (regression)
    regressed = ours_up & ~base_up
    canvas[regressed] = [60, 255, 60]

    return canvas


def _render_posenet_chart(
    *,
    xs: list[int],
    primary: list[float],
    secondary: list[float] | None,
    mode: str,
    total_frames: int,
) -> np.ndarray:
    """Render PoseNet error chart as (PANEL_H, PANEL_W, 3) uint8 array.

    Matches comma style: black background, minimal decoration, thin line,
    no axis labels, tiny tick labels, dark gray spines.
    """
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(
            figsize=(CHART_W_IN, CHART_H_IN), dpi=CHART_DPI,
        )
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        # Always show baseline as faint solid reference when available
        if secondary is not None and len(secondary) > 0:
            ax.plot(xs[:len(secondary)], secondary, color="#888888",
                    linestyle="-", linewidth=0.7, label="baseline", alpha=0.5)

        if len(xs) > 0 and len(primary) > 0:
            line_color = "#00ff88" if mode in ("ours", "comparison") else "#ffffff"
            ax.plot(xs[:len(primary)], primary, color=line_color,
                    linewidth=0.9, label="ours" if mode != "baseline" else "compressed")

        if secondary is not None and len(secondary) > 0:
            ax.legend(
                loc="upper right", fontsize=5, frameon=True,
                facecolor="black", edgecolor="#333333", labelcolor="white",
                handlelength=1.2, borderpad=0.3, labelspacing=0.2,
            )

        # X axis: full video range
        ax.set_xlim(0, total_frames)

        # Y axis: auto-scale with headroom, floor at a small positive value
        all_vals = list(primary)
        if secondary:
            all_vals.extend(secondary)
        if all_vals:
            ymax = max(all_vals) * 1.4
            ymax = max(ymax, 0.01)
        else:
            ymax = 1.0
        ax.set_ylim(0, ymax)

        # Minimal tick styling -- tiny white labels, no axis labels
        ax.tick_params(
            axis="both", which="both",
            colors="#888888", labelsize=5,
            length=2, width=0.5, pad=1,
        )
        # Only a few y-ticks
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, min_n_ticks=2))
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, min_n_ticks=2))

        # Dark gray spines, thin
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.5)

        # No axis labels (the panel label says "posenet errors")
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Tight layout to minimize wasted space
        fig.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.12)

        # Render to exact pixel buffer using raw RGBA format (no PNG encode/decode
        # overhead) -- gives us a flat numpy-friendly byte array at exact DPI.
        buf = io.BytesIO()
        fig.savefig(buf, format="raw", dpi=CHART_DPI, facecolor="black")
        plt.close(fig)
        buf.seek(0)

        w = int(fig.get_size_inches()[0] * CHART_DPI)
        h = int(fig.get_size_inches()[1] * CHART_DPI)
        raw = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(h, w, 4)
        return raw[:PANEL_H, :PANEL_W, :3]


def _composite_frame(
    *,
    top_left_img: np.ndarray,
    top_right_img: np.ndarray,
    segnet_panel: np.ndarray,
    chart_rgb: np.ndarray,
    top_left_label: str,
    top_right_label: str,
) -> np.ndarray:
    """Compose the 4 panels into a single 512x384 canvas with text labels.

    All panels are already (PANEL_H, PANEL_W, 3) uint8.
    Text labels are drawn directly onto the image using PIL (not matplotlib
    titles) for a cleaner, comma-style look.

    Returns (CANVAS_H, CANVAS_W, 3) uint8 array.
    """
    from PIL import Image, ImageDraw

    # Resize video frames to panel size
    tl = _resize_frame_to_panel(top_left_img)
    tr = _resize_frame_to_panel(top_right_img)
    bl = segnet_panel  # already (PANEL_H, PANEL_W, 3) — either mono or color-coded diff
    br = chart_rgb  # already at panel size

    # Assemble 2x2 grid
    top_row = np.concatenate([tl, tr], axis=1)   # (PANEL_H, CANVAS_W, 3)
    bot_row = np.concatenate([bl, br], axis=1)
    canvas = np.concatenate([top_row, bot_row], axis=0)  # (CANVAS_H, CANVAS_W, 3)

    # Draw text labels with PIL
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    font = _get_mono_font(LABEL_FONT_SIZE)

    labels = [
        (LABEL_PAD_X, LABEL_PAD_Y, top_left_label),
        (PANEL_W + LABEL_PAD_X, LABEL_PAD_Y, top_right_label),
        (LABEL_PAD_X, PANEL_H + LABEL_PAD_Y, "segnet errors"),
        (PANEL_W + LABEL_PAD_X, PANEL_H + LABEL_PAD_Y, "posenet errors"),
    ]
    for x, y, text in labels:
        # Shadow for readability on bright frames
        draw.text((x + 1, y + 1), text, fill=LABEL_SHADOW_COLOR, font=font)
        draw.text((x, y), text, fill=LABEL_COLOR, font=font)

    return np.array(pil_img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Path setup for tac imports
    project_root = Path(__file__).resolve().parent.parent
    src_dir = str(project_root / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    upstream_dir = Path(args.upstream)
    upstream_str = str(upstream_dir)
    if upstream_str not in sys.path:
        sys.path.insert(0, upstream_str)

    from PIL import Image

    from tac.data import decode_archive, decode_video
    from tac.scorer import detect_device, load_scorers

    need_postfilter = args.mode in ("ours", "comparison")

    device = args.device or str(detect_device())
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    model = None
    if need_postfilter:
        from tac.architectures import build_postfilter
        from tac.quantization import load_int8

        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Loading post-filter: {args.variant} h={args.hidden} k={args.kernel}")
        model = build_postfilter(args.variant, hidden=args.hidden, kernel=args.kernel)
        load_int8(str(checkpoint_path), model, device=device)
        model = model.eval().to(device)

    print("Loading scorers...")
    posenet, segnet = load_scorers(
        upstream_dir / "models" / "posenet.safetensors",
        upstream_dir / "models" / "segnet.safetensors",
        device=device,
        upstream_dir=upstream_str,
    )

    # ------------------------------------------------------------------
    # Decode frames
    # ------------------------------------------------------------------
    archive_path = Path(args.archive)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    gt_video_path = upstream_dir / "videos" / "0.mkv"
    if not gt_video_path.exists():
        raise FileNotFoundError(f"GT video not found: {gt_video_path}")

    print(f"Decoding archive: {archive_path}")
    comp_frames = decode_archive(str(archive_path))
    print(f"Decoding GT video: {gt_video_path}")
    gt_frames = decode_video(str(gt_video_path))

    if len(comp_frames) != len(gt_frames):
        raise ValueError(f"Frame count mismatch: {len(comp_frames)} vs {len(gt_frames)}")
    total_frames = len(comp_frames)
    print(f"Total video frames: {total_frames}")

    # Stride: sample every Nth frame to get exactly args.num_frames
    stride = max(1, total_frames // args.num_frames)
    sample_indices = list(range(0, total_frames, stride))[: args.num_frames]
    print(f"Sampling {len(sample_indices)} frames (stride={stride})")

    # ------------------------------------------------------------------
    # Compute file sizes for labels
    # ------------------------------------------------------------------
    archive_mb = archive_path.stat().st_size / (1024 * 1024)
    gt_mb = gt_video_path.stat().st_size / (1024 * 1024)

    # Labels per mode -- short, comma-style
    if args.mode == "baseline":
        top_left_label = f"original  {gt_mb:.1f}MB"
        top_right_label = f"compressed  {archive_mb:.1f}MB"
    elif args.mode == "ours":
        top_left_label = f"original  {gt_mb:.1f}MB"
        top_right_label = f"ours  {archive_mb:.1f}MB"
    else:  # comparison
        top_left_label = f"compressed  {archive_mb:.1f}MB"
        top_right_label = f"ours  {archive_mb:.1f}MB"

    # ------------------------------------------------------------------
    # Generate frames
    # ------------------------------------------------------------------
    gif_images: list[Image.Image] = []
    posenet_primary_mses: list[float] = []
    posenet_secondary_mses: list[float] = []
    posenet_xs: list[int] = []

    # Cache to avoid double post-filter inference on the same frame
    _gif_cached_idx: int = -1
    _gif_cached_filtered: torch.Tensor | None = None
    _gif_cached_baseline: torch.Tensor | None = None
    _gif_cached_gt: torch.Tensor | None = None

    print("Generating GIF frames...")
    with torch.no_grad():
        for fi, idx in enumerate(sample_indices):
            comp = comp_frames[idx].to(device)   # (H, W, 3) uint8
            gt = gt_frames[idx].to(device)

            # CHW float for models
            if idx == _gif_cached_idx and _gif_cached_baseline is not None:
                baseline_chw = _gif_cached_baseline
                gt_chw = _gif_cached_gt
                filtered_chw = _gif_cached_filtered
            else:
                baseline_chw = comp.float().permute(2, 0, 1).unsqueeze(0).contiguous()
                gt_chw = gt.float().permute(2, 0, 1).unsqueeze(0).contiguous()
                if need_postfilter:
                    filtered_chw = model(baseline_chw).round().clamp(0, 255)
                else:
                    filtered_chw = None

            # ----- Determine panels per mode -----
            gt_np = gt.cpu().numpy()
            baseline_np = comp.cpu().numpy()

            if args.mode == "baseline":
                top_left_img = gt_np
                top_right_img = baseline_np
                # SegNet: compressed vs GT
                seg_test_chw = baseline_chw
            elif args.mode == "ours":
                top_left_img = gt_np
                top_right_img = filtered_chw[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
                seg_test_chw = filtered_chw
            else:  # comparison
                top_left_img = baseline_np
                top_right_img = filtered_chw[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
                seg_test_chw = filtered_chw

            # SegNet error mask
            gt_cls = get_segnet_classes(gt_chw, segnet)
            test_cls = get_segnet_classes(seg_test_chw, segnet)
            ours_seg_diff = ((test_cls != gt_cls).astype(np.uint8) * 255)  # (Hs, Ws)

            # Color-coded diff in ours/comparison mode (red=fixed, green=regression, white=both)
            if args.mode in ("ours", "comparison") and need_postfilter:
                baseline_cls = get_segnet_classes(baseline_chw, segnet)
                baseline_seg_diff = ((baseline_cls != gt_cls).astype(np.uint8) * 255)
                seg_panel = _upscale_segnet_diff(ours_seg_diff, baseline_seg_diff)
            else:
                seg_panel = _upscale_segnet_mask(ours_seg_diff)

            # PoseNet: compute on consecutive frame pairs
            if idx + stride < total_frames:
                next_idx = min(idx + 1, total_frames - 1)
                next_comp = comp_frames[next_idx].to(device)
                next_gt = gt_frames[next_idx].to(device)
                next_baseline_chw = next_comp.float().permute(2, 0, 1).unsqueeze(0).contiguous()
                next_gt_chw = next_gt.float().permute(2, 0, 1).unsqueeze(0).contiguous()

                if args.mode == "baseline":
                    mse = get_posenet_mse(baseline_chw, next_baseline_chw, gt_chw, next_gt_chw, posenet)
                    posenet_primary_mses.append(mse)
                elif args.mode == "ours":
                    next_filtered_chw = model(next_baseline_chw).round().clamp(0, 255)
                    ours_mse = get_posenet_mse(filtered_chw, next_filtered_chw, gt_chw, next_gt_chw, posenet)
                    bl_mse = get_posenet_mse(baseline_chw, next_baseline_chw, gt_chw, next_gt_chw, posenet)
                    posenet_primary_mses.append(ours_mse)
                    posenet_secondary_mses.append(bl_mse)
                    # Cache the next frame's filtered output for reuse
                    _gif_cached_idx = next_idx
                    _gif_cached_baseline = next_baseline_chw
                    _gif_cached_gt = next_gt_chw
                    _gif_cached_filtered = next_filtered_chw
                else:  # comparison
                    next_filtered_chw = model(next_baseline_chw).round().clamp(0, 255)
                    ours_mse = get_posenet_mse(filtered_chw, next_filtered_chw, gt_chw, next_gt_chw, posenet)
                    bl_mse = get_posenet_mse(baseline_chw, next_baseline_chw, gt_chw, next_gt_chw, posenet)
                    posenet_primary_mses.append(ours_mse)
                    posenet_secondary_mses.append(bl_mse)
                    # Cache the next frame's filtered output for reuse
                    _gif_cached_idx = next_idx
                    _gif_cached_baseline = next_baseline_chw
                    _gif_cached_gt = next_gt_chw
                    _gif_cached_filtered = next_filtered_chw

                posenet_xs.append(idx)

            # Render PoseNet chart panel (matplotlib, exact panel size)
            chart_rgb = _render_posenet_chart(
                xs=posenet_xs,
                primary=posenet_primary_mses,
                secondary=posenet_secondary_mses if args.mode in ("comparison", "ours") else None,
                mode=args.mode,
                total_frames=args.total_video_frames,
            )

            # Composite all 4 panels + text labels into 512x384
            frame_rgb = _composite_frame(
                top_left_img=top_left_img,
                top_right_img=top_right_img,
                segnet_panel=seg_panel,
                chart_rgb=chart_rgb,
                top_left_label=top_left_label,
                top_right_label=top_right_label,
            )

            gif_images.append(Image.fromarray(frame_rgb))

            if (fi + 1) % 25 == 0 or fi == 0:
                print(f"  [{fi + 1}/{len(sample_indices)}] video frame {idx}")

    # ------------------------------------------------------------------
    # Save GIF
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gif_images[0].save(
        str(out_path),
        save_all=True,
        append_images=gif_images[1:],
        duration=args.frame_duration,
        loop=0,
        optimize=True,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote GIF: {out_path} ({size_mb:.1f} MB)")
    print(f"  {len(gif_images)} frames, {args.frame_duration}ms/frame, 512x384 px")

    # Summary
    if posenet_primary_mses:
        mean_primary = sum(posenet_primary_mses) / len(posenet_primary_mses)
        print(f"  Primary PoseNet MSE mean: {mean_primary:.6f}")
        if posenet_secondary_mses:
            mean_secondary = sum(posenet_secondary_mses) / len(posenet_secondary_mses)
            reduction = (1 - mean_primary / mean_secondary) * 100 if mean_secondary > 0 else 0.0
            print(f"  Baseline PoseNet MSE mean: {mean_secondary:.6f}")
            print(f"  Reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()
