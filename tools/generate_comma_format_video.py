#!/usr/bin/env python3
"""Generate a 6-panel comma-challenge-style comparison video.

Replicates the exact visual format from the comma.ai video compression
challenge README GIF, extended to three columns so the post-filter result
appears alongside the baseline.

Layout (3 columns x 2 rows):

  Top row:
    Original (GT)  |  Baseline (compressed, no filter)  |  Ours (compressed + post-filter)

  Bottom row:
    PoseNet chart  |  SegNet errors (baseline vs GT)    |  SegNet errors (ours vs GT)

Panel details:
  - Top panels show the actual RGB video frames.
  - SegNet error panels are binary white-on-black masks at SegNet resolution
    (384x512) upscaled to panel size: white = argmax(compressed) != argmax(GT).
  - PoseNet chart is a running line chart with two overlaid traces:
      gray dashed = baseline per-pair MSE
      green solid = ours per-pair MSE
  - Labels at top of each panel in white monospace font on dark background.

Output: MP4 (4x speed) and animated GIF.

Usage:
    python tools/generate_comma_format_video.py \\
        --upstream workspace/upstream/comma_video_compression_challenge \\
        --archive submissions/robust_current/archive.zip \\
        --checkpoint submissions/robust_current/postfilter_int8.pt \\
        --output-dir reports/graphs/site/comma_format \\
        --variant standard --hidden 64

Requires: torch, av (PyAV), PIL/Pillow, numpy, matplotlib, tac library.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate 6-panel comma-format comparison video.",
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
        default="reports/graphs/site/comma_format",
        help="Output directory for video and GIF",
    )
    p.add_argument("--variant", type=str, default="standard", help="Post-filter architecture variant")
    p.add_argument("--hidden", type=int, default=64, help="Post-filter hidden channel count")
    p.add_argument("--kernel", type=int, default=3, help="Post-filter kernel size")
    p.add_argument("--device", type=str, default=None, help="Compute device (auto-detected if omitted)")
    p.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    p.add_argument("--fps-playback", type=int, default=120, help="Playback FPS (4x speed = 120)")
    p.add_argument("--gif-stride", type=int, default=4, help="Only include every Nth frame in GIF")
    p.add_argument("--gif-fps", type=int, default=30, help="GIF playback FPS")
    p.add_argument("--gif-scale", type=float, default=0.5, help="GIF downscale factor (0.5 = half res)")
    p.add_argument("--label-height", type=int, default=40, help="Height of label strip in pixels")
    p.add_argument(
        "--chart-window", type=int, default=120,
        help="PoseNet chart rolling window (number of recent pairs to show)",
    )
    p.add_argument(
        "--baseline-label", type=str, default=None,
        help="Override baseline column label (e.g. 'compressed -- 1.4MB')",
    )
    p.add_argument(
        "--ours-label", type=str, default=None,
        help="Override ours column label (e.g. 'ours + post-filter -- 1.4MB')",
    )
    p.add_argument(
        "--original-label", type=str, default=None,
        help="Override original column label (e.g. 'original -- 37.5MB')",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _load_mono_font(size: int = 16):
    """Try to load a monospace font; fall back to PIL default."""
    from PIL import ImageFont

    for candidate in [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]:
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_label_bar(
    width: int,
    height: int,
    labels: list[str],
    bg: tuple[int, int, int] = (10, 10, 10),
    fg: tuple[int, int, int] = (255, 255, 255),
    font_size: int = 16,
) -> np.ndarray:
    """Render a horizontal bar with evenly-spaced centered text labels."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = _load_mono_font(font_size)

    section_w = width // max(len(labels), 1)
    for i, text in enumerate(labels):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = i * section_w + (section_w - tw) // 2
        y = (height - th) // 2
        draw.text((x, y), text, fill=fg, font=font)

    return np.array(img)


def make_segnet_error_mask(
    seg_classes: np.ndarray,
    gt_classes: np.ndarray,
    display_h: int,
    display_w: int,
) -> np.ndarray:
    """Binary white-on-black error mask: white where argmax differs from GT.

    Args:
        seg_classes: (Hs, Ws) int array of SegNet argmax classes.
        gt_classes:  (Hs, Ws) int array of GT SegNet argmax classes.
        display_h, display_w: target display size.

    Returns:
        (display_h, display_w, 3) uint8 RGB array.
    """
    from PIL import Image

    diff = (seg_classes != gt_classes).astype(np.uint8) * 255  # (Hs, Ws)
    # Upscale to display resolution with nearest neighbor (keeps binary look)
    diff_img = Image.fromarray(diff, mode="L").resize(
        (display_w, display_h), Image.NEAREST
    )
    diff_np = np.array(diff_img)  # (display_h, display_w)
    return np.stack([diff_np, diff_np, diff_np], axis=-1)  # (H, W, 3)


def render_posenet_chart(
    baseline_mses: list[float],
    ours_mses: list[float],
    chart_w: int,
    chart_h: int,
    window: int = 120,
) -> np.ndarray:
    """Render a running PoseNet MSE line chart as an RGB numpy array.

    Uses matplotlib with Agg backend so no display is needed.

    Args:
        baseline_mses: all baseline per-pair MSE values so far.
        ours_mses: all ours per-pair MSE values so far.
        chart_w, chart_h: output pixel dimensions.
        window: number of recent pairs to display.

    Returns:
        (chart_h, chart_w, 3) uint8 RGB array.
    """
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    fig_w = chart_w / dpi
    fig_h = chart_h / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Slice to window
    n = len(baseline_mses)
    start = max(0, n - window)
    xs = list(range(start, n))
    bl_slice = baseline_mses[start:]
    ours_slice = ours_mses[start:]

    ax.plot(xs, bl_slice, color="gray", linestyle="--", linewidth=1.0, label="baseline")
    ax.plot(xs, ours_slice, color="#00ff88", linestyle="-", linewidth=1.5, label="ours")

    # Style
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.set_ylabel("PoseNet MSE", color="white", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, facecolor="black", edgecolor="#333333", labelcolor="white")

    # Y limits: auto with some padding
    all_vals = bl_slice + ours_slice
    if all_vals:
        ymax = max(all_vals) * 1.3
        ax.set_ylim(0, max(ymax, 1e-6))

    fig.tight_layout(pad=0.3)

    # Render to numpy
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    from PIL import Image

    chart_img = Image.open(buf).convert("RGB").resize((chart_w, chart_h), Image.LANCZOS)
    return np.array(chart_img)


def get_segnet_classes(frame_chw: torch.Tensor, segnet) -> np.ndarray:
    """Run SegNet on a single frame, return (Hs, Ws) int argmax classes.

    Args:
        frame_chw: (1, 3, H, W) float tensor.
        segnet: loaded SegNet model.

    Returns:
        (Hs, Ws) numpy int array of per-pixel class indices.
    """
    # SegNet.preprocess_input expects (B, T, C, H, W); uses last frame only.
    pair = frame_chw.unsqueeze(1).expand(-1, 2, -1, -1, -1)
    seg_input = segnet.preprocess_input(pair)
    seg_output = segnet(seg_input)
    # seg_output: (B, num_classes, Hs, Ws)
    classes = seg_output[0].argmax(dim=0).cpu().numpy()  # (Hs, Ws)
    return classes


def get_posenet_mse(
    frame_chw_a: torch.Tensor,
    frame_chw_b: torch.Tensor,
    gt_chw_a: torch.Tensor,
    gt_chw_b: torch.Tensor,
    posenet,
) -> float:
    """Compute PoseNet distortion (MSE) between a frame pair and GT pair.

    PoseNet expects pairs of consecutive frames. We build (1, 2, C, H, W)
    inputs and use compute_distortion.

    Args:
        frame_chw_a, frame_chw_b: (1, 3, H, W) float tensors (consecutive frames).
        gt_chw_a, gt_chw_b: (1, 3, H, W) float tensors (consecutive GT frames).
        posenet: loaded PoseNet model.

    Returns:
        Scalar MSE float.
    """
    # Build (1, 2, C, H, W)
    comp_pair = torch.cat([frame_chw_a, frame_chw_b], dim=0).unsqueeze(0)  # (1, 2, C, H, W)
    gt_pair = torch.cat([gt_chw_a, gt_chw_b], dim=0).unsqueeze(0)

    comp_input = posenet.preprocess_input(comp_pair)
    gt_input = posenet.preprocess_input(gt_pair)

    comp_out = posenet(comp_input)
    gt_out = posenet(gt_input)

    dist = posenet.compute_distortion(comp_out, gt_out)  # (1,) tensor
    return dist.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Add src/ to path for tac imports
    project_root = Path(__file__).resolve().parent.parent
    src_dir = str(project_root / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

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

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    checkpoint_path = Path(args.checkpoint)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
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
    print(f"Processing {num_frames} of {len(comp_frames)} frames, resolution {W}x{H}")

    # ------------------------------------------------------------------
    # Compute archive size for labels
    # ------------------------------------------------------------------
    archive_mb = archive_path.stat().st_size / (1024 * 1024)
    gt_mb = gt_video_path.stat().st_size / (1024 * 1024)

    original_label = args.original_label or f"speed 4x -- original -- {gt_mb:.1f}MB"
    baseline_label = args.baseline_label or f"speed 4x -- compressed -- {archive_mb:.1f}MB"
    ours_label = args.ours_label or f"speed 4x -- ours (post-filter) -- {archive_mb:.1f}MB"

    # ------------------------------------------------------------------
    # Layout: 3 columns x 2 rows
    # ------------------------------------------------------------------
    panel_w = W
    panel_h = H
    n_cols = 3
    n_rows = 2
    label_h = args.label_height

    # Top labels row + top video row + bottom labels row + bottom panels row
    total_w = panel_w * n_cols
    # Row 1: label bar + video panels.  Row 2: label bar + error/chart panels.
    total_h = label_h + panel_h + label_h + panel_h

    # ------------------------------------------------------------------
    # Output setup
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = out_dir / "comma_format.mp4"
    out_container = av.open(str(mp4_path), mode="w")
    stream = out_container.add_stream("libx264", rate=args.fps_playback)
    stream.width = total_w
    stream.height = total_h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}

    gif_frames: list[Image.Image] = []

    # PoseNet MSE accumulators (per pair, not per frame)
    baseline_pose_mses: list[float] = []
    ours_pose_mses: list[float] = []

    # Pre-render the two label bars (they are static)
    top_label_bar = draw_label_bar(
        total_w, label_h,
        [original_label, baseline_label, ours_label],
        font_size=15,
    )
    bottom_labels = draw_label_bar(
        total_w, label_h,
        ["posenet errors", "segnet errors (baseline)", "segnet errors (ours)"],
        font_size=14,
        fg=(180, 180, 180),
    )

    print("Generating comma-format frames...")
    with torch.no_grad():
        for idx in range(num_frames):
            comp = comp_frames[idx].to(device)  # (H, W, 3) uint8
            gt = gt_frames[idx].to(device)

            # CHW float tensors for model input
            baseline_chw = comp.float().permute(2, 0, 1).unsqueeze(0).contiguous()
            gt_chw = gt.float().permute(2, 0, 1).unsqueeze(0).contiguous()

            # Post-filter output
            filtered_chw = model(baseline_chw).round().clamp(0, 255)

            # SegNet class maps (all at SegNet resolution)
            gt_cls = get_segnet_classes(gt_chw, segnet)
            baseline_cls = get_segnet_classes(baseline_chw, segnet)
            filtered_cls = get_segnet_classes(filtered_chw, segnet)

            # PoseNet: compute on consecutive pairs (idx, idx+1)
            if idx < num_frames - 1:
                next_comp = comp_frames[idx + 1].to(device)
                next_gt = gt_frames[idx + 1].to(device)
                next_baseline_chw = next_comp.float().permute(2, 0, 1).unsqueeze(0).contiguous()
                next_gt_chw = next_gt.float().permute(2, 0, 1).unsqueeze(0).contiguous()
                next_filtered_chw = model(next_baseline_chw).round().clamp(0, 255)

                bl_mse = get_posenet_mse(
                    baseline_chw, next_baseline_chw, gt_chw, next_gt_chw, posenet,
                )
                ours_mse = get_posenet_mse(
                    filtered_chw, next_filtered_chw, gt_chw, next_gt_chw, posenet,
                )
                baseline_pose_mses.append(bl_mse)
                ours_pose_mses.append(ours_mse)

            # ----------------------------------------------------------
            # Numpy frames for compositing
            # ----------------------------------------------------------
            gt_np = gt.cpu().numpy()  # (H, W, 3) uint8
            baseline_np = comp.cpu().numpy()
            filtered_np = (
                filtered_chw[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            )

            # Top row: Original | Baseline | Ours
            top_row = np.concatenate([gt_np, baseline_np, filtered_np], axis=1)

            # Bottom row panels
            # Panel 1: PoseNet chart
            if baseline_pose_mses:
                chart_panel = render_posenet_chart(
                    baseline_pose_mses,
                    ours_pose_mses,
                    chart_w=panel_w,
                    chart_h=panel_h,
                    window=args.chart_window,
                )
            else:
                # First frame: blank black panel
                chart_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

            # Panel 2: SegNet errors (baseline vs GT)
            baseline_err = make_segnet_error_mask(baseline_cls, gt_cls, panel_h, panel_w)

            # Panel 3: SegNet errors (ours vs GT)
            ours_err = make_segnet_error_mask(filtered_cls, gt_cls, panel_h, panel_w)

            bottom_row = np.concatenate([chart_panel, baseline_err, ours_err], axis=1)

            # Full composite: top_label + top_row + bottom_label + bottom_row
            composite = np.concatenate(
                [top_label_bar, top_row, bottom_labels, bottom_row], axis=0
            )

            # Write to MP4
            av_frame = av.VideoFrame.from_ndarray(composite, format="rgb24")
            for packet in stream.encode(av_frame):
                out_container.mux(packet)

            # Collect for GIF
            if idx % args.gif_stride == 0:
                gif_frames.append(Image.fromarray(composite))

            if (idx + 1) % 100 == 0 or idx == 0:
                bl_seg_err = int((baseline_cls != gt_cls).sum())
                ours_seg_err = int((filtered_cls != gt_cls).sum())
                bl_pose = baseline_pose_mses[-1] if baseline_pose_mses else 0.0
                our_pose = ours_pose_mses[-1] if ours_pose_mses else 0.0
                print(
                    f"  [{(idx + 1) / num_frames * 100:5.1f}%] frame {idx}: "
                    f"seg_err baseline={bl_seg_err:,} ours={ours_seg_err:,} | "
                    f"pose_mse baseline={bl_pose:.6f} ours={our_pose:.6f}"
                )

    # Flush MP4
    for packet in stream.encode():
        out_container.mux(packet)
    out_container.close()
    print(f"Wrote MP4: {mp4_path} ({mp4_path.stat().st_size / (1024 * 1024):.1f} MB)")

    # ------------------------------------------------------------------
    # Write GIF
    # ------------------------------------------------------------------
    if gif_frames:
        gif_path = out_dir / "comma_format.gif"
        s = args.gif_scale
        gif_w = int(total_w * s)
        gif_h = int(total_h * s)
        resized = [f.resize((gif_w, gif_h), Image.LANCZOS) for f in gif_frames]
        duration_ms = int(1000 / args.gif_fps)
        resized[0].save(
            str(gif_path),
            save_all=True,
            append_images=resized[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        print(f"Wrote GIF: {gif_path} ({gif_path.stat().st_size / (1024 * 1024):.1f} MB)")

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    if baseline_pose_mses:
        bl_mean = sum(baseline_pose_mses) / len(baseline_pose_mses)
        ours_mean = sum(ours_pose_mses) / len(ours_pose_mses)
        reduction = (1 - ours_mean / bl_mean) * 100 if bl_mean > 0 else 0.0
        print(f"\nPoseNet MSE -- baseline mean: {bl_mean:.6f}, ours mean: {ours_mean:.6f}")
        print(f"PoseNet MSE reduction: {reduction:.1f}%")

    print(f"\nOutputs in: {out_dir}")


if __name__ == "__main__":
    main()
