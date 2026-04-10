#!/usr/bin/env python3
"""Generate a 512x384 animated GIF replicating the comma.ai challenge README format.

Produces a pixel-perfect 2x2 matplotlib grid matching the original comma
challenge GIF specs:

  512x384 px, 150 frames, 100ms/frame (10fps), dark_background style.

  Top-left:     video frame (original or baseline)
  Top-right:    video frame (compressed or ours)
  Bottom-left:  segnet errors (binary white-on-black)
  Bottom-right: posenet errors (progressive line chart)

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
# Rendering helpers
# ---------------------------------------------------------------------------

# Exact figure dimensions: 5.12 x 3.84 inches at 100 dpi = 512 x 384 px.
FIG_W_IN = 5.12
FIG_H_IN = 3.84
DPI = 100


def _render_frame_matplotlib(
    *,
    top_left_img: np.ndarray,
    top_right_img: np.ndarray,
    segnet_mask: np.ndarray,
    posenet_xs: list[int],
    posenet_primary: list[float],
    posenet_secondary: list[float] | None,
    top_left_label: str,
    top_right_label: str,
    mode: str,
    total_video_frames: int,
) -> np.ndarray:
    """Render one 512x384 frame using matplotlib dark_background style.

    Returns (384, 512, 3) uint8 numpy array.
    """
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 2, figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)

        # --- Top-left: first video panel ---
        ax_tl = axes[0, 0]
        ax_tl.imshow(top_left_img)
        ax_tl.set_title(top_left_label, fontsize=7, fontfamily="monospace",
                        color="#cccccc", pad=2)
        ax_tl.axis("off")

        # --- Top-right: second video panel ---
        ax_tr = axes[0, 1]
        ax_tr.imshow(top_right_img)
        ax_tr.set_title(top_right_label, fontsize=7, fontfamily="monospace",
                        color="#cccccc", pad=2)
        ax_tr.axis("off")

        # --- Bottom-left: segnet errors ---
        ax_bl = axes[1, 0]
        ax_bl.imshow(segnet_mask, cmap="gray", vmin=0, vmax=255)
        ax_bl.set_title("segnet errors", fontsize=7, fontfamily="monospace",
                        color="#cccccc", pad=2)
        ax_bl.axis("off")

        # --- Bottom-right: posenet errors (progressive line chart) ---
        ax_br = axes[1, 1]
        ax_br.set_title("posenet errors", fontsize=7, fontfamily="monospace",
                        color="#cccccc", pad=2)

        if mode == "comparison" and posenet_secondary is not None:
            # Dual trace: baseline gray dashed, ours green solid
            ax_br.plot(posenet_xs, posenet_secondary, color="gray",
                       linestyle="--", linewidth=0.8, label="baseline")
            ax_br.plot(posenet_xs, posenet_primary, color="#00ff88",
                       linestyle="-", linewidth=1.0, label="ours")
            ax_br.legend(loc="upper right", fontsize=5, facecolor="black",
                         edgecolor="#333333", labelcolor="white")
        else:
            line_color = "#00ff88" if mode == "ours" else "#ffffff"
            ax_br.plot(posenet_xs, posenet_primary, color=line_color,
                       linewidth=1.0)

        ax_br.set_xlim(0, total_video_frames)
        ax_br.set_ylim(0, 2.0)
        ax_br.tick_params(colors="white", labelsize=5)
        ax_br.set_ylabel("MSE", color="white", fontsize=6)
        ax_br.set_xlabel("frame", color="white", fontsize=6)
        for spine in ax_br.spines.values():
            spine.set_color("#333333")

        fig.tight_layout(pad=0.4)

        # Render to numpy array
        buf = io.BytesIO()
        fig.savefig(buf, format="raw", dpi=DPI, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)

        # raw format: RGBA, width x height from get_size_inches * dpi
        w = int(fig.get_size_inches()[0] * DPI)
        h = int(fig.get_size_inches()[1] * DPI)
        raw = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(h, w, 4)
        return raw[:, :, :3]  # drop alpha -> (384, 512, 3)


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

    # Labels per mode
    if args.mode == "baseline":
        top_left_label = f"speed 4x \u2014 original \u2014 {gt_mb:.1f}MB"
        top_right_label = f"speed 4x \u2014 compressed \u2014 {archive_mb:.1f}MB"
    elif args.mode == "ours":
        top_left_label = f"speed 4x \u2014 original \u2014 {gt_mb:.1f}MB"
        top_right_label = f"speed 4x \u2014 ours (post-filter) \u2014 {archive_mb:.1f}MB"
    else:  # comparison
        top_left_label = f"speed 4x \u2014 compressed \u2014 {archive_mb:.1f}MB"
        top_right_label = f"speed 4x \u2014 ours (post-filter) \u2014 {archive_mb:.1f}MB"

    # ------------------------------------------------------------------
    # Generate frames
    # ------------------------------------------------------------------
    gif_images: list[Image.Image] = []
    posenet_primary_mses: list[float] = []
    posenet_secondary_mses: list[float] = []
    posenet_xs: list[int] = []

    print("Generating GIF frames...")
    with torch.no_grad():
        for fi, idx in enumerate(sample_indices):
            comp = comp_frames[idx].to(device)   # (H, W, 3) uint8
            gt = gt_frames[idx].to(device)

            # CHW float for models
            baseline_chw = comp.float().permute(2, 0, 1).unsqueeze(0).contiguous()
            gt_chw = gt.float().permute(2, 0, 1).unsqueeze(0).contiguous()

            # Post-filter if needed
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
            seg_diff = ((test_cls != gt_cls).astype(np.uint8) * 255)  # (Hs, Ws)

            # PoseNet: need consecutive pair
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
                    mse = get_posenet_mse(filtered_chw, next_filtered_chw, gt_chw, next_gt_chw, posenet)
                    posenet_primary_mses.append(mse)
                else:  # comparison
                    next_filtered_chw = model(next_baseline_chw).round().clamp(0, 255)
                    ours_mse = get_posenet_mse(filtered_chw, next_filtered_chw, gt_chw, next_gt_chw, posenet)
                    bl_mse = get_posenet_mse(baseline_chw, next_baseline_chw, gt_chw, next_gt_chw, posenet)
                    posenet_primary_mses.append(ours_mse)
                    posenet_secondary_mses.append(bl_mse)

                posenet_xs.append(idx)

            # Render matplotlib frame
            frame_rgb = _render_frame_matplotlib(
                top_left_img=top_left_img,
                top_right_img=top_right_img,
                segnet_mask=seg_diff,
                posenet_xs=posenet_xs,
                posenet_primary=posenet_primary_mses,
                posenet_secondary=posenet_secondary_mses if args.mode == "comparison" else None,
                top_left_label=top_left_label,
                top_right_label=top_right_label,
                mode=args.mode,
                total_video_frames=args.total_video_frames,
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
