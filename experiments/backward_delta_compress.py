#!/usr/bin/env python3
"""Backward Delta Compression: anchor frame + temporal deltas.

Radically different paradigm from mask-based rendering:
    1. Generate ONE perfect anchor frame (the last frame, 1199)
    2. Compute backward optical flow between consecutive frames
    3. Compute deltas: delta_i = frame_i - warp(frame_{i+1}, flow_{i->i+1})
    4. The deltas are sparse and small (most pixels don't change between frames)
    5. Encode deltas compactly

At inflate time:
    1. Decode the perfect anchor frame
    2. Apply backward flow warps + deltas sequentially to reconstruct all frames

This analysis script measures:
    - How compressible are the backward deltas? (entropy, sparsity, L1)
    - What is the flow field quality? (how much residual after warping?)
    - Rate-distortion tradeoff: how many bits for the deltas at various quality levels?
    - Comparison with forward (temporal) vs backward (anchor) approaches

Usage:
    # Smoke test (local MPS):
    PYTHONPATH=src:upstream python experiments/backward_delta_compress.py \
        --device mps --smoke

    # Full analysis:
    PYTHONPATH=src:upstream python experiments/backward_delta_compress.py \
        --device mps --n-frames 1200

    # With specific flow method:
    PYTHONPATH=src:upstream python experiments/backward_delta_compress.py \
        --device cuda --n-frames 1200 --flow-method raft
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tac.utils import find_project_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backward delta compression analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", type=str, default="mps", choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=1200, help="Number of frames to analyze")
    p.add_argument("--flow-method", type=str, default="farneback",
                   choices=["farneback", "raft", "simple"],
                   help="Optical flow method (farneback=OpenCV, raft=learned, simple=pixel diff)")
    p.add_argument("--upstream", type=str, default=None, help="Path to upstream repo")
    p.add_argument("--video", type=str, default=None, help="Path to GT video")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--smoke", action="store_true", help="Smoke test: 40 frames")
    return p.parse_args()


def compute_farneback_flow(
    frame_src: torch.Tensor,
    frame_dst: torch.Tensor,
) -> torch.Tensor:
    """Compute dense optical flow from src to dst using Farneback (OpenCV).

    Args:
        frame_src: (H, W, 3) uint8 tensor (source frame)
        frame_dst: (H, W, 3) uint8 tensor (destination frame)

    Returns:
        (H, W, 2) float tensor of flow vectors (dx, dy)
    """
    try:
        import cv2
    except ImportError:
        import warnings
        warnings.warn(
            "cv2 (OpenCV) not installed — returning zero flow. "
            "Install with: uv pip install opencv-python-headless",
            stacklevel=2,
        )
        return torch.zeros(frame_src.shape[0], frame_src.shape[1], 2)

    # Convert to grayscale
    src_np = frame_src.numpy().astype(np.uint8)
    dst_np = frame_dst.numpy().astype(np.uint8)
    src_gray = cv2.cvtColor(src_np, cv2.COLOR_RGB2GRAY)
    dst_gray = cv2.cvtColor(dst_np, cv2.COLOR_RGB2GRAY)

    # Compute flow
    flow = cv2.calcOpticalFlowFarneback(
        src_gray, dst_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    return torch.from_numpy(flow).float()


def warp_frame(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp a frame using optical flow (backward warp via grid_sample).

    Args:
        frame: (H, W, 3) float tensor
        flow: (H, W, 2) float tensor (pixel displacement)

    Returns:
        (H, W, 3) warped frame
    """
    H, W, _ = frame.shape

    # Create base grid
    gy = torch.arange(H, dtype=torch.float32)
    gx = torch.arange(W, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

    # Apply flow displacement
    new_x = grid_x + flow[:, :, 0]
    new_y = grid_y + flow[:, :, 1]

    # Normalize to [-1, 1] for grid_sample
    new_x = 2.0 * new_x / (W - 1) - 1.0
    new_y = 2.0 * new_y / (H - 1) - 1.0

    grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    frame_chw = frame.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    warped = F.grid_sample(
        frame_chw, grid, mode="bilinear", padding_mode="border", align_corners=True
    )

    return warped[0].permute(1, 2, 0)  # (H, W, 3)


def estimate_entropy(tensor: torch.Tensor, n_bins: int = 256) -> float:
    """Estimate the entropy of a tensor (bits per element).

    Uses histogram-based probability estimation.

    Args:
        tensor: flat tensor of values
        n_bins: number of histogram bins

    Returns:
        Estimated entropy in bits per element.
    """
    # Normalize to [0, 1]
    t = tensor.float().flatten()
    t_min, t_max = t.min(), t.max()
    if t_max - t_min < 1e-8:
        return 0.0
    t_norm = (t - t_min) / (t_max - t_min)

    # Histogram
    hist = torch.histc(t_norm, bins=n_bins, min=0.0, max=1.0)
    probs = hist / hist.sum()
    probs = probs[probs > 0]

    # Shannon entropy
    entropy = -(probs * torch.log2(probs)).sum().item()
    return entropy


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.n_frames = 40

    device = torch.device(args.device)

    # Resolve paths
    root = find_project_root()
    upstream = Path(args.upstream) if args.upstream else root / "upstream"
    video_path = Path(args.video) if args.video else upstream / "videos" / "0.mkv"
    output_dir = Path(args.output_dir) if args.output_dir else (
        root / "experiments" / "results" / "backward_delta"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Decode video
    from tac.data import decode_video
    print(f"[backward-delta] Decoding video: {video_path}")
    gt_frames = decode_video(str(video_path))[:args.n_frames]
    n_frames = len(gt_frames)
    H, W = gt_frames[0].shape[:2]
    print(f"[backward-delta] {n_frames} frames, {H}x{W}")

    # Compute backward deltas: delta_i = frame_i - warp(frame_{i+1}, flow_{i->i+1})
    print(f"\n[backward-delta] Computing backward flow + deltas...")
    t0 = time.time()

    backward_deltas = []  # (N-1,) list of (H, W, 3) tensors
    flow_magnitudes = []
    warp_errors = []  # L1 error after warping (without delta)

    for i in range(n_frames - 1):
        # Flow from frame i to frame i+1
        flow = compute_farneback_flow(gt_frames[i], gt_frames[i + 1])

        # Warp frame i+1 back to frame i's position
        warped = warp_frame(gt_frames[i + 1].float(), flow)

        # Delta = actual - warped
        delta = gt_frames[i].float() - warped
        backward_deltas.append(delta)

        # Statistics
        flow_mag = flow.pow(2).sum(dim=-1).sqrt().mean().item()
        flow_magnitudes.append(flow_mag)
        warp_error = delta.abs().mean().item()
        warp_errors.append(warp_error)

        if i % 50 == 0:
            print(f"  Frame {i}/{n_frames-1} | flow_mag={flow_mag:.2f}px | warp_error={warp_error:.2f}")

    flow_time = time.time() - t0
    print(f"[backward-delta] Flow computation: {flow_time:.1f}s")

    # Analysis
    print(f"\n[backward-delta] === DELTA STATISTICS ===")

    delta_tensor = torch.stack(backward_deltas, dim=0)  # (N-1, H, W, 3)

    # Basic statistics
    delta_abs = delta_tensor.abs()
    mean_l1 = delta_abs.mean().item()
    mean_linf = delta_abs.amax(dim=(1, 2, 3)).mean().item()

    # Sparsity: what fraction of pixels have |delta| < threshold?
    thresholds = [1.0, 2.0, 5.0, 10.0, 20.0]
    sparsity = {}
    for thresh in thresholds:
        sparse_frac = (delta_abs < thresh).float().mean().item()
        sparsity[f"below_{thresh}"] = sparse_frac

    # Entropy of deltas (bits per channel per pixel)
    delta_entropy = estimate_entropy(delta_tensor.clamp(-128, 127))

    # Entropy of raw frames (for comparison)
    frame_tensor = torch.stack([f.float() for f in gt_frames], dim=0)
    frame_entropy = estimate_entropy(frame_tensor)

    # Estimate archive size for deltas
    # If we quantize deltas to int8 and use arithmetic coding at entropy rate:
    n_delta_elements = delta_tensor.numel()
    bits_at_entropy = delta_entropy * n_delta_elements
    bytes_at_entropy = bits_at_entropy / 8

    # Flow field size: (N-1) × H × W × 2 × fp16 = huge, need quantized flow
    # Practical: store flow as int8 with scale = 15KB per frame pair
    flow_bytes_per_frame = H * W * 2 * 1  # int8 flow
    total_flow_bytes = flow_bytes_per_frame * (n_frames - 1)

    # Anchor frame: one RGB frame = H × W × 3 bytes
    anchor_bytes = H * W * 3

    print(f"  Mean L1 error (after warp): {mean_l1:.2f}")
    print(f"  Mean L-inf error: {mean_linf:.2f}")
    print(f"  Mean flow magnitude: {np.mean(flow_magnitudes):.2f} pixels")
    print(f"  Delta entropy: {delta_entropy:.2f} bits/element (vs {frame_entropy:.2f} for raw)")
    print(f"  Compression gain from flow: {frame_entropy / max(delta_entropy, 0.01):.1f}x")
    print(f"\n  Sparsity:")
    for thresh, frac in sparsity.items():
        print(f"    |delta| < {thresh.split('_')[1]}: {frac:.1%}")

    print(f"\n  Estimated archive sizes (full {n_frames} frames):")
    print(f"    Anchor frame: {anchor_bytes/1024:.1f}KB")
    print(f"    Deltas (entropy-coded): {bytes_at_entropy/1024:.1f}KB")
    print(f"    Flow fields (int8): {total_flow_bytes/1024:.1f}KB")
    print(f"    TOTAL: {(anchor_bytes + bytes_at_entropy + total_flow_bytes)/1024:.1f}KB")
    print(f"    (Context: contest budget ~250KB target)")

    # Compare with direct approaches
    renderer_archive_kb = 195.0  # FP4 renderer
    print(f"\n  Comparison:")
    print(f"    Renderer (FP4): ~{renderer_archive_kb:.0f}KB")
    delta_total_kb = (anchor_bytes + bytes_at_entropy + total_flow_bytes) / 1024
    print(f"    Backward delta: ~{delta_total_kb:.0f}KB")
    print(f"    Verdict: {'Feasible' if delta_total_kb < 250 else 'TOO LARGE'}")

    # Quality analysis: what's the maximum achievable quality with this approach?
    # If we store GT anchor + perfect flow + deltas, reconstruction is perfect
    # But we need to compress the deltas...
    # Threshold deltas below a cutoff for rate savings:
    print(f"\n[backward-delta] === RATE-QUALITY TRADEOFF ===")
    for thresh in [1.0, 2.0, 5.0, 10.0]:
        # Zero out deltas below threshold (treat as free)
        thresholded = delta_tensor.clone()
        thresholded[thresholded.abs() < thresh] = 0.0
        nonzero_frac = (thresholded != 0).float().mean().item()
        remaining_entropy = estimate_entropy(thresholded[thresholded != 0]) if nonzero_frac > 0 else 0.0
        # Estimated rate: sparse storage
        sparse_elements = int(thresholded.numel() * nonzero_frac)
        sparse_bytes = sparse_elements * 1  # int8 values
        index_bytes = sparse_elements * 2  # int16 indices (within block)
        total_bytes = sparse_bytes + index_bytes + total_flow_bytes + anchor_bytes
        # Reconstruction error from zeroing deltas
        recon_error = delta_tensor[delta_tensor.abs() < thresh].abs().mean().item() if (delta_tensor.abs() < thresh).any() else 0.0

        print(f"  Threshold={thresh:.0f}: "
              f"nonzero={nonzero_frac:.1%}, "
              f"archive~{total_bytes/1024:.0f}KB, "
              f"zeroed_error={recon_error:.2f}")

    # Save results
    results = {
        "n_frames": n_frames,
        "resolution": f"{H}x{W}",
        "flow_method": args.flow_method,
        "flow_time_s": flow_time,
        "delta_statistics": {
            "mean_l1": mean_l1,
            "mean_linf": mean_linf,
            "entropy_bits_per_element": delta_entropy,
            "frame_entropy_bits_per_element": frame_entropy,
            "compression_gain_from_flow": frame_entropy / max(delta_entropy, 0.01),
        },
        "sparsity": sparsity,
        "flow_statistics": {
            "mean_magnitude_px": float(np.mean(flow_magnitudes)),
            "max_magnitude_px": float(np.max(flow_magnitudes)),
            "std_magnitude_px": float(np.std(flow_magnitudes)),
        },
        "archive_estimate": {
            "anchor_bytes": anchor_bytes,
            "deltas_entropy_bytes": int(bytes_at_entropy),
            "flow_bytes_int8": total_flow_bytes,
            "total_bytes": int(anchor_bytes + bytes_at_entropy + total_flow_bytes),
            "total_kb": (anchor_bytes + bytes_at_entropy + total_flow_bytes) / 1024,
            "feasible_under_250kb": delta_total_kb < 250,
        },
        "warp_errors_per_frame": warp_errors[:50],  # first 50 for analysis
        "flow_magnitudes_per_frame": flow_magnitudes[:50],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save sample deltas for visualization
    torch.save({
        "sample_deltas": backward_deltas[:10],  # first 10 for viz
        "sample_flows": flow_magnitudes[:10],
    }, output_dir / "sample_data.pt")

    # Visualization
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Backward Delta Compression Analysis", fontsize=14)

        # Delta magnitude histogram
        axes[0, 0].hist(delta_tensor.abs().flatten().numpy(), bins=100, range=(0, 50), density=True)
        axes[0, 0].set_xlabel("Delta magnitude")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].set_title("Delta Magnitude Distribution")
        axes[0, 0].axvline(x=5, color="r", linestyle="--", label="threshold=5")
        axes[0, 0].legend()

        # Flow magnitude over time
        axes[0, 1].plot(flow_magnitudes)
        axes[0, 1].set_xlabel("Frame index")
        axes[0, 1].set_ylabel("Mean flow (pixels)")
        axes[0, 1].set_title("Optical Flow Magnitude Over Time")

        # Warp error over time
        axes[0, 2].plot(warp_errors)
        axes[0, 2].set_xlabel("Frame index")
        axes[0, 2].set_ylabel("Mean |delta| after warp")
        axes[0, 2].set_title("Warp Residual Error Over Time")

        # Sparsity curve
        threshs = list(range(1, 30))
        sparsities = [(delta_abs < t).float().mean().item() for t in threshs]
        axes[1, 0].plot(threshs, sparsities)
        axes[1, 0].set_xlabel("Threshold")
        axes[1, 0].set_ylabel("Fraction of pixels below threshold")
        axes[1, 0].set_title("Delta Sparsity Curve")
        axes[1, 0].grid(True)

        # Sample delta heatmap
        if len(backward_deltas) > 0:
            sample_delta = backward_deltas[0].abs().mean(dim=-1)  # (H, W)
            axes[1, 1].imshow(sample_delta.numpy(), cmap="hot", vmax=30)
            axes[1, 1].set_title("Sample Delta Magnitude (frame 0)")

        # Cumulative rate-distortion
        threshs_rd = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        nonzero_fracs = [(delta_abs >= t).float().mean().item() for t in threshs_rd]
        axes[1, 2].plot(threshs_rd, nonzero_fracs, "o-")
        axes[1, 2].set_xlabel("Threshold (delta cutoff)")
        axes[1, 2].set_ylabel("Fraction of non-zero deltas (rate)")
        axes[1, 2].set_title("Rate vs Threshold")
        axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(output_dir / "analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n[backward-delta] Visualization saved to {output_dir / 'analysis.png'}")
    except ImportError:
        print("[backward-delta] matplotlib not available, skipping visualization")

    # Final summary
    print(f"\n{'='*60}")
    print(f"[backward-delta] CONCLUSION")
    if delta_total_kb < 250:
        print(f"  FEASIBLE: {delta_total_kb:.0f}KB < 250KB budget")
        print(f"  But requires: perfect anchor + flow compression + sparse delta coding")
    else:
        print(f"  TOO LARGE: {delta_total_kb:.0f}KB > 250KB budget")
        print(f"  Would need aggressive thresholding (lossy) or better flow compression")
    print(f"  Flow compression gain: {frame_entropy / max(delta_entropy, 0.01):.1f}x over raw")
    print(f"  Key finding: {sparsity.get('below_5.0', 0):.1%} of deltas are below 5 pixels")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
