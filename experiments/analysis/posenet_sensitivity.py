#!/usr/bin/env python3
"""PoseNet Sensitivity Map: per-pixel gradient magnitude through PoseNet.

Computes the Jacobian of PoseNet's pose output w.r.t. input pixels for each
non-overlapping pair, then aggregates the gradient magnitude into a spatial
heatmap. This map identifies which pixels PoseNet is sensitive to (high gradient)
vs insensitive to (low gradient).

The insensitive patches are SAFE to perturb for compressibility without
changing the PoseNet score. This is the foundation for GT-initialized sparse
patch TTO (#1).

Usage::

    PYTHONPATH=src:upstream python experiments/analysis/posenet_sensitivity.py \
        --device mps --n-frames 40

    PYTHONPATH=src:upstream python experiments/analysis/posenet_sensitivity.py \
        --device cuda --n-frames 1200

Outputs:
    experiments/results/posenet_sensitivity/sensitivity_map.pt   — (H, W) float tensor
    experiments/results/posenet_sensitivity/sensitivity_map.png  — matplotlib heatmap
    experiments/results/posenet_sensitivity/patch_ranking.json   — ranked 7x7 patches
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def find_project_root() -> Path:
    """Walk up from this file to find the project root (contains src/)."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "src").is_dir() and (p / "upstream").is_dir():
            return p
        p = p.parent
    raise RuntimeError("Cannot find project root (expected src/ and upstream/ dirs)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PoseNet per-pixel gradient sensitivity map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=1200, help="Number of frames to analyze")
    p.add_argument("--patch-size", type=int, default=7, help="Patch size for ranking")
    p.add_argument("--upstream", type=str, default=None, help="Path to upstream repo")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--video", type=str, default=None, help="Path to GT video")
    p.add_argument("--batch-pairs", type=int, default=4,
                   help="Pairs per batch for gradient computation (small for VRAM)")
    p.add_argument("--smoke", action="store_true", help="Smoke test: 20 frames")
    return p.parse_args()


def compute_posenet_gradient_map(
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    device: torch.device,
    batch_pairs: int = 4,
) -> torch.Tensor:
    """Compute per-pixel PoseNet gradient magnitude averaged over all pairs.

    For each non-overlapping pair (2k, 2k+1), computes the gradient of the
    PoseNet pose output w.r.t. the input pixels, then takes the L2 norm
    across pose dimensions and channels to get a per-pixel scalar.

    Args:
        gt_frames: list of (H, W, 3) uint8 tensors.
        posenet: frozen PoseNet model (on device).
        device: computation device.
        batch_pairs: pairs per gradient computation (small to avoid OOM).

    Returns:
        (H, W) float tensor of average gradient magnitude at scorer resolution.
    """
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W

    N = len(gt_frames)
    P = N // 2

    # Accumulator at scorer resolution
    grad_accum = torch.zeros(SEGNET_INPUT_H, SEGNET_INPUT_W, device="cpu", dtype=torch.float64)
    n_accumulated = 0

    for start in range(0, P, batch_pairs):
        end = min(start + batch_pairs, P)
        B = end - start

        # Build pair tensor with gradients enabled
        pair_list = []
        for k in range(start, end):
            f0 = gt_frames[2 * k].float()      # (H, W, 3)
            f1 = gt_frames[2 * k + 1].float()  # (H, W, 3)
            pair_list.append(torch.stack([f0, f1], dim=0))  # (2, H, W, 3)

        pairs_hwc = torch.stack(pair_list).to(device)  # (B, 2, H, W, 3)
        pairs_hwc.requires_grad_(True)

        # Convert to (B, 2, C, H, W) for PoseNet
        pairs_chw = pairs_hwc.permute(0, 1, 4, 2, 3).contiguous()

        # Resize to scorer resolution if needed
        _, T, C, H, W = pairs_chw.shape
        if H != SEGNET_INPUT_H or W != SEGNET_INPUT_W:
            pairs_flat = pairs_chw.reshape(B * T, C, H, W)
            pairs_flat = F.interpolate(
                pairs_flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear", align_corners=False,
            )
            pairs_chw = pairs_flat.reshape(B, T, C, SEGNET_INPUT_H, SEGNET_INPUT_W)

        # PoseNet forward
        posenet_in = posenet.preprocess_input(pairs_chw)
        posenet_out = posenet(posenet_in)
        pose = posenet_out["pose"][..., :6]  # (B, 6)

        # Gradient of sum-of-squares of all pose outputs w.r.t. input pixels.
        # Using pose.pow(2).sum() instead of pose.sum() avoids sign cancellation:
        # if tx gradient is +g and ty gradient is -g, sum() would cancel them,
        # making a high-sensitivity pixel appear insensitive. pow(2) ensures all
        # gradient directions contribute positively.
        scalar_loss = pose.pow(2).sum()
        scalar_loss.backward()

        # pairs_hwc.grad is (B, 2, H, W, 3) — gradient at input resolution
        grad = pairs_hwc.grad.detach()  # (B, 2, H_in, W_in, 3)

        # L2 norm across channels to get per-pixel magnitude, then average over batch and time
        # Shape: (B, 2, H_in, W_in)
        grad_mag = grad.norm(dim=-1)  # L2 across RGB channels
        # Average over batch (B) and time (2 frames per pair)
        grad_mag_avg = grad_mag.mean(dim=(0, 1))  # (H_in, W_in)

        # If input wasn't at scorer resolution, resize the gradient map
        H_in, W_in = grad_mag_avg.shape
        if H_in != SEGNET_INPUT_H or W_in != SEGNET_INPUT_W:
            grad_mag_avg = F.interpolate(
                grad_mag_avg.unsqueeze(0).unsqueeze(0),
                size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear", align_corners=False,
            ).squeeze()

        grad_accum += grad_mag_avg.cpu().double() * B
        n_accumulated += B

        # Free memory
        del pairs_hwc, pairs_chw, posenet_in, posenet_out, pose, scalar_loss, grad
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    # Average across all batches
    sensitivity_map = (grad_accum / max(n_accumulated, 1)).float()
    return sensitivity_map


def rank_patches(
    sensitivity_map: torch.Tensor,
    patch_size: int = 7,
) -> list[dict]:
    """Rank non-overlapping patches by sensitivity (ascending = most insensitive first).

    Args:
        sensitivity_map: (H, W) float tensor of gradient magnitudes.
        patch_size: size of square patches.

    Returns:
        List of dicts with patch info, sorted by mean sensitivity (ascending).
        Each dict: {row, col, mean_sensitivity, max_sensitivity, h, w}.
    """
    H, W = sensitivity_map.shape
    patches = []

    for r in range(0, H - patch_size + 1, patch_size):
        for c in range(0, W - patch_size + 1, patch_size):
            patch = sensitivity_map[r:r + patch_size, c:c + patch_size]
            patches.append({
                "row": r,
                "col": c,
                "h": patch_size,
                "w": patch_size,
                "mean_sensitivity": float(patch.mean()),
                "max_sensitivity": float(patch.max()),
            })

    # Sort ascending: lowest sensitivity = safest to perturb
    patches.sort(key=lambda x: x["mean_sensitivity"])
    return patches


def visualize_sensitivity(
    sensitivity_map: torch.Tensor,
    output_path: Path,
    top_k_patches: list[dict] | None = None,
    n_show: int = 50,
) -> None:
    """Save sensitivity heatmap with optional top-K insensitive patch overlay.

    Args:
        sensitivity_map: (H, W) float tensor.
        output_path: path for PNG output.
        top_k_patches: optional ranked patches to overlay.
        n_show: number of patches to highlight.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: raw sensitivity heatmap
    sm_np = sensitivity_map.numpy()
    im = axes[0].imshow(sm_np, cmap="hot", aspect="auto")
    axes[0].set_title("PoseNet Gradient Sensitivity Map")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")
    plt.colorbar(im, ax=axes[0], label="Gradient Magnitude")

    # Right: log-scale with patch overlay
    sm_log = np.log1p(sm_np)
    im2 = axes[1].imshow(sm_log, cmap="hot", aspect="auto")
    axes[1].set_title(f"Log Sensitivity + Top-{n_show} Insensitive Patches")
    axes[1].set_xlabel("Width")
    axes[1].set_ylabel("Height")
    plt.colorbar(im2, ax=axes[1], label="log(1 + Gradient Magnitude)")

    if top_k_patches:
        for i, p in enumerate(top_k_patches[:n_show]):
            rect = mpatches.Rectangle(
                (p["col"], p["row"]), p["w"], p["h"],
                linewidth=1, edgecolor="cyan", facecolor="none", alpha=0.7,
            )
            axes[1].add_patch(rect)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved sensitivity heatmap to {output_path}")


def main():
    args = parse_args()

    if args.smoke:
        args.n_frames = 20
        print("[smoke] Smoke test: 20 frames")

    # Ensure even frame count
    args.n_frames = args.n_frames - (args.n_frames % 2)

    root = find_project_root()
    device = torch.device(args.device)
    upstream = Path(args.upstream) if args.upstream else root / "upstream"

    if args.output_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%S")
        args.output_dir = str(root / "experiments" / "results" / f"posenet_sensitivity_{ts}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = args.video or str(upstream / "videos" / "0.mkv")

    print(f"[config] device={device}, n_frames={args.n_frames}, "
          f"patch_size={args.patch_size}, batch_pairs={args.batch_pairs}")

    # ── Step 1: Load scorers ─────────────────────────────────────────────
    print("\n[1/4] Loading PoseNet...")
    t0 = time.monotonic()
    from tac.scorer import load_scorers
    posenet, _ = load_scorers(
        posenet_path=upstream / "models" / "posenet.safetensors",
        segnet_path=upstream / "models" / "segnet.safetensors",
        device=str(device),
        upstream_dir=str(upstream),
    )
    print(f"[1/4] PoseNet loaded in {time.monotonic() - t0:.1f}s")

    # ── Step 2: Decode GT video ──────────────────────────────────────────
    print(f"\n[2/4] Decoding GT video ({args.n_frames} frames)...")
    t0 = time.monotonic()
    from tac.data import decode_video
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W

    gt_frames_full = decode_video(video_path, target_h=SEGNET_INPUT_H, target_w=SEGNET_INPUT_W)
    gt_frames = gt_frames_full[:args.n_frames]
    args.n_frames = len(gt_frames) - (len(gt_frames) % 2)
    gt_frames = gt_frames[:args.n_frames]
    assert args.n_frames >= 2, f"Need at least 2 frames, got {len(gt_frames)}"
    print(f"[2/4] Decoded {args.n_frames} frames ({gt_frames[0].shape}) in {time.monotonic() - t0:.1f}s")

    # ── Step 3: Compute sensitivity map ──────────────────────────────────
    print(f"\n[3/4] Computing PoseNet gradient sensitivity ({args.n_frames // 2} pairs)...")
    t0 = time.monotonic()
    sensitivity_map = compute_posenet_gradient_map(
        gt_frames, posenet, device, batch_pairs=args.batch_pairs,
    )
    dt = time.monotonic() - t0
    print(f"[3/4] Sensitivity map computed in {dt:.1f}s ({dt / (args.n_frames // 2):.2f}s/pair)")
    print(f"[3/4] Map shape: {sensitivity_map.shape}, "
          f"min={sensitivity_map.min():.6f}, max={sensitivity_map.max():.6f}, "
          f"mean={sensitivity_map.mean():.6f}")

    # ── Step 4: Rank patches and save ────────────────────────────────────
    print(f"\n[4/4] Ranking {args.patch_size}x{args.patch_size} patches...")
    patches = rank_patches(sensitivity_map, patch_size=args.patch_size)
    n_patches = len(patches)
    print(f"[4/4] {n_patches} patches ranked. "
          f"Most insensitive: mean={patches[0]['mean_sensitivity']:.6f}, "
          f"Most sensitive: mean={patches[-1]['mean_sensitivity']:.6f}")

    # Sensitivity ratio: how many patches have <10% of max sensitivity
    if n_patches > 0:
        max_sens = patches[-1]["mean_sensitivity"]
        threshold = max_sens * 0.1
        n_low = sum(1 for p in patches if p["mean_sensitivity"] < threshold)
        print(f"[4/4] {n_low}/{n_patches} patches ({100 * n_low / n_patches:.1f}%) "
              f"below 10% threshold = safe to perturb")
    else:
        n_low = 0
        print("[4/4] WARNING: no patches found (map too small for patch size)")

    # Save artifacts
    torch.save(sensitivity_map, output_dir / "sensitivity_map.pt")
    print(f"[save] Sensitivity map: {output_dir / 'sensitivity_map.pt'}")

    # Save patch ranking (top 200 + bottom 20)
    ranking_data = {
        "patch_size": args.patch_size,
        "n_frames": args.n_frames,
        "n_pairs": args.n_frames // 2,
        "n_patches": n_patches,
        "map_shape": list(sensitivity_map.shape),
        "sensitivity_stats": {
            "min": float(sensitivity_map.min()),
            "max": float(sensitivity_map.max()),
            "mean": float(sensitivity_map.mean()),
            "std": float(sensitivity_map.std()),
        },
        "n_low_sensitivity": n_low,
        "low_sensitivity_fraction": n_low / max(n_patches, 1),
        "top_insensitive": patches[:200],
        "most_sensitive": patches[-20:],
    }
    with open(output_dir / "patch_ranking.json", "w") as f:
        json.dump(ranking_data, f, indent=2)
    print(f"[save] Patch ranking: {output_dir / 'patch_ranking.json'}")

    # Visualization
    visualize_sensitivity(
        sensitivity_map,
        output_dir / "sensitivity_map.png",
        top_k_patches=patches,
        n_show=50,
    )

    # Also save to standard reports location
    reports_path = root / "reports" / "graphs" / "posenet_sensitivity.png"
    visualize_sensitivity(sensitivity_map, reports_path, top_k_patches=patches, n_show=50)

    print(f"\n{'=' * 60}")
    print("POSENET SENSITIVITY ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Frames analyzed: {args.n_frames}")
    print(f"  Patch size: {args.patch_size}x{args.patch_size}")
    print(f"  Total patches: {n_patches}")
    print(f"  Safe patches (<10% max): {n_low} ({100 * n_low / max(n_patches, 1):.1f}%)")
    print(f"  Sensitivity range: [{sensitivity_map.min():.6f}, {sensitivity_map.max():.6f}]")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
