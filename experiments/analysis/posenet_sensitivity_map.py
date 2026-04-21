#!/usr/bin/env python3
"""PoseNet Sensitivity Map: full Jacobian analysis per pair.

Computes the Jacobian of PoseNet's 6D pose output w.r.t. input pixels for each
non-overlapping pair. Unlike the existing posenet_sensitivity.py (which computes
scalar sensitivity), this produces the FULL gradient magnitude map showing exactly
which pixels PoseNet cares about and how much.

Output structure:
    - per_pair_sensitivity: (600, 2, H, W) gradient magnitude per frame per pair
    - aggregate_map: (H, W) mean sensitivity across all pairs
    - class_sensitivity: (5, H, W) sensitivity per semantic class
    - pair_ranking: which pairs have highest total sensitivity (hardest for TTO)

Application: weight renderer training loss by sensitivity map so high-sensitivity
pixels get proportionally more gradient. This is Fridrich's "invest bits where
they matter" principle applied to neural rendering.

Usage:
    # Smoke test (local MPS):
    PYTHONPATH=src:upstream python experiments/analysis/posenet_sensitivity_map.py \
        --device mps --smoke

    # Full analysis:
    PYTHONPATH=src:upstream python experiments/analysis/posenet_sensitivity_map.py \
        --device mps --n-frames 1200

    # CUDA (faster):
    PYTHONPATH=src:upstream python experiments/analysis/posenet_sensitivity_map.py \
        --device cuda --n-frames 1200 --batch-pairs 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tac.utils import find_project_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full PoseNet Jacobian sensitivity map per pair",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", type=str, default="mps", choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=1200, help="Number of frames to analyze")
    p.add_argument("--batch-pairs", type=int, default=2,
                   help="Pairs per batch (small for VRAM — Jacobian is memory-heavy)")
    p.add_argument("--upstream", type=str, default=None, help="Path to upstream repo")
    p.add_argument("--video", type=str, default=None, help="Path to GT video")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--save-per-pair", action="store_true",
                   help="Save individual per-pair sensitivity maps (large)")
    p.add_argument("--smoke", action="store_true", help="Smoke test: 20 frames, 2 pairs")
    return p.parse_args()


def compute_pair_sensitivity(
    frames: tuple[torch.Tensor, torch.Tensor],
    posenet: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-pixel gradient magnitude through PoseNet for one pair.

    The gradient of PoseNet's 6D output w.r.t. each input pixel tells us
    how much that pixel affects the pose estimate. High gradient = PoseNet
    is sensitive to changes there.

    Args:
        frames: tuple of (H, W, 3) float tensors (frame0, frame1)
        posenet: frozen PoseNet (but we need gradients w.r.t. input)
        device: compute device

    Returns:
        Tuple of (H, W) gradient magnitude maps for frame0 and frame1.
    """
    f0, f1 = frames
    # Create input pair: (1, 2, C, H, W) — requires grad on input
    f0_t = f0.to(device).unsqueeze(0).permute(0, 3, 1, 2).contiguous()  # (1, 3, H, W)
    f1_t = f1.to(device).unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    pair = torch.cat([f0_t, f1_t], dim=0).unsqueeze(0)  # (1, 2, 3, H, W)
    pair.requires_grad_(True)

    # Forward through PoseNet
    posenet_in = posenet.preprocess_input(pair)
    posenet_out = posenet(posenet_in)
    pose_6 = posenet_out["pose"][..., :6]  # (1, 6)

    # Compute gradient of sum of pose outputs w.r.t. input pixels
    # This gives per-pixel influence on the total pose estimate
    grad = torch.autograd.grad(
        outputs=pose_6.sum(),
        inputs=pair,
        create_graph=False,
        retain_graph=False,
    )[0]  # (1, 2, 3, H, W)

    # Gradient magnitude: L2 norm across RGB channels
    grad_mag = grad[0].pow(2).sum(dim=1).sqrt()  # (2, H, W)
    return grad_mag[0].detach().cpu(), grad_mag[1].detach().cpu()


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.n_frames = 20

    device = torch.device(args.device)

    # Resolve paths
    root = find_project_root()
    upstream = Path(args.upstream) if args.upstream else root / "upstream"
    video_path = Path(args.video) if args.video else upstream / "videos" / "0.mkv"
    output_dir = Path(args.output_dir) if args.output_dir else (
        root / "experiments" / "results" / "posenet_sensitivity_full"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scorers (differentiable for gradient computation)
    from tac.scorer import load_differentiable_scorers, extract_gt_masks
    posenet, segnet = load_differentiable_scorers(str(upstream), device=device)

    # Decode video
    from tac.data import decode_video
    print(f"[sensitivity] Decoding video: {video_path}")
    gt_frames = decode_video(str(video_path))[:args.n_frames]
    n_pairs = len(gt_frames) // 2
    print(f"[sensitivity] Processing {n_pairs} pairs on {device}")

    # Extract masks for class-conditional analysis
    print(f"[sensitivity] Extracting SegNet masks...")
    gt_masks = extract_gt_masks(gt_frames, segnet, device=device)

    # Compute sensitivity maps
    all_sensitivities = []  # (n_pairs, 2, H, W)
    pair_total_sensitivity = []  # scalar per pair

    t0 = time.time()
    for pair_idx in range(n_pairs):
        f0 = gt_frames[pair_idx * 2].float()
        f1 = gt_frames[pair_idx * 2 + 1].float()

        s0, s1 = compute_pair_sensitivity((f0, f1), posenet, device)
        all_sensitivities.append(torch.stack([s0, s1], dim=0))
        pair_total_sensitivity.append((s0.sum() + s1.sum()).item())

        if pair_idx % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (pair_idx + 1) * (n_pairs - pair_idx - 1)
            print(f"  Pair {pair_idx}/{n_pairs} | total_sens={pair_total_sensitivity[-1]:.1f} | ETA: {eta:.0f}s")

    total_time = time.time() - t0
    print(f"[sensitivity] Complete in {total_time:.1f}s")

    # Stack into tensor
    sensitivity_tensor = torch.stack(all_sensitivities, dim=0)  # (n_pairs, 2, H, W)

    # Aggregate statistics
    aggregate_map = sensitivity_tensor.mean(dim=(0, 1))  # (H, W)

    # Per-class sensitivity
    H, W = aggregate_map.shape
    class_sensitivity = torch.zeros(5, H, W)
    class_counts = torch.zeros(5, H, W)
    for pair_idx in range(n_pairs):
        for frame_offset in range(2):
            frame_idx = pair_idx * 2 + frame_offset
            mask = gt_masks[frame_idx].cpu()  # (mask_H, mask_W) — may differ from (H, W)
            sens = sensitivity_tensor[pair_idx, frame_offset]  # (H, W)
            # Resize mask to match sensitivity resolution if needed
            if mask.shape != sens.shape:
                mask = F.interpolate(
                    mask.float().unsqueeze(0).unsqueeze(0),
                    size=sens.shape, mode="nearest",
                ).squeeze().long()
            for c in range(5):
                c_mask = (mask == c)
                class_sensitivity[c][c_mask] += sens[c_mask]
                class_counts[c][c_mask] += 1.0

    class_sensitivity = class_sensitivity / class_counts.clamp(min=1)

    # Pair ranking (hardest = highest total sensitivity)
    pair_ranking = sorted(
        enumerate(pair_total_sensitivity),
        key=lambda x: x[1],
        reverse=True,
    )

    # Save results
    torch.save(aggregate_map, output_dir / "aggregate_sensitivity.pt")
    torch.save(class_sensitivity, output_dir / "class_sensitivity.pt")

    if args.save_per_pair:
        torch.save(sensitivity_tensor, output_dir / "per_pair_sensitivity.pt")

    # Statistics
    stats = {
        "n_pairs": n_pairs,
        "total_time_s": total_time,
        "aggregate_stats": {
            "mean": aggregate_map.mean().item(),
            "std": aggregate_map.std().item(),
            "max": aggregate_map.max().item(),
            "min": aggregate_map.min().item(),
        },
        "per_class_mean_sensitivity": {
            f"class_{c}": class_sensitivity[c].mean().item() for c in range(5)
        },
        "top_20_hardest_pairs": [
            {"pair_idx": idx, "total_sensitivity": val}
            for idx, val in pair_ranking[:20]
        ],
        "bottom_20_easiest_pairs": [
            {"pair_idx": idx, "total_sensitivity": val}
            for idx, val in pair_ranking[-20:]
        ],
        "sensitivity_ratio_hard_vs_easy": (
            pair_ranking[0][1] / max(pair_ranking[-1][1], 1e-8)
        ),
    }

    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Visualization (matplotlib)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("PoseNet Sensitivity Analysis", fontsize=14)

        # Aggregate map
        im = axes[0, 0].imshow(aggregate_map.numpy(), cmap="hot", aspect="auto")
        axes[0, 0].set_title("Aggregate Sensitivity")
        plt.colorbar(im, ax=axes[0, 0])

        # Top 5 class sensitivities
        class_names = ["Road", "Lane", "Vehicle", "Movable", "Other"]
        for c in range(min(5, class_sensitivity.shape[0])):
            row, col = divmod(c + 1, 3)
            if row < 2 and col < 3:
                im = axes[row, col].imshow(class_sensitivity[c].numpy(), cmap="hot", aspect="auto")
                axes[row, col].set_title(f"Class {c}: {class_names[c]}")
                plt.colorbar(im, ax=axes[row, col])

        plt.tight_layout()
        plt.savefig(output_dir / "sensitivity_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[sensitivity] Visualization saved to {output_dir / 'sensitivity_analysis.png'}")
    except ImportError:
        print("[sensitivity] matplotlib not available, skipping visualization")

    # Print summary
    print(f"\n{'='*60}")
    print(f"[sensitivity] RESULTS")
    print(f"  Aggregate mean sensitivity: {stats['aggregate_stats']['mean']:.4f}")
    print(f"  Hardest pair: idx={pair_ranking[0][0]} (sens={pair_ranking[0][1]:.1f})")
    print(f"  Easiest pair: idx={pair_ranking[-1][0]} (sens={pair_ranking[-1][1]:.1f})")
    print(f"  Hard/Easy ratio: {stats['sensitivity_ratio_hard_vs_easy']:.1f}x")
    print(f"  Per-class:")
    for c in range(5):
        print(f"    {class_names[c]}: {stats['per_class_mean_sensitivity'][f'class_{c}']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
