#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Scorer Internal Analysis: layer-by-layer activation comparison GT vs TTO.

Hooks all conv layers in both SegNet and PoseNet, then computes per-layer
activation differences between GT frames and TTO/renderer frames. This reveals:

    1. Which layers show the largest activation difference (the "bottleneck layers")
    2. Which spatial regions have the largest activation mismatch
    3. Whether the problem is in early features (texture/edges) or late features (semantics)

This informs targeted optimization: if Layer 7 has 10x the activation distance
of other layers, our TTO loss should directly target Layer 7's activations
(feature matching loss at that specific layer).

Usage:
    # Smoke test (local MPS):
    PYTHONPATH=src:upstream python experiments/analysis/scorer_internals.py \
        --device mps --smoke

    # Full analysis with renderer frames:
    PYTHONPATH=src:upstream python experiments/analysis/scorer_internals.py \
        --device mps --n-frames 1200 \
        --rendered-frames experiments/results/renderer_tto_20260414T142644/tto_frames.pt

    # Compare TTO v7 frames:
    PYTHONPATH=src:upstream python experiments/analysis/scorer_internals.py \
        --device cuda --n-frames 1200 \
        --rendered-frames experiments/results/tto_v7_hinge_500/tto_frames.pt
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from tac.utils import find_project_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scorer internal activation analysis: GT vs rendered",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", type=str, default="mps", choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=1200, help="Number of frames to analyze")
    p.add_argument("--rendered-frames", type=str, default=None,
                   help="Path to rendered/TTO frames .pt file (if None, uses renderer checkpoint)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Renderer checkpoint (used if --rendered-frames not provided)")
    p.add_argument("--upstream", type=str, default=None, help="Path to upstream repo")
    p.add_argument("--video", type=str, default=None, help="Path to GT video")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--top-k-layers", type=int, default=10,
                   help="Number of top layers to report in detail")
    p.add_argument("--smoke", action="store_true", help="Smoke test: 20 frames")
    return p.parse_args()


class ActivationHook:
    """Hook that captures intermediate activations from a model.

    Registers forward hooks on all Conv2d layers and stores their outputs.
    Memory-efficient: stores only L2 norms per spatial location, not full activations.
    """

    def __init__(self, model: nn.Module, model_name: str = "model"):
        self.model_name = model_name
        self.activations: dict[str, torch.Tensor] = {}
        self._hooks = []

        # Register hooks on all Conv2d layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            # Store L2 norm per spatial position: (B, H, W)
            self.activations[name] = output.detach().pow(2).sum(dim=1).sqrt()
        return hook_fn

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @property
    def layer_names(self) -> list[str]:
        return list(self.activations.keys())


def compute_activation_distances(
    gt_frames: list[torch.Tensor],
    rendered_frames: list[torch.Tensor],
    posenet: nn.Module,
    segnet: nn.Module,
    device: torch.device,
    n_frames: int | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Compute per-layer activation distances between GT and rendered frames.

    Args:
        gt_frames: list of (H, W, 3) uint8 tensors
        rendered_frames: list of (H, W, 3) float tensors (renderer output)
        posenet: PoseNet model
        segnet: SegNet model
        device: compute device
        n_frames: limit analysis to first N frames

    Returns:
        Dict with per-layer distance statistics for both PoseNet and SegNet.
    """
    if n_frames:
        gt_frames = gt_frames[:n_frames]
        rendered_frames = rendered_frames[:n_frames]

    n_pairs = min(len(gt_frames), len(rendered_frames)) // 2

    # Set up hooks
    posenet_hook = ActivationHook(posenet, "posenet")
    segnet_hook = ActivationHook(segnet, "segnet")

    # Track per-layer distances
    posenet_dists: dict[str, list[float]] = defaultdict(list)
    segnet_dists: dict[str, list[float]] = defaultdict(list)

    for pair_idx in range(n_pairs):
        # GT pair
        f0_gt = gt_frames[pair_idx * 2].float().to(device)
        f1_gt = gt_frames[pair_idx * 2 + 1].float().to(device)
        gt_pair = torch.stack([f0_gt, f1_gt], dim=0).unsqueeze(0).permute(0, 1, 4, 2, 3).contiguous()

        # Rendered pair
        f0_r = rendered_frames[pair_idx * 2].float().to(device)
        f1_r = rendered_frames[pair_idx * 2 + 1].float().to(device)
        r_pair = torch.stack([f0_r, f1_r], dim=0).unsqueeze(0).permute(0, 1, 4, 2, 3).contiguous()

        with torch.no_grad():
            # Run GT through both scorers
            posenet_hook.clear()
            segnet_hook.clear()
            posenet(posenet.preprocess_input(gt_pair))
            gt_posenet_acts = {k: v.clone() for k, v in posenet_hook.activations.items()}

            segnet_hook.clear()
            segnet(segnet.preprocess_input(gt_pair))
            gt_segnet_acts = {k: v.clone() for k, v in segnet_hook.activations.items()}

            # Run rendered through both scorers
            posenet_hook.clear()
            segnet_hook.clear()
            posenet(posenet.preprocess_input(r_pair))
            r_posenet_acts = {k: v.clone() for k, v in posenet_hook.activations.items()}

            segnet_hook.clear()
            segnet(segnet.preprocess_input(r_pair))
            r_segnet_acts = {k: v.clone() for k, v in segnet_hook.activations.items()}

        # Compute per-layer L2 distance
        for layer_name in gt_posenet_acts:
            if layer_name in r_posenet_acts:
                dist = (gt_posenet_acts[layer_name] - r_posenet_acts[layer_name]).pow(2).mean().sqrt()
                posenet_dists[layer_name].append(dist.item())

        for layer_name in gt_segnet_acts:
            if layer_name in r_segnet_acts:
                dist = (gt_segnet_acts[layer_name] - r_segnet_acts[layer_name]).pow(2).mean().sqrt()
                segnet_dists[layer_name].append(dist.item())

        if pair_idx % 20 == 0:
            print(f"  Pair {pair_idx}/{n_pairs}")

    # Clean up hooks
    posenet_hook.remove_hooks()
    segnet_hook.remove_hooks()

    return {
        "posenet": dict(posenet_dists),
        "segnet": dict(segnet_dists),
    }


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
        root / "experiments" / "results" / "scorer_internals"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scorers
    from tac.scorer import load_differentiable_scorers, extract_gt_masks
    posenet, segnet = load_differentiable_scorers(str(upstream), device=device)

    # Decode GT video
    from tac.data import decode_video
    print(f"[internals] Decoding GT video: {video_path}")
    gt_frames = decode_video(str(video_path))[:args.n_frames]

    # Get rendered frames
    if args.rendered_frames:
        print(f"[internals] Loading rendered frames from {args.rendered_frames}")
        rendered_data = torch.load(args.rendered_frames, map_location="cpu", weights_only=False)
        if isinstance(rendered_data, dict) and "frames" in rendered_data:
            rendered_frames = rendered_data["frames"]
        elif isinstance(rendered_data, torch.Tensor):
            # Assume (N, H, W, 3) or (N, 3, H, W)
            if rendered_data.shape[-1] == 3:
                rendered_frames = [rendered_data[i] for i in range(rendered_data.shape[0])]
            else:
                rendered_frames = [rendered_data[i].permute(1, 2, 0) for i in range(rendered_data.shape[0])]
        else:
            rendered_frames = rendered_data
    else:
        # Use renderer to generate frames
        checkpoint = Path(args.checkpoint) if args.checkpoint else (
            root / "experiments" / "results" / "v5_lagrangian_renderer" / "renderer_best.pt"
        )
        from tac.checkpoint import verify_checkpoint_identity
        verify_checkpoint_identity(str(checkpoint))

        from tac.renderer import MaskRenderer
        print(f"[internals] Generating frames with renderer: {checkpoint}")
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            renderer_state = state["model_state_dict"]
            config = state.get("config", {})
        else:
            renderer_state = state
            config = {}

        renderer = MaskRenderer(
            embed_dim=config.get("embed_dim", 6),
            base_ch=config.get("base_ch", 36),
            mid_ch=config.get("mid_ch", 60),
            depth=config.get("depth", 1),
            pose_dim=config.get("pose_dim", 0),
        )
        renderer.load_state_dict(renderer_state, strict=False)
        renderer = renderer.to(device).eval()

        gt_masks = extract_gt_masks(gt_frames, segnet, device=device)
        rendered_frames = []
        with torch.no_grad():
            for i in range(len(gt_frames)):
                mask = gt_masks[i].to(device).unsqueeze(0)
                rgb = renderer(mask)  # (1, 3, H, W)
                rendered_frames.append(rgb[0].permute(1, 2, 0).cpu())  # (H, W, 3)

    print(f"[internals] GT frames: {len(gt_frames)}, Rendered frames: {len(rendered_frames)}")

    # Compute activation distances
    print("\n[internals] Computing per-layer activation distances...")
    t0 = time.time()
    distances = compute_activation_distances(
        gt_frames, rendered_frames, posenet, segnet, device,
        n_frames=args.n_frames,
    )
    total_time = time.time() - t0
    print(f"[internals] Complete in {total_time:.1f}s")

    # Analyze results
    results = {"posenet_layers": {}, "segnet_layers": {}, "summary": {}}

    for scorer_name in ["posenet", "segnet"]:
        layer_stats = []
        for layer_name, dists in distances[scorer_name].items():
            import numpy as np
            dists_arr = np.array(dists)
            stat = {
                "layer_name": layer_name,
                "mean_distance": float(dists_arr.mean()),
                "std_distance": float(dists_arr.std()),
                "max_distance": float(dists_arr.max()),
                "min_distance": float(dists_arr.min()),
            }
            layer_stats.append(stat)

        # Sort by mean distance (highest = most problematic)
        layer_stats.sort(key=lambda x: x["mean_distance"], reverse=True)
        results[f"{scorer_name}_layers"] = layer_stats

    # Top-K analysis
    print(f"\n[internals] === TOP-{args.top_k_layers} PROBLEMATIC LAYERS ===")

    for scorer_name in ["posenet", "segnet"]:
        print(f"\n  {scorer_name.upper()}:")
        for i, stat in enumerate(results[f"{scorer_name}_layers"][:args.top_k_layers]):
            print(f"    {i+1}. {stat['layer_name']}: mean={stat['mean_distance']:.4f} "
                  f"(std={stat['std_distance']:.4f}, max={stat['max_distance']:.4f})")

    # Summary statistics
    posenet_total = sum(s["mean_distance"] for s in results["posenet_layers"])
    segnet_total = sum(s["mean_distance"] for s in results["segnet_layers"])

    results["summary"] = {
        "posenet_total_activation_distance": posenet_total,
        "segnet_total_activation_distance": segnet_total,
        "posenet_n_layers": len(results["posenet_layers"]),
        "segnet_n_layers": len(results["segnet_layers"]),
        "posenet_top3_contribution_pct": (
            sum(s["mean_distance"] for s in results["posenet_layers"][:3]) / max(posenet_total, 1e-8) * 100
        ),
        "segnet_top3_contribution_pct": (
            sum(s["mean_distance"] for s in results["segnet_layers"][:3]) / max(segnet_total, 1e-8) * 100
        ),
        "total_time_s": total_time,
        "n_pairs_analyzed": min(len(gt_frames), len(rendered_frames)) // 2,
    }

    # Save
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Visualization
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        for ax_idx, scorer_name in enumerate(["posenet", "segnet"]):
            layers = results[f"{scorer_name}_layers"][:20]
            names = [s["layer_name"].split(".")[-1][:15] for s in layers]
            means = [s["mean_distance"] for s in layers]
            stds = [s["std_distance"] for s in layers]

            axes[ax_idx].barh(range(len(names)), means, xerr=stds, alpha=0.7)
            axes[ax_idx].set_yticks(range(len(names)))
            axes[ax_idx].set_yticklabels(names, fontsize=8)
            axes[ax_idx].set_xlabel("Mean Activation Distance (L2)")
            axes[ax_idx].set_title(f"{scorer_name.upper()} — Top 20 Layers by Distance")
            axes[ax_idx].invert_yaxis()

        plt.tight_layout()
        plt.savefig(output_dir / "activation_distances.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n[internals] Visualization saved to {output_dir / 'activation_distances.png'}")
    except ImportError:
        print("[internals] matplotlib not available, skipping visualization")

    # Final summary
    print(f"\n{'='*60}")
    print("[internals] SUMMARY")
    print(f"  PoseNet: {len(results['posenet_layers'])} layers, total dist={posenet_total:.2f}")
    print(f"    Top 3 layers account for {results['summary']['posenet_top3_contribution_pct']:.1f}% of distance")
    print(f"  SegNet: {len(results['segnet_layers'])} layers, total dist={segnet_total:.2f}")
    print(f"    Top 3 layers account for {results['summary']['segnet_top3_contribution_pct']:.1f}% of distance")
    print("  Implication: feature matching loss on top-3 layers could close the gap")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
