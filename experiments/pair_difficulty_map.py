#!/usr/bin/env python3
"""Per-pair PoseNet/SegNet difficulty map for adaptive TTO budget allocation.

Computes per-pair PoseNet MSE and SegNet disagreement for ALL 600 pairs
using the renderer output. The difficulty map enables adaptive TTO budget
allocation: hard pairs (high PoseNet distortion) get more TTO steps while
easy pairs (already low distortion) get zero.

Key insight from the step curve experiment: PoseNet saturates at ~100 steps
while SegNet dominates the score at 77:1 leverage ratio. This script
identifies which pairs are already easy (skip TTO) vs hard (allocate budget).

Runs locally on MPS (no GPU cost) or on any CUDA device.

Usage:
    # Full run on MPS (all 600 pairs):
    PYTHONPATH=src:upstream python experiments/pair_difficulty_map.py \
        --checkpoint /path/to/renderer_best.pt --device mps

    # Smoke test (first 20 pairs):
    PYTHONPATH=src:upstream python experiments/pair_difficulty_map.py \
        --checkpoint /path/to/renderer_best.pt --device mps --smoke

    # CUDA:
    PYTHONPATH=src:upstream python experiments/pair_difficulty_map.py \
        --checkpoint /path/to/renderer_best.pt --device cuda

Output:
    experiments/results/pair_difficulty_map.json  -- per-pair metrics
    experiments/results/pair_difficulty_distribution.png  -- histogram
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Per-pair PoseNet/SegNet difficulty map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to renderer .pt checkpoint",
    )
    p.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["cuda", "mps", "cpu"],
        help="Computation device",
    )
    p.add_argument(
        "--upstream",
        type=str,
        default="upstream/",
        help="Path to upstream repo",
    )
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to GT video (default: upstream/videos/0.mkv)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Output directory for results",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Pairs per forward pass for scoring (lower = less VRAM)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: first 20 pairs only",
    )
    # CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True. Removed
    # `--no-eval-roundtrip` flag; only escape hatch is TAC_ALLOW_NO_ROUNDTRIP=1.
    p.add_argument(
        "--eval-roundtrip", action="store_true", default=True,
        help="Simulate contest eval resize chain in scorer loss. ALWAYS True; "
             "disabling requires TAC_ALLOW_NO_ROUNDTRIP=1.",
    )
    return p.parse_args()


def compute_per_pair_distortions(
    rendered_frames: torch.Tensor,
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    batch_size: int = 8,
    eval_roundtrip: bool = True,
) -> list[dict[str, float]]:
    """Compute PoseNet MSE and SegNet disagreement for each pair independently.

    Unlike compute_proxy_score which averages across all pairs, this returns
    per-pair metrics for difficulty analysis.

    Args:
        rendered_frames: (N, H, W, 3) float tensor of rendered frames.
        gt_frames: list of (H, W, 3) uint8 GT frames.
        posenet: frozen PoseNet with differentiable preprocessing.
        segnet: frozen SegNet.
        device: computation device.
        batch_size: pairs per forward pass.
        eval_roundtrip: simulate official scorer resolution pipeline.

    Returns:
        List of dicts, one per pair, with 'posenet_mse', 'segnet_disagree',
        'pose_contribution', 'seg_contribution', 'score', 'pair_idx' keys.
    """
    from tac.camera import CAMERA_H, CAMERA_W, SEGNET_INPUT_H, SEGNET_INPUT_W

    N = rendered_frames.shape[0]
    n_pairs = N // 2
    pair_results: list[dict[str, float]] = []

    for batch_start in range(0, n_pairs, batch_size):
        batch_end = min(batch_start + batch_size, n_pairs)
        B = batch_end - batch_start

        # Build pair tensors for this batch
        cand_pairs: list[torch.Tensor] = []
        gt_pairs: list[torch.Tensor] = []
        for k in range(batch_start, batch_end):
            cand_pairs.append(
                torch.stack(
                    [rendered_frames[2 * k], rendered_frames[2 * k + 1]], dim=0
                )
            )
            gt_pairs.append(
                torch.stack(
                    [gt_frames[2 * k].float(), gt_frames[2 * k + 1].float()],
                    dim=0,
                )
            )

        cand_t = torch.stack(cand_pairs).to(device)  # (B, 2, H, W, 3)
        gt_t = torch.stack(gt_pairs).to(device)

        # Convert to CHW format for scorers
        cand_chw = cand_t.permute(0, 1, 4, 2, 3).contiguous()
        gt_chw = gt_t.permute(0, 1, 4, 2, 3).contiguous()

        # Resize to scorer input resolution if needed
        _, _, C, H, W = cand_chw.shape
        if H != SEGNET_INPUT_H or W != SEGNET_INPUT_W:
            cand_flat = cand_chw.reshape(B * 2, C, H, W)
            gt_flat = gt_chw.reshape(B * 2, C, H, W)
            cand_flat = F.interpolate(
                cand_flat,
                size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear",
                align_corners=False,
            )
            gt_flat = F.interpolate(
                gt_flat,
                size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear",
                align_corners=False,
            )
            cand_chw = cand_flat.reshape(B, 2, C, SEGNET_INPUT_H, SEGNET_INPUT_W)
            gt_chw = gt_flat.reshape(B, 2, C, SEGNET_INPUT_H, SEGNET_INPUT_W)

        cand_chw = cand_chw.round().clamp(0, 255)

        if eval_roundtrip:
            flat = cand_chw.reshape(-1, *cand_chw.shape[2:])
            flat = F.interpolate(
                flat,
                size=(CAMERA_H, CAMERA_W),
                mode="bilinear",
                align_corners=False,
            )
            flat = flat.round().clamp(0, 255)
            flat = F.interpolate(
                flat,
                size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear",
                align_corners=False,
            )
            cand_chw = flat.reshape(B, 2, *flat.shape[1:])

        with torch.no_grad():
            # PoseNet: per-pair MSE on pose output (6-d)
            fp_in = posenet.preprocess_input(cand_chw)
            gp_in = posenet.preprocess_input(gt_chw)
            fp_out = posenet(fp_in)
            gp_out = posenet(gp_in)
            pose_mse_per_pair = (
                (fp_out["pose"][..., :6] - gp_out["pose"][..., :6])
                .pow(2)
                .mean(dim=-1)
            )  # (B,)

            # SegNet: per-pair hard disagreement
            # SegNet.preprocess_input selects only the LAST frame per pair
            # (x[:, -1, ...]), so output is (B, num_classes, H, W) -- one
            # disagreement value per pair, matching the official scorer.
            fs_in = segnet.preprocess_input(cand_chw)
            gs_in = segnet.preprocess_input(gt_chw)
            fs_out = segnet(fs_in)
            gs_out = segnet(gs_in)
            diff = (fs_out.argmax(dim=1) != gs_out.argmax(dim=1)).float()
            seg_disagree_per_pair = diff.mean(
                dim=tuple(range(1, diff.ndim))
            )  # (B,)

        # Record per-pair results
        for i in range(B):
            pair_idx = batch_start + i
            pose_val = pose_mse_per_pair[i].item()
            seg_val = seg_disagree_per_pair[i].item()
            pose_contrib = math.sqrt(max(0.0, 10.0 * pose_val))
            seg_contrib = 100.0 * seg_val
            score = seg_contrib + pose_contrib

            pair_results.append(
                {
                    "pair_idx": pair_idx,
                    "posenet_mse": pose_val,
                    "segnet_disagree": seg_val,
                    "pose_contribution": round(pose_contrib, 6),
                    "seg_contribution": round(seg_contrib, 6),
                    "score": round(score, 6),
                }
            )

    return pair_results


def compute_statistics(
    pair_results: list[dict[str, float]],
) -> dict[str, Any]:
    """Compute summary statistics from per-pair results.

    Args:
        pair_results: list of per-pair metric dicts.

    Returns:
        Dict with mean, std, min, max, percentiles, and tier assignments.
    """
    import numpy as np

    n = len(pair_results)
    pose_vals = np.array([r["posenet_mse"] for r in pair_results])
    seg_vals = np.array([r["segnet_disagree"] for r in pair_results])
    scores = np.array([r["score"] for r in pair_results])

    def _stats(arr: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
        }

    # Tier assignments based on PoseNet difficulty
    pose_p50 = float(np.percentile(pose_vals, 50))
    pose_p80 = float(np.percentile(pose_vals, 80))

    easy_pairs = [r["pair_idx"] for r in pair_results if r["posenet_mse"] <= pose_p50]
    medium_pairs = [
        r["pair_idx"]
        for r in pair_results
        if pose_p50 < r["posenet_mse"] <= pose_p80
    ]
    hard_pairs = [r["pair_idx"] for r in pair_results if r["posenet_mse"] > pose_p80]

    # Also tier by SegNet (the dominant term)
    seg_p50 = float(np.percentile(seg_vals, 50))
    seg_p80 = float(np.percentile(seg_vals, 80))

    seg_easy = [r["pair_idx"] for r in pair_results if r["segnet_disagree"] <= seg_p50]
    seg_hard = [r["pair_idx"] for r in pair_results if r["segnet_disagree"] > seg_p80]

    return {
        "n_pairs": n,
        "posenet": _stats(pose_vals),
        "segnet": _stats(seg_vals),
        "score": _stats(scores),
        "tiers": {
            "posenet": {
                "easy_count": len(easy_pairs),
                "easy_threshold": pose_p50,
                "medium_count": len(medium_pairs),
                "hard_count": len(hard_pairs),
                "hard_threshold": pose_p80,
                "easy_pair_indices": sorted(easy_pairs),
                "hard_pair_indices": sorted(hard_pairs),
            },
            "segnet": {
                "easy_count": len(seg_easy),
                "easy_threshold": seg_p50,
                "hard_count": len(seg_hard),
                "hard_threshold": seg_p80,
                "easy_pair_indices": sorted(seg_easy),
                "hard_pair_indices": sorted(seg_hard),
            },
        },
    }


def save_visualization(
    pair_results: list[dict[str, float]],
    stats: dict[str, Any],
    output_path: Path,
) -> None:
    """Save difficulty distribution plots.

    Creates a 2x2 panel:
      - Top-left: PoseNet MSE distribution (histogram)
      - Top-right: SegNet disagreement distribution (histogram)
      - Bottom-left: PoseNet vs SegNet scatter (per pair)
      - Bottom-right: Score contribution breakdown (sorted)

    Args:
        pair_results: list of per-pair metric dicts.
        stats: summary statistics dict.
        output_path: path to save the PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pose_vals = np.array([r["posenet_mse"] for r in pair_results])
    seg_vals = np.array([r["segnet_disagree"] for r in pair_results])
    pose_contribs = np.array([r["pose_contribution"] for r in pair_results])
    seg_contribs = np.array([r["seg_contribution"] for r in pair_results])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Per-Pair Difficulty Map: Renderer Output\n"
        f"({len(pair_results)} pairs, renderer checkpoint)",
        fontsize=13,
        fontweight="bold",
    )

    # Top-left: PoseNet MSE histogram
    ax = axes[0, 0]
    ax.hist(pose_vals, bins=40, color="#2196F3", alpha=0.8, edgecolor="white")
    ax.axvline(
        stats["posenet"]["median"],
        color="red",
        linestyle="--",
        label=f'Median: {stats["posenet"]["median"]:.4f}',
    )
    ax.axvline(
        stats["tiers"]["posenet"]["hard_threshold"],
        color="orange",
        linestyle="--",
        label=f'P80: {stats["tiers"]["posenet"]["hard_threshold"]:.4f}',
    )
    ax.set_xlabel("PoseNet MSE (per pair)")
    ax.set_ylabel("Count")
    ax.set_title("PoseNet Difficulty Distribution")
    ax.legend(fontsize=9)

    # Top-right: SegNet disagreement histogram
    ax = axes[0, 1]
    ax.hist(seg_vals, bins=40, color="#4CAF50", alpha=0.8, edgecolor="white")
    ax.axvline(
        stats["segnet"]["median"],
        color="red",
        linestyle="--",
        label=f'Median: {stats["segnet"]["median"]:.6f}',
    )
    ax.set_xlabel("SegNet Disagreement (per pair)")
    ax.set_ylabel("Count")
    ax.set_title("SegNet Difficulty Distribution")
    ax.legend(fontsize=9)

    # Bottom-left: PoseNet vs SegNet scatter
    ax = axes[1, 0]
    sc = ax.scatter(
        pose_vals,
        seg_vals,
        c=pose_contribs + seg_contribs,
        cmap="YlOrRd",
        alpha=0.6,
        s=15,
        edgecolors="none",
    )
    ax.set_xlabel("PoseNet MSE")
    ax.set_ylabel("SegNet Disagreement")
    ax.set_title("PoseNet vs SegNet (color = total score)")
    plt.colorbar(sc, ax=ax, label="Score contribution")

    # Bottom-right: Score contribution breakdown (sorted by total)
    ax = axes[1, 1]
    sorted_indices = np.argsort(pose_contribs + seg_contribs)[::-1]
    x = np.arange(len(sorted_indices))
    ax.bar(x, seg_contribs[sorted_indices], label="SegNet (100x)", color="#4CAF50", alpha=0.8)
    ax.bar(
        x,
        pose_contribs[sorted_indices],
        bottom=seg_contribs[sorted_indices],
        label="PoseNet (sqrt(10x))",
        color="#2196F3",
        alpha=0.8,
    )
    ax.set_xlabel("Pair (sorted by total score)")
    ax.set_ylabel("Score Contribution")
    ax.set_title("Per-Pair Score Breakdown")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved difficulty distribution plot to {output_path}")


def _enforce_eval_roundtrip(args) -> None:
    """CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True; only escape hatch
    is TAC_ALLOW_NO_ROUNDTRIP=1 env var with loud banner.

    2026-04-27 codex R5-4 #4: delegated to the centralised
    `tac.eval_roundtrip_gate.enforce_eval_roundtrip` helper. The previous
    per-script copies were sticky — they only printed the warning when
    `args.eval_roundtrip` was already False, so a leftover env var in a
    shell / tmux session silently relaxed later runs without acknowledgement.
    The centralised helper warns whenever the env var is present and
    records it in run provenance.
    """
    from tac.eval_roundtrip_gate import enforce_eval_roundtrip
    output_dir = getattr(args, "output_dir", None)
    enforce_eval_roundtrip(args, output_dir=output_dir, write_provenance=output_dir is not None)


def main() -> None:
    """Run the per-pair difficulty map computation."""
    args = parse_args()

    device = torch.device(args.device)
    upstream = Path(args.upstream)
    video_path = args.video or str(upstream / "videos" / "0.mkv")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # codex R5-r6 #3: gate AFTER output_dir is materialised so the
    # sidecar provenance lands in the resolved run directory.
    _enforce_eval_roundtrip(args)

    n_frames = 40 if args.smoke else 1200

    print(f"[config] device={device}, n_frames={n_frames}")
    print(f"[config] checkpoint={args.checkpoint}")
    print(f"[config] video={video_path}")
    print(f"[config] output_dir={output_dir}")
    print(f"[config] eval_roundtrip={args.eval_roundtrip}")

    # ---- Checkpoint sanity check ----
    from tac.checkpoint import verify_checkpoint_identity

    md5 = verify_checkpoint_identity(args.checkpoint)
    print(f"[checkpoint] Verified MD5 prefix: {md5}")

    t_total_start = time.monotonic()

    # ---- Step 1: Load scorers with differentiable preprocessing ----
    print("\n[1/5] Loading scorers...")
    t0 = time.monotonic()
    from tac.scorer import load_differentiable_scorers

    posenet, segnet = load_differentiable_scorers(upstream, device=str(device))
    print(f"[1/5] Scorers loaded in {time.monotonic() - t0:.1f}s")

    # ---- Step 2: Load renderer ----
    print("\n[2/5] Loading renderer...")
    t0 = time.monotonic()
    # Import from sibling module using importlib (experiments/ is not a package)
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "renderer_tto",
        Path(__file__).parent / "renderer_tto.py",
    )
    _renderer_tto = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_renderer_tto)
    load_renderer = _renderer_tto.load_renderer
    generate_renderer_frames = _renderer_tto.generate_renderer_frames

    renderer = load_renderer(args.checkpoint, device)
    print(f"[2/5] Renderer loaded in {time.monotonic() - t0:.1f}s")

    # ---- Step 3: Decode GT video ----
    print(f"\n[3/5] Decoding GT video ({n_frames} frames)...")
    t0 = time.monotonic()
    from tac.data import load_gt_video

    gt_frames = load_gt_video(video_path, n_frames=n_frames)
    n_actual = len(gt_frames)
    print(f"[3/5] Decoded {n_actual} frames in {time.monotonic() - t0:.1f}s")

    # ---- Step 4: Extract masks and generate renderer frames ----
    print("\n[4/5] Extracting masks and generating renderer frames...")
    t0 = time.monotonic()
    from tac.scorer import extract_gt_masks

    masks = extract_gt_masks(gt_frames, segnet, device)
    print(f"[4/5] Extracted {masks.shape[0]} masks in {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    # generate_renderer_frames already imported above via importlib
    rendered_frames = generate_renderer_frames(renderer, masks, device)
    print(
        f"[4/5] Generated {rendered_frames.shape[0]} rendered frames in "
        f"{time.monotonic() - t0:.1f}s"
    )

    # Free renderer from memory
    del renderer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    # ---- Step 5: Compute per-pair distortions ----
    n_pairs = n_actual // 2
    print(f"\n[5/5] Computing per-pair distortions for {n_pairs} pairs...")
    t0 = time.monotonic()
    pair_results = compute_per_pair_distortions(
        rendered_frames,
        gt_frames,
        posenet,
        segnet,
        device,
        batch_size=args.batch_size,
        eval_roundtrip=args.eval_roundtrip,
    )
    t_distort = time.monotonic() - t0
    print(f"[5/5] Computed {len(pair_results)} pair distortions in {t_distort:.1f}s")

    # ---- Compute statistics ----
    stats = compute_statistics(pair_results)

    # ---- Sort by difficulty (highest score first) ----
    pair_results_sorted = sorted(pair_results, key=lambda r: r["score"], reverse=True)

    # ---- Save results ----
    t_total = time.monotonic() - t_total_start

    output_data = {
        "experiment": "pair_difficulty_map",
        "config": {
            "checkpoint": args.checkpoint,
            "device": args.device,
            "n_frames": n_actual,
            "n_pairs": n_pairs,
            "batch_size": args.batch_size,
            "eval_roundtrip": args.eval_roundtrip,
            "smoke": args.smoke,
        },
        "statistics": stats,
        "pairs_by_difficulty": pair_results_sorted,
        "total_time_s": round(t_total, 2),
    }

    json_path = output_dir / "pair_difficulty_map.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[output] Saved per-pair metrics to {json_path}")

    # ---- Visualization ----
    try:
        viz_path = output_dir / "pair_difficulty_distribution.png"
        save_visualization(pair_results, stats, viz_path)
    except ImportError as e:
        print(f"[viz] Skipping visualization (matplotlib not available): {e}")

    # ---- Print summary ----
    print("\n" + "=" * 72)
    print("PER-PAIR DIFFICULTY MAP SUMMARY")
    print("=" * 72)
    print(f"  Total pairs: {stats['n_pairs']}")
    print(f"  Total time:  {t_total:.1f}s")
    print()
    print("  PoseNet MSE:")
    print(f"    Mean:   {stats['posenet']['mean']:.6f}")
    print(f"    Std:    {stats['posenet']['std']:.6f}")
    print(f"    Min:    {stats['posenet']['min']:.6f}")
    print(f"    Max:    {stats['posenet']['max']:.6f}")
    print(f"    Median: {stats['posenet']['median']:.6f}")
    print(f"    P10:    {stats['posenet']['p10']:.6f}")
    print(f"    P90:    {stats['posenet']['p90']:.6f}")
    print()
    print("  SegNet Disagreement:")
    print(f"    Mean:   {stats['segnet']['mean']:.6f}")
    print(f"    Std:    {stats['segnet']['std']:.6f}")
    print(f"    Min:    {stats['segnet']['min']:.6f}")
    print(f"    Max:    {stats['segnet']['max']:.6f}")
    print(f"    Median: {stats['segnet']['median']:.6f}")
    print(f"    P10:    {stats['segnet']['p10']:.6f}")
    print(f"    P90:    {stats['segnet']['p90']:.6f}")
    print()
    print("  Difficulty Tiers (by PoseNet):")
    pt = stats["tiers"]["posenet"]
    print(f"    Easy  (bottom 50%): {pt['easy_count']} pairs  "
          f"(PoseNet MSE <= {pt['easy_threshold']:.6f})")
    print(f"    Medium (50-80%):    {pt['medium_count']} pairs")
    print(f"    Hard   (top 20%):   {pt['hard_count']} pairs  "
          f"(PoseNet MSE > {pt['hard_threshold']:.6f})")
    print()
    print("  Difficulty Tiers (by SegNet):")
    st = stats["tiers"]["segnet"]
    print(f"    Easy  (bottom 50%): {st['easy_count']} pairs  "
          f"(SegNet disagree <= {st['easy_threshold']:.6f})")
    print(f"    Hard   (top 20%):   {st['hard_count']} pairs  "
          f"(SegNet disagree > {st['hard_threshold']:.6f})")
    print()
    print("  Top 10 hardest pairs (by total score):")
    for r in pair_results_sorted[:10]:
        print(f"    Pair {r['pair_idx']:>3d}: score={r['score']:.4f}  "
              f"pose={r['posenet_mse']:.6f}  seg={r['segnet_disagree']:.6f}")
    print()
    print("  Bottom 10 easiest pairs (by total score):")
    for r in pair_results_sorted[-10:]:
        print(f"    Pair {r['pair_idx']:>3d}: score={r['score']:.4f}  "
              f"pose={r['posenet_mse']:.6f}  seg={r['segnet_disagree']:.6f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
