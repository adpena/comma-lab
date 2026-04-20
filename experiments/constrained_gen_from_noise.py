#!/usr/bin/env python3
"""Constrained generation from noise using mini-scorers.

Fourth lane: no renderer needed. Directly optimize pixel values from a noise
seed against mini-scorer gradients. The mini-scorers (50KB total) are stored
in the archive, making this contest-compliant.

Key differences from renderer_tto.py:
  - No renderer checkpoint needed (no neural network generating frames)
  - Starts from class-mean-colored noise instead of renderer output
  - Can use either mini-scorers (contest-compliant) or full scorers (reference)
  - Archive is SMALLER (138KB vs 184KB), so rate is BETTER
  - The entire "generation model" is gradient descent itself

Score projection: 0.15-0.25 (GPU Eureka projected 0.135 with full scorers)

Architecture of the archive:
  - mini_segnet.bin: ~25KB (FP16 weights)
  - mini_posenet.bin: ~25KB (FP16 weights)
  - poses.pt: ~8.7KB (GT pose targets, 600 pairs x 6 dims x fp16)
  - masks.mkv: ~79KB (GT mask targets, compressed)
  - seed: 64 bytes (deterministic noise initialization)
  - Total: ~138KB
  - Rate: 25 * 138000 / 37545489 = 0.092

Usage:
    # Smoke test (10 frames, 50 steps, MPS):
    PYTHONPATH=src:upstream .venv/bin/python experiments/constrained_gen_from_noise.py \
        --mode mini --mini-scorer-dir experiments/results/mini_scorer \
        --device mps --smoke

    # Full comparison (mini vs full scorers):
    PYTHONPATH=src:upstream .venv/bin/python experiments/constrained_gen_from_noise.py \
        --mode both --mini-scorer-dir experiments/results/mini_scorer \
        --device cuda --n-frames 1200 --steps 1000

    # Full scorers only (reference, not contest-compliant):
    PYTHONPATH=src:upstream .venv/bin/python experiments/constrained_gen_from_noise.py \
        --mode full --device cuda --n-frames 1200 --steps 1000

    # Hybrid: renderer warm-start + mini-scorer TTO:
    PYTHONPATH=src:upstream .venv/bin/python experiments/constrained_gen_from_noise.py \
        --mode mini --mini-scorer-dir experiments/results/mini_scorer \
        --checkpoint /path/to/renderer_best.pt --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = REPO_ROOT / "upstream"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Constrained generation from noise — fourth lane experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Mode
    p.add_argument("--mode", type=str, default="mini",
                   choices=["mini", "full", "both"],
                   help="'mini' = mini-scorers only (contest-compliant), "
                        "'full' = full scorers (reference), "
                        "'both' = run both and compare gradient directions.")

    # Mini-scorer config
    p.add_argument("--mini-scorer-dir", type=str, default=None,
                   help="Directory with mini_segnet.bin + mini_posenet.bin")

    # Optional renderer warm-start (hybrid mode)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Renderer checkpoint for warm-start (hybrid). "
                        "If None, starts from class-mean noise (pure from-noise).")

    # Optimization params
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=1200,
                   help="Number of frames to generate")
    p.add_argument("--steps", type=int, default=1000,
                   help="Gradient descent steps")
    p.add_argument("--lr", type=float, default=0.02,
                   help="Adam learning rate (higher than TTO since starting from noise)")
    p.add_argument("--seg-weight", type=float, default=100.0,
                   help="SegNet loss weight")
    p.add_argument("--pose-weight", type=float, default=10.0,
                   help="PoseNet loss weight")
    p.add_argument("--compress-weight", type=float, default=1.0,
                   help="Compressibility loss weight")
    p.add_argument("--batch-pairs", type=int, default=20,
                   help="Pairs per optimization batch (memory vs speed)")
    p.add_argument("--noise-seed", type=int, default=42,
                   help="Deterministic noise seed for initialization")

    # Advanced options
    p.add_argument("--segnet-loss-mode", type=str, default="hinge",
                   choices=["xent", "hinge"],
                   help="SegNet loss function (hinge is 24-49%% better)")
    p.add_argument("--hinge-margin", type=float, default=0.5)
    p.add_argument("--lr-schedule", type=str, default="cosine",
                   choices=["constant", "cosine"])
    p.add_argument("--use-null-space", action="store_true",
                   help="Apply YUV null space projection for free SegNet improvement")
    p.add_argument("--null-space-every", type=int, default=10)
    p.add_argument("--null-space-step", type=float, default=0.5)
    p.add_argument("--eval-roundtrip", action="store_true",
                   help="Simulate scorer resize roundtrip in loss")
    p.add_argument("--early-stop-patience", type=int, default=200,
                   help="Early stop on PoseNet plateau (higher for from-noise)")
    p.add_argument("--antialias-weight", type=float, default=0.2,
                   help="2x2 block variance penalty (PoseNet-invisible noise)")

    # I/O
    p.add_argument("--upstream", type=str, default="upstream/")
    p.add_argument("--video", type=str, default=None,
                   help="GT video path (default: upstream/videos/0.mkv)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: timestamped)")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 10 frames, 50 steps")

    return p.parse_args()


def load_gt_targets(
    video_path: Path,
    n_frames: int,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Load GT video and extract mask + pose targets.

    Returns:
        (masks, pose_targets, gt_frames_list)
        - masks: (N, 384, 512) long
        - pose_targets: (P, 6) float where P = N//2
        - gt_frames_list: list of (H, W, 3) uint8 tensors
    """
    from tac.scorer import extract_gt_masks, extract_gt_pose_targets

    logger.info("Loading GT video: %s (first %d frames)", video_path, n_frames)
    import av

    container = av.open(str(video_path))
    gt_frames = []
    for frame in container.decode(video=0):
        gt_frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
        if len(gt_frames) >= n_frames:
            break
    container.close()
    logger.info("  Loaded %d GT frames, shape %s", len(gt_frames), gt_frames[0].shape)

    # Ensure even number for pairs
    if len(gt_frames) % 2 != 0:
        gt_frames = gt_frames[:-1]

    n_frames_actual = len(gt_frames)

    # Extract masks using full SegNet
    logger.info("Extracting SegNet masks...")
    masks = extract_gt_masks(gt_frames, segnet, device=device)
    logger.info("  Masks: %s, classes: %s", masks.shape, masks.unique().tolist())

    # Extract pose targets using full PoseNet
    logger.info("Extracting PoseNet targets...")
    pose_targets = extract_gt_pose_targets(gt_frames, posenet, device=device)
    logger.info("  Pose targets: %s", pose_targets.shape)

    return masks, pose_targets, gt_frames


def run_full_scorer_optimization(
    masks: torch.Tensor,
    pose_targets: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    init_frames: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Run constrained gen using FULL scorers (reference, not contest-compliant).

    Uses coupled_trajectory_optimize from tac.constrained_gen directly.
    """
    from tac.constrained_gen import coupled_trajectory_optimize

    logger.info("Running FULL scorer optimization (%d steps, lr=%.4f)", args.steps, args.lr)
    t0 = time.time()

    frames = coupled_trajectory_optimize(
        masks=masks,
        expected_pose=pose_targets,
        posenet=posenet,
        segnet=segnet,
        num_steps=args.steps,
        lr=args.lr,
        seg_weight=args.seg_weight,
        pose_weight=args.pose_weight,
        compress_weight=args.compress_weight,
        noise_seed=args.noise_seed,
        device=str(device),
        log_every=50,
        init_frames=init_frames,
        early_stop_patience=args.early_stop_patience,
        segnet_loss_mode=args.segnet_loss_mode,
        hinge_margin=args.hinge_margin,
        lr_schedule=args.lr_schedule,
        eval_roundtrip=args.eval_roundtrip,
        use_null_space=args.use_null_space,
        null_space_step=args.null_space_step,
        null_space_every=args.null_space_every,
        antialias_weight=args.antialias_weight,
    )

    elapsed = time.time() - t0
    logger.info("Full scorer optimization done in %.1fs", elapsed)

    metrics = {"elapsed_s": elapsed, "mode": "full"}
    return frames, metrics


def run_mini_scorer_optimization(
    masks: torch.Tensor,
    pose_targets: torch.Tensor,
    mini_seg: torch.nn.Module,
    mini_pose: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    init_frames: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Run constrained gen using MINI-scorers (contest-compliant, 50KB total).

    Uses MiniScorerTTO.optimize for batched optimization.
    """
    from tac.constrained_gen import generate_initial_frames
    from tac.mini_scorer import MiniScorerTTO, MINI_SEG_H, MINI_SEG_W

    logger.info("Running MINI scorer optimization (%d steps, lr=%.4f)", args.steps, args.lr)
    t0 = time.time()

    N = masks.shape[0]

    # Prepare targets at mini-scorer resolution
    # Masks: downsample from (N, 384, 512) to (N, 96, 128)
    masks_mini = F.interpolate(
        masks.float().unsqueeze(1),
        size=(MINI_SEG_H, MINI_SEG_W),
        mode="nearest",
    ).squeeze(1).long()

    # Initialize frames
    if init_frames is not None:
        frames_init = init_frames.float()
        logger.info("  Warm-starting from provided frames")
    else:
        frames_init = generate_initial_frames(
            masks, args.noise_seed, device="cpu",
        )
        logger.info("  Starting from class-mean noise (seed=%d)", args.noise_seed)

    # Run mini-scorer TTO
    mini_tto = MiniScorerTTO(mini_seg, mini_pose, device=device)
    frames = mini_tto.optimize(
        init_frames=frames_init,
        target_masks=masks_mini,
        target_poses=pose_targets,
        num_steps=args.steps,
        lr=args.lr,
        seg_weight=args.seg_weight,
        pose_weight=args.pose_weight,
        log_every=50,
        batch_pairs=args.batch_pairs,
    )

    elapsed = time.time() - t0
    logger.info("Mini scorer optimization done in %.1fs", elapsed)

    metrics = {"elapsed_s": elapsed, "mode": "mini"}
    return frames, metrics


def score_frames(
    frames: torch.Tensor,
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Score generated frames against GT using full scorers.

    Returns dict with posenet_dist, segnet_dist, proxy_score.
    """
    from tac.scorer import compute_proxy_score

    result = compute_proxy_score(
        frames=frames,
        gt_frames=gt_frames,
        posenet=posenet,
        segnet=segnet,
        device=device,
    )
    return result


def compare_gradient_directions(
    frames: torch.Tensor,
    masks: torch.Tensor,
    pose_targets: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    mini_seg: torch.nn.Module,
    mini_pose: torch.nn.Module,
    device: torch.device,
    n_samples: int = 5,
) -> dict:
    """Compare gradient directions between mini and full scorers.

    Measures cosine similarity of gradients at sample frames to quantify
    how well mini-scorer optimization aligns with full-scorer optima.
    """
    from tac.constrained_gen import (
        compute_segnet_constraint_loss,
        compute_posenet_constraint_loss,
    )
    from tac.mini_scorer import MiniScorerTTO, MINI_SEG_H, MINI_SEG_W

    logger.info("Comparing gradient directions (mini vs full scorers)...")

    # Take a small sample for gradient comparison
    sample_n = min(n_samples * 2, frames.shape[0])  # pairs
    sample_frames = frames[:sample_n].to(device).float().detach().clone()
    sample_frames.requires_grad_(True)
    sample_masks = masks[:sample_n].to(device)
    sample_poses = pose_targets[:sample_n // 2].to(device)

    # Full scorer gradients
    seg_loss_full = compute_segnet_constraint_loss(
        sample_frames, sample_masks, segnet, loss_mode="hinge",
    )
    pose_loss_full = compute_posenet_constraint_loss(
        sample_frames, sample_poses, posenet,
    )
    total_full = 100.0 * seg_loss_full + 10.0 * pose_loss_full
    grad_full = torch.autograd.grad(total_full, sample_frames, retain_graph=False)[0]

    # Mini scorer gradients
    sample_frames2 = sample_frames.detach().clone().requires_grad_(True)
    masks_mini = F.interpolate(
        sample_masks.float().unsqueeze(1),
        size=(MINI_SEG_H, MINI_SEG_W),
        mode="nearest",
    ).squeeze(1).long()

    mini_tto = MiniScorerTTO(mini_seg, mini_pose, device=device)
    seg_loss_mini = mini_tto.compute_seg_loss(sample_frames2, masks_mini)
    pose_loss_mini = mini_tto.compute_pose_loss(sample_frames2, sample_poses)
    total_mini = 100.0 * seg_loss_mini + 10.0 * pose_loss_mini
    grad_mini = torch.autograd.grad(total_mini, sample_frames2, retain_graph=False)[0]

    # Cosine similarity
    grad_full_flat = grad_full.reshape(-1)
    grad_mini_flat = grad_mini.reshape(-1)

    cos_sim = F.cosine_similarity(
        grad_full_flat.unsqueeze(0),
        grad_mini_flat.unsqueeze(0),
    ).item()

    # Per-pixel magnitude correlation
    mag_full = grad_full.abs().mean(dim=-1)  # (N, H, W)
    mag_mini = grad_mini.abs().mean(dim=-1)

    # Flatten for correlation
    f_flat = mag_full.reshape(-1).cpu()
    m_flat = mag_mini.reshape(-1).cpu()
    pearson_r = torch.corrcoef(torch.stack([f_flat, m_flat]))[0, 1].item()

    results = {
        "cosine_similarity": cos_sim,
        "magnitude_correlation": pearson_r,
        "full_grad_norm": grad_full_flat.norm().item(),
        "mini_grad_norm": grad_mini_flat.norm().item(),
        "full_seg_loss": seg_loss_full.item(),
        "mini_seg_loss": seg_loss_mini.item(),
        "full_pose_loss": pose_loss_full.item(),
        "mini_pose_loss": pose_loss_mini.item(),
    }

    logger.info("  Gradient cosine similarity: %.4f", cos_sim)
    logger.info("  Magnitude correlation: %.4f", pearson_r)
    logger.info("  Full grad norm: %.4f, Mini grad norm: %.4f",
                results["full_grad_norm"], results["mini_grad_norm"])

    return results


def main() -> None:
    args = parse_args()
    t_start = time.time()

    if args.smoke:
        args.n_frames = 10
        args.steps = 50
        args.batch_pairs = 5
        logger.info("SMOKE TEST: 10 frames, 50 steps, 5 batch_pairs")

    device = torch.device(args.device)

    # Output directory
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        args.output_dir = f"experiments/results/constrained_gen_noise/{ts}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Video path
    video_path = Path(args.video) if args.video else Path(args.upstream) / "videos" / "0.mkv"

    # Load full scorers (always needed for scoring; also for full-mode optimization)
    logger.info("Loading full scorers...")
    from tac.scorer import load_differentiable_scorers
    posenet, segnet = load_differentiable_scorers(args.upstream, device=device)

    # Load GT targets
    masks, pose_targets, gt_frames = load_gt_targets(
        video_path, args.n_frames, posenet, segnet, device,
    )

    # Optionally load renderer for warm-start
    init_frames = None
    if args.checkpoint:
        logger.info("Loading renderer for warm-start...")
        sys.path.insert(0, str(REPO_ROOT / "experiments"))
        from renderer_tto import load_renderer, generate_renderer_frames
        renderer = load_renderer(args.checkpoint, device)
        init_frames = generate_renderer_frames(renderer, masks, device)
        logger.info("  Renderer frames: %s", init_frames.shape)
        del renderer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Load mini-scorers if needed
    mini_seg, mini_pose = None, None
    if args.mode in ("mini", "both"):
        if args.mini_scorer_dir is None:
            logger.error("--mini-scorer-dir required for mode=%s", args.mode)
            sys.exit(1)
        from tac.mini_scorer import load_mini_scorers
        mini_seg, mini_pose = load_mini_scorers(args.mini_scorer_dir, device=device)
        seg_params = sum(p.numel() for p in mini_seg.parameters())
        pose_params = sum(p.numel() for p in mini_pose.parameters())
        logger.info("  Mini-scorers: seg=%d params, pose=%d params", seg_params, pose_params)

    results = {}

    # Run full scorer optimization
    if args.mode in ("full", "both"):
        frames_full, metrics_full = run_full_scorer_optimization(
            masks, pose_targets, posenet, segnet, device, args, init_frames,
        )
        score_full = score_frames(frames_full, gt_frames, posenet, segnet, device)
        results["full_scorer"] = {**metrics_full, **score_full}
        logger.info("FULL SCORER: proxy=%.4f (pose=%.6f, seg=%.6f)",
                    score_full["proxy_score"], score_full["posenet_dist"],
                    score_full["segnet_dist"])
        # Save frames
        torch.save(frames_full.cpu(), str(output_dir / "frames_full.pt"))

    # Run mini scorer optimization
    if args.mode in ("mini", "both"):
        frames_mini, metrics_mini = run_mini_scorer_optimization(
            masks, pose_targets, mini_seg, mini_pose, device, args, init_frames,
        )
        # Score with FULL scorers (authoritative measure)
        score_mini = score_frames(frames_mini, gt_frames, posenet, segnet, device)
        results["mini_scorer"] = {**metrics_mini, **score_mini}
        logger.info("MINI SCORER: proxy=%.4f (pose=%.6f, seg=%.6f)",
                    score_mini["proxy_score"], score_mini["posenet_dist"],
                    score_mini["segnet_dist"])
        # Save frames
        torch.save(frames_mini.cpu(), str(output_dir / "frames_mini.pt"))

    # Compare gradient directions
    if args.mode == "both" and mini_seg is not None:
        # Use the mini-scorer optimized frames as the comparison point
        grad_comparison = compare_gradient_directions(
            frames_mini, masks, pose_targets,
            posenet, segnet, mini_seg, mini_pose, device,
        )
        results["gradient_comparison"] = grad_comparison

    # Estimate archive size and rate
    archive_size_estimate = 138_000  # 25KB + 25KB + 8.7KB + 79KB + 64B
    rate = archive_size_estimate / 37_545_489
    rate_contribution = 25 * rate
    results["archive_estimate"] = {
        "total_bytes": archive_size_estimate,
        "rate": rate,
        "rate_score_contribution": rate_contribution,
    }
    logger.info("Archive estimate: %d bytes, rate=%.4f, rate_score=%.3f",
                archive_size_estimate, rate, rate_contribution)

    # Total projected score
    if "mini_scorer" in results:
        total = (100 * results["mini_scorer"]["segnet_dist"]
                 + (10 * results["mini_scorer"]["posenet_dist"]) ** 0.5
                 + rate_contribution)
        results["mini_scorer"]["projected_auth_score"] = total
        logger.info("PROJECTED AUTH (mini): %.4f", total)
    if "full_scorer" in results:
        total = (100 * results["full_scorer"]["segnet_dist"]
                 + (10 * results["full_scorer"]["posenet_dist"]) ** 0.5
                 + rate_contribution)
        results["full_scorer"]["projected_auth_score"] = total
        logger.info("PROJECTED AUTH (full): %.4f", total)

    # Save results
    results["config"] = {
        "mode": args.mode,
        "n_frames": args.n_frames,
        "steps": args.steps,
        "lr": args.lr,
        "seg_weight": args.seg_weight,
        "pose_weight": args.pose_weight,
        "compress_weight": args.compress_weight,
        "noise_seed": args.noise_seed,
        "segnet_loss_mode": args.segnet_loss_mode,
        "lr_schedule": args.lr_schedule,
        "warm_start": args.checkpoint is not None,
        "device": args.device,
    }
    results["total_elapsed_s"] = time.time() - t_start

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("CONSTRAINED GEN FROM NOISE — RESULTS")
    logger.info("=" * 60)
    if "full_scorer" in results:
        logger.info("  Full scorers:  proxy=%.4f  projected_auth=%.4f  time=%.1fs",
                    results["full_scorer"]["proxy_score"],
                    results["full_scorer"]["projected_auth_score"],
                    results["full_scorer"]["elapsed_s"])
    if "mini_scorer" in results:
        logger.info("  Mini scorers:  proxy=%.4f  projected_auth=%.4f  time=%.1fs",
                    results["mini_scorer"]["proxy_score"],
                    results["mini_scorer"]["projected_auth_score"],
                    results["mini_scorer"]["elapsed_s"])
    if "gradient_comparison" in results:
        gc = results["gradient_comparison"]
        logger.info("  Gradient cosine sim: %.4f  Magnitude correlation: %.4f",
                    gc["cosine_similarity"], gc["magnitude_correlation"])
    logger.info("  Archive: %dKB  Rate contribution: %.3f",
                archive_size_estimate // 1024, rate_contribution)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
