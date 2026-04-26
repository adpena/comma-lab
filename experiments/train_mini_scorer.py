#!/usr/bin/env python3
"""Train mini-scorers: distill PoseNet+SegNet into tiny models for inflate-time TTO.

The pivotal experiment: if mini-scorers achieve >98% agreement with full scorers,
we can run TTO at inflate time using only ~50KB of scorer weights in the archive.
This bridges our unlimited-compute 0.145 proxy to a contest-compliant score.

Usage:
    # MPS smoke test (20 frames, 50 epochs):
    PYTHONPATH=src:upstream python experiments/train_mini_scorer.py \
        --checkpoint /path/to/renderer_best.pt --device mps --smoke

    # Full training on MPS:
    PYTHONPATH=src:upstream python experiments/train_mini_scorer.py \
        --checkpoint /path/to/renderer_best.pt --device mps

    # CUDA full training:
    PYTHONPATH=src:upstream python experiments/train_mini_scorer.py \
        --checkpoint /path/to/renderer_best.pt --device cuda

Outputs:
    experiments/results/mini_scorer/mini_segnet.bin    (~25KB INT8)
    experiments/results/mini_scorer/mini_posenet.bin   (~12KB INT8)
    experiments/results/mini_scorer/metrics.json       (fidelity metrics)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train mini-scorers via knowledge distillation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to renderer .pt checkpoint")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=1200, help="Number of frames to generate")
    p.add_argument("--upstream", type=str, default="upstream/", help="Path to upstream repo")
    p.add_argument("--video", type=str, default=None, help="Path to GT video (for masks/poses)")
    p.add_argument("--output-dir", type=str, default="experiments/results/mini_scorer",
                   help="Output directory")
    p.add_argument("--seg-epochs", type=int, default=200, help="MiniSegNet training epochs")
    p.add_argument("--pose-epochs", type=int, default=300, help="MiniPoseNet training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    p.add_argument("--hidden", type=int, default=16, help="Hidden channels for mini models")
    p.add_argument("--no-quantize", action="store_true", help="Skip INT8 quantization")
    p.add_argument("--smoke", action="store_true", help="Smoke test: 20 frames, 50 epochs")
    return p.parse_args()


def load_renderer(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load the trained renderer with content-based format dispatch.

    Defers to experiments.precompute_gradient_corrections.load_renderer so all
    consumers share one canonical content-detecting loader. Suffix-blind
    torch.load on FP4 .bin files was the 2026-04-26 DEN-V2 bug surface.
    """
    from experiments.precompute_gradient_corrections import (
        load_renderer as _canonical_load_renderer,
    )
    renderer = _canonical_load_renderer(checkpoint_path, device)
    logger.info("Loaded renderer from %s", checkpoint_path)
    return renderer


def generate_renderer_frames(
    renderer: torch.nn.Module,
    masks: torch.Tensor,
    poses: torch.Tensor,
    device: torch.device,
    batch_size: int = 20,
) -> torch.Tensor:
    """Generate all frames from renderer.

    Handles both model types:
        - AsymmetricPairGenerator: processes consecutive mask PAIRS, returns (B, 2, H, W, 3)
        - DPSIMSRenderer/MaskRenderer: processes individual masks, returns (B, 3, H, W)

    Args:
        renderer: trained renderer model.
        masks: (N, H, W) long — input masks.
        poses: (P, 6) float — GT poses for FiLM conditioning.
        device: computation device.
        batch_size: pairs per batch (for asymmetric) or frames per batch (for single-frame).

    Returns:
        (N, 3, H, W) float tensor in [0, 255].
    """
    N = masks.shape[0]
    all_frames = []

    # Detect model type by checking for motion sub-module (AsymmetricPairGenerator signature)
    is_asymmetric = (
        hasattr(renderer, "renderer") and hasattr(renderer, "motion")
        and hasattr(renderer.motion, "output_channels")
    )

    with torch.no_grad():
        if is_asymmetric:
            # AsymmetricPairGenerator: forward(mask_t, mask_t1, pose=...) -> (B, 2, H, W, 3)
            P = N // 2
            for start in range(0, P, batch_size):
                end = min(start + batch_size, P)
                masks_t = masks[2 * start:2 * end:2].to(device)
                masks_t1 = masks[2 * start + 1:2 * end + 1:2].to(device)

                # Pass pose conditioning if available and model supports it
                batch_pose = None
                if poses is not None and hasattr(renderer, "pose_dim") and renderer.pose_dim > 0:
                    if end <= poses.shape[0]:
                        batch_pose = poses[start:end].to(device)

                if batch_pose is not None:
                    pairs = renderer(masks_t, masks_t1, pose=batch_pose)
                else:
                    pairs = renderer(masks_t, masks_t1)  # (B, 2, H, W, 3)

                # Convert pair output (B, 2, H, W, 3) to interleaved (2B, 3, H, W)
                B = pairs.shape[0]
                f0 = pairs[:, 0].permute(0, 3, 1, 2)  # (B, 3, H, W)
                f1 = pairs[:, 1].permute(0, 3, 1, 2)  # (B, 3, H, W)
                # Interleave: [f0_0, f1_0, f0_1, f1_1, ...]
                interleaved = torch.stack([f0, f1], dim=1).reshape(2 * B, 3, *f0.shape[2:])
                all_frames.append(interleaved.cpu())

            # Handle odd trailing mask via the renderer sub-module directly
            if N % 2 != 0:
                last_mask = masks[N - 1:N].to(device)
                frame = renderer.renderer(last_mask)  # (1, 3, H, W)
                all_frames.append(frame.cpu())
        else:
            # Single-frame renderer (DPSIMSRenderer / MaskRenderer)
            for i in range(0, N, batch_size):
                batch_masks = masks[i:i + batch_size].to(device)
                frames = renderer(batch_masks)  # (B, 3, H, W)

                # Ensure (B, 3, H, W) format
                if frames.ndim == 4 and frames.shape[1] == 3:
                    pass  # already correct
                elif frames.ndim == 4 and frames.shape[-1] == 3:
                    frames = frames.permute(0, 3, 1, 2)
                else:
                    raise ValueError(f"Unexpected renderer output shape: {frames.shape}")

                all_frames.append(frames.cpu())

    return torch.cat(all_frames, dim=0)


def main() -> None:
    args = parse_args()
    t0 = time.time()

    if args.smoke:
        args.n_frames = 20
        args.seg_epochs = 50
        args.pose_epochs = 50
        logger.info("SMOKE TEST MODE: 20 frames, 50 epochs each")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load scorers ─────────────────────────────────────────
    logger.info("Loading full scorers...")
    from tac.scorer import load_differentiable_scorers
    posenet, segnet = load_differentiable_scorers(args.upstream, device=device)
    logger.info("Scorers loaded on %s", device)

    # ── Step 2: Load renderer and generate frames ────────────────────
    logger.info("Loading renderer from %s...", args.checkpoint)
    renderer = load_renderer(args.checkpoint, device)

    # Load GT frames, then extract masks and poses
    from tac.data import load_gt_video
    from tac.scorer import extract_gt_masks, extract_gt_pose_targets

    video_path = args.video or str(Path(args.upstream) / "videos" / "0.mkv")
    logger.info("Loading GT frames from %s...", video_path)
    gt_frames = load_gt_video(video_path, n_frames=args.n_frames)
    logger.info("Loaded %d GT frames", len(gt_frames))

    logger.info("Extracting GT masks...")
    gt_masks = extract_gt_masks(gt_frames, segnet, device=device)
    logger.info("Extracting GT poses...")
    gt_poses = extract_gt_pose_targets(gt_frames, posenet, device=device)

    logger.info("Generating %d renderer frames...", args.n_frames)
    frames_chw = generate_renderer_frames(renderer, gt_masks, gt_poses, device, batch_size=20)
    logger.info("Generated frames: shape=%s", frames_chw.shape)

    # Free renderer memory
    del renderer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Step 3: Train MiniSegNet ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING MiniSegNet (%d epochs, hidden=%d)", args.seg_epochs, args.hidden)
    logger.info("=" * 60)

    from tac.mini_scorer import (
        train_mini_segnet,
        train_mini_posenet,
        validate_mini_segnet_fidelity,
        validate_mini_posenet_fidelity,
        save_mini_scorers,
    )

    mini_seg, seg_metrics = train_mini_segnet(
        frames_chw,
        segnet,
        epochs=args.seg_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden=args.hidden,
        device=device,
        verbose=True,
    )

    # Validate SegNet fidelity
    logger.info("Validating MiniSegNet fidelity...")
    seg_fidelity = validate_mini_segnet_fidelity(mini_seg, frames_chw, segnet, device=device)
    logger.info(
        "MiniSegNet fidelity: %.4f%% agreement (%d/%d pixels)",
        seg_fidelity["agreement"] * 100,
        seg_fidelity["correct_pixels"],
        seg_fidelity["total_pixels"],
    )
    for cls, acc in seg_fidelity["per_class_agreement"].items():
        logger.info("  class %d: %.4f%%", cls, acc * 100)

    # ── Step 4: Train MiniPoseNet ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING MiniPoseNet (%d epochs, hidden=%d)", args.pose_epochs, args.hidden)
    logger.info("=" * 60)

    mini_pose, pose_metrics = train_mini_posenet(
        frames_chw,
        posenet,
        epochs=args.pose_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden=args.hidden,
        device=device,
        verbose=True,
    )

    # Validate PoseNet fidelity
    logger.info("Validating MiniPoseNet fidelity...")
    pose_fidelity = validate_mini_posenet_fidelity(mini_pose, frames_chw, posenet, device=device)
    logger.info(
        "MiniPoseNet fidelity: R2=%.4f, MSE=%.6f (%d pairs)",
        pose_fidelity["r_squared"],
        pose_fidelity["mse"],
        pose_fidelity["num_pairs"],
    )
    for d, r2 in enumerate(pose_fidelity["per_dim_r2"]):
        dim_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
        logger.info("  %s: R2=%.4f", dim_names[d], r2)

    # ── Step 5: Save mini-scorers ────────────────────────────────────
    logger.info("Saving mini-scorers to %s...", output_dir)
    sizes = save_mini_scorers(
        mini_seg, mini_pose, output_dir,
        quantize_int8=not args.no_quantize,
    )

    # ── Step 6: Compile results ──────────────────────────────────────
    elapsed = time.time() - t0
    results = {
        "experiment": "train_mini_scorer",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": vars(args),
        "elapsed_seconds": elapsed,
        "segnet": {
            "param_count": seg_metrics["param_count"],
            "final_val_accuracy": seg_metrics["final_val_acc"],
            "fidelity": seg_fidelity,
        },
        "posenet": {
            "param_count": pose_metrics["param_count"],
            "final_val_r2": pose_metrics["final_val_r2"],
            "fidelity": pose_fidelity,
        },
        "archive_sizes": sizes,
        "rate_cost_estimate": 25 * sizes["total_bytes"] / 37545489,
        "verdict": {
            "segnet_pass": seg_fidelity["agreement"] >= 0.98,
            "posenet_pass": pose_fidelity["r_squared"] >= 0.95,
            "both_pass": (
                seg_fidelity["agreement"] >= 0.98
                and pose_fidelity["r_squared"] >= 0.95
            ),
        },
    }

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MINI-SCORER TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("  Time: %.1f seconds", elapsed)
    logger.info("  MiniSegNet: %d params, %.2f%% agreement",
                seg_metrics["param_count"], seg_fidelity["agreement"] * 100)
    logger.info("  MiniPoseNet: %d params, R2=%.4f",
                pose_metrics["param_count"], pose_fidelity["r_squared"])
    logger.info("  Archive cost: %d bytes (rate += %.4f)",
                sizes["total_bytes"], results["rate_cost_estimate"])
    logger.info("  VERDICT: %s",
                "PASS - ready for mini-TTO" if results["verdict"]["both_pass"]
                else "FAIL - fidelity too low")

    if not results["verdict"]["both_pass"]:
        if not results["verdict"]["segnet_pass"]:
            logger.warning("  SegNet agreement %.2f%% < 98%% threshold",
                           seg_fidelity["agreement"] * 100)
        if not results["verdict"]["posenet_pass"]:
            logger.warning("  PoseNet R2 %.4f < 0.95 threshold",
                           pose_fidelity["r_squared"])
        sys.exit(1)


if __name__ == "__main__":
    main()
