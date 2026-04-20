#!/usr/bin/env python3
"""Mini-TTO inflate: proof of concept for inflate-time TTO with mini-scorers.

Validates the full pipeline:
    1. Load renderer + mini-scorers from archive
    2. Generate frames via renderer
    3. Run TTO using mini-scorers (NOT full scorers) -- 100 steps
    4. Write .raw output
    5. Score with FULL upstream evaluate.py

This answers the critical question: does TTO against mini-scorers improve
the REAL (full-scorer) score? If yes, mini-scorer distillation is viable
for contest-compliant submission.

Usage:
    # Smoke test on MPS:
    PYTHONPATH=src:upstream python experiments/mini_tto_inflate.py \
        --checkpoint /path/to/renderer_best.pt \
        --mini-scorer-dir experiments/results/mini_scorer \
        --device mps --smoke

    # Full validation:
    PYTHONPATH=src:upstream python experiments/mini_tto_inflate.py \
        --checkpoint /path/to/renderer_best.pt \
        --mini-scorer-dir experiments/results/mini_scorer \
        --device cuda --n-frames 1200

Outputs:
    experiments/results/mini_tto/frames_before.pt    -- renderer output
    experiments/results/mini_tto/frames_after.pt     -- after mini-TTO
    experiments/results/mini_tto/results.json        -- score comparison
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mini-TTO inflate proof of concept",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Renderer checkpoint")
    p.add_argument("--mini-scorer-dir", type=str, required=True,
                   help="Directory with mini_segnet.bin + mini_posenet.bin")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=1200, help="Number of frames")
    p.add_argument("--tto-steps", type=int, default=100, help="Mini-TTO optimization steps")
    p.add_argument("--tto-lr", type=float, default=0.01, help="Mini-TTO learning rate")
    p.add_argument("--seg-weight", type=float, default=100.0, help="SegNet loss weight")
    p.add_argument("--pose-weight", type=float, default=10.0, help="PoseNet loss weight")
    p.add_argument("--upstream", type=str, default="upstream/", help="Upstream dir")
    p.add_argument("--video", type=str, default=None, help="GT video path")
    p.add_argument("--output-dir", type=str, default="experiments/results/mini_tto")
    p.add_argument("--smoke", action="store_true", help="Smoke: 20 frames, 20 steps")
    p.add_argument("--pair-weights-path", type=str, default=None,
                   help="Path to pair difficulty weights (from pair_difficulty_map)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    if args.smoke:
        args.n_frames = 20
        args.tto_steps = 20
        logger.info("SMOKE TEST: 20 frames, 20 TTO steps")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load full scorers (for scoring, not for TTO) ─────────────────
    logger.info("Loading full scorers for evaluation...")
    from tac.scorer import load_differentiable_scorers, extract_gt_masks, extract_gt_pose_targets
    posenet, segnet = load_differentiable_scorers(args.upstream, device=device)

    # ── Load mini-scorers (for TTO) ─────────────────────────────────
    logger.info("Loading mini-scorers from %s...", args.mini_scorer_dir)
    from tac.mini_scorer import load_mini_scorers, MiniScorerTTO, MINI_SEG_H, MINI_SEG_W
    mini_seg, mini_pose = load_mini_scorers(args.mini_scorer_dir, device=device)
    mini_tto = MiniScorerTTO(mini_seg, mini_pose, device=device)
    logger.info("Mini-scorers loaded (seg params=%d, pose params=%d)",
                sum(p.numel() for p in mini_seg.parameters()),
                sum(p.numel() for p in mini_pose.parameters()))

    # ── Generate renderer frames ─────────────────────────────────────
    logger.info("Loading renderer and generating frames...")
    from experiments.train_mini_scorer import load_renderer, generate_renderer_frames

    renderer = load_renderer(args.checkpoint, device)
    video_path = args.video or str(Path(args.upstream) / "videos" / "0.mkv")
    gt_masks = extract_gt_masks(video_path, segnet, device=device, n_frames=args.n_frames)
    gt_poses = extract_gt_pose_targets(video_path, posenet, device=device, n_frames=args.n_frames)

    frames_chw = generate_renderer_frames(renderer, gt_masks, gt_poses, device, batch_size=20)
    del renderer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("Renderer frames: %s", frames_chw.shape)

    # ── Score BEFORE mini-TTO ────────────────────────────────────────
    logger.info("Scoring renderer output (before mini-TTO)...")
    from tac.scorer import compute_proxy_score
    frames_hwc_before = frames_chw.permute(0, 2, 3, 1).contiguous()  # (N, H, W, 3)
    score_before = compute_proxy_score(frames_hwc_before, posenet, segnet, device=device)
    logger.info("Score BEFORE mini-TTO: %.4f (seg=%.4f, pose=%.6f)",
                score_before["total"], score_before.get("segnet", 0), score_before.get("posenet", 0))

    # ── Prepare mini-TTO targets ─────────────────────────────────────
    # Target masks at mini resolution (from full SegNet on renderer frames)
    logger.info("Computing mini-TTO targets...")
    with torch.no_grad():
        # Get full SegNet labels, downsample to mini res
        target_masks_list = []
        for i in range(0, args.n_frames, 32):
            batch = frames_chw[i:i+32].to(device)
            if batch.shape[-2] != 384 or batch.shape[-1] != 512:
                batch_resized = F.interpolate(batch, size=(384, 512), mode="bilinear", align_corners=False)
            else:
                batch_resized = batch
            logits = segnet(batch_resized)
            labels = logits.argmax(dim=1)  # (B, 384, 512)
            labels_mini = F.interpolate(
                labels.float().unsqueeze(1),
                size=(MINI_SEG_H, MINI_SEG_W),
                mode="nearest",
            ).squeeze(1).long()
            target_masks_list.append(labels_mini.cpu())
        target_masks = torch.cat(target_masks_list, dim=0)

    # Target poses from full PoseNet
    target_poses = gt_poses  # (P, 6)

    # Load pair weights if available
    pair_weights = None
    if args.pair_weights_path:
        pw_data = torch.load(args.pair_weights_path, map_location="cpu", weights_only=True)
        pair_weights = pw_data if isinstance(pw_data, torch.Tensor) else pw_data.get("weights")
        logger.info("Loaded pair weights: %s", pair_weights.shape)

    # ── Run mini-TTO ─────────────────────────────────────────────────
    logger.info("Running mini-TTO (%d steps, lr=%.4f)...", args.tto_steps, args.tto_lr)
    t_tto = time.time()

    frames_hwc_after = mini_tto.optimize(
        init_frames=frames_hwc_before,
        target_masks=target_masks,
        target_poses=target_poses,
        num_steps=args.tto_steps,
        lr=args.tto_lr,
        seg_weight=args.seg_weight,
        pose_weight=args.pose_weight,
        pair_weights=pair_weights,
        log_every=max(1, args.tto_steps // 5),
    )

    tto_time = time.time() - t_tto
    logger.info("Mini-TTO complete in %.1f seconds", tto_time)

    # ── Score AFTER mini-TTO ─────────────────────────────────────────
    logger.info("Scoring after mini-TTO...")
    score_after = compute_proxy_score(frames_hwc_after, posenet, segnet, device=device)
    logger.info("Score AFTER mini-TTO: %.4f (seg=%.4f, pose=%.6f)",
                score_after["total"], score_after.get("segnet", 0), score_after.get("posenet", 0))

    # ── Results ──────────────────────────────────────────────────────
    improvement = score_before["total"] - score_after["total"]
    logger.info("=" * 60)
    logger.info("MINI-TTO RESULTS")
    logger.info("=" * 60)
    logger.info("  Before: %.4f", score_before["total"])
    logger.info("  After:  %.4f", score_after["total"])
    logger.info("  Improvement: %.4f (%+.1f%%)",
                improvement, -100 * improvement / max(score_before["total"], 1e-8))
    logger.info("  TTO time: %.1f seconds", tto_time)
    logger.info("  Steps: %d", args.tto_steps)

    # Rate cost of mini-scorers
    mini_scorer_dir = Path(args.mini_scorer_dir)
    seg_size = (mini_scorer_dir / "mini_segnet.bin").stat().st_size if (mini_scorer_dir / "mini_segnet.bin").exists() else 0
    pose_size = (mini_scorer_dir / "mini_posenet.bin").stat().st_size if (mini_scorer_dir / "mini_posenet.bin").exists() else 0
    total_scorer_bytes = seg_size + pose_size
    rate_cost = 25 * total_scorer_bytes / 37545489
    net_benefit = improvement - rate_cost

    logger.info("  Mini-scorer size: %d bytes (rate cost = %.4f)", total_scorer_bytes, rate_cost)
    logger.info("  NET BENEFIT: %.4f (improvement %.4f - rate cost %.4f)",
                net_benefit, improvement, rate_cost)
    logger.info("  VERDICT: %s",
                "VIABLE" if net_benefit > 0 else "NOT VIABLE (rate cost > improvement)")

    # Save results
    results = {
        "experiment": "mini_tto_inflate",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": vars(args),
        "score_before": score_before,
        "score_after": score_after,
        "improvement": improvement,
        "tto_time_seconds": tto_time,
        "mini_scorer_bytes": total_scorer_bytes,
        "rate_cost": rate_cost,
        "net_benefit": net_benefit,
        "viable": net_benefit > 0,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    # Save frame tensors for analysis
    torch.save(frames_hwc_before.cpu(), str(output_dir / "frames_before.pt"))
    torch.save(frames_hwc_after.cpu(), str(output_dir / "frames_after.pt"))
    logger.info("Saved frame tensors to %s", output_dir)


if __name__ == "__main__":
    main()
