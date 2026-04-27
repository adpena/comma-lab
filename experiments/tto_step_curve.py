#!/usr/bin/env python3
"""TTO Step-vs-Improvement Curve Experiment.

Measures the diminishing-returns relationship between TTO step count and
scorer improvement. This empirical curve is essential for adaptive TTO
budget allocation: knowing exactly when additional steps stop helping
determines how many pairs we can afford to TTO within the contest budget.

Runs TTO at multiple step counts (10, 25, 50, 100, 150, 200, 300, 500)
and measures PoseNet + SegNet distortion at each checkpoint. Uses v5b
config (embedding loss, seg_odd_only) which is our current best TTO recipe.

Output: JSON with step_count -> {posenet, segnet, score, elapsed_s} mapping.

Design rationale:
    - Non-overlapping pairs are independent, so we can run different step
      counts on different pair subsets without contamination.
    - We use a FIXED subset of 30 pairs (60 frames) for all step counts
      to ensure comparability. This is 5% of the full 600-pair set --
      enough for statistical signal without excessive compute.
    - Each step count is run from the SAME renderer initialization (no
      warm-starting from prior step counts) to isolate the marginal value.

Usage:
    # Local smoke test (5 pairs, 3 step counts):
    PYTHONPATH=src:upstream python experiments/tto_step_curve.py \
        --checkpoint path/to/renderer_best.pt --device mps --smoke

    # Full experiment on Modal/Vast T4/4090:
    PYTHONPATH=src:upstream python experiments/tto_step_curve.py \
        --checkpoint path/to/renderer_best.pt --device cuda

    # Custom step counts:
    PYTHONPATH=src:upstream python experiments/tto_step_curve.py \
        --checkpoint path/to/renderer_best.pt --device cuda \
        --step-counts 10,50,100,200,500,1000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Default step counts for the full sweep
DEFAULT_STEP_COUNTS: list[int] = [10, 25, 50, 100, 150, 200, 300, 500]
SMOKE_STEP_COUNTS: list[int] = [10, 50, 100]

# Fixed evaluation subset: 30 pairs (60 frames) from the middle of the video
# to avoid edge effects from the first/last frames.
EVAL_PAIR_START: int = 100
EVAL_N_PAIRS: int = 30
SMOKE_N_PAIRS: int = 5


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="TTO Step-vs-Improvement Curve Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to renderer .pt checkpoint")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "mps", "cpu"])
    p.add_argument("--upstream", type=str, default="upstream/",
                   help="Path to upstream repo")
    p.add_argument("--video", type=str, default=None,
                   help="Path to GT video (default: upstream/videos/0.mkv)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: timestamped)")
    p.add_argument("--step-counts", type=str, default=None,
                   help="Comma-separated step counts (default: 10,25,50,100,150,200,300,500)")
    p.add_argument("--n-pairs", type=int, default=EVAL_N_PAIRS,
                   help="Number of pairs to evaluate at each step count")
    p.add_argument("--pair-start", type=int, default=EVAL_PAIR_START,
                   help="Starting pair index for the evaluation subset")
    p.add_argument("--batch-pairs", type=int, default=10,
                   help="Pairs per TTO optimization batch (VRAM constrained)")
    p.add_argument("--tto-lr", type=float, default=0.005,
                   help="TTO learning rate")
    p.add_argument("--seg-weight", type=float, default=100.0,
                   help="SegNet loss weight")
    p.add_argument("--pose-weight", type=float, default=10.0,
                   help="PoseNet loss weight")
    p.add_argument("--compress-weight", type=float, default=0.5,
                   help="Compressibility weight")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 5 pairs, 3 step counts")
    p.add_argument("--use-embedding-loss", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Use PoseNet embedding MSE (v5b default: enabled)")
    p.add_argument("--seg-odd-only", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="SegNet loss on odd frames only (v5b default: enabled)")
    # CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True. Removed
    # `--no-eval-roundtrip` flag; only escape hatch is TAC_ALLOW_NO_ROUNDTRIP=1.
    p.add_argument("--eval-roundtrip", action="store_true", default=True,
                   help="Simulate scorer resolution round-trip in proxy scoring. "
                        "ALWAYS True; disabling requires TAC_ALLOW_NO_ROUNDTRIP=1.")
    p.add_argument("--lr-schedule", type=str, default="constant",
                   choices=["constant", "cosine"],
                   help="LR schedule for TTO. 'cosine' = warmup then cosine decay.")
    p.add_argument("--segnet-loss-mode", type=str, default="xent",
                   choices=["xent", "hinge"],
                   help="SegNet loss function. 'xent' = cross-entropy (default). "
                        "'hinge' = logit-margin hinge loss.")
    p.add_argument("--hinge-margin", type=float, default=0.5,
                   help="Margin for hinge loss (only used with --segnet-loss-mode hinge).")
    p.add_argument("--use-null-space", action="store_true",
                   help="Apply adaptive per-pixel YUV null space projection for free SegNet improvement.")
    p.add_argument("--null-space-step", type=float, default=0.5,
                   help="Step size for null space projection (only with --use-null-space).")
    p.add_argument("--null-space-every", type=int, default=10,
                   help="Apply null space projection every N steps (only with --use-null-space).")
    return p.parse_args()


def compute_pair_distortions(
    frames: torch.Tensor,
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    pair_start: int,
    n_pairs: int,
    eval_roundtrip: bool = True,
) -> dict[str, float]:
    """Compute PoseNet and SegNet distortions for a subset of pairs.

    Uses the same scoring logic as the official evaluate.py but on a
    subset of frames for efficiency.

    Args:
        frames: (N, H, W, 3) float tensor of candidate frames.
        gt_frames: list of (H, W, 3) uint8 tensors (ground truth).
        posenet: PoseNet scorer.
        segnet: SegNet scorer.
        device: computation device.
        pair_start: first pair index in the subset.
        n_pairs: number of pairs to evaluate.
        eval_roundtrip: simulate official scorer resolution pipeline.

    Returns:
        Dict with 'posenet', 'segnet', 'score' keys.
    """
    from tac.scorer import compute_proxy_score

    frame_start = 2 * pair_start
    frame_end = 2 * (pair_start + n_pairs)

    subset_frames = frames[frame_start:frame_end]
    subset_gt = gt_frames[frame_start:frame_end]

    result = compute_proxy_score(
        subset_frames, subset_gt, posenet, segnet, device,
        eval_roundtrip=eval_roundtrip,
    )
    return {
        "posenet": result["pose"],
        "segnet": result["seg"],
        "score": result["score"],
        "pose_contribution": result["pose_contribution"],
        "seg_contribution": result["seg_contribution"],
    }


def run_single_step_count(
    step_count: int,
    renderer_frames: torch.Tensor,
    masks: torch.Tensor,
    pose_targets: torch.Tensor,
    pose_embeddings: torch.Tensor | None,
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    pair_start: int,
    n_pairs: int,
    batch_pairs: int,
    tto_lr: float,
    seg_weight: float,
    pose_weight: float,
    compress_weight: float,
    use_embedding_loss: bool,
    seg_odd_only: bool,
    eval_roundtrip: bool,
    lr_schedule: str = "constant",
    segnet_loss_mode: str = "xent",
    hinge_margin: float = 0.5,
    use_null_space: bool = False,
    null_space_step: float = 0.5,
    null_space_every: int = 10,
) -> dict[str, float]:
    """Run TTO at a single step count and measure distortion.

    Each call starts from the SAME renderer frames (no warm-starting
    from prior runs) to isolate the marginal value of additional steps.

    Args:
        step_count: number of TTO gradient steps.
        renderer_frames: (N, H, W, 3) renderer output (full video).
        masks: (N, H, W) segmentation masks (full video).
        pose_targets: (P, 6) GT pose targets (full video).
        pose_embeddings: (P, D) GT pose embeddings or None.
        gt_frames: list of GT frames for scoring.
        posenet: frozen PoseNet.
        segnet: frozen SegNet.
        device: computation device.
        pair_start: starting pair index for the evaluation subset.
        n_pairs: number of pairs to process.
        batch_pairs: pairs per optimization batch.
        tto_lr: learning rate.
        seg_weight: SegNet loss weight.
        pose_weight: PoseNet loss weight.
        compress_weight: compressibility weight.
        use_embedding_loss: use embedding MSE.
        seg_odd_only: SegNet on odd frames only.
        eval_roundtrip: simulate scorer resolution pipeline.
        lr_schedule: LR schedule ('constant' or 'cosine').
        segnet_loss_mode: SegNet loss function (``"xent"`` or ``"hinge"``).
        hinge_margin: margin for hinge loss.
        use_null_space: apply adaptive per-pixel YUV null space projection.
        null_space_step: step size for null space projection.
        null_space_every: apply null space projection every N steps.

    Returns:
        Dict with posenet, segnet, score, elapsed_s, steps.
    """
    from tac.constrained_gen import coupled_trajectory_optimize

    frame_start = 2 * pair_start
    frame_end = 2 * (pair_start + n_pairs)

    batch_masks = masks[frame_start:frame_end]
    batch_init = renderer_frames[frame_start:frame_end].clone()
    batch_pose = pose_targets[pair_start:pair_start + n_pairs]
    batch_emb = None
    if use_embedding_loss and pose_embeddings is not None:
        batch_emb = pose_embeddings[pair_start:pair_start + n_pairs]

    t0 = time.monotonic()

    # Run TTO with the specified step count
    # Process in sub-batches for VRAM safety
    n_sub_batches = math.ceil(n_pairs / batch_pairs)
    refined = torch.zeros_like(batch_init)

    for sub_idx in range(n_sub_batches):
        sp_start = sub_idx * batch_pairs
        sp_end = min(sp_start + batch_pairs, n_pairs)
        sf_start = 2 * sp_start
        sf_end = 2 * sp_end

        sub_masks = batch_masks[sf_start:sf_end]
        sub_init = batch_init[sf_start:sf_end]
        sub_pose = batch_pose[sp_start:sp_end]
        sub_emb = batch_emb[sp_start:sp_end] if batch_emb is not None else None

        sub_result = coupled_trajectory_optimize(
            masks=sub_masks,
            expected_pose=sub_pose,
            posenet=posenet,
            segnet=segnet,
            num_steps=step_count,
            lr=tto_lr,
            seg_weight=seg_weight,
            pose_weight=pose_weight,
            compress_weight=compress_weight,
            noise_seed=42 + sub_idx,
            device=str(device),
            log_every=max(step_count // 3, 1),
            init_frames=sub_init,
            early_stop_patience=step_count + 1,  # disable early stop for curve
            use_embedding_loss=use_embedding_loss,
            expected_pose_embeddings=sub_emb,
            seg_odd_only=seg_odd_only,
            lr_schedule=lr_schedule,
            segnet_loss_mode=segnet_loss_mode,
            hinge_margin=hinge_margin,
            use_null_space=use_null_space,
            null_space_step=null_space_step,
            null_space_every=null_space_every,
        )
        refined[sf_start:sf_end] = sub_result.cpu()

        # Free GPU
        del sub_result
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    elapsed = time.monotonic() - t0

    # Measure distortion — both refined and GT must be LOCAL subsets
    # starting at index 0, with matching pairs aligned.
    # refined is already a local subset [0:2*n_pairs].
    # gt_frames must be sliced to the same range.
    gt_slice = gt_frames[frame_start:frame_end]
    distortions = compute_pair_distortions(
        refined, gt_slice, posenet, segnet, device,
        pair_start=0, n_pairs=n_pairs,
        eval_roundtrip=eval_roundtrip,
    )
    distortions["elapsed_s"] = round(elapsed, 2)
    distortions["steps"] = step_count
    distortions["s_per_frame"] = round(elapsed / max(2 * n_pairs, 1), 3)

    return distortions


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
    """Run the TTO step-vs-improvement curve experiment."""
    args = parse_args()

    # Smoke test overrides
    if args.smoke:
        args.n_pairs = SMOKE_N_PAIRS
        step_counts = SMOKE_STEP_COUNTS
        print("[smoke] Smoke test: 5 pairs, steps=[10, 50, 100]")
    elif args.step_counts:
        step_counts = [int(s.strip()) for s in args.step_counts.split(",")]
    else:
        step_counts = DEFAULT_STEP_COUNTS

    device = torch.device(args.device)
    upstream = Path(args.upstream)
    video_path = args.video or str(upstream / "videos" / "0.mkv")

    if args.output_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%S")
        args.output_dir = f"experiments/results/tto_step_curve_{ts}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # codex R5-r6 #3: gate AFTER output_dir resolution so sidecar lands.
    _enforce_eval_roundtrip(args)

    print(f"[config] device={device}, n_pairs={args.n_pairs}, "
          f"pair_start={args.pair_start}")
    print(f"[config] step_counts={step_counts}")
    print(f"[config] tto_lr={args.tto_lr}, seg_weight={args.seg_weight}, "
          f"pose_weight={args.pose_weight}, compress_weight={args.compress_weight}")
    print(f"[config] use_embedding_loss={args.use_embedding_loss}, "
          f"seg_odd_only={args.seg_odd_only}, "
          f"segnet_loss_mode={args.segnet_loss_mode}, "
          f"hinge_margin={args.hinge_margin}")
    print(f"[config] checkpoint={args.checkpoint}")
    print(f"[config] video={video_path}")
    print(f"[config] output_dir={output_dir}")

    # ---- Checkpoint sanity check ----
    from tac.checkpoint import verify_checkpoint_identity

    md5 = verify_checkpoint_identity(args.checkpoint)
    print(f"[checkpoint] Verified MD5 prefix: {md5}")

    t_total_start = time.monotonic()

    # ---- Step 1: Load scorers ----
    print("\n[1/6] Loading scorers...")
    t0 = time.monotonic()
    from tac.scorer import (
        compute_proxy_score,
        extract_gt_masks,
        extract_gt_pose_targets,
        load_differentiable_scorers,
    )
    posenet, segnet = load_differentiable_scorers(upstream, device=str(device))
    t_scorers = time.monotonic() - t0
    print(f"[1/6] Scorers loaded in {t_scorers:.1f}s")

    # ---- Step 2: Load renderer ----
    print("\n[2/6] Loading renderer...")
    t0 = time.monotonic()
    from experiments.renderer_tto import load_renderer
    renderer = load_renderer(args.checkpoint, device)
    t_renderer = time.monotonic() - t0
    print(f"[2/6] Renderer loaded in {t_renderer:.1f}s")

    # ---- Step 3: Decode GT video ----
    # We need enough frames to cover the evaluation subset
    n_frames_needed = 2 * (args.pair_start + args.n_pairs)
    print(f"\n[3/6] Decoding GT video ({n_frames_needed} frames)...")
    t0 = time.monotonic()
    from tac.data import load_gt_video
    gt_frames = load_gt_video(video_path, n_frames=n_frames_needed)
    n_frames_actual = len(gt_frames)
    t_decode = time.monotonic() - t0
    print(f"[3/6] Decoded {n_frames_actual} frames in {t_decode:.1f}s")

    # Adjust pair range if video is shorter than expected
    max_pairs = n_frames_actual // 2
    if args.pair_start + args.n_pairs > max_pairs:
        args.pair_start = max(0, max_pairs - args.n_pairs)
        args.n_pairs = min(args.n_pairs, max_pairs - args.pair_start)
        print(f"[3/6] Adjusted: pair_start={args.pair_start}, n_pairs={args.n_pairs}")

    # ---- Step 4: Extract masks ----
    print("\n[4/6] Extracting SegNet masks...")
    t0 = time.monotonic()
    masks = extract_gt_masks(gt_frames, segnet, device)
    t_masks = time.monotonic() - t0
    print(f"[4/6] Extracted {masks.shape[0]} masks in {t_masks:.1f}s")

    # ---- Step 5: Generate renderer frames ----
    print("\n[5/6] Generating renderer frames...")
    t0 = time.monotonic()
    from experiments.renderer_tto import generate_renderer_frames
    renderer_frames = generate_renderer_frames(renderer, masks, device)
    t_render = time.monotonic() - t0
    print(f"[5/6] Generated {renderer_frames.shape[0]} frames in {t_render:.1f}s")

    del renderer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- Extract GT targets ----
    print("\n[5.5/6] Extracting GT pose targets...")
    t0 = time.monotonic()
    pose_targets = extract_gt_pose_targets(gt_frames, posenet, device)
    t_targets = time.monotonic() - t0
    print(f"[5.5/6] Extracted {pose_targets.shape[0]} targets in {t_targets:.1f}s")

    pose_embeddings = None
    if args.use_embedding_loss:
        print("[5.5/6] Extracting GT pose embeddings...")
        t0 = time.monotonic()
        from experiments.renderer_tto import extract_gt_pose_embeddings
        pose_embeddings = extract_gt_pose_embeddings(gt_frames, posenet, device)
        t_emb = time.monotonic() - t0
        print(f"[5.5/6] Extracted {pose_embeddings.shape[0]} embeddings in {t_emb:.1f}s")

    # ---- Baseline (0 steps = renderer only) ----
    print("\n[baseline] Computing renderer-only distortion...")
    baseline = compute_pair_distortions(
        renderer_frames, gt_frames, posenet, segnet, device,
        pair_start=args.pair_start, n_pairs=args.n_pairs,
        eval_roundtrip=args.eval_roundtrip,
    )
    baseline["elapsed_s"] = 0.0
    baseline["steps"] = 0
    baseline["s_per_frame"] = 0.0
    print(f"[baseline] posenet={baseline['posenet']:.6f}, "
          f"segnet={baseline['segnet']:.6f}, score={baseline['score']:.4f}")

    # ---- Step 6: Run TTO at each step count ----
    results: dict[str, dict[str, float]] = {"0": baseline}

    for i, step_count in enumerate(step_counts):
        print(f"\n{'=' * 60}")
        print(f"[6/6] TTO sweep {i + 1}/{len(step_counts)}: {step_count} steps")
        print(f"{'=' * 60}")

        distortions = run_single_step_count(
            step_count=step_count,
            renderer_frames=renderer_frames,
            masks=masks,
            pose_targets=pose_targets,
            pose_embeddings=pose_embeddings,
            gt_frames=gt_frames,
            posenet=posenet,
            segnet=segnet,
            device=device,
            pair_start=args.pair_start,
            n_pairs=args.n_pairs,
            batch_pairs=args.batch_pairs,
            tto_lr=args.tto_lr,
            seg_weight=args.seg_weight,
            pose_weight=args.pose_weight,
            compress_weight=args.compress_weight,
            use_embedding_loss=args.use_embedding_loss,
            seg_odd_only=args.seg_odd_only,
            eval_roundtrip=args.eval_roundtrip,
            lr_schedule=args.lr_schedule,
            segnet_loss_mode=args.segnet_loss_mode,
            hinge_margin=args.hinge_margin,
            use_null_space=args.use_null_space,
            null_space_step=args.null_space_step,
            null_space_every=args.null_space_every,
        )

        results[str(step_count)] = distortions

        delta_pose = baseline["posenet"] - distortions["posenet"]
        delta_seg = baseline["segnet"] - distortions["segnet"]
        delta_score = baseline["score"] - distortions["score"]

        print(f"[result] steps={step_count}: "
              f"posenet={distortions['posenet']:.6f} (delta={delta_pose:+.6f}), "
              f"segnet={distortions['segnet']:.6f} (delta={delta_seg:+.6f}), "
              f"score={distortions['score']:.4f} (delta={delta_score:+.4f}), "
              f"elapsed={distortions['elapsed_s']:.1f}s "
              f"({distortions['s_per_frame']:.2f}s/frame)")

        # Save incremental results after each step count
        _save_results(output_dir, results, args)

    t_total = time.monotonic() - t_total_start

    # ---- Final summary ----
    print("\n" + "=" * 72)
    print("TTO STEP-VS-IMPROVEMENT CURVE")
    print("=" * 72)
    print(f"  {'Steps':>6s}  {'PoseNet':>12s}  {'SegNet':>12s}  {'Score':>10s}  "
          f"{'Time (s)':>10s}  {'s/frame':>8s}")
    print(f"  {'-' * 6}  {'-' * 12}  {'-' * 12}  {'-' * 10}  "
          f"{'-' * 10}  {'-' * 8}")

    for key in sorted(results.keys(), key=int):
        r = results[key]
        print(f"  {r['steps']:>6d}  {r['posenet']:>12.6f}  {r['segnet']:>12.6f}  "
              f"{r['score']:>10.4f}  {r['elapsed_s']:>10.2f}  {r['s_per_frame']:>8.3f}")

    print(f"\n  Total experiment time: {t_total:.1f}s")
    print(f"  Output: {output_dir}")
    print("=" * 72)


def _save_results(
    output_dir: Path,
    results: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> None:
    """Save current results to JSON."""
    output = {
        "experiment": "tto_step_curve",
        "config": {
            "checkpoint": args.checkpoint,
            "device": args.device,
            "n_pairs": args.n_pairs,
            "pair_start": args.pair_start,
            "batch_pairs": args.batch_pairs,
            "tto_lr": args.tto_lr,
            "seg_weight": args.seg_weight,
            "pose_weight": args.pose_weight,
            "compress_weight": args.compress_weight,
            "use_embedding_loss": args.use_embedding_loss,
            "seg_odd_only": args.seg_odd_only,
            "eval_roundtrip": args.eval_roundtrip,
            "segnet_loss_mode": args.segnet_loss_mode,
            "hinge_margin": args.hinge_margin,
            "use_null_space": args.use_null_space,
            "null_space_step": args.null_space_step,
            "null_space_every": args.null_space_every,
        },
        "results": results,
    }
    with open(output_dir / "step_curve.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
