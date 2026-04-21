#!/usr/bin/env python3
"""Pose-Space TTO: optimize FiLM conditioning vectors per pair.

Instead of pixel-level TTO (707M values, 40 min on 4090), optimize the
6D FiLM pose vectors (3,600 values total, should converge in seconds).

The "optimized poses" are not physically meaningful -- they are 6D instructions
to the FiLM layer for how to render each pair optimally for the scorers.

Archive: 600 x 6 x 4 = 14.4KB (same size as GT poses, just different values).
Contest-compliant: no scorers at inflate time, single forward pass.

Usage:
    # Smoke test (local MPS, 10 pairs, 100 steps):
    PYTHONPATH=src:upstream python experiments/optimize_poses.py \
        --checkpoint path/to/renderer_best.pt --device mps --smoke

    # Full run (4090):
    PYTHONPATH=src:upstream python experiments/optimize_poses.py \
        --checkpoint path/to/renderer_best.pt --device cuda

    # Extended conditioning (pose 6D + latent 16D = 22D):
    PYTHONPATH=src:upstream python experiments/optimize_poses.py \
        --checkpoint path/to/renderer_best.pt --device cuda --latent-dim 16
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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_CANDIDATE_UPSTREAM = [
    Path(os.environ["TAC_UPSTREAM_DIR"]) if os.environ.get("TAC_UPSTREAM_DIR") else None,
    Path(os.environ["UPSTREAM_ROOT"]) if os.environ.get("UPSTREAM_ROOT") else None,
    Path(__file__).resolve().parent.parent / "upstream",
]
UPSTREAM_ROOT: Path | None = None
for _p in _CANDIDATE_UPSTREAM:
    if _p is not None and (_p / "modules.py").exists():
        UPSTREAM_ROOT = _p
        break
if UPSTREAM_ROOT is not None and str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

RESULTS_DIR = (
    Path(os.environ.get("TAC_RESULTS_DIR", ""))
    if os.environ.get("TAC_RESULTS_DIR")
    else Path(__file__).resolve().parent / "results" / "pose_tto"
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEGNET_INPUT_H, SEGNET_INPUT_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
NUM_FRAMES = 1200
NUM_PAIRS = NUM_FRAMES // 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pose-Space TTO: optimize FiLM conditioning vectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to renderer .pt or .bin checkpoint")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=NUM_FRAMES,
                   help="Number of frames to process")
    p.add_argument("--steps", type=int, default=500,
                   help="Optimization steps per batch")
    p.add_argument("--lr", type=float, default=0.01,
                   help="Adam learning rate for pose vectors")
    p.add_argument("--batch-pairs", type=int, default=50,
                   help="Pairs per optimization batch")
    p.add_argument("--seg-weight", type=float, default=100.0,
                   help="SegNet loss weight (hinge)")
    p.add_argument("--pose-weight", type=float, default=10.0,
                   help="PoseNet loss weight (MSE)")
    p.add_argument("--hinge-margin", type=float, default=0.5,
                   help="Margin for SegNet hinge loss")
    p.add_argument("--upstream", type=str, default=None,
                   help="Path to upstream repo (auto-detected if None)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: timestamped)")
    p.add_argument("--video", type=str, default=None,
                   help="Path to GT video (default: upstream/videos/0.mkv)")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 20 frames, 100 steps")
    p.add_argument("--eval-roundtrip", action="store_true",
                   help="Simulate contest eval resolution roundtrip in loss")
    p.add_argument("--early-stop-patience", type=int, default=100,
                   help="Stop if loss hasn't improved in this many steps")
    p.add_argument("--latent-dim", type=int, default=0,
                   help="Extra latent dimensions per pair (0=pose only, 16=22D total)")
    p.add_argument("--gt-poses-path", type=str, default=None,
                   help="Path to pre-extracted GT poses (poses.pt). "
                        "If not provided, extracts from GT video.")
    p.add_argument("--log-every", type=int, default=25,
                   help="Log metrics every N steps")
    return p.parse_args()


def load_renderer(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load AsymmetricPairGenerator from checkpoint (reuses renderer_tto logic)."""
    from tac.renderer import AsymmetricPairGenerator

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("model_config", ckpt.get("config", {}))
    model = AsymmetricPairGenerator(
        num_classes=model_cfg.get("num_classes", 5),
        embed_dim=model_cfg.get("embed_dim", 6),
        base_ch=model_cfg.get("base_ch", 36),
        mid_ch=model_cfg.get("mid_ch", 60),
        motion_hidden=model_cfg.get("motion_hidden", 32),
        depth=model_cfg.get("depth", 1),
        max_flow_px=model_cfg.get("max_flow_px", 20.0),
        max_residual=model_cfg.get("max_residual", 20.0),
        flow_only=model_cfg.get("flow_only", False),
        pose_dim=model_cfg.get("pose_dim", 0),
        use_dsconv=model_cfg.get("use_dsconv", False),
    )

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model = model.eval().to(device)
    # Freeze ALL renderer parameters
    for p_param in model.parameters():
        p_param.requires_grad = False

    n_params = sum(p_param.numel() for p_param in model.parameters())
    pose_dim = model_cfg.get("pose_dim", 0)
    print(f"[renderer] Loaded {n_params:,} params, pose_dim={pose_dim} from {checkpoint_path}")

    if pose_dim == 0:
        print("[WARNING] Renderer has pose_dim=0 -- FiLM layers are disabled.")
        print("[WARNING] Pose optimization will have NO effect on output.")
        print("[WARNING] Consider training a renderer with --pose-dim 6.")
    return model


def simulate_eval_roundtrip(frames_chw: torch.Tensor) -> torch.Tensor:
    """Simulate contest eval resolution roundtrip: scorer_res -> camera_res -> uint8 -> scorer_res.

    Gradients flow through via STE (Straight-Through Estimator).
    """
    orig_h, orig_w = frames_chw.shape[2], frames_chw.shape[3]
    up = F.interpolate(frames_chw, size=(CAMERA_H, CAMERA_W),
                       mode="bilinear", align_corners=False)
    # STE quantization: forward = round+clamp, backward = identity
    up_q = up + (up.round().clamp(0, 255) - up).detach()
    down = F.interpolate(up_q, size=(orig_h, orig_w),
                         mode="bilinear", align_corners=False)
    return down


def segnet_hinge_loss(
    logits: torch.Tensor,
    gt_masks: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Hinge loss on SegNet logits: penalize pixels at risk of argmax flip.

    For each pixel, the loss is max(0, margin - (correct_logit - max_other_logit)).
    This focuses gradient on boundary pixels where argmax might flip, which is
    much more efficient than cross-entropy (2-5x empirically).

    Args:
        logits: (B, C, H, W) raw SegNet output
        gt_masks: (B, H, W) long tensor of GT class indices
        margin: desired minimum gap between correct and runner-up logit

    Returns:
        Scalar mean hinge loss
    """
    B, C, H, W = logits.shape
    # Gather correct class logits
    correct = logits.gather(1, gt_masks.unsqueeze(1)).squeeze(1)  # (B, H, W)
    # Mask out correct class to find runner-up
    mask_inf = torch.zeros_like(logits)
    mask_inf.scatter_(1, gt_masks.unsqueeze(1), float("-inf"))
    runner_up = (logits + mask_inf).max(dim=1).values  # (B, H, W)
    # Hinge: penalize when gap < margin
    loss = F.relu(margin - (correct - runner_up))
    return loss.mean()


def optimize_poses_batch(
    renderer: torch.nn.Module,
    masks_t: torch.Tensor,
    masks_t1: torch.Tensor,
    gt_frames: list[torch.Tensor],
    gt_masks: torch.Tensor,
    pose_targets: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    init_poses: torch.Tensor,
    device: torch.device,
    steps: int = 500,
    lr: float = 0.01,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    hinge_margin: float = 0.5,
    eval_roundtrip: bool = False,
    early_stop_patience: int = 100,
    latent_dim: int = 0,
    log_every: int = 25,
) -> tuple[torch.Tensor, dict]:
    """Optimize pose vectors (and optional latent codes) for a batch of pairs.

    Args:
        renderer: FROZEN AsymmetricPairGenerator
        masks_t, masks_t1: (B, H, W) even/odd masks
        gt_frames: list of (H, W, 3) uint8 tensors (for scorer comparison)
        gt_masks: (2*B, H, W) GT SegNet masks for the batch frames
        pose_targets: (B, 6) GT PoseNet outputs for these pairs
        posenet, segnet: FROZEN differentiable scorers
        init_poses: (B, 6) initial pose vectors (warm start from GT)
        device: computation device
        steps: optimization steps
        lr: Adam learning rate
        seg_weight: SegNet hinge loss weight
        pose_weight: PoseNet MSE loss weight
        hinge_margin: margin for hinge loss
        eval_roundtrip: simulate eval resolution roundtrip
        early_stop_patience: stop if no improvement for this many steps
        latent_dim: extra latent dimensions (0 = pose only)
        log_every: logging frequency

    Returns:
        (optimized_conditioning, metrics_dict)
        conditioning is (B, pose_dim + latent_dim)
    """
    B = masks_t.shape[0]
    pose_dim = init_poses.shape[1]
    cond_dim = pose_dim + latent_dim

    # Initialize conditioning vector: [pose (warm start) | latent (zeros)]
    conditioning = torch.zeros(B, cond_dim, device=device, dtype=torch.float32)
    conditioning[:, :pose_dim] = init_poses.to(device)
    conditioning.requires_grad_(True)

    optimizer = torch.optim.Adam([conditioning], lr=lr)

    best_loss = float("inf")
    best_cond = conditioning.detach().clone()
    patience_counter = 0

    metrics = {
        "steps_run": 0,
        "final_loss": float("inf"),
        "final_pose_loss": float("inf"),
        "final_seg_loss": float("inf"),
        "initial_loss": float("inf"),
        "improvement_pct": 0.0,
    }

    for step in range(steps):
        optimizer.zero_grad()

        # Extract pose part of conditioning
        pose_part = conditioning[:, :pose_dim]

        # Forward: renderer produces (B, 2, H, W, 3) HWC pairs
        pairs = renderer(masks_t, masks_t1, pose=pose_part)  # (B, 2, H, W, 3)

        # Convert to CHW for scorer input
        # pairs shape: (B, 2, H, W, 3) -> need (2*B, 3, H, W) for scorers
        frame_t = pairs[:, 0]   # (B, H, W, 3)
        frame_t1 = pairs[:, 1]  # (B, H, W, 3)
        frames_hwc = torch.cat([frame_t, frame_t1], dim=0)  # (2*B, H, W, 3)
        frames_chw = frames_hwc.permute(0, 3, 1, 2).contiguous()  # (2*B, 3, H, W)

        # Optional eval roundtrip
        if eval_roundtrip:
            frames_chw = simulate_eval_roundtrip(frames_chw)

        # --- SegNet loss (hinge) ---
        # SegNet expects (B, 1, 3, H, W) via preprocess_input
        seg_in = segnet.preprocess_input(frames_chw.unsqueeze(1))
        seg_logits = segnet(seg_in)  # (2*B, 5, H, W)

        seg_loss = segnet_hinge_loss(seg_logits, gt_masks.to(device), margin=hinge_margin)

        # --- PoseNet loss (MSE on 6D output) ---
        # PoseNet expects (B, 2, 3, H, W) via preprocess_input
        pairs_chw = torch.stack([
            frame_t.permute(0, 3, 1, 2),
            frame_t1.permute(0, 3, 1, 2),
        ], dim=1)  # (B, 2, 3, H, W)

        if eval_roundtrip:
            # Apply roundtrip to the pair format too
            B_p, T_p, C_p, H_p, W_p = pairs_chw.shape
            flat = pairs_chw.reshape(B_p * T_p, C_p, H_p, W_p)
            flat = simulate_eval_roundtrip(flat)
            pairs_chw = flat.reshape(B_p, T_p, C_p, H_p, W_p)

        pose_in = posenet.preprocess_input(pairs_chw)
        pose_out = posenet(pose_in)["pose"][..., :6]  # (B, 6)
        pose_loss = F.mse_loss(pose_out, pose_targets.to(device))

        # Combined loss
        total_loss = seg_weight * seg_loss + pose_weight * pose_loss

        total_loss.backward()
        optimizer.step()

        loss_val = total_loss.item()

        if step == 0:
            metrics["initial_loss"] = loss_val

        if loss_val < best_loss:
            best_loss = loss_val
            best_cond = conditioning.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1

        if step % log_every == 0 or step == steps - 1:
            grad_norm = conditioning.grad.norm().item() if conditioning.grad is not None else 0.0
            pose_change = (conditioning[:, :pose_dim].detach() - init_poses.to(device)).norm(dim=1).mean().item()
            print(f"  step {step:4d}/{steps}: loss={loss_val:.6f} "
                  f"(seg={seg_loss.item():.6f}, pose={pose_loss.item():.6f}) "
                  f"|grad|={grad_norm:.4f} |dpose|={pose_change:.4f}")

        if patience_counter >= early_stop_patience:
            print(f"  Early stop at step {step} (no improvement for {early_stop_patience} steps)")
            break

    metrics["steps_run"] = step + 1
    metrics["final_loss"] = best_loss
    metrics["final_seg_loss"] = seg_loss.item()
    metrics["final_pose_loss"] = pose_loss.item()
    if metrics["initial_loss"] > 0:
        metrics["improvement_pct"] = (
            (metrics["initial_loss"] - best_loss) / metrics["initial_loss"] * 100
        )

    return best_cond.cpu(), metrics


def main():
    args = parse_args()

    if args.smoke:
        args.n_frames = 20
        args.steps = 100
        args.batch_pairs = 10
        args.log_every = 10
        print("[smoke] Smoke test: 20 frames, 100 steps, 10 pairs/batch")

    # Ensure even frame count
    args.n_frames = args.n_frames - (args.n_frames % 2)
    n_pairs = args.n_frames // 2

    device = torch.device(args.device)
    upstream = Path(args.upstream) if args.upstream else UPSTREAM_ROOT
    if upstream is None:
        print("ERROR: Cannot find upstream root. Set --upstream or TAC_UPSTREAM_DIR.", file=sys.stderr)
        sys.exit(1)

    if args.output_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%S")
        args.output_dir = str(RESULTS_DIR / f"pose_tto_{ts}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = args.video or str(upstream / "videos" / "0.mkv")
    cond_dim = 6 + args.latent_dim

    print(f"[config] device={device}, n_frames={args.n_frames}, steps={args.steps}")
    print(f"[config] lr={args.lr}, batch_pairs={args.batch_pairs}")
    print(f"[config] seg_weight={args.seg_weight}, pose_weight={args.pose_weight}")
    print(f"[config] hinge_margin={args.hinge_margin}, eval_roundtrip={args.eval_roundtrip}")
    print(f"[config] latent_dim={args.latent_dim} (conditioning dim={cond_dim})")
    print(f"[config] checkpoint={args.checkpoint}")
    print(f"[config] output_dir={output_dir}")

    t_total = time.monotonic()

    # ── Step 1: Load scorers ─────────────────────────────────────────────
    print("\n[1/6] Loading differentiable scorers...")
    t0 = time.monotonic()
    from tac.scorer import load_differentiable_scorers
    posenet, segnet = load_differentiable_scorers(upstream, device=str(device))
    print(f"[1/6] Scorers loaded in {time.monotonic() - t0:.1f}s")

    # ── Step 2: Load renderer ────────────────────────────────────────────
    print("\n[2/6] Loading renderer...")
    t0 = time.monotonic()
    renderer = load_renderer(args.checkpoint, device)
    print(f"[2/6] Renderer loaded in {time.monotonic() - t0:.1f}s")

    # Verify pose_dim compatibility
    renderer_pose_dim = getattr(renderer, "pose_dim", 0)
    if renderer_pose_dim == 0:
        print("\n" + "=" * 70)
        print("FATAL: Renderer has pose_dim=0. FiLM layers are not present.")
        print("Pose-space TTO requires a renderer trained with --pose-dim 6.")
        print("Aborting.")
        print("=" * 70)
        sys.exit(1)

    # ── Step 3: Decode GT video + extract masks + pose targets ──────────
    print(f"\n[3/6] Decoding GT video ({args.n_frames} frames)...")
    t0 = time.monotonic()
    from tac.data import load_gt_video
    gt_frames = load_gt_video(video_path, n_frames=args.n_frames)
    args.n_frames = len(gt_frames)
    n_pairs = args.n_frames // 2
    print(f"[3/6] Decoded {args.n_frames} frames in {time.monotonic() - t0:.1f}s")

    print("\n[4/6] Extracting GT masks and pose targets...")
    t0 = time.monotonic()
    from tac.scorer import extract_gt_masks, extract_gt_pose_targets
    gt_masks = extract_gt_masks(gt_frames, segnet, device)
    pose_targets = extract_gt_pose_targets(gt_frames, posenet, device)
    print(f"[4/6] Masks: {gt_masks.shape}, Poses: {pose_targets.shape} in {time.monotonic() - t0:.1f}s")

    # ── Step 4: Load or extract initial GT poses ────────────────────────
    print("\n[5/6] Loading initial pose vectors...")
    if args.gt_poses_path and Path(args.gt_poses_path).exists():
        init_poses = torch.load(args.gt_poses_path, map_location="cpu", weights_only=True).float()
        print(f"  Loaded GT poses from {args.gt_poses_path}: {init_poses.shape}")
    else:
        # Use pose_targets as warm start (these are PoseNet outputs on GT frames)
        init_poses = pose_targets.clone()
        print(f"  Using PoseNet GT targets as warm start: {init_poses.shape}")
    init_poses = init_poses[:n_pairs]

    # ── Step 5: Batched pose optimization ────────────────────────────────
    print(f"\n[6/6] Optimizing {n_pairs} pose vectors in batches of {args.batch_pairs}...")
    n_batches = math.ceil(n_pairs / args.batch_pairs)

    all_optimized = torch.zeros(n_pairs, cond_dim)
    all_metrics = []
    total_steps = 0

    for batch_idx in range(n_batches):
        pair_start = batch_idx * args.batch_pairs
        pair_end = min(pair_start + args.batch_pairs, n_pairs)
        frame_start = 2 * pair_start
        frame_end = 2 * pair_end
        n_batch_pairs = pair_end - pair_start

        print(f"\n--- Batch {batch_idx + 1}/{n_batches}: pairs [{pair_start}:{pair_end}] "
              f"({n_batch_pairs} pairs, {n_batch_pairs * 2} frames) ---")

        # Prepare batch data
        # Masks: even-indexed for mask_t, odd-indexed for mask_t1
        batch_masks_t = gt_masks[frame_start:frame_end:2].to(device)
        batch_masks_t1 = gt_masks[frame_start + 1:frame_end + 1:2].to(device)
        batch_gt_masks = gt_masks[frame_start:frame_end].to(device)
        batch_gt_frames = gt_frames[frame_start:frame_end]
        batch_pose_targets = pose_targets[pair_start:pair_end]
        batch_init_poses = init_poses[pair_start:pair_end]

        t0 = time.monotonic()
        optimized_cond, batch_metrics = optimize_poses_batch(
            renderer=renderer,
            masks_t=batch_masks_t,
            masks_t1=batch_masks_t1,
            gt_frames=batch_gt_frames,
            gt_masks=batch_gt_masks,
            pose_targets=batch_pose_targets,
            posenet=posenet,
            segnet=segnet,
            init_poses=batch_init_poses,
            device=device,
            steps=args.steps,
            lr=args.lr,
            seg_weight=args.seg_weight,
            pose_weight=args.pose_weight,
            hinge_margin=args.hinge_margin,
            eval_roundtrip=args.eval_roundtrip,
            early_stop_patience=args.early_stop_patience,
            latent_dim=args.latent_dim,
            log_every=args.log_every,
        )
        dt = time.monotonic() - t0

        all_optimized[pair_start:pair_end] = optimized_cond
        batch_metrics["batch_idx"] = batch_idx
        batch_metrics["time_s"] = dt
        all_metrics.append(batch_metrics)
        total_steps += batch_metrics["steps_run"]

        print(f"  Batch {batch_idx + 1} done in {dt:.1f}s "
              f"(improvement: {batch_metrics['improvement_pct']:.1f}%)")

        # Save intermediate results
        torch.save(all_optimized[:pair_end], output_dir / "optimized_poses_partial.pt")

        # Free GPU memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    # ── Step 6: Compare GT poses vs optimized poses ─────────────────────
    print("\n" + "=" * 70)
    print("RESULTS: GT poses vs optimized poses")
    print("=" * 70)

    # Save optimized poses
    optimized_poses = all_optimized[:, :6]  # pose part only
    torch.save(optimized_poses, output_dir / "optimized_poses.pt")
    print(f"  Saved optimized_poses.pt: {optimized_poses.shape}")

    if args.latent_dim > 0:
        optimized_latents = all_optimized[:, 6:]
        torch.save(optimized_latents, output_dir / "optimized_latents.pt")
        print(f"  Saved optimized_latents.pt: {optimized_latents.shape}")

    # Compute proxy score with GT poses vs optimized poses
    print("\n  Computing proxy scores (GT poses vs optimized poses)...")
    from tac.scorer import compute_proxy_score

    # Generate frames with GT poses
    gt_pose_frames = _generate_frames(renderer, gt_masks, init_poses[:n_pairs], device, args.batch_pairs)
    gt_score = compute_proxy_score(
        gt_pose_frames, gt_frames, posenet, segnet, device, rate=0.0,
    )

    # Generate frames with optimized poses
    opt_pose_frames = _generate_frames(renderer, gt_masks, optimized_poses, device, args.batch_pairs)
    opt_score = compute_proxy_score(
        opt_pose_frames, gt_frames, posenet, segnet, device, rate=0.0,
    )

    print(f"\n  GT Poses:        score={gt_score['score']:.4f} "
          f"(pose={gt_score['pose']:.6f}, seg={gt_score['seg']:.6f})")
    print(f"  Optimized Poses: score={opt_score['score']:.4f} "
          f"(pose={opt_score['pose']:.6f}, seg={opt_score['seg']:.6f})")
    print(f"  Delta:           {opt_score['score'] - gt_score['score']:+.4f} "
          f"(pose: {opt_score['pose'] - gt_score['pose']:+.6f}, "
          f"seg: {opt_score['seg'] - gt_score['seg']:+.6f})")

    # Pose vector statistics
    pose_delta = (optimized_poses - init_poses[:n_pairs]).norm(dim=1)
    print(f"\n  Pose delta: mean={pose_delta.mean():.4f}, "
          f"max={pose_delta.max():.4f}, min={pose_delta.min():.4f}")

    # Archive size estimate
    archive_bytes = n_pairs * cond_dim * 4  # float32
    archive_fp16_bytes = n_pairs * cond_dim * 2  # float16
    print(f"\n  Archive size: {archive_bytes:,} bytes (fp32), {archive_fp16_bytes:,} bytes (fp16)")

    # Save summary
    total_time = time.monotonic() - t_total
    summary = {
        "config": vars(args),
        "gt_score": gt_score,
        "optimized_score": opt_score,
        "delta_score": opt_score["score"] - gt_score["score"],
        "total_time_s": total_time,
        "total_steps": total_steps,
        "n_pairs": n_pairs,
        "pose_delta_mean": pose_delta.mean().item(),
        "pose_delta_max": pose_delta.max().item(),
        "batch_metrics": all_metrics,
        "archive_bytes_fp32": archive_bytes,
        "archive_bytes_fp16": archive_fp16_bytes,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Results saved to: {output_dir}")
    print("=" * 70)


def _generate_frames(
    renderer: torch.nn.Module,
    masks: torch.Tensor,
    poses: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Generate frames using renderer with pose conditioning.

    Args:
        renderer: AsymmetricPairGenerator
        masks: (N, H, W) long masks
        poses: (P, 6) pose vectors, P = N//2
        device: computation device
        batch_size: pairs per forward pass

    Returns:
        (N, H, W, 3) float tensor of rendered frames in [0, 255]
    """
    N = masks.shape[0]
    P = N // 2
    all_frames = []

    with torch.inference_mode():
        for start in range(0, P, batch_size):
            end = min(start + batch_size, P)
            mask_t = masks[2 * start:2 * end:2].to(device)
            mask_t1 = masks[2 * start + 1:2 * end + 1:2].to(device)
            batch_pose = poses[start:end].to(device)

            pairs = renderer(mask_t, mask_t1, pose=batch_pose)  # (B, 2, H, W, 3)
            f0 = pairs[:, 0]
            f1 = pairs[:, 1]
            B = f0.shape[0]
            interleaved = torch.stack([f0, f1], dim=1).reshape(2 * B, *f0.shape[1:])
            all_frames.append(interleaved.cpu())

    return torch.cat(all_frames, dim=0).float()


if __name__ == "__main__":
    main()
