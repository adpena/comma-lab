#!/usr/bin/env python3
"""GT-Initialized Sparse Patch TTO: the path to sub-0.2 auth score.

Core insight: Start from GT video (score=0.0000 by definition) and perturb
ONLY the patches that PoseNet does not read. This gives us:
  1. A MUCH better optimization basin than renderer warm-start (0.87)
  2. Reduced dimensionality: ~7K params (50 patches * 7*7*3) vs 589K (all pixels)
  3. Rate minimization on insensitive patches to reduce archive size

The optimization uses L-BFGS (not SGD) because:
  - Low dimensionality (~7K) makes L-BFGS practical and fast
  - L-BFGS has better convergence on smooth, low-dimensional landscapes
  - Stochastic restarts escape saddle points that SGD cannot

Architecture:
  - Load GT video frames as float32 tensors
  - Load precomputed PoseNet sensitivity map (from posenet_sensitivity.py)
  - Select top-K INSENSITIVE patches (safe to perturb)
  - Create a sparse perturbation tensor (only selected patches are optimized)
  - Optimize perturbations via L-BFGS with:
      * PoseNet preservation: keep pose score near zero
      * SegNet preservation: keep SegNet score near zero
      * Rate minimization: make perturbed frames more compressible
  - Stochastic restarts: save best, add noise, re-optimize

Usage::

    # Smoke test (local MPS):
    PYTHONPATH=src:upstream python experiments/gt_sparse_tto.py \
        --device mps --smoke

    # Full run (Modal T4):
    PYTHONPATH=src:upstream python experiments/gt_sparse_tto.py \
        --device cuda --n-frames 1200 --n-patches 50 --n-restarts 5 \
        --steps-per-restart 200

    # With precomputed sensitivity map:
    PYTHONPATH=src:upstream python experiments/gt_sparse_tto.py \
        --device cuda --sensitivity-map experiments/results/posenet_sensitivity_*/sensitivity_map.pt
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F


from tac.scorer import compute_proxy_score, extract_gt_masks, extract_gt_pose_targets
from tac.utils import find_project_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GT-Initialized Sparse Patch TTO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=1200, help="Number of frames to process")
    p.add_argument("--n-patches", type=int, default=50,
                   help="Number of insensitive patches to optimize per frame")
    p.add_argument("--patch-size", type=int, default=7, help="Patch size (must match sensitivity map)")
    p.add_argument("--n-restarts", type=int, default=5, help="Stochastic restarts")
    p.add_argument("--steps-per-restart", type=int, default=200, help="L-BFGS steps per restart")
    p.add_argument("--lbfgs-lr", type=float, default=1.0, help="L-BFGS learning rate")
    p.add_argument("--lbfgs-history", type=int, default=20, help="L-BFGS history size")
    p.add_argument("--restart-noise", type=float, default=5.0,
                   help="Noise magnitude for stochastic restarts (pixels)")
    p.add_argument("--seg-weight", type=float, default=100.0, help="SegNet preservation weight")
    p.add_argument("--pose-weight", type=float, default=10.0, help="PoseNet preservation weight")
    p.add_argument("--rate-weight", type=float, default=2.0, help="Rate/compressibility weight")
    p.add_argument("--perturbation-budget", type=float, default=30.0,
                   help="Max per-pixel perturbation magnitude (clamp)")
    p.add_argument("--batch-pairs", type=int, default=50, help="Pairs per optimization batch")
    p.add_argument("--upstream", type=str, default=None, help="Path to upstream repo")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--video", type=str, default=None, help="Path to GT video")
    p.add_argument("--sensitivity-map", type=str, default=None,
                   help="Path to precomputed sensitivity_map.pt (skips recomputation)")
    p.add_argument("--smoke", action="store_true", help="Smoke test: 20 frames, 2 restarts, 20 steps")
    p.add_argument("--simulate-resize", action="store_true",
                   help="Simulate official scorer resolution round-trip in proxy scoring")
    return p.parse_args()



class SparsePatchPerturbation(torch.nn.Module):
    """Sparse perturbation applied only to selected patches.

    Maintains a dense perturbation buffer but masks it to only affect
    the selected (insensitive) patches. The mask is fixed after init.

    The optimization variable is the perturbation delta, not the frame pixels.
    This keeps the perturbation bounded and makes L-BFGS more effective.
    """

    def __init__(
        self,
        n_frames: int,
        height: int,
        width: int,
        patch_locations: list[dict],
        perturbation_budget: float = 30.0,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.height = height
        self.width = width
        self.perturbation_budget = perturbation_budget

        # The optimizable perturbation (dense, but masked during forward)
        self.delta = torch.nn.Parameter(
            torch.zeros(n_frames, height, width, 3)
        )

        # Build binary mask: 1.0 at selected patches, 0.0 elsewhere
        mask = torch.zeros(height, width, dtype=torch.float32)
        for loc in patch_locations:
            r, c, h, w = loc["row"], loc["col"], loc["h"], loc["w"]
            mask[r:r + h, c:c + w] = 1.0
        # (H, W) -> (1, H, W, 1) for broadcasting with (N, H, W, 3)
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(-1))

        n_active_pixels = int(mask.sum().item())
        total_pixels = height * width
        print(f"[sparse] {n_active_pixels}/{total_pixels} active pixels "
              f"({100 * n_active_pixels / total_pixels:.1f}%), "
              f"{len(patch_locations)} patches, budget=+/-{perturbation_budget}")

    def forward(self, gt_frames: torch.Tensor) -> torch.Tensor:
        """Apply sparse perturbation to GT frames.

        Args:
            gt_frames: (N, H, W, 3) float tensor of GT frames in [0, 255].

        Returns:
            (N, H, W, 3) float tensor of perturbed frames in [0, 255].
        """
        # Clamp perturbation to budget, then mask to selected patches only
        clamped_delta = self.delta.clamp(-self.perturbation_budget, self.perturbation_budget)
        masked_delta = clamped_delta * self.mask  # (N, H, W, 3)

        # Apply perturbation and clamp to valid pixel range
        perturbed = (gt_frames + masked_delta).clamp(0.0, 255.0)
        return perturbed

    def n_parameters(self) -> int:
        """Number of effectively active parameters (within mask)."""
        return int(self.mask.sum().item()) * 3 * self.n_frames


def select_patches_for_frames(
    sensitivity_map: torch.Tensor,
    n_patches: int,
    patch_size: int,
) -> list[dict]:
    """Select top-K insensitive patches from the sensitivity map.

    Returns patches sorted by ascending sensitivity (most insensitive first).

    Args:
        sensitivity_map: (H, W) float tensor of PoseNet gradient magnitudes.
        n_patches: number of patches to select.
        patch_size: size of square patches.

    Returns:
        List of dicts: [{row, col, h, w, mean_sensitivity}, ...].
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
            })

    patches.sort(key=lambda x: x["mean_sensitivity"])
    if n_patches > len(patches):
        import warnings
        warnings.warn(
            f"Requested {n_patches} patches but only {len(patches)} available "
            f"(map {H}x{W}, patch_size={patch_size}). Using all {len(patches)}."
        )
    selected = patches[:n_patches]

    if selected:
        print(f"[patch-select] Selected {len(selected)} patches, "
              f"sensitivity range: [{selected[0]['mean_sensitivity']:.6f}, "
              f"{selected[-1]['mean_sensitivity']:.6f}]")

    return selected


def compute_sensitivity_map_live(
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    device: torch.device,
    batch_pairs: int = 4,
) -> torch.Tensor:
    """Compute PoseNet sensitivity map (inline version).

    Same algorithm as posenet_sensitivity.py but without file I/O.
    Uses pose.pow(2).sum() to avoid sign cancellation.
    """
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W

    N = len(gt_frames)
    P = N // 2

    grad_accum = torch.zeros(SEGNET_INPUT_H, SEGNET_INPUT_W, device="cpu", dtype=torch.float64)
    n_accumulated = 0

    for start in range(0, P, batch_pairs):
        end = min(start + batch_pairs, P)
        B = end - start

        pair_list = []
        for k in range(start, end):
            f0 = gt_frames[2 * k].float()
            f1 = gt_frames[2 * k + 1].float()
            pair_list.append(torch.stack([f0, f1], dim=0))

        pairs_hwc = torch.stack(pair_list).to(device)
        pairs_hwc.requires_grad_(True)

        pairs_chw = pairs_hwc.permute(0, 1, 4, 2, 3).contiguous()

        _, T, C, H, W = pairs_chw.shape
        if H != SEGNET_INPUT_H or W != SEGNET_INPUT_W:
            pairs_flat = pairs_chw.reshape(B * T, C, H, W)
            pairs_flat = F.interpolate(
                pairs_flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear", align_corners=False,
            )
            pairs_chw = pairs_flat.reshape(B, T, C, SEGNET_INPUT_H, SEGNET_INPUT_W)

        posenet_in = posenet.preprocess_input(pairs_chw)
        posenet_out = posenet(posenet_in)
        pose = posenet_out["pose"][..., :6]

        scalar_loss = pose.pow(2).sum()
        scalar_loss.backward()

        grad = pairs_hwc.grad.detach()
        grad_mag = grad.norm(dim=-1)
        grad_mag_avg = grad_mag.mean(dim=(0, 1))

        H_in, W_in = grad_mag_avg.shape
        if H_in != SEGNET_INPUT_H or W_in != SEGNET_INPUT_W:
            grad_mag_avg = F.interpolate(
                grad_mag_avg.unsqueeze(0).unsqueeze(0),
                size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear", align_corners=False,
            ).squeeze()

        grad_accum += grad_mag_avg.cpu().double() * B
        n_accumulated += B

        del pairs_hwc, pairs_chw, posenet_in, posenet_out, pose, scalar_loss, grad
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    return (grad_accum / max(n_accumulated, 1)).float()


def run_sparse_tto_batch(
    gt_frames_batch: torch.Tensor,
    patch_locations: list[dict],
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    gt_pose_targets: torch.Tensor,
    gt_seg_masks: torch.Tensor,
    device: torch.device,
    steps_per_restart: int = 200,
    n_restarts: int = 5,
    lbfgs_lr: float = 1.0,
    lbfgs_history: int = 20,
    restart_noise: float = 5.0,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    rate_weight: float = 2.0,
    perturbation_budget: float = 30.0,
    log_every: int = 50,
) -> torch.Tensor:
    """Run sparse patch TTO on a batch of frame pairs.

    Uses L-BFGS optimizer with stochastic restarts. Only patches in
    patch_locations are perturbed; all other pixels stay at GT values.

    Args:
        gt_frames_batch: (N, H, W, 3) float tensor of GT frames for this batch.
        patch_locations: list of {row, col, h, w} for selected insensitive patches.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        gt_pose_targets: (P, 6) float tensor of GT pose values for this batch.
        gt_seg_masks: (N, H, W) long tensor of GT SegNet masks for this batch.
        device: computation device.
        steps_per_restart: L-BFGS steps per restart cycle.
        n_restarts: number of stochastic restart cycles.
        lbfgs_lr: L-BFGS learning rate.
        lbfgs_history: L-BFGS history size.
        restart_noise: noise magnitude for restarts (pixel units).
        seg_weight: SegNet preservation weight.
        pose_weight: PoseNet preservation weight.
        rate_weight: compressibility weight.
        perturbation_budget: max per-pixel perturbation.
        log_every: print loss every N steps.

    Returns:
        (N, H, W, 3) float tensor of optimized frames in [0, 255].
    """
    from tac.constrained_gen import (
        compute_compressibility_loss,
        compute_posenet_constraint_loss,
        compute_segnet_constraint_loss,
    )

    N = gt_frames_batch.shape[0]
    H, W = gt_frames_batch.shape[1], gt_frames_batch.shape[2]

    # Create sparse perturbation module
    perturbation = SparsePatchPerturbation(
        n_frames=N,
        height=H,
        width=W,
        patch_locations=patch_locations,
        perturbation_budget=perturbation_budget,
    ).to(device)

    # GT frames on device (not optimized, just the base)
    gt_base = gt_frames_batch.to(device).detach()

    # Move targets to device
    gt_pose_dev = gt_pose_targets.to(device)
    gt_masks_dev = gt_seg_masks.to(device)

    # Track global best across restarts.
    # Initialize best_delta_state to zeros (GT unperturbed = safest fallback).
    best_loss = float("inf")
    best_delta_state = torch.zeros_like(perturbation.delta.data)

    for restart in range(n_restarts):
        # Stochastic restart: add calibrated noise to perturbation
        if restart > 0:
            with torch.no_grad():
                noise = torch.randn_like(perturbation.delta) * restart_noise
                # Apply mask to noise (only perturb active patches)
                noise = noise * perturbation.mask
                perturbation.delta.data = best_delta_state.clone() + noise

        optimizer = torch.optim.LBFGS(
            [perturbation.delta],
            lr=lbfgs_lr,
            max_iter=1,  # We control the outer loop
            history_size=lbfgs_history,
            line_search_fn="strong_wolfe",
        )

        for step in range(steps_per_restart):
            def closure():
                optimizer.zero_grad()
                perturbed = perturbation(gt_base)

                # PoseNet preservation: keep pose output matching GT
                pose_loss = compute_posenet_constraint_loss(
                    perturbed, gt_pose_dev, posenet, batch_size=32,
                )

                # SegNet preservation: keep segmentation matching GT
                seg_loss = compute_segnet_constraint_loss(
                    perturbed, gt_masks_dev, segnet, batch_size=32,
                )

                # Rate: encourage compressibility in perturbed regions
                compress_loss = compute_compressibility_loss(perturbed)

                total = (
                    pose_weight * pose_loss
                    + seg_weight * seg_loss
                    + rate_weight * compress_loss
                )

                total.backward()
                return total

            loss = optimizer.step(closure)

            if loss is None:
                loss_val = float("inf")
            elif isinstance(loss, torch.Tensor):
                loss_val = loss.item()
            else:
                loss_val = float(loss)

            if step % log_every == 0 or step == steps_per_restart - 1:
                # Re-evaluate for logging (closure may have been called multiple times by LBFGS)
                with torch.no_grad():
                    perturbed = perturbation(gt_base)
                    p_loss = compute_posenet_constraint_loss(perturbed, gt_pose_dev, posenet, batch_size=32).item()
                    s_loss = compute_segnet_constraint_loss(perturbed, gt_masks_dev, segnet, batch_size=32).item()
                    c_loss = compute_compressibility_loss(perturbed).item()
                print(f"  [restart {restart + 1}/{n_restarts}] step {step + 1}/{steps_per_restart}: "
                      f"pose={p_loss:.6f} seg={s_loss:.4f} compress={c_loss:.4f}")

            # Track best (skip NaN — L-BFGS can produce NaN on degenerate line search)
            if math.isfinite(loss_val) and loss_val < best_loss:
                best_loss = loss_val
                best_delta_state = perturbation.delta.data.detach().clone()

        print(f"  [restart {restart + 1}/{n_restarts}] best_loss={best_loss:.6f}")

    # Apply best perturbation
    with torch.no_grad():
        perturbation.delta.data = best_delta_state
        result = perturbation(gt_base)
        result = result.round().clamp(0.0, 255.0)

    return result.cpu()



def main():
    args = parse_args()

    if args.smoke:
        args.n_frames = 20
        args.n_restarts = 2
        args.steps_per_restart = 20
        args.batch_pairs = 10
        print("[smoke] Smoke test: 20 frames, 2 restarts, 20 steps/restart")

    # Ensure even frame count
    args.n_frames = args.n_frames - (args.n_frames % 2)

    root = find_project_root()
    device = torch.device(args.device)
    upstream = Path(args.upstream) if args.upstream else root / "upstream"

    if args.output_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%S")
        args.output_dir = str(root / "experiments" / "results" / f"gt_sparse_tto_{ts}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = args.video or str(upstream / "videos" / "0.mkv")

    print(f"[config] device={device}, n_frames={args.n_frames}")
    print(f"[config] n_patches={args.n_patches}, patch_size={args.patch_size}")
    print(f"[config] n_restarts={args.n_restarts}, steps/restart={args.steps_per_restart}")
    print(f"[config] lbfgs_lr={args.lbfgs_lr}, lbfgs_history={args.lbfgs_history}")
    print(f"[config] restart_noise={args.restart_noise}, perturbation_budget={args.perturbation_budget}")
    print(f"[config] pose_weight={args.pose_weight}, seg_weight={args.seg_weight}, "
          f"rate_weight={args.rate_weight}")

    t_total_start = time.monotonic()

    # ── Step 1: Load scorers ─────────────────────────────────────────────
    print("\n[1/6] Loading scorers...")
    t0 = time.monotonic()
    from tac.scorer import load_default_scorers
    posenet, segnet = load_default_scorers(upstream, device=str(device))
    print(f"[1/6] Scorers loaded in {time.monotonic() - t0:.1f}s")

    # ── Step 2: Decode GT video ──────────────────────────────────────────
    print(f"\n[2/6] Decoding GT video ({args.n_frames} frames)...")
    t0 = time.monotonic()
    from tac.data import load_gt_video

    gt_frames = load_gt_video(video_path, n_frames=args.n_frames)
    args.n_frames = len(gt_frames)
    print(f"[2/6] Decoded {args.n_frames} frames ({gt_frames[0].shape}) in {time.monotonic() - t0:.1f}s")

    # ── Step 3: Compute or load sensitivity map ──────────────────────────
    if args.sensitivity_map and Path(args.sensitivity_map).exists():
        print(f"\n[3/6] Loading precomputed sensitivity map: {args.sensitivity_map}")
        sensitivity_map = torch.load(args.sensitivity_map, map_location="cpu", weights_only=True)
    else:
        print(f"\n[3/6] Computing PoseNet sensitivity map ({args.n_frames // 2} pairs)...")
        t0 = time.monotonic()
        sensitivity_map = compute_sensitivity_map_live(
            gt_frames, posenet, device, batch_pairs=4,
        )
        dt = time.monotonic() - t0
        print(f"[3/6] Sensitivity map computed in {dt:.1f}s")
        # Save for reuse
        torch.save(sensitivity_map, output_dir / "sensitivity_map.pt")
        print(f"[3/6] Saved to {output_dir / 'sensitivity_map.pt'}")

    print(f"[3/6] Sensitivity: shape={sensitivity_map.shape}, "
          f"min={sensitivity_map.min():.6f}, max={sensitivity_map.max():.6f}")

    # ── Step 4: Select insensitive patches ───────────────────────────────
    print(f"\n[4/6] Selecting {args.n_patches} insensitive patches...")
    patch_locations = select_patches_for_frames(
        sensitivity_map, n_patches=args.n_patches, patch_size=args.patch_size,
    )

    # ── Step 5: Extract GT targets ───────────────────────────────────────
    print(f"\n[5/6] Extracting GT targets...")
    t0 = time.monotonic()

    gt_masks = extract_gt_masks(gt_frames, segnet, device)
    gt_pose = extract_gt_pose_targets(gt_frames, posenet, device)
    print(f"[5/6] Extracted {gt_masks.shape[0]} masks, {gt_pose.shape[0]} pose targets "
          f"in {time.monotonic() - t0:.1f}s")

    # ── Baseline: GT frames scored against GT (should be near 0) ─────────
    print("\n[...] Computing GT baseline proxy score...")
    gt_frames_tensor = torch.stack(gt_frames).float()
    gt_baseline = compute_proxy_score(
        gt_frames_tensor, gt_frames, posenet, segnet, device,
        simulate_resize=args.simulate_resize,
    )
    print(f"[baseline] GT score={gt_baseline['score']:.6f} | "
          f"seg={gt_baseline['seg']:.6f} | pose={gt_baseline['pose']:.6f}")

    # ── Step 6: Run sparse patch TTO in batches ──────────────────────────
    print(f"\n[6/6] Running GT-sparse TTO ({args.n_restarts} restarts x "
          f"{args.steps_per_restart} steps)...")
    t0 = time.monotonic()

    N = args.n_frames
    P = N // 2
    n_batches = math.ceil(P / args.batch_pairs)
    all_optimized = torch.zeros_like(gt_frames_tensor)

    for batch_idx in range(n_batches):
        pair_start = batch_idx * args.batch_pairs
        pair_end = min(pair_start + args.batch_pairs, P)
        frame_start = 2 * pair_start
        frame_end = 2 * pair_end
        n_pairs_this = pair_end - pair_start

        print(f"\n[6/6] Batch {batch_idx + 1}/{n_batches}: "
              f"pairs [{pair_start}:{pair_end}] = {n_pairs_this} pairs")

        batch_frames = gt_frames_tensor[frame_start:frame_end]
        batch_pose = gt_pose[pair_start:pair_end]
        batch_masks = gt_masks[frame_start:frame_end]

        batch_result = run_sparse_tto_batch(
            gt_frames_batch=batch_frames,
            patch_locations=patch_locations,
            posenet=posenet,
            segnet=segnet,
            gt_pose_targets=batch_pose,
            gt_seg_masks=batch_masks,
            device=device,
            steps_per_restart=args.steps_per_restart,
            n_restarts=args.n_restarts,
            lbfgs_lr=args.lbfgs_lr,
            lbfgs_history=args.lbfgs_history,
            restart_noise=args.restart_noise,
            seg_weight=args.seg_weight,
            pose_weight=args.pose_weight,
            rate_weight=args.rate_weight,
            perturbation_budget=args.perturbation_budget,
        )

        all_optimized[frame_start:frame_end] = batch_result

        # Save batch checkpoint
        ckpt_path = output_dir / f"sparse_tto_batch_{batch_idx:03d}.pt"
        torch.save(batch_result, ckpt_path)

        # Free memory
        del batch_result
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    t_tto = time.monotonic() - t0
    print(f"\n[6/6] Sparse TTO completed in {t_tto:.1f}s ({t_tto / N:.2f}s/frame)")

    # ── Proxy score on optimized frames ──────────────────────────────────
    print("\n[eval] Computing proxy score on optimized frames...")
    optimized_result = compute_proxy_score(
        all_optimized, gt_frames, posenet, segnet, device,
        simulate_resize=args.simulate_resize,
    )

    t_total = time.monotonic() - t_total_start

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("GT-SPARSE TTO RESULTS")
    print(f"{'=' * 72}")
    print(f"  GT Baseline (ground truth):")
    print(f"    score = {gt_baseline['score']:.6f}")
    print(f"    seg   = {gt_baseline['seg']:.6f}  ({gt_baseline['seg_contribution']:.6f})")
    print(f"    pose  = {gt_baseline['pose']:.6f}  ({gt_baseline['pose_contribution']:.6f})")
    print(f"  After Sparse TTO:")
    print(f"    score = {optimized_result['score']:.6f}")
    print(f"    seg   = {optimized_result['seg']:.6f}  ({optimized_result['seg_contribution']:.6f})")
    print(f"    pose  = {optimized_result['pose']:.6f}  ({optimized_result['pose_contribution']:.6f})")
    print(f"  Delta from GT:")
    d_score = optimized_result['score'] - gt_baseline['score']
    d_pose = optimized_result['pose'] - gt_baseline['pose']
    d_seg = optimized_result['seg'] - gt_baseline['seg']
    print(f"    score: {d_score:+.6f} ({'regression' if d_score > 0 else 'improved'})")
    print(f"    pose:  {d_pose:+.6f}")
    print(f"    seg:   {d_seg:+.6f}")
    print(f"  Config:")
    print(f"    patches: {args.n_patches} x {args.patch_size}x{args.patch_size}")
    print(f"    restarts: {args.n_restarts} x {args.steps_per_restart} steps")
    print(f"    perturbation budget: +/-{args.perturbation_budget}")
    print(f"  Timing:")
    print(f"    total = {t_total:.1f}s | TTO = {t_tto:.1f}s")
    print(f"{'=' * 72}")

    # ── Save results ─────────────────────────────────────────────────────
    torch.save(all_optimized.to(torch.uint8), output_dir / "sparse_tto_frames.pt")
    print(f"\n[save] Frames: {output_dir / 'sparse_tto_frames.pt'}")

    results = {
        "gt_baseline": gt_baseline,
        "optimized": optimized_result,
        "delta": {"score": d_score, "pose": d_pose, "seg": d_seg},
        "config": {
            "n_frames": args.n_frames,
            "n_patches": args.n_patches,
            "patch_size": args.patch_size,
            "n_restarts": args.n_restarts,
            "steps_per_restart": args.steps_per_restart,
            "lbfgs_lr": args.lbfgs_lr,
            "restart_noise": args.restart_noise,
            "perturbation_budget": args.perturbation_budget,
            "pose_weight": args.pose_weight,
            "seg_weight": args.seg_weight,
            "rate_weight": args.rate_weight,
        },
        "timing": {"total_s": round(t_total, 2), "tto_s": round(t_tto, 2)},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] Results: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
