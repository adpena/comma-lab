#!/usr/bin/env python
"""Compute the mathematical minimum PoseNet distortion achievable by pixel correction.

This is NOT a training run. It's a closed-form optimization that computes,
for each decoded frame pair, the **optimal pixel correction** that minimizes
the PoseNet distortion under a linear approximation around the decoded frame.

The math:

  Let x = decoded pixels (flattened), y = pose output (6-dim).
  Let J = dy/dx evaluated at x = R^(6 x N) where N = 2*3*874*1164 ~ 6.1M pixels.
  The PoseNet distortion we minimize is:

    d(x) = ||pose(x) - pose_gt||^2

  Linearized around x_decoded:

    pose(x_decoded + delta) ~= pose(x_decoded) + J @ delta

  Setting gradient to zero for the minimum of d w.r.t. delta:

    J^T @ (pose(x_decoded) + J @ delta - pose_gt) = 0
    J^T J @ delta = J^T (pose_gt - pose(x_decoded))
    delta = (J^T J)^+ J^T (pose_gt - pose(x_decoded))

  Since J has shape (6, N) with N >> 6, J^T J is (N, N) which is intractable,
  but the minimum-norm solution is:

    delta = J^T (J J^T)^+ (pose_gt - pose(x_decoded))

  where J J^T is only (6, 6). This is the Moore-Penrose pseudoinverse
  closed-form, and it's O(6N) to compute per pair.

  This delta is the ABSOLUTE OPTIMAL pixel correction under the linear
  approximation. If it achieves pose distortion near 0, then the CNN
  approach has room to improve. If it achieves a large residual, that
  is the mathematical LOWER BOUND on what any pixel correction can do.

We then apply delta (possibly with step-size clipping to respect pixel
value ranges) and measure the resulting PoseNet distortion via the
real scorer forward pass, to see how well the linear approximation holds.

Usage::

    cd /tmp/pact-mine
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy python -u experiments/jacobian_optimal.py
"""
from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.data import build_pairs, decode_archive, decode_video
from tac.losses import scorer_forward_pair
from tac.scorer import detect_device, load_scorers
from tac.proxy_eval import _default_paths

_PROJECT, _UPSTREAM, VIDEOS_DIR, _LIVE_ARCHIVE, ARCHIVE_ZIP = _default_paths()
UPSTREAM = _UPSTREAM
DEVICE = detect_device()


@torch.no_grad()
def posenet_output_6dim(posenet: nn.Module, pair_uint8_btwhc: torch.Tensor) -> torch.Tensor:
    """Run posenet and return the 6-dim pose vector used for distortion.

    Input: (1, 2, H, W, 3) uint8 tensor
    Output: (6,) float tensor — the pose[:6] used by posenet.compute_distortion
    """
    x = pair_uint8_btwhc.float().permute(0, 1, 4, 2, 3).contiguous()  # (1, 2, 3, H, W)
    inp = posenet.preprocess_input(x)
    out = posenet(inp)
    return out["pose"][..., :6].squeeze()  # (6,)


def posenet_output_with_grad(posenet: nn.Module, pair_float: torch.Tensor) -> torch.Tensor:
    """Same as above but for a differentiable float tensor (shape (1, 2, H, W, 3))."""
    x = pair_float.permute(0, 1, 4, 2, 3).contiguous()  # (1, 2, 3, H, W)
    inp = posenet.preprocess_input(x)
    out = posenet(inp)
    return out["pose"][..., :6].squeeze()  # (6,)


def compute_jacobian(posenet: nn.Module, pair_float: torch.Tensor) -> torch.Tensor:
    """Compute J = dy/dx where y is 6-dim pose and x is flattened pair pixels.

    Uses torch.autograd.grad with 6 backward passes (one per output dim).
    This is O(6 * forward_cost) which is acceptable for our 600 pairs.

    Returns: J of shape (6, N) where N = prod(pair_float.shape[1:]).
    """
    pair_float = pair_float.detach().clone().requires_grad_(True)
    y = posenet_output_with_grad(posenet, pair_float)  # (6,)
    N = pair_float.numel()
    J = torch.zeros(6, N, device=pair_float.device, dtype=pair_float.dtype)
    for k in range(6):
        grad = torch.autograd.grad(
            y[k], pair_float, retain_graph=(k < 5),
            create_graph=False,
        )[0]
        J[k] = grad.reshape(-1)
    return J, y.detach()


def optimal_correction(J: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Compute the minimum-norm delta that drives (pose(x) + J @ delta) = pose_gt.

    J: (6, N)
    residual: (6,) = pose_gt - pose_decoded
    Returns: delta (N,) minimum-norm solution such that J @ delta ~= residual.
    """
    # delta = J^T (J J^T)^+ residual
    # J J^T is (6, 6), easy to invert/pinv
    JJT = J @ J.T  # (6, 6)
    # Add small regularization to avoid numerical issues
    eps = 1e-6
    JJT_inv = torch.linalg.pinv(JJT + eps * torch.eye(6, device=J.device, dtype=J.dtype))
    delta = J.T @ (JJT_inv @ residual)  # (N,)
    return delta


def main():
    print(f"[jacobian] device={DEVICE}")
    print(f"[jacobian] Loading PoseNet (only — we don't need SegNet here)...")
    posenet, segnet = load_scorers(DEVICE)

    print(f"[jacobian] Decoding archive + GT...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]
    print(f"[jacobian] {n} frames")

    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)
    n_pairs = min(len(comp_pairs), len(gt_pairs))
    print(f"[jacobian] {n_pairs} frame pairs")

    del comp_frames, gt_frames
    gc.collect()

    # We evaluate on a subsample for speed. Each pair takes ~6 backward passes
    # of posenet which is ~1-2 seconds on MPS.
    subsample = 4  # every 4th pair = 150 pairs total
    indices = list(range(0, n_pairs, subsample))
    print(f"[jacobian] Analyzing {len(indices)}/{n_pairs} pairs (stride={subsample})")

    # Baseline: pose distortion without any correction
    print(f"\n[jacobian] Baseline (no correction):")
    total_baseline_dist = 0.0
    total_optimal_dist = 0.0
    total_delta_norm = 0.0
    total_delta_max = 0.0
    delta_mean_pixel_change = 0.0
    applied_count = 0

    for i, idx in enumerate(indices):
        comp_pair = comp_pairs[idx].to(DEVICE).float()  # (1, 2, H, W, 3) float
        gt_pair = gt_pairs[idx].to(DEVICE).float()

        # Baseline distortion at the raw decoded pair
        with torch.no_grad():
            pose_comp = posenet_output_with_grad(posenet, comp_pair)
            pose_gt = posenet_output_with_grad(posenet, gt_pair)
            baseline_dist = (pose_comp - pose_gt).pow(2).mean().item()
        total_baseline_dist += baseline_dist

        # Compute Jacobian at the decoded pair
        J, pose_comp_grad = compute_jacobian(posenet, comp_pair)
        residual = (pose_gt - pose_comp_grad).detach()

        # Optimal correction (closed form)
        delta = optimal_correction(J, residual)  # (N,)
        delta_reshaped = delta.reshape(comp_pair.shape)

        # Norm stats
        delta_norm_l2 = delta.pow(2).mean().sqrt().item()
        delta_max = delta.abs().max().item()
        total_delta_norm += delta_norm_l2
        total_delta_max = max(total_delta_max, delta_max)

        # Apply correction. Clip to [0, 255] to stay in valid pixel range.
        corrected = (comp_pair + delta_reshaped).clamp(0, 255)

        # Count how many pixels actually changed meaningfully (>0.5 LSB)
        changed_frac = ((corrected - comp_pair).abs() > 0.5).float().mean().item()

        # Measure the PoseNet distortion on the corrected pair
        with torch.no_grad():
            pose_corrected = posenet_output_with_grad(posenet, corrected)
            optimal_dist = (pose_corrected - pose_gt).pow(2).mean().item()
        total_optimal_dist += optimal_dist
        delta_mean_pixel_change += changed_frac
        applied_count += 1

        if (i + 1) % 10 == 0:
            print(f"  pair {i+1:3d}/{len(indices)}: "
                  f"baseline={baseline_dist:.6f} optimal={optimal_dist:.6f} "
                  f"||δ||_rms={delta_norm_l2:.3f} max_δ={delta_max:.2f} "
                  f"frac_changed={changed_frac:.4f}")

    n_eval = applied_count
    avg_baseline = total_baseline_dist / n_eval
    avg_optimal = total_optimal_dist / n_eval
    avg_delta_norm = total_delta_norm / n_eval
    avg_frac_changed = delta_mean_pixel_change / n_eval

    print(f"\n{'=' * 70}")
    print(f"RESULTS: Closed-form Jacobian-optimal correction")
    print(f"{'=' * 70}")
    print(f"Baseline PoseNet distortion (uncorrected): {avg_baseline:.6f}")
    print(f"Optimal PoseNet distortion (J pseudoinv):  {avg_optimal:.6f}")
    print(f"Reduction: {(avg_baseline - avg_optimal) / avg_baseline * 100:.1f}%")
    print(f"")
    print(f"Correction statistics:")
    print(f"  Mean RMS delta: {avg_delta_norm:.3f}")
    print(f"  Max delta seen: {total_delta_max:.2f}")
    print(f"  Mean fraction of pixels changed (>0.5 LSB): {avg_frac_changed:.4f}")
    print(f"")
    print(f"Comparison with our current winner (h=32 long1000):")
    print(f"  CNN-based PoseNet:   0.048092")
    print(f"  Closed-form optimal: {avg_optimal:.6f}")
    print(f"  Gap:                 {0.048092 - avg_optimal:.6f}")
    print(f"")
    print(f"Score projection (holding SegNet=0.00576, rate=0.5754):")
    est_score = 100 * 0.00576 + math.sqrt(10 * avg_optimal) + 0.5754
    print(f"  Score if this PoseNet is achievable: {est_score:.4f}")
    print(f"  Current best (CNN):                  1.8453")
    print(f"  Implied headroom:                    {1.8453 - est_score:.4f}")

    result = {
        "avg_baseline_pose_dist": avg_baseline,
        "avg_optimal_pose_dist": avg_optimal,
        "reduction_pct": (avg_baseline - avg_optimal) / avg_baseline * 100,
        "avg_delta_rms": avg_delta_norm,
        "max_delta": total_delta_max,
        "avg_frac_pixels_changed": avg_frac_changed,
        "estimated_score_if_optimal": est_score,
        "current_cnn_score": 1.8453,
        "n_pairs_analyzed": n_eval,
    }
    print(f"\nJSON:\n{json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    main()
