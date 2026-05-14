#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Empirical SegNet boundary-band floor measurement (Tao's proposal).

Before pouring more compute into the SegNet attack lane, measure the
irreducible per-pixel disagreement floor. Two quantities:

1. **Boundary-band fraction**: what fraction of GT label pixels lie
   within 1 stride of a class boundary? Any compression that preserves
   pixel perceptual quality but shifts boundaries by <1 px will flip
   those pixels. This is the Bayes-risk lower bound.

2. **Self-disagreement under jitter**: if we slightly perturb the GT
   frame (1-pixel shift, or adding BT.601 quantization noise) and
   re-run SegNet, how much does the output change? This measures the
   intrinsic instability of SegNet at its operating point.

These two numbers together give us a realistic floor for SegNet
distortion. If our current 0.00576 is already near the floor, we should
stop chasing SegNet gains and pivot. If there's 30-50% headroom, keep
pushing the SegNet-attack lane.

Usage::

    cd /tmp/pact-mine
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy python -u experiments/segnet_boundary_floor.py
"""
from __future__ import annotations

import gc
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.data import build_pairs, decode_video
from tac.scorer import detect_device, load_scorers
from tac.proxy_eval import _default_paths

_PROJECT, _UPSTREAM, VIDEOS_DIR, _LIVE_ARCHIVE, _LEGACY_ARCHIVE = _default_paths()
DEVICE = detect_device()
segnet_model_input_size = (512, 384)  # (W, H)


def compute_boundary_band_fraction(labels: torch.Tensor, kernel_size: int = 3) -> float:
    """Fraction of pixels that are within 1 stride of a class boundary.

    labels: (B, H, W) int64 class labels
    Returns: float in [0, 1]
    """
    B, H, W = labels.shape
    # A pixel is in the boundary band if any of its 8-neighbors has a different label.
    # We compute this via a max/min dilation trick:
    labels_f = labels.float().unsqueeze(1)  # (B, 1, H, W)
    max_pool = F.max_pool2d(labels_f, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    min_pool = -F.max_pool2d(-labels_f, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    is_boundary = (max_pool != min_pool).float()
    return is_boundary.mean().item()


def measure_self_disagreement(
    segnet: nn.Module,
    gt_pair: torch.Tensor,  # (1, 2, H, W, 3) uint8 float
    jitter: float = 0.5,
    n_trials: int = 5,
    device: str = "cpu",
) -> float:
    """Run SegNet on the GT pair, then on GT+noise, measure argmax disagreement.

    jitter is the std of uniform noise added to pixel values (in LSB units).
    """
    gt_pair = gt_pair.float()
    with torch.no_grad():
        # Reference: clean GT
        x_ref = gt_pair.permute(0, 1, 4, 2, 3).contiguous()
        in_ref = segnet.preprocess_input(x_ref)
        out_ref = segnet(in_ref)
        labels_ref = out_ref.argmax(dim=1)  # (B, H_seg, W_seg)

        disagreements = []
        for trial in range(n_trials):
            noise = (torch.rand_like(gt_pair) - 0.5) * 2 * jitter
            x_noisy = (gt_pair + noise).clamp(0, 255)
            x_noisy = x_noisy.permute(0, 1, 4, 2, 3).contiguous()
            in_noisy = segnet.preprocess_input(x_noisy)
            out_noisy = segnet(in_noisy)
            labels_noisy = out_noisy.argmax(dim=1)
            disagree = (labels_ref != labels_noisy).float().mean().item()
            disagreements.append(disagree)
        return float(np.mean(disagreements))


def main():
    print(f"[segnet-floor] device={DEVICE}")
    posenet, segnet = load_scorers(DEVICE)

    print("[segnet-floor] Loading GT video...")
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = len(gt_frames)
    gt_pairs = build_pairs(gt_frames)
    n_pairs = len(gt_pairs)
    print(f"[segnet-floor] {n_pairs} frame pairs")
    del gt_frames
    gc.collect()

    # Subsample for speed
    n_analyze = 60
    indices = np.linspace(0, n_pairs - 1, n_analyze, dtype=int).tolist()

    # ── Pass 1: boundary-band fraction of GT labels ────────────────────
    print(f"\n[segnet-floor] Computing boundary-band fraction on {n_analyze} pairs...")
    boundary_fracs = []
    for i, idx in enumerate(indices):
        gt_pair = gt_pairs[idx].to(DEVICE).float()
        with torch.no_grad():
            x = gt_pair.permute(0, 1, 4, 2, 3).contiguous()
            in_seg = segnet.preprocess_input(x)  # (B, 3, H, W)
            out = segnet(in_seg)  # (B, 5, H, W)
            labels = out.argmax(dim=1)  # (B, H, W)
            frac = compute_boundary_band_fraction(labels, kernel_size=3)
            boundary_fracs.append(frac)
    boundary_frac_mean = float(np.mean(boundary_fracs))
    boundary_frac_median = float(np.median(boundary_fracs))
    print(f"[segnet-floor] Boundary fraction: mean={boundary_frac_mean:.4f}  "
          f"median={boundary_frac_median:.4f}")

    # ── Pass 2: self-disagreement under small jitter ───────────────────
    print("\n[segnet-floor] Measuring self-disagreement under pixel jitter...")
    jitter_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
    jitter_results = {}
    for jitter in jitter_levels:
        disagreements = []
        for idx in indices[:30]:  # smaller sample for speed
            gt_pair = gt_pairs[idx].to(DEVICE).float()
            d = measure_self_disagreement(segnet, gt_pair, jitter=jitter, n_trials=3, device=str(DEVICE))
            disagreements.append(d)
        mean_d = float(np.mean(disagreements))
        median_d = float(np.median(disagreements))
        jitter_results[jitter] = {"mean": mean_d, "median": median_d}
        print(f"  jitter={jitter:>5.2f} LSB: mean_disagree={mean_d:.6f}  median={median_d:.6f}")

    # ── Interpretation ────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("INTERPRETATION")
    print(f"{'=' * 78}")
    print("Current promoted SegNet distortion: 0.005764")
    print(f"Boundary-band fraction of GT labels: {boundary_frac_mean:.4f}")
    print(f"Self-disagreement at jitter=0.5 LSB: {jitter_results[0.5]['mean']:.6f}")
    print(f"Self-disagreement at jitter=1.0 LSB: {jitter_results[1.0]['mean']:.6f}")
    print("")
    print("A reasonable SegNet floor is the self-disagreement at the LSB scale")
    print("corresponding to the compressor's quantization noise (~0.5-1.0 LSB).")

    # Tao's hypothesis: boundary fraction * intrinsic flip rate = floor
    tao_floor = boundary_frac_mean * 0.05  # 5% intrinsic boundary noise
    print("")
    print(f"Tao's closed-form floor (5% boundary noise): {tao_floor:.6f}")
    print(f"Empirical jitter-based floor (jitter=0.5):   {jitter_results[0.5]['mean']:.6f}")
    print("")

    current = 0.005764
    empirical_floor = jitter_results[0.5]['mean']
    headroom = current - empirical_floor
    headroom_frac = headroom / current if current > 0 else 0
    print(f"Current SegNet:  {current:.6f}")
    print(f"Empirical floor: {empirical_floor:.6f}")
    print(f"Headroom:        {headroom:.6f}  ({headroom_frac * 100:.1f}% of current)")
    print("")
    if headroom_frac > 0.3:
        print("✓ Plenty of SegNet headroom. The SegNet-attack lane should continue.")
    elif headroom_frac > 0.1:
        print("○ Moderate SegNet headroom. Worth pushing but with diminishing returns.")
    else:
        print("⚠ SegNet is near its empirical floor. Pivot compute elsewhere.")

    result = {
        "current_seg_distortion": current,
        "boundary_frac_mean": boundary_frac_mean,
        "boundary_frac_median": boundary_frac_median,
        "jitter_disagreements": jitter_results,
        "tao_closed_form_floor": tao_floor,
        "empirical_floor_jitter_0p5": empirical_floor,
        "headroom": headroom,
        "headroom_fraction": headroom_frac,
        "n_pairs_analyzed": n_analyze,
    }
    print(f"\nJSON:\n{json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    main()
