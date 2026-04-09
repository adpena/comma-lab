#!/usr/bin/env python
"""Train a post-filter with Kalman filter weight averaging (user suggestion).

EMA treats every optimizer step as equally informative and blends with a
fixed decay. But late-epoch STE or QAT oscillations have much higher
variance than early-epoch smooth descent, which pollutes the running
average. A Kalman filter replaces the fixed decay with an **inverse-
variance weighted** update:

    For each parameter tensor, maintain a state estimate `mu` and
    uncertainty `sigma^2`. At each optimizer step:
        - Prediction step: sigma^2 <- sigma^2 + process_noise (small)
        - Observation: z = current model weights
        - Observation noise: estimated from recent gradient magnitude
          (high gradients -> high obs noise)
        - Kalman gain K = sigma^2 / (sigma^2 + obs_noise)
        - Update: mu <- mu + K * (z - mu)
                  sigma^2 <- (1 - K) * sigma^2

When the model is oscillating (high grad norms), the observation noise
is high, K is small, and mu barely changes — filtering out the noise.
When the model is descending smoothly (low grad norms), obs noise is
low, K approaches 1, and mu tracks the weights closely.

This is strictly more information-theoretic than EMA: EMA is the special
case where obs_noise is constant and equal to (1 - decay) * process_noise.

Layered on the winning QAT+saliency recipe, otherwise identical.

Usage::

    cd /tmp/pact-mine
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy python -u experiments/train_postfilter_kalman.py \\
        --alpha 20 --hidden 32 --epochs 1000 --tag kalman_long1000_h32
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_postfilter_saliency import (  # type: ignore
    ARCHIVE_ZIP,
    DEFAULT_HIDDEN,
    DEFAULT_KERNEL,
    DEVICE,
    OUTPUT_DIR,
    SALIENCY_PATH,
    UPSTREAM,
    VIDEOS_DIR,
    apply_filter_to_pair,
    build_pairs,
    compute_combined_loss,
    compute_pair_loss,
    count_params,
    decode_archive,
    decode_video,
    load_saliency_weights,
    load_scorers,
    normalize_postfilter_meta,
    save_model_int8,
)
from train_postfilter_qat_ema import QATPostFilter  # type: ignore
from frame_utils import seq_len  # noqa: E402


# ── Kalman weight filter ─────────────────────────────────────────────────


class KalmanWeightFilter:
    """Per-parameter scalar Kalman filter on model weights.

    Each tensor has a single scalar sigma^2 (uncertainty) rather than a
    full covariance — cheap and empirically sufficient. The observation
    noise is estimated from the L2 norm of the weight delta since the
    last update (large deltas suggest oscillation / high observation noise).
    """

    def __init__(
        self,
        model: nn.Module,
        process_noise: float = 1e-6,
        obs_noise_base: float = 1e-4,
        obs_noise_scale: float = 10.0,
    ):
        self.process_noise = process_noise
        self.obs_noise_base = obs_noise_base
        self.obs_noise_scale = obs_noise_scale
        # Shadow state (the filtered estimate)
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        # Per-tensor uncertainty (scalar sigma^2)
        self.sigma2: dict[str, float] = {
            k: 1.0 for k in self.shadow.keys()  # start uninformative
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if not v.dtype.is_floating_point:
                self.shadow[k].copy_(v)
                continue
            z = v.detach()
            # Prediction step: sigma^2 grows by process noise
            sigma2 = self.sigma2[k] + self.process_noise
            # Observation noise: proportional to weight delta magnitude
            # (high delta = oscillating / noisy update = high obs noise)
            delta = (z - self.shadow[k]).pow(2).mean().item()
            obs_noise = self.obs_noise_base + self.obs_noise_scale * delta
            # Kalman gain
            K = sigma2 / (sigma2 + obs_noise)
            # Update state and uncertainty
            self.shadow[k].mul_(1 - K).add_(z, alpha=K)
            self.sigma2[k] = (1 - K) * sigma2

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)


# ── Main ──────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kalman-filtered weight averaging")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--sal-lambda", type=float, default=0.1)
    p.add_argument("--train-subsample", type=int, default=8)
    p.add_argument("--eval-subsample", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--process-noise", type=float, default=1e-6)
    p.add_argument("--obs-noise-base", type=float, default=1e-4)
    p.add_argument("--obs-noise-scale", type=float, default=10.0)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--tag", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    alpha = args.alpha
    tag = args.tag or f"kalman_h{args.hidden}"
    meta = normalize_postfilter_meta(args.hidden, args.kernel, alpha)

    print(f"[kalman] device={DEVICE} alpha={alpha} hidden={args.hidden} tag={tag}")
    print(f"[kalman] process_noise={args.process_noise} obs_base={args.obs_noise_base} "
          f"obs_scale={args.obs_noise_scale}")

    print("[kalman] Loading scorer models...")
    posenet, segnet = load_scorers(DEVICE)

    print("[kalman] Decoding archive + ground truth...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]
    print(f"[kalman] {n} frames each")

    sal_all = load_saliency_weights(alpha, n, DEVICE)
    sal_pairs = []
    for i in range(0, n - 1, seq_len):
        if i + seq_len > n:
            break
        sal_pairs.append(sal_all[i:i + seq_len])

    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)
    n_pairs = len(comp_pairs)
    print(f"[kalman] {n_pairs} frame pairs")

    del comp_frames, gt_frames, sal_all
    gc.collect()

    comp_pairs = [p.to(DEVICE) for p in comp_pairs]
    gt_pairs = [p.to(DEVICE) for p in gt_pairs]

    model = QATPostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    print(f"[kalman] Model: {count_params(model)} params")

    kalman = KalmanWeightFilter(
        model,
        process_noise=args.process_noise,
        obs_noise_base=args.obs_noise_base,
        obs_noise_scale=args.obs_noise_scale,
    )

    eval_indices = list(range(0, n_pairs, args.eval_subsample))
    n_eval = len(eval_indices)

    # Baseline
    print(f"[kalman] Baseline on {n_eval}/{n_pairs} pairs...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            _, pd, sd = compute_pair_loss(
                comp_pairs[idx].float(), gt_pairs[idx], posenet, segnet
            )
            total_pose += pd
            total_seg += sd
    baseline_pose = total_pose / n_eval
    baseline_seg = total_seg / n_eval
    baseline_loss = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)
    print(f"[kalman] Baseline: loss={baseline_loss:.4f} "
          f"pose={baseline_pose:.6f} seg={baseline_seg:.6f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def lr_at(epoch_idx: int) -> float:
        if epoch_idx < args.warmup_epochs:
            return (epoch_idx + 1) / max(1, args.warmup_epochs)
        progress = (epoch_idx - args.warmup_epochs) / max(
            1, args.epochs - args.warmup_epochs
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress)) * (1 - 0.02) + 0.02

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    train_size = max(1, n_pairs // args.train_subsample)
    print(f"[kalman] Training: {args.epochs} epochs, {train_size} pairs/epoch")
    print(f"{'epoch':>5} {'total':>10} {'scorer':>10} {'pose':>12} "
          f"{'seg':>12} {'sal_recon':>10} {'lr':>10}")
    print("-" * 78)

    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(n_pairs)[:train_size].tolist()
        ep_loss = ep_scorer = ep_pose = ep_seg = ep_sal = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(indices):
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            total, scorer, pd, sd, sal_recon = compute_combined_loss(
                filtered, gt_pairs[idx], comp_pairs[idx],
                posenet, segnet, sal_pairs[idx], args.sal_lambda,
            )
            (total / args.accum_steps).backward()
            ep_loss += total.item()
            ep_scorer += scorer
            ep_pose += pd
            ep_seg += sd
            ep_sal += sal_recon

            if (step_i + 1) % args.accum_steps == 0 or (step_i + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                kalman.update(model)  # <- Kalman instead of EMA

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            ns = len(indices)
            lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch + 1:>5} {ep_loss / ns:>10.4f} {ep_scorer / ns:>10.4f} "
                  f"{ep_pose / ns:>12.6f} {ep_seg / ns:>12.6f} "
                  f"{ep_sal / ns:>10.4f} {lr:>10.6f}")

    kalman.copy_to(model)
    model.eval()

    print(f"\n[kalman] Final eval on Kalman-filtered weights ({n_eval} pairs)...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            _, pd, sd = compute_pair_loss(filtered, gt_pairs[idx], posenet, segnet)
            total_pose += pd
            total_seg += sd
    final_pose = total_pose / n_eval
    final_seg = total_seg / n_eval
    final_loss = 100.0 * final_seg + math.sqrt(10.0 * final_pose)

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {tag}")
    print(f"{'=' * 70}")
    print(f"Baseline: loss={baseline_loss:.4f}  "
          f"pose={baseline_pose:.6f}  seg={baseline_seg:.6f}")
    print(f"Filtered: loss={final_loss:.4f}  "
          f"pose={final_pose:.6f}  seg={final_seg:.6f}")
    delta = final_loss - baseline_loss
    print(f"Delta:    {delta:+.4f}")
    if delta < 0:
        print(f"*** IMPROVEMENT: {-delta:.4f} points ***")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32_path = OUTPUT_DIR / f"postfilter_{tag}_fp32.pt"
    int8_path = OUTPUT_DIR / f"postfilter_{tag}_int8.pt"
    torch.save(model.state_dict(), fp32_path)
    int8_size = save_model_int8(model, int8_path, meta=meta)
    print(f"\nSaved fp32: {fp32_path}")
    print(f"Saved int8: {int8_path} ({int8_size} bytes)")
    return {"tag": tag, "baseline_loss": baseline_loss, "final_loss": final_loss}


if __name__ == "__main__":
    main()
