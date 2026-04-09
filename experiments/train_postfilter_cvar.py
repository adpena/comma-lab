#!/usr/bin/env python
"""Train a post-filter with worst-decile CVaR objective (von Neumann).

Standard training averages the loss across all sampled pairs, which means
outliers (hard pairs) can be swamped by easy pairs. The distribution of
per-pair scorer losses is heavy-tailed: a small fraction of pairs
dominates the final score. A CVaR (Conditional Value at Risk) objective
only averages the **worst alpha fraction** of pair losses, forcing the
optimizer to fix the hard cases.

This is distributionally robust optimization as proposed by the expert
panel on 2026-04-09. It directly addresses the train/eval gap that
Monte Carlo overfitting revealed — ensuring we optimize for the
distribution we actually get scored on, not the easy subset.

Concretely:
  CVaR_alpha(loss) = mean(top-alpha-worst pair losses per batch)

We use alpha=0.2 (worst 20% of pairs per epoch).

Layered on the QAT+EMA recipe that gave us 1.845. Expected improvement
is 0.005-0.015 from closing the train/eval gap; combined with SegNet
attack this could land us at 1.80 or below.

Usage::

    cd /tmp/pact-mine
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy python -u experiments/train_postfilter_cvar.py \\
        --alpha 20 --hidden 32 --epochs 1000 --cvar-alpha 0.2 \\
        --tag cvar_long1000_h32
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
from train_postfilter_qat_ema import EMA, QATPostFilter  # type: ignore
from frame_utils import seq_len  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CVaR post-filter trainer")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=20.0,
                   help="Saliency weighting alpha (inherited from base recipe)")
    p.add_argument("--sal-lambda", type=float, default=0.1)
    p.add_argument("--train-subsample", type=int, default=8)
    p.add_argument("--eval-subsample", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--cvar-alpha", type=float, default=0.2,
                   help="CVaR quantile: average only the worst alpha fraction")
    p.add_argument("--tag", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    alpha = args.alpha
    tag = args.tag or f"cvar_a{int(args.cvar_alpha*100)}_h{args.hidden}"
    meta = normalize_postfilter_meta(args.hidden, args.kernel, alpha)

    print(f"[cvar] device={DEVICE} alpha={alpha} cvar_alpha={args.cvar_alpha} "
          f"hidden={args.hidden} tag={tag}")

    print("[cvar] Loading scorer models...")
    posenet, segnet = load_scorers(DEVICE)

    print("[cvar] Decoding archive + ground truth...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]
    print(f"[cvar] {n} frames each")

    sal_all = load_saliency_weights(alpha, n, DEVICE)
    sal_pairs = []
    for i in range(0, n - 1, seq_len):
        if i + seq_len > n:
            break
        sal_pairs.append(sal_all[i:i + seq_len])

    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)
    n_pairs = len(comp_pairs)
    print(f"[cvar] {n_pairs} frame pairs")

    del comp_frames, gt_frames, sal_all
    gc.collect()

    comp_pairs = [p.to(DEVICE) for p in comp_pairs]
    gt_pairs = [p.to(DEVICE) for p in gt_pairs]

    model = QATPostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    print(f"[cvar] Model: {count_params(model)} params")

    ema = EMA(model, decay=args.ema_decay)

    eval_indices = list(range(0, n_pairs, args.eval_subsample))
    n_eval = len(eval_indices)

    print(f"[cvar] Baseline on {n_eval}/{n_pairs} pairs...")
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
    print(f"[cvar] Baseline: loss={baseline_loss:.4f} "
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
    print(f"[cvar] Training: {args.epochs} epochs, {train_size} pairs/epoch, "
          f"cvar_alpha={args.cvar_alpha} (worst {int(args.cvar_alpha*100)}%)")
    print(f"{'epoch':>5} {'total':>10} {'scorer':>10} {'pose':>12} "
          f"{'seg':>12} {'sal_recon':>10} {'cvar_k':>8} {'lr':>10}")
    print("-" * 88)

    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(n_pairs)[:train_size].tolist()

        # First pass: compute per-pair losses without backprop to rank them
        # (this is the CVaR selection step)
        pair_losses = []
        with torch.no_grad():
            for idx in indices:
                filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
                total_loss, _, _, _, _ = compute_combined_loss(
                    filtered, gt_pairs[idx], comp_pairs[idx],
                    posenet, segnet, sal_pairs[idx], args.sal_lambda,
                )
                pair_losses.append(total_loss.item())

        # Select worst cvar_alpha fraction
        pair_losses_t = torch.tensor(pair_losses)
        k = max(1, int(round(args.cvar_alpha * len(indices))))
        worst_pair_positions = torch.topk(pair_losses_t, k).indices.tolist()
        cvar_indices = [indices[i] for i in worst_pair_positions]

        # Second pass: backprop only on the worst pairs
        ep_loss = ep_scorer = ep_pose = ep_seg = ep_sal = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(cvar_indices):
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

            if (step_i + 1) % args.accum_steps == 0 or (step_i + 1) == len(cvar_indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            ns = len(cvar_indices)
            lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch + 1:>5} {ep_loss / ns:>10.4f} {ep_scorer / ns:>10.4f} "
                  f"{ep_pose / ns:>12.6f} {ep_seg / ns:>12.6f} "
                  f"{ep_sal / ns:>10.4f} {k:>8d} {lr:>10.6f}")

    ema.copy_to(model)
    model.eval()

    # Final eval (unconditional mean, not CVaR)
    print(f"\n[cvar] Final eval on EMA weights ({n_eval} pairs, unconditional mean)...")
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
