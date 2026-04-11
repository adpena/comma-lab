#!/usr/bin/env python
# ============================================================================
# LEGACY — This script predates the tac library and is superseded by:
#   python experiments/train_tac.py --profile film_conditioned_smoke
# Unique logic has been migrated to src/tac/. Kept for git history reference.
# ============================================================================
"""Train a FiLM-conditioned post-filter on top of the QAT+EMA recipe."""
from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_postfilter_qat_ema import EMA, FakeQuantSTE, save_best_checkpoint  # type: ignore
from train_postfilter_saliency import (  # type: ignore
    ARCHIVE_ZIP,
    DEFAULT_HIDDEN,
    DEFAULT_KERNEL,
    DEVICE,
    OUTPUT_DIR,
    SALIENCY_PATH,
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
)
from frame_utils import seq_len  # noqa: E402


def fake_quant(t: torch.Tensor) -> torch.Tensor:
    return FakeQuantSTE.apply(t)


class FiLMQATPostFilter(nn.Module):
    def __init__(self, hidden: int = DEFAULT_HIDDEN, kernel: int = DEFAULT_KERNEL):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.film = nn.Linear(3, hidden * 2, bias=True)
        self.act = nn.ReLU(inplace=False)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def _qconv(self, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        wq = fake_quant(conv.weight)
        bq = fake_quant(conv.bias) if conv.bias is not None else None
        return F.conv2d(x, wq, bq, padding=conv.padding, stride=conv.stride)

    def _descriptor(self, x: torch.Tensor) -> torch.Tensor:
        y = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
        y_norm = y / 255.0
        mean = y_norm.mean(dim=(2, 3))
        std = y_norm.std(dim=(2, 3), unbiased=False)
        dx = y_norm[..., :, 1:] - y_norm[..., :, :-1]
        dy = y_norm[..., 1:, :] - y_norm[..., :-1, :]
        edge = 0.5 * (dx.abs().mean(dim=(2, 3)) + dy.abs().mean(dim=(2, 3)))
        return torch.cat([mean, std, edge], dim=1)

    def _film_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fq = fake_quant(self.film.weight)
        bq = fake_quant(self.film.bias)
        film = F.linear(self._descriptor(x), fq, bq)
        gamma, beta = film.chunk(2, dim=1)
        gamma = 1.0 + 0.25 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = 8.0 * torch.tanh(beta).unsqueeze(-1).unsqueeze(-1)
        return gamma, beta

    def forward(self, x):
        gamma, beta = self._film_params(x)
        residual = self.act(self._qconv(self.conv1, x))
        residual = residual * gamma + beta
        residual = self.act(self._qconv(self.conv2, residual))
        residual = residual * gamma + beta
        residual = self._qconv(self.conv3, residual)
        return (x + residual).clamp(0, 255)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FiLM-conditioned QAT+EMA post-filter")
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--sal-lambda", type=float, default=0.1)
    p.add_argument("--train-subsample", type=int, default=8)
    p.add_argument("--eval-subsample", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--tag", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_arg_parser().parse_args(argv)
    alpha = args.alpha
    tag = args.tag or f"film_conditioned_alpha{int(alpha)}_h{args.hidden}"
    meta = normalize_postfilter_meta(args.hidden, args.kernel, alpha)
    meta["variant"] = "film_conditioned"

    print(f"[film] device={DEVICE} alpha={alpha} hidden={args.hidden} tag={tag}")
    posenet, segnet = load_scorers(DEVICE)

    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]
    sal_all = load_saliency_weights(alpha, n, DEVICE)
    sal_pairs = []
    for i in range(0, n - 1, seq_len):
        if i + seq_len > n:
            break
        sal_pairs.append(sal_all[i:i + seq_len])
    comp_pairs = [p.to(DEVICE) for p in build_pairs(comp_frames)]
    gt_pairs = [p.to(DEVICE) for p in build_pairs(gt_frames)]
    n_pairs = len(comp_pairs)
    print(f"[film] {n_pairs} frame pairs")
    del comp_frames, gt_frames, sal_all
    gc.collect()

    model = FiLMQATPostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    print(f"[film] Model: {count_params(model)} params")
    ema = EMA(model, decay=args.ema_decay)

    eval_indices = list(range(0, n_pairs, args.eval_subsample))
    baseline_pose = baseline_seg = 0.0
    with torch.no_grad():
        for idx in eval_indices:
            _, pd, sd = compute_pair_loss(comp_pairs[idx].float(), gt_pairs[idx], posenet, segnet)
            baseline_pose += pd
            baseline_seg += sd
    baseline_pose /= len(eval_indices)
    baseline_seg /= len(eval_indices)
    baseline_loss = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)
    print(f"[film] Baseline: loss={baseline_loss:.4f} pose={baseline_pose:.6f} seg={baseline_seg:.6f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def lr_at(epoch_idx: int) -> float:
        if epoch_idx < args.warmup_epochs:
            return (epoch_idx + 1) / max(1, args.warmup_epochs)
        progress = (epoch_idx - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress)) * (1 - 0.02) + 0.02

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    train_size = max(1, n_pairs // args.train_subsample)
    print(f"[film] Training: {args.epochs} epochs, {train_size} pairs/epoch")
    print(f"{'epoch':>5} {'total':>10} {'scorer':>10} {'pose':>12} {'seg':>12} {'sal_recon':>10} {'lr':>10}")
    print("-" * 78)

    best_scorer = float("inf")
    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(n_pairs)[:train_size].tolist()
        ep_loss = ep_scorer = ep_pose = ep_seg = ep_sal = 0.0
        optimizer.zero_grad()
        for step_i, idx in enumerate(indices):
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            total_loss, scorer_loss, pd, sd, sal_recon = compute_combined_loss(
                filtered, gt_pairs[idx], comp_pairs[idx], posenet, segnet, sal_pairs[idx], args.sal_lambda
            )
            (total_loss / args.accum_steps).backward()
            ep_loss += total_loss.item()
            ep_scorer += scorer_loss
            ep_pose += pd
            ep_seg += sd
            ep_sal += sal_recon
            if (step_i + 1) % args.accum_steps == 0 or (step_i + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)
        scheduler.step()
        avg_scorer = ep_scorer / len(indices)
        if epoch == 0 or avg_scorer < best_scorer:
            best_scorer = avg_scorer
            save_best_checkpoint(
                model=model,
                ema=ema,
                output_dir=OUTPUT_DIR,
                tag=tag,
                meta=meta,
                epoch=epoch + 1,
                scorer=avg_scorer,
            )
        if (epoch + 1) % 10 == 0 or epoch == 0:
            n_steps = len(indices)
            lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch + 1:>5} {ep_loss / n_steps:>10.4f} {ep_scorer / n_steps:>10.4f} {ep_pose / n_steps:>12.6f} {ep_seg / n_steps:>12.6f} {ep_sal / n_steps:>10.4f} {lr:>10.6f}")
    return {"tag": tag, "baseline_loss": baseline_loss}


if __name__ == "__main__":
    main()
