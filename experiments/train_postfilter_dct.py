#!/usr/bin/env python
# ============================================================================
# LEGACY — This script predates the tac library and is superseded by:
#   python experiments/train_tac.py --profile dct_midband_smoke
# Unique logic has been migrated to src/tac/. Kept for git history reference.
# ============================================================================
"""Train a tiny mid-frequency DCT-basis post-filter.

This is a research lane inspired by the residual-analysis finding that the
winning CNN writes most of its energy into mid frequencies. Instead of learning
pixel-space convolutions, this model:

1. splits each frame into fixed-size blocks,
2. transforms each block with a fixed orthonormal DCT,
3. applies learnable per-band gains only in the mid-frequency region,
4. reconstructs a pixel-space residual with the inverse DCT.

The model is intentionally tiny and non-shipping by default. It exists to test
whether making the spectral prior explicit improves convergence or efficiency.
"""
from __future__ import annotations

import argparse
import gc
import json
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
    DEVICE,
    OUTPUT_DIR,
    SALIENCY_PATH,
    VIDEOS_DIR,
    apply_filter_to_pair,
    compute_combined_loss,
    compute_pair_loss,
    count_params,
    decode_archive,
    decode_video,
    load_scorers,
    normalize_postfilter_meta,
    save_model_int8,
)
from train_postfilter_qat_ema import (  # type: ignore
    build_pair_start_indices,
    pair_from_frames,
    saliency_pair_at,
)
from frame_utils import seq_len  # noqa: E402


def orthonormal_dct_matrix(size: int) -> torch.Tensor:
    n = torch.arange(size, dtype=torch.float32)
    k = n.unsqueeze(1)
    matrix = torch.cos(math.pi / size * (n + 0.5) * k)
    matrix[0] *= 1.0 / math.sqrt(2.0)
    matrix *= math.sqrt(2.0 / size)
    return matrix


def build_mid_frequency_mask(size: int, *, low: float = 0.18, high: float = 0.72) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    radius = torch.sqrt(yy.float().pow(2) + xx.float().pow(2))
    radius /= radius.max().clamp(min=1.0)
    mask = ((radius >= low) & (radius <= high)).float()
    mask[0, 0] = 0.0
    return mask


class BlockDCTMidbandFilter(nn.Module):
    def __init__(self, block: int = 8, channels: int = 3, low: float = 0.18, high: float = 0.72):
        super().__init__()
        self.block = block
        self.channels = channels
        self.register_buffer("dct", orthonormal_dct_matrix(block))
        self.register_buffer("mid_mask", build_mid_frequency_mask(block, low=low, high=high))
        self.gain = nn.Parameter(torch.zeros(channels, block, block))
        self.bias = nn.Parameter(torch.zeros(channels, block, block))
        self.mix = nn.Parameter(torch.zeros(()))

    def _blockify(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        b, c, h, w = x.shape
        pad_h = (self.block - h % self.block) % self.block
        pad_w = (self.block - w % self.block) % self.block
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        hp, wp = x.shape[-2:]
        blocks = x.reshape(b, c, hp // self.block, self.block, wp // self.block, self.block)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        return blocks, pad_h, pad_w

    def _deblockify(self, blocks: torch.Tensor, *, pad_h: int, pad_w: int, orig_h: int, orig_w: int) -> torch.Tensor:
        b, c, nh, nw, _, _ = blocks.shape
        x = blocks.permute(0, 1, 2, 4, 3, 5).reshape(b, c, nh * self.block, nw * self.block)
        if pad_h or pad_w:
            x = x[..., :orig_h, :orig_w]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[-2:]
        blocks, pad_h, pad_w = self._blockify(x)
        flat = blocks.reshape(-1, self.block, self.block)
        coeff = torch.matmul(self.dct, flat)
        coeff = torch.matmul(coeff, self.dct.t())
        coeff = coeff.reshape(*blocks.shape)

        mask = self.mid_mask.view(1, 1, 1, 1, self.block, self.block)
        gain = torch.tanh(self.gain).view(1, self.channels, 1, 1, self.block, self.block)
        bias = (0.05 * torch.tanh(self.bias)).view(1, self.channels, 1, 1, self.block, self.block)
        delta_coeff = mask * (coeff * gain + bias)

        delta_flat = delta_coeff.reshape(-1, self.block, self.block)
        delta_blocks = torch.matmul(self.dct.t(), delta_flat)
        delta_blocks = torch.matmul(delta_blocks, self.dct)
        delta_blocks = delta_blocks.reshape_as(blocks)
        delta = self._deblockify(delta_blocks, pad_h=pad_h, pad_w=pad_w, orig_h=orig_h, orig_w=orig_w)
        return (x + torch.tanh(self.mix) * delta).clamp(0.0, 255.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mid-frequency DCT-basis post-filter experiment")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--sal-lambda", type=float, default=0.1)
    parser.add_argument("--train-subsample", type=int, default=8)
    parser.add_argument("--eval-subsample", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--tag", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_arg_parser().parse_args(argv)
    tag = args.tag or f"dct_midband_alpha{int(args.alpha)}_b{args.block}"
    meta = normalize_postfilter_meta(hidden=args.block, kernel=args.block, alpha=args.alpha)
    meta["variant"] = "dct_midband"
    meta["block"] = int(args.block)

    print(
        f"[dct] device={DEVICE} alpha={args.alpha} block={args.block} "
        f"epochs={args.epochs} tag={tag}"
    )

    print("[dct] Loading scorer models...")
    posenet, segnet = load_scorers(DEVICE)

    print("[dct] Decoding compressed archive + ground truth...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]
    pair_starts = build_pair_start_indices(n, seq_len)
    n_pairs = len(pair_starts)
    print(f"[dct] {n} frames each, {n_pairs} frame pairs")

    sal_base = torch.from_numpy(np.load(str(SALIENCY_PATH))).float()
    print(f"[dct] Saliency base: mean={sal_base.mean().item():.3f} max={sal_base.max().item():.1f}")

    model = BlockDCTMidbandFilter(block=args.block).to(DEVICE)
    param_count = count_params(model)
    print(f"[dct] Model: {param_count} params")

    eval_indices = list(range(0, n_pairs, args.eval_subsample))
    baseline_pose, baseline_seg = 0.0, 0.0
    with torch.no_grad():
        for start in eval_indices:
            comp_pair = pair_from_frames(comp_frames, pair_starts[start]).to(DEVICE).float()
            gt_pair = pair_from_frames(gt_frames, pair_starts[start]).to(DEVICE)
            _, pd, sd = compute_pair_loss(comp_pair, gt_pair, posenet, segnet)
            baseline_pose += pd
            baseline_seg += sd
    baseline_pose /= len(eval_indices)
    baseline_seg /= len(eval_indices)
    baseline_score = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)
    print(f"[dct] Baseline: score={baseline_score:.4f}, pose={baseline_pose:.6f}, seg={baseline_seg:.6f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_score = math.inf
    best_payload: dict[str, object] | None = None
    train_indices = list(range(0, n_pairs, args.train_subsample))

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(epoch)
        rng.shuffle(train_indices)
        optimizer.zero_grad(set_to_none=True)
        step_count = 0
        for pair_idx in train_indices:
            start = pair_starts[pair_idx]
            comp_pair = pair_from_frames(comp_frames, start).to(DEVICE)
            gt_pair = pair_from_frames(gt_frames, start).to(DEVICE)
            sal_pair = saliency_pair_at(sal_base, start_idx=start, alpha=args.alpha, device=DEVICE)
            filtered = apply_filter_to_pair(model, comp_pair, DEVICE)
            total_loss, _, _, _, _ = compute_combined_loss(
                filtered,
                gt_pair,
                comp_pair,
                posenet,
                segnet,
                sal_pair,
                args.sal_lambda,
            )
            (total_loss / args.accum_steps).backward()
            step_count += 1
            if step_count % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if step_count % args.accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if epoch % 10 != 0 and epoch != args.epochs:
            continue

        model.eval()
        total_pose, total_seg = 0.0, 0.0
        with torch.no_grad():
            for pair_idx in eval_indices:
                start = pair_starts[pair_idx]
                comp_pair = pair_from_frames(comp_frames, start).to(DEVICE)
                gt_pair = pair_from_frames(gt_frames, start).to(DEVICE)
                filtered = apply_filter_to_pair(model, comp_pair, DEVICE)
                _, pd, sd = compute_pair_loss(filtered, gt_pair, posenet, segnet)
                total_pose += pd
                total_seg += sd
        avg_pose = total_pose / len(eval_indices)
        avg_seg = total_seg / len(eval_indices)
        score = 100.0 * avg_seg + math.sqrt(10.0 * avg_pose)
        print(
            f"{epoch:>5d} {score:>10.4f} {avg_pose:>12.6f} "
            f"{avg_seg:>12.6f} {scheduler.get_last_lr()[0]:>10.6f}"
        )
        if score < best_score:
            best_score = score
            fp32_path = OUTPUT_DIR / f"postfilter_{tag}_best_fp32.pt"
            int8_path = OUTPUT_DIR / f"postfilter_{tag}_best_int8.pt"
            meta_path = OUTPUT_DIR / f"postfilter_{tag}_best_meta.json"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), fp32_path)
            int8_size = save_model_int8(model, int8_path, meta=meta)
            best_payload = {
                "epoch": epoch,
                "score": score,
                "pose": avg_pose,
                "seg": avg_seg,
                "fp32_path": str(fp32_path),
                "int8_path": str(int8_path),
                "int8_size": int8_size,
                "meta": meta,
            }
            meta_path.write_text(json.dumps(best_payload, indent=2))

    gc.collect()
    result = {
        "tag": tag,
        "baseline_score": baseline_score,
        "best": best_payload,
        "params": param_count,
    }
    print("[dct] JSON:")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
