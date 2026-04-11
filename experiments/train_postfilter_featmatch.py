#!/usr/bin/env python
# ============================================================================
# LEGACY — This script predates the tac library and is superseded by:
#   python experiments/train_tac.py --profile proven_baseline --loss-mode feature_match
# Unique logic has been migrated to src/tac/. Kept for git history reference.
# ============================================================================
"""Train a post-filter with PoseNet feature-matching loss + EMA + QAT.

The baseline saliency trainer minimizes only the *scalar* PoseNet
distortion (a 6-dim MSE between pose vectors).  That gives a relatively
weak gradient signal — only 6 numbers per pair.  This script adds a
**perceptual / feature-matching** term that minimizes the distance
between PoseNet's intermediate 512-d *summary* features for the
reconstructed pair vs the ground-truth pair.

Why this should help:

* PoseNet's summarizer output is a high-dimensional embedding that
  captures everything the head uses to compute pose.  Matching it
  forces the post-filter to preserve features the network actually
  cares about, not just the final scalar.
* It's analogous to perceptual / VGG-feature losses in image
  super-resolution — except tuned to the *exact* model that judges us.

We layer this on top of the QAT + EMA training pipeline (so all the
stability improvements still apply) and keep the existing saliency
reconstruction penalty for SegNet protection.

Usage::

    cd /tmp/pact-mine
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy python -u experiments/train_postfilter_featmatch.py \\
        --alpha 20 --feat-lambda 0.5 --epochs 120
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from pathlib import Path

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
    compute_saliency_reconstruction_loss,
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


# ── Feature-matching hook on PoseNet ─────────────────────────────────────


class PoseNetFeatureCapture:
    """Capture PoseNet's penultimate (post-summarizer) feature vector
    for each forward pass via a hook.
    """

    def __init__(self, posenet: nn.Module):
        self.posenet = posenet
        self.feat: torch.Tensor | None = None
        self._handle = posenet.summarizer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.feat = output

    def close(self):
        self._handle.remove()


def compute_loss_with_featmatch(
    filtered_pair_hwc, gt_pair_hwc, comp_pair_hwc,
    posenet, segnet, capture: PoseNetFeatureCapture,
    sal_weights_pair, sal_lambda: float, feat_lambda: float,
):
    """Combined loss = scorer + sal_recon + feat_match.

    The feature-matching term is the L2 distance between PoseNet's
    summary embedding for the filtered pair and the ground-truth pair.
    """
    # Forward filtered pair through scorers (gradient flows)
    fx = filtered_pair_hwc.permute(0, 1, 4, 2, 3).contiguous()
    fp_in = posenet.preprocess_input(fx)
    fp_out = posenet(fp_in)
    f_feat = capture.feat  # captured during fp_out call

    fs_in = segnet.preprocess_input(fx)
    fs_out = segnet(fs_in)

    # Forward GT pair without gradients
    with torch.no_grad():
        gx = gt_pair_hwc.float().permute(0, 1, 4, 2, 3).contiguous()
        gp_in = posenet.preprocess_input(gx)
        gp_out = posenet(gp_in)
        g_feat = capture.feat.detach().clone()
        gs_in = segnet.preprocess_input(gx)
        gs_out = segnet(gs_in)

    # Standard scorer loss
    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()
    pred_soft = F.softmax(fs_out, dim=1)
    gt_soft = F.softmax(gs_out, dim=1)
    seg_dist = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()
    scorer_loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)

    # Feature-matching loss (PoseNet summary features, 512-d)
    # Normalize by feature norm so the magnitude is comparable across
    # examples and across α settings.
    feat_diff = f_feat - g_feat
    feat_norm = g_feat.detach().pow(2).mean().sqrt() + 1e-6
    feat_match_loss = feat_diff.pow(2).mean() / feat_norm

    # Saliency reconstruction penalty (unchanged)
    B, T, H, W, C = filtered_pair_hwc.shape
    filtered_bchw = filtered_pair_hwc.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
    comp_bchw = comp_pair_hwc.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2)
    sal_recon = compute_saliency_reconstruction_loss(filtered_bchw, comp_bchw, sal_weights_pair)

    total = scorer_loss + sal_lambda * sal_recon + feat_lambda * feat_match_loss
    return (
        total,
        scorer_loss.item(),
        pose_dist.item(),
        seg_dist.item(),
        sal_recon.item(),
        feat_match_loss.item(),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Feature-matching post-filter trainer")
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--sal-lambda", type=float, default=0.1)
    p.add_argument("--feat-lambda", type=float, default=0.5,
                   help="Weight for the PoseNet feature-matching loss")
    p.add_argument("--train-subsample", type=int, default=8)
    p.add_argument("--eval-subsample", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--tag", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    alpha = args.alpha
    tag = args.tag or f"featmatch_alpha{int(alpha)}_h{args.hidden}_fl{args.feat_lambda}"
    meta = normalize_postfilter_meta(args.hidden, args.kernel, alpha)

    print(f"[featmatch] device={DEVICE} alpha={alpha} hidden={args.hidden} "
          f"feat_lambda={args.feat_lambda} ema={args.ema_decay} clip={args.grad_clip}")

    print("[featmatch] Loading scorer models...")
    posenet, segnet = load_scorers(DEVICE)
    capture = PoseNetFeatureCapture(posenet)

    print("[featmatch] Decoding archive + GT...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]
    print(f"[featmatch] {n} frames")

    sal_all = load_saliency_weights(alpha, n, DEVICE)
    sal_pairs = []
    for i in range(0, n - 1, seq_len):
        if i + seq_len > n:
            break
        sal_pairs.append(sal_all[i:i + seq_len])

    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)
    n_pairs = len(comp_pairs)
    print(f"[featmatch] {n_pairs} frame pairs")

    del comp_frames, gt_frames, sal_all
    gc.collect()

    comp_pairs = [p.to(DEVICE) for p in comp_pairs]
    gt_pairs = [p.to(DEVICE) for p in gt_pairs]

    model = QATPostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    print(f"[featmatch] Model: {count_params(model)} params")

    ema = EMA(model, decay=args.ema_decay)

    eval_indices = list(range(0, n_pairs, args.eval_subsample))
    n_eval = len(eval_indices)

    # Baseline (no filter)
    print(f"[featmatch] Baseline on {n_eval} pairs...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            unfiltered = comp_pairs[idx].float()
            fx = unfiltered.permute(0, 1, 4, 2, 3).contiguous()
            fp_out = posenet(posenet.preprocess_input(fx))
            fs_out = segnet(segnet.preprocess_input(fx))
            gx = gt_pairs[idx].float().permute(0, 1, 4, 2, 3).contiguous()
            gp_out = posenet(posenet.preprocess_input(gx))
            gs_out = segnet(segnet.preprocess_input(gx))
            pd = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean().item()
            ps = F.softmax(fs_out, dim=1)
            gs = F.softmax(gs_out, dim=1)
            sd = (1.0 - (ps * gs).sum(dim=1).mean()).item()
            total_pose += pd
            total_seg += sd
    baseline_pose = total_pose / n_eval
    baseline_seg = total_seg / n_eval
    baseline_loss = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)
    print(f"[featmatch] Baseline: loss={baseline_loss:.4f} "
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
    print(f"[featmatch] Training: {args.epochs} epochs, {train_size} pairs/epoch")
    print(f"{'epoch':>5} {'total':>10} {'scorer':>10} {'pose':>12} {'seg':>12} "
          f"{'sal':>10} {'feat':>10} {'lr':>10}")
    print("-" * 88)

    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(n_pairs)[:train_size].tolist()
        ep_loss = ep_scorer = ep_pose = ep_seg = ep_sal = ep_feat = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(indices):
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            total, scorer, pd, sd, sal_recon, fmatch = compute_loss_with_featmatch(
                filtered, gt_pairs[idx], comp_pairs[idx],
                posenet, segnet, capture,
                sal_pairs[idx], args.sal_lambda, args.feat_lambda,
            )
            (total / args.accum_steps).backward()
            ep_loss += total.item()
            ep_scorer += scorer
            ep_pose += pd
            ep_seg += sd
            ep_sal += sal_recon
            ep_feat += fmatch

            if (step_i + 1) % args.accum_steps == 0 or (step_i + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            ns = len(indices)
            lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch + 1:>5} {ep_loss / ns:>10.4f} {ep_scorer / ns:>10.4f} "
                  f"{ep_pose / ns:>12.6f} {ep_seg / ns:>12.6f} "
                  f"{ep_sal / ns:>10.4f} {ep_feat / ns:>10.4f} {lr:>10.6f}")

    capture.close()
    ema.copy_to(model)
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32_path = OUTPUT_DIR / f"postfilter_{tag}_fp32.pt"
    int8_path = OUTPUT_DIR / f"postfilter_{tag}_int8.pt"
    torch.save(model.state_dict(), fp32_path)
    int8_size = save_model_int8(model, int8_path, meta=meta)
    print(f"\nSaved fp32: {fp32_path}")
    print(f"Saved int8: {int8_path} ({int8_size} bytes)")
    return {"tag": tag, "baseline_loss": baseline_loss}


if __name__ == "__main__":
    main()
