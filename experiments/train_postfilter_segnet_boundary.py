#!/usr/bin/env python
"""SegNet boundary-band attack — council recommendation #2.

Modification of the SegNet STE attack that weights the cross-entropy loss
by proximity to SegNet class boundaries. The insight (Tao):

- SegNet scoring uses hard argmax disagreement
- Only pixels near class boundaries can actually flip
- 97.39% of pixels are deep inside a class and can't be changed
- Focusing the gradient on the 2.61% boundary band is 38x more efficient

The boundary mask is computed from the ground-truth SegNet output by finding
pixels where the argmax class changes within a 3-pixel dilation radius.

This uses the standard h=64 PostFilter (NOT PixelShuffle) to preserve
our PoseNet advantage.

Usage:
    .venv/bin/python experiments/train_postfilter_segnet_boundary.py \
        --hidden 64 --epochs 1000 --alpha 20 --boundary-weight 20 \
        --tag segnet_boundary_h64
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

# Import the base training infrastructure
sys.path.insert(0, str(Path(__file__).parent))
from train_postfilter_saliency import (
    PROJECT, ARCHIVE_ZIP, GT_VIDEO,
    DEVICE, PostFilter, QATPostFilter,
    decode_archive, decode_gt_video, build_pairs,
    load_scorer_models, compute_saliency_reconstruction_loss,
    pair_from_frames, maybe_to_device,
    save_best_checkpoint, evaluate_ema_model,
)


def compute_boundary_masks(gt_pairs, segnet, device="cpu"):
    """Compute SegNet class boundary masks from ground-truth pairs.

    For each GT pair, run SegNet to get class labels, then find pixels
    where the argmax class differs from any of its neighbors (3x3 dilation).
    These are the pixels where small changes can flip the SegNet score.

    Returns: list of (H, W) float tensors, one per pair. Value = 1.0 at
    boundaries, 0.0 elsewhere.
    """
    print("[boundary] Computing SegNet boundary masks from GT...")
    masks = []

    for i, pair in enumerate(gt_pairs):
        # pair is (1, 2, H, W, 3) uint8
        p = pair.to(device).float()
        # SegNet only uses the last frame (frame index 1)
        frame = p[:, 1]  # (1, H, W, 3)
        frame_chw = frame.permute(0, 3, 1, 2)  # (1, 3, H, W)

        with torch.no_grad():
            seg_out = segnet(frame_chw)  # (1, C, H, W)
            labels = seg_out.argmax(dim=1)  # (1, H, W)

        # Find boundaries: pixels where any neighbor has a different label
        # Use max-pool of (label != center) over a 5x5 window
        lab = labels.float().unsqueeze(1)  # (1, 1, H, W)

        # Dilate: check if any pixel in a 5x5 neighborhood differs
        # We do this by checking if the range within the window > 0
        max_pool = F.max_pool2d(lab, kernel_size=5, stride=1, padding=2)
        min_pool = -F.max_pool2d(-lab, kernel_size=5, stride=1, padding=2)
        boundary = (max_pool != min_pool).float().squeeze()  # (H, W)

        masks.append(boundary.cpu())

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(gt_pairs)}] boundary fraction: {boundary.mean():.4f}")

    # Stats
    all_fracs = [m.mean().item() for m in masks]
    avg_frac = sum(all_fracs) / len(all_fracs)
    print(f"[boundary] Average boundary fraction: {avg_frac:.4f} ({avg_frac*100:.2f}%)")
    return masks


def compute_pair_loss_segnet_boundary(
    filtered_pair_hwc, gt_pair_hwc, posenet, segnet,
    boundary_mask, boundary_weight=20.0,
):
    """Compute loss with boundary-weighted SegNet STE.

    The SegNet cross-entropy is weighted: boundary pixels get `boundary_weight`x
    more gradient than non-boundary pixels. This focuses learning on the pixels
    that can actually flip the scorer.
    """
    from train_postfilter_saliency import scorer_forward_pair

    B, T, H, W, C = filtered_pair_hwc.shape
    fx = filtered_pair_hwc
    gx = gt_pair_hwc

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)
    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

    # PoseNet: unchanged MSE
    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()

    # SegNet: boundary-weighted STE
    with torch.no_grad():
        gt_labels = gs_out.argmax(dim=1)  # (B, H_seg, W_seg)
        pred_labels = fs_out.argmax(dim=1)
        hard_disagree = (pred_labels != gt_labels).float().mean()

    B_seg, C_seg, H_seg, W_seg = fs_out.shape

    # Resize boundary mask to SegNet output resolution
    bm = boundary_mask.to(fs_out.device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    bm_resized = F.interpolate(bm, size=(H_seg, W_seg), mode="nearest").squeeze()  # (H_seg, W_seg)

    # Per-pixel weights: boundary gets boundary_weight, non-boundary gets 1.0
    pixel_weights = torch.where(bm_resized > 0.5, boundary_weight, 1.0)
    pixel_weights = pixel_weights / pixel_weights.mean()  # normalize so mean = 1

    # Weighted cross-entropy
    flat_labels = gt_labels.reshape(-1)
    flat_logits = fs_out.permute(0, 2, 3, 1).reshape(-1, C_seg)
    flat_weights = pixel_weights.expand(B_seg, -1, -1).reshape(-1)

    per_pixel_ce = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    soft_ce = (per_pixel_ce * flat_weights).mean()

    # STE: forward = hard_disagree, backward = weighted soft_ce
    seg_dist = soft_ce + (hard_disagree - soft_ce).detach()

    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), hard_disagree.item()


def main():
    parser = argparse.ArgumentParser(description="SegNet boundary-band attack")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--sal-lambda", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=20.0)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--tag", type=str, default="segnet_boundary_h64")
    args = parser.parse_args()

    print(f"[segnet-boundary] device: {DEVICE}")
    print(f"[segnet-boundary] h={args.hidden}, boundary_weight={args.boundary_weight}")

    # Load models
    posenet, segnet = load_scorer_models()

    # Decode frames
    print("[segnet-boundary] Decoding compressed archive...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    print(f"[segnet-boundary] Decoding GT video...")
    gt_frames = decode_gt_video(str(GT_VIDEO))

    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)
    n_pairs = len(comp_pairs)
    print(f"[segnet-boundary] {n_pairs} pairs")

    # Compute boundary masks from GT
    boundary_masks = compute_boundary_masks(gt_pairs, segnet, device=DEVICE)

    # Move pairs to device
    comp_pairs = [p.to(DEVICE) for p in comp_pairs]
    gt_pairs = [p.to(DEVICE) for p in gt_pairs]

    # Compute saliency weights
    from train_postfilter_saliency import compute_saliency_weights
    sal_weights = compute_saliency_weights(gt_frames, posenet, DEVICE, args.alpha)

    # Build model
    model = QATPostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    ema_state = {k: v.clone() for k, v in model.state_dict().items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=1e-6
    )

    best_scorer = float("inf")
    warmup_epochs = 10

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_pose = 0.0
        total_seg = 0.0

        # Warmup LR
        if epoch < warmup_epochs:
            lr = args.lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        indices = torch.randperm(n_pairs)
        for step_i, idx in enumerate(indices):
            idx = idx.item()
            filtered = model(comp_pairs[idx][:, 1:2].permute(0, 1, 4, 2, 3).reshape(-1, 3, *comp_pairs[idx].shape[2:4]).float())
            # Rebuild the pair with filtered frame
            filtered_pair = comp_pairs[idx].clone().float()
            filtered_pair[:, 1] = filtered.permute(0, 2, 3, 1)

            loss, pd, sd = compute_pair_loss_segnet_boundary(
                filtered_pair, gt_pairs[idx], posenet, segnet,
                boundary_masks[idx], args.boundary_weight,
            )

            # Add saliency reconstruction
            comp_bchw = comp_pairs[idx][:, 1].permute(0, 3, 1, 2).float()
            sal_recon = compute_saliency_reconstruction_loss(
                filtered, comp_bchw,
                sal_weights[idx * 2 + 1: idx * 2 + 2] if idx * 2 + 1 < len(sal_weights) else sal_weights[-1:]
            )
            total = loss + args.sal_lambda * sal_recon

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # EMA update
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema_state[k].mul_(args.ema_decay).add_(v, alpha=1 - args.ema_decay)

            total_loss += loss.item()
            total_pose += pd
            total_seg += sd

        if epoch >= warmup_epochs:
            scheduler.step()

        avg_loss = total_loss / n_pairs
        avg_pose = total_pose / n_pairs
        avg_seg = total_seg / n_pairs

        # Evaluate EMA with int8
        scorer_val = evaluate_ema_model(model, ema_state, comp_pairs, gt_pairs, posenet, segnet, DEVICE)

        if scorer_val < best_scorer:
            best_scorer = scorer_val
            save_best_checkpoint(
                model, ema_state, epoch, scorer_val, args,
                tag=args.tag,
                meta={"variant": "standard", "hidden": args.hidden,
                      "kernel": args.kernel, "alpha": args.alpha,
                      "boundary_weight": args.boundary_weight},
            )

        print(f"[ep {epoch:4d}] loss={avg_loss:.4f} pose={avg_pose:.6f} seg={avg_seg:.6f} "
              f"scorer={scorer_val:.4f} best={best_scorer:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

    print(f"[segnet-boundary] Training complete. Best scorer: {best_scorer:.4f}")


if __name__ == "__main__":
    main()
