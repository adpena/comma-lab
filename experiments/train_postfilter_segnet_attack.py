#!/usr/bin/env python
"""Train a post-filter with direct SegNet argmax-disagreement attack.

This is the 'SegNet attack' experiment recommended unanimously by the expert
panel (Tao, Karpathy, LeCun, Collier, Einstein).  The insight:

- The scorer term is ``100*S + sqrt(10*P) + 25*R``
- At our current operating point (P=0.048, S=0.00576, R=0.575):
    dScore/dS = 100  (the multiplier)
    dScore/dP = sqrt(10)/(2*sqrt(P)) ~ 7.21  (the sqrt's derivative)
- So **S is ~14x more leveraged than P** per unit distortion
- We have aggressively optimized P (-23% vs the 2.01 floor) but barely
  touched S (~0% change).  A 20% cut in S alone would yield -0.115 points.
- The winning trainer uses a soft cosine proxy for SegNet, which doesn't
  correspond 1:1 with the scorer's hard argmax disagreement metric.

This trainer swaps the SegNet loss for a straight-through estimator on the
hard argmax disagreement, layered on top of the QAT+EMA recipe that gave
us the 1.845 floor.

Everything else (scorer loss composition, saliency weighting, QAT, EMA,
warmup, cosine LR) is identical to train_postfilter_qat_ema.py so that
any score delta is attributable to the SegNet loss change only.

Usage::

    cd /tmp/pact-mine
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy python -u experiments/train_postfilter_segnet_attack.py \\
        --alpha 20 --hidden 32 --epochs 1000 --tag segnet_attack_h32
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

# Reuse everything from the QAT+EMA trainer. We will replace only the
# SegNet component of the loss via monkey-patching
# ``train_postfilter_saliency.compute_pair_loss``.
import train_postfilter_saliency as _base
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
    scorer_forward_pair,
)
from train_postfilter_qat_ema import EMA, QATPostFilter, save_best_checkpoint  # type: ignore
from frame_utils import seq_len  # noqa: E402


# ── SegNet attack: straight-through hard argmax disagreement ─────────────


def compute_pair_loss_segnet_attack(
    filtered_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    posenet: nn.Module,
    segnet: nn.Module,
):
    """Drop-in replacement for train_postfilter_saliency.compute_pair_loss.

    PoseNet loss: unchanged (scorer-faithful MSE of pose[:6]).
    SegNet loss: straight-through argmax disagreement rate matching the
                 scorer's hard metric exactly. Forward value is the
                 non-differentiable ``(pred.argmax != gt.argmax).float().mean()``.
                 Backward gradient flows through a softened cross-entropy-like
                 proxy so SGD still has a useful signal.
    """
    fx = filtered_pair_hwc.permute(0, 1, 4, 2, 3).contiguous()
    gx = gt_pair_hwc.float().permute(0, 1, 4, 2, 3).contiguous()

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)

    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

    # PoseNet: unchanged
    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()

    # SegNet STE: forward = hard disagreement, backward = soft cross-entropy
    # gradient. We use categorical cross-entropy since gt is treated as a
    # hard label (argmax of gs_out) in the forward pass.
    with torch.no_grad():
        gt_labels = gs_out.argmax(dim=1)
        pred_labels = fs_out.argmax(dim=1)
        hard_disagree = (pred_labels != gt_labels).float().mean()

    # Soft proxy for the backward pass. Previously we scaled soft_ce by 0.05
    # to magnitude-match hard_disagree, but that CRUSHED the gradient by 20x
    # relative to the cross-entropy that should flow (SegNet research agent
    # diagnosis, 2026-04-09). STE works by passing the backward of the soft
    # term through unchanged — the forward value is replaced but the gradient
    # must have the full magnitude.
    #
    # We use the canonical STE: forward = hard_disagree, backward = soft_ce.
    B, C, H, W = fs_out.shape
    flat_labels = gt_labels.reshape(-1)
    flat_logits = fs_out.permute(0, 2, 3, 1).reshape(-1, C)
    soft_ce = F.cross_entropy(flat_logits, flat_labels, reduction="mean")

    # Canonical STE: seg_dist has the hard_disagree forward value, but its
    # backward gradient is exactly the soft_ce gradient at full scale.
    seg_dist = soft_ce + (hard_disagree - soft_ce).detach()

    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), hard_disagree.item()


def compute_combined_loss_segnet_attack(
    filtered_pair_hwc, gt_pair_hwc, comp_pair_hwc,
    posenet, segnet, sal_weights_pair, sal_lambda,
):
    """Combined loss with SegNet-attack scorer loss.

    Layers:
    - scorer loss (SegNet-attack variant) -- PoseNet MSE + STE argmax SegNet
    - saliency-weighted reconstruction penalty (unchanged) protects low-saliency
      pixels from drifting
    """
    scorer_loss, pose_dist, seg_dist = compute_pair_loss_segnet_attack(
        filtered_pair_hwc, gt_pair_hwc, posenet, segnet
    )

    B, T, H, W, C = filtered_pair_hwc.shape
    filtered_bchw = filtered_pair_hwc.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
    comp_bchw = comp_pair_hwc.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2)
    sal_recon = compute_saliency_reconstruction_loss(
        filtered_bchw, comp_bchw, sal_weights_pair
    )

    total = scorer_loss + sal_lambda * sal_recon
    return total, scorer_loss.item(), pose_dist, seg_dist, sal_recon.item()


def evaluate_ema_model(
    *,
    model: nn.Module,
    ema: EMA,
    eval_indices: list[int],
    comp_pairs: list[torch.Tensor],
    gt_pairs: list[torch.Tensor],
    posenet: nn.Module,
    segnet: nn.Module,
) -> tuple[float, float, float]:
    """Evaluate the EMA shadow weights on the held-out eval subset."""
    original_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    ema.copy_to(model)
    model.eval()
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            _, pd, sd = compute_pair_loss_segnet_attack(
                filtered, gt_pairs[idx], posenet, segnet
            )
            total_pose += pd
            total_seg += sd
    model.load_state_dict(original_state)
    avg_pose = total_pose / len(eval_indices)
    avg_seg = total_seg / len(eval_indices)
    score = 100.0 * avg_seg + math.sqrt(10.0 * avg_pose)
    return score, avg_pose, avg_seg


# ── Main ──────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SegNet-attack post-filter trainer")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    p.add_argument("--epochs", type=int, default=1000)
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


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    alpha = args.alpha
    tag = args.tag or f"segnet_attack_h{args.hidden}"
    meta = normalize_postfilter_meta(args.hidden, args.kernel, alpha)

    print(f"[segnet-attack] device={DEVICE} alpha={alpha} hidden={args.hidden} "
          f"ema={args.ema_decay} tag={tag}")

    print("[segnet-attack] Loading scorer models...")
    posenet, segnet = load_scorers(DEVICE)

    print("[segnet-attack] Decoding archive + ground truth...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]
    print(f"[segnet-attack] {n} frames each")

    sal_all = load_saliency_weights(alpha, n, DEVICE)
    sal_pairs = []
    for i in range(0, n - 1, seq_len):
        if i + seq_len > n:
            break
        sal_pairs.append(sal_all[i:i + seq_len])

    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)
    n_pairs = len(comp_pairs)
    print(f"[segnet-attack] {n_pairs} frame pairs")

    del comp_frames, gt_frames, sal_all
    gc.collect()

    comp_pairs = [p.to(DEVICE) for p in comp_pairs]
    gt_pairs = [p.to(DEVICE) for p in gt_pairs]

    model = QATPostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    print(f"[segnet-attack] Model: {count_params(model)} params")

    ema = EMA(model, decay=args.ema_decay)

    eval_indices = list(range(0, n_pairs, args.eval_subsample))
    n_eval = len(eval_indices)

    # Baseline
    print(f"[segnet-attack] Baseline on {n_eval}/{n_pairs} pairs...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            _, pd, sd = compute_pair_loss_segnet_attack(
                comp_pairs[idx].float(), gt_pairs[idx], posenet, segnet
            )
            total_pose += pd
            total_seg += sd
    baseline_pose = total_pose / n_eval
    baseline_seg = total_seg / n_eval
    baseline_loss = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)
    print(f"[segnet-attack] Baseline: loss={baseline_loss:.4f} "
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
    print(f"[segnet-attack] Training: {args.epochs} epochs, {train_size} pairs/epoch")
    print(f"{'epoch':>5} {'total':>10} {'scorer':>10} {'pose':>12} "
          f"{'seg':>12} {'sal_recon':>10} {'lr':>10}")
    print("-" * 78)
    best_eval_score = math.inf
    best_eval_payload: dict[str, object] | None = None

    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(n_pairs)[:train_size].tolist()
        ep_loss = ep_scorer = ep_pose = ep_seg = ep_sal = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(indices):
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            total, scorer, pd, sd, sal_recon = compute_combined_loss_segnet_attack(
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
                ema.update(model)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            ns = len(indices)
            lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch + 1:>5} {ep_loss / ns:>10.4f} {ep_scorer / ns:>10.4f} "
                  f"{ep_pose / ns:>12.6f} {ep_seg / ns:>12.6f} "
                  f"{ep_sal / ns:>10.4f} {lr:>10.6f}")
            eval_score, eval_pose, eval_seg = evaluate_ema_model(
                model=model,
                ema=ema,
                eval_indices=eval_indices,
                comp_pairs=comp_pairs,
                gt_pairs=gt_pairs,
                posenet=posenet,
                segnet=segnet,
            )
            print(f"      eval score={eval_score:>10.4f} pose={eval_pose:>12.6f} seg={eval_seg:>12.6f}")
            if eval_score < best_eval_score:
                best_eval_score = eval_score
                best_eval_payload = save_best_checkpoint(
                    model=model,
                    ema=ema,
                    output_dir=OUTPUT_DIR,
                    tag=tag,
                    meta=meta,
                    epoch=epoch + 1,
                    scorer=eval_score,
                )
                print(
                    f"      best checkpoint -> epoch {epoch + 1} score={eval_score:.4f} "
                    f"int8={best_eval_payload['int8_size']} bytes"
                )

    ema.copy_to(model)
    model.eval()

    # Final eval
    print(f"\n[segnet-attack] Final eval on EMA weights ({n_eval} pairs)...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            _, pd, sd = compute_pair_loss_segnet_attack(
                filtered, gt_pairs[idx], posenet, segnet
            )
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
    if best_eval_payload is not None:
        print(
            f"Best EMA checkpoint: epoch {best_eval_payload['epoch']} "
            f"score={best_eval_payload['scorer']:.4f} "
            f"int8={best_eval_payload['int8_size']} bytes"
        )

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
