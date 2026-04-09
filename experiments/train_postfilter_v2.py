#!/usr/bin/env python
"""Train post-filter v2: scorer-faithful loss + QAT + EMA.

Key improvements over train_postfilter_saliency.py:

1. **SegNet loss uses straight-through hard argmax** instead of soft
   cosine similarity.  The actual scorer computes argmax disagreement
   rate: ``(pred.argmax(1) != gt.argmax(1)).float().mean()``.  Our
   training used a soft proxy ``1 - (softmax(pred)*softmax(gt)).sum(1).mean()``
   which can diverge significantly.  Now we use:
     - Forward: hard argmax disagreement (same as scorer)
     - Backward: gradient flows through the softmax (STE)

2. **PoseNet uses DistortionNet.compute_distortion** directly instead of
   a manual reimplementation — eliminates any subtle differences in how
   the MSE is aggregated.

3. **Full-pair evaluation (no subsampling)** at regular checkpoints to
   close the subsample bias gap and drive checkpoint selection.

4. **QAT + EMA** (from train_postfilter_qat_ema.py) for stability and
   to close the fp32→int8 deployment gap.

Usage::

    cd /tmp/pact-mine
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy python -u experiments/train_postfilter_v2.py \\
        --alpha 20 --epochs 200
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
    build_pairs,
    compute_saliency_reconstruction_loss,
    count_params,
    decode_archive,
    decode_video,
    load_saliency_weights,
    normalize_postfilter_meta,
    save_model_int8,
)
from train_postfilter_qat_ema import EMA, QATPostFilter  # type: ignore
from frame_utils import seq_len  # noqa: E402

sys.path.insert(0, str(UPSTREAM))
from modules import DistortionNet, posenet_sd_path, segnet_sd_path
from safetensors.torch import load_file
import einops


# ── Scorer-faithful loss ─────────────────────────────────────────────────


def load_distortion_net(device) -> DistortionNet:
    net = DistortionNet().eval().to(device)
    net.load_state_dicts(str(posenet_sd_path), str(segnet_sd_path), device)
    # Freeze all parameters
    for p in net.parameters():
        p.requires_grad = False
    return net


class ScorerFaithfulLoss(nn.Module):
    """Loss that matches the upstream evaluator as closely as possible.

    PoseNet: exact same compute_distortion (MSE of pose[:6] per sample).
    SegNet: straight-through hard argmax disagreement rate.
    """

    def __init__(self, distortion_net: DistortionNet):
        super().__init__()
        self.posenet = distortion_net.posenet
        self.segnet = distortion_net.segnet

    def forward(self, filtered_btchw: torch.Tensor, gt_btchw: torch.Tensor):
        """
        filtered_btchw: (B, T, C, H, W) float, gradient-connected
        gt_btchw: (B, T, C, H, W) float, detached
        Returns (total_loss, pose_dist_scalar, seg_dist_scalar)
        """
        # PoseNet (gradient flows through filtered)
        fp_in = self.posenet.preprocess_input(filtered_btchw)
        fp_out = self.posenet(fp_in)

        with torch.no_grad():
            gp_in = self.posenet.preprocess_input(gt_btchw)
            gp_out = self.posenet(gp_in)

        # Use the exact same distortion as the scorer
        pose_dist = self.posenet.compute_distortion(fp_out, gp_out).mean()

        # SegNet with straight-through hard argmax
        fs_in = self.segnet.preprocess_input(filtered_btchw)
        fs_out = self.segnet(fs_in)

        with torch.no_grad():
            gs_in = self.segnet.preprocess_input(gt_btchw)
            gs_out = self.segnet(gs_in)

        # Straight-through estimator for argmax disagreement:
        # Forward: hard argmax disagreement (non-differentiable)
        # Backward: gradient through softmax cross-entropy proxy
        seg_dist_hard = self._segnet_ste(fs_out, gs_out)

        loss = 100.0 * seg_dist_hard + torch.sqrt(10.0 * pose_dist + 1e-8)
        return loss, pose_dist.item(), seg_dist_hard.item()

    def _segnet_ste(self, pred_logits, gt_logits):
        """Straight-through estimator for SegNet argmax disagreement.

        Forward value = hard argmax disagreement rate (matches scorer)
        Backward gradient = gradient of soft cross-entropy proxy
        """
        # Soft differentiable proxy (for backward)
        pred_soft = F.softmax(pred_logits, dim=1)
        gt_soft = F.softmax(gt_logits, dim=1)
        soft_agreement = (pred_soft * gt_soft).sum(dim=1).mean()
        soft_dist = 1.0 - soft_agreement

        # Hard scorer value (for forward, detached)
        with torch.no_grad():
            hard_disagree = (
                pred_logits.argmax(dim=1) != gt_logits.argmax(dim=1)
            ).float().mean()

        # STE: use hard value but gradient of soft
        return soft_dist + (hard_disagree - soft_dist).detach()


# ── Apply filter to pair ─────────────────────────────────────────────────


def apply_filter_to_pair(model, pair_uint8, device):
    """Apply post-filter to a (1, T, H, W, 3) uint8 pair.
    Returns (1, T, H, W, 3) float filtered pair (gradient-connected).
    """
    B, T, H, W, C = pair_uint8.shape
    x = pair_uint8.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
    y = model(x)
    return y.permute(0, 2, 3, 1).reshape(B, T, H, W, C)


def evaluate_model_pairs(
    model: nn.Module,
    comp_pairs: list[torch.Tensor],
    gt_pairs: list[torch.Tensor],
    *,
    eval_indices: list[int],
    loss_fn,
    device,
) -> tuple[float, float, float]:
    total_loss = 0.0
    total_pose = 0.0
    total_seg = 0.0

    with torch.no_grad():
        for idx in eval_indices:
            filtered = apply_filter_to_pair(model, comp_pairs[idx], device)
            fx = filtered.permute(0, 1, 4, 2, 3).contiguous()
            gx = gt_pairs[idx].float().permute(0, 1, 4, 2, 3).contiguous()
            loss, pose, seg = loss_fn(fx, gx)
            total_loss += float(loss.item() if hasattr(loss, "item") else loss)
            total_pose += float(pose)
            total_seg += float(seg)

    n_eval = len(eval_indices)
    return total_loss / n_eval, total_pose / n_eval, total_seg / n_eval


def quantize_state_dict_like_saved_int8(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    quantized_state = {}
    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            quantized_state[name] = tensor.clone()
            continue
        scale = tensor.detach().abs().max() / 127.0
        if float(scale) == 0.0:
            quantized_state[name] = tensor.clone()
            continue
        q = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
        quantized_state[name] = (q.float() * scale).to(dtype=tensor.dtype)
    return quantized_state


# ── Main ──────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="V2 scorer-faithful post-filter")
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--sal-lambda", type=float, default=0.1)
    p.add_argument("--train-subsample", type=int, default=8)
    p.add_argument("--eval-subsample", type=int, default=1,
                   help="1 = evaluate ALL pairs (faithful to scorer)")
    p.add_argument("--checkpoint-eval-every", type=int, default=10,
                   help="Run full eval on EMA weights every N epochs for checkpoint selection")
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
    tag = args.tag or f"v2_alpha{int(alpha)}_h{args.hidden}"
    meta = normalize_postfilter_meta(args.hidden, args.kernel, alpha)

    print(f"[v2] device={DEVICE} alpha={alpha} hidden={args.hidden} "
          f"ema={args.ema_decay} clip={args.grad_clip} eval_sub={args.eval_subsample}")

    print("[v2] Loading DistortionNet (scorer-faithful)...")
    dist_net = load_distortion_net(DEVICE)
    loss_fn = ScorerFaithfulLoss(dist_net)

    print("[v2] Decoding archive + GT...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]
    print(f"[v2] {n} frames")

    sal_all = load_saliency_weights(alpha, n, DEVICE)
    sal_pairs = []
    for i in range(0, n - 1, seq_len):
        if i + seq_len > n:
            break
        sal_pairs.append(sal_all[i:i + seq_len])

    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)
    n_pairs = len(comp_pairs)
    print(f"[v2] {n_pairs} frame pairs")

    del comp_frames, gt_frames, sal_all
    gc.collect()

    comp_pairs = [p.to(DEVICE) for p in comp_pairs]
    gt_pairs = [p.to(DEVICE) for p in gt_pairs]

    model = QATPostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    print(f"[v2] Model: {count_params(model)} params")

    ema = EMA(model, decay=args.ema_decay)

    eval_indices = list(range(0, n_pairs, args.eval_subsample))
    n_eval = len(eval_indices)

    # Baseline
    print(f"[v2] Baseline on {n_eval}/{n_pairs} pairs...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            unfiltered = comp_pairs[idx].float()
            fx = unfiltered.permute(0, 1, 4, 2, 3).contiguous()
            gx = gt_pairs[idx].float().permute(0, 1, 4, 2, 3).contiguous()
            _, pd, sd = loss_fn(fx, gx)
            total_pose += pd
            total_seg += sd
    baseline_pose = total_pose / n_eval
    baseline_seg = total_seg / n_eval
    baseline_loss = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)
    print(f"[v2] Baseline: loss={baseline_loss:.4f} "
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
    print(f"[v2] Training: {args.epochs} epochs, {train_size} pairs/epoch, "
          f"eval on {n_eval} pairs")
    print(f"{'epoch':>5} {'total':>10} {'scorer':>10} {'pose':>12} {'seg':>12} "
          f"{'sal':>10} {'eval':>10} {'lr':>10}")
    print("-" * 86)

    eval_model = QATPostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    best_eval_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(n_pairs)[:train_size].tolist()
        ep_loss = ep_scorer = ep_pose = ep_seg = ep_sal = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(indices):
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)

            # Scorer-faithful loss
            fx = filtered.permute(0, 1, 4, 2, 3).contiguous()
            gx = gt_pairs[idx].float().permute(0, 1, 4, 2, 3).contiguous()
            scorer_loss, pd, sd = loss_fn(fx, gx)

            # Saliency reconstruction penalty
            B, T, H, W, C = filtered.shape
            filtered_bchw = filtered.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
            comp_bchw = comp_pairs[idx].float().reshape(B * T, H, W, C).permute(0, 3, 1, 2)
            sal_recon = compute_saliency_reconstruction_loss(
                filtered_bchw, comp_bchw, sal_pairs[idx]
            )

            total_loss = scorer_loss + args.sal_lambda * sal_recon
            (total_loss / args.accum_steps).backward()

            ep_loss += total_loss.item()
            ep_scorer += scorer_loss.item()
            ep_pose += pd
            ep_seg += sd
            ep_sal += sal_recon.item()

            if (step_i + 1) % args.accum_steps == 0 or (step_i + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

        scheduler.step()

        ns = len(indices)
        eval_loss = None
        if (
            (epoch + 1) % args.checkpoint_eval_every == 0
            or epoch == 0
            or (epoch + 1) == args.epochs
        ):
            eval_model.load_state_dict(quantize_state_dict_like_saved_int8(ema.shadow))
            eval_model.eval()
            eval_loss, _, _ = evaluate_model_pairs(
                eval_model,
                comp_pairs,
                gt_pairs,
                eval_indices=eval_indices,
                loss_fn=loss_fn,
                device=DEVICE,
            )
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_state = {k: v.clone() for k, v in ema.shadow.items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            eval_display = f"{eval_loss:.4f}" if eval_loss is not None else "-"
            print(f"{epoch + 1:>5} {ep_loss / ns:>10.4f} {ep_scorer / ns:>10.4f} "
                  f"{ep_pose / ns:>12.6f} {ep_seg / ns:>12.6f} "
                  f"{ep_sal / ns:>10.4f} {eval_display:>10} {lr:>10.6f}")

    # Use best EMA state
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        ema.copy_to(model)
    model.eval()

    # Final full evaluation
    print(f"\n[v2] Final eval on EMA weights ({n_eval} pairs)...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            fx = filtered.permute(0, 1, 4, 2, 3).contiguous()
            gx = gt_pairs[idx].float().permute(0, 1, 4, 2, 3).contiguous()
            _, pd, sd = loss_fn(fx, gx)
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
