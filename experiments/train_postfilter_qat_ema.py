#!/usr/bin/env python
"""Train a saliency-weighted post-filter with three additions over the baseline:

1. **Quantization-Aware Training (QAT)** — fake-quantize weights on every
   forward pass using the exact same per-tensor symmetric scheme that
   ``save_model_int8`` uses at deployment time, with straight-through
   estimator for the gradient.  This closes the train/deploy gap that
   currently costs us up to ~0.01 of score on the round-trip.

2. **Polyak / EMA weight averaging** — maintain a shadow copy of model
   weights updated as ``ema = decay * ema + (1 - decay) * model`` after
   every optimizer step.  At evaluation and save time we use the EMA
   weights, which are noticeably more stable than the raw "best-epoch"
   snapshot we relied on before.  This is the direct fix for the
   late-epoch oscillation we observed at α=30 in
   ``train_alpha30.log``.

3. **Tighter gradient clipping + warmup** — the baseline already clips at
   1.0, but we drop to 0.5 because the saliency-weighted loss has heavy
   tails that benefit from stricter clipping.  We also add a short
   linear warmup so the first few steps don't blow up the EMA.

The model architecture, data pipeline, scorer setup, and saliency
mask are all reused unchanged from ``train_postfilter_saliency.py``.

Usage::

    cd /tmp/pact-mine
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy python -u experiments/train_postfilter_qat_ema.py \\
        --alpha 20 --hidden 16 --epochs 120
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the entire infrastructure from the baseline trainer
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_postfilter_saliency import (  # type: ignore
    ARCHIVE_ZIP,
    DEFAULT_HIDDEN,
    DEFAULT_KERNEL,
    DEVICE,
    OUTPUT_DIR,
    PostFilter,
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
from frame_utils import seq_len  # noqa: E402  (UPSTREAM is on sys.path)


# ── Quantization-Aware Training helpers ──────────────────────────────────


class FakeQuantSTE(torch.autograd.Function):
    """Straight-through estimator for symmetric per-tensor int8 quantization.

    Forward: q = round(clamp(w / s, -128, 127)) * s,  s = max|w| / 127
    Backward: gradient passes through unchanged inside the clamp range,
              zeroed for values that hit the saturation boundary.
    """

    @staticmethod
    def forward(ctx, w):
        with torch.no_grad():
            scale = w.detach().abs().max() / 127.0
            if scale.item() == 0.0:
                ctx.save_for_backward(torch.zeros_like(w, dtype=torch.bool))
                return w
            q = (w / scale).round().clamp(-128.0, 127.0)
            saturated = (q.abs() >= 127.0)
            ctx.save_for_backward(saturated)
            return q * scale

    @staticmethod
    def backward(ctx, grad_out):
        (saturated,) = ctx.saved_tensors
        # Pass gradient through except where the quantizer saturated.
        return grad_out * (~saturated).to(grad_out.dtype)


def fake_quant(t: torch.Tensor) -> torch.Tensor:
    return FakeQuantSTE.apply(t)


class QATPostFilter(nn.Module):
    """PostFilter with fake-quantized weights on every forward.

    Mirrors the architecture of ``train_postfilter_saliency.PostFilter``
    so that the fp32 state dict is interchangeable.
    """

    def __init__(self, hidden: int = DEFAULT_HIDDEN, kernel: int = DEFAULT_KERNEL):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=False)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def _qconv(self, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        wq = fake_quant(conv.weight)
        bq = fake_quant(conv.bias) if conv.bias is not None else None
        return F.conv2d(x, wq, bq, padding=conv.padding, stride=conv.stride)

    def forward(self, x):
        residual = self.act(self._qconv(self.conv1, x))
        residual = self.act(self._qconv(self.conv2, residual))
        residual = self._qconv(self.conv3, residual)
        return (x + residual).clamp(0, 255)


# ── EMA helper ────────────────────────────────────────────────────────────


class EMA:
    """Polyak weight averaging.  Stored on the same device as the source."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)


def maybe_transfer_pairs_to_device(
    pairs: list[torch.Tensor],
    device: torch.device,
    *,
    eager: bool,
) -> list[torch.Tensor]:
    if not eager:
        return pairs
    return [pair.to(device) for pair in pairs]


def maybe_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    return tensor.to(device)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_pair_start_indices(frame_count: int, pair_len: int) -> list[int]:
    starts: list[int] = []
    for start in range(0, frame_count - 1, pair_len):
        if start + pair_len > frame_count:
            break
        starts.append(start)
    return starts


def pair_from_frames(frames: list[torch.Tensor], start_idx: int) -> torch.Tensor:
    return torch.stack(frames[start_idx:start_idx + seq_len]).unsqueeze(0)


def saliency_pair_at(
    base_saliency: torch.Tensor,
    *,
    start_idx: int,
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    slices = []
    last = base_saliency[-1]
    for offset in range(seq_len):
        frame_idx = start_idx + offset
        if frame_idx < base_saliency.shape[0]:
            sal = base_saliency[frame_idx]
        else:
            sal = last
        slices.append((1.0 + alpha * sal).unsqueeze(0))
    weights = torch.stack(slices, dim=0)
    return weights.to(device)


def save_best_checkpoint(
    *,
    model: nn.Module,
    ema: EMA,
    output_dir: Path,
    tag: str,
    meta: dict,
    epoch: int,
    scorer: float,
    shadow_state: dict[str, torch.Tensor] | None = None,
    per_channel_int8: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / f"postfilter_{tag}_best_fp32.pt"
    int8_path = output_dir / f"postfilter_{tag}_best_int8.pt"
    meta_path = output_dir / f"postfilter_{tag}_best_meta.json"

    source_shadow = shadow_state if shadow_state is not None else ema.shadow
    shadow = {name: tensor.detach().clone() for name, tensor in source_shadow.items()}
    torch.save(shadow, fp32_path)

    original_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    model.load_state_dict(shadow)
    int8_size = save_model_int8(model, int8_path, meta=meta, per_channel=per_channel_int8)
    model.load_state_dict(original_state)

    payload = {
        "epoch": epoch,
        "scorer": scorer,
        "fp32_path": str(fp32_path),
        "int8_path": str(int8_path),
        "int8_size": int8_size,
        "meta": meta,
    }
    meta_path.write_text(json.dumps(payload, indent=2))
    return payload


def quantize_state_dict_like_saved_int8(
    state_dict: dict[str, torch.Tensor],
    *,
    per_channel: bool = False,
) -> dict[str, torch.Tensor]:
    quantized_state: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            quantized_state[name] = tensor.clone()
            continue
        if per_channel and tensor.ndim >= 2 and not name.endswith("bias"):
            flattened = tensor.detach().reshape(tensor.shape[0], -1)
            scale = flattened.abs().max(dim=1).values / 127.0
            scale[scale == 0] = 1.0
            shape = [tensor.shape[0]] + [1] * (tensor.ndim - 1)
            q = torch.clamp(torch.round(tensor / scale.view(*shape)), -128, 127).to(torch.int8)
            quantized_state[name] = (q.float() * scale.view(*shape)).to(dtype=tensor.dtype)
            continue
        if per_channel and name.endswith("bias"):
            quantized_state[name] = tensor.clone()
            continue
        scale = tensor.detach().abs().max() / 127.0
        if float(scale) == 0.0:
            quantized_state[name] = tensor.clone()
            continue
        q = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
        quantized_state[name] = (q.float() * scale).to(dtype=tensor.dtype)
    return quantized_state


def init_average_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in state_dict.items()}


@torch.no_grad()
def update_average_state(
    average_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
    *,
    count: int,
) -> dict[str, torch.Tensor]:
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    alpha = 1.0 / float(count + 1)
    for name, tensor in source_state.items():
        if tensor.dtype.is_floating_point:
            average_state[name].mul_(1.0 - alpha).add_(tensor.detach(), alpha=alpha)
        else:
            average_state[name].copy_(tensor)
    return average_state


# ── Main ──────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QAT + EMA saliency post-filter")
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--alpha", type=float, default=20.0,
                   help="Saliency emphasis: weight = 1 + alpha * saliency")
    p.add_argument("--sal-lambda", type=float, default=0.1)
    p.add_argument("--train-subsample", type=int, default=8)
    p.add_argument("--eval-subsample", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--eager-pair-transfer", action="store_true",
                   help="Move all decoded pairs to DEVICE up front. Default keeps pairs on CPU and transfers per step.")
    p.add_argument("--cuda-autocast", action="store_true",
                   help="Use fp16 autocast for scorer/model forwards on CUDA to reduce memory pressure.")
    p.add_argument("--restart-t0", type=int, default=0,
                   help="Enable CosineAnnealingWarmRestarts with this initial cycle length when > 0.")
    p.add_argument("--restart-tmult", type=int, default=2)
    p.add_argument("--restart-eta-min", type=float, default=1e-5)
    p.add_argument("--swa-start-epoch", type=int, default=0,
                   help="If > 0, start averaging EMA shadows into an SWA shadow from this epoch onward.")
    p.add_argument("--swa-every", type=int, default=10,
                   help="Average one EMA shadow into SWA every N epochs once SWA starts.")
    p.add_argument("--checkpoint-eval-every", type=int, default=10,
                   help="Run eval-based checkpoint selection every N epochs.")
    p.add_argument("--checkpoint-select-int8", action="store_true",
                   help="Select best checkpoints after quantizing the EMA shadow like the saved int8 payload.")
    p.add_argument("--per-channel-int8", action="store_true",
                   help="Save int8 checkpoints with per-channel conv scales and fp32 biases.")
    p.add_argument("--tag", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    alpha = args.alpha
    tag = args.tag or f"qat_ema_alpha{int(alpha)}_h{args.hidden}"
    meta = normalize_postfilter_meta(args.hidden, args.kernel, alpha)

    print(f"[qat-ema] device={DEVICE} alpha={alpha} hidden={args.hidden} "
          f"ema={args.ema_decay} clip={args.grad_clip} tag={tag}")

    print("[qat-ema] Loading scorer models...")
    posenet, segnet = load_scorers(DEVICE)

    print("[qat-ema] Decoding compressed archive + ground truth...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]
    print(f"[qat-ema] {n} frames each")

    sal_base = torch.from_numpy(np.load(str(SALIENCY_PATH))).float()
    print(f"[qat-ema] Saliency base: mean={sal_base.mean().item():.3f} "
          f"max={sal_base.max().item():.1f}")

    pair_starts = build_pair_start_indices(n, seq_len)
    n_pairs = len(pair_starts)
    print(f"[qat-ema] {n_pairs} frame pairs")

    model = QATPostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    print(f"[qat-ema] Model: {count_params(model)} params (QAT-wrapped)")
    eval_model = PostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)

    ema = EMA(model, decay=args.ema_decay)

    eval_indices = list(range(0, n_pairs, args.eval_subsample))
    n_eval = len(eval_indices)

    # Baseline
    print(f"[qat-ema] Baseline (no filter) on {n_eval}/{n_pairs} pairs...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            start = pair_starts[idx]
            comp_pair = maybe_to_device(pair_from_frames(comp_frames, start), DEVICE)
            gt_pair = maybe_to_device(pair_from_frames(gt_frames, start), DEVICE)
            with autocast_context(DEVICE, args.cuda_autocast):
                _, pd, sd = compute_pair_loss(
                    comp_pair.float(), gt_pair, posenet, segnet
                )
            total_pose += pd
            total_seg += sd
    baseline_pose = total_pose / n_eval
    baseline_seg = total_seg / n_eval
    baseline_loss = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)
    print(f"[qat-ema] Baseline: loss={baseline_loss:.4f} "
          f"pose={baseline_pose:.6f} seg={baseline_seg:.6f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.restart_t0 > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.restart_t0,
            T_mult=args.restart_tmult,
            eta_min=args.restart_eta_min,
        )
        scheduler_mode = "warm_restarts"
    else:
        def lr_at(epoch_idx: int) -> float:
            if epoch_idx < args.warmup_epochs:
                return (epoch_idx + 1) / max(1, args.warmup_epochs)
            progress = (epoch_idx - args.warmup_epochs) / max(
                1, args.epochs - args.warmup_epochs
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress)) * (1 - 0.02) + 0.02
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
        scheduler_mode = "cosine_once"

    train_size = max(1, n_pairs // args.train_subsample)
    print(f"[qat-ema] Training: {args.epochs} epochs, {train_size} pairs/epoch, "
          f"warmup={args.warmup_epochs}, scheduler={scheduler_mode}")
    print(f"{'epoch':>5} {'total':>10} {'scorer':>10} {'pose':>12} "
          f"{'seg':>12} {'sal_recon':>10} {'lr':>10}")
    print("-" * 75)

    output_dir = OUTPUT_DIR
    best_scorer = float("inf")
    best_shadow_state: dict[str, torch.Tensor] | None = None
    swa_state: dict[str, torch.Tensor] | None = None
    swa_count = 0

    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(n_pairs)[:train_size].tolist()
        ep_loss = ep_scorer = ep_pose = ep_seg = ep_sal = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(indices):
            start = pair_starts[idx]
            comp_pair = maybe_to_device(pair_from_frames(comp_frames, start), DEVICE)
            gt_pair = maybe_to_device(pair_from_frames(gt_frames, start), DEVICE)
            sal_pair = saliency_pair_at(sal_base, start_idx=start, alpha=alpha, device=DEVICE)
            with autocast_context(DEVICE, args.cuda_autocast):
                filtered = apply_filter_to_pair(model, comp_pair, DEVICE)
                total_loss, scorer_loss, pd, sd, sal_recon = compute_combined_loss(
                    filtered, gt_pair, comp_pair,
                    posenet, segnet, sal_pair, args.sal_lambda,
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

        if scheduler_mode == "warm_restarts":
            scheduler.step(epoch + 1)
        else:
            scheduler.step()

        if args.swa_start_epoch > 0 and (epoch + 1) >= args.swa_start_epoch:
            if ((epoch + 1) - args.swa_start_epoch) % max(1, args.swa_every) == 0:
                if swa_state is None:
                    swa_state = init_average_state(ema.shadow)
                    swa_count = 1
                else:
                    update_average_state(swa_state, ema.shadow, count=swa_count)
                    swa_count += 1

        avg_scorer = ep_scorer / len(indices)
        score_for_checkpoint = avg_scorer
        shadow_for_checkpoint: dict[str, torch.Tensor] | None = None
        if swa_state is not None:
            score_for_checkpoint = min(score_for_checkpoint, avg_scorer)
            shadow_for_checkpoint = swa_state
        candidate_shadow = shadow_for_checkpoint or ema.shadow
        if (
            args.checkpoint_select_int8
            and (
                (epoch + 1) % args.checkpoint_eval_every == 0
                or epoch == 0
                or (epoch + 1) == args.epochs
            )
        ):
            eval_state = quantize_state_dict_like_saved_int8(
                candidate_shadow,
                per_channel=args.per_channel_int8,
            )
            eval_model.load_state_dict(eval_state)
            eval_model.eval()
            total_pose_eval, total_seg_eval = 0.0, 0.0
            with torch.no_grad():
                for idx in eval_indices:
                    start = pair_starts[idx]
                    comp_pair = maybe_to_device(pair_from_frames(comp_frames, start), DEVICE)
                    gt_pair = maybe_to_device(pair_from_frames(gt_frames, start), DEVICE)
                    with autocast_context(DEVICE, args.cuda_autocast):
                        filtered = apply_filter_to_pair(eval_model, comp_pair, DEVICE)
                        _, pd, sd = compute_pair_loss(filtered, gt_pair, posenet, segnet)
                    total_pose_eval += pd
                    total_seg_eval += sd
            pose_eval = total_pose_eval / n_eval
            seg_eval = total_seg_eval / n_eval
            score_for_checkpoint = 100.0 * seg_eval + math.sqrt(10.0 * pose_eval)
        if epoch == 0 or score_for_checkpoint < best_scorer:
            best_scorer = score_for_checkpoint
            payload = save_best_checkpoint(
                model=model,
                ema=ema,
                output_dir=output_dir,
                tag=tag,
                meta=meta,
                epoch=epoch + 1,
                scorer=score_for_checkpoint,
                shadow_state=shadow_for_checkpoint,
                per_channel_int8=args.per_channel_int8,
            )
            best_shadow_state = {
                name: tensor.detach().clone()
                for name, tensor in candidate_shadow.items()
            }

        if (epoch + 1) % 10 == 0 or epoch == 0:
            n_steps = len(indices)
            lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch + 1:>5} {ep_loss / n_steps:>10.4f} "
                  f"{ep_scorer / n_steps:>10.4f} {ep_pose / n_steps:>12.6f} "
                  f"{ep_seg / n_steps:>12.6f} {ep_sal / n_steps:>10.4f} {lr:>10.6f}")

    # Use the EMA weights as the deployed model
    if best_shadow_state is not None:
        model.load_state_dict(best_shadow_state)
    else:
        ema.copy_to(model)
    model.eval()

    print(f"\n[qat-ema] Final eval on EMA weights ({n_eval} pairs)...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            start = pair_starts[idx]
            comp_pair = maybe_to_device(pair_from_frames(comp_frames, start), DEVICE)
            gt_pair = maybe_to_device(pair_from_frames(gt_frames, start), DEVICE)
            with autocast_context(DEVICE, args.cuda_autocast):
                filtered = apply_filter_to_pair(model, comp_pair, DEVICE)
                _, pd, sd = compute_pair_loss(filtered, gt_pair, posenet, segnet)
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
    print(f"Delta:    {delta:+.4f}  pose={final_pose - baseline_pose:+.6f}  "
          f"seg={final_seg - baseline_seg:+.6f}")
    if delta < 0:
        print(f"*** IMPROVEMENT: {-delta:.4f} points ***")
    else:
        print("*** NO IMPROVEMENT ***")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32_path = OUTPUT_DIR / f"postfilter_{tag}_fp32.pt"
    int8_path = OUTPUT_DIR / f"postfilter_{tag}_int8.pt"
    torch.save(model.state_dict(), fp32_path)
    int8_size = save_model_int8(model, int8_path, meta=meta, per_channel=args.per_channel_int8)
    print(f"\nSaved fp32:  {fp32_path}")
    print(f"Saved int8:  {int8_path} ({int8_size} bytes)")
    return {"tag": tag, "baseline_loss": baseline_loss, "final_loss": final_loss}


if __name__ == "__main__":
    main()
