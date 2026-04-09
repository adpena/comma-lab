#!/usr/bin/env python
"""Jacob Collier's two-voice counterpoint post-filter.

Train two h=16 filters A, B JOINTLY with three losses:
1. Standard saliency-weighted scorer MSE on the SUM A(x)+B(x)
2. Band-orthogonality penalty forcing disjoint DCT frequency registers
3. Output-decorrelation term preventing the two voices from collapsing

Collier's hypothesis: the sum hits a residual shape no single h=32 model
can reach because it's literally a *chord*, not a doubled note. The
ensemble weight averaging failure (2.0469) happened because two unisons
don't make a chord — they phase-cancel. Explicit band orthogonality fixes
this.

Deployed size: 2 × h=16 int8 ~= 16 KB total (comparable to one h=32).
Deployed inference: run both filters, sum the residuals, clamp.

Collier's prediction: 1.82.

Usage::

    cd /tmp/pact-mine  # or tertiary
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy python -u experiments/train_postfilter_counterpoint.py \\
        --alpha 20 --epochs 800 --tag counterpoint_long800
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
    compute_combined_loss,
    compute_pair_loss,
    compute_saliency_reconstruction_loss,
    count_params,
    decode_archive,
    decode_video,
    load_scorers,
    normalize_postfilter_meta,
    save_model_int8,
    scorer_forward_pair,
)
from train_postfilter_qat_ema import (  # type: ignore
    EMA,
    QATPostFilter,
    build_pair_start_indices,
    maybe_to_device,
    pair_from_frames,
    saliency_pair_at,
)
from frame_utils import seq_len  # noqa: E402


class CounterpointVoice(nn.Module):
    """A single voice: same topology as QATPostFilter but exposes the raw
    residual directly via `forward_residual` so we can sum residuals
    without the lossy `filtered - input` subtraction trick.
    """

    def __init__(self, hidden: int = 16, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=False)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def _qconv(self, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        from train_postfilter_qat_ema import fake_quant
        wq = fake_quant(conv.weight)
        bq = fake_quant(conv.bias) if conv.bias is not None else None
        return F.conv2d(x, wq, bq, padding=conv.padding, stride=conv.stride)

    def forward_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Return the learned residual only (no +x, no clamp)."""
        h = self.act(self._qconv(self.conv1, x))
        h = self.act(self._qconv(self.conv2, h))
        return self._qconv(self.conv3, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.forward_residual(x)).clamp(0, 255)


class CounterpointPostFilter(nn.Module):
    """Two voices whose raw residuals are summed:
        out = clamp(x + residual_a + residual_b, 0, 255)

    Each voice is a separate CounterpointVoice (3-layer conv with
    fake-quantized weights and fp32 residual path). Residuals are
    exposed directly — no subtraction trick, no clamp inside the voice.
    """

    def __init__(self, hidden: int = 16, kernel: int = 3):
        super().__init__()
        self.voice_a = CounterpointVoice(hidden=hidden, kernel=kernel)
        self.voice_b = CounterpointVoice(hidden=hidden, kernel=kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resid_a = self.voice_a.forward_residual(x)
        resid_b = self.voice_b.forward_residual(x)
        return (x + resid_a + resid_b).clamp(0, 255)

    def residual_a(self, x: torch.Tensor) -> torch.Tensor:
        return self.voice_a.forward_residual(x)

    def residual_b(self, x: torch.Tensor) -> torch.Tensor:
        return self.voice_b.forward_residual(x)


def dct2_of_residual(residual_bchw: torch.Tensor) -> torch.Tensor:
    """Return the 2D rFFT magnitude of the luma channel of the residual.

    We use rFFT as a cheap proxy for DCT; the orthogonality relationship
    between two frequency-domain representations is qualitatively the same.
    Returns (B, H, W_half) magnitude tensor.
    """
    # Convert residual from (B, 3, H, W) to luma via BT.601 coefficients
    r = residual_bchw[:, 0]
    g = residual_bchw[:, 1]
    b = residual_bchw[:, 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    fft = torch.fft.rfft2(luma)
    return fft.abs()


def compute_counterpoint_loss(
    filtered_pair_hwc,
    gt_pair_hwc,
    comp_pair_hwc,
    posenet,
    segnet,
    sal_weights_pair,
    sal_lambda,
    resid_a_bchw,
    resid_b_bchw,
    band_lambda: float,
    decor_lambda: float,
):
    """Composite loss: scorer + saliency recon + band-orthogonality + decorrelation."""
    total, scorer, pose_dist, seg_dist, sal_recon = compute_combined_loss(
        filtered_pair_hwc,
        gt_pair_hwc,
        comp_pair_hwc,
        posenet,
        segnet,
        sal_weights_pair,
        sal_lambda,
    )

    # Band-orthogonality: ||DCT(A_resid) * DCT(B_resid)||^2 per element.
    # Guard against zero residuals (epoch 0 with zero-init): if either
    # residual norm is below epsilon, the penalty is zero and we skip the
    # normalization entirely to avoid NaN gradients through 0/eps.
    resid_a_norm_sq = resid_a_bchw.pow(2).sum()
    resid_b_norm_sq = resid_b_bchw.pow(2).sum()
    eps_active = 1e-8
    if resid_a_norm_sq.item() > eps_active and resid_b_norm_sq.item() > eps_active:
        spec_a = dct2_of_residual(resid_a_bchw)
        spec_b = dct2_of_residual(resid_b_bchw)
        # Normalize each so scale differences don't dominate. Use add-eps
        # rather than clamp-min so gradient flows smoothly through the norm.
        a_scale = spec_a.pow(2).sum().sqrt() + 1e-6
        b_scale = spec_b.pow(2).sum().sqrt() + 1e-6
        spec_a_n = spec_a / a_scale
        spec_b_n = spec_b / b_scale
        band_penalty = (spec_a_n * spec_b_n).pow(2).sum()

        # Output decorrelation: cosine similarity between flattened residuals
        a_flat = resid_a_bchw.reshape(-1)
        b_flat = resid_b_bchw.reshape(-1)
        a_norm_scale = a_flat.pow(2).sum().sqrt() + 1e-6
        b_norm_scale = b_flat.pow(2).sum().sqrt() + 1e-6
        a_norm = a_flat / a_norm_scale
        b_norm = b_flat / b_norm_scale
        cos_sim = (a_norm * b_norm).sum()
        decor_penalty = cos_sim.pow(2)
    else:
        # Residuals are essentially zero — no meaningful orthogonality to enforce yet.
        band_penalty = torch.zeros((), device=resid_a_bchw.device, dtype=resid_a_bchw.dtype)
        decor_penalty = torch.zeros((), device=resid_a_bchw.device, dtype=resid_a_bchw.dtype)

    total_counterpoint = total + band_lambda * band_penalty + decor_lambda * decor_penalty
    return (
        total_counterpoint,
        scorer,
        pose_dist,
        seg_dist,
        sal_recon,
        band_penalty.item(),
        decor_penalty.item(),
    )


def apply_counterpoint_to_pair(model: CounterpointPostFilter, pair_uint8, device):
    """Apply the two-voice filter. Returns (filtered_hwc, resid_a_bchw, resid_b_bchw)."""
    B, T, H, W, C = pair_uint8.shape
    x = pair_uint8.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
    resid_a = model.residual_a(x)  # (B*T, 3, H, W)
    resid_b = model.residual_b(x)
    out_bchw = (x + resid_a + resid_b).clamp(0, 255)
    filtered_hwc = out_bchw.permute(0, 2, 3, 1).reshape(B, T, H, W, C)
    return filtered_hwc, resid_a, resid_b


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collier two-voice counterpoint trainer")
    p.add_argument("--hidden", type=int, default=16, help="per-voice hidden width")
    p.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--sal-lambda", type=float, default=0.1)
    p.add_argument("--band-lambda", type=float, default=1.0,
                   help="weight for the band-orthogonality penalty")
    p.add_argument("--decor-lambda", type=float, default=0.5,
                   help="weight for the output decorrelation penalty")
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
    tag = args.tag or f"counterpoint_h{args.hidden}x2"
    meta = {
        "variant": "counterpoint",
        "hidden": args.hidden,
        "kernel": args.kernel,
        "alpha": alpha,
        "voices": 2,
    }

    print(f"[counterpoint] device={DEVICE} alpha={alpha} hidden={args.hidden}x2 "
          f"band_lambda={args.band_lambda} decor_lambda={args.decor_lambda} tag={tag}")

    print("[counterpoint] Loading scorer models...")
    posenet, segnet = load_scorers(DEVICE)

    print("[counterpoint] Decoding archive + ground truth (frames kept on CPU)...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]
    print(f"[counterpoint] {n} frames each")

    # Lazy pair loading — pairs materialize on DEVICE only during use.
    # This is critical for small-memory machines (WSL2 at 12GB, tertiary 8GB).
    # The base saliency map stays on CPU and per-pair weights are moved
    # lazily via saliency_pair_at.
    sal_base = torch.from_numpy(np.load(str(SALIENCY_PATH))).float()
    print(f"[counterpoint] Saliency base: mean={sal_base.mean().item():.3f} "
          f"max={sal_base.max().item():.1f}")

    pair_starts = build_pair_start_indices(n, seq_len)
    n_pairs = len(pair_starts)
    print(f"[counterpoint] {n_pairs} frame pairs (lazy)")

    gc.collect()

    model = CounterpointPostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    total_params = count_params(model)
    print(f"[counterpoint] Model: {total_params} params (two h={args.hidden} voices)")

    ema = EMA(model, decay=args.ema_decay)

    eval_indices = list(range(0, n_pairs, args.eval_subsample))
    n_eval = len(eval_indices)

    # Baseline: no filter (lazy-load pairs, only what we need)
    print(f"[counterpoint] Baseline on {n_eval}/{n_pairs} pairs...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            start = pair_starts[idx]
            comp_pair = maybe_to_device(pair_from_frames(comp_frames, start), DEVICE)
            gt_pair = maybe_to_device(pair_from_frames(gt_frames, start), DEVICE)
            _, pd, sd = compute_pair_loss(
                comp_pair.float(), gt_pair, posenet, segnet
            )
            total_pose += pd
            total_seg += sd
    baseline_pose = total_pose / n_eval
    baseline_seg = total_seg / n_eval
    baseline_loss = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)
    print(f"[counterpoint] Baseline: loss={baseline_loss:.4f} "
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
    print(f"[counterpoint] Training: {args.epochs} epochs, {train_size} pairs/epoch")
    print(f"{'epoch':>5} {'total':>10} {'scorer':>10} {'pose':>12} "
          f"{'seg':>12} {'band':>10} {'decor':>10} {'lr':>10}")
    print("-" * 92)

    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(n_pairs)[:train_size].tolist()
        ep_loss = ep_scorer = ep_pose = ep_seg = ep_band = ep_decor = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(indices):
            start = pair_starts[idx]
            comp_pair = maybe_to_device(pair_from_frames(comp_frames, start), DEVICE)
            gt_pair = maybe_to_device(pair_from_frames(gt_frames, start), DEVICE)
            sal_weights = saliency_pair_at(
                sal_base, start_idx=start, alpha=alpha, device=DEVICE
            )
            filtered, resid_a, resid_b = apply_counterpoint_to_pair(
                model, comp_pair, DEVICE
            )
            total, scorer, pd, sd, sal_recon, band, decor = compute_counterpoint_loss(
                filtered, gt_pair, comp_pair,
                posenet, segnet, sal_weights, args.sal_lambda,
                resid_a, resid_b, args.band_lambda, args.decor_lambda,
            )
            (total / args.accum_steps).backward()
            ep_loss += total.item()
            ep_scorer += scorer
            ep_pose += pd
            ep_seg += sd
            ep_band += band
            ep_decor += decor

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
                  f"{ep_band / ns:>10.4f} {ep_decor / ns:>10.4f} {lr:>10.6f}")

    ema.copy_to(model)
    model.eval()

    print(f"\n[counterpoint] Final eval on EMA weights ({n_eval} pairs)...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            start = pair_starts[idx]
            comp_pair = maybe_to_device(pair_from_frames(comp_frames, start), DEVICE)
            gt_pair = maybe_to_device(pair_from_frames(gt_frames, start), DEVICE)
            filtered, _, _ = apply_counterpoint_to_pair(model, comp_pair, DEVICE)
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
    print(f"Delta:    {delta:+.4f}")

    # Save each voice separately as its own int8 + combined meta
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32_path = OUTPUT_DIR / f"postfilter_{tag}_fp32.pt"
    torch.save(model.state_dict(), fp32_path)
    print(f"\nSaved fp32 (both voices): {fp32_path}")

    # Save each voice as its own int8 file (can be bundled together)
    voice_a_int8 = OUTPUT_DIR / f"postfilter_{tag}_voiceA_int8.pt"
    voice_b_int8 = OUTPUT_DIR / f"postfilter_{tag}_voiceB_int8.pt"
    size_a = save_model_int8(model.voice_a, voice_a_int8, meta={**meta, "voice": "a"})
    size_b = save_model_int8(model.voice_b, voice_b_int8, meta={**meta, "voice": "b"})
    total_size = size_a + size_b
    print(f"Saved voice A int8: {voice_a_int8} ({size_a} bytes)")
    print(f"Saved voice B int8: {voice_b_int8} ({size_b} bytes)")
    print(f"Total int8 size: {total_size} bytes")

    return {
        "tag": tag,
        "baseline_loss": baseline_loss,
        "final_loss": final_loss,
        "final_pose": final_pose,
        "final_seg": final_seg,
        "voice_a_int8_size": size_a,
        "voice_b_int8_size": size_b,
        "total_int8_size": total_size,
    }


if __name__ == "__main__":
    main()
