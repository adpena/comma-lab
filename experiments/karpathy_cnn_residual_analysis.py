#!/usr/bin/env python
"""Measure what our trained CNN post-filter actually does to pixels.

Karpathy's revised hypothesis after the Jacobian failure:
- The CNN's residual (filtered - decoded) is DENSE (>50% of pixels nudged)
- SMALL-AMPLITUDE (sub-LSB mean)
- SPECTRALLY BIASED toward mid-frequencies (where PoseNet's early convs respond)
- In contrast, the Jacobian's minimum-norm delta is SPARSE, HIGH-AMPLITUDE, SPECTRALLY WHITE

If the prediction holds, the CNN is doing "network-aware spreading" —
keeping corrections inside every ReLU region it touches.

This is a diagnostic measurement, not a training run. It analyzes the
1.845 winning CNN against decoded frames and reports:

1. Pixel-change histogram (magnitude distribution)
2. Spatial sparsity (fraction of pixels with |delta| > threshold)
3. 2D-DCT spectrum of the residual (frequency band energies)
4. Comparison with the Jacobian delta from the failed experiment
5. Per-frame statistics

Expected outputs inform whether the CNN is truly doing network-aware
spreading and whether a frequency-domain filter would work.

Usage::

    cd /tmp/pact-mine
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy python -u experiments/karpathy_cnn_residual_analysis.py
"""
from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "submissions" / "robust_current"))

from train_postfilter_saliency import (  # type: ignore
    ARCHIVE_ZIP,
    DEVICE,
    VIDEOS_DIR,
    build_pairs,
    decode_archive,
    decode_video,
    load_scorers,
)
from jacobian_optimal import compute_jacobian, optimal_correction  # type: ignore
from inflate_postfilter import load_postfilter_int8  # type: ignore
from frame_utils import seq_len  # noqa: E402


def dct2_power_spectrum(x: torch.Tensor) -> torch.Tensor:
    """Compute the 2D DCT magnitude spectrum of a grayscale image.

    Uses the FFT as a cheap proxy — absolute value of 2D rFFT gives us
    spectral energy in mid/high/low frequency bands without needing a
    true DCT implementation (the bands we care about are qualitatively
    the same).
    """
    # x: (H, W) float
    fft = torch.fft.rfft2(x.float())
    return fft.abs()


def band_energies(spectrum: torch.Tensor) -> dict:
    """Split a 2D spectrum into low/mid/high frequency bands by radius."""
    device = spectrum.device
    # Use magnitude (it's already |fft|, but we want squared energy)
    energy = spectrum.pow(2).cpu()
    H, W = energy.shape
    cy, cx = H // 2, 0  # rFFT is not shifted on the W axis; origin is at (H/2, 0)
    yy = torch.arange(H).view(-1, 1).expand(H, W).float()
    xx = torch.arange(W).view(1, -1).expand(H, W).float()
    # Distance from DC (origin)
    dist = torch.sqrt((yy - cy).pow(2) + xx.pow(2))
    max_dist = math.sqrt((H/2) ** 2 + W ** 2)
    low = (dist < max_dist / 3).float()
    mid = ((dist >= max_dist / 3) & (dist < 2 * max_dist / 3)).float()
    high = (dist >= 2 * max_dist / 3).float()
    spectrum = energy  # now CPU
    total = spectrum.sum().item() + 1e-12
    return {
        "low_frac": float((spectrum * low).sum().item() / total),
        "mid_frac": float((spectrum * mid).sum().item() / total),
        "high_frac": float((spectrum * high).sum().item() / total),
    }


def main():
    print(f"[karpathy] device={DEVICE}")
    print(f"[karpathy] Loading winning CNN from submissions/robust_current/")
    cnn_path = PROJECT / "submissions" / "robust_current" / "postfilter_int8.pt"
    print(f"[karpathy] CNN weights: {cnn_path}")
    cnn = load_postfilter_int8(str(cnn_path), device=DEVICE)
    cnn.eval()
    print(f"[karpathy] CNN loaded, params={sum(p.numel() for p in cnn.parameters())}")

    print(f"[karpathy] Loading PoseNet...")
    posenet, _ = load_scorers(DEVICE)

    print(f"[karpathy] Decoding archive + GT...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_pairs = build_pairs(comp_frames[:n])
    gt_pairs = build_pairs(gt_frames[:n])
    n_pairs = len(comp_pairs)
    del comp_frames, gt_frames
    gc.collect()

    # Analyze on a small subsample (30 pairs is enough for stable stats)
    n_analyze = 20
    indices = np.linspace(0, n_pairs - 1, n_analyze, dtype=int).tolist()

    # Collect CNN residual statistics
    cnn_delta_abs_mean = []
    cnn_delta_abs_max = []
    cnn_frac_changed_0_5 = []
    cnn_frac_changed_2_0 = []
    cnn_band_low = []
    cnn_band_mid = []
    cnn_band_high = []

    # Collect Jacobian delta statistics (for comparison)
    jac_delta_abs_mean = []
    jac_delta_abs_max = []
    jac_frac_changed_0_5 = []
    jac_band_low = []
    jac_band_mid = []
    jac_band_high = []

    print(f"\n[karpathy] Analyzing {n_analyze} pairs...")

    for i, idx in enumerate(indices):
        comp_pair = comp_pairs[idx].to(DEVICE).float()  # (1, 2, H, W, 3)
        gt_pair = gt_pairs[idx].to(DEVICE).float()

        # Apply CNN: (1, 2, H, W, 3) -> float, permute, filter, permute back
        B, T, H, W, C = comp_pair.shape
        x = comp_pair.reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            y = cnn(x)
        cnn_filtered = y.permute(0, 2, 3, 1).reshape(B, T, H, W, C)
        cnn_delta = cnn_filtered - comp_pair  # the residual correction

        # CNN stats
        cnn_delta_abs = cnn_delta.abs()
        cnn_delta_abs_mean.append(cnn_delta_abs.mean().item())
        cnn_delta_abs_max.append(cnn_delta_abs.max().item())
        cnn_frac_changed_0_5.append((cnn_delta_abs > 0.5).float().mean().item())
        cnn_frac_changed_2_0.append((cnn_delta_abs > 2.0).float().mean().item())

        # Spectrum on luma channel of first frame
        cnn_luma = (0.299 * cnn_delta[0, 0, :, :, 0] +
                    0.587 * cnn_delta[0, 0, :, :, 1] +
                    0.114 * cnn_delta[0, 0, :, :, 2])
        spec = dct2_power_spectrum(cnn_luma)
        bands = band_energies(spec)
        cnn_band_low.append(bands["low_frac"])
        cnn_band_mid.append(bands["mid_frac"])
        cnn_band_high.append(bands["high_frac"])

        # Jacobian delta for comparison
        J, pose_comp = compute_jacobian(posenet, comp_pair)
        with torch.no_grad():
            pose_gt = posenet(posenet.preprocess_input(
                gt_pair.permute(0, 1, 4, 2, 3).contiguous()
            ))["pose"][..., :6].squeeze()
        residual = pose_gt - pose_comp
        jac_delta = optimal_correction(J, residual).reshape(comp_pair.shape)

        jac_delta_abs = jac_delta.abs()
        jac_delta_abs_mean.append(jac_delta_abs.mean().item())
        jac_delta_abs_max.append(jac_delta_abs.max().item())
        jac_frac_changed_0_5.append((jac_delta_abs > 0.5).float().mean().item())

        jac_luma = (0.299 * jac_delta[0, 0, :, :, 0] +
                    0.587 * jac_delta[0, 0, :, :, 1] +
                    0.114 * jac_delta[0, 0, :, :, 2])
        spec = dct2_power_spectrum(jac_luma)
        bands = band_energies(spec)
        jac_band_low.append(bands["low_frac"])
        jac_band_mid.append(bands["mid_frac"])
        jac_band_high.append(bands["high_frac"])

        if (i + 1) % 5 == 0:
            print(f"  pair {i+1:3d}/{n_analyze}: "
                  f"CNN |δ|={cnn_delta_abs_mean[-1]:.4f} max={cnn_delta_abs_max[-1]:.1f} "
                  f"frac>0.5={cnn_frac_changed_0_5[-1]:.3f} | "
                  f"Jac |δ|={jac_delta_abs_mean[-1]:.6f} max={jac_delta_abs_max[-1]:.3f} "
                  f"frac>0.5={jac_frac_changed_0_5[-1]:.5f}")

    # Summary
    def stats(arr):
        arr = np.array(arr)
        return {"mean": float(arr.mean()), "median": float(np.median(arr)), "std": float(arr.std())}

    cnn_summary = {
        "abs_mean": stats(cnn_delta_abs_mean),
        "abs_max": stats(cnn_delta_abs_max),
        "frac_changed_0_5": stats(cnn_frac_changed_0_5),
        "frac_changed_2_0": stats(cnn_frac_changed_2_0),
        "band_low": stats(cnn_band_low),
        "band_mid": stats(cnn_band_mid),
        "band_high": stats(cnn_band_high),
    }
    jac_summary = {
        "abs_mean": stats(jac_delta_abs_mean),
        "abs_max": stats(jac_delta_abs_max),
        "frac_changed_0_5": stats(jac_frac_changed_0_5),
        "band_low": stats(jac_band_low),
        "band_mid": stats(jac_band_mid),
        "band_high": stats(jac_band_high),
    }

    print(f"\n{'=' * 78}")
    print(f"RESULTS: CNN vs Jacobian delta characterization")
    print(f"{'=' * 78}")
    print(f"\n--- Delta magnitude ---")
    print(f"  CNN mean |δ|:        {cnn_summary['abs_mean']['mean']:.4f}")
    print(f"  Jacobian mean |δ|:   {jac_summary['abs_mean']['mean']:.6f}")
    print(f"  CNN max |δ|:         {cnn_summary['abs_max']['mean']:.2f}")
    print(f"  Jacobian max |δ|:    {jac_summary['abs_max']['mean']:.3f}")

    print(f"\n--- Spatial density (pixels moved > 0.5 LSB) ---")
    print(f"  CNN frac > 0.5:      {cnn_summary['frac_changed_0_5']['mean']:.4f}  "
          f"({cnn_summary['frac_changed_0_5']['mean']*100:.1f}%)")
    print(f"  Jacobian frac > 0.5: {jac_summary['frac_changed_0_5']['mean']:.6f}  "
          f"({jac_summary['frac_changed_0_5']['mean']*100:.4f}%)")
    ratio = (cnn_summary['frac_changed_0_5']['mean'] /
             max(jac_summary['frac_changed_0_5']['mean'], 1e-9))
    print(f"  CNN is {ratio:.1f}x denser than Jacobian")

    print(f"\n--- Frequency band energy (luma DCT) ---")
    print(f"  CNN low:  {cnn_summary['band_low']['mean']:.3f}  "
          f"mid: {cnn_summary['band_mid']['mean']:.3f}  "
          f"high: {cnn_summary['band_high']['mean']:.3f}")
    print(f"  Jac low:  {jac_summary['band_low']['mean']:.3f}  "
          f"mid: {jac_summary['band_mid']['mean']:.3f}  "
          f"high: {jac_summary['band_high']['mean']:.3f}")

    print(f"\n{'=' * 78}")
    print(f"KARPATHY PREDICTION VERIFICATION")
    print(f"{'=' * 78}")
    pred_dense = cnn_summary['frac_changed_0_5']['mean'] > 0.5
    pred_small_amp = cnn_summary['abs_mean']['mean'] < 2.0
    pred_mid_bias = cnn_summary['band_mid']['mean'] > cnn_summary['band_low']['mean']
    pred_jac_sparse = jac_summary['frac_changed_0_5']['mean'] < 0.01
    pred_jac_spike = jac_summary['abs_max']['mean'] > 1.0

    print(f"  CNN is dense (>50% pixels moved):         {'✓ TRUE' if pred_dense else '✗ FALSE'}")
    print(f"  CNN is small-amplitude (mean |δ|<2):      {'✓ TRUE' if pred_small_amp else '✗ FALSE'}")
    print(f"  CNN is mid-freq biased (mid > low):       {'✓ TRUE' if pred_mid_bias else '✗ FALSE'}")
    print(f"  Jacobian is sparse (<1% pixels moved):    {'✓ TRUE' if pred_jac_sparse else '✗ FALSE'}")
    print(f"  Jacobian is spike-like (max > 1.0):       {'✓ TRUE' if pred_jac_spike else '✗ FALSE'}")

    all_confirmed = pred_dense and pred_small_amp and pred_mid_bias and pred_jac_sparse and pred_jac_spike
    if all_confirmed:
        print(f"\n*** Karpathy's inverse-rendering hypothesis CONFIRMED ***")
        print(f"The CNN spreads corrections while the Jacobian concentrates.")
        print(f"This is why the linear approach fails and the CNN succeeds.")
    else:
        print(f"\n*** PARTIAL confirmation — some predictions missed ***")

    result = {"cnn": cnn_summary, "jacobian": jac_summary, "karpathy_confirmed": all_confirmed}
    print(f"\nJSON:\n{json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    main()
