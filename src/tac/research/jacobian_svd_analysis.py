#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""SVD analysis of the per-pair Jacobian J = dPose/dPixel.

Even though applying a single-step pseudoinverse correction fails (the
nonlinearity dominates), the SVD of J still tells us useful things:

1. **Effective rank**: If J has rank < 6, the pose-output dimensions are
   not all independently controllable at this pair. That suggests
   PoseNet's decision depends on fewer than 6 distinct pixel-space
   directions.

2. **Singular value spectrum**: The relative magnitudes tell us how much
   each pose direction costs in pixel perturbation. Large σ = cheap,
   small σ = expensive.

3. **Cross-pair averaging**: If we stack J across many pairs, the top
   singular vectors of the combined matrix reveal global pixel directions
   that consistently affect pose — a kind of "universal adversarial
   perturbation" direction.

This is a diagnostic, not a training run. It informs architecture
choices (e.g. "the effective pose direction dimension is 4, so a rank-4
parameter factorization might suffice") and tells us if there's hidden
structure our CNN is wasting capacity to discover.

Usage::

    cd /tmp/pact-mine
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy python -u experiments/jacobian_svd_analysis.py
"""
from __future__ import annotations

import gc
import json
import math

import numpy as np
import torch

from tac.data import build_pairs, decode_archive, decode_video
from tac.scorer import detect_device, load_scorers
from tac.proxy_eval import _default_paths
from tac.research.jacobian_optimal import compute_jacobian

_PROJECT, _UPSTREAM, VIDEOS_DIR, _LIVE_ARCHIVE, ARCHIVE_ZIP = _default_paths()
UPSTREAM = _UPSTREAM
DEVICE = detect_device()


def main():
    print(f"[svd] device={DEVICE}")
    print("[svd] Loading PoseNet...")
    posenet, segnet = load_scorers(DEVICE)

    print("[svd] Decoding archive + GT...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_pairs = build_pairs(comp_frames[:n])
    gt_pairs = build_pairs(gt_frames[:n])
    n_pairs = len(comp_pairs)
    print(f"[svd] {n_pairs} frame pairs")

    del comp_frames, gt_frames
    gc.collect()

    # Analyze per-pair SVDs and collect effective ranks
    n_analyze = 30  # subsample for speed: 30 pairs is enough for a stable average
    indices = np.linspace(0, n_pairs - 1, n_analyze, dtype=int).tolist()

    per_pair_singular_values = []
    per_pair_effective_rank = []
    jjt_traces = []

    print(f"\n[svd] Per-pair SVD analysis on {n_analyze} pairs...")
    print(f"{'pair':>5} {'sv1':>10} {'sv2':>10} {'sv3':>10} {'sv4':>10} "
          f"{'sv5':>10} {'sv6':>10} {'eff_rank':>10}")
    print("-" * 80)

    for i, idx in enumerate(indices):
        comp_pair = comp_pairs[idx].to(DEVICE).float()
        J, _ = compute_jacobian(posenet, comp_pair)  # (6, N)
        # Singular values via eigendecomposition of J J^T (6x6)
        JJT = (J @ J.T).cpu().double()
        # Protect against numerical issues
        eigvals, _ = torch.linalg.eigh(JJT)
        # Sort descending, clip negatives to zero, take sqrt
        eigvals = eigvals.flip(0).clamp(min=0.0)
        svals = eigvals.sqrt()
        per_pair_singular_values.append(svals.numpy())
        # Effective rank via entropy of normalized squared singular values
        s2 = eigvals / eigvals.sum().clamp(min=1e-12)
        entropy = -(s2 * s2.clamp(min=1e-12).log()).sum().item()
        effective_rank = math.exp(entropy)
        per_pair_effective_rank.append(effective_rank)
        jjt_traces.append(eigvals.sum().item())
        if (i + 1) % 5 == 0 or i < 3:
            sv_list = svals.tolist()
            print(f"{i+1:>5d} " + " ".join(f"{v:>10.4f}" for v in sv_list[:6]) +
                  f" {effective_rank:>10.3f}")

    per_pair_singular_values = np.stack(per_pair_singular_values)  # (n_analyze, 6)
    mean_svs = per_pair_singular_values.mean(axis=0)
    median_svs = np.median(per_pair_singular_values, axis=0)
    mean_eff_rank = float(np.mean(per_pair_effective_rank))
    median_eff_rank = float(np.median(per_pair_effective_rank))

    print(f"\n{'=' * 70}")
    print("RESULTS: Per-pair J singular value statistics")
    print(f"{'=' * 70}")
    print(f"Mean singular values  (n={n_analyze}): {mean_svs}")
    print(f"Median singular values:                {median_svs}")
    print(f"Mean effective rank (entropy-based):   {mean_eff_rank:.3f} / 6")
    print(f"Median effective rank:                 {median_eff_rank:.3f} / 6")
    print(f"Mean ratio sv_top/sv_6:                {mean_svs[0]/mean_svs[5]:.2f}")
    print(f"Fraction of variance in top 3:         "
          f"{(mean_svs[:3]**2).sum() / (mean_svs**2).sum():.3f}")
    print(f"Fraction of variance in top 4:         "
          f"{(mean_svs[:4]**2).sum() / (mean_svs**2).sum():.3f}")

    # Interpretation
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")
    if median_eff_rank < 4:
        print(f"⚠ Effective rank is LOW ({median_eff_rank:.2f} < 4).")
        print("  This means PoseNet's 6-dim pose output is really controlled")
        print("  by fewer than 4 independent pixel-space directions at our operating point.")
        print("  A rank-reduced filter architecture could match full performance")
        print("  with fewer parameters.")
    elif median_eff_rank > 5.5:
        print(f"✓ Effective rank is HIGH ({median_eff_rank:.2f} > 5.5).")
        print("  All 6 pose dimensions are independently controllable —")
        print("  our h=32 architecture has adequate dimensionality.")
    else:
        print(f"○ Effective rank is MODERATE ({median_eff_rank:.2f}).")
        print("  Some pose dimensions are coupled. A mild rank reduction")
        print("  (e.g. factorized conv) might save a few params without loss.")

    condition = mean_svs[0] / mean_svs[5]
    if condition > 100:
        print(f"\n⚠ Condition number is LARGE ({condition:.0f}).")
        print("  The linear model is ILL-CONDITIONED — this explains why")
        print("  the single-step pseudoinverse failed: a tiny error in the")
        print(f"  estimated residual gets amplified by ~{condition:.0f}x in the")
        print("  direction of the smallest singular value.")
    else:
        print(f"\n○ Condition number is moderate ({condition:.0f}).")

    result = {
        "n_pairs_analyzed": n_analyze,
        "mean_singular_values": mean_svs.tolist(),
        "median_singular_values": median_svs.tolist(),
        "mean_effective_rank": mean_eff_rank,
        "median_effective_rank": median_eff_rank,
        "mean_condition_number": float(mean_svs[0] / mean_svs[5]),
        "fraction_variance_top_3": float((mean_svs[:3]**2).sum() / (mean_svs**2).sum()),
        "fraction_variance_top_4": float((mean_svs[:4]**2).sum() / (mean_svs**2).sum()),
    }
    print(f"\nJSON:\n{json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    main()
