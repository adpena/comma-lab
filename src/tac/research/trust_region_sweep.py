#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Measure PoseNet's honest linear trust radius in pixel units.

Karpathy's specific request: pick a random direction `d`, scan `alpha`
over [1e-4, 1e-3, 1e-2, 1e-1, 1, 10] pixel units, and plot

    linearization_error(alpha) = ||pose(x + alpha*d) - pose(x) - alpha * J @ d||

vs alpha. The knee of that curve is the honest linear radius of PoseNet.

Karpathy's prediction: the knee is well under 0.01 pixels. If true, this
kills any single-step closed-form correction family (Newton, quasi-Newton,
single-step trust region) and explains exactly why the Jacobian failed.

This also informs the test-time Newton experiment (panel #6): the trust
radius measured here dictates the max step size we can take without
leaving the linear region. Too small -> impractical. Workable -> worth
building the iterative Newton inflate method.

Usage::

    cd /tmp/pact-mine
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy python -u experiments/trust_region_sweep.py
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
from tac.research.jacobian_optimal import compute_jacobian, posenet_output_with_grad

_PROJECT, _UPSTREAM, VIDEOS_DIR, _LIVE_ARCHIVE, ARCHIVE_ZIP = _default_paths()
DEVICE = detect_device()


def main():
    print(f"[trust-region] device={DEVICE}")
    posenet, _ = load_scorers(DEVICE)

    print("[trust-region] Decoding archive + GT...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_pairs = build_pairs(comp_frames[:n])
    n_pairs = len(comp_pairs)
    del comp_frames, gt_frames
    gc.collect()

    # Test points: 10 pairs spread across the video
    test_indices = np.linspace(0, n_pairs - 1, 10, dtype=int).tolist()
    # Alpha values (pixel units along a random unit direction)
    alphas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]

    print(f"[trust-region] Measuring linearization error at {len(test_indices)} pairs "
          f"across {len(alphas)} alpha values")
    print("[trust-region] Alpha units are pixel RMS (direction is random unit vector)")

    # For each pair and each alpha, compute:
    #   exact_delta = pose(x + alpha*d) - pose(x)
    #   linear_delta = alpha * J @ d
    #   error = ||exact_delta - linear_delta|| / ||exact_delta||  (relative error)
    # Average over 3 random directions for stability.

    results = {alpha: [] for alpha in alphas}
    n_dirs = 3

    for i, idx in enumerate(test_indices):
        comp_pair = comp_pairs[idx].to(DEVICE).float()  # (1, 2, H, W, 3)
        J, pose_at_x = compute_jacobian(posenet, comp_pair)  # J: (6, N)
        N = J.shape[1]

        # Reference pose norm (for relative error)
        pose_norm = pose_at_x.pow(2).sum().sqrt().clamp(min=1e-12)

        for d_idx in range(n_dirs):
            # Random unit direction in pixel space
            torch.manual_seed(idx * 7 + d_idx)
            d = torch.randn(N, device=DEVICE, dtype=comp_pair.dtype)
            d = d / d.pow(2).sum().sqrt().clamp(min=1e-12)  # unit L2 vector
            # d has RMS ~1/sqrt(N), so alpha*d has per-pixel scale ~alpha/sqrt(N)
            # To make alpha the interpretable "pixel RMS" we rescale:
            d = d * math.sqrt(N)  # now d has RMS = 1
            # Now (alpha * d).rms() == alpha in pixel units

            for alpha in alphas:
                delta = alpha * d
                delta_reshaped = delta.reshape(comp_pair.shape)
                x_plus = (comp_pair + delta_reshaped).clamp(0, 255)

                # Exact delta
                with torch.no_grad():
                    pose_exact = posenet_output_with_grad(posenet, x_plus)
                    exact_delta_pose = pose_exact - pose_at_x

                # Linear prediction
                linear_delta_pose = (J @ delta).squeeze()

                # Error (absolute and relative)
                abs_err = (exact_delta_pose - linear_delta_pose).pow(2).sum().sqrt().item()
                exact_mag = exact_delta_pose.pow(2).sum().sqrt().item()
                rel_err = abs_err / (exact_mag + 1e-12)  # REL_ERR_NON_CANONICAL_OK: L2 ratio of pose deltas in trust-region sweep (Bayesian sensitivity); not allocator-fed

                results[alpha].append({
                    "abs_err": abs_err,
                    "exact_mag": exact_mag,
                    "rel_err": rel_err,
                })

        if (i + 1) % 3 == 0:
            print(f"  processed {i+1}/{len(test_indices)} pairs")

    # Summarize
    print(f"\n{'=' * 78}")
    print("TRUST REGION SWEEP RESULTS")
    print(f"{'=' * 78}")
    print(f"{'alpha (px)':>12} {'mean_abs_err':>14} {'mean_exact':>14} "
          f"{'rel_err':>10} {'samples':>8}")
    print("-" * 78)
    summary = []
    for alpha in alphas:
        entries = results[alpha]
        mean_abs = np.mean([e["abs_err"] for e in entries])
        mean_exact = np.mean([e["exact_mag"] for e in entries])
        mean_rel = np.mean([e["rel_err"] for e in entries])
        median_rel = np.median([e["rel_err"] for e in entries])
        print(f"{alpha:>12.4g} {mean_abs:>14.6e} {mean_exact:>14.6e} "
              f"{mean_rel:>10.3f} {len(entries):>8d}")
        summary.append({
            "alpha": alpha,
            "mean_abs_err": mean_abs,
            "mean_exact_mag": mean_exact,
            "mean_rel_err": mean_rel,
            "median_rel_err": median_rel,
        })

    # Find the knee: first alpha where relative error > 0.5 (50%)
    knee_alpha = None
    for s in summary:
        if s["median_rel_err"] > 0.5:
            knee_alpha = s["alpha"]
            break

    print(f"\n{'=' * 78}")
    print("INTERPRETATION")
    print(f"{'=' * 78}")
    if knee_alpha is None:
        print(f"⚠ Relative error never exceeded 50% even at alpha={alphas[-1]} pixels.")
        print("  PoseNet is remarkably linear at this scale. Single-step methods")
        print("  could actually work — the Jacobian failure may have been due to")
        print("  something else (conditioning, numerical precision).")
    else:
        print(f"⚑ Trust radius knee: alpha ≈ {knee_alpha:.4g} pixels")
        if knee_alpha < 0.01:
            print("  VERY SMALL. Karpathy's prediction CONFIRMED.")
            print("  Single-step methods are dead on arrival. Only iterative")
            print("  small-step descent (i.e., SGD) works. The CNN is the right tool.")
            print("  Test-time Newton is NOT viable — step size would be too small")
            print("  to make progress within inflate time budget.")
        elif knee_alpha < 0.1:
            print("  SMALL but workable. Iterative Newton at inflate time is")
            print("  possible if we can fit ~50 steps per frame in the budget.")
        else:
            print("  WORKABLE. Newton at inflate time is viable with ~5-10 steps.")

    print("\nFor comparison, the Jacobian experiment's delta was 0.012 RMS")
    print("with max 3.07 pixels — way outside the linear region.")

    result = {
        "summary": summary,
        "knee_alpha": knee_alpha,
        "n_test_pairs": len(test_indices),
        "n_directions": n_dirs,
    }
    print(f"\nJSON:\n{json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    main()
