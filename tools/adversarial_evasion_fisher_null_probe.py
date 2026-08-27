# SPDX-License-Identifier: MIT
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""Adversarial / detection-evasion (UNIWARD dual) probe on the frozen-SegNet Fisher field.

FRAME (operator P0 2026-07-15). The witness = PROJECTION onto the correct argmax
cell. Some necessary flips are uint8-UNREACHABLE (the realization-limited sub-LSB
tail, `segnet_head_rank4_linear_flipdist_v1`). For that IRREDUCIBLE residual,
projection fails; the DUAL strategy (Yousfi / UNIWARD, inverse-steganalysis) is
EVASION: place the unavoidable d_seg cost where the detector (SegNet) is
Fisher-FLAT (large margin, low sensitivity), so the same visual error costs less
DETECTION. The scorer IS a detector; d_seg IS its detection rate; the margin field
IS the Fisher surrogate (`½·sech²(m/2)`, curvature↔(−margin) Pearson 0.978,
`tac.information_geometry.optimal_metric`).

This probe is CACHED-ONLY and standalone (no live trainer, no SegNet forward): it
reads the n96 cached per-pixel margin field + labels (`gt_n96.npz` margins/lstars,
bit-exact vs the frozen SegNet per `segnet_recursive_fractal_factorization_20260715`)
and the closed-form head gain anchor, and computes:

  A. the DETECTABILITY landscape (Fisher trace concentration = the annulus, #333);
  B. the IRREDUCIBLE realization-floor residual (quantization-only, DERIVED scale);
  C. the UNIWARD detectability COST density cost(m) = minimal render-L2 to flip a
     pixel of margin m, and the EVADABLE FRACTION counterfactual: naive (uniform)
     error placement vs UNIWARD-optimal (route budget to the Fisher-flat interior).

Authority: `[macOS-CPU advisory]` — the cached margins reproduce the frozen SegNet
argmax bit-exactly (0 mismatches / 96 frames, stage_b1.json). research_only;
score_claim=false; pointer 0.19108 UNMOVED (MEANS). Numbers labelled
MEASURED (from the cached field) vs DERIVED (from the closed-form gain model).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

GT_N96 = Path("experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
OUT = Path("experiments/results/adversarial_evasion_fisher_null_20260715")

# comma10k class order (measured, CLAUDE.md).
CLASSES = ["Road", "Lane", "Undrivable", "Movable", "MyCar"]

# --- Closed-form head gain anchor (MEASURED, segnet_head_rank4_linear_flipdist_v1) ---
# Median Road-Lane boundary pixel: logit margin 0.516, first-order minimal-flip
# input perturbation L2 = 8.8 over the (384,512,3) frame (memo §3).
ANCHOR_MARGIN = 0.516
ANCHOR_FLIP_L2 = 8.8
FRAME_ELEMS = 384 * 512 * 3
# Aligned input->margin gradient norm ||grad m|| = margin / ||delta_flip|| (exact
# minimal aligned perturbation = margin / gradient-norm).
GRAD_M_NORM_AT_ANCHOR = ANCHOR_MARGIN / ANCHOR_FLIP_L2  # ~0.0586 margin per unit input-L2
# margin<->Fisher (0.978): ||grad m_p|| ~= G * sech(m_p/2); solve G at the anchor.
G_GAIN = GRAD_M_NORM_AT_ANCHOR / (1.0 / math.cosh(ANCHOR_MARGIN / 2.0))  # ~0.0606

# uint8 half-LSB quantization: uniform[-0.5,0.5] -> RMS per element = 1/sqrt(12).
HALF_LSB_RMS = 1.0 / math.sqrt(12.0)  # ~0.2887 (0-255 units)


def fisher_trace(m: np.ndarray) -> np.ndarray:
    """Exact 2-class Fisher trace surrogate tr g = 1/2 sech^2(m/2) (optimal_metric)."""
    return 0.5 / np.cosh(m / 2.0) ** 2


def grad_m_norm(m: np.ndarray) -> np.ndarray:
    """DERIVED per-pixel input->margin gradient norm from the 0.978 relation + anchor."""
    return G_GAIN / np.cosh(m / 2.0)


def flip_cost_render_l2(m: np.ndarray) -> np.ndarray:
    """Minimal ALIGNED render-L2 (0-255 units) to flip a pixel of margin m.

    cost(m) = m / ||grad m|| = (m / G) * cosh(m/2)  -- grows ~ m*e^{m/2}: the
    UNIWARD detectability cost. Annulus (m->0): cost->0 (free to flip = detectable);
    interior (m large): cost->inf exponentially (evasion-safe).
    """
    return (m / G_GAIN) * np.cosh(m / 2.0)


def margin_jitter_from_render_rms(rho: float, m: np.ndarray) -> np.ndarray:
    """Expected margin perturbation (1-sigma) from per-element render error RMS rho.

    Random (unaligned) error: E[dm^2]^{1/2} = rho * ||grad m_p||. (DERIVED, first-order.)
    """
    return rho * grad_m_norm(m)


def main() -> None:
    d = np.load(GT_N96)
    margins = d["margins"].astype(np.float64)  # (96,384,512) winner-vs-runnerup logit margin
    lstars = d["lstars"]  # (96,384,512) argmax labels (target)
    n, H, W = margins.shape
    npix = margins.size
    m = margins.ravel()
    lab = lstars.ravel()

    report: dict[str, object] = {
        "scope": {
            "authority": "[macOS-CPU advisory] cached frozen-SegNet margin field (bit-exact argmax vs gt_n96)",
            "frames": f"n96 real GT frame_1 ({n} frames, {H}x{W})",
            "npix": int(npix),
            "research_only": True,
            "score_claim": False,
            "pointer": "0.19108 UNMOVED (MEANS)",
            "gain_anchor": {
                "anchor_margin": ANCHOR_MARGIN,
                "anchor_flip_L2_frame": ANCHOR_FLIP_L2,
                "grad_m_norm_at_anchor_MEASURED": GRAD_M_NORM_AT_ANCHOR,
                "G_gain_DERIVED": G_GAIN,
                "half_lsb_rms": HALF_LSB_RMS,
            },
        }
    }

    # ---- A. DETECTABILITY landscape (Fisher trace concentration = annulus, #333) ----
    F = fisher_trace(m)
    order = np.argsort(m)  # ascending margin = descending detectability
    Fsorted = F[order]
    cumF = np.cumsum(Fsorted) / F.sum()
    # area fraction holding X% of total Fisher detectability mass
    def area_for_mass(frac: float) -> float:
        idx = int(np.searchsorted(cumF, frac))
        return idx / npix
    # margin thresholds and the area + detectability-mass below them
    marg_thresholds = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    annulus_rows = []
    for t in marg_thresholds:
        below = m < t
        area = float(below.mean())
        fmass = float(F[below].sum() / F.sum())
        annulus_rows.append({"margin_below": t, "area_frac": area, "fisher_mass_frac": fmass})
    report["A_detectability_landscape"] = {
        "fisher_trace_law": "tr g = 0.5*sech^2(m/2) (optimal_metric, exact)",
        "margin_pctiles": {
            p: float(np.percentile(m, p)) for p in [0.1, 1, 4.7, 10, 25, 50, 90]
        },
        "area_holding_fisher_mass": {
            "50pct": area_for_mass(0.50),
            "90pct": area_for_mass(0.90),
            "97pct": area_for_mass(0.97),
            "99pct": area_for_mass(0.99),
        },
        "margin_threshold_rows": annulus_rows,
    }

    # ---- B. IRREDUCIBLE realization-floor residual (quantization only) ----
    # A perfect render still suffers uint8 half-LSB quantization. Margin jitter
    # RMS at pixel p = HALF_LSB_RMS * ||grad m_p|| (random error). A pixel flips
    # (~1-2 sigma) when jitter >~ m_p. DERIVED scale.
    for k_sigma in (1.0, 2.0):
        jit = margin_jitter_from_render_rms(HALF_LSB_RMS, m)  # per-pixel 1-sigma margin jitter
        flip = m < k_sigma * jit  # margin below k-sigma of quantization jitter
        rate = float(flip.mean())
        per_class = {
            CLASSES[c]: float(flip[lab == c].mean()) if (lab == c).any() else 0.0
            for c in range(5)
        }
        report.setdefault("B_realization_floor", {})[f"k_sigma_{k_sigma:g}"] = {
            "d_seg_floor_rate": rate,
            "median_quant_margin_jitter": float(np.median(jit)),
            "per_class_flip_rate": per_class,
            "note": "quantization-only irreducible residual: no render precision (uint8) can save these",
        }

    # ---- C. UNIWARD cost density + EVADABLE-FRACTION counterfactual ----
    cost = flip_cost_render_l2(m)  # minimal aligned render-L2 to flip each pixel
    # Intrinsic (unevadable) annulus = pixels whose flip cost is below the
    # quantization floor: even a boundary-focused render cannot keep them unflipped.
    quant_floor_cost = HALF_LSB_RMS * math.sqrt(3.0)  # ~0.5 (a half-LSB aligned nudge, order-of-magnitude)
    intrinsic = cost < quant_floor_cost
    intrinsic_rate = float(intrinsic.mean())

    # Sweep a naive per-element render-error RMS rho (0-255 units). Under NAIVE
    # (MSE, margin-blind) placement the same rho hits every pixel -> flips where
    # jitter > margin. Under UNIWARD the same TOTAL budget rho^2*npix is routed
    # to the Fisher-flat interior (highest-cost pixels, which never flip), so the
    # only remaining flips are the intrinsic annulus (cost below floor). Evadable
    # fraction = (naive - intrinsic)/naive.
    rho_rows = []
    for rho in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        jit = margin_jitter_from_render_rms(rho, m)
        naive_flip = m < jit  # 1-sigma naive
        naive_rate = float(naive_flip.mean())
        # UNIWARD floor = intrinsic annulus (reallocation cannot save cost<floor)
        uniward_rate = float((naive_flip & intrinsic).mean())
        evadable = 0.0 if naive_rate == 0 else (naive_rate - uniward_rate) / naive_rate
        # of the naive flips, how many sit in the Fisher-flat interior (margin above
        # the 4.7%-area annulus threshold) = pure interior leak (fully evadable)
        annulus_thr = float(np.percentile(m, 4.7))
        interior_leak = float((naive_flip & (m >= annulus_thr)).mean())
        rho_rows.append({
            "rho_render_rms_0_255": rho,
            "naive_d_seg_rate": naive_rate,
            "uniward_floor_rate": uniward_rate,
            "evadable_fraction": evadable,
            "interior_leak_rate": interior_leak,
            "interior_leak_frac_of_naive": 0.0 if naive_rate == 0 else interior_leak / naive_rate,
        })
    report["C_uniward_evasion"] = {
        "cost_density_law": "cost(m) = (m/G)*cosh(m/2)  [minimal aligned render-L2 to flip], DERIVED",
        "intrinsic_annulus_rate_MEASURED": intrinsic_rate,
        "quant_floor_cost_used": quant_floor_cost,
        "annulus_thr_margin_4p7pct": float(np.percentile(m, 4.7)),
        "rho_sweep": rho_rows,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    outp = OUT / "evasion_probe.json"
    outp.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n[written] {outp}")


if __name__ == "__main__":
    main()
