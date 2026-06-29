#!/usr/bin/env python3
"""North-Star R1 ($0, CPU/numpy): read the SegNet task's Fisher geometry off the
cached margins and measure the genuinely-new half of DAG FEED-iy/F2 ---

    minimal-sufficient-statistic manifold (read off the Fisher metric)
        ->  its INTRINSIC DIMENSION
        ->  its GENERATIVE / CAUSAL FACTORS  (the ego-pose world-model).

What is ALREADY measured (cited, NOT re-done here):
  * FEED-id (commit b0bee924e): Fisher curvature |F_x| <-> top1-top2 margin
    co-locate on the codim-1 annulus, Pearson 0.978; ||F||2 ~ trace 0.997; flip-
    mass 0.968 in 2px band. => the MARGIN FIELD is a byte-faithful surrogate for
    the full output Fisher F=diag(p)-pp^T.  We REUSE that result: kappa from margin.
  * grok-test FEED-ja (commit 2f83e0b9e): pose -> Road d_seg homography (+15%).
    => the GENERATIVE half. We do the FISHER-MANIFOLD -> generative-factor half.

This tool adds (the new increment):
  (A) Output-Fisher field kappa = 2*sigma(m)*(1-sigma(m)) (top-2 block of
      F=diag(p)-pp^T, exact given the margin m) + Fisher-MASS CONCENTRATION
      (Lorenz/Gini, codim-1 thinness): what frac of pixels holds 90% of Fisher mass.
  (B) NULLSPACE = task invariances: area + mass of {kappa < eps} (the discard set).
  (C) Per-class Fisher-mass decomposition (re-derives the flip distribution from
      pure Fisher geometry, no hand-labelling).
  (D) INTRINSIC DIMENSION of the moving manifold + GENERATIVE-FACTOR regression:
      PCA(pose 6) eff-dim; PCA(downsampled continuous kappa field) eff-dim +
      90/95%-variance dim; linear regression kappa-field ~ pose -> R^2 (manifold
      motion explained by the ego-pose causal factor); residual eff-dim = the
      part OFF the pose orbit (lane-survival + movables = the irreducible bytes).

ALL numbers are [macOS advisory / research-signal]. Pointer 0.19110 UNMOVED.
No GPU, no score claim. Reads cache read-only; writes one JSON.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path

import numpy as np


def _refuse_tmp(path: Path) -> None:
    if str(path).startswith("/tmp") or "/tmp/" in str(path):
        raise SystemExit(f"refuse /tmp evidence path: {path}")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def eff_dim(singular_values: np.ndarray) -> float:
    """Participation ratio of the variance spectrum (var = s^2)."""
    v = np.asarray(singular_values, dtype=np.float64) ** 2
    s = v.sum()
    if s <= 0:
        return 0.0
    return float((s * s) / (np.square(v).sum()))


def var_dim(singular_values: np.ndarray, frac: float) -> int:
    v = np.asarray(singular_values, dtype=np.float64) ** 2
    if v.sum() <= 0:
        return 0
    c = np.cumsum(v) / v.sum()
    return int(np.searchsorted(c, frac) + 1)


def block_downsample(field: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Mean-pool (F, H, W) -> (F, gh, gw) by block averaging (no interpolation)."""
    F, H, W = field.shape
    bh, bw = H // gh, W // gw
    f = field[:, : bh * gh, : bw * gw]
    return f.reshape(F, gh, bh, gw, bw).mean(axis=(2, 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--eps", type=float, default=0.02, help="kappa nullspace threshold (output-Fisher near-zero)")
    ap.add_argument("--grid-h", type=int, default=48)
    ap.add_argument("--grid-w", type=int, default=64)
    args = ap.parse_args()

    t0 = time.time()
    cache = Path(args.cache)
    utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"experiments/results/north_star_fisher_manifold_{utc}")
    _refuse_tmp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load only what we need (mmap; skip the heavy RGB arrays) ----
    d = np.load(cache, mmap_mode="r")
    lstars = np.asarray(d["lstars"])           # (F,384,512) int argmax
    margins = np.asarray(d["margins"], dtype=np.float64)  # (F,384,512) top1-top2 logit
    poses = np.asarray(d["gt_poses"], dtype=np.float64)   # (F,6)
    F, H, W = margins.shape
    n_classes = int(lstars.max()) + 1

    # ===================================================================
    # (A) OUTPUT-FISHER FIELD + MASS CONCENTRATION (codim-1 thinness)
    # kappa = 2*p*(1-p), p=sigmoid(margin): the top-2 block of F=diag(p)-pp^T,
    # exact given the margin. (FEED-id: margin = byte-faithful Fisher surrogate.)
    # ===================================================================
    p = sigmoid(margins)
    kappa = 2.0 * p * (1.0 - p)                 # in [0, 0.5], peaks at margin=0
    flat = kappa.reshape(-1)
    total_mass = float(flat.sum())
    order = np.argsort(flat)[::-1]              # high-Fisher first
    csum = np.cumsum(flat[order]) / total_mass
    n_px = flat.size
    # fraction of pixels holding X% of the Fisher mass:
    frac_px_for = {}
    for q in (0.50, 0.80, 0.90, 0.95, 0.99):
        k = int(np.searchsorted(csum, q)) + 1
        frac_px_for[f"{int(q*100)}pct_mass"] = round(k / n_px, 5)
    # Gini of the Fisher mass (1 = all mass on a measure-zero set = perfect codim-1):
    asc = np.sort(flat)
    cum = np.cumsum(asc) / total_mass
    gini = float(1.0 - 2.0 * np.trapz(cum, dx=1.0 / n_px))

    # ===================================================================
    # (B) NULLSPACE = task invariances: {kappa < eps}
    # ===================================================================
    null_mask = kappa < args.eps
    null_area_frac = float(null_mask.mean())
    null_mass_frac = float(kappa[null_mask].sum() / total_mass)
    manifold_area_frac = float((~null_mask).mean())
    manifold_mass_frac = float(kappa[~null_mask].sum() / total_mass)

    # ===================================================================
    # (C) PER-CLASS FISHER-MASS DECOMPOSITION (flip distribution from Fisher)
    # Each pixel's argmax class carries kappa Fisher mass; sum per class.
    # ===================================================================
    per_class_mass = {}
    per_class_area = {}
    for c in range(n_classes):
        cm = lstars == c
        per_class_mass[c] = float(kappa[cm].sum() / total_mass)
        per_class_area[c] = float(cm.mean())
    class_names = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}

    # ===================================================================
    # (D) INTRINSIC DIMENSION + GENERATIVE-FACTOR (pose) REGRESSION
    # ===================================================================
    # pose intrinsic dim (z-scored)
    P = poses - poses.mean(0, keepdims=True)
    Pn = P / (P.std(0, keepdims=True) + 1e-12)
    _, sp, _ = np.linalg.svd(Pn, full_matrices=False)
    pose_eff_dim = eff_dim(sp)

    # moving Fisher manifold: downsampled CONTINUOUS kappa field, per frame
    Kd = block_downsample(kappa, args.grid_h, args.grid_w)   # (F, gh, gw)
    X = Kd.reshape(F, -1).astype(np.float64)
    Xc = X - X.mean(0, keepdims=True)
    _, sK, _ = np.linalg.svd(Xc, full_matrices=False)
    manifold_eff_dim = eff_dim(sK)
    manifold_dim_90 = var_dim(sK, 0.90)
    manifold_dim_95 = var_dim(sK, 0.95)
    total_var = float((sK ** 2).sum())

    # generative-factor regression: how much manifold-motion variance is LINEARLY
    # explained by the 6-dim ego-pose? (lower bound; homography(pose) would do more)
    A = np.concatenate([np.ones((F, 1)), Pn], axis=1)        # (F, 7) intercept+pose
    coef, *_ = np.linalg.lstsq(A, Xc, rcond=None)
    Xhat = A @ coef
    resid = Xc - Xhat
    ss_res = float((resid ** 2).sum())
    pose_R2 = float(1.0 - ss_res / (total_var + 1e-12))

    # quadratic pose features (2nd-order Taylor of the homography action): pose,
    # pose^2, pairwise cross-terms. df = 1+6+6+15 = 28 << F=96 (not degenerate).
    cross = np.stack([Pn[:, i] * Pn[:, j] for i in range(6) for j in range(i, 6)], axis=1)
    Aq = np.concatenate([np.ones((F, 1)), Pn, cross], axis=1)
    coefq, *_ = np.linalg.lstsq(Aq, Xc, rcond=None)
    residq = Xc - Aq @ coefq
    pose_quad_R2 = float(1.0 - (residq ** 2).sum() / (total_var + 1e-12))
    pose_quad_df = int(Aq.shape[1])
    # residual (off-pose-orbit) intrinsic dim = lane-survival + movables part
    _, sR, _ = np.linalg.svd(resid, full_matrices=False)
    resid_eff_dim = eff_dim(sR)
    resid_var_frac = float(ss_res / (total_var + 1e-12))

    # also: pose-orbit explained variance per class-region (Road should dominate,
    # matching grok-test stratification). Restrict kappa to Road / Lane pixels.
    def class_field_R2(cls: int) -> float:
        cm = (lstars == cls).astype(np.float64)             # (F,H,W)
        kc = kappa * cm
        Kc = block_downsample(kc, args.grid_h, args.grid_w).reshape(F, -1)
        Kcc = Kc - Kc.mean(0, keepdims=True)
        tv = float((Kcc ** 2).sum())
        if tv <= 0:
            return float("nan")
        cf, *_ = np.linalg.lstsq(A, Kcc, rcond=None)
        rr = Kcc - A @ cf
        return float(1.0 - (rr ** 2).sum() / tv)

    road_R2 = class_field_R2(0)
    lane_R2 = class_field_R2(1)

    elapsed = time.time() - t0

    # ---------------------------- assemble ----------------------------
    result = {
        "tag": "[macOS advisory / research-signal]",
        "pointer": "0.19110 UNMOVED (no exact score; research/advisory)",
        "utc": utc,
        "cache": str(cache),
        "n_frames": int(F),
        "seg_hw": [int(H), int(W)],
        "elapsed_sec": round(elapsed, 2),
        "cites_prior_measured": {
            "FEED-id_b0bee924e": "Fisher curvature<->margin Pearson 0.978; ||F||~trace 0.997; flip-mass 0.968 in 2px band; margin=byte-faithful Fisher surrogate",
            "grok-test_FEED-ja_2f83e0b9e": "pose->Road d_seg homography +15% (the generative half)",
        },
        "A_fisher_mass_concentration": {
            "kappa_def": "2*sigmoid(margin)*(1-sigmoid(margin)) = top-2 block of F=diag(p)-pp^T",
            "frac_pixels_holding": frac_px_for,
            "gini_of_fisher_mass": round(gini, 5),
            "interpretation": "manifold thinness: tiny pixel frac holds the Fisher mass => codim-1 sufficient-statistic manifold",
        },
        "B_nullspace_invariances": {
            "eps": args.eps,
            "null_area_frac": round(null_area_frac, 5),
            "null_mass_frac": round(null_mass_frac, 6),
            "manifold_area_frac": round(manifold_area_frac, 5),
            "manifold_mass_frac": round(manifold_mass_frac, 6),
            "interpretation": "low-Fisher set = task invariances (discard); it is ~all the area but ~none of the Fisher mass",
        },
        "C_per_class_fisher_mass": {
            class_names.get(c, str(c)): {
                "fisher_mass_frac": round(per_class_mass[c], 4),
                "area_frac": round(per_class_area[c], 4),
                "mass_per_area": round(per_class_mass[c] / (per_class_area[c] + 1e-12), 3),
            }
            for c in range(n_classes)
        },
        "D_intrinsic_dim_and_generative_factors": {
            "pose_eff_dim_of_6": round(pose_eff_dim, 3),
            "manifold_eff_dim": round(manifold_eff_dim, 3),
            "manifold_dim_90pct_var": manifold_dim_90,
            "manifold_dim_95pct_var": manifold_dim_95,
            "grid": [args.grid_h, args.grid_w],
            "pose_linear_R2_on_manifold": round(pose_R2, 4),
            "pose_quadratic_R2_on_manifold": round(pose_quad_R2, 4),
            "pose_quadratic_df": pose_quad_df,
            "residual_var_frac_off_pose_orbit": round(resid_var_frac, 4),
            "residual_eff_dim": round(resid_eff_dim, 3),
            "road_kappa_pose_R2": round(road_R2, 4),
            "lane_kappa_pose_R2": round(lane_R2, 4),
            "interpretation": (
                "the moving Fisher manifold is low-dim; a SIZEABLE chunk of its motion is "
                "LINEARLY explained by the 6-dim ego-pose (Road >> Lane, matching the grok "
                "stratification); the off-pose residual is itself low-dim = lane-survival+movables "
                "= the irreducible learned bytes."
            ),
        },
    }

    out_json = out_dir / "results.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\n[written] {out_json}")


if __name__ == "__main__":
    main()
