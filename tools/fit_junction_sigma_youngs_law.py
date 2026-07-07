#!/usr/bin/env python
"""$0 junction sigma_ij Young's-law fit from the cached frozen-SegNet GT argmax (n600).

Theory (viscosity_theory_alignment_hunt_20260707.md §7, EUREKA candidate #2): the multiphase
Modica-Mortola Gamma-limit imposes Herring angle conditions at triple junctions set by the
surface-tension matrix sigma_ij; with the all-ones length weight (today's ``--length-weight``,
a single scalar) that means 120-120-120. The frozen SegNet's partition has NO reason to satisfy
equal-tension Herring angles — the perimeter regularizer therefore imposes a WRONG junction
boundary condition (Imbert-Monneau: the junction condition is a FREE parameter). The $0 fit:

  1. detect triple junctions in the cached GT argmax (2x2 plaquettes with exactly 3 distinct
     classes; canonical class order Road0/Lane1/Undrivable2/Movable3/MyCar4 — NEVER luma-sorted);
  2. measure the three interior angles per junction by walking a radius-r circle and requiring a
     CLEAN junction (exactly 3 circular label transitions => contiguous arcs);
  3. invert Young's law  sigma_jk/sin(theta_i) = sigma_ik/sin(theta_j) = sigma_ij/sin(theta_k)
     per class-triple, then solve a weighted least-squares in log sigma across all triples
     (gauge: geometric mean of observed sigma_ij = 1, so all-ones IS the null hypothesis);
  4. bootstrap CIs over junctions.

Sub-pixel junction refinement is SKIPPED (stated per the task): angles are measured on a
radius-r circle around the integer plaquette corner; the +-0.5 px center error adds per-junction
angle noise that averages out over thousands of junctions (and is absorbed by the bootstrap CI).

All numbers [macOS-CPU advisory] — research signal, never a score; pointer 0.19110 moves only
via upstream/evaluate.py. Peak RSS ~1.2 GiB (the int64 lstars load), well under the 10 GiB cap.

Usage:
  .venv/bin/python tools/fit_junction_sigma_youngs_law.py \
      --out-dir experiments/results/solver_pack_20260707/junction_sigma
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"

CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
ADVISORY = "[macOS-CPU advisory] NON-PROMOTABLE research-signal; pointer 0.19110 UNMOVED"

_POPCOUNT = np.array([bin(i).count("1") for i in range(32)], dtype=np.uint8)


def detect_triple_junctions(lstars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2x2-plaquette triple-junction detector on (P,H,W) uint8 labels (5 classes).

    Returns (coords, quad_count): coords is (N,3) int32 [pair, y, x] where the junction sits at
    plaquette corner (y+0.5, x+0.5); quad_count is the number of 4-distinct-class plaquettes
    (excluded from the fit, reported for completeness).
    """
    a = lstars[:, :-1, :-1]
    b = lstars[:, :-1, 1:]
    c = lstars[:, 1:, :-1]
    d = lstars[:, 1:, 1:]
    mask = ((1 << a.astype(np.int32)) | (1 << b.astype(np.int32))
            | (1 << c.astype(np.int32)) | (1 << d.astype(np.int32)))
    ndistinct = _POPCOUNT[mask]
    triple = np.argwhere(ndistinct == 3).astype(np.int32)
    quad_count = int((ndistinct == 4).sum())
    return triple, quad_count


def measure_junction_angles(lstars: np.ndarray, coords: np.ndarray, *,
                            radius: float = 4.0, n_samples: int = 120,
                            ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Interior angles at each junction from a radius-r circle walk (vectorized).

    For each junction, sample the label at n_samples points on the circle of ``radius`` around
    (y+0.5, x+0.5). A junction is CLEAN iff the circular label sequence has exactly 3 transitions
    (=> each class's arc is contiguous) and the circle's class set equals a 3-class set. Returns
    (triples (M,3) sorted class ids, angles_deg (M,3) aligned to the sorted triple, stats).
    """
    P, H, W = lstars.shape
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    dy = radius * np.sin(theta)
    dx = radius * np.cos(theta)

    pi = coords[:, 0].astype(np.int64)
    yc = coords[:, 1].astype(np.float64) + 0.5
    xc = coords[:, 2].astype(np.float64) + 0.5
    yy = np.rint(yc[:, None] + dy[None, :]).astype(np.int64)
    xx = np.rint(xc[:, None] + dx[None, :]).astype(np.int64)
    in_bounds = ((yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)).all(axis=1)

    stats = {"n_candidates": int(coords.shape[0]),
             "n_out_of_bounds": int((~in_bounds).sum())}
    pi, yy, xx = pi[in_bounds], yy[in_bounds], xx[in_bounds]

    labels = lstars[pi[:, None], yy, xx]          # (N, n_samples) uint8
    rolled = np.roll(labels, -1, axis=1)
    transitions = (labels != rolled).sum(axis=1)
    clean = transitions == 3
    stats["n_not_clean_at_radius"] = int((~clean).sum())
    labels = labels[clean]

    # class sets: exactly 3 distinct classes on the circle
    onehot_counts = np.zeros((labels.shape[0], 5), dtype=np.int32)
    for k in range(5):
        onehot_counts[:, k] = (labels == k).sum(axis=1)
    n_classes = (onehot_counts > 0).sum(axis=1)
    ok = n_classes == 3
    stats["n_circle_class_set_not_3"] = int((~ok).sum())
    onehot_counts = onehot_counts[ok]
    stats["n_clean_junctions"] = int(onehot_counts.shape[0])

    deg_per_sample = 360.0 / labels.shape[1] if labels.shape[1] else 0.0
    triples = np.zeros((onehot_counts.shape[0], 3), dtype=np.int8)
    angles = np.zeros((onehot_counts.shape[0], 3), dtype=np.float64)
    present = onehot_counts > 0
    for row in range(onehot_counts.shape[0]):
        cls = np.flatnonzero(present[row])
        triples[row] = cls
        angles[row] = onehot_counts[row, cls] * deg_per_sample
    return triples, angles, stats


def fit_sigma_from_angles(triples: np.ndarray, angles: np.ndarray, *,
                          min_junctions_per_triple: int = 30) -> dict:
    """Per-triple mean angles -> Young's-law log-least-squares for the 5x5 sigma matrix.

    Young's law at a junction of classes (i,j,k) with interior angles (th_i,th_j,th_k):
    sigma_jk/sin(th_i) = sigma_ik/sin(th_j) = sigma_ij/sin(th_k). Per triple this gives two
    independent linear equations in x_pq = log sigma_pq, weighted by sqrt(junction count).
    Gauge: mean over OBSERVED pairs of x = 0 (geometric mean 1 => all-ones is the null).
    Junctions with any arc >= 180 deg violate positive-tension equilibrium (reported, dropped).
    """
    keys = [tuple(t) for t in triples.tolist()]
    uniq = sorted(set(keys))
    per_triple: dict = {}
    eqs: list[tuple[dict, float, float]] = []  # (coeffs {pair: +-1}, rhs, weight)
    pair_ids: set[tuple[int, int]] = set()

    for tri in uniq:
        sel = np.array([k == tri for k in keys])
        A = angles[sel]                                   # (n,3) aligned to sorted (i,j,k)
        wide = (A >= 180.0).any(axis=1)
        n_wide = int(wide.sum())
        A = A[~wide]
        n = A.shape[0]
        i, j, k = tri
        entry = {
            "triple": [CLASS_NAMES[i], CLASS_NAMES[j], CLASS_NAMES[k]],
            "class_ids": [int(i), int(j), int(k)],
            "n_junctions": n,
            "n_dropped_arc_ge_180": n_wide,
            "mean_angles_deg": [float(x) for x in A.mean(axis=0)] if n else None,
            "median_angles_deg": [float(x) for x in np.median(A, axis=0)] if n else None,
            "std_angles_deg": [float(x) for x in A.std(axis=0)] if n else None,
            "herring_120_deviation_deg": ([float(x - 120.0) for x in A.mean(axis=0)]
                                          if n else None),
        }
        per_triple[f"{CLASS_NAMES[i]}-{CLASS_NAMES[j]}-{CLASS_NAMES[k]}"] = entry
        if n < min_junctions_per_triple:
            entry["used_in_fit"] = False
            continue
        entry["used_in_fit"] = True
        th = np.deg2rad(A.mean(axis=0))
        s = np.sin(th)
        if (s <= 1e-6).any():
            entry["used_in_fit"] = False
            continue
        w = float(np.sqrt(n))
        p_jk, p_ik, p_ij = (min(j, k), max(j, k)), (min(i, k), max(i, k)), (min(i, j), max(i, j))
        # x_jk - x_ik = ln sin th_i - ln sin th_j ; x_ik - x_ij = ln sin th_j - ln sin th_k
        eqs.append(({p_jk: 1.0, p_ik: -1.0}, float(np.log(s[0]) - np.log(s[1])), w))
        eqs.append(({p_ik: 1.0, p_ij: -1.0}, float(np.log(s[1]) - np.log(s[2])), w))
        pair_ids.update((p_jk, p_ik, p_ij))

    pairs = sorted(pair_ids)
    if not pairs:
        return {"per_triple": per_triple, "sigma": None,
                "note": "no triple met min_junctions_per_triple"}
    idx = {p: c for c, p in enumerate(pairs)}
    M = np.zeros((len(eqs) + 1, len(pairs)))
    rhs = np.zeros(len(eqs) + 1)
    for r, (coeffs, b, w) in enumerate(eqs):
        for p, c in coeffs.items():
            M[r, idx[p]] = c * w
        rhs[r] = b * w
    gauge_w = 10.0 * max(np.sqrt(len(eqs)), 1.0)   # soft gauge row: mean log sigma = 0
    M[-1, :] = gauge_w / len(pairs)
    rhs[-1] = 0.0
    x, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    # cross-triple consistency: unweighted per-equation log-space residuals (a pair constrained by
    # 2+ triples must agree; large residuals = the SAME pair demands different tensions at
    # different triples — model misfit the bootstrap CI does NOT cover)
    resid = [float(abs(sum(c * x[idx[p]] for p, c in coeffs.items()) - b))
             for coeffs, b, _w in eqs]

    sigma = np.full((5, 5), np.nan)
    np.fill_diagonal(sigma, 0.0)
    for p, c in idx.items():
        sigma[p[0], p[1]] = sigma[p[1], p[0]] = float(np.exp(x[c]))
    return {"per_triple": per_triple, "pairs": [list(p) for p in pairs],
            "log_sigma": {f"{CLASS_NAMES[p[0]]}-{CLASS_NAMES[p[1]]}": float(x[idx[p]])
                          for p in pairs},
            "sigma_matrix_5x5": sigma.tolist(),
            "n_equations": len(eqs),
            "log_space_eq_residuals_abs": resid,
            "max_abs_log_residual": max(resid) if resid else None}


def bootstrap_sigma_ci(triples: np.ndarray, angles: np.ndarray, *, n_boot: int = 200,
                       seed: int = 0, min_junctions_per_triple: int = 30) -> dict:
    """Percentile bootstrap over junctions (resampled within each triple)."""
    rng = np.random.default_rng(seed)
    keys = np.array([t[0] * 25 + t[1] * 5 + t[2] for t in triples])
    samples: dict[tuple[int, int], list[float]] = {}
    for _ in range(n_boot):
        parts_t, parts_a = [], []
        for u in np.unique(keys):
            sel = np.flatnonzero(keys == u)
            pick = rng.choice(sel, size=sel.size, replace=True)
            parts_t.append(triples[pick])
            parts_a.append(angles[pick])
        fit = fit_sigma_from_angles(np.concatenate(parts_t), np.concatenate(parts_a),
                                    min_junctions_per_triple=min_junctions_per_triple)
        if fit.get("sigma_matrix_5x5") is None:
            continue
        S = np.array(fit["sigma_matrix_5x5"])
        for p in fit["pairs"]:
            samples.setdefault((p[0], p[1]), []).append(S[p[0], p[1]])
    out = {}
    for p, vals in samples.items():
        vals = sorted(vals)
        lo = vals[max(int(0.025 * len(vals)) - 1, 0)]
        hi = vals[min(int(0.975 * len(vals)), len(vals) - 1)]
        out[f"{CLASS_NAMES[p[0]]}-{CLASS_NAMES[p[1]]}"] = {
            "ci95": [lo, hi], "n_boot_effective": len(vals),
            "excludes_1.0": bool(hi < 1.0 or lo > 1.0)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-cache", type=Path, default=GT_CACHE)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "experiments/results/solver_pack_20260707/junction_sigma")
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--n-circle-samples", type=int, default=120)
    ap.add_argument("--min-junctions-per-triple", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.gt_cache)
    lstars = z["lstars"].astype(np.uint8)   # (600,384,512); canonical class order — never re-derive
    n_pairs = int(lstars.shape[0])

    coords, quad_count = detect_triple_junctions(lstars)
    triples, angles, stats = measure_junction_angles(
        lstars, coords, radius=args.radius, n_samples=args.n_circle_samples)
    fit = fit_sigma_from_angles(triples, angles,
                                min_junctions_per_triple=args.min_junctions_per_triple)
    ci = bootstrap_sigma_ci(triples, angles, n_boot=args.n_boot, seed=args.seed,
                            min_junctions_per_triple=args.min_junctions_per_triple)

    all_angles = angles.ravel()
    result = {
        "tool": "tools/fit_junction_sigma_youngs_law.py",
        "axis": ADVISORY,
        "theory": ("Imbert-Monneau flux-limited junction condition = free parameter; Young's law "
                   "inverts measured scorer junction angles to sigma_ij — "
                   "viscosity_theory_alignment_hunt_20260707.md §7 (EUREKA candidate #2)"),
        "inputs": {"gt_cache": str(args.gt_cache), "n_pairs": n_pairs,
                   "class_order": CLASS_NAMES,
                   "radius_px": args.radius, "n_circle_samples": args.n_circle_samples,
                   "subpixel_refine": "SKIPPED (stated): integer plaquette corners + radius-r "
                                      "circle; +-0.5px center noise averaged over junctions and "
                                      "absorbed by the bootstrap CI",
                   "seed": args.seed, "n_boot": args.n_boot,
                   "min_junctions_per_triple": args.min_junctions_per_triple},
        "detection": {**stats, "n_quad_plaquettes_excluded": quad_count},
        "angle_distribution_overall": {
            "mean_deg": float(all_angles.mean()) if all_angles.size else None,
            "std_deg": float(all_angles.std()) if all_angles.size else None,
            "herring_null": 120.0,
            "abs_dev_from_120_mean_deg": (float(np.abs(all_angles - 120.0).mean())
                                          if all_angles.size else None),
        },
        "fit": fit,
        "sigma_ci95_bootstrap": ci,
        "null_hypothesis": ("all-ones sigma (today's scalar --length-weight) == Herring "
                            "120-120-120; gauge geometric-mean(sigma)=1 makes 1.0 the null value "
                            "per pair"),
        "consumption_path": {
            "status": "TrainerSupportGap (flag does NOT exist; do not invent flags)",
            "proposed_flag": "--length-sigma-matrix <path.json|15 comma floats upper-tri>",
            "dsl_holder": ("tac.witness_dsl.curriculum_dsl Regularizer('--length-weight', ...) "
                           "factory extended with a sigma_matrix argument (default all-ones = "
                           "byte-identical OFF) — a Lever factory, never a hand flag"),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "junction_sigma_fit.json"
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=1)
    os.replace(tmp, out_path)
    print(f"wrote {out_path}")
    print(f"junctions: {stats['n_clean_junctions']} clean / {stats['n_candidates']} candidates "
          f"({quad_count} quad plaquettes excluded)")
    ad = result["angle_distribution_overall"]
    print(f"angles: mean {ad['mean_deg']:.1f} deg, |dev from 120| mean {ad['abs_dev_from_120_mean_deg']:.1f} deg")
    if fit.get("sigma_matrix_5x5") is not None:
        for pair, ls in fit["log_sigma"].items():
            c = ci.get(pair, {})
            print(f"  sigma[{pair}] = {np.exp(ls):.3f}  ci95={c.get('ci95')} "
                  f"excludes_all_ones={c.get('excludes_1.0')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
