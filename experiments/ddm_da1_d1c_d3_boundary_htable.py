#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_da1 D1c + D3 -- boundary-curve temporal coherence + H-table MASS decomposition.

D1c (worldsheet): how many px does the flip-support boundary move pair-to-pair? For each consecutive
  pair, EDT distance from each flip pixel in t to nearest flip pixel in t-1 (median/p90). Small median
  => a persistent worldsheet the support geometry could ride (temporal prior).

D3 (H-table 0.325 b/flip MASS): reuse fc1's exact label context (copy_argmax x bdist_bucket x adj_class).
  (a) TOP-10 cells by total label bytes (cell -> flips, b/flip, total B) -- WHERE the 41,392 B live.
  (b) b/flip vs COPY-MARGIN decile -- structure of the label cost across margin.
  (c) DETERMINISM threshold: at margin deciles, what fraction of flips have H(label|cell) ~ 0
      (label derivable => support could be restricted to low-margin sites, rest derived).

Reuses cached copy_argmax + copy_margin + lstars. No SegNet re-run. `[macOS-CPU advisory]`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SCHEMA = "ddm_da1_d1c_d3.v1"
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
MARGIN_EDGES = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, np.inf], dtype=np.float64)
BDIST_EDGES = np.array([0.0, 1.0, 1.5, 2.5, 3.5, 5.5, 10.5, np.inf], dtype=np.float64)
N_MARGIN = len(MARGIN_EDGES) - 1
N_BDIST = len(BDIST_EDGES) - 1
N_CLASS = 5


def _bucket(x, edges):
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2).astype(np.int64)


def _boundary_distance(a):
    from scipy.ndimage import distance_transform_edt
    boundary = np.zeros(a.shape, dtype=bool)
    boundary[:, :-1] |= a[:, :-1] != a[:, 1:]
    boundary[:, 1:] |= a[:, :-1] != a[:, 1:]
    boundary[:-1, :] |= a[:-1, :] != a[1:, :]
    boundary[1:, :] |= a[:-1, :] != a[1:, :]
    return distance_transform_edt(~boundary).astype(np.float32)


def _adjacent_class(a):
    counts = np.zeros((N_CLASS,) + a.shape, dtype=np.int16)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(np.roll(a, dr, axis=0), dc, axis=1)
            for k in range(N_CLASS):
                counts[k] += (shifted == k)
    for k in range(N_CLASS):
        counts[k][a == k] = -1
    return counts.argmax(axis=0).astype(np.int64)


def main(argv=None) -> int:
    from scipy.ndimage import distance_transform_edt
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx-dir", type=Path, required=True)
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-pairs", type=int, default=600)
    args = ap.parse_args(argv)
    t0 = time.time()

    gt = np.load(str(args.gt_cache))
    lstars = gt["lstars"]
    chunks = sorted(args.ctx_dir.glob("ctx_*.npz"))

    # accumulators: label cell = (copy_argmax, bdist_bucket, adj_class) -> lstars histogram
    lab_counts = np.zeros((N_CLASS * N_BDIST * N_CLASS, N_CLASS), dtype=np.int64)
    # per-flip: (label self-info via lab model, margin) -> need 2nd pass; accumulate margin-bucket x
    # for D3(b/c) we accumulate, per margin bucket, the label histogram to get H(label|marginbucket)
    margin_lab = np.zeros((N_MARGIN, N_CLASS), dtype=np.int64)  # flips only
    # fine copy-margin deciles need the raw margins at flips; collect a subsample of (margin,label)
    # but to get b/flip via decile we accumulate per-fine-margin-bucket label hist. Use 40 fine bins.
    FINE_EDGES = np.concatenate([np.linspace(0, 2.0, 41), [np.inf]])
    n_fine = len(FINE_EDGES) - 1
    fine_lab = np.zeros((n_fine, N_CLASS), dtype=np.int64)
    fine_tot = np.zeros(n_fine, dtype=np.int64)

    # D1c boundary motion accumulators (per consecutive pair)
    med_list, p90_list = [], []
    prev_flip = None
    prev_idx = None

    total_flips = 0
    total_sites = 0
    P_done = 0
    for ch in chunks:
        d = np.load(str(ch))
        c_arg = d["copy_argmax"]
        c_mar = d["copy_margin"].astype(np.float32)
        s0, s1 = int(d["start"]), int(d["end"])
        s1 = min(s1, args.max_pairs)
        if s0 >= args.max_pairs:
            continue
        ls = lstars[s0:s1].astype(np.int64)
        m = s1 - s0
        for i in range(m):
            gidx = s0 + i
            a = c_arg[i].astype(np.int64)
            mar = c_mar[i]
            l = ls[i]
            flip = a != l
            total_sites += a.size
            nf = int(flip.sum())
            total_flips += nf
            # D1c: boundary motion vs previous pair's flip mask
            if prev_flip is not None and prev_idx == gidx - 1 and nf > 0 and prev_flip.any():
                dist_to_prev = distance_transform_edt(~prev_flip)
                dvals = dist_to_prev[flip]
                med_list.append(float(np.median(dvals)))
                p90_list.append(float(np.percentile(dvals, 90)))
            prev_flip = flip
            prev_idx = gidx
            if nf == 0:
                continue
            bdist = _boundary_distance(a)
            adj = _adjacent_class(a)
            db = _bucket(bdist, BDIST_EDGES)
            mb = _bucket(mar, MARGIN_EDGES)
            fa = a[flip]; fdb = db[flip]; fadj = adj[flip]; fl = l[flip]
            lab_cell = (fa * N_BDIST + fdb) * N_CLASS + fadj
            np.add.at(lab_counts, (lab_cell, fl), 1)
            np.add.at(margin_lab, (mb[flip], fl), 1)
            fb = _bucket(mar[flip], FINE_EDGES)
            np.add.at(fine_lab, (fb, fl), 1)
            np.add.at(fine_tot, fb, 1)
        P_done += m
        print(f"[d1c/d3] pairs {s0}-{s1} done ({time.time()-t0:.0f}s) flips={total_flips}", flush=True)

    # ---- D3(a) top-10 cells by total label bytes ----
    cell_tot = lab_counts.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(cell_tot[:, None] > 0, lab_counts / cell_tot[:, None], 0.0)
        h_cell = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)  # bits/flip in cell
    cell_bytes = cell_tot * h_cell / 8.0
    order = np.argsort(cell_bytes)[::-1]
    top10 = []
    for ci in order[:10]:
        if cell_tot[ci] == 0:
            continue
        a_k = ci // (N_BDIST * N_CLASS)
        rem = ci % (N_BDIST * N_CLASS)
        dbk = rem // N_CLASS
        adjk = rem % N_CLASS
        top10.append({
            "cell": {"copy_class": CLASS_NAMES[int(a_k)], "bdist_bucket": int(dbk), "adj_class": CLASS_NAMES[int(adjk)]},
            "flips": int(cell_tot[ci]),
            "bits_per_flip": float(h_cell[ci]),
            "total_bytes": float(cell_bytes[ci]),
            "dominant_label": CLASS_NAMES[int(lab_counts[ci].argmax())],
            "dominant_label_frac": float(lab_counts[ci].max() / cell_tot[ci]),
        })
    total_label_bytes = float(cell_bytes.sum())
    top10_bytes = sum(c["total_bytes"] for c in top10)

    # ---- D3(b) b/flip vs copy-margin decile ----
    margin_decile = []
    for mb in range(N_MARGIN):
        tot = margin_lab[mb].sum()
        if tot == 0:
            continue
        pp = margin_lab[mb] / tot
        hh = -np.where(pp > 0, pp * np.log2(pp), 0.0).sum()
        margin_decile.append({
            "margin_lo": float(MARGIN_EDGES[mb]),
            "margin_hi": float(MARGIN_EDGES[mb + 1]),
            "flips": int(tot),
            "bits_per_flip_H_label_given_marginbucket": float(hh),
            "share_of_flips": float(tot / max(1, total_flips)),
        })

    # ---- D3(c) determinism: fraction of flips whose fine-margin-cell H(label) < eps ----
    with np.errstate(divide="ignore", invalid="ignore"):
        pf = np.where(fine_tot[:, None] > 0, fine_lab / fine_tot[:, None], 0.0)
        hf = -np.where(pf > 0, pf * np.log2(pf), 0.0).sum(axis=1)
    det_curve = []
    for eps in (0.05, 0.1, 0.25, 0.5):
        # flips living in fine-margin cells with H<eps (label ~ deterministic given margin)
        mask = hf < eps
        det_flips = int(fine_tot[mask].sum())
        det_curve.append({
            "H_eps_bits": eps,
            "flips_in_near_deterministic_marginbins": det_flips,
            "share": float(det_flips / max(1, total_flips)),
        })
    # margin threshold where H(label|finemargin) crosses below 0.1 bit
    thr_idx = None
    for j in range(n_fine):
        if fine_tot[j] > 100 and hf[j] < 0.1:
            thr_idx = j
            break

    d1c = {
        "consecutive_pairs_measured": len(med_list),
        "boundary_move_px_median_of_pair_medians": float(np.median(med_list)) if med_list else None,
        "boundary_move_px_median_of_pair_p90": float(np.median(p90_list)) if p90_list else None,
        "boundary_move_px_mean_median": float(np.mean(med_list)) if med_list else None,
        "interpretation": "median px a flip site sits from the nearest prior-pair flip site; small => worldsheet persistence",
    }

    result = {
        "schema": SCHEMA,
        "evidence_axis": "[macOS-CPU advisory] exact plug-in label entropy over decoder-derivable cells + EDT boundary motion; NOT a byte-closed evaluate.py row",
        "pairs": P_done,
        "total_flips": total_flips,
        "D1c_boundary_motion": d1c,
        "D3a_total_label_bytes_from_cells": total_label_bytes,
        "D3a_top10_cells": top10,
        "D3a_top10_share_of_label_bytes": top10_bytes / max(1e-9, total_label_bytes),
        "D3b_margin_decile": margin_decile,
        "D3c_determinism": {
            "near_deterministic_curve": det_curve,
            "margin_threshold_H_below_0p1_bit": (float(FINE_EDGES[thr_idx]) if thr_idx is not None else None),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"[d1c/d3] done ({time.time()-t0:.0f}s) -> {args.out}", flush=True)
    print(json.dumps(result["D1c_boundary_motion"], indent=2), flush=True)
    print(json.dumps(result["D3b_margin_decile"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
