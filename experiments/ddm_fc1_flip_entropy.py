#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DDM-FC1 STAGE 1 -- THE TIER-MOVER SCALAR: H(flip | free decoder context), n600.

This is pantheon F6 (NULL/UNMEASURED): the conditional entropy of the argmax-flip correction against
the copy-base decoder-side context. It decides the correction stream's floor. Wyner-Ziv/DSC codes the
COSET H(X|Y), not H(X): X = the true relabeling field, Y = decoder-derivable context (copy argmax +
copy margin + boundary distance + nearest-adjacent class). If H is well below ~1 bit/flip the bar is
reachable; if it is ~>=1.2 bit/flip the correction stream alone busts it.

TWO honest floors are measured with EXACT plug-in conditional entropy over DECODER-DERIVABLE context
cells (the cell partition is decoder-computable, so a conditional arithmetic coder using these cell
frequencies achieves exactly this rate; the frequency TABLE is a tiny counted side stream, reported):

  (A) SUPPORT floor  H(flip? | copy_argmax, margin_bucket, bdist_bucket) x 117.9M sites -> WHERE bytes.
  (B) LABEL floor    H(lstars | flip, copy_argmax, bdist_bucket, adj_class) x 1.02M flips -> WHAT bytes.
      == the charter's "bits/flip" tier-mover, overall + per class.

Plus the CONCESSION CURVE (waterfill input): flip sites sorted by label self-information; cumulative
label bits vs fraction of flips fixed -> the fix-vs-concede water level (1.273 B/error).

Self-check: flip mask (copy_argmax != lstars) must reproduce oc1/r2s copy support EXACTLY. `[macOS-CPU
advisory]` -- entropy floors are REAL achievable conditional-coder rates (decoder-derivable cells), NOT
a byte-closed evaluate.py row.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SCHEMA = "ddm_fc1_flip_entropy.v1"
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
OC1_SUPPORT_FRACTION = 0.00864212883843316  # r2s/oc1 aggregate self-check
OC1_FLIP_SITES = 1_019_467
CONCESSION_WATER_B_PER_ERR = 1.273  # region_merge MDL water level (B/error)

# margin (top1-top2 logit) bucket edges -- fine at low margin where flips live
MARGIN_EDGES = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, np.inf], dtype=np.float64)
# distance-to-nearest-class-boundary bucket edges (pixels)
BDIST_EDGES = np.array([0.0, 1.0, 1.5, 2.5, 3.5, 5.5, 10.5, np.inf], dtype=np.float64)
N_MARGIN = len(MARGIN_EDGES) - 1
N_BDIST = len(BDIST_EDGES) - 1
N_CLASS = 5


def _bucket(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # returns index in [0, len(edges)-2]
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2).astype(np.int64)


def _boundary_distance(argmax_hw: np.ndarray) -> np.ndarray:
    """Distance (px) from each pixel to the nearest copy-argmax class boundary (decoder-derivable)."""
    from scipy.ndimage import distance_transform_edt

    a = argmax_hw
    # a pixel is on a boundary if any 4-neighbor differs
    boundary = np.zeros(a.shape, dtype=bool)
    boundary[:, :-1] |= a[:, :-1] != a[:, 1:]
    boundary[:, 1:] |= a[:, :-1] != a[:, 1:]
    boundary[:-1, :] |= a[:-1, :] != a[1:, :]
    boundary[1:, :] |= a[:-1, :] != a[1:, :]
    # distance to nearest boundary pixel: EDT of the complement (non-boundary)
    return distance_transform_edt(~boundary).astype(np.float32)


def _adjacent_class(argmax_hw: np.ndarray) -> np.ndarray:
    """Most-frequent 8-neighbor class != self (the class the flip most likely relabels TO)."""
    a = argmax_hw
    counts = np.zeros((N_CLASS,) + a.shape, dtype=np.int16)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(np.roll(a, dr, axis=0), dc, axis=1)
            for k in range(N_CLASS):
                counts[k] += (shifted == k)
    # forbid self-class
    for k in range(N_CLASS):
        counts[k][a == k] = -1
    return counts.argmax(axis=0).astype(np.int64)


def _cond_entropy_bits(counts: np.ndarray) -> tuple[float, int]:
    """Plug-in H(Y|X) in bits and total N, from counts[cell, y]. H = sum P(cell) H(Y|cell)."""
    total = int(counts.sum())
    if total == 0:
        return 0.0, 0
    cell_tot = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(cell_tot > 0, counts / cell_tot, 0.0)
        h_cell = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)  # bits per site in each cell
    weighted = (cell_tot.ravel() * h_cell).sum()
    return float(weighted / total), total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx-dir", type=Path, required=True, help="dir of ctx_*.npz from context cache")
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-pairs", type=int, default=600)
    args = ap.parse_args(argv)

    t0 = time.time()
    gt = np.load(str(args.gt_cache))
    lstars_all = gt["lstars"]  # (600,384,512) int64
    chunks = sorted(args.ctx_dir.glob("ctx_*.npz"))
    if not chunks:
        raise SystemExit(f"no ctx chunks in {args.ctx_dir}")

    H, W = lstars_all.shape[1], lstars_all.shape[2]
    px_per_frame = H * W

    # accumulators
    # (A) support: cell = (A, margin_bucket, bdist_bucket) -> [nonflip, flip]
    supp_counts = np.zeros((N_CLASS * N_MARGIN * N_BDIST, 2), dtype=np.int64)
    # (B) label: cell = (A, bdist_bucket, adj_class) -> lstars in {0..4}, at flip sites only
    lab_counts = np.zeros((N_CLASS * N_BDIST * N_CLASS, N_CLASS), dtype=np.int64)
    # held-out 2-fold: even-pair vs odd-pair label counts (defends headline vs overfit)
    lab_counts_even = np.zeros_like(lab_counts)
    lab_counts_odd = np.zeros_like(lab_counts)
    # per-class flip totals
    flip_per_class = np.zeros(N_CLASS, dtype=np.int64)
    total_sites = 0
    total_flips = 0

    # concession curve: per-flip label self-information -> we accumulate a histogram over fine bits bins
    # (store bits-per-flip histogram; also per-class)
    BIT_EDGES = np.linspace(0.0, 6.0, 121)  # 0.05-bit resolution up to 6 bits
    selfinfo_hist = np.zeros(len(BIT_EDGES) - 1, dtype=np.int64)
    selfinfo_hist_byclass = np.zeros((N_CLASS, len(BIT_EDGES) - 1), dtype=np.int64)

    n_done = 0
    for ch in chunks:
        d = np.load(str(ch))
        c_arg = d["copy_argmax"]  # (m,384,512) uint8
        c_mar = d["copy_margin"].astype(np.float32)  # (m,384,512)
        s0, s1 = int(d["start"]), int(d["end"])
        s1 = min(s1, args.max_pairs)
        if s0 >= args.max_pairs:
            continue
        m = s1 - s0
        ls = lstars_all[s0:s1]

        for i in range(m):
            a = c_arg[i].astype(np.int64)
            mar = c_mar[i]
            l = ls[i]
            flip = a != l
            bdist = _boundary_distance(a)
            adj = _adjacent_class(a)
            mb = _bucket(mar, MARGIN_EDGES)
            db = _bucket(bdist, BDIST_EDGES)

            # (A) support counts (dense)
            supp_cell = (a * N_MARGIN + mb) * N_BDIST + db
            flat_cell = supp_cell.ravel()
            flat_flip = flip.ravel().astype(np.int64)
            np.add.at(supp_counts, (flat_cell, flat_flip), 1)

            # (B) label counts (flip sites only)
            fa = a[flip]
            fdb = db[flip]
            fadj = adj[flip]
            fl = l[flip]
            lab_cell = (fa * N_BDIST + fdb) * N_CLASS + fadj
            np.add.at(lab_counts, (lab_cell, fl), 1)
            if (s0 + i) % 2 == 0:
                np.add.at(lab_counts_even, (lab_cell, fl), 1)
            else:
                np.add.at(lab_counts_odd, (lab_cell, fl), 1)

            for k in range(N_CLASS):
                flip_per_class[k] += int((flip & (l == k)).sum())
            total_sites += px_per_frame
            total_flips += int(flip.sum())
        n_done += m
        print(f"[entropy] pairs {s0}-{s1} done ({time.time()-t0:.0f}s) flips_so_far={total_flips}", flush=True)

    # ---- entropies ----
    h_support, n_supp = _cond_entropy_bits(supp_counts)  # bits/pixel
    h_label, n_lab = _cond_entropy_bits(lab_counts)      # bits/flip

    # held-out 2-fold cross-entropy: train on one parity, pay -log2 p_train on the other (add-0.5 KT
    # smoothing so unseen (cell,label) never costs infinity). Average both directions. If ~= h_label
    # the plug-in floor generalizes (not an overfit artifact).
    def _xent(train: np.ndarray, test: np.ndarray, alpha: float = 0.5) -> float:
        tt = train + alpha
        p = tt / tt.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore"):
            nll = -(test * np.log2(p)).sum()
        n = test.sum()
        return float(nll / n) if n else 0.0
    h_label_heldout = 0.5 * (_xent(lab_counts_even, lab_counts_odd) + _xent(lab_counts_odd, lab_counts_even))

    support_geom_floor_bytes = h_support * total_sites / 8.0
    label_floor_bytes = h_label * total_flips / 8.0

    # per-class label entropy: restrict lab_counts rows whose observed... recompute per-class via a
    # second pass masking by lstars is not directly available; instead measure per-class by the
    # dominant lstars in each cell weighted -- simpler: recompute per-class conditional entropy
    # H(lstars | context, lstars==k) is degenerate; the useful per-class number is the mean
    # self-information of flips whose TRUE label is k. Build from selfinfo below.

    # label self-information per flip (for concession curve + per-class), second pass using the
    # fitted lab model (decoder-derivable cells; the model table is counted separately).
    cell_tot = lab_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        lab_p = np.where(cell_tot > 0, lab_counts / cell_tot, 0.0)
    # model table size: nonzero cells * (5 counts) -- store as varint-ish; report cells and a byte est.
    nonzero_cells = int((cell_tot.ravel() > 0).sum())

    n_done2 = 0
    for ch in chunks:
        d = np.load(str(ch))
        c_arg = d["copy_argmax"]
        s0, s1 = int(d["start"]), int(d["end"])
        s1 = min(s1, args.max_pairs)
        if s0 >= args.max_pairs:
            continue
        m = s1 - s0
        ls = lstars_all[s0:s1]
        for i in range(m):
            a = c_arg[i].astype(np.int64)
            l = ls[i]
            flip = a != l
            if not flip.any():
                continue
            bdist = _boundary_distance(a)
            adj = _adjacent_class(a)
            db = _bucket(bdist, BDIST_EDGES)
            fa = a[flip]; fdb = db[flip]; fadj = adj[flip]; fl = l[flip]
            lab_cell = (fa * N_BDIST + fdb) * N_CLASS + fadj
            p_true = lab_p[lab_cell, fl]
            si = np.where(p_true > 0, -np.log2(p_true), 12.0)  # clip impossible to 12 bits
            selfinfo_hist += np.histogram(si, bins=BIT_EDGES)[0]
            for k in range(N_CLASS):
                mk = fl == k
                if mk.any():
                    selfinfo_hist_byclass[k] += np.histogram(si[mk], bins=BIT_EDGES)[0]
        n_done2 += m

    # concession curve: cumulative label bits vs fraction of flips fixed (cheapest first)
    bit_centers = 0.5 * (BIT_EDGES[:-1] + BIT_EDGES[1:])
    order_bits = bit_centers  # already ascending
    cum_flips = np.cumsum(selfinfo_hist)
    cum_bits = np.cumsum(selfinfo_hist * order_bits)
    frac_fixed = cum_flips / max(1, total_flips)
    curve = []
    for target in (0.5, 0.8, 0.9, 0.95, 0.982, 0.99, 1.0):
        idx = int(np.searchsorted(frac_fixed, target))
        idx = min(idx, len(cum_bits) - 1)
        curve.append({
            "fraction_fixed": target,
            "cum_label_bits": float(cum_bits[idx]),
            "cum_label_bytes": float(cum_bits[idx] / 8.0),
            "marginal_bits_per_flip_at_this_frac": float(order_bits[idx]),
        })

    per_class_mean_si = {}
    for k in range(N_CLASS):
        cnt = selfinfo_hist_byclass[k]
        tot = cnt.sum()
        mean_si = float((cnt * bit_centers).sum() / tot) if tot else 0.0
        per_class_mean_si[CLASS_NAMES[k]] = {
            "flips": int(flip_per_class[k]),
            "mean_bits_per_flip": mean_si,
            "share_of_support": float(flip_per_class[k] / max(1, total_flips)),
        }

    support_fraction = total_flips / max(1, total_sites)
    result = {
        "schema": SCHEMA,
        "evidence_axis": "[macOS-CPU advisory] real plug-in conditional entropy over decoder-derivable cells; NOT a byte-closed evaluate.py row",
        "pairs": n_done,
        "total_sites": total_sites,
        "total_flips": total_flips,
        "support_fraction": support_fraction,
        "self_check_vs_oc1": {
            "expected_fraction": OC1_SUPPORT_FRACTION,
            "expected_sites_n600": OC1_FLIP_SITES,
            "match_fraction": abs(support_fraction - OC1_SUPPORT_FRACTION) < 1e-5 if n_done == 600 else None,
        },
        "STAGE1_HEADLINE": {
            "H_label_bits_per_flip": h_label,
            "H_label_bits_per_flip_HELDOUT_2fold": h_label_heldout,
            "H_support_bits_per_pixel": h_support,
            "label_floor_bytes": label_floor_bytes,
            "support_geom_floor_bytes": support_geom_floor_bytes,
            "label_model_table_cells_nonzero": nonzero_cells,
            "label_model_table_bytes_est": nonzero_cells * 5,  # ~1 byte/count varint upper-ish
            "total_correction_floor_bytes": support_geom_floor_bytes + label_floor_bytes + nonzero_cells * 5,
            "bar_bytes_0p172": 187_727,
            "bar_bytes_0p15": 154_522,
        },
        "per_class_label": per_class_mean_si,
        "concession_curve": curve,
        "concession_water_B_per_error": CONCESSION_WATER_B_PER_ERR,
        "context_features": {
            "support": "copy_argmax x margin_bucket x bdist_bucket",
            "label": "copy_argmax x bdist_bucket x adjacent_class",
            "margin_edges": MARGIN_EDGES.tolist(),
            "bdist_edges": BDIST_EDGES.tolist(),
            "note_g4_stationarity": "NOT included (copy base has no decoder temporal signal); adding it can only lower H further",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["STAGE1_HEADLINE"], indent=2), flush=True)
    print(json.dumps(result["per_class_label"], indent=2), flush=True)
    print(json.dumps(result["concession_curve"], indent=2), flush=True)
    print(f"[done] entropy stage in {time.time()-t0:.0f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
