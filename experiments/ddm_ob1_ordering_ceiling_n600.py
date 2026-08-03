#!/usr/bin/env python
"""ddm_ob1 Job B -- the ORDERING ceiling, split into free vs oracle, at n600.

Scorer-free ($0). ZERO SegNet/PoseNet forwards.
Axis: [macOS-CPU cache-derived advisory] NON-PROMOTABLE. score_claim=False.
pointer 0.1910828242 [contest-CPU] UNMOVED.

THE FRAME
---------
gp1 priced the address as a HARD-THRESHOLD BAND: pick a radius r, spend
`log2 C(band_px, flips_in_band)` bits to say which band pixels flip, and pay nothing for
pixels outside.  That is a crude use of the label field.  An optimal coder does not threshold
-- it assigns every pixel a probability and arithmetic-codes the flip indicator:

    address_bits = SUM_i  -[ f_i log2 p_i + (1-f_i) log2 (1-p_i) ]

Given a feature X computable by the receiver, the best achievable is the empirical conditional
entropy N * H(f | X).  So the address question is not "which r" but "how much does the receiver
KNOW about where the flips are".  Hard-threshold banding is the special case X = 1[d <= r].

WHAT THIS MEASURES ($0, from the label fields already on disk)
-------------------------------------------------------------
  X_free  = features derived from the DECODER'S OWN L* alone:
              d      = distance to L*'s label boundary (bucketed 0..15, 16+)
              own    = the decoder's own class at the pixel
              nbr    = the nearest differing class (the EDGE identity -- pc2's hub law says
                       decompose per EDGE, never per class)
  H(f | X_free) * N  is the address cost of an OPTIMAL free coder, i.e. the ceiling of
  ordering-within-the-band that needs NO student and NO scorer weights.

  PAYLOAD is priced in the same frame: H(gt | own) and H(gt | own, nbr).

  The model table itself is COUNTED (a per-video probability table is video-derived payload
  under rule 118, not generic code), and a held-out split by pair is reported so the
  empirical conditional entropy is not quoted as if fitting were free.

The remaining gap -- between the best FREE feature and the frozen scorer's MARGIN -- is
exactly gp1's R5 student rung.  It needs the scorer and is NOT measured here; this unit
brackets it from below.

DENOMINATORS (m66)
------------------
live best S = 0.7910689 @ 353,805 B (pu2) [macOS-CPU advisory]; gap to PR130 bar = 0.6189279
1% of gap = 0.0061893 S = 9,295 B
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

ARGMAX_DIR = "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"
OUT_DIR = "/Volumes/VertigoDataTier/pact/ddm_ob1_20260803"

SEG_H, SEG_W = 384, 512
PIX_PER_PAIR = SEG_H * SEG_W
N_PAIRS = 600
RATE_PER_BYTE = 25.0 / 37_545_489.0
LIVE_BEST_S = 0.7910689
GAP = 0.6189279
S_PER_FLIP = 100.0 / (N_PAIRS * PIX_PER_PAIR)
NC = 5
DMAX = 15          # distance buckets 0..15, with 16 = "further than 15"
NB_D = DMAX + 2    # 17 buckets
ETA_POSE_NEUTRAL_N32 = 0.5406
ETA_UNCONSTRAINED_N32 = 0.7895


def boundary(lab: np.ndarray) -> np.ndarray:
    b = np.zeros(lab.shape, dtype=bool)
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :] |= lab[:-1, :] != lab[1:, :]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    return b


def dilate1(m: np.ndarray) -> np.ndarray:
    o = m.copy()
    o[:-1, :] |= m[1:, :]
    o[1:, :] |= m[:-1, :]
    o[:, :-1] |= m[:, 1:]
    o[:, 1:] |= m[:, :-1]
    return o


def dist_to_boundary(lab: np.ndarray) -> np.ndarray:
    """L1 distance to the label boundary, bucketed; matches gp1's dilation geometry exactly."""
    d = np.full(lab.shape, DMAX + 1, dtype=np.uint8)
    m = boundary(lab)
    d[m] = 0
    for r in range(1, DMAX + 1):
        nm = dilate1(m)
        new = nm & ~m
        d[new] = r
        m = nm
        if m.all():
            break
    return d


def nearest_diff_label(lab: np.ndarray) -> np.ndarray:
    """For each pixel, the smallest differing class among its 4-neighbours; NC if none.

    This is the EDGE identity at the boundary. Away from the boundary it is NC ("interior"),
    which is the correct behaviour: interiors carry no edge information.
    """
    out = np.full(lab.shape, NC, dtype=np.uint8)
    for c in range(NC - 1, -1, -1):
        hit = np.zeros(lab.shape, dtype=bool)
        eq = lab == c
        hit[:-1, :] |= eq[1:, :]
        hit[1:, :] |= eq[:-1, :]
        hit[:, :-1] |= eq[:, 1:]
        hit[:, 1:] |= eq[:, :-1]
        out[hit & (lab != c)] = c
    return out


def cond_bits(counts_n: np.ndarray, counts_k: np.ndarray) -> float:
    """SUM over cells of n*H(k/n): the ideal arithmetic-coded cost of the flip indicator."""
    n = counts_n.astype(np.float64)
    k = counts_k.astype(np.float64)
    out = 0.0
    ok = (n > 0) & (k > 0) & (k < n)
    p = np.zeros_like(n)
    p[ok] = k[ok] / n[ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(ok, -(p * np.log2(np.where(p > 0, p, 1))
                           + (1 - p) * np.log2(np.where(p < 1, 1 - p, 1))), 0.0)
    out = float((n * h).sum())
    return out


def xent_bits(counts_n: np.ndarray, counts_k: np.ndarray,
              p_model: np.ndarray, eps: float = 1e-9) -> float:
    """Cross-entropy of a HELD-OUT split's counts under a model fit on the other split."""
    p = np.clip(p_model, eps, 1 - eps)
    n = counts_n.astype(np.float64)
    k = counts_k.astype(np.float64)
    return float(-(k * np.log2(p) + (n - k) * np.log2(1 - p)).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "ob1_ordering_ceiling_n600.json"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    n = args.n_pairs

    gt = np.load(os.path.join(ARGMAX_DIR, "gt_argmax_n600.npy"), mmap_mode="r")
    rd = np.load(os.path.join(ARGMAX_DIR, "cx1_argmax_n600.npy"), mmap_mode="r")

    # ---- flip-indicator contingency tables, split even/odd pairs for held-out honesty -------
    # cells: [split, d_bucket, own_class, nbr_class]
    cn = np.zeros((2, NB_D, NC, NC + 1), dtype=np.int64)
    ck = np.zeros((2, NB_D, NC, NC + 1), dtype=np.int64)
    # payload tables: [split, own_class, nbr_class, gt_class] over FLIPS only
    pay = np.zeros((2, NC, NC + 1, NC), dtype=np.int64)

    total_flips = 0
    t0 = time.time()
    for p in range(n):
        s = p & 1
        g = np.asarray(gt[p])
        lstar = np.asarray(rd[p])
        f = g != lstar
        total_flips += int(f.sum())
        d = dist_to_boundary(lstar)
        nb = nearest_diff_label(lstar)
        idx = (d.astype(np.int64) * NC + lstar.astype(np.int64)) * (NC + 1) + nb.astype(np.int64)
        flat_n = np.bincount(idx.ravel(), minlength=NB_D * NC * (NC + 1))
        flat_k = np.bincount(idx[f].ravel(), minlength=NB_D * NC * (NC + 1))
        cn[s] += flat_n.reshape(NB_D, NC, NC + 1)
        ck[s] += flat_k.reshape(NB_D, NC, NC + 1)
        pidx = ((lstar[f].astype(np.int64) * (NC + 1) + nb[f].astype(np.int64)) * NC
                + g[f].astype(np.int64))
        pay[s] += np.bincount(pidx, minlength=NC * (NC + 1) * NC).reshape(NC, NC + 1, NC)
        if (p + 1) % 100 == 0:
            print(f"  ob1-B {p+1}/{n}  {time.time()-t0:.1f}s", flush=True)

    N = n * PIX_PER_PAIR
    CN, CK = cn.sum(0), ck.sum(0)
    PAY = pay.sum(0)

    # ---- address cost under nested receiver-legal feature sets -----------------------------
    def marg(axes):
        return CN.sum(axis=axes), CK.sum(axis=axes)

    feats = {}
    nn, kk = CN.sum(), CK.sum()
    feats["none (uniform)"] = cond_bits(np.array([nn]), np.array([kk]))
    a, b = marg((1, 2))
    feats["d only"] = cond_bits(a, b)
    a, b = marg((0, 2))
    feats["own only"] = cond_bits(a, b)
    a, b = marg((2,))
    feats["d x own"] = cond_bits(a, b)
    a, b = marg((0,))
    feats["own x nbr (edge)"] = cond_bits(a, b)
    feats["d x own x nbr"] = cond_bits(CN, CK)

    # ---- held-out: fit on even pairs, price on odd pairs, and vice versa -------------------
    ho = 0.0
    for s in (0, 1):
        fit_n, fit_k = cn[1 - s], ck[1 - s]
        # Krichevsky-Trofimov (add-1/2) so unseen cells cannot cost infinity
        pm = (fit_k + 0.5) / (fit_n + 1.0)
        ho += xent_bits(cn[s], ck[s], pm)
    # model table: one fp16 probability per non-empty cell, brotli'd -- COUNTED as payload
    nonempty = int((CN > 0).sum())

    # ---- payload cost under the same feature sets -------------------------------------------
    def pay_bits(tab):
        t = tab.astype(np.float64)
        tot = t.sum(axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.where(t > 0, np.log2(np.where(tot > 0, t / np.where(tot > 0, tot, 1), 1)), 0.0)
        return float(-(t * lg).sum())

    pay_own = pay_bits(PAY.sum(axis=1))
    pay_own_nbr = pay_bits(PAY)

    rows = []
    for name, addr in feats.items():
        for pname, pb in (("H(gt|own)", pay_own), ("H(gt|own,nbr)", pay_own_nbr)):
            byts = (addr + pb) / 8.0
            gross = S_PER_FLIP * total_flips
            rows.append({
                "address_feature": name, "payload_feature": pname,
                "address_bits": addr, "address_bits_per_field_px": addr / N,
                "payload_bits": pb, "payload_bits_per_flip": pb / total_flips,
                "total_bytes": byts,
                "rate_cost_S": byts * RATE_PER_BYTE,
                "capture_rate": 1.0,
                "net_dS_eta1": byts * RATE_PER_BYTE - gross,
                "net_dS_eta_unconstrained_n32": byts * RATE_PER_BYTE
                - ETA_UNCONSTRAINED_N32 * gross,
                "net_dS_eta_pose_neutral_n32": byts * RATE_PER_BYTE
                - ETA_POSE_NEUTRAL_N32 * gross,
            })

    best = min(rows, key=lambda r: r["net_dS_eta_pose_neutral_n32"])
    gross = S_PER_FLIP * total_flips
    # eta needed to break even with the best free model
    eta_be = best["rate_cost_S"] / gross

    out = {
        "schema": "ddm_ob1_ordering_ceiling.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[macOS-CPU cache-derived advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "scorer_forwards": 0, "n_pairs": n,
        "baseline": {"live_best_S": LIVE_BEST_S, "gap_to_floor": GAP,
                     "rate_S_per_byte": RATE_PER_BYTE, "S_per_flip": S_PER_FLIP,
                     "one_pct_of_gap_bytes": (GAP / 100.0) / RATE_PER_BYTE},
        "totals": {"flips": total_flips, "field_pixels": N,
                   "d_seg_reproduced": total_flips / N,
                   "gross_dS_at_eta1_full_capture": gross},
        "held_out": {
            "note": "fit on even pairs price on odd and vice versa, Krichevsky-Trofimov 1/2 prior; "
                    "this is the honest address cost, in-sample is optimistic",
            "address_bits_heldout": ho,
            "address_bits_insample": feats["d x own x nbr"],
            "optimism_bits": feats["d x own x nbr"] - ho,
            "model_cells_nonempty": nonempty,
            "model_table_bytes_fp16_upper": nonempty * 2,
        },
        "eta_break_even_for_best_free_model": eta_be,
        "eta_measured_n32": {"unconstrained": ETA_UNCONSTRAINED_N32,
                             "pose_neutral_P7": ETA_POSE_NEUTRAL_N32,
                             "scope": "sq1 n=32 on sq1's ANISOTROPIC r=1 band; transfer to a "
                                      "soft full-field model is UNMEASURED"},
        "rows": rows,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\nwrote {args.out}")
    print(f"flips {total_flips}  d_seg {total_flips/N:.18f}")
    print(f"gross dS at eta=1, FULL capture: {gross:.5f}\n")
    print("  address feature        addr bits/px   payload feat     bytes     net@1     net@.5406")
    for r in rows:
        print(f"  {r['address_feature']:<20s}  {r['address_bits_per_field_px']:.5f}      "
              f"{r['payload_feature']:<14s} {r['total_bytes']:8.0f}  "
              f"{r['net_dS_eta1']:+.5f}  {r['net_dS_eta_pose_neutral_n32']:+.5f}")
    print(f"\nheld-out address bits {ho:,.0f} vs in-sample {feats['d x own x nbr']:,.0f} "
          f"(optimism {feats['d x own x nbr']-ho:,.0f} bits = "
          f"{(feats['d x own x nbr']-ho)/8:,.0f} B)")
    print(f"model table upper bound: {nonempty*2:,} B ({nonempty} cells)")
    print(f"eta needed to break even on the best FREE model: {eta_be:.4f} "
          f"(measured n32: {ETA_POSE_NEUTRAL_N32} pose-neutral / {ETA_UNCONSTRAINED_N32} uncon.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
