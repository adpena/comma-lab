#!/usr/bin/env python
"""ddm_et1 Job 0 -- RECALIBRATE the free band that every seg-address rung is priced on.

Two arms measured eta/price on bands that are NOT the same object, and neither memo says so:

  * gp1  (`ddm_gp1_free_band_and_net.py:77`)  dilate = ONE 4-neighbour OR per r
         -> structuring element is the L1/von-Neumann PLUS (5 px), NOT Chebyshev
         seed = FIELD_DIR `argmax`, the decoder's own distilled label field (receiver-held)
  * sq1  (`ddm_sq1_eta_seg_realization.py:149`) dilate = 4-neighbour OR THEN a vertical-only OR
         -> structuring element is PLUS (+) VERTICAL-3 (11 px) = 2.2x gp1's SE area
         seed = `sc.seg_argmax(dec)`, the FROZEN SEGNET's argmax of the decoded frames

Both docstrings say "Chebyshev"; neither implements it.  So this script measures all THREE
structuring elements against BOTH seeds, at n600, and re-prices under gp1's own byte model so
the comparison is apples-to-apples.

The two questions it answers, which no existing receipt does:
  Q1  How much of gp1's priced A3 address does sq1's eta actually stand on? (area ratio)
  Q2  Is the band sq1 measured on receiver-legal at all? (seed provenance)

No scorer, no GPU: pure numpy over cached argmax fields.  n600, no subsetting -> the m96
prefix-bias class cannot apply here by construction.

Axis: [macOS-CPU cache-derived advisory] NON-PROMOTABLE.  score_claim=false.
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import os
import time

import numpy as np

SEG_H, SEG_W = 384, 512
PIX_PER_PAIR = SEG_H * SEG_W
N_PAIRS = 600
RATE_PER_BYTE = 25.0 / 37_545_489.0
# CURRENT denominators (charter): sq1 priced against a STALE live-best 0.826496209256714.
LIVE_BEST_S = 0.7910689
PR130_FLOOR_S = 0.172141
GAP_S = LIVE_BEST_S - PR130_FLOOR_S            # 0.6189279
S_PER_FLIP = 100.0 / (N_PAIRS * SEG_H * SEG_W)
H_GT_GIVEN_REND = 1.1011521270613085           # gp1's measured payload entropy, bits/flip

ARGMAX_DIR = "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"
FIELD_DIR = "/Volumes/VertigoDataTier/pact/ddm_b2b_qa75_field_20260730"
OUT_DIR = "/Volumes/VertigoDataTier/pact/ddm_et1_20260803"

CLASS_NAMES = ["Road", "Lane", "Undrivable", "Movable", "MyCar"]


def log2_binom(n: int, k: int) -> float:
    if k <= 0 or k >= n:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2.0)


def lzma1_raw(b: bytes) -> int:
    filt = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}]
    return len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=filt))


def boundary(lab: np.ndarray) -> np.ndarray:
    """4-neighbour label boundary -- IDENTICAL in gp1 and sq1 (verified by inspection)."""
    b = np.zeros(lab.shape, dtype=bool)
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :] |= lab[:-1, :] != lab[1:, :]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    return b


def _or4(m: np.ndarray) -> np.ndarray:
    o = m.copy()
    o[:-1, :] |= m[1:, :]
    o[1:, :] |= m[:-1, :]
    o[:, :-1] |= m[:, 1:]
    o[:, 1:] |= m[:, :-1]
    return o


def _or_vert(m: np.ndarray) -> np.ndarray:
    o = m.copy()
    o[:-1, :] |= m[1:, :]
    o[1:, :] |= m[:-1, :]
    return o


def _or8(m: np.ndarray) -> np.ndarray:
    o = _or4(m)
    o[:-1, :-1] |= m[1:, 1:]
    o[:-1, 1:] |= m[1:, :-1]
    o[1:, :-1] |= m[:-1, 1:]
    o[1:, 1:] |= m[:-1, :-1]
    return o


def dilate_gp1(mask: np.ndarray, r: int) -> np.ndarray:
    """gp1's ACTUAL convention: L1/von-Neumann ball of radius r (SE area 2r^2+2r+1)."""
    m = mask
    for _ in range(r):
        m = _or4(m)
    return m


def dilate_sq1(mask: np.ndarray, r: int) -> np.ndarray:
    """sq1's ACTUAL convention: (L1 dilate by 1) then (vertical dilate by 1), r times."""
    m = mask
    for _ in range(r):
        m = _or_vert(_or4(m))
    return m


def dilate_cheb(mask: np.ndarray, r: int) -> np.ndarray:
    """TRUE Chebyshev/L-infinity: the (2r+1)^2 square both docstrings claim."""
    m = mask
    for _ in range(r):
        m = _or8(m)
    return m


SES = {"gp1_L1": dilate_gp1, "sq1_L1plusVert": dilate_sq1, "true_chebyshev": dilate_cheb}


def se_footprint(fn, r: int) -> int:
    """Area of the structuring element: dilate a lone centre pixel in a big empty field."""
    k = 4 * r + 9
    m = np.zeros((k, k), dtype=bool)
    m[k // 2, k // 2] = True
    return int(fn(m, r).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--radii", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "et1_band_recalibration.json"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    n = args.n_pairs

    gt = np.load(os.path.join(ARGMAX_DIR, "gt_argmax_n600.npy"), mmap_mode="r")
    rd = np.load(os.path.join(ARGMAX_DIR, "cx1_argmax_n600.npy"), mmap_mode="r")

    # SE footprints -- closed-form check of the 2.2x claim, independent of any data
    foot = {name: {r: se_footprint(fn, r) for r in args.radii} for name, fn in SES.items()}

    combos = [(seed, se, r) for seed in ("distill_field", "segnet_argmax")
              for se in SES for r in args.radii]
    acc = {c: {"px": 0, "hit": 0, "bits": 0} for c in combos}
    # uncaptured-flip decomposition, gp1's own convention only (the priced one)
    edge_tot = np.zeros((5, 5), dtype=np.int64)
    edge_unc = np.zeros((5, 5), dtype=np.int64)
    total_flips = 0
    seed_agree = 0
    seed_px = 0

    t0 = time.time()
    for p in range(n):
        g = np.asarray(gt[p])
        r_ = np.asarray(rd[p])
        diff = g != r_
        total_flips += int(diff.sum())

        with np.load(os.path.join(FIELD_DIR, f"pair-{p:06d}.npz")) as z:
            lab_distill = np.asarray(z["argmax"])
        seeds = {"distill_field": lab_distill, "segnet_argmax": r_}
        seed_agree += int((lab_distill == r_).sum())
        seed_px += lab_distill.size

        bnd = {k: boundary(v) for k, v in seeds.items()}
        for (seed, se, rr) in combos:
            band = SES[se](bnd[seed], rr)
            a = acc[(seed, se, rr)]
            a["px"] += int(band.sum())
            hit = diff & band
            a["hit"] += int(hit.sum())
            a["bits"] += lzma1_raw(np.packbits(diff[band].astype(np.uint8)).tobytes()) * 8

        # per-EDGE (pc2 hub law: never per class alone) on gp1's priced band, r=1
        band_priced = dilate_gp1(bnd["distill_field"], 1)
        unc = diff & ~band_priced
        np.add.at(edge_tot, (g[diff], r_[diff]), 1)
        np.add.at(edge_unc, (g[unc], r_[unc]), 1)

        if (p + 1) % 100 == 0:
            print(f"  et1 {p+1}/{n}  {time.time()-t0:.1f}s", flush=True)

    total_px = n * PIX_PER_PAIR
    rows = []
    for (seed, se, rr) in combos:
        a = acc[(seed, se, rr)]
        nb, hit, real_bits = a["px"], a["hit"], a["bits"]
        bound = log2_binom(nb, hit)
        ratio = (real_bits / bound) if bound else float("nan")
        addr_bits = bound * ratio                      # == real_bits at n600 (subset==population)
        pay_bits = hit * H_GT_GIVEN_REND
        by = (addr_bits + pay_bits) / 8.0
        rows.append({
            "seed": seed, "structuring_element": se, "dilate_r": rr,
            "se_footprint_px": foot[se][rr],
            "band_pixels": nb, "band_fraction_of_field": nb / total_px,
            "flips_captured": hit, "capture_rate": hit / total_flips,
            "enrichment_x": (hit / total_flips) / (nb / total_px),
            "real_lzma1_bits": real_bits, "setcoding_bound_bits": bound,
            "real_over_bound_ratio": ratio,
            "address_bytes": addr_bits / 8.0, "payload_bytes": pay_bits / 8.0,
            "total_bytes": by, "rate_S": by * RATE_PER_BYTE,
        })

    # normalise against gp1's PRICED row (A3 = distill seed, L1 SE, r=1) when it is in scope
    base = next((r for r in rows if r["seed"] == "distill_field"
                 and r["structuring_element"] == "gp1_L1" and r["dilate_r"] == 1), None)
    for r in rows:
        r["band_px_x_gp1_r1"] = (r["band_pixels"] / base["band_pixels"]) if base else None
        r["bytes_x_gp1_r1"] = (r["total_bytes"] / base["total_bytes"]) if base else None

    edges = []
    for i in range(5):
        for j in range(5):
            if edge_tot[i, j]:
                edges.append({
                    "gt_class": CLASS_NAMES[i], "rendered_class": CLASS_NAMES[j],
                    "flips": int(edge_tot[i, j]),
                    "uncaptured_at_gp1_r1": int(edge_unc[i, j]),
                    "uncaptured_frac": float(edge_unc[i, j] / edge_tot[i, j]),
                    "share_of_all_flips": float(edge_tot[i, j] / total_flips),
                })
    edges.sort(key=lambda e: -e["flips"])

    out = {
        "schema": "ddm_et1_band_recalibration.v1",
        "axis": "[macOS-CPU cache-derived advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n_pairs": n, "denominator_note": "n600 FULL -- no subsetting, m96 cannot apply",
        "denominators": {"live_best_S": LIVE_BEST_S, "floor_S": PR130_FLOOR_S,
                         "gap_S": GAP_S, "S_per_flip": S_PER_FLIP,
                         "stale_gap_used_by_sq1": 0.654355209256714},
        "se_footprints_px": {k: {str(a): b for a, b in v.items()} for k, v in foot.items()},
        "total_flips": total_flips,
        "seed_agreement_distill_vs_segnet": seed_agree / seed_px,
        "rows": rows, "edges": edges,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print("\n=== SE footprints (closed form, data-independent) ===")
    for name, v in foot.items():
        print(f"  {name:18s} " + "  ".join(f"r{a}={b}px" for a, b in v.items()))
    print(f"\nseed agreement distill vs segnet-argmax: {seed_agree/seed_px:.6f}")
    print(f"total flips n600: {total_flips}")
    print(f"\n{'seed':15s} {'SE':16s} {'r':>2s} {'band%':>7s} {'capture':>8s} "
          f"{'bytes':>10s} {'xGP1px':>7s} {'xGP1B':>7s}")
    def _f(v: float | None) -> str:
        return "    n/a" if v is None else f"{v:7.3f}"

    for r in rows:
        print(f"{r['seed']:15s} {r['structuring_element']:16s} {r['dilate_r']:2d} "
              f"{100*r['band_fraction_of_field']:7.3f} {100*r['capture_rate']:8.3f} "
              f"{r['total_bytes']:10.0f} {_f(r['band_px_x_gp1_r1'])} {_f(r['bytes_x_gp1_r1'])}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
