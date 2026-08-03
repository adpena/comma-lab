#!/usr/bin/env python
"""ddm_gp1 -- price the SELECTIVITY LADDER for shipping ground-truth corrections.

Scorer-free ($0): consumes only cached argmax arrays and cached frozen-scorer
margin fields. Performs ZERO SegNet/PoseNet forwards.

Axis: [macOS-CPU cache-derived advisory] NON-PROMOTABLE.
score_claim=False. pointer 0.1910828242 [contest-CPU] UNMOVED.

What this measures
------------------
The operator's selectivity ladder asks: how cheap does a ground-truth
correction get when we ship it only for selected pairs / classes / entities /
dimensions?  Cost has two separable halves (gt2r ADDRESS LAW: address is
76-78% of explicit cost):

    cost = ADDRESS (where the correction applies) + PAYLOAD (what it says)

This script prices ADDRESS under five modes and PAYLOAD under measured
conditional entropy, using real coders (brotli-Q11, LZMA1-raw) plus exact
combinatorial set-coding bounds.

Address modes priced
--------------------
  NAIVE_COORD   : explicit (y,x) per flip, delta-coded raster order + LZMA1.
  DENSE_MASK    : per-pair 384x512 label map, brotli-Q11 (spatial coherence).
  BAND_SET      : receiver ranks its own margin field, we ship a SET of ranks
                  within the low-margin band -> log2 C(n_band, k) exactly.
  BAND_RUN      : band membership + run-structure, real coder on the band-
                  restricted bitstring.
  STATIC_UNION  : one shared address template across all pairs, then per-pair
                  selection inside it (amortises address over the population).

BOUND labelling
---------------
Every S column is BOUND-IF-REALIZED: it is the seg contribution that would be
eliminated if the shipped correction perfectly zeroed the selected flips. It
is an UPPER bound on gain. Byte columns are MEASURED (real coder output) or
EXACT-BOUND (combinatorial log2 C(n,k), which no coder can beat).

The margin band is derived from a frozen-scorer margin field of a near-GT
render. A legal receiver cannot run the frozen scorer (73 MB); it would need a
distilled student. So BAND_* rows are ORACLE-BAND bounds, and the student
parameter cost that would make them legal is priced separately (rung R5).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np

ARGMAX_DIR = "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"
FIELD_DIR = "/Volumes/VertigoDataTier/pact/ddm_b2b_qa75_field_20260730"
OUT_DIR = "/Volumes/VertigoDataTier/pact/ddm_gp1_20260803"

SEG_H, SEG_W = 384, 512
PIX_PER_PAIR = SEG_H * SEG_W  # 196608
N_PAIRS = 600
CLASS_ORDER = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

# Receipt-verified positive control (ddm_pu2 cx1_directed_flip_receipt.json).
EXPECTED_TOTAL_FLIPS = 508640
EXPECTED_D_SEG = 0.004311794704861111


# --------------------------------------------------------------------------
# exact information-theoretic bounds
# --------------------------------------------------------------------------
def log2_binom(n: int, k: int) -> float:
    """Exact log2 C(n, k) in bits. No coder can beat this for a set of k of n."""
    if k <= 0 or k >= n:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2.0)


def entropy_bits(counts: np.ndarray) -> float:
    """Shannon entropy (bits/symbol) of an empirical count vector."""
    c = np.asarray(counts, dtype=np.float64)
    tot = c.sum()
    if tot <= 0:
        return 0.0
    p = c[c > 0] / tot
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------
# real coders
# --------------------------------------------------------------------------
def brotli_q11(payload: bytes) -> int:
    import brotli

    return len(brotli.compress(payload, quality=11))


def lzma1_raw(payload: bytes) -> int:
    """PR95-family L24 canonical: FORMAT_RAW + FILTER_LZMA1, no container header."""
    import lzma

    filt = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}]
    return len(lzma.compress(payload, format=lzma.FORMAT_RAW, filters=filt))


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def pack_bits(bools: np.ndarray) -> bytes:
    return np.packbits(bools.astype(np.uint8)).tobytes()


# --------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------
def load_argmax():
    gt = np.load(os.path.join(ARGMAX_DIR, "gt_argmax_n600.npy"), mmap_mode="r")
    rd = np.load(os.path.join(ARGMAX_DIR, "cx1_argmax_n600.npy"), mmap_mode="r")
    return gt, rd


def load_margin(pair: int) -> np.ndarray:
    with np.load(os.path.join(FIELD_DIR, f"pair-{pair:06d}.npz")) as z:
        return np.asarray(z["distill_margin"], dtype=np.float32)


# --------------------------------------------------------------------------
# pass 1 -- flip census + margin-rank enrichment
# --------------------------------------------------------------------------
@dataclass
class PairStat:
    pair: int
    flips: int
    # count of this pair's flips whose margin percentile-rank falls below each
    # band fraction in BAND_Q
    in_band: list


BAND_Q = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]


def pass1(n_pairs: int, verbose: bool = True):
    gt, rd = load_argmax()
    per_flips = np.zeros(n_pairs, dtype=np.int64)
    in_band = np.zeros((n_pairs, len(BAND_Q)), dtype=np.int64)
    # directed transition counts, gt_class x rendered_class
    trans = np.zeros((5, 5), dtype=np.int64)
    t0 = time.time()
    for p in range(n_pairs):
        g = np.asarray(gt[p])
        r = np.asarray(rd[p])
        diff = g != r
        k = int(diff.sum())
        per_flips[p] = k
        np.add.at(trans, (g[diff].astype(np.int64), r[diff].astype(np.int64)), 1)
        if k:
            m = load_margin(p).ravel()
            order = np.argsort(m, kind="stable")
            # percentile rank of every pixel under ascending margin
            rank = np.empty(m.size, dtype=np.int64)
            rank[order] = np.arange(m.size)
            fr = rank[diff.ravel()]
            for j, q in enumerate(BAND_Q):
                in_band[p, j] = int((fr < q * m.size).sum())
        if verbose and (p + 1) % 100 == 0:
            print(f"  pass1 {p+1}/{n_pairs}  {time.time()-t0:.1f}s", flush=True)
    return per_flips, in_band, trans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "gp1_pass1.json"))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    per_flips, in_band, trans = pass1(args.n_pairs)

    total = int(per_flips.sum())
    d_seg = total / (args.n_pairs * PIX_PER_PAIR)
    control_ok = (args.n_pairs == N_PAIRS) and (total == EXPECTED_TOTAL_FLIPS)

    # ---- enrichment: flips captured vs band fraction, population-wide
    cap = in_band.sum(axis=0)
    enrich = {
        f"{q:g}": {
            "band_fraction": q,
            "flips_captured": int(cap[j]),
            "capture_rate": float(cap[j] / total),
            "enrichment_x": float((cap[j] / total) / q),
            "band_pixels_total": int(round(q * PIX_PER_PAIR * args.n_pairs)),
        }
        for j, q in enumerate(BAND_Q)
    }

    # ---- payload entropy: bits to name the corrected class given the wrong one
    # receiver knows its own rendered class r; we must name gt class g != r.
    off = trans.copy()
    np.fill_diagonal(off, 0)
    cond_bits = 0.0
    tot_off = off.sum()
    for r in range(5):
        col = off[:, r]
        if col.sum():
            cond_bits += (col.sum() / tot_off) * entropy_bits(col)
    marginal_bits = entropy_bits(off.ravel())

    out = {
        "schema": "ddm_gp1_selectivity_pass1.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[macOS-CPU cache-derived advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "scorer_forwards": 0,
        "n_pairs": args.n_pairs,
        "positive_control": {
            "total_flips": total,
            "expected_total_flips": EXPECTED_TOTAL_FLIPS,
            "d_seg": d_seg,
            "expected_d_seg": EXPECTED_D_SEG,
            "verdict": "ARGMAX_VERIFIED" if control_ok else "SUBSET_OR_MISMATCH",
        },
        "band_source": {
            "field": "distill_logit_margin (frozen scorer, qa75_solve render)",
            "note": (
                "near-GT render (99.9% GT-agreeing); band is scene-geometry margin. "
                "Frozen scorer is NOT receiver-legal (73 MB) -> BAND_* rows are "
                "ORACLE-BAND bounds; legal use requires a distilled student (rung R5)."
            ),
        },
        "flip_concentration_by_pair": {
            f"top{k}": float(np.sort(per_flips)[::-1][:k].sum() / total)
            for k in (6, 25, 56, 112, 200, 300)
        },
        "margin_band_enrichment": enrich,
        "payload_entropy_bits_per_flip": {
            "marginal_H_transition": marginal_bits,
            "conditional_H_gt_given_rendered": cond_bits,
            "note": "receiver knows its own rendered class; conditional is the payable rate",
        },
        "directed_transitions_gt_by_rendered": trans.tolist(),
        "class_order": list(CLASS_ORDER),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    np.save(os.path.join(OUT_DIR, "gp1_per_pair_flips.npy"), per_flips)
    np.save(os.path.join(OUT_DIR, "gp1_in_band.npy"), in_band)
    print(json.dumps({k: out[k] for k in
                      ("positive_control", "flip_concentration_by_pair",
                       "payload_entropy_bits_per_flip")}, indent=1))
    for q, v in enrich.items():
        print(f"  band q={q:>6}  capture={v['capture_rate']:.4f}  enrich={v['enrichment_x']:.1f}x")


if __name__ == "__main__":
    main()
