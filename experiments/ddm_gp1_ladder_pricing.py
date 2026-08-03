#!/usr/bin/env python
"""ddm_gp1 pass2 -- price every rung of the selectivity ladder with real coders.

Scorer-free ($0). ZERO SegNet/PoseNet forwards. Consumes cached argmax +
cached frozen-scorer margin fields only.

Axis: [macOS-CPU cache-derived advisory] NON-PROMOTABLE. score_claim=False.
pointer 0.1910828242 [contest-CPU] UNMOVED.

Prices, for each rung, ADDRESS and PAYLOAD separately (gt2r ADDRESS LAW), under
both the NAIVE COORDINATE mode and the CHEAP composed mode (margin-rank band +
set coding + rank-conditioned prior).

Cost model, all MEASURED or EXACT-BOUND
---------------------------------------
NAIVE_COORD  address = k * log2(PIX_PER_PAIR)         [uniform coordinate]
             payload = k * H(gt | rendered)
BAND_SET     address = log2 C(n_band, k)               [exact set-coding bound]
             payload = k * H(gt | rendered)
BAND_PRIOR   address = sum_buckets n_b * H(p_b)        [rank-conditioned prior;
             the receiver knows the margin rank, so flip density is not iid]
             payload = k * H(gt | rendered)
Real coders (brotli-Q11, LZMA1-raw) are run on the actual bitstrings/maps to
confirm the bounds are approachable rather than merely asserted.

Every S column is BOUND-IF-REALIZED: the seg contribution eliminated if the
shipped correction perfectly zeroed the captured flips. UPPER bound on gain.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

ARGMAX_DIR = "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"
FIELD_DIR = "/Volumes/VertigoDataTier/pact/ddm_b2b_qa75_field_20260730"
OUT_DIR = "/Volumes/VertigoDataTier/pact/ddm_gp1_20260803"

SEG_H, SEG_W = 384, 512
PIX_PER_PAIR = SEG_H * SEG_W
N_PAIRS = 600
CLASS_ORDER = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
COORD_BITS = math.log2(PIX_PER_PAIR)  # 17.585

# live-best baseline supplied by charter (own-vehicle frontier)
LIVE_BEST_S = 0.7910689
LIVE_BEST_BYTES = 353805
GAP_TO_FLOOR = 0.6189279
BYTES_PER_PCT_GAP = 9295

# rank buckets used for the rank-conditioned prior (fractions of a pair)
RANK_EDGES = np.array(
    [0.0, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]
)
BAND_Q = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 1.00]


def log2_binom(n: int, k: int) -> float:
    if k <= 0 or k >= n:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2.0)


def hbin(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def entropy_bits(counts) -> float:
    c = np.asarray(counts, dtype=np.float64)
    t = c.sum()
    if t <= 0:
        return 0.0
    p = c[c > 0] / t
    return float(-(p * np.log2(p)).sum())


def brotli_q11(b: bytes) -> int:
    import brotli

    return len(brotli.compress(b, quality=11))


def lzma1_raw(b: bytes) -> int:
    import lzma

    filt = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}]
    return len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=filt))


def load_margin(p: int) -> np.ndarray:
    with np.load(os.path.join(FIELD_DIR, f"pair-{p:06d}.npz")) as z:
        return np.asarray(z["distill_margin"], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--coder-sample", type=int, default=40,
                    help="pairs to run real coders on (full run is slow)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "gp1_pass2.json"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    n = args.n_pairs

    gt = np.load(os.path.join(ARGMAX_DIR, "gt_argmax_n600.npy"), mmap_mode="r")
    rd = np.load(os.path.join(ARGMAX_DIR, "cx1_argmax_n600.npy"), mmap_mode="r")

    nb = len(RANK_EDGES) - 1
    # global accumulators
    bucket_pixels = np.zeros(nb, dtype=np.int64)
    bucket_flips = np.zeros(nb, dtype=np.int64)
    per_pair_flips = np.zeros(n, dtype=np.int64)
    # per (class-pair-restriction, bucket) flips, for R2
    # restrictions: 0=all, 1=Road<->Lane edge, 2=Lane involved, 3=Movable involved
    NR = 4
    r_bucket_flips = np.zeros((NR, nb), dtype=np.int64)
    r_total = np.zeros(NR, dtype=np.int64)
    r_per_pair = np.zeros((NR, n), dtype=np.int64)
    trans = np.zeros((5, 5), dtype=np.int64)
    # payload conditional entropy accumulators per restriction
    r_trans = np.zeros((NR, 5, 5), dtype=np.int64)

    # real-coder accumulators, per band q
    coder = {q: {"band_bits_lzma": 0, "band_bits_brotli": 0, "n_band": 0, "k": 0,
                 "setbound_bits": 0.0} for q in BAND_Q}
    dense_brotli = 0
    dense_pairs = 0
    naive_lzma_bytes = 0

    t0 = time.time()
    edges_pix = (RANK_EDGES * PIX_PER_PAIR).astype(np.int64)
    for p in range(n):
        g = np.asarray(gt[p]).ravel()
        r = np.asarray(rd[p]).ravel()
        diff = g != r
        k = int(diff.sum())
        per_pair_flips[p] = k
        if k == 0:
            bucket_pixels += np.diff(edges_pix)
            continue
        np.add.at(trans, (g[diff].astype(np.int64), r[diff].astype(np.int64)), 1)

        m = load_margin(p).ravel()
        order = np.argsort(m, kind="stable")
        rank = np.empty(m.size, dtype=np.int64)
        rank[order] = np.arange(m.size)

        bucket_pixels += np.diff(edges_pix)
        fr = rank[diff]
        bi = np.searchsorted(edges_pix, fr, side="right") - 1
        np.add.at(bucket_flips, bi, 1)

        gg = g[diff].astype(np.int64)
        rr = r[diff].astype(np.int64)
        # restriction masks
        masks = {
            0: np.ones(k, dtype=bool),
            1: ((gg == 0) & (rr == 1)) | ((gg == 1) & (rr == 0)),
            2: (gg == 1) | (rr == 1),
            3: (gg == 3) | (rr == 3),
        }
        for ri, msk in masks.items():
            cnt = int(msk.sum())
            r_total[ri] += cnt
            r_per_pair[ri, p] = cnt
            if cnt:
                np.add.at(r_bucket_flips[ri], bi[msk], 1)
                np.add.at(r_trans[ri], (gg[msk], rr[msk]), 1)

        # ---- real coders on a sample of pairs
        if p < args.coder_sample:
            # dense label map: 0 = no correction, 1+gt_class otherwise
            dm = np.zeros(m.size, dtype=np.uint8)
            dm[diff] = (g[diff] + 1)
            dense_brotli += brotli_q11(dm.reshape(SEG_H, SEG_W).tobytes())
            dense_pairs += 1
            # naive sparse: delta-coded raster positions (varint) + labels, LZMA1
            pos = np.flatnonzero(diff)
            d = np.diff(np.concatenate(([0], pos)))
            buf = bytearray()
            for v in d.tolist():
                x = int(v)
                while True:
                    b = x & 0x7F
                    x >>= 7
                    buf.append(b | 0x80 if x else b)
                    if not x:
                        break
            naive_lzma_bytes += lzma1_raw(bytes(buf)) + lzma1_raw(g[diff].tobytes())
            # band-restricted bitstring in margin-rank order
            flip_by_rank = np.zeros(m.size, dtype=np.uint8)
            flip_by_rank[rank[diff]] = 1
            for q in BAND_Q:
                nbq = int(q * m.size)
                seg = flip_by_rank[:nbq]
                kk = int(seg.sum())
                packed = np.packbits(seg).tobytes()
                coder[q]["band_bits_lzma"] += lzma1_raw(packed) * 8
                coder[q]["band_bits_brotli"] += brotli_q11(packed) * 8
                coder[q]["n_band"] += nbq
                coder[q]["k"] += kk
                coder[q]["setbound_bits"] += log2_binom(nbq, kk)
        if (p + 1) % 100 == 0:
            print(f"  pass2 {p+1}/{n}  {time.time()-t0:.1f}s", flush=True)

    total = int(per_pair_flips.sum())
    total_pixels = n * PIX_PER_PAIR

    # ---- payload rate per restriction: H(gt | rendered)
    def cond_H(tm):
        off = tm.copy()
        np.fill_diagonal(off, 0)
        tot = off.sum()
        if tot == 0:
            return 0.0
        h = 0.0
        for rc in range(5):
            col = off[:, rc]
            if col.sum():
                h += (col.sum() / tot) * entropy_bits(col)
        return float(h)

    payload_bits = cond_H(trans)

    # ---- RD curve over band fraction, three address modes
    cum_pix = np.cumsum(bucket_pixels).astype(np.float64)
    cum_flip = np.cumsum(bucket_flips).astype(np.float64)

    def band_at(q):
        """interpolate cumulative pixels/flips at band fraction q"""
        target = q * total_pixels
        idx = int(np.searchsorted(cum_pix, target))
        if idx == 0:
            frac = target / cum_pix[0] if cum_pix[0] else 0.0
            return target, cum_flip[0] * frac
        if idx >= nb:
            return cum_pix[-1], cum_flip[-1]
        lo_p, hi_p = cum_pix[idx - 1], cum_pix[idx]
        lo_f, hi_f = cum_flip[idx - 1], cum_flip[idx]
        w = (target - lo_p) / (hi_p - lo_p) if hi_p > lo_p else 0.0
        return target, lo_f + w * (hi_f - lo_f)

    def prior_bits(q):
        """rank-conditioned address bits: sum over buckets inside the band"""
        target = q * total_pixels
        bits = 0.0
        acc = 0.0
        for j in range(nb):
            npx = float(bucket_pixels[j])
            if acc + npx <= target:
                use = npx
            else:
                use = max(0.0, target - acc)
            if use > 0:
                pj = bucket_flips[j] / npx if npx else 0.0
                bits += use * hbin(pj)
            acc += npx
            if acc >= target:
                break
        return bits

    rd_curve = []
    for q in BAND_Q:
        npx, kf = band_at(q)
        kf_i = int(round(kf))
        set_bits = log2_binom(int(npx), kf_i)
        pr_bits = prior_bits(q)
        pay_bits = kf * payload_bits
        naive_addr = kf * COORD_BITS
        dseg_removed = kf / total_pixels
        rd_curve.append({
            "band_fraction": q,
            "band_pixels": int(npx),
            "flips_captured": kf_i,
            "capture_rate": kf / total,
            "enrichment_x": (kf / total) / q,
            "bound_if_realized_d_seg_removed": dseg_removed,
            "bound_if_realized_delta_S": 100.0 * dseg_removed,
            "payload_bytes": pay_bits / 8.0,
            "address_bytes_NAIVE_COORD": naive_addr / 8.0,
            "address_bytes_BAND_SET": set_bits / 8.0,
            "address_bytes_BAND_PRIOR": pr_bits / 8.0,
            "total_bytes_NAIVE_COORD": (naive_addr + pay_bits) / 8.0,
            "total_bytes_BAND_SET": (set_bits + pay_bits) / 8.0,
            "total_bytes_BAND_PRIOR": (pr_bits + pay_bits) / 8.0,
            "bits_per_band_pixel_SET": set_bits / npx if npx else 0.0,
            "bits_per_band_pixel_PRIOR": pr_bits / npx if npx else 0.0,
            "bits_per_flip_NAIVE": (naive_addr + pay_bits) / kf if kf else 0.0,
            "bits_per_flip_PRIOR": (pr_bits + pay_bits) / kf if kf else 0.0,
        })

    # ---- per-class restrictions (R2)
    r_names = ["ALL", "Road<->Lane edge", "Lane involved", "Movable involved"]
    r_rows = []
    for ri in range(NR):
        kt = int(r_total[ri])
        if kt == 0:
            continue
        pay = cond_H(r_trans[ri])
        cumf = np.cumsum(r_bucket_flips[ri]).astype(np.float64)
        rows = []
        for q in (0.02, 0.05):
            target = q * total_pixels
            idx = int(np.searchsorted(cum_pix, target))
            if idx == 0:
                kf = cumf[0] * (target / cum_pix[0])
            elif idx >= nb:
                kf = cumf[-1]
            else:
                w = (target - cum_pix[idx - 1]) / (cum_pix[idx] - cum_pix[idx - 1])
                kf = cumf[idx - 1] + w * (cumf[idx] - cumf[idx - 1])
            set_bits = log2_binom(int(target), int(round(kf)))
            pr = prior_bits(q)  # band cost is shared/unchanged by class filter
            rows.append({
                "band_fraction": q,
                "flips_captured": int(round(kf)),
                "capture_rate_within_restriction": kf / kt,
                "bound_if_realized_delta_S": 100.0 * kf / total_pixels,
                "payload_bits_per_flip": pay,
                "address_bytes_BAND_SET_selecting_only_this_class": set_bits / 8.0,
                "address_bytes_if_band_shared_with_ALL": pr / 8.0,
                "total_bytes_BAND_SET": (set_bits + kf * pay) / 8.0,
            })
        r_rows.append({
            "restriction": r_names[ri],
            "total_flips": kt,
            "share_of_all_flips": kt / total,
            "payload_H_gt_given_rendered_bits": pay,
            "pair_concentration_top25": float(
                np.sort(r_per_pair[ri])[::-1][:25].sum() / kt),
            "pair_concentration_top112": float(
                np.sort(r_per_pair[ri])[::-1][:112].sum() / kt),
            "bands": rows,
        })

    # ---- real coder verdicts
    coder_out = {}
    for q, c in coder.items():
        if c["n_band"] == 0:
            continue
        coder_out[f"{q:g}"] = {
            "sample_pairs": args.coder_sample,
            "band_pixels": c["n_band"],
            "flips_in_band": c["k"],
            "setcoding_bound_bytes": c["setbound_bits"] / 8.0,
            "LZMA1_raw_bytes": c["band_bits_lzma"] / 8.0,
            "brotli_q11_bytes": c["band_bits_brotli"] / 8.0,
            "best_real_coder_vs_bound": min(c["band_bits_lzma"], c["band_bits_brotli"])
            / c["setbound_bits"] if c["setbound_bits"] else None,
        }

    out = {
        "schema": "ddm_gp1_ladder_pricing.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[macOS-CPU cache-derived advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "scorer_forwards": 0,
        "baseline": {
            "live_best_S": LIVE_BEST_S,
            "live_best_bytes": LIVE_BEST_BYTES,
            "gap_to_floor": GAP_TO_FLOOR,
            "bytes_per_pct_of_gap": BYTES_PER_PCT_GAP,
        },
        "positive_control": {
            "total_flips": total,
            "expected": 508640,
            "verdict": "ARGMAX_VERIFIED" if total == 508640 and n == 600 else "SUBSET",
        },
        "payload_H_gt_given_rendered_bits_per_flip": payload_bits,
        "naive_coordinate_bits": COORD_BITS,
        "rd_curve_band": rd_curve,
        "per_class_restrictions": r_rows,
        "real_coder_vs_setcoding_bound": coder_out,
        "real_coder_dense_and_naive": {
            "sample_pairs": dense_pairs,
            "dense_labelmap_brotli_q11_bytes": dense_brotli,
            "naive_sparse_deltacoord_lzma1_bytes": naive_lzma_bytes,
            "flips_in_sample": int(per_pair_flips[:dense_pairs].sum()),
        },
        "pair_selectivity_seg": {
            f"top{k}": float(np.sort(per_pair_flips)[::-1][:k].sum() / total)
            for k in (6, 25, 56, 112, 200, 300, 600)
        },
        "rank_buckets": {
            "edges_fraction": RANK_EDGES.tolist(),
            "pixels": bucket_pixels.tolist(),
            "flips": bucket_flips.tolist(),
            "flip_density": (bucket_flips / np.maximum(bucket_pixels, 1)).tolist(),
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")
    print(f"payload H(gt|rendered) = {payload_bits:.4f} bits/flip")
    print("\n band     capture  dS_bound   NAIVE B    SET B     PRIOR B   b/bandpx")
    for r in rd_curve:
        print(f" {r['band_fraction']:<7g} {r['capture_rate']:.4f}  "
              f"{r['bound_if_realized_delta_S']:.5f}  "
              f"{r['total_bytes_NAIVE_COORD']:9.0f} {r['total_bytes_BAND_SET']:9.0f} "
              f"{r['total_bytes_BAND_PRIOR']:9.0f}  {r['bits_per_band_pixel_PRIOR']:.4f}")
    print("\nreal coder vs set-coding bound:")
    for q, c in coder_out.items():
        print(f"  q={q:<6} bound={c['setcoding_bound_bytes']:9.0f}B  "
              f"lzma={c['LZMA1_raw_bytes']:9.0f}B  brotli={c['brotli_q11_bytes']:9.0f}B  "
              f"ratio={c['best_real_coder_vs_bound']:.3f}")


if __name__ == "__main__":
    main()
