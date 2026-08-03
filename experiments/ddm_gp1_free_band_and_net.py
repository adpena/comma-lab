#!/usr/bin/env python
"""ddm_gp1 pass3 -- is the ADDRESS BAND free, and what is the NET score?

Scorer-free ($0). ZERO SegNet/PoseNet forwards.
Axis: [macOS-CPU cache-derived advisory] NON-PROMOTABLE. score_claim=False.
pointer 0.1910828242 [contest-CPU] UNMOVED.

Three questions, all measured:

1. FREE BAND vs ORACLE BAND.
   The ORACLE band ranks pixels by the frozen scorer's margin -- not receiver
   legal (73 MB scorer). The FREE band is dilate(boundary(L)) where L is a
   label field the receiver already holds after decoding (rz1 measured that the
   decoder's own L* recovers 93.53% of the margin separatrix). If the FREE band
   captures flips about as well as the ORACLE band, the whole micro-student rung
   is unnecessary for ADDRESSING and GT-shipping wins outright.
   Here L is the cached near-GT render argmax -- a label field of exactly the
   kind a receiver holds. Cross-vehicle to cx1's flips, so BOUND-flavoured, and
   labelled as such.

2. NET score, not gross.
   Every gross BOUND-IF-REALIZED dS must pay its own rate. Net is
       net_dS = -(gain) + 25*bytes/37_545_489
   Negative net = score goes down = worth doing. The sweep finds the optimum.

3. RUNNER-UP PAYLOAD.
   ru1 measured 98.8% of flips have GT == the render's runner-up class. If the
   receiver can name its own runner-up, the payload collapses from H(gt|rend)
   to ~H_b(0.988). Measured here on the field's OWN error set (within-vehicle).
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
RATE_PER_BYTE = 25.0 / 37_545_489.0  # S per archive byte
LIVE_BEST_S = 0.7910689
GAP = 0.6189279
DILATE_R = (1, 2, 3, 5, 8)


def log2_binom(n: int, k: int) -> float:
    if k <= 0 or k >= n:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2.0)


def hbin(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def boundary(lab: np.ndarray) -> np.ndarray:
    """4-neighbour label boundary: pixel differs from any orthogonal neighbour."""
    b = np.zeros(lab.shape, dtype=bool)
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :] |= lab[:-1, :] != lab[1:, :]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    return b


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Chebyshev dilation by r via iterated 3x3 max (separable OR)."""
    m = mask
    for _ in range(r):
        o = m.copy()
        o[:-1, :] |= m[1:, :]
        o[1:, :] |= m[:-1, :]
        o[:, :-1] |= m[:, 1:]
        o[:, 1:] |= m[:, :-1]
        m = o
    return m


def lzma1_raw(b: bytes) -> int:
    import lzma

    filt = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}]
    return len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=filt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--coder-pairs", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "gp1_pass3.json"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    n = args.n_pairs

    gt = np.load(os.path.join(ARGMAX_DIR, "gt_argmax_n600.npy"), mmap_mode="r")
    rd = np.load(os.path.join(ARGMAX_DIR, "cx1_argmax_n600.npy"), mmap_mode="r")

    # accumulators
    free_band_px = {r: 0 for r in DILATE_R}
    free_band_hit = {r: 0 for r in DILATE_R}
    free_band_hit_rl = {r: 0 for r in DILATE_R}
    # real-coder accumulators are over the CODER SUBSET only; they must be
    # compared against a bound computed on the SAME subset, never against the
    # full-population bound (that mismatch understates bytes by n/coder_pairs).
    free_band_bits_real = {r: 0 for r in DILATE_R}
    sub_band_px = {r: 0 for r in DILATE_R}
    sub_band_hit = {r: 0 for r in DILATE_R}
    total_flips = 0
    total_rl = 0
    # runner-up structure on the field's own error set
    ru_err = 0
    ru_is_runner = 0
    # real coder on free-band bitstring
    t0 = time.time()
    for p in range(n):
        g = np.asarray(gt[p])
        r_ = np.asarray(rd[p])
        diff = g != r_
        k = int(diff.sum())
        total_flips += k
        rl = ((g == 0) & (r_ == 1)) | ((g == 1) & (r_ == 0))
        total_rl += int(rl.sum())

        with np.load(os.path.join(FIELD_DIR, f"pair-{p:06d}.npz")) as z:
            lab = np.asarray(z["argmax"])
            if p < 60:
                lg = np.asarray(z["distill_logits"], dtype=np.float32)
        # ---- FREE band from the receiver-held label field
        bnd = boundary(lab)
        for rr in DILATE_R:
            band = dilate(bnd, rr)
            nb = int(band.sum())
            free_band_px[rr] += nb
            free_band_hit[rr] += int((diff & band).sum())
            free_band_hit_rl[rr] += int((rl & band).sum())
            if p < args.coder_pairs:
                # bitstring of flip-or-not restricted to band pixels, raster order
                seg = diff[band].astype(np.uint8)
                free_band_bits_real[rr] += lzma1_raw(np.packbits(seg).tobytes()) * 8
                sub_band_px[rr] += nb
                sub_band_hit[rr] += int((diff & band).sum())

        # ---- runner-up structure on the field's OWN errors (within-vehicle)
        if p < 60:
            ferr = lab != g
            ne = int(ferr.sum())
            if ne:
                srt = np.argsort(-lg, axis=0)  # (5,H,W) desc
                runner = srt[1]
                ru_err += ne
                ru_is_runner += int((g[ferr] == runner[ferr]).sum())
        if (p + 1) % 100 == 0:
            print(f"  pass3 {p+1}/{n}  {time.time()-t0:.1f}s", flush=True)

    total_px = n * PIX_PER_PAIR

    free_rows = []
    for rr in DILATE_R:
        nb = free_band_px[rr]
        hit = free_band_hit[rr]
        hit_rl = free_band_hit_rl[rr]
        set_bits = log2_binom(nb, hit)
        p_ = hit / nb if nb else 0.0
        iid_bits = nb * hbin(p_)
        # like-for-like: real coder on the subset vs the SAME subset's bound.
        sub_bound = log2_binom(sub_band_px[rr], sub_band_hit[rr])
        real_bits_sub = free_band_bits_real[rr]
        ratio = (real_bits_sub / sub_bound) if sub_bound else float("nan")
        # full-population address estimate = population bound x measured ratio
        addr_bits = set_bits * ratio
        # payload: ALL restriction uses H(gt|rend)=1.1012; RL restriction is 0.0
        pay_all = hit * 1.1011521270613085
        b_all = (addr_bits + pay_all) / 8.0
        b_rl_set = log2_binom(nb, hit_rl)
        # payload is EXACTLY zero for the 2-class restriction (measured pass2)
        b_rl = (b_rl_set * ratio) / 8.0
        free_rows.append({
            "dilate_r": rr,
            "band_pixels": nb,
            "band_fraction_of_field": nb / total_px,
            "flips_captured_ALL": hit,
            "capture_rate_ALL": hit / total_flips,
            "enrichment_x": (hit / total_flips) / (nb / total_px),
            "flips_captured_RoadLane": hit_rl,
            "capture_rate_RoadLane": hit_rl / total_rl,
            "setcoding_bits_population": set_bits,
            "iid_entropy_bits": iid_bits,
            "coder_subset_pairs": args.coder_pairs,
            "coder_subset_bound_bits": sub_bound,
            "coder_subset_LZMA1_bits": real_bits_sub,
            "real_over_bound_ratio": ratio,
            "address_bits_population_est": addr_bits,
            "bits_per_band_pixel_real": real_bits_sub / sub_band_px[rr] if sub_band_px[rr] else 0.0,
            "ALL_total_bytes": b_all,
            "ALL_gross_dS": 100.0 * hit / total_px,
            "ALL_rate_cost_S": b_all * RATE_PER_BYTE,
            "ALL_net_dS": b_all * RATE_PER_BYTE - 100.0 * hit / total_px,
            "RoadLane_total_bytes": b_rl,
            "RoadLane_gross_dS": 100.0 * hit_rl / total_px,
            "RoadLane_rate_cost_S": b_rl * RATE_PER_BYTE,
            "RoadLane_net_dS": b_rl * RATE_PER_BYTE - 100.0 * hit_rl / total_px,
        })

    out = {
        "schema": "ddm_gp1_free_band_and_net.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[macOS-CPU cache-derived advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "scorer_forwards": 0,
        "baseline": {"live_best_S": LIVE_BEST_S, "gap_to_floor": GAP,
                     "rate_S_per_byte": RATE_PER_BYTE},
        "totals": {"flips_ALL": total_flips, "flips_RoadLane": total_rl,
                   "field_pixels": total_px},
        "free_band_from_receiver_label_boundary": free_rows,
        "runner_up_payload": {
            "sample_pairs": 60,
            "field_own_errors": ru_err,
            "gt_is_runner_up": ru_is_runner,
            "rate": ru_is_runner / ru_err if ru_err else None,
            "payload_bits_if_receiver_names_runner_up":
                hbin(ru_is_runner / ru_err) + (1 - ru_is_runner / ru_err) * math.log2(3)
                if ru_err else None,
            "note": "within-vehicle on the near-GT field's own error set; ru1 "
                    "measured 98.8% on the endpoint vehicle",
        },
        "labels": {
            "bytes": "MEASURED (LZMA1-raw) or EXACT-BOUND (log2 C(n,k))",
            "dS": "BOUND-IF-REALIZED -- upper bound on gain; realization unbuilt",
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\nwrote {args.out}")
    ru = out["runner_up_payload"]
    print(f"runner-up: {ru['gt_is_runner_up']}/{ru['field_own_errors']} = "
          f"{ru['rate']:.4f} -> payload {ru['payload_bits_if_receiver_names_runner_up']:.4f} bits/flip")
    print("\nFREE band (receiver's own label boundary, dilated):")
    print("  r  band%   capALL  capRL   b/bandpx   ALL_bytes  ALL_net_dS   RL_bytes  RL_net_dS")
    for r in free_rows:
        print(f"  {r['dilate_r']}  {r['band_fraction_of_field']*100:5.2f}%  "
              f"{r['capture_rate_ALL']:.4f}  {r['capture_rate_RoadLane']:.4f}  "
              f"{r['bits_per_band_pixel_real']:.4f}   "
              f"{r['ALL_total_bytes']:9.0f}  {r['ALL_net_dS']:+.5f}   "
              f"{r['RoadLane_total_bytes']:8.0f}  {r['RoadLane_net_dS']:+.5f}")


if __name__ == "__main__":
    main()
