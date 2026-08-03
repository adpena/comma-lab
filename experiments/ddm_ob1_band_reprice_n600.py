#!/usr/bin/env python
"""ddm_ob1 Job A -- re-derive gp1's band prices at n600 on the DECODER'S ACTUAL L*.

Scorer-free ($0). ZERO SegNet/PoseNet forwards.
Axis: [macOS-CPU cache-derived advisory] NON-PROMOTABLE. score_claim=False.
pointer 0.1910828242 [contest-CPU] UNMOVED.

WHY THIS IS $0 AND STILL AUTHORITATIVE
--------------------------------------
gp1 computed its "FREE label-boundary band" from `ddm_b2b_qa75_field_20260730/pair-*.npz`
["argmax"] -- the argmax of a *qa75_solve* render, which gp1 itself labelled a NEAR-GT PROXY
for the decoder's own label field.  But the flips it addressed come from `cx1_argmax_n600.npy`
vs `gt_argmax_n600.npy`.  Band and flips therefore came from DIFFERENT vehicles.

sq1's positive control C2 measured `frozen_SegNet(decoded frames).argmax == cx1_argmax_n600[p]`
EXACT on 32/32 pairs.  So `cx1_argmax_n600.npy` IS the decoder's actual L*, already on disk.
Recomputing the band from it needs no scorer forward at all -- gp1's own F4 falsifier is a $0
n600 measurement, and sq1 answered it only on n=32 (and, see below, with a different dilation).

WHAT ELSE THIS CORRECTS
-----------------------
sq1's `dilate()` docstring claims "Chebyshev dilation by r -- gp1's convention"; it is neither.
It applies a 4-neighbour dilation AND THEN a vertical-only dilation per iteration, giving an
ANISOTROPIC band (+/-r horizontal, +/-2r vertical) that is 2.20x larger than gp1's at r=1.
So sq1's "byte-for-byte the SAME object gp1 priced at 367,523 B" is false.  Both conventions
are measured here on both label fields, so the label-field effect and the dilation effect are
separated instead of confounded.

DENOMINATORS (m66: never a bare delta)
--------------------------------------
live best S = 0.7910689 @ 353,805 B (pu2, sha c72ef357) [macOS-CPU advisory]
PR130 bar   = 0.172141   ->   gap = 0.6189279 ;  1% of gap = 0.0061893 S = 9,295 B
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import os
import time
from collections import Counter

import numpy as np

ARGMAX_DIR = "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"
FIELD_DIR = "/Volumes/VertigoDataTier/pact/ddm_b2b_qa75_field_20260730"
OUT_DIR = "/Volumes/VertigoDataTier/pact/ddm_ob1_20260803"

SEG_H, SEG_W = 384, 512
PIX_PER_PAIR = SEG_H * SEG_W
N_PAIRS = 600
RATE_PER_BYTE = 25.0 / 37_545_489.0
LIVE_BEST_S = 0.7910689
LIVE_BEST_BYTES = 353_805
GAP = 0.6189279
PR130_FLOOR = 0.172141
S_PER_FLIP = 100.0 / (N_PAIRS * PIX_PER_PAIR)
RADII = (1, 2, 3, 5, 8)
ROAD, LANE = 0, 1

# sq1 §2.8 measured eta at n=32 for the pose-neutral (P7 yuv6-null) realizer, and §2.4 for the
# unconstrained one.  Carried here as MULTIPLIERS ONLY, with their n and scope attached.
ETA_POSE_NEUTRAL_N32 = 0.5406
ETA_UNCONSTRAINED_N32 = 0.7895


def log2_binom(n: int, k: int) -> float:
    if k <= 0 or k >= n:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2.0)


def boundary(lab: np.ndarray) -> np.ndarray:
    """4-neighbour label boundary -- VERBATIM from ddm_gp1_free_band_and_net.py."""
    b = np.zeros(lab.shape, dtype=bool)
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :] |= lab[:-1, :] != lab[1:, :]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    return b


def dilate_gp1(mask: np.ndarray, r: int) -> np.ndarray:
    """VERBATIM ddm_gp1_free_band_and_net.py: one 4-neighbour OR per r (L1/diamond)."""
    m = mask
    for _ in range(r):
        o = m.copy()
        o[:-1, :] |= m[1:, :]
        o[1:, :] |= m[:-1, :]
        o[:, :-1] |= m[:, 1:]
        o[:, 1:] |= m[:, :-1]
        m = o
    return m


def dilate_sq1(mask: np.ndarray, r: int) -> np.ndarray:
    """VERBATIM ddm_sq1_eta_seg_realization.py: 4-neighbour OR then a VERTICAL-ONLY OR per r.

    Anisotropic (+/-r horizontal, +/-2r vertical).  Reproduced here so sq1's n=32 numbers can
    be placed on the same axis as gp1's, not to endorse it.
    """
    out = mask.copy()
    for _ in range(r):
        acc = out.copy()
        acc[:-1, :] |= out[1:, :]
        acc[1:, :] |= out[:-1, :]
        acc[:, :-1] |= out[:, 1:]
        acc[:, 1:] |= out[:, :-1]
        cur = acc.copy()
        cur[:-1, :] |= acc[1:, :]
        cur[1:, :] |= acc[:-1, :]
        out = cur
    return out


DILATIONS = {"gp1_L1": dilate_gp1, "sq1_aniso": dilate_sq1}


def lzma1_raw_bits(bits: np.ndarray) -> int:
    filt = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}]
    return len(lzma.compress(np.packbits(bits).tobytes(), format=lzma.FORMAT_RAW,
                             filters=filt)) * 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "ob1_band_reprice_n600.json"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    n = args.n_pairs

    gt = np.load(os.path.join(ARGMAX_DIR, "gt_argmax_n600.npy"), mmap_mode="r")
    rd = np.load(os.path.join(ARGMAX_DIR, "cx1_argmax_n600.npy"), mmap_mode="r")

    # keys: (label_src, dilation, r)
    keys = [(ls, dn, r) for ls in ("actual_Lstar", "gp1_proxy") for dn in DILATIONS
            for r in RADII]
    band_px = Counter()
    band_hit = Counter()
    band_hit_rl = Counter()
    band_bits = Counter()

    total_flips = 0
    total_rl = 0
    # payload control: H(gt | rendered) on the ACTUAL flip set (gp1 reports 1.1012)
    joint = Counter()
    # per-pair for the m88 governing-quantity control
    per_pair_flips = np.zeros(n, dtype=np.int64)
    proxy_agree_gt = 0
    proxy_agree_lstar = 0

    t0 = time.time()
    for p in range(n):
        g = np.asarray(gt[p])
        lstar = np.asarray(rd[p])
        diff = g != lstar
        k = int(diff.sum())
        total_flips += k
        per_pair_flips[p] = k
        rl = ((g == ROAD) & (lstar == LANE)) | ((g == LANE) & (lstar == ROAD))
        total_rl += int(rl.sum())
        for a, b in zip(lstar[diff].tolist(), g[diff].tolist(), strict=True):
            joint[(a, b)] += 1

        with np.load(os.path.join(FIELD_DIR, f"pair-{p:06d}.npz")) as z:
            proxy = np.asarray(z["argmax"])
        proxy_agree_gt += int((proxy == g).sum())
        proxy_agree_lstar += int((proxy == lstar).sum())

        for ls, lab in (("actual_Lstar", lstar), ("gp1_proxy", proxy)):
            bnd = boundary(lab)
            for dn, fn in DILATIONS.items():
                for r in RADII:
                    band = fn(bnd, r)
                    kk = (ls, dn, r)
                    band_px[kk] += int(band.sum())
                    band_hit[kk] += int((diff & band).sum())
                    band_hit_rl[kk] += int((rl & band).sum())
                    band_bits[kk] += lzma1_raw_bits(diff[band])
        if (p + 1) % 50 == 0:
            print(f"  ob1 {p+1}/{n}  {time.time()-t0:.1f}s", flush=True)

    total_px = n * PIX_PER_PAIR

    # payload entropy H(gt | rendered), measured on the ACTUAL flip set
    by_rend = Counter()
    for (a, _b), c in joint.items():
        by_rend[a] += c
    h_cond = 0.0
    for (a, _b), c in joint.items():
        h_cond += -c * math.log2(c / by_rend[a])
    h_cond /= total_flips

    rows = []
    for kk in keys:
        ls, dn, r = kk
        nb, hit, hit_rl = band_px[kk], band_hit[kk], band_hit_rl[kk]
        bound = log2_binom(nb, hit)
        real = band_bits[kk]
        ratio = real / bound if bound else float("nan")
        # coder subset == full population here, so NO prefix scaling and no #875 caveat.
        addr_bits = real
        pay_all = hit * h_cond
        b_all = (addr_bits + pay_all) / 8.0
        gross = S_PER_FLIP * hit
        bound_rl = log2_binom(nb, hit_rl)
        b_rl = (bound_rl * ratio) / 8.0  # payload EXACTLY 0 for the 2-class restriction
        gross_rl = S_PER_FLIP * hit_rl
        rows.append({
            "label_source": ls, "dilation": dn, "dilate_r": r,
            "band_pixels": nb, "band_fraction_of_field": nb / total_px,
            "flips_captured_ALL": hit, "capture_rate_ALL": hit / total_flips,
            "enrichment_x": (hit / total_flips) / (nb / total_px),
            "flips_captured_RoadLane": hit_rl,
            "capture_rate_RoadLane": hit_rl / total_rl,
            "setcoding_bound_bits": bound, "LZMA1_real_bits": real,
            "real_over_bound_ratio": ratio,
            "bits_per_band_pixel_real": real / nb if nb else 0.0,
            "ALL_total_bytes": b_all,
            "ALL_gross_dS_eta1": gross,
            "ALL_rate_cost_S": b_all * RATE_PER_BYTE,
            "ALL_net_dS_eta1": b_all * RATE_PER_BYTE - gross,
            "ALL_net_dS_eta_unconstrained_n32":
                b_all * RATE_PER_BYTE - ETA_UNCONSTRAINED_N32 * gross,
            "ALL_net_dS_eta_pose_neutral_n32":
                b_all * RATE_PER_BYTE - ETA_POSE_NEUTRAL_N32 * gross,
            "RoadLane_total_bytes": b_rl,
            "RoadLane_gross_dS_eta1": gross_rl,
            "RoadLane_net_dS_eta1": b_rl * RATE_PER_BYTE - gross_rl,
            "RoadLane_net_dS_eta_pose_neutral_n32":
                b_rl * RATE_PER_BYTE - ETA_POSE_NEUTRAL_N32 * gross_rl,
        })

    # the NO-BAND anchor: code the flip set over the whole field, no address structure
    noband_bound_bits = log2_binom(total_px, total_flips)
    noband_bytes = (noband_bound_bits + total_flips * h_cond) / 8.0

    out = {
        "schema": "ddm_ob1_band_reprice.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[macOS-CPU cache-derived advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "scorer_forwards": 0,
        "n_pairs": n,
        "baseline": {"live_best_S": LIVE_BEST_S, "live_best_bytes": LIVE_BEST_BYTES,
                     "floor_S": PR130_FLOOR, "gap_to_floor": GAP,
                     "rate_S_per_byte": RATE_PER_BYTE, "S_per_flip": S_PER_FLIP,
                     "one_pct_of_gap_S": GAP / 100.0,
                     "one_pct_of_gap_bytes": (GAP / 100.0) / RATE_PER_BYTE},
        "totals": {"flips_ALL": total_flips, "flips_RoadLane": total_rl,
                   "field_pixels": total_px,
                   "d_seg_reproduced": total_flips / total_px,
                   "payload_H_gt_given_rendered_bits_per_flip": h_cond},
        "proxy_field_agreement": {
            "note": "how close gp1's b2b_qa75 render argmax is to GT and to the decoder's L*",
            "agree_with_GT_frac": proxy_agree_gt / total_px,
            "agree_with_decoder_Lstar_frac": proxy_agree_lstar / total_px,
        },
        "no_band_anchor": {
            "note": "cost of coding the flip set over the whole field with no address band",
            "setcoding_bound_bits": noband_bound_bits, "total_bytes": noband_bytes,
            "net_dS_eta1": noband_bytes * RATE_PER_BYTE - S_PER_FLIP * total_flips,
        },
        "eta_multipliers_carried": {
            "unconstrained_n32": ETA_UNCONSTRAINED_N32,
            "pose_neutral_P7_n32": ETA_POSE_NEUTRAL_N32,
            "scope": "sq1 n=32, [macOS-CPU frozen-scorer advisory]; measured on sq1's ANISOTROPIC "
                     "r=1 band, NOT on gp1's L1 band -- see ob1 memo for the transfer caveat",
        },
        "per_pair_flips": per_pair_flips.tolist(),
        "rows": rows,
        "labels": {"bytes": "MEASURED LZMA1-raw address over ALL 600 pairs (no prefix scaling) "
                            "+ exact payload at measured H(gt|rendered)",
                   "dS": "BOUND-IF-REALIZED at eta=1; eta-scaled columns carry sq1's n=32 eta"},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\nwrote {args.out}")
    print(f"d_seg reproduced: {total_flips/total_px:.18f}  (flips {total_flips})")
    print(f"H(gt|rendered) = {h_cond:.6f} bits/flip   (gp1: 1.1012)")
    print(f"proxy field agrees with GT {proxy_agree_gt/total_px:.4%}, "
          f"with decoder L* {proxy_agree_lstar/total_px:.4%}")
    print(f"NO-BAND anchor: {noband_bytes:,.0f} B\n")
    hdr = ("  label_src     dil        r  band%   capALL  capRL   b/px    bytes    "
           "net@1      net@.5406")
    print(hdr)
    for r in rows:
        print(f"  {r['label_source']:<12s} {r['dilation']:<10s} {r['dilate_r']}  "
              f"{r['band_fraction_of_field']*100:5.2f}%  {r['capture_rate_ALL']:.4f}  "
              f"{r['capture_rate_RoadLane']:.4f}  {r['bits_per_band_pixel_real']:.4f}  "
              f"{r['ALL_total_bytes']:8.0f}  {r['ALL_net_dS_eta1']:+.5f}  "
              f"{r['ALL_net_dS_eta_pose_neutral_n32']:+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
