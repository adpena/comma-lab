#!/usr/bin/env python
"""ddm_et1 Job 2 -- ph1's regional-phase reach ceiling, RE-MEASURED on the vehicle we ship.

ph1 published block16 at gross +0.15511 S / net +0.11186 S (39.89% reach, 64,953 B).  Its own
arithmetic identifies its baseline: `baseline_flips - flips_left = 458,739 - 275,766 = 182,973`
flips, and `182,973 * S_per_flip = 0.15511` exactly.  So **ph1's base field carries 458,739
flips = d_seg 0.0038888** -- the burn/ep399 field -- while **our live-best ships 508,640 flips =
d_seg 0.0043118**.  Different vehicle, 10.9% fewer flips to start from.  Per L18 (ANCESTOR =
LESSONS not NUMBERS) the ceiling does not transfer as a number.

This script measures the same mechanism on OUR field, two ways:

  (A) TRANSFER  -- apply ph1's published offsets to our field.  Honest but pessimistic: those
      offsets were solved against a different field, so this conflates mechanism with transfer.
  (B) RE-SOLVE  -- solve the per-block offset exhaustively over the same (2*rmax+1)^2 lattice
      against OUR field.  This is the mechanism's true ceiling on our vehicle, and it is the
      apples-to-apples analogue of what ph1 did on its own.

Both are EXACT-REALIZATION ceilings in label space: they assume eta = 1, i.e. that RGB can be
produced whose frozen-SegNet argmax equals the translated field.  Realizing that is a separate,
unmeasured step -- the whole point of this unit -- so nothing here is a score.

Offsets are GT-solved and therefore COUNTED video-derived payload; they are priced with a real
coder, never asserted.

Axis: [macOS-CPU cache-derived advisory] NON-PROMOTABLE.  score_claim=false.
"""
from __future__ import annotations

import argparse
import json
import lzma
import os
import time

import numpy as np

SEG_H, SEG_W = 384, 512
N_PAIRS = 600
S_PER_FLIP = 100.0 / (N_PAIRS * SEG_H * SEG_W)
RATE_PER_BYTE = 25.0 / 37_545_489.0
LIVE_BEST_S = 0.7910689
GAP_S = LIVE_BEST_S - 0.172141
ARGMAX_DIR = "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"
PH1_OFFSETS = "/Volumes/VertigoDataTier/pact/ddm_ph1_20260803/offsets_n600_rmax5.npz"
OUT_DIR = "/Volumes/VertigoDataTier/pact/ddm_et1_20260803"
PH1_BASELINE_FLIPS = 458739          # DERIVED from ph1's own published arithmetic
PH1_BLOCK16_GROSS_S = 0.15511
PH1_BLOCK16_BYTES = 64953


def lzma1_raw(b: bytes) -> int:
    filt = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}]
    return len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=filt))


def translate_blocks(lab: np.ndarray, off: np.ndarray, block: int) -> np.ndarray:
    """out[y,x] = lab[y+dy, x+dx] for the block containing (y,x)  -- ph1's convention.

    ph1 memo line 62: "R(y+dy,x+dx)=R(y,x)=G(y,x)", i.e. the field SAMPLES from the shifted
    location.  Edge sampling is clamped, matching a border-replicate translation.
    """
    nby, nbx = SEG_H // block, SEG_W // block
    out = lab.copy()
    for bi in range(nby):
        for bj in range(nbx):
            dy, dx = int(off[bi * nbx + bj][0]), int(off[bi * nbx + bj][1])
            if dy == 0 and dx == 0:
                continue
            ys, ye, xs, xe = bi * block, (bi + 1) * block, bj * block, (bj + 1) * block
            yy = np.clip(np.arange(ys, ye) + dy, 0, SEG_H - 1)
            xx = np.clip(np.arange(xs, xe) + dx, 0, SEG_W - 1)
            out[ys:ye, xs:xe] = lab[np.ix_(yy, xx)]
    return out


def solve_blocks(lab: np.ndarray, gt: np.ndarray, block: int, rmax: int) -> np.ndarray:
    """Exhaustive per-block argmax-agreement offset over the (2*rmax+1)^2 lattice.

    This is a SEARCH, not a solver, and is named accordingly (NO-FAKE #6): it enumerates every
    admissible integer offset and keeps the one maximising per-block agreement with GT.  For a
    lattice this small, exhaustive search IS the exact optimum, so the ceiling it reports is
    tight rather than a heuristic's lower bound.
    """
    nby, nbx = SEG_H // block, SEG_W // block
    # Seed with the ZERO offset so ties resolve to (0,0).  Not cosmetic: reach is identical
    # across tied offsets but a zero-biased field has strictly lower entropy, so a -rmax-biased
    # tie-break would pay real bytes for nothing.  Caught in adversarial review.
    best = (lab == gt).reshape(nby, block, nbx, block).sum(axis=(1, 3)).astype(np.int32)
    best_off = np.zeros((nby, nbx, 2), dtype=np.int8)
    ar_y, ar_x = np.arange(SEG_H), np.arange(SEG_W)
    for dy in range(-rmax, rmax + 1):
        yy = np.clip(ar_y + dy, 0, SEG_H - 1)
        rows = lab[yy, :]
        for dx in range(-rmax, rmax + 1):
            xx = np.clip(ar_x + dx, 0, SEG_W - 1)
            ag = (rows[:, xx] == gt)
            blk = ag.reshape(nby, block, nbx, block).sum(axis=(1, 3)).astype(np.int32)
            upd = blk > best
            if upd.any():
                best[upd] = blk[upd]
                best_off[upd] = (dy, dx)
    return best_off


def price(offs: np.ndarray) -> dict:
    """Real-coder price of the offset field. LZMA1 bound; ph1 measured SMEVR 15-24% better."""
    a = np.ascontiguousarray(offs.astype(np.int8))
    raw = a.tobytes()
    by_lzma = lzma1_raw(raw)
    # ph1's own measured SMEVR advantage over the best generic coder (its 114), applied as a
    # LABELLED projection -- never presented as a measurement of a coder we did not run here.
    return {"lzma1_bytes": by_lzma, "smevr_projected_bytes": round(by_lzma * 0.80),
            "raw_bytes": len(raw)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--rmax", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "et1_ph1_block16_our_vehicle.json"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    n = args.n_pairs

    gt = np.load(os.path.join(ARGMAX_DIR, "gt_argmax_n600.npy"), mmap_mode="r")
    cx = np.load(os.path.join(ARGMAX_DIR, "cx1_argmax_n600.npy"), mmap_mode="r")
    ph_off = np.load(PH1_OFFSETS)[f"block{args.block}"]

    base = xfer = resolved = 0
    solved = np.zeros((n, (SEG_H // args.block) * (SEG_W // args.block), 2), dtype=np.int8)
    t0 = time.time()
    for p in range(n):
        g = np.asarray(gt[p])
        c = np.asarray(cx[p])
        base += int((g != c).sum())
        xfer += int((g != translate_blocks(c, ph_off[p], args.block)).sum())
        so = solve_blocks(c, g, args.block, args.rmax)
        solved[p] = so.reshape(-1, 2)
        resolved += int((g != translate_blocks(c, solved[p], args.block)).sum())
        if (p + 1) % 50 == 0:
            print(f"  {p+1}/{n}  {time.time()-t0:.1f}s", flush=True)

    pr_x = price(ph_off[:n])
    pr_s = price(solved)
    out = {
        "schema": "ddm_et1_ph1_block16_our_vehicle.v1",
        "axis": "[macOS-CPU cache-derived advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n_pairs": n, "block": args.block, "rmax": args.rmax,
        "denominators": {"gap_S": GAP_S, "S_per_flip": S_PER_FLIP,
                         "rate_per_byte": RATE_PER_BYTE},
        "ph1_published": {"baseline_flips_DERIVED": PH1_BASELINE_FLIPS,
                          "baseline_d_seg": PH1_BASELINE_FLIPS * S_PER_FLIP / 100,
                          "gross_S": PH1_BLOCK16_GROSS_S, "bytes": PH1_BLOCK16_BYTES,
                          "net_S": 0.11186, "reach": 0.3989},
        "our_vehicle": {"baseline_flips": base, "baseline_d_seg": base * S_PER_FLIP / 100},
        "transfer_ph1_offsets": {"flips_left": xfer, "reach": (base - xfer) / base,
                                 "gross_S": (base - xfer) * S_PER_FLIP, "price": pr_x},
        "resolved_on_our_field": {"flips_left": resolved, "reach": (base - resolved) / base,
                                  "gross_S": (base - resolved) * S_PER_FLIP, "price": pr_s},
    }
    for k in ("transfer_ph1_offsets", "resolved_on_our_field"):
        b = out[k]["price"]["smevr_projected_bytes"]
        out[k]["rate_S_at_projected_bytes"] = b * RATE_PER_BYTE
        out[k]["net_S_at_eta1"] = b * RATE_PER_BYTE - out[k]["gross_S"]
        out[k]["breakeven_eta"] = (b * RATE_PER_BYTE) / out[k]["gross_S"]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\n=== ph1 block{args.block} on OUR vehicle  (n={n}, DENOMINATOR stated) ===")
    print(f"ph1 baseline (DERIVED from its own arithmetic): {PH1_BASELINE_FLIPS} flips "
          f"= d_seg {PH1_BASELINE_FLIPS*S_PER_FLIP/100:.7f}  [burn/ep399]")
    print(f"OUR live-best baseline                        : {base} flips "
          f"= d_seg {base*S_PER_FLIP/100:.7f}  [cx1, what we ship]")
    print(f"  -> ph1's base has {100*(1-PH1_BASELINE_FLIPS/base):.1f}% FEWER flips to start from\n")
    print(f"{'arm':34s} {'flips left':>10s} {'reach':>8s} {'gross S':>9s} "
          f"{'bytes':>8s} {'net@eta1':>9s} {'BE eta':>7s}")
    print(f"{'ph1 PUBLISHED (its own vehicle)':34s} {275766:10d} {0.3989:8.4f} "
          f"{PH1_BLOCK16_GROSS_S:9.5f} {PH1_BLOCK16_BYTES:8d} {0.04325-PH1_BLOCK16_GROSS_S:+9.5f} "
          f"{0.04325/PH1_BLOCK16_GROSS_S:7.4f}")
    for k, lbl in (("transfer_ph1_offsets", "(A) TRANSFER ph1 offsets -> ours"),
                   ("resolved_on_our_field", "(B) RE-SOLVED on our field")):
        r = out[k]
        print(f"{lbl:34s} {r['flips_left']:10d} {r['reach']:8.4f} {r['gross_S']:9.5f} "
              f"{r['price']['smevr_projected_bytes']:8d} {r['net_S_at_eta1']:+9.5f} "
              f"{r['breakeven_eta']:7.4f}")
    print("\nnet@eta1 NEGATIVE = worth doing IF realization were free. BE eta = the realized")
    print("efficiency the rung needs. Realization is NOT measured here.")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
