#!/usr/bin/env python
"""ddm_et1 Job 1 -- eta on the band that is actually PRICED, matched pair-for-pair to sq1.

sq1 measured eta on `dilate_sq1(boundary(lstar), 1)`.  gp1 priced row A3 on
`dilate_gp1(boundary(seed), 1)`.  Job 0 (`ddm_et1_band_convention_recalibration.py`, n600)
established these are different objects:

    SE footprint   gp1 = 5 px (L1/von-Neumann)   sq1 = 11 px (L1 + vertical)   2.200x
    realized band  gp1 = 3.619% of field         sq1 = 5.143%                  1.299x
    bytes          gp1 = 331,824 B               sq1 = 369,414 B               1.113x
    capture        gp1 = 83.334%                 sq1 = 86.701%                 +3.37 pp

So sq1's eta = 0.7895 / 0.5406 does NOT transfer to gp1's priced address.  This script
measures eta on gp1's SE, on the SAME 32 pairs sq1 used, with the SAME solver budgets --
a PAIRED A/B in which the structuring element is the only thing that changes.

Three realizer arms (LAW B / m95: the level is per-ROLE, so each arm is a different role):
    (a) v0     truth paint            -- the content control (sq1: -3.7640)
    (b) solved unconstrained          -- steps=25, 2 starts   (sq1 Job 1b: +0.7895)
    (c) solved + Q3 pose-null         -- steps=15, 1 start    (sq1 Job 1c: +0.5406)

Also folds in sm1/#935: sq1's solve stopped AT its step cap on 31/32 pairs.  Every arm here
records the iterate index its best came from, so cap-pinning is MEASURED rather than assumed.
Per the #874 lesson the cap is NOT blindly raised -- `--extended-steps` runs a separate,
explicitly-labelled headroom probe.

And it discharges sq1 §2.8's snap-tax debt: arm (c)'s band is snapped to whole 2x2 blocks,
so the SNAPPED band is priced directly (real LZMA1) rather than left as an IOU.

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.  score_claim=false.
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from ddm_sq1_eta_seg_realization import (
    N_PAIRS_TOTAL,
    SEG_H,
    SEG_W,
    Scorer,
    _assert_private_support,
    boundary,
    decode_gt_frames,
    paste_truth,
    seq_len,
)
from ddm_sq1_eta_seg_realization import dilate as dilate_sq1
from ddm_sq1_pose_null_constrained_paint import (
    pose_null_projector,
    snap_band_to_blocks,
    solve_null_constrained,
)
from ddm_sq1_stage_decomposition_and_solved_paint import (
    realize_scorer_paint_to_camera,
    solve_margin_optimal_paint,
)

RATE_PER_BYTE = 25.0 / 37_545_489.0
LIVE_BEST_S = 0.7910689          # pu2, [macOS-CPU advisory]; sq1 used a STALE 0.826496
PR130_FLOOR_S = 0.172141
GAP_S = LIVE_BEST_S - PR130_FLOOR_S
BASE_D_POSE = 0.0025513987495742437
S_PER_FLIP = 100.0 / (N_PAIRS_TOTAL * SEG_H * SEG_W)
DS_DDPOSE = 5.0 / math.sqrt(10.0 * BASE_D_POSE)   # CURRENT slope (K3), never a shelf price
H_GT_GIVEN_REND = 1.1011521270613085


def dilate_gp1(mask: np.ndarray, r: int) -> np.ndarray:
    """gp1's ACTUAL convention (`ddm_gp1_free_band_and_net.py:77`): L1 ball, SE 5 px at r=1."""
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
    filt = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}]
    return len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=filt))


def band_price_bits(band: np.ndarray, flips: np.ndarray) -> tuple:
    """gp1's own byte model, applied to whatever band it is handed. (addr_bits, pay_bits)."""
    addr = lzma1_raw(np.packbits(flips[band].astype(np.uint8)).tobytes()) * 8
    return addr, int((flips & band).sum()) * H_GT_GIVEN_REND


def score_arm(sc, rec: dict, tag: str, *, dec, lgt, flips0_map, flips0, pose_gt,
              edited_f1, described, pose: bool = True) -> None:
    """Score one realization arm through the REAL scorer and write its row fields.

    Module-level (not a closure over the pair loop) so every dependency is an explicit
    argument -- the B023 class, and the reason sq1's per-pair state could not be audited
    from the call site alone.

    `eta_net` divides the WHOLE-FRAME flip reduction by the flips the description ADDRESSES,
    so collateral damage outside the band is charged against the arm; `eta_inband_raw` ignores
    collateral, and the gap between the two IS the collateral.
    """
    pair_e = np.stack([dec[0], edited_f1])
    lam = sc.seg_argmax(pair_e)
    fa = int((lam != lgt).sum())
    nd = int((flips0_map & described).sum())
    rec[f"{tag}_flips_after"] = fa
    rec[f"{tag}_n_described"] = nd
    rec[f"{tag}_eta_net"] = ((flips0 - fa) / nd) if nd else None
    rec[f"{tag}_eta_inband_raw"] = (
        int((flips0_map & described & (lam == lgt)).sum()) / nd) if nd else None
    if pose:
        rec[f"{tag}_d_pose_after"] = sc.d_pose(pose_gt, sc.pose_out(pair_e))


def iterate_of(tag: str) -> int:
    """sq1's solvers tag the retained iterate '<start>@<it>'.  -1 if unparseable."""
    try:
        return int(str(tag).rsplit("@", 1)[1])
    except (IndexError, ValueError):
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", type=Path, required=True)
    ap.add_argument("--gt-mkv", type=Path, required=True)
    ap.add_argument("--pairs-npy", type=Path, required=True)
    ap.add_argument("--argmax-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--uncon-steps", type=int, default=25)   # sq1 Job 1b budget
    ap.add_argument("--null-steps", type=int, default=15)    # sq1 Job 1c budget
    ap.add_argument("--extended-steps", type=int, default=0,
                    help="headroom probe ONLY -- a separate labelled arm, never the verdict")
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    pairs = np.load(args.pairs_npy).tolist()
    if args.limit:
        pairs = pairs[: args.limit]
    geom = _assert_private_support()
    print(f"[et1] {len(pairs)} pairs (matched to sq1)  D-blind {geom['frac_blind_to_scorers']:.6f}",
          flush=True)

    raw = np.memmap(args.sub_dir / "inflated" / "0.raw", dtype=np.uint8, mode="r",
                    shape=(N_PAIRS_TOTAL * seq_len, 874, 1164, 3))
    cx1 = np.load(args.argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    gtc = np.load(args.argmax_cache / "gt_argmax_n600.npy", mmap_mode="r")

    wanted = set()
    for p in pairs:
        wanted.update({seq_len * p, seq_len * p + 1})
    gt_frames = decode_gt_frames(args.gt_mkv, wanted)
    sc = Scorer(args.threads)
    P = pose_null_projector()
    print(f"[et1] scorer + rank-6 null projector ready t={time.time()-t0:.1f}s", flush=True)

    rows = []
    for n, p in enumerate(pairs):
        tp = time.time()
        dec = np.stack([raw[seq_len * p], raw[seq_len * p + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * p], gt_frames[seq_len * p + 1]])

        lstar = sc.seg_argmax(dec)
        lgt = sc.seg_argmax(gt)
        rec = {
            "pair": int(p),
            "C2_lstar_matches_cache": bool((lstar == np.asarray(cx1[p])).all()),
            "C3_lgt_matches_cache": bool((lgt == np.asarray(gtc[p])).all()),
        }
        flips0_map = lstar != lgt
        flips0 = int(flips0_map.sum())
        rec["flips_before"] = flips0
        pose_gt = sc.pose_out(gt)
        rec["d_pose_before"] = sc.d_pose(pose_gt, sc.pose_out(dec))

        bnd = boundary(lstar)
        band_gp1 = dilate_gp1(bnd, 1)          # the PRICED SE -- charter's corrected band
        band_sq1 = dilate_sq1(bnd, 1)          # sq1's SE -- reproduction control
        band_snap = snap_band_to_blocks(band_gp1)   # what arm (c) actually actuates

        for nm, b in (("gp1", band_gp1), ("sq1", band_sq1), ("gp1snap", band_snap)):
            a_bits, p_bits = band_price_bits(b, flips0_map)
            rec[f"band_{nm}_px"] = int(b.sum())
            rec[f"band_{nm}_frac"] = float(b.mean())
            rec[f"band_{nm}_capture"] = float((flips0_map & b).sum() / max(flips0, 1))
            rec[f"band_{nm}_addr_bits"] = a_bits
            rec[f"band_{nm}_pay_bits"] = p_bits

        def score(tag, edited_f1, described, pose=True, _r=rec, _d=dec, _l=lgt,
                  _fm=flips0_map, _f0=flips0, _pg=pose_gt):
            score_arm(sc, _r, tag, dec=_d, lgt=_l, flips0_map=_fm, flips0=_f0, pose_gt=_pg,
                      edited_f1=edited_f1, described=described, pose=pose)

        # ---- arm (a): the content control, on the PRICED SE --------------------------------
        score("a_truth_gp1", paste_truth(dec[1], gt[1], band_gp1), band_gp1)

        # ---- arm (b): solved unconstrained, sq1 Job-1b budget, on the PRICED SE ------------
        nb, paint, tag = solve_margin_optimal_paint(
            sc.net.segnet, dec[1], gt[1], band_gp1, lgt,
            steps=args.uncon_steps, lr=args.lr, eval_every=args.eval_every)
        rec["b_solved_gp1_tag"] = tag
        rec["b_solved_gp1_iterate"] = iterate_of(tag)
        rec["b_solved_gp1_cap_pinned"] = bool(iterate_of(tag) >= args.uncon_steps)
        score("b_solved_gp1", realize_scorer_paint_to_camera(dec[1], band_gp1, paint), band_gp1)

        # ---- arm (c): solved + Q3 pose-null, sq1 Job-1c budget, on the PRICED SE -----------
        nc, paint_c, tag_c = solve_null_constrained(
            sc.net.segnet, dec[1], band_gp1, lgt, P,
            steps=args.null_steps, lr=args.lr, eval_every=args.eval_every)
        rec["c_null_gp1_tag"] = tag_c
        rec["c_null_gp1_iterate"] = iterate_of(tag_c)
        rec["c_null_gp1_cap_pinned"] = bool(iterate_of(tag_c) >= args.null_steps)
        # arm (c) actuates the SNAPPED band, so eta is denominated on what it actually touched
        score("c_null_gp1", realize_scorer_paint_to_camera(dec[1], band_snap, paint_c), band_snap)

        # ---- headroom probe (labelled, never the verdict) ---------------------------------
        if args.extended_steps:
            ne, paint_e, tag_e = solve_margin_optimal_paint(
                sc.net.segnet, dec[1], gt[1], band_gp1, lgt,
                steps=args.extended_steps, lr=args.lr, eval_every=args.eval_every)
            rec["x_extended_tag"] = tag_e
            rec["x_extended_iterate"] = iterate_of(tag_e)
            rec["x_extended_cap_pinned"] = bool(iterate_of(tag_e) >= args.extended_steps)
            score("x_extended", realize_scorer_paint_to_camera(dec[1], band_gp1, paint_e),
                  band_gp1, pose=False)

        rows.append(rec)
        print(f"[et1] pair {p:3d} ({n+1}/{len(pairs)}) flips {flips0:5d} | "
              f"a {rec['a_truth_gp1_eta_net']:+.4f} "
              f"b {rec['b_solved_gp1_eta_net']:+.4f}@{rec['b_solved_gp1_iterate']} "
              f"c {rec['c_null_gp1_eta_net']:+.4f}@{rec['c_null_gp1_iterate']} "
              f"| dpose {rec['d_pose_before']:.6f}->{rec['c_null_gp1_d_pose_after']:.6f} "
              f"[{time.time()-tp:.1f}s]", flush=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"schema": "ddm_et1_eta_priced_band.v1",
                       "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
                       "score_claim": False, "promotion_eligible": False,
                       "pointer": "0.1910828242 [contest-CPU] UNMOVED",
                       "matched_to": "ddm_sq1 32-pair stratified systematic selection",
                       "budgets": {"uncon_steps": args.uncon_steps,
                                   "null_steps": args.null_steps,
                                   "extended_steps": args.extended_steps,
                                   "lr": args.lr, "eval_every": args.eval_every},
                       "denominators": {"live_best_S": LIVE_BEST_S, "gap_S": GAP_S,
                                        "S_per_flip": S_PER_FLIP,
                                        "dS_ddpose_current": DS_DDPOSE,
                                        "rate_per_byte": RATE_PER_BYTE,
                                        "stale_gap_used_by_sq1": 0.654355209256714},
                       "D_geometry": geom, "pairs": pairs, "rows": rows}, f, indent=1)

    print(f"[et1] DONE {len(rows)} pairs t={time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
