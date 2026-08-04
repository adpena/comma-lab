#!/usr/bin/env python
"""ddm_et1 Job 3 -- REALIZED eta on ph1's regional phase field (block16), our vehicle.

ph1 measured a reach CEILING in label space: translate the argmax field per 16x16 block and it
agrees with GT far better.  Job 2 re-measured that on OUR field (re-solved offsets: reach
41.84%, gross 0.18039 S, break-even eta 0.2279).  Neither is a score, because both assume the
translated LABEL FIELD can be produced as RGB the frozen SegNet actually argmaxes that way.

This script measures that step -- the same instrument as Job 1, pointed at a different field.

    target  = translate_blocks(L*, resolved_block16_offsets)     [NOT GT: this is what the
                                                                  ~62 KB actually buys]
    band    = {p : target(p) != L*(p)}                           [only pixels the field moves]
    realizer= solve-from-frozen-head, multi-start, best-realized-iterate  (arm (b) machinery)

    eta_net = (flips_before - flips_after) / n_described
    n_described = flips the translated field would FIX if realized exactly (= its reach mass)

Truth-paint is deliberately NOT used: sq1 measured it at eta -3.7640 through the real receiver
(inserting true texture into a thin region makes the net hallucinate large-area classes).  The
-3.764 lesson is the reason this arm solves rather than pastes.

Pose collateral is mandatory and reported per pair: frame_1 is read by BOTH scorers (pz1 -- they
make the identical interpolate call), so a seg-only verdict is forbidden.

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.  score_claim=false.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from ddm_et1_ph1_block16_on_our_vehicle import solve_blocks, translate_blocks
from ddm_sq1_eta_seg_realization import (
    N_PAIRS_TOTAL,
    SEG_H,
    SEG_W,
    Scorer,
    _assert_private_support,
    decode_gt_frames,
    seq_len,
)
from ddm_sq1_stage_decomposition_and_solved_paint import (
    realize_scorer_paint_to_camera,
    solve_margin_optimal_paint,
)

S_PER_FLIP = 100.0 / (N_PAIRS_TOTAL * SEG_H * SEG_W)
RATE_PER_BYTE = 25.0 / 37_545_489.0
GAP_S = 0.7910689 - 0.172141


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", type=Path, required=True)
    ap.add_argument("--gt-mkv", type=Path, required=True)
    ap.add_argument("--pairs-npy", type=Path, required=True)
    ap.add_argument("--argmax-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--rmax", type=int, default=5)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    pairs = np.load(args.pairs_npy).tolist()
    if args.limit:
        pairs = pairs[: args.limit]
    geom = _assert_private_support()
    print(f"[b16] {len(pairs)} pairs (matched to sq1)  block{args.block} rmax{args.rmax}",
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
    print(f"[b16] scorer ready t={time.time()-t0:.1f}s", flush=True)

    rows = []
    for n, p in enumerate(pairs):
        tp = time.time()
        dec = np.stack([raw[seq_len * p], raw[seq_len * p + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * p], gt_frames[seq_len * p + 1]])
        lstar = sc.seg_argmax(dec)
        lgt = sc.seg_argmax(gt)
        rec = {"pair": int(p),
               "C2_lstar_matches_cache": bool((lstar == np.asarray(cx1[p])).all()),
               "C3_lgt_matches_cache": bool((lgt == np.asarray(gtc[p])).all())}

        flips0_map = lstar != lgt
        flips0 = int(flips0_map.sum())
        rec["flips_before"] = flips0

        off = solve_blocks(lstar, lgt, args.block, args.rmax)
        target = translate_blocks(lstar, off.reshape(-1, 2), args.block)
        band = target != lstar
        # the CEILING this pair's field buys, in label space, if realized exactly
        fixed_ceiling = int((flips0_map & (target == lgt)).sum())
        broken_ceiling = int((~flips0_map & (target != lgt)).sum())
        rec["band_px"] = int(band.sum())
        rec["band_frac"] = float(band.mean())
        rec["label_ceiling_flips_left"] = int((target != lgt).sum())
        rec["label_ceiling_net_fixed"] = flips0 - int((target != lgt).sum())
        rec["label_ceiling_fixed"] = fixed_ceiling
        rec["label_ceiling_broken"] = broken_ceiling

        pose_gt = sc.pose_out(gt)
        rec["d_pose_before"] = sc.d_pose(pose_gt, sc.pose_out(dec))

        # ---- realize the TRANSLATED FIELD (target), not GT --------------------------------
        nb, paint, tag = solve_margin_optimal_paint(
            sc.net.segnet, dec[1], gt[1], band, target,
            steps=args.steps, lr=args.lr, eval_every=args.eval_every)
        edited = realize_scorer_paint_to_camera(dec[1], band, paint)
        pair_e = np.stack([dec[0], edited])
        lam = sc.seg_argmax(pair_e)
        fa = int((lam != lgt).sum())
        rec["solve_tag"] = tag
        rec["cap_pinned"] = bool(str(tag).rsplit("@", 1)[-1] == str(args.steps))
        rec["flips_after"] = fa
        # denominator = the ceiling mass the ~62 KB field buys (net label-space reduction)
        nd = rec["label_ceiling_net_fixed"]
        rec["n_described"] = nd
        rec["eta_realized"] = ((flips0 - fa) / nd) if nd else None
        # how faithfully did the RGB reproduce the target field itself?
        rec["target_fidelity"] = float((lam == target).mean())
        rec["d_pose_after"] = sc.d_pose(pose_gt, sc.pose_out(pair_e))
        rec["d_pose_ratio"] = rec["d_pose_after"] / rec["d_pose_before"]

        rows.append(rec)
        e = rec["eta_realized"]
        print(f"[b16] pair {p:3d} ({n+1}/{len(pairs)}) flips {flips0:5d} "
              f"ceiling {nd:5d} after {fa:5d} | eta_realized "
              f"{f'{e:+.4f}' if e is not None else '  n/a'} @{tag} "
              f"| tgt_fid {rec['target_fidelity']:.4f} "
              f"| dpose {rec['d_pose_ratio']:.3f}x [{time.time()-tp:.1f}s]", flush=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"schema": "ddm_et1_block16_realization.v1",
                       "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
                       "score_claim": False, "promotion_eligible": False,
                       "pointer": "0.1910828242 [contest-CPU] UNMOVED",
                       "block": args.block, "rmax": args.rmax,
                       "budget": {"steps": args.steps, "lr": args.lr,
                                  "eval_every": args.eval_every},
                       "denominators": {"gap_S": GAP_S, "S_per_flip": S_PER_FLIP,
                                        "rate_per_byte": RATE_PER_BYTE},
                       "D_geometry": geom, "pairs": pairs, "rows": rows}, f, indent=1)

    print(f"[b16] DONE {len(rows)} t={time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
