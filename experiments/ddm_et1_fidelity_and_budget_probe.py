#!/usr/bin/env python
"""ddm_et1 -- two owed corrections in one run: band-restricted fidelity, and eta-vs-budget.

(3) FIDELITY.  `ddm_et1_block16_realization.py` reports `target_fidelity` over the WHOLE frame,
    where it is dominated by the ~97% of pixels the band never touches.  0.9984 whole-frame is
    therefore not evidence that the realizer reproduced the target field.  The meaningful
    quantity is fidelity RESTRICTED TO THE BAND -- the pixels the description actually asks to
    move -- plus the off-band collateral rate.  Both are computed here.

(4) BUDGET / CONVERGENCE.  Two different solvers are involved and only one of them has a
    convergence question at all:

      * the OFFSET search (`solve_blocks`) is EXHAUSTIVE over the (2*rmax+1)^2 integer lattice.
        It has no iteration and no stopping rule: it evaluates every admissible offset and keeps
        the argmax.  It is therefore EXACT, not converged-to -- the sm1/#874 class cannot apply.
      * the PAINT solve (`solve_margin_optimal_paint`, inherited from sq1) is Adam on a
        continuous delta with a STEP CAP and best-realized-iterate retention.  This one is
        cap-pinned in every run so far, so its eta is a FLOOR, not an optimum.

    Per #874 the answer is not to raise the cap and quote the new number -- it is to MEASURE the
    budget response.  This runs the same pair at 3 budgets so the slope is visible.  A budget
    ladder that is still rising at the top is reported as STILL-RISING (a floor), never as
    converged.

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
from ddm_sq1_eta_seg_realization import N_PAIRS_TOTAL, Scorer, decode_gt_frames, seq_len
from ddm_sq1_stage_decomposition_and_solved_paint import (
    realize_scorer_paint_to_camera,
    solve_margin_optimal_paint,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", type=Path, required=True)
    ap.add_argument("--gt-mkv", type=Path, required=True)
    ap.add_argument("--argmax-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--budgets", type=int, nargs="+", default=[10, 25, 50])
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--rmax", type=int, default=5)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    t0 = time.time()
    raw = np.memmap(args.sub_dir / "inflated" / "0.raw", dtype=np.uint8, mode="r",
                    shape=(N_PAIRS_TOTAL * seq_len, 874, 1164, 3))
    p = args.pair
    gt_frames = decode_gt_frames(args.gt_mkv, {seq_len * p, seq_len * p + 1})
    dec = np.stack([raw[seq_len * p], raw[seq_len * p + 1]]).astype(np.uint8)
    gt = np.stack([gt_frames[seq_len * p], gt_frames[seq_len * p + 1]])

    sc = Scorer(args.threads)
    lstar = sc.seg_argmax(dec)
    lgt = sc.seg_argmax(gt)
    flips0_map = lstar != lgt
    flips0 = int(flips0_map.sum())

    off = solve_blocks(lstar, lgt, args.block, args.rmax)
    target = translate_blocks(lstar, off.reshape(-1, 2), args.block)
    band = target != lstar
    nd = flips0 - int((target != lgt).sum())
    pose_gt = sc.pose_out(gt)
    dp0 = sc.d_pose(pose_gt, sc.pose_out(dec))

    print(f"[fb] pair {p}  flips {flips0}  band_px {int(band.sum())}  ceiling {nd}  "
          f"t={time.time()-t0:.1f}s", flush=True)

    rows = []
    for steps in args.budgets:
        ts = time.time()
        _, paint, tag = solve_margin_optimal_paint(
            sc.net.segnet, dec[1], gt[1], band, target, steps=steps, lr=2.0, eval_every=5)
        edited = realize_scorer_paint_to_camera(dec[1], band, paint)
        pe = np.stack([dec[0], edited])
        lam = sc.seg_argmax(pe)
        fa = int((lam != lgt).sum())
        it = int(str(tag).rsplit("@", 1)[1])
        r = {
            "steps": steps, "tag": tag, "best_iterate": it,
            "cap_pinned": it >= steps,
            "flips_after": fa, "eta_realized": (flips0 - fa) / nd if nd else None,
            # (3) the CORRECTED metric, and the misleading one kept beside it for the record
            "fidelity_IN_BAND": float((lam[band] == target[band]).mean()),
            "fidelity_whole_frame_MISLEADING": float((lam == target).mean()),
            "offband_collateral_rate": float((lam[~band] != lstar[~band]).mean()),
            "d_pose_ratio": sc.d_pose(pose_gt, sc.pose_out(pe)) / dp0,
        }
        rows.append(r)
        print(f"[fb] steps {steps:3d} -> eta {r['eta_realized']:+.4f} @{tag} "
              f"pinned={r['cap_pinned']} | fid_in_band {r['fidelity_IN_BAND']:.4f} "
              f"(whole-frame {r['fidelity_whole_frame_MISLEADING']:.4f}) | "
              f"offband_collateral {r['offband_collateral_rate']:.5f} | "
              f"dpose {r['d_pose_ratio']:.3f}x [{time.time()-ts:.1f}s]", flush=True)

    etas = [r["eta_realized"] for r in rows]
    still_rising = len(etas) > 1 and etas[-1] > etas[-2]
    verdict = ("STILL-RISING at the top budget -- every eta here is a FLOOR, not an optimum"
               if still_rising else "PLATEAUED between the top two budgets")
    print(f"\n[fb] BUDGET RESPONSE: {verdict}")
    print("[fb] offset search is EXHAUSTIVE (exact); only the paint solve has a budget.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema": "ddm_et1_fidelity_budget.v1",
        "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "pair": p, "n_pairs_DENOMINATOR": 1,
        "caveat": "n=1 -- a budget SLOPE probe, not a population eta",
        "offset_search": "EXHAUSTIVE over (2*rmax+1)^2 -- exact, no stopping rule",
        "paint_solver": "Adam + best-realized-iterate, STEP-CAPPED (sm1/#874 class)",
        "budget_verdict": verdict, "rows": rows}, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
