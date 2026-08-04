# SPDX-License-Identifier: MIT
"""ddm_fz1 -- the PROVEN frame_0 k=4 DCT pose repair applied to the REAL byte-closed seg base.

TARGET BASE (the #827 composition, cr2_ep854): the byte-closed archive
`v4d_composed_cr2_ep854_archive.zip` (285,529 B, sha 6edf45fa...) whose n600 row is MEASURED by
the real evaluator: seg 0.3944070 S + rate 0.1901220 S + pose 19.4621 S (d_pose 37.877).  Its
delivered frames are on disk (`.../v4d_cr2_ep854/inflated/0.raw`) -- the EXACT frames the n600
evaluator saw.  The raw ep854 TR1 archive (#881) shares the IDENTICAL seg packet (cr2 control:
receiver-rebuilt packet byte-identical, sha f78334bfdd3daff3), so this composition IS the
byte-closed realization of both #827 and #881's stranded seg+rate prize.

PRE-REGISTERED BAR (fixed before any measurement; every ΔS vs LIVE BEST 0.7910689):
    banks  <=>  sqrt(10*d_pose_n600) + RATE_PER_BYTE*B_stream < 0.7910689 - 0.3944070 - 0.1901220
                                                              = 0.2065399
    at the full k=4 stream (57,634 B):  d_pose_n600 < 0.0028287
    at ZERO stream bytes (absolute):    d_pose_n600 < 0.0042659
    FLOOR CLOSURE (selection-immune):   if sum of best-achievable d_pose over the MEASURED pairs
    alone exceeds 600*0.0042659 = 2.5595, the composition cannot bank under ANY stream size or
    solver improvement on the unmeasured pairs (d_pose >= 0 everywhere) -- REJECT is proven by
    arithmetic, uv1-style, with no extrapolation.

WHY THIS IS NOT A RERUN OF cr2r/uv1 (m94 -- a negative measures the instrument).  cr2r/uv1
refuted PARAMETRIC post-hoc carriers on this base (warp/theta/two-plane GN: best 11.59 mean,
~880x over break-even).  The direct frame_0 solve (js1/bz1's proven mechanism: additive DCT
delta on the structurally seg-invisible frame_0, best-iterate ON the int16-quantised synthesis)
is a DIFFERENT instrument that was never pointed at this base.  bz1's fire-order-1 asks exactly
this measurement.  Arms, per pair:

    CENSUS   d_pose of the DELIVERED pair (the damage census bz1 named)
             + the GT-f0 probe: d_pose(PoseNet(GT_f0, delivered_f1)) -- the perfect-photometric-
             frame_0 point, 1 forward, free (mechanism evidence, not a sup bound)
    K4       js1's quantised k=4 DCT solve (96 B/pair counted), lr ladder, camera re-scored
    ORACLE   instrument-capacity in the claim's units: free-form frame_0 solve (unpriceable,
             589,824 DOF) + k=16 DCT, lr ladders -- if even the best of ALL arms stays orders
             above the bar on every pair, the frame_0-repair FAMILY on this base is closed by
             the floor arithmetic, not just k=4.

Seg-exactness is BY CONSTRUCTION (SegNet reads x[:,-1] only) and ASSERTED per pair anyway.
GT decode ONLY via frame_utils.yuv420_to_rgb (canonical).  No scorer at inflate is implied:
the counted object is int16 coefficients; the solve is ENCODE-side only.

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.  score_claim=False.  Only
upstream/evaluate.py on the exact shipped bytes is a score.
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

from ddm_js1_staging_discriminator import (
    CAM_H,
    CAM_W,
    N_PAIRS_TOTAL,
    RATE_PER_BYTE,
    Scorer,
    _assert_private_support,
    decode_gt_frames,
    patch_yuv6_and_assert,
    realize_full_frame,
    seq_len,
    solve_pose_repair_frame0,
    solve_pose_repair_frame0_cheap_dct,
)

# ---- pre-registered denominators (fixed BEFORE measurement; sources cited) --------------------
LIVE_BEST_S = 0.7910689                # pu2 @ 353,805 B [macOS-CPU advisory]
BASE_SEG_S = 0.3944070                 # cr2_ep854 report.txt, n600 MEASURED (0.00394407*100)
BASE_RATE_S = 0.1901220                # 25*285,529/37,545,489 (n600 report)
HEADROOM_S = LIVE_BEST_S - BASE_SEG_S - BASE_RATE_S          # 0.2065399
K4_STREAM_BYTES = 600 * 96 + 8 + 9 + 4 * 600                 # coefs + header + pair index
D_POSE_BAR_WITH_STREAM = ((HEADROOM_S - RATE_PER_BYTE * K4_STREAM_BYTES) ** 2) / 10.0
D_POSE_BAR_ZERO_BYTES = (HEADROOM_S ** 2) / 10.0             # 0.0042659...
FLOOR_BUDGET_TOTAL = 600.0 * D_POSE_BAR_ZERO_BYTES           # 2.5595...


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", type=Path, required=True,
                    help="evaluated submission dir containing inflated/0.raw (cr2_ep854)")
    ap.add_argument("--gt-mkv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", type=str, required=True, help="comma-separated pair ids")
    ap.add_argument("--oracle-pairs", type=str, default="",
                    help="subset of --pairs that also get free-form + k16 oracle arms")
    ap.add_argument("--k4-lrs", type=str, default="20,200")
    ap.add_argument("--oracle-lrs", type=str, default="2,8")
    ap.add_argument("--k16-lrs", type=str, default="20,200")
    ap.add_argument("--pose-steps", type=int, default=60)
    ap.add_argument("--oracle-steps", type=int, default=80)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    pairs = [int(x) for x in args.pairs.split(",")]
    oracle_pairs = {int(x) for x in args.oracle_pairs.split(",")} if args.oracle_pairs else set()
    k4_lrs = [float(x) for x in args.k4_lrs.split(",")]
    oracle_lrs = [float(x) for x in args.oracle_lrs.split(",")]
    k16_lrs = [float(x) for x in args.k16_lrs.split(",")]

    geom = _assert_private_support()
    raw = np.memmap(args.sub_dir / "inflated" / "0.raw", dtype=np.uint8, mode="r",
                    shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3))

    wanted = set()
    for p in pairs:
        wanted.update({seq_len * p, seq_len * p + 1})
    gt_frames = decode_gt_frames(args.gt_mkv, wanted)
    sc = Scorer(args.threads)
    patch = patch_yuv6_and_assert(sc)
    print(f"[fz1] scorer+patch ready t={time.time()-t0:.1f}s  {patch}", flush=True)

    rows: list[dict] = []
    if args.resume and args.out.exists():
        rows = json.loads(args.out.read_text()).get("rows", [])
        done = {int(r["pair"]) for r in rows}
        pairs = [p for p in pairs if p not in done]
        print(f"[fz1] resume: {len(rows)} on disk, {len(pairs)} left", flush=True)

    def flush() -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        best_sum = sum(r["best_achievable_d_pose"] for r in rows)
        with open(args.out, "w") as f:
            json.dump({
                "schema": "ddm_fz1_frame0_repair_real_base.v1",
                "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
                "score_claim": False, "promotion_eligible": False,
                "base": {
                    "archive": "v4d_composed_cr2_ep854_archive.zip",
                    "bytes": 285529,
                    "sha256_prefix": "6edf45fa5052949d",
                    "n600_seg_S": BASE_SEG_S, "n600_rate_S": BASE_RATE_S,
                    "n600_d_pose_delivered": 37.87713242,
                },
                "pre_registered_bar": {
                    "live_best_S": LIVE_BEST_S,
                    "headroom_S": HEADROOM_S,
                    "k4_stream_bytes": K4_STREAM_BYTES,
                    "d_pose_bar_with_k4_stream": D_POSE_BAR_WITH_STREAM,
                    "d_pose_bar_zero_bytes": D_POSE_BAR_ZERO_BYTES,
                    "floor_budget_total_n600": FLOOR_BUDGET_TOTAL,
                    "rule": "REJECT proven if sum(best_achievable_d_pose over measured pairs) "
                            "> floor_budget_total_n600 (d_pose>=0 on unmeasured pairs; "
                            "selection-immune)",
                },
                "floor_closure": {
                    "sum_best_achievable_measured": best_sum,
                    "budget": FLOOR_BUDGET_TOTAL,
                    "exceeded": bool(best_sum > FLOOR_BUDGET_TOTAL),
                    "n_pairs_measured": len(rows),
                },
                "yuv6_patch": patch, "D_geometry": geom,
                "rows": rows}, f, indent=1)

    for n, p in enumerate(pairs):
        tp = time.time()
        dec = np.stack([raw[seq_len * p], raw[seq_len * p + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * p], gt_frames[seq_len * p + 1]])
        pose_gt = sc.pose_out(gt)
        lstar = sc.seg_argmax(dec)
        lgt = sc.seg_argmax(gt)
        flips = int((lstar != lgt).sum())

        # ---- CENSUS ----------------------------------------------------------------------
        dp_delivered = sc.d_pose(pose_gt, sc.pose_out(dec))
        # GT-f0 probe: perfect photometric frame_0 against the delivered (palette) frame_1
        dp_gtf0 = sc.d_pose(pose_gt, sc.pose_out(np.stack([gt[0], dec[1]])))

        rec: dict = {"pair": int(p), "flips_delivered": flips,
                     "d_pose_delivered": dp_delivered,
                     "d_pose_gt_f0_probe": dp_gtf0}
        arms: dict[str, dict] = {}

        # ---- K4 (the priced mechanism, 96 B/pair) ------------------------------------------
        best_k4 = None
        for lr in k4_lrs:
            dp_s, paint, tag, counted, cmax = solve_pose_repair_frame0_cheap_dct(
                sc, dec[0], dec[1], pose_gt, k=4, steps=args.pose_steps, lr=lr,
                eval_every=args.eval_every)
            if paint is not None:
                pair_f = np.stack([realize_full_frame(dec[0], paint), dec[1]])
                dp_cam = sc.d_pose(pose_gt, sc.pose_out(pair_f))
                seg_ok = bool((sc.seg_argmax(pair_f) == lstar).all())
            else:
                dp_cam, seg_ok = dp_delivered, True
            arm = {"lr": lr, "tag": tag, "d_pose_scorer": dp_s, "d_pose_camera": dp_cam,
                   "counted_bytes": counted, "coef_max_abs": cmax, "seg_exact": seg_ok,
                   "all_zero": paint is None}
            if best_k4 is None or dp_cam < best_k4["d_pose_camera"]:
                best_k4 = arm
            arms[f"k4_lr{lr:g}"] = arm
        rec["k4_best"] = best_k4

        # ---- ORACLE (instrument capacity, unpriceable) --------------------------------------
        best_oracle = dict(best_k4)
        best_oracle["arm"] = "k4"
        if p in oracle_pairs:
            for lr in k16_lrs:
                dp_s, paint, tag, counted, cmax = solve_pose_repair_frame0_cheap_dct(
                    sc, dec[0], dec[1], pose_gt, k=16, steps=args.pose_steps, lr=lr,
                    eval_every=args.eval_every)
                if paint is not None:
                    pair_f = np.stack([realize_full_frame(dec[0], paint), dec[1]])
                    dp_cam = sc.d_pose(pose_gt, sc.pose_out(pair_f))
                else:
                    dp_cam = dp_delivered
                arms[f"k16_lr{lr:g}"] = {"lr": lr, "tag": tag, "d_pose_camera": dp_cam,
                                         "counted_bytes": counted, "coef_max_abs": cmax}
                if dp_cam < best_oracle["d_pose_camera"]:
                    best_oracle = {"arm": "k16", "lr": lr, "d_pose_camera": dp_cam,
                                   "tag": tag}
            for lr in oracle_lrs:
                dp_s, paint, tag = solve_pose_repair_frame0(
                    sc, dec[0], dec[1], pose_gt, steps=args.oracle_steps, lr=lr,
                    eval_every=args.eval_every, linf=0.0)
                if str(tag).startswith("identity"):
                    # js1's identity paint placeholder is zeros, not the base frame -- do not
                    # realize it (it would score a black frame_0); identity == delivered.
                    dp_cam = dp_delivered
                else:
                    pair_f = np.stack([realize_full_frame(dec[0], paint), dec[1]])
                    dp_cam = sc.d_pose(pose_gt, sc.pose_out(pair_f))
                arms[f"free_lr{lr:g}"] = {"lr": lr, "tag": tag, "d_pose_scorer": dp_s,
                                          "d_pose_camera": dp_cam}
                if dp_cam < best_oracle["d_pose_camera"]:
                    best_oracle = {"arm": "free", "lr": lr, "d_pose_camera": dp_cam,
                                   "tag": tag}
        rec["arms"] = arms
        rec["oracle_ran"] = p in oracle_pairs
        rec["best_achievable_d_pose"] = float(best_oracle["d_pose_camera"])
        rec["best_arm"] = best_oracle.get("arm", "k4")
        rec["k4_ratio_vs_bar"] = best_k4["d_pose_camera"] / D_POSE_BAR_WITH_STREAM
        rows.append(rec)
        flush()
        print(f"[fz1] pair {p:3d} ({n+1}/{len(pairs)}) delivered {dp_delivered:9.4f} "
              f"gtf0 {dp_gtf0:9.4f} | k4 {best_k4['d_pose_camera']:9.4f} "
              f"({best_k4['tag']}) | best {rec['best_achievable_d_pose']:9.4f} "
              f"[{rec['best_arm']}] | bar {D_POSE_BAR_WITH_STREAM:.6f} "
              f"[{time.time()-tp:.1f}s]", flush=True)

    flush()
    print(f"[fz1] done {len(rows)} rows t={time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
