#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""#330 measurement harness — does the parent RSS return to baseline after a verdict?

Runs the REAL chunked CPU-torch scorer forward (SegNet argmax + PoseNet MSE) over REAL gt-cache
frames and measures RSS before / during / after for THREE paths, so the honest simplest-fix-wins
comparison is a MEASUREMENT, not an assertion:

    A. in-process, NO reclaim         (the current ratchet)
    B. in-process, + reclaim_process_memory()   (cheap: gc + malloc_trim/pressure_relief + torch cache)
    C. subprocess (killpg-reclaimed child)       (parent never sees the transient)

If B returns the pages, the subprocess (C) is unnecessary complexity. Governor-gated: refuses to run
under warn/critical macOS memory pressure (heavy real torch forwards). Reduced-n by default (RSS-return
is n-independent in mechanism); pass --num-pairs 600 beside a quiet run only if the envelope allows.

Advisory / NON-PROMOTABLE (never a score). Emits a human table + a JSON block.

Usage:
    .venv/bin/python tools/measure_verdict_memory_reclaim.py \
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n24.npz --num-pairs 24 --verdict-batch 8
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.witness_control.verdict_reclaim import (  # noqa: E402
    reclaim_process_memory,
    rss_gib,
    run_verdict_in_subprocess,
)


def _governor_ok() -> tuple[bool, str]:
    try:
        import system_memory_governor as gov

        lvl = int(gov.read_memory_pressure_level())
    except Exception as exc:
        return False, f"pressure_unreadable:{type(exc).__name__}"
    return (lvl == 1, "normal" if lvl == 1 else f"pressure_level_{lvl}")


def _inprocess_verdict(seg_cpu, posenet_cpu, f0s, f1s, lstars, poses, vbatch: int) -> tuple[float, float]:
    from train_witness_realized_through_R_mlx import (
        cpu_verdict_d_pose_batch,
        cpu_verdict_d_seg_batch,
    )
    n = len(f1s)
    vb = n if (vbatch <= 0 or vbatch >= n) else vbatch
    ds: list[float] = []
    dp: list[float] = []
    for s in range(0, n, vb):
        e = min(s + vb, n)
        ds.extend(cpu_verdict_d_seg_batch(seg_cpu, f1s[s:e], lstars[s:e]))
        dp.extend(cpu_verdict_d_pose_batch(posenet_cpu, f0s[s:e], f1s[s:e], poses[s:e]))
    return float(np.mean(ds)), float(np.mean(dp))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n24.npz")
    ap.add_argument("--num-pairs", type=int, default=24)
    ap.add_argument("--verdict-batch", type=int, default=8)
    ap.add_argument("--allow-pressure", action="store_true", help="bypass the governor gate (unsafe)")
    args = ap.parse_args(argv)

    ok, why = _governor_ok()
    if not ok and not args.allow_pressure:
        print(json.dumps({"stage": "measure_verdict_reclaim", "skipped": True, "reason": why,
                          "note": "governor-gated (heavy real torch forwards); pass --allow-pressure to force"}))
        return 0

    from train_witness_realized_through_R_mlx import load_gt_from_cache

    gt, seg_cpu, posenet_cpu = load_gt_from_cache(Path(args.gt_cache), args.num_pairs)
    # The verdict's INPUT is the RENDERED frames; use the real GT frames as a faithful stand-in for
    # the render (identical uint8 shape/dtype => identical scorer transient; the d_seg/d_pose values
    # are irrelevant to the MEMORY measurement, which is the point of this harness).
    n = gt.n_pairs
    f0s = [gt.gt_f0[i] for i in range(n)]
    f1s = [gt.gt_f1[i] for i in range(n)]
    lstars = [gt.lstars[i] for i in range(n)]
    poses = [gt.gt_poses[i] for i in range(n)]

    gc.collect()
    baseline = rss_gib()
    rows: dict[str, object] = {"stage": "measure_verdict_reclaim", "num_pairs": n,
                               "verdict_batch": int(args.verdict_batch),
                               "baseline_rss_gib": round(baseline, 3), "governor": why}

    # C FIRST — on a CLEAN parent (before any in-process forward pollutes the parent RSS). This is the
    # only ordering under which the "parent returns to baseline" claim is honest: A/B ratchet the parent
    # in-place, so they must run AFTER C or C would inherit their high-water mark.
    sub = run_verdict_in_subprocess(f0s, f1s, lstars, poses, vbatch=int(args.verdict_batch))
    gc.collect()
    c_after = rss_gib()
    rows["C_subprocess"] = {"rss_after_gib": round(c_after, 3),
                            "ratchet_gib": round(c_after - baseline, 3),
                            "child_rss_peak_gib": sub.get("child_rss_peak_gib"),
                            "elapsed_s": sub.get("elapsed_s"),
                            "d_seg_mean": round(float(sub["d_seg_mean"]), 6),
                            "d_pose_mean": round(float(sub["d_pose_mean"]), 6),
                            "note": "parent RSS delta vs baseline; child held the transient"}

    # A. in-process, NO reclaim (the ratchet demo). RSS delta measured from the post-C parent level.
    pre_a = rss_gib()
    _ds, _dp = _inprocess_verdict(seg_cpu, posenet_cpu, f0s, f1s, lstars, poses, args.verdict_batch)
    a_after = rss_gib()
    rows["A_inprocess_noreclaim"] = {"rss_before_gib": round(pre_a, 3), "rss_after_gib": round(a_after, 3),
                                     "ratchet_gib": round(a_after - pre_a, 3),
                                     "d_seg": round(_ds, 6), "d_pose": round(_dp, 6)}

    # B. in-process, + cheap reclaim (does the allocator return the pages the in-process forward left?).
    _ds, _dp = _inprocess_verdict(seg_cpu, posenet_cpu, f0s, f1s, lstars, poses, args.verdict_batch)
    b_recl = reclaim_process_memory()
    rows["B_inprocess_reclaim"] = {"rss_before_gib": b_recl["rss_before_gib"],
                                   "rss_after_gib": b_recl["rss_after_gib"],
                                   "reclaimed_gib": b_recl["reclaimed_gib"],
                                   "trim_method": b_recl["trim_method"],
                                   "note": "reclaimed_gib>0 => cheap fix suffices; ~0 => subprocess needed"}

    # Bit-identity cross-check: subprocess d_seg/d_pose vs the in-process values on the same inputs.
    ip_ds, ip_dp = _inprocess_verdict(seg_cpu, posenet_cpu, f0s, f1s, lstars, poses, args.verdict_batch)
    rows["bit_identity"] = {
        "d_seg_bit_equal": bool(np.float64(ip_ds).tobytes() == np.float64(sub["d_seg_mean"]).tobytes()),
        "d_pose_bit_equal": bool(np.float64(ip_dp).tobytes() == np.float64(sub["d_pose_mean"]).tobytes()),
        "d_seg_abs_diff": abs(ip_ds - float(sub["d_seg_mean"])),
        "d_pose_abs_diff": abs(ip_dp - float(sub["d_pose_mean"])),
    }
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
