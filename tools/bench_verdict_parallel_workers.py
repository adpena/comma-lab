#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""#509 batch 3 / m5max mem-for-compute (operator fire-now 2026-07-15): measured wall-clock
row for the ``--verdict-parallel-workers`` lever.

Runs the trainer's OWN ``_verdict_dseg_dpose_chunked`` (the exact function the launch path
calls) over the REAL ``gt_n600`` frames on the frozen CPU-torch scorers, torch intra-op
threads pinned to the training standard (``segnet_exact_forward_cpu_thread_law_20260713``),
``vbatch`` chunking as sealed (32). The GT frames are used AS the candidate frames — the
verdict wall is content-independent (same tensor shapes/dtypes through the same forwards),
and using real data lets the bench ALSO assert w=0 vs w=N value equality on real inputs
(the bit-identity claim, checked not assumed).

Emits a JSON receipt (per-arm wall seconds, ratios, host/thread/git provenance) under
``experiments/results/``. ADVISORY [macOS-CPU-torch timing] NON-PROMOTABLE; the verdict is
the ADVISORY path (never read into training) so this lever is score-neutral by construction.
means != ends: a verdict-wall sec lever row; NO score claim; pointer UNMOVED.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--vbatch", type=int, default=32)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 8],
                    help="worker counts to time, in order (0 = sequential incumbent)")
    ap.add_argument("--warmup-pairs", type=int, default=32,
                    help="untimed warmup forward (torch kernel/init cost excluded from ALL arms)")
    ap.add_argument("--out", default=None,
                    help="receipt JSON path (default experiments/results/"
                         "verdict_parallel_bench_<utc>/receipt.json)")
    args = ap.parse_args()

    import sys
    for p in ("experiments", "src", "upstream"):
        sys.path.insert(0, str(REPO / p))

    import torch
    from tac.canonical_equations.segnet_exact_forward_cpu_thread_law_20260713 import (
        SELECTED_THREADS,
    )
    torch.set_num_threads(int(SELECTED_THREADS))

    from train_levelset_witness_realized_through_R_mlx import _verdict_dseg_dpose_chunked
    from train_witness_realized_through_R_mlx import load_gt_from_cache

    gt, seg_cpu, posenet_cpu = load_gt_from_cache(REPO / args.gt_cache, int(args.pairs))
    P = gt.n_pairs
    f0s, f1s, lstars, poses = gt.gt_f0, gt.gt_f1, gt.lstars, gt.gt_poses

    # untimed warmup (first-forward torch init would bias whichever arm runs first)
    w = max(1, min(int(args.warmup_pairs), P))
    _verdict_dseg_dpose_chunked(
        seg_cpu, posenet_cpu, f0s[:w], f1s[:w], lstars[:w], poses[:w],
        vbatch=int(args.vbatch), workers=0)

    rows = []
    baseline = {}
    for wk in args.workers:
        t0 = time.perf_counter()
        d_seg, d_pose = _verdict_dseg_dpose_chunked(
            seg_cpu, posenet_cpu, f0s, f1s, lstars, poses,
            vbatch=int(args.vbatch), workers=int(wk))
        wall = time.perf_counter() - t0
        row = {"workers": int(wk), "wall_s": round(wall, 2),
               "d_seg": float(d_seg), "d_pose": float(d_pose),
               "peak_rss_gib": round(
                   resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)}
        if not baseline:
            baseline = {"wall_s": wall, "d_seg": d_seg, "d_pose": d_pose}
        else:
            row["speedup_vs_first_arm"] = round(baseline["wall_s"] / wall, 3)
            row["values_identical_to_first_arm"] = bool(
                d_seg == baseline["d_seg"] and d_pose == baseline["d_pose"])
        rows.append(row)
        print(json.dumps({"stage": "verdict_parallel_bench_arm", **row}), flush=True)

    utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else (
        REPO / f"experiments/results/verdict_parallel_bench_{utc}/receipt.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],  # subprocess-no-check-OK: git-head provenance capture; except arm records 'unknown'
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except OSError:
        git = "unknown"
    receipt = {
        "schema": "verdict_parallel_workers_bench.v1",
        "axis": "[macOS-CPU-torch timing] ADVISORY NON-PROMOTABLE",
        "score_claim": False, "promotable": False,
        "gt_cache": str(args.gt_cache), "pairs": int(P), "vbatch": int(args.vbatch),
        "torch_intraop_threads": int(SELECTED_THREADS),
        "warmup_pairs": int(w), "arms": rows,
        "note": ("GT frames used as candidate frames: wall is content-independent; value "
                 "equality across arms asserted on real inputs (the bit-identity claim). "
                 "Record concurrent host load with the receipt when citing the ratio."),
        "host": platform.node(), "git": git, "utc": utc,
    }
    out.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({"stage": "verdict_parallel_bench_receipt", "path": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
