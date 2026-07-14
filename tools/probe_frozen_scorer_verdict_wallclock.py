#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the REAL frozen-scorer VERDICT wall-clock — the campaign's "95%-kill" target.

The in-loop component timer (spec_throughput_component_timer) showed the *in-loop MLX teacher* is
fast (~0.02-0.05s/call) and the async CPU-torch *verdict* was only ~4.3s at n24 verdict-pairs=2.
The campaign's "95% = frozen-SegNet fwd+bwd" is really the **CPU-torch authority VERDICT** (the
d_seg SegNet-argmax + d_pose PoseNet forward over all pairs), which at n600 is the #205 OOM spike.

This probe times the ACTUAL chunked verdict (``_verdict_dseg_dpose_chunked``) on the real GT frames,
under the canonical 1-thread CPU-torch law, at a given ``--num-pairs``. Scoring GT-vs-GT gives
d_seg≈0 but the FORWARD COST is input-independent, so the wall-clock is the true authority-forward
cost. Reports the SegNet(d_seg) vs PoseNet(d_pose) split and per-pair cost -> linear-extrapolates to
n600 (the accumulation loop is per-pair linear; #306 cross-calibration).

Authority: ``[macOS-CPU-torch 1-thread advisory wall-clock]`` NON-PROMOTABLE. No score claim, no
training, no paid job. MEANS only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Time the frozen-scorer verdict (the 95%-kill target).")
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--num-pairs", type=int, default=96)
    ap.add_argument("--vbatch", type=int, default=32, help="chunk size (memory-safe; #205 OOM fix)")
    ap.add_argument("--extrapolate-to", type=int, default=600)
    ap.add_argument("--output", default=".omx/research/frozen_scorer_verdict_wallclock_probe.json")
    args = ap.parse_args(argv)

    # Canonical 1-thread CPU-torch law (training teacher/verdict forwards).
    import torch
    torch.set_num_threads(1)

    from train_levelset_witness_realized_through_R_mlx import (  # type: ignore
        _verdict_dseg_dpose_chunked,
        load_gt_from_cache,
    )
    from train_witness_realized_through_R_mlx import (  # type: ignore
        cpu_verdict_d_pose_batch,
        cpu_verdict_d_seg_batch,
    )

    t_load0 = time.perf_counter()
    gt, seg_cpu, posenet_cpu = load_gt_from_cache(Path(args.gt_cache), args.num_pairs)
    t_load = time.perf_counter() - t_load0

    # Extract the per-pair frame/label/pose lists the verdict consumes.
    f0s = list(gt.gt_f0)
    f1s = list(gt.gt_f1)
    lstars = list(gt.lstars)
    poses = list(gt.gt_poses)
    n = len(f1s)

    # Warm one chunk (first-call graph/BN warmup) so the timed pass is steady-state.
    _ = cpu_verdict_d_seg_batch(seg_cpu, f1s[:min(4, n)], lstars[:min(4, n)])

    # Time d_seg (SegNet argmax) alone, d_pose (PoseNet) alone, and the combined chunked verdict.
    t0 = time.perf_counter()
    ds_all = []
    for s in range(0, n, args.vbatch):
        e = min(s + args.vbatch, n)
        ds_all.extend(cpu_verdict_d_seg_batch(seg_cpu, f1s[s:e], lstars[s:e]))
    t_seg = time.perf_counter() - t0

    t0 = time.perf_counter()
    dp_all = []
    for s in range(0, n, args.vbatch):
        e = min(s + args.vbatch, n)
        dp_all.extend(cpu_verdict_d_pose_batch(posenet_cpu, f0s[s:e], f1s[s:e], poses[s:e]))
    t_pose = time.perf_counter() - t0

    t0 = time.perf_counter()
    d_seg, d_pose = _verdict_dseg_dpose_chunked(
        seg_cpu, posenet_cpu, f0s, f1s, lstars, poses, vbatch=args.vbatch)
    t_verdict = time.perf_counter() - t0

    per_pair_verdict = t_verdict / n
    extrap = per_pair_verdict * args.extrapolate_to
    out = {
        "schema": "frozen_scorer_verdict_wallclock.v1",
        "axis": "[macOS-CPU-torch 1-thread advisory wall-clock] NON-PROMOTABLE",
        "score_claim": False, "promotable": False, "means_only": True,
        "gt_cache": args.gt_cache, "num_pairs": n, "vbatch": args.vbatch,
        "torch_threads": 1,
        "load_s": round(t_load, 3),
        "seg_verdict_s_total": round(t_seg, 3),
        "pose_verdict_s_total": round(t_pose, 3),
        "combined_verdict_s_total": round(t_verdict, 3),
        "per_pair_verdict_s": round(per_pair_verdict, 4),
        "seg_per_pair_s": round(t_seg / n, 4),
        "pose_per_pair_s": round(t_pose / n, 4),
        "seg_fraction_of_verdict": round(t_seg / (t_seg + t_pose), 3),
        "pose_fraction_of_verdict": round(t_pose / (t_seg + t_pose), 3),
        f"extrapolated_n{args.extrapolate_to}_verdict_s": round(extrap, 1),
        f"extrapolated_n{args.extrapolate_to}_verdict_min": round(extrap / 60.0, 2),
        "d_seg_measured": round(float(d_seg), 6),
        "d_pose_measured": round(float(d_pose), 6),
        "note": (
            "GT-vs-GT so d_seg/d_pose ~ label-noise floor, NOT a score; the FORWARD wall-clock is "
            "input-independent = the real authority-verdict cost. per-pair linear -> n600 extrapolation "
            "is legitimate for wall-clock (#306). SegNet=d_seg forward, PoseNet=d_pose forward."
        ),
    }
    outp = REPO / args.output
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
