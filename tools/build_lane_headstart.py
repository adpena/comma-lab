#!/usr/bin/env python3
"""Build + measure the Wyner-Ziv lane-centerline HEAD-START for the v2 witness.

Thin CLI over :mod:`tac.boundary_math.lane_headstart` (all real implementation lives
in the library, per "tac stays clean").  Reproduces the conditioned lane-base d_seg
(the a99f41f0 ~0.00214 / 0.00207 head-start), measures the Wyner-Ziv conditional
residual support + exact round-trip, and estimates the COUNTED centerline bytes
(parametric, exact-preserving delta-entropy, and a MEASURED zlib temporal estimate).

The head-start = the ResidualGauge ``CONDITIONAL_ON_LANE_PRIOR`` cell: the eventual
through-R lane GPU run STARTS from the conditioned base, not from scratch.

rule-118: the centerline RASTERIZER is a GENERIC algorithm -> FREE in inflate.py; the
stored coeffs are VIDEO-DERIVED -> COUNTED (tiny); the learned residual is PENDING-GPU.

Authority: [macOS research-signal] / advisory. score_claim=false; promotable=false.
Pointer UNMOVED contest-CPU 0.19110.  This is MEANS, not a score.

Disk hygiene: writes only a small JSON summary (<10 KB); produces NO large artifacts,
so no SSD-spill / cleanup hook is required (the GT cache is read-only, lazily).

Usage:
    .venv/bin/python tools/build_lane_headstart.py \
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
        --degree 4 --json
    # write a durable summary artifact:
    .venv/bin/python tools/build_lane_headstart.py --gt-cache ... \
        --out experiments/results/lane_headstart_$(date -u +%Y%m%dT%H%M%SZ)
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from tac.boundary_math import lane_headstart as lh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gt-cache",
        default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        help="GT cache npz containing 'lstars' (frozen-SegNet argmax).",
    )
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument(
        "--dash-bridge-rows",
        type=int,
        default=1,
        help="1 = per-dash (the 0.00214 head-start); >1 bridges dashes (0.00341).",
    )
    ap.add_argument("--min-component-pixels", type=int, default=12)
    ap.add_argument("--max-half-width", type=int, default=4)
    ap.add_argument("--n-target-frames", type=int, default=600)
    ap.add_argument("--out", default=None, help="optional dir for a durable JSON summary.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    d = np.load(args.gt_cache)  # npz members are lazy; only 'lstars' is materialized
    if "lstars" not in d:
        raise SystemExit(f"{args.gt_cache} has no 'lstars'; keys={list(d.keys())}")
    lstars = d["lstars"]

    result = lh.build_lane_headstart(
        lstars,
        degree=args.degree,
        dash_bridge_rows=args.dash_bridge_rows,
        min_component_pixels=args.min_component_pixels,
        max_half_width=args.max_half_width,
        n_target_frames=args.n_target_frames,
    )
    cell = lh.gauge_cost_cell(result)

    summary = {
        "gt_cache": args.gt_cache,
        "config": {
            "degree": args.degree,
            "dash_bridge_rows": args.dash_bridge_rows,
            "min_component_pixels": args.min_component_pixels,
            "max_half_width": args.max_half_width,
            "n_target_frames": args.n_target_frames,
        },
        "result": {k: v for k, v in asdict(result).items() if k != "centerlines_per_frame"},
        "gauge_cost_cell": cell,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authority": "[macOS research-signal] advisory; score_claim=false; pointer UNMOVED 0.19110",
    }

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, "lane_headstart_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        summary["written_to"] = out_path

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return

    r = result
    print(f"\n=== lane head-start (Wyner-Ziv CONDITIONAL_ON_LANE_PRIOR) — {r.n_frames} frames ===")
    print(f"  from-scratch lane d_seg (drop-all): {r.from_scratch_lane_dseg:.6f}")
    print(f"  conditioned base lane d_seg       : {r.base_lane_dseg:.6f}")
    print(f"  recovered fraction                : {r.recovered_frac:.4f}")
    print(f"  residual d_seg the witness must fix: {r.base_lane_dseg:.6f} "
          f"(target sub-015 <= {lh.SUB015_LANE_DSEG:.2e}; sub-019 <= {lh.SUB019_LANE_DSEG:.2e})")
    print(f"  residual round-trip exact         : {r.roundtrip_exact}")
    print(f"  IoU(true,base) mean               : {r.iou_mean:.4f}")
    b = r.bytes
    print(f"  centerline bytes/600 parametric   : {b['parametric_bytes_600']:.0f}")
    print(f"  centerline bytes/600 delta-entropy: {b['delta_entropy_bytes_600']:.0f}")
    print(f"  centerline bytes/600 zlib temporal: {b['zlib_temporal_bytes_600']:.0f}  (image-space iid)")
    print(f"  achievable iid rate term          : {b['achievable_iid_rate_term_600']:.4f}")
    print(f"  ground-frame target               : {b['temporal_groundframe_target_bytes_600']}")
    print(f"  learned residual cost             : {cell['learned_residual_cost']}")
    if args.out:
        print(f"  summary written to                : {summary.get('written_to')}")
    print("\nNOTE: advisory [macOS research-signal]; the centerline base is the FREE rasterizer's\n"
          "input; the LEARNED ragged-boundary residual is the real (PENDING-GPU) budget.\n"
          "Pointer UNMOVED 0.19110. MEANS toward a byte-closed exact row, not a score.")


if __name__ == "__main__":
    main()
