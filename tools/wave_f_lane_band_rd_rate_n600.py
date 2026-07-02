#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Wave-F: MEASURE the real n600 lane-band rate (naive LBND1 vs optimal LBND2 RD).

Fits the per-pair lane lines from the REAL frozen SegNet argmax cache (``gt_nN.npz``
``lstars``, the same cache the byte-close tool uses) and measures the COUNTED archive
bytes of BOTH codecs + the Shannon floor + the PTC1 range-coder comparison + the
induced geometric lateral RMS. This is the "map the curve" evidence for the Wave-F
rate claim -- REAL byte counts, never asserted. NO GPU, NO scorer, CPU, ``$0``.

    .venv/bin/python tools/wave_f_lane_band_rd_rate_n600.py \
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --n-pairs 600

The rate_term is ``25 * counted_brotli_bytes / 37_545_489``. Advisory / build-only;
pointer 0.19110 UNMOVED (moves only via a byte-closed upstream/evaluate.py exact row).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.boundary_math.analytic_lane_render_band import (  # noqa: E402
    LaneBandRDTolerance,
    LaneBandRenderConfig,
    build_lane_band_pairs_from_lstars,
    lane_band_rd_rate_report,
)

RATE_DENOM = 37_545_489.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Wave-F n600 lane-band RD rate measurement")
    ap.add_argument("--gt-cache", type=Path, required=True,
                    help="frozen SegNet argmax cache (gt_nN.npz with 'lstars').")
    ap.add_argument("--n-pairs", type=int, default=600, help="cap pairs (default 600 = full n600).")
    ap.add_argument("--u-mask", action="store_true", help="enable the witness-margin u_mask cfg.")
    ap.add_argument("--lat-tol-m", type=float, default=0.02, help="centerline lateral tolerance (m).")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here (durable path).")
    args = ap.parse_args()

    if not args.gt_cache.exists():
        raise FileNotFoundError(f"--gt-cache {args.gt_cache} not found.")
    t0 = time.time()
    z = np.load(args.gt_cache, allow_pickle=False)
    if "lstars" not in z.files:
        raise ValueError(f"{args.gt_cache} lacks 'lstars'.")
    lstars = z["lstars"]
    ncap = min(int(args.n_pairs), int(len(lstars)))
    lst_list = [np.asarray(lstars[i], np.int64) for i in range(ncap)]
    t_load = time.time() - t0

    cfg = LaneBandRenderConfig(u_mask_enabled=bool(args.u_mask))
    tol = LaneBandRDTolerance(lat_tol_m=float(args.lat_tol_m))

    t1 = time.time()
    pairs_lines, fit_stats = build_lane_band_pairs_from_lstars(lst_list, cfg)
    t_fit = time.time() - t1

    t2 = time.time()
    rep = lane_band_rd_rate_report(pairs_lines, cfg, tol=tol)
    t_rate = time.time() - t2

    out = {
        "gt_cache": str(args.gt_cache),
        "n_pairs": ncap,
        "lat_tol_m": float(args.lat_tol_m),
        "u_mask": bool(args.u_mask),
        "fit_stats": fit_stats,
        "rate_report": rep,
        "timing_s": {"load": round(t_load, 1), "fit": round(t_fit, 1), "rate": round(t_rate, 1)},
        "headline": {
            "naive_LBND1_brotli_bytes": rep["naive_lbnd1_brotli_bytes"],
            "naive_rate_term": round(rep["naive_rate_term"], 6),
            "rd_LBND2_brotli_bytes": rep["rd_lbnd2_brotli_bytes"],
            "rd_rate_term": round(rep["rd_rate_term"], 6),
            "rd_vs_naive_ratio": round(rep["rd_vs_naive_ratio"], 4),
            "ptc1_range_coded_bytes": rep["ptc1_range_coded_bytes"],
            "rate_floor_plus_0p005_bytes": int(0.005 * RATE_DENOM / 25.0),
            "under_0p005_floor": bool(rep["rd_lbnd2_brotli_bytes"] < 0.005 * RATE_DENOM / 25.0),
        },
    }
    js = json.dumps(out, indent=2)
    print(js)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js)
        print(f"\n[wrote] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
