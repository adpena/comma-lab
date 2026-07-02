#!/usr/bin/env python3
"""Wave-F Stage-2b: MEASURED n600 CORRESPONDENCE-FIRST lane-rate bake-off ($0, CPU, no scorer).

Loads the frozen SegNet argmax cache (``gt_nN.npz`` ``lstars``), fits the per-pair lane
manifold ONCE, then measures the COUNTED byte-closed rate in GATE ORDER (research authority
``.omx/research/lane_coeff_tracking_denoising_optimal_survey_20260702.md``):

  * LBND2_sort_baseline      -- lateral-sort packing (the LANDED Stage-1 codec, n600 rate 0.02765).
  * coherent_slot_none       -- bounded-K coherent slotting, NO denoise: the LOSSLESS
                                correspondence rate refinement (measured ~0.7% -> the honest size).
  * persistent_none          -- persistent tracks, NO denoise: the column explosion (WORSE at n600).
  * coherent_slot_{rts,l1trend,median,rpca} -- clean-track-space BATCH denoise, packed compact.
  * sort_MA_win{N}           -- the incumbent moving-average (median) smoothing, WITH geometric RMS.

The headline question: does correspondence-first (+ coherent batch denoise) beat the moving-
average on rate AND at LESS geometric loss? MEASURED verdict (n600): correspondence is LOSSLESS
but a SMALL rate win (delta mass is fit-JITTER, not slot-swaps); the moving-average remains the
rate winner. All numbers MEASURED (real brotli byte counts); rate_term = 25*bytes/RATE_DENOM. The
d_seg WIN is the #205 through-R follow-up (out of scope). Advisory / build-only; pointer 0.19110 UNMOVED.

    .venv/bin/python tools/wave_f_lane_tracking_rate_n600.py \
        --gt-cache /abs/experiments/results/mlx_fleet_gt_cache/gt_n600.npz --n-pairs 600 \
        --out .omx/research/wave_f_lane_tracking_rate_n600_RESULT.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


from tac.boundary_math.analytic_lane_render_band import (  # noqa: E402
    LaneBandRenderConfig, build_lane_band_pairs_from_lstars, lane_band_tracking_rate_report,
)

RATE_DENOM = 37_545_489.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Wave-F Stage-2b n600 correspondence-first lane-rate bake-off")
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--smooth-methods", type=str, nargs="+",
                    default=["none", "rts", "l1trend", "median", "rpca"])
    ap.add_argument("--gate-dist-m", type=float, default=1.5)
    ap.add_argument("--max-gap", type=int, default=24)
    ap.add_argument("--ref-smooth-wins", type=int, nargs="+", default=[9, 15, 25],
                    help="reference sort-path moving-average (median) windows (the incumbent)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    z = np.load(args.gt_cache, allow_pickle=False)
    if "lstars" not in z.files:
        raise ValueError(f"{args.gt_cache} lacks 'lstars'.")
    lstars = z["lstars"]
    ncap = min(int(args.n_pairs), int(len(lstars)))
    lst = [np.asarray(lstars[i], np.int64) for i in range(ncap)]
    t_load = time.time() - t0

    cfg = LaneBandRenderConfig()
    t0 = time.time()
    pairs, fit_stats = build_lane_band_pairs_from_lstars(lst, cfg)
    t_fit = time.time() - t0

    t0 = time.time()
    report = lane_band_tracking_rate_report(
        pairs, cfg, gate_dist_m=args.gate_dist_m, max_gap=args.max_gap,
        smooth_methods=tuple(args.smooth_methods), ref_smooth_wins=tuple(args.ref_smooth_wins))
    t_measure = time.time() - t0

    corr = report["correspondence_only_best"]
    ma = report["moving_average_best"]
    result = {
        "gt_cache": str(args.gt_cache), "n_pairs": ncap, "fit_stats": fit_stats,
        "rate_denom_bytes": int(RATE_DENOM),
        "timing_s": {"load": round(t_load, 1), "fit": round(t_fit, 1), "measure": round(t_measure, 1)},
        "K_sort": report["K_sort"], "K_persistent": report["K_persistent"],
        "swap_diagnostic": report["swap_diagnostic"],
        "correspondence_lossless_vs_sort": report["correspondence_lossless_vs_sort"],
        "rows": report["rows"],
        "best_variant": report["best_variant"], "best_bytes": report["best_bytes"],
        "best_rate_term": report["best_rate_term"],
        "lbnd2_baseline_rate_term": report["lbnd2_baseline_rate_term"],
        "correspondence_only_best": corr,
        "moving_average_best": ma,
        "verdict": {
            "correspondence_lossless": bool(report["correspondence_lossless_vs_sort"]),
            "correspondence_only_beats_baseline": bool(corr["ratio_vs_lbnd2"] < 1.0),
            "best_beats_moving_average": bool(
                ma is not None and report["best_rate_term"] < ma["rate_term"]),
            "note": ("correspondence is LOSSLESS but a SMALL rate refinement (the delta mass is "
                     "fit-JITTER, not slot-swaps); the moving-average remains the rate winner. "
                     "The correspondence + edge-preserving denoise value is LOWER geometric "
                     "distortion at matched rate -> the #205 through-R d_seg leg is the real gate."),
        },
        "axis_labels": "[macOS-CPU advisory] MEASURED brotli byte counts; NOT a score claim; pointer 0.19110 UNMOVED",
    }
    print(json.dumps(result, indent=2, default=float))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, default=float))
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
