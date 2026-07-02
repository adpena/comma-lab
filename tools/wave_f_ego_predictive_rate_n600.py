#!/usr/bin/env python3
"""Wave-F unified-ξ: MEASURED n600 lane-rate bake-off ($0, CPU, no scorer).

Loads the frozen SegNet argmax cache (``gt_nN.npz`` ``lstars``), fits the per-pair lane
manifold ONCE, then measures the COUNTED byte-closed rate for every lane-rate variant:

  * LBND2  -- temporal-delta RD baseline (the LANDED Stage-1 codec).
  * LBND3  -- ego-motion-compensated predictive coding, per estimator:
              lane-optimal (i, achievable floor), PoseNet-physical (ii, gt_poses).
  * LBND2-on-SMOOTHED -- the L1 source re-parameterization (temporal denoising of the
              coeff trajectory) the ego-predictor negative REVEALED, over a window sweep.
  * LBND3-ego-on-SMOOTHED -- combined (does ego help ON TOP of denoising?).

All numbers MEASURED (real brotli byte counts). rate_term = 25*bytes/RATE_DENOM. This is
the primary research measurement (design Q2/Q3): does ego-compensation collapse the
camera-frame floor, and which estimator wins the lane-rate axis. Advisory / build-only;
pointer 0.19110 UNMOVED.

    .venv/bin/python tools/wave_f_ego_predictive_rate_n600.py \
        --gt-cache /abs/experiments/results/mlx_fleet_gt_cache/gt_n600.npz --n-pairs 600 \
        --out .omx/research/wave_f_ego_predictive_rate_n600_RESULT.json
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

import brotli  # noqa: E402

from tac.boundary_math.analytic_lane_render_band import (  # noqa: E402
    LaneBandRenderConfig, build_lane_band_pairs_from_lstars, temporal_smooth_pairs_lines,
    serialize_lane_band_rd, lane_band_rd3_rate_report, _pack_pairs_to_matrix,
)
from tac.boundary_math.ego_xi_trajectory import (  # noqa: E402
    LaneOptimalEgoEstimator, PoseTargetEgoEstimator, bspline_fit_error_curve,
)

RATE_DENOM = 37_545_489.0


def _rd2_bytes(pairs_lines, cfg) -> int:
    return len(brotli.compress(serialize_lane_band_rd(pairs_lines, cfg), quality=11))


def main() -> int:
    ap = argparse.ArgumentParser(description="Wave-F unified-ξ n600 lane-rate bake-off")
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--smooth-wins", type=int, nargs="+", default=[3, 5, 9, 15])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    z = np.load(args.gt_cache, allow_pickle=False)
    if "lstars" not in z.files:
        raise ValueError(f"{args.gt_cache} lacks 'lstars'.")
    lstars = z["lstars"]
    ncap = min(int(args.n_pairs), int(len(lstars)))
    lst = [np.asarray(lstars[i], np.int64) for i in range(ncap)]
    gt_poses = np.asarray(z["gt_poses"], np.float64)[:ncap] if "gt_poses" in z.files else None
    t_load = time.time() - t0

    cfg = LaneBandRenderConfig()
    t0 = time.time()
    pairs, fit_stats = build_lane_band_pairs_from_lstars(lst, cfg)
    t_fit = time.time() - t0
    M, presence, K = _pack_pairs_to_matrix(pairs)

    rows: list[dict] = []
    # baseline
    b2 = _rd2_bytes(pairs, cfg)
    rows.append({"variant": "LBND2_baseline", "brotli_bytes": b2, "rate_term": 25.0 * b2 / RATE_DENOM,
                 "ratio_vs_lbnd2": 1.0})

    # LBND3 ego -- lane-optimal (full 3-DOF and dy+yaw ablation)
    for name, kw in (("lane_optimal_full", dict(fit_forward=True)),
                     ("lane_optimal_dy_yaw", dict(fit_forward=False)),
                     ("lane_optimal_dy_yaw_smooth5", dict(fit_forward=False, smooth_win=5))):
        xi = LaneOptimalEgoEstimator(**kw).estimate(M, presence, K)
        rep = lane_band_rd3_rate_report(pairs, cfg, xi)
        rows.append({"variant": f"LBND3_ego_{name}", "brotli_bytes": rep["rd3_lbnd3_brotli_bytes"],
                     "rate_term": rep["rd3_rate_term"], "ratio_vs_lbnd2": rep["rd3_vs_rd2_ratio"],
                     "ego_payload_brotli_bytes": rep["ego_payload_brotli_bytes"],
                     "ds_med_m": float(np.median(np.abs(xi.ds))), "dy_med_m": float(np.median(np.abs(xi.dy)))})

    # LBND3 ego -- PoseNet physical (gt_poses; affine-to-lane calibration = design Q4)
    xi_bspline_curve = None
    if gt_poses is not None:
        ref = LaneOptimalEgoEstimator(fit_forward=True).estimate(M, presence, K)
        xi_phys = PoseTargetEgoEstimator(calib="affine_to_lane").estimate(gt_poses, ref=ref)
        rep = lane_band_rd3_rate_report(pairs, cfg, xi_phys)
        rows.append({"variant": "LBND3_ego_posenet_affine", "brotli_bytes": rep["rd3_lbnd3_brotli_bytes"],
                     "rate_term": rep["rd3_rate_term"], "ratio_vs_lbnd2": rep["rd3_vs_rd2_ratio"],
                     "ego_payload_brotli_bytes": rep["ego_payload_brotli_bytes"],
                     "affine_calib": xi_phys.provenance})
        # design Q5: SE(3) B-spline fit-error-vs-M curve of the physical trajectory
        try:
            xi_bspline_curve = bspline_fit_error_curve(xi_phys.dense_xi, [8, 12, 16, 24, 32, 48])
        except Exception as e:  # noqa: BLE001
            xi_bspline_curve = {"error": f"{type(e).__name__}: {e}"}

    # SOURCE-SMOOTHING (the L1 re-param the ego negative revealed) -- window sweep
    for w in args.smooth_wins:
        sm = temporal_smooth_pairs_lines(pairs, win=w)
        bs = _rd2_bytes(sm, cfg)
        rows.append({"variant": f"LBND2_smoothed_win{w}", "brotli_bytes": bs, "rate_term": 25.0 * bs / RATE_DENOM,
                     "ratio_vs_lbnd2": bs / b2, "n_lines": int(sum(len(l) for l in sm))})

    # combined: ego on smoothed (does ego add value ON TOP of denoising?)
    if gt_poses is not None:
        w_best = min(args.smooth_wins, key=lambda w: _rd2_bytes(temporal_smooth_pairs_lines(pairs, win=w), cfg))
        sm = temporal_smooth_pairs_lines(pairs, win=w_best)
        Ms, presm, Km = _pack_pairs_to_matrix(sm)
        xi_sm = LaneOptimalEgoEstimator(fit_forward=True).estimate(Ms, presm, Km)
        rep = lane_band_rd3_rate_report(sm, cfg, xi_sm)
        rows.append({"variant": f"LBND3_ego_on_smoothed_win{w_best}",
                     "brotli_bytes": rep["rd3_lbnd3_brotli_bytes"], "rate_term": rep["rd3_rate_term"],
                     "ratio_vs_lbnd2": rep["rd3_vs_rd2_ratio"], "smoothed_lbnd2_bytes": _rd2_bytes(sm, cfg)})

    best = min(rows, key=lambda r: r["brotli_bytes"])
    result = {
        "gt_cache": str(args.gt_cache), "n_pairs": ncap, "K_slots": int(K),
        "fit_stats": fit_stats, "rate_denom_bytes": int(RATE_DENOM),
        "timing_s": {"load": round(t_load, 1), "fit": round(t_fit, 1)},
        "rows": rows,
        "best_variant": best["variant"], "best_bytes": best["brotli_bytes"], "best_rate_term": best["rate_term"],
        "bspline_fit_error_curve": xi_bspline_curve,
        "verdict": {
            "ego_predictive_helps": bool(min((r["ratio_vs_lbnd2"] for r in rows if r["variant"].startswith("LBND3_ego_lane") or r["variant"].startswith("LBND3_ego_posenet")), default=1.0) < 1.0),
            "source_smoothing_best_ratio": float(min((r["ratio_vs_lbnd2"] for r in rows if "smoothed" in r["variant"] and "ego" not in r["variant"]), default=1.0)),
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
