"""FEED-fs $0/CPU — OPTIMAL Road<->Lane separatrix attack for the witness.

The binding sub-0.15 endgame residual is the Road<->Lane (lane-marking) boundary
(FEED-fr: 98-99% of the witness residual at the operating point). FEED-er solved the
lane SHAPE (FN 0.000133) via per-frame deg-3 openpilot poly x homography, road-plane
IPM, but left an FP over-paint residual (lane_fp_from_road 0.000306) -> total
lane-attributable 0.000439 = 1.87x ABOVE the witness's lane contribution (~0.000236).

This script attacks the FP over-paint (the open FEED-er problem) by deep-math:
the FP is a BAND-WIDTH calibration problem (the 90th-pct half-width over-covers the
thin lane ribbon). We:
  1. reproduce the FEED-er baseline (continuous band, fitted 90th-pct half-width);
  2. CALIBRATE the half-width (global scale sweep) -> the FN/FP water-fill optimum;
  3. fixed thin-ribbon widths (centerline +/- const px);
  4. ORACLE per-row L*-matched width (the achievable lower bound of the ribbon model);
  5. report floats/frame (byte cost) for each.

NO-FAKE: every number is the REAL argmax disagreement of the REAL rasterized polynomial
fit to the REAL class-1 pixels of the REAL frozen CPU-torch SegNet argmax L* (lstars in
gt_n96.npz, bit-exact). Reuses the in-tree FEED-dm component lane_sdf_component.py +
the IDEAL signed_distance_fields (argmax==L* exactly) so the test ISOLATES the lane
field. [macOS-CPU advisory] research-signal; score_claim=false; NOT a byte-closed row.
Pointer UNMOVED 0.19110.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from tac.boundary_math.lane_sdf_component import (
    cluster_lane_lines, fit_lane_line, rasterize_lane_band, lane_signed_distance,
    inject_lane_sdf, decompose_argmax_disagreement, ground_to_image_row,
    _SEG_H, _SEG_W, _V_HORIZON,
)
from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields

_GT = Path("experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
_N, _LANE, _ROAD = 5, 1, 0


def scaled_band(lines, scale, *, fixed_hw=None):
    """Rasterize union band with half-width scaled (or fixed) — reuses the FREE rasterizer."""
    out = []
    for ln in lines:
        import copy
        l2 = copy.copy(ln)
        if fixed_hw is not None:
            l2.halfwidth_coeffs = np.array([0.0, float(fixed_hw)])
        else:
            l2.halfwidth_coeffs = np.asarray(ln.halfwidth_coeffs, np.float64) * float(scale)
        out.append(l2)
    return rasterize_lane_band(out, dash_gate=False)


def oracle_rowwidth_band(L, lines):
    """Per-row L*-matched ribbon: for each line centerline u_c(v), paint the connected
    class-1 run of L* that contains/abuts u_c. This is the ACHIEVABLE lower bound of the
    centerline+row-width ribbon model (the row-width is then itself fittable as a poly)."""
    band = np.zeros((_SEG_H, _SEG_W), bool)
    rows = np.arange(_SEG_H, dtype=np.float64)
    below = rows > (_V_HORIZON + 1.0)
    vr = rows[below]
    is_lane = (L == _LANE)
    for ln in lines:
        _, u_c = ground_to_image_row(vr, ln.lateral_of_forward)
        f0, f1 = ln.forward_range
        fwd, _ = ground_to_image_row(vr, ln.lateral_of_forward)
        for j, vv in enumerate(vr):
            vi = int(vv)
            uc = u_c[j]
            if not np.isfinite(uc):
                continue
            u0 = int(round(uc))
            if u0 < 0 or u0 >= _SEG_W:
                continue
            row = is_lane[vi]
            # grow a run around the nearest lane pixel to u0 (within +/-12 px search)
            srch = 12
            lo = max(0, u0 - srch); hi = min(_SEG_W, u0 + srch + 1)
            local = np.where(row[lo:hi])[0]
            if local.size == 0:
                continue
            # nearest lane px to u0
            cand = local + lo
            nearest = cand[np.argmin(np.abs(cand - u0))]
            # grow connected run
            a = nearest
            while a - 1 >= 0 and row[a - 1]:
                a -= 1
            b = nearest
            while b + 1 < _SEG_W and row[b + 1]:
                b += 1
            band[vi, a:b + 1] = True
    return band


def attrib(band, phi_ideal, L):
    phi1 = lane_signed_distance(band)
    pred = inject_lane_sdf(phi_ideal, phi1, lane_cls=_LANE, mode="replace").argmax(-1)
    return decompose_argmax_disagreement(pred, L, lane_cls=_LANE, road_cls=_ROAD)


def main():
    t0 = time.time()
    npz = np.load(_GT)
    lstars = npz["lstars"]
    P = min(96, lstars.shape[0])
    print(f"[FEED-fs] Road<->Lane width-calibration; n={P}", flush=True)

    scales = [0.30, 0.45, 0.60, 0.75, 0.90, 1.00, 1.20]
    fixed = [0.75, 1.0, 1.5, 2.0, 3.0]
    acc = {f"scale_{s}": {"fn": [], "fp": [], "attr": []} for s in scales}
    acc.update({f"fixed_{x}": {"fn": [], "fp": [], "attr": []} for x in fixed})
    acc["oracle_rowwidth"] = {"fn": [], "fp": [], "attr": []}
    floats_pf = []

    for i in range(P):
        L = np.asarray(lstars[i]).astype(np.int64)
        phi_ideal = signed_distance_fields(L, _N)
        clusters = cluster_lane_lines(L, lane_cls=_LANE)
        lines = [fit_lane_line(c, centerline_deg=3, fit_dash=False) for c in clusters]
        lines = [ln for ln in lines if ln is not None]
        if not lines:
            continue
        floats_pf.append(sum(ln.n_floats() for ln in lines))
        for s in scales:
            d = attrib(scaled_band(lines, s), phi_ideal, L)
            k = f"scale_{s}"
            acc[k]["fn"].append(d.lane_fn)
            acc[k]["fp"].append(d.lane_fp_from_road + d.lane_fp_from_other)
            acc[k]["attr"].append(d.lane_attributable)
        for x in fixed:
            d = attrib(scaled_band(lines, 1.0, fixed_hw=x), phi_ideal, L)
            k = f"fixed_{x}"
            acc[k]["fn"].append(d.lane_fn)
            acc[k]["fp"].append(d.lane_fp_from_road + d.lane_fp_from_other)
            acc[k]["attr"].append(d.lane_attributable)
        d = attrib(oracle_rowwidth_band(L, lines), phi_ideal, L)
        acc["oracle_rowwidth"]["fn"].append(d.lane_fn)
        acc["oracle_rowwidth"]["fp"].append(d.lane_fp_from_road + d.lane_fp_from_other)
        acc["oracle_rowwidth"]["attr"].append(d.lane_attributable)
        if (i + 1) % 24 == 0:
            print(f"  ... {i+1}/{P}  ({time.time()-t0:.1f}s)", flush=True)

    res = {}
    for k, v in acc.items():
        res[k] = {
            "fn": float(np.mean(v["fn"])),
            "fp": float(np.mean(v["fp"])),
            "attr": float(np.mean(v["attr"])),
        }
    # find the argmin-attr config
    best = min(res.items(), key=lambda kv: kv[1]["attr"])
    out = {
        "feed": "FEED-fs", "n": P, "authority": "macOS-CPU advisory",
        "witness_lane_ref": 0.000236, "feed_er_baseline_attr": 0.000439,
        "feed_er_fn": 0.000133, "feed_er_fp": 0.000306,
        "floats_per_frame": float(np.mean(floats_pf)),
        "results": res, "best_config": best[0], "best_attr": best[1]["attr"],
        "elapsed_s": time.time() - t0,
    }
    Path("experiments/results/lane_sdf_FEED-er").mkdir(parents=True, exist_ok=True)
    op = Path("experiments/results/lane_sdf_FEED-er/feed_fs_width_calib.json")
    op.write_text(json.dumps(out, indent=2))
    print("\n=== FEED-fs RESULTS (mean over n96; lane-attributable d_seg) ===")
    print(f"{'config':<20} {'FN':>10} {'FP':>10} {'ATTR':>10}")
    for k in list(acc.keys()):
        r = res[k]
        print(f"{k:<20} {r['fn']:>10.6f} {r['fp']:>10.6f} {r['attr']:>10.6f}")
    print(f"\nBEST: {best[0]}  attr={best[1]['attr']:.6f}  "
          f"(FEED-er baseline 0.000439; witness lane ref 0.000236)")
    print(f"floats/frame={out['floats_per_frame']:.1f}  elapsed={out['elapsed_s']:.1f}s")
    print(f"JSON -> {op}")


if __name__ == "__main__":
    main()
