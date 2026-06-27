"""$0 CPU — FEED-er: per-frame PARAMETRIC lane-boundary SOLVE feasibility.

Reactivation of the FEED-el (a56e3816) deferral: the STATIC frozen-mask decomposition
was DEFERRED because d_seg flip-mass is on per-frame-MOVING boundaries. This script
tests whether a PER-FRAME PARAMETRIC lane solve (openpilot ground-frame polynomial x
the known camera homography K, scaled to scorer res in lane_sdf_component) CAPTURES that
motion across the FULL n96 (curved + moving frames), not just the n48 straight segment
FEED-dm/ds measured.

It REUSES the in-tree FEED-dm component (src/tac/boundary_math/lane_sdf_component.py),
adds per-frame distribution + temporal-smoothness (the motion -> delta-codeable bytes
question) + an image-coords vs road-plane fit comparison.

NO-FAKE: every number is the REAL argmax disagreement of the REAL rasterized polynomial
fit to the REAL class-1 pixels of the REAL frozen CPU-torch SegNet argmax L* (lstars in
gt_n96.npz, bit-exact). [macOS-CPU advisory] research-signal; score_claim=false,
promotable=false; NOT a byte-closed row. Pointer UNMOVED 0.19110.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tac.boundary_math.lane_sdf_component import (
    build_structured_lane_sdf,
    cluster_lane_lines,
    decompose_argmax_disagreement,
    fit_lane_line,
    inject_lane_sdf,
    rasterize_lane_band,
    _SEG_H,
    _SEG_W,
    _V_HORIZON,
)
from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields

_GT = Path("experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
_N_CLASSES = 5
_LANE = 1
_ROAD = 0


# ---------------------------------------------------------------------------
# Image-coords lane fit (u = poly(v)) — the comparison arm to the road-plane fit.
# Same clustering (ground-frame, to merge dashes) but the CENTERLINE is fit
# directly in image coords u=poly(v) instead of lateral=poly(forward). Tests the
# hypothesis that the road-plane (inverse-homography) fit is the right space.
# ---------------------------------------------------------------------------
def rasterize_image_coords_lanes(L: np.ndarray, *, deg: int = 3,
                                 v_h: float = _V_HORIZON) -> tuple[np.ndarray, int]:
    """Cluster lane pixels (ground-frame grouping), fit u=poly(v) per cluster in IMAGE
    coords, rasterize a per-row half-width band. Returns (band_mask, total_floats)."""
    clusters = cluster_lane_lines(L, lane_cls=_LANE, v_h=v_h)
    band = np.zeros((_SEG_H, _SEG_W), bool)
    total_floats = 0
    for px in clusters:
        if px.shape[0] < 12:
            continue
        v = px[:, 0].astype(np.float64)
        u = px[:, 1].astype(np.float64)
        d = int(min(deg, max(1, np.unique(np.round(v)).size - 1)))
        c = np.polyfit(v, u, d)                       # u = poly(v) ; d+1 floats
        # per-row half-width from residual spread (1 float, median) -> +1 float
        u_c_all = np.polyval(c, v)
        hw = float(np.clip(np.percentile(np.abs(u - u_c_all), 90), 0.5, 12.0))
        total_floats += (d + 1) + 1
        vlo, vhi = int(np.floor(v.min())), int(np.ceil(v.max()))
        vlo = max(vlo, int(v_h + 1))
        for vv in range(vlo, min(vhi + 1, _SEG_H)):
            uc = float(np.polyval(c, float(vv)))
            lo = int(max(0, np.floor(uc - hw)))
            hi = int(min(_SEG_W, np.ceil(uc + hw) + 1))
            if hi > lo:
                band[vv, lo:hi] = True
    return band, total_floats


def _attrib_from_band(band: np.ndarray, L: np.ndarray, phi_ideal: np.ndarray):
    """Inject a band (as a crisp SDF) into the ideal field stack and decompose vs L*."""
    from tac.boundary_math.lane_sdf_component import lane_signed_distance
    phi1 = lane_signed_distance(band)
    pred = inject_lane_sdf(phi_ideal, phi1, lane_cls=_LANE, mode="replace").argmax(-1)
    return decompose_argmax_disagreement(pred, L, lane_cls=_LANE, road_cls=_ROAD)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--deg", type=int, default=3)
    ap.add_argument("--json-out", type=str,
                    default="experiments/results/lane_sdf_FEED-er/perframe_n96.json")
    args = ap.parse_args()

    if not _GT.exists():
        raise SystemExit(f"GT cache missing: {_GT}")
    t0 = time.time()
    npz = np.load(_GT)
    lstars = npz["lstars"]
    P = min(int(args.n), lstars.shape[0])
    print(f"[FEED-er $0] per-frame parametric lane solve; n={P} deg={args.deg}", flush=True)

    rp_attr, rp_fn, rp_fp = [], [], []     # road-plane fit (the component)
    ic_attr, ic_fn, ic_fp = [], [], []     # image-coords fit (comparison arm)
    floats_rp, floats_ic = [], []
    curvature = []                          # |2nd-order lateral coeff| of the dominant line
    lane_area = []
    bands = []                              # store rasterized band for temporal IoU
    n_lines = []

    for i in range(P):
        L = np.asarray(lstars[i]).astype(np.int64)
        phi_ideal = signed_distance_fields(L, _N_CLASSES)
        lane_area.append(float((L == _LANE).mean()))

        # ---- road-plane (the FEED-dm component), continuous (dash off = optimum) ----
        phi1_c, mc = build_structured_lane_sdf(L, lane_cls=_LANE, dash_gate=False,
                                               centerline_deg=args.deg)
        pred_c = inject_lane_sdf(phi_ideal, phi1_c, lane_cls=_LANE, mode="replace").argmax(-1)
        dc = decompose_argmax_disagreement(pred_c, L, lane_cls=_LANE, road_cls=_ROAD)
        rp_attr.append(dc.lane_attributable); rp_fn.append(dc.lane_fn)
        rp_fp.append(dc.lane_fp_from_road + dc.lane_fp_from_other)
        floats_rp.append(mc["total_floats"]); n_lines.append(mc["n_lines"])

        # store the rasterized band (road-plane) for temporal smoothness
        clusters = cluster_lane_lines(L, lane_cls=_LANE)
        lines = [fit_lane_line(c, centerline_deg=args.deg, fit_dash=False) for c in clusters]
        lines = [ln for ln in lines if ln is not None]
        band_rp = rasterize_lane_band(lines, dash_gate=False)
        bands.append(band_rp)
        # curvature proxy: max |quadratic coeff| across lines (deg>=2)
        cv = 0.0
        for ln in lines:
            cc = ln.centerline_coeffs
            if len(cc) >= 3:
                cv = max(cv, abs(float(cc[-3])))
        curvature.append(cv)

        # ---- image-coords fit (comparison) ----
        band_ic, tf_ic = rasterize_image_coords_lanes(L, deg=args.deg)
        di = _attrib_from_band(band_ic, L, phi_ideal)
        ic_attr.append(di.lane_attributable); ic_fn.append(di.lane_fn)
        ic_fp.append(di.lane_fp_from_road + di.lane_fp_from_other)
        floats_ic.append(tf_ic)

        if (i + 1) % 24 == 0:
            print(f"  ... {i+1}/{P}", flush=True)

    rp_attr = np.array(rp_attr); ic_attr = np.array(ic_attr)
    curvature = np.array(curvature); lane_area = np.array(lane_area)

    # temporal smoothness: band IoU frame-to-frame (how much the lane MOVES)
    band_iou = []
    for i in range(1, P):
        a, b = bands[i - 1], bands[i]
        inter = (a & b).sum(); uni = (a | b).sum()
        band_iou.append(float(inter / uni) if uni > 0 else 1.0)
    band_iou = np.array(band_iou)

    # curvature dependence: correlate per-frame curvature with per-frame error
    if curvature.std() > 1e-12 and rp_attr.std() > 1e-12:
        corr_curv = float(np.corrcoef(curvature, rp_attr)[0, 1])
    else:
        corr_curv = 0.0

    # --- byte accounting: ~floats/frame -> temporal-delta + AR estimate ---
    # smooth ego-motion => coeffs are temporally correlated. Conservative AR estimate:
    # quantize each float to ~10 bits raw; temporal-delta typically saves ~2x; further
    # AR/entropy ~1.5x. We REPORT raw + a conservative compressed estimate (NOT a
    # byte-closed row; the IPM+EDT rasterizer is FREE rule-118; only coeffs counted).
    fpp_rp = float(np.mean(floats_rp))
    raw_bytes = fpp_rp * P * 10 / 8.0                      # 10 bits/float raw
    ar_bytes = raw_bytes / 3.0                             # ~3x temporal-delta+AR (conservative)

    # witness lane contribution reference (advisory; pointer-derived numbers in prompt)
    witness_total = 0.00124
    witness_lane = witness_total * 0.19                   # ~0.000236

    out = {
        "feed": "FEED-er", "n": P, "deg": args.deg,
        "authority": "macOS-CPU advisory", "score_claim": False, "promotable": False,
        "byte_closed_row": False,
        "road_plane": {
            "lane_attributable_mean": float(rp_attr.mean()),
            "lane_attributable_median": float(np.median(rp_attr)),
            "lane_attributable_p90": float(np.percentile(rp_attr, 90)),
            "lane_attributable_max": float(rp_attr.max()),
            "lane_fn_mean": float(np.mean(rp_fn)),
            "lane_fp_mean": float(np.mean(rp_fp)),
            "floats_per_frame": fpp_rp,
            "lines_per_frame": float(np.mean(n_lines)),
        },
        "image_coords": {
            "lane_attributable_mean": float(ic_attr.mean()),
            "lane_fn_mean": float(np.mean(ic_fn)),
            "lane_fp_mean": float(np.mean(ic_fp)),
            "floats_per_frame": float(np.mean(floats_ic)),
        },
        "motion": {
            "band_iou_frame_to_frame_mean": float(band_iou.mean()),
            "band_iou_min": float(band_iou.min()),
            "curvature_mean": float(curvature.mean()),
            "curvature_max": float(curvature.max()),
            "corr_curvature_vs_error": corr_curv,
            "frac_frames_with_curvature": float((curvature > 1e-6).mean()),
        },
        "bytes": {
            "raw_bytes_10bit": round(raw_bytes),
            "ar_estimate_bytes": round(ar_bytes),
            "rate_term_ar_estimate": round(25 * ar_bytes / 37_545_489, 8),
            "note": "IPM+EDT rasterizer FREE (rule-118); only per-frame coeffs counted; estimate NOT byte-closed",
        },
        "witness_reference": {
            "witness_total_dseg": witness_total,
            "witness_lane_contribution_19pct": witness_lane,
            "parametric_over_witness_ratio": float(rp_attr.mean() / witness_lane),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }

    print("\n=== ROAD-PLANE fit (lateral=poly(forward), the FEED-dm component) ===")
    print(f"  lane_attributable: mean {rp_attr.mean():.6f}  median {np.median(rp_attr):.6f}  "
          f"p90 {np.percentile(rp_attr,90):.6f}  max {rp_attr.max():.6f}")
    print(f"  lane_fn (shape) {np.mean(rp_fn):.6f}   lane_fp {np.mean(rp_fp):.6f}")
    print(f"  floats/frame {fpp_rp:.1f}  lines/frame {np.mean(n_lines):.1f}")
    print("\n=== IMAGE-COORDS fit (u=poly(v), comparison arm) ===")
    print(f"  lane_attributable mean {ic_attr.mean():.6f}   fn {np.mean(ic_fn):.6f}  "
          f"fp {np.mean(ic_fp):.6f}  floats/frame {np.mean(floats_ic):.1f}")
    print("\n=== MOTION (the FEED-el per-frame-moving reactivation test) ===")
    print(f"  band IoU frame-to-frame: mean {band_iou.mean():.3f}  min {band_iou.min():.3f}")
    print(f"  curvature: mean {curvature.mean():.4f} max {curvature.max():.4f}  "
          f"frac-frames-curved {(curvature>1e-6).mean():.2f}")
    print(f"  corr(curvature, error) {corr_curv:+.3f}  "
          f"(near 0 => fit robust to curvature/motion)")
    print("\n=== BYTES (rule-118: rasterizer FREE; coeffs COUNTED) ===")
    print(f"  raw ~{round(raw_bytes)} B  AR-estimate ~{round(ar_bytes)} B  "
          f"rate-term ~{25*ar_bytes/37_545_489:.6f}")
    print("\n=== VERDICT REFERENCE ===")
    print(f"  parametric lane-attributable {rp_attr.mean():.6f}  vs  witness lane "
          f"contribution {witness_lane:.6f}  -> ratio {rp_attr.mean()/witness_lane:.2f}x")

    jp = Path(args.json_out)
    if "/tmp" in str(jp):
        raise SystemExit("refuse /tmp json-out")
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(out, indent=2))
    print(f"\n[json] {jp}\n[done] {out['elapsed_s']}s [macOS-CPU advisory]; pointer UNMOVED 0.19110")


if __name__ == "__main__":
    main()
