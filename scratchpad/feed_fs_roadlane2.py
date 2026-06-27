"""FEED-fs measurement 2 — which CHEAP per-row width model recovers the oracle ribbon?

Mtmt-1 found: global half-width scale is already near-optimum (0.000439 @ scale 1.0);
the per-row EXACT ribbon (oracle) hits 0.000019. So the residual is per-row WIDTH MODELING.
This script prices the recovery:
  - perspective-physics width hw(v)=k*(v-v_h)  [1 float/line; lane width ~const in road-plane]
  - free width poly_2 / poly_3 in v            [3-4 floats/line]
  - centerline-OFFSET isolation: grow oracle per-row width but SYMMETRIC around the
    PARAMETRIC centerline (isolates width-model error from centerline-offset error)
  - finer clustering (gap_m 0.4) to test merged-line over-paint
  - centerline deg 4
All vs the IDEAL SDF stack (argmax==L* exact), n96, frozen CPU-torch L*. NO-FAKE.
"""
from __future__ import annotations
import copy, json, time
from pathlib import Path
import numpy as np

from tac.boundary_math.lane_sdf_component import (
    cluster_lane_lines, fit_lane_line, rasterize_lane_band, lane_signed_distance,
    inject_lane_sdf, decompose_argmax_disagreement, ground_to_image_row,
    _SEG_H, _SEG_W, _V_HORIZON, _CAM_H, _FY,
)
from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields

_GT = Path("experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
_N, _LANE, _ROAD = 5, 1, 0
_ROWS = np.arange(_SEG_H, dtype=np.float64)
_BELOW = _ROWS > (_V_HORIZON + 1.0)
_VR = _ROWS[_BELOW]


def band_from_hwfun(lines, hw_fun):
    """hw_fun(line, vr)->halfwidth per row; paint symmetric band around parametric centerline."""
    band = np.zeros((_SEG_H, _SEG_W), bool)
    for ln in lines:
        _, u_c = ground_to_image_row(_VR, ln.lateral_of_forward)
        hw = hw_fun(ln, _VR)
        f0, f1 = ln.forward_range
        fwd, _ = ground_to_image_row(_VR, ln.lateral_of_forward)
        in_rng = (fwd >= f0 - 1.0) & (fwd <= f1 + 5.0)
        for j, vv in enumerate(_VR):
            if not in_rng[j] or not np.isfinite(u_c[j]):
                continue
            lo = int(max(0, np.floor(u_c[j] - hw[j])))
            hi = int(min(_SEG_W, np.ceil(u_c[j] + hw[j]) + 1))
            if hi > lo:
                band[int(vv), lo:hi] = True
    return band


def fit_perspective_k(ln, L):
    """1-float perspective width: hw(v)=k*(v-v_h); k fit to the line's class-1 pixel spread."""
    # use the rasterized centerline residuals over true lane pixels near the line
    return None  # placeholder; computed inline below


def oracle_width_around_parametric(L, lines):
    """Per-row: grow the connected class-1 run that the PARAMETRIC centerline lands in/abuts,
    but only SYMMETRIC half-width = max(|center-left|,|center-right|)?  No -- paint the run
    (same as oracle) BUT keyed to the parametric centerline (isolates centerline quality)."""
    band = np.zeros((_SEG_H, _SEG_W), bool)
    is_lane = (L == _LANE)
    for ln in lines:
        _, u_c = ground_to_image_row(_VR, ln.lateral_of_forward)
        for j, vv in enumerate(_VR):
            vi = int(vv); uc = u_c[j]
            if not np.isfinite(uc):
                continue
            u0 = int(round(uc))
            if u0 < 0 or u0 >= _SEG_W:
                continue
            row = is_lane[vi]
            lo = max(0, u0 - 12); hi = min(_SEG_W, u0 + 13)
            local = np.where(row[lo:hi])[0]
            if local.size == 0:
                continue
            cand = local + lo
            nearest = cand[np.argmin(np.abs(cand - u0))]
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
    lstars = np.load(_GT)["lstars"]
    P = min(96, lstars.shape[0])
    print(f"[FEED-fs m2] per-row width-model recovery; n={P}", flush=True)

    keys = ["persp_k", "wpoly2", "wpoly3", "centerdeg4_wpoly1",
            "oracle_param_center", "gap04_wpoly1"]
    acc = {k: {"fn": [], "fp": [], "attr": []} for k in keys}
    fl = {k: [] for k in keys}

    for i in range(P):
        L = np.asarray(lstars[i]).astype(np.int64)
        phi_ideal = signed_distance_fields(L, _N)
        clusters = cluster_lane_lines(L, lane_cls=_LANE)
        lines = [fit_lane_line(c, centerline_deg=3, fit_dash=False) for c in clusters]
        lines = [ln for ln in lines if ln is not None]
        if not lines:
            continue

        # --- per-line width fits from the centerline residual over the cluster pixels ---
        # recompute residual |u - u_c(v)| for the cluster's pixels to fit width models
        lines_w2, lines_w3, lines_pk = [], [], []
        for ln, c in zip(lines, clusters):
            v = c[:, 0].astype(np.float64); u = c[:, 1].astype(np.float64)
            _, uc = ground_to_image_row(v, ln.lateral_of_forward)
            resid = np.abs(u - uc)
            ok = np.isfinite(resid)
            v, resid = v[ok], resid[ok]
            # perspective k: hw = k*(v - v_h) ; k = median(resid/(v-v_h))
            dv = np.maximum(v - _V_HORIZON, 1.0)
            k = float(np.clip(np.median(resid / dv), 1e-4, 0.2))
            lp = copy.copy(ln); lp._pk = k; lines_pk.append(lp)
            # width poly_2 / poly_3 of the per-row 90th-pct residual
            vr2 = np.round(v).astype(np.int64); rows = np.unique(vr2)
            l2 = copy.copy(ln); l3 = copy.copy(ln)
            if rows.size >= 4:
                hwr = np.array([np.percentile(resid[vr2 == r], 90) for r in rows])
                hwr = np.clip(hwr, 0.5, 20.0)
                l2._wc = np.polyfit(rows.astype(float), hwr, min(2, rows.size - 1))
                l3._wc = np.polyfit(rows.astype(float), hwr, min(3, rows.size - 1))
            else:
                med = float(np.clip(np.median(resid) + 0.5, 0.5, 8.0))
                l2._wc = np.array([med]); l3._wc = np.array([med])
            lines_w2.append(l2); lines_w3.append(l3)

        # persp_k
        b = band_from_hwfun(lines_pk, lambda ln, vr: np.maximum(ln._pk * (vr - _V_HORIZON), 0.5))
        d = attrib(b, phi_ideal, L)
        acc["persp_k"]["fn"].append(d.lane_fn); acc["persp_k"]["fp"].append(d.lane_fp_from_road + d.lane_fp_from_other); acc["persp_k"]["attr"].append(d.lane_attributable)
        fl["persp_k"].append(sum(len(ln.centerline_coeffs) + 1 for ln in lines_pk))
        # wpoly2
        b = band_from_hwfun(lines_w2, lambda ln, vr: np.maximum(np.polyval(ln._wc, vr), 0.5))
        d = attrib(b, phi_ideal, L)
        acc["wpoly2"]["fn"].append(d.lane_fn); acc["wpoly2"]["fp"].append(d.lane_fp_from_road + d.lane_fp_from_other); acc["wpoly2"]["attr"].append(d.lane_attributable)
        fl["wpoly2"].append(sum(len(ln.centerline_coeffs) + len(ln._wc) for ln in lines_w2))
        # wpoly3
        b = band_from_hwfun(lines_w3, lambda ln, vr: np.maximum(np.polyval(ln._wc, vr), 0.5))
        d = attrib(b, phi_ideal, L)
        acc["wpoly3"]["fn"].append(d.lane_fn); acc["wpoly3"]["fp"].append(d.lane_fp_from_road + d.lane_fp_from_other); acc["wpoly3"]["attr"].append(d.lane_attributable)
        fl["wpoly3"].append(sum(len(ln.centerline_coeffs) + len(ln._wc) for ln in lines_w3))

        # centerline deg4 + width poly1 (baseline width)
        lines4 = [fit_lane_line(c, centerline_deg=4, fit_dash=False) for c in clusters]
        lines4 = [ln for ln in lines4 if ln is not None]
        b = rasterize_lane_band(lines4, dash_gate=False)
        d = attrib(b, phi_ideal, L)
        acc["centerdeg4_wpoly1"]["fn"].append(d.lane_fn); acc["centerdeg4_wpoly1"]["fp"].append(d.lane_fp_from_road + d.lane_fp_from_other); acc["centerdeg4_wpoly1"]["attr"].append(d.lane_attributable)
        fl["centerdeg4_wpoly1"].append(sum(ln.n_floats() for ln in lines4))

        # oracle width but PARAMETRIC centerline (isolate centerline offset)
        b = oracle_width_around_parametric(L, lines)
        d = attrib(b, phi_ideal, L)
        acc["oracle_param_center"]["fn"].append(d.lane_fn); acc["oracle_param_center"]["fp"].append(d.lane_fp_from_road + d.lane_fp_from_other); acc["oracle_param_center"]["attr"].append(d.lane_attributable)
        fl["oracle_param_center"].append(0)

        # finer clustering gap 0.4 + width poly1
        cl4 = cluster_lane_lines(L, lane_cls=_LANE, gap_m=0.4)
        ln4 = [fit_lane_line(c, centerline_deg=3, fit_dash=False) for c in cl4]
        ln4 = [x for x in ln4 if x is not None]
        b = rasterize_lane_band(ln4, dash_gate=False)
        d = attrib(b, phi_ideal, L)
        acc["gap04_wpoly1"]["fn"].append(d.lane_fn); acc["gap04_wpoly1"]["fp"].append(d.lane_fp_from_road + d.lane_fp_from_other); acc["gap04_wpoly1"]["attr"].append(d.lane_attributable)
        fl["gap04_wpoly1"].append(sum(x.n_floats() for x in ln4))

        if (i + 1) % 32 == 0:
            print(f"  ... {i+1}/{P} ({time.time()-t0:.1f}s)", flush=True)

    print("\n=== FEED-fs m2 (mean n96) ===")
    print(f"{'config':<22}{'FN':>10}{'FP':>10}{'ATTR':>10}{'floats/fr':>11}")
    res = {}
    for k in keys:
        r = {"fn": float(np.mean(acc[k]["fn"])), "fp": float(np.mean(acc[k]["fp"])),
             "attr": float(np.mean(acc[k]["attr"])), "floats": float(np.mean(fl[k]))}
        res[k] = r
        print(f"{k:<22}{r['fn']:>10.6f}{r['fp']:>10.6f}{r['attr']:>10.6f}{r['floats']:>11.1f}")
    print(f"\n(ref: FEED-er baseline wpoly1 0.000439; oracle exact 0.000019; witness lane 0.000236)")
    Path("experiments/results/lane_sdf_FEED-er/feed_fs_width_model2.json").write_text(
        json.dumps({"feed": "FEED-fs-m2", "n": P, "results": res,
                    "authority": "macOS-CPU advisory"}, indent=2))
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
