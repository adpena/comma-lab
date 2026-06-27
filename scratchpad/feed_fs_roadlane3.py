"""FEED-fs measurement 3 — composition grounding:
 (A) does the openpilot parametric centerline TANGENT field match the actual Road<->Lane
     boundary normal (i.e. does it GROUND the directional curvelet orientation, the -48% lever)?
 (B) per-row extent residual entropy: is storing the per-row width cheaper than learning it?
     (confirms the witness must LEARN the high-freq boundary placement.)
n96, frozen CPU-torch L*. NO-FAKE (real boundary pixels, real L*).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

from tac.boundary_math.lane_sdf_component import (
    cluster_lane_lines, fit_lane_line, ground_to_image_row, _SEG_H, _SEG_W, _V_HORIZON,
)

_GT = Path("experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
_LANE = 1


def main():
    t0 = time.time()
    lstars = np.load(_GT)["lstars"]
    P = min(96, lstars.shape[0])
    print(f"[FEED-fs m3] tangent grounding + extent entropy; n={P}", flush=True)

    align_cos = []           # |cos angle| between centerline-perp and actual boundary normal
    extent_resid_px = []     # per-row deviation of true ribbon edge from smooth-band edge (px)
    vert_grad_frac = []      # fraction of lane-boundary edges that are ~horizontal (along tangent)

    for i in range(P):
        L = np.asarray(lstars[i]).astype(np.int64)
        is_lane = (L == _LANE)
        if is_lane.sum() < 50:
            continue
        clusters = cluster_lane_lines(L, lane_cls=_LANE)
        lines = [fit_lane_line(c, centerline_deg=3, fit_dash=False) for c in clusters]
        lines = [ln for ln in lines if ln is not None]
        if not lines:
            continue

        # boundary normal of the lane region (grad of the binary mask)
        gy, gx = np.gradient(is_lane.astype(np.float64))
        gmag = np.hypot(gx, gy)
        bnd = gmag > 1e-6
        bv, bu = np.where(bnd)
        # for each boundary pixel assign to nearest line by centerline u at that row
        rows = np.arange(_SEG_H, dtype=np.float64)
        for ln in lines:
            _, uc_all = ground_to_image_row(rows, ln.lateral_of_forward)
            # centerline tangent in image: d u_c / d v  -> tangent (1, du/dv); perp = (-du/dv,1) norm
            duv = np.gradient(uc_all)
            for (vv, uu) in zip(bv, bu):
                if not np.isfinite(uc_all[vv]):
                    continue
                if abs(uu - uc_all[vv]) > 6:   # only boundary pixels near THIS line
                    continue
                # actual boundary normal at (vv,uu)
                n = np.array([gx[vv, uu], gy[vv, uu]])
                if np.linalg.norm(n) < 1e-6:
                    continue
                n = n / np.linalg.norm(n)
                # centerline perp (the lane-marking normal predicted by openpilot poly)
                t = np.array([duv[vv], 1.0]); t = t / np.linalg.norm(t)
                perp = np.array([-t[1], t[0]])
                align_cos.append(abs(float(np.dot(n, perp))))
                vert_grad_frac.append(1.0 if abs(n[0]) > abs(n[1]) else 0.0)

        # extent residual: per row, true ribbon edge vs smooth (poly1) band edge
        for ln, c in zip(lines, clusters):
            v = c[:, 0].astype(np.float64); u = c[:, 1].astype(np.float64)
            _, uc = ground_to_image_row(v, ln.lateral_of_forward)
            resid = u - uc                       # signed lateral offset of true lane px from centerline
            ok = np.isfinite(resid)
            v, resid = v[ok], resid[ok]
            vr = np.round(v).astype(np.int64)
            for r in np.unique(vr):
                rr = resid[vr == r]
                if rr.size == 0:
                    continue
                # true ribbon half-extent this row (right side)
                true_hw = float(np.max(np.abs(rr)))
                smooth_hw = float(np.maximum(np.polyval(ln.halfwidth_coeffs, r), 0.5))
                extent_resid_px.append(true_hw - smooth_hw)

        if (i + 1) % 32 == 0:
            print(f"  ... {i+1}/{P} ({time.time()-t0:.1f}s)", flush=True)

    align_cos = np.array(align_cos)
    er = np.array(extent_resid_px)
    vgf = np.array(vert_grad_frac)
    # bits estimate to store per-row extent residual (quantize to 1px, range needed)
    rng = float(np.percentile(np.abs(er), 99)) if er.size else 0.0
    bits_per_row = float(np.log2(2 * max(rng, 1.0) + 1))
    out = {
        "feed": "FEED-fs-m3", "n": P, "authority": "macOS-CPU advisory",
        "tangent_alignment_cos_mean": float(align_cos.mean()) if align_cos.size else None,
        "tangent_alignment_cos_median": float(np.median(align_cos)) if align_cos.size else None,
        "tangent_alignment_frac_above_0.9": float((align_cos > 0.9).mean()) if align_cos.size else None,
        "lane_boundary_horizontal_normal_frac": float(vgf.mean()) if vgf.size else None,
        "extent_resid_px_mean": float(er.mean()) if er.size else None,
        "extent_resid_px_std": float(er.std()) if er.size else None,
        "extent_resid_px_abs_p90": float(np.percentile(np.abs(er), 90)) if er.size else None,
        "extent_resid_bits_per_active_row": bits_per_row,
        "n_boundary_samples": int(align_cos.size), "n_extent_rows": int(er.size),
        "elapsed_s": time.time() - t0,
    }
    Path("experiments/results/lane_sdf_FEED-er/feed_fs_grounding3.json").write_text(json.dumps(out, indent=2))
    print("\n=== FEED-fs m3 ===")
    for k, v in out.items():
        if k not in ("feed", "n", "authority"):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
