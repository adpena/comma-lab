#!/usr/bin/env python
"""Measure the ANISOTROPY MAP of the SegNet argmax partition — $0, memory-light.

Loads the cached gt argmax labels + margin field (NO SegNet/PoseNet forward, NO model
inference) and computes the per-edge / per-saddle / temporal anisotropy d_H = log(lam_max/
lam_min) of the local margin structure tensor, ranking WHERE the geometry+factorization+
SPD-cone treatment (proven on pose, tested on lane) has leverage beyond the lane.

Authority: ``[macOS-MLX advisory]`` — geometric map on cached argmax; NOT byte-closed; moves
no score. Pointer UNMOVED.

Usage::

    .venv/bin/python tools/probe_partition_anisotropy_map.py \
        --cache experiments/results/mlx_fleet_gt_cache/gt_n96.npz \
        --out experiments/results/partition_anisotropy_map_20260710/probe.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tac.boundary_math.ground_frame_chart import intrinsics_for_grid
from tac.boundary_math.partition_anisotropy_map import compute_anisotropy_map


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache",
        default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz",
        help="npz with lstars (T,H,W) + margins (T,H,W)",
    )
    ap.add_argument("--out", default="experiments/results/partition_anisotropy_map_20260710/probe.json")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames in the cache")
    ap.add_argument("--sigma", type=float, default=2.0, help="structure-tensor / Hessian Gaussian sigma")
    ap.add_argument("--saddle-radius", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    d = np.load(args.cache)
    L = d["lstars"]
    M = d["margins"]
    T, H, W = L.shape
    max_frames = None if args.max_frames <= 0 else args.max_frames

    # vanishing point for the ego-radial/tangential temporal split: principal point (u=cx,v=cy)
    K = intrinsics_for_grid(W, H)
    vp_uv = (float(K[0, 2]), float(K[1, 2]))

    amap = compute_anisotropy_map(
        L,
        M,
        sigma_st=args.sigma,
        sigma_hess=args.sigma,
        saddle_radius=args.saddle_radius,
        vp_uv=vp_uv,
        max_frames=max_frames,
    )
    out = amap.to_json()
    out["meta"]["cache"] = str(args.cache)
    out["meta"]["vanishing_point_uv"] = list(vp_uv)
    out["meta"]["intrinsics_cx_cy"] = [float(K[0, 2]), float(K[1, 2])]
    out["meta"]["wall_seconds"] = round(time.time() - t0, 1)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))

    # --- human-readable summary ---
    print(f"# Partition anisotropy map  ({args.cache}, T={amap.meta['n_frames']} frames)")
    print(f"# class order: {amap.class_order['order']}")
    print()
    print("EDGES (ranked by leverage = dH_mean x pixel_share):")
    print(f"  {'edge':22s} {'dH_mean':>8s} {'dH_ewt':>8s} {'share':>7s} {'leverage':>9s}  {'n_cracks':>9s}")
    for e in sorted(amap.edges, key=lambda x: -x["leverage_dH_x_share"]):
        print(
            f"  {e['name']:22s} {e['dH_mean']:8.3f} {e['dH_energy_weighted']:8.3f} "
            f"{e['pixel_share']:7.3f} {e['leverage_dH_x_share']:9.4f}  {e['n_cracks']:9d}"
        )
    print()
    s = amap.saddles
    print("SADDLES (triple junctions):")
    print(f"  per-frame mean: {s['triple_junctions_per_frame_mean']:.1f}")
    print(f"  hyperbolic (mixed-sign Hessian, neighborhood) confirmed fraction: {s['hyperbolic_fraction']:.3f}")
    es = s.get("eigenstructure", {})
    if es.get("n_junctions_sampled"):
        print(
            f"  margin-Hessian AT junction: lam1_med={es['hess_lam1_median']:.3f} "
            f"lam2_med={es['hess_lam2_median']:.3f}  mixed-sign_frac={es['mixed_sign_fraction']:.3f}"
        )
        print(
            f"  Hessian isotropy |lam_min|/|lam_max| median (among mixed)={es['hess_isotropy_ratio_median']:.3f} "
            f"(->1 = genuine 2D); struct-tensor d_H at saddles median={es['structure_tensor_dH_at_saddles_median']:.3f}"
        )
        print(
            f"  ROUTING: directionally-codeable={es['frac_directionally_codeable']:.2f}  "
            f"genuine-2D-hyperbolic (needs saddle-aware code, #1 lever FAILS)={es['frac_genuine_2d_hyperbolic']:.2f}"
        )
    print(
        f"  hard-mass concentration ratio (low-margin near saddle / overall): "
        f"{s['hard_mass_concentration_ratio']:.2f}x"
    )
    print()
    h = amap.horizon
    print("HORIZON (Road<->Undrivable boundary line fit):")
    if h.get("fit_ok"):
        print(
            f"  v_at_center_row_mean={h.get('v_at_center_row_mean', h.get('v_at_center_row')):.1f}"
            f"  (intrinsics cy={out['meta']['intrinsics_cx_cy'][1]:.1f})"
        )
        print(
            f"  slope_mean={h.get('slope_mean', h.get('slope')):.4f}"
            f"  residual_rows_rms_mean={h.get('residual_rows_rms_mean', h.get('residual_rows_rms')):.2f}"
            f"  coverage={h.get('coverage', 0):.2f}"
        )
    print()
    tt = amap.temporal
    print("TEMPORAL (spatio-temporal structure tensor over boundary pixels):")
    print(f"  dH_spatiotemporal={tt['dH_spatiotemporal']:.3f}  eigs={['%.3g' % x for x in tt['eigs']]}")
    if tt.get("radial_tangential"):
        rt = tt["radial_tangential"]
        print(f"  ego radial/tangential energy ratio={rt['radial_over_tangential']:.3f}")
    print()
    print(f"wrote {outp}  ({out['meta']['wall_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
