#!/usr/bin/env python3
"""$0 MEASUREMENT DRIVER for the v8 residual-kit (de-share Lever-1 + curve-relative Lever-2).

Runs the two recess probes the P8 rate row needs, on the frozen SegNet-argmax cache (n600):
  * PROBE 1 (de-share magnitude): ``movable_deshare.measure_deshare_magnitude`` -> per-edge bytes
    double-counted (Movable-first attribution) + the general pairwise ARCHIVE-DEDUP AUDIT
    (operator 2026-07-09 no-duplicate-data binding).
  * PROBE 2 (curve-relative δ(s)): per-edge curve-relative coded bytes vs the absolute-2-D baseline,
    + the δ(s) offset entropy / Haar N-term, with a bit-exact roundtrip assertion per edge.

Read-only on ``gt_n600.npz['lstars']``; pure numpy/scipy/brotli; NO GPU, NO training; pointer
0.19110 UNMOVED.  Emits a small JSON report (durable, ``.omx/research/``).  Axis label travels with
every number: ``[macOS-CPU advisory · NON-PROMOTABLE]`` -- a measurement moves no pointer.

Run:
  PYTHONPATH=src:upstream:$PWD .venv/bin/python \
      tools/measure_v8_residual_kit_deshare_curverel.py \
      --cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
      --out .omx/research/residual_kit_measured_20260709.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from tac.boundary_math import curve_relative_offset_coder as C
from tac.boundary_math import movable_deshare as M


def _build_edges(lstars: np.ndarray, roles: M.SegRoles, *, deshare: bool):
    """Per-frame residual sets + generator curves for the two curve-relative edges.

    ``deshare=True`` subtracts the Movable footprint FIRST (Lever-1 composes with Lever-2): the
    curve-relative coder should see the near-curve poly-fit wiggle, NOT the far Movable secondary
    arcs that Lever-1 attributes to G3.
    """
    from tac.boundary_math.analytic_lane_render_band import build_analytic_lane_band_prior

    n, h, w = lstars.shape
    hz_resid, hz_curves = [], []
    ln_resid, ln_curves = [], []
    for i in range(n):
        lst = lstars[i]
        fp = M.movable_footprint(lst, roles.movable, dilate=2).reshape(-1) if deshare else None
        # horizon (Road<->Undriv): column-param curve = the deg-3 poly rows
        yrows = M._horizon_poly_rows(lst, roles.road, roles.undriv)
        hz_curves.append([C.curve_from_column_function(yrows, seg_id=0)])
        hz = M.horizon_residual_idx(lst, roles)
        if fp is not None and hz.size:
            hz = hz[~fp[hz]]
        hz_resid.append(hz)
        # lane (Road<->Lane): row-param curves from the analytic band coverage
        cov = build_analytic_lane_band_prior(lst, lane_cls=roles.lane).coverage
        ln_curves.append(C.curves_from_coverage_mask(cov >= 0.5, axis="row", min_len=8))
        ln = M.lane_residual_idx(lst, roles, cov)
        if fp is not None and ln.size:
            ln = ln[~fp[ln]]
        ln_resid.append(ln)
    return (h, w), (hz_resid, hz_curves), (ln_resid, ln_curves)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--out", default=".omx/research/residual_kit_measured_20260709.json")
    ap.add_argument("--n-frames", type=int, default=0, help="0 = all frames in the cache")
    args = ap.parse_args()

    t0 = time.time()
    lstars = np.load(args.cache)["lstars"]
    if args.n_frames:
        lstars = lstars[: args.n_frames]
    n = lstars.shape[0]
    roles = M.detect_seg_roles(lstars)
    print(f"[residual-kit] n={n} roles={roles.as_dict()}  ({time.time()-t0:.1f}s)", flush=True)

    # ---- PROBE 1: de-share magnitude + general pairwise dedup audit ----
    deshare = M.measure_deshare_magnitude(lstars, seed=0, roles=roles)
    print(f"[probe-1 de-share] total_bytes_double_counted={deshare['total_bytes_double_counted']} "
          f"S={deshare['total_S_deflation']:.5f}  ({time.time()-t0:.1f}s)", flush=True)
    dedup = M.pairwise_dedup_audit(lstars, roles=roles)
    print(f"[probe-1b dedup-audit] top pair: {dedup['pairs'][0]['row_a']}<->{dedup['pairs'][0]['row_b']} "
          f"= {dedup['pairs'][0]['overlap_bytes_double_counted']} B  ({time.time()-t0:.1f}s)", flush=True)

    # ---- PROBE 2: curve-relative vs absolute-2-D baseline, per edge ----
    # Levers COMPOSE: measure on the DE-SHARED residual (Lever-1 removes far Movable arcs first) as
    # primary; also measure on the raw residual for the composition delta.
    probe2 = {}
    for mode, ds in (("deshared", True), ("raw", False)):
        (h, w), (hz_resid, hz_curves), (ln_resid, ln_curves) = _build_edges(lstars, roles, deshare=ds)
        crh = C.measure_curve_relative(hz_resid, hz_curves, h, w, edge_name="horizon_Road_Undriv")
        crl = C.measure_curve_relative(ln_resid, ln_curves, h, w, edge_name="lane_Road_Lane")
        probe2[mode] = {"horizon": crh, "lane": crl}
        for nm, cr in (("horizon", crh), ("lane", crl)):
            sp = cr["delta_s_spectrum"]
            print(f"[probe-2 {mode:8s} {nm:7s}] curve-rel {cr['bytes_curve_relative']} B vs abs "
                  f"{cr['bytes_absolute_2d_baseline']} B  ratio={cr['savings_ratio']:.2f} "
                  f"|n|max={sp['offset_abs_max_px']} |n|mean={sp['offset_abs_mean_px']:.2f} "
                  f"H(n)={sp['offset_alphabet_entropy_bits']:.2f}b Nterm={sp['haar_nterm_frac_mean']:.2f} "
                  f"on_curve={cr['frac_on_curve']:.2f} bit_exact={cr['curve_relative_bit_exact']} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    report = {
        "n_frames": int(n),
        "roles": roles.as_dict(),
        "elapsed_sec": round(time.time() - t0, 1),
        "axis_label": "[macOS-CPU advisory · NON-PROMOTABLE]",
        "pointer": "0.19110 UNMOVED",
        "probe1_deshare": deshare,
        "probe1b_pairwise_dedup_audit": dedup,
        "probe2_curve_relative": probe2,
    }
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    print(f"[residual-kit] wrote {outp}  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
