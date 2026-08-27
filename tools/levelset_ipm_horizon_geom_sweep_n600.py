#!/usr/bin/env python
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""$0 GEOMETRY-ONLY n600 IPM-horizon sweep (de-risk (a) T0 SCREEN for #205 Phase-3).

[macOS-CPU advisory] NON-PROMOTABLE. Question (seeding audit R1): does moving the
lane-IPM vanishing-point horizon v_h 174 (shipped) -> 188 (FEED-dj) IMPROVE the
analytic-lane band's geometric placement (band-vs-GT-lane recall UP, band FP frac
DOWN) at the config's dash_forward_max_m=55? This is the DIRECT mechanism the audit
names (horizon sets the image-row->forward-meter map the dash-gate uses; the dash-gap
FP is 90% of the recon lane d_seg). Pure geometry (build_analytic_lane_band_prior),
NO render, NO SegNet, NO R -> genuinely $0. If 188 does NOT beat 174 geometrically,
the ~2h realized render+SegNet sweep is MOOT (don't pin 188). If it DOES, the realized
confirm becomes worth its cost + a module/flag edit to APPLY.

n600 (all 600 pairs, real gt lstars). dash_forward_max_m=55 (config value)."""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))
sys.path.insert(0, str(REPO))
import numpy as np
from tac.boundary_math.analytic_lane_render_band import build_analytic_lane_band_prior

CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
HORIZONS = [174.0, 182.0, 188.0, 194.0]
DASH_FWD = 55.0
LANE_CLS = 1

def main():
    t0 = time.time()
    z = np.load(CACHE, allow_pickle=False)
    lst_all = z["lstars"]
    P = int(z["n_pairs"])
    print(f"[{time.time()-t0:.1f}s] loaded lstars P={P} shape[0]={lst_all[0].shape if hasattr(lst_all[0],'shape') else 'na'}", flush=True)

    res = {vh: {"recall": np.full(P, np.nan), "fp": np.full(P, np.nan),
                "nlines": np.full(P, np.nan)} for vh in HORIZONS}
    for pi in range(P):
        lst = np.asarray(lst_all[pi], np.int64)
        is_lane = lst == LANE_CLS
        nlane = int(is_lane.sum())
        ntot = float(lst.size)
        for vh in HORIZONS:
            pr = build_analytic_lane_band_prior(
                lst, lane_cls=LANE_CLS, softness=1.0, dash_gate=True,
                dash_forward_max_m=DASH_FWD, centerline_deg=3, v_h=vh)
            band = pr.coverage >= 0.5
            res[vh]["recall"][pi] = (float((band & is_lane).sum()) / nlane) if nlane else np.nan
            res[vh]["fp"][pi] = float((band & (~is_lane)).sum()) / ntot
            res[vh]["nlines"][pi] = pr.n_lines
        if (pi + 1) % 100 == 0:
            print(f"[{time.time()-t0:.1f}s] {pi+1}/{P}", flush=True)

    print(f"\n[{time.time()-t0:.1f}s] === n600 IPM-horizon GEOMETRY sweep (dash_fwd={DASH_FWD}) ===", flush=True)
    print(f"{'v_h':>6} {'band_recall':>12} {'band_fp_frac':>13} {'n_lines':>9}", flush=True)
    base = None
    for vh in HORIZONS:
        r = float(np.nanmean(res[vh]["recall"]))
        fp = float(np.nanmean(res[vh]["fp"]))
        nl = float(np.nanmean(res[vh]["nlines"]))
        tag = ""
        if base is None:
            base = (r, fp)
        else:
            tag = f"  dRecall={r-base[0]:+.4f} dFP={fp-base[1]:+.6f}"
        print(f"{vh:>6.0f} {r:>12.4f} {fp:>13.6f} {nl:>9.3f}{tag}", flush=True)
    # A better horizon => recall UP and fp DOWN vs 174.
    r174, fp174 = float(np.nanmean(res[174.0]["recall"])), float(np.nanmean(res[174.0]["fp"]))
    r188, fp188 = float(np.nanmean(res[188.0]["recall"])), float(np.nanmean(res[188.0]["fp"]))
    print(f"\nVERDICT 188-vs-174: dRecall={r188-r174:+.4f}  dFP={fp188-fp174:+.6f}", flush=True)
    better = (r188 >= r174 - 1e-4) and (fp188 <= fp174 + 1e-6)
    print(f"188 geometrically >= 174 (recall not-worse AND fp not-worse): {better}", flush=True)

if __name__ == "__main__":
    main()
