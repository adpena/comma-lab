#!/usr/bin/env python3
"""$0 n600 measurement of the lane-IPM v_horizon reconciliation (task #327, read-only).

Audit #326 found `lane_sdf_component._V_HORIZON=174.0` is the SALIENCY-VP row, wrongly
reused as the lane-IPM ground-projection horizon. openpilot geometric horizon = 192
(cy·384/874), module's own note says 188 IPM-optimal. v_h enters the analytic lane band
via the rasterization cutoff `below = rows > v_h + 1` — 174 paints ~18 extra near-horizon
rows that are far-field, where lane lines are thin/uncertain (the "whole enemy" = far-field
FALSE POSITIVES). This turns the INFERRED top d_seg lever into a MEASURED verdict.

Method (n600, real cached GT SegNet argmax, NO training, NO GPU): for each candidate v_h,
build the analytic lane band from each GT argmax frame (fit lane lines → AA-SDF raster) and
score the band>=0.5 mask against the true GT lane mask (class 1):
  recall     = |band ∧ lane| / |lane|              (fit quality — must NOT drop)
  precision  = |band ∧ lane| / |band|
  FP_far     = |band ∧ ¬lane ∧ rows∈[174,200]| / frame   (the near-horizon far-field FP)
  FP_total   = |band ∧ ¬lane| / frame
  band_err   = |band ⊕ lane| / frame               (symmetric proxy for lane d_seg cost)
The winner CUTS FP_far while HOLDING recall. cam_h∈{1.2,1.22} is 2nd-order (fit↔render
cancels it except in dash-distance gating) — measured via module monkeypatch for honesty.

NOTE (round-trip caveat, audit #326): v_h's effect is NOT band placement (fit and render
use the same v_h so most of it cancels) — it is the raster CUTOFF + poly reparam + dash
scale. This measurement captures exactly that (band built AND scored at each v_h).

Advisory only: `[macOS-CPU advisory]`, NON-PROMOTABLE, no score claim (this is a lane-prior
quality proxy vs the frozen GT argmax, NOT an exact upstream/evaluate.py row).

Run: .venv/bin/python tools/measure_lane_ipm_vhorizon_reconciliation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tac.boundary_math import analytic_lane_render_band as alrb
from tac.boundary_math import lane_sdf_component as lsc

_REPO = Path(__file__).resolve().parents[1]
_CACHE = _REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
_OUT = _REPO / "experiments/results/lane_ipm_vhorizon_reconciliation"
LANE = 1
FARFIELD_ROWS = (174, 200)   # the near-horizon zone 174 paints and 188/192 do not


def _score_frame(lstar: np.ndarray, v_h: float) -> dict:
    prior = alrb.build_analytic_lane_band_prior(lstar, lane_cls=LANE, v_h=v_h)
    band = prior.coverage >= 0.5
    lane = lstar == LANE
    inter = band & lane
    nlane = int(lane.sum())
    nband = int(band.sum())
    size = float(lstar.size)
    rows = np.arange(lstar.shape[0])[:, None]
    farzone = (rows >= FARFIELD_ROWS[0]) & (rows < FARFIELD_ROWS[1])
    fp = band & ~lane
    return {
        "recall": float(inter.sum() / nlane) if nlane else float("nan"),
        "precision": float(inter.sum() / nband) if nband else float("nan"),
        "fp_far_frac": float((fp & farzone).sum() / size),
        "fp_total_frac": float(fp.sum() / size),
        "band_err_frac": float((band ^ lane).sum() / size),
        "band_area_frac": float(nband / size),
        "has_lane": 1.0 if nlane else 0.0,
    }


def _sweep_vh(lstars: np.ndarray, v_list: list[float]) -> list[dict]:
    n = lstars.shape[0]
    out = []
    for v_h in v_list:
        rows = [_score_frame(lstars[i], v_h) for i in range(n)]
        keys = [k for k in rows[0] if k != "has_lane"]
        agg = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
        agg["v_h"] = v_h
        agg["frames_with_lane"] = int(sum(r["has_lane"] for r in rows))
        out.append(agg)
    return out


def main() -> int:
    z = np.load(_CACHE)
    lstars = np.asarray(z["lstars"])
    n = lstars.shape[0]
    res: dict = {"n": n, "farfield_rows": list(FARFIELD_ROWS),
                 "axis_tag": "[macOS-CPU advisory]", "promotable": False,
                 "note": "lane-prior quality vs frozen GT argmax, NOT an exact evaluate.py row"}

    # ── PRIMARY: v_horizon sweep (cam_h at its current default) ──────────────
    print(f"=== v_horizon sweep (n={n}, lane class {LANE}, far-field rows {FARFIELD_ROWS}) ===")
    print(f"  {'v_h':>5} {'recall':>8} {'precis':>8} {'FP_far':>9} {'FP_tot':>9} {'band_err':>9} {'band_area':>10}")
    vh_rows = _sweep_vh(lstars, [174.0, 188.0, 192.0])
    for r in vh_rows:
        print(f"  {r['v_h']:>5.0f} {r['recall']:>8.4f} {r['precision']:>8.4f} "
              f"{r['fp_far_frac']:>9.5f} {r['fp_total_frac']:>9.5f} {r['band_err_frac']:>9.5f} "
              f"{r['band_area_frac']:>10.5f}")
    res["vh_sweep"] = vh_rows

    # verdict: pick min band_err among v_h that hold recall within 1% of the best recall
    best_recall = max(r["recall"] for r in vh_rows)
    eligible = [r for r in vh_rows if r["recall"] >= best_recall - 0.01]
    winner = min(eligible, key=lambda r: r["band_err_frac"])
    base = next(r for r in vh_rows if r["v_h"] == 174.0)
    res["verdict"] = {
        "winner_v_h": winner["v_h"],
        "recall_hold": f"{winner['recall']:.4f} vs 174={base['recall']:.4f}",
        "fp_far_cut": f"{base['fp_far_frac']:.5f}->{winner['fp_far_frac']:.5f} "
                      f"({100*(1-winner['fp_far_frac']/base['fp_far_frac']):.0f}% cut)" if base['fp_far_frac'] > 0 else "n/a",
        "band_err": f"{base['band_err_frac']:.5f}->{winner['band_err_frac']:.5f}",
    }
    print(f"\n  VERDICT: v_h={winner['v_h']:.0f} (recall held within 1% of best; min band_err). "
          f"FP_far {res['verdict']['fp_far_cut']}")

    # ── SECONDARY: cam_h 1.2 vs 1.22 at the winning v_h (2nd-order, monkeypatch) ──
    print(f"\n=== cam_h 1.2 vs 1.22 at v_h={winner['v_h']:.0f} (2nd-order: fit<->render cancels; dash-gate only) ===")
    cam_rows = []
    orig_a, orig_l = alrb._CAM_H, lsc._CAM_H
    try:
        for cam_h in (1.2, 1.22):
            alrb._CAM_H = cam_h
            lsc._CAM_H = cam_h
            rows = [_score_frame(lstars[i], winner["v_h"]) for i in range(n)]
            keys = [k for k in rows[0] if k != "has_lane"]
            agg = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
            agg["cam_h"] = cam_h
            cam_rows.append(agg)
            print(f"  cam_h={cam_h}: recall={agg['recall']:.4f} FP_far={agg['fp_far_frac']:.5f} "
                  f"band_err={agg['band_err_frac']:.5f}")
    finally:
        alrb._CAM_H, lsc._CAM_H = orig_a, orig_l
    res["cam_h_sweep"] = cam_rows
    d_err = abs(cam_rows[0]["band_err_frac"] - cam_rows[1]["band_err_frac"])
    res["cam_h_verdict"] = (f"cam_h effect on band_err = {d_err:.2e} "
                            f"({'2nd-order as predicted' if d_err < 1e-4 else 'NON-trivial — investigate'}); "
                            "fix 1.2->1.22 for frame consistency regardless")
    print(f"  cam_h band_err delta = {d_err:.2e} ({res['cam_h_verdict'].split(';')[0].split('=')[-1].strip()})")

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "result.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {_OUT / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
