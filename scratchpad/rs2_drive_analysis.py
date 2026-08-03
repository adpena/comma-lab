#!/usr/bin/env python
"""ddm_rs2 §1.6 — what the n600 DRIVE sweep says, against every incumbent key.

DRIVE is the half of flip damage that no existing key measures: how hard a cell's
drop actually pushes the SegNet's input plane.  This reads the sweep's output and
answers, at n600 and exactly:

  * the per-cell receptive field, measured 384 times instead of once (does the
    single-cell 84x82 anchor generalise?);
  * how DRIVE ranks against bytes, activity, and all three damage keys;
  * whether any live cell has DRIVE == 0 (an exact scorer-free zero-flip certificate);
  * how the two BUILT A/B arms compare on the DRIVE currency.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SSD = Path("/Volumes/VertigoDataTier/pact")
WORK = SSD / "ddm_rs2_20260803"
R, Cc, H, W_ = 24, 32, 384, 512


def sp(a, b):
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    z = np.load(WORK / "rs2_drive_sweep/cell_drive.npz")
    receipt = json.loads((WORK / "rs2_drive_sweep/receipt.json").read_text())
    live = z["live_cell"].astype(bool)
    drive = z["drive_L1"]
    px = z["px_over"]
    thr = z["thresholds"]
    keys = np.load(WORK / "rs2_thin_margin_keys.npz")
    marg = np.load(WORK / "rs2_cell_byte_marginal.npy")
    ab = json.loads((WORK / "rs2_ab_build_receipt.json").read_text())

    rf_h = (z["rf_r1"] - z["rf_r0"] + 1).astype(float)
    rf_w = (z["rf_c1"] - z["rf_c0"] + 1).astype(float)
    rf_area = rf_h * rf_w
    lv = live
    out: dict = {
        "axis": "[byte-closed, scorer-free]", "score_claim": False,
        "promotion_eligible": False,
        "n_pairs": receipt["n_pairs"], "live_cells": int(lv.sum()),
        "groups_run": len(receipt["groups"]),
        "max_abs_leak_L1": receipt["max_abs_leak_L1"],
        "leak_relative_to_total_drive": receipt["max_abs_leak_L1"] / float(drive.sum()),
        "receptive_field_over_384_live_cells": {
            "rows_min": float(rf_h[lv].min()), "rows_median": float(np.median(rf_h[lv])),
            "rows_max": float(rf_h[lv].max()),
            "cols_min": float(rf_w[lv].min()), "cols_median": float(np.median(rf_w[lv])),
            "cols_max": float(rf_w[lv].max()),
            "area_median_px": float(np.median(rf_area[lv])),
            "area_max_px": float(rf_area[lv].max()),
            "pilot_anchor_bbox_px": 84 * 82,
            "support_ratio_median_vs_wr1_tile": float(np.median(rf_area[lv])) / 256.0,
        },
        "drive_over_live_cells": {
            "min": float(drive[lv].min()), "median": float(np.median(drive[lv])),
            "mean": float(drive[lv].mean()), "max": float(drive[lv].max()),
            "spread_max_over_min": float(drive[lv].max() / drive[lv].min())
            if drive[lv].min() > 0 else None,
            "n_zero_drive_cells": int((drive[lv] == 0).sum()),
        },
        "px_over_thresholds_summed_over_600_pairs": {
            f"gt_{t:g}_LSB": {"median": float(np.median(px[lv, i])),
                              "max": float(px[lv, i].max())}
            for i, t in enumerate(thr)
        },
    }

    tm = keys["tau0p2_all"]
    cand = {
        "bytes_marginal": marg, "wr1_tile": keys["fm16"], "rs2_rf_ambient": keys["fmrf"],
        "gr1_gsum": keys["gsum"], "thin_margin_tau0.2": tm,
    }
    out["spearman_drive_vs"] = {k: sp(drive[lv], v[lv]) for k, v in cand.items()}

    # separable flip-damage proxy: DRIVE-side count x susceptibility density
    box = np.maximum(rf_area, 1.0)
    dens = tm / box
    for i, t in enumerate(thr):
        out.setdefault("spearman_fdproxy_vs_incumbents", {})[f"px_gt_{t:g}_x_thin_density"] = {
            k: sp((px[:, i] * dens)[lv], v[lv]) for k, v in cand.items()
        }

    cells = ab["cells"]
    selA = np.array(cells["A_gr1_drop63"])
    selB = np.array(cells["B_rs2_bytematched"])
    out["arms_on_the_drive_currency"] = {
        "A_gr1_drop63_total_drive": float(drive[selA].sum()),
        "B_rs2_bytematched_total_drive": float(drive[selB].sum()),
        "B_over_A": float(drive[selB].sum() / drive[selA].sum()),
        "A_total_px_gt1LSB": float(px[selA, 1].sum()),
        "B_total_px_gt1LSB": float(px[selB, 1].sum()),
        "B_over_A_px_gt1LSB": float(px[selB, 1].sum() / px[selA, 1].sum()),
    }
    (WORK / "rs2_drive_analysis.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
