#!/usr/bin/env python
"""ddm_rs2 — the FOOTPRINT correction of task #766's reverse waterfill.

THE CLAIM UNDER TEST (verified at source, `experiments/ddm_wr1_reverse_waterfill.py`)

    :89   cell = (atlas["y"] // 16) * 32 + (atlas["x"] // 16)
    :90   flip_mass = np.bincount(cell, minlength=768)
    :93   order = np.lexsort((-residual_mass, flip_mass))
    :131  dseg_ceiling = REF_DSEG + dropped_flip_mass / TOTAL_PX
    :211  --knee-a 486  "safe-floor tranche (all zero-flip)"

wr1 already ranks by FLIP DAMAGE first and by bytes only as a tie-break, so the
charter's premise ("ranking by the wrong currency = bytes") is FALSE.  What IS
wrong is the SUPPORT: `flip_mass` attributes a flip to the 16x16 tile it lands
in, i.e. it prices a cell's drop by the flips inside the cell's OWN tile.  But
dropping a cell perturbs its whole RECEPTIVE FIELD through the decoder's 4
upsample+conv stages -- MEASURED in the rs2 pilot at 84 x 82 scorer pixels
(6,192 px) for cell (13,17), 24x the 256-pixel tile.

This script re-prices every cell on the MEASURED support, keeping wr1's own
ambient-flip proxy fixed so the comparison is apples-to-apples, and asks what
changes -- above all, how many of the 486 cells wr1 calls "provably safe (all
zero-flip)" still have zero flips in their real footprint.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SSD = Path("/Volumes/VertigoDataTier/pact")
WORK = SSD / "ddm_rs2_20260803"
ATLAS = SSD / "ddm_ru1_20260729/atlas_flat.npz"
WR1 = SSD / "ddm_wr1_20260729/wr1_cell_sensitivity_atlas.npz"
OUT = WORK / "rs2_footprint_rerank.json"

sys.path.insert(0, "src")
from tac.optimization import ddm_ix2_archive_container as C  # noqa: E402

LEVELS, R, Cc = 16, 24, 32
SEG_H, SEG_W = 384, 512
DEN = 37_545_489
PX = 196_608 * 600
W = 4.0 * DEN / PX
RF_HALF = 34          # MEASURED in the pilot: bbox rows [174,257] for cell row 13 (px 208..223)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    rep: dict = {"axis": "[byte-closed, scorer-free: cached atlas + real encoder]",
                 "score_claim": False, "promotion_eligible": False}
    atlas = np.load(ATLAS)
    y, x = atlas["y"].astype(np.int64), atlas["x"].astype(np.int64)
    rep["atlas_flips"] = len(y)

    # dense 384x512 ambient-flip histogram (summed over all 600 pairs)
    dense = np.bincount(y * SEG_W + x, minlength=SEG_H * SEG_W).reshape(SEG_H, SEG_W)
    assert dense.sum() == len(y)

    # --- wr1's key, reproduced exactly -----------------------------------
    fm16 = np.bincount((y // 16) * Cc + (x // 16), minlength=R * Cc).astype(np.float64)
    wr1 = np.load(WR1)
    rep["wr1_flip_mass_reproduced"] = bool(np.array_equal(fm16, wr1["flip_mass"]))
    resid = wr1["residual_mass"].astype(np.float64)

    # --- the corrected key: same proxy, MEASURED support ------------------
    ii = np.zeros((SEG_H + 1, SEG_W + 1), dtype=np.int64)
    ii[1:, 1:] = dense.cumsum(0).cumsum(1)

    def box_sum(r0, r1, c0, c1):
        return int(ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0])

    fmrf = np.zeros(R * Cc, dtype=np.float64)
    boxes = {}
    for r_ in range(R):
        for c_ in range(Cc):
            r0, r1 = max(0, r_ * 16 - RF_HALF), min(SEG_H, (r_ + 1) * 16 + RF_HALF)
            c0, c1 = max(0, c_ * 16 - RF_HALF), min(SEG_W, (c_ + 1) * 16 + RF_HALF)
            fmrf[r_ * Cc + c_] = box_sum(r0, r1, c0, c1)
            boxes[r_ * Cc + c_] = (r0, r1, c0, c1)
    rep["support_px"] = {"wr1_tile": 256, "measured_rf_typical": (2 * RF_HALF + 16) ** 2,
                         "ratio": round((2 * RF_HALF + 16) ** 2 / 256.0, 2)}

    # --- the zero-flip "provably safe" set --------------------------------
    z16, zrf = fm16 == 0, fmrf == 0
    rep["zero_flip_sets"] = {
        "wr1_tile_zero_cells": int(z16.sum()),
        "measured_rf_zero_cells": int(zrf.sum()),
        "wr1_zero_that_are_NOT_rf_zero": int((z16 & ~zrf).sum()),
        "rf_zero_subset_of_wr1_zero": bool((zrf & ~z16).sum() == 0),
    }
    order16 = np.lexsort((-resid, fm16))
    orderrf = np.lexsort((-resid, fmrf))
    kneeA = order16[:486]
    rep["kneeA_486"] = {
        "cells": 486,
        "wr1_flip_mass_sum": float(fm16[kneeA].sum()),
        "measured_rf_flip_mass_sum": float(fmrf[kneeA].sum()),
        "cells_with_zero_rf_flip_mass": int((fmrf[kneeA] == 0).sum()),
        "cells_wr1_calls_safe_but_rf_says_not": int((fmrf[kneeA] > 0).sum()),
        "rf_flip_mass_as_frac_of_all_flips": float(fmrf[kneeA].sum() / len(y)),
    }

    # --- rank agreement ---------------------------------------------------
    rep["spearman"] = {
        "wr1_tile_vs_measured_rf_flipmass": spearman(fm16, fmrf),
        "wr1_tile_vs_bytes_proxy": spearman(fm16, resid),
        "measured_rf_vs_bytes_proxy": spearman(fmrf, resid),
        "droporder_wr1_vs_rf": spearman(
            np.argsort(order16).astype(float), np.argsort(orderrf).astype(float)
        ),
    }
    for k in (100, 200, 300, 384, 486, 600):
        rep.setdefault("prefix_overlap", {})[str(k)] = len(np.intersect1d(order16[:k], orderrf[:k]))

    # --- the EXACT byte side, per cell, on the real encoder ---------------
    codes = np.load(SSD / "ddm_br1_20260803/cx1_tokens.npy")
    base, delta = C._factor_mode_delta(codes, LEVELS)
    live_cell = (delta != 0).any(axis=0).any(axis=2).reshape(-1)
    base_bytes = len(C.encode_token_frame(codes, levels=LEVELS))
    rep["token_member_bytes"] = int(base_bytes)
    rep["live_cells"] = int(live_cell.sum())

    t0 = time.time()
    marg = np.zeros(R * Cc)
    for idx in np.nonzero(live_cell)[0]:
        r_, c_ = divmod(int(idx), Cc)
        mod = codes.copy()
        mod[:, r_, c_, :] = base[r_, c_, :][None]
        marg[idx] = base_bytes - len(C.encode_token_frame(mod, levels=LEVELS))
    rep["byte_marginal_seconds"] = round(time.time() - t0, 1)
    lm = marg[live_cell]
    rep["byte_marginal_per_live_cell"] = {
        "min": float(lm.min()), "median": float(np.median(lm)),
        "mean": float(lm.mean()), "max": float(lm.max()),
        "n_negative": int((lm < 0).sum()), "n_zero": int((lm == 0).sum()),
        "spearman_vs_wr1_tile_flipmass": spearman(lm, fm16[live_cell]),
        "spearman_vs_measured_rf_flipmass": spearman(lm, fmrf[live_cell]),
        "spearman_vs_bytes_proxy_residual_mass": spearman(lm, resid[live_cell]),
    }
    np.save(WORK / "rs2_cell_byte_marginal.npy", marg)
    np.save(WORK / "rs2_cell_flipmass_rf.npy", fmrf)

    # --- the priced consequence: exchange rate of the two orders ----------
    live_ids = np.nonzero(live_cell)[0]
    priced = {}
    for name, key in (("wr1_tile", fm16), ("measured_rf", fmrf)):
        o = live_ids[np.lexsort((-marg[live_ids], key[live_ids]))]
        rows = []
        for k in (50, 100, 150, 200, 250, 300, 384):
            sel = o[:k]
            mod = codes.copy()
            mod[:, sel // Cc, sel % Cc, :] = base[sel // Cc, sel % Cc, :][None]
            b = base_bytes - len(C.encode_token_frame(mod, levels=LEVELS))
            f = float(key[sel].sum())
            rows.append({"k": k, "bytes_saved": int(b), "ambient_flip_mass_in_support": f,
                         "flip_budget_at_W": round(b / W, 1),
                         "B_per_ambient_flip": round(b / f, 4) if f > 0 else None})
        priced[name] = rows
    rep["priced_prefixes"] = priced
    rep["W"] = W

    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
