#!/usr/bin/env python
"""ddm_rs2 — the THIN-MARGIN susceptibility key, and why it is better than ambient flips.

`ddm_wr1`'s damage key is AMBIENT FLIP MASS: the number of pixels already disagreeing
with GT inside a cell's tile.  Two things are wrong with it as a susceptibility proxy,
independent of the support bug this arm's §1 measures:

  1. A pixel that ALREADY flips cannot flip again -- d_seg counts disagreements, so
     ambient-flip pixels are exactly the pixels whose damage is already CAPPED (a drop
     there can only leave them wrong or accidentally fix them).  The pixels at risk of
     NEW damage are the CURRENTLY-CORRECT ones with a THIN margin.
  2. The ambient-flip atlas is measured at the `ru1`/`sg1` endpoint (458,738 flips),
     not at the live cx1 base (508,639 flips) -- a 10.9% different vehicle state.

The GT margin field `gt_n600.npz['margins']` (top1-top2 SegNet logit gap on the GT
frames) is endpoint-INDEPENDENT and covers every pixel, so a thin-margin key is both
better-typed and free of the borrowed-endpoint caveat.  This script builds it on the
MEASURED receptive-field support and reports how it ranks against the three incumbents.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

SSD = Path("/Volumes/VertigoDataTier/pact")
OUT = SSD / "ddm_rs2_20260803"
R, Cc, H, W_ = 24, 32, 384, 512
RF_HALF = 34
TAUS = (0.05, 0.2, 0.5, 1.0, 2.0)


def boxsum(dense: np.ndarray, half: int) -> np.ndarray:
    ii = np.zeros((H + 1, W_ + 1), np.float64)
    ii[1:, 1:] = dense.cumsum(0).cumsum(1)
    out = np.zeros(R * Cc)
    for r_ in range(R):
        for c_ in range(Cc):
            r0, r1 = max(0, r_ * 16 - half), min(H, (r_ + 1) * 16 + half)
            c0, c1 = max(0, c_ * 16 - half), min(W_, (c_ + 1) * 16 + half)
            out[r_ * Cc + c_] = ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]
    return out


def sp(a, b):
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    t0 = time.time()
    z = np.load("experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    marg = z["margins"]
    lst = z["lstars"]
    print("loaded", marg.shape, lst.shape, round(time.time() - t0, 1), "s", flush=True)
    real = np.concatenate(
        [np.load(SSD / f"ddm_sg1_20260731/argmax/chunk_{i:04d}.npy") for i in (0, 120, 240, 360, 480)], 0
    )
    correct = real == lst.astype(np.uint8)
    print("realized flips at sg1 endpoint:", int((~correct).sum()), flush=True)

    at = np.load(SSD / "ddm_ru1_20260729/atlas_flat.npz")
    y, x = at["y"].astype(np.int64), at["x"].astype(np.int64)
    fm16 = np.bincount((y // 16) * Cc + (x // 16), minlength=R * Cc).astype(float)
    fmrf = boxsum(np.bincount(y * W_ + x, minlength=H * W_).reshape(H, W_).astype(float), RF_HALF)
    gsum = np.load(SSD / "ddm_sg1_20260731/gr1_cell_gsum.npy").reshape(-1)

    out = {
        "axis": "[byte-closed, scorer-free: cached GT margin field + cached realized argmax]",
        "score_claim": False,
        "promotion_eligible": False,
        "rf_half_px": RF_HALF,
        "endpoint_caveat": (
            "ambient-flip keys (wr1 tile, rs2 rf) come from the ru1/sg1 endpoint "
            "(458,738 flips); the live cx1 base has 508,639. The thin-margin key uses the "
            "GT margin field and is endpoint-independent."
        ),
        "realized_flips_sg1_endpoint": int((~correct).sum()),
        "taus": {},
    }
    keys = {}
    for tau in TAUS:
        thin = marg < tau
        k_all = boxsum(thin.sum(0).astype(np.float64), RF_HALF)
        k_cor = boxsum((thin & correct).sum(0).astype(np.float64), RF_HALF)
        keys[f"tau{str(tau).replace('.', 'p')}_all"] = k_all
        keys[f"tau{str(tau).replace('.', 'p')}_correct"] = k_cor
        out["taus"][str(tau)] = {
            "n_thin_px": int(thin.sum()),
            "n_thin_and_correct": int((thin & correct).sum()),
            "rho_all_vs_wr1tile": sp(k_all, fm16),
            "rho_all_vs_rs2rf_ambient": sp(k_all, fmrf),
            "rho_all_vs_gr1gsum": sp(k_all, gsum),
            "rho_correct_vs_rs2rf_ambient": sp(k_cor, fmrf),
            "rho_correct_vs_gr1gsum": sp(k_cor, gsum),
            "rho_all_vs_correct": sp(k_all, k_cor),
        }
        print("tau", tau, out["taus"][str(tau)], round(time.time() - t0, 1), "s", flush=True)
    np.savez_compressed(OUT / "rs2_thin_margin_keys.npz", **keys, fm16=fm16, fmrf=fmrf, gsum=gsum)
    (OUT / "rs2_thin_margin_keys.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print("WROTE", OUT / "rs2_thin_margin_keys.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
