#!/usr/bin/env python
"""ddm_rs2 — the EXACT visibility operator a gradient carries for free.

THE QUESTION
------------
`ddm_gr1`'s damage key is a backprop gradient; mine is a hand-built geometric box.
Both are support-correct, yet they agree only at rho 0.829.  The coordinator's
hypothesis for the missing 17%: **a gradient is computed in the space the score
actually sees**, so it carries the resampling VISIBILITY structure for free, while
a geometric proxy has no way to know about it.

That hypothesis is testable exactly and scorer-free, because the operator between
the renderer's output and the scorer's input is LINEAR and SEPARABLE:

    render [384,512]  --U-->  camera [874,1164]  --D-->  scorer input [384,512]

U is PyTorch bicubic (A=-0.75, align_corners=False) and D is PyTorch bilinear
point sampling (antialias=False), and both act independently on rows and columns.
So the composite M = D.U factorises as M_row (384x384) acting on rows and M_col
(512x512) acting on columns, and the full 196,608-dimensional operator's spectrum
is exactly the outer product of their singular values.  No approximation.

WHAT THIS BOUNDS.  M is the PRE-quantisation operator; the shipped path inserts
`clip(rint(.))` between U and D, whose dead zone can only annihilate MORE.  So the
null/attenuated fraction computed here is a LOWER BOUND on invisibility.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SSD = Path("/Volumes/VertigoDataTier/pact")
OUT = SSD / "ddm_rs2_20260803" / "rs2_visibility_operator.json"
PF = SSD / "ddm_pfs1_20260729/d1/eval_root/submissions/pfs1"

for _p in ("src", str(PF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ddm_tr1_runtime import _cubic_indices_weights  # noqa: E402

from tac.optimization.ddm_ll1_window_solve import _bilinear_taps  # noqa: E402

SEG_H, SEG_W = 384, 512
CAM_H, CAM_W = 874, 1164


def up_matrix(src: int, dst: int) -> np.ndarray:
    """Dense (dst, src) bicubic upsample matrix, exactly as the receiver applies it."""
    idx, w = _cubic_indices_weights(src, dst)
    m = np.zeros((dst, src), dtype=np.float64)
    for t in range(idx.shape[1]):
        np.add.at(m, (np.arange(dst), idx[:, t]), w[:, t].astype(np.float64))
    return m


def down_matrix(src: int, dst: int) -> np.ndarray:
    """Dense (dst, src) bilinear point-sampling downsample matrix (antialias=False)."""
    idx, w = _bilinear_taps(src, dst)
    idx = np.asarray(idx)
    w = np.asarray(w, dtype=np.float64)
    m = np.zeros((dst, src), dtype=np.float64)
    for t in range(idx.shape[1]):
        np.add.at(m, (np.arange(dst), idx[:, t]), w[:, t])
    return m


def analyse(name: str, n: int, cam: int) -> dict:
    U = up_matrix(n, cam)
    Dn = down_matrix(cam, n)
    M = Dn @ U
    s = np.linalg.svd(M, compute_uv=False)
    ident = float(np.abs(M - np.eye(n)).max())
    return {
        "axis": name, "render_n": n, "camera_n": cam,
        "singular_max": float(s.max()), "singular_min": float(s.min()),
        "singular_median": float(np.median(s)),
        "condition_number": float(s.max() / s.min()) if s.min() > 0 else None,
        "max_abs_deviation_from_identity": ident,
        "frac_sv_below_1e-3": float((s < 1e-3).mean()),
        "frac_sv_below_1e-2": float((s < 1e-2).mean()),
        "frac_sv_below_0.1": float((s < 0.1).mean()),
        "frac_sv_below_0.5": float((s < 0.5).mean()),
        "energy_frac_in_top_half": float((s[: n // 2] ** 2).sum() / (s**2).sum()),
        "_sv": s,
    }


def main() -> int:
    rows = analyse("rows", SEG_H, CAM_H)
    cols = analyse("cols", SEG_W, CAM_W)
    sr, sc = rows.pop("_sv"), cols.pop("_sv")

    # full 2-D operator spectrum = outer product of the two axes' singular values
    full = np.outer(sr, sc).reshape(-1)
    full.sort()
    total = SEG_H * SEG_W
    # per-pixel uint8 quantisation step relative to the 0..255 signal range: a render-space
    # direction whose gain falls below this cannot survive clip(rint(.)) on its own.
    q = 0.5 / 255.0
    rep = {
        "axis": "[exact, scorer-free: the frozen resamplers only]",
        "score_claim": False, "promotion_eligible": False,
        "operator": "M = D.U  (render 384x512 -> camera 874x1164 -> scorer 384x512)",
        "separable": True,
        "rows": rows, "cols": cols,
        "full_operator_dims": total,
        "full_spectrum": {
            "max": float(full.max()), "min": float(full.min()),
            "median": float(np.median(full)),
            "frac_gain_below_1e-3": float((full < 1e-3).mean()),
            "frac_gain_below_1e-2": float((full < 1e-2).mean()),
            "frac_gain_below_0.1": float((full < 0.1).mean()),
            "frac_gain_below_0.5": float((full < 0.5).mean()),
            "frac_gain_below_uint8_step": float((full < q).mean()),
            "uint8_step_used": q,
            "energy_frac_in_strongest_1pct": float(
                (full[-total // 100:] ** 2).sum() / (full**2).sum()),
            "energy_frac_in_strongest_10pct": float(
                (full[-total // 10:] ** 2).sum() / (full**2).sum()),
            "dims_carrying_99pct_energy": int(
                total - np.searchsorted(np.cumsum(full**2) / (full**2).sum(), 0.01)),
        },
        "bound_direction": (
            "M is the PRE-quantisation operator; the shipped path inserts clip(rint(.)) "
            "between U and D, whose dead zone can only annihilate MORE. Every 'invisible' "
            "fraction here is a LOWER BOUND."
        ),
    }
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
