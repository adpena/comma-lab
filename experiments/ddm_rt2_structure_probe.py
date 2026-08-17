#!/usr/bin/env python3
"""ddm_rt2 -- structural probe: is the shipped camera decode an UPSAMPLE of a scorer-res field?

Scorer-free, $0.  Answers one question before any SegNet forward is spent:

    the round trip is  render -> (lift to 874x1164) -> uint8 -> D_bilinear -> 384x512 -> SegNet

If the render is NATIVELY 384x512 and lifted, then `D o U` is a fixed blur the receiver could
in principle pre-compensate for FREE.  If the render is natively camera-res, no such lever
exists and the loss must be attacked elsewhere.

Method (no renderer source needed -- read the bytes):
  1. take the shipped camera frame X (874x1164x3 uint8)
  2. build Y = D_bilinear(X)                    -- exactly what the scorer sees, 384x512
  3. re-lift Y with each candidate kernel and compare to X

Also measures the exact READ SUPPORT of the scorer's bilinear downsample: which camera pixels
carry non-zero weight into the 384x512 lattice, and how much of the camera field is never read
(reproduces rn1's 768 rows / 1024 cols / 22.6969% blind, from geometry alone).

TWO LIMITS OF THIS PROBE, recorded because both bit and neither is load-bearing below:

  * The re-lift residual does NOT test the upsample hypothesis.  If X = U(m) then D X = A m
    with A = D.U != I, so U(D X) != X and a large residual is EXPECTED.  The correct test is
    the ORTHOGONAL PROJECTION of X onto range(U), which ddm_rt2 ran separately and which
    settles it: residual rms 0.2334 / max 0.825 = pure uint8 residue, so the camera field IS a
    bilinear lift of a native 384x512 render.
  * `row_rank_numeric` is computed on a COLUMN-SUBSAMPLED matrix `g[:, ::8]` (146 columns), so
    it can never exceed 146 and reports the subsample width, not the field's rank.  It is
    retained only so the receipt is reproducible; it carries no information and no finding in
    this arm cites it.

axis: [macOS-CPU advisory] -- structural, scorer-free.  NEVER a score.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

FRAMES = 600
SEG_H, SEG_W = 384, 512
CAM_H, CAM_W = 874, 1164

DEFAULT_RAW = Path(
    "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/runs/"
    "base_optimized_n600_r3/output/0.raw"
)
DEFAULT_WORK = Path("/Volumes/APDataStore/pact/ddm_rt2")


class Rt2Error(RuntimeError):
    """Fail-closed error."""


def open_raw(raw: Path) -> np.memmap:
    n = raw.stat().st_size // (CAM_H * CAM_W * 3)
    if n * CAM_H * CAM_W * 3 != raw.stat().st_size:
        raise Rt2Error(f"{raw} is not a whole number of {CAM_H}x{CAM_W}x3 frames")
    return np.memmap(raw, dtype=np.uint8, mode="r", shape=(n, CAM_H, CAM_W, 3))


def bilinear_weights(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """torch `interpolate(mode='bilinear', align_corners=False)` 1-D weights, no antialias.

    Returns (idx, w) with idx (n_out, 2) int64 source indices and w (n_out, 2) float64 weights.
    Matches ATen's `area_pixel_compute_source_index` with align_corners=False:
        src = scale * (dst + 0.5) - 0.5, clamped at 0.
    """
    scale = n_in / n_out
    src = scale * (np.arange(n_out, dtype=np.float64) + 0.5) - 0.5
    src = np.maximum(src, 0.0)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    frac = src - i0
    idx = np.stack([i0, i1], axis=1)
    w = np.stack([1.0 - frac, frac], axis=1)
    return idx, w


def down_bilinear_np(frame: np.ndarray) -> np.ndarray:
    """D: camera uint8 (H,W,3) -> float64 (SEG_H,SEG_W,3), torch-bilinear semantics."""
    ri, rw = bilinear_weights(CAM_H, SEG_H)
    ci, cw = bilinear_weights(CAM_W, SEG_W)
    x = frame.astype(np.float64)
    rows = x[ri[:, 0]] * rw[:, 0, None, None] + x[ri[:, 1]] * rw[:, 1, None, None]
    out = rows[:, ci[:, 0]] * cw[None, :, 0, None] + rows[:, ci[:, 1]] * cw[None, :, 1, None]
    return out


def read_support() -> dict:
    """Which camera pixels carry non-zero weight into the scorer lattice."""
    ri, rw = bilinear_weights(CAM_H, SEG_H)
    ci, cw = bilinear_weights(CAM_W, SEG_W)
    rows_used = np.zeros(CAM_H, dtype=bool)
    cols_used = np.zeros(CAM_W, dtype=bool)
    for k in range(2):
        rows_used[ri[rw[:, k] > 0, k]] = True
        cols_used[ci[cw[:, k] > 0, k]] = True
    n_rows, n_cols = int(rows_used.sum()), int(cols_used.sum())
    return {
        "camera_rows_read": n_rows,
        "camera_rows_total": CAM_H,
        "camera_cols_read": n_cols,
        "camera_cols_total": CAM_W,
        "camera_pixels_read": n_rows * n_cols,
        "camera_pixels_total": CAM_H * CAM_W,
        "read_fraction": n_rows * n_cols / (CAM_H * CAM_W),
        "never_read_fraction": 1.0 - n_rows * n_cols / (CAM_H * CAM_W),
        "scorer_cells": SEG_H * SEG_W,
        "dof_ratio_scorer_over_camera": SEG_H * SEG_W / (CAM_H * CAM_W),
    }


def torch_ops():
    import torch

    torch.set_grad_enabled(False)
    return torch


def probe_frames(raw: Path, frame_ids: list[int]) -> dict:
    torch = torch_ops()
    import torch.nn.functional as F

    arr = open_raw(raw)
    rows = []
    for fid in frame_ids:
        X = np.asarray(arr[fid], dtype=np.uint8)
        xt = torch.from_numpy(X).permute(2, 0, 1)[None].float()

        # what the scorer actually sees
        Y = F.interpolate(xt, size=(SEG_H, SEG_W), mode="bilinear")
        # numpy reimplementation cross-check (proves my D matches torch bit-for-bit enough)
        Y_np = down_bilinear_np(X)
        d_impl = float(np.abs(Y[0].permute(1, 2, 0).numpy().astype(np.float64) - Y_np).max())

        # re-lift with the two candidate lift kernels and compare to the real camera frame
        rec = {"frame": fid, "D_impl_max_abs_err_vs_torch": d_impl}
        for mode in ("bicubic", "bilinear", "nearest"):
            kw = {} if mode == "nearest" else {"align_corners": False}
            Xp = F.interpolate(Y, size=(CAM_H, CAM_W), mode=mode, **kw)
            resid = (Xp - xt)[0].numpy()
            rec[f"relift_{mode}_max_abs"] = float(np.abs(resid).max())
            rec[f"relift_{mode}_rms"] = float(np.sqrt((resid ** 2).mean()))
            rec[f"relift_{mode}_frac_gt_0p5"] = float((np.abs(resid) > 0.5).mean())

        # rank probe: how many of the 874 camera rows are (numerically) independent?
        g = X.astype(np.float64).mean(axis=2)  # (874, 1164) luma-ish
        # row-space rank via SVD on a column subsample (cheap, exact enough for a rank read)
        sub = g[:, ::8]
        s = np.linalg.svd(sub, compute_uv=False)
        tol = s.max() * max(sub.shape) * np.finfo(np.float64).eps
        rec["row_rank_numeric"] = int((s > tol).sum())
        rec["row_rank_1pct_energy"] = int((s > 0.01 * s.max()).sum())
        rec["camera_rows"] = CAM_H
        rec["scorer_rows"] = SEG_H
        rows.append(rec)
    return {"frames": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--frames", type=int, nargs="*", default=[1, 201, 601, 999, 1199])
    args = ap.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    receipt = {
        "schema": "ddm_rt2_structure_probe.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "raw": str(args.raw),
        "read_support": read_support(),
    }
    receipt.update(probe_frames(args.raw, args.frames))
    receipt["wall_s"] = time.time() - t0
    out = args.work / "RT2_STRUCTURE_PROBE.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
