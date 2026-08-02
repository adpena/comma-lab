#!/usr/bin/env python3
"""ddm_ll1 -- the per-scorer-pixel WINDOW LATTICE SOLVE probe (task #897).

WHY THIS EXISTS
---------------
``ddm_mp1`` asked us to "swap ``clip(rint(up))`` for the generic dither".  Its own
decomposition says rounding is only ~6% of the error variance while ``e_resample``
(U o D != I) is 76% ideal / 93.5% live.  A dither perturbs the ROUNDING of U(r) and
therefore attacks the 6%.

MAIN measured D directly (2026-08-02) and found the reason the bigger lever exists:

    D = F.interpolate(x,(384,512),'bilinear'), align_corners=False, antialias=False
      => POINT SAMPLING with bilinear weights, NOT an area average.
    stride y = 874/384 = 2.2760 ; x = 1164/512 = 2.2734 ; both > 2
      => consecutive 2x2 read-windows CANNOT overlap.
    reads-per-camera-pixel histogram has exactly two bins:
        0x : 230,904 px (22.70%)  <- blind to BOTH scorers (reproduces task #401)
        1x : 786,432 px (77.30%)  = 196,608 scorer px * 4 EXACTLY

So every scorer pixel owns a PRIVATE, DISJOINT 2x2 camera window and the uint8
preimage problem DECOUPLES EXACTLY into 196,608 independent 4-variable problems.
That is what mp1's otherwise-unexplained "256^4" literally is.

This probe measures the thing the ask could not: solving each window for
``sum_ij w_ij * cam_ij = r`` instead of rounding U(r) attacks e_quant AND
e_resample together.  Receiver-side generic algorithm => rule-118 free => ZERO
counted bytes.

WHAT IT REPORTS
---------------
rms/max of the delivered-vs-intended residual ``D(cam) - r`` under
  (a) the CURRENT path  cam = clip(rint(U(r)))
  (b) the SOLVED path   cam = argmin over the private window
plus the camera-domain divergence between the two rasters (the input to the
pose-half falsifier: v4d warps frame_0 FROM frame_1's camera pixels).

This is a MEASUREMENT, not a wiring.  It touches no runtime path.
Axis: [macOS-CPU advisory] -- geometry + arithmetic only, no scorer is run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512


def bilinear_sample_geometry(
    src: int, dst: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (i0, frac) for torch bilinear ``align_corners=False`` downsampling.

    Each output index reads src indices ``i0`` and ``i0+1`` with weights
    ``(1-frac, frac)``.  This is the EXACT geometry ``F.interpolate`` uses; it is
    re-derived here rather than recalled so the probe carries its own authority.
    """
    centers = (np.arange(dst, dtype=np.float64) + 0.5) * (src / dst) - 0.5
    centers = np.clip(centers, 0.0, None)
    i0 = np.floor(centers).astype(np.int64)
    frac = centers - i0
    # torch clamps the upper tap at the edge; when i0 == src-1 the second tap
    # collapses onto the first, which the weight (frac) handles only if we also
    # clamp the index.  Both are done here so the probe matches torch exactly.
    i1 = np.clip(i0 + 1, 0, src - 1)
    return np.stack([np.clip(i0, 0, src - 1), i1], axis=1), frac


def build_window_index() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ys, xs, wy, wx): the private 2x2 window index + separable weights."""
    ys, fy = bilinear_sample_geometry(CAMERA_H, SEG_H)
    xs, fx = bilinear_sample_geometry(CAMERA_W, SEG_W)
    wy = np.stack([1.0 - fy, fy], axis=1)  # (384,2)
    wx = np.stack([1.0 - fx, fx], axis=1)  # (512,2)
    return ys, xs, wy, wx


def apply_D(cam: np.ndarray, ys, xs, wy, wx) -> np.ndarray:
    """Exact bilinear-sample downsample 874x1164 -> 384x512, per-channel."""
    src = cam.astype(np.float64)
    out = np.zeros((SEG_H, SEG_W, src.shape[2]), dtype=np.float64)
    for a in range(2):
        for b in range(2):
            gathered = src[np.ix_(ys[:, a], xs[:, b])]
            out += gathered * (wy[:, a][:, None, None] * wx[:, b][None, :, None])
    return out


def solve_windows(
    cam0: np.ndarray,
    target: np.ndarray,
    ys,
    xs,
    wy,
    wx,
    radius: int = 2,
) -> np.ndarray:
    """Choose each private window's 4 uint8 values to hit ``target`` exactly.

    Exhaustive over d in {-radius..radius}^4 around the rounded-bicubic start.
    The windows are DISJOINT (measured), so this is 196,608 independent exact
    little problems and a scatter-back cannot collide.  Deterministic: ties break
    by first-encountered combo in a fixed lexicographic order.
    """
    n_ch = cam0.shape[2]
    # gather the private window: c[a][b] has shape (384,512,ch)
    c = [[cam0[np.ix_(ys[:, a], xs[:, b])].astype(np.float64) for b in range(2)] for a in range(2)]
    w = [[(wy[:, a][:, None] * wx[:, b][None, :])[..., None] for b in range(2)] for a in range(2)]

    current = sum(w[a][b] * c[a][b] for a in range(2) for b in range(2))
    err = target - current  # (384,512,ch) what the window still owes

    best_cost = np.abs(err).copy()
    best = np.zeros((SEG_H, SEG_W, n_ch, 4), dtype=np.int16)

    offsets = range(-radius, radius + 1)
    for combo in itertools.product(offsets, offsets, offsets, offsets):
        if combo == (0, 0, 0, 0):
            continue
        d = np.array(combo, dtype=np.float64).reshape(2, 2)
        delta = sum(w[a][b] * d[a, b] for a in range(2) for b in range(2))
        cost = np.abs(err - delta)
        # validity: every adjusted value must stay in [0,255]
        valid = np.ones_like(cost, dtype=bool)
        for a in range(2):
            for b in range(2):
                v = c[a][b] + d[a, b]
                valid &= (v >= 0.0) & (v <= 255.0)
        take = valid & (cost < best_cost)
        if not take.any():
            continue
        best_cost = np.where(take, cost, best_cost)
        for k, val in enumerate(combo):
            best[..., k] = np.where(take, np.int16(val), best[..., k])

    out = cam0.astype(np.int16).copy()
    for k, (a, b) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        rows = ys[:, a][:, None]
        cols = xs[:, b][None, :]
        out[rows, cols, :] = np.clip(
            c[a][b].astype(np.int16) + best[..., k], 0, 255
        ).astype(np.int16)
    return out.astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from tac.optimization.ddm_tr1_runtime import bicubic_up_to_camera_float

    ys, xs, wy, wx = build_window_index()

    # REAL content: decode real camera frames and use D(real) as the render target
    # r.  That is exactly the kind of target the renderer aims at, with real
    # spatial statistics -- no synthetic fixture (NO-FAKE class 3).
    sys.path.insert(0, str(REPO_ROOT / "upstream"))
    import av  # noqa: PLC0415

    from frame_utils import yuv420_to_rgb  # noqa: PLC0415

    rows = []
    container = av.open(str(REPO_ROOT / "upstream" / "videos" / "0.mkv"))
    taken = 0
    for frame in container.decode(video=0):
        if taken >= args.frames:
            break
        # CANONICAL decode only: yuv420_to_rgb takes the PyAV FRAME object.
        # PyAV rgb24 manufactures ~100x phantom pose (CLAUDE.md), so never that.
        rgb = yuv420_to_rgb(frame)
        cam_real = np.asarray(rgb.numpy() if hasattr(rgb, "numpy") else rgb, dtype=np.uint8)
        if cam_real.shape[:2] != (CAMERA_H, CAMERA_W):
            continue
        target = apply_D(cam_real, ys, xs, wy, wx)  # (384,512,3) float target r

        up = bicubic_up_to_camera_float(target.astype(np.float32))
        cam_round = np.clip(np.rint(up), 0, 255).astype(np.uint8)
        delivered_round = apply_D(cam_round, ys, xs, wy, wx)
        e_round = delivered_round - target

        cam_solved = solve_windows(
            cam_round, target, ys, xs, wy, wx, radius=args.radius
        )
        delivered_solved = apply_D(cam_solved, ys, xs, wy, wx)
        e_solved = delivered_solved - target

        cam_div = np.abs(cam_solved.astype(np.int32) - cam_round.astype(np.int32))
        row = {
            "frame": taken,
            "rms_lsb_round": float(np.sqrt((e_round**2).mean())),
            "rms_lsb_solved": float(np.sqrt((e_solved**2).mean())),
            "max_lsb_round": float(np.abs(e_round).max()),
            "max_lsb_solved": float(np.abs(e_solved).max()),
            "camera_divergence_mean_lsb": float(cam_div.mean()),
            "camera_divergence_max_lsb": int(cam_div.max()),
            "camera_px_changed_frac": float((cam_div > 0).mean()),
        }
        row["rms_improvement_x"] = (
            row["rms_lsb_round"] / row["rms_lsb_solved"]
            if row["rms_lsb_solved"] > 0
            else float("inf")
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        taken += 1
    container.close()

    if rows:
        agg = {
            "n_frames": len(rows),
            "radius": args.radius,
            "mean_rms_round": float(np.mean([r["rms_lsb_round"] for r in rows])),
            "mean_rms_solved": float(np.mean([r["rms_lsb_solved"] for r in rows])),
            "mean_improvement_x": float(np.mean([r["rms_improvement_x"] for r in rows])),
            "axis": "[macOS-CPU advisory] geometry+arithmetic, no scorer run",
        }
        print("AGG " + json.dumps(agg), flush=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"rows": rows, "agg": agg}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
