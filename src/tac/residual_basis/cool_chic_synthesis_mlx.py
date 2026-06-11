# SPDX-License-Identifier: MIT
"""MLX fast-path Cool-Chic synthesis (portability-contract sibling of numpy ref).

Per operator binding 2026-06-09 ("MLX-first everything; numpy reference =
portability contract; torch via tinygrad-like primitives"): this is the MLX
fast path for the Cool-Chic synthesis forward. It MUST match
:func:`tac.residual_basis.cool_chic_synthesis_numpy.synthesize_rgb_numpy`
bit-for-bit (within float tolerance) on REAL (non-zero) inputs — the parity gate
is in ``test_cool_chic_basis.py`` and feeds NON-zero grids+weights (the grid-PE
fake-parity lesson: zeros pass any parity check).

MLX runs on the M5 Max unified memory; the GPU may be busy, so default to
``mx.set_default_device(mx.cpu)`` at call time per the operator "MLX-CPU or
torch-CPU; do NOT contend for the GPU" constraint.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def bilinear_upsample_mlx(grid: np.ndarray, out_h: int, out_w: int) -> mx.array:
    """Bilinear-upsample ``(C, h, w)`` numpy -> MLX ``(C, out_h, out_w)``.

    align_corners=False convention (matches numpy ref + torch). Implemented via
    explicit gather so MLX (which lacks a direct align_corners=False resize that
    we can guarantee matches torch) agrees with the reference exactly.
    """
    c, h, w = grid.shape
    if h == out_h and w == out_w:
        return mx.array(grid.astype(np.float32))
    ys = (np.arange(out_h, dtype=np.float64) + 0.5) * (h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float64) + 0.5) * (w / out_w) - 0.5
    ys = np.clip(ys, 0.0, h - 1)
    xs = np.clip(xs, 0.0, w - 1)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]
    g = grid.astype(np.float32)
    top = g[:, y0][:, :, x0] * (1 - wx) + g[:, y0][:, :, x1] * wx
    bot = g[:, y1][:, :, x0] * (1 - wx) + g[:, y1][:, :, x1] * wx
    out = top * (1 - wy) + bot * wy
    return mx.array(out)


def _gelu_mlx(x: mx.array) -> mx.array:
    # exact GELU via erf (matches numpy ref _gelu and torch F.gelu default).
    return 0.5 * x * (1.0 + mx.erf(x / mx.sqrt(mx.array(2.0))))


def synthesize_rgb_mlx(
    grids: list[np.ndarray],
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    out_h: int,
    out_w: int,
    *,
    on_cpu: bool = True,
) -> np.ndarray:
    """MLX synthesis forward -> ``(3, out_h, out_w)`` numpy in [0,1].

    ``on_cpu`` forces ``mx.cpu`` so we never contend for the busy Metal GPU.
    """
    if on_cpu:
        mx.set_default_device(mx.cpu)
    ups = [bilinear_upsample_mlx(g, out_h, out_w) for g in grids]
    feat = mx.concatenate(ups, axis=0)  # (c_in, H, W)
    c_in, h, w = feat.shape
    flat = feat.reshape(c_in, h * w)
    hidden = mx.array(w1.astype(np.float32)) @ flat + mx.array(b1.astype(np.float32))[:, None]
    hidden = _gelu_mlx(hidden)
    out = mx.array(w2.astype(np.float32)) @ hidden + mx.array(b2.astype(np.float32))[:, None]
    out = mx.sigmoid(out)
    mx.eval(out)
    return np.array(out).reshape(3, h, w)
