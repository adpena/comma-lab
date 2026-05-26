# SPDX-License-Identifier: MIT
"""tac.local_acceleration.pr95_hnerv_numpy_reference — CANONICAL numpy reference primitives.

Per CONSOLIDATE-OP-1 META-CONSOLIDATE-OP-2 extraction wave 2026-05-26
(operator NON-NEGOTIABLE *"adversarial review against all landing recursive
for math and scientific and engineering rigor and for MLX drift minimization
and portability via numpy"* + Catalog #335 cathedral consumer canonical
contract paradigm-shift).

Canonical sister-substrate-reusable numpy reference implementations for the
7 PR95/HNeRV-family primitives that every Path 3 MLX substrate uses. Sister
substrates MUST import from this canonical module rather than re-implement
local copies (per CONSOLIDATE-OP-1 META extraction wave; prevents primitive
drift class across future Path 3 candidates).

Original canonical-reference anchor: G=NIRVANA established the 7-primitive
canonical pattern at
``src/tac/substrates/nirvana_cascading_nerv/numpy_reference.py`` per axis-3
portability discipline (operator directive #3 2026-05-26). This module
extracts those 7 primitives to one canonical location so future Path 3
substrates (L/M/N/O + future) can reference (not duplicate).

The G=NIRVANA-specific composition helper ``cascade_reconstruct`` is NOT
extracted here — it is substrate-specific and stays in
``nirvana_cascading_nerv.numpy_reference``.

This module is canonical-portable: numpy only, no MLX import, no torch
import. Operable on any Python+numpy install.

Per-primitive parity bound vs MLX/PyTorch reference: ≤ 1e-5 for fp32
deterministic ops; ≤ 1e-3 for fp16 accumulation.

Use cases enabled:
- GHA CPU CI testing per Catalog #178 + #179 without MLX install
- Sister cathedral consumer cross-validation per Catalog #335
- Operator-portable diagnostic on non-Apple-Silicon hardware
- Sister substrate empirical drift validation
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Primitive 1: dtype cast
# ---------------------------------------------------------------------------


def to_float32(x: Any) -> np.ndarray:
    """Cast input to numpy float32. Lossless from int/bool; safe from fp16/fp64."""
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Primitive 2: linear (matmul)
# ---------------------------------------------------------------------------


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """Apply linear layer: y = x @ weight.T + bias.

    Args:
        x: input shape (..., in_features)
        weight: shape (out_features, in_features); MLX/PyTorch canonical
        bias: optional shape (out_features,)

    Returns:
        Output shape (..., out_features). fp32 accumulation.
    """
    x32 = to_float32(x)
    w32 = to_float32(weight)
    y = np.matmul(x32, w32.T)
    if bias is not None:
        y = y + to_float32(bias)
    return y


# ---------------------------------------------------------------------------
# Primitive 3: conv2d NHWC (naive correctness reference)
# ---------------------------------------------------------------------------


def conv2d_nhwc(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    padding: int = 0,
) -> np.ndarray:
    """Apply 2D convolution in NHWC layout.

    Args:
        x: input shape (N, H, W, C_in)
        weight: shape (C_out, kH, kW, C_in) — MLX canonical NHWC layout
        bias: optional shape (C_out,)
        padding: symmetric int padding (default 0; canonical 'same' for 3x3 is 1)

    Returns:
        Output shape (N, H, W, C_out) when padding preserves spatial extent.

    Notes:
        - fp32 accumulation per Catalog #962 / slot 16 engineering corrections
        - canonical NHWC matches MLX layout (PyTorch's NCHW requires transpose
          at the export bridge per sister substrate dreamer_v3_rssm pattern)
        - This is a naive O(N*H*W*C_out*kH*kW*C_in) implementation; intended
          for correctness reference + small test fixtures, NOT production
          throughput.
    """
    x32 = to_float32(x)
    w32 = to_float32(weight)
    N, H, W, C_in = x32.shape
    C_out, kH, kW, C_in_w = w32.shape
    if C_in != C_in_w:
        raise ValueError(
            f"conv2d_nhwc: input C_in={C_in} != weight C_in={C_in_w}"
        )

    # Symmetric padding
    if padding > 0:
        x_padded = np.pad(
            x32, ((0, 0), (padding, padding), (padding, padding), (0, 0))
        )
    else:
        x_padded = x32

    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    H_out = H_padded - kH + 1
    W_out = W_padded - kW + 1

    # Naive nested-loop conv (correctness reference; not throughput-optimal)
    y = np.zeros((N, H_out, W_out, C_out), dtype=np.float32)
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                patch = x_padded[n, i:i + kH, j:j + kW, :]
                patch_flat = patch.reshape(-1)
                w_flat = w32.reshape(C_out, kH * kW * C_in)
                y[n, i, j, :] = w_flat @ patch_flat

    if bias is not None:
        y = y + to_float32(bias)[None, None, None, :]
    return y


# ---------------------------------------------------------------------------
# Primitive 4: bilinear upsample ×2 NHWC, align_corners=False (PyTorch default)
# ---------------------------------------------------------------------------


def bilinear_upsample_2x_nhwc(x: np.ndarray) -> np.ndarray:
    """Bilinear upsample input by 2× spatially in NHWC layout, align_corners=False.

    Args:
        x: input shape (N, H, W, C)

    Returns:
        Output shape (N, 2*H, 2*W, C).

    Notes:
        - **CANONICAL**: align_corners=False (PyTorch default + canonical
          per CLAUDE.md sister A=DreamerV3 documented max_abs=24.34 gap
          caused by align_corners=True via mx.repeat substitution).
        - Each output pixel maps to input via formula:
            src_x = (dst_x + 0.5) / 2.0 - 0.5
            src_y = (dst_y + 0.5) / 2.0 - 0.5
          then clamped to [0, H-1] × [0, W-1].
        - fp32 accumulation; ≤ 1e-5 parity vs PyTorch
          F.interpolate(scale_factor=2, mode='bilinear', align_corners=False).
        - Sister of MLX canonical
          ``tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc``;
          the MLX helper produces 0.0 absolute drift vs PyTorch; this numpy
          reference produces ≤ 1e-5 drift (fp32 accumulation noise floor).
    """
    x32 = to_float32(x)
    N, H, W, C = x32.shape
    H_out = 2 * H
    W_out = 2 * W

    dst_y = np.arange(H_out, dtype=np.float32)
    dst_x = np.arange(W_out, dtype=np.float32)
    src_y = (dst_y + 0.5) * 0.5 - 0.5
    src_x = (dst_x + 0.5) * 0.5 - 0.5

    src_y = np.clip(src_y, 0.0, H - 1.0)
    src_x = np.clip(src_x, 0.0, W - 1.0)

    y0 = np.floor(src_y).astype(np.int32)
    y1 = np.minimum(y0 + 1, H - 1)
    x0 = np.floor(src_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    fy = src_y - y0.astype(np.float32)
    fx = src_x - x0.astype(np.float32)

    y0_grid, x0_grid = np.meshgrid(y0, x0, indexing='ij')
    y1_grid, x1_grid = np.meshgrid(y1, x1, indexing='ij')
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing='ij')

    w00 = (1.0 - fx_grid) * (1.0 - fy_grid)
    w01 = fx_grid * (1.0 - fy_grid)
    w10 = (1.0 - fx_grid) * fy_grid
    w11 = fx_grid * fy_grid

    p00 = x32[:, y0_grid, x0_grid, :]
    p01 = x32[:, y0_grid, x1_grid, :]
    p10 = x32[:, y1_grid, x0_grid, :]
    p11 = x32[:, y1_grid, x1_grid, :]

    w00 = w00[None, :, :, None]
    w01 = w01[None, :, :, None]
    w10 = w10[None, :, :, None]
    w11 = w11[None, :, :, None]

    return w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11


# ---------------------------------------------------------------------------
# Primitive 5: sigmoid (numerically stable)
# ---------------------------------------------------------------------------


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: 1/(1+exp(-x)).

    For large negative x, exp(-x) overflows; use stable form:
        sigmoid(x) = 1/(1+exp(-x))                  for x >= 0
        sigmoid(x) = exp(x)/(1+exp(x))              for x < 0
    """
    x32 = to_float32(x)
    pos_mask = x32 >= 0
    out = np.empty_like(x32)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x32[pos_mask]))
    exp_x = np.exp(x32[~pos_mask])
    out[~pos_mask] = exp_x / (1.0 + exp_x)
    return out


# ---------------------------------------------------------------------------
# Primitive 6: sin (NeRV-canonical positional encoding)
# ---------------------------------------------------------------------------


def sin(x: np.ndarray) -> np.ndarray:
    """Element-wise sin; fp32 reference."""
    return np.sin(to_float32(x))


# ---------------------------------------------------------------------------
# Primitive 7: mean reduction (with optional Kahan summation)
# ---------------------------------------------------------------------------


def mean(x: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    """Standard mean reduction; fp32 accumulation."""
    return np.mean(to_float32(x), axis=axis)


def kahan_mean(x: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    """Kahan-summation mean for large-N batch aggregations.

    Per Catalog #962 / slot 16 engineering corrections: non-Kahan summation
    accumulates rounding error at large N (>1e6 elements). Kahan summation
    bounds the accumulation error to ~machine-epsilon per element regardless
    of N.

    This is the recommended canonical helper for batch-level loss
    aggregation when reproducibility across hardware/runtime is required.
    """
    x32 = to_float32(x).flatten() if axis is None else to_float32(x)
    if axis is None:
        # Kahan summation along flattened array
        total = np.float32(0.0)
        comp = np.float32(0.0)
        for val in x32:
            y = val - comp
            t = total + y
            comp = (t - total) - y
            total = t
        return np.float32(total / float(len(x32)))
    # For non-None axis, fall back to numpy mean (Kahan along axis would be
    # significantly more complex; the canonical pattern is whole-array
    # reduction)
    return np.mean(x32, axis=axis)


__all__ = [
    "bilinear_upsample_2x_nhwc",
    "conv2d_nhwc",
    "kahan_mean",
    "linear",
    "mean",
    "sigmoid",
    "sin",
    "to_float32",
]
