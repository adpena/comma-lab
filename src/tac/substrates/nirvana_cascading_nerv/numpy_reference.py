# SPDX-License-Identifier: MIT
"""nirvana_cascading_nerv.numpy_reference — sister numpy reference implementations.

Per operator directive #3 2026-05-26 verbatim:
*"adversarial review against all landing recursive for math and scientific
and engineering rigor and for MLX drift minimization and portability via
numpy"*

Every MLX primitive used by ``mlx_renderer.py`` MUST have a sister numpy
reference implementation in this module OR documented non-portability
rationale.

Portability per Catalog #1 device-selection-defaults discipline; enables:
(a) GHA CPU CI testing per Catalog #178 + #179 without MLX install,
(b) sister cathedral consumer cross-validation per Catalog #335,
(c) operator-portable diagnostic on non-Apple-Silicon hardware.

This module is canonical-portable: numpy only, no MLX import, no torch
import. Operable on any Python+numpy install.

Per-primitive parity bound vs MLX/PyTorch reference: ≤ 1e-5 for fp32
deterministic ops; ≤ 1e-3 for fp16 accumulation.
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
# Primitive 3: conv2d NHWC
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
                # Extract patch: shape (kH, kW, C_in)
                patch = x_padded[n, i:i + kH, j:j + kW, :]
                # Reshape to (C_in * kH * kW,)
                patch_flat = patch.reshape(-1)
                # Weight reshape: (C_out, C_in * kH * kW)
                # But weight is (C_out, kH, kW, C_in); flatten last 3 dims
                w_flat = w32.reshape(C_out, kH * kW * C_in)
                # Wait — patch is (kH, kW, C_in); flatten matches if order matches
                # Reorder weight to (C_out, kH, kW, C_in) flattened same way:
                # patch.reshape(-1) goes (kH, kW, C_in) row-major
                # weight.reshape(C_out, -1) goes (kH, kW, C_in) row-major
                # So weight @ patch is correct
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
    """
    x32 = to_float32(x)
    N, H, W, C = x32.shape
    H_out = 2 * H
    W_out = 2 * W

    # Compute output → input coordinate mapping per align_corners=False
    # PyTorch formula: src = (dst + 0.5) / scale - 0.5
    dst_y = np.arange(H_out, dtype=np.float32)
    dst_x = np.arange(W_out, dtype=np.float32)
    src_y = (dst_y + 0.5) * 0.5 - 0.5
    src_x = (dst_x + 0.5) * 0.5 - 0.5

    # Clamp to valid range
    src_y = np.clip(src_y, 0.0, H - 1.0)
    src_x = np.clip(src_x, 0.0, W - 1.0)

    # Get integer corner indices + fractional offsets
    y0 = np.floor(src_y).astype(np.int32)
    y1 = np.minimum(y0 + 1, H - 1)
    x0 = np.floor(src_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    fy = src_y - y0.astype(np.float32)
    fx = src_x - x0.astype(np.float32)

    # Bilinear interpolation: f(x, y) = (1-fx)(1-fy) f(x0,y0) + fx(1-fy) f(x1,y0)
    #                                  + (1-fx)fy f(x0,y1) + fx fy f(x1,y1)
    # Broadcast for all H_out x W_out output positions
    # Build per-position weight tensors
    y0_grid, x0_grid = np.meshgrid(y0, x0, indexing='ij')  # (H_out, W_out)
    y1_grid, x1_grid = np.meshgrid(y1, x1, indexing='ij')
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing='ij')

    w00 = (1.0 - fx_grid) * (1.0 - fy_grid)  # (H_out, W_out)
    w01 = fx_grid * (1.0 - fy_grid)
    w10 = (1.0 - fx_grid) * fy_grid
    w11 = fx_grid * fy_grid

    # Index: x32[n, y, x, c]; for each output (i, j): need x32[:, y0[i], x0[j], :]
    # Use advanced indexing
    # x32 shape (N, H, W, C); we want output (N, H_out, W_out, C)
    p00 = x32[:, y0_grid, x0_grid, :]  # (N, H_out, W_out, C)
    p01 = x32[:, y0_grid, x1_grid, :]
    p10 = x32[:, y1_grid, x0_grid, :]
    p11 = x32[:, y1_grid, x1_grid, :]

    w00 = w00[None, :, :, None]  # (1, H_out, W_out, 1)
    w01 = w01[None, :, :, None]
    w10 = w10[None, :, :, None]
    w11 = w11[None, :, :, None]

    return w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11


# ---------------------------------------------------------------------------
# Primitive 5: sigmoid
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


# ---------------------------------------------------------------------------
# Hierarchical residual cascade reference (composition of primitives)
# ---------------------------------------------------------------------------

def cascade_reconstruct(
    base_rgb_nhwc: np.ndarray,
    residuals_nhwc: list[np.ndarray],
) -> np.ndarray:
    """Reference reconstruction for hierarchical residual cascade.

    Mirrors the canonical sequence in the MLX renderer:
        level0 → upsample → +level1_residual → upsample → +level2_residual → ...

    Args:
        base_rgb_nhwc: shape (N, H0, W0, 3) — level 0 base RGB in [0, 1]
        residuals_nhwc: list of (N, H_i, W_i, 3) residual tensors; each
            level i's H_i = 2 × H_{i-1}, W_i = 2 × W_{i-1}; residuals
            already dequantized to fp32

    Returns:
        Final RGB shape (N, H_final, W_final, 3) in [0, 1] (clamped).
    """
    current = to_float32(base_rgb_nhwc)
    for residual in residuals_nhwc:
        residual32 = to_float32(residual)
        upsampled = bilinear_upsample_2x_nhwc(current)
        if upsampled.shape != residual32.shape:
            raise ValueError(
                f"cascade_reconstruct: upsampled shape {upsampled.shape} != "
                f"residual shape {residual32.shape}"
            )
        current = upsampled + residual32
        current = np.clip(current, 0.0, 1.0)
    return current


__all__ = [
    "bilinear_upsample_2x_nhwc",
    "cascade_reconstruct",
    "conv2d_nhwc",
    "kahan_mean",
    "linear",
    "mean",
    "sigmoid",
    "sin",
    "to_float32",
]
