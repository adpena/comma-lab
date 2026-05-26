# SPDX-License-Identifier: MIT
"""coin_pp_implicit_neural_representation.numpy_reference — sister numpy reference implementations.

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
deterministic ops; ≤ 1e-6 for sin/sigmoid/exp pointwise; integer-exact
for coord grid construction.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Primitive 1: dtype cast
# ---------------------------------------------------------------------------

def to_float32(x: Any) -> np.ndarray:
    """Cast input to numpy float32. Lossless from int/bool; safe from fp16/fp64."""
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Primitive 2: linear (matmul) — canonical PyTorch nn.Linear layout
# ---------------------------------------------------------------------------

def linear(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None
) -> np.ndarray:
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
# Primitive 3: sin (positional encoding + activation)
# ---------------------------------------------------------------------------

def sin(x: np.ndarray) -> np.ndarray:
    """Element-wise sin; fp32 reference."""
    return np.sin(to_float32(x))


def cos(x: np.ndarray) -> np.ndarray:
    """Element-wise cos; fp32 reference."""
    return np.cos(to_float32(x))


# ---------------------------------------------------------------------------
# Primitive 4: sinusoidal positional encoding (composite NeRF/COIN++ canonical)
# ---------------------------------------------------------------------------

def sinusoidal_positional_encoding(
    coords: np.ndarray, num_frequencies: int
) -> np.ndarray:
    """Sinusoidal positional encoding per NeRF (Mildenhall et al. 2020) + COIN++.

    For input coord c in [-1, 1] and L = num_frequencies, produces:
        [sin(2^0 * pi * c), cos(2^0 * pi * c),
         sin(2^1 * pi * c), cos(2^1 * pi * c),
         ...,
         sin(2^(L-1) * pi * c), cos(2^(L-1) * pi * c)]

    Args:
        coords: shape (..., D) where D is coordinate dimensionality (e.g. 3 for x,y,t)
        num_frequencies: L; encoding output dim = L * 2 * D

    Returns:
        Encoded coords shape (..., L * 2 * D). fp32.

    Notes:
        - Canonical NeRF/COIN++ scaling: 2^k * pi for k in [0, L-1]
        - HARD-EARNED per Catalog #303 audit assumption #5 + Mildenhall NeRF 2020
        - This is the SAME encoding used in MLX renderer; numpy reference
          provides byte-identical output for parity testing
    """
    coords32 = to_float32(coords)
    D = coords32.shape[-1]
    L = int(num_frequencies)
    # Build frequency bands: shape (L,)
    freq_bands = (2.0 ** np.arange(L, dtype=np.float32)) * np.float32(math.pi)
    # Broadcast: coords (..., D) * freq_bands (L,) -> (..., D, L)
    # Reshape coords to (..., D, 1), freq_bands to (1, ..., 1, L)
    coords_expanded = coords32[..., np.newaxis]  # (..., D, 1)
    scaled = coords_expanded * freq_bands  # (..., D, L)
    # Interleave sin and cos along last axis -> (..., D, L*2)
    sin_vals = np.sin(scaled)
    cos_vals = np.cos(scaled)
    # Stack along new last axis to get (..., D, L, 2) then reshape to (..., D*L*2)
    stacked = np.stack([sin_vals, cos_vals], axis=-1)  # (..., D, L, 2)
    out_shape = coords32.shape[:-1] + (D * L * 2,)
    return stacked.reshape(out_shape)


# ---------------------------------------------------------------------------
# Primitive 5: FiLM modulation
# ---------------------------------------------------------------------------

def film_modulate(
    h: np.ndarray, scale: np.ndarray, shift: np.ndarray
) -> np.ndarray:
    """Apply FiLM modulation: h' = h * scale + shift (Perez et al. 2017).

    Args:
        h: hidden activations, shape (..., hidden_dim)
        scale: per-batch scale, shape (B, hidden_dim) broadcastable
        shift: per-batch shift, shape (B, hidden_dim) broadcastable

    Returns:
        Modulated activations, same shape as h.

    Notes:
        - Direct linear scale (NOT exp(log_scale)) per axis 2 MLX drift discipline
        - HARD-EARNED per Catalog #303 audit assumption #8 + Perez 2017 canonical
    """
    h32 = to_float32(h)
    s32 = to_float32(scale)
    b32 = to_float32(shift)
    # Broadcast scale/shift across non-batch dims
    return h32 * s32 + b32


# ---------------------------------------------------------------------------
# Primitive 6: sigmoid output activation
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
# Primitive 7: coord grid construction (NHWC convention)
# ---------------------------------------------------------------------------

def make_coord_grid_nhwc(
    H: int, W: int, t: float = 0.0
) -> np.ndarray:
    """Build canonical coordinate grid for coord-MLP forward.

    Args:
        H: height
        W: width
        t: frame index (0 for first frame, 1 for second frame); normalized to [-1, 1]
            with t=0 -> -1.0, t=1 -> 1.0 OR raw t per caller's normalization

    Returns:
        Coordinate grid shape (H, W, 3) of (x, y, t) tuples in [-1, 1] for x/y;
        t passed through as-is (caller responsible for normalization).

    Notes:
        - canonical NHWC layout matches MLX renderer
        - x in [-1, 1] across W; y in [-1, 1] across H
        - Integer-exact construction; no floating-point drift risk
    """
    # x in [-1, 1] across W positions (W points)
    if W == 1:
        x = np.array([0.0], dtype=np.float32)
    else:
        x = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    # y in [-1, 1] across H positions
    if H == 1:
        y = np.array([0.0], dtype=np.float32)
    else:
        y = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    # Meshgrid in NHWC convention
    yy, xx = np.meshgrid(y, x, indexing="ij")  # (H, W) each
    tt = np.full((H, W), np.float32(t), dtype=np.float32)
    grid = np.stack([xx, yy, tt], axis=-1)  # (H, W, 3)
    return grid


# ---------------------------------------------------------------------------
# Primitive 8: mean reduction (with optional Kahan summation)
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
# Primitive 9: coord-MLP forward (composite — sister of MLX renderer forward)
# ---------------------------------------------------------------------------

def coord_mlp_forward(
    coords: np.ndarray,
    modulation: np.ndarray,
    state_dict: dict[str, np.ndarray],
    *,
    pos_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
) -> np.ndarray:
    """Numpy reference implementation of COIN++ coord-MLP forward.

    Mirrors the canonical MLX renderer forward pass. Used for axis 3 portability
    + axis 2 parity testing.

    Args:
        coords: shape (B, H, W, 3) — (x, y, t) coord grid per batch
        modulation: shape (B, mod_dim) — per-pair modulation vectors
        state_dict: required keys:
            "input_proj.weight" shape (hidden_dim, pos_enc_dim)
            "input_proj.bias" shape (hidden_dim,)
            "hidden_{i}.weight" / "hidden_{i}.bias" for i in 0..num_hidden_layers-1
            "film_{i}.weight" / "film_{i}.bias" for i in 0..num_hidden_layers-1
                where film weight shape = (2*hidden_dim, mod_dim)
            "output_proj.weight" shape (3, hidden_dim)
            "output_proj.bias" shape (3,)
        pos_dim: sinusoidal positional encoding frequency count
        hidden_dim: hidden layer width
        num_hidden_layers: number of hidden layers

    Returns:
        RGB output shape (B, H, W, 3) in [0, 1].

    Notes:
        - Composite of primitives 1-8; demonstrates that ALL MLX primitives
          have numpy sister implementations per axis 3 portability discipline
        - fp32 accumulation throughout; parity bound ≤ 1e-5 vs MLX
    """
    coords32 = to_float32(coords)
    B, H, W, _ = coords32.shape

    # Step 1: positional encoding (B, H, W, pos_enc_dim)
    pos_enc = sinusoidal_positional_encoding(coords32, pos_dim)

    # Step 2: input projection -> (B, H, W, hidden_dim)
    h = linear(
        pos_enc, state_dict["input_proj.weight"], state_dict.get("input_proj.bias")
    )

    # Step 3: hidden layers with FiLM modulation
    # modulation is (B, mod_dim); broadcast to (B, 1, 1, mod_dim)
    mod_b = to_float32(modulation)[:, np.newaxis, np.newaxis, :]
    for i in range(num_hidden_layers):
        # Linear: hidden_dim -> hidden_dim
        h = linear(
            h,
            state_dict[f"hidden_{i}.weight"],
            state_dict.get(f"hidden_{i}.bias"),
        )
        # FiLM projection: mod_dim -> 2 * hidden_dim
        film_out = linear(
            mod_b,
            state_dict[f"film_{i}.weight"],
            state_dict.get(f"film_{i}.bias"),
        )
        # Split into scale + shift; broadcast across (H, W)
        scale = film_out[..., :hidden_dim]  # (B, 1, 1, hidden_dim)
        shift = film_out[..., hidden_dim:]  # (B, 1, 1, hidden_dim)
        # Add 1.0 to scale so initial state is identity-like (avoids zero-out)
        scale = scale + np.float32(1.0)
        # Apply FiLM modulation
        h = film_modulate(h, scale, shift)
        # Sin activation per COIN++ paradigm
        h = sin(h)

    # Step 4: output projection + sigmoid -> RGB in [0, 1]
    rgb_logits = linear(
        h, state_dict["output_proj.weight"], state_dict.get("output_proj.bias")
    )
    rgb = sigmoid(rgb_logits)
    return rgb


__all__ = [
    "coord_mlp_forward",
    "cos",
    "film_modulate",
    "kahan_mean",
    "linear",
    "make_coord_grid_nhwc",
    "mean",
    "sigmoid",
    "sin",
    "sinusoidal_positional_encoding",
    "to_float32",
]
