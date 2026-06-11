# SPDX-License-Identifier: MIT
"""NumPy reference oracle for the Cool-Chic synthesis + ARM rate math.

This is the **portability-contract reference** for the Cool-Chic non-conv basis
(lane ``lane_cool_chic_score_aware_basis_20260611``). Per the operator binding
2026-06-09 ("MLX-first everything; numpy reference = portability contract"),
the MLX/torch fast paths in :mod:`tac.residual_basis.cool_chic_carrier` must
match THIS module bit-for-bit (within float tolerance) on REAL inputs — NOT on
zero-init (the grid-PE fake-parity lesson: a zero-init tensor passes any
parity check because every backend renders zeros identically).

Cool-Chic (Ladune et al. 2023, ICCV) represents an image as:

  1. ``n_grids`` multiresolution latent grids ``z_l`` of shape
     ``(C_l, H/2^l, W/2^l)`` (coarse-to-fine). The total latent count is the
     ONLY large charged payload — the synthesis net is ~hundreds of params.
  2. an **autoregressive module (ARM)** that predicts, for each latent value,
     a ``(mu, log_scale)`` from its already-decoded causal spatial neighbors.
     The entropy-coded rate of a grid is ``sum -log2 p(z | mu, scale)`` under
     a Laplacian (or logistic) density. This is the charged *latent* bytes.
  3. a tiny **synthesis** ConvNet that maps the bilinear-upsampled,
     channel-concatenated grids to RGB.

We implement the forward synthesis + the ARM rate estimate in pure numpy so a
deterministic, backend-independent oracle exists. The MLX path is the fast
fitter; the torch path is what plugs into the live-SegNet ``ScoreAwareTrainer``;
this numpy path is the proof both agree.

References
----------
Ladune, T., Philippe, P., Hamidouche, W., Henry, F., & Deforges, O. (2023).
"Cool-Chic: Coordinate-based Low Complexity Hierarchical Image Codec." ICCV.
Leguay et al. (2024) "Cool-chic video" (hal.science) — competitive fidelity at
~hundreds of synthesis params (the basis-specific-wall hypothesis source).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LOG2_E = 1.0 / math.log(2.0)


def bilinear_upsample_numpy(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear-upsample ``(C, h, w)`` -> ``(C, out_h, out_w)``.

    Uses ``align_corners=False`` sampling — the SAME convention as
    ``torch.nn.functional.interpolate(..., mode='bilinear', align_corners=False)``
    and the MLX path — so all three backends agree.
    """
    if grid.ndim != 3:
        raise ValueError(f"grid must be (C, h, w); got {grid.shape}")
    c, h, w = grid.shape
    if h == out_h and w == out_w:
        return grid.astype(np.float64, copy=True)
    # align_corners=False source coordinate mapping.
    ys = (np.arange(out_h, dtype=np.float64) + 0.5) * (h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float64) + 0.5) * (w / out_w) - 0.5
    ys = np.clip(ys, 0.0, h - 1)
    xs = np.clip(xs, 0.0, w - 1)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None]  # (out_h, 1)
    wx = (xs - x0)[None, :]  # (1, out_w)
    g = grid.astype(np.float64, copy=False)
    # gather corners -> (C, out_h, out_w)
    top = g[:, y0][:, :, x0] * (1 - wx) + g[:, y0][:, :, x1] * wx
    bot = g[:, y1][:, :, x0] * (1 - wx) + g[:, y1][:, :, x1] * wx
    return top * (1 - wy) + bot * wy


def laplacian_rate_bits_numpy(
    z: np.ndarray, mu: np.ndarray, log_scale: np.ndarray, *, quant_step: float = 1.0
) -> np.ndarray:
    """Per-element entropy-coded rate (bits) under a Laplacian density.

    Cool-Chic charges ``-log2 P(z_q)`` where ``z_q`` is the quantized latent and
    ``P`` integrates the Laplacian density over the quantization bin of width
    ``quant_step`` centered at ``z_q``. We use the standard continuous-density
    approximation ``P ≈ density(z) * quant_step`` (valid when scale >> step),
    which is the rate term Cool-Chic back-props through. Returns per-element
    bits ``>= 0``.
    """
    scale = np.exp(log_scale.astype(np.float64))
    scale = np.maximum(scale, 1e-6)
    abs_dev = np.abs(z.astype(np.float64) - mu.astype(np.float64))
    # Laplacian density p(x) = 1/(2b) exp(-|x-mu|/b); rate = -log2(p * step).
    density = (1.0 / (2.0 * scale)) * np.exp(-abs_dev / scale)
    prob = np.clip(density * quant_step, 1e-12, 1.0)
    return -np.log2(prob)


@dataclass(frozen=True)
class SynthesisWeightsNumpy:
    """Weights for the tiny synthesis net (two 1x1 conv layers + GELU).

    A 1x1 conv over the concatenated upsampled grids is the smallest faithful
    Cool-Chic synthesis (the spatial structure lives in the grids; the synth
    just mixes channels -> RGB). ``w1`` mixes ``c_in -> c_hidden``; ``w2`` mixes
    ``c_hidden -> 3``. This keeps the synthesis param count at
    ``c_in*c_hidden + c_hidden*3`` — hundreds, not the conv-HNeRV's 162K+.
    """

    w1: np.ndarray  # (c_hidden, c_in)
    b1: np.ndarray  # (c_hidden,)
    w2: np.ndarray  # (3, c_hidden)
    b2: np.ndarray  # (3,)


def _gelu(x: np.ndarray) -> np.ndarray:
    # erf via scipy-free vectorize; clip the argument so a pathological (un-fit)
    # weight perturbation cannot overflow the reference oracle itself.
    arg = np.clip(x / math.sqrt(2.0), -30.0, 30.0)
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(arg))


def synthesize_rgb_numpy(
    grids: list[np.ndarray],
    weights: SynthesisWeightsNumpy,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Forward synthesis: multiresolution grids -> ``(3, out_h, out_w)`` in [0,1].

    Each grid is bilinear-upsampled to ``(out_h, out_w)``, channel-concatenated,
    then a 1x1 conv (matmul over channels) -> GELU -> 1x1 conv -> sigmoid.
    """
    ups = [bilinear_upsample_numpy(g, out_h, out_w) for g in grids]
    feat = np.concatenate(ups, axis=0)  # (c_in, H, W)
    c_in, h, w = feat.shape
    flat = feat.reshape(c_in, h * w)  # (c_in, HW)
    hidden = weights.w1 @ flat + weights.b1[:, None]  # (c_hidden, HW)
    hidden = _gelu(hidden)
    out = weights.w2 @ hidden + weights.b2[:, None]  # (3, HW)
    out = 1.0 / (1.0 + np.exp(-out))  # sigmoid -> [0,1]
    return out.reshape(3, h, w)


def total_synthesis_param_count(c_in: int, c_hidden: int) -> int:
    """Charged synthesis-net param count (w1 + b1 + w2 + b2)."""
    return c_in * c_hidden + c_hidden + c_hidden * 3 + 3
