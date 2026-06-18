# SPDX-License-Identifier: MIT
"""Post-hoc weight bit-shrink primitives — REUSABLE per-tensor quantize-dequantize.

The shipped torch-vehicle codec (``submissions/hnerv_muon/src/codec.py``) stores the
HNeRV decoder weights as per-tensor symmetric INT8 (``n_quant=127``) → zigzag → brotli.
This module exposes the *quant grid* knobs that lower the effective bit-width WITHOUT
retraining, so a $0 PTQ feasibility sweep can ask "does the decoder TOLERATE bit-shrink?"
(d_seg/d_pose hold) and "what is the measured rate saving?" (re-encode + brotli the
shrunk weights through the SAME shipping codec).

Two grids are provided, both pure post-hoc (no QAT, no STE — these are quantize→dequantize
round-trips that return float tensors on the SAME grid the codec will store):

  * :func:`intn_qdq` — per-tensor symmetric int-N (n_quant = 2**(nbits-1) - 1). At nbits=8
    this is the shipped codec's exact grid (n_quant=127). Lower nbits = sparser grid =
    fewer distinct stored int8 levels = better brotli (the measured rate lever).
  * :func:`e2m1_qdq` — FP4 E2M1 (8 signed magnitudes {0,.5,1,1.5,2,3,4,6}) per-output-channel
    or per-tensor. A genuine 4-bit floating grid (denser near zero than int4).

NO hardware gate (these are float qdq round-trips, CPU-safe by construction; the
hardware-FP4/FP8 gate in :mod:`tac.quantization` is for ACTIVE low-bit kernels, a
different surface). NO score claim: a qdq round-trip changes the weights; whether that
costs d_seg/d_pose is an empirical question the caller measures with the real scorer.
"""

from __future__ import annotations

import torch

# FP4 E2M1 absolute magnitudes (the 8 representable |value|s of a 1-2-1 sign-exp-mantissa
# 4-bit float). The sign bit doubles this to 15 distinct values + signed-zero.
E2M1_ABS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def intn_qdq(t: torch.Tensor, nbits: int) -> torch.Tensor:
    """Per-tensor symmetric int-N quantize→dequantize (returns a float tensor).

    ``nbits=8`` reproduces the shipped codec grid (n_quant = 2**7 - 1 = 127). The scale
    is max-abs / n_quant; values are rounded to the int grid then scaled back. A zero
    tensor (or all-zero scale) is returned unchanged.

    Args:
        t: weight tensor (any shape).
        nbits: target bit-width (>= 2). 8 = codec baseline; 4/5/6 = shrink.

    Returns:
        Float tensor on the int-N grid (same shape/dtype/device as ``t``).
    """
    if nbits < 2:
        raise ValueError(f"nbits must be >= 2, got {nbits}")
    qmax = 2 ** (nbits - 1) - 1
    s = t.abs().max() / qmax
    if float(s) == 0.0:
        return t.clone()
    return (t / s).round().clamp_(-qmax, qmax) * s


def e2m1_qdq(t: torch.Tensor, *, per_channel: bool = True) -> torch.Tensor:
    """FP4 E2M1 quantize→dequantize (returns a float tensor on the 4-bit float grid).

    The 8 magnitudes :data:`E2M1_ABS` (× sign) form a real 4-bit floating grid (denser
    near zero than int4's uniform grid). Scale = max-abs / 6 (6 is the E2M1 max magnitude),
    per-output-channel (dim 0) by default for weight tensors, else per-tensor.

    Args:
        t: weight tensor.
        per_channel: per-output-channel scaling (dim 0) when ``t.dim() >= 2``.

    Returns:
        Float tensor on the E2M1 grid (same shape/dtype/device as ``t``).
    """
    grid = torch.tensor(E2M1_ABS, dtype=t.dtype, device=t.device)
    if per_channel and t.dim() >= 2:
        red = tuple(range(1, t.dim()))
        s = t.abs().amax(dim=red, keepdim=True) / 6.0
    else:
        s = t.abs().max() / 6.0
    s = s.clamp_min(1e-12)
    tn = t / s
    idx = (tn.abs().unsqueeze(-1) - grid).abs().argmin(dim=-1)
    return tn.sign() * grid[idx] * s


def requantize_decoder_state_dict(
    dec_sd: dict[str, torch.Tensor],
    mode: str,
    *,
    head_prefixes: tuple[str, ...] = ("skips.", "refine.", "rgb_0.", "rgb_1."),
    weight_min_dim: int = 2,
) -> dict[str, torch.Tensor]:
    """Return a NEW state dict with weight tensors re-quantized post-hoc per ``mode``.

    Only tensors with ``dim() >= weight_min_dim`` are quantized (biases / 1-D params keep
    fp32 — they are tiny and output-proximal). Modes:

      * ``"fp32"``        — passthrough (no quantization).
      * ``"int8".."int4"``— per-tensor symmetric int-N via :func:`intn_qdq`.
      * ``"fp4_all"``     — every weight tensor on E2M1.
      * ``"fp4_mixed"``   — output-proximal heads (``head_prefixes``) stay int8, interior
                            tensors go FP4 E2M1 (the macro-bridge hypothesis: d_seg lives
                            in the output-proximal 384×512 band so heads must stay precise).

    Args:
        dec_sd: decoder state dict (float tensors).
        mode: one of the modes above.
        head_prefixes: key prefixes treated as output-proximal heads (int8-kept in mixed).
        weight_min_dim: minimum tensor rank to quantize.

    Returns:
        A new state dict (does not mutate the input).
    """
    intn_modes = {"int8": 8, "int7": 7, "int6": 6, "int5": 5, "int4": 4, "int3": 3}
    out: dict[str, torch.Tensor] = {}
    for k, v in dec_sd.items():
        if not torch.is_tensor(v) or v.dim() < weight_min_dim:
            out[k] = v.clone()
            continue
        if mode == "fp32":
            out[k] = v.clone()
        elif mode in intn_modes:
            out[k] = intn_qdq(v, intn_modes[mode])
        elif mode == "fp4_all":
            out[k] = e2m1_qdq(v)
        elif mode == "fp4_mixed":
            is_head = any(k.startswith(p) for p in head_prefixes)
            out[k] = intn_qdq(v, 8) if is_head else e2m1_qdq(v)
        else:
            raise ValueError(f"unknown requant mode: {mode!r}")
    return out
