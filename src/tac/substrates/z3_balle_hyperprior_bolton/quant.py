# SPDX-License-Identifier: MIT
"""Shared int8 quantization helpers for Z3 archive grammars."""
from __future__ import annotations

import numpy as np
import torch


def quantize_int8_with_scale(
    tensor: torch.Tensor, *, scale_clip_range: float = 7.0
) -> tuple[bytes, float]:
    """Quantize a float tensor to int8 with one scale factor."""
    abs_max = float(tensor.detach().abs().max().item())
    if abs_max <= 1e-12:
        return bytes(tensor.numel()), 1.0
    scale = scale_clip_range / abs_max
    q = (tensor.detach() * scale).round().clamp(-128, 127).to(torch.int8)
    return q.cpu().numpy().tobytes(), scale


def dequantize_int8_with_scale(
    int8_bytes: bytes,
    scale: float,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Inverse of :func:`quantize_int8_with_scale`."""
    arr = np.frombuffer(int8_bytes, dtype=np.int8).reshape(shape)
    return torch.from_numpy(arr.copy()).to(dtype) / scale


__all__ = ["dequantize_int8_with_scale", "quantize_int8_with_scale"]
