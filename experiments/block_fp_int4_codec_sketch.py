#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""SKETCH: int4 signed block-FP variant (PR106 revival path).

Per memory feedback_block_fp_codec_pr106_unviable_at_default_ternary_20260504.md,
the existing src/tac/block_fp_codec.py is ternary-only ({-1, 0, +1}) — purpose-built
for Selfcomp's bimodal-regularized weight distribution. PR106's HNeRV decoder has
continuous-distribution weights → ternary destroys them (max_err ~0.5-1.0).

This sketch implements an INT4 SIGNED variant: qint values in {-7..+7} (15 levels)
with per-block shared float32 exponent. Expected effective bit cost:
    4 bits / weight (qint) + 32 bits / block / block_size (exponent) ≈ 4.25 bits/weight at block_size=128
Compared to ternary: 1.58 bits/weight + same exponent overhead.

SKETCH-ONLY status — NOT in src/tac/ yet. Per CLAUDE.md "Adversarial council review
of design decisions" rule: adding int4 to production block_fp_codec requires council
review before promotion (signed vs unsigned, asymmetric vs symmetric, block size
calibration). This experimental file lets us measure compressibility + round-trip
fidelity on PR106 weights at zero risk to the production codec.

Usage:
    .venv/bin/python experiments/block_fp_int4_codec_sketch.py
"""
from __future__ import annotations

import struct
from typing import Tuple

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

INT4_MIN = -7  # signed 4-bit range: -7..+7 (15 levels; -8 reserved for "reserved" / future use)
INT4_MAX = +7
INT4_LEVELS = INT4_MAX - INT4_MIN  # = 14 (number of step intervals)


def _pack_block_fp_int4_arrays(
    weight: torch.Tensor,
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode along axis-0 in blocks of `block_size` rows.

    For each block, find max(|w|), set scale = max(|w|) / 7. Then qint = round(w / scale)
    clamped to [-7, +7]. Reconstructed weight = qint * scale.

    Returns:
        qint: int8 array (one int per weight; only -7..+7 used; packed to 4-bit at write time).
        scales: float32 array (num_blocks,) — one scale per block.
    """
    if weight.numel() == 0:
        return np.zeros(weight.shape, dtype=np.int8), np.zeros((0,), dtype=np.float32)
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")

    w = weight.detach().to(torch.float32)
    n_rows = w.shape[0]
    n_blocks = (n_rows + block_size - 1) // block_size

    qint = torch.zeros_like(w, dtype=torch.int8)
    scales = torch.zeros((n_blocks,), dtype=torch.float32)

    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n_rows)
        block = w[start:end]
        max_abs = block.abs().max().item()
        if max_abs == 0:
            scales[b] = 0.0
            continue
        scale = max_abs / INT4_MAX
        scales[b] = scale
        qint[start:end] = (block / scale).round().clamp(INT4_MIN, INT4_MAX).to(torch.int8)

    return qint.numpy(), scales.numpy()


def _unpack_block_fp_int4_arrays(
    qint: np.ndarray,
    scales: np.ndarray,
    block_size: int,
) -> torch.Tensor:
    """Inverse of _pack_block_fp_int4_arrays."""
    qint_t = torch.from_numpy(qint).to(torch.float32)
    n_rows = qint_t.shape[0]
    n_blocks = scales.shape[0]
    out = torch.zeros_like(qint_t)
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n_rows)
        out[start:end] = qint_t[start:end] * float(scales[b])
    return out


def _pack_int4_signed_to_bytes(qint: np.ndarray) -> bytes:
    """Pack signed int4 (range -7..+7) into 4-bit nibbles. Two values per byte.
    Encoding: shift signed -7..+7 to unsigned 0..14 then pack nibbles big-endian.
    """
    flat = qint.astype(np.int32).flatten()
    if flat.size % 2 != 0:
        flat = np.concatenate([flat, np.array([0], dtype=np.int32)])  # pad
    unsigned = (flat - INT4_MIN).astype(np.uint8)  # 0..14
    high = unsigned[::2] << 4
    low = unsigned[1::2] & 0x0F
    return (high | low).tobytes()


def _unpack_int4_signed_from_bytes(data: bytes, count: int) -> np.ndarray:
    """Inverse of _pack_int4_signed_to_bytes; truncates pad if `count` is odd."""
    arr = np.frombuffer(data, dtype=np.uint8)
    high = (arr >> 4) & 0x0F
    low = arr & 0x0F
    interleaved = np.empty(arr.size * 2, dtype=np.uint8)
    interleaved[0::2] = high
    interleaved[1::2] = low
    interleaved = interleaved[:count]
    return (interleaved.astype(np.int32) + INT4_MIN).astype(np.int8)


def encode_int4_block_fp(weight: torch.Tensor, block_size: int = 128) -> bytes:
    """Sketch: encode a Conv2d-shaped weight via int4 signed block-FP."""
    qint, scales = _pack_block_fp_int4_arrays(weight, block_size)
    qint_bytes = _pack_int4_signed_to_bytes(qint)
    scales_bytes = scales.tobytes()
    # 16-byte header: ndim(1B) + shape(ndim*4B) + block_size(4B) + n_blocks(4B) + qint_count(4B) + qint_byte_len(4B)
    shape = tuple(weight.shape)
    ndim = len(shape)
    qint_count = int(weight.numel())
    header = struct.pack("<B", ndim)
    for s in shape:
        header += struct.pack("<I", int(s))
    header += struct.pack("<IIII", int(block_size), int(scales.shape[0]), qint_count, len(qint_bytes))
    return header + qint_bytes + scales_bytes


def decode_int4_block_fp(data: bytes) -> torch.Tensor:
    """Inverse of encode_int4_block_fp."""
    pos = 0
    ndim = struct.unpack_from("<B", data, pos)[0]; pos += 1
    shape = struct.unpack_from(f"<{ndim}I", data, pos); pos += 4 * ndim
    block_size = struct.unpack_from("<I", data, pos)[0]; pos += 4
    n_blocks = struct.unpack_from("<I", data, pos)[0]; pos += 4
    qint_count = struct.unpack_from("<I", data, pos)[0]; pos += 4
    qint_byte_len = struct.unpack_from("<I", data, pos)[0]; pos += 4
    qint_bytes = data[pos:pos + qint_byte_len]; pos += qint_byte_len
    scales_bytes = data[pos:pos + 4 * n_blocks]; pos += 4 * n_blocks
    qint = _unpack_int4_signed_from_bytes(qint_bytes, qint_count).reshape(shape)
    scales = np.frombuffer(scales_bytes, dtype=np.float32)
    return _unpack_block_fp_int4_arrays(qint, scales, block_size).reshape(shape)


def _smoke_test() -> int:
    """Run sketch on PR106 Conv2d weights and report bytes + round-trip error."""
    sd = torch.load(
        "experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt",
        map_location="cpu",
        weights_only=False,
    )
    print(f"PR106 state_dict: {len(sd)} tensors, {sum(t.numel() for t in sd.values())} params")
    print()
    print("--- per-tensor int4 block-FP + brotli (PR106 Conv2d weights) ---")
    for block_size in [16, 32, 64, 128]:
        total_pack = 0
        total_brotli = 0
        max_err_global = 0.0
        for name, t in sd.items():
            if t.dim() != 4:
                continue
            blob = encode_int4_block_fp(t.to(torch.float32), block_size=block_size)
            br = brotli.compress(blob, quality=11)
            rt = decode_int4_block_fp(blob)
            err = (rt - t).abs().max().item()
            total_pack += len(blob)
            total_brotli += len(br)
            max_err_global = max(max_err_global, err)
        print(
            f"  block_size={block_size}: pack={total_pack} brotli={total_brotli} "
            f"max_err={max_err_global:.4g}"
        )
    print()
    print("Reference: PR106 brotli encodes whole decoder (~228K params Conv2d) at 170,278 bytes.")
    print("OWV2 stub (Lane Ω-W-V3 commit b2f958a4) saved -22,152 bytes vs brotli (147KB Conv2d portion).")
    print("If int4 block-FP brotli total is < 147KB AND max_err is < 1e-2, this is a viable Lane #04 revival.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
