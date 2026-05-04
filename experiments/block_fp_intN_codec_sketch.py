#!/usr/bin/env python3
"""SKETCH: signed int-N block-FP variant (parameterized; supports int3..int8).

Generalizes experiments/block_fp_int4_codec_sketch.py to arbitrary signed bit width.

Compared to int4 (-7..+7, ~7.09% intrinsic relative error per weight):
- int3 (-3..+3, 3 levels per side) — ~14% rel error, ~3 bits/weight
- int4 (-7..+7, 7 levels per side) — ~7.09% rel error, ~4 bits/weight
- int5 (-15..+15, 15 levels per side) — ~3.33% rel error, ~5 bits/weight
- int6 (-31..+31, 31 levels per side) — ~1.61% rel error, ~6 bits/weight

The Pareto frontier of bytes vs error is what we're mapping. Each tighter
quantizer trades bytes for fidelity.

Usage:
    .venv/bin/python experiments/block_fp_intN_codec_sketch.py
"""
from __future__ import annotations

import struct
from typing import Tuple

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch


def _intN_range(bits: int) -> tuple[int, int]:
    """Return (min, max) for signed N-bit qint with -2^(N-1)+1 .. +2^(N-1)-1.
    Reserves -2^(N-1) for future use (matching int4 sketch convention)."""
    half = 1 << (bits - 1)
    return -(half - 1), (half - 1)


def _pack_block_fp_intN_arrays(
    weight: torch.Tensor,
    block_size: int,
    bits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode along axis-0 in blocks of `block_size` rows, signed int-N quantization."""
    if weight.numel() == 0:
        return np.zeros(weight.shape, dtype=np.int8), np.zeros((0,), dtype=np.float32)
    if not 2 <= bits <= 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")

    qmin, qmax = _intN_range(bits)
    w = weight.detach().to(torch.float32)
    n_rows = w.shape[0]
    n_blocks = (n_rows + block_size - 1) // block_size

    qint = torch.zeros_like(w, dtype=torch.int8 if bits <= 8 else torch.int16)
    scales = torch.zeros((n_blocks,), dtype=torch.float32)

    for b in range(n_blocks):
        start, end = b * block_size, min((b + 1) * block_size, n_rows)
        block = w[start:end]
        max_abs = block.abs().max().item()
        if max_abs == 0:
            scales[b] = 0.0
            continue
        scale = max_abs / qmax
        scales[b] = scale
        qint[start:end] = (block / scale).round().clamp(qmin, qmax).to(qint.dtype)

    return qint.numpy(), scales.numpy()


def _pack_intN_to_bytes(qint: np.ndarray, bits: int) -> bytes:
    """Pack signed intN values into a packed bit stream.
    For bits=4: 2 values per byte (high nibble + low nibble), bits=8: 1 byte each.
    For bits in {3, 5, 6, 7}: pack 8 values into bits bytes via bit-shifts."""
    qmin, _ = _intN_range(bits)
    flat = qint.astype(np.int32).flatten()
    unsigned = (flat - qmin).astype(np.uint32)  # 0..(2^bits-2)
    if bits == 8:
        return unsigned.astype(np.uint8).tobytes()
    if bits == 4:
        if flat.size % 2 != 0:
            unsigned = np.concatenate([unsigned, np.zeros(1, dtype=np.uint32)])
        high = (unsigned[::2] & 0x0F) << 4
        low = unsigned[1::2] & 0x0F
        return ((high | low).astype(np.uint8)).tobytes()
    # General bit-stream packing (less efficient but correct).
    bit_buf = 0
    bit_count = 0
    out = bytearray()
    mask = (1 << bits) - 1
    for v in unsigned:
        bit_buf |= (int(v) & mask) << bit_count
        bit_count += bits
        while bit_count >= 8:
            out.append(bit_buf & 0xFF)
            bit_buf >>= 8
            bit_count -= 8
    if bit_count > 0:
        out.append(bit_buf & 0xFF)
    return bytes(out)


def _unpack_intN_from_bytes(data: bytes, count: int, bits: int) -> np.ndarray:
    qmin, _ = _intN_range(bits)
    if bits == 8:
        return (np.frombuffer(data, dtype=np.uint8).astype(np.int32) + qmin).astype(np.int8)
    if bits == 4:
        arr = np.frombuffer(data, dtype=np.uint8)
        high = (arr >> 4) & 0x0F
        low = arr & 0x0F
        interleaved = np.empty(arr.size * 2, dtype=np.uint8)
        interleaved[0::2] = high
        interleaved[1::2] = low
        return (interleaved[:count].astype(np.int32) + qmin).astype(np.int8)
    # General bit-stream unpack
    bit_buf = 0
    bit_count = 0
    out = np.empty(count, dtype=np.int32)
    mask = (1 << bits) - 1
    pos = 0
    byte_iter = iter(data)
    for i in range(count):
        while bit_count < bits:
            bit_buf |= next(byte_iter) << bit_count
            bit_count += 8
        out[i] = bit_buf & mask
        bit_buf >>= bits
        bit_count -= bits
    return (out + qmin).astype(np.int8)


def encode_intN_block_fp(weight: torch.Tensor, block_size: int = 128, bits: int = 5) -> bytes:
    qint, scales = _pack_block_fp_intN_arrays(weight, block_size, bits)
    qint_bytes = _pack_intN_to_bytes(qint, bits)
    scales_bytes = scales.tobytes()
    shape = tuple(weight.shape)
    ndim = len(shape)
    qint_count = int(weight.numel())
    header = struct.pack("<BB", ndim, bits)
    for s in shape:
        header += struct.pack("<I", int(s))
    header += struct.pack("<IIII", int(block_size), int(scales.shape[0]), qint_count, len(qint_bytes))
    return header + qint_bytes + scales_bytes


def decode_intN_block_fp(data: bytes) -> torch.Tensor:
    pos = 0
    ndim, bits = struct.unpack_from("<BB", data, pos); pos += 2
    shape = struct.unpack_from(f"<{ndim}I", data, pos); pos += 4 * ndim
    block_size, n_blocks, qint_count, qint_byte_len = struct.unpack_from("<IIII", data, pos); pos += 16
    qint_bytes = data[pos:pos + qint_byte_len]; pos += qint_byte_len
    scales_bytes = data[pos:pos + 4 * n_blocks]
    qint = _unpack_intN_from_bytes(qint_bytes, qint_count, bits).reshape(shape)
    scales = np.frombuffer(scales_bytes, dtype=np.float32)
    qint_t = torch.from_numpy(qint).to(torch.float32)
    out = torch.zeros_like(qint_t)
    for b in range(n_blocks):
        start, end = b * block_size, min((b + 1) * block_size, shape[0])
        out[start:end] = qint_t[start:end] * float(scales[b])
    return out.reshape(shape)


def _smoke_test() -> int:
    sd = torch.load(
        "experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt",
        map_location="cpu", weights_only=False,
    )
    print(f"PR106 state_dict: {len(sd)} tensors")
    print()
    print("--- Pareto sweep: bits × block_size on PR106 Conv2d weights (brotli compressed) ---")
    print(f"{'bits':>4} {'bs':>4} {'brotli':>8} {'max_err':>8} {'rel_err':>8}")
    for bits in [3, 4, 5, 6, 7, 8]:
        for bs in [64, 128]:
            total_brotli = 0
            max_err_global = 0.0
            for name, t in sd.items():
                if t.dim() != 4: continue
                blob = encode_intN_block_fp(t.to(torch.float32), block_size=bs, bits=bits)
                br = brotli.compress(blob, quality=11)
                rt = decode_intN_block_fp(blob)
                err = (rt - t).abs().max().item()
                total_brotli += len(br)
                max_err_global = max(max_err_global, err)
            print(f"{bits:>4d} {bs:>4d} {total_brotli:>8d} {max_err_global:>8.4g} {100*max_err_global/0.9683:>7.2f}%")
    print()
    print("Reference: PR106 brotli encodes whole decoder at 170,278 bytes (Conv2d portion ~138KB).")
    print("OWV2 stub Conv2d portion: ~115KB (saves -22KB total archive).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
