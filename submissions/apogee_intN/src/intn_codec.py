"""Vendored intN block-FP decoder for apogee_intN inflate runtime.

Mirrors `experiments/block_fp_intN_codec_sketch.decode_intN_block_fp`
byte-for-byte. Vendored here so the inflate script has no out-of-tree
imports at runtime (per CLAUDE.md submission discipline + sympathy with the
contest-CUDA `inflate.sh` constraint).

Format per encoder output (also mirrored in
`experiments/block_fp_intN_codec_sketch._pack_block_fp_intN_arrays` +
`_pack_intN_to_bytes`):

    header[2]                 ndim (uint8) + bits (uint8)
    shape[ndim * 4]           uint32 LE per dim
    block_meta[16]            block_size, n_blocks, qint_count, qint_byte_len (uint32 LE)
    qint_bytes[qint_byte_len] bit-packed signed intN values (8/4/general bit-stream)
    scales[n_blocks * 4]      float32 LE per block
"""
from __future__ import annotations

import struct
from typing import Tuple

import numpy as np
import torch


def _intN_range(bits: int) -> tuple[int, int]:
    half = 1 << (bits - 1)
    return -(half - 1), (half - 1)


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
    bit_buf = 0
    bit_count = 0
    out = np.empty(count, dtype=np.int32)
    mask = (1 << bits) - 1
    byte_iter = iter(data)
    for i in range(count):
        while bit_count < bits:
            bit_buf |= next(byte_iter) << bit_count
            bit_count += 8
        out[i] = bit_buf & mask
        bit_buf >>= bits
        bit_count -= bits
    return (out + qmin).astype(np.int8)


def decode_intN_block_fp(data: bytes) -> torch.Tensor:
    """Decode an intN block-FP-encoded weight tensor back to FP32.

    Inverse of encode_intN_block_fp from
    experiments/block_fp_intN_codec_sketch.py.
    """
    pos = 0
    ndim, bits = struct.unpack_from("<BB", data, pos); pos += 2
    if not 4 <= bits <= 8:
        raise ValueError(f"decode_intN_block_fp: bits {bits} out of [4, 8]")
    shape = struct.unpack_from(f"<{ndim}I", data, pos); pos += 4 * ndim
    block_size, n_blocks, qint_count, qint_byte_len = struct.unpack_from("<IIII", data, pos); pos += 16
    qint_bytes = data[pos:pos + qint_byte_len]; pos += qint_byte_len
    scales_bytes = data[pos:pos + 4 * n_blocks]
    qint = _unpack_intN_from_bytes(qint_bytes, qint_count, bits).reshape(shape)
    scales = np.frombuffer(scales_bytes, dtype=np.float32)
    qint_t = torch.from_numpy(qint).to(torch.float32)
    out = torch.zeros_like(qint_t)
    for b in range(n_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, shape[0])
        out[start:end] = qint_t[start:end] * float(scales[b])
    return out.reshape(shape)


def decode_intN_blockfp_from_brotli(payload: bytes) -> torch.Tensor:
    """Brotli-decompress + decode_intN_block_fp. The producer encodes via
    `brotli.compress(encode_intN_block_fp(t, bits))` so the inverse must
    decompress before parsing the intN header."""
    import brotli  # type: ignore[import-not-found]
    blob = brotli.decompress(payload)
    return decode_intN_block_fp(blob)


__all__ = [
    "decode_intN_block_fp",
    "decode_intN_blockfp_from_brotli",
]
