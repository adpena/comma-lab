"""Minimal inflate-side sparse PacketIR decoder (no constriction dep).

Mirrors the encoder in `tac.packet_compiler.sparse_packet_ir` but constrained
to the decode subset that inflate.py needs. Pure-stdlib + numpy.

This file is intentionally inline + duplicated per family submission to keep
each `submissions/pr106_<family>_residual_sidecar/` self-contained (no
cross-submission imports). The shared oracle lives at
`src/tac/packet_compiler/sparse_packet_ir.py`; the per-family inline file
ports the decode-only subset and excludes the encode-time AC dependency on
constriction.

The AC-coded coefficient stream IS available as `decode_arithmetic_coefficients`
when constriction is importable; otherwise inflate.py falls back to a refusal
(family scripts using AC must add a dependency). For now wavelet/c3/cool_chic
inflate paths use only RLE + temporal-subsampled which are constriction-free.
"""
from __future__ import annotations

import struct

import numpy as np

SPARSE_RLE_MAGIC = b"SRL1"
SPARSE_AC_MAGIC = b"SAC1"
SPARSE_TEMPORAL_MAGIC = b"STS1"

_DTYPE_BY_CODE = {0: np.int8, 1: np.int16, 2: np.int32}
_DTYPE_WIRE = {0: "<i1", 1: "<i2", 2: "<i4"}


class SparseDecodeError(ValueError):
    """Raised on sparse wire-format violation."""


def decode_rle_of_zeros_bytes(blob: bytes) -> np.ndarray:
    """Decode a serialised RLE-of-zeros stream to a dense int array."""
    if len(blob) < 13:
        raise SparseDecodeError(f"RLE blob too short: {len(blob)}")
    magic, total_length, n_nonzero, dtype_code = struct.unpack_from(
        "<4sIIB", blob, 0
    )
    if magic != SPARSE_RLE_MAGIC:
        raise SparseDecodeError(f"RLE magic mismatch: {magic!r}")
    if dtype_code not in _DTYPE_BY_CODE:
        raise SparseDecodeError(f"RLE unknown dtype code {dtype_code}")
    dtype = _DTYPE_BY_CODE[dtype_code]
    pos = 13
    idx_size = 4 * n_nonzero
    val_size = np.dtype(dtype).itemsize * n_nonzero
    if pos + idx_size + val_size != len(blob):
        raise SparseDecodeError(
            f"RLE length mismatch: pos={pos} idx={idx_size} val={val_size} total={len(blob)}"
        )
    out = np.zeros(total_length, dtype=dtype)
    if n_nonzero == 0:
        return out
    indices = np.frombuffer(blob, dtype="<u4", count=n_nonzero, offset=pos)
    if indices.size:
        if int(indices.max()) >= total_length:
            raise SparseDecodeError(
                f"RLE index out of range: {int(indices.max())} >= {total_length}"
            )
        if np.any(np.diff(indices.astype(np.int64)) <= 0):
            raise SparseDecodeError("RLE indices must be strictly increasing")
    pos += idx_size
    values = np.frombuffer(
        blob, dtype=_DTYPE_WIRE[dtype_code], count=n_nonzero, offset=pos
    ).astype(dtype, copy=False)
    out[indices] = values
    return out


def decode_temporal_subsampled_bytes(
    blob: bytes,
    *,
    dtype: np.dtype = np.dtype(np.int8),
    frame_shape: tuple[int, ...] | None = None,
) -> list[np.ndarray | None]:
    """Decode a serialised temporal-subsampled stream."""
    if len(blob) < 16:
        raise SparseDecodeError(f"temporal blob too short: {len(blob)}")
    magic, N, K, per_frame_bytes = struct.unpack_from("<4sIII", blob, 0)
    if magic != SPARSE_TEMPORAL_MAGIC:
        raise SparseDecodeError(f"temporal magic mismatch: {magic!r}")
    pos = 16
    bitmap_size = (N + 7) // 8
    if pos + bitmap_size > len(blob):
        raise SparseDecodeError("temporal bitmap truncated")
    bitmap = blob[pos : pos + bitmap_size]
    if N % 8 and bitmap:
        used_bits = N % 8
        padding_mask = (~((1 << used_bits) - 1)) & 0xFF
        if bitmap[-1] & padding_mask:
            raise SparseDecodeError("temporal bitmap has non-zero padding bits")
    pos += bitmap_size
    if pos + K * per_frame_bytes != len(blob):
        raise SparseDecodeError(
            f"temporal length mismatch: pos+packed={pos + K*per_frame_bytes} total={len(blob)}"
        )
    out: list[np.ndarray | None] = [None] * N
    target = np.dtype(dtype)
    items = per_frame_bytes // target.itemsize
    if items * target.itemsize != per_frame_bytes:
        raise SparseDecodeError("per_frame_bytes not divisible by dtype itemsize")
    k_index = 0
    for i in range(N):
        if bitmap[i // 8] & (1 << (i % 8)):
            start = pos + k_index * per_frame_bytes
            end = start + per_frame_bytes
            arr = np.frombuffer(blob, dtype=target, count=items, offset=start).copy()
            if frame_shape is not None:
                arr = arr.reshape(frame_shape)
            out[i] = arr
            k_index += 1
    if k_index != K:
        raise SparseDecodeError(f"popcount mismatch: decoded={k_index} K={K}")
    return out
