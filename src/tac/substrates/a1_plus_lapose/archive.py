"""Wire format for the A1 + LAPose composition substrate.

The composition substrate appends a sidecar blob to A1's existing wire
format. The full inner wire (single ZIP member ``x`` in archive.zip)
is::

    [A1 wire format] (verbatim — uint32 LE decoder section total +
                       decoder blob + 15387 B latent blob + sidecar blob)
    [LAPOSE sidecar] (this module)
        magic       : 4 bytes ASCII ``"LPA1"``
        version     : uint8 (== 1)
        num_selected: uint16 LE
        foveal_h    : uint16 LE
        foveal_w    : uint16 LE
        residual_rank: uint8
        int8_scale  : float32 LE (4 bytes; q_residual / int8_scale = real)
        selected_indices: num_selected * uint16 LE
        residual_blob: Brotli-compressed int8 residual params

        residual params (pre-Brotli) layout per slot::
            [U_left rank * 3 * fov_h int8]
            [V_left rank * fov_w int8]
            [U_right rank * 3 * fov_h int8]
            [V_right rank * fov_w int8]

Per Catalog #146 the inflate runtime that consumes this format is at
``tac.substrates.a1_plus_lapose.inflate`` (≤ 100 LOC + glue per HNeRV
parity discipline lesson 4).

Per Catalog #19 (deterministic zip): the archive builder in the trainer
uses ZipInfo + writestr with fixed timestamps.
"""

from __future__ import annotations

import struct

import brotli
import torch

LAPOSE_SIDECAR_MAGIC = b"LPA1"
LAPOSE_SIDECAR_VERSION = 1
LAPOSE_HEADER_STRUCT = struct.Struct("<4sBHHHBf")  # magic, ver, nsel, fh, fw, rank, scale


def encode_lapose_sidecar(
    selected_indices: tuple[int, ...],
    residuals: torch.Tensor,
    *,
    foveal_h: int,
    foveal_w: int,
    residual_rank: int,
    int8_scale: float,
) -> bytes:
    """Pack the LAPose composition residual sidecar.

    Args:
        selected_indices: pair indices receiving a composition residual.
            Length N. Each in [0, 65535).
        residuals: float tensor (N, 2, rank, 3*fov_h + fov_w).  Holds
            U_left/V_left/U_right/V_right concatenated along the last
            dim per slot.
        foveal_h, foveal_w: foveal patch dims.
        residual_rank: per-frame rank K.
        int8_scale: q = round(real * scale).clamp(-128, 127).

    Returns:
        Sidecar bytes (header + brotli-compressed int8 params).
    """
    if residuals.dim() != 4:
        raise ValueError(
            f"residuals must be 4D (N, 2, rank, 3*fov_h + fov_w); got "
            f"{tuple(residuals.shape)}"
        )
    num = len(selected_indices)
    if residuals.shape[0] != num:
        raise ValueError(
            f"residuals shape mismatch: {residuals.shape[0]} != selected count {num}"
        )
    if residuals.shape[1] != 2 or residuals.shape[2] != residual_rank:
        raise ValueError(
            f"residuals must be (N, 2, rank={residual_rank}, ...); got "
            f"{tuple(residuals.shape)}"
        )
    expected_last = 3 * foveal_h + foveal_w
    if residuals.shape[3] != expected_last:
        raise ValueError(
            f"residuals last dim must be 3*fov_h + fov_w = {expected_last}; "
            f"got {residuals.shape[3]}"
        )

    header = LAPOSE_HEADER_STRUCT.pack(
        LAPOSE_SIDECAR_MAGIC,
        LAPOSE_SIDECAR_VERSION,
        num,
        foveal_h,
        foveal_w,
        residual_rank,
        float(int8_scale),
    )
    index_bytes = b"".join(struct.pack("<H", int(i) & 0xFFFF) for i in selected_indices)

    # Int8 quantize the residual params.
    quantized = (
        (residuals.detach().cpu().float() * float(int8_scale))
        .round()
        .clamp(-128, 127)
        .to(torch.int8)
    )
    raw_bytes = bytes(quantized.reshape(-1).numpy().tobytes())
    compressed = brotli.compress(raw_bytes, quality=11)
    return header + index_bytes + compressed


def decode_lapose_sidecar(blob: bytes) -> tuple[tuple[int, ...], torch.Tensor, dict]:
    """Inverse of :func:`encode_lapose_sidecar`.

    Returns ``(selected_indices, residuals_float, meta)`` where
    ``residuals_float`` is a float tensor (N, 2, rank, 3*fov_h + fov_w).
    """
    if len(blob) < LAPOSE_HEADER_STRUCT.size:
        raise ValueError("lapose sidecar too short for header")
    magic, ver, num, foveal_h, foveal_w, rank, scale = LAPOSE_HEADER_STRUCT.unpack_from(
        blob, 0
    )
    if magic != LAPOSE_SIDECAR_MAGIC:
        raise ValueError(f"bad lapose magic: {magic!r}")
    if ver != LAPOSE_SIDECAR_VERSION:
        raise ValueError(f"unsupported lapose version: {ver}")
    offset = LAPOSE_HEADER_STRUCT.size
    indices: list[int] = []
    for _ in range(num):
        (idx,) = struct.unpack_from("<H", blob, offset)
        indices.append(int(idx))
        offset += 2
    compressed = blob[offset:]
    raw = brotli.decompress(compressed)
    last = 3 * int(foveal_h) + int(foveal_w)
    expected_count = int(num) * 2 * int(rank) * last
    if len(raw) != expected_count:
        raise ValueError(
            f"lapose residual byte count mismatch: got {len(raw)} expected "
            f"{expected_count}"
        )
    import numpy as np

    arr = np.frombuffer(raw, dtype=np.int8).reshape(num, 2, rank, last)
    residuals_float = torch.from_numpy(arr.copy()).float() / float(scale)
    meta = {
        "foveal_h": int(foveal_h),
        "foveal_w": int(foveal_w),
        "residual_rank": int(rank),
        "int8_scale": float(scale),
        "num_selected": int(num),
    }
    return tuple(indices), residuals_float, meta


def split_composition_archive(archive_bytes: bytes) -> tuple[bytes, bytes]:
    """Split the composition archive into (a1_bytes, lapose_sidecar_bytes).

    The split is done by scanning for the LAPose magic from the END of the
    blob (the LAPose sidecar is always the suffix). If no LAPose magic is
    found, the entire blob is treated as A1 and an empty LAPose suffix is
    returned (graceful for back-compat with pure A1 archives).
    """
    # Search backwards for the magic; the header is 16 bytes so the magic
    # appears at position len - tail_len for some tail_len. We scan from
    # the rightmost half of the blob to keep this O(blob/2).
    idx = archive_bytes.rfind(LAPOSE_SIDECAR_MAGIC)
    if idx <= 0:
        return archive_bytes, b""
    # Validate header sanity at idx.
    if idx + LAPOSE_HEADER_STRUCT.size > len(archive_bytes):
        return archive_bytes, b""
    try:
        magic, ver, _num, _fh, _fw, _rank, _scale = LAPOSE_HEADER_STRUCT.unpack_from(
            archive_bytes, idx
        )
    except struct.error:
        return archive_bytes, b""
    if magic != LAPOSE_SIDECAR_MAGIC or ver != LAPOSE_SIDECAR_VERSION:
        return archive_bytes, b""
    try:
        decode_lapose_sidecar(archive_bytes[idx:])
    except Exception:
        return archive_bytes, b""
    return archive_bytes[:idx], archive_bytes[idx:]


def pack_composition_archive(
    a1_bytes: bytes,
    *,
    selected_indices: tuple[int, ...],
    residuals: torch.Tensor,
    foveal_h: int,
    foveal_w: int,
    residual_rank: int,
    int8_scale: float,
) -> bytes:
    """Concatenate A1 base bytes with the LAPose sidecar.

    The A1 bytes are passed through verbatim — they are an immutable
    reference per the "Apples-to-apples evidence discipline" rule.
    """
    if not a1_bytes:
        raise ValueError("a1_bytes must be non-empty")
    sidecar = encode_lapose_sidecar(
        selected_indices,
        residuals,
        foveal_h=foveal_h,
        foveal_w=foveal_w,
        residual_rank=residual_rank,
        int8_scale=int8_scale,
    )
    return a1_bytes + sidecar


__all__ = [
    "LAPOSE_HEADER_STRUCT",
    "LAPOSE_SIDECAR_MAGIC",
    "LAPOSE_SIDECAR_VERSION",
    "decode_lapose_sidecar",
    "encode_lapose_sidecar",
    "pack_composition_archive",
    "split_composition_archive",
]
