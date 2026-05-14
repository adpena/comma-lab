# SPDX-License-Identifier: MIT
"""Wire format for the A1 + wavelet residual composition substrate.

The composition substrate appends a sidecar blob (magic-byte trailer D4.B
pattern) to A1's existing wire format.  The full inner wire (single ZIP
member ``x`` in archive.zip) is::

    [A1 wire format] (verbatim — uint32 LE decoder section total +
                       decoder blob + 15387 B latent blob + sidecar blob)
    [WAVELET sidecar] (this module)
        magic        : 4 bytes ASCII ``"WAV1"``
        version      : uint8 (== 1)
        num_selected : uint16 LE
        foveal_h     : uint16 LE  (HALF-camera-resolution patch height)
        foveal_w     : uint16 LE  (HALF-camera-resolution patch width)
        coeff_rank   : uint8
        int8_scale   : float32 LE (4 bytes; q_residual / int8_scale = real)
        reserved     : 2 bytes (=0)
        selected_indices : num_selected * uint16 LE
        coeffs_blob  : brotli-compressed int8 detail-band coefficients

    coeffs (pre-brotli) layout per selected slot::
        For band in (LH, HL, HH):
            For frame in (0, 1):
                For channel in (R, G, B):
                    U_band_frame_channel : rank * foveal_h int8
                    V_band_frame_channel : rank * foveal_w int8

Per Catalog #146 the inflate runtime that consumes this format is at
``tac.substrates.a1_plus_wavelet_residual.inflate`` (≤ 200 LOC per HNeRV
parity discipline L4 substrate-engineering exemption).

Per Catalog #19 (deterministic zip): the archive builder uses ZipInfo +
writestr with fixed timestamps.

Magic distinct from ``LPA1`` (A1+LAPose) ensures cross-substrate split
operations never confuse the trailer (each substrate-scoped inflate
refuses unknown magic loudly).
"""

from __future__ import annotations

import struct

import brotli
import torch

WAVELET_SIDECAR_MAGIC = b"WAV1"
WAVELET_SIDECAR_VERSION = 1
# magic(4s) + version(B) + nsel(H) + fh(H) + fw(H) + rank(B) + scale(f) + reserved(2s)
WAVELET_HEADER_STRUCT = struct.Struct("<4sBHHHBf2s")

NUM_DETAIL_BANDS = 3  # LH, HL, HH
NUM_FRAMES_PER_PAIR = 2
NUM_RGB_CHANNELS = 3


def _pre_brotli_byte_count(
    num_selected: int, coeff_rank: int, foveal_h: int, foveal_w: int
) -> int:
    """Closed-form pre-brotli coefficient byte count.

    Used by both ``encode`` (sanity check after concat) and ``decode``
    (post-brotli decompress sanity check).
    """
    per_band_per_frame_per_chan = coeff_rank * (foveal_h + foveal_w)
    bands_per_frame = NUM_DETAIL_BANDS * NUM_RGB_CHANNELS * per_band_per_frame_per_chan
    return int(num_selected * NUM_FRAMES_PER_PAIR * bands_per_frame)


def encode_wavelet_sidecar(
    selected_indices: tuple[int, ...],
    coeffs: torch.Tensor,
    *,
    foveal_h: int,
    foveal_w: int,
    coeff_rank: int,
    int8_scale: float,
) -> bytes:
    """Pack the wavelet composition residual sidecar.

    Args:
        selected_indices: pair indices receiving a wavelet residual; length N.
            Each in [0, 65535).
        coeffs: float tensor (N, 3 bands, 2 frames, 3 RGB, rank, foveal_h +
            foveal_w).  Holds per-band U/V factors concatenated along the
            last axis (rows 0..foveal_h-1 = U, foveal_h..foveal_h+foveal_w-1
            = V).
        foveal_h, foveal_w: foveal patch dims at HALF-camera resolution.
        coeff_rank: per-band low-rank K.
        int8_scale: q = clamp(round(real * scale), -128, 127).

    Returns:
        Sidecar bytes (header + indices + brotli-compressed int8 coeffs).
    """
    if coeffs.dim() != 6:
        raise ValueError(
            f"coeffs must be 6D (N, bands=3, frames=2, RGB=3, rank, fh+fw); got "
            f"{tuple(coeffs.shape)}"
        )
    num = len(selected_indices)
    if coeffs.shape[0] != num:
        raise ValueError(
            f"coeffs shape mismatch: {coeffs.shape[0]} != selected count {num}"
        )
    if (
        coeffs.shape[1] != NUM_DETAIL_BANDS
        or coeffs.shape[2] != NUM_FRAMES_PER_PAIR
        or coeffs.shape[3] != NUM_RGB_CHANNELS
        or coeffs.shape[4] != coeff_rank
    ):
        raise ValueError(
            f"coeffs axes mismatch: shape={tuple(coeffs.shape)} expected "
            f"(N={num}, bands={NUM_DETAIL_BANDS}, frames={NUM_FRAMES_PER_PAIR}, "
            f"RGB={NUM_RGB_CHANNELS}, rank={coeff_rank}, ...)"
        )
    expected_last = foveal_h + foveal_w
    if coeffs.shape[5] != expected_last:
        raise ValueError(
            f"coeffs last dim must be foveal_h + foveal_w = {expected_last}; "
            f"got {coeffs.shape[5]}"
        )

    header = WAVELET_HEADER_STRUCT.pack(
        WAVELET_SIDECAR_MAGIC,
        WAVELET_SIDECAR_VERSION,
        num,
        foveal_h,
        foveal_w,
        coeff_rank,
        float(int8_scale),
        b"\x00\x00",
    )
    index_bytes = b"".join(struct.pack("<H", int(i) & 0xFFFF) for i in selected_indices)

    quantized = (
        (coeffs.detach().cpu().float() * float(int8_scale))
        .round()
        .clamp(-128, 127)
        .to(torch.int8)
    )
    raw_bytes = bytes(quantized.reshape(-1).numpy().tobytes())
    expected_raw = _pre_brotli_byte_count(num, coeff_rank, foveal_h, foveal_w)
    if len(raw_bytes) != expected_raw:
        raise ValueError(
            f"int8 coeff byte count mismatch: got {len(raw_bytes)} expected "
            f"{expected_raw}"
        )
    compressed = brotli.compress(raw_bytes, quality=11)
    return header + index_bytes + compressed


def decode_wavelet_sidecar(blob: bytes) -> tuple[tuple[int, ...], torch.Tensor, dict]:
    """Inverse of :func:`encode_wavelet_sidecar`.

    Returns ``(selected_indices, coeffs_float, meta)`` where ``coeffs_float``
    is a float tensor (N, 3 bands, 2 frames, 3 RGB, rank, foveal_h+foveal_w).
    """
    if len(blob) < WAVELET_HEADER_STRUCT.size:
        raise ValueError("wavelet sidecar too short for header")
    magic, ver, num, foveal_h, foveal_w, rank, scale, _reserved = (
        WAVELET_HEADER_STRUCT.unpack_from(blob, 0)
    )
    if magic != WAVELET_SIDECAR_MAGIC:
        raise ValueError(f"bad wavelet magic: {magic!r}")
    if ver != WAVELET_SIDECAR_VERSION:
        raise ValueError(f"unsupported wavelet version: {ver}")
    offset = WAVELET_HEADER_STRUCT.size
    indices: list[int] = []
    for _ in range(num):
        (idx,) = struct.unpack_from("<H", blob, offset)
        indices.append(int(idx))
        offset += 2
    compressed = blob[offset:]
    raw = brotli.decompress(compressed)
    expected_count = _pre_brotli_byte_count(
        int(num), int(rank), int(foveal_h), int(foveal_w)
    )
    if len(raw) != expected_count:
        raise ValueError(
            f"wavelet coeff byte count mismatch: got {len(raw)} expected "
            f"{expected_count}"
        )
    import numpy as np

    arr = np.frombuffer(raw, dtype=np.int8).reshape(
        int(num),
        NUM_DETAIL_BANDS,
        NUM_FRAMES_PER_PAIR,
        NUM_RGB_CHANNELS,
        int(rank),
        int(foveal_h) + int(foveal_w),
    )
    coeffs_float = torch.from_numpy(arr.copy()).float() / float(scale)
    meta = {
        "foveal_h": int(foveal_h),
        "foveal_w": int(foveal_w),
        "coeff_rank": int(rank),
        "int8_scale": float(scale),
        "num_selected": int(num),
    }
    return tuple(indices), coeffs_float, meta


def split_composition_archive(archive_bytes: bytes) -> tuple[bytes, bytes]:
    """Split the composition archive into (a1_bytes, wavelet_sidecar_bytes).

    The split is done by scanning for the WAV1 magic from the END of the
    blob (the wavelet sidecar is always the suffix).  If no WAV1 magic is
    found, the entire blob is treated as A1 and an empty wavelet suffix is
    returned (graceful for back-compat with pure A1 archives).
    """
    idx = archive_bytes.rfind(WAVELET_SIDECAR_MAGIC)
    if idx <= 0:
        return archive_bytes, b""
    if idx + WAVELET_HEADER_STRUCT.size > len(archive_bytes):
        return archive_bytes, b""
    try:
        magic, ver, _num, _fh, _fw, _rank, _scale, _reserved = (
            WAVELET_HEADER_STRUCT.unpack_from(archive_bytes, idx)
        )
    except struct.error:
        return archive_bytes, b""
    if magic != WAVELET_SIDECAR_MAGIC or ver != WAVELET_SIDECAR_VERSION:
        return archive_bytes, b""
    try:
        decode_wavelet_sidecar(archive_bytes[idx:])
    except Exception:
        return archive_bytes, b""
    return archive_bytes[:idx], archive_bytes[idx:]


def pack_composition_archive(
    a1_bytes: bytes,
    *,
    selected_indices: tuple[int, ...],
    coeffs: torch.Tensor,
    foveal_h: int,
    foveal_w: int,
    coeff_rank: int,
    int8_scale: float,
) -> bytes:
    """Concatenate A1 base bytes with the wavelet sidecar (magic-byte trailer).

    The A1 bytes are passed through verbatim — they are an immutable reference
    per the "Apples-to-apples evidence discipline" rule.
    """
    if not a1_bytes:
        raise ValueError("a1_bytes must be non-empty")
    sidecar = encode_wavelet_sidecar(
        selected_indices,
        coeffs,
        foveal_h=foveal_h,
        foveal_w=foveal_w,
        coeff_rank=coeff_rank,
        int8_scale=int8_scale,
    )
    return a1_bytes + sidecar


__all__ = [
    "NUM_DETAIL_BANDS",
    "NUM_FRAMES_PER_PAIR",
    "NUM_RGB_CHANNELS",
    "WAVELET_HEADER_STRUCT",
    "WAVELET_SIDECAR_MAGIC",
    "WAVELET_SIDECAR_VERSION",
    "decode_wavelet_sidecar",
    "encode_wavelet_sidecar",
    "pack_composition_archive",
    "split_composition_archive",
]
