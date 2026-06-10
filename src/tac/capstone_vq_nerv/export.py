# SPDX-License-Identifier: MIT
"""Byte-closed archive export for the original VQ-NeRV capstone (Task #78).

The archive grammar is a single monolithic ``0.bin`` with length-prefixed
sections (the PR95-family L20 monolithic-4-section discipline, extended for the
VQ-index + stored-pose sections):

  ``(u32 dec_len, dec_blob brotli)``    decoder + FiLM weights (fp16, brotli q11)
  ``(u32 cb_len,  cb_blob  brotli)``    VQ codebook (fp16, brotli q11) — FREE in
                                        the native decode (paid once), NOT per-pair
  ``(u32 idx_len, idx_blob raw)``       bit-packed per-pair VQ indices
                                        (ceil(log2(K)) bits/pair)
  ``(u32 pose_len, pose_blob brotli)``  stored 6-dim GT pose scalars (fp16, brotli)

The rate lever (#67): the per-pair *index* costs ``ceil(log2(K))`` bits/pair,
NOT a continuous 28-d fp16 latent (448 bits/pair). At K=256 that is 8 bits/pair
= 600 bytes for 600 pairs vs ~33.6 KB for fp16 latents — a 56x cut on the
per-pair carrier, with the codebook paid ONCE.

The stored pose (#81 / Quantizr): 600 pairs * 6 fp16 = 7.2 KB raw, brotli'd to
~kilobytes. This is the explicit-pose carrier the FiLM injects; it buys the pose
term WITHOUT a per-pixel pose reconstruction.

This module is pure-numpy + brotli (NO torch, NO MLX) so the inflate path is
``numpy-portable`` per the CLAUDE.md "MLX-first numpy-portable" contract; the
byte accounting is exact and machine-checkable.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - import guard
    import brotli
except Exception:  # pragma: no cover
    brotli = None  # type: ignore[assignment]


def _require_brotli() -> None:
    if brotli is None:  # pragma: no cover
        raise RuntimeError("tac.capstone_vq_nerv.export requires brotli.")


def bits_per_index(codebook_size: int) -> int:
    """Bits needed to store one VQ index in ``[0, codebook_size)``."""
    if codebook_size <= 1:
        return 1
    return int(np.ceil(np.log2(codebook_size)))


def bit_pack_vq_indices(indices: np.ndarray, codebook_size: int) -> bytes:
    """Bit-pack per-pair VQ indices to ``ceil(log2(K))`` bits each (MSB-first).

    The index stream is the per-pair carrier. Packing to the exact bit width
    (not a byte each) is the #67 rate discipline: at K=256 it is one byte/pair;
    at K=16 it is a nibble/pair (300 bytes for 600 pairs).
    """
    idx = np.asarray(indices, dtype=np.int64).ravel()
    if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= codebook_size):
        raise ValueError(
            f"indices out of range [0,{codebook_size}): "
            f"min={int(idx.min())} max={int(idx.max())}"
        )
    nbits = bits_per_index(codebook_size)
    bit_buf = 0
    bit_count = 0
    out = bytearray()
    for v in idx:
        bit_buf = (bit_buf << nbits) | int(v)
        bit_count += nbits
        while bit_count >= 8:
            bit_count -= 8
            out.append((bit_buf >> bit_count) & 0xFF)
    if bit_count > 0:
        out.append((bit_buf << (8 - bit_count)) & 0xFF)
    return bytes(out)


def bit_unpack_vq_indices(
    packed: bytes, n_indices: int, codebook_size: int
) -> np.ndarray:
    """Inverse of :func:`bit_pack_vq_indices` — recover ``n_indices`` indices."""
    nbits = bits_per_index(codebook_size)
    out = np.empty(n_indices, dtype=np.int32)
    bit_buf = 0
    bit_count = 0
    byte_iter = iter(packed)
    for i in range(n_indices):
        while bit_count < nbits:
            bit_buf = (bit_buf << 8) | next(byte_iter)
            bit_count += 8
        bit_count -= nbits
        out[i] = (bit_buf >> bit_count) & ((1 << nbits) - 1)
    return out


def _section(blob: bytes) -> bytes:
    """Length-prefix a blob: ``u32 len`` + ``blob``."""
    return struct.pack("<I", len(blob)) + blob


@dataclass
class CapstoneArchiveAccount:
    """Exact per-section byte accounting for the capstone archive."""

    decoder_bytes: int
    codebook_bytes: int
    index_bytes: int
    pose_bytes: int
    total_bytes: int
    num_pairs: int
    codebook_size: int
    bits_per_index: int

    @property
    def rate(self) -> float:
        """Contest rate term ``25 * archive_bytes / 37_545_489`` (canonical)."""
        return 25.0 * self.total_bytes / 37_545_489.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "decoder_bytes": self.decoder_bytes,
            "codebook_bytes": self.codebook_bytes,
            "index_bytes": self.index_bytes,
            "pose_bytes": self.pose_bytes,
            "total_bytes": self.total_bytes,
            "num_pairs": self.num_pairs,
            "codebook_size": self.codebook_size,
            "bits_per_index": self.bits_per_index,
            "rate": self.rate,
        }


def _fp16_brotli(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize a name->array dict as fp16 + brotli (q11). Deterministic order."""
    _require_brotli()
    parts = bytearray()
    for name in sorted(arrays):
        arr = np.asarray(arrays[name], dtype=np.float16)
        name_b = name.encode("utf-8")
        parts += struct.pack("<H", len(name_b)) + name_b
        parts += struct.pack("<B", arr.ndim)
        for d in arr.shape:
            parts += struct.pack("<I", int(d))
        parts += arr.tobytes()
    return brotli.compress(bytes(parts), quality=11)


def _zigzag_int8(arr_i8: np.ndarray) -> np.ndarray:
    """Map signed int8 [-128,127] to unsigned uint8 via zigzag (PR95 L21).

    Zigzag interleaves small-magnitude positive/negative values to small
    unsigned codes, which brotli compresses far better than two's-complement
    (the negative tail otherwise spans 0x80..0xFF).
    """
    x = arr_i8.astype(np.int16)
    return ((x << 1) ^ (x >> 15)).astype(np.uint8)


def _unzigzag_uint8(arr_u8: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_zigzag_int8` -> signed int8."""
    u = arr_u8.astype(np.int16)
    return ((u >> 1) ^ -(u & 1)).astype(np.int8)


def _int8_brotli(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize a name->array dict as per-tensor symmetric int8 + fp16 scale.

    Each tensor is quantized ``q = round(w / scale)`` with
    ``scale = max(|w|) / 127`` (PR95 L29 fp16-per-tensor-scale), zigzag-mapped
    (L21), and the whole stream is brotli q11 (L32). This is the ~1-byte/param
    entropy-coded path that the sub-0.15 byte budget requires (vs 2 bytes/param
    fp16). The codec is exact-invertible (verified by the round-trip test).
    """
    _require_brotli()
    parts = bytearray()
    for name in sorted(arrays):
        arr = np.asarray(arrays[name], dtype=np.float32)
        amax = float(np.max(np.abs(arr))) if arr.size else 0.0
        scale = (amax / 127.0) if amax > 0 else 1.0
        q = np.round(arr / scale).astype(np.int32)
        q = np.clip(q, -127, 127).astype(np.int8)
        zz = _zigzag_int8(q)
        name_b = name.encode("utf-8")
        parts += struct.pack("<H", len(name_b)) + name_b
        parts += struct.pack("<e", np.float16(scale))  # fp16 per-tensor scale
        parts += struct.pack("<B", arr.ndim)
        for d in arr.shape:
            parts += struct.pack("<I", int(d))
        parts += zz.tobytes()
    return brotli.compress(bytes(parts), quality=11)


def _decode_int8_brotli(blob: bytes) -> dict[str, np.ndarray]:
    """Inverse of :func:`_int8_brotli` -> name->dequantized fp32 array (parity)."""
    _require_brotli()
    raw = brotli.decompress(blob)
    out: dict[str, np.ndarray] = {}
    off = 0
    while off < len(raw):
        (nlen,) = struct.unpack_from("<H", raw, off)
        off += 2
        name = raw[off : off + nlen].decode("utf-8")
        off += nlen
        (scale,) = struct.unpack_from("<e", raw, off)
        off += 2
        (ndim,) = struct.unpack_from("<B", raw, off)
        off += 1
        shape = []
        for _ in range(ndim):
            (d,) = struct.unpack_from("<I", raw, off)
            off += 4
            shape.append(int(d))
        n = int(np.prod(shape)) if shape else 1
        zz = np.frombuffer(raw[off : off + n], dtype=np.uint8)
        off += n
        q = _unzigzag_uint8(zz).astype(np.float32)
        out[name] = (q * float(scale)).reshape(shape)
    return out


def build_capstone_archive_bytes(
    *,
    decoder_weights: dict[str, np.ndarray],
    codebook: np.ndarray,
    vq_indices: np.ndarray,
    pose_scalars: np.ndarray,
    codebook_size: int,
    decoder_dtype: str = "fp16",
) -> tuple[bytes, CapstoneArchiveAccount]:
    """Byte-close the capstone archive and return ``(bytes, account)``.

    Args:
        decoder_weights: name->array of decoder + FiLM params (the "free" basis).
        codebook: ``(K, latent_dim)`` VQ codebook (free in decode, paid once).
        vq_indices: ``(num_pairs,)`` per-pair codebook index (the bit-packed carrier).
        pose_scalars: ``(num_pairs, 6)`` stored GT pose (the FiLM carrier).
        codebook_size: K (determines the index bit width).
        decoder_dtype: ``"fp16"`` (2 B/param, lossless baseline) or ``"int8"``
            (per-tensor symmetric int8 + fp16 scale + zigzag + brotli, ~1 B/param
            — the sub-0.15 byte-budget enabler; the PR95 L21/L29/L32 stack).
    """
    _require_brotli()
    if decoder_dtype not in {"fp16", "int8"}:
        raise ValueError(f"decoder_dtype must be 'fp16' or 'int8'; got {decoder_dtype!r}")
    num_pairs = int(np.asarray(vq_indices).shape[0])

    serialize = _int8_brotli if decoder_dtype == "int8" else _fp16_brotli
    dec_blob = serialize(decoder_weights)
    cb_blob = serialize({"codebook": np.asarray(codebook)})
    idx_blob = bit_pack_vq_indices(vq_indices, codebook_size)
    pose_blob = brotli.compress(
        np.asarray(pose_scalars, dtype=np.float16).tobytes(), quality=11
    )

    archive = (
        _section(dec_blob)
        + _section(cb_blob)
        + _section(idx_blob)
        + _section(pose_blob)
    )
    account = CapstoneArchiveAccount(
        decoder_bytes=len(dec_blob),
        codebook_bytes=len(cb_blob),
        index_bytes=len(idx_blob),
        pose_bytes=len(pose_blob),
        total_bytes=len(archive),
        num_pairs=num_pairs,
        codebook_size=int(codebook_size),
        bits_per_index=bits_per_index(codebook_size),
    )
    return archive, account


def parse_capstone_archive_bytes(archive: bytes) -> dict[str, bytes]:
    """Parse the 4 length-prefixed sections back to raw blobs (parse-back proof)."""
    off = 0
    blobs: list[bytes] = []
    for _ in range(4):
        (length,) = struct.unpack_from("<I", archive, off)
        off += 4
        blobs.append(archive[off : off + length])
        off += length
    return {
        "decoder": blobs[0],
        "codebook": blobs[1],
        "index": blobs[2],
        "pose": blobs[3],
    }


__all__ = [
    "CapstoneArchiveAccount",
    "bit_pack_vq_indices",
    "bit_unpack_vq_indices",
    "bits_per_index",
    "build_capstone_archive_bytes",
    "parse_capstone_archive_bytes",
]


# Re-exported for the round-trip parity test (NO-FAKE: the int8 codec is exact-
# invertible up to the per-tensor quant step).
_INT8_CODEC = (_int8_brotli, _decode_int8_brotli)
