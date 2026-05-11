"""PR93 ``flatpup`` pose-codec primitives — reusable byte-grammar pieces.

This module extracts the REUSABLE pose-axis byte-grammar primitives from the
PR93 public submission (``submissions/flatpup/inflate.py``) into typed,
golden-vector-backed transducers that compose orthogonally with the existing
PR101 sidecar grammar and PR103 arithmetic-coding primitives.

Two primitives land here:

1. **Delta-varint pose codec** (:func:`encode_delta_varint_pose` /
   :func:`decode_delta_varint_pose`)

   PR93's ``QZPDV1`` (``b"QZPDV1\\0\\0"``) magic-prefixed payload encodes a
   pose tensor ``(n, d)`` as

   * fp32 ``lo`` (length ``d``) — per-dimension offset.
   * fp32 ``scale`` (length ``d``) — per-dimension multiplier.
   * uint8 OR uint16 ``first[d]`` — absolute quantised first row.
   * signed-varint stream over the remaining ``(n - 1) * d`` deltas in
     row-major order; each integer is *zigzag*-encoded then LEB128-packed.

   This is the POSE-AXIS primitive identified as the highest-EV-per-byte
   port in ``feedback_public_pr_nonhnerv_mechanism_backlog_landed_20260511``
   at the PR106 r2 operating point (pose marginal value is 2.79× SegNet's).

2. **QZMB1 / QZPDV1 magic grammar** (:func:`pack_qzmb1_block` /
   :func:`unpack_qzmb1_block` plus the magic constants)

   The block grammar PR93 uses to frame the model + pose payloads. Each block
   is ``MAGIC || body``; the magic alone identifies the body grammar so a
   future archive can carry multiple QZ* blocks without per-block length
   prefixes (the wrapping container — typically a brotli + ZIP member —
   provides the length).

Source: ``experiments/results/public_pr_archive_kaggle_mirror/public_pr93_intake_20260505_auto/source/submissions/flatpup/inflate.py``
(SHA pinned via ``check_public_pr_intake_clones_pristine``-protected intake).

CLAUDE.md compliance
====================

* No scorer load — pure numpy + struct + stdlib.
* No MPS / torch import.
* No ``/tmp`` paths.
* Frozen dataclasses; ``encode → decode`` is bit-exact on the
  ``pr93_delta_varint_pose_v1`` and ``pr93_qzmb1_v1`` golden vectors.
* OSS-friendly: public surface is the 7 names re-exported from
  ``tac.packet_compiler``; everything else is ``_``-prefixed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# ── Magic constants ─────────────────────────────────────────────────────────
#: 8-byte magic identifying a QZPDV1 (delta-varint pose) payload.
MAGIC_POSE_DV: bytes = b"QZPDV1\x00\x00"
#: 8-byte magic identifying a QZMB1 (compact model block) payload.
MAGIC_MODEL_COMPACT: bytes = b"QZMB1\x00\x00\x00"


# ── Public dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class DeltaVarintPoseStream:
    """Encoded delta-varint pose payload.

    Attributes
    ----------
    payload:
        Bytes that begin with :data:`MAGIC_POSE_DV` and end with the last
        varint byte; ready to brotli-compress + place in an archive member.
    n_rows, n_dims:
        Pose tensor shape (``n_rows`` is the temporal axis; PR93 uses 600).
    bits:
        Width of the absolute first-row code (8 or 16). PR93 derives this
        from the empirical max of the quantised first row.
    lo, scale:
        Per-dim fp32 affine recovery parameters; the decoded values satisfy
        ``recovered = lo + q * scale`` where ``q`` is the int32 reconstruction.
    """

    payload: bytes
    n_rows: int
    n_dims: int
    bits: int
    lo: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class QZMB1Block:
    """Output of :func:`pack_qzmb1_block`.

    A QZMB1 block is ``MAGIC_MODEL_COMPACT || H || body`` where ``H`` is the
    fixed PR93 header ``<HH`` (``block_size, arch_len``) followed by an
    ``arch_len``-byte JSON arch-config blob. The rest of the body is opaque
    to this primitive (the consumer parses tensor records).

    Attributes
    ----------
    payload:
        Concatenation of magic + header + arch-config + body bytes.
    block_size:
        FP4 codebook block size (PR93 uses 32).
    arch_config_json:
        JSON-encoded architecture configuration (the bytes used to compute
        ``arch_len``).
    body:
        Opaque tensor-record body bytes (caller supplies; the primitive
        only frames it). Empty bytes are accepted.
    """

    payload: bytes
    block_size: int
    arch_config_json: bytes
    body: bytes


# ── Internal varint helpers ─────────────────────────────────────────────────


def _zigzag_encode_i32(value: int) -> int:
    """Zigzag-encode a signed 32-bit integer to unsigned (LEB128-friendly)."""
    if not (-(1 << 31) <= value < (1 << 31)):
        raise ValueError(f"value {value} out of int32 range")
    return ((value << 1) ^ (value >> 31)) & 0xFFFFFFFF


def _zigzag_decode_u32(value: int) -> int:
    """Inverse of :func:`_zigzag_encode_i32`."""
    return (value >> 1) ^ -(value & 1)


def _encode_unsigned_varint(value: int) -> bytes:
    """LEB128 unsigned varint encode of a non-negative integer."""
    if value < 0:
        raise ValueError(f"unsigned varint value must be >= 0; got {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _decode_unsigned_varint(buf: bytes, offset: int) -> tuple[int, int]:
    """Decode one unsigned varint. Returns ``(value, new_offset)``."""
    value = 0
    shift = 0
    while True:
        if offset >= len(buf):
            raise ValueError("truncated varint")
        byte = buf[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift >= 64:
            raise ValueError("varint too large (>64 bits)")


# ── Public delta-varint pose codec ──────────────────────────────────────────


def encode_delta_varint_pose(
    poses: np.ndarray,
    *,
    lo: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    bits: int | None = None,
) -> DeltaVarintPoseStream:
    """Encode a pose tensor under the PR93 ``QZPDV1`` delta-varint grammar.

    Parameters
    ----------
    poses:
        Float pose tensor of shape ``(n_rows, n_dims)``. Values are quantised
        per-column as ``round((poses - lo) / scale)``.
    lo, scale:
        Optional per-column fp32 offset/scale. If not supplied they are
        derived from the column min / column range so the quantisation fits
        a non-negative integer grid.
    bits:
        Width of the absolute first-row code. Must be 8 or 16. If ``None``,
        the encoder picks 8 when the first row's max value fits in
        ``[0, 255]``; otherwise 16. PR93 uses the same rule.

    Returns
    -------
    DeltaVarintPoseStream
        Container with the magic-prefixed payload bytes plus the recovery
        parameters.
    """
    arr = np.asarray(poses, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"poses must be 2D (n_rows, n_dims); got shape {arr.shape}"
        )
    n_rows, n_dims = arr.shape
    if n_rows == 0 or n_dims == 0:
        raise ValueError(
            f"poses must be non-empty; got shape {arr.shape}"
        )

    if lo is None:
        lo = arr.min(axis=0).astype(np.float32)
    else:
        lo = np.asarray(lo, dtype=np.float32)
    if lo.shape != (n_dims,):
        raise ValueError(f"lo shape {lo.shape} != ({n_dims},)")

    if scale is None:
        rng = (arr.max(axis=0) - lo).astype(np.float32)
        rng = np.where(rng <= 0.0, np.float32(1.0), rng)
        scale = (rng / 255.0).astype(np.float32)
    else:
        scale = np.asarray(scale, dtype=np.float32)
    if scale.shape != (n_dims,):
        raise ValueError(f"scale shape {scale.shape} != ({n_dims},)")
    if not np.all(scale > 0):
        raise ValueError("scale must be strictly positive everywhere")

    q = np.round((arr - lo) / scale).astype(np.int64)
    if q.min() < 0:
        raise ValueError(
            "quantised value < 0; provide lo such that all poses >= lo"
        )
    q_max = int(q.max())

    if bits is None:
        bits = 8 if q_max <= 0xFF else 16
    if bits not in (8, 16):
        raise ValueError(f"bits must be 8 or 16; got {bits}")
    if bits == 8 and q_max > 0xFF:
        raise ValueError(
            f"quantised max {q_max} does not fit in 8 bits; pass bits=16"
        )
    if bits == 16 and q_max > 0xFFFF:
        raise ValueError(
            f"quantised max {q_max} does not fit in 16 bits"
        )

    first = q[0].astype(np.uint8 if bits == 8 else np.uint16)
    if n_rows > 1:
        deltas = (q[1:] - q[:-1]).astype(np.int64).reshape(-1)
    else:
        deltas = np.zeros(0, dtype=np.int64)

    body = bytearray()
    body += struct.pack("<III", n_rows, n_dims, bits)
    body += lo.astype(np.float32).tobytes()
    body += scale.astype(np.float32).tobytes()
    body += first.tobytes()
    for d in deltas:
        body += _encode_unsigned_varint(_zigzag_encode_i32(int(d)))

    payload = bytes(MAGIC_POSE_DV + bytes(body))
    return DeltaVarintPoseStream(
        payload=payload,
        n_rows=int(n_rows),
        n_dims=int(n_dims),
        bits=int(bits),
        lo=lo,
        scale=scale,
    )


def decode_delta_varint_pose(payload: bytes) -> np.ndarray:
    """Decode a ``QZPDV1`` magic-prefixed payload back to float poses.

    Returns
    -------
    np.ndarray
        Recovered pose tensor of shape ``(n_rows, n_dims)`` and dtype
        ``float32``.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"payload must be bytes-like; got {type(payload)!r}")
    payload = bytes(payload)
    if payload[: len(MAGIC_POSE_DV)] != MAGIC_POSE_DV:
        raise ValueError(
            f"missing QZPDV1 magic; got prefix {payload[:8]!r}"
        )
    off = len(MAGIC_POSE_DV)
    n_rows, n_dims, bits = struct.unpack_from("<III", payload, off)
    off += 12
    if bits not in (8, 16):
        raise ValueError(f"unsupported bits {bits}; expected 8 or 16")
    if n_dims == 0:
        raise ValueError("n_dims must be > 0")
    lo = np.frombuffer(payload, dtype=np.float32, count=n_dims, offset=off).copy()
    off += n_dims * 4
    scale = np.frombuffer(payload, dtype=np.float32, count=n_dims, offset=off).copy()
    off += n_dims * 4
    dtype = np.uint8 if bits == 8 else np.uint16
    item_size = 1 if bits == 8 else 2
    first = np.frombuffer(payload, dtype=dtype, count=n_dims, offset=off).astype(np.int64)
    off += n_dims * item_size

    n_deltas = (n_rows - 1) * n_dims
    deltas = np.empty(n_deltas, dtype=np.int64)
    for i in range(n_deltas):
        u, off = _decode_unsigned_varint(payload, off)
        deltas[i] = _zigzag_decode_u32(u)
    if off != len(payload):
        raise ValueError(
            f"pose payload has {len(payload) - off} trailing bytes"
        )

    q = np.empty((n_rows, n_dims), dtype=np.int64)
    q[0] = first
    if n_rows > 1:
        deltas2d = deltas.reshape(n_rows - 1, n_dims)
        q[1:] = first + np.cumsum(deltas2d, axis=0)
    return (lo + q.astype(np.float32) * scale).astype(np.float32)


# ── QZMB1 grammar ───────────────────────────────────────────────────────────


def pack_qzmb1_block(
    *,
    block_size: int,
    arch_config_json: bytes,
    body: bytes = b"",
) -> QZMB1Block:
    """Frame a QZMB1 compact-model block from its header + body bytes.

    Parameters
    ----------
    block_size:
        FP4 codebook block size (PR93 uses 32). Stored as ``uint16``.
    arch_config_json:
        UTF-8 JSON bytes naming the architecture configuration. Stored
        ``uint16``-length-prefixed.
    body:
        Opaque post-arch-config body (typically tensor records). The
        primitive does not interpret these bytes.

    Returns
    -------
    QZMB1Block
        Container with the framed payload + the inputs (for diagnostics).
    """
    if not (0 < block_size < 65536):
        raise ValueError(
            f"block_size must satisfy 0 < block_size < 65536; got {block_size}"
        )
    if not isinstance(arch_config_json, (bytes, bytearray)):
        raise TypeError(
            f"arch_config_json must be bytes; got {type(arch_config_json)!r}"
        )
    arch_config_json = bytes(arch_config_json)
    if len(arch_config_json) >= 65536:
        raise ValueError(
            f"arch_config_json too long ({len(arch_config_json)} bytes); "
            "max 65535"
        )
    header = struct.pack("<HH", block_size, len(arch_config_json))
    payload = MAGIC_MODEL_COMPACT + header + arch_config_json + bytes(body)
    return QZMB1Block(
        payload=payload,
        block_size=int(block_size),
        arch_config_json=arch_config_json,
        body=bytes(body),
    )


def unpack_qzmb1_block(payload: bytes) -> QZMB1Block:
    """Inverse of :func:`pack_qzmb1_block` — parse the framing."""
    payload = bytes(payload)
    if payload[: len(MAGIC_MODEL_COMPACT)] != MAGIC_MODEL_COMPACT:
        raise ValueError(
            f"missing QZMB1 magic; got prefix {payload[:8]!r}"
        )
    off = len(MAGIC_MODEL_COMPACT)
    if len(payload) < off + 4:
        raise ValueError("truncated QZMB1 header")
    block_size, arch_len = struct.unpack_from("<HH", payload, off)
    off += 4
    end = off + arch_len
    if end > len(payload):
        raise ValueError(
            f"arch_config truncated: header says {arch_len} bytes but "
            f"only {len(payload) - off} available"
        )
    arch_config_json = bytes(payload[off:end])
    body = bytes(payload[end:])
    return QZMB1Block(
        payload=payload,
        block_size=int(block_size),
        arch_config_json=arch_config_json,
        body=body,
    )


__all__ = [
    "DeltaVarintPoseStream",
    "MAGIC_MODEL_COMPACT",
    "MAGIC_POSE_DV",
    "QZMB1Block",
    "decode_delta_varint_pose",
    "encode_delta_varint_pose",
    "pack_qzmb1_block",
    "unpack_qzmb1_block",
]
