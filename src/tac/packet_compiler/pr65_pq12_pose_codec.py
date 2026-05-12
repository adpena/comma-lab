"""PR65 ``henosis`` PQ12 pose codec — 12-bit / 3-byte / 2-value packed grammar.

This module extracts the REUSABLE 12-bit-per-value packed pose codec from the
PR65 public submission
(``submissions/henosis_qz_n3z_r25_clean/inflate.py``, lines 383-397) into a
typed, golden-vector-backed transducer.

The wire format (magic ``b"PQ12"``):

* ``magic[4]`` — ``b"PQ12"`` literal.
* ``uint16 n_frames`` — number of rows (PR65 uses 600).
* ``uint16 n_dims`` — number of columns (PR65 uses 6).
* ``f32[n_dims] mn`` — per-column minimum (offset).
* ``f32[n_dims] scale`` — per-column quantum.
* ``uint8[ceil(total / 2) * 3] packed`` — flat 12-bit pairs:
  every 3 bytes encodes 2 consecutive uint12 values (q0 = b0 | (b1[3:0] << 8);
  q1 = (b1[7:4]) | (b2 << 4)).

Recovery formula (PR65 source, lines 391-397): ::

    p = np.frombuffer(raw, dtype=np.uint8, offset=off).reshape(-1, 3).astype(np.uint16)
    q0 = p[:, 0] | ((p[:, 1] & 0x0F) << 8)
    q1 = (p[:, 1] >> 4) | (p[:, 2] << 4)
    q = np.empty(p.shape[0] * 2, dtype=np.uint16)
    q[0::2] = q0
    q[1::2] = q1
    return mn[None, :] + q[: n * d].reshape(n, d).astype(np.float32) * scale[None, :]

PR65 stores 12 bits per quantum (vs PR63's 16 bits-per-quantum uint16 view),
so it's 25% smaller per quantum at the cost of (a) a small alphabet ceiling
(4096 levels per column) and (b) a 12-bit unpack at inflate. PR65 hardcodes
column 0 with the same ``q / 512.0 + 20.0`` velocity recovery via the
``(mn, scale)`` per-column affine.

Source: ``experiments/results/public_pr_archive_kaggle_mirror/public_pr65_intake_20260505_auto/source/submissions/henosis_qz_n3z_r25_clean/inflate.py``
(SHA pinned via ``check_public_pr_intake_clones_pristine``-protected intake).

CLAUDE.md compliance
====================

* No scorer load — pure numpy + struct + stdlib.
* No MPS / torch import.
* No ``/tmp`` paths.
* Frozen dataclass; ``encode → decode`` is bit-exact on the
  ``pr65_pq12_pose_v1`` golden vector.
* OSS-friendly: public surface is the 4 names re-exported from
  ``tac.packet_compiler``.

[empirical:src/tac/packet_compiler/golden_vectors/pr65_pq12_pose_v1.json]

score_claim=false; promotion_eligible=false; ready_for_exact_eval_dispatch=false
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# ── Constants ───────────────────────────────────────────────────────────────

#: 4-byte magic identifying a PQ12 (12-bit / 3-byte / 2-value packed) payload.
MAGIC_PQ12: bytes = b"PQ12"
#: 12-bit max value (alphabet ceiling for PQ12-coded values).
PQ12_MAX_QUANTUM: int = 0xFFF  # = 4095


# ── Public dataclass ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PQ12PoseStream:
    """Encoded PR65 PQ12 pose payload.

    Attributes
    ----------
    payload:
        Bytes that begin with :data:`MAGIC_PQ12` and end with the last
        packed-pair byte; ready to brotli-compress + place in an archive
        member.
    n_frames, n_dims:
        Pose tensor shape (PR65 uses 600, 6).
    mn, scale:
        Per-column fp32 affine recovery: ``recovered = mn + q * scale``.
    """

    payload: bytes
    n_frames: int
    n_dims: int
    mn: np.ndarray
    scale: np.ndarray


# ── Internal pack/unpack helpers ────────────────────────────────────────────


def _pack_12bit_pairs(values: np.ndarray) -> bytes:
    """Pack a 1D uint12 array into 3-bytes-per-2-values little-endian.

    If the value count is odd, the trailing single value is packed into a
    final 3-byte cell with the upper 12 bits set to zero (PR65 grammar
    rounds the payload up to a multiple of 3 bytes).
    """
    if values.dtype != np.uint16:
        raise TypeError(
            f"values must be uint16 (12-bit values fit in 16-bit slot); "
            f"got {values.dtype}"
        )
    if values.size == 0:
        return b""
    if int(values.max()) > PQ12_MAX_QUANTUM:
        raise ValueError(
            f"value exceeds 12-bit range (max {PQ12_MAX_QUANTUM}); "
            f"got max {int(values.max())}"
        )
    # Pad to even count for the q0/q1 pairing.
    if values.size % 2 == 1:
        values = np.concatenate([values, np.zeros(1, dtype=np.uint16)])
    q0 = values[0::2].astype(np.uint16)
    q1 = values[1::2].astype(np.uint16)
    n_pairs = q0.size
    out = np.empty((n_pairs, 3), dtype=np.uint8)
    out[:, 0] = (q0 & 0xFF).astype(np.uint8)
    out[:, 1] = (((q0 >> 8) & 0x0F) | ((q1 & 0x0F) << 4)).astype(np.uint8)
    out[:, 2] = ((q1 >> 4) & 0xFF).astype(np.uint8)
    return out.tobytes()


def _unpack_12bit_pairs(packed: bytes, expected_count: int) -> np.ndarray:
    """Inverse of :func:`_pack_12bit_pairs`.

    Returns the first ``expected_count`` values from the packed stream.
    The packed stream length must be a multiple of 3 and large enough to
    contain ``expected_count`` values.
    """
    if len(packed) % 3 != 0:
        raise ValueError(
            f"packed length {len(packed)} is not a multiple of 3"
        )
    n_pairs = len(packed) // 3
    if expected_count > n_pairs * 2:
        raise ValueError(
            f"expected {expected_count} values but packed buffer "
            f"only fits {n_pairs * 2}"
        )
    p = np.frombuffer(packed, dtype=np.uint8).reshape(n_pairs, 3).astype(np.uint16)
    q0 = p[:, 0] | ((p[:, 1] & 0x0F) << 8)
    q1 = (p[:, 1] >> 4) | (p[:, 2] << 4)
    q = np.empty(n_pairs * 2, dtype=np.uint16)
    q[0::2] = q0
    q[1::2] = q1
    return q[:expected_count].copy()


# ── Encoder / decoder ───────────────────────────────────────────────────────


def encode_pq12_pose(
    poses: np.ndarray,
    *,
    mn: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> PQ12PoseStream:
    """Encode a ``(n_frames, n_dims)`` pose tensor under PR65's PQ12 grammar.

    Parameters
    ----------
    poses:
        Float pose tensor of shape ``(n_frames, n_dims)``. Values are
        quantised per-column as ``round((p - mn) / scale)``; the result
        must fit in 12 bits ``[0, 4095]``.
    mn:
        Optional per-column fp32 offset. If ``None``, derived from
        column minima so quantisation fits a non-negative grid.
    scale:
        Optional per-column fp32 quantum. If ``None``, derived from
        column range so the max value fits in 12 bits.

    Returns
    -------
    PQ12PoseStream
        Container with the magic-prefixed payload + recovery parameters.

    Raises
    ------
    ValueError
        On wrong shape, empty input, non-positive scale, or out-of-range
        quantisation.
    """
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"poses must be 2D; got shape {arr.shape}")
    n_frames, n_dims = arr.shape
    if n_frames == 0 or n_dims == 0:
        raise ValueError(f"poses must be non-empty; got shape {arr.shape}")
    if n_frames > 0xFFFF:
        raise ValueError(
            f"n_frames {n_frames} exceeds uint16 wire-format max 65535"
        )
    if n_dims > 0xFFFF:
        raise ValueError(
            f"n_dims {n_dims} exceeds uint16 wire-format max 65535"
        )

    if mn is None:
        mn = arr.min(axis=0).astype(np.float32)
    else:
        mn = np.asarray(mn, dtype=np.float32)
    if mn.shape != (n_dims,):
        raise ValueError(f"mn shape {mn.shape} != ({n_dims},)")

    if scale is None:
        rng = (arr.max(axis=0) - mn).astype(np.float32)
        rng = np.where(rng <= 0.0, np.float32(1.0), rng)
        scale = (rng / float(PQ12_MAX_QUANTUM)).astype(np.float32)
    else:
        scale = np.asarray(scale, dtype=np.float32)
    if scale.shape != (n_dims,):
        raise ValueError(f"scale shape {scale.shape} != ({n_dims},)")
    if not np.all(scale > 0):
        raise ValueError("scale must be strictly positive everywhere")

    q = np.round((arr - mn) / scale).astype(np.int64)
    if q.min() < 0 or q.max() > PQ12_MAX_QUANTUM:
        raise ValueError(
            f"quantised values out of 12-bit range [0, {PQ12_MAX_QUANTUM}]; "
            f"got [{q.min()}, {q.max()}]"
        )

    flat = q.astype(np.uint16).reshape(-1)
    packed = _pack_12bit_pairs(flat)

    body = bytearray()
    body += MAGIC_PQ12
    body += struct.pack("<HH", n_frames, n_dims)
    body += mn.astype(np.float32).tobytes()
    body += scale.astype(np.float32).tobytes()
    body += packed

    return PQ12PoseStream(
        payload=bytes(body),
        n_frames=int(n_frames),
        n_dims=int(n_dims),
        mn=mn,
        scale=scale,
    )


def decode_pq12_pose(payload: bytes) -> np.ndarray:
    """Decode a PR65 PQ12 magic-prefixed payload back to a float pose tensor.

    Returns
    -------
    np.ndarray
        Float32 pose tensor of shape ``(n_frames, n_dims)``.

    Raises
    ------
    ValueError
        On wrong magic, truncated header, truncated mn/scale vectors,
        non-positive scale, truncated packed body, or trailing bytes
        beyond the per-value capacity.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"payload must be bytes-like; got {type(payload)!r}")
    payload = bytes(payload)
    if payload[:4] != MAGIC_PQ12:
        raise ValueError(f"missing PQ12 magic; got prefix {payload[:4]!r}")
    off = 4
    if len(payload) < off + 4:
        raise ValueError("truncated PQ12 shape header")
    n_frames, n_dims = struct.unpack_from("<HH", payload, off)
    off += 4
    if n_frames == 0 or n_dims == 0:
        raise ValueError(
            f"n_frames and n_dims must be > 0; got ({n_frames}, {n_dims})"
        )
    if len(payload) < off + 4 * n_dims:
        raise ValueError("truncated PQ12 mn vector")
    mn = np.frombuffer(payload, dtype=np.float32, count=n_dims, offset=off).copy()
    off += 4 * n_dims
    if len(payload) < off + 4 * n_dims:
        raise ValueError("truncated PQ12 scale vector")
    scale = np.frombuffer(payload, dtype=np.float32, count=n_dims, offset=off).copy()
    if not np.all(scale > 0):
        raise ValueError("scale must be strictly positive everywhere")
    off += 4 * n_dims
    total = int(n_frames) * int(n_dims)
    # PR65 packs 2 values per 3 bytes; payload rounds up to next 3-byte cell.
    n_pairs = (total + 1) // 2
    packed_len = n_pairs * 3
    if len(payload) < off + packed_len:
        raise ValueError(
            f"truncated PQ12 packed body: need {packed_len} bytes, "
            f"have {len(payload) - off}"
        )
    if len(payload) != off + packed_len:
        raise ValueError(
            f"trailing bytes after PQ12 packed body: "
            f"{len(payload) - off - packed_len} extra"
        )
    flat = _unpack_12bit_pairs(payload[off : off + packed_len], total)
    q = flat.reshape(n_frames, n_dims)
    return (
        mn[None, :].astype(np.float32)
        + q.astype(np.float32) * scale[None, :].astype(np.float32)
    ).astype(np.float32)


__all__ = [
    "MAGIC_PQ12",
    "PQ12_MAX_QUANTUM",
    "PQ12PoseStream",
    "decode_pq12_pose",
    "encode_pq12_pose",
]
