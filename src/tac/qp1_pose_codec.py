"""QP1 pose codec — PR #67 EthanYang qpose14_qzs3 byte-identical.

PR #67's archive (rank-1 leaderboard ~0.31) ships poses as ~1140 bytes
(uncompressed, ~899 brotli-compressed) for 600 pose-pairs encoded as:

    Layout
    ------
    bytes [0:3]  : magic = b"QP1"
    bytes [3:5]  : little-endian uint16 first quantized pose-0 word
    bytes [5:]   : ZigZag-VLQ-encoded deltas of subsequent pose-0 words

    Quantization contract (matches pr67_inflate.py:809-811)
    --------------------------------------------------------
        q0 = round((velocity - 20.0) * 512.0)        # uint16
        q[i>0] = round(pose[i] * 2048.0)             # int16

    Decode-time, columns 1-5 are zeroed (only column 0 = velocity is
    encoded). PR #67 confirmed that PoseNet's distortion contribution
    from cols 1-5 is negligible at this operating point.

    Variable-length quantity (VLQ) is little-endian base-128:
    each byte's bottom 7 bits are payload, top bit is continuation.
    ZigZag maps signed -> unsigned via:
        encode: (n << 1) ^ (n >> 31)
        decode: (n >> 1) ^ -(n & 1)

This module implements the WRITER side. The reader (pr67_inflate.py:789-806)
is reproduced verbatim by ``decode_qp1`` for parity testing.

This is intentionally LOSSY (cols 1-5 are dropped). The existing
``encode_pose_qpose14_col_delta`` in
``experiments/build_renderer_packed_payload_archive.py`` is a DIFFERENT
codec (magic ``b"QP14"``, all-6-columns col-deltas) and is preserved
for callers that need the broader contract.
"""
from __future__ import annotations

import struct
from typing import Iterable

import numpy as np
import torch

QP1_MAGIC = b"QP1"
VELOCITY_OFFSET = 20.0
VELOCITY_SCALE = 512.0
POSE_SCALE = 2048.0


__all__ = [
    "QP1_MAGIC",
    "VELOCITY_OFFSET",
    "VELOCITY_SCALE",
    "POSE_SCALE",
    "encode_qp1",
    "decode_qp1",
]


def _zigzag_encode(n: int) -> int:
    """Map signed int -> unsigned int via ZigZag (matches PR #67 line 803 inverse)."""

    if n >= 0:
        return n << 1
    return ((-n) << 1) - 1


def _vlq_encode(n: int) -> bytes:
    """Encode unsigned int as little-endian base-128 VLQ."""

    out = bytearray()
    while True:
        if n < 0x80:
            out.append(n & 0x7F)
            return bytes(out)
        out.append((n & 0x7F) | 0x80)
        n >>= 7


def encode_qp1(
    poses: torch.Tensor | np.ndarray | Iterable[Iterable[float]],
    *,
    velocity_offset: float = VELOCITY_OFFSET,
    velocity_scale: float = VELOCITY_SCALE,
) -> bytes:
    """Encode an (N, 6) pose array as PR #67 QP1 bytes.

    ``poses[:, 0]`` is the velocity column; ``poses[:, 1:]`` are dropped.
    The output is the uncompressed QP1 stream — callers that want PR #67's
    on-disk ``pose_q_br`` should brotli-compress the result.

    Raises:
        ValueError if any quantized velocity falls outside [0, 0xFFFF].
    """

    if isinstance(poses, torch.Tensor):
        arr = poses.detach().cpu().numpy()
    else:
        arr = np.asarray(poses)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"poses must be (N, >=1); got shape {arr.shape}")
    n_rows = int(arr.shape[0])
    if n_rows == 0:
        raise ValueError("poses must have at least one row")

    velocities = arr[:, 0].astype(np.float64)
    qs = np.rint((velocities - velocity_offset) * velocity_scale).astype(np.int64)
    if (qs < 0).any() or (qs > 0xFFFF).any():
        bad = int(np.argmax((qs < 0) | (qs > 0xFFFF)))
        raise ValueError(
            f"qp1 velocity at row {bad} -> {int(qs[bad])} outside [0, 65535]; "
            "input velocity is outside the qpose14 quantization band"
        )
    qs_u16 = qs.astype(np.uint16)

    out = bytearray(QP1_MAGIC)
    out += struct.pack("<H", int(qs_u16[0]))
    prev = int(qs_u16[0])
    for i in range(1, n_rows):
        cur = int(qs_u16[i])
        delta = cur - prev
        out += _vlq_encode(_zigzag_encode(delta))
        prev = cur
    return bytes(out)


def decode_qp1(
    payload: bytes,
    *,
    velocity_offset: float = VELOCITY_OFFSET,
    velocity_scale: float = VELOCITY_SCALE,
    pose_dim: int = 6,
) -> np.ndarray:
    """Decode QP1 bytes -> (N, pose_dim) float32 array (cols 1+ zeroed).

    Mirrors pr67_inflate.py:789-811 verbatim so packer round-trip tests can
    assert byte-identical behavior without importing PR #67.
    """

    if not payload.startswith(QP1_MAGIC):
        raise ValueError(f"bad QP1 magic: {payload[:3]!r}")
    if len(payload) < 5:
        raise ValueError("QP1 payload too short")
    first = int.from_bytes(payload[3:5], "little")
    vals = [first]
    cursor = 5
    while cursor < len(payload):
        shift = 0
        acc = 0
        while True:
            if cursor >= len(payload):
                raise ValueError("truncated VLQ at end of QP1 stream")
            byte = payload[cursor]
            cursor += 1
            acc |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        delta = (acc >> 1) ^ -(acc & 1)
        vals.append((vals[-1] + delta) & 0xFFFF)

    n = len(vals)
    q_pose = np.zeros((n, pose_dim), dtype=np.uint16)
    q_pose[:, 0] = np.asarray(vals, dtype=np.uint16)
    pose_np = np.empty(q_pose.shape, dtype=np.float32)
    pose_np[:, 0] = q_pose[:, 0].astype(np.float32) / velocity_scale + velocity_offset
    if pose_dim > 1:
        pose_np[:, 1:] = q_pose[:, 1:].view(np.int16).astype(np.float32) / POSE_SCALE
    return pose_np
