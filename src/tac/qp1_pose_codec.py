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
from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable

import numpy as np
import torch

QP1_MAGIC = b"QP1"
QPV1_MAGIC = b"QPV1"
QP2_MAGIC = b"PVR1"
VELOCITY_OFFSET = 20.0
VELOCITY_SCALE = 512.0
POSE_SCALE = 2048.0


__all__ = [
    "QP1_MAGIC",
    "QPV1_MAGIC",
    "QP2_MAGIC",
    "VELOCITY_OFFSET",
    "VELOCITY_SCALE",
    "POSE_SCALE",
    "QPV1DimensionStream",
    "QPV1Payload",
    "encode_qp1",
    "decode_qp1",
    "parse_qpv1",
    "decode_qpv1",
    "encode_qp2_residual_topk",
    "decode_qp2_residual_topk",
    "qp2_residual_atom_plan",
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


def _read_zigzag_vlq(payload: bytes, cursor: int) -> tuple[int, int]:
    acc = 0
    shift = 0
    while True:
        if cursor >= len(payload):
            raise ValueError("truncated ZigZag-VLQ payload")
        byte = payload[cursor]
        cursor += 1
        acc |= (byte & 0x7F) << shift
        if byte < 0x80:
            break
        shift += 7
        if shift > 63:
            raise ValueError("overlong ZigZag-VLQ payload")
    return (acc >> 1) ^ -(acc & 1), cursor


@dataclass(frozen=True)
class QPV1DimensionStream:
    """One PR77/QPV1 integer delta stream for a single pose dimension."""

    dim: int
    offset: float
    scale: float
    values: tuple[int, ...]


@dataclass(frozen=True)
class QPV1Payload:
    """Parsed PR77/QPV1 multidimensional pose stream.

    Wire format, from PR77's public inflate reference:
    ``QPV1`` + ``u16 count`` + ``u8 dim_count`` followed by per-dimension
    records: ``u8 dim``, ``f32 offset``, ``f32 scale``, ``i32 first_value``,
    then ZigZag-VLQ deltas until ``count`` values are present.
    """

    count: int
    pose_dim: int
    streams: tuple[QPV1DimensionStream, ...]

    def values_by_dim(self) -> dict[int, list[int]]:
        return {stream.dim: list(stream.values) for stream in self.streams}

    def with_values(self, values_by_dim: dict[int, list[int]]) -> "QPV1Payload":
        streams: list[QPV1DimensionStream] = []
        for stream in self.streams:
            values = tuple(int(value) for value in values_by_dim[stream.dim])
            if len(values) != self.count:
                raise ValueError(
                    f"QPV1 dim {stream.dim} expected {self.count} values, got {len(values)}"
                )
            streams.append(
                QPV1DimensionStream(
                    dim=stream.dim,
                    offset=stream.offset,
                    scale=stream.scale,
                    values=values,
                )
            )
        return QPV1Payload(count=self.count, pose_dim=self.pose_dim, streams=tuple(streams))

    def to_numpy(self) -> np.ndarray:
        poses = np.zeros((self.count, self.pose_dim), dtype=np.float32)
        for stream in self.streams:
            values = np.asarray(stream.values, dtype=np.float32)
            poses[:, stream.dim] = np.float32(stream.offset) + values / np.float32(stream.scale)
        return poses

    def to_bytes(self) -> bytes:
        if self.count <= 0 or self.count > 0xFFFF:
            raise ValueError(f"invalid QPV1 row count: {self.count}")
        if not self.streams:
            raise ValueError("QPV1 payload must contain at least one dimension stream")
        out = bytearray(QPV1_MAGIC)
        out.extend(struct.pack("<HB", self.count, len(self.streams)))
        seen: set[int] = set()
        for stream in self.streams:
            if stream.dim in seen:
                raise ValueError(f"duplicate QPV1 dimension: {stream.dim}")
            seen.add(stream.dim)
            if stream.dim < 0 or stream.dim >= self.pose_dim or stream.dim > 0xFF:
                raise ValueError(f"invalid QPV1 dimension: {stream.dim}")
            if not isfinite(float(stream.offset)):
                raise ValueError(f"non-finite QPV1 offset for dim {stream.dim}")
            if not isfinite(float(stream.scale)) or float(stream.scale) == 0.0:
                raise ValueError(f"invalid QPV1 scale for dim {stream.dim}: {stream.scale}")
            if len(stream.values) != self.count:
                raise ValueError(
                    f"QPV1 dim {stream.dim} expected {self.count} values, got {len(stream.values)}"
                )
            values = [int(value) for value in stream.values]
            if any(value < -(2**31) or value > 2**31 - 1 for value in values):
                raise ValueError(f"QPV1 dim {stream.dim} value outside signed int32 range")
            out.append(stream.dim)
            out.extend(struct.pack("<ff", float(stream.offset), float(stream.scale)))
            out.extend(struct.pack("<i", values[0]))
            previous = values[0]
            for value in values[1:]:
                out.extend(_vlq_encode(_zigzag_encode(value - previous)))
                previous = value
        return bytes(out)


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


def parse_qpv1(payload: bytes, *, pose_dim: int = 6) -> QPV1Payload:
    """Parse PR77/QPV1 multidimensional integer-delta pose bytes.

    The parser is intentionally strict: malformed headers, duplicate
    dimensions, truncated VLQs, trailing bytes, and non-finite scale metadata
    all fail closed before any candidate builder can dispatch or score.
    """

    if not payload.startswith(QPV1_MAGIC):
        raise ValueError(f"bad QPV1 magic: {payload[:4]!r}")
    if len(payload) < 7:
        raise ValueError("QPV1 payload too short")
    count = struct.unpack_from("<H", payload, 4)[0]
    dim_count = payload[6]
    if count <= 0:
        raise ValueError("QPV1 row count must be positive")
    if dim_count <= 0:
        raise ValueError("QPV1 dimension count must be positive")
    cursor = 7
    seen: set[int] = set()
    streams: list[QPV1DimensionStream] = []
    for _ in range(dim_count):
        if cursor + 13 > len(payload):
            raise ValueError("truncated QPV1 dimension header")
        dim = payload[cursor]
        cursor += 1
        if dim in seen:
            raise ValueError(f"duplicate QPV1 dimension: {dim}")
        if dim >= pose_dim:
            raise ValueError(f"QPV1 dimension {dim} outside pose_dim {pose_dim}")
        seen.add(dim)
        offset = struct.unpack_from("<f", payload, cursor)[0]
        cursor += 4
        scale = struct.unpack_from("<f", payload, cursor)[0]
        cursor += 4
        if not isfinite(float(offset)):
            raise ValueError(f"non-finite QPV1 offset for dim {dim}")
        if not isfinite(float(scale)) or float(scale) == 0.0:
            raise ValueError(f"invalid QPV1 scale for dim {dim}: {scale}")
        first = struct.unpack_from("<i", payload, cursor)[0]
        cursor += 4
        values = [int(first)]
        while len(values) < count:
            delta, cursor = _read_zigzag_vlq(payload, cursor)
            values.append(values[-1] + delta)
        streams.append(
            QPV1DimensionStream(
                dim=int(dim),
                offset=float(offset),
                scale=float(scale),
                values=tuple(values),
            )
        )
    if cursor != len(payload):
        raise ValueError(f"QPV1 payload has {len(payload) - cursor} trailing bytes")
    return QPV1Payload(count=int(count), pose_dim=int(pose_dim), streams=tuple(streams))


def decode_qpv1(payload: bytes, *, pose_dim: int = 6) -> np.ndarray:
    """Decode PR77/QPV1 bytes to an ``(N, pose_dim)`` float32 pose array."""

    return parse_qpv1(payload, pose_dim=pose_dim).to_numpy()


def _as_pose_array(
    poses: torch.Tensor | np.ndarray | Iterable[Iterable[float]],
) -> np.ndarray:
    if isinstance(poses, torch.Tensor):
        arr = poses.detach().cpu().numpy()
    else:
        arr = np.asarray(poses)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"poses must be (N, >=2); got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("poses must have at least one row")
    return arr.astype(np.float64, copy=False)


def _quantize_velocity_words(
    arr: np.ndarray,
    *,
    velocity_offset: float,
    velocity_scale: float,
) -> np.ndarray:
    qs = np.rint((arr[:, 0] - velocity_offset) * velocity_scale).astype(np.int64)
    if (qs < 0).any() or (qs > 0xFFFF).any():
        bad = int(np.argmax((qs < 0) | (qs > 0xFFFF)))
        raise ValueError(
            f"qp2 velocity at row {bad} -> {int(qs[bad])} outside [0, 65535]; "
            "input velocity is outside the qpose quantization band"
        )
    return qs.astype(np.uint16)


def _float_to_half_word(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", float(value)))[0]


def _half_word_to_float(word: int) -> float:
    return struct.unpack("<e", struct.pack("<H", int(word) & 0xFFFF))[0]


def qp2_residual_atom_plan(
    poses: torch.Tensor | np.ndarray | Iterable[Iterable[float]],
    *,
    topk: int,
    velocity_offset: float = VELOCITY_OFFSET,
    velocity_scale: float = VELOCITY_SCALE,
) -> dict[str, Any]:
    """Return the deterministic residual-atom plan used by QP2/PVR1.

    QP2 is a contest-runtime-compatible pose atom family: the wire payload is
    the existing ``PVR1`` contract handled by ``unpack_renderer_payload.py``.
    Velocity remains one low-dimensional QP1-style scalar per frame; non-
    velocity dimensions are reconstructed from per-dimension fp16 means plus
    charged top-K fp16 residual atoms.
    """

    if topk < 0:
        raise ValueError(f"topk must be non-negative, got {topk}")
    arr = _as_pose_array(poses)
    n_rows, pose_dim = int(arr.shape[0]), int(arr.shape[1])
    max_atoms = n_rows * max(0, pose_dim - 1)
    if topk > max_atoms:
        raise ValueError(f"topk {topk} exceeds available non-velocity atoms {max_atoms}")

    velocity_words = _quantize_velocity_words(
        arr,
        velocity_offset=velocity_offset,
        velocity_scale=velocity_scale,
    )
    mean_words: list[int] = []
    mean_values: list[float] = []
    for dim in range(1, pose_dim):
        mean_word = _float_to_half_word(float(arr[:, dim].mean()))
        mean_words.append(mean_word)
        mean_values.append(_half_word_to_float(mean_word))

    residual_atoms: list[tuple[float, int, int, int]] = []
    for row in range(n_rows):
        for dim in range(1, pose_dim):
            value = float(arr[row, dim])
            residual = abs(value - mean_values[dim - 1])
            half_word = _float_to_half_word(value)
            residual_atoms.append((residual, row, dim, half_word))
    residual_atoms.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = residual_atoms[:topk]
    return {
        "n_rows": n_rows,
        "pose_dim": pose_dim,
        "topk": int(topk),
        "velocity_words": velocity_words.astype(np.uint16, copy=False),
        "mean_words": mean_words,
        "atoms": [
            {
                "row": int(row),
                "dim": int(dim),
                "key": int(row * pose_dim + dim),
                "half_word": int(half_word),
                "abs_residual": float(residual),
            }
            for residual, row, dim, half_word in selected
        ],
    }


def encode_qp2_residual_topk(
    poses: torch.Tensor | np.ndarray | Iterable[Iterable[float]],
    *,
    topk: int,
    velocity_offset: float = VELOCITY_OFFSET,
    velocity_scale: float = VELOCITY_SCALE,
) -> bytes:
    """Encode QP2 residual atoms using the contest-supported ``PVR1`` wire format."""

    plan = qp2_residual_atom_plan(
        poses,
        topk=topk,
        velocity_offset=velocity_offset,
        velocity_scale=velocity_scale,
    )
    n_rows = int(plan["n_rows"])
    pose_dim = int(plan["pose_dim"])
    velocity_words = plan["velocity_words"]

    out = bytearray(QP2_MAGIC)
    out += struct.pack("<HHHH", n_rows, pose_dim, int(topk), int(velocity_words[0]))
    for word in plan["mean_words"]:
        out += struct.pack("<H", int(word))
    prev = int(velocity_words[0])
    for row, cur_word in enumerate(velocity_words[1:], start=1):
        cur = int(cur_word)
        delta = cur - prev
        if not -32768 <= delta <= 32767:
            raise ValueError(f"velocity delta out of int16 range at row {row}: {delta}")
        out += struct.pack("<h", delta)
        prev = cur
    for atom in plan["atoms"]:
        out += struct.pack("<HH", int(atom["key"]), int(atom["half_word"]))
    return bytes(out)


def decode_qp2_residual_topk(
    payload: bytes,
    *,
    velocity_offset: float = VELOCITY_OFFSET,
    velocity_scale: float = VELOCITY_SCALE,
) -> np.ndarray:
    """Decode QP2/PVR1 bytes into a float32 pose array."""

    if len(payload) < 12 or payload[:4] != QP2_MAGIC:
        raise ValueError(f"bad QP2/PVR1 magic: {payload[:4]!r}")
    n_rows, pose_dim, topk, first_velocity_q = struct.unpack_from("<HHHH", payload, 4)
    if n_rows <= 0 or pose_dim <= 1 or n_rows > 10_000 or pose_dim > 64:
        raise ValueError(f"invalid QP2 pose shape: rows={n_rows}, cols={pose_dim}")
    max_atoms = n_rows * (pose_dim - 1)
    if topk > max_atoms:
        raise ValueError(f"topk {topk} exceeds available non-velocity atoms {max_atoms}")

    means_start = 12
    means_end = means_start + (pose_dim - 1) * 2
    deltas_end = means_end + max(0, n_rows - 1) * 2
    expected = deltas_end + topk * 4
    if len(payload) != expected:
        raise ValueError(
            f"QP2 payload length mismatch: expected {expected}, got {len(payload)}"
        )
    mean_words = [
        struct.unpack_from("<H", payload, means_start + (dim - 1) * 2)[0]
        for dim in range(1, pose_dim)
    ]

    out = np.zeros((n_rows, pose_dim), dtype=np.float32)
    velocity_q = int(first_velocity_q)
    delta_offset = means_end
    for row in range(n_rows):
        if row > 0:
            delta = struct.unpack_from("<h", payload, delta_offset)[0]
            delta_offset += 2
            velocity_q = (velocity_q + delta) & 0xFFFF
        out[row, 0] = velocity_q / velocity_scale + velocity_offset
        for dim in range(1, pose_dim):
            out[row, dim] = _half_word_to_float(mean_words[dim - 1])

    atom_offset = deltas_end
    seen: set[int] = set()
    for _ in range(topk):
        key, half_word = struct.unpack_from("<HH", payload, atom_offset)
        atom_offset += 4
        row, dim = divmod(int(key), pose_dim)
        if row >= n_rows or dim <= 0 or dim >= pose_dim:
            raise ValueError(f"QP2 residual atom out of bounds: row={row}, dim={dim}")
        if key in seen:
            raise ValueError(f"duplicate QP2 residual atom key: {key}")
        seen.add(key)
        out[row, dim] = _half_word_to_float(half_word)
    return out
