# SPDX-License-Identifier: MIT
"""Receiver-visible lossless integer-plane codec primitives.

This module is NumPy-only and scorer-free so archive runtimes can vendor it.
It provides the reusable SNeRV LF payload rate codec: raster deltas, zigzag,
unsigned LEB128, and deterministic shared-shape headers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

SPATIAL_DELTA_ZIGZAG_LEB128_CODEC = "spatial_delta_zigzag_leb128_lzma"


class ReceiverIntegerPlaneCodecError(ValueError):
    """Raised when receiver integer-plane codec payloads are malformed."""


@dataclass(frozen=True)
class IntegerPlanePayload:
    """Encoded lossless integer planes and receiver-visible metadata."""

    codec: str
    raw: bytes
    header: dict[str, Any]
    canonical_int64_raw: bytes

    @property
    def raw_bytes(self) -> int:
        return len(self.raw)

    @property
    def canonical_int64_raw_bytes(self) -> int:
        return len(self.canonical_int64_raw)

    @property
    def canonical_int64_raw_sha256(self) -> str:
        return hashlib.sha256(self.canonical_int64_raw).hexdigest()


def encode_spatial_delta_zigzag_leb128_planes(
    planes: list[np.ndarray],
) -> IntegerPlanePayload:
    """Encode integer planes as raster deltas plus zigzag ULEB128 bytes."""

    arrays = [_validate_integer_plane(plane) for plane in planes]
    if not arrays:
        raise ReceiverIntegerPlaneCodecError("integer planes must be non-empty")
    canonical_raw = canonical_int64_raw(arrays)
    encoded_parts = []
    for arr in arrays:
        flat = np.asarray(arr, dtype=np.int64).reshape(-1)
        delta = np.empty_like(flat)
        delta[0] = flat[0]
        if flat.size > 1:
            delta[1:] = flat[1:] - flat[:-1]
        encoded_parts.append(encode_uleb128(zigzag_encode_i64(delta)))
    raw = b"".join(encoded_parts)
    header = {
        "codec": SPATIAL_DELTA_ZIGZAG_LEB128_CODEC,
        "dtype": "spatial_delta_zigzag_uleb128",
        **integer_plane_shape_header(arrays),
        "canonical_int64_raw_bytes": len(canonical_raw),
        "canonical_int64_raw_sha256": hashlib.sha256(canonical_raw).hexdigest(),
        "raw_bytes": len(raw),
    }
    return IntegerPlanePayload(
        codec=SPATIAL_DELTA_ZIGZAG_LEB128_CODEC,
        raw=raw,
        header=header,
        canonical_int64_raw=canonical_raw,
    )


def decode_spatial_delta_zigzag_leb128_planes(
    raw: bytes,
    *,
    header: dict[str, Any],
) -> list[np.ndarray]:
    """Decode planes encoded by :func:`encode_spatial_delta_zigzag_leb128_planes`."""

    if str(header.get("codec")) != SPATIAL_DELTA_ZIGZAG_LEB128_CODEC:
        raise ReceiverIntegerPlaneCodecError(
            f"unsupported integer-plane codec: {header.get('codec')!r}"
        )
    if len(raw) != int(header["raw_bytes"]):
        raise ReceiverIntegerPlaneCodecError("integer-plane raw byte count mismatch")
    out = []
    cursor = 0
    for shape in integer_plane_shapes_from_header(header):
        out_shape = tuple(int(v) for v in shape)
        count = int(np.prod(out_shape))
        zz, cursor = decode_uleb128(raw, count=count, offset=cursor)
        delta = zigzag_decode_u64(zz)
        values = np.cumsum(delta, dtype=np.int64)
        out.append(values.reshape(out_shape))
    if cursor != len(raw):
        raise ReceiverIntegerPlaneCodecError("integer-plane payload has unused bytes")
    canonical_raw = canonical_int64_raw(out)
    expected_sha = str(header.get("canonical_int64_raw_sha256", ""))
    if expected_sha and hashlib.sha256(canonical_raw).hexdigest() != expected_sha:
        raise ReceiverIntegerPlaneCodecError("integer-plane canonical sha256 mismatch")
    expected_bytes = header.get("canonical_int64_raw_bytes")
    if expected_bytes is not None and len(canonical_raw) != int(expected_bytes):
        raise ReceiverIntegerPlaneCodecError(
            "integer-plane canonical byte count mismatch"
        )
    return out


def integer_plane_shape_header(arrays: list[np.ndarray]) -> dict[str, Any]:
    """Return a compact shape header, sharing shape when every plane matches."""

    shapes = [list(_validate_integer_plane(a).shape) for a in arrays]
    if not shapes:
        raise ReceiverIntegerPlaneCodecError("integer planes must be non-empty")
    if all(shape == shapes[0] for shape in shapes):
        return {"shared_shape": shapes[0], "shape_count": len(shapes)}
    return {"shapes": shapes}


def integer_plane_shapes_from_header(header: dict[str, Any]) -> list[list[int]]:
    """Expand explicit or shared shape metadata into one shape per plane."""

    if "shared_shape" in header:
        count = int(header.get("shape_count", 0))
        if count <= 0:
            raise ReceiverIntegerPlaneCodecError("shared shape count must be positive")
        shape = [int(v) for v in header["shared_shape"]]
        _validate_shape(shape)
        return [shape for _ in range(count)]
    shapes = header.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        raise ReceiverIntegerPlaneCodecError("integer-plane header missing shapes")
    out = []
    for shape in shapes:
        row = [int(v) for v in shape]
        _validate_shape(row)
        out.append(row)
    return out


def canonical_int64_raw(planes: list[np.ndarray]) -> bytes:
    """Return the canonical int64 little-endian bytes for checksum custody."""

    arrays = [_validate_integer_plane(a) for a in planes]
    if not arrays:
        raise ReceiverIntegerPlaneCodecError("integer planes must be non-empty")
    return b"".join(np.asarray(a, dtype="<i8").reshape(-1).tobytes() for a in arrays)


def zigzag_encode_i64(values: np.ndarray) -> np.ndarray:
    """Map signed int64 values to unsigned zigzag values."""

    arr = np.asarray(values, dtype=np.int64)
    return np.where(arr >= 0, arr * 2, -arr * 2 - 1).astype(np.uint64)


def zigzag_decode_u64(values: np.ndarray) -> np.ndarray:
    """Map unsigned zigzag values back to signed int64 values."""

    arr = np.asarray(values, dtype=np.uint64)
    positive = (arr >> np.uint64(1)).astype(np.int64)
    negative = -positive - 1
    return np.where((arr & np.uint64(1)) == 0, positive, negative).astype(np.int64)


def encode_uleb128(values: np.ndarray) -> bytes:
    """Encode unsigned integers as LEB128 bytes."""

    out = bytearray()
    for value in np.asarray(values, dtype=np.uint64).reshape(-1):
        x = int(value)
        while x >= 0x80:
            out.append((x & 0x7F) | 0x80)
            x >>= 7
        out.append(x)
    return bytes(out)


def decode_uleb128(
    payload: bytes,
    *,
    count: int,
    offset: int = 0,
) -> tuple[np.ndarray, int]:
    """Decode ``count`` unsigned LEB128 values from ``payload``."""

    values = np.empty(int(count), dtype=np.uint64)
    cursor = int(offset)
    for idx in range(int(count)):
        shift = 0
        value = 0
        while True:
            if cursor >= len(payload):
                raise ReceiverIntegerPlaneCodecError("payload ended inside LEB128 value")
            byte = payload[cursor]
            cursor += 1
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
            if shift >= 64:
                raise ReceiverIntegerPlaneCodecError("LEB128 value exceeds uint64")
        values[idx] = value
    return values, cursor


def _validate_integer_plane(plane: np.ndarray) -> np.ndarray:
    arr = np.asarray(plane)
    if arr.size == 0:
        raise ReceiverIntegerPlaneCodecError("integer planes must be non-empty")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ReceiverIntegerPlaneCodecError("integer planes must contain integers")
    return arr.astype("<i8", copy=False)


def _validate_shape(shape: list[int]) -> None:
    if not shape or any(int(v) <= 0 for v in shape):
        raise ReceiverIntegerPlaneCodecError("integer-plane shapes must be positive")


__all__ = [
    "SPATIAL_DELTA_ZIGZAG_LEB128_CODEC",
    "IntegerPlanePayload",
    "ReceiverIntegerPlaneCodecError",
    "canonical_int64_raw",
    "decode_spatial_delta_zigzag_leb128_planes",
    "decode_uleb128",
    "encode_spatial_delta_zigzag_leb128_planes",
    "encode_uleb128",
    "integer_plane_shape_header",
    "integer_plane_shapes_from_header",
    "zigzag_decode_u64",
    "zigzag_encode_i64",
]
