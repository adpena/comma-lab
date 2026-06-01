# SPDX-License-Identifier: MIT
"""Deterministic integer-stream codec portfolio for compact receiver sections."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

INT_STREAM_CODEC_MAGIC = b"ISC1"
_HEADER = struct.Struct("<4sBI")


@dataclass(frozen=True)
class IntStreamCodecStats:
    """Byte accounting for one unsigned integer stream."""

    mode: str
    count: int
    max_value: int
    payload_bytes: int
    envelope_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "count": self.count,
            "max_value": self.max_value,
            "payload_bytes": self.payload_bytes,
            "envelope_bytes": self.envelope_bytes,
        }


def encode_uint_stream(
    values: Any,
    *,
    mode: str = "auto",
    max_value: int | None = None,
) -> bytes:
    """Encode an unsigned integer stream with a measured mode portfolio.

    Modes are intentionally small and contest-runtime friendly:
    ``raw_u32``, ``raw_u16``, ``varint``, ``varint_delta_zigzag``,
    ``zero_run_varints``, ``bitpack_fixed``, and ``packed_bitmask``.
    ``auto`` chooses the shortest charged envelope among legal modes.
    """

    arr = _as_uint_array(values)
    max_v = int(arr.max()) if arr.size else 0
    if max_value is not None:
        max_v = max(max_v, int(max_value))
    normalized = str(mode).strip().lower()
    if normalized in {"raw_u16_legacy", "raw_uint16_legacy", "u16_legacy"}:
        if max_v > 0xFFFF:
            raise ValueError(f"raw_uint16_legacy cannot encode max_value {max_v}")
        return arr.astype(np.uint16, copy=False).tobytes()

    modes: tuple[str, ...]
    if normalized in {"auto", "portfolio_auto"}:
        candidates = ["raw_u32", "varint", "varint_delta_zigzag", "zero_run_varints"]
        if max_v <= 0xFFFF:
            candidates.append("raw_u16")
        if max_v <= 1:
            candidates.append("packed_bitmask")
        if max_v >= 0:
            candidates.append("bitpack_fixed")
        modes = tuple(candidates)
    else:
        modes = (normalized,)
    encoded = [_encode_enveloped(arr, mode=item, max_value=max_v) for item in modes]
    return min(encoded, key=lambda item: len(item[0]))[0]


def decode_uint_stream(
    blob: bytes,
    *,
    count: int | None = None,
    max_value: int | None = None,
) -> np.ndarray:
    """Decode an integer stream emitted by :func:`encode_uint_stream`."""

    if not blob.startswith(INT_STREAM_CODEC_MAGIC):
        if count is None:
            raise ValueError("legacy integer stream decode requires count")
        if len(blob) == int(count) * 2:
            arr = np.frombuffer(blob, dtype=np.uint16).astype(np.int64)
        elif len(blob) == int(count) * 4:
            arr = np.frombuffer(blob, dtype=np.uint32).astype(np.int64)
        else:
            raise ValueError(
                f"legacy integer stream length {len(blob)} does not match "
                f"count {count} as raw_u16/raw_u32"
            )
        _validate_decoded(arr, count=count, max_value=max_value)
        return arr

    header, payload = _read_envelope(blob)
    mode = str(header["mode"])
    expected_count = int(header["count"])
    expected_max = int(header["max_value"])
    if count is not None and int(count) != expected_count:
        raise ValueError(f"integer stream count {expected_count} != expected {count}")
    if max_value is not None and expected_max > int(max_value):
        raise ValueError(
            f"integer stream max_value {expected_max} exceeds expected {max_value}"
        )
    arr = _decode_payload(payload, mode=mode, count=expected_count, max_value=expected_max)
    _validate_decoded(arr, count=expected_count, max_value=expected_max)
    return arr


def int_stream_codec_stats(blob: bytes, *, count: int | None = None) -> IntStreamCodecStats:
    """Return integer stream byte metadata without expanding the stream."""

    if not blob.startswith(INT_STREAM_CODEC_MAGIC):
        if count is None:
            raise ValueError("legacy integer stream stats require count")
        mode = "raw_u16_legacy" if len(blob) == int(count) * 2 else "raw_u32_legacy"
        return IntStreamCodecStats(
            mode=mode,
            count=int(count),
            max_value=-1,
            payload_bytes=len(blob),
            envelope_bytes=0,
        )
    header, _payload = _read_envelope(blob)
    return IntStreamCodecStats(
        mode=str(header["mode"]),
        count=int(header["count"]),
        max_value=int(header["max_value"]),
        payload_bytes=int(header["payload_bytes"]),
        envelope_bytes=len(blob),
    )


def pack_fixed_width_uints(values: Any, *, bits: int) -> bytes:
    """Pack unsigned integers LSB-first using exactly ``bits`` per value."""

    if bits <= 0 or bits > 32:
        raise ValueError(f"bits must be in [1, 32]; got {bits}")
    arr = _as_uint_array(values)
    limit = (1 << bits) - 1
    if arr.size and int(arr.max()) > limit:
        raise ValueError(f"value {int(arr.max())} exceeds {bits}-bit limit {limit}")
    out = bytearray()
    acc = 0
    nbits = 0
    for raw in arr.tolist():
        acc |= int(raw) << nbits
        nbits += bits
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
    if nbits:
        out.append(acc & 0xFF)
    return bytes(out)


def unpack_fixed_width_uints(blob: bytes, *, bits: int, count: int) -> np.ndarray:
    """Unpack LSB-first fixed-width unsigned integers."""

    if bits <= 0 or bits > 32:
        raise ValueError(f"bits must be in [1, 32]; got {bits}")
    out: list[int] = []
    acc = 0
    nbits = 0
    pos = 0
    mask = (1 << bits) - 1
    while len(out) < int(count):
        while nbits < bits:
            if pos >= len(blob):
                raise ValueError("truncated fixed-width integer stream")
            acc |= int(blob[pos]) << nbits
            pos += 1
            nbits += 8
        out.append(acc & mask)
        acc >>= bits
        nbits -= bits
    return np.asarray(out, dtype=np.int64)


def encode_varint(n: int) -> bytes:
    """LEB128-style unsigned varint."""

    if n < 0:
        raise ValueError(f"varint cannot encode negative value {n}")
    out = bytearray()
    value = int(n)
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            byte |= 0x80
        out.append(byte)
        if not value:
            return bytes(out)


def decode_varint(blob: bytes, pos: int = 0) -> tuple[int, int]:
    """Decode one unsigned LEB128 varint from ``blob[pos:]``."""

    value = 0
    shift = 0
    while True:
        if pos >= len(blob):
            raise ValueError("truncated varint")
        byte = int(blob[pos])
        pos += 1
        value |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _encode_enveloped(
    arr: np.ndarray,
    *,
    mode: str,
    max_value: int,
) -> tuple[bytes, IntStreamCodecStats]:
    payload = _encode_payload(arr, mode=mode, max_value=max_value)
    header = {
        "schema": "int_stream_codec_envelope.v1",
        "mode": mode,
        "count": int(arr.size),
        "max_value": int(max_value),
        "payload_bytes": len(payload),
        "false_authority": True,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    blob = _HEADER.pack(INT_STREAM_CODEC_MAGIC, 1, len(header_bytes)) + header_bytes + payload
    return (
        blob,
        IntStreamCodecStats(
            mode=mode,
            count=int(arr.size),
            max_value=int(max_value),
            payload_bytes=len(payload),
            envelope_bytes=len(blob),
        ),
    )


def _encode_payload(arr: np.ndarray, *, mode: str, max_value: int) -> bytes:
    if mode == "raw_u32":
        return arr.astype(np.uint32, copy=False).tobytes()
    if mode == "raw_u16":
        if max_value > 0xFFFF:
            raise ValueError(f"raw_u16 cannot encode max_value {max_value}")
        return arr.astype(np.uint16, copy=False).tobytes()
    if mode == "varint":
        return b"".join(encode_varint(int(value)) for value in arr.tolist())
    if mode == "varint_delta_zigzag":
        prev = 0
        out = bytearray()
        for idx, raw in enumerate(arr.tolist()):
            value = int(raw)
            if idx == 0:
                out.extend(encode_varint(value))
            else:
                out.extend(encode_varint(_zigzag_encode(value - prev)))
            prev = value
        return bytes(out)
    if mode == "zero_run_varints":
        out = bytearray()
        i = 0
        values = arr.tolist()
        while i < len(values):
            value = int(values[i])
            if value == 0:
                run = 1
                while i + run < len(values) and int(values[i + run]) == 0:
                    run += 1
                out.extend(encode_varint(0))
                out.extend(encode_varint(run))
                i += run
            else:
                out.extend(encode_varint(value + 1))
                i += 1
        return bytes(out)
    if mode == "bitpack_fixed":
        bits = max(1, math.ceil(math.log2(max(2, max_value + 1))))
        return pack_fixed_width_uints(arr, bits=bits)
    if mode == "packed_bitmask":
        if max_value > 1:
            raise ValueError("packed_bitmask requires max_value <= 1")
        return pack_fixed_width_uints(arr, bits=1)
    raise ValueError(f"unsupported integer stream mode: {mode!r}")


def _decode_payload(
    payload: bytes,
    *,
    mode: str,
    count: int,
    max_value: int,
) -> np.ndarray:
    if mode == "raw_u32":
        arr = np.frombuffer(payload, dtype=np.uint32).astype(np.int64)
    elif mode == "raw_u16":
        arr = np.frombuffer(payload, dtype=np.uint16).astype(np.int64)
    elif mode == "varint":
        pos = 0
        out: list[int] = []
        while pos < len(payload):
            value, pos = decode_varint(payload, pos)
            out.append(value)
        arr = np.asarray(out, dtype=np.int64)
    elif mode == "varint_delta_zigzag":
        pos = 0
        out = []
        prev = 0
        idx = 0
        while pos < len(payload):
            raw, pos = decode_varint(payload, pos)
            value = raw if idx == 0 else prev + _zigzag_decode(raw)
            out.append(value)
            prev = value
            idx += 1
        arr = np.asarray(out, dtype=np.int64)
    elif mode == "zero_run_varints":
        pos = 0
        out = []
        while pos < len(payload):
            token, pos = decode_varint(payload, pos)
            if token == 0:
                run, pos = decode_varint(payload, pos)
                out.extend([0] * run)
            else:
                out.append(token - 1)
        arr = np.asarray(out, dtype=np.int64)
    elif mode == "bitpack_fixed":
        bits = max(1, math.ceil(math.log2(max(2, max_value + 1))))
        arr = unpack_fixed_width_uints(payload, bits=bits, count=count)
    elif mode == "packed_bitmask":
        arr = unpack_fixed_width_uints(payload, bits=1, count=count)
    else:
        raise ValueError(f"unsupported integer stream mode: {mode!r}")
    _validate_decoded(arr, count=count, max_value=max_value)
    return arr


def _read_envelope(blob: bytes) -> tuple[dict[str, Any], bytes]:
    if len(blob) < _HEADER.size:
        raise ValueError("integer stream envelope too short")
    magic, version, header_len = _HEADER.unpack(blob[: _HEADER.size])
    if magic != INT_STREAM_CODEC_MAGIC or version != 1:
        raise ValueError("unsupported integer stream envelope")
    header_start = _HEADER.size
    header_end = header_start + int(header_len)
    header = json.loads(blob[header_start:header_end].decode("utf-8"))
    return header, blob[header_end:]


def _as_uint_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"integer stream must be 1-D; got shape {arr.shape}")
    if arr.size and int(arr.min()) < 0:
        raise ValueError("integer stream contains negative values")
    return arr.astype(np.int64, copy=False)


def _validate_decoded(
    arr: np.ndarray,
    *,
    count: int | None,
    max_value: int | None,
) -> None:
    if count is not None and int(arr.size) != int(count):
        raise ValueError(f"decoded integer stream count {arr.size} != {count}")
    if arr.size and int(arr.min()) < 0:
        raise ValueError("decoded integer stream contains negative values")
    if max_value is not None and arr.size and int(arr.max()) > int(max_value):
        raise ValueError(
            f"decoded integer stream max {int(arr.max())} exceeds {int(max_value)}"
        )


def _zigzag_encode(value: int) -> int:
    return (int(value) << 1) ^ (int(value) >> 63)


def _zigzag_decode(value: int) -> int:
    raw = int(value)
    return (raw >> 1) ^ -(raw & 1)


__all__ = [
    "INT_STREAM_CODEC_MAGIC",
    "IntStreamCodecStats",
    "decode_uint_stream",
    "decode_varint",
    "encode_uint_stream",
    "encode_varint",
    "int_stream_codec_stats",
    "pack_fixed_width_uints",
    "unpack_fixed_width_uints",
]
