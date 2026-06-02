# SPDX-License-Identifier: MIT
"""Lossless receiver-visible SNeRV LF payload codec portfolio.

The legacy SNAR1 LF section stores signed LF coefficient planes as raw int64
followed by XZ.  This module keeps the signal identical, but gives the receiver
smaller integer-stream grammars that match what the planes actually need:
delta-varints, zero-run varints, and exact signed int2/int4/int8 bit-packing.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import struct
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tac.substrates._shared.int_stream_codec import (
    decode_varint,
    encode_varint,
    pack_fixed_width_uints,
    unpack_fixed_width_uints,
)

try:  # pragma: no cover - absence is exercised by wrapper filtering.
    import brotli
except Exception:  # pragma: no cover
    brotli = None

SNERV_LF_QUANT_V2_MAGIC = b"SQL2"
SNERV_LF_QUANT_V2_SCHEMA = "snerv_lf_quant_payload.v2"
SNERV_LF_PAYLOAD_INTN_CODEC_PROOF = (
    "snerv_lf_quant_payload.v2_receiver_visible_exact_intn_codec"
)
_HEADER = struct.Struct("<4sBI")
_SUPPORTED_MODES = (
    "raw_i64",
    "zigzag_delta_varint",
    "zero_run_varint",
    "signed_int2_bitpack",
    "signed_int4_bitpack",
    "signed_int8_bitpack",
)
_SUPPORTED_WRAPPERS = ("none", "brotli", "lzma")


class SnervLfPayloadCodecError(ValueError):
    """Raised when an LF payload cannot be encoded or decoded exactly."""


@dataclass(frozen=True)
class LfPlaneCodecRow:
    """Byte accounting and custody metadata for one encoded LF plane."""

    plane_index: int
    shape: tuple[int, ...]
    mode: str
    wrapper: str
    raw_i64_bytes: int
    payload_bytes: int
    wrapped_payload_bytes: int
    decoded_sha256: str

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LfPayloadCodecReport:
    """Packet-level report for a v2 LF payload."""

    schema: str
    plane_count: int
    packet_bytes: int
    raw_i64_bytes: int
    payload_bytes: int
    mode_histogram: dict[str, int]
    wrapper_histogram: dict[str, int]
    plane_rows: tuple[LfPlaneCodecRow, ...]
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def as_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["plane_rows"] = [row.as_jsonable() for row in self.plane_rows]
        return d


def encode_lf_quant_payload_v2(
    lf_quant_planes: Sequence[np.ndarray],
    *,
    mode: str = "portfolio_auto",
    wrapper: str = "portfolio_auto",
) -> bytes:
    """Encode signed LF planes losslessly with a measured integer portfolio."""

    packet, _report = encode_lf_quant_payload_v2_with_report(
        lf_quant_planes,
        mode=mode,
        wrapper=wrapper,
    )
    return packet


def encode_lf_quant_payload_v2_with_report(
    lf_quant_planes: Sequence[np.ndarray],
    *,
    mode: str = "portfolio_auto",
    wrapper: str = "portfolio_auto",
) -> tuple[bytes, LfPayloadCodecReport]:
    """Encode and return packet plus byte-accounting report."""

    planes = [_validate_lf_plane(plane) for plane in lf_quant_planes]
    if not planes:
        raise SnervLfPayloadCodecError("lf_quant_planes must be non-empty")

    plane_headers: list[dict[str, Any]] = []
    plane_rows: list[LfPlaneCodecRow] = []
    payload = bytearray()
    for idx, plane in enumerate(planes):
        encoded = _best_plane_encoding(
            plane.reshape(-1),
            mode=_normalize_mode(mode),
            wrapper=_normalize_wrapper(wrapper),
        )
        decoded = _decode_plane_payload(
            encoded["payload"],
            mode=encoded["mode"],
            wrapper=encoded["wrapper"],
            count=int(plane.size),
        ).reshape(plane.shape)
        if not np.array_equal(decoded, plane):
            raise SnervLfPayloadCodecError("internal LF codec roundtrip mismatch")
        offset = len(payload)
        payload.extend(encoded["payload"])
        raw = plane.astype("<i8", copy=False).reshape(-1).tobytes()
        decoded_sha = _sha256(raw)
        plane_headers.append(
            {
                "shape": list(plane.shape),
                "mode": encoded["mode"],
                "wrapper": encoded["wrapper"],
                "payload_offset": offset,
                "payload_bytes": len(encoded["payload"]),
                "raw_i64_bytes": len(raw),
                "decoded_sha256": decoded_sha,
            }
        )
        plane_rows.append(
            LfPlaneCodecRow(
                plane_index=idx,
                shape=tuple(int(v) for v in plane.shape),
                mode=encoded["mode"],
                wrapper=encoded["wrapper"],
                raw_i64_bytes=len(raw),
                payload_bytes=int(encoded["payload_unwrapped_bytes"]),
                wrapped_payload_bytes=len(encoded["payload"]),
                decoded_sha256=decoded_sha,
            )
        )

    header = {
        "schema": SNERV_LF_QUANT_V2_SCHEMA,
        "proof": SNERV_LF_PAYLOAD_INTN_CODEC_PROOF,
        "plane_count": len(planes),
        "dtype": "int64_logical",
        "planes": plane_headers,
        "payload_bytes": len(payload),
        "payload_sha256": _sha256(bytes(payload)),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    packet = _HEADER.pack(SNERV_LF_QUANT_V2_MAGIC, 1, len(header_bytes))
    packet += header_bytes + bytes(payload)
    report = _build_report(packet, plane_rows, payload_bytes=len(payload))
    return packet, report


def decode_lf_quant_payload_v2(payload: bytes) -> list[np.ndarray]:
    """Decode a v2 LF payload into signed int64 planes."""

    header, body = _read_packet(payload)
    planes = []
    ranges: list[tuple[int, int, int]] = []
    for plane_idx, plane in enumerate(header["planes"]):
        shape = tuple(int(v) for v in plane["shape"])
        count = int(np.prod(shape))
        start = int(plane["payload_offset"])
        end = start + int(plane["payload_bytes"])
        if start < 0 or end > len(body) or end < start:
            raise SnervLfPayloadCodecError("LF v2 plane payload bounds invalid")
        ranges.append((start, end, plane_idx))
        arr = _decode_plane_payload(
            body[start:end],
            mode=str(plane["mode"]),
            wrapper=str(plane["wrapper"]),
            count=count,
        ).reshape(shape)
        raw = arr.astype("<i8", copy=False).reshape(-1).tobytes()
        if _sha256(raw) != str(plane["decoded_sha256"]):
            raise SnervLfPayloadCodecError("LF v2 plane decoded sha256 mismatch")
        planes.append(arr)
    _validate_payload_coverage(ranges, payload_len=len(body))
    return planes


def inspect_lf_quant_payload_v2(payload: bytes) -> LfPayloadCodecReport:
    """Return byte-accounting metadata without trusting it as score authority."""

    header, body = _read_packet(payload)
    rows = []
    for idx, plane in enumerate(header["planes"]):
        rows.append(
            LfPlaneCodecRow(
                plane_index=idx,
                shape=tuple(int(v) for v in plane["shape"]),
                mode=str(plane["mode"]),
                wrapper=str(plane["wrapper"]),
                raw_i64_bytes=int(plane["raw_i64_bytes"]),
                payload_bytes=-1,
                wrapped_payload_bytes=int(plane["payload_bytes"]),
                decoded_sha256=str(plane["decoded_sha256"]),
            )
        )
    return _build_report(payload, rows, payload_bytes=len(body))


def is_lf_quant_payload_v2(payload: bytes) -> bool:
    """Return true if ``payload`` starts with the v2 LF codec envelope."""

    return bytes(payload).startswith(SNERV_LF_QUANT_V2_MAGIC)


def _best_plane_encoding(
    arr: np.ndarray,
    *,
    mode: tuple[str, ...],
    wrapper: tuple[str, ...],
) -> dict[str, Any]:
    candidates = []
    last_error: SnervLfPayloadCodecError | None = None
    for candidate_mode in mode:
        try:
            encoded = _encode_plane_unwrapped(arr, mode=candidate_mode)
        except SnervLfPayloadCodecError as exc:
            last_error = exc
            continue
        for candidate_wrapper in wrapper:
            try:
                wrapped = _wrap_payload(encoded, wrapper=candidate_wrapper)
            except SnervLfPayloadCodecError as exc:
                last_error = exc
                continue
            candidates.append(
                {
                    "mode": candidate_mode,
                    "wrapper": candidate_wrapper,
                    "payload": wrapped,
                    "payload_unwrapped_bytes": len(encoded),
                }
            )
    if not candidates:
        if last_error is not None:
            raise last_error
        raise SnervLfPayloadCodecError("no legal LF payload codec candidate")
    return min(candidates, key=lambda item: (len(item["payload"]), item["mode"], item["wrapper"]))


def _encode_plane_unwrapped(arr: np.ndarray, *, mode: str) -> bytes:
    if mode == "raw_i64":
        return arr.astype("<i8", copy=False).tobytes()
    if mode == "zigzag_delta_varint":
        out = bytearray()
        prev = 0
        for raw in arr.tolist():
            value = int(raw)
            out.extend(encode_varint(_zigzag_encode(value - prev)))
            prev = value
        return bytes(out)
    if mode == "zero_run_varint":
        out = bytearray()
        values = arr.tolist()
        idx = 0
        while idx < len(values):
            value = int(values[idx])
            if value == 0:
                run = 1
                while idx + run < len(values) and int(values[idx + run]) == 0:
                    run += 1
                out.extend(encode_varint(0))
                out.extend(encode_varint(run))
                idx += run
                continue
            out.extend(encode_varint(_zigzag_encode(value) + 1))
            idx += 1
        return bytes(out)
    bits = _signed_bitpack_bits(mode)
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    if arr.size and (int(arr.min()) < qmin or int(arr.max()) > qmax):
        raise SnervLfPayloadCodecError(
            f"{mode} requires values in [{qmin}, {qmax}]"
        )
    return pack_fixed_width_uints(arr.astype(np.int64) - qmin, bits=bits)


def _decode_plane_unwrapped(blob: bytes, *, mode: str, count: int) -> np.ndarray:
    if mode == "raw_i64":
        arr = np.frombuffer(blob, dtype="<i8").astype(np.int64)
    elif mode == "zigzag_delta_varint":
        out = []
        pos = 0
        prev = 0
        while pos < len(blob):
            token, pos = decode_varint(blob, pos)
            value = prev + _zigzag_decode(token)
            out.append(value)
            prev = value
        arr = np.asarray(out, dtype=np.int64)
    elif mode == "zero_run_varint":
        out = []
        pos = 0
        while pos < len(blob):
            token, pos = decode_varint(blob, pos)
            if token == 0:
                run, pos = decode_varint(blob, pos)
                out.extend([0] * run)
            else:
                out.append(_zigzag_decode(token - 1))
        arr = np.asarray(out, dtype=np.int64)
    else:
        bits = _signed_bitpack_bits(mode)
        qmin = -(1 << (bits - 1))
        arr = unpack_fixed_width_uints(blob, bits=bits, count=count).astype(np.int64)
        arr += qmin
    if int(arr.size) != int(count):
        raise SnervLfPayloadCodecError(
            f"LF v2 decoded count {arr.size} != expected {count}"
        )
    return arr


def _wrap_payload(payload: bytes, *, wrapper: str) -> bytes:
    if wrapper == "none":
        return payload
    if wrapper == "brotli":
        if brotli is None:
            raise SnervLfPayloadCodecError("brotli unavailable")
        return brotli.compress(payload, quality=11)
    if wrapper == "lzma":
        return lzma.compress(payload, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    raise SnervLfPayloadCodecError(f"unsupported LF wrapper: {wrapper!r}")


def _unwrap_payload(payload: bytes, *, wrapper: str) -> bytes:
    if wrapper == "none":
        return payload
    if wrapper == "brotli":
        if brotli is None:
            raise SnervLfPayloadCodecError("brotli unavailable")
        return brotli.decompress(payload)
    if wrapper == "lzma":
        return lzma.decompress(payload)
    raise SnervLfPayloadCodecError(f"unsupported LF wrapper: {wrapper!r}")


def _decode_plane_payload(
    payload: bytes,
    *,
    mode: str,
    wrapper: str,
    count: int,
) -> np.ndarray:
    raw = _unwrap_payload(payload, wrapper=wrapper)
    return _decode_plane_unwrapped(raw, mode=mode, count=count)


def _read_packet(packet: bytes) -> tuple[dict[str, Any], bytes]:
    blob = bytes(packet)
    if len(blob) < _HEADER.size:
        raise SnervLfPayloadCodecError("LF v2 payload too short")
    magic, version, header_len = _HEADER.unpack(blob[: _HEADER.size])
    if magic != SNERV_LF_QUANT_V2_MAGIC or version != 1:
        raise SnervLfPayloadCodecError("unsupported LF v2 payload envelope")
    header_start = _HEADER.size
    header_end = header_start + int(header_len)
    if header_end > len(blob):
        raise SnervLfPayloadCodecError("LF v2 header exceeds packet")
    header = json.loads(blob[header_start:header_end].decode("utf-8"))
    if header.get("schema") != SNERV_LF_QUANT_V2_SCHEMA:
        raise SnervLfPayloadCodecError("unsupported LF v2 schema")
    body = blob[header_end:]
    if int(header.get("payload_bytes", -1)) != len(body):
        raise SnervLfPayloadCodecError("LF v2 payload byte count mismatch")
    if _sha256(body) != str(header.get("payload_sha256")):
        raise SnervLfPayloadCodecError("LF v2 payload sha256 mismatch")
    return header, body


def _build_report(
    packet: bytes,
    rows: Sequence[LfPlaneCodecRow],
    *,
    payload_bytes: int,
) -> LfPayloadCodecReport:
    mode_histogram = _histogram(row.mode for row in rows)
    wrapper_histogram = _histogram(row.wrapper for row in rows)
    return LfPayloadCodecReport(
        schema=SNERV_LF_QUANT_V2_SCHEMA,
        plane_count=len(rows),
        packet_bytes=len(packet),
        raw_i64_bytes=sum(row.raw_i64_bytes for row in rows),
        payload_bytes=int(payload_bytes),
        mode_histogram=mode_histogram,
        wrapper_histogram=wrapper_histogram,
        plane_rows=tuple(rows),
    )


def _histogram(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _validate_lf_plane(plane: np.ndarray) -> np.ndarray:
    arr = np.asarray(plane)
    if arr.ndim != 2:
        raise SnervLfPayloadCodecError(f"LF plane must be 2-D; got shape {arr.shape}")
    if arr.size == 0:
        raise SnervLfPayloadCodecError("LF plane must be non-empty")
    if not np.issubdtype(arr.dtype, np.integer):
        raise SnervLfPayloadCodecError("LF plane must contain integers")
    return arr.astype(np.int64, copy=False)


def _normalize_mode(mode: str) -> tuple[str, ...]:
    normalized = str(mode).strip().lower()
    if normalized in {"auto", "portfolio", "portfolio_auto"}:
        return _SUPPORTED_MODES
    aliases = {
        "raw": "raw_i64",
        "delta_varint": "zigzag_delta_varint",
        "zero_run": "zero_run_varint",
        "int2": "signed_int2_bitpack",
        "int4": "signed_int4_bitpack",
        "int8": "signed_int8_bitpack",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _SUPPORTED_MODES:
        raise SnervLfPayloadCodecError(f"unsupported LF mode: {mode!r}")
    return (normalized,)


def _normalize_wrapper(wrapper: str) -> tuple[str, ...]:
    normalized = str(wrapper).strip().lower()
    if normalized in {"auto", "portfolio", "portfolio_auto"}:
        if brotli is None:
            return ("none", "lzma")
        return _SUPPORTED_WRAPPERS
    if normalized not in _SUPPORTED_WRAPPERS:
        raise SnervLfPayloadCodecError(f"unsupported LF wrapper: {wrapper!r}")
    if normalized == "brotli" and brotli is None:
        raise SnervLfPayloadCodecError("brotli unavailable")
    return (normalized,)


def _signed_bitpack_bits(mode: str) -> int:
    if mode == "signed_int2_bitpack":
        return 2
    if mode == "signed_int4_bitpack":
        return 4
    if mode == "signed_int8_bitpack":
        return 8
    raise SnervLfPayloadCodecError(f"not a signed bitpack mode: {mode!r}")


def _validate_payload_coverage(
    ranges: Sequence[tuple[int, int, int]],
    *,
    payload_len: int,
) -> None:
    cursor = 0
    for start, end, idx in sorted(ranges):
        if start != cursor:
            raise SnervLfPayloadCodecError(
                f"LF v2 payload range for plane {idx} starts at {start}, expected {cursor}"
            )
        if end <= start:
            raise SnervLfPayloadCodecError(f"LF v2 payload range for plane {idx} is empty")
        cursor = end
    if cursor != int(payload_len):
        raise SnervLfPayloadCodecError(
            f"LF v2 payload consumed {cursor} bytes, expected {payload_len}"
        )


def _zigzag_encode(value: int) -> int:
    return (int(value) << 1) ^ (int(value) >> 63)


def _zigzag_decode(value: int) -> int:
    raw = int(value)
    return (raw >> 1) ^ -(raw & 1)


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


__all__ = [
    "SNERV_LF_PAYLOAD_INTN_CODEC_PROOF",
    "SNERV_LF_QUANT_V2_MAGIC",
    "SNERV_LF_QUANT_V2_SCHEMA",
    "LfPayloadCodecReport",
    "LfPlaneCodecRow",
    "SnervLfPayloadCodecError",
    "decode_lf_quant_payload_v2",
    "encode_lf_quant_payload_v2",
    "encode_lf_quant_payload_v2_with_report",
    "inspect_lf_quant_payload_v2",
    "is_lf_quant_payload_v2",
]
