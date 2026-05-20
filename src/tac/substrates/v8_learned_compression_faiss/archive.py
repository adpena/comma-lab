# SPDX-License-Identifier: MIT
"""Minimal byte-closed V8 learned-compression/Faiss archive grammar.

This is the local conformance grammar for the first V8 runtime gate. It is a
deterministic raw-u8 frame fixture container, not a score or promotion claim.

Schema v1:

    MAGIC(8)        b"V8FAISS1"
    VERSION(2)      u16 == 1
    HEADER_LEN(2)   u16 == 56
    FRAME_COUNT(2)  u16 > 0
    HEIGHT(2)       u16 > 0
    WIDTH(2)        u16 > 0
    CHANNELS(1)     u8 in {1, 3}
    FLAGS(1)        u8 == 0
    PAYLOAD_LEN(4)  u32, must equal frames*height*width*channels
    SHA256(32)      sha256(payload)
    PAYLOAD         raw frame bytes in frame,height,width,channel order

Inflate consumes every payload byte by copying the validated raw payload to the
requested output path. Future learned V8 codecs can add new versions without
weakening this fixture contract.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

V8_MAGIC = b"V8FAISS1"
V8_ARCHIVE_VERSION = 1
V8_HEADER_FORMAT = ">8sHHHHHBBI32s"
V8_HEADER_LEN = struct.calcsize(V8_HEADER_FORMAT)
V8_ALLOWED_CHANNELS = {1, 3}
V8_MAX_DIMENSION = 8192
V8_MAX_FRAMES = 10_000
V8_RAW_FLAGS = 0

assert V8_HEADER_LEN == 56


class V8ArchiveError(ValueError):
    """Raised when a V8 archive fails closed before payload consumption."""


@dataclass(frozen=True)
class V8ArchiveHeader:
    """Parsed V8 header and non-promotable custody metadata."""

    version: int
    header_len: int
    frame_count: int
    height: int
    width: int
    channels: int
    flags: int
    payload_len: int
    payload_sha256: str

    @property
    def expected_payload_len(self) -> int:
        return self.frame_count * self.height * self.width * self.channels

    def as_dict(self) -> dict[str, object]:
        return {
            "magic": V8_MAGIC.decode("ascii"),
            "version": self.version,
            "header_len": self.header_len,
            "frame_count": self.frame_count,
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
            "flags": self.flags,
            "payload_bytes": self.payload_len,
            "payload_sha256": self.payload_sha256,
            "codec": "raw_u8_fixture_v1",
            "research_only": True,
            "dispatch_enabled": False,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_header_fields(header: V8ArchiveHeader) -> None:
    if header.version != V8_ARCHIVE_VERSION:
        raise V8ArchiveError(
            f"unsupported V8 archive version {header.version}; expected {V8_ARCHIVE_VERSION}"
        )
    if header.header_len != V8_HEADER_LEN:
        raise V8ArchiveError(
            f"V8 header length mismatch: {header.header_len} != {V8_HEADER_LEN}"
        )
    if header.frame_count <= 0 or header.frame_count > V8_MAX_FRAMES:
        raise V8ArchiveError(f"invalid V8 frame_count {header.frame_count}")
    if header.height <= 0 or header.height > V8_MAX_DIMENSION:
        raise V8ArchiveError(f"invalid V8 height {header.height}")
    if header.width <= 0 or header.width > V8_MAX_DIMENSION:
        raise V8ArchiveError(f"invalid V8 width {header.width}")
    if header.channels not in V8_ALLOWED_CHANNELS:
        raise V8ArchiveError(
            f"invalid V8 channels {header.channels}; expected one of {sorted(V8_ALLOWED_CHANNELS)}"
        )
    if header.flags != V8_RAW_FLAGS:
        raise V8ArchiveError(f"unsupported V8 flags {header.flags}; expected 0")
    if header.payload_len != header.expected_payload_len:
        raise V8ArchiveError(
            "V8 payload length does not match frame dimensions: "
            f"{header.payload_len} != {header.expected_payload_len}"
        )


def parse_v8_header(data: bytes) -> V8ArchiveHeader:
    """Parse and validate the fixed-size V8 header."""
    if len(data) < V8_HEADER_LEN:
        raise V8ArchiveError(
            f"V8 archive too short for v1 header: {len(data)} < {V8_HEADER_LEN}"
        )
    (
        magic,
        version,
        header_len,
        frame_count,
        height,
        width,
        channels,
        flags,
        payload_len,
        payload_sha256,
    ) = struct.unpack(V8_HEADER_FORMAT, data[:V8_HEADER_LEN])
    if magic != V8_MAGIC:
        raise V8ArchiveError("V8 archive magic mismatch; expected V8FAISS1")
    header = V8ArchiveHeader(
        version=version,
        header_len=header_len,
        frame_count=frame_count,
        height=height,
        width=width,
        channels=channels,
        flags=flags,
        payload_len=payload_len,
        payload_sha256=payload_sha256.hex(),
    )
    _validate_header_fields(header)
    return header


def parse_v8_archive(data: bytes) -> tuple[V8ArchiveHeader, bytes]:
    """Parse a complete V8 archive and return validated raw frame bytes."""
    header = parse_v8_header(data)
    expected_total = header.header_len + header.payload_len
    if len(data) != expected_total:
        raise V8ArchiveError(
            f"V8 archive length mismatch: {len(data)} != {expected_total}"
        )
    payload = data[header.header_len:]
    actual_sha = _sha256_hex(payload)
    if actual_sha != header.payload_sha256:
        raise V8ArchiveError(
            f"V8 payload sha256 mismatch: {actual_sha} != {header.payload_sha256}"
        )
    return header, payload


def decode_raw_frame_archive(data: bytes) -> bytes:
    """Return validated raw frame bytes from a V8 v1 archive."""
    _header, payload = parse_v8_archive(data)
    return payload


def build_raw_frame_archive(
    payload: bytes,
    *,
    frame_count: int,
    height: int,
    width: int,
    channels: int,
) -> bytes:
    """Build a deterministic v1 raw-frame fixture archive."""
    header = V8ArchiveHeader(
        version=V8_ARCHIVE_VERSION,
        header_len=V8_HEADER_LEN,
        frame_count=int(frame_count),
        height=int(height),
        width=int(width),
        channels=int(channels),
        flags=V8_RAW_FLAGS,
        payload_len=len(payload),
        payload_sha256=_sha256_hex(payload),
    )
    _validate_header_fields(header)
    packed = struct.pack(
        V8_HEADER_FORMAT,
        V8_MAGIC,
        header.version,
        header.header_len,
        header.frame_count,
        header.height,
        header.width,
        header.channels,
        header.flags,
        header.payload_len,
        bytes.fromhex(header.payload_sha256),
    )
    return packed + payload
