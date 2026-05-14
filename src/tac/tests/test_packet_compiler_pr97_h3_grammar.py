# SPDX-License-Identifier: MIT
"""Tests for the PR97 H3 wire-format grammar primitives.

Covers round-trip identity, golden-vector pinning, edge cases (empty
sections, trailing-bytes detection), and failure modes (truncated headers,
chunk-count mismatch, max-section overflow).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tac.packet_compiler.pr97_h3_grammar import (
    LengthPrefixedSectionPayload,
    TileBandStreamPayload,
    decode_length_prefixed_sections,
    decode_tile_band_streams,
    encode_length_prefixed_sections,
    encode_tile_band_streams,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


# ── Length-prefixed multi-section payload ─────────────────────────────────


def test_pr97_length_prefixed_roundtrips_pr97_typical_4_sections() -> None:
    """PR97 packs mask/pose/model/sidecar = 4 sections."""
    sections = [
        b"mask-bytes-go-here" * 8,
        b"pose-bytes" * 16,
        b"model-bytes" * 32,
        b"sidecar-bytes" * 4,
    ]
    blob = encode_length_prefixed_sections(sections)
    decoded = decode_length_prefixed_sections(blob, min_sections=3)
    assert decoded.sections == tuple(sections)
    assert decoded.total_bytes == len(blob)


def test_pr97_length_prefixed_optional_sidecar_absent() -> None:
    """PR97 tolerates missing sidecar slot (3-section payload)."""
    sections = [b"mask", b"pose", b"model"]
    blob = encode_length_prefixed_sections(sections)
    decoded = decode_length_prefixed_sections(blob, min_sections=3)
    assert decoded.sections == tuple(sections)


def test_pr97_length_prefixed_max_sections_caps_count() -> None:
    """When max_sections cap is hit, extra trailing sections cause trailing-byte rejection."""
    sections = [b"a", b"b", b"c", b"d"]
    blob = encode_length_prefixed_sections(sections)
    with pytest.raises(ValueError, match="trailing"):
        decode_length_prefixed_sections(blob, min_sections=2, max_sections=2)


def test_pr97_length_prefixed_max_sections_caps_count_exact() -> None:
    """When max_sections equals actual count, no trailing bytes — clean decode."""
    sections = [b"a", b"b"]
    blob = encode_length_prefixed_sections(sections)
    decoded = decode_length_prefixed_sections(blob, min_sections=2, max_sections=2)
    assert decoded.sections == tuple(sections)


def test_pr97_length_prefixed_empty_section_allowed() -> None:
    """Zero-length sections are valid (e.g. PR97 with an empty optional sidecar)."""
    sections = [b"mask", b"", b"model"]
    blob = encode_length_prefixed_sections(sections)
    decoded = decode_length_prefixed_sections(blob, min_sections=3)
    assert decoded.sections == tuple(sections)


def test_pr97_length_prefixed_below_min_raises() -> None:
    sections = [b"only-one"]
    blob = encode_length_prefixed_sections(sections)
    with pytest.raises(ValueError, match="required min"):
        decode_length_prefixed_sections(blob, min_sections=2)


def test_pr97_length_prefixed_truncated_header_raises() -> None:
    bad = b"\x10\x00"  # 2 bytes — can't fit the 4-byte length prefix
    with pytest.raises(ValueError, match="truncated length prefix"):
        decode_length_prefixed_sections(bad, min_sections=1)


def test_pr97_length_prefixed_truncated_body_raises() -> None:
    import struct

    bad = struct.pack("<I", 100) + b"short"  # claims 100 bytes, has 5
    with pytest.raises(ValueError, match="truncated section"):
        decode_length_prefixed_sections(bad, min_sections=1)


def test_pr97_length_prefixed_max_min_invariant() -> None:
    with pytest.raises(ValueError, match="cannot be less than min_sections"):
        decode_length_prefixed_sections(b"", min_sections=3, max_sections=2)


def test_pr97_length_prefixed_rejects_non_bytes() -> None:
    with pytest.raises(TypeError):
        encode_length_prefixed_sections(["not-bytes"])  # type: ignore[list-item]


def test_pr97_length_prefixed_bytearray_input_works() -> None:
    """Encoder must accept bytes-like (bytearray, memoryview) per spec."""
    sections = [bytearray(b"mask"), memoryview(b"pose"), b"model"]
    blob = encode_length_prefixed_sections(sections)
    decoded = decode_length_prefixed_sections(blob, min_sections=3)
    assert decoded.sections == (b"mask", b"pose", b"model")


# ── Tile-band multi-stream wire format ────────────────────────────────────


def test_pr97_tile_band_roundtrips_22_chunk_pr97_layout() -> None:
    """PR97 uses 4 bands × per-band W-splits = 3+8+8+3 = 22 chunks."""
    streams = [b"chunk-%d-data" % i for i in range(22)]
    blob = encode_tile_band_streams(streams)
    decoded = decode_tile_band_streams(blob, expected_n_chunks=22)
    assert decoded.n_chunks == 22
    assert decoded.streams == tuple(streams)


def test_pr97_tile_band_roundtrips_without_expected_count() -> None:
    streams = [b"a", b"b", b"c"]
    blob = encode_tile_band_streams(streams)
    decoded = decode_tile_band_streams(blob)
    assert decoded.n_chunks == 3
    assert decoded.streams == tuple(streams)


def test_pr97_tile_band_chunk_count_mismatch_raises() -> None:
    streams = [b"a", b"b"]
    blob = encode_tile_band_streams(streams)
    with pytest.raises(ValueError, match="chunk count mismatch"):
        decode_tile_band_streams(blob, expected_n_chunks=3)


def test_pr97_tile_band_zero_chunks_roundtrips() -> None:
    """Empty stream list should encode/decode cleanly."""
    blob = encode_tile_band_streams([])
    decoded = decode_tile_band_streams(blob, expected_n_chunks=0)
    assert decoded.n_chunks == 0
    assert decoded.streams == ()


def test_pr97_tile_band_truncated_header_raises() -> None:
    with pytest.raises(ValueError, match="truncated header"):
        decode_tile_band_streams(b"\x10")


def test_pr97_tile_band_truncated_stream_body_raises() -> None:
    import struct

    bad = struct.pack("<I", 1) + struct.pack("<I", 100) + b"short"
    with pytest.raises(ValueError, match="truncated stream-0 body"):
        decode_tile_band_streams(bad)


def test_pr97_tile_band_truncated_stream_header_raises() -> None:
    import struct

    bad = struct.pack("<I", 2) + struct.pack("<I", 0) + b"\x10"  # only 1 byte after 1st stream
    with pytest.raises(ValueError, match="truncated stream-1 length prefix"):
        decode_tile_band_streams(bad)


def test_pr97_tile_band_trailing_bytes_raises() -> None:
    streams = [b"a"]
    blob = encode_tile_band_streams(streams) + b"extra"
    with pytest.raises(ValueError, match="trailing bytes"):
        decode_tile_band_streams(blob)


def test_pr97_tile_band_rejects_non_bytes() -> None:
    with pytest.raises(TypeError):
        encode_tile_band_streams([42])  # type: ignore[list-item]


def test_pr97_tile_band_empty_stream_in_list_allowed() -> None:
    streams = [b"a", b"", b"c"]
    blob = encode_tile_band_streams(streams)
    decoded = decode_tile_band_streams(blob)
    assert decoded.streams == tuple(streams)


# ── Frozen dataclass discipline ───────────────────────────────────────────


def test_pr97_length_prefixed_payload_is_frozen() -> None:
    p = LengthPrefixedSectionPayload(sections=(b"x",), total_bytes=5)
    with pytest.raises(Exception):
        p.total_bytes = 99  # type: ignore[misc]


def test_pr97_tile_band_payload_is_frozen() -> None:
    p = TileBandStreamPayload(streams=(b"x",), n_chunks=1, total_bytes=9)
    with pytest.raises(Exception):
        p.n_chunks = 99  # type: ignore[misc]


# ── Golden vectors (SHA-pinned) ───────────────────────────────────────────


def test_pr97_length_prefixed_golden_vector_pins_sha() -> None:
    """SHA-pinned conformance for native ports."""
    sections = [
        b"\x00" * 16,
        b"\x01" * 32,
        b"\x02" * 8,
        b"\x03" * 4,
    ]
    blob = encode_length_prefixed_sections(sections)
    sha = hashlib.sha256(blob).hexdigest()
    golden_path = GOLDEN_DIR / "pr97_h3_length_prefixed_sections_v1.json"
    golden = json.loads(golden_path.read_text())
    assert sha == golden["sha256"]
    assert len(blob) == golden["payload_len"]
    assert golden["n_sections"] == len(sections)


def test_pr97_tile_band_golden_vector_pins_sha() -> None:
    streams = [bytes([i]) * (i + 1) for i in range(22)]
    blob = encode_tile_band_streams(streams)
    sha = hashlib.sha256(blob).hexdigest()
    golden_path = GOLDEN_DIR / "pr97_h3_tile_band_streams_v1.json"
    golden = json.loads(golden_path.read_text())
    assert sha == golden["sha256"]
    assert len(blob) == golden["payload_len"]
    assert golden["n_chunks"] == 22
