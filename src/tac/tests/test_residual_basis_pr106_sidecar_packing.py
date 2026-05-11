"""Conformance tests for the PR106 + non-HNeRV residual sidecar packing grammar.

Covers ``tac.residual_basis.pr106_sidecar_packing``:

* round-trip: ``parse_archive(build_archive(...).archive_bytes) == inputs``
  for all 5 families (wavelet / cool_chic / c3 / siren / coord_mlp);
* byte-length budget enforcement;
* magic-byte mismatch refused (avoids accidental PR100 latent-sidecar consumption);
* unknown format_id refused at parse time;
* per-family ``expect_format_id`` refuses other families;
* truncated archives refused at every section boundary;
* trailing-bytes after residual refused;
* score_claim / promotion_eligible / ready_for_exact_eval_dispatch pinned False;
* deterministic byte parity: identical inputs produce identical archive bytes.
"""

from __future__ import annotations

import struct

import pytest

from tac.residual_basis.pr106_sidecar_packing import (
    PR106_RESIDUAL_FORMAT_IDS,
    PR106_RESIDUAL_MAGIC,
    ResidualArchiveError,
    build_archive,
    expect_format_id,
    parse_archive,
)


@pytest.fixture
def pr106_payload() -> bytes:
    """A plausible PR106 r2 payload size: ~178 KB monolithic 0.bin."""
    return bytes(b"\x01\x02\x03\x04" * 1024 * 32)  # 128 KB synthetic


@pytest.fixture
def residual_payload() -> bytes:
    """A typical residual blob: 4 KB."""
    return bytes(range(256)) * 16  # 4 KB synthetic


# ── Round-trip + byte parity ─────────────────────────────────────────────────


@pytest.mark.parametrize("family", sorted(PR106_RESIDUAL_FORMAT_IDS))
def test_round_trip_all_families(family: str, pr106_payload: bytes, residual_payload: bytes) -> None:
    """Every registered family round-trips byte-identical."""
    result = build_archive(family=family, pr106_bytes=pr106_payload, residual_bytes=residual_payload)
    parsed = parse_archive(result.archive_bytes)
    assert parsed.family == family
    assert parsed.format_id == PR106_RESIDUAL_FORMAT_IDS[family]
    assert parsed.pr106_bytes == pr106_payload
    assert parsed.residual_bytes == residual_payload
    assert parsed.total_bytes == len(result.archive_bytes)


def test_deterministic_archive_bytes(pr106_payload: bytes, residual_payload: bytes) -> None:
    """Identical inputs produce identical archive bytes."""
    a = build_archive(family="wavelet", pr106_bytes=pr106_payload, residual_bytes=residual_payload)
    b = build_archive(family="wavelet", pr106_bytes=pr106_payload, residual_bytes=residual_payload)
    assert a.archive_bytes == b.archive_bytes


def test_byte_lengths(pr106_payload: bytes, residual_payload: bytes) -> None:
    """Archive size = header (6B) + pr106 + residual_len_prefix (4B) + residual."""
    result = build_archive(
        family="cool_chic", pr106_bytes=pr106_payload, residual_bytes=residual_payload
    )
    assert len(result.archive_bytes) == 6 + len(pr106_payload) + 4 + len(residual_payload)
    assert result.pr106_len == len(pr106_payload)
    assert result.residual_len == len(residual_payload)


# ── Promotion-status invariants ──────────────────────────────────────────────


@pytest.mark.parametrize("family", sorted(PR106_RESIDUAL_FORMAT_IDS))
def test_promotion_invariants_frozen(family: str, pr106_payload: bytes) -> None:
    """No build_archive() result may have promotion-status fields True."""
    result = build_archive(family=family, pr106_bytes=pr106_payload, residual_bytes=b"\x00")
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.ready_for_exact_eval_dispatch is False
    assert result.evidence_grade == "research_signal"


# ── Negative cases: contract violations ──────────────────────────────────────


def test_empty_pr106_refused(residual_payload: bytes) -> None:
    with pytest.raises(ResidualArchiveError, match="pr106_bytes is empty"):
        build_archive(family="wavelet", pr106_bytes=b"", residual_bytes=residual_payload)


def test_unknown_family_refused() -> None:
    with pytest.raises(ResidualArchiveError, match="unknown family"):
        build_archive(family="nerv", pr106_bytes=b"\x01", residual_bytes=b"")


def test_non_bytes_pr106_refused() -> None:
    with pytest.raises(ResidualArchiveError, match="must be bytes"):
        build_archive(family="wavelet", pr106_bytes="not bytes", residual_bytes=b"")  # type: ignore[arg-type]


def test_non_bytes_residual_refused() -> None:
    with pytest.raises(ResidualArchiveError, match="must be bytes"):
        build_archive(family="wavelet", pr106_bytes=b"\x01", residual_bytes="not bytes")  # type: ignore[arg-type]


def test_residual_zero_bytes_ok(pr106_payload: bytes) -> None:
    """Empty residual is legal — represents the family-no-op archive."""
    result = build_archive(family="c3", pr106_bytes=pr106_payload, residual_bytes=b"")
    parsed = parse_archive(result.archive_bytes)
    assert parsed.residual_bytes == b""


def test_empty_archive_refused() -> None:
    with pytest.raises(ResidualArchiveError, match="empty archive"):
        parse_archive(b"")


def test_truncated_header_refused() -> None:
    with pytest.raises(ResidualArchiveError, match="archive too short"):
        parse_archive(b"\xFD\x10\x00")  # 3 bytes; header is 6


def test_bad_magic_refused() -> None:
    """0xFE (PR100 latent sidecar magic) must be refused."""
    bad = struct.pack("<BBI", 0xFE, 0x10, 0) + b""
    with pytest.raises(ResidualArchiveError, match="magic mismatch"):
        parse_archive(bad)


def test_unknown_format_id_refused() -> None:
    """An unknown format_id is refused at parse time."""
    blob = (
        struct.pack("<BBI", PR106_RESIDUAL_MAGIC, 0xAA, 1)
        + b"\x42"
        + struct.pack("<I", 0)
    )
    with pytest.raises(ResidualArchiveError, match="not in registered family set"):
        parse_archive(blob)


def test_truncated_pr106_payload_refused() -> None:
    """A claimed pr106_len longer than the buffer is refused."""
    blob = struct.pack("<BBI", PR106_RESIDUAL_MAGIC, 0x10, 100)
    with pytest.raises(ResidualArchiveError, match="truncated PR106 payload"):
        parse_archive(blob)


def test_truncated_residual_len_prefix_refused() -> None:
    """A buffer that ends before the residual length prefix is refused."""
    blob = struct.pack("<BBI", PR106_RESIDUAL_MAGIC, 0x10, 4) + b"\x01\x02\x03\x04"
    # No residual_len prefix bytes.
    with pytest.raises(ResidualArchiveError, match="truncated residual length prefix"):
        parse_archive(blob)


def test_truncated_residual_payload_refused() -> None:
    """A claimed residual_len longer than the buffer is refused."""
    blob = (
        struct.pack("<BBI", PR106_RESIDUAL_MAGIC, 0x10, 4)
        + b"\x01\x02\x03\x04"
        + struct.pack("<I", 100)  # claims 100B residual; provides 0
    )
    with pytest.raises(ResidualArchiveError, match="truncated residual payload"):
        parse_archive(blob)


def test_trailing_bytes_after_residual_refused() -> None:
    """Extra bytes after the residual payload are refused (deterministic parse)."""
    blob = (
        struct.pack("<BBI", PR106_RESIDUAL_MAGIC, 0x10, 4)
        + b"\x01\x02\x03\x04"
        + struct.pack("<I", 2)
        + b"\x05\x06"
        + b"\xAA"  # trailing
    )
    with pytest.raises(ResidualArchiveError, match="trailing bytes"):
        parse_archive(blob)


# ── Family-scoped expect_format_id ───────────────────────────────────────────


@pytest.mark.parametrize("family", sorted(PR106_RESIDUAL_FORMAT_IDS))
def test_expect_format_id_passes_own_family(family: str, pr106_payload: bytes) -> None:
    result = build_archive(family=family, pr106_bytes=pr106_payload, residual_bytes=b"\x00\x01")
    parsed = expect_format_id(result.archive_bytes, family=family)
    assert parsed.family == family


def test_expect_format_id_refuses_other_family(pr106_payload: bytes) -> None:
    """A wavelet archive must be refused by a cool_chic inflate scope."""
    result = build_archive(family="wavelet", pr106_bytes=pr106_payload, residual_bytes=b"\x00")
    with pytest.raises(ResidualArchiveError, match="family-scoped"):
        expect_format_id(result.archive_bytes, family="cool_chic")


def test_expect_format_id_refuses_unknown_family_argument() -> None:
    with pytest.raises(ResidualArchiveError, match="unknown family"):
        expect_format_id(b"\xFD\x10\x00\x00\x00\x00\x00\x00\x00\x00", family="nerv")


# ── No-op detector primitive: mutating a byte must change parse output ──────


def test_byte_mutation_changes_parsed_residual(
    pr106_payload: bytes, residual_payload: bytes
) -> None:
    """The byte-mutation no-op detector primitive: flipping a residual byte
    must produce a different ``parsed.residual_bytes`` value than the original."""
    result = build_archive(
        family="wavelet", pr106_bytes=pr106_payload, residual_bytes=residual_payload
    )
    blob = bytearray(result.archive_bytes)
    # Locate the first residual byte (header 6B + pr106_len + residual_len_prefix 4B).
    offset = 6 + len(pr106_payload) + 4
    original = blob[offset]
    blob[offset] = (original + 1) & 0xFF
    parsed = parse_archive(bytes(blob))
    assert parsed.residual_bytes[0] != residual_payload[0]


def test_byte_mutation_in_pr106_payload_changes_parsed_pr106(
    pr106_payload: bytes, residual_payload: bytes
) -> None:
    """Mutating a PR106 byte must change parsed.pr106_bytes."""
    result = build_archive(
        family="cool_chic", pr106_bytes=pr106_payload, residual_bytes=residual_payload
    )
    blob = bytearray(result.archive_bytes)
    offset = 6  # first PR106 byte
    blob[offset] = (blob[offset] + 1) & 0xFF
    parsed = parse_archive(bytes(blob))
    assert parsed.pr106_bytes[0] != pr106_payload[0]
