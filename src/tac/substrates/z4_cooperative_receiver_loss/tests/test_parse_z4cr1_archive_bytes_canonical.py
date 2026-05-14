# SPDX-License-Identifier: MIT
"""Canonical-surface tests for ``parse_z4cr1_archive_bytes``.

Promoted to canonical surface 2026-05-14 in lane
``lane_sister_parser_p1_wave_class_shift_5_substrates_20260514`` (Decision G
of `feedback_ibps1_parser_wave_p0_landed_20260514.md` 8-step template applied
to the 5 P1 class-shift substrates).

Z4CR1 is bytewise-compatible with sister C6 IBPS1 grammar; differentiated at
the META section level by the ``cooperative_receiver_meta`` provenance tag.
"""

from __future__ import annotations

import struct

import pytest

from tac.substrates.z4_cooperative_receiver_loss.archive import (
    Z4CR1_HEADER_FMT,
    Z4CR1_HEADER_SIZE,
    Z4CR1_MAGIC,
    Z4CR1_SCHEMA_VERSION,
    Z4CR1_SECTION_ROLES,
    parse_z4cr1_archive_bytes,
)


def _build_synthetic_z4cr1_inner(
    *,
    latent_dim: int = 24,
    num_pairs: int = 600,
    encoder_blob: bytes = b"\xaa" * 100,
    decoder_blob: bytes = b"\xbb" * 200,
    meta_blob: bytes = b'{"k":"v"}',
) -> bytes:
    """Build a synthetic well-formed Z4CR1 inner blob for parser tests."""
    latent_bytes = num_pairs * latent_dim
    latent_blob = b"\xcc" * latent_bytes
    header = struct.pack(
        Z4CR1_HEADER_FMT,
        Z4CR1_MAGIC,
        Z4CR1_SCHEMA_VERSION,
        latent_dim,
        num_pairs,
        len(encoder_blob),
        len(decoder_blob),
        latent_bytes,
        len(meta_blob),
    )
    return header + encoder_blob + decoder_blob + latent_blob + meta_blob


# --------------------------------------------------------------------------
# 1. Positive parse — section offsets match writer layout
# --------------------------------------------------------------------------


def test_parse_z4cr1_canonical_returns_five_canonical_sections() -> None:
    inner = _build_synthetic_z4cr1_inner(
        latent_dim=24,
        num_pairs=600,
        encoder_blob=b"\xaa" * 100,
        decoder_blob=b"\xbb" * 200,
        meta_blob=b'{"k":"v"}',
    )
    sections = parse_z4cr1_archive_bytes(inner)
    assert set(sections.keys()) == {
        "z4cr1_header",
        "encoder_blob",
        "decoder_blob",
        "latent_blob",
        "meta_blob",
    }
    assert sections["z4cr1_header"] == (0, Z4CR1_HEADER_SIZE)
    assert sections["encoder_blob"] == (25, 100)
    assert sections["decoder_blob"] == (125, 200)
    # latent_blob = 600 * 24 = 14400
    assert sections["latent_blob"] == (325, 14_400)
    assert sections["meta_blob"] == (14_725, 9)


def test_parse_z4cr1_canonical_sections_contiguous_cover_archive() -> None:
    inner = _build_synthetic_z4cr1_inner()
    sections = parse_z4cr1_archive_bytes(inner)
    cursor = 0
    for name in (
        "z4cr1_header",
        "encoder_blob",
        "decoder_blob",
        "latent_blob",
        "meta_blob",
    ):
        offset, length = sections[name]
        assert offset == cursor, f"section {name!r} not contiguous"
        cursor += length
    assert cursor == len(inner)


# --------------------------------------------------------------------------
# 2. Negative cases — wire-format protection
# --------------------------------------------------------------------------


def test_parse_z4cr1_canonical_rejects_short_header() -> None:
    with pytest.raises(ValueError, match="z4cr1 archive too short"):
        parse_z4cr1_archive_bytes(b"Z4CR\x01")


def test_parse_z4cr1_canonical_rejects_bad_magic() -> None:
    bad = struct.pack(Z4CR1_HEADER_FMT, b"XXXX", 1, 24, 600, 100, 100, 14_400, 10)
    bad += b"\x00" * (100 + 100 + 14_400 + 10)
    with pytest.raises(ValueError, match="bad magic"):
        parse_z4cr1_archive_bytes(bad)


def test_parse_z4cr1_canonical_rejects_unsupported_version() -> None:
    bad = struct.pack(
        Z4CR1_HEADER_FMT, Z4CR1_MAGIC, 99, 24, 600, 100, 100, 14_400, 10
    ) + b"\x00" * (100 + 100 + 14_400 + 10)
    with pytest.raises(ValueError, match="unsupported schema version"):
        parse_z4cr1_archive_bytes(bad)


def test_parse_z4cr1_canonical_rejects_latent_length_mismatch() -> None:
    bad = struct.pack(
        Z4CR1_HEADER_FMT, Z4CR1_MAGIC, 1, 24, 600, 50, 50, 12, 5
    ) + b"\x00" * (50 + 50 + 12 + 5)
    with pytest.raises(ValueError, match="latent_len"):
        parse_z4cr1_archive_bytes(bad)


def test_parse_z4cr1_canonical_rejects_truncated_archive() -> None:
    inner = _build_synthetic_z4cr1_inner()
    truncated = inner[:-100]
    with pytest.raises(ValueError, match="archive size"):
        parse_z4cr1_archive_bytes(truncated)


def test_parse_z4cr1_canonical_rejects_trailing_schema_drift() -> None:
    inner = _build_synthetic_z4cr1_inner()
    with pytest.raises(ValueError, match="archive size"):
        parse_z4cr1_archive_bytes(inner + b"tail")


# --------------------------------------------------------------------------
# 3. Section-roles map invariant
# --------------------------------------------------------------------------


def test_z4cr1_section_roles_map_covers_all_canonical_sections() -> None:
    inner = _build_synthetic_z4cr1_inner()
    sections = parse_z4cr1_archive_bytes(inner)
    assert set(Z4CR1_SECTION_ROLES.keys()) == set(sections.keys())


def test_z4cr1_section_roles_use_canonical_role_taxonomy() -> None:
    """The roles must match tac.analysis.scorer_conditional_mdl ROLE_WEIGHTS."""
    from tac.analysis.scorer_conditional_mdl import ROLE_WEIGHTS

    for section, role in Z4CR1_SECTION_ROLES.items():
        assert role in ROLE_WEIGHTS, (
            f"Z4CR1 section {section!r} role {role!r} not in canonical "
            f"ROLE_WEIGHTS taxonomy"
        )


def test_z4cr1_encoder_blob_is_training_provenance_only() -> None:
    """Z4CR1 inherits IBPS1 role taxonomy — encoder not score-affecting at inflate."""
    assert Z4CR1_SECTION_ROLES["encoder_blob"] == "training_provenance_only"
    assert Z4CR1_SECTION_ROLES["decoder_blob"] == "decoder_weight_stream"
    assert Z4CR1_SECTION_ROLES["latent_blob"] == "latent_stream"


def test_z4cr1_constants_pinned() -> None:
    assert Z4CR1_MAGIC == b"Z4CR"
    assert Z4CR1_SCHEMA_VERSION == 1
    assert Z4CR1_HEADER_FMT == "<4sBHHIIII"
    assert Z4CR1_HEADER_SIZE == 25
