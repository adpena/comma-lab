# SPDX-License-Identifier: MIT
"""Canonical-surface tests for ``parse_wz1_archive_bytes``.

Promoted to canonical surface 2026-05-14 in lane
``lane_sister_parser_p1_wave_class_shift_5_substrates_20260514`` (Decision G
of `feedback_ibps1_parser_wave_p0_landed_20260514.md` 8-step template applied
to the 5 P1 class-shift substrates).

These tests verify the canonical parser stays in lock-step with the WZ1
archive writer (:func:`tac.substrates.wyner_ziv_cooperative_receiver.archive.pack_archive`)
so the section-offset contract consumed by:

- :mod:`tac.analysis.scorer_conditional_mdl`
- :mod:`tac.analysis.hnerv_packet_sections` (WZ1 auto-detection by
  ``b"WZ1\\x00"`` magic prefix)

cannot drift if the WZ1 schema ever versions.
"""

from __future__ import annotations

import struct

import pytest

from tac.substrates.wyner_ziv_cooperative_receiver.archive import (
    WZ1_HEADER_FMT,
    WZ1_HEADER_SIZE,
    WZ1_MAGIC,
    WZ1_SCHEMA_VERSION,
    WZ1_SECTION_ROLES,
    parse_wz1_archive_bytes,
)


def _build_synthetic_wz1_inner(
    *,
    num_pairs: int = 600,
    hidden_dim: int = 64,
    num_hidden_layers: int = 2,
    side_info_hidden_dim: int = 32,
    side_info_num_layers: int = 2,
    output_h: int = 384,
    output_w: int = 512,
    pose_dim: int = 6,
    coset_index_bits: int = 4,
    renderer_blob: bytes = b"\xaa" * 100,
    side_info_blob: bytes = b"\xbb" * 100,
    coset_blob: bytes = b"\xcc" * 50,
    meta_blob: bytes = b'{"k":"v"}',
) -> bytes:
    """Build a synthetic well-formed WZ1 inner blob for parser tests."""
    header = struct.pack(
        WZ1_HEADER_FMT,
        WZ1_MAGIC,
        WZ1_SCHEMA_VERSION,
        num_pairs,
        hidden_dim,
        num_hidden_layers,
        side_info_hidden_dim,
        side_info_num_layers,
        output_h,
        output_w,
        pose_dim,
        coset_index_bits,
        len(renderer_blob),
        len(side_info_blob),
        len(coset_blob),
        len(meta_blob),
    )
    return header + renderer_blob + side_info_blob + coset_blob + meta_blob


# --------------------------------------------------------------------------
# 1. Positive parse — section offsets match writer layout
# --------------------------------------------------------------------------


def test_parse_wz1_canonical_returns_five_canonical_sections() -> None:
    inner = _build_synthetic_wz1_inner()
    sections = parse_wz1_archive_bytes(inner)
    assert set(sections.keys()) == {
        "wz1_header",
        "renderer_blob",
        "side_info_predictor_blob",
        "coset_indices_blob",
        "meta_blob",
    }
    assert sections["wz1_header"] == (0, WZ1_HEADER_SIZE)
    assert sections["renderer_blob"] == (35, 100)
    assert sections["side_info_predictor_blob"] == (135, 100)
    assert sections["coset_indices_blob"] == (235, 50)
    assert sections["meta_blob"] == (285, 9)


def test_parse_wz1_canonical_sections_contiguous_cover_archive() -> None:
    inner = _build_synthetic_wz1_inner()
    sections = parse_wz1_archive_bytes(inner)
    cursor = 0
    for name in (
        "wz1_header",
        "renderer_blob",
        "side_info_predictor_blob",
        "coset_indices_blob",
        "meta_blob",
    ):
        offset, length = sections[name]
        assert offset == cursor, f"section {name!r} not contiguous"
        cursor += length
    assert cursor == len(inner)


# --------------------------------------------------------------------------
# 2. Negative cases — wire-format protection
# --------------------------------------------------------------------------


def test_parse_wz1_canonical_rejects_short_header() -> None:
    with pytest.raises(ValueError, match="wz1 archive too short"):
        parse_wz1_archive_bytes(b"WZ1\x00\x01")


def test_parse_wz1_canonical_rejects_bad_magic() -> None:
    bad = struct.pack(
        WZ1_HEADER_FMT, b"XXXX", 1, 600, 64, 2, 32, 2, 384, 512, 6, 4, 100, 100, 50, 10
    ) + b"\x00" * (100 + 100 + 50 + 10)
    with pytest.raises(ValueError, match="bad magic"):
        parse_wz1_archive_bytes(bad)


def test_parse_wz1_canonical_rejects_unsupported_version() -> None:
    bad = struct.pack(
        WZ1_HEADER_FMT, WZ1_MAGIC, 99, 600, 64, 2, 32, 2, 384, 512, 6, 4, 100, 100, 50, 10
    ) + b"\x00" * (100 + 100 + 50 + 10)
    with pytest.raises(ValueError, match="unsupported schema version"):
        parse_wz1_archive_bytes(bad)


def test_parse_wz1_canonical_rejects_truncated_archive() -> None:
    inner = _build_synthetic_wz1_inner()
    truncated = inner[:-30]
    with pytest.raises(ValueError, match="archive size"):
        parse_wz1_archive_bytes(truncated)


def test_parse_wz1_canonical_rejects_trailing_schema_drift() -> None:
    inner = _build_synthetic_wz1_inner()
    with pytest.raises(ValueError, match="archive size"):
        parse_wz1_archive_bytes(inner + b"tail")


# --------------------------------------------------------------------------
# 3. Section-roles map invariant
# --------------------------------------------------------------------------


def test_wz1_section_roles_map_covers_all_canonical_sections() -> None:
    inner = _build_synthetic_wz1_inner()
    sections = parse_wz1_archive_bytes(inner)
    assert set(WZ1_SECTION_ROLES.keys()) == set(sections.keys())


def test_wz1_section_roles_use_canonical_role_taxonomy() -> None:
    """The roles must match tac.analysis.scorer_conditional_mdl ROLE_WEIGHTS."""
    from tac.analysis.scorer_conditional_mdl import ROLE_WEIGHTS

    for section, role in WZ1_SECTION_ROLES.items():
        assert role in ROLE_WEIGHTS, (
            f"WZ1 section {section!r} role {role!r} not in canonical "
            f"ROLE_WEIGHTS taxonomy"
        )


def test_wz1_coset_indices_blob_is_entropy_model_role() -> None:
    """Coset coding IS the Wyner-Ziv rate-distortion-with-side-info entropy primitive."""
    assert WZ1_SECTION_ROLES["coset_indices_blob"] == "entropy_model_or_range_stream"
    assert WZ1_SECTION_ROLES["renderer_blob"] == "decoder_weight_stream"
    assert WZ1_SECTION_ROLES["side_info_predictor_blob"] == "sidecar_or_correction_stream"


def test_wz1_constants_pinned() -> None:
    assert WZ1_MAGIC == b"WZ1\x00"
    assert WZ1_SCHEMA_VERSION == 1
    assert WZ1_HEADER_FMT == "<4sBHHBHBHHBBIIII"
    assert WZ1_HEADER_SIZE == 35
