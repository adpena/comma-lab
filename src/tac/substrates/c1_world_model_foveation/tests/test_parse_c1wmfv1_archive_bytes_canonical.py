# SPDX-License-Identifier: MIT
"""Canonical-surface tests for ``parse_c1wmfv1_archive_bytes``.

Promoted to canonical surface 2026-05-14 in lane
``lane_sister_parser_p1_wave_class_shift_5_substrates_20260514`` (Decision G
of `feedback_ibps1_parser_wave_p0_landed_20260514.md` 8-step template applied
to the 5 P1 class-shift substrates).

These tests verify the canonical parser stays in lock-step with the C1WMFV1
archive writer (:func:`tac.substrates.c1_world_model_foveation.archive.pack_archive`)
so the section-offset contract consumed by:

- :mod:`tac.analysis.scorer_conditional_mdl` (ScorerConditionalMDLEstimator
  section-aware Tier A density estimation)
- :mod:`tac.analysis.hnerv_packet_sections` (parser-section manifest C1WMFV1
  auto-detection by ``b"WMF\\x01"`` magic prefix)

cannot drift if the C1WMFV1 schema ever versions.
"""

from __future__ import annotations

import struct

import pytest

from tac.substrates.c1_world_model_foveation.archive import (
    C1WMFV1_HEADER_FMT,
    C1WMFV1_HEADER_SIZE,
    C1WMFV1_MAGIC,
    C1WMFV1_SCHEMA_VERSION,
    C1WMFV1_SECTION_ROLES,
    parse_c1wmfv1_archive_bytes,
)


def _build_synthetic_c1wmfv1_inner(
    *,
    num_pairs: int = 600,
    recurrence_mode: int = 0,
    foveation_strategy: int = 1,
    latent_dim: int = 64,
    output_h: int = 384,
    output_w: int = 512,
    wm_blob: bytes = b"\xaa" * 100,
    decoder_blob: bytes = b"\xbb" * 200,
    z_init_blob: bytes = b"\xcc" * 50,
    fov_meta_blob: bytes = b"\xdd" * 30,
    residual_blob: bytes = b"\xee" * 120,
    meta_blob: bytes = b'{"k":"v"}',
) -> bytes:
    """Build a synthetic well-formed C1WMFV1 inner blob for parser tests."""
    header = struct.pack(
        C1WMFV1_HEADER_FMT,
        C1WMFV1_MAGIC,
        C1WMFV1_SCHEMA_VERSION,
        num_pairs,
        recurrence_mode,
        foveation_strategy,
        latent_dim,
        output_h,
        output_w,
        len(wm_blob),
        len(decoder_blob),
        len(z_init_blob),
        len(fov_meta_blob),
        len(residual_blob),
        len(meta_blob),
    )
    return (
        header
        + wm_blob
        + decoder_blob
        + z_init_blob
        + fov_meta_blob
        + residual_blob
        + meta_blob
    )


# --------------------------------------------------------------------------
# 1. Positive parse — section offsets match writer layout
# --------------------------------------------------------------------------


def test_parse_c1wmfv1_canonical_returns_seven_canonical_sections() -> None:
    inner = _build_synthetic_c1wmfv1_inner()
    sections = parse_c1wmfv1_archive_bytes(inner)
    assert set(sections.keys()) == {
        "c1wmfv1_header",
        "world_model_blob",
        "decoder_blob",
        "z_init_blob",
        "foveation_meta_blob",
        "residual_blob",
        "meta_blob",
    }
    assert sections["c1wmfv1_header"] == (0, C1WMFV1_HEADER_SIZE)
    assert sections["world_model_blob"] == (39, 100)
    assert sections["decoder_blob"] == (139, 200)
    assert sections["z_init_blob"] == (339, 50)
    assert sections["foveation_meta_blob"] == (389, 30)
    assert sections["residual_blob"] == (419, 120)
    assert sections["meta_blob"] == (539, 9)


def test_parse_c1wmfv1_canonical_sections_contiguous_cover_archive() -> None:
    inner = _build_synthetic_c1wmfv1_inner()
    sections = parse_c1wmfv1_archive_bytes(inner)
    cursor = 0
    for name in (
        "c1wmfv1_header",
        "world_model_blob",
        "decoder_blob",
        "z_init_blob",
        "foveation_meta_blob",
        "residual_blob",
        "meta_blob",
    ):
        offset, length = sections[name]
        assert offset == cursor, f"section {name!r} not contiguous"
        cursor += length
    assert cursor == len(inner)


# --------------------------------------------------------------------------
# 2. Negative cases — wire-format protection
# --------------------------------------------------------------------------


def test_parse_c1wmfv1_canonical_rejects_short_header() -> None:
    with pytest.raises(ValueError, match="c1wmfv1 archive too short"):
        parse_c1wmfv1_archive_bytes(b"WMF\x01\x01")


def test_parse_c1wmfv1_canonical_rejects_bad_magic() -> None:
    bad = struct.pack(
        C1WMFV1_HEADER_FMT, b"XXXX", 1, 600, 0, 1, 64, 384, 512, 100, 200, 50, 30, 120, 10
    ) + b"\x00" * (100 + 200 + 50 + 30 + 120 + 10)
    with pytest.raises(ValueError, match="bad magic"):
        parse_c1wmfv1_archive_bytes(bad)


def test_parse_c1wmfv1_canonical_rejects_unsupported_version() -> None:
    bad = struct.pack(
        C1WMFV1_HEADER_FMT, C1WMFV1_MAGIC, 99, 600, 0, 1, 64, 384, 512, 100, 200, 50, 30, 120, 10
    ) + b"\x00" * (100 + 200 + 50 + 30 + 120 + 10)
    with pytest.raises(ValueError, match="unsupported schema version"):
        parse_c1wmfv1_archive_bytes(bad)


def test_parse_c1wmfv1_canonical_rejects_truncated_archive() -> None:
    inner = _build_synthetic_c1wmfv1_inner()
    truncated = inner[:-50]
    with pytest.raises(ValueError, match="archive size"):
        parse_c1wmfv1_archive_bytes(truncated)


def test_parse_c1wmfv1_canonical_rejects_trailing_schema_drift() -> None:
    inner = _build_synthetic_c1wmfv1_inner()
    with pytest.raises(ValueError, match="archive size"):
        parse_c1wmfv1_archive_bytes(inner + b"tail")


# --------------------------------------------------------------------------
# 3. Section-roles map invariant
# --------------------------------------------------------------------------


def test_c1wmfv1_section_roles_map_covers_all_canonical_sections() -> None:
    inner = _build_synthetic_c1wmfv1_inner()
    sections = parse_c1wmfv1_archive_bytes(inner)
    assert set(C1WMFV1_SECTION_ROLES.keys()) == set(sections.keys())


def test_c1wmfv1_section_roles_use_canonical_role_taxonomy() -> None:
    """The roles must match tac.analysis.scorer_conditional_mdl ROLE_WEIGHTS."""
    from tac.analysis.scorer_conditional_mdl import ROLE_WEIGHTS

    for section, role in C1WMFV1_SECTION_ROLES.items():
        assert role in ROLE_WEIGHTS, (
            f"C1WMFV1 section {section!r} role {role!r} not in canonical "
            f"ROLE_WEIGHTS taxonomy"
        )


def test_c1wmfv1_world_model_and_decoder_blobs_are_decoder_weight_streams() -> None:
    """C1 class-shift design: world-model IS substrate decoder; renderer head is the per-frame decoder."""
    assert C1WMFV1_SECTION_ROLES["world_model_blob"] == "decoder_weight_stream"
    assert C1WMFV1_SECTION_ROLES["decoder_blob"] == "decoder_weight_stream"


def test_c1wmfv1_constants_pinned() -> None:
    assert C1WMFV1_MAGIC == b"WMF\x01"
    assert C1WMFV1_SCHEMA_VERSION == 1
    assert C1WMFV1_HEADER_FMT == "<4sBHBBHHHIIIIII"
    assert C1WMFV1_HEADER_SIZE == 39
