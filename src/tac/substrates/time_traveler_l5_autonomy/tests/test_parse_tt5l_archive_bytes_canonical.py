# SPDX-License-Identifier: MIT
"""Canonical-surface tests for ``parse_tt5l_archive_bytes``.

Promoted to canonical surface 2026-05-14 in lane
``lane_sister_parser_p1_wave_class_shift_5_substrates_20260514`` (Decision G).

Time-Traveler L5 is the predictive-receiver class-shift substrate (the
"staircase Step 3+" end-asymptote per zen-floor band v2).
"""

from __future__ import annotations

import struct

import pytest

from tac.substrates.time_traveler_l5_autonomy.archive import (
    TT5L_HEADER_FMT,
    TT5L_HEADER_SIZE,
    TT5L_MAGIC,
    TT5L_SCHEMA_VERSION,
    TT5L_SECTION_ROLES,
    parse_tt5l_archive_bytes,
)


def _build_synthetic_tt5l_inner(
    *,
    num_pairs: int = 600,
    hidden_dim: int = 64,
    num_hidden_layers: int = 2,
    output_h: int = 384,
    output_w: int = 512,
    foveation_grid_h: int = 8,
    foveation_grid_w: int = 8,
    pose_dim: int = 6,
    per_pair_bytes: int = 45,
    world_blob: bytes = b"\xaa" * 100,
    side_blob: bytes = b"\xbb" * 200,
    ac_blob: bytes = b"\xcc" * 40,
    meta_blob: bytes = b'{"k":"v"}',
) -> bytes:
    """Build a synthetic well-formed TT5L inner blob for parser tests."""
    header = struct.pack(
        TT5L_HEADER_FMT,
        TT5L_MAGIC,
        TT5L_SCHEMA_VERSION,
        num_pairs,
        hidden_dim,
        num_hidden_layers,
        output_h,
        output_w,
        foveation_grid_h,
        foveation_grid_w,
        pose_dim,
        per_pair_bytes,
        len(world_blob),
        len(side_blob),
        len(ac_blob),
        len(meta_blob),
    )
    return header + world_blob + side_blob + ac_blob + meta_blob


# --------------------------------------------------------------------------
# 1. Positive parse — section offsets match writer layout
# --------------------------------------------------------------------------


def test_parse_tt5l_canonical_returns_five_canonical_sections() -> None:
    inner = _build_synthetic_tt5l_inner()
    sections = parse_tt5l_archive_bytes(inner)
    assert set(sections.keys()) == {
        "tt5l_header",
        "world_model_blob",
        "per_pair_side_info_blob",
        "ac_state_blob",
        "meta_blob",
    }
    assert sections["tt5l_header"] == (0, TT5L_HEADER_SIZE)
    assert sections["world_model_blob"] == (34, 100)
    assert sections["per_pair_side_info_blob"] == (134, 200)
    assert sections["ac_state_blob"] == (334, 40)
    assert sections["meta_blob"] == (374, 9)


def test_parse_tt5l_canonical_sections_contiguous_cover_archive() -> None:
    inner = _build_synthetic_tt5l_inner()
    sections = parse_tt5l_archive_bytes(inner)
    cursor = 0
    for name in (
        "tt5l_header",
        "world_model_blob",
        "per_pair_side_info_blob",
        "ac_state_blob",
        "meta_blob",
    ):
        offset, length = sections[name]
        assert offset == cursor, f"section {name!r} not contiguous"
        cursor += length
    assert cursor == len(inner)


# --------------------------------------------------------------------------
# 2. Negative cases — wire-format protection
# --------------------------------------------------------------------------


def test_parse_tt5l_canonical_rejects_short_header() -> None:
    with pytest.raises(ValueError, match="tt5l archive too short"):
        parse_tt5l_archive_bytes(b"TT5L\x01")


def test_parse_tt5l_canonical_rejects_bad_magic() -> None:
    bad = struct.pack(
        TT5L_HEADER_FMT, b"XXXX", 1, 600, 64, 2, 384, 512, 8, 8, 6, 45, 100, 200, 40, 10
    ) + b"\x00" * (100 + 200 + 40 + 10)
    with pytest.raises(ValueError, match="bad magic"):
        parse_tt5l_archive_bytes(bad)


def test_parse_tt5l_canonical_rejects_unsupported_version() -> None:
    bad = struct.pack(
        TT5L_HEADER_FMT, TT5L_MAGIC, 99, 600, 64, 2, 384, 512, 8, 8, 6, 45, 100, 200, 40, 10
    ) + b"\x00" * (100 + 200 + 40 + 10)
    with pytest.raises(ValueError, match="unsupported schema version"):
        parse_tt5l_archive_bytes(bad)


def test_parse_tt5l_canonical_rejects_truncated_archive() -> None:
    inner = _build_synthetic_tt5l_inner()
    truncated = inner[:-30]
    with pytest.raises(ValueError, match="archive size"):
        parse_tt5l_archive_bytes(truncated)


def test_parse_tt5l_canonical_rejects_trailing_schema_drift() -> None:
    inner = _build_synthetic_tt5l_inner()
    with pytest.raises(ValueError, match="archive size"):
        parse_tt5l_archive_bytes(inner + b"tail")


# --------------------------------------------------------------------------
# 3. Section-roles map invariant
# --------------------------------------------------------------------------


def test_tt5l_section_roles_map_covers_all_canonical_sections() -> None:
    inner = _build_synthetic_tt5l_inner()
    sections = parse_tt5l_archive_bytes(inner)
    assert set(TT5L_SECTION_ROLES.keys()) == set(sections.keys())


def test_tt5l_section_roles_use_canonical_role_taxonomy() -> None:
    """The roles must match tac.analysis.scorer_conditional_mdl ROLE_WEIGHTS."""
    from tac.analysis.scorer_conditional_mdl import ROLE_WEIGHTS

    for section, role in TT5L_SECTION_ROLES.items():
        assert role in ROLE_WEIGHTS, (
            f"TT5L section {section!r} role {role!r} not in canonical "
            f"ROLE_WEIGHTS taxonomy"
        )


def test_tt5l_world_model_is_decoder_weight_stream() -> None:
    """Time-Traveler world-model IS substrate decoder in predictive-receiver class."""
    assert TT5L_SECTION_ROLES["world_model_blob"] == "decoder_weight_stream"
    assert TT5L_SECTION_ROLES["per_pair_side_info_blob"] == "sidecar_or_correction_stream"
    assert TT5L_SECTION_ROLES["ac_state_blob"] == "sidecar_or_correction_stream"
    assert "entropy_model_or_range_stream" not in TT5L_SECTION_ROLES.values()


def test_tt5l_constants_pinned() -> None:
    assert TT5L_MAGIC == b"TT5L"
    assert TT5L_SCHEMA_VERSION == 1
    assert TT5L_HEADER_FMT == "<4sBHHBHHBBBBIIII"
    assert TT5L_HEADER_SIZE == 34
