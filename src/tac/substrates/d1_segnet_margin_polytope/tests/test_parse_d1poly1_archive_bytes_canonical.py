# SPDX-License-Identifier: MIT
"""Canonical-surface tests for ``parse_d1poly1_archive_bytes``.

Promoted to canonical surface 2026-05-14 in lane
``lane_ibps1_parser_wave_p0_d1_d4_dp1_20260514`` (operator-routed decision
#1 from ``feedback_ibps1_canonical_surface_landed_20260514.md`` —
P0 follow-up: D1 / D4 / DP1).

These tests verify the D1 (D1POLY1) canonical parser stays in lock-step
with the D1 archive writer
(:func:`tac.substrates.d1_segnet_margin_polytope.archive.pack_archive`)
so the section-offset contract consumed by:

- :mod:`tac.analysis.scorer_conditional_mdl` (ScorerConditionalMDLEstimator
  section-aware Tier A density estimation)
- :mod:`tac.analysis.hnerv_packet_sections` (parser-section manifest
  D1POLY1 auto-detection by ``b"D1PY"`` magic prefix)

cannot drift if the D1POLY1 schema ever versions. Sister of
``test_parse_ibps1_archive_bytes_canonical.py`` (the IBPS1 reference)
and ``test_parse_wzf01_archive_bytes_canonical.py`` (D4).
"""

from __future__ import annotations

import struct

import pytest

from tac.substrates.d1_segnet_margin_polytope.archive import (
    D1POLY1_HEADER_FMT,
    D1POLY1_HEADER_SIZE,
    D1POLY1_MAGIC,
    D1POLY1_SCHEMA_VERSION,
    D1POLY1_SECTION_ROLES,
    parse_d1poly1_archive_bytes,
)


def _build_synthetic_d1poly1_inner(
    *,
    height: int = 96,
    width: int = 128,
    scale: float = 1.0,
    jacobian_lipschitz: float = 0.5,
    base_id: bytes = b"a1",
    base_sha: bytes = b"0123456789abcdef",  # exactly 16 chars
    margin_map_blob: bytes = b"\xaa" * 200,
    polytope_payload_blob: bytes = b"\xbb" * 100,
    meta_blob: bytes = b'{"k":"v"}',
) -> bytes:
    """Build a synthetic well-formed D1POLY1 inner blob for parser tests."""
    header = struct.pack(
        D1POLY1_HEADER_FMT,
        D1POLY1_MAGIC,
        D1POLY1_SCHEMA_VERSION,
        height,
        width,
        scale,
        jacobian_lipschitz,
        len(base_id),
        len(base_sha),
        len(margin_map_blob),
        len(polytope_payload_blob),
        len(meta_blob),
    )
    return (
        header
        + base_id
        + base_sha
        + margin_map_blob
        + polytope_payload_blob
        + meta_blob
    )


# --------------------------------------------------------------------------
# 1. Positive parse — section offsets match writer layout
# --------------------------------------------------------------------------


def test_parse_d1poly1_canonical_returns_six_canonical_sections() -> None:
    inner = _build_synthetic_d1poly1_inner()
    sections = parse_d1poly1_archive_bytes(inner)
    assert set(sections.keys()) == {
        "d1poly1_header",
        "base_substrate_id",
        "base_archive_sha256_truncated",
        "margin_map_blob",
        "polytope_payload_blob",
        "meta_blob",
    }
    assert sections["d1poly1_header"] == (0, D1POLY1_HEADER_SIZE)
    # base_id is "a1" = 2 bytes
    assert sections["base_substrate_id"] == (31, 2)
    assert sections["base_archive_sha256_truncated"] == (33, 16)
    assert sections["margin_map_blob"] == (49, 200)
    assert sections["polytope_payload_blob"] == (249, 100)
    assert sections["meta_blob"] == (349, 9)


# --------------------------------------------------------------------------
# 2. Negative cases — wire-format protection
# --------------------------------------------------------------------------


def test_parse_d1poly1_canonical_rejects_short_header() -> None:
    with pytest.raises(ValueError, match="d1poly1 archive too short"):
        parse_d1poly1_archive_bytes(b"D1PY\x01")


def test_parse_d1poly1_canonical_rejects_bad_magic() -> None:
    bad = _build_synthetic_d1poly1_inner()
    # Mutate the magic
    bad = b"XXXX" + bad[4:]
    with pytest.raises(ValueError, match="bad magic"):
        parse_d1poly1_archive_bytes(bad)


def test_parse_d1poly1_canonical_rejects_unsupported_version() -> None:
    bad = struct.pack(
        D1POLY1_HEADER_FMT,
        D1POLY1_MAGIC,
        99,
        96,
        128,
        1.0,
        0.5,
        2,
        16,
        200,
        100,
        9,
    ) + b"\x00" * (2 + 16 + 200 + 100 + 9)
    with pytest.raises(ValueError, match="unsupported schema version"):
        parse_d1poly1_archive_bytes(bad)


def test_parse_d1poly1_canonical_rejects_wrong_sha_length() -> None:
    # sha_len must be exactly 16
    bad = struct.pack(
        D1POLY1_HEADER_FMT,
        D1POLY1_MAGIC,
        1,
        96,
        128,
        1.0,
        0.5,
        2,
        8,  # WRONG: should be 16
        200,
        100,
        9,
    ) + b"\x00" * (2 + 8 + 200 + 100 + 9)
    with pytest.raises(ValueError, match="sha truncated len"):
        parse_d1poly1_archive_bytes(bad)


def test_parse_d1poly1_canonical_rejects_zero_base_id_length() -> None:
    bad = struct.pack(
        D1POLY1_HEADER_FMT,
        D1POLY1_MAGIC,
        1,
        96,
        128,
        1.0,
        0.5,
        0,  # WRONG: must be > 0
        16,
        200,
        100,
        9,
    ) + b"\x00" * (0 + 16 + 200 + 100 + 9)
    with pytest.raises(ValueError, match="base_substrate_id length"):
        parse_d1poly1_archive_bytes(bad)


def test_parse_d1poly1_canonical_rejects_truncated_archive() -> None:
    inner = _build_synthetic_d1poly1_inner()
    truncated = inner[:-50]
    with pytest.raises(ValueError, match="archive size"):
        parse_d1poly1_archive_bytes(truncated)


def test_parse_d1poly1_canonical_rejects_trailing_schema_drift() -> None:
    inner = _build_synthetic_d1poly1_inner()
    with pytest.raises(ValueError, match="archive size"):
        parse_d1poly1_archive_bytes(inner + b"trailing")


# --------------------------------------------------------------------------
# 3. Section-roles map invariant
# --------------------------------------------------------------------------


def test_d1poly1_section_roles_map_covers_all_canonical_sections() -> None:
    """The roles map must include every section parse_d1poly1 returns."""
    inner = _build_synthetic_d1poly1_inner()
    sections = parse_d1poly1_archive_bytes(inner)
    assert set(D1POLY1_SECTION_ROLES.keys()) == set(sections.keys())


def test_d1poly1_section_roles_use_canonical_role_taxonomy() -> None:
    """The roles must match tac.analysis.scorer_conditional_mdl ROLE_WEIGHTS."""
    from tac.analysis.scorer_conditional_mdl import ROLE_WEIGHTS

    for section, role in D1POLY1_SECTION_ROLES.items():
        assert role in ROLE_WEIGHTS, (
            f"D1POLY1 section {section!r} role {role!r} not in canonical "
            f"ROLE_WEIGHTS taxonomy"
        )


def test_d1poly1_margin_map_role_is_sidecar_correction() -> None:
    """The margin map IS the per-pixel safe-noise budget — sidecar/correction."""
    assert D1POLY1_SECTION_ROLES["margin_map_blob"] == "sidecar_or_correction_stream"
    assert (
        D1POLY1_SECTION_ROLES["polytope_payload_blob"]
        == "sidecar_or_correction_stream"
    )


# --------------------------------------------------------------------------
# 4. Constants pinned
# --------------------------------------------------------------------------


def test_parse_d1poly1_canonical_constants_pinned() -> None:
    """The canonical constants must not drift from the documented contract."""
    assert D1POLY1_MAGIC == b"D1PY"
    assert D1POLY1_SCHEMA_VERSION == 1
    assert D1POLY1_HEADER_FMT == "<4sBHHffBBIII"
    assert D1POLY1_HEADER_SIZE == 31
