# SPDX-License-Identifier: MIT
"""Canonical-surface tests for ``parse_wzf01_archive_bytes``.

Promoted to canonical surface 2026-05-14 in lane
``lane_ibps1_parser_wave_p0_d1_d4_dp1_20260514`` (operator-routed decision
#1 from ``feedback_ibps1_canonical_surface_landed_20260514.md`` —
P0 follow-up: D1 / D4 / DP1).

These tests verify the D4 (WZF01) canonical parser stays in lock-step
with the D4 archive writer
(:func:`tac.substrates.d4_wyner_ziv_frame_0.archive.pack_archive`) so the
section-offset contract consumed by:

- :mod:`tac.analysis.scorer_conditional_mdl` (ScorerConditionalMDLEstimator
  section-aware Tier A density estimation)
- :mod:`tac.analysis.hnerv_packet_sections` (parser-section manifest
  WZF01 auto-detection by ``b"WZF\\x01"`` magic prefix)

cannot drift if the WZF01 schema ever versions. Sister of the IBPS1 / D1
canonical-surface test suites.
"""

from __future__ import annotations

import struct

import pytest

from tac.substrates.d4_wyner_ziv_frame_0.archive import (
    BASE_SHA_HEX_LEN,
    WZF01_HEADER_FMT,
    WZF01_HEADER_SIZE,
    WZF01_MAGIC,
    WZF01_SCHEMA_VERSION,
    WZF01_SECTION_ROLES,
    parse_wzf01_archive_bytes,
)


def _build_synthetic_wzf01_inner(
    *,
    num_pairs: int = 600,
    motion_mode: int = 0,  # 0=SE3
    flow_grid_h: int = 0,
    flow_grid_w: int = 0,
    residual_coarse_h: int = 12,
    residual_coarse_w: int = 16,
    base_sha: bytes = b"0" * 64,
    base_bytes: bytes = b"\xff" * 500,
    motion_blob: bytes = b"\xaa" * 1000,
    residual_blob: bytes = b"\xbb" * 2000,
    meta_blob: bytes = b'{"motion_mode":0}',
) -> bytes:
    """Build a synthetic well-formed WZF01 inner blob for parser tests."""
    header = struct.pack(
        WZF01_HEADER_FMT,
        WZF01_MAGIC,
        WZF01_SCHEMA_VERSION,
        num_pairs,
        motion_mode,
        flow_grid_h,
        flow_grid_w,
        residual_coarse_h,
        residual_coarse_w,
        BASE_SHA_HEX_LEN,
        len(base_bytes),
        len(motion_blob),
        len(residual_blob),
        len(meta_blob),
    )
    return (
        header
        + base_sha
        + base_bytes
        + motion_blob
        + residual_blob
        + meta_blob
    )


# --------------------------------------------------------------------------
# 1. Positive parse — section offsets match writer layout
# --------------------------------------------------------------------------


def test_parse_wzf01_canonical_returns_six_canonical_sections() -> None:
    inner = _build_synthetic_wzf01_inner()
    sections = parse_wzf01_archive_bytes(inner)
    assert set(sections.keys()) == {
        "wzf01_header",
        "base_substrate_archive_sha256",
        "base_substrate_bytes",
        "motion_blob",
        "residual_blob",
        "meta_blob",
    }
    assert sections["wzf01_header"] == (0, WZF01_HEADER_SIZE)
    assert sections["base_substrate_archive_sha256"] == (33, 64)
    assert sections["base_substrate_bytes"] == (97, 500)
    assert sections["motion_blob"] == (597, 1000)
    assert sections["residual_blob"] == (1597, 2000)
    assert sections["meta_blob"] == (3597, 17)


def test_parse_wzf01_canonical_supports_optical_flow_mode() -> None:
    """Optical-flow mode (motion_mode=1) parses identically to SE3."""
    inner = _build_synthetic_wzf01_inner(
        motion_mode=1, flow_grid_h=12, flow_grid_w=16
    )
    sections = parse_wzf01_archive_bytes(inner)
    # Section layout is unchanged; only the motion_blob CONTENT differs
    assert set(sections.keys()) == set(WZF01_SECTION_ROLES.keys())


# --------------------------------------------------------------------------
# 2. Negative cases — wire-format protection
# --------------------------------------------------------------------------


def test_parse_wzf01_canonical_rejects_short_header() -> None:
    with pytest.raises(ValueError, match="wzf01 archive too short"):
        parse_wzf01_archive_bytes(b"WZF\x01\x01")


def test_parse_wzf01_canonical_rejects_bad_magic() -> None:
    bad = _build_synthetic_wzf01_inner()
    bad = b"XXXX" + bad[4:]
    with pytest.raises(ValueError, match="bad magic"):
        parse_wzf01_archive_bytes(bad)


def test_parse_wzf01_canonical_rejects_unsupported_version() -> None:
    bad = struct.pack(
        WZF01_HEADER_FMT,
        WZF01_MAGIC,
        99,
        600, 0, 0, 0, 12, 16,
        BASE_SHA_HEX_LEN,
        500, 1000, 2000, 17,
    ) + b"\x00" * (64 + 500 + 1000 + 2000 + 17)
    with pytest.raises(ValueError, match="unsupported schema version"):
        parse_wzf01_archive_bytes(bad)


def test_parse_wzf01_canonical_rejects_wrong_base_sha_length() -> None:
    bad = struct.pack(
        WZF01_HEADER_FMT,
        WZF01_MAGIC, 1,
        600, 0, 0, 0, 12, 16,
        32,  # WRONG: should be 64
        500, 1000, 2000, 17,
    ) + b"\x00" * (32 + 500 + 1000 + 2000 + 17)
    with pytest.raises(ValueError, match="base_sha_len"):
        parse_wzf01_archive_bytes(bad)


def test_parse_wzf01_canonical_rejects_unknown_motion_mode() -> None:
    bad = struct.pack(
        WZF01_HEADER_FMT,
        WZF01_MAGIC, 1,
        600,
        7,  # WRONG: must be 0 or 1
        0, 0, 12, 16,
        BASE_SHA_HEX_LEN,
        500, 1000, 2000, 17,
    ) + b"\x00" * (64 + 500 + 1000 + 2000 + 17)
    with pytest.raises(ValueError, match="motion_mode"):
        parse_wzf01_archive_bytes(bad)


def test_parse_wzf01_canonical_rejects_truncated_archive() -> None:
    inner = _build_synthetic_wzf01_inner()
    truncated = inner[:-100]
    with pytest.raises(ValueError, match="archive size"):
        parse_wzf01_archive_bytes(truncated)


def test_parse_wzf01_canonical_rejects_trailing_schema_drift() -> None:
    inner = _build_synthetic_wzf01_inner()
    with pytest.raises(ValueError, match="archive size"):
        parse_wzf01_archive_bytes(inner + b"trailing")


# --------------------------------------------------------------------------
# 3. Section-roles map invariant
# --------------------------------------------------------------------------


def test_wzf01_section_roles_map_covers_all_canonical_sections() -> None:
    """The roles map must include every section parse_wzf01 returns."""
    inner = _build_synthetic_wzf01_inner()
    sections = parse_wzf01_archive_bytes(inner)
    assert set(WZF01_SECTION_ROLES.keys()) == set(sections.keys())


def test_wzf01_section_roles_use_canonical_role_taxonomy() -> None:
    """The roles must match tac.analysis.scorer_conditional_mdl ROLE_WEIGHTS."""
    from tac.analysis.scorer_conditional_mdl import ROLE_WEIGHTS

    for section, role in WZF01_SECTION_ROLES.items():
        assert role in ROLE_WEIGHTS, (
            f"WZF01 section {section!r} role {role!r} not in canonical "
            f"ROLE_WEIGHTS taxonomy"
        )


def test_wzf01_role_semantics_match_wyner_ziv_model() -> None:
    """Section-role taxonomy must reflect Wyner-Ziv decoder structure.

    - base_substrate_bytes reconstructs frame 1 (cooperative receiver
      side-information) -> decoder_weight_stream
    - motion_blob is per-pair correction to the warp -> sidecar_or_correction_stream
    - residual_blob is the rate-limited frame_0-warp(frame_1) energy -> latent_stream
    """
    assert WZF01_SECTION_ROLES["base_substrate_bytes"] == "decoder_weight_stream"
    assert WZF01_SECTION_ROLES["motion_blob"] == "sidecar_or_correction_stream"
    assert WZF01_SECTION_ROLES["residual_blob"] == "latent_stream"


# --------------------------------------------------------------------------
# 4. Constants pinned
# --------------------------------------------------------------------------


def test_parse_wzf01_canonical_constants_pinned() -> None:
    """The canonical constants must not drift from the documented contract."""
    assert WZF01_MAGIC == b"WZF\x01"
    assert WZF01_SCHEMA_VERSION == 1
    assert WZF01_HEADER_FMT == "<4sBHBHHHHBIIII"
    assert WZF01_HEADER_SIZE == 33
    assert BASE_SHA_HEX_LEN == 64
