# SPDX-License-Identifier: MIT
"""Canonical-surface tests for ``parse_dp1_archive_bytes``.

Promoted to canonical surface 2026-05-14 in lane
``lane_ibps1_parser_wave_p0_d1_d4_dp1_20260514`` (operator-routed decision
#1 from ``feedback_ibps1_canonical_surface_landed_20260514.md`` —
P0 follow-up: D1 / D4 / DP1).

These tests verify the DP1 canonical parser stays in lock-step with the
DP1 archive writer
(:func:`tac.substrates.pretrained_driving_prior.archive.pack_archive`)
so the section-offset contract consumed by:

- :mod:`tac.analysis.scorer_conditional_mdl` (ScorerConditionalMDLEstimator
  section-aware Tier A density estimation)
- :mod:`tac.analysis.hnerv_packet_sections` (parser-section manifest
  DP1 auto-detection by ``b"DP1\\x00"`` magic prefix)

cannot drift if the DP1 schema ever versions. Sister of the IBPS1 / D1 /
WZF01 canonical-surface test suites.

Critically: ``parse_dp1_archive_bytes`` MUST NEVER call
:func:`_deserialize_state_dict` (which invokes ``pickle.loads`` on the
renderer blob). The canonical parser returns ONLY byte ranges so it is
safe to call on adversarially-tampered archives without risking
arbitrary-code-execution via pickle.
"""

from __future__ import annotations

import struct

import pytest

from tac.substrates.pretrained_driving_prior.archive import (
    DP1_HEADER_FMT,
    DP1_HEADER_SIZE,
    DP1_MAGIC,
    DP1_SCHEMA_VERSION,
    DP1_SECTION_ROLES,
    parse_dp1_archive_bytes,
)


def _build_synthetic_dp1_inner(
    *,
    num_pairs: int = 600,
    output_height: int = 384,
    output_width: int = 512,
    per_pair_bytes: int = 1,
    codebook_blob: bytes = b"\xaa" * 500,
    renderer_blob: bytes = b"\xbb" * 2000,
    residual_blob: bytes = b"\xcc" * 600,
    meta_blob: bytes = b'{"k":"v"}',
) -> bytes:
    """Build a synthetic well-formed DP1 inner blob for parser tests."""
    header = struct.pack(
        DP1_HEADER_FMT,
        DP1_MAGIC,
        DP1_SCHEMA_VERSION,
        num_pairs,
        output_height,
        output_width,
        per_pair_bytes,
        len(codebook_blob),
        len(renderer_blob),
        len(residual_blob),
        len(meta_blob),
    )
    return (
        header + codebook_blob + renderer_blob + residual_blob + meta_blob
    )


# --------------------------------------------------------------------------
# 1. Positive parse — section offsets match writer layout
# --------------------------------------------------------------------------


def test_parse_dp1_canonical_returns_five_canonical_sections() -> None:
    inner = _build_synthetic_dp1_inner()
    sections = parse_dp1_archive_bytes(inner)
    assert set(sections.keys()) == {
        "dp1_header",
        "codebook_blob",
        "renderer_blob",
        "residual_blob",
        "meta_blob",
    }
    assert sections["dp1_header"] == (0, DP1_HEADER_SIZE)
    assert sections["codebook_blob"] == (28, 500)
    assert sections["renderer_blob"] == (528, 2000)
    assert sections["residual_blob"] == (2528, 600)
    assert sections["meta_blob"] == (3128, 9)


# --------------------------------------------------------------------------
# 2. Negative cases — wire-format protection
# --------------------------------------------------------------------------


def test_parse_dp1_canonical_rejects_short_header() -> None:
    with pytest.raises(ValueError, match="dp1 archive too short"):
        parse_dp1_archive_bytes(b"DP1\x00\x01")


def test_parse_dp1_canonical_rejects_bad_magic() -> None:
    bad = _build_synthetic_dp1_inner()
    bad = b"XXXX" + bad[4:]
    with pytest.raises(ValueError, match="bad magic"):
        parse_dp1_archive_bytes(bad)


def test_parse_dp1_canonical_rejects_unsupported_version() -> None:
    bad = struct.pack(
        DP1_HEADER_FMT,
        DP1_MAGIC,
        99,
        600, 384, 512, 1,
        500, 2000, 600, 9,
    ) + b"\x00" * (500 + 2000 + 600 + 9)
    with pytest.raises(ValueError, match="unsupported schema version"):
        parse_dp1_archive_bytes(bad)


def test_parse_dp1_canonical_rejects_truncated_archive() -> None:
    inner = _build_synthetic_dp1_inner()
    truncated = inner[:-100]
    with pytest.raises(ValueError, match="archive size"):
        parse_dp1_archive_bytes(truncated)


def test_parse_dp1_canonical_rejects_trailing_schema_drift() -> None:
    inner = _build_synthetic_dp1_inner()
    with pytest.raises(ValueError, match="archive size"):
        parse_dp1_archive_bytes(inner + b"trailing")


# --------------------------------------------------------------------------
# 3. Section-roles map invariant
# --------------------------------------------------------------------------


def test_dp1_section_roles_map_covers_all_canonical_sections() -> None:
    """The roles map must include every section parse_dp1 returns."""
    inner = _build_synthetic_dp1_inner()
    sections = parse_dp1_archive_bytes(inner)
    assert set(DP1_SECTION_ROLES.keys()) == set(sections.keys())


def test_dp1_section_roles_use_canonical_role_taxonomy() -> None:
    """The roles must match tac.analysis.scorer_conditional_mdl ROLE_WEIGHTS."""
    from tac.analysis.scorer_conditional_mdl import ROLE_WEIGHTS

    for section, role in DP1_SECTION_ROLES.items():
        assert role in ROLE_WEIGHTS, (
            f"DP1 section {section!r} role {role!r} not in canonical "
            f"ROLE_WEIGHTS taxonomy"
        )


def test_dp1_codebook_and_renderer_both_decoder_weight_streams() -> None:
    """Codebook (fixed lookup) + renderer (overfit weights) BOTH decoder streams.

    Both contribute to inflate-time decode; both are structurally weight-like.
    """
    assert DP1_SECTION_ROLES["codebook_blob"] == "decoder_weight_stream"
    assert DP1_SECTION_ROLES["renderer_blob"] == "decoder_weight_stream"
    assert DP1_SECTION_ROLES["residual_blob"] == "latent_stream"


# --------------------------------------------------------------------------
# 4. Pickle safety — parser never invokes pickle.loads
# --------------------------------------------------------------------------


def test_parse_dp1_canonical_does_not_pickle_renderer_blob() -> None:
    """Canonical parser must not pickle.loads the renderer blob.

    A tampered DP1 archive could embed a malicious pickle payload as the
    renderer blob. The canonical section-offset parser must return byte
    ranges WITHOUT deserializing — only :func:`parse_archive` (the full
    decoder) calls :func:`_deserialize_state_dict`.
    """
    # Embed a deliberately bogus byte sequence in the renderer slot.
    # If the canonical parser invoked pickle.loads, this would raise
    # UnpicklingError; instead it should successfully return offsets.
    bogus_renderer = b"\x80\x05not_a_real_pickle_payload"
    inner = _build_synthetic_dp1_inner(renderer_blob=bogus_renderer)
    sections = parse_dp1_archive_bytes(inner)
    # Parser succeeded — it returned only byte ranges.
    assert sections["renderer_blob"][1] == len(bogus_renderer)


# --------------------------------------------------------------------------
# 5. Constants pinned
# --------------------------------------------------------------------------


def test_parse_dp1_canonical_constants_pinned() -> None:
    """The canonical constants must not drift from the documented contract."""
    assert DP1_MAGIC == b"DP1\x00"
    assert DP1_SCHEMA_VERSION == 1
    assert DP1_HEADER_FMT == "<4sBHHHBIIII"
    assert DP1_HEADER_SIZE == 28
