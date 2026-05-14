# SPDX-License-Identifier: MIT
"""Canonical-surface tests for ``parse_z5pcwm1_archive_bytes``.

Promoted to canonical surface 2026-05-14 in lane
``lane_sister_parser_p1_wave_class_shift_5_substrates_20260514`` (Decision G).

Z5PCWM1 extends Z4CR1 by adding a predictor section and replacing the
per-pair-latent section with (latent_init + residuals + ego_motion). The
predictor implements Rao-Ballard 1999 predictive-coding world-model.
"""

from __future__ import annotations

import struct

import pytest

from tac.substrates.z5_predictive_coding_world_model.archive import (
    Z5PCWM1_HEADER_FMT,
    Z5PCWM1_HEADER_SIZE,
    Z5PCWM1_MAGIC,
    Z5PCWM1_SCHEMA_VERSION,
    Z5PCWM1_SECTION_ROLES,
    parse_z5pcwm1_archive_bytes,
)


def _build_synthetic_z5pcwm1_inner(
    *,
    latent_dim: int = 24,
    ego_motion_dim: int = 8,
    num_pairs: int = 600,
    encoder_blob: bytes = b"\xaa" * 100,
    decoder_blob: bytes = b"\xbb" * 200,
    predictor_blob: bytes = b"\xcc" * 150,
    meta_blob: bytes = b'{"k":"v"}',
) -> bytes:
    """Build a synthetic well-formed Z5PCWM1 inner blob for parser tests."""
    latent_init_bytes = latent_dim  # int8 (latent_dim,)
    residuals_bytes = num_pairs * latent_dim
    ego_motion_bytes = num_pairs * ego_motion_dim
    latent_init_blob = b"\xdd" * latent_init_bytes
    residuals_blob = b"\xee" * residuals_bytes
    ego_motion_blob = b"\xff" * ego_motion_bytes
    header = struct.pack(
        Z5PCWM1_HEADER_FMT,
        Z5PCWM1_MAGIC,
        Z5PCWM1_SCHEMA_VERSION,
        latent_dim,
        ego_motion_dim,
        num_pairs,
        len(encoder_blob),
        len(decoder_blob),
        len(predictor_blob),
        latent_init_bytes,
        residuals_bytes,
        ego_motion_bytes,
        len(meta_blob),
    )
    return (
        header
        + encoder_blob
        + decoder_blob
        + predictor_blob
        + latent_init_blob
        + residuals_blob
        + ego_motion_blob
        + meta_blob
    )


# --------------------------------------------------------------------------
# 1. Positive parse — section offsets match writer layout
# --------------------------------------------------------------------------


def test_parse_z5pcwm1_canonical_returns_eight_canonical_sections() -> None:
    inner = _build_synthetic_z5pcwm1_inner()
    sections = parse_z5pcwm1_archive_bytes(inner)
    assert set(sections.keys()) == {
        "z5pcwm1_header",
        "encoder_blob",
        "decoder_blob",
        "predictor_blob",
        "latent_init_blob",
        "residuals_blob",
        "ego_motion_blob",
        "meta_blob",
    }
    assert sections["z5pcwm1_header"] == (0, Z5PCWM1_HEADER_SIZE)
    assert sections["encoder_blob"] == (39, 100)
    assert sections["decoder_blob"] == (139, 200)
    assert sections["predictor_blob"] == (339, 150)
    # latent_init = 24 (latent_dim,)
    assert sections["latent_init_blob"] == (489, 24)
    # residuals = 600 * 24 = 14400
    assert sections["residuals_blob"] == (513, 14_400)
    # ego_motion = 600 * 8 = 4800
    assert sections["ego_motion_blob"] == (14_913, 4_800)
    assert sections["meta_blob"] == (19_713, 9)


def test_parse_z5pcwm1_canonical_sections_contiguous_cover_archive() -> None:
    inner = _build_synthetic_z5pcwm1_inner()
    sections = parse_z5pcwm1_archive_bytes(inner)
    cursor = 0
    for name in (
        "z5pcwm1_header",
        "encoder_blob",
        "decoder_blob",
        "predictor_blob",
        "latent_init_blob",
        "residuals_blob",
        "ego_motion_blob",
        "meta_blob",
    ):
        offset, length = sections[name]
        assert offset == cursor, f"section {name!r} not contiguous"
        cursor += length
    assert cursor == len(inner)


# --------------------------------------------------------------------------
# 2. Negative cases — wire-format protection
# --------------------------------------------------------------------------


def test_parse_z5pcwm1_canonical_rejects_short_header() -> None:
    with pytest.raises(ValueError, match="z5pcwm1 archive too short"):
        parse_z5pcwm1_archive_bytes(b"Z5WM\x01")


def test_parse_z5pcwm1_canonical_rejects_bad_magic() -> None:
    bad = struct.pack(
        Z5PCWM1_HEADER_FMT, b"XXXX", 1, 24, 8, 600, 100, 200, 150, 24, 14_400, 4_800, 10
    ) + b"\x00" * (100 + 200 + 150 + 24 + 14_400 + 4_800 + 10)
    with pytest.raises(ValueError, match="bad magic"):
        parse_z5pcwm1_archive_bytes(bad)


def test_parse_z5pcwm1_canonical_rejects_unsupported_version() -> None:
    bad = struct.pack(
        Z5PCWM1_HEADER_FMT, Z5PCWM1_MAGIC, 99, 24, 8, 600, 100, 200, 150, 24, 14_400, 4_800, 10
    ) + b"\x00" * (100 + 200 + 150 + 24 + 14_400 + 4_800 + 10)
    with pytest.raises(ValueError, match="unsupported schema version"):
        parse_z5pcwm1_archive_bytes(bad)


def test_parse_z5pcwm1_canonical_rejects_latent_init_length_mismatch() -> None:
    bad = struct.pack(
        Z5PCWM1_HEADER_FMT, Z5PCWM1_MAGIC, 1, 24, 8, 600, 50, 50, 50, 99, 14_400, 4_800, 5
    ) + b"\x00" * (50 + 50 + 50 + 99 + 14_400 + 4_800 + 5)
    with pytest.raises(ValueError, match="latent_init_len"):
        parse_z5pcwm1_archive_bytes(bad)


def test_parse_z5pcwm1_canonical_rejects_residuals_length_mismatch() -> None:
    bad = struct.pack(
        Z5PCWM1_HEADER_FMT, Z5PCWM1_MAGIC, 1, 24, 8, 600, 50, 50, 50, 24, 99, 4_800, 5
    ) + b"\x00" * (50 + 50 + 50 + 24 + 99 + 4_800 + 5)
    with pytest.raises(ValueError, match="residuals_len"):
        parse_z5pcwm1_archive_bytes(bad)


def test_parse_z5pcwm1_canonical_rejects_ego_motion_length_mismatch() -> None:
    bad = struct.pack(
        Z5PCWM1_HEADER_FMT, Z5PCWM1_MAGIC, 1, 24, 8, 600, 50, 50, 50, 24, 14_400, 99, 5
    ) + b"\x00" * (50 + 50 + 50 + 24 + 14_400 + 99 + 5)
    with pytest.raises(ValueError, match="ego_motion_len"):
        parse_z5pcwm1_archive_bytes(bad)


def test_parse_z5pcwm1_canonical_rejects_truncated_archive() -> None:
    inner = _build_synthetic_z5pcwm1_inner()
    truncated = inner[:-100]
    with pytest.raises(ValueError, match="archive size"):
        parse_z5pcwm1_archive_bytes(truncated)


def test_parse_z5pcwm1_canonical_rejects_trailing_schema_drift() -> None:
    inner = _build_synthetic_z5pcwm1_inner()
    with pytest.raises(ValueError, match="archive size"):
        parse_z5pcwm1_archive_bytes(inner + b"tail")


# --------------------------------------------------------------------------
# 3. Section-roles map invariant
# --------------------------------------------------------------------------


def test_z5pcwm1_section_roles_map_covers_all_canonical_sections() -> None:
    inner = _build_synthetic_z5pcwm1_inner()
    sections = parse_z5pcwm1_archive_bytes(inner)
    assert set(Z5PCWM1_SECTION_ROLES.keys()) == set(sections.keys())


def test_z5pcwm1_section_roles_use_canonical_role_taxonomy() -> None:
    """The roles must match tac.analysis.scorer_conditional_mdl ROLE_WEIGHTS."""
    from tac.analysis.scorer_conditional_mdl import ROLE_WEIGHTS

    for section, role in Z5PCWM1_SECTION_ROLES.items():
        assert role in ROLE_WEIGHTS, (
            f"Z5PCWM1 section {section!r} role {role!r} not in canonical "
            f"ROLE_WEIGHTS taxonomy"
        )


def test_z5pcwm1_predictor_is_decoder_weight_stream() -> None:
    """Rao-Ballard predictor is part of inference path — decoder_weight_stream role."""
    assert Z5PCWM1_SECTION_ROLES["predictor_blob"] == "decoder_weight_stream"
    assert Z5PCWM1_SECTION_ROLES["decoder_blob"] == "decoder_weight_stream"
    assert Z5PCWM1_SECTION_ROLES["latent_init_blob"] == "latent_stream"
    assert Z5PCWM1_SECTION_ROLES["residuals_blob"] == "latent_stream"
    assert Z5PCWM1_SECTION_ROLES["ego_motion_blob"] == "sidecar_or_correction_stream"
    assert Z5PCWM1_SECTION_ROLES["encoder_blob"] == "training_provenance_only"


def test_z5pcwm1_constants_pinned() -> None:
    assert Z5PCWM1_MAGIC == b"Z5WM"
    assert Z5PCWM1_SCHEMA_VERSION == 1
    assert Z5PCWM1_HEADER_FMT == "<4sBHHHIIIIIII"
    assert Z5PCWM1_HEADER_SIZE == 39
