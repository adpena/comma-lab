# SPDX-License-Identifier: MIT
"""Canonical-surface tests for ``parse_ibps1_archive_bytes``.

Promoted to canonical surface 2026-05-14 in lane
``lane_ibps1_canonical_surface_promotion_20260514`` (operator decision #4 from
``feedback_c6_next_wave_landed_20260514.md``).

These tests verify the canonical parser stays in lock-step with the IBPS1
archive writer (:func:`tac.substrates.c6_e4_mdl_ibps.archive.pack_archive`)
so the section-offset contract consumed by:

- :mod:`tools.mdl_scorer_conditional_ablation` (CLI Tier A/B/C ablation)
- :mod:`tac.analysis.scorer_conditional_mdl` (ScorerConditionalMDLEstimator
  section-aware MDL density)
- :mod:`tac.analysis.hnerv_packet_sections` (parser-section manifest IBPS1
  auto-detection by ``b"IBPS"`` magic prefix)

cannot drift if the IBPS1 schema ever versions. Sister of the historical
test in ``src/tac/tests/test_mdl_scorer_conditional_ablation.py``
(which exercises the tool's delegating shim).
"""

from __future__ import annotations

import struct

import pytest
import torch

from tac.substrates.c6_e4_mdl_ibps.archive import (
    IBPS1_HEADER_FMT,
    IBPS1_HEADER_SIZE,
    IBPS1_MAGIC,
    IBPS1_SCHEMA_VERSION,
    IBPS1_SECTION_ROLES,
    pack_archive,
    parse_ibps1_archive_bytes,
)


def _build_synthetic_ibps1_inner(
    *,
    latent_dim: int = 24,
    num_pairs: int = 600,
    encoder_blob: bytes = b"\xaa" * 100,
    decoder_blob: bytes = b"\xbb" * 200,
    meta_blob: bytes = b'{"k":"v"}',
) -> bytes:
    """Build a synthetic well-formed IBPS1 inner blob for parser tests."""
    latent_bytes = num_pairs * latent_dim
    latent_blob = b"\xcc" * latent_bytes
    header = struct.pack(
        IBPS1_HEADER_FMT,
        IBPS1_MAGIC,
        IBPS1_SCHEMA_VERSION,
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


def test_parse_ibps1_canonical_returns_five_canonical_sections() -> None:
    inner = _build_synthetic_ibps1_inner(
        latent_dim=24,
        num_pairs=600,
        encoder_blob=b"\xaa" * 100,
        decoder_blob=b"\xbb" * 200,
        meta_blob=b'{"k":"v"}',
    )
    sections = parse_ibps1_archive_bytes(inner)
    assert set(sections.keys()) == {
        "ibps1_header",
        "encoder_blob",
        "decoder_blob",
        "latent_blob",
        "meta_blob",
    }
    assert sections["ibps1_header"] == (0, IBPS1_HEADER_SIZE)
    assert sections["encoder_blob"] == (25, 100)
    assert sections["decoder_blob"] == (125, 200)
    # latent_blob = 600 * 24 = 14400
    assert sections["latent_blob"] == (325, 14_400)
    assert sections["meta_blob"] == (14_725, 9)


# --------------------------------------------------------------------------
# 2. Negative cases — wire-format protection
# --------------------------------------------------------------------------


def test_parse_ibps1_canonical_rejects_short_header() -> None:
    with pytest.raises(ValueError, match="ibps1 archive too short"):
        parse_ibps1_archive_bytes(b"IBPS\x01")


def test_parse_ibps1_canonical_rejects_bad_magic() -> None:
    bad = struct.pack(IBPS1_HEADER_FMT, b"XXXX", 1, 24, 600, 100, 100, 14_400, 10)
    bad += b"\x00" * (100 + 100 + 14_400 + 10)
    with pytest.raises(ValueError, match="bad magic"):
        parse_ibps1_archive_bytes(bad)


def test_parse_ibps1_canonical_rejects_unsupported_version() -> None:
    bad = struct.pack(
        IBPS1_HEADER_FMT, IBPS1_MAGIC, 99, 24, 600, 100, 100, 14_400, 10
    ) + b"\x00" * (100 + 100 + 14_400 + 10)
    with pytest.raises(ValueError, match="unsupported schema version"):
        parse_ibps1_archive_bytes(bad)


def test_parse_ibps1_canonical_rejects_latent_length_mismatch() -> None:
    # Header declares latent_len=12 but num_pairs*latent_dim=14400
    bad = struct.pack(
        IBPS1_HEADER_FMT, IBPS1_MAGIC, 1, 24, 600, 50, 50, 12, 5
    ) + b"\x00" * (50 + 50 + 12 + 5)
    with pytest.raises(ValueError, match="latent_len"):
        parse_ibps1_archive_bytes(bad)


def test_parse_ibps1_canonical_rejects_truncated_archive() -> None:
    inner = _build_synthetic_ibps1_inner()
    truncated = inner[:-100]
    with pytest.raises(ValueError, match="archive size"):
        parse_ibps1_archive_bytes(truncated)


def test_parse_ibps1_canonical_rejects_trailing_schema_drift() -> None:
    inner = _build_synthetic_ibps1_inner()
    with pytest.raises(ValueError, match="archive size"):
        parse_ibps1_archive_bytes(inner + b"tail")


# --------------------------------------------------------------------------
# 3. Section-roles map invariant
# --------------------------------------------------------------------------


def test_ibps1_section_roles_map_covers_all_canonical_sections() -> None:
    """The roles map must include every section parse_ibps1 returns."""
    inner = _build_synthetic_ibps1_inner()
    sections = parse_ibps1_archive_bytes(inner)
    assert set(IBPS1_SECTION_ROLES.keys()) == set(sections.keys())


def test_ibps1_section_roles_use_canonical_role_taxonomy() -> None:
    """The roles must match tac.analysis.scorer_conditional_mdl ROLE_WEIGHTS."""
    from tac.analysis.scorer_conditional_mdl import ROLE_WEIGHTS

    for section, role in IBPS1_SECTION_ROLES.items():
        assert role in ROLE_WEIGHTS, (
            f"IBPS1 section {section!r} role {role!r} not in canonical "
            f"ROLE_WEIGHTS taxonomy"
        )


def test_ibps1_encoder_blob_role_is_training_provenance_only() -> None:
    """Encoder bytes are not score-affecting on inflate with frames_for_encoder=None."""
    assert IBPS1_SECTION_ROLES["encoder_blob"] == "training_provenance_only"
    assert IBPS1_SECTION_ROLES["decoder_blob"] == "decoder_weight_stream"


# --------------------------------------------------------------------------
# 4. Lock-step with the canonical writer (pack_archive)
# --------------------------------------------------------------------------


def test_parse_ibps1_canonical_matches_pack_archive_byte_ranges() -> None:
    """The parser's offsets must agree with what pack_archive actually writes.

    Builds a real archive via pack_archive, then verifies that the offsets
    returned by parse_ibps1_archive_bytes carve into the same byte ranges
    that the writer laid down.
    """
    torch.manual_seed(0)
    # Construct minimal but valid encoder/decoder state_dicts + latents
    encoder_sd = {"layer.weight": torch.randn(8, 4, dtype=torch.float16)}
    decoder_sd = {"layer.weight": torch.randn(4, 8, dtype=torch.float16)}
    latents = torch.randn(10, 6, dtype=torch.float32)
    meta = {"beta_ib": 0.01, "decoder_channels": [16, 32]}

    archive_bytes = pack_archive(encoder_sd, decoder_sd, latents, meta)
    sections = parse_ibps1_archive_bytes(archive_bytes)

    # The header section must occupy the canonical fixed prefix
    assert sections["ibps1_header"] == (0, IBPS1_HEADER_SIZE)
    # And the very first bytes must be the magic
    h_start, h_len = sections["ibps1_header"]
    assert archive_bytes[h_start : h_start + 4] == IBPS1_MAGIC

    # The sum of (offset + length) for the last section must equal the
    # total archive size — no orphaned bytes
    end_meta_start, end_meta_len = sections["meta_blob"]
    assert end_meta_start + end_meta_len == len(archive_bytes)

    # And the latent blob length must equal num_pairs * latent_dim (10 * 6 = 60)
    _, latent_len = sections["latent_blob"]
    assert latent_len == 60


def test_parse_ibps1_canonical_constants_pinned() -> None:
    """The canonical constants must not drift from the documented contract."""
    assert IBPS1_MAGIC == b"IBPS"
    assert IBPS1_SCHEMA_VERSION == 1
    assert IBPS1_HEADER_FMT == "<4sBHHIIII"
    assert IBPS1_HEADER_SIZE == 25


# --------------------------------------------------------------------------
# 5. Backward compatibility — tools/ shim still delegates
# --------------------------------------------------------------------------


def test_tools_shim_delegates_to_canonical_parser() -> None:
    """tools.mdl_scorer_conditional_ablation.parse_ibps1_archive_bytes
    must produce identical output to the canonical parser."""
    import importlib.util
    import sys
    from pathlib import Path

    tool_path = Path(__file__).resolve().parents[5] / "tools" / "mdl_scorer_conditional_ablation.py"
    if not tool_path.is_file():
        pytest.skip(f"tools/mdl_scorer_conditional_ablation.py missing at {tool_path}")
    spec = importlib.util.spec_from_file_location(
        "_mdl_ablation_test_module", tool_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_mdl_ablation_test_module"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("_mdl_ablation_test_module", None)

    inner = _build_synthetic_ibps1_inner()
    canonical = parse_ibps1_archive_bytes(inner)
    via_shim = mod.parse_ibps1_archive_bytes(inner)
    assert canonical == via_shim
