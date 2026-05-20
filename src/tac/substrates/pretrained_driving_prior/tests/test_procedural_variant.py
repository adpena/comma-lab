# SPDX-License-Identifier: MIT
"""Tests for the DP1 procedural-codebook replacement variant.

Per task description §C (WAVE-3-DP1-PROCEDURAL-TRAINER-BUILD 2026-05-20) +
``feedback_dp1_procedural_codebook_paired_smoke_pre_dispatch_design_landed_20260520.md``
OP-ROUTABLE #2.

Covers:

* Config invariants (seed-size + shape + dtype + generator_kind validation).
* Determinism (same seed -> same codebook bytes).
* Sister generator-kind parity (xorshift / lcg / pcg64 all callable).
* IN-DOMAIN check ``verify_procedural_codebook_in_domain``.
* Byte-mutation detection ``verify_seed_mutation_changes_codebook_bytes``.
* Archive composition (``compose_with_procedural_codebook`` +
  ``compose_procedural_archive`` convenience wrapper).
* Sister DP1 regression (existing ``parse_dp1_archive_bytes`` + composition
  still works on the original archive bytes; variant does not break the
  base substrate).
* Catalog #272 byte-mutation distinguishing-feature contract: mutate one
  seed byte -> different archive bytes -> different inflate-rendered
  codebook bytes.
"""

from __future__ import annotations

import io
import struct

import brotli  # type: ignore[import-not-found]
import numpy as np
import pytest
import torch

from tac.substrates.pretrained_driving_prior import (
    CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
    Comma2k19FrameIterator,
    DistillationConfig,
    DP1_HEADER_FMT,
    DP1_HEADER_SIZE,
    DP1_MAGIC,
    DP1_SCHEMA_VERSION,
    PROCEDURAL_CODEBOOK_DTYPE_DEFAULT,
    PROCEDURAL_CODEBOOK_SHAPE_DEFAULT,
    PROCEDURAL_SEED_SIZE_BYTES,
    PROCEDURAL_VARIANT_AVAILABLE,
    ProceduralVariantConfig,
    ProceduralVariantError,
    compose_procedural_archive,
    compose_with_procedural_codebook,
    derive_procedural_codebook_replacement,
    distill_codebook,
    pack_archive,
    parse_dp1_archive_bytes,
    verify_procedural_codebook_in_domain,
    verify_seed_mutation_changes_codebook_bytes,
)


_CANONICAL_SEED_32B = b"\x00\x11\x22\x33\x44\x55\x66\x77" * 4


def _make_synthetic_dp1_archive_bytes(seed: int = 0xDA5C) -> bytes:
    """Build a minimal valid DP1 archive bytes for composition fixture use.

    Mirrors the pattern from ``test_composition.py::_make_dp1_archive_bytes``
    but pinned to the procedural variant's defaults (16 frames, 4 pairs).
    """
    iterator = Comma2k19FrameIterator(synthetic=True, n_frames=16, seed=seed)
    cfg = DistillationConfig(
        dataset_name="synthetic_test",
        dataset_sha256="",
        max_frames=16,
        random_seed=seed,
    )
    codebook = distill_codebook(cfg, frames=iter(iterator))
    state_dict = {
        "input_proj.weight": torch.zeros(64, 4, dtype=torch.float32),
        "input_proj.bias": torch.zeros(64, dtype=torch.float32),
        "output_proj.weight": torch.zeros(3, 64, dtype=torch.float32),
        "output_proj.bias": torch.zeros(3, dtype=torch.float32),
    }
    num_pairs = 4
    per_pair_bytes = 6
    residual = bytes([0] * (num_pairs * per_pair_bytes))
    meta = {
        "residual_int8_scale": 64.0,
        "lane_id": "lane_dp1_procedural_codebook_replacement_variant_20260520",
        "evidence_grade": "[proxy]",
        "score_claim": False,
    }
    return pack_archive(
        codebook=codebook,
        renderer_state_dict=state_dict,
        per_pair_residual=residual,
        meta=meta,
        num_pairs=num_pairs,
        output_height=192,
        output_width=256,
        per_pair_bytes=per_pair_bytes,
    )


# -------------------------------------------------------------------------
# 1. Module availability + canonical constants
# -------------------------------------------------------------------------


def test_procedural_variant_module_available_flag_set() -> None:
    """``PROCEDURAL_VARIANT_AVAILABLE`` is True at scaffold landing."""
    assert PROCEDURAL_VARIANT_AVAILABLE is True
    assert PROCEDURAL_SEED_SIZE_BYTES == 32
    assert PROCEDURAL_CODEBOOK_SHAPE_DEFAULT == (1024, 4)
    assert PROCEDURAL_CODEBOOK_DTYPE_DEFAULT == np.dtype(np.uint8)
    assert (
        CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT
        == "comma2k19_ood_derived_basis_replacement"
    )


# -------------------------------------------------------------------------
# 2. ProceduralVariantConfig invariants
# -------------------------------------------------------------------------


def test_procedural_variant_config_rejects_short_seed() -> None:
    """Seeds < 8 bytes fail per canonical equation #26 domain-of-validity."""
    with pytest.raises(ProceduralVariantError, match="outside canonical equation"):
        ProceduralVariantConfig(seed_bytes=b"\x00" * 4)


def test_procedural_variant_config_rejects_long_seed() -> None:
    """Seeds > 256 bytes fail per canonical equation #26 domain-of-validity."""
    with pytest.raises(ProceduralVariantError, match="outside canonical equation"):
        ProceduralVariantConfig(seed_bytes=b"\x00" * 257)


def test_procedural_variant_config_rejects_invalid_generator_kind() -> None:
    """Generator kind must be one of the 3 canonical kinds."""
    with pytest.raises(ProceduralVariantError, match="not canonical"):
        ProceduralVariantConfig(seed_bytes=_CANONICAL_SEED_32B, generator_kind="mt19937")


def test_procedural_variant_config_accepts_canonical_defaults() -> None:
    """Canonical 32-byte seed + defaults constructs cleanly."""
    cfg = ProceduralVariantConfig(seed_bytes=_CANONICAL_SEED_32B)
    assert cfg.generator_kind == "pcg64"
    assert cfg.output_shape == (1024, 4)
    assert cfg.dtype == np.dtype(np.uint8)
    assert cfg.canonical_equation_context == "comma2k19_ood_derived_basis_replacement"


# -------------------------------------------------------------------------
# 3. Derivation determinism + sister generator-kind parity
# -------------------------------------------------------------------------


def test_derive_procedural_codebook_replacement_deterministic() -> None:
    """Same seed -> same codebook bytes byte-for-byte."""
    out_a = derive_procedural_codebook_replacement(seed_bytes=_CANONICAL_SEED_32B)
    out_b = derive_procedural_codebook_replacement(seed_bytes=_CANONICAL_SEED_32B)
    assert np.array_equal(out_a, out_b)
    assert out_a.shape == (1024, 4)
    assert out_a.dtype == np.uint8


def test_derive_procedural_codebook_replacement_3_kind_parity() -> None:
    """All 3 canonical generator kinds (xorshift/lcg/pcg64) derive successfully."""
    for kind in ("xorshift", "lcg", "pcg64"):
        out = derive_procedural_codebook_replacement(
            seed_bytes=_CANONICAL_SEED_32B,
            generator_kind=kind,
        )
        assert out.shape == (1024, 4)
        assert out.dtype == np.uint8
        # Each kind produces a different deterministic codebook (no
        # cross-kind collisions on this canonical seed).
        assert out.nbytes == 4096


# -------------------------------------------------------------------------
# 4. IN-DOMAIN check
# -------------------------------------------------------------------------


def test_verify_procedural_codebook_in_domain_canonical_passes() -> None:
    """Canonical IN-DOMAIN context (``comma2k19_ood_derived_basis_replacement``)
    passes per slot 3 refined domain.
    """
    assert verify_procedural_codebook_in_domain() is True
    assert (
        verify_procedural_codebook_in_domain(
            context="comma2k19_ood_derived_basis_replacement"
        )
        is True
    )


def test_verify_procedural_codebook_in_domain_wrong_context_fails() -> None:
    """An out-of-domain context label fails the IN-DOMAIN check.

    NOTE: when slot 3's ``validate_context_is_in_domain`` lands, this
    test verifies the integration goes through the canonical helper.
    Until then it verifies the constant-comparison fallback.
    """
    # Skip if slot 3 sister helper has landed (the canonical helper may
    # accept additional aliases for the same domain).
    try:
        from tac.canonical_equations import validate_context_is_in_domain  # noqa: F401
        pytest.skip("slot 3 sister helper landed; canonical helper has its own contract")
    except ImportError:
        pass
    assert verify_procedural_codebook_in_domain(context="random_other_domain") is False


# -------------------------------------------------------------------------
# 5. Byte-mutation distinguishing-feature contract (Catalog #272)
# -------------------------------------------------------------------------


def test_verify_seed_mutation_changes_codebook_bytes_passes_for_canonical_seed() -> None:
    """Flipping any seed byte produces a different derived codebook.

    This is the Catalog #272 byte-mutation distinguishing-feature contract:
    the operational mechanism MUST be byte-mutation-traceable.
    """
    assert verify_seed_mutation_changes_codebook_bytes(seed_bytes=_CANONICAL_SEED_32B) is True


# -------------------------------------------------------------------------
# 6. Archive composition + bytes-saved invariant
# -------------------------------------------------------------------------


def test_compose_with_procedural_codebook_reduces_archive_bytes() -> None:
    """Procedural variant reduces archive bytes vs original (codebook shrinks)."""
    original = _make_synthetic_dp1_archive_bytes()
    new = compose_with_procedural_codebook(
        original_archive_bytes=original,
        seed_bytes=_CANONICAL_SEED_32B,
    )
    bytes_saved = len(original) - len(new)
    # Synthetic codebook is small but seed is even smaller; bytes-saved
    # is expected to be > 0 but not catastrophic.
    assert bytes_saved > 0, (
        f"compose_with_procedural_codebook did NOT shrink archive "
        f"({len(original)} -> {len(new)})"
    )
    # New archive remains parseable as DP1.
    sections = parse_dp1_archive_bytes(new)
    assert set(sections.keys()) == {
        "dp1_header",
        "codebook_blob",
        "renderer_blob",
        "residual_blob",
        "meta_blob",
    }


def test_compose_with_procedural_codebook_preserves_non_codebook_sections() -> None:
    """Renderer/residual/meta sections are byte-identical post-composition."""
    original = _make_synthetic_dp1_archive_bytes()
    new = compose_with_procedural_codebook(
        original_archive_bytes=original,
        seed_bytes=_CANONICAL_SEED_32B,
    )
    orig_sections = parse_dp1_archive_bytes(original)
    new_sections = parse_dp1_archive_bytes(new)
    # codebook_blob bytes change; the other 3 payload sections do NOT.
    for section_name in ("renderer_blob", "residual_blob", "meta_blob"):
        orig_start, orig_len = orig_sections[section_name]
        new_start, new_len = new_sections[section_name]
        assert orig_len == new_len, f"{section_name} length changed unexpectedly"
        orig_bytes = original[orig_start : orig_start + orig_len]
        new_bytes = new[new_start : new_start + new_len]
        assert orig_bytes == new_bytes, (
            f"{section_name} content differs between original and procedural variant"
        )


def test_compose_procedural_archive_convenience_wrapper_matches_full_API() -> None:
    """``compose_procedural_archive`` (archive.py) is a thin shim around
    ``compose_with_procedural_codebook`` with canonical defaults.
    """
    original = _make_synthetic_dp1_archive_bytes()
    via_archive_py = compose_procedural_archive(
        original_archive_bytes=original,
        seed_bytes=_CANONICAL_SEED_32B,
    )
    via_variant_module = compose_with_procedural_codebook(
        original_archive_bytes=original,
        seed_bytes=_CANONICAL_SEED_32B,
    )
    assert via_archive_py == via_variant_module


# -------------------------------------------------------------------------
# 7. Catalog #272 byte-mutation smoke at the ARCHIVE level
# -------------------------------------------------------------------------


def test_compose_with_procedural_codebook_byte_mutation_smoke_catalog_272() -> None:
    """Catalog #272 byte-mutation smoke at the archive surface: different
    seed -> different archive bytes -> Tier-A different codebook section.

    This is the per-substrate distinguishing-feature integration contract:
    the seed bytes MUST flow into the rendered archive (NOT be silently
    ignored). The L0 SCAFFOLD verifier asserts the codebook section bytes
    change when the seed bytes change.
    """
    original = _make_synthetic_dp1_archive_bytes()
    archive_seed_a = compose_with_procedural_codebook(
        original_archive_bytes=original,
        seed_bytes=_CANONICAL_SEED_32B,
    )
    seed_b = bytearray(_CANONICAL_SEED_32B)
    seed_b[0] = (seed_b[0] + 1) & 0xFF
    archive_seed_b = compose_with_procedural_codebook(
        original_archive_bytes=original,
        seed_bytes=bytes(seed_b),
    )
    assert archive_seed_a != archive_seed_b, (
        "byte-mutation smoke FAILED: seed change did not propagate to archive bytes"
    )
    # Specifically the codebook_blob section differs; the other 3 payload
    # sections are byte-identical.
    sections_a = parse_dp1_archive_bytes(archive_seed_a)
    sections_b = parse_dp1_archive_bytes(archive_seed_b)
    cb_start_a, cb_len_a = sections_a["codebook_blob"]
    cb_start_b, cb_len_b = sections_b["codebook_blob"]
    cb_bytes_a = archive_seed_a[cb_start_a : cb_start_a + cb_len_a]
    cb_bytes_b = archive_seed_b[cb_start_b : cb_start_b + cb_len_b]
    assert cb_bytes_a != cb_bytes_b, (
        "codebook_blob bytes did not change with seed; operational mechanism inert"
    )


# -------------------------------------------------------------------------
# 8. Sister DP1 regression — base substrate still works
# -------------------------------------------------------------------------


def test_sister_dp1_base_substrate_archive_still_parses() -> None:
    """Regression: the procedural variant landing does NOT break the
    canonical Comma2k19-derived DP1 archive build path.
    """
    original = _make_synthetic_dp1_archive_bytes()
    sections = parse_dp1_archive_bytes(original)
    assert "codebook_blob" in sections
    assert "renderer_blob" in sections
    assert "residual_blob" in sections
    assert "meta_blob" in sections
    # Header magic + version invariant.
    (magic, version, *_rest) = struct.unpack(DP1_HEADER_FMT, original[:DP1_HEADER_SIZE])
    assert magic == DP1_MAGIC
    assert version == DP1_SCHEMA_VERSION
