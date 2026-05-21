# SPDX-License-Identifier: MIT
"""Tests for the grayscale_lut procedural-LUT replacement variant.

Per WAVE-3-GRAYSCALE-LUT-PROCEDURAL-TRAINER-BUILD 2026-05-20 + PR101/PR106
BUILD DESIGN landing commit ``086d3ac1d`` Top-3 #1 PIVOT + DP1 PROCEDURAL
TRAINER BUILD canonical pattern landing commit ``9cbfa471c`` + VQ-VAE
PROCEDURAL VARIANT BUILD canonical sister landing commit ``6fea30f22``.

Covers:

* Module availability + canonical constants (``PROCEDURAL_VARIANT_AVAILABLE``,
  ``PROCEDURAL_SEED_SIZE_BYTES``, ``PROCEDURAL_LUT_BYTES_DEFAULT``,
  ``CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT``).
* Config invariants (seed-size + shape + dtype + generator_kind validation).
* Determinism of derivation (same seed -> same LUT bytes).
* Sister generator-kind parity (xorshift / lcg / pcg64 all callable).
* IN-DOMAIN check ``verify_procedural_lut_in_domain``.
* Byte-mutation detection ``verify_seed_mutation_changes_lut_bytes``.
* Canonical equation #26 closed-form predictions
  (``predicted_archive_bytes_saved`` + ``predicted_delta_s`` matches
  ``-25 * 224 / 37_545_489 = -0.000149``).
* Archive composition (``compose_with_procedural_lut`` +
  ``compose_procedural_archive`` convenience wrapper).
* Sister grayscale_lut regression (existing ``parse_archive`` + ``pack_archive``
  roundtrip still works; variant does not break the base substrate).
* Catalog #272 byte-mutation distinguishing-feature contract: mutate one
  seed byte -> different archive bytes -> different envelope bytes.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from tac.substrates.grayscale_lut import (
    CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
    PROCEDURAL_LUT_BYTES_DEFAULT,
    PROCEDURAL_LUT_DTYPE_DEFAULT,
    PROCEDURAL_LUT_SENTINEL,
    PROCEDURAL_SEED_SIZE_BYTES,
    PROCEDURAL_VARIANT_AVAILABLE,
    GrayscaleLutConfig,
    GrayscaleLutSubstrate,
    ProceduralVariantConfig,
    ProceduralVariantError,
    compose_procedural_archive,
    compose_with_procedural_lut,
    derive_procedural_lut_replacement,
    pack_archive,
    parse_archive,
    predicted_archive_bytes_saved,
    predicted_delta_s,
    verify_procedural_lut_in_domain,
    verify_seed_mutation_changes_lut_bytes,
)


def _smoke_cfg() -> GrayscaleLutConfig:
    """Tiny config for synthetic-archive roundtrip tests."""
    return GrayscaleLutConfig(
        grayscale_downsample=4,
        decoder_hidden=8,
        decoder_blocks=2,
        embedding_dim=4,
        num_pairs=4,
        output_height=16,
        output_width=16,
    )


def _make_synthetic_glv1_archive_bytes(seed: int = 0xBEEF) -> bytes:
    """Build a tiny synthetic GLV1 archive for compose-roundtrip tests."""
    torch.manual_seed(seed)
    cfg = _smoke_cfg()
    sub = GrayscaleLutSubstrate(cfg)
    sd = sub.runtime_state_dict_for_archive()
    gs_u8 = sub.quantize_grayscale_for_archive()
    meta = {
        "decoder_hidden": cfg.decoder_hidden,
        "decoder_blocks": cfg.decoder_blocks,
        "synthetic": True,
        "seed": int(seed),
    }
    return pack_archive(
        decoder_state_dict=sd,
        grayscale=gs_u8,
        meta=meta,
        num_pairs=cfg.num_pairs,
        grayscale_downsample=cfg.grayscale_downsample,
        embedding_dim=cfg.embedding_dim,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
    )


# ---------------------------------------------------------------------------
# Module flag + canonical constants
# ---------------------------------------------------------------------------


def test_procedural_variant_module_available_flag_set() -> None:
    assert PROCEDURAL_VARIANT_AVAILABLE is True
    assert isinstance(PROCEDURAL_LUT_SENTINEL, bytes)
    assert PROCEDURAL_LUT_SENTINEL == b"GLPV"


def test_canonical_constants_match_task_spec() -> None:
    """Per task spec: LUT 256 B → 32 B seed → predicted ΔS = -0.000149."""
    assert PROCEDURAL_SEED_SIZE_BYTES == 32
    assert PROCEDURAL_LUT_BYTES_DEFAULT == 256
    assert CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT == "chroma_lut_replacement"


# ---------------------------------------------------------------------------
# Canonical equation #26 closed-form predictions
# ---------------------------------------------------------------------------


def test_predicted_archive_bytes_saved_canonical_256_minus_32() -> None:
    assert predicted_archive_bytes_saved() == 224
    assert predicted_archive_bytes_saved(lut_bytes=256, seed_bytes=32) == 224


def test_predicted_delta_s_canonical_matches_canonical_equation_26() -> None:
    """ΔS = -25 * 224 / 37_545_489 = -0.000149."""
    expected = -25.0 * 224 / 37_545_489
    assert predicted_delta_s() == pytest.approx(expected, rel=1e-9)
    # The task spec literal -0.000149 must round-match.
    assert round(predicted_delta_s(), 6) == round(expected, 6)
    assert predicted_delta_s() < 0  # negative = improvement


def test_predicted_delta_s_scales_linearly_with_lut_bytes() -> None:
    """Doubling N - K bytes -> double the (negative) ΔS magnitude."""
    delta_256 = predicted_delta_s(lut_bytes=256, seed_bytes=32)  # -25*224 / denom
    delta_512 = predicted_delta_s(lut_bytes=512, seed_bytes=32)  # -25*480 / denom
    # 480 / 224 ratio scales linearly.
    assert delta_512 == pytest.approx(delta_256 * (480.0 / 224.0), rel=1e-9)


def test_predicted_delta_s_zero_when_seed_geq_lut() -> None:
    assert predicted_delta_s(lut_bytes=32, seed_bytes=32) == 0.0
    assert predicted_delta_s(lut_bytes=16, seed_bytes=32) == 0.0
    assert predicted_archive_bytes_saved(lut_bytes=32, seed_bytes=32) == 0


# ---------------------------------------------------------------------------
# ProceduralVariantConfig invariants
# ---------------------------------------------------------------------------


def test_procedural_variant_config_rejects_short_seed() -> None:
    with pytest.raises(ProceduralVariantError, match="seed_bytes length"):
        ProceduralVariantConfig(seed_bytes=b"x" * 4)  # below 8-byte min


def test_procedural_variant_config_rejects_long_seed() -> None:
    with pytest.raises(ProceduralVariantError, match="seed_bytes length"):
        ProceduralVariantConfig(seed_bytes=b"x" * 300)  # above 256-byte max


def test_procedural_variant_config_rejects_invalid_generator_kind() -> None:
    with pytest.raises(ProceduralVariantError, match="generator_kind"):
        ProceduralVariantConfig(seed_bytes=bytes(range(32)), generator_kind="not_a_real_prng")


def test_procedural_variant_config_rejects_non_bytes_seed() -> None:
    with pytest.raises(ProceduralVariantError, match="bytes-like"):
        ProceduralVariantConfig(seed_bytes="not_bytes_string")  # type: ignore[arg-type]


def test_procedural_variant_config_accepts_canonical_defaults() -> None:
    cfg = ProceduralVariantConfig(seed_bytes=bytes(range(32)))
    assert cfg.canonical_equation_context == CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT
    assert cfg.dtype == PROCEDURAL_LUT_DTYPE_DEFAULT
    assert cfg.output_shape == (256,)
    assert cfg.generator_kind == "pcg64"


# ---------------------------------------------------------------------------
# Derivation determinism + sister generator-kind parity
# ---------------------------------------------------------------------------


def test_derive_procedural_lut_replacement_deterministic() -> None:
    """Same seed -> same LUT bytes (canonical equation #26 determinism)."""
    seed = bytes(range(32))
    lut_a = derive_procedural_lut_replacement(seed)
    lut_b = derive_procedural_lut_replacement(seed)
    np.testing.assert_array_equal(lut_a, lut_b)
    assert lut_a.shape == (256,)
    assert lut_a.dtype == np.uint8


def test_derive_procedural_lut_replacement_3_kind_parity() -> None:
    """All 3 generator kinds callable + produce different bytes for same seed."""
    seed = bytes(range(32))
    lut_xs = derive_procedural_lut_replacement(seed, generator_kind="xorshift")
    lut_lcg = derive_procedural_lut_replacement(seed, generator_kind="lcg")
    lut_pcg = derive_procedural_lut_replacement(seed, generator_kind="pcg64")
    # All three shapes match
    assert lut_xs.shape == lut_lcg.shape == lut_pcg.shape == (256,)
    # And produce structurally different outputs (different PRNG algorithms)
    assert not np.array_equal(lut_xs, lut_lcg)
    assert not np.array_equal(lut_lcg, lut_pcg)


# ---------------------------------------------------------------------------
# IN-DOMAIN check
# ---------------------------------------------------------------------------


def test_verify_procedural_lut_in_domain_canonical_passes() -> None:
    """chroma_lut_replacement is the canonical IN-DOMAIN context."""
    assert verify_procedural_lut_in_domain() is True
    assert verify_procedural_lut_in_domain("chroma_lut_replacement") is True


def test_verify_procedural_lut_in_domain_sister_dp1_context_passes() -> None:
    """DP1 sister context comma2k19_ood_derived_basis_replacement is also IN-DOMAIN."""
    assert (
        verify_procedural_lut_in_domain("comma2k19_ood_derived_basis_replacement") is True
    )


def test_verify_procedural_lut_in_domain_sister_vq_vae_context_passes() -> None:
    """VQ-VAE sister context intermediate_transform_quantizer is also IN-DOMAIN."""
    assert (
        verify_procedural_lut_in_domain("intermediate_transform_quantizer") is True
    )


def test_verify_procedural_lut_in_domain_wrong_context_fails() -> None:
    assert verify_procedural_lut_in_domain("not_a_real_context_token") is False


def test_verify_procedural_lut_in_domain_uses_slot_3_helper_if_available() -> None:
    """Smoke that slot 3 helper is called when present (no-op assertion)."""
    try:
        from tac.canonical_equations import validate_context_is_in_domain  # noqa: F401
    except ImportError:
        pytest.skip("slot 3 helper not yet landed; fallback path is exercised elsewhere")
    # If imported successfully, the verify function delegates to it.
    assert verify_procedural_lut_in_domain() is True


# ---------------------------------------------------------------------------
# Byte-mutation distinguishing-feature contract (Catalog #272)
# ---------------------------------------------------------------------------


def test_verify_seed_mutation_changes_lut_bytes_passes_for_canonical_seed() -> None:
    """Mutating any seed byte produces a different LUT (operational mechanism)."""
    seed = bytes(range(32))
    assert verify_seed_mutation_changes_lut_bytes(seed) is True


# ---------------------------------------------------------------------------
# Archive composition + sister regression
# ---------------------------------------------------------------------------


def test_compose_with_procedural_lut_extends_archive_with_envelope() -> None:
    """L0 scaffold: compose APPENDS envelope (no in-place LUT section to replace yet)."""
    original = _make_synthetic_glv1_archive_bytes()
    seed = bytes(range(32))
    new = compose_with_procedural_lut(original, seed)
    # Envelope is appended; new archive is strictly longer.
    assert len(new) > len(original)
    # Original GLV1 bytes preserved as the prefix.
    assert new.startswith(original)
    # Envelope starts with the canonical sentinel.
    envelope = new[len(original):]
    assert envelope.startswith(PROCEDURAL_LUT_SENTINEL)


def test_compose_with_procedural_lut_envelope_layout() -> None:
    """Envelope = SENTINEL(4) + LUT_BYTES(u16) + GEN_KIND_TAG(u8) + SEED_LEN(u16) + seed."""
    original = _make_synthetic_glv1_archive_bytes()
    seed = bytes(range(32))
    new = compose_with_procedural_lut(original, seed)
    envelope = new[len(original):]
    assert envelope[:4] == PROCEDURAL_LUT_SENTINEL
    lut_bytes_field, gen_kind_tag, seed_len = struct.unpack("<HBH", envelope[4:9])
    assert lut_bytes_field == 256
    assert gen_kind_tag == 2  # pcg64 default tag
    assert seed_len == 32
    assert envelope[9:9 + 32] == seed


def test_compose_with_procedural_lut_preserves_original_archive_parse() -> None:
    """Original archive bytes prefix still parses as canonical GLV1."""
    original = _make_synthetic_glv1_archive_bytes()
    seed = bytes(range(32))
    new = compose_with_procedural_lut(original, seed)
    # The original archive prefix still parses (current inflate is GLV1-aware
    # only; the appended envelope is invisible to canonical parser).
    arc_original = parse_archive(original)
    # Sister: the canonical parser ignores the appended bytes (the envelope
    # would surface as "trailing bytes" if the parser were stricter; per the
    # current parse_archive contract, it asserts pos == len(blob) so we
    # cannot reparse the composed archive as canonical GLV1 directly).
    # Test passes when the original bytes are still valid GLV1.
    assert arc_original.num_pairs == 4
    assert arc_original.schema_version == 1


def test_compose_procedural_archive_convenience_wrapper_matches_full_API() -> None:
    """Thin convenience wrapper delegates to compose_with_procedural_lut."""
    original = _make_synthetic_glv1_archive_bytes()
    seed = bytes(range(32))
    new_full = compose_with_procedural_lut(original, seed)
    new_thin = compose_procedural_archive(original, seed)
    assert new_full == new_thin


def test_compose_with_procedural_lut_byte_mutation_smoke_catalog_272() -> None:
    """Catalog #272: mutate one seed byte -> different envelope bytes."""
    original = _make_synthetic_glv1_archive_bytes()
    seed = bytes(range(32))
    mutated = bytearray(seed)
    mutated[0] = (mutated[0] + 1) & 0xFF
    new_orig = compose_with_procedural_lut(original, seed)
    new_mut = compose_with_procedural_lut(original, bytes(mutated))
    assert new_orig != new_mut
    # The original archive prefix is identical; only the envelope differs.
    orig_envelope = new_orig[len(original):]
    mut_envelope = new_mut[len(original):]
    assert orig_envelope != mut_envelope


def test_compose_with_procedural_lut_rejects_non_glv1_archive() -> None:
    """Bad archive bytes raise ValueError (sister discipline with VQ-VAE)."""
    bad = b"NOTGLV1" + b"\x00" * 100
    with pytest.raises((ValueError, ProceduralVariantError)):
        compose_with_procedural_lut(bad, bytes(range(32)))


def test_compose_with_procedural_lut_rejects_short_seed() -> None:
    """ProceduralVariantConfig.__post_init__ rejects short seed."""
    original = _make_synthetic_glv1_archive_bytes()
    with pytest.raises(ProceduralVariantError, match="seed_bytes length"):
        compose_with_procedural_lut(original, b"x" * 4)


def test_sister_grayscale_lut_base_substrate_archive_still_parses() -> None:
    """Sister regression: canonical pack_archive + parse_archive roundtrip unaffected."""
    original = _make_synthetic_glv1_archive_bytes()
    arc = parse_archive(original)
    assert arc.num_pairs == 4
    assert arc.grayscale.dtype == torch.uint8
    assert arc.grayscale.shape == (4, 1, 4, 4)  # num_pairs=4, H/D=16/4=4, W/D=16/4=4
    assert "decoder_hidden" in arc.meta
    assert arc.schema_version == 1
