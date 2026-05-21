# SPDX-License-Identifier: MIT
"""Tests for the nscs06 v8 chroma-LUT substrate L0 SCAFFOLD.

Per WAVE-3-NSCS06-V8-CHROMA-LUT-SUBSTRATE-BUILD 2026-05-21. Covers:

* Module + canonical contract import + SubstrateContract validation.
* Canonical equation #26 predictions
  (``predicted_archive_bytes_saved`` + ``predicted_delta_s``).
* ``ProceduralVariantConfig`` invariants (seed-size + dtype + generator_kind).
* Determinism of derivation (same seed -> same LUT bytes).
* Catalog #272 byte-mutation smoke
  (mutating any seed byte -> different derived LUT bytes).
* Canonical equation #26 IN-DOMAIN context membership.
* CH08 v1 inline-LUT pack/parse roundtrip.
* CH08 v2 procedural-seed pack/parse roundtrip + empirical bytes-saved
  matches canonical-equation prediction.
* End-to-end inflate roundtrip (CH08 v1 + CH08 v2 produce contest .raw files).
* Catalog #205 device-fork (select_inflate_device rejects "mps", accepts auto/cpu/cuda).
* ``build_chroma_lut_from_ground_truth`` produces deterministic (levels, classes, 3) uint8 LUT.
* ``lookup_rgb_via_chroma_lut`` per-pixel lookup correctness.
* Catalog #240 recipe-vs-trainer-state consistency (research_only=True; cost_band_epochs=1).
* Catalog #220 sentinel: archive_bytes_added <= 1 KB AND schema scaffold-deferred.
* Catalog #161 degenerate-range guard (pack rejects num_pairs=0).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tac.substrates.nscs06_v8_chroma_lut import (
    CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
    CH08_HEADER_SIZE,
    CH08_MAGIC,
    CH08_SCHEMA_VERSION_INLINE_LUT,
    CH08_SCHEMA_VERSION_PROCEDURAL_SEED,
    CHROMA_LUT_BYTES_DEFAULT,
    GENERATOR_KIND_TAG,
    GRAYSCALE_LEVELS_DEFAULT,
    NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT,
    NUM_SEGNET_CLASSES,
    Nscs06V8Archive,
    Nscs06V8ChromaLutConfig,
    POSE_DIMS,
    PROCEDURAL_LUT_SENTINEL,
    PROCEDURAL_SEED_SIZE_BYTES,
    PROCEDURAL_VARIANT_AVAILABLE,
    ProceduralVariantConfig,
    ProceduralVariantError,
    build_chroma_lut_from_ground_truth,
    derive_procedural_chroma_lut_replacement,
    inflate_one_video,
    lookup_rgb_via_chroma_lut,
    pack_archive,
    parse_archive,
    predicted_archive_bytes_saved,
    predicted_delta_s,
    select_inflate_device,
    verify_procedural_lut_in_domain,
    verify_seed_mutation_changes_lut_bytes,
)


# ---------------------------------------------------------------------------
# Canonical constants + import
# ---------------------------------------------------------------------------


def test_procedural_variant_available_true_at_landing() -> None:
    assert PROCEDURAL_VARIANT_AVAILABLE is True


def test_canonical_constants_pinned() -> None:
    assert CHROMA_LUT_BYTES_DEFAULT == 4096
    assert PROCEDURAL_SEED_SIZE_BYTES == 32
    assert GRAYSCALE_LEVELS_DEFAULT == 16
    assert NUM_SEGNET_CLASSES == 5
    assert POSE_DIMS == 6
    assert CH08_MAGIC == b"CH08"
    assert CH08_SCHEMA_VERSION_INLINE_LUT == 1
    assert CH08_SCHEMA_VERSION_PROCEDURAL_SEED == 2
    assert CH08_HEADER_SIZE == 35
    assert PROCEDURAL_LUT_SENTINEL == b"NV8C"
    assert CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT == "nscs06_v8_chroma_lut"


def test_generator_kind_tag_canonical_three_kinds() -> None:
    assert set(GENERATOR_KIND_TAG) == {"xorshift", "lcg", "pcg64"}
    assert GENERATOR_KIND_TAG["pcg64"] == 2


# ---------------------------------------------------------------------------
# Canonical equation #26 closed-form predictions
# ---------------------------------------------------------------------------


def test_predicted_archive_bytes_saved_default() -> None:
    assert predicted_archive_bytes_saved() == CHROMA_LUT_BYTES_DEFAULT - PROCEDURAL_SEED_SIZE_BYTES
    assert predicted_archive_bytes_saved() == 4064


def test_predicted_delta_s_canonical_value() -> None:
    """Closed-form ΔS == -25 * (4096 - 32) / 37_545_489 ≈ -0.002706."""
    expected = -25.0 * 4064 / 37_545_489
    assert abs(predicted_delta_s() - expected) < 1e-9
    assert abs(predicted_delta_s() - (-0.002706)) < 1e-4


def test_predicted_savings_zero_when_seed_larger_than_lut() -> None:
    assert predicted_archive_bytes_saved(lut_bytes=10, seed_bytes=20) == 0
    assert predicted_delta_s(lut_bytes=10, seed_bytes=20) == 0.0


# ---------------------------------------------------------------------------
# Canonical equation #26 IN-DOMAIN context
# ---------------------------------------------------------------------------


def test_canonical_in_domain_default_passes() -> None:
    assert verify_procedural_lut_in_domain() is True


def test_canonical_in_domain_explicit_v8_context_passes() -> None:
    assert verify_procedural_lut_in_domain("nscs06_v8_chroma_lut") is True


def test_canonical_in_domain_excluded_context_returns_false() -> None:
    assert (
        verify_procedural_lut_in_domain(
            "direct_dwt_detail_subband_byte_substitution"
        )
        is False
    )


# ---------------------------------------------------------------------------
# ProceduralVariantConfig invariants
# ---------------------------------------------------------------------------


def test_procedural_variant_config_default_construction() -> None:
    cfg = ProceduralVariantConfig(seed_bytes=b"\x00" * 32)
    assert cfg.chroma_lut_bytes == CHROMA_LUT_BYTES_DEFAULT
    assert cfg.dtype == np.dtype(np.uint8)
    assert cfg.generator_kind == "pcg64"
    assert cfg.canonical_equation_context == CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT


def test_procedural_variant_config_rejects_short_seed() -> None:
    with pytest.raises(ProceduralVariantError):
        ProceduralVariantConfig(seed_bytes=b"\x00" * 4)


def test_procedural_variant_config_rejects_long_seed() -> None:
    with pytest.raises(ProceduralVariantError):
        ProceduralVariantConfig(seed_bytes=b"\x00" * 512)


def test_procedural_variant_config_rejects_non_bytes_seed() -> None:
    with pytest.raises(ProceduralVariantError):
        ProceduralVariantConfig(seed_bytes="not-bytes")  # type: ignore[arg-type]


def test_procedural_variant_config_rejects_unknown_generator_kind() -> None:
    with pytest.raises(ProceduralVariantError):
        ProceduralVariantConfig(seed_bytes=b"\x00" * 32, generator_kind="mersenne")


# ---------------------------------------------------------------------------
# Determinism of derivation
# ---------------------------------------------------------------------------


def test_derive_procedural_lut_same_seed_same_bytes() -> None:
    seed = b"abcdefgh" * 4
    lut_a = derive_procedural_chroma_lut_replacement(seed)
    lut_b = derive_procedural_chroma_lut_replacement(seed)
    assert np.array_equal(lut_a, lut_b)


def test_derive_procedural_lut_shape_and_dtype() -> None:
    lut = derive_procedural_chroma_lut_replacement(b"\x00" * 32)
    assert lut.shape == (GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3)
    assert lut.dtype == np.uint8


def test_derive_procedural_lut_all_three_generator_kinds_callable() -> None:
    seed = b"x" * 32
    for kind in ("xorshift", "lcg", "pcg64"):
        lut = derive_procedural_chroma_lut_replacement(seed, generator_kind=kind)
        assert lut.shape == (GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3)


# ---------------------------------------------------------------------------
# Catalog #272 byte-mutation smoke
# ---------------------------------------------------------------------------


def test_byte_mutation_changes_lut_bytes() -> None:
    seed_a = b"\x00" * 32
    seed_b = b"\x01" + b"\x00" * 31
    assert verify_seed_mutation_changes_lut_bytes(seed_a, seed_b) is True


def test_byte_mutation_smoke_rejects_identical_seeds() -> None:
    seed = b"\x00" * 32
    with pytest.raises(ProceduralVariantError):
        verify_seed_mutation_changes_lut_bytes(seed, seed)


# ---------------------------------------------------------------------------
# CH08 v1 inline-LUT pack/parse roundtrip
# ---------------------------------------------------------------------------


def _smoke_pose_grayscale(num_pairs: int = 4, gh: int = 4, gw: int = 6) -> tuple[bytes, bytes]:
    rng = np.random.RandomState(42)
    pose = rng.randint(0, 256, size=num_pairs * POSE_DIMS, dtype=np.uint8).tobytes()
    grayscale = rng.randint(0, 256, size=num_pairs * gh * gw, dtype=np.uint8).tobytes()
    return pose, grayscale


def test_pack_parse_v1_inline_lut_roundtrip() -> None:
    pose, grayscale = _smoke_pose_grayscale()
    rng = np.random.RandomState(42)
    lut = rng.randint(
        0, 256, size=(GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3), dtype=np.uint8
    )
    blob = pack_archive(
        num_pairs=4,
        grayscale_h=4,
        grayscale_w=6,
        output_height=16,
        output_width=24,
        pose_bytes=pose,
        grayscale_bytes=grayscale,
        chroma_lut=lut,
    )
    arc = parse_archive(blob)
    assert arc.schema_version == 1
    assert arc.num_pairs == 4
    assert arc.chroma_lut is not None
    assert np.array_equal(arc.chroma_lut, lut)
    assert arc.chroma_seed is None
    assert arc.generator_kind is None
    assert arc.pose_bytes == pose
    assert arc.grayscale_bytes == grayscale


def test_pack_v1_rejects_when_both_lut_and_seed_supplied() -> None:
    pose, grayscale = _smoke_pose_grayscale()
    lut = np.zeros((GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        pack_archive(
            num_pairs=4, grayscale_h=4, grayscale_w=6,
            output_height=16, output_width=24,
            pose_bytes=pose, grayscale_bytes=grayscale,
            chroma_lut=lut, chroma_seed=b"\x00" * 32,
        )


def test_pack_rejects_when_neither_lut_nor_seed() -> None:
    pose, grayscale = _smoke_pose_grayscale()
    with pytest.raises(ValueError):
        pack_archive(
            num_pairs=4, grayscale_h=4, grayscale_w=6,
            output_height=16, output_width=24,
            pose_bytes=pose, grayscale_bytes=grayscale,
        )


def test_pack_v1_rejects_chroma_lut_wrong_shape() -> None:
    pose, grayscale = _smoke_pose_grayscale()
    lut = np.zeros((10, 5, 3), dtype=np.uint8)  # wrong levels dim
    with pytest.raises(ValueError):
        pack_archive(
            num_pairs=4, grayscale_h=4, grayscale_w=6,
            output_height=16, output_width=24,
            pose_bytes=pose, grayscale_bytes=grayscale,
            chroma_lut=lut,
        )


# ---------------------------------------------------------------------------
# CH08 v2 procedural-seed pack/parse roundtrip
# ---------------------------------------------------------------------------


def test_pack_parse_v2_procedural_seed_roundtrip() -> None:
    pose, grayscale = _smoke_pose_grayscale()
    seed = b"\x01" * PROCEDURAL_SEED_SIZE_BYTES
    blob = pack_archive(
        num_pairs=4,
        grayscale_h=4,
        grayscale_w=6,
        output_height=16,
        output_width=24,
        pose_bytes=pose,
        grayscale_bytes=grayscale,
        chroma_seed=seed,
        generator_kind="pcg64",
    )
    arc = parse_archive(blob)
    assert arc.schema_version == 2
    assert arc.chroma_lut is None
    assert arc.chroma_seed == seed
    assert arc.generator_kind == "pcg64"


def test_pack_v2_rejects_wrong_size_seed() -> None:
    pose, grayscale = _smoke_pose_grayscale()
    with pytest.raises(ValueError):
        pack_archive(
            num_pairs=4, grayscale_h=4, grayscale_w=6,
            output_height=16, output_width=24,
            pose_bytes=pose, grayscale_bytes=grayscale,
            chroma_seed=b"\x00" * 8,  # too short for PROCEDURAL_SEED_SIZE_BYTES=32
        )


def test_pack_v2_rejects_unknown_generator_kind() -> None:
    pose, grayscale = _smoke_pose_grayscale()
    with pytest.raises(ValueError):
        pack_archive(
            num_pairs=4, grayscale_h=4, grayscale_w=6,
            output_height=16, output_width=24,
            pose_bytes=pose, grayscale_bytes=grayscale,
            chroma_seed=b"\x00" * 32, generator_kind="mersenne",
        )


def test_v2_blob_smaller_than_v1_by_canonical_savings() -> None:
    """Empirical bytes-saved IS the canonical equation #26 prediction."""
    pose, grayscale = _smoke_pose_grayscale()
    rng = np.random.RandomState(7)
    lut = rng.randint(
        0, 256, size=(GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3), dtype=np.uint8
    )
    blob_v1 = pack_archive(
        num_pairs=4, grayscale_h=4, grayscale_w=6,
        output_height=16, output_width=24,
        pose_bytes=pose, grayscale_bytes=grayscale,
        chroma_lut=lut,
    )
    blob_v2 = pack_archive(
        num_pairs=4, grayscale_h=4, grayscale_w=6,
        output_height=16, output_width=24,
        pose_bytes=pose, grayscale_bytes=grayscale,
        chroma_seed=b"\x00" * PROCEDURAL_SEED_SIZE_BYTES,
    )
    empirical_savings = len(blob_v1) - len(blob_v2)
    predicted_savings = predicted_archive_bytes_saved()
    assert empirical_savings == predicted_savings == 4064


# ---------------------------------------------------------------------------
# End-to-end inflate roundtrip
# ---------------------------------------------------------------------------


def _inflate_one(blob: bytes, num_pairs: int, out_h: int, out_w: int) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out_stem = Path(tmp) / "video0"
        raw = inflate_one_video(blob, out_stem)
        return len(raw.read_bytes())


def test_inflate_v1_emits_canonical_raw_bytes() -> None:
    pose, grayscale = _smoke_pose_grayscale()
    rng = np.random.RandomState(1)
    lut = rng.randint(
        0, 256, size=(GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3), dtype=np.uint8
    )
    blob = pack_archive(
        num_pairs=4, grayscale_h=4, grayscale_w=6,
        output_height=16, output_width=24,
        pose_bytes=pose, grayscale_bytes=grayscale,
        chroma_lut=lut,
    )
    raw_bytes = _inflate_one(blob, 4, 16, 24)
    expected = 4 * 2 * 16 * 24 * 3
    assert raw_bytes == expected


def test_inflate_v2_emits_canonical_raw_bytes() -> None:
    pose, grayscale = _smoke_pose_grayscale()
    seed = b"v8seed!!" * 4
    blob = pack_archive(
        num_pairs=4, grayscale_h=4, grayscale_w=6,
        output_height=16, output_width=24,
        pose_bytes=pose, grayscale_bytes=grayscale,
        chroma_seed=seed,
    )
    raw_bytes = _inflate_one(blob, 4, 16, 24)
    expected = 4 * 2 * 16 * 24 * 3
    assert raw_bytes == expected


def test_inflate_v1_and_v2_byte_stable_with_matching_lut() -> None:
    """Catalog #272 byte-mutation distinguishing-feature: v1 with explicit LUT
    matches v2 with the seed that derives the same LUT bytes-for-bytes."""
    pose, grayscale = _smoke_pose_grayscale()
    seed = b"\x42" * 32
    derived = derive_procedural_chroma_lut_replacement(seed)
    blob_v1 = pack_archive(
        num_pairs=4, grayscale_h=4, grayscale_w=6,
        output_height=16, output_width=24,
        pose_bytes=pose, grayscale_bytes=grayscale,
        chroma_lut=derived,
    )
    blob_v2 = pack_archive(
        num_pairs=4, grayscale_h=4, grayscale_w=6,
        output_height=16, output_width=24,
        pose_bytes=pose, grayscale_bytes=grayscale,
        chroma_seed=seed,
    )
    with tempfile.TemporaryDirectory() as tmp:
        out_v1 = Path(tmp) / "v1"
        out_v2 = Path(tmp) / "v2"
        raw_v1 = inflate_one_video(blob_v1, out_v1).read_bytes()
        raw_v2 = inflate_one_video(blob_v2, out_v2).read_bytes()
        assert raw_v1 == raw_v2, (
            "v1 inline LUT and v2 procedural-seed must produce identical rendered "
            "frames when v2 seed derives the same LUT bytes as v1 inline (canonical "
            "equation #26 byte-identity invariant)"
        )


# ---------------------------------------------------------------------------
# Catalog #272 byte-mutation distinguishing-feature contract
# ---------------------------------------------------------------------------


def test_distinct_seeds_produce_distinct_rendered_frames() -> None:
    pose, grayscale = _smoke_pose_grayscale()
    blob_a = pack_archive(
        num_pairs=4, grayscale_h=4, grayscale_w=6,
        output_height=16, output_width=24,
        pose_bytes=pose, grayscale_bytes=grayscale,
        chroma_seed=b"\x00" * 32,
    )
    blob_b = pack_archive(
        num_pairs=4, grayscale_h=4, grayscale_w=6,
        output_height=16, output_width=24,
        pose_bytes=pose, grayscale_bytes=grayscale,
        chroma_seed=b"\x01" * 32,
    )
    with tempfile.TemporaryDirectory() as tmp:
        raw_a = inflate_one_video(blob_a, Path(tmp) / "a").read_bytes()
        raw_b = inflate_one_video(blob_b, Path(tmp) / "b").read_bytes()
        assert raw_a != raw_b, (
            "Catalog #272: distinct procedural seeds MUST produce distinct "
            "rendered frames (byte-mutation distinguishing-feature contract)"
        )


# ---------------------------------------------------------------------------
# Catalog #205 inflate-device selector
# ---------------------------------------------------------------------------


def test_select_inflate_device_default_auto() -> None:
    # Force unset
    os.environ.pop("PACT_INFLATE_DEVICE", None)
    assert select_inflate_device() == "auto"


def test_select_inflate_device_cpu() -> None:
    os.environ["PACT_INFLATE_DEVICE"] = "cpu"
    try:
        assert select_inflate_device() == "cpu"
    finally:
        os.environ.pop("PACT_INFLATE_DEVICE", None)


def test_select_inflate_device_rejects_mps() -> None:
    os.environ["PACT_INFLATE_DEVICE"] = "mps"
    try:
        with pytest.raises(RuntimeError, match="MPS auth eval is NOISE"):
            select_inflate_device()
    finally:
        os.environ.pop("PACT_INFLATE_DEVICE", None)


def test_select_inflate_device_rejects_invalid() -> None:
    os.environ["PACT_INFLATE_DEVICE"] = "tpu"
    try:
        with pytest.raises(RuntimeError):
            select_inflate_device()
    finally:
        os.environ.pop("PACT_INFLATE_DEVICE", None)


# ---------------------------------------------------------------------------
# build_chroma_lut_from_ground_truth
# ---------------------------------------------------------------------------


def test_build_chroma_lut_from_ground_truth_shape() -> None:
    rng = np.random.RandomState(7)
    rgb = rng.randint(0, 256, size=(4, 3, 8, 12), dtype=np.uint8)
    cls = rng.randint(0, NUM_SEGNET_CLASSES, size=(4, 8, 12), dtype=np.uint8)
    lut = build_chroma_lut_from_ground_truth(rgb, cls)
    assert lut.shape == (GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3)
    assert lut.dtype == np.uint8


def test_build_chroma_lut_rejects_non_uint8_rgb() -> None:
    rgb = np.zeros((4, 3, 8, 12), dtype=np.float32)
    cls = np.zeros((4, 8, 12), dtype=np.uint8)
    with pytest.raises(ValueError):
        build_chroma_lut_from_ground_truth(rgb, cls)


def test_build_chroma_lut_empty_class_returns_fallback() -> None:
    """When a class has zero pixels, the per-class GLOBAL median fills the bins."""
    rgb = np.zeros((1, 3, 4, 4), dtype=np.uint8)
    rgb[:, :, 2, :] = 200  # half the pixels have RGB=(200, 200, 200)
    cls = np.zeros((1, 4, 4), dtype=np.uint8)  # only class 0 present
    cls[0, 0, 0] = 4  # one pixel as class 4 (rare)
    lut = build_chroma_lut_from_ground_truth(rgb, cls)
    # Classes 1-3 have ZERO pixels; their bins should be fallback (128).
    for c in (1, 2, 3):
        for lvl in range(GRAYSCALE_LEVELS_DEFAULT):
            assert tuple(lut[lvl, c]) == (128, 128, 128)


# ---------------------------------------------------------------------------
# lookup_rgb_via_chroma_lut
# ---------------------------------------------------------------------------


def test_lookup_rgb_via_chroma_lut_shape_dtype() -> None:
    gray = np.array([[0, 64, 128, 255]], dtype=np.uint8)
    cls = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    lut = np.zeros((GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3), dtype=np.uint8)
    out = lookup_rgb_via_chroma_lut(gray, cls, lut)
    assert out.shape == (1, 4, 3)
    assert out.dtype == np.uint8


def test_lookup_rgb_via_chroma_lut_correctness() -> None:
    """Verify the LUT lookup picks the right (level, class, channel) entry."""
    lut = np.zeros((GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3), dtype=np.uint8)
    # Set a distinctive value at (level=4, class=2)
    lut[4, 2] = (123, 45, 67)
    # gray=72 -> level 72//16 = 4; cls=2 -> class 2.
    gray = np.array([[72]], dtype=np.uint8)
    cls = np.array([[2]], dtype=np.uint8)
    out = lookup_rgb_via_chroma_lut(gray, cls, lut)
    assert tuple(out[0, 0]) == (123, 45, 67)


def test_lookup_rgb_rejects_shape_mismatch() -> None:
    gray = np.zeros((4, 4), dtype=np.uint8)
    cls = np.zeros((4, 5), dtype=np.uint8)  # mismatch
    lut = np.zeros((GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        lookup_rgb_via_chroma_lut(gray, cls, lut)


# ---------------------------------------------------------------------------
# Nscs06V8ChromaLutConfig invariants
# ---------------------------------------------------------------------------


def test_v8_config_defaults() -> None:
    cfg = Nscs06V8ChromaLutConfig()
    assert cfg.grayscale_levels == GRAYSCALE_LEVELS_DEFAULT
    assert cfg.num_segnet_classes == NUM_SEGNET_CLASSES
    assert cfg.chroma_lut_bytes == CHROMA_LUT_BYTES_DEFAULT
    assert cfg.chroma_lut_shape == (16, 5, 3)


def test_v8_config_rejects_lut_bytes_below_dense_minimum() -> None:
    with pytest.raises(ValueError):
        Nscs06V8ChromaLutConfig(chroma_lut_bytes=100)  # below 16*5*3=240


def test_v8_config_rejects_out_of_range_grayscale_levels() -> None:
    with pytest.raises(ValueError):
        Nscs06V8ChromaLutConfig(grayscale_levels=0)
    with pytest.raises(ValueError):
        Nscs06V8ChromaLutConfig(grayscale_levels=300)


# ---------------------------------------------------------------------------
# SubstrateContract validation (Catalog #241/#242)
# ---------------------------------------------------------------------------


def test_substrate_contract_validates() -> None:
    contract = NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT
    assert contract.id == "nscs06_v8_chroma_lut"
    assert contract.lane_id == "lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521"
    assert contract.score_improvement_mechanism_status == "SCAFFOLD_DEFERRED_INTEGRATION"
    # Catalog #220 sentinel: archive_bytes_added <= 1 KB AND scaffold-deferred.
    assert contract.archive_bytes_added is not None
    assert "32 bytes" in contract.archive_bytes_added
    # Catalog #240: research_only AND smoke_only (NOT contest-promotable yet).
    assert contract.recipe_research_only is True
    assert contract.recipe_smoke_only is True
    # Catalog #125 hook wire-in declaration
    assert contract.hook_pareto_constraint == "rate_distortion_v1"
    assert contract.hook_continual_learning_anchor_kind == "cuda_only"


def test_substrate_contract_target_modes_research_only() -> None:
    contract = NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT
    assert "research_substrate" in contract.target_modes
    # NOT contest_one_video_replay until per-substrate symposium per Catalog #325
    assert "contest_one_video_replay" not in contract.target_modes


# ---------------------------------------------------------------------------
# Parser sanity
# ---------------------------------------------------------------------------


def test_parse_rejects_bad_magic() -> None:
    blob = b"BAD!" + b"\x00" * (CH08_HEADER_SIZE - 4)
    with pytest.raises(ValueError):
        parse_archive(blob)


def test_parse_rejects_short_blob() -> None:
    with pytest.raises(ValueError):
        parse_archive(b"")


def test_parse_rejects_unsupported_version() -> None:
    pose, grayscale = _smoke_pose_grayscale()
    blob = pack_archive(
        num_pairs=4, grayscale_h=4, grayscale_w=6,
        output_height=16, output_width=24,
        pose_bytes=pose, grayscale_bytes=grayscale,
        chroma_seed=b"\x00" * 32,
    )
    # Mutate version byte to 99
    bad = bytearray(blob)
    bad[4] = 99
    with pytest.raises(ValueError):
        parse_archive(bytes(bad))
