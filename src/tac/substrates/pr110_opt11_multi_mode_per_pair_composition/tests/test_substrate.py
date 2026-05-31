# SPDX-License-Identifier: MIT
"""Dedicated tests for PR110-OPT-11 multi-mode-per-pair composition L0 SCAFFOLD.

Per CLAUDE.md NO FAKE IMPLEMENTATIONS non-negotiable 2026-05-30 + Slot EEE
fake-implementation audit anchor 2026-05-29: this test suite verifies ACTUAL
multi-mode composition behavior — NOT just canonical menu size constants. The
distinguishing-feature integration contract per Catalog #272 is empirically
validated via behavioral evidence (composed frame differs from both single-
mode outputs when both modes are non-identity).

NOT a tests-verify-constants-not-behavior suite per Slot EEE 5 forbidden
classes (Class 2).
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tac.substrates.pr110_opt11_multi_mode_per_pair_composition import (
    ARCHIVE_MAGIC,
    ARCHIVE_VERSION,
    CANONICAL_ORTHOGONAL_FAMILY_PAIRS,
    DEFAULT_MODES_PER_PAIR,
    DEFAULT_PR110_BASE_PAIRS,
    DEFAULT_SELECTOR_BITS_PER_MODE,
    OPT11MMP_HEADER_FMT,
    OPT11MMP_HEADER_LEN,
    PR110OPT11Config,
    PR110OPT11Result,
    apply_substrate_to_pr110_canonical,
    build_substrate_default_config,
    verify_canonical_multi_mode_composition,
)
from tac.substrates.pr110_opt11_multi_mode_per_pair_composition.archive_grammar import (
    pack_header,
    pack_selector_stream,
    unpack_header,
    unpack_selector_stream,
)
from tac.substrates.pr110_opt11_multi_mode_per_pair_composition.substrate import (
    CANONICAL_MODE_MENU,
    _apply_canonical_perturbation,
    _compose_two_modes_on_frame,
    _mode_indices_in_family,
)


# =========================================================================
# Canonical archive grammar contracts (Catalog #146 frozen-offset discipline)
# =========================================================================


def test_canonical_magic_is_8_bytes() -> None:
    assert ARCHIVE_MAGIC == b"OPT11MMP"
    assert len(ARCHIVE_MAGIC) == 8


def test_canonical_header_is_exactly_32_bytes() -> None:
    assert OPT11MMP_HEADER_LEN == 32
    assert struct.calcsize(OPT11MMP_HEADER_FMT) == 32


def test_pack_unpack_header_roundtrip() -> None:
    sha_prefix = bytes(range(16))
    packed = pack_header(
        version=ARCHIVE_VERSION,
        modes_per_pair=DEFAULT_MODES_PER_PAIR,
        selector_bits_per_mode=DEFAULT_SELECTOR_BITS_PER_MODE,
        family_pair_index=1,
        pr110_base_sha256_prefix=sha_prefix,
    )
    assert len(packed) == OPT11MMP_HEADER_LEN
    unpacked = unpack_header(packed)
    assert unpacked["magic"] == ARCHIVE_MAGIC
    assert unpacked["version"] == ARCHIVE_VERSION
    assert unpacked["modes_per_pair"] == DEFAULT_MODES_PER_PAIR
    assert unpacked["selector_bits_per_mode"] == DEFAULT_SELECTOR_BITS_PER_MODE
    assert unpacked["family_pair_index"] == 1
    assert unpacked["pr110_base_sha256_prefix"] == sha_prefix
    assert unpacked["reserved"] == b"\x00\x00\x00\x00"


def test_pack_header_rejects_wrong_sha_prefix_length() -> None:
    with pytest.raises(ValueError, match="exactly 16 bytes"):
        pack_header(
            version=1,
            modes_per_pair=2,
            selector_bits_per_mode=4,
            family_pair_index=0,
            pr110_base_sha256_prefix=b"\x00" * 8,  # WRONG: 8 bytes
        )


def test_unpack_header_rejects_wrong_magic() -> None:
    sha_prefix = bytes(range(16))
    packed = pack_header(
        version=1,
        modes_per_pair=2,
        selector_bits_per_mode=4,
        family_pair_index=0,
        pr110_base_sha256_prefix=sha_prefix,
    )
    # Corrupt the magic
    corrupted = b"NOTAMMP1" + packed[8:]
    with pytest.raises(ValueError, match="magic mismatch"):
        unpack_header(corrupted)


# =========================================================================
# Canonical selector stream packing (4-bit + >4-bit paths)
# =========================================================================


def test_pack_unpack_selector_stream_4bit_roundtrip() -> None:
    # K=16 → 4-bit selectors → 2 selectors per byte
    selectors = [(0, 1), (15, 14), (7, 8), (2, 3)]
    stream = pack_selector_stream(selectors, selector_bits_per_mode=4)
    # 4 pairs × 1 byte each (4-bit packed) = 4 bytes
    assert len(stream) == 4
    unpacked = unpack_selector_stream(
        stream, n_pairs=4, selector_bits_per_mode=4
    )
    assert unpacked == selectors


def test_pack_unpack_selector_stream_8bit_roundtrip() -> None:
    # K=256 → 8-bit selectors → 2 bytes per pair
    selectors = [(0, 255), (128, 64), (200, 1)]
    stream = pack_selector_stream(selectors, selector_bits_per_mode=8)
    assert len(stream) == 6  # 3 pairs × 2 bytes each
    unpacked = unpack_selector_stream(
        stream, n_pairs=3, selector_bits_per_mode=8
    )
    assert unpacked == selectors


def test_pack_selector_rejects_overflow_4bit() -> None:
    with pytest.raises(ValueError, match="exceeds 4-bit budget"):
        pack_selector_stream([(16, 0)], selector_bits_per_mode=4)
    with pytest.raises(ValueError, match="exceeds 4-bit budget"):
        pack_selector_stream([(0, 16)], selector_bits_per_mode=4)


def test_unpack_selector_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="stream length"):
        unpack_selector_stream(b"\x00", n_pairs=5, selector_bits_per_mode=4)


# =========================================================================
# Canonical mode menu + orthogonal family pair enumeration
# =========================================================================


def test_canonical_mode_menu_matches_wave_n34_22_modes_plus_identity() -> None:
    # Per Wave N+34 source: 22 distinct modes across 4 families + 1 identity
    families = {fam for fam, _mid in CANONICAL_MODE_MENU}
    assert "identity" in families
    assert "frame0_luma_bias" in families
    assert "frame0_blue_chroma" in families
    assert "frame0_rgb_bias" in families
    assert "frame0_roll" in families
    # The menu carries 22 perturbation modes (Wave N+34 anchor)
    non_identity = [m for f, m in CANONICAL_MODE_MENU if f != "identity"]
    assert len(non_identity) == 21  # 22 menu - 1 identity sentinel


def test_canonical_orthogonal_family_pairs_is_6_combinations() -> None:
    # 4 perturbation families choose 2 = 6 pairs
    assert len(CANONICAL_ORTHOGONAL_FAMILY_PAIRS) == 6
    # All entries are distinct families
    for f1, f2 in CANONICAL_ORTHOGONAL_FAMILY_PAIRS:
        assert f1 != f2
        assert f1 != "identity"
        assert f2 != "identity"


def test_mode_indices_in_family_returns_all_family_members() -> None:
    luma_modes = _mode_indices_in_family("frame0_luma_bias")
    assert len(luma_modes) == 6  # +1, +2, +4, -1, -2, -4
    chroma_modes = _mode_indices_in_family("frame0_blue_chroma")
    assert len(chroma_modes) == 3
    rgb_modes = _mode_indices_in_family("frame0_rgb_bias")
    assert len(rgb_modes) == 8
    roll_modes = _mode_indices_in_family("frame0_roll")
    assert len(roll_modes) == 4


# =========================================================================
# CANONICAL BEHAVIORAL TESTS (NO FAKE IMPLEMENTATIONS per Slot EEE)
# =========================================================================


def _make_test_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=256, size=(48, 64, 3), dtype=np.uint8)


def test_identity_mode_returns_unchanged_frame() -> None:
    # mode_idx=0 is identity (none)
    frame = _make_test_frame(42)
    out = _apply_canonical_perturbation(frame, mode_idx=0)
    assert np.array_equal(out, frame)


def test_luma_bias_actually_changes_R_channel() -> None:
    # frame0_luma_bias_+4 must actually add +4 to R channel
    frame = _make_test_frame(42)
    # Find a frame where R is in [0, 250] so +4 doesn't clip
    frame[..., 0] = 100  # constant R = 100
    luma_4_idx = next(
        i for i, (f, m) in enumerate(CANONICAL_MODE_MENU)
        if m == "frame0_luma_bias_+4"
    )
    out = _apply_canonical_perturbation(frame, mode_idx=luma_4_idx)
    # R should now be 104 everywhere (clipped if originally > 251)
    assert out[..., 0].mean() == pytest.approx(104.0)
    # G and B unchanged
    assert np.array_equal(out[..., 1], frame[..., 1])
    assert np.array_equal(out[..., 2], frame[..., 2])


def test_blue_chroma_actually_scales_B_channel() -> None:
    frame = _make_test_frame(42)
    frame[..., 2] = 100  # constant B = 100
    blue_2_idx = next(
        i for i, (f, m) in enumerate(CANONICAL_MODE_MENU)
        if m == "frame0_blue_chroma_amp_2"
    )
    out = _apply_canonical_perturbation(frame, mode_idx=blue_2_idx)
    # B should be ~ 100 * 1.10 = 110
    assert out[..., 2].mean() == pytest.approx(110.0, abs=1.0)
    # R and G unchanged
    assert np.array_equal(out[..., 0], frame[..., 0])
    assert np.array_equal(out[..., 1], frame[..., 1])


def test_rgb_bias_actually_applies_per_channel_offsets() -> None:
    frame = _make_test_frame(42)
    frame[..., :] = 100  # all channels = 100
    # frame0_rgb_bias_m4_p2_p2 → R -= 4, G += 2, B += 2
    rgb_idx = next(
        i for i, (f, m) in enumerate(CANONICAL_MODE_MENU)
        if m == "frame0_rgb_bias_m4_p2_p2"
    )
    out = _apply_canonical_perturbation(frame, mode_idx=rgb_idx)
    assert out[..., 0].mean() == pytest.approx(96.0)
    assert out[..., 1].mean() == pytest.approx(102.0)
    assert out[..., 2].mean() == pytest.approx(102.0)


def test_roll_actually_shifts_spatially() -> None:
    frame = _make_test_frame(42)
    # frame0_roll_h_1 → np.roll(shift=1, axis=1)
    h_1_idx = next(
        i for i, (f, m) in enumerate(CANONICAL_MODE_MENU)
        if m == "frame0_roll_h_1"
    )
    out = _apply_canonical_perturbation(frame, mode_idx=h_1_idx)
    expected = np.roll(frame, shift=1, axis=1)
    assert np.array_equal(out, expected)


def test_compose_two_modes_differs_from_both_single_mode_outputs() -> None:
    """Slot EEE NO FAKE IMPLEMENTATIONS canonical behavioral test.

    The substrate's distinguishing-feature integration contract per Catalog
    #272: composition of (mode_a, mode_b) MUST produce a frame different from
    BOTH single-mode outputs when both modes are non-identity and orthogonal-
    family. This empirically proves composition actually happens, not no-op.
    """
    frame = _make_test_frame(42)
    luma_4_idx = next(
        i for i, (f, m) in enumerate(CANONICAL_MODE_MENU)
        if m == "frame0_luma_bias_+4"
    )
    rgb_idx = next(
        i for i, (f, m) in enumerate(CANONICAL_MODE_MENU)
        if m == "frame0_rgb_bias_m4_p2_p2"
    )

    after_a = _apply_canonical_perturbation(frame, luma_4_idx)
    after_b = _apply_canonical_perturbation(frame, rgb_idx)
    composed = _compose_two_modes_on_frame(frame, luma_4_idx, rgb_idx)

    # Composition is non-identity:
    assert not np.array_equal(composed, frame)
    # Composition differs from EACH single-mode output:
    assert not np.array_equal(composed, after_a)
    assert not np.array_equal(composed, after_b)


def test_compose_with_identity_b_equals_single_mode_a() -> None:
    """Identity element invariant: A ∘ identity == A."""
    frame = _make_test_frame(42)
    luma_4_idx = next(
        i for i, (f, m) in enumerate(CANONICAL_MODE_MENU)
        if m == "frame0_luma_bias_+4"
    )
    after_a = _apply_canonical_perturbation(frame, luma_4_idx)
    composed = _compose_two_modes_on_frame(frame, luma_4_idx, 0)  # identity = 0
    assert np.array_equal(composed, after_a)


# =========================================================================
# Canonical config + result + entry point integration tests
# =========================================================================


def test_default_config_carries_canonical_constants() -> None:
    cfg = build_substrate_default_config()
    assert cfg.n_pairs == DEFAULT_PR110_BASE_PAIRS
    assert cfg.modes_per_pair == DEFAULT_MODES_PER_PAIR
    assert cfg.selector_bits_per_mode == DEFAULT_SELECTOR_BITS_PER_MODE
    assert 0 <= cfg.family_pair_index < len(CANONICAL_ORTHOGONAL_FAMILY_PAIRS)


def test_config_rejects_invalid_modes_per_pair() -> None:
    with pytest.raises(ValueError, match="modes_per_pair must be in"):
        PR110OPT11Config(modes_per_pair=1)
    with pytest.raises(ValueError, match="modes_per_pair must be in"):
        PR110OPT11Config(modes_per_pair=9)


def test_config_rejects_invalid_selector_bits() -> None:
    with pytest.raises(ValueError, match="selector_bits_per_mode must be in"):
        PR110OPT11Config(selector_bits_per_mode=0)
    with pytest.raises(ValueError, match="selector_bits_per_mode must be in"):
        PR110OPT11Config(selector_bits_per_mode=9)


def test_config_rejects_invalid_family_pair_index() -> None:
    with pytest.raises(ValueError, match="family_pair_index must be in"):
        PR110OPT11Config(family_pair_index=99)


def test_apply_substrate_returns_canonical_tier_a_result() -> None:
    """Catalog #341 Tier A canonical-routing-markers contract."""
    result = apply_substrate_to_pr110_canonical()
    assert isinstance(result, PR110OPT11Result)
    assert result.predicted_delta_adjustment == 0.0
    assert result.promotable is False
    assert result.axis_tag == "[predicted]"
    assert "DEFERRED" in result.verdict


def test_apply_substrate_produces_per_pair_selectors_for_all_pairs() -> None:
    cfg = PR110OPT11Config(n_pairs=600)
    result = apply_substrate_to_pr110_canonical(cfg)
    assert len(result.per_pair_selectors) == 600
    # All selectors are valid (within bit budget)
    max_value = (1 << cfg.selector_bits_per_mode) - 1
    for sel_a, sel_b in result.per_pair_selectors:
        assert 0 <= sel_a <= max_value
        assert 0 <= sel_b <= max_value


def test_apply_substrate_actually_invokes_canonical_helpers() -> None:
    """Slot EEE NO FAKE IMPLEMENTATIONS: helpers must ACTUALLY fire."""
    result = apply_substrate_to_pr110_canonical()
    expected_helpers = {
        "apply_canonical_perturbation_family_a",
        "apply_canonical_perturbation_family_b",
        "compose_two_modes_on_frame",
        "build_canonical_per_pair_selectors",
        "build_provenance_for_predicted",
    }
    assert set(result.canonical_helpers_invoked.keys()) == expected_helpers
    # All MUST be True (actually invoked, NOT just declared)
    for helper, invoked in result.canonical_helpers_invoked.items():
        assert invoked is True, f"helper {helper} was NOT actually invoked"


def test_apply_substrate_composition_produces_distinct_output() -> None:
    """Slot EEE NO FAKE Class 1: composition must actually change frames."""
    result = apply_substrate_to_pr110_canonical()
    evidence = result.composition_behavioral_evidence
    assert evidence["all_compositions_produced_distinct_output"] is True
    assert evidence["mean_sum_abs_delta_ab_vs_base"] > 0.0
    # The per-sample deltas carry actual numeric evidence
    for sample in evidence["per_sample_deltas"]:
        assert sample["sum_abs_delta_ab_vs_base"] >= 0.0


def test_apply_substrate_carries_canonical_provenance_per_catalog_323() -> None:
    result = apply_substrate_to_pr110_canonical()
    prov = result.canonical_provenance
    assert isinstance(prov, dict)
    assert "artifact_kind" in prov
    assert "score_claim_valid" in prov
    # Provenance artifact_kind is predicted_from_model per Catalog #323 + Tier A
    assert prov["artifact_kind"] == "predicted_from_model"
    # score_claim_valid is False for predicted (non-promotable)
    assert prov["score_claim_valid"] is False


def test_apply_substrate_carries_cross_reference_matrix() -> None:
    result = apply_substrate_to_pr110_canonical()
    xref = result.cross_reference_matrix
    # Must reference the 5 canonical anchors
    assert "wave_n34_analytical_investigator" in xref
    assert "task_1323_pending_operator_routable" in xref
    assert "sister_pr110_opt_7_l1_promotion" in xref
    assert "pair_component_rows_canonical_22_mode_menu" in xref
    assert "canonical_orthogonal_family_pairs_enumeration" in xref


def test_verify_canonical_multi_mode_composition_passes_clean_substrate() -> None:
    result = apply_substrate_to_pr110_canonical()
    verdict = verify_canonical_multi_mode_composition(result)
    assert verdict["all_invoked"] is True
    assert verdict["missing_helpers"] == []
    assert verdict["composition_distinct_output_verdict"] == "PASS"


def test_apply_substrate_with_explicit_canonical_base_frames() -> None:
    cfg = PR110OPT11Config(n_pairs=600)
    frames = np.zeros((8, 48, 64, 3), dtype=np.uint8)
    # Make frames distinct
    for i in range(8):
        frames[i] = i * 30
    result = apply_substrate_to_pr110_canonical(cfg, canonical_base_frames=frames)
    assert result.composition_behavioral_evidence["n_samples_composed"] == 8


def test_apply_substrate_rejects_wrong_frame_dtype() -> None:
    frames = np.zeros((4, 48, 64, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must be uint8"):
        apply_substrate_to_pr110_canonical(canonical_base_frames=frames)


def test_apply_substrate_rejects_wrong_frame_shape() -> None:
    frames = np.zeros((4, 48, 64), dtype=np.uint8)  # missing C
    with pytest.raises(ValueError, match="must be"):
        apply_substrate_to_pr110_canonical(canonical_base_frames=frames)


# =========================================================================
# Deterministic per-seed reproducibility (canonical L0 contract)
# =========================================================================


def test_substrate_is_deterministic_per_rng_seed() -> None:
    cfg1 = PR110OPT11Config(rng_seed=42)
    cfg2 = PR110OPT11Config(rng_seed=42)
    r1 = apply_substrate_to_pr110_canonical(cfg1)
    r2 = apply_substrate_to_pr110_canonical(cfg2)
    assert r1.per_pair_selectors == r2.per_pair_selectors


def test_substrate_differs_across_distinct_rng_seeds() -> None:
    r1 = apply_substrate_to_pr110_canonical(PR110OPT11Config(rng_seed=1))
    r2 = apply_substrate_to_pr110_canonical(PR110OPT11Config(rng_seed=2))
    # Per-pair selectors differ across seeds
    assert r1.per_pair_selectors != r2.per_pair_selectors


def test_substrate_differs_across_family_pair_indices() -> None:
    r0 = apply_substrate_to_pr110_canonical(PR110OPT11Config(family_pair_index=0))
    r1 = apply_substrate_to_pr110_canonical(PR110OPT11Config(family_pair_index=1))
    assert r0.family_pair != r1.family_pair
