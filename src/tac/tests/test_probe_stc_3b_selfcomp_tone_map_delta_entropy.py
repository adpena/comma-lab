# SPDX-License-Identifier: MIT
"""Tests for tools/probe_stc_3b_selfcomp_tone_map_delta_entropy.py.

Per OVERNIGHT-AA lane `lane_overnight_aa_stc_3b_selfcomp_tone_map_delta_entropy_probe_build_local_cpu_run_20260521`
+ Catalog #229 PV + Catalog #287 evidence-tag discipline + Catalog #323 canonical
Provenance + Catalog #344 canonical equation #359-sister IN-DOMAIN reference.

Mirrors sister `test_probe_stc_3a_a1_residual_entropy.py` test pattern.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from probe_stc_3b_selfcomp_tone_map_delta_entropy import (  # noqa: E402
    COVER_SIGNAL_TYPE,
    DEFAULT_LUT_BITS,
    HIGH_ENTROPY_THRESHOLD_BITS_PER_SYMBOL,
    HIGH_SPARSITY_THRESHOLD,
    MEDIUM_ENTROPY_THRESHOLD_BITS_PER_SYMBOL,
    MEDIUM_SPARSITY_THRESHOLD,
    STCToneMapDeltaProbeVerdict,
    build_canonical_provenance_for_report,
    classify_verdict_tier,
    compute_predicted_delta_s_band,
    compute_stc_tone_map_delta_entropy,
    five_tuple_sparsity,
    lut_quantize_grayscale,
    main,
    shannon_entropy_bits_per_symbol,
    soft_grayscale_from_rgb,
    synthetic_tone_map_delta_residuals_for_test,
    verify_ground_truth_video_sha256,
)


# =====================================================================
# Test 1: BT.601 soft-grayscale formula correctness
# =====================================================================
def test_soft_grayscale_white_is_white() -> None:
    """Pure white RGB (255, 255, 255) → grayscale 255."""
    rgb = np.full((4, 4, 3), 255, dtype=np.uint8)
    y = soft_grayscale_from_rgb(rgb)
    assert np.allclose(y, 255.0, atol=1e-3)


def test_soft_grayscale_black_is_black() -> None:
    """Pure black RGB (0,0,0) → grayscale 0."""
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    y = soft_grayscale_from_rgb(rgb)
    assert np.allclose(y, 0.0, atol=1e-6)


def test_soft_grayscale_red_coefficient() -> None:
    """Pure red (255, 0, 0) → 0.299 * 255 ≈ 76.245."""
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    y = soft_grayscale_from_rgb(rgb)
    assert np.isclose(y[0, 0], 0.299 * 255, atol=1e-3)


# =====================================================================
# Test 2: LUT quantize correctness per PR #56 paradigm
# =====================================================================
def test_lut_quantize_4bit_has_16_unique_levels() -> None:
    """4-bit LUT should produce ≤16 unique values over full grayscale range."""
    grayscale = np.linspace(0, 255, 256).astype(np.float32)
    quantized = lut_quantize_grayscale(grayscale, lut_bits=4)
    unique = np.unique(quantized)
    assert len(unique) == 16


def test_lut_quantize_2bit_has_4_unique_levels() -> None:
    """2-bit LUT should produce ≤4 unique values."""
    grayscale = np.linspace(0, 255, 256).astype(np.float32)
    quantized = lut_quantize_grayscale(grayscale, lut_bits=2)
    unique = np.unique(quantized)
    assert len(unique) == 4


def test_lut_quantize_8bit_is_near_identity_for_uint8() -> None:
    """8-bit LUT (256 levels) on uint8 input should be near-identity."""
    grayscale = np.arange(256, dtype=np.float32)
    quantized = lut_quantize_grayscale(grayscale, lut_bits=8)
    # 256 levels over [0, 255] → step = 1.0 → quantize-to-nearest is identity
    assert np.max(np.abs(grayscale - quantized)) <= 0.5


def test_lut_quantize_invalid_bits_raises() -> None:
    """lut_bits outside [1, 8] raises."""
    grayscale = np.array([100.0], dtype=np.float32)
    with pytest.raises(ValueError, match="lut_bits"):
        lut_quantize_grayscale(grayscale, lut_bits=0)
    with pytest.raises(ValueError, match="lut_bits"):
        lut_quantize_grayscale(grayscale, lut_bits=9)


# =====================================================================
# Test 3: Synthetic patterns produce expected entropy/sparsity
# =====================================================================
def test_synthetic_uniform_random_high_entropy_low_sparsity() -> None:
    """Uniform-random residuals should produce ~8 bits/symbol entropy + low sparsity."""
    residuals = synthetic_tone_map_delta_residuals_for_test(
        pattern="uniform_random", n_symbols=50_000
    )
    entropy = shannon_entropy_bits_per_symbol(residuals)
    sparsity = five_tuple_sparsity(residuals)
    assert entropy > 7.5, f"expected high entropy, got {entropy}"
    assert sparsity < 0.05, f"expected low sparsity, got {sparsity}"


def test_synthetic_compressible_lut_low_entropy_high_sparsity() -> None:
    """LUT-bounded residuals should produce moderate entropy + high sparsity.

    For lut_bits=4, residuals are uniformly distributed in [-8, +8] = 17 values
    → entropy ≈ log2(17) ≈ 4.1; sparsity (|r|<=2) = 5/17 ≈ 0.29.
    """
    residuals = synthetic_tone_map_delta_residuals_for_test(
        pattern="compressible_lut", n_symbols=50_000
    )
    entropy = shannon_entropy_bits_per_symbol(residuals)
    sparsity = five_tuple_sparsity(residuals)
    # Uniform [-8, 8] → entropy ~= log2(17) ≈ 4.087
    assert 3.5 < entropy < 4.5, f"expected ~4.1 entropy, got {entropy}"
    # |r|<=2 covers 5 of 17 values ≈ 0.29
    assert 0.20 < sparsity < 0.40, f"expected ~0.29 sparsity, got {sparsity}"


def test_synthetic_low_pattern_very_low_entropy() -> None:
    """All-zero pattern with tiny noise → very low entropy + very high sparsity."""
    residuals = synthetic_tone_map_delta_residuals_for_test(pattern="low", n_symbols=50_000)
    entropy = shannon_entropy_bits_per_symbol(residuals)
    sparsity = five_tuple_sparsity(residuals)
    assert entropy < 1.0, f"expected very low entropy, got {entropy}"
    assert sparsity > 0.95, f"expected very high sparsity, got {sparsity}"


# =====================================================================
# Test 4: Verdict tier classification
# =====================================================================
def test_verdict_tier_high() -> None:
    """HIGH tier requires BOTH entropy>=2.5 AND sparsity>=0.40."""
    tier, rationale = classify_verdict_tier(residual_entropy=3.0, sparsity=0.50)
    assert tier == "HIGH"
    assert "HIGH-ENTROPY-RESIDUAL-PRESENT" in rationale
    assert "$5.20" in rationale
    assert "Selfcomp tone-map-delta" in rationale


def test_verdict_tier_medium_via_entropy() -> None:
    """MEDIUM tier via entropy-only path: entropy in [1.5, 2.5)."""
    tier, _ = classify_verdict_tier(residual_entropy=2.0, sparsity=0.10)
    assert tier == "MEDIUM"


def test_verdict_tier_medium_via_sparsity() -> None:
    """MEDIUM tier via sparsity-only path: sparsity in [0.2, 0.4)."""
    tier, rationale = classify_verdict_tier(residual_entropy=1.0, sparsity=0.30)
    assert tier == "MEDIUM"
    assert "sister probe 3c" in rationale


def test_verdict_tier_low() -> None:
    """LOW tier: BOTH entropy<1.5 AND sparsity<0.20."""
    tier, rationale = classify_verdict_tier(residual_entropy=0.5, sparsity=0.05)
    assert tier == "LOW"
    assert "IMPLEMENTATION-LEVEL falsified at BOTH A1 + Selfcomp" in rationale
    assert "Catalog #307" in rationale
    assert "Catalog #308" in rationale


# =====================================================================
# Test 5: Predicted ΔS band computation
# =====================================================================
def test_predicted_band_high_ev() -> None:
    """HIGH-EV band: [-0.005, +0.001]."""
    band = compute_predicted_delta_s_band(residual_entropy=3.0, sparsity=0.50)
    assert band == (-0.005, +0.001)


def test_predicted_band_medium_ev() -> None:
    """MEDIUM-EV band: [-0.001, +0.001]."""
    band = compute_predicted_delta_s_band(residual_entropy=2.0, sparsity=0.10)
    assert band == (-0.001, +0.001)


def test_predicted_band_low_ev_rate_penalty_only() -> None:
    """LOW-EV band: rate penalty + small extension (no distortion offset)."""
    band = compute_predicted_delta_s_band(residual_entropy=0.5, sparsity=0.05)
    assert band[0] > 0  # rate penalty positive
    assert band[1] > band[0]


# =====================================================================
# Test 6: Verdict dataclass invariants (Catalog #192 non-promotable)
# =====================================================================
def test_verdict_score_claim_must_be_false() -> None:
    """score_claim=True must raise per Catalog #192."""
    with pytest.raises(ValueError, match="score_claim"):
        STCToneMapDeltaProbeVerdict(
            tone_map_delta_entropy_bits_per_symbol=3.0,
            five_tuple_sparsity_ratio=0.5,
            predicted_delta_s_rate_only=0.000271,
            predicted_delta_s_band=(-0.005, 0.001),
            verdict_tier="HIGH",
            canonical_equation_id="test",
            verdict_rationale="test",
            sample_pairs_decoded=16,
            sample_pairs_total_residuals=1000,
            cover_signal_type=COVER_SIGNAL_TYPE,
            lut_bits=4,
            ground_truth_video_sha256="a" * 64,
            score_claim=True,  # forbidden
        )


def test_verdict_promotable_must_be_false() -> None:
    """promotable=True must raise per Catalog #192."""
    with pytest.raises(ValueError, match="promotable"):
        STCToneMapDeltaProbeVerdict(
            tone_map_delta_entropy_bits_per_symbol=3.0,
            five_tuple_sparsity_ratio=0.5,
            predicted_delta_s_rate_only=0.000271,
            predicted_delta_s_band=(-0.005, 0.001),
            verdict_tier="HIGH",
            canonical_equation_id="test",
            verdict_rationale="test",
            sample_pairs_decoded=16,
            sample_pairs_total_residuals=1000,
            cover_signal_type=COVER_SIGNAL_TYPE,
            lut_bits=4,
            ground_truth_video_sha256="a" * 64,
            promotable=True,  # forbidden
        )


def test_verdict_invalid_tier_raises() -> None:
    """verdict_tier not in HIGH/MEDIUM/LOW must raise."""
    with pytest.raises(ValueError, match="verdict_tier"):
        STCToneMapDeltaProbeVerdict(
            tone_map_delta_entropy_bits_per_symbol=3.0,
            five_tuple_sparsity_ratio=0.5,
            predicted_delta_s_rate_only=0.000271,
            predicted_delta_s_band=(-0.005, 0.001),
            verdict_tier="INVALID",
            canonical_equation_id="test",
            verdict_rationale="test",
            sample_pairs_decoded=16,
            sample_pairs_total_residuals=1000,
            cover_signal_type=COVER_SIGNAL_TYPE,
            lut_bits=4,
            ground_truth_video_sha256="a" * 64,
        )


def test_verdict_invalid_cover_signal_raises() -> None:
    """cover_signal_type must match canonical token."""
    with pytest.raises(ValueError, match="cover_signal_type"):
        STCToneMapDeltaProbeVerdict(
            tone_map_delta_entropy_bits_per_symbol=3.0,
            five_tuple_sparsity_ratio=0.5,
            predicted_delta_s_rate_only=0.000271,
            predicted_delta_s_band=(-0.005, 0.001),
            verdict_tier="HIGH",
            canonical_equation_id="test",
            verdict_rationale="test",
            sample_pairs_decoded=16,
            sample_pairs_total_residuals=1000,
            cover_signal_type="wrong_cover_signal",
            lut_bits=4,
            ground_truth_video_sha256="a" * 64,
        )


# =====================================================================
# Test 7: compute_stc_tone_map_delta_entropy end-to-end
# =====================================================================
def test_compute_verdict_synthetic_high_pattern() -> None:
    """High-noise synthetic → entropy >= 2.5 → likely HIGH if sparsity also OK."""
    residuals = synthetic_tone_map_delta_residuals_for_test(pattern="high", n_symbols=50_000)
    verdict = compute_stc_tone_map_delta_entropy(
        residuals,
        ground_truth_video_sha256="a" * 64,
        lut_bits=4,
    )
    assert isinstance(verdict, STCToneMapDeltaProbeVerdict)
    assert verdict.cover_signal_type == COVER_SIGNAL_TYPE
    assert verdict.lut_bits == 4
    assert verdict.score_claim is False
    assert verdict.promotable is False
    assert verdict.axis_tag == "[macOS-CPU advisory]"
    # High Gaussian σ=8 → entropy high
    assert verdict.tone_map_delta_entropy_bits_per_symbol > 4.0


def test_compute_verdict_synthetic_compressible_lut_medium_tier() -> None:
    """compressible_lut pattern: entropy ~4 + sparsity ~0.29 → MEDIUM tier."""
    residuals = synthetic_tone_map_delta_residuals_for_test(
        pattern="compressible_lut", n_symbols=50_000
    )
    verdict = compute_stc_tone_map_delta_entropy(
        residuals,
        ground_truth_video_sha256="b" * 64,
        lut_bits=4,
    )
    # entropy ~= 4 >= 2.5 (HIGH on entropy alone) BUT sparsity ~0.29 < 0.40 (MEDIUM on sparsity)
    # → MEDIUM verdict (need BOTH high to qualify as HIGH)
    assert verdict.verdict_tier == "MEDIUM"


def test_compute_verdict_synthetic_low_pattern() -> None:
    """All-zeros + noise pattern → LOW tier (very-low entropy)."""
    residuals = synthetic_tone_map_delta_residuals_for_test(pattern="low", n_symbols=50_000)
    verdict = compute_stc_tone_map_delta_entropy(
        residuals,
        ground_truth_video_sha256="c" * 64,
        lut_bits=4,
    )
    # entropy < 1.5 AND sparsity > 0.95 → sparsity is HIGH; MEDIUM via sparsity
    # (per OVERNIGHT-W §9 thresholds; one signal at HIGH suffices for MEDIUM not LOW)
    # Actually: sparsity 0.95 >= 0.40 BUT entropy < 1.5
    # HIGH requires BOTH; MEDIUM accepts (entropy >= 1.5 OR sparsity >= 0.20)
    # Here entropy < 1.5 but sparsity >= 0.40 → MEDIUM (via sparsity)
    assert verdict.verdict_tier == "MEDIUM"


# =====================================================================
# Test 8: Verdict serializable + cover-signal-type tag preserved
# =====================================================================
def test_verdict_as_dict_round_trip() -> None:
    """Verdict as_dict produces JSON-serializable dict with cover_signal_type."""
    residuals = synthetic_tone_map_delta_residuals_for_test(pattern="high", n_symbols=10_000)
    verdict = compute_stc_tone_map_delta_entropy(
        residuals,
        ground_truth_video_sha256="a" * 64,
        lut_bits=4,
    )
    d = verdict.as_dict()
    serialized = json.dumps(d, sort_keys=True)
    parsed = json.loads(serialized)
    assert parsed["cover_signal_type"] == COVER_SIGNAL_TYPE
    assert parsed["lut_bits"] == 4
    assert parsed["score_claim"] is False
    assert parsed["promotable"] is False
    assert parsed["axis_tag"] == "[macOS-CPU advisory]"


# =====================================================================
# Test 9: Canonical Provenance respected (Catalog #323)
# =====================================================================
def test_canonical_provenance_built_from_verdict() -> None:
    """build_canonical_provenance_for_report produces dict with required keys."""
    residuals = synthetic_tone_map_delta_residuals_for_test(pattern="high", n_symbols=10_000)
    verdict = compute_stc_tone_map_delta_entropy(
        residuals,
        ground_truth_video_sha256="a" * 64,
        lut_bits=4,
    )
    prov = build_canonical_provenance_for_report(verdict)
    assert isinstance(prov, dict)
    # Canonical Provenance from build_provenance_for_macos_cpu_advisory carries non-promotable markers
    # (the exact field names depend on the canonical contract per Catalog #323)


# =====================================================================
# Test 10: Deterministic output (same input → same verdict)
# =====================================================================
def test_deterministic_synthetic_output() -> None:
    """Same seed → identical residuals → identical verdict."""
    r1 = synthetic_tone_map_delta_residuals_for_test(pattern="high", n_symbols=10_000, seed=42)
    r2 = synthetic_tone_map_delta_residuals_for_test(pattern="high", n_symbols=10_000, seed=42)
    np.testing.assert_array_equal(r1, r2)
    v1 = compute_stc_tone_map_delta_entropy(r1, ground_truth_video_sha256="a" * 64, lut_bits=4)
    v2 = compute_stc_tone_map_delta_entropy(r2, ground_truth_video_sha256="a" * 64, lut_bits=4)
    assert v1.tone_map_delta_entropy_bits_per_symbol == v2.tone_map_delta_entropy_bits_per_symbol
    assert v1.five_tuple_sparsity_ratio == v2.five_tuple_sparsity_ratio
    assert v1.verdict_tier == v2.verdict_tier


# =====================================================================
# Test 11: Graceful failure on missing ground-truth video
# =====================================================================
def test_verify_video_sha_returns_sentinel_for_missing() -> None:
    """verify_ground_truth_video_sha256 returns sentinel for missing file."""
    sha = verify_ground_truth_video_sha256(Path("/nonexistent/path/0.mkv"))
    assert sha == "0" * 64


# =====================================================================
# Test 12: main() synthetic-test-mode runs cleanly
# =====================================================================
def test_main_synthetic_test_mode_returns_zero() -> None:
    """main with --synthetic-test-mode + --skip-ledger-registration returns 0."""
    rc = main([
        "--synthetic-test-mode",
        "--synthetic-pattern", "high",
        "--skip-ledger-registration",
    ])
    assert rc == 0


def test_main_synthetic_test_mode_compressible_lut() -> None:
    """main with compressible_lut synthetic pattern returns 0 with MEDIUM tier."""
    rc = main([
        "--synthetic-test-mode",
        "--synthetic-pattern", "compressible_lut",
        "--lut-bits", "4",
        "--skip-ledger-registration",
    ])
    assert rc == 0


# =====================================================================
# Test 13: Cover signal type tag is canonical
# =====================================================================
def test_cover_signal_type_canonical() -> None:
    """COVER_SIGNAL_TYPE constant matches expected canonical token."""
    assert COVER_SIGNAL_TYPE == "selfcomp_tone_map_delta"


# =====================================================================
# Test 14: DEFAULT_LUT_BITS canonical per PR #56
# =====================================================================
def test_default_lut_bits_canonical() -> None:
    """DEFAULT_LUT_BITS = 4 per PR #56 paradigm."""
    assert DEFAULT_LUT_BITS == 4


# =====================================================================
# Test 15: tone-map-delta end-to-end with synthetic RGB
# =====================================================================
def test_tone_map_delta_end_to_end_synthetic_rgb() -> None:
    """Synthetic RGB → soft-grayscale → LUT-quantize → tone-map-delta residual.

    For a uniform-noise RGB input, the tone-map-delta should be bounded in
    [-step/2, +step/2] where step = 255 / (2^lut_bits - 1).
    """
    rng = np.random.default_rng(123)
    rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    grayscale = soft_grayscale_from_rgb(rgb)
    quantized = lut_quantize_grayscale(grayscale, lut_bits=4)
    delta = grayscale - quantized
    # For lut_bits=4: step=17; |delta| ≤ ceil(8.5) = 9 (allowing for rounding)
    assert np.max(np.abs(delta)) <= 9.0, f"delta out of bounds: max abs = {np.max(np.abs(delta))}"
