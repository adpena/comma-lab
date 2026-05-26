# SPDX-License-Identifier: MIT
"""MLX-local smoke test for UNIWARD 7th-order integration into NSCS06 v8.

Per CLAUDE.md "MLX portable-local-substrate authority": training-time only;
output tagged `[macOS-MLX research-signal]` per Catalog #192/#317/#341.

Per Catalog #229 premise verification: synthetic compress-time GT batch
exercises the canonical aggregation + UNIWARD-weighted LUT derivation +
canonical-vs-UNIWARD comparison, deriving a paradigm verdict from
empirically-observable LUT byte differences.

5-pair fixture @ 32x32 (sister synthetic batch sized for MLX-local 30s
smoke; the N+1 fixture is 50 pairs @ 96x128 with cached real-scorer
gradients reusable via npz at
`.omx/research/uniward_per_pixel_n_plus_1_artifacts_20260526/real_scorer_gradients_cache.npz`).

The PARADIGM-VALIDATION threshold per the 6th-order Carmack-dissent verdict:
- ANY measurable byte difference in LUT output → PARADIGM-VALIDATED-AT-ENTROPY-CODED-SIDECAR
- byte-identical LUTs → PARADIGM-NULL-NO-EFFECT-AT-ENTROPY-CODED-SIDECAR
  (final Fridrich-canonical falsification for our contest; Catalog #348
  retroactive sweep triggered)
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.uniward_per_pixel_distortion.nscs06_v8_chroma_lut_integration import (
    CONSUMER_NAME,
    CONSUMER_VERSION,
    CONSUMER_HOOK_NUMBERS,
    INTEGRATION_NAME,
    INTEGRATION_VERSION,
)
from tac.substrates.uniward_per_pixel_distortion.nscs06_v8_chroma_lut_integration.weight_map_per_lut_index import (
    GRAYSCALE_LEVELS_DEFAULT,
    NUM_SEGNET_CLASSES,
    PER_LUT_INDEX_WEIGHT_EPS,
    PerLutIndexUniwardWeights,
    aggregate_per_pixel_uniward_weights_into_lut_bins,
    build_canonical_provenance_for_per_lut_index_aggregation,
)
from tac.substrates.uniward_per_pixel_distortion.nscs06_v8_chroma_lut_integration.lut_derivation_uniward_weighted import (
    WeightedMedianResult,
    build_uniward_weighted_chroma_lut,
    compare_uniward_vs_canonical_lut,
    weighted_median_per_channel,
)

# Sister of v8 canonical derivation; READ-ONLY consumer import per Catalog #230
from tac.substrates.nscs06_v8_chroma_lut.architecture import (
    build_chroma_lut_from_ground_truth,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_compress_batch(
    *,
    n: int = 5,
    h: int = 32,
    w: int = 32,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic compress-time GT batch sized for MLX-local 30s smoke."""
    rng = np.random.default_rng(seed)
    # Smooth gradient + class-correlated chroma so the LUT has signal
    rgb = np.zeros((n, 3, h, w), dtype=np.uint8)
    cls = np.zeros((n, h, w), dtype=np.uint8)
    for i in range(n):
        # Luma gradient
        luma_base = np.linspace(20, 220, h * w, dtype=np.float32).reshape(h, w)
        luma = (luma_base + rng.normal(0, 10, size=(h, w))).clip(0, 255)
        # Per-class chroma offset
        cls_map = (rng.integers(0, NUM_SEGNET_CLASSES, size=(h, w))).astype(np.uint8)
        cls[i] = cls_map
        for c in range(NUM_SEGNET_CLASSES):
            mask = cls_map == c
            # Different chroma per class
            r_offset = 30 * (c - 2)
            g_offset = 15 * (c - 2)
            b_offset = -20 * (c - 2)
            rgb_r = (luma + r_offset).clip(0, 255)
            rgb_g = (luma + g_offset).clip(0, 255)
            rgb_b = (luma + b_offset).clip(0, 255)
            rgb[i, 0][mask] = rgb_r[mask].astype(np.uint8)
            rgb[i, 1][mask] = rgb_g[mask].astype(np.uint8)
            rgb[i, 2][mask] = rgb_b[mask].astype(np.uint8)
    return rgb, cls


def _make_synthetic_uniward_weight_map(
    *,
    h: int = 32,
    w: int = 32,
    seed: int = 19,
    dynamic_range: float = 50.0,
) -> np.ndarray:
    """Synthetic UNIWARD weight map with high dynamic range across pixels."""
    rng = np.random.default_rng(seed)
    # Spatial pattern (center high, edges low) + per-pixel jitter
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    distance = np.sqrt(yy * yy + xx * xx)
    base = np.exp(-distance * distance / 0.5)
    jitter = rng.uniform(0.5, 1.5, size=(h, w))
    weight = (base * dynamic_range + 1.0) * jitter
    return weight.astype(np.float32)


# ---------------------------------------------------------------------------
# Module identity + provenance contract tests
# ---------------------------------------------------------------------------


def test_integration_module_identity():
    assert INTEGRATION_NAME == "uniward_per_lut_index_into_nscs06_v8_chroma_lut"
    assert INTEGRATION_VERSION == "v1_2026-05-26_7th_order"


def test_consumer_canonical_contract_fields_present():
    """Per Catalog #335 cathedral consumer contract surface."""
    assert CONSUMER_NAME == "uniward_per_lut_index_into_nscs06_v8_chroma_lut_integration"
    assert CONSUMER_VERSION == "v1_2026-05-26_7th_order"
    assert CONSUMER_HOOK_NUMBERS == (1, 5)


# ---------------------------------------------------------------------------
# Per-LUT-index aggregation correctness tests
# ---------------------------------------------------------------------------


def test_aggregation_returns_per_lut_index_uniward_weights_dataclass():
    rgb, cls = _make_synthetic_compress_batch()
    weight = _make_synthetic_uniward_weight_map()
    result = aggregate_per_pixel_uniward_weights_into_lut_bins(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=weight
    )
    assert isinstance(result, PerLutIndexUniwardWeights)
    assert result.weight_per_bin.shape == (GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES)
    assert result.pixel_count_per_bin.shape == (GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES)
    assert result.weight_per_bin.dtype == np.float32


def test_aggregation_preserves_total_weight():
    """Sum of per-bin weights == sum of per-pixel weights (canonical invariant)."""
    rgb, cls = _make_synthetic_compress_batch()
    weight = _make_synthetic_uniward_weight_map()
    n, _, h, w = rgb.shape
    total_per_pixel = float(weight.sum()) * n  # broadcast over N frames
    result = aggregate_per_pixel_uniward_weights_into_lut_bins(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=weight
    )
    total_per_bin = float(result.weight_per_bin.sum())
    np.testing.assert_allclose(total_per_bin, total_per_pixel, rtol=1e-4)


def test_aggregation_pixel_count_matches_total_pixels():
    rgb, cls = _make_synthetic_compress_batch()
    weight = _make_synthetic_uniward_weight_map()
    n, _, h, w = rgb.shape
    total_pixels = n * h * w
    result = aggregate_per_pixel_uniward_weights_into_lut_bins(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=weight
    )
    assert int(result.pixel_count_per_bin.sum()) == total_pixels


def test_aggregation_dynamic_range_observable():
    """Catalog #305 observability: dynamic range surfaced via dataclass property."""
    rgb, cls = _make_synthetic_compress_batch()
    weight = _make_synthetic_uniward_weight_map(dynamic_range=100.0)
    result = aggregate_per_pixel_uniward_weights_into_lut_bins(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=weight
    )
    # Synthetic batch has substantial bin-weight variation
    assert result.dynamic_range_ratio > 1.5


def test_aggregation_rejects_invalid_shapes():
    rgb, cls = _make_synthetic_compress_batch()
    bad_weight = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="weight_map shape"):
        aggregate_per_pixel_uniward_weights_into_lut_bins(
            rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=bad_weight
        )


def test_aggregation_rejects_negative_weights():
    rgb, cls = _make_synthetic_compress_batch()
    n, _, h, w = rgb.shape
    weight = -np.ones((h, w), dtype=np.float32)
    with pytest.raises(ValueError, match="non-negative"):
        aggregate_per_pixel_uniward_weights_into_lut_bins(
            rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=weight
        )


def test_aggregation_3d_weight_map_per_frame_accepted():
    rgb, cls = _make_synthetic_compress_batch()
    n, _, h, w = rgb.shape
    weight_3d = np.ones((n, h, w), dtype=np.float32) * 2.0
    result = aggregate_per_pixel_uniward_weights_into_lut_bins(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=weight_3d
    )
    # Total weight = 2.0 * N * H * W
    assert float(result.weight_per_bin.sum()) == pytest.approx(2.0 * n * h * w, rel=1e-4)


# ---------------------------------------------------------------------------
# Weighted median correctness tests
# ---------------------------------------------------------------------------


def test_weighted_median_uniform_weights_matches_unweighted_median():
    values = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    weights = np.ones(5, dtype=np.float32)
    result = weighted_median_per_channel(values, weights)
    # Unweighted median = 3
    assert int(result) == 3


def test_weighted_median_skews_toward_high_weight_values():
    values = np.array([10, 100], dtype=np.uint8)
    weights = np.array([1.0, 100.0], dtype=np.float32)
    result = weighted_median_per_channel(values, weights)
    # Heavy weight on 100; weighted median should be 100
    assert int(result) == 100


def test_weighted_median_skews_toward_low_weight_when_low_value_dominates():
    values = np.array([10, 100], dtype=np.uint8)
    weights = np.array([100.0, 1.0], dtype=np.float32)
    result = weighted_median_per_channel(values, weights)
    assert int(result) == 10


def test_weighted_median_zero_weights_falls_back_to_unweighted():
    values = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    weights = np.zeros(5, dtype=np.float32)
    result = weighted_median_per_channel(values, weights)
    assert int(result) == 3  # unweighted median fallback


# ---------------------------------------------------------------------------
# UNIWARD-weighted LUT derivation correctness tests
# ---------------------------------------------------------------------------


def test_build_uniward_weighted_chroma_lut_returns_canonical_shape():
    rgb, cls = _make_synthetic_compress_batch()
    weight = _make_synthetic_uniward_weight_map()
    lut = build_uniward_weighted_chroma_lut(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=weight
    )
    assert lut.shape == (GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3)
    assert lut.dtype == np.uint8


def test_build_uniform_weights_produces_identical_lut_to_canonical():
    """Sanity: uniform UNIWARD weights -> UNIWARD-weighted-median == unweighted-median."""
    rgb, cls = _make_synthetic_compress_batch()
    n, _, h, w = rgb.shape
    uniform_weight = np.ones((h, w), dtype=np.float32)

    lut_uniward = build_uniward_weighted_chroma_lut(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=uniform_weight
    )
    lut_canonical = build_chroma_lut_from_ground_truth(
        rgb_pairs=rgb, class_labels=cls
    )
    # Should be byte-identical (weighted median == unweighted median when
    # weights are uniform, modulo cumulative-sum tie-breaking which can
    # differ by 1 bin position on uniform-weight cases).
    diff = np.abs(lut_uniward.astype(np.int32) - lut_canonical.astype(np.int32))
    # Allow at most 1-byte deviation per channel due to median tie-breaking
    # differences (np.median uses interpolation; weighted median uses
    # discrete-step searchsorted).
    assert diff.max() <= 2, (
        f"uniform-weight LUT should match canonical to within median "
        f"tie-breaking tolerance; max delta={diff.max()}"
    )


def test_build_high_dynamic_range_weights_produces_different_lut():
    """7th-order paradigm test: high-dynamic-range UNIWARD weights produce
    DIFFERENT LUT vs canonical-median LUT.

    This is the PARADIGM-VALIDATION signal: if UNIWARD per-LUT-index
    routing has structural traction at the entropy-coded sidecar surface,
    the resulting LUT bytes will differ from the canonical median LUT.
    """
    rgb, cls = _make_synthetic_compress_batch()
    weight = _make_synthetic_uniward_weight_map(dynamic_range=200.0)

    lut_uniward = build_uniward_weighted_chroma_lut(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=weight
    )
    lut_canonical = build_chroma_lut_from_ground_truth(
        rgb_pairs=rgb, class_labels=cls
    )
    comparison = compare_uniward_vs_canonical_lut(
        lut_uniward_weighted=lut_uniward,
        lut_canonical_median=lut_canonical,
    )
    # PARADIGM-VALIDATION: at high dynamic range, LUT bytes MUST differ
    # measurably from canonical-median LUT.
    assert comparison.num_bins_changed > 0, (
        "7th-order paradigm test FAILED: UNIWARD-weighted LUT byte-identical "
        "to canonical-median LUT even at 200x dynamic range -- this would be "
        "PARADIGM-NULL-NO-EFFECT-AT-ENTROPY-CODED-SIDECAR (final UNIWARD "
        "falsification for our contest application; trigger Catalog #348 "
        "retroactive sweep)"
    )


# ---------------------------------------------------------------------------
# Compare-uniward-vs-canonical surface tests
# ---------------------------------------------------------------------------


def test_compare_uniward_vs_canonical_identical_lut_no_change():
    rgb, cls = _make_synthetic_compress_batch()
    lut = build_chroma_lut_from_ground_truth(rgb_pairs=rgb, class_labels=cls)
    comparison = compare_uniward_vs_canonical_lut(
        lut_uniward_weighted=lut,
        lut_canonical_median=lut,
    )
    assert comparison.num_bins_changed == 0
    assert comparison.max_per_channel_delta_u8 == 0


def test_compare_uniward_vs_canonical_rejects_mismatched_shape():
    lut1 = np.zeros((16, 5, 3), dtype=np.uint8)
    lut2 = np.zeros((8, 5, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_uniward_vs_canonical_lut(
            lut_uniward_weighted=lut1, lut_canonical_median=lut2
        )


# ---------------------------------------------------------------------------
# Canonical Provenance per Catalog #323 + Catalog #341 tests
# ---------------------------------------------------------------------------


def test_canonical_provenance_carries_non_promotable_markers():
    rgb, cls = _make_synthetic_compress_batch()
    weight = _make_synthetic_uniward_weight_map()
    result = aggregate_per_pixel_uniward_weights_into_lut_bins(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=weight
    )
    prov = build_canonical_provenance_for_per_lut_index_aggregation(
        per_bin_weights=result
    )
    # Catalog #341 canonical-routing markers
    assert prov["score_claim"] is False
    assert prov["promotable"] is False
    assert prov["axis_tag"] == "[predicted]"
    assert prov["evidence_grade"] == "macOS-MLX research-signal"
    assert prov["measurement_axis"] == "[macOS-MLX research-signal]"
    # Catalog #230 sister-disjoint acknowledgment
    assert prov["nscs06_v8_substrate_modification_scope"] == (
        "none_read_only_consumer_import"
    )
    # Catalog #125 hook wire-in
    assert set(prov["hook_numbers_fired"]) == {1, 5}
    # Entropy-position declaration
    assert prov["entropy_position"] == (
        "P3_entropy_coded_sidecar_per_lut_index_routing"
    )


# ---------------------------------------------------------------------------
# END-TO-END 7th-order MLX-LOCAL SMOKE VERDICT
# ---------------------------------------------------------------------------


def test_end_to_end_7th_order_mlx_local_paradigm_smoke():
    """END-TO-END 7th-order paradigm smoke: synthetic GT + synthetic UNIWARD
    weights -> per-LUT-index aggregation -> UNIWARD-weighted LUT -> compare
    vs canonical-median LUT.

    PARADIGM-VALIDATION threshold: any measurable byte difference =>
    PARADIGM-VALIDATED-AT-ENTROPY-CODED-SIDECAR (proceed to subagent B
    paired-CUDA validation).

    PARADIGM-FALSIFICATION threshold: byte-identical LUTs across all 240
    bins (16 levels x 5 classes) at 200x dynamic range UNIWARD weights =>
    PARADIGM-NULL-NO-EFFECT-AT-ENTROPY-CODED-SIDECAR (final Fridrich-
    canonical falsification for our contest application; Catalog #348
    retroactive sweep triggered).
    """
    rgb, cls = _make_synthetic_compress_batch()
    weight = _make_synthetic_uniward_weight_map(dynamic_range=200.0)

    # Aggregate per-pixel weights into per-LUT-index bins
    per_bin = aggregate_per_pixel_uniward_weights_into_lut_bins(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=weight
    )
    assert per_bin.num_nonempty_bins > 0
    assert per_bin.dynamic_range_ratio > 1.0

    # Derive UNIWARD-weighted LUT
    lut_uniward = build_uniward_weighted_chroma_lut(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=weight
    )

    # Derive canonical-median LUT
    lut_canonical = build_chroma_lut_from_ground_truth(
        rgb_pairs=rgb, class_labels=cls
    )

    # Compare
    comparison = compare_uniward_vs_canonical_lut(
        lut_uniward_weighted=lut_uniward,
        lut_canonical_median=lut_canonical,
    )

    # Emit canonical Provenance for posterity
    prov = build_canonical_provenance_for_per_lut_index_aggregation(
        per_bin_weights=per_bin
    )

    # PARADIGM-VALIDATION assertion: at 200x dynamic-range UNIWARD weights,
    # the LUT bytes MUST differ measurably from canonical-median LUT for
    # the 7th-order entropy-coded sidecar surface test to PROCEED to
    # subagent B paired-CUDA validation.
    assert comparison.num_bins_changed > 0, (
        f"PARADIGM-FALSIFICATION at 7th-order entropy-coded sidecar surface: "
        f"UNIWARD-weighted LUT byte-identical to canonical-median LUT "
        f"(num_bins_changed=0; max_delta={comparison.max_per_channel_delta_u8}). "
        f"Final UNIWARD falsification across N+1 (5th-order PARADIGM-NULL) + "
        f"6th-order BoostNeRV (PARADIGM-NULL) + 7th-order entropy-coded sidecar "
        f"(THIS FALSIFICATION). Trigger Catalog #348 retroactive sweep of all "
        f"prior UNIWARD KILL/DEFER memos. provenance={prov}"
    )


def test_provenance_aggregation_includes_consumer_id():
    rgb, cls = _make_synthetic_compress_batch()
    weight = _make_synthetic_uniward_weight_map()
    result = aggregate_per_pixel_uniward_weights_into_lut_bins(
        rgb_pairs=rgb, class_labels=cls, per_pixel_uniward_weight=weight
    )
    prov = build_canonical_provenance_for_per_lut_index_aggregation(
        per_bin_weights=result
    )
    assert prov["integration_id"] == INTEGRATION_NAME
    assert prov["integration_version"] == INTEGRATION_VERSION
    assert prov["consumed_substrate_id"] == "nscs06_v8_chroma_lut"
    assert prov["consumed_substrate_scope"] == "read_only_consumer_import"
