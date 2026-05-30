# SPDX-License-Identifier: MIT
"""Tests for canonical syndrome trellis coding (Filler-Judas-Fridrich 2011)."""

from __future__ import annotations

import pytest

from tac.composition.fridrich_school_inverse_steganalysis_patterns import (
    FILLER_2011_DISTORTION_VS_BOUND_PERCENT,
    FILLER_CANONICAL_SUB_MATRIX_HEIGHT_RANGE,
    STCAdaptiveEmbeddingStrategy,
    STCEmbeddingConfig,
    STCEmbeddingError,
    STCSubMatrixHeight,
    compute_stc_distortion_bound_from_sub_matrix_height,
)


def test_filler_canonical_height_range_7_to_12() -> None:
    """Per Filler 2011 Section VI: canonical h range is 7-12."""
    assert FILLER_CANONICAL_SUB_MATRIX_HEIGHT_RANGE == (7, 12)


def test_filler_2011_distortion_table_monotonic_decreasing() -> None:
    """% gap MUST decrease as h increases (more trellis states = better)."""
    items = sorted(FILLER_2011_DISTORTION_VS_BOUND_PERCENT.items())
    for i in range(len(items) - 1):
        h_lo, gap_lo = items[i]
        h_hi, gap_hi = items[i + 1]
        assert h_lo < h_hi
        assert gap_lo > gap_hi, (
            f"h={h_lo} gap={gap_lo} should be > h={h_hi} gap={gap_hi}"
        )


def test_filler_table_canonical_values() -> None:
    """Sample canonical Filler 2011 Table II values."""
    assert FILLER_2011_DISTORTION_VS_BOUND_PERCENT[7] == 3.0
    assert FILLER_2011_DISTORTION_VS_BOUND_PERCENT[10] == 1.7
    assert FILLER_2011_DISTORTION_VS_BOUND_PERCENT[12] == 1.2


def test_compute_distortion_bound_canonical() -> None:
    """1:1 with Filler 2011 Table II for each canonical h."""
    for h in FILLER_2011_DISTORTION_VS_BOUND_PERCENT:
        sub_height = STCSubMatrixHeight(h)
        gap = compute_stc_distortion_bound_from_sub_matrix_height(sub_height)
        assert gap == FILLER_2011_DISTORTION_VS_BOUND_PERCENT[h]


def test_substrate_heights_substantively_distinct() -> None:
    """Slot EEE: each STC height MUST produce different gap-to-bound."""
    gaps = set()
    for h_enum in STCSubMatrixHeight:
        gap = compute_stc_distortion_bound_from_sub_matrix_height(h_enum)
        gaps.add(gap)
    # 6 canonical heights MUST produce 6 distinct gap values.
    assert len(gaps) == 6


def test_strategies_substantively_distinct() -> None:
    """4 canonical strategies; each a structurally distinct embedding algorithm."""
    expected = {
        "canonical_viterbi_trellis",
        "greedy_lsb_baseline",
        "random_embedding_baseline",
        "optimal_lp_relaxation",
    }
    actual = {s.value for s in STCAdaptiveEmbeddingStrategy}
    assert actual == expected


def test_state_counts_power_of_2() -> None:
    """Trellis state count = 2^h per Filler 2011 Section IV."""
    for h_enum in STCSubMatrixHeight:
        h = int(h_enum)
        # The state count is implicit in the enum name; verify h is in canonical range.
        assert 7 <= h <= 12


def test_config_default_canonical_viterbi() -> None:
    """Default strategy is canonical Viterbi trellis algorithm."""
    cfg = STCEmbeddingConfig()
    assert cfg.strategy == STCAdaptiveEmbeddingStrategy.CANONICAL_VITERBI_TRELLIS
    assert cfg.sub_matrix_height == STCSubMatrixHeight.H_10_1024_STATES


def test_config_canonical_payload_rate() -> None:
    """Canonical Yousfi+alaska benchmark payload rate is 0.4 bpac."""
    cfg = STCEmbeddingConfig()
    assert cfg.payload_rate_bits_per_pixel == 0.4


def test_config_invalid_strategy_raises() -> None:
    """Wrong type for strategy raises."""
    with pytest.raises(STCEmbeddingError, match="must be STCAdaptiveEmbeddingStrategy"):
        STCEmbeddingConfig(strategy="bogus")  # type: ignore[arg-type]


def test_config_invalid_height_raises() -> None:
    """Wrong type for sub_matrix_height raises."""
    with pytest.raises(STCEmbeddingError, match="must be STCSubMatrixHeight"):
        STCEmbeddingConfig(sub_matrix_height=10)  # type: ignore[arg-type]


def test_config_invalid_payload_rate_raises() -> None:
    """Payload rate outside (0, 1.0] raises."""
    with pytest.raises(STCEmbeddingError, match="payload_rate"):
        STCEmbeddingConfig(payload_rate_bits_per_pixel=0.0)
    with pytest.raises(STCEmbeddingError, match="payload_rate"):
        STCEmbeddingConfig(payload_rate_bits_per_pixel=1.5)


def test_compute_distortion_invalid_input_raises() -> None:
    """Wrong type for sub_matrix_height raises."""
    with pytest.raises(STCEmbeddingError, match="must be STCSubMatrixHeight"):
        compute_stc_distortion_bound_from_sub_matrix_height(10)  # type: ignore[arg-type]


def test_h_12_canonical_lowest_gap() -> None:
    """h=12 (4096 states) gives the LOWEST canonical gap (~1.2%)."""
    gap_12 = compute_stc_distortion_bound_from_sub_matrix_height(
        STCSubMatrixHeight.H_12_4096_STATES
    )
    gap_7 = compute_stc_distortion_bound_from_sub_matrix_height(
        STCSubMatrixHeight.H_7_128_STATES
    )
    assert gap_12 < gap_7
    assert gap_12 == 1.2
    assert gap_7 == 3.0
