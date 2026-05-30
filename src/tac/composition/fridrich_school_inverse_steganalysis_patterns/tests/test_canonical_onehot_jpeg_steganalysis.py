# SPDX-License-Identifier: MIT
"""Tests for canonical OneHot JPEG steganalysis (Yousfi-Fridrich 2020)."""

from __future__ import annotations

import numpy as np
import pytest

from tac.composition.fridrich_school_inverse_steganalysis_patterns import (
    DCT_COEFFICIENT_CANONICAL_RANGE,
    OneHotEncodingConfig,
    OneHotEncodingError,
    OneHotEncodingStrategy,
    compute_one_hot_input_channels,
    encode_value_one_hot,
)


def test_dct_canonical_range_yousfi_fridrich_2020() -> None:
    """Canonical [-1024, +1024] range per Yousfi-Fridrich 2020 IEEE SPL."""
    assert DCT_COEFFICIENT_CANONICAL_RANGE == (-1024, 1024)


def test_full_range_2049_channels() -> None:
    """FULL_RANGE_2049_CHANNELS strategy produces 2049 channels canonical."""
    cfg = OneHotEncodingConfig(
        strategy=OneHotEncodingStrategy.FULL_RANGE_2049_CHANNELS
    )
    assert compute_one_hot_input_channels(cfg) == 2049


def test_clipped_range_512_channels() -> None:
    """CLIPPED_RANGE_512_CHANNELS produces ~513 channels (quarter of full range)."""
    cfg = OneHotEncodingConfig(
        strategy=OneHotEncodingStrategy.CLIPPED_RANGE_512_CHANNELS
    )
    n = compute_one_hot_input_channels(cfg)
    # (2048 // 4) + 1 = 513
    assert n == 513


def test_binarized_lsb_2_channels() -> None:
    """LSB encoding produces 2 channels."""
    cfg = OneHotEncodingConfig(
        strategy=OneHotEncodingStrategy.BINARIZED_LSB_2_CHANNELS
    )
    assert compute_one_hot_input_channels(cfg) == 2


def test_multi_scale_pyramid_8_channels() -> None:
    """Octave pyramid produces 8 channels (bits 0-7)."""
    cfg = OneHotEncodingConfig(
        strategy=OneHotEncodingStrategy.MULTI_SCALE_PYRAMID_OCTAVE_8
    )
    assert compute_one_hot_input_channels(cfg) == 8


def test_strategies_substantively_distinct_channel_counts() -> None:
    """Slot EEE: each canonical strategy MUST produce different channel count."""
    counts = []
    for strategy in OneHotEncodingStrategy:
        cfg = OneHotEncodingConfig(strategy=strategy)
        counts.append(compute_one_hot_input_channels(cfg))
    # 2049 (full), 513 (clipped), 2 (LSB), 8 (octave)
    assert len(set(counts)) == len(counts), f"All channel counts must be distinct: {counts}"


def test_encode_full_range_3d_input() -> None:
    """Encode (B, H, W) integer tensor produces (B, 2049, H, W) one-hot."""
    rng = np.random.default_rng(seed=42)
    values = rng.integers(-100, 100, size=(2, 4, 5)).astype(np.int32)
    cfg = OneHotEncodingConfig(strategy=OneHotEncodingStrategy.FULL_RANGE_2049_CHANNELS)
    encoded = encode_value_one_hot(values, cfg)
    assert encoded.shape == (2, 2049, 4, 5)
    # Each (B, :, h, w) MUST sum to 1 (one-hot invariant).
    sums = encoded.sum(axis=1)
    np.testing.assert_array_equal(sums, np.ones((2, 4, 5)))


def test_encode_binarized_lsb_recovers_lsb() -> None:
    """LSB encoding correctly extracts the least-significant bit."""
    values = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]], dtype=np.int32)  # (1, 2, 4)
    cfg = OneHotEncodingConfig(strategy=OneHotEncodingStrategy.BINARIZED_LSB_2_CHANNELS)
    encoded = encode_value_one_hot(values, cfg)
    # Shape: (1, 2, 2, 4)
    assert encoded.shape == (1, 2, 2, 4)
    # Channel 1 = LSB
    expected_lsb = np.array([[[0, 1, 0, 1], [0, 1, 0, 1]]])
    np.testing.assert_array_equal(encoded[:, 1], expected_lsb)
    # Channel 0 = 1 - LSB
    np.testing.assert_array_equal(encoded[:, 0], 1 - expected_lsb)


def test_encode_octave_recovers_bits() -> None:
    """Octave encoding correctly extracts each bit-plane."""
    # Value 7 = 0b00000111; bits 0,1,2 are 1; bits 3-7 are 0.
    values = np.array([[[7]]], dtype=np.int32)  # (1, 1, 1)
    cfg = OneHotEncodingConfig(strategy=OneHotEncodingStrategy.MULTI_SCALE_PYRAMID_OCTAVE_8)
    encoded = encode_value_one_hot(values, cfg)
    assert encoded.shape == (1, 8, 1, 1)
    expected_bits = np.array([[[[1]], [[1]], [[1]], [[0]], [[0]], [[0]], [[0]], [[0]]]])
    np.testing.assert_array_equal(encoded, expected_bits)


def test_encode_full_range_4d_input() -> None:
    """Encode (B, C, H, W) integer tensor produces (B, C*2049, H, W) one-hot."""
    rng = np.random.default_rng(seed=42)
    values = rng.integers(-10, 10, size=(1, 2, 3, 3)).astype(np.int32)
    cfg = OneHotEncodingConfig(strategy=OneHotEncodingStrategy.FULL_RANGE_2049_CHANNELS)
    encoded = encode_value_one_hot(values, cfg)
    assert encoded.shape == (1, 2 * 2049, 3, 3)


def test_encode_invalid_input_raises() -> None:
    """Invalid input raises OneHotEncodingError."""
    cfg = OneHotEncodingConfig()
    with pytest.raises(OneHotEncodingError, match="must be np.ndarray"):
        encode_value_one_hot([1, 2, 3], cfg)  # type: ignore[arg-type]
    with pytest.raises(OneHotEncodingError, match="integer dtype"):
        encode_value_one_hot(np.array([1.0, 2.0]).reshape(1, 1, 2), cfg)
    with pytest.raises(OneHotEncodingError, match="not supported"):
        encode_value_one_hot(np.array([[1]], dtype=np.int32), cfg)  # 2-D not allowed


def test_config_invalid_strategy_raises() -> None:
    """Wrong type for strategy raises."""
    with pytest.raises(OneHotEncodingError, match="must be OneHotEncodingStrategy"):
        OneHotEncodingConfig(strategy="bogus")  # type: ignore[arg-type]


def test_config_invalid_range_raises() -> None:
    """Inverted range raises."""
    with pytest.raises(OneHotEncodingError, match="requires lo < hi"):
        OneHotEncodingConfig(dct_range=(100, 50))


def test_all_4_canonical_strategies_present() -> None:
    """4 canonical strategies per the Yousfi-Fridrich 2020 taxonomy."""
    expected = {
        "full_range_2049_channels",
        "clipped_range_512_channels",
        "binarized_lsb_2_channels",
        "multi_scale_pyramid_octave_8",
    }
    assert {s.value for s in OneHotEncodingStrategy} == expected
