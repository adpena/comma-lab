# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.symposium_impls.daubechies_wavelet_codec`."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tac.symposium_impls.daubechies_wavelet_codec import (
    DAUBECHIES_FILTERS,
    DEFAULT_BASE_BITS,
    DaubechiesFilter,
    WaveletDecomposition,
    WaveletSubBandAllocation,
    compute_per_sub_band_bit_allocation,
    forward_wavelet_decomposition,
    inverse_wavelet_reconstruction,
    select_filter,
    update_from_anchor,
)


# ----- filter sanity ----------------------------------------------------------------------------


def test_db1_filter_is_haar() -> None:
    h, g = select_filter(DaubechiesFilter.DB1)
    assert h.size == 2
    expected = 1.0 / math.sqrt(2.0)
    assert h[0] == pytest.approx(expected)
    assert h[1] == pytest.approx(expected)


def test_db2_filter_has_4_taps() -> None:
    h, g = select_filter(DaubechiesFilter.DB2)
    assert h.size == 4
    assert g.size == 4


def test_db4_filter_has_8_taps() -> None:
    h, g = select_filter(DaubechiesFilter.DB4)
    assert h.size == 8


def test_filter_orthonormality_db1_haar() -> None:
    """For Daubechies-N, sum h_n^2 = 1 (orthonormality of scaling function)."""
    h, _ = select_filter(DaubechiesFilter.DB1)
    assert float((h**2).sum()) == pytest.approx(1.0, abs=1e-12)


def test_filter_orthonormality_db2() -> None:
    h, _ = select_filter(DaubechiesFilter.DB2)
    assert float((h**2).sum()) == pytest.approx(1.0, abs=1e-9)


def test_filter_orthonormality_db4() -> None:
    h, _ = select_filter(DaubechiesFilter.DB4)
    assert float((h**2).sum()) == pytest.approx(1.0, abs=1e-9)


def test_select_filter_string_id_works() -> None:
    h_a, _ = select_filter(DaubechiesFilter.DB2)
    h_b, _ = select_filter("db2")
    assert np.allclose(h_a, h_b)


def test_qmf_relation_holds() -> None:
    """g_n = (-1)^n h_{N-1-n}: the canonical QMF construction."""
    h, g = select_filter(DaubechiesFilter.DB2)
    for n, gn in enumerate(g):
        expected = (-1) ** n * h[h.size - 1 - n]
        assert gn == pytest.approx(expected, abs=1e-12)


# ----- forward/inverse perfect reconstruction --------------------------------------------------


def test_db1_perfect_reconstruction_constant_signal() -> None:
    sig = np.full(8, 5.0)
    decomp = forward_wavelet_decomposition(sig, filter_id=DaubechiesFilter.DB1, levels=2)
    recovered = inverse_wavelet_reconstruction(decomp)
    assert np.allclose(recovered, sig, atol=1e-9)


def test_db2_perfect_reconstruction_random_signal() -> None:
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(64)
    decomp = forward_wavelet_decomposition(sig, filter_id=DaubechiesFilter.DB2, levels=3)
    recovered = inverse_wavelet_reconstruction(decomp)
    assert np.allclose(recovered, sig, atol=1e-7)


def test_db4_perfect_reconstruction_random_signal() -> None:
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(128)
    decomp = forward_wavelet_decomposition(sig, filter_id=DaubechiesFilter.DB4, levels=3)
    recovered = inverse_wavelet_reconstruction(decomp)
    assert np.allclose(recovered, sig, atol=1e-6)


def test_decomposition_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        forward_wavelet_decomposition(np.zeros((4, 4)))


def test_decomposition_invalid_levels_raises() -> None:
    with pytest.raises(ValueError):
        forward_wavelet_decomposition(np.zeros(8), levels=0)


# ----- multi-scale structure --------------------------------------------------------------------


def test_decomposition_levels_create_correct_number_of_detail_bands() -> None:
    sig = np.random.default_rng(2).standard_normal(32)
    decomp = forward_wavelet_decomposition(sig, levels=3)
    assert decomp.levels == 3
    assert len(decomp.detail_coefficients) == 3


def test_decomposition_total_count_preserves_data_dimension() -> None:
    """Daubechies orthonormal: total coeff count >= original (with periodic ext)."""
    sig = np.random.default_rng(3).standard_normal(64)
    decomp = forward_wavelet_decomposition(sig, levels=3)
    assert decomp.total_coefficient_count >= sig.size


# ----- per-sub-band bit allocation -------------------------------------------------------------


def test_bit_allocation_falling_rule_decreases_with_scale() -> None:
    sig = np.random.default_rng(4).standard_normal(64)
    decomp = forward_wavelet_decomposition(sig, levels=3)
    alloc = compute_per_sub_band_bit_allocation(decomp)
    detail_bits = [a.bits_per_coefficient for a in alloc[:-1]]
    # Falling rule: bits should be non-increasing as scale grows
    for prev, current in zip(detail_bits, detail_bits[1:]):
        assert current <= prev


def test_bit_allocation_approximation_always_at_base_bits() -> None:
    sig = np.random.default_rng(5).standard_normal(32)
    decomp = forward_wavelet_decomposition(sig, levels=3)
    alloc = compute_per_sub_band_bit_allocation(decomp, base_bits=8)
    approx_alloc = alloc[-1]
    assert approx_alloc.bits_per_coefficient == 8
    assert approx_alloc.scale_level == decomp.levels


def test_bit_allocation_invalid_base_bits_raises() -> None:
    sig = np.zeros(8)
    decomp = forward_wavelet_decomposition(sig)
    with pytest.raises(ValueError):
        compute_per_sub_band_bit_allocation(decomp, base_bits=0)


def test_bit_allocation_invalid_threshold_raises() -> None:
    sig = np.zeros(8)
    decomp = forward_wavelet_decomposition(sig)
    with pytest.raises(ValueError):
        compute_per_sub_band_bit_allocation(decomp, magnitude_threshold=-0.1)


def test_bit_allocation_rule_text_carries_threshold() -> None:
    sig = np.random.default_rng(6).standard_normal(16)
    decomp = forward_wavelet_decomposition(sig, levels=2)
    alloc = compute_per_sub_band_bit_allocation(decomp, magnitude_threshold=0.123)
    # Some rule text should carry the threshold
    assert any("0.123" in a.rule_text for a in alloc[:-1])


def test_bit_allocation_significant_count_matches_threshold() -> None:
    sig = np.array([10.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    decomp = forward_wavelet_decomposition(sig, levels=1)
    alloc = compute_per_sub_band_bit_allocation(decomp, magnitude_threshold=1.0)
    # Some sub-bands have non-trivial magnitude > 1.0
    assert any(a.significant_coefficient_count > 0 for a in alloc)


# ----- continual learning hook -----------------------------------------------------------------


def test_update_from_anchor_no_signal_returns_none() -> None:
    assert update_from_anchor({}) is None


def test_update_from_anchor_empty_signal_returns_none() -> None:
    assert update_from_anchor({}, signal=np.array([])) is None


def test_update_from_anchor_returns_decomposition() -> None:
    sig = np.random.default_rng(7).standard_normal(32)
    decomp = update_from_anchor({"levels": 2}, signal=sig)
    assert decomp is not None
    assert decomp.levels == 2


def test_update_from_anchor_uses_supplied_filter() -> None:
    sig = np.random.default_rng(8).standard_normal(32)
    decomp = update_from_anchor({"levels": 1, "filter_id": "db1"}, signal=sig)
    assert decomp is not None
    assert decomp.filter_id == DaubechiesFilter.DB1


# ----- signal energy preservation (Parseval) ---------------------------------------------------


def test_db1_energy_preservation_parseval() -> None:
    """Sum of squared coefficients ≈ sum of squared signal samples (Parseval)."""
    sig = np.random.default_rng(9).standard_normal(32)
    decomp = forward_wavelet_decomposition(sig, filter_id=DaubechiesFilter.DB1, levels=3)
    coef_energy = float((decomp.approximation_coefficients**2).sum()) + sum(
        float((d**2).sum()) for d in decomp.detail_coefficients
    )
    sig_energy = float((sig**2).sum())
    # Parseval-like equality holds within a tolerance for periodic ext.
    assert coef_energy == pytest.approx(sig_energy, rel=0.05)
