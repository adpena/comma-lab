"""Regression tests for the framework-free FreSh selector."""

from __future__ import annotations

import numpy as np
import pytest

from tac.witness_init.fresh_frequency_shift import (
    derived_tangent_frequency_multipliers,
    deterministic_first_layer_bias_candidates,
    fresh_spectrum,
    inclusive_bias_width_grid,
    label_boundary_target_map,
    select_fresh_configuration,
    select_fresh_spectra,
    tangent_frequency_candidates,
    wasserstein1_cdf_l1,
)


def test_manual_antidiagonal_spectrum_is_per_channel_fft_magnitude() -> None:
    # Inverse DFT produces known coefficients at (0, 1), (1, 0), and (1, 1).
    coeffs = np.zeros((2, 4, 4), dtype=np.complex128)
    coeffs[0, 0, 1] = coeffs[0, 0, 3] = 2.0
    coeffs[0, 1, 0] = coeffs[0, 3, 0] = 3.0
    coeffs[1, 1, 1] = coeffs[1, 3, 3] = 5.0
    signal = np.fft.ifft2(  # FFT_TOOL_USE_OK:constructs synthetic test fixtures only
        coeffs, axes=(-2, -1)
    ).real

    np.testing.assert_allclose(fresh_spectrum(signal, 2), [0.5, 0.5])


def test_dc_is_omitted_and_signal_scaling_does_not_change_spectrum() -> None:
    coeffs = np.zeros((4, 4), dtype=np.complex128)
    coeffs[0, 0] = 1_000.0  # Must not influence the retained distribution.
    coeffs[0, 1] = 1.0
    coeffs[1, 1] = 3.0
    signal = np.fft.ifft2(coeffs).real  # FFT_TOOL_USE_OK:constructs synthetic test fixtures only

    expected = np.array([0.25, 0.75])
    np.testing.assert_allclose(fresh_spectrum(signal, 2), expected)
    np.testing.assert_allclose(fresh_spectrum(signal * 17.0, 2), expected)


def test_fresh_selects_exact_spectral_match_and_reports_cdf_l1_distance() -> None:
    target_coeffs = np.zeros((4, 4), dtype=np.complex128)
    target_coeffs[0, 1] = 1.0
    target_coeffs[1, 1] = 3.0
    target = np.fft.ifft2(target_coeffs).real  # FFT_TOOL_USE_OK:constructs synthetic test fixtures only

    low_coeffs = np.zeros((4, 4), dtype=np.complex128)
    low_coeffs[0, 1] = 4.0
    low = np.fft.ifft2(low_coeffs).real  # FFT_TOOL_USE_OK:constructs synthetic test fixtures only

    selection = select_fresh_configuration(target, {"low": low, "match": target}, 2)
    assert selection.candidate == "match"
    assert selection.index == 1
    assert selection.mean_distance == 0.0
    assert wasserstein1_cdf_l1([1.0, 0.0], [0.0, 1.0]) == 1.0


def test_fresh_tie_break_is_mapping_order_stable() -> None:
    target = np.fft.ifft2(  # FFT_TOOL_USE_OK:constructs synthetic test fixtures only
        np.array([[0.0, 1.0], [0.0, 0.0]])
    ).real
    # Both candidates are the same spectrum, so strict '<' preserves "first".
    selection = select_fresh_configuration(target, {"first": target, "second": target}, 1)
    assert selection.candidate == "first"
    assert selection.index == 0


def test_precomputed_spectrum_api_reuses_one_cold_output_without_image_storage() -> None:
    target_spectra = (np.array([0.8, 0.2]), np.array([0.7, 0.3]))
    selection = select_fresh_spectra(
        target_spectra,
        {"high": np.array([0.2, 0.8]), "near": np.array([0.75, 0.25])},
    )
    assert selection.candidate == "near"
    assert selection.distances == pytest.approx((0.05, 0.05))
    matrix_selection = select_fresh_spectra(
        np.asarray(target_spectra),
        {"paired": np.asarray(((0.8, 0.2), (0.7, 0.3)))},
    )
    assert matrix_selection.mean_distance == 0.0


def test_transposition_invariance_holds_for_antidiagonal_degree_bins() -> None:
    rng = np.random.default_rng(4)
    signal = rng.normal(size=(3, 5, 6))
    np.testing.assert_allclose(fresh_spectrum(signal, 7), fresh_spectrum(signal.transpose(0, 2, 1), 7))


def test_boundary_map_marks_both_sides_of_every_class_transition() -> None:
    labels = np.array([[0, 0, 1], [0, 2, 2]], dtype=np.int64)
    expected = np.array([[False, True, True], [True, True, True]])
    np.testing.assert_array_equal(label_boundary_target_map(labels), expected)


def test_derived_tangent_grid_and_dedicated_bias_rng_are_deterministic() -> None:
    assert derived_tangent_frequency_multipliers() == pytest.approx((1.0, np.sqrt(3.2), 3.2))
    assert tangent_frequency_candidates(6.0) == pytest.approx((6.0, 8.0, 8.0 * np.sqrt(3.2), 25.6))
    assert tangent_frequency_candidates(8.0) == pytest.approx((8.0, 8.0 * np.sqrt(3.2), 25.6))
    widths = inclusive_bias_width_grid()
    assert len(widths) == 31
    assert widths[0] == 0.0
    assert widths[-1] == 3.0
    np.random.seed(19)
    before = np.random.random()
    a = deterministic_first_layer_bias_candidates((0.0, 0.5, 3.0), 4, seed=7)
    after = np.random.random()
    np.random.seed(19)
    assert before == np.random.random()
    assert after == np.random.random()
    np.testing.assert_array_equal(a, deterministic_first_layer_bias_candidates((0.0, 0.5, 3.0), 4, seed=7))
    assert not np.array_equal(a, deterministic_first_layer_bias_candidates((0.0, 0.5, 3.0), 4, seed=8))
    np.testing.assert_array_equal(a[0], np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(a[2], 6.0 * a[1], rtol=0.0, atol=3e-7)
    reference_rng = np.random.default_rng(7 + 20260707)
    np.testing.assert_array_equal(
        a[1], reference_rng.uniform(-0.5, 0.5, size=4).astype(np.float32)
    )


def test_fresh_fails_closed_on_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        fresh_spectrum(np.ones((2, 2)), 0)
    with pytest.raises(ValueError, match="zero"):
        fresh_spectrum(np.zeros((2, 2)), 1)
    with pytest.raises(ValueError, match="NaN or Inf"):
        fresh_spectrum(np.array([[np.nan, 0.0], [0.0, 0.0]]), 1)
    with pytest.raises(ValueError, match="integer dtype"):
        label_boundary_target_map(np.ones((2, 2), dtype=np.float64))
    target = np.arange(16, dtype=np.float64).reshape(4, 4)
    with pytest.raises(ValueError, match="does not match target shape"):
        select_fresh_configuration(target, {"wrong": np.arange(25).reshape(5, 5)}, 2)
    with pytest.raises(ValueError, match="divide"):
        inclusive_bias_width_grid(0.0, 1.0, 0.3)
