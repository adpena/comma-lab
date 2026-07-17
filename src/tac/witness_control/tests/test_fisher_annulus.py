# SPDX-License-Identifier: MIT
"""Tests for the Fisher-mass-in-annulus observable (behavior, not constants)."""
from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control.fisher_annulus import (
    DEFAULT_ANNULUS_BAND,
    fisher_mass_in_annulus,
    fisher_mass_report_over_pairs,
    fisher_trace_from_margin,
)


def test_trace_peak_at_zero_margin_is_half():
    assert fisher_trace_from_margin(np.array([0.0]))[0] == pytest.approx(0.5, abs=1e-12)


def test_trace_matches_naive_sech2_formula():
    m = np.linspace(-6, 6, 101)
    naive = 0.5 / np.cosh(m / 2.0) ** 2
    got = fisher_trace_from_margin(m)
    np.testing.assert_allclose(got, naive, rtol=1e-12, atol=1e-15)


def test_trace_even_in_margin_sign():
    m = np.array([-3.0, -0.7, 0.7, 3.0])
    tr = fisher_trace_from_margin(m)
    np.testing.assert_allclose(tr[:2], tr[3:1:-1], rtol=1e-14)


def test_trace_overflow_stable_at_huge_margin():
    tr = fisher_trace_from_margin(np.array([1e4, -1e4]))
    assert np.all(np.isfinite(tr))
    assert np.all(tr >= 0.0)
    assert np.all(tr < 1e-300 + 1e-10)  # essentially 0, never NaN/inf


def test_trace_monotone_decreasing_in_abs_margin():
    m = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    tr = fisher_trace_from_margin(m)
    assert np.all(np.diff(tr) < 0)


def test_annulus_report_all_inside():
    m = np.zeros((4, 4))
    r = fisher_mass_in_annulus(m, band=2.0)
    assert r.annulus_px == 16
    assert r.annulus_mass_fraction == pytest.approx(1.0)
    assert r.total_fisher_mass == pytest.approx(16 * 0.5)
    assert r.mean_fisher_in_annulus == pytest.approx(0.5)


def test_annulus_report_none_inside():
    m = np.full((3, 3), 10.0)
    r = fisher_mass_in_annulus(m, band=2.0)
    assert r.annulus_px == 0
    assert r.annulus_fisher_mass == 0.0
    assert r.mean_fisher_in_annulus == 0.0
    assert 0.0 <= r.annulus_mass_fraction < 1e-3


def test_annulus_mass_fraction_exceeds_px_fraction_when_boundary_present():
    # Fisher mass CONCENTRATES on the low-margin annulus: with half the pixels
    # at m=0.1 and half at m=6, mass fraction must far exceed px fraction... px
    # fraction is 0.5 here, so check mass fraction > px fraction.
    m = np.concatenate([np.full(50, 0.1), np.full(50, 6.0)])
    r = fisher_mass_in_annulus(m, band=2.0)
    assert r.annulus_px_fraction == pytest.approx(0.5)
    assert r.annulus_mass_fraction > 0.9  # concentration, the point of the observable


def test_annulus_uses_abs_margin():
    m = np.array([-1.0, 1.0, -5.0, 5.0])
    r = fisher_mass_in_annulus(m, band=2.0)
    assert r.annulus_px == 2


def test_report_is_observer_only_markers():
    r = fisher_mass_in_annulus(np.zeros(4))
    assert r.promotable is False
    assert r.score_claim is False
    assert r.axis_tag == "[observer]"
    d = r.to_dict()
    assert d["promotable"] is False and d["score_claim"] is False


def test_fail_closed_on_nan():
    with pytest.raises(ValueError, match="non-finite"):
        fisher_mass_in_annulus(np.array([0.0, np.nan]))


def test_fail_closed_on_empty():
    with pytest.raises(ValueError, match="empty"):
        fisher_mass_in_annulus(np.zeros((0,)))


def test_fail_closed_on_bad_band():
    with pytest.raises(ValueError, match="band"):
        fisher_mass_in_annulus(np.zeros(4), band=0.0)
    with pytest.raises(ValueError, match="band"):
        fisher_mass_in_annulus(np.zeros(4), band=float("nan"))


def test_over_pairs_equals_concatenated():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 2, (8, 8))
    b = rng.normal(0, 2, (8, 8))
    agg = fisher_mass_report_over_pairs([a, b])
    one = fisher_mass_in_annulus(np.concatenate([a.ravel(), b.ravel()]))
    assert agg.total_fisher_mass == pytest.approx(one.total_fisher_mass)
    assert agg.annulus_fisher_mass == pytest.approx(one.annulus_fisher_mass)
    assert agg.annulus_mass_fraction == pytest.approx(one.annulus_mass_fraction)
    assert agg.total_px == one.total_px and agg.annulus_px == one.annulus_px


def test_over_pairs_empty_fails_closed():
    with pytest.raises(ValueError, match="no margin"):
        fisher_mass_report_over_pairs([])


def test_default_band_is_two():
    assert DEFAULT_ANNULUS_BAND == 2.0
    r = fisher_mass_in_annulus(np.array([1.9, 2.1]))
    assert r.annulus_px == 1
