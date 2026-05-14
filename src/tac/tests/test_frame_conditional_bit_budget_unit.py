# SPDX-License-Identifier: MIT
"""Unit tests for ``tac.codec.frame_conditional_bit_budget``.

Covers the 10 properties enumerated in the council prescription:

  1. Sum invariant — allocated bits sum to total_bit_budget within tolerance.
  2. Floor enforced — no frame below ``floor * avg``.
  3. Cap enforced — no frame above ``cap * avg``.
  4. eta=0 → uniform allocation regardless of complexity (sanity).
  5. eta=1 → proportional to complexity for unclamped frames.
  6. Higher eta → more concentrated on high-complexity frames.
  7. Edge: single frame (n=1) → all bits to that frame.
  8. Edge: identical complexities → uniform regardless of eta.
  9. Edge: zero complexity total → uniform fallback (still honours floor).
 10. Reproducibility — same inputs → identical outputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.codec.frame_conditional_bit_budget import (
    ComplexityComponents,
    allocate_per_frame_bits,
)

# Tolerance for "sums to total" checks: clamp/redistribute is a fixed-point
# iteration with float arithmetic; ±0.5 bits is the documented contract.
SUM_TOLERANCE = 0.5


def _uniform_mean(total: float, n: int) -> float:
    return total / n


# ─── 1. Sum invariant ──────────────────────────────────────────────────────


def test_sum_invariant_proportional() -> None:
    complexities = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    total = 1000.0
    bits = allocate_per_frame_bits(complexities, total, eta=1.0, floor=0.5, cap=2.0)
    assert abs(bits.sum() - total) < SUM_TOLERANCE


def test_sum_invariant_under_heavy_clamping() -> None:
    # Very skewed complexities should force cap-clamping; sum still holds.
    complexities = np.array([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    total = 8000.0
    bits = allocate_per_frame_bits(complexities, total, eta=2.0, floor=0.5, cap=1.5)
    assert abs(bits.sum() - total) < SUM_TOLERANCE


# ─── 2. Floor enforced ─────────────────────────────────────────────────────


def test_floor_enforced() -> None:
    complexities = np.array([1000.0, 0.001, 0.001, 0.001, 0.001])
    total = 5000.0
    floor = 0.5
    bits = allocate_per_frame_bits(complexities, total, eta=1.0, floor=floor, cap=2.0)
    mu = _uniform_mean(total, len(complexities))
    assert (bits >= floor * mu - 1e-6).all()


# ─── 3. Cap enforced ───────────────────────────────────────────────────────


def test_cap_enforced() -> None:
    complexities = np.array([0.001, 1000.0, 0.001, 0.001, 0.001])
    total = 5000.0
    cap = 1.5
    bits = allocate_per_frame_bits(complexities, total, eta=1.0, floor=0.5, cap=cap)
    mu = _uniform_mean(total, len(complexities))
    assert (bits <= cap * mu + 1e-6).all()


# ─── 4. eta=0 → uniform (sanity) ───────────────────────────────────────────


def test_eta_zero_is_uniform() -> None:
    complexities = np.array([1.0, 5.0, 100.0, 0.5, 17.0])
    total = 500.0
    bits = allocate_per_frame_bits(complexities, total, eta=0.0, floor=0.5, cap=2.0)
    mu = _uniform_mean(total, len(complexities))
    assert np.allclose(bits, mu, atol=1e-9)


# ─── 5. eta=1 → proportional to complexity for unclamped frames ────────────


def test_eta_one_is_proportional_when_unclamped() -> None:
    # Complexities chosen so the proportional allocation does NOT touch
    # the floor (0.5) or cap (2.0): all values within [0.5, 2.0]*mean.
    complexities = np.array([1.0, 1.2, 1.4, 1.6, 0.8])
    total = 1000.0
    bits = allocate_per_frame_bits(complexities, total, eta=1.0, floor=0.5, cap=2.0)
    expected = total * complexities / complexities.sum()
    np.testing.assert_allclose(bits, expected, rtol=1e-6, atol=1e-3)


# ─── 6. Higher eta → more concentration on high-complexity frames ──────────


def test_higher_eta_concentrates_on_high_complexity() -> None:
    # Small floor / large cap so concentration is observable at the extremes.
    complexities = np.array([1.0, 2.0, 3.0])
    total = 600.0
    bits_low = allocate_per_frame_bits(complexities, total, eta=0.5, floor=0.0, cap=10.0)
    bits_high = allocate_per_frame_bits(complexities, total, eta=2.0, floor=0.0, cap=10.0)
    # The largest-complexity frame (idx 2) gets MORE bits at eta=2 vs eta=0.5.
    assert bits_high[2] > bits_low[2]
    # The smallest-complexity frame (idx 0) gets FEWER bits at eta=2 vs eta=0.5.
    assert bits_high[0] < bits_low[0]


# ─── 7. Edge: single frame → all bits to that frame ─────────────────────────


def test_single_frame_gets_all_bits() -> None:
    bits = allocate_per_frame_bits([7.5], 1234.5, eta=1.0, floor=0.5, cap=2.0)
    assert bits.shape == (1,)
    assert bits[0] == pytest.approx(1234.5)


# ─── 8. Edge: identical complexities → uniform regardless of eta ───────────


def test_identical_complexities_yield_uniform() -> None:
    complexities = np.full(10, 7.0)
    total = 1000.0
    for eta in (0.0, 0.5, 1.0, 2.0, 4.0):
        bits = allocate_per_frame_bits(complexities, total, eta=eta, floor=0.5, cap=2.0)
        mu = _uniform_mean(total, 10)
        np.testing.assert_allclose(bits, mu, atol=1e-9)


# ─── 9. Edge: zero complexity total → uniform fallback (honours floor) ─────


def test_zero_complexity_total_uniform_fallback() -> None:
    complexities = np.zeros(8)
    total = 800.0
    bits = allocate_per_frame_bits(complexities, total, eta=1.0, floor=0.5, cap=2.0)
    mu = _uniform_mean(total, 8)
    np.testing.assert_allclose(bits, mu, atol=1e-9)
    # Floor ≤ mu (0.5 * mu ≤ mu) so floor is honoured by symmetry.
    assert (bits >= 0.5 * mu).all()


# ─── 10. Reproducibility — same inputs → identical outputs ─────────────────


def test_reproducibility_identical_outputs() -> None:
    complexities = np.array([1.7, 2.3, 0.4, 9.1, 5.5, 3.3, 0.001, 12.0])
    total = 1000.0
    bits_a = allocate_per_frame_bits(complexities, total, eta=1.5, floor=0.4, cap=2.5)
    bits_b = allocate_per_frame_bits(complexities, total, eta=1.5, floor=0.4, cap=2.5)
    bits_c = allocate_per_frame_bits(
        complexities.copy(), total, eta=1.5, floor=0.4, cap=2.5
    )
    np.testing.assert_array_equal(bits_a, bits_b)
    np.testing.assert_array_equal(bits_a, bits_c)


# ─── Bonus coverage: argument validation and ComplexityComponents ──────────


def test_negative_complexity_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        allocate_per_frame_bits([1.0, -0.1, 2.0], 100.0)


def test_non_finite_eta_rejected() -> None:
    with pytest.raises(ValueError, match="eta must be finite"):
        allocate_per_frame_bits([1.0, 2.0], 100.0, eta=float("inf"))


def test_floor_outside_unit_interval_rejected() -> None:
    with pytest.raises(ValueError, match="floor"):
        allocate_per_frame_bits([1.0, 2.0], 100.0, floor=1.5)


def test_cap_below_one_rejected() -> None:
    with pytest.raises(ValueError, match="cap"):
        allocate_per_frame_bits([1.0, 2.0], 100.0, cap=0.5)


def test_complexity_components_multiplies_factors() -> None:
    edge = np.array([1.0, 2.0, 3.0])
    var = np.array([4.0, 5.0, 6.0])
    diff = np.array([7.0, 8.0, 9.0])
    cc = ComplexityComponents(edge_density=edge, pixel_variance=var, frame_difference=diff)
    np.testing.assert_array_equal(cc.complexity, edge * var * diff)


def test_negative_eta_handles_zero_complexity_without_inf() -> None:
    # eta < 0 with a zero entry: implementation substitutes a tiny epsilon
    # so 0**(-1) doesn't blow up. Result must still sum to total.
    complexities = np.array([0.0, 1.0, 2.0])
    total = 600.0
    bits = allocate_per_frame_bits(complexities, total, eta=-1.0, floor=0.0, cap=10.0)
    assert np.isfinite(bits).all()
    assert abs(bits.sum() - total) < SUM_TOLERANCE
    # The zero-complexity entry gets the LARGEST allocation under eta<0
    # (heuristic only valid because we substitute epsilon).
    assert bits[0] >= bits[1]
    assert bits[0] >= bits[2]
