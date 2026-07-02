# SPDX-License-Identifier: MIT
"""Focused tests for the through-R waterfill store-vs-witness-capacity canonical equation (FEED-wf)."""
from __future__ import annotations

import math

import pytest

from tac.canonical_equations.waterfill_annulus_through_r_store_vs_capacity_20260701 import (
    EQUATION_ID,
    build_waterfill_annulus_through_R_store_vs_capacity_v1 as build_eq,
    flip_concentration_enrichment,
    predict_store_realized_residual,
)


def test_equation_builds_and_validates() -> None:
    eq = build_eq()
    assert eq.equation_id == EQUATION_ID
    assert eq.equation_id == eq.equation_id.lower()
    assert len(eq.empirical_anchors) == 2
    # non-orphan contract: producers AND consumers present
    assert eq.canonical_producers and eq.canonical_consumers
    assert "tools/measure_waterfill_through_R.py" in eq.canonical_producers
    # the DEFERRED port module is a declared consumer (its reactivation criterion IS this equation)
    assert "tac.torch_vehicle.boundary_routing" in eq.canonical_consumers


def test_flip_concentration_enrichment_positive_finding() -> None:
    # margin<3 annulus: 5.71% of pixels captures 98.23% of flips -> ~17x enrichment (routing key valid)
    e = flip_concentration_enrichment(0.9823, 0.0571)
    assert e == pytest.approx(17.2, abs=0.2)
    assert e > 1.0  # flips ARE denser in the low-margin band than uniform


def test_realization_efficiency_reproduces_measured_negative() -> None:
    # eta_R ~ 0.35: only ~35% of the idealized correction survives R -> targeted paste dominated
    res_noov, res_ideal, res_realized = 0.004487, 0.000489, 0.003088
    eta = (res_noov - res_realized) / (res_noov - res_ideal)
    assert eta == pytest.approx(0.35, abs=0.02)
    pred = predict_store_realized_residual(res_noov, res_ideal, eta)
    assert pred == pytest.approx(res_realized, abs=1e-9)


def test_store_side_negative_full_store_dominates_targeted() -> None:
    # realized store-side S at a targeted tau (100*B_res + rate) exceeds the full-store S 0.10575
    res_realized_t12, rate_t12 = 0.003088, 0.0828
    s_targeted = 100.0 * res_realized_t12 + rate_t12
    s_full_store = 0.10575
    assert s_targeted > s_full_store  # targeted is DOMINATED -> STORE lever NEGATIVE


def test_efficiency_clamped_and_finite() -> None:
    assert predict_store_realized_residual(0.004, 0.0004, 5.0) == pytest.approx(0.0004)  # clamp>1 -> full
    assert predict_store_realized_residual(0.004, 0.0004, -1.0) == pytest.approx(0.004)  # clamp<0 -> noop
    with pytest.raises(ValueError):
        flip_concentration_enrichment(0.9, 0.0)
