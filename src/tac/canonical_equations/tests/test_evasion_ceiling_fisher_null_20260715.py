# SPDX-License-Identifier: MIT
"""Isolated tests for evasion_ceiling_fisher_null_20260715 (locked-registry pattern)."""

from __future__ import annotations

import math

import pytest

from tac.canonical_equations.evasion_ceiling_fisher_null_20260715 import (
    ANNULUS_THRESHOLD_MARGIN,
    EQUATION_ID,
    G_GAIN,
    build_evasion_ceiling_fisher_null_interior_v1,
    flip_cost_render_l2,
    render_rms_to_flip_interior,
)


def test_cost_monotone_and_exponential() -> None:
    # cost(m) = (m/G)cosh(m/2) is strictly increasing and grows super-linearly.
    c1 = flip_cost_render_l2(0.5)
    c2 = flip_cost_render_l2(2.0)
    c3 = flip_cost_render_l2(5.0)
    assert 0.0 < c1 < c2 < c3
    # super-linear: doubling+ margin more than doubles cost
    assert c2 / c1 > 4.0


def test_cost_symmetric_in_sign() -> None:
    assert flip_cost_render_l2(-1.3) == pytest.approx(flip_cost_render_l2(1.3))


def test_cost_zero_at_boundary() -> None:
    assert flip_cost_render_l2(0.0) == pytest.approx(0.0)


def test_cost_matches_closed_form() -> None:
    m = 2.0
    assert flip_cost_render_l2(m) == pytest.approx((m / G_GAIN) * math.cosh(m / 2.0))


def test_interior_flip_rms_is_catastrophic() -> None:
    # Flipping the 4.7%-area annulus-threshold interior pixel needs ~50 LSB RMS ==> null.
    rms = render_rms_to_flip_interior(ANNULUS_THRESHOLD_MARGIN)
    assert rms > 40.0  # far above any realistic render error (<=4 LSB)


def test_bad_gain_rejected() -> None:
    with pytest.raises(ValueError):
        flip_cost_render_l2(1.0, g_gain=0.0)


def test_equation_builds_and_is_consistent() -> None:
    eq = build_evasion_ceiling_fisher_null_interior_v1()
    assert eq.equation_id == EQUATION_ID
    assert eq.domain_of_validity["research_only"] is True
    assert eq.domain_of_validity["score_claim"] is False
    # the load-bearing MEASURED anchor: interior leak == 0
    ids = {a.anchor_id: a for a in eq.empirical_anchors}
    interior = ids["evasion_interior_null_20260715"]
    assert interior.empirical_output["interior_leak_rate_at_rho_0p25_to_4p0_LSB"] == 0.0
    assert interior.empirical_output["evadable_by_relocation_fraction"] == 0.0
    assert interior.residual == 0.0
