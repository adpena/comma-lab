from __future__ import annotations

import math

import numpy as np

from tac.analysis.ddm_db1_decay_bounds import (
    fit_decay_family,
    required_delta_bounds,
    unique_count_bounds,
)


def test_unique_count_bounds_use_global_duplicate_budget() -> None:
    # 105 total incidences represent 100 unique pixels, so any prefix can
    # contain at most five duplicate incidences.
    assert unique_count_bounds(
        3, total_incidence_count=105, unique_total=100
    ) == (0, 3)
    assert unique_count_bounds(
        30, total_incidence_count=105, unique_total=100
    ) == (25, 30)
    assert unique_count_bounds(
        105, total_incidence_count=105, unique_total=100
    ) == (100, 100)


def test_required_delta_bounds_are_order_statistic_bounds() -> None:
    distances = np.arange(1, 106, dtype=np.float64) / 1000.0
    row = required_delta_bounds(
        distances,
        unique_total=100,
        required_unique_count=30,
    )
    assert row == {
        "status": "BOUNDED_FROM_MARGIN_ORDER_STATISTICS",
        "necessary_delta_lower_bound": 0.030,
        "duplicate_budget_sufficient_delta_upper_bound": 0.035,
    }
    impossible = required_delta_bounds(
        distances,
        unique_total=100,
        required_unique_count=101,
    )
    assert impossible["status"] == "EXCEEDS_COMPLETE_FIXED_BOUNDARY_SUPPORT"


def test_exponential_fit_recovers_synthetic_asymptote_and_target_refusal() -> None:
    admissions = np.arange(1, 105, dtype=np.float64)
    values = 0.0244 + 0.002 * np.exp(-0.019 * admissions)
    fitted = fit_decay_family(admissions, values, "exponential")
    assert math.isclose(fitted.asymptote, 0.0244, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(fitted.exponent, 0.019, rel_tol=0.0, abs_tol=1e-8)
    assert fitted.target_projection(8.684e-4, 104)["status"] == (
        "ASYMPTOTE_AT_OR_ABOVE_TARGET"
    )


def test_power_fit_exposes_extreme_finite_extrapolation() -> None:
    admissions = np.arange(1, 105, dtype=np.float64)
    values = 0.03 * admissions ** (-0.02)
    fitted = fit_decay_family(admissions, values, "power")
    projection = fitted.target_projection(8.684e-4, 104)
    assert fitted.asymptote < 1e-9
    assert projection["status"] == "FINITE_MODEL_EXTRAPOLATION"
    assert projection["log10_total_admissions"] > 70
