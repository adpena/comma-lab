# SPDX-License-Identifier: MIT
"""Behaviour tests for ddm_fs1 coordinate-fit staleness.

Pins the two verdicts that carry the measured signal (extrapolated vs interpolated)
and the refusal that keeps a context-free row from reading as clean.
"""

from __future__ import annotations

from tac.canonical_equations.ddm_fs1_coordinate_fit_staleness_20260802 import (
    FRESH,
    STALE_EXTRAPOLATED,
    STALE_INTERPOLATED,
    UNDETERMINED_NO_CONTEXT,
    V4C_AB_FIT_MENU,
    V4C_AB_STALENESS_CENSUS,
    census,
    fit_staleness,
)


def _ab(beta_at_fit: float, beta_shipped: float) -> dict:
    return dict(
        coefficient="ab_gain_bias",
        fitted_against={"beta": beta_at_fit},
        shipped_partners={"beta": beta_shipped},
        fit_menu=V4C_AB_FIT_MENU,
        partner_key="beta",
    )


def test_partner_unchanged_is_fresh() -> None:
    r = fit_staleness(**_ab(0.0, 0.0))
    assert r["verdict"] == FRESH and r["drifted"] == ()


def test_drift_inside_the_fitted_hull_is_interpolated() -> None:
    """beta moved 0.0 -> 0.5, still inside {0, 0.5, 1.0}: stale but bounded."""
    r = fit_staleness(**_ab(0.0, 0.5))
    assert r["verdict"] == STALE_INTERPOLATED
    assert r["outside_fit_hull"] is False
    assert r["drifted"] == ("beta",)


def test_drift_outside_the_hull_is_extrapolated() -> None:
    """The measured 59-pair class: mq1 shipped |beta| up to 7.5 against a 1.0 hull."""
    r = fit_staleness(**_ab(0.0, 7.5))
    assert r["verdict"] == STALE_EXTRAPOLATED
    assert r["outside_fit_hull"] is True


def test_missing_context_refuses_rather_than_passing() -> None:
    """The vacuity rule: an unstamped row is UNKNOWN, never FRESH."""
    r = fit_staleness(
        coefficient="ab_gain_bias", fitted_against=None, shipped_partners={"beta": 3.0}
    )
    assert r["verdict"] == UNDETERMINED_NO_CONTEXT
    assert r["verdict"] != FRESH
    assert r["sufficient_for_verdict"] is False
    assert "not FRESH" in r["insufficiency_reason"]


def test_no_menu_still_detects_drift_but_cannot_classify_hull() -> None:
    r = fit_staleness(
        coefficient="c", fitted_against={"b": 0.0}, shipped_partners={"b": 9.0}
    )
    assert r["verdict"] == STALE_INTERPOLATED
    assert r["outside_fit_hull"] is None  # unknowable without the fitted menu


def test_census_reports_every_verdict_including_the_unstamped() -> None:
    rows = [
        {"fitted_against": {"beta": 0.0}, "shipped_partners": {"beta": 0.0}},
        {"fitted_against": {"beta": 0.0}, "shipped_partners": {"beta": 0.5}},
        {"fitted_against": {"beta": 0.0}, "shipped_partners": {"beta": 7.5}},
        {"shipped_partners": {"beta": 2.0}},  # unstamped
    ]
    c = census(rows, fit_menu=V4C_AB_FIT_MENU, partner_key="beta")
    assert c["rows"] == 4
    assert c[FRESH] == 1
    assert c[STALE_INTERPOLATED] == 1
    assert c[STALE_EXTRAPOLATED] == 1
    assert c[UNDETERMINED_NO_CONTEXT] == 1


def test_measured_census_anchor_is_internally_consistent() -> None:
    c = V4C_AB_STALENESS_CENSUS
    assert c["extrapolated"] <= c["stale_set"] <= c["pairs"]
    assert c["stale_set"] <= min(c["beta_nonzero"], c["ab_non_identity"])
