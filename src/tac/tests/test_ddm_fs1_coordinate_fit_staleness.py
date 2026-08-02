# SPDX-License-Identifier: MIT
"""Behaviour tests for ddm_fs1 coordinate-fit staleness.

Pins the two verdicts that carry the measured signal (extrapolated vs interpolated)
and the refusal that keeps a context-free row from reading as clean.
"""

from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_fs1_coordinate_fit_staleness_20260802 import (
    FIT_CONTEXT_KEY,
    FRESH,
    STALE_EXTRAPOLATED,
    STALE_FOREIGN_PARTNER,
    STALE_INTERPOLATED,
    UNDETERMINED_NO_CONTEXT,
    V4C_AB_FIT_MENU,
    V4C_AB_STALENESS_CENSUS,
    StaleFitContextError,
    census,
    fit_staleness,
    require_fit_context,
    stamp_fit_context,
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
    # The ft1 supersession widened stale_set from the beta-only 244 to 267 by
    # counting the POSE partner too, so stale_set is no longer bounded by
    # beta_nonzero -- a pair can be stale via pose drift with beta untouched.
    # The beta-only historical count still obeys the original bound.
    assert c["stale_beta_only"] <= min(c["beta_nonzero"], c["ab_non_identity"])
    assert c["stale_set"] <= c["ab_non_identity"]
    assert c["extrapolated_magnitude_only"] <= c["extrapolated"]


# --- ddm_sf1 extension: identity partners, the stamp, and the refusal --------


def test_identity_partner_drift_is_foreign_not_a_float_crash() -> None:
    """The gap that made 2 of the 3 measured instances unrepresentable.

    Before this, a base/vehicle partner raised ValueError on the float cast, so
    the pose-solved-on-celldrop50-applied-to-ep854 class and the C1-lattice-as-
    teacher-for-a-TR1-student class could not even be expressed by the ladder.
    """
    r = fit_staleness(
        coefficient="pose",
        fitted_against={"base": "celldrop50"},
        shipped_partners={"base": "ep854"},
    )
    assert r["verdict"] == STALE_FOREIGN_PARTNER
    assert r["drifted_identity"] == ("base",)
    # No hull to be inside of, so no Taylor step home: re-solve is the only cure.
    assert r["transport_admissible"] is False


def test_foreign_partner_outranks_numeric_drift() -> None:
    r = fit_staleness(
        coefficient="ab",
        fitted_against={"beta": 0.0, "vehicle": "C1"},
        shipped_partners={"beta": 0.5, "vehicle": "TR1"},
        fit_menu=V4C_AB_FIT_MENU,
        partner_key="beta",
    )
    assert r["verdict"] == STALE_FOREIGN_PARTNER  # not STALE_INTERPOLATED
    assert r["drifted_numeric"] == ("beta",)


def test_identity_partner_unchanged_is_fresh() -> None:
    r = fit_staleness(
        coefficient="pose",
        fitted_against={"base": "celldrop50", "beta": 0.0},
        shipped_partners={"base": "celldrop50", "beta": 0.0},
    )
    assert r["verdict"] == FRESH


def test_bool_partner_is_categorical_not_numeric() -> None:
    """A flag is an identity partner even though bool subclasses int."""
    r = fit_staleness(
        coefficient="c", fitted_against={"far": False}, shipped_partners={"far": True}
    )
    assert r["verdict"] == STALE_FOREIGN_PARTNER


def test_stamp_then_require_round_trips_to_fresh() -> None:
    ctx = stamp_fit_context(
        coefficient="ab_gain_bias",
        partners={"beta": 0.0, "p0": 1.5},
        base="celldrop50",
        fit_menu=V4C_AB_FIT_MENU,
    )
    v = require_fit_context(
        {FIT_CONTEXT_KEY: ctx},
        coefficient="ab_gain_bias",
        shipped_partners={"beta": 0.0, "p0": 1.5, "base": "celldrop50"},
        partner_key="beta",
    )
    assert v["verdict"] == FRESH


def test_stamped_menu_is_recovered_from_the_row() -> None:
    """The consumer need not know the producer's grid -- the stamp carries it."""
    ctx = stamp_fit_context(coefficient="ab", partners={"beta": 0.0},
                            fit_menu=V4C_AB_FIT_MENU)
    v = require_fit_context({FIT_CONTEXT_KEY: ctx}, coefficient="ab",
                            shipped_partners={"beta": 7.5}, partner_key="beta")
    assert v["verdict"] == STALE_EXTRAPOLATED


def test_unstamped_row_raises_instead_of_reading_fresh() -> None:
    with pytest.raises(StaleFitContextError, match="not freshness"):
        require_fit_context({}, coefficient="ab", shipped_partners={"beta": 7.5})


def test_fresh_only_boundary_rejects_a_known_stale_value() -> None:
    ctx = stamp_fit_context(coefficient="ab", partners={"beta": 0.0})
    with pytest.raises(StaleFitContextError, match="requires FRESH"):
        require_fit_context({FIT_CONTEXT_KEY: ctx}, coefficient="ab",
                            shipped_partners={"beta": 0.5}, allow_stale=False)


def test_empty_stamp_is_refused_because_it_would_certify_freshness() -> None:
    """An empty context is strictly worse than none: it reads FRESH against all."""
    with pytest.raises(ValueError, match="empty fit context"):
        stamp_fit_context(coefficient="ab", partners={})


def test_census_counts_the_foreign_class() -> None:
    rows = [
        {"fitted_against": {"base": "celldrop50"}, "shipped_partners": {"base": "ep854"}},
        {"fitted_against": {"base": "celldrop50"}, "shipped_partners": {"base": "celldrop50"}},
    ]
    c = census(rows)
    assert c[STALE_FOREIGN_PARTNER] == 1 and c[FRESH] == 1
