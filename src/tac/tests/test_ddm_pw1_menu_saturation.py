# SPDX-License-Identifier: MIT
"""Behaviour tests for the menu-saturation discriminator (ddm_pw1).

These test BEHAVIOUR on the four measured menus, not constants.  In particular the
suite pins the two cases where the instrument must REFUSE rather than guess -- the
measured false negative is a first-class property here, not a bug to be smoothed over.
"""

from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_pw1_menu_saturation_discriminator_20260801 import (
    DIM0_MOVE_HISTOGRAM,
    build_ddm_pw1_menu_saturation_discriminator_v1,
    menu_saturation,
)

SAT = "SATURATED_MEASURE_BEYOND_BOUND"
CLOSED = "CLOSED_INTERIOR_OPTIMUM"


def test_dim0_the_measured_positive_is_saturated_and_certifiable() -> None:
    """The strongest measured positive: 8.27x terminal spike over a falling interior."""
    r = menu_saturation(DIM0_MOVE_HISTOGRAM)
    assert r["verdict"] == SAT
    assert r["sufficient_for_verdict"] is True
    assert r["terminal_to_last_interior_ratio"] == pytest.approx(124 / 15, rel=1e-9)
    assert r["verdict_is_a_measurement_request"] is True


def test_s_t_the_negative_control_is_closed() -> None:
    """s_t occupied 6-9 only, terminal bin EMPTY -- the control that makes it specific."""
    st = [0] * 6 + [1, 1, 1, 1, 0]
    r = menu_saturation(st)
    assert r["verdict"] == CLOSED
    assert r["sufficient_for_verdict"] is True
    assert r["terminal_count"] == 0


def test_short_menu_refuses_rather_than_guessing() -> None:
    """beta is a REAL positive (26.4% of mass) that a 3-entry histogram cannot see.

    The instrument must decline to certify, naming the reason -- never emit a
    confident CLOSED on a menu with no interior trend.
    """
    r = menu_saturation([262, 262, 76])
    assert r["sufficient_for_verdict"] is False
    assert "shorter_than_5" in str(r["insufficiency_reason"])


def test_objective_mass_overrides_count_when_supplied() -> None:
    """Count and mass can disagree ~2x; mass is the authority when present."""
    counts = [50, 40, 30, 20, 10, 25]
    # terminal carries little count but dominant objective mass
    r_mass = menu_saturation(counts, objective_mass=[1.0, 1.0, 1.0, 1.0, 1.0, 40.0])
    assert r_mass["decided_by"] == "objective_mass"
    assert r_mass["verdict"] == SAT
    r_count = menu_saturation(counts)
    assert r_count["decided_by"] == "count"


def test_empty_scope_is_vacuous_never_closed() -> None:
    """The vacuity genus: no selections is UNDETERMINED, never a clean CLOSED."""
    r = menu_saturation([0, 0, 0, 0, 0, 0])
    assert r["verdict"] == "UNDETERMINED_EMPTY"
    assert r["verdict"] != CLOSED


def test_zero_length_histogram_raises() -> None:
    with pytest.raises(ValueError, match="VACUOUS"):
        menu_saturation([])


def test_negative_counts_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        menu_saturation([1, -1, 3])


def test_misaligned_objective_mass_rejected() -> None:
    with pytest.raises(ValueError, match="align 1:1"):
        menu_saturation([1, 2, 3, 4, 5], objective_mass=[1.0, 2.0])


def test_explicit_terminal_index_handles_non_last_bounds() -> None:
    """A signed table bounded at both ends is called twice, once per bound."""
    h = [40, 10, 8, 6, 30]
    lo = menu_saturation(h, terminal_index=0)
    hi = menu_saturation(h, terminal_index=4)
    assert lo["terminal_count"] == 40
    assert hi["terminal_count"] == 30


def test_equation_builds_with_both_measured_anchors() -> None:
    """Two independent instances (a value menu and a solver cap) lift it off INSTANCE."""
    eq = build_ddm_pw1_menu_saturation_discriminator_v1()
    assert len(eq.empirical_anchors) == 2
    ids = {a.anchor_id for a in eq.empirical_anchors}
    assert any("pw1" in i for i in ids)
    assert any("dc1" in i for i in ids)


def test_exact_eval_anchor_carries_the_realized_delta_not_the_prediction() -> None:
    """The anchor's residual is prediction-vs-exact, and the exact row is authority."""
    eq = build_ddm_pw1_menu_saturation_discriminator_v1()
    pw1 = next(a for a in eq.empirical_anchors if "pw1" in a.anchor_id)
    assert pw1.empirical_output["exact_eval_composed_S"] == pytest.approx(0.9476091)
    assert pw1.empirical_output["d_seg_change"] == 0.0
    # the win is entirely pose; rate went UP
    assert pw1.empirical_output["archive_bytes_added"] > 0
    assert pw1.residual < 1e-5
