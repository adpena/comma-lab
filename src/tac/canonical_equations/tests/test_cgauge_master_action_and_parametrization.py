# SPDX-License-Identifier: MIT
"""Behavior tests for the V9·CGauge master action + #223 parametrization optima."""
from __future__ import annotations

import math

import pytest

from tac.canonical_equations.cgauge_master_action_20260711 import (
    EDGE_LABELS,
    MEASURED_INPUT,
    VALUE_FORMS,
    axiom_ids,
    axioms,
    build_cgauge_master_action_v1,
    closed_loop_roles,
    derivation_edges,
    edge_label_of,
)
from tac.canonical_equations.cgauge_parametrization_optima_20260711 import (
    ALL_CGAUGE_PARAMETRIZATION_BUILDERS,
    along_tangent_deficit_ratio,
    bank_frequency_is_wasted,
    beta1_beta2_guard_ok,
    beta2_window,
    moddim_under_embeds,
    nyquist_bank_ceiling_cyc_per_unit,
    parabolic_along_tangent_allocation,
    whitney_mod_dim,
)


# ---- #223 Law 1: Whitney mod-dim ---------------------------------------------------
def test_whitney_mod_dim_derivation_not_constant() -> None:
    assert whitney_mod_dim(8, gauge_margin=0) == 17
    assert whitney_mod_dim(8, gauge_margin=2) == 19
    # behavior: scales with intrinsic dim (joint-10 hypothetical => 21)
    assert whitney_mod_dim(10, gauge_margin=0) == 21
    with pytest.raises(ValueError):
        whitney_mod_dim(0)


def test_mod16_under_embeds_mod17_does_not() -> None:
    assert moddim_under_embeds(16, 8) is True
    assert moddim_under_embeds(17, 8) is False
    # live mod-32 comfortably embeds (rate slack, not necessity)
    assert moddim_under_embeds(32, 8) is False


# ---- #223 Law 2: Nyquist ceiling ---------------------------------------------------
def test_nyquist_ceiling_matches_measured_mtf_zero() -> None:
    assert nyquist_bank_ceiling_cyc_per_unit() == pytest.approx(128.0)
    # parameter dependence: a later MTF zero (4px) halves the ceiling
    assert nyquist_bank_ceiling_cyc_per_unit(mtf_zero_period_px=4) == pytest.approx(64.0)


def test_deficit_band_is_below_ceiling_not_a_sampling_wall() -> None:
    assert bank_frequency_is_wasted(25.0) is False  # the dash line is usable through R
    assert bank_frequency_is_wasted(200.0) is True  # beyond the MTF zero => wasted


# ---- #223 Law 3: parabolic scaling --------------------------------------------------
def test_parabolic_allocation_reproduces_live_bank() -> None:
    # sqrt(64) = 8 == the live along-tangent allocation (n-dir-freqs=2)
    assert parabolic_along_tangent_allocation(64.0) == pytest.approx(8.0)
    with pytest.raises(ValueError):
        parabolic_along_tangent_allocation(0.0)


def test_deficit_ratio_matches_measured_3p2x_within_tolerance() -> None:
    r = along_tangent_deficit_ratio()
    assert r == pytest.approx(3.125)
    assert abs(r - 3.2) / 3.2 < 0.05  # derived 25/8 vs measured 3.2x


# ---- #223 Law 4: beta2 window --------------------------------------------------------
def test_beta2_window_floor_and_ceiling() -> None:
    lo, hi = beta2_window(75, curvature_timescale_epochs=100.0)
    assert lo == pytest.approx(1 - 1 / 75)
    assert hi == pytest.approx(1 - 3 / 7500)
    assert lo < 0.999 < hi          # the measured anchor is admissible
    assert not (lo < 0.9999999 < hi)  # the T0 candidate is derived-REJECTED
    with pytest.raises(ValueError):
        beta2_window(0)


def test_beta1_guard() -> None:
    assert beta1_beta2_guard_ok(0.9, 0.999) is True
    assert beta1_beta2_guard_ok(0.9995, 0.999) is False
    assert math.isclose(math.sqrt(0.999), 0.9994998749, rel_tol=1e-9)


# ---- all four #223 equations build + validate ---------------------------------------
def test_all_parametrization_equations_build() -> None:
    eqs = [b() for b in ALL_CGAUGE_PARAMETRIZATION_BUILDERS]
    assert len({e.equation_id for e in eqs}) == 4
    for e in eqs:
        assert e.empirical_anchors, e.equation_id
        assert e.canonical_producers and e.canonical_consumers, e.equation_id


# ---- master action system ------------------------------------------------------------
def test_master_action_builds_and_links_children_without_duplicating_anchors() -> None:
    eq = build_cgauge_master_action_v1()
    assert eq.equation_id == "cgauge_master_action_v1"
    assert len(eq.empirical_anchors) == 1  # ONE system anchor; children keep their own


def test_axioms_are_six_and_labeled() -> None:
    ax = axioms()
    assert axiom_ids() == ("A1", "A2", "A3", "A4", "A5", "A6")
    for a in ax:
        assert a["label"] in EDGE_LABELS
    # the ASSUMED axioms carry an audit note where audited
    assumed = [a for a in ax if a["label"] == "ASSUMED"]
    assert any("audit" in a for a in assumed)


def test_every_edge_is_labeled_and_parents_resolve() -> None:
    ax_ids = set(axiom_ids()) | {"S", "descent-of-S_tau"}
    children = {e["child"] for e in derivation_edges()}
    for e in derivation_edges():
        assert e["label"] in EDGE_LABELS, e["child"]
        assert e["operation"], e["child"]
        for p in e["parents"]:
            assert p in ax_ids or p in children, f"unresolved parent {p} of {e['child']}"


def test_new_223_laws_are_in_the_tree_as_derived() -> None:
    for cid in (
        "cgauge_whitney_moddim_v1",
        "cgauge_nyquist_bank_frequency_v1",
        "cgauge_curvelet_parabolic_bank_v1",
        "cgauge_beta2_window_v1",
    ):
        assert edge_label_of(cid) == "DERIVED", cid


def test_honesty_edges() -> None:
    # the eikonal CFL cure is honestly marked FALSIFIED_MECHANISM (n600 FEED-06g)
    assert edge_label_of("adaptive_eps_cfl_edge_tracking_v1") == "FALSIFIED_MECHANISM"
    # optimizer empirics are NOT derived from S
    assert edge_label_of("muon_finisher_schedule_warmstart_and_lr_anneal_v1") == MEASURED_INPUT
    # the totality audit is a measurement OF the axiom, not a derivation
    assert edge_label_of("witness_general_covariance_totality_v1") == MEASURED_INPUT


def test_closed_loop_has_costate_nexus_and_containment() -> None:
    roles = closed_loop_roles()
    assert "costate" in roles["adjoint"]
    assert "operator-GO" in roles["containment"]
    assert set(roles) == {"forward", "adjoint", "sense", "act", "law", "containment"}


def test_value_forms_vocabulary() -> None:
    assert len(VALUE_FORMS) == 7
    assert "SELF_DERIVING" in VALUE_FORMS and "POLYTOPE_KKT" in VALUE_FORMS
