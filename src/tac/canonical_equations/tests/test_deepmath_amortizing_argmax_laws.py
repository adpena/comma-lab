# SPDX-License-Identifier: MIT
"""Focused tests for the 8 proven 'Amortizing the Argmax' canonical equations (task #284 / A2).

Verifies: (a) every build_*_v1() returns a valid CanonicalEquation (id + non-empty anchors +
producers-or-consumers present); (b) the helper callables compute the STATED values (tau*ln5 for
tau=0.05,K=5; the exact annulus Fisher identity 1/2 sech^2(m/2) = 0.5 at m=0; etc.); (c) NO-FAKE
discipline: conjectures are recorded ASSUMED_AWAITING_VERIFICATION (not headline); the anisotropy
198:1 is DISPUTED; the module imports cleanly; the se(3) law is REFERENCED not duplicated.
"""
from __future__ import annotations

import math

import pytest

import tac.canonical_equations as ce
from tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704 import (
    ALL_DEEPMATH_AMORTIZING_ARGMAX_BUILDERS,
    ANISOTROPY_DISPUTE_EQUATION_ID,
    ANISOTROPY_GRADIENT_PROJECTION,
    ANISOTROPY_ON_THE_FLY_DISPUTED,
    ANISOTROPY_STRUCTURE_TENSOR,
    FISHER_CAUSTIC_EQUATION_ID,
    FISHER_CURVATURE_MARGIN_PEARSON_BAND,
    MASLOV_EQUATION_ID,
    MBO_SMOOTHING_LANE_FRACTION,
    MCF_ERASURE_EQUATION_ID,
    MD_NG_EQUATION_ID,
    MODICA_MORTOLA_EQUATION_ID,
    SHEARLET_RATE_EQUATION_ID,
    TAU_EPS_HBAR_EQUATION_ID,
    annulus_anisotropy_ratio,
    annulus_fisher_trace,
    build_annulus_anisotropy_magnitude_disputed_v1,
    build_ce_softmax_mirror_descent_natural_gradient_v1,
    build_fisher_curvature_equals_categorical_fisher_trace_caustic_v1,
    build_maslov_dequantization_bound_v1,
    build_shearlet_nterm_upper_bounds_task_rate_v1,
    categorical_bregman_divergence,
    categorical_fisher_trace_two_class,
    herring_triple_junction_angle_deg,
    maslov_dequantization_bound,
    mbo_smoothing_cost_lane_fraction,
    shearlet_nterm_error,
    tau_interface_halfwidth,
)
from tac.canonical_equations.equation import CanonicalEquation

_ALL_BUILDERS = ALL_DEEPMATH_AMORTIZING_ARGMAX_BUILDERS
_EXPECTED_IDS = {
    MASLOV_EQUATION_ID,
    FISHER_CAUSTIC_EQUATION_ID,
    MD_NG_EQUATION_ID,
    SHEARLET_RATE_EQUATION_ID,
    TAU_EPS_HBAR_EQUATION_ID,
    MODICA_MORTOLA_EQUATION_ID,
    MCF_ERASURE_EQUATION_ID,
    ANISOTROPY_DISPUTE_EQUATION_ID,
}


def test_eight_builders_registered() -> None:
    assert len(_ALL_BUILDERS) == 8
    ids = {b().equation_id for b in _ALL_BUILDERS}
    assert ids == _EXPECTED_IDS


@pytest.mark.parametrize("builder", _ALL_BUILDERS)
def test_each_builder_returns_valid_canonical_equation(builder) -> None:
    eq = builder()
    assert isinstance(eq, CanonicalEquation)
    assert eq.equation_id == eq.equation_id.lower()
    # non-empty anchors
    assert len(eq.empirical_anchors) >= 1
    # non-orphan: producers OR consumers present (CanonicalEquation.__post_init__ enforces)
    assert eq.canonical_producers or eq.canonical_consumers
    # every anchor carries a verification status (this module never leaves it None)
    for a in eq.empirical_anchors:
        assert a.empirical_verification_status in (
            "VERIFIED_VIA_EMPIRICAL_ANCHOR",
            "INFERRED_FROM_DOMAIN_LITERATURE",
            "ASSUMED_AWAITING_VERIFICATION",
        )
    # means/ends firewall recorded in domain_of_validity
    assert "0.19110" in str(eq.domain_of_validity)


def test_module_imports_cleanly_via_package() -> None:
    # exported from the package __init__ (queryable surface)
    assert ce.build_maslov_dequantization_bound_v1 is build_maslov_dequantization_bound_v1
    assert ce.build_annulus_anisotropy_magnitude_disputed_v1 is build_annulus_anisotropy_magnitude_disputed_v1
    assert ce.annulus_fisher_trace is annulus_fisher_trace


# ---- helper value checks (the laws compute the STATED numbers) ------------------------------------
def test_maslov_bound_tau_ln_k() -> None:
    # tau*ln K for tau=0.05, K=5.
    assert maslov_dequantization_bound(0.05, 5) == pytest.approx(0.05 * math.log(5))
    assert maslov_dequantization_bound(0.05, 5) == pytest.approx(0.0804718956, abs=1e-9)
    # monotone in tau -> 0 as tau -> 0.
    assert maslov_dequantization_bound(0.0, 5) == 0.0
    with pytest.raises(ValueError):
        maslov_dequantization_bound(-0.1, 5)
    with pytest.raises(ValueError):
        maslov_dequantization_bound(0.05, 1)


def test_annulus_fisher_trace_exact_identity() -> None:
    # tr F = 1/2 sech^2(m/2); at m=0 (boundary, p=0.5) the MAX Fisher = 1/2 * sech^2(0) = 0.5.
    # (NOTE: 1/2 * sech^2(0) = 0.5, NOT 0.125 -- the categorical Fisher trace 1 - sum p^2 at
    # p=0.5 is 1 - 0.5 = 0.5; the two agree exactly, which is the whole identity.)
    assert annulus_fisher_trace(0.0) == pytest.approx(0.5)
    assert annulus_fisher_trace(0.0) == pytest.approx(0.5 / math.cosh(0.0) ** 2)
    # the EXACT identity: annulus tr F(m) == categorical trace(sigma(m)).
    for m in (-2.0, -0.5, 0.0, 0.5, 2.0):
        p = 1.0 / (1.0 + math.exp(-m))  # sigma(m)
        assert annulus_fisher_trace(m) == pytest.approx(categorical_fisher_trace_two_class(p))
    # symmetric + decays off the boundary.
    assert annulus_fisher_trace(2.0) == pytest.approx(annulus_fisher_trace(-2.0))
    assert annulus_fisher_trace(2.0) < annulus_fisher_trace(0.0)


def test_categorical_fisher_trace_two_class_boundary() -> None:
    assert categorical_fisher_trace_two_class(0.5) == pytest.approx(0.5)  # 1 - 0.25 - 0.25
    assert categorical_fisher_trace_two_class(1.0) == pytest.approx(0.0)  # argmax stable -> dark
    assert categorical_fisher_trace_two_class(0.0) == pytest.approx(0.0)


def test_categorical_bregman_is_kl_and_zero_at_self() -> None:
    p = [0.2, 0.2, 0.2, 0.2, 0.2]
    assert categorical_bregman_divergence(p, p) == pytest.approx(0.0)
    # matches KL for a concrete pair.
    p2 = [0.5, 0.5]
    q2 = [0.25, 0.75]
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert categorical_bregman_divergence(p2, q2) == pytest.approx(expected)
    assert categorical_bregman_divergence(p2, q2) > 0.0  # non-negativity of KL


def test_shearlet_nterm_rate_and_asymptotics() -> None:
    # O(N^-2 (log N)^3): at N=64 it is N^-2 * (ln 64)^3.
    assert shearlet_nterm_error(64) == pytest.approx(64 ** -2.0 * math.log(64) ** 3)
    # monotone decreasing at large N; beats wavelet N^-1 ASYMPTOTICALLY (crossover is large N).
    assert shearlet_nterm_error(1_000_000) < 1_000_000 ** -1.0
    assert shearlet_nterm_error(10_000) < shearlet_nterm_error(1_000)


def test_tau_interface_halfwidth_is_tau_over_two() -> None:
    assert tau_interface_halfwidth(0.05) == pytest.approx(0.025)
    assert tau_interface_halfwidth(1.0) == pytest.approx(0.5)


def test_herring_equal_tension_angle_is_120() -> None:
    assert herring_triple_junction_angle_deg() == pytest.approx(120.0)
    assert herring_triple_junction_angle_deg((1.0, 1.0, 1.0)) == pytest.approx(120.0)
    with pytest.raises(ValueError):
        herring_triple_junction_angle_deg((1.0, 2.0, 1.0))


def test_mbo_lane_fraction_measured() -> None:
    assert mbo_smoothing_cost_lane_fraction() == pytest.approx(0.957)
    assert pytest.approx(0.957) == MBO_SMOOTHING_LANE_FRACTION


def test_anisotropy_settled_methods_and_disputed_198() -> None:
    assert annulus_anisotropy_ratio("gradient_projection") == pytest.approx(9.56)
    assert annulus_anisotropy_ratio("structure_tensor") == pytest.approx(37.8)
    assert pytest.approx(9.56) == ANISOTROPY_GRADIENT_PROJECTION
    assert pytest.approx(37.8) == ANISOTROPY_STRUCTURE_TENSOR
    # the 198:1 is DISPUTED -> NOT returned as a settled value.
    assert pytest.approx(198.0) == ANISOTROPY_ON_THE_FLY_DISPUTED
    with pytest.raises(ValueError):
        annulus_anisotropy_ratio("on_the_fly_disputed")


# ---- NO-FAKE discipline anchors ------------------------------------------------------------------
def test_fisher_caustic_records_measured_0978_and_derived_identity() -> None:
    eq = build_fisher_curvature_equals_categorical_fisher_trace_caustic_v1()
    assert len(eq.empirical_anchors) == 2
    identity, measured = eq.empirical_anchors
    assert identity.empirical_verification_status == "INFERRED_FROM_DOMAIN_LITERATURE"
    assert measured.empirical_verification_status == "VERIFIED_VIA_EMPIRICAL_ANCHOR"
    assert measured.empirical_output["pearson_band"] == pytest.approx(
        FISHER_CURVATURE_MARGIN_PEARSON_BAND)
    assert measured.empirical_output["spearman_global"] == pytest.approx(0.908)


def test_shearlet_tightness_is_conjecture_not_headline() -> None:
    eq = build_shearlet_nterm_upper_bounds_task_rate_v1()
    statuses = [a.empirical_verification_status for a in eq.empirical_anchors]
    # the UPPER BOUND (derived) + the D1 -48% (measured) + tightness (conjecture, NOT a claim).
    assert "INFERRED_FROM_DOMAIN_LITERATURE" in statuses
    assert "VERIFIED_VIA_EMPIRICAL_ANCHOR" in statuses
    assert "ASSUMED_AWAITING_VERIFICATION" in statuses
    tightness = eq.empirical_anchors[-1]
    assert tightness.empirical_output["tightness_proven"] is False
    assert "CONJECTURED" in str(eq.domain_of_validity)


def test_se3_referenced_not_duplicated() -> None:
    # the store-nothing se(3) law is the canonical home; the shearlet-rate law LINKS to it as a
    # consumer (rate-half payload |xi B-spline|) rather than registering a second se(3) law.
    eq = build_shearlet_nterm_upper_bounds_task_rate_v1()
    assert ("tac.canonical_equations.store_nothing_pose_carrier_rate_dpose_20260702"
            in eq.canonical_consumers)
    # and no builder here is an se(3) / screw / temporal-sufficiency law.
    ids = {b().equation_id for b in _ALL_BUILDERS}
    assert not any("se3" in i or "screw" in i or "temporal" in i for i in ids)


def test_anisotropy_correction_flags_198_disputed() -> None:
    eq = build_annulus_anisotropy_magnitude_disputed_v1()
    remeasured, disputed = eq.empirical_anchors
    assert remeasured.empirical_output["gradient_projection_ratio"] == pytest.approx(9.56)
    assert remeasured.empirical_output["structure_tensor_ratio"] == pytest.approx(37.8)
    assert disputed.empirical_verification_status == "ASSUMED_AWAITING_VERIFICATION"
    assert disputed.empirical_output["settled"] is False
    assert "DISPUTED" in str(eq.domain_of_validity)


def test_md_ng_conjecture_recorded_in_domain_not_as_law() -> None:
    eq = build_ce_softmax_mirror_descent_natural_gradient_v1()
    # the "trajectory literally IS NG-along-Maslov" conjecture is recorded in the domain, NOT a law.
    assert "CONJECTURE" in str(eq.domain_of_validity)
    assert eq.empirical_anchors[0].empirical_verification_status == "INFERRED_FROM_DOMAIN_LITERATURE"
