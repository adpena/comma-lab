from __future__ import annotations

import math

import numpy as np
import pytest

from tac.canonical_equations.optimal_metric_unification_20260714 import (
    MARGIN_FISHER_PEARSON,
    RIPO_SPEARMAN_BINARY_VS_DIRECTIONAL,
    build_categorical_fisher_trust_region_winner_rival_v1,
    build_optimal_metric_unification_v1,
    optimal_metric_unification_law,
    populate_optimal_metric_unification_v1,
    winner_rival_trust_radius,
)
from tac.canonical_equations.registry import (
    get_equation_by_id,
    load_registry_events_lenient,
)
from tac.information_geometry.optimal_metric import (
    MetricGeometryError,
    annulus_fisher_trace_surrogate,
    log_partition_hessian,
    metric_directional_quadratic,
    softmax,
    squared_metric_quadratic,
    tempered_metric,
    tempered_winner_rival_curvature,
    winner_rival_curvature_via_metric,
)
from tac.optimization.ripo_fisher_trust_region import (
    winner_rival_curvature,
    winner_rival_radius,
)

# ---------------------------------------------------------------------------
# The metric IS diag(p) - p p^T (Bregman Hessian of logsumexp = categorical Fisher)
# ---------------------------------------------------------------------------


def test_log_partition_hessian_is_diag_minus_outer():
    p = np.array([0.5, 0.3, 0.2])
    g = log_partition_hessian(p)
    expected = np.diag(p) - np.outer(p, p)
    assert np.allclose(g, expected)
    # symmetric + PSD (Fisher information)
    assert np.allclose(g, g.T)
    assert np.min(np.linalg.eigvalsh(g)) >= -1e-12


def test_hessian_rejects_non_simplex():
    with pytest.raises(MetricGeometryError):
        log_partition_hessian(np.array([0.5, 0.4]))  # sums to 0.9
    with pytest.raises(MetricGeometryError):
        log_partition_hessian(np.array([1.5, -0.5]))  # negative entry


# ---------------------------------------------------------------------------
# FIDELITY reduction: C_wr is a DIRECTIONAL quadratic form of g == RIPO (bit-equal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p",
    [
        np.array([0.7, 0.2, 0.05, 0.03, 0.02]),
        np.array([0.994, 0.004, 0.001, 0.0007, 0.0003]),  # real interior operating point
        np.array([0.34, 0.33, 0.2, 0.08, 0.05]),  # near-tie annulus
        np.array([0.5, 0.5]),  # boundary two-class
    ],
)
def test_fidelity_reduction_bit_equal_to_ripo(p):
    """The unification's directional quadratic (e_w-e_r)^T g (e_w-e_r) equals the
    independently-implemented RIPO winner_rival_curvature. Proves the reduction is
    REAL (computed), not asserted."""
    via_metric = winner_rival_curvature_via_metric(p)
    via_ripo = float(winner_rival_curvature(p))
    assert math.isclose(via_metric, via_ripo, rel_tol=0.0, abs_tol=1e-12)


def test_fidelity_reduction_matches_closed_form():
    p = np.array([0.7, 0.2, 0.05, 0.03, 0.02])
    p_w, p_r = 0.7, 0.2
    assert math.isclose(
        winner_rival_curvature_via_metric(p),
        p_w + p_r - (p_w - p_r) ** 2,
        abs_tol=1e-12,
    )


def test_trust_radius_callable_matches_ripo():
    p = np.array([0.6, 0.25, 0.1, 0.03, 0.02])
    assert np.allclose(
        winner_rival_trust_radius(p, delta=0.02),
        winner_rival_radius(p, delta=0.02, delta_convention="delta_kl"),
    )


# ---------------------------------------------------------------------------
# TRAINING-LOSS reduction: tr g|2-class = 1/2 sech^2(m/2), monotone in margin
# ---------------------------------------------------------------------------


def test_annulus_trace_max_at_boundary():
    # boundary (m=0, p=0.5) is MAX Fisher = 0.5
    assert math.isclose(annulus_fisher_trace_surrogate(0.0), 0.5, abs_tol=1e-12)


def test_annulus_trace_monotone_decreasing_in_abs_margin():
    margins = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    vals = [annulus_fisher_trace_surrogate(m) for m in margins]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    # decays toward 0 far from the boundary (interior is Fisher-flat)
    assert vals[-1] < 1e-3


def test_annulus_trace_equals_two_class_fisher_trace():
    for m in (-3.0, -0.7, 0.0, 1.3, 5.0):
        p = 1.0 / (1.0 + math.exp(-m))
        two_class = 1.0 - p * p - (1.0 - p) ** 2  # = 2 p (1-p)
        assert math.isclose(annulus_fisher_trace_surrogate(m), two_class, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# CURRICULUM reduction: g(tau) = tau^-2 (diag(p_tau) - p_tau p_tau^T); concentrates
# ---------------------------------------------------------------------------


def test_tempered_metric_formula():
    logits = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
    tau = 0.7
    p_tau = softmax(logits, tau)
    expected = (np.diag(p_tau) - np.outer(p_tau, p_tau)) / (tau**2)
    assert np.allclose(tempered_metric(logits, tau), expected)


def test_curriculum_concentrates_on_separatrix_as_tau_drops():
    """As tau falls, the interior operating point sharpens (p_w -> 1) and the
    natural-coordinate winner-rival curvature vanishes -> the metric concentrates on
    the boundary. This is the DERIVED tau-dependence (not asserted)."""
    logits = np.array([4.0, 1.0, -1.0, -2.0, -3.0])
    readings = [tempered_winner_rival_curvature(logits, tau) for tau in (1.0, 0.5, 0.1)]
    p_ws = [r["p_w"] for r in readings]
    c_nat = [r["c_wr_natural"] for r in readings]
    assert p_ws[0] < p_ws[1] < p_ws[2]  # p_w increases as tau drops
    assert c_nat[0] > c_nat[1] > c_nat[2]  # interior curvature vanishes


def test_tempered_metric_rejects_nonpositive_tau():
    with pytest.raises(MetricGeometryError):
        tempered_metric(np.array([1.0, 0.0, -1.0]), 0.0)
    with pytest.raises(MetricGeometryError):
        tempered_metric(np.array([1.0, 0.0, -1.0]), -0.3)


# ---------------------------------------------------------------------------
# NO-FAKE: squared-Hessian dual no-solve is NOT the Fisher-natural primal length
# ---------------------------------------------------------------------------


def test_squared_hessian_differs_from_primal_quadratic():
    """The DUAL raw-mean no-solve length u^T g^2 u must differ from the PRIMAL
    Fisher-natural length u^T g u for a general SPD-restricted g (the landed guard).
    Conflating them is the forbidden squared-Hessian-as-Fisher-natural fake."""
    p = np.array([0.7, 0.2, 0.05, 0.03, 0.02])
    u = np.array([1.0, -1.0, 0.0, 0.0, 0.0])
    primal = metric_directional_quadratic(p, u)
    squared = squared_metric_quadratic(p, u)
    assert not math.isclose(primal, squared, rel_tol=1e-3)
    # squared-Hessian is u^T g^2 u = ||g u||^2 >= 0
    assert squared >= 0.0


# ---------------------------------------------------------------------------
# Canonical-equation build + registration (one event each, no dup/orphan)
# ---------------------------------------------------------------------------


def test_both_equations_build_and_validate():
    e_unif = build_optimal_metric_unification_v1()
    e_ripo = build_categorical_fisher_trust_region_winner_rival_v1()
    assert e_unif.equation_id == "optimal_metric_unification_v1"
    assert e_ripo.equation_id == "categorical_fisher_trust_region_winner_rival_v1"
    # neither is an orphan (both have producers AND consumers)
    for e in (e_unif, e_ripo):
        assert e.canonical_producers
        assert e.canonical_consumers
    # unification cites the measured margin<->Fisher band as an anchor
    assert any(
        a.empirical_output.get("pearson_band") == MARGIN_FISHER_PEARSON
        for a in e_unif.empirical_anchors
    )
    # RIPO eq carries the MEASURED spearman falsification
    assert any(
        a.empirical_output.get("spearman_binary_vs_directional")
        == RIPO_SPEARMAN_BINARY_VS_DIRECTIONAL
        for a in e_ripo.empirical_anchors
    )


def test_unification_callable_evaluates():
    out = optimal_metric_unification_law([3.0, 1.0, -1.0, -2.0, -3.0], tau=0.5)
    assert set(out) == {
        "fidelity_directional_curvature",
        "training_loss_margin_surrogate_trace",
        "curriculum_tau_reading",
        "logit_margin",
    }
    assert out["fidelity_directional_curvature"] >= 0.0
    assert 0.0 <= out["training_loss_margin_surrogate_trace"] <= 0.5


def test_populate_registers_both_idempotently(tmp_path):
    reg = tmp_path / "registry.jsonl"
    lock = tmp_path / "registry.lock"
    populate_optimal_metric_unification_v1(path=reg, lock_path=lock)
    e1 = get_equation_by_id("optimal_metric_unification_v1", path=reg)
    e2 = get_equation_by_id("categorical_fisher_trust_region_winner_rival_v1", path=reg)
    assert e1 is not None and e2 is not None
    # idempotent APPEND-ONLY: re-running appends new registered events, still resolvable
    populate_optimal_metric_unification_v1(path=reg, lock_path=lock)
    events = load_registry_events_lenient(path=reg)
    ids = [
        ev.get("equation_id")
        for ev in events
        if ev.get("event_type") == "registered"
    ]
    assert ids.count("optimal_metric_unification_v1") == 2
    assert ids.count("categorical_fisher_trust_region_winner_rival_v1") == 2
