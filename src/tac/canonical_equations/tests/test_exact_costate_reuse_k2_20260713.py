import pytest

from tac.canonical_equations.exact_costate_reuse_k2_20260713 import (
    EQUATION_ID,
    amortized_cost_fraction,
    build_exact_costate_reuse_k2_guarded_v1,
    exact_backward_call_amortization,
    exact_costate_reuse_k2_laws,
    full_facet_guard,
    terminal_costate_skip_admitted,
)


def test_k2_cost_law_and_non_k2_refusal():
    assert amortized_cost_fraction(alpha=0.2) == pytest.approx(0.6)
    assert amortized_cost_fraction(alpha=0.2, fallback_rate=0.25) == pytest.approx(0.7)
    assert exact_backward_call_amortization(reuse_accept_fraction=1.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        amortized_cost_fraction(alpha=0.2, cadence=3)


def test_full_facet_guard_and_terminal_method_distinction():
    assert full_facet_guard(
        anchor_ce=1.0,
        candidate_ce=0.9,
        anchor_d_seg=0.2,
        candidate_d_seg=0.2,
        anchor_d_pose=0.3,
        candidate_d_pose=0.3,
    )
    assert not full_facet_guard(
        anchor_ce=1.0,
        candidate_ce=0.9,
        anchor_d_seg=0.2,
        candidate_d_seg=0.21,
        anchor_d_pose=0.3,
        candidate_d_pose=0.3,
    )
    assert terminal_costate_skip_admitted(
        exact_metric_accept_reject=True,
        effective_dimension=None,
        deterministic_dimension_certificate=False,
        n_pairs=600,
        receipt_custody_valid=True,
    )
    assert not terminal_costate_skip_admitted(
        exact_metric_accept_reject=False,
        effective_dimension=3,
        deterministic_dimension_certificate=True,
        n_pairs=600,
        receipt_custody_valid=True,
    )


def test_laws_inject_cost_guard_and_terminal_skip():
    laws = exact_costate_reuse_k2_laws(
        alpha=0.25,
        anchor_ce=1.0,
        candidate_ce=0.8,
        anchor_d_seg=0.2,
        candidate_d_seg=0.2,
        anchor_d_pose=0.3,
        candidate_d_pose=0.29,
        effective_dimension=2,
        deterministic_dimension_certificate=True,
        terminal_n_pairs=600,
        terminal_receipt_custody_valid=True,
        reuse_accept_fraction=1.0,
    )
    assert laws == {
        "cadence": 2,
        "n_pairs": 600,
        "amortized_cost_fraction": 0.625,
        "teacher_slice_speedup": 1.6,
        "exact_backward_call_amortization": 2.0,
        "full_facet_guard_admitted": True,
        "terminal_costate_skip_admitted": True,
    }


def test_equation_declares_consumers_producer_and_pending_authority():
    equation = build_exact_costate_reuse_k2_guarded_v1()
    assert equation.equation_id == EQUATION_ID
    assert set(equation.canonical_consumers) == {
        "tac.witness_control.exact_costate_reuse",
        "tac.witness_dsl.exact_costate_reuse_policy",
        "tac.through_r.terminal_costate_skip",
    }
    assert equation.canonical_producers == ("tools.probe_p0_costate_reuse_k2",)
    assert equation.domain_of_validity["provider_current"] is False
    assert equation.empirical_anchors == ()
