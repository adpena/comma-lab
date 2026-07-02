# SPDX-License-Identifier: MIT
"""Focused tests for the POWERPLAY (arXiv:1112.5309) Variant-II cost isomorphism canonical equation."""
from __future__ import annotations

from tac.canonical_equations.powerplay_variant_ii_cost_isomorphism_20260702 import (
    EQUATION_ID,
    build_powerplay_variant_ii_cost_isomorphism_v1 as build_eq,
    contest_score_as_powerplay_cost,
)
from tac.contest_score import compute_contest_score


def test_equation_builds_and_validates() -> None:
    eq = build_eq()
    assert eq.equation_id == EQUATION_ID == eq.equation_id.lower()
    assert len(eq.empirical_anchors) == 1
    # non-orphan contract: producers AND consumers present
    assert eq.canonical_producers and eq.canonical_consumers
    # the anchored callable is the canonical DSL powerplay cost (single source of truth)
    assert eq.python_callable_module_path == "tac.witness_dsl.powerplay:powerplay_cost"
    # campaign-meta consumers: the executable powerplay surface + the campaign decide loop
    assert "tac.witness_dsl.powerplay" in eq.canonical_consumers
    assert "tac.witness_dsl.campaign" in eq.canonical_consumers


def test_isomorphism_is_an_exact_identity_for_arbitrary_inputs() -> None:
    # S == POWERPLAY Variant-II cost EXACTLY, for ALL inputs (both call seg/pose/rate_term).
    for d_seg, d_pose, b in [(0.0, 0.0, 0), (0.006655, 0.096572, 83062),
                             (0.001, 3.4e-5, 37_545_489), (0.5, 12.0, 1_000_000)]:
        cost = contest_score_as_powerplay_cost(d_seg, d_pose, b)
        assert cost.S == compute_contest_score(d_seg, d_pose, b)
        # L(s) (rate) + task deficit == S
        assert abs((cost.description_bits_term + cost.task_deficit_term) - cost.S) < 1e-12


def test_anchor_records_verified_via_source_inspection_zero_residual() -> None:
    eq = build_eq()
    a = eq.empirical_anchors[0]
    assert a.residual == 0.0  # exact algebraic identity
    assert a.empirical_verification_status == "VERIFIED_VIA_SOURCE_INSPECTION"
    assert a.empirical_output["identity_holds"] is True
    # honest: an illustrative identity witness, NOT a frontier score row
    assert a.empirical_output["identity_witness_not_a_frontier_row"] is True


def test_domain_names_the_three_campaign_mechanisms() -> None:
    eq = build_eq()
    mech = eq.domain_of_validity["mechanisms_named"]
    assert "correctness_demonstration" in mech   # review axis-9
    assert "variant_ii_acceptance" in mech        # compose-without-regression / net-S gate
    assert "simplest_unsolvable_order" in mech     # the #216 instrument ordering
    # it is a structural isomorphism, NOT a contest lever
    assert "NOT a contest lever" in eq.domain_of_validity["result_type"]
