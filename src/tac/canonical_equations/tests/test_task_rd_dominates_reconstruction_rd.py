# SPDX-License-Identifier: MIT
"""Focused tests for the task-R(D) < reconstruction-R(D) dominance canonical equation."""
from __future__ import annotations

from tac.canonical_equations.task_rd_dominates_reconstruction_rd_20260702 import (
    EQUATION_ID,
    build_task_rd_dominates_reconstruction_rd_v1 as build_eq,
    task_rd_dominance_gap,
    task_rd_le_reconstruction_rd,
)


def test_equation_builds_and_validates() -> None:
    eq = build_eq()
    assert eq.equation_id == EQUATION_ID == eq.equation_id.lower()
    assert len(eq.empirical_anchors) == 1
    assert eq.canonical_producers and eq.canonical_consumers
    # the task-space witness generator consumes the dominance rationale
    assert "tac.boundary_math.lever_b_generator" in eq.canonical_consumers


def test_dominance_gap_is_nonnegative_by_theorem() -> None:
    # gap = R_X(D) - R_T(D) >= 0 at equal task-distortion
    assert task_rd_dominance_gap(0.118, 0.059) == 0.059
    assert task_rd_dominance_gap(1.0, 1.0) == 0.0
    # a raw negative (would violate the theorem) clamps to 0; the sign helper reports the violation
    assert task_rd_dominance_gap(0.5, 0.9) == 0.0
    assert task_rd_le_reconstruction_rd(0.118, 0.059) is True
    assert task_rd_le_reconstruction_rd(0.5, 0.9) is False


def test_anchor_is_citation_inferred_from_literature() -> None:
    eq = build_eq()
    a = eq.empirical_anchors[0]
    assert a.residual == 0.0  # a cited inequality bound, not a measured residual
    assert a.empirical_verification_status == "INFERRED_FROM_DOMAIN_LITERATURE"
    cites = a.inputs["citations"]
    assert any("2602.12866" in c for c in cites)
    assert any("Dobrushin" in c for c in cites)


def test_it_is_framing_not_a_contest_lever_with_honest_corroboration_caveat() -> None:
    eq = build_eq()
    assert "FRAMING THEOREM" in eq.domain_of_validity["result_type"]
    # the bc20/bc36 corroboration is honestly labeled as a reskin, not equal-distortion
    corr = eq.domain_of_validity["corroboration_only"]
    assert "caveat" in corr and "reskin" in corr["caveat"]
    assert corr["bc20_task_space_rate"] < corr["bc36_reskin_rate"]
