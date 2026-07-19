# SPDX-License-Identifier: MIT
"""Focused checks for the Einstein--Kolmogorov action/rate contract."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tac.canonical_equations.einstein_kolmogorov_crux_20260719 import (
    EQUATION_ID,
    SOURCE_MEASUREMENT,
    InfeasibleByteBudgetError,
    MeasuredHardRReceipt,
    build_einstein_kolmogorov_crux_action_rate_contract_v1,
    contest_action,
    derive_research_only_decision,
    fixed_byte_palette_delta,
    maximum_byte_budget,
    populate_einstein_kolmogorov_crux_action_rate_contract_v1,
)


@pytest.mark.parametrize(
    ("target", "d_pose", "expected"),
    [
        (0.1910828242, 0.0, 264_150),
        (0.1910828242, 1.0184e-4, 216_223),
        (0.15, 0.0, 202_451),
        (0.15, 1.0184e-4, 154_524),
    ],
)
def test_authority_derived_byte_cap_examples(target: float, d_pose: float, expected: int) -> None:
    # Values are formula examples from the design memo, not a tournament result.
    budget = maximum_byte_budget(target_action=target, d_seg=1.5196e-4, d_pose=d_pose)
    assert abs(budget - expected) <= 1


def test_action_and_budget_are_monotone() -> None:
    assert contest_action(d_seg=0.001, d_pose=0.0, archive_bytes=101) > contest_action(
        d_seg=0.001, d_pose=0.0, archive_bytes=100
    )
    assert maximum_byte_budget(target_action=0.2, d_seg=0.001, d_pose=0.0) > maximum_byte_budget(
        target_action=0.19, d_seg=0.001, d_pose=0.0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_action": 0.1, "d_seg": 0.002, "d_pose": 0.0},
        {"target_action": float("nan"), "d_seg": 0.0, "d_pose": 0.0},
    ],
)
def test_budget_rejects_infeasible_or_invalid_inputs(kwargs: dict[str, float]) -> None:
    expected = InfeasibleByteBudgetError if kwargs["target_action"] == 0.1 else ValueError
    with pytest.raises(expected):
        maximum_byte_budget(**kwargs)


def test_fixed_byte_palette_delta_has_no_rate_term() -> None:
    delta = fixed_byte_palette_delta(
        before_d_seg=0.02,
        before_d_pose=0.01,
        after_d_seg=0.015,
        after_d_pose=0.01,
        before_bytes=19_859,
        after_bytes=19_859,
    )
    assert math.isclose(delta, -0.5, abs_tol=1e-12)
    with pytest.raises(ValueError, match="identical packet bytes"):
        fixed_byte_palette_delta(
            before_d_seg=0.02,
            before_d_pose=0.0,
            after_d_seg=0.01,
            after_d_pose=0.0,
            before_bytes=1,
            after_bytes=2,
        )


def test_receipt_derivation_edges_keep_research_only_scope() -> None:
    receipt = MeasuredHardRReceipt(
        receipt_id="hard-r-receipt:caller-supplied",
        verdict_scope="n24 hard-R research-only",
        d_seg=0.001,
        d_pose=0.0,
        archive_bytes=100,
    )
    decision = derive_research_only_decision(receipt=receipt, target_action=0.2)
    assert decision.equation_id == EQUATION_ID
    assert decision.research_only is True
    assert decision.promotion_eligible is False
    assert [edge.relation for edge in decision.derivation_edges] == ["MEASURED_HARD_R_INPUT", "DERIVES", "SCOPES"]


def test_hash_bound_canonical_equation_builds_from_frozen_measurement() -> None:
    equation = build_einstein_kolmogorov_crux_action_rate_contract_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.provenance.source_path == SOURCE_MEASUREMENT
    assert equation.domain_of_validity["anchor_measurement_sha256"] == (equation.provenance.source_sha256)
    assert equation.domain_of_validity["research_only"] is True
    assert equation.domain_of_validity["promotion_eligible"] is False
    assert len(equation.empirical_anchors) == 1
    anchor = equation.empirical_anchors[0]
    assert anchor.provenance.source_sha256 == equation.provenance.source_sha256
    assert anchor.empirical_output["winner_hard_mismatch_px"] < (anchor.empirical_output["source_hard_mismatch_px"])
    assert anchor.empirical_output["full_archive_or_contest_score_claim"] is False


def test_canonical_equation_registry_query_roundtrip(tmp_path: Path) -> None:
    from tac.canonical_equations.registry import query_equations

    registry = tmp_path / "canonical_equations.jsonl"
    equation = populate_einstein_kolmogorov_crux_action_rate_contract_v1(
        path=registry,
        lock_path=tmp_path / "canonical_equations.lock",
        agent="pytest",
        subagent_id="einstein-kolmogorov-registry-test",
    )
    loaded = query_equations(path=registry)
    assert [item.equation_id for item in loaded] == [EQUATION_ID]
    assert loaded[0].provenance.source_sha256 == equation.provenance.source_sha256
    assert loaded[0].empirical_anchors[0].empirical_output == (equation.empirical_anchors[0].empirical_output)
