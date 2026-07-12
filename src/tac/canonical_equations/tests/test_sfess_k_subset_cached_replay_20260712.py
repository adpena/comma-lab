# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import pytest

from tac.canonical_equations import get_equation_by_id
from tac.canonical_equations.sfess_k_subset_cached_replay_20260712 import (
    BEST_NONDEGENERATE_K,
    BEST_SFESS_S,
    COMPARISON_NOISE_FLOOR_S,
    EQUATION_ID,
    EXACT_ENUMERATION_S,
    SFESS_EXACT_GAP_S,
    build_sfess_fixed_k_cached_replay_ranking_v1,
    populate_sfess_fixed_k_cached_replay_equation,
)


def test_measured_law_keeps_degenerate_control_out_of_sfess_verdict() -> None:
    equation = build_sfess_fixed_k_cached_replay_ranking_v1()
    anchor = equation.empirical_anchors[0]

    assert equation.equation_id == EQUATION_ID
    assert BEST_NONDEGENERATE_K == 5
    assert pytest.approx(7.052840218513268e-7) == BEST_SFESS_S - EXACT_ENUMERATION_S
    assert SFESS_EXACT_GAP_S > COMPARISON_NOISE_FLOOR_S
    assert anchor.empirical_output["verdict"] == "NO-GO"
    assert anchor.empirical_output["same_budget_ranking_changed"] is False
    assert anchor.empirical_output["k6_is_one_state_control_not_estimator_evidence"] is True
    assert anchor.empirical_output["scorer_calls"] == 0
    assert equation.domain_of_validity["scope_level"] == "instance/formulation"
    assert "not a SFESS-family death verdict" in equation.domain_of_validity["exclusions"]


def test_populate_uses_append_only_registry(tmp_path) -> None:
    path = tmp_path / "equations.jsonl"
    lock_path = tmp_path / "equations.lock"

    populate_sfess_fixed_k_cached_replay_equation(
        path=path,
        lock_path=lock_path,
        agent="pytest",
        subagent_id="sfess-test",
    )
    populate_sfess_fixed_k_cached_replay_equation(
        path=path,
        lock_path=lock_path,
        agent="pytest",
        subagent_id="sfess-test",
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert {row["equation_id"] for row in rows} == {EQUATION_ID}
    assert all(row["event_type"] == "registered" for row in rows)


def test_measured_equation_is_discoverable_in_default_canonical_registry() -> None:
    equation = get_equation_by_id(EQUATION_ID)
    assert equation is not None
    assert equation.equation_id == EQUATION_ID
    assert equation.empirical_anchors[0].source_artifact.endswith(
        "sfess_cached_replay_ugc64_20260712T214520Z/measurement_receipt.json"
    )
