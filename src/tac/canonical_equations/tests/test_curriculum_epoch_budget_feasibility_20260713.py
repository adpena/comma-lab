# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import pytest

from tac.canonical_equations.curriculum_epoch_budget_feasibility_20260713 import (
    EQUATION_ID,
    build_curriculum_epoch_budget_feasibility_v1,
    curriculum_epoch_budget_feasibility,
    populate_curriculum_epoch_budget_feasibility_v1,
)


def test_enabled_margin_is_exact_and_boundary_is_feasible() -> None:
    result = curriculum_epoch_budget_feasibility(
        4, (1, 4), curriculum_enabled=True
    )
    assert result.margin_epochs == 0
    assert result.feasible is True
    assert result.status == "PASS"


def test_enabled_negative_margin_refuses() -> None:
    result = curriculum_epoch_budget_feasibility(
        4, (1, 726), curriculum_enabled=True
    )
    assert result.margin_epochs == -722
    assert result.feasible is False
    assert result.status == "REFUSE_OUT_OF_BUDGET_STAGE"


def test_disabled_curriculum_is_explicit_vacuous_pass() -> None:
    result = curriculum_epoch_budget_feasibility(
        4, (726,), curriculum_enabled=False
    )
    assert result.feasible is True
    assert result.margin_epochs is None
    assert result.status == "VACUOUS_CURRICULUM_DISABLED"


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="epochs must be"):
        curriculum_epoch_budget_feasibility(0, (), curriculum_enabled=True)
    with pytest.raises(ValueError, match="start epochs"):
        curriculum_epoch_budget_feasibility(4, (-1,), curriculum_enabled=True)


def test_canonical_entity_is_wired_and_scoped() -> None:
    equation = build_curriculum_epoch_budget_feasibility_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.domain_of_validity["verdict_scope"] == "config/boot-runnability-only"
    assert "schedule_epoch_budget_violations" in equation.canonical_consumers[0]


def test_population_uses_locked_registry_writer(tmp_path) -> None:
    registry = tmp_path / "equations.jsonl"
    equation = populate_curriculum_epoch_budget_feasibility_v1(
        path=registry,
        lock_path=tmp_path / "equations.lock",
        agent="codex",
        subagent_id="timer_curriculum_complete_test",
    )
    assert equation.equation_id == EQUATION_ID
    row = json.loads(registry.read_text())
    assert row["equation_id"] == EQUATION_ID
    assert row["subagent_id"] == "timer_curriculum_complete_test"
