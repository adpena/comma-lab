from __future__ import annotations

import json

import pytest

from tac.canonical_equations.ugc_terminal_polish_variance_cost_20260712 import (
    EQUATION_ID,
    UGC_TO_DISARM_VARIANCE_RATIO,
    UGC_TO_ES_PROGRESS_RATIO,
    build_ugc_terminal_polish_variance_cost_progress_separation_v1,
    populate_ugc_terminal_polish_variance_cost_equation,
)


def test_measured_law_keeps_variance_and_progress_verdicts_separate() -> None:
    equation = build_ugc_terminal_polish_variance_cost_progress_separation_v1()

    assert equation.equation_id == EQUATION_ID
    assert pytest.approx(0.6822352788355922) == UGC_TO_DISARM_VARIANCE_RATIO
    assert pytest.approx(0.9142958312837861) == UGC_TO_ES_PROGRESS_RATIO
    assert UGC_TO_DISARM_VARIANCE_RATIO < 1.0
    assert UGC_TO_ES_PROGRESS_RATIO < 1.0
    assert equation.empirical_anchors[0].empirical_output["verdict"] == (
        "UGC_LOSES_INSTANCE_FORMULATION_SCOPED"
    )
    assert equation.domain_of_validity["scope_level"] == "instance/formulation"


def test_populate_uses_append_only_registry(tmp_path) -> None:
    path = tmp_path / "equations.jsonl"
    lock_path = tmp_path / "equations.lock"

    populate_ugc_terminal_polish_variance_cost_equation(
        path=path,
        lock_path=lock_path,
        agent="pytest",
        subagent_id="ugc-test",
    )
    populate_ugc_terminal_polish_variance_cost_equation(
        path=path,
        lock_path=lock_path,
        agent="pytest",
        subagent_id="ugc-test",
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert {row["equation_id"] for row in rows} == {EQUATION_ID}
    assert all(row["event_type"] == "registered" for row in rows)
