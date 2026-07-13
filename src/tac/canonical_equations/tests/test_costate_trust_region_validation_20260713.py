from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.costate_trust_region_validation_20260713 import (
    ACCEPTED_EXACT_CE_DELTA,
    ACCEPTED_EXACT_DSEG_DELTA,
    BASELINE_VALIDATION_FORWARDS,
    BASELINE_VALIDATIONS_PER_TEACHER_CALL,
    EQUATION_ID,
    NEW_OPERATIONAL_VALIDATIONS_PER_ANCHOR,
    NORMALIZED_VALIDATION_REDUCTION_FACTOR,
    NORMALIZED_VALIDATION_REDUCTION_FRACTION,
    PROXY_CANDIDATES,
    PROXY_REUSES,
    build_frozen_segnet_costate_trust_region_v1,
    populate_frozen_segnet_costate_trust_region_v1,
)


def _receipt() -> dict[str, object]:
    repo = Path(__file__).resolve().parents[4]
    path = (
        repo
        / "experiments/results/costate_trust_region_economics_20260713T032000Z/measurement_receipt.json"
    )
    return json.loads(path.read_text())


def test_equation_separates_rigorous_certificate_from_empirical_proxy() -> None:
    equation = build_frozen_segnet_costate_trust_region_v1()
    empirical = equation.empirical_anchors[0].empirical_output

    assert equation.equation_id == EQUATION_ID
    assert "B_R E(r)<\\gamma_\\theta" in equation.latex_form
    assert empirical["rigorous_certificate"] == "BLOCKED_MISSING_BOUND_ARTIFACTS"
    assert empirical["empirical_margin_fisher_formulation"].startswith("NO-GO")
    assert empirical["score_claim"] is False
    assert empirical["pointer_moved"] is False
    assert any("correlation" in item for item in equation.domain_of_validity["excluded"])


def test_equation_constants_rederive_from_primary_receipt() -> None:
    receipt = _receipt()
    baseline = receipt["baseline_counts"]
    economics = receipt["economics"]
    accepted = [
        decision
        for regime in receipt["regimes"]
        for decision in regime["decisions"]
        if decision["decision"]["status"] == "PROXY_REUSE"
    ]
    exact = accepted[0]["fresh_shadow_exact_control"]
    anchor = receipt["regimes"][0]["anchor"]

    assert baseline["operational_validation_forwards"] == BASELINE_VALIDATION_FORWARDS
    assert economics["baseline_validations_per_teacher_call"] == BASELINE_VALIDATIONS_PER_TEACHER_CALL
    assert economics["new_operational_validations_per_anchor"] == NEW_OPERATIONAL_VALIDATIONS_PER_ANCHOR
    assert economics["normalized_validation_reduction_factor"] == NORMALIZED_VALIDATION_REDUCTION_FACTOR
    assert economics["normalized_validation_reduction_fraction"] == pytest.approx(
        NORMALIZED_VALIDATION_REDUCTION_FRACTION
    )
    assert sum(row["decision_counts"]["candidates"] for row in receipt["regimes"]) == PROXY_CANDIDATES
    assert len(accepted) == PROXY_REUSES
    assert exact["ce"] - anchor["exact_ce"] == ACCEPTED_EXACT_CE_DELTA
    assert exact["dseg"] - anchor["exact_dseg"] == ACCEPTED_EXACT_DSEG_DELTA


def test_populate_uses_append_only_registry(tmp_path: Path) -> None:
    path = tmp_path / "equations.jsonl"
    lock_path = tmp_path / "equations.lock"

    populate_frozen_segnet_costate_trust_region_v1(
        path=path,
        lock_path=lock_path,
        agent="pytest",
        subagent_id="task454-test",
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(rows) == 1
    assert rows[0]["equation_id"] == EQUATION_ID
    assert rows[0]["event_type"] == "registered"
