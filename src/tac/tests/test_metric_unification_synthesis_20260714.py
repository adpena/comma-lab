from __future__ import annotations

from tac.research.metric_unification_synthesis_20260714 import (
    METRIC_ID,
    build_metric_unification_spec,
)


def _components() -> dict[str, object]:
    return {
        "canonical_metric": {
            "metric_id": METRIC_ID,
            "state_receipt_schema": "reachable_decision_geometry_fidelity.v1",
            "selection_receipt_schema": "reachable_decision_preconditioner_selection.v1",
            "candidate_preconditioner": "winner_rival_margin_fisher_natural",
            "n600_selection_status": "MEASURED_COMPLETE",
            "fisher_margin_pearson": 0.978,
            "fisher_margin_source_sha256": "a" * 64,
        },
        "trust_region": {
            "law": "full_k_categorical_fisher_exact_kl",
            "implementation_status": "MEASURED_COMPLETE",
            "falsified_scalar_p1_law_excluded": True,
        },
        "basis": {
            "metric_id": METRIC_ID,
            "gram_metric_custody_status": "MEASURED_COMPLETE",
            "equal_budget_n600_status": "MEASURED_COMPLETE",
        },
        "anneal": {"comparison_status": "AGREE"},
        "v9_source_closure": {"source_closure_status": "GREEN_STABLE"},
    }


def test_complete_contract_can_activate() -> None:
    result = build_metric_unification_spec(_components())
    assert result["activation_allowed"]
    assert result["blockers"] == []
    assert "Psi_B^T G_dec Psi_B" in result["coordinate_definition"]["basis_pullback"]


def test_current_missing_custody_fails_closed_without_family_negative() -> None:
    components = _components()
    components["canonical_metric"]["n600_selection_status"] = "NO_VERDICT_DATA_CUSTODY"  # type: ignore[index]
    components["trust_region"]["implementation_status"] = "IN_PROGRESS"  # type: ignore[index]
    components["basis"]["equal_budget_n600_status"] = "PENDING_N600"  # type: ignore[index]
    components["anneal"]["comparison_status"] = "NO_VERDICT_SOURCE_CUSTODY"  # type: ignore[index]
    components["v9_source_closure"]["source_closure_status"] = "TOCTOU_BLOCKED"  # type: ignore[index]
    result = build_metric_unification_spec(components)
    assert not result["activation_allowed"]
    assert len(result["blockers"]) == 5
    assert "preserves every specialist family" in result["verdict_scope"]


def test_falsified_scalar_ripo_law_is_rejected() -> None:
    components = _components()
    components["trust_region"] = {
        "law": "sqrt_delta_over_p1",
        "implementation_status": "MEASURED_COMPLETE",
        "falsified_scalar_p1_law_excluded": False,
    }
    result = build_metric_unification_spec(components)
    assert not result["activation_allowed"]
    assert any("corrected full-K" in item for item in result["blockers"])
    assert any("sqrt(delta/p1)" in item for item in result["blockers"])


def test_fisher_margin_value_requires_source_custody() -> None:
    components = _components()
    del components["canonical_metric"]["fisher_margin_source_sha256"]  # type: ignore[index]
    result = build_metric_unification_spec(components)
    assert not result["activation_allowed"]
    assert any("Fisher-margin source" in item for item in result["blockers"])


def test_basis_must_name_the_canonical_metric() -> None:
    components = _components()
    del components["basis"]["metric_id"]  # type: ignore[index]
    result = build_metric_unification_spec(components)
    assert not result["activation_allowed"]
    assert any("basis metric_id" in item for item in result["blockers"])
