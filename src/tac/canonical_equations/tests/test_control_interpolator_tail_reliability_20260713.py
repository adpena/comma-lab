from __future__ import annotations

import numpy as np
import pytest

from tac.canonical_equations.control_interpolator_tail_reliability_20260713 import (
    TailReliabilityError,
    build_control_interpolator_tail_cvar_mean_gate_v1,
    build_fixed_design_correlated_gaussian_ridge_tail_v1,
    correlated_gaussian_quadratic_upper_tail,
    empirical_cvar,
    retained_mass_tail_summary,
    select_tail_lambda,
)
from tac.canonical_equations.registry import load_equation_registry_strict
from tac.witness_dsl.control_tail_reliability_policy_20260713 import (
    ControlTailReliabilityPolicy,
)


def test_empirical_cvar_keeps_fractional_boundary_mass() -> None:
    losses = [0.0, 1.0, 2.0, 3.0]
    assert empirical_cvar(losses, alpha=0.50) == pytest.approx(2.5)
    assert empirical_cvar(losses, alpha=0.75) == pytest.approx(3.0)
    assert empirical_cvar(losses, alpha=0.90) == pytest.approx(3.0)


def test_retained_mass_summary_reports_harmful_upper_tail() -> None:
    summary = retained_mass_tail_summary([0.1, 0.2, 0.3, 0.4], alpha=0.75)
    assert summary["retained_mass_worst"] == pytest.approx(0.1)
    assert summary["shortfall_worst"] == pytest.approx(0.9)
    assert summary["shortfall_cvar"] == pytest.approx(0.9)


def test_tail_selection_enforces_positive_lambda_and_mean_gate() -> None:
    rows = [
        {"lambda": 0.0, "losses": [0.1, 0.9]},
        {"lambda": 0.1, "losses": [0.4, 0.4]},
        {"lambda": 1.0, "losses": [0.1, 0.5]},
        {"lambda": 10.0, "losses": [0.2, 0.2]},
    ]
    selected = select_tail_lambda(
        rows,
        mean_reference=0.3,
        alpha=0.5,
        require_positive=True,
    )
    assert selected.lambda_value == pytest.approx(10.0)
    assert selected.mean_loss == pytest.approx(0.2)
    with pytest.raises(TailReliabilityError, match="no lambda"):
        select_tail_lambda(rows, mean_reference=0.05, alpha=0.5)


def test_fixed_design_gaussian_bound_has_auditable_scalar_constants() -> None:
    bound = correlated_gaussian_quadratic_upper_tail([[2.0]], [3.0], delta=0.05)
    t = np.log(20.0)
    expected = 2.0 * np.sqrt((4.0 + 18.0) * t) + 4.0 * t
    assert bound["upper_tail_excess"] == pytest.approx(expected)
    assert bound["operator_A"] == pytest.approx(2.0)


def test_policy_is_default_off_and_records_cache_limitation() -> None:
    contract = ControlTailReliabilityPolicy().compile_measurement_contract()
    assert contract["lambda_grid"][0] == 0.0
    assert contract["fit_resample_seeds"] == (455, 456, 457)
    assert contract["scorer_calls_enabled"] is False
    assert contract["live_run_mutation_enabled"] is False
    assert "cannot be reconstructed" in contract["pre_se_measurement_scope"][
        "official_heldout_limitation"
    ]


def test_canonical_equation_builders_validate_and_keep_authority_narrow() -> None:
    selection = build_control_interpolator_tail_cvar_mean_gate_v1()
    bound = build_fixed_design_correlated_gaussian_ridge_tail_v1()
    assert selection.equation_id.endswith("_v1")
    assert selection.empirical_anchors[0].empirical_output["pointer_moved"] is False
    assert bound.empirical_anchors == ()
    assert "numeric close=false" in bound.domain_of_validity["verdict_scope"]


def test_population_uses_locked_registry_contract_without_shared_file(tmp_path) -> None:
    from tac.canonical_equations.control_interpolator_tail_reliability_20260713 import (
        populate_control_tail_reliability_equations,
    )

    registry = tmp_path / "equations.jsonl"
    lock = tmp_path / "equations.lock"
    populate_control_tail_reliability_equations(
        path=registry,
        lock_path=lock,
        agent="test",
        subagent_id="quant_tail_reliability_test",
    )
    rows = load_equation_registry_strict(registry)
    assert [row["equation_id"] for row in rows] == [
        "control_interpolator_tail_cvar_mean_gate_v1",
        "fixed_design_correlated_gaussian_ridge_tail_v1",
    ]
