# SPDX-License-Identifier: MIT
"""Tests for tac.cathedral_consumers.dykstra_pareto_solver_consumer (Catalog #335/#341)."""
from __future__ import annotations

import pytest

from tac.cathedral.consumer_contract import (
    ConsumerTier,
    DEFAULT_CONSUMER_TIER,
    HookNumber,
    validate_consumer_module,
)


def test_consumer_module_imports():
    import tac.cathedral_consumers.dykstra_pareto_solver_consumer as mod
    assert mod is not None


def test_consumer_canonical_contract_compliant():
    import tac.cathedral_consumers.dykstra_pareto_solver_consumer as mod
    result = validate_consumer_module(mod)
    assert result.contract_compliant is True
    assert result.consumer_name == "dykstra_pareto_solver_consumer"
    assert result.consumer_version == "1.0.0"


def test_consumer_declared_tier_is_default_tier_a():
    import tac.cathedral_consumers.dykstra_pareto_solver_consumer as mod
    result = validate_consumer_module(mod)
    # No explicit CONSUMER_TIER => defaults to TIER_A_OBSERVABILITY_ONLY.
    assert result.consumer_tier == DEFAULT_CONSUMER_TIER
    assert result.consumer_tier == ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def test_consumer_hook_numbers_include_pareto_constraint_primary():
    import tac.cathedral_consumers.dykstra_pareto_solver_consumer as mod
    result = validate_consumer_module(mod)
    assert HookNumber.PARETO_CONSTRAINT in result.consumer_hook_numbers


def test_consumer_hook_numbers_include_sensitivity_map():
    import tac.cathedral_consumers.dykstra_pareto_solver_consumer as mod
    result = validate_consumer_module(mod)
    assert HookNumber.SENSITIVITY_MAP in result.consumer_hook_numbers


def test_consumer_hook_numbers_include_continual_learning():
    import tac.cathedral_consumers.dykstra_pareto_solver_consumer as mod
    result = validate_consumer_module(mod)
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in result.consumer_hook_numbers


def test_consume_candidate_returns_tier_a_routing_markers():
    """Per Catalog #341: every Tier A contribution returns
    predicted_delta_adjustment=0.0 + axis_tag='[predicted]' + promotable=False."""
    from tac.cathedral_consumers.dykstra_pareto_solver_consumer import consume_candidate
    out = consume_candidate({"candidate_id": "test_candidate"})
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["axis_tag"] == "[predicted]"
    assert out["promotable"] is False


def test_consume_candidate_emits_canonical_helper_module():
    """Sister Catalog #344: canonical equation lookup consumer pattern —
    emit canonical_helper_module + canonical_equation_id so downstream
    consumers can audit which canonical predictor governs the prediction."""
    from tac.cathedral_consumers.dykstra_pareto_solver_consumer import consume_candidate
    out = consume_candidate({"candidate_id": "test_helper_module"})
    assert out["canonical_helper_module"] == "tac.dykstra_pareto_solver"
    assert (
        out["canonical_equation_id"]
        == "dykstra_pareto_polytope_intersection_compounding_v1"
    )


def test_consume_candidate_includes_rationale():
    from tac.cathedral_consumers.dykstra_pareto_solver_consumer import consume_candidate
    out = consume_candidate({"candidate_id": "rationale_test"})
    assert "rationale" in out
    assert "dykstra_pareto_solver_consumer" in out["rationale"]
    assert "Tier A" in out["rationale"]


def test_consume_candidate_handles_missing_candidate_id():
    from tac.cathedral_consumers.dykstra_pareto_solver_consumer import consume_candidate
    out = consume_candidate({})
    assert out["predicted_delta_adjustment"] == 0.0
    # Falls back to '?' gracefully.
    assert "'?'" in out["rationale"] or "?" in out["rationale"]


def test_update_from_anchor_is_noop_at_tier_a():
    """Tier A consumer does not mutate canonical state on its own per
    CLAUDE.md 'Apples-to-apples evidence discipline' + Catalog #341."""
    from tac.cathedral_consumers.dykstra_pareto_solver_consumer import update_from_anchor
    # Should accept any anchor without raising.
    update_from_anchor({"any": "anchor", "shape": "irrelevant_at_tier_a"})
    update_from_anchor(None)
    update_from_anchor(object())


def test_consumer_auto_discoverable_by_catalog_335_gate():
    """Verify Catalog #335 STRICT preflight gate sees this consumer."""
    from tac.preflight import (
        check_cathedral_consumer_directory_package_exposes_canonical_contract,
    )
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        strict=False
    )
    # The dykstra_pareto_solver_consumer must NOT contribute any violation.
    for v in violations:
        assert "dykstra_pareto_solver_consumer" not in v, (
            f"Catalog #335 reports violation involving our consumer: {v}"
        )
