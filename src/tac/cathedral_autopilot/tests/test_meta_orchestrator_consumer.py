# SPDX-License-Identifier: MIT
"""Tests for tac.cathedral_consumers.meta_orchestrator_consumer (Catalog #335)."""
from __future__ import annotations

import importlib


def _load_consumer():
    return importlib.import_module("tac.cathedral_consumers.meta_orchestrator_consumer")


def test_consumer_satisfies_canonical_contract() -> None:
    from tac.cathedral.consumer_contract import validate_consumer_module

    m = _load_consumer()
    reg = validate_consumer_module(m)
    assert reg.contract_compliant is True
    assert list(reg.validation_errors) == []


def test_consumer_canonical_name_pinned() -> None:
    m = _load_consumer()
    assert m.CONSUMER_NAME == "meta_orchestrator_consumer"
    assert m.CONSUMER_VERSION


def test_consumer_hook_numbers_active_4_5_6() -> None:
    """Hooks 4 (cathedral autopilot dispatch) + 5 (continual-learning) + 6 (probe-disambiguator)."""
    from tac.cathedral.consumer_contract import HookNumber

    m = _load_consumer()
    hooks = m.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in hooks
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in hooks
    assert HookNumber.PROBE_DISAMBIGUATOR in hooks


def test_consumer_tier_a_observability_only() -> None:
    """Per Catalog #341: observability-only at landing."""
    from tac.cathedral.consumer_contract import ConsumerTier

    m = _load_consumer()
    assert m.CONSUMER_TIER == ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def test_consume_candidate_returns_canonical_routing_markers() -> None:
    """Per Catalog #341: predicted_delta_adjustment=0.0 + promotable=False + axis_tag='[predicted]'."""
    m = _load_consumer()
    result = m.consume_candidate({
        "candidate_id": "test_candidate",
        "predicted_score_delta": -0.5,
        "probability_materializes": 0.5,
        "wall_clock_to_validation_hours": 1.0,
        "hygiene_lessons_honored": 7,
    })
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["axis_tag"] == "[predicted]"
    assert result["promotable"] is False
    assert result["confidence"] == 0.0


def test_consume_candidate_surfaces_3_metric_trichotomy() -> None:
    m = _load_consumer()
    result = m.consume_candidate({
        "candidate_id": "test_candidate",
        "predicted_score_delta": -0.5,
        "probability_materializes": 0.5,
        "wall_clock_to_validation_hours": 1.0,
        "hygiene_lessons_honored": 7,
    })
    assert "three_metric_trichotomy" in result
    trichotomy = result["three_metric_trichotomy"]
    assert "hygiene_ev" in trichotomy
    assert "frontier_breaking_ev" in trichotomy
    assert "highest_ev_shortest_wall_clock_ev" in trichotomy
    assert trichotomy["operator_canonical_metric"] == "highest_ev_shortest_wall_clock"


def test_consume_candidate_handles_empty_candidate_gracefully() -> None:
    """Defensive: empty candidate should produce zero EVs but not crash."""
    m = _load_consumer()
    result = m.consume_candidate({})
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["axis_tag"] == "[predicted]"


def test_update_from_anchor_is_noop() -> None:
    """Per docstring: refresh is operator-triggered, not auto-fired."""
    m = _load_consumer()
    # Should not raise; observability-only hook
    m.update_from_anchor(None)
    m.update_from_anchor({"some": "anchor"})


def test_consumer_auto_discovered_in_cathedral_consumers_dir() -> None:
    """Per Catalog #335 paradigm: convention-over-configuration auto-discovery."""
    from tac.cathedral.consumer_contract import validate_consumer_module
    import importlib

    m = importlib.import_module("tac.cathedral_consumers.meta_orchestrator_consumer")
    reg = validate_consumer_module(m)
    assert reg.contract_compliant is True
    assert reg.consumer_name == "meta_orchestrator_consumer"
