# SPDX-License-Identifier: MIT
"""Tests for cathedral consumer
:mod:`tac.cathedral_consumers.mlx_canonicalization_audit_consumer`.

Per Catalog #335 + Catalog #341 + Catalog #383 canonical contracts.
"""
from __future__ import annotations


from tac.cathedral.consumer_contract import HookNumber
from tac.cathedral_consumers.mlx_canonicalization_audit_consumer import (
    CONSUMER_HOOK_NUMBERS,
    CONSUMER_NAME,
    CONSUMER_VERSION,
    consume_candidate,
    update_from_anchor,
)


class TestCanonicalContract:
    """Catalog #335 canonical contract presence."""

    def test_consumer_name_is_string(self):
        assert isinstance(CONSUMER_NAME, str)
        assert len(CONSUMER_NAME) > 0

    def test_consumer_version_is_string(self):
        assert isinstance(CONSUMER_VERSION, str)

    def test_consumer_hook_numbers_is_tuple_of_enum(self):
        assert isinstance(CONSUMER_HOOK_NUMBERS, tuple)
        for hook in CONSUMER_HOOK_NUMBERS:
            assert isinstance(hook, HookNumber)

    def test_consume_candidate_returns_mapping(self):
        result = consume_candidate({"id": "test_candidate"})
        assert isinstance(result, dict)

    def test_update_from_anchor_callable(self):
        # NO-OP; just ensure callable
        update_from_anchor({"anchor_id": "test"})


class TestCatalog341TierAMarkers:
    """Catalog #341 canonical Tier A non-promotable markers."""

    def test_predicted_delta_adjustment_zero(self):
        result = consume_candidate({})
        assert result["predicted_delta_adjustment"] == 0.0

    def test_promotable_false(self):
        result = consume_candidate({})
        assert result["promotable"] is False

    def test_axis_tag_predicted(self):
        result = consume_candidate({})
        assert result["axis_tag"] == "[predicted]"

    def test_rationale_substantive(self):
        result = consume_candidate({})
        rationale = result.get("rationale", "")
        # Per Catalog #287 sister discipline
        assert len(rationale) > 10
        assert "<rationale>" not in rationale.lower()
        assert "<reason>" not in rationale.lower()


class TestHook4PrimaryActive:
    """Hook #4 cathedral autopilot dispatch is ACTIVE PRIMARY."""

    def test_dispatch_hook_present(self):
        assert (
            HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in CONSUMER_HOOK_NUMBERS
        )


class TestHook5ActiveViaCanonicalEquations:
    """Hook #5 continual-learning posterior is ACTIVE via canonical
    equations consumed list."""

    def test_posterior_hook_present(self):
        assert (
            HookNumber.CONTINUAL_LEARNING_POSTERIOR in CONSUMER_HOOK_NUMBERS
        )

    def test_consumed_canonical_equations_present(self):
        result = consume_candidate({})
        assert "consumed_canonical_equations" in result
        consumed = result["consumed_canonical_equations"]
        assert len(consumed) >= 1


class TestAutoDiscovery:
    """Catalog #335 auto-discovery surface."""

    def test_consumer_discoverable_via_auto_discovery_loop(self):
        from pathlib import Path
        package_dir = Path(__file__).resolve().parents[1]
        assert package_dir.name == "mlx_canonicalization_audit_consumer"
        # Required __init__.py
        assert (package_dir / "__init__.py").exists()
