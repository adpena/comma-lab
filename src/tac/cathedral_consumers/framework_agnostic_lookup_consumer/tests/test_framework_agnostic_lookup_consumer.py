# SPDX-License-Identifier: MIT
"""Tests for framework_agnostic_lookup_consumer (Catalog #335 sister)."""
from __future__ import annotations

import pytest

from tac.cathedral.consumer_contract import (
    CathedralConsumerContract,
    ConsumerTier,
    HookNumber,
    validate_consumer_module,
)


@pytest.fixture
def consumer():
    """Import the consumer module fresh."""
    from tac.cathedral_consumers import framework_agnostic_lookup_consumer
    return framework_agnostic_lookup_consumer


# -----------------------------------------------------------------------------
# Canonical contract compliance (Catalog #335)
# -----------------------------------------------------------------------------


def test_consumer_satisfies_canonical_contract(consumer):
    """Module satisfies CathedralConsumerContract Protocol."""
    assert isinstance(consumer, CathedralConsumerContract)


def test_consumer_passes_validate_consumer_module(consumer):
    """Module passes the canonical contract validator with no errors."""
    registration = validate_consumer_module(consumer)
    assert registration.contract_compliant is True
    assert registration.validation_errors == ()


def test_consumer_name_canonical(consumer):
    assert consumer.CONSUMER_NAME == "framework_agnostic_lookup_consumer"


def test_consumer_version_pinned(consumer):
    assert consumer.CONSUMER_VERSION == "1.0.0"


def test_consumer_hook_numbers_include_dispatch_and_disambiguator(consumer):
    """Per canonical 6-hook declaration: hooks #4 + #6 ACTIVE."""
    hooks = consumer.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in hooks
    assert HookNumber.PROBE_DISAMBIGUATOR in hooks


def test_consumer_tier_is_tier_a_observability_only(consumer):
    """Per Catalog #357: this is a Tier A observability-only consumer."""
    assert consumer.CONSUMER_TIER is ConsumerTier.TIER_A_OBSERVABILITY_ONLY


# -----------------------------------------------------------------------------
# update_from_anchor hook
# -----------------------------------------------------------------------------


def test_update_from_anchor_noop_does_not_raise(consumer):
    """Reference no-op MUST NOT raise on any anchor shape."""
    consumer.update_from_anchor({"any": "anchor"})
    consumer.update_from_anchor(None)
    consumer.update_from_anchor(object())


# -----------------------------------------------------------------------------
# consume_candidate hook + canonical routing markers (Catalog #341)
# -----------------------------------------------------------------------------


def test_consume_candidate_returns_tier_a_canonical_markers(consumer):
    """Catalog #341 Tier A canonical-routing markers always present."""
    result = consumer.consume_candidate({})
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"


def test_consume_candidate_returns_rationale(consumer):
    result = consumer.consume_candidate({})
    assert isinstance(result["rationale"], str)
    assert len(result["rationale"]) >= 4


def test_consume_candidate_inference_explicit_framework_mlx(consumer):
    """Explicit framework_backend=mlx → non_promotable_research_signal."""
    result = consumer.consume_candidate({"framework_backend": "mlx"})
    assert result["framework_backend"] == "mlx"
    assert result["routing_class"] == "non_promotable_research_signal"


def test_consume_candidate_inference_explicit_framework_pytorch(consumer):
    """Explicit framework_backend=pytorch → promotable_contest_resolution."""
    result = consumer.consume_candidate({"framework_backend": "pytorch"})
    assert result["framework_backend"] == "pytorch"
    assert result["routing_class"] == "promotable_contest_resolution"


def test_consume_candidate_inference_explicit_backend_numpy(consumer):
    """Explicit backend=numpy → diagnostic_reference."""
    result = consumer.consume_candidate({"backend": "numpy"})
    assert result["framework_backend"] == "numpy"
    assert result["routing_class"] == "diagnostic_reference"


def test_consume_candidate_inference_trainer_path_mlx_local(consumer):
    """trainer_path containing _mlx_local → mlx + non_promotable_research_signal."""
    result = consumer.consume_candidate(
        {"trainer_path": "experiments/train_substrate_pact_nerv_selector_v3_mlx_local.py"}
    )
    assert result["framework_backend"] == "mlx"
    assert result["routing_class"] == "non_promotable_research_signal"


def test_consume_candidate_inference_recipe_modal_t4(consumer):
    """recipe containing _modal_t4 → pytorch + promotable_contest_resolution."""
    result = consumer.consume_candidate(
        {"recipe": ".omx/operator_authorize_recipes/substrate_x_modal_t4_dispatch.yaml"}
    )
    assert result["framework_backend"] == "pytorch"
    assert result["routing_class"] == "promotable_contest_resolution"


def test_consume_candidate_inference_unknown_when_no_signal(consumer):
    """Empty candidate yields routing_class=unknown."""
    result = consumer.consume_candidate({"unrelated_key": "value"})
    assert result["framework_backend"] is None
    assert result["routing_class"] == "unknown"


def test_consume_candidate_v3_sister_trainer_pair_anchor(consumer):
    """V3 sister trainer pair empirical anchor: MLX-LOCAL vs Modal_T4 disambiguation."""
    mlx_result = consumer.consume_candidate(
        {"trainer_path": "experiments/train_substrate_pact_nerv_selector_v3_mlx_local.py"}
    )
    torch_result = consumer.consume_candidate(
        {"trainer_path": "experiments/train_substrate_pact_nerv_selector_v3.py",
         "recipe": "substrate_pact_nerv_selector_v3_modal_t4_dispatch.yaml"}
    )
    assert mlx_result["routing_class"] == "non_promotable_research_signal"
    assert torch_result["routing_class"] == "promotable_contest_resolution"


# -----------------------------------------------------------------------------
# Catalog #341 routing markers full structural check
# -----------------------------------------------------------------------------


def test_all_routing_branches_carry_canonical_markers(consumer):
    """Every routing branch (mlx / pytorch / numpy / unknown) → canonical Tier A markers."""
    test_candidates = [
        {"framework_backend": "mlx"},
        {"framework_backend": "pytorch"},
        {"framework_backend": "numpy"},
        {"unrelated": "value"},  # unknown
        {"trainer_path": "experiments/train_substrate_x_mlx_local.py"},
        {"recipe": "substrate_y_modal_a100.yaml"},
    ]
    for candidate in test_candidates:
        result = consumer.consume_candidate(candidate)
        assert result["predicted_delta_adjustment"] == 0.0, candidate
        assert result["promotable"] is False, candidate
        assert result["axis_tag"] == "[predicted]", candidate
        assert "routing_class" in result, candidate
