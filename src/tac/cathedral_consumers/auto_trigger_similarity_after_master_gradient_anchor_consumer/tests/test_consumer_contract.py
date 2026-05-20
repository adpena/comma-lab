# SPDX-License-Identifier: MIT
"""Tests for auto_trigger_similarity_after_master_gradient_anchor_consumer.

Per Catalog #335 contract + Catalog #341 canonical routing markers
+ Catalog #344 canonical-equation consumption discipline.
"""
from __future__ import annotations

import pytest

from tac.cathedral.consumer_contract import (
    HookNumber,
    validate_consumer_module,
)
from tac.cathedral_consumers import (
    auto_trigger_similarity_after_master_gradient_anchor_consumer as consumer,
)


def test_canonical_contract_satisfied() -> None:
    """Catalog #335 STRICT contract - all 5 canonical fields + 2 callables."""
    validate_consumer_module(consumer)


def test_canonical_name_and_version_pinned() -> None:
    assert consumer.CONSUMER_NAME == (
        "auto_trigger_similarity_after_master_gradient_anchor_consumer"
    )
    assert isinstance(consumer.CONSUMER_VERSION, str)
    assert len(consumer.CONSUMER_VERSION.split(".")) == 3


def test_canonical_hook_numbers_includes_continual_learning_and_dispatch() -> None:
    """Per docstring hook declaration: #4 + #5 + #6 ACTIVE."""
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in consumer.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in consumer.CONSUMER_HOOK_NUMBERS
    assert HookNumber.PROBE_DISAMBIGUATOR in consumer.CONSUMER_HOOK_NUMBERS


def test_update_from_anchor_does_not_raise_on_synthetic_anchor() -> None:
    """Stub implementation must not propagate exceptions on any input."""
    consumer.update_from_anchor({"archive_sha256": "fake", "substrate_id": "synthetic"})
    consumer.update_from_anchor(None)
    consumer.update_from_anchor("not even a dict")


def test_consume_candidate_returns_canonical_tier_a_markers() -> None:
    """Per Catalog #341: predicted_delta_adjustment=0.0 + promotable=False + axis_tag=[predicted]."""
    result = consumer.consume_candidate({"substrate_a": "pr101", "substrate_b": "fec6"})
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"
    assert "canonical_equations_consumed" in result


def test_consume_candidate_pre_classifies_pr106_vs_hnerv_as_super_additive() -> None:
    """Per Equation 9 cross_codec_super_additive_orthogonality_predictor_v1."""
    result = consumer.consume_candidate(
        {"substrate_a": "pr106_format0d", "substrate_b": "pr101_gold"}
    )
    assert result["predicted_classification"] == "SUPER_ADDITIVE"


def test_consume_candidate_pre_classifies_pr106_vs_fec6_as_super_additive() -> None:
    result = consumer.consume_candidate(
        {"substrate_a": "fec6_frontier_cuda_t4", "substrate_b": "pr106_format0d"}
    )
    assert result["predicted_classification"] == "SUPER_ADDITIVE"


def test_consume_candidate_pre_classifies_within_hnerv_as_sub_additive() -> None:
    """Per Equation 8 hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1."""
    result = consumer.consume_candidate(
        {"substrate_a": "pr101_gold", "substrate_b": "a1_finetuned"}
    )
    assert result["predicted_classification"] == "SUB_ADDITIVE"


def test_consume_candidate_returns_none_classification_for_unknown_pair() -> None:
    result = consumer.consume_candidate(
        {"substrate_a": "unknown_substrate_x", "substrate_b": "unknown_substrate_y"}
    )
    assert result["predicted_classification"] is None


def test_consume_candidate_handles_empty_input() -> None:
    """Defensive contract: empty dict accepted without raising."""
    result = consumer.consume_candidate({})
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["predicted_classification"] is None


def test_canonical_equations_consumed_includes_all_three_strategic_findings() -> None:
    """Per STRATEGIC-FINDINGS commit 80484241f."""
    result = consumer.consume_candidate({"substrate_a": "pr101", "substrate_b": "fec6"})
    consumed = set(result["canonical_equations_consumed"])
    assert "per_byte_leverage_cross_hardware_aware_v2" in consumed
    assert "hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1" in consumed
    assert "cross_codec_super_additive_orthogonality_predictor_v1" in consumed
