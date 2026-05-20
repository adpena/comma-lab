# SPDX-License-Identifier: MIT
"""Tests for cross_substrate_similarity_consumer canonical contract.

Per Catalog #335 + #341 + WAVE-3-CROSS-CANDIDATE-SENSITIVITY-COMPARISON-DIAGNOSTIC
task spec deliverable E.
"""
from __future__ import annotations

import json
from pathlib import Path

from tac.cathedral.consumer_contract import (
    HookNumber,
    validate_consumer_module,
)
from tac.cathedral_consumers import cross_substrate_similarity_consumer as csc


def test_consumer_module_satisfies_canonical_contract() -> None:
    """Catalog #335 STRICT contract satisfied."""
    registration = validate_consumer_module(csc)
    assert registration.contract_compliant, (
        f"contract violations: {registration.validation_errors}"
    )


def test_consumer_declares_hook_numbers() -> None:
    """Catalog #125 hook declarations match."""
    assert csc.CONSUMER_NAME == "cross_substrate_similarity_consumer"
    assert csc.CONSUMER_VERSION == "0.1.0"
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in csc.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in csc.CONSUMER_HOOK_NUMBERS


def test_consume_candidate_returns_canonical_routing_markers_per_catalog_341() -> None:
    """Catalog #341 canonical Tier A markers in every return."""
    result = csc.consume_candidate({"substrate": "fec6_frontier_cuda_t4"})
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"


def test_consume_candidate_handles_missing_matrix_gracefully() -> None:
    """Graceful failure when matrix not on disk (Catalog #138 sister)."""
    # This will read whatever is currently in .omx/state/; if no matrix
    # exists the consumer should NOT crash.
    result = csc.consume_candidate({"substrate": "fec6_frontier_cuda_t4"})
    # Result is always a Mapping with canonical markers
    assert isinstance(result, dict)
    assert "rationale" in result
    assert "matrix_available" in result


def test_consume_candidate_extracts_substrate_label_from_candidate() -> None:
    """The candidate-text walker matches known substrate tokens."""
    # The matrix may or may not exist; either way the substrate label
    # extraction should be deterministic
    result = csc.consume_candidate({"lane_id": "lane_pr101_frame_exploit_fec6"})
    assert isinstance(result, dict)
    # Should either match (pr101 -> pr101_gold OR fec6 -> fec6_frontier_cuda_t4)
    # The candidate text contains both 'pr101' AND 'fec6' so we expect a match
    if result.get("matrix_available"):
        # Matrix is on disk; extract should succeed
        assert result.get("matched_substrate") is not None


def test_update_from_anchor_is_noop() -> None:
    """Per Catalog #287/#323: manual refresh required for new anchors."""
    # Should not raise
    csc.update_from_anchor({"archive_sha256": "test"})


def test_state_dir_resolution() -> None:
    """Helper resolves to the canonical .omx/state directory."""
    state = csc._state_dir()
    # State dir resolution is best-effort; if found, must be a directory
    if state.is_dir():
        assert state.name == "state"
        assert state.parent.name == ".omx"


def test_load_latest_matrix_when_present_returns_dict() -> None:
    """When matrix exists, loader returns parsed JSON dict."""
    state = csc._state_dir()
    matrices = list(state.glob("cross_substrate_sensitivity_similarity_matrix_*.json"))
    if matrices:
        matrix = csc._load_latest_matrix()
        assert matrix is not None
        assert "pairs" in matrix
        assert "schema_version" in matrix


def test_classifications_for_substrate_handles_empty_pairs() -> None:
    """No pairs -> empty result list."""
    result = csc._classifications_for_substrate({"pairs": []}, "pr101_gold")
    assert result == []


def test_classifications_for_substrate_handles_non_list_pairs() -> None:
    """Defensive: non-list pairs -> empty result."""
    result = csc._classifications_for_substrate({"pairs": "not a list"}, "pr101_gold")
    assert result == []


def test_candidate_substrate_label_returns_none_for_no_match() -> None:
    """No known substrate token -> None."""
    label = csc._candidate_substrate_label({"unrelated_field": "no_substrate_here"})
    assert label is None


def test_candidate_substrate_label_finds_canonical_match() -> None:
    """Known substrate token surfaces canonical full label."""
    label = csc._candidate_substrate_label({"substrate": "pr101"})
    assert label == "pr101_gold"

    label = csc._candidate_substrate_label({"substrate": "fec6"})
    assert label == "fec6_frontier_cuda_t4"
