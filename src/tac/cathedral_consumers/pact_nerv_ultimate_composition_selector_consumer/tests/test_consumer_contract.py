# SPDX-License-Identifier: MIT
"""Tests for pact_nerv_ultimate_composition_selector_consumer canonical contract.

Per Catalog #335 + #341 + WAVE-3-PACT-NERV-ULTIMATE-RESEARCH-AND-DESIGN
task spec deliverable D.
"""
from __future__ import annotations

from tac.cathedral.consumer_contract import HookNumber, validate_consumer_module
from tac.cathedral_consumers import (
    pact_nerv_ultimate_composition_selector_consumer as pnusc,
)


def test_consumer_module_satisfies_canonical_contract() -> None:
    """Catalog #335 STRICT contract satisfied."""
    registration = validate_consumer_module(pnusc)
    assert registration.contract_compliant, (
        f"contract violations: {registration.validation_errors}"
    )


def test_consumer_declares_hook_numbers() -> None:
    """Catalog #125 hook declarations match."""
    assert pnusc.CONSUMER_NAME == "pact_nerv_ultimate_composition_selector_consumer"
    assert pnusc.CONSUMER_VERSION == "0.1.0"
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in pnusc.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in pnusc.CONSUMER_HOOK_NUMBERS


def test_consume_candidate_returns_canonical_routing_markers_per_catalog_341() -> None:
    """Catalog #341 canonical Tier A markers in every return."""
    result = pnusc.consume_candidate({"substrate": "pact_nerv_selector_v2"})
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"


def test_consume_candidate_returns_canonical_markers_for_unknown_variant() -> None:
    """Unknown variant still returns canonical Tier A markers (no crash)."""
    result = pnusc.consume_candidate({"substrate": "some_unrelated_substrate"})
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"
    assert result["matched_variant"] is None


def test_consume_candidate_handles_missing_matrix_gracefully() -> None:
    """Graceful when matrix not on disk (Catalog #138 sister)."""
    result = pnusc.consume_candidate({"substrate": "pact_nerv_ia3"})
    assert isinstance(result, dict)
    assert "rationale" in result
    assert "matrix_available" in result


def test_taxonomy_has_at_least_15_variants() -> None:
    """Per task spec: 10+ NEW variants + 5 from FILM-FAMILY-RESEARCH = 15+ total."""
    assert len(pnusc.PACT_NERV_VARIANT_TAXONOMY) >= 15


def test_ultimate_dimensions_has_8() -> None:
    """Per task spec: 7 dimensions + 1 CROSS-CODEC from CROSS-CANDIDATE finding #3."""
    assert len(pnusc.ULTIMATE_DIMENSIONS) == 8
    assert "FRONTIER" in pnusc.ULTIMATE_DIMENSIONS
    assert "CROSS-CODEC" in pnusc.ULTIMATE_DIMENSIONS


def test_priority_1_variants_include_selector_extensions() -> None:
    """PRIORITY 1 must include SELECTOR-EXTENSIONS per CROSS-CANDIDATE finding #1."""
    priority_1_names = [
        row["name"] for row in pnusc.PACT_NERV_VARIANT_TAXONOMY if row["priority"] == 1
    ]
    assert "pact_nerv_selector_v2" in priority_1_names
    assert "pact_nerv_selector_v3" in priority_1_names
    assert "pact_nerv_selector_v4" in priority_1_names
    assert "pact_nerv_cross_codec_a" in priority_1_names


def test_cross_codec_variants_present_per_finding_3() -> None:
    """Variants 16-18 CROSS-CODEC composition per CROSS-CANDIDATE finding #3."""
    variant_names = [row["name"] for row in pnusc.PACT_NERV_VARIANT_TAXONOMY]
    assert "pact_nerv_cross_codec_a" in variant_names
    assert "pact_nerv_cross_codec_b" in variant_names
    assert "pact_nerv_neural_codec_e2e_cross" in variant_names


def test_consume_candidate_for_priority_1_selector_v2_returns_canonical() -> None:
    """SELECTOR-V2 must surface FRONTIER + EFFICIENCY ULTIMATE eligibility."""
    result = pnusc.consume_candidate({"substrate": "pact_nerv_selector_v2"})
    assert result["matched_variant"] == "pact_nerv_selector_v2"
    assert "FRONTIER" in result["ultimate_eligibility"]
    assert "EFFICIENCY" in result["ultimate_eligibility"]
    assert result["priority"] == 1


def test_consume_candidate_for_cross_codec_a_returns_canonical() -> None:
    """CROSS-CODEC-A must surface CROSS-CODEC + FRONTIER ULTIMATE eligibility."""
    result = pnusc.consume_candidate({"substrate": "pact_nerv_cross_codec_a"})
    assert result["matched_variant"] == "pact_nerv_cross_codec_a"
    assert "CROSS-CODEC" in result["ultimate_eligibility"]
    assert "FRONTIER" in result["ultimate_eligibility"]
    assert result["priority"] == 1


def test_alias_pact_nerv_defaults_to_a1() -> None:
    """Bare 'pact_nerv' candidate defaults to canonical baseline A1 variant."""
    result = pnusc.consume_candidate({"substrate": "pact_nerv"})
    assert result["matched_variant"] == "pact_nerv_a1"
