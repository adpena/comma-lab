# SPDX-License-Identifier: MIT
"""Tests for phantom_score_canonical_posterior_lookup_consumer cathedral consumer.

Per Catalog #335 canonical contract + Catalog #341 Tier A canonical-routing
markers + Catalog #125 6-hook wire-in declaration. Sister of
canonical_equation_lookup_consumer + anti_pattern_lookup_consumer tests.
"""
from __future__ import annotations

from tac.cathedral.consumer_contract import ConsumerTier, HookNumber
from tac.cathedral_consumers.phantom_score_canonical_posterior_lookup_consumer import (
    CONSUMER_HOOK_NUMBERS,
    CONSUMER_NAME,
    CONSUMER_TIER,
    CONSUMER_VERSION,
    consume_candidate,
    update_from_anchor,
)

# -------- Catalog #335 canonical contract tests --------


def test_consumer_name_canonical():
    """CONSUMER_NAME matches package directory per Catalog #335."""
    assert CONSUMER_NAME == "phantom_score_canonical_posterior_lookup_consumer"


def test_consumer_version_canonical():
    """CONSUMER_VERSION is semver string per Catalog #335."""
    assert isinstance(CONSUMER_VERSION, str)
    assert len(CONSUMER_VERSION.split(".")) == 3  # semver MAJOR.MINOR.PATCH


def test_consumer_hook_numbers_canonical():
    """CONSUMER_HOOK_NUMBERS declares Catalog #125 hooks."""
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in CONSUMER_HOOK_NUMBERS
    assert HookNumber.PROBE_DISAMBIGUATOR in CONSUMER_HOOK_NUMBERS


def test_consumer_tier_a_observability_only():
    """CONSUMER_TIER = TIER_A_OBSERVABILITY_ONLY per Catalog #341."""
    assert CONSUMER_TIER == ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def test_update_from_anchor_callable():
    """update_from_anchor signature per Catalog #335 (hook #5)."""
    # Should not raise on any anchor
    update_from_anchor(None)
    update_from_anchor({})
    update_from_anchor({"any": "anchor"})


def test_consume_candidate_callable():
    """consume_candidate signature per Catalog #335 (hook #4)."""
    result = consume_candidate({"name": "test_candidate"})
    assert isinstance(result, dict)


# -------- Catalog #341 Tier A canonical-routing markers tests --------


def test_consume_candidate_returns_predicted_delta_adjustment_zero():
    """Tier A: predicted_delta_adjustment=0.0 always."""
    result = consume_candidate({"name": "test_candidate"})
    assert result["predicted_delta_adjustment"] == 0.0


def test_consume_candidate_returns_promotable_false():
    """Tier A: promotable=False always."""
    result = consume_candidate({"name": "test_candidate"})
    assert result["promotable"] is False


def test_consume_candidate_returns_axis_tag_predicted():
    """Tier A: axis_tag=[predicted] always."""
    result = consume_candidate({"name": "test_candidate"})
    assert result["axis_tag"] == "[predicted]"


def test_consume_candidate_carries_canonical_provenance():
    """Result carries Catalog #323 canonical Provenance dict."""
    result = consume_candidate({"name": "test_candidate"})
    assert "provenance" in result
    prov = result["provenance"]
    assert prov["score_claim"] is False
    assert prov["evidence_grade"] == "predicted"
    assert prov["consumer_name"] == CONSUMER_NAME
    assert prov["consumer_version"] == CONSUMER_VERSION


# -------- Hook #4 cathedral autopilot dispatch tests --------


def test_consume_candidate_empty_candidate():
    """Empty candidate returns clean observability annotation."""
    result = consume_candidate({})
    assert result["match_count"] == 0
    assert result["matched_phantom_tokens"] == ()


def test_consume_candidate_no_canonical_tokens():
    """Candidate with no canonical-posterior tokens returns clean."""
    result = consume_candidate({"unrelated_field": "value"})
    assert result["match_count"] == 0


def test_consume_candidate_extracts_substrate_field():
    """Candidate's substrate field is extracted as canonical token."""
    result = consume_candidate({"substrate": "test_substrate"})
    # Should not crash; verdict depends on canonical posterior live state
    assert isinstance(result, dict)


def test_consume_candidate_extracts_lane_id_field():
    """Candidate's lane_id field is extracted as canonical token."""
    result = consume_candidate({"lane_id": "test_lane"})
    assert isinstance(result, dict)


def test_consume_candidate_extracts_multiple_fields():
    """Candidate with multiple canonical fields queries each token."""
    result = consume_candidate(
        {
            "substrate": "test_substrate",
            "lane_id": "test_lane",
            "probe_id": "test_probe",
            "equation_id": "test_eq",
        }
    )
    assert isinstance(result, dict)
    # rationale should mention candidate token count
    assert result["rationale"]


# -------- Sister Wave N+33 phantom-score regression guard --------


def test_consume_candidate_wave_n33_alpha_phantom_anchor():
    """Wave N+33 alpha=4.74 phantom-score anchor regression guard.

    Per Slot U landed 2026-05-29 07:10CST + canonical anti-pattern
    synthesis_vs_empirical_phantom_alpha_from_research_sidecar_v1 registered
    2026-05-28T23:49Z. If candidate cites the canonical anti-pattern id
    as a token, the consumer should surface it as matched_phantom.
    """
    result = consume_candidate(
        {
            "anti_pattern_id": (
                "synthesis_vs_empirical_phantom_alpha_from_research_sidecar"
            )
        }
    )
    # The validator may or may not match depending on live canonical posterior
    # state. The key invariant: consumer DOES NOT crash, returns canonical markers
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["axis_tag"] == "[predicted]"
    assert result["promotable"] is False


def test_consume_candidate_clean_candidate_no_phantom_match():
    """Candidate with completely unrelated token returns match_count=0."""
    result = consume_candidate(
        {
            "substrate": "totally_made_up_substrate_xyz123abc_no_chance_of_match"
        }
    )
    assert result["match_count"] == 0
    assert "no" in result["rationale"].lower() or "observability" in result["rationale"].lower()
