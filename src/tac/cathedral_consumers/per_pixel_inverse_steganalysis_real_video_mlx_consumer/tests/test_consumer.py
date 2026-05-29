# SPDX-License-Identifier: MIT
"""Tests for the per-pixel inverse-steganalysis real-video MLX cathedral consumer.

Per Catalog #335 canonical contract + Catalog #341 Tier A canonical-routing
markers + Catalog #357 dual-tier consumer architecture.
"""
from __future__ import annotations

import pytest

from tac.cathedral.consumer_contract import (
    HookNumber,
    validate_consumer_module,
)
from tac.cathedral_consumers import (
    per_pixel_inverse_steganalysis_real_video_mlx_consumer as consumer,
)


class TestCanonicalContract:
    """Catalog #335 canonical contract compliance."""

    def test_consumer_name(self):
        assert consumer.CONSUMER_NAME == "per_pixel_inverse_steganalysis_real_video_mlx_consumer"

    def test_consumer_version(self):
        assert consumer.CONSUMER_VERSION == "1.0.0"

    def test_consumer_hook_numbers_include_dispatch_and_continual_learning(self):
        assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in consumer.CONSUMER_HOOK_NUMBERS
        assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in consumer.CONSUMER_HOOK_NUMBERS
        assert HookNumber.PROBE_DISAMBIGUATOR in consumer.CONSUMER_HOOK_NUMBERS

    def test_consumer_satisfies_catalog_335_contract(self):
        """validate_consumer_module accepts the consumer per Catalog #335."""
        verdict = validate_consumer_module(consumer)
        assert verdict.contract_compliant is True
        assert verdict.validation_errors == ()

    def test_update_from_anchor_callable(self):
        # Should not raise
        consumer.update_from_anchor({"foo": "bar"})
        consumer.update_from_anchor(None)


class TestConsumeCandidate:
    """consume_candidate returns canonical Tier A contributions per Catalog #341."""

    def test_consume_candidate_returns_tier_a_markers(self):
        result = consumer.consume_candidate({"name": "test_candidate"})
        assert result["predicted_delta_adjustment"] == 0.0
        assert result["promotable"] is False
        assert result["score_claim"] is False
        assert result["axis_tag"] == "[predicted]"

    def test_consume_candidate_includes_canonical_helper_module(self):
        result = consumer.consume_candidate({"name": "test"})
        assert result["canonical_helper_module"] == "tac.inverse_steganalysis_real_video_mlx"

    def test_hill_paradigm_matched(self):
        candidate = {
            "name": "test_hill_candidate",
            "description": "HILL canonical inverse-steganalysis cost matrix",
        }
        result = consumer.consume_candidate(candidate)
        assert result["matched_paradigm"] == "hill_per_pixel_mlx"
        assert "hill_per_pixel_mlx" in result["rationale"]

    def test_li_wang_li_huang_paradigm_matched(self):
        candidate = {
            "name": "test_lw",
            "description": "Li-Wang-Li-Huang 2014 cascade",
        }
        result = consumer.consume_candidate(candidate)
        assert result["matched_paradigm"] == "hill_per_pixel_mlx"

    def test_mipod_paradigm_matched(self):
        candidate = {
            "description": "mipod_canonical_inverse_steganalysis_sedighi_cogranne_fridrich_2016",
        }
        result = consumer.consume_candidate(candidate)
        assert result["matched_paradigm"] == "mipod_per_pixel_mlx"

    def test_uniward_paradigm_matched(self):
        candidate = {"description": "uniward inverse-scorer basis"}
        result = consumer.consume_candidate(candidate)
        assert result["matched_paradigm"] == "uniward_per_pixel_mlx"

    def test_hugo_paradigm_matched(self):
        candidate = {"description": "hugo_canonical_inverse_steganalysis spam features"}
        result = consumer.consume_candidate(candidate)
        assert result["matched_paradigm"] == "hugo_per_pixel_mlx"

    def test_no_paradigm_match_returns_none(self):
        candidate = {"description": "totally unrelated candidate"}
        result = consumer.consume_candidate(candidate)
        assert result["matched_paradigm"] == "none"
        assert "No canonical inverse-steganalysis paradigm token matched" in result["rationale"]

    def test_rationale_mentions_macos_cpu_advisory_when_matched(self):
        candidate = {"description": "hill_canonical_inverse_steganalysis"}
        result = consumer.consume_candidate(candidate)
        assert "macOS-CPU advisory" in result["rationale"]
        assert "paired-CUDA" in result["rationale"]


class TestObservabilityOnlySemantics:
    """Per Catalog #341 + #192: this consumer NEVER promotes."""

    def test_no_score_claim(self):
        for desc in [
            "hill", "mipod", "uniward", "hugo",
            "totally_unrelated_candidate",
        ]:
            result = consumer.consume_candidate({"description": desc})
            assert result["score_claim"] is False, f"score_claim must be False for {desc}"
            assert result["promotable"] is False, f"promotable must be False for {desc}"
            assert result["predicted_delta_adjustment"] == 0.0
            assert result["axis_tag"] == "[predicted]"
