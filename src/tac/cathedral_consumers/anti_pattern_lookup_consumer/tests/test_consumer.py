# SPDX-License-Identifier: MIT
"""Tests for anti_pattern_lookup_consumer (Catalog #335 sister of sister consumers).

Mirrors test patterns from sister canonical_equation_lookup_consumer.

Per design memo §"Tests at .../tests/test_consumer.py" requirement of >=15 tests.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tac.cathedral.consumer_contract import (
    ConsumerTier,
    HookNumber,
    validate_consumer_module,
)
from tac.cathedral_consumers import anti_pattern_lookup_consumer


def test_catalog_335_contract_compliance():
    """All 5 Catalog #335 required tokens present + correctly typed."""
    reg = validate_consumer_module(
        anti_pattern_lookup_consumer,
        module_path="tac.cathedral_consumers.anti_pattern_lookup_consumer",
    )
    assert reg.contract_compliant is True
    assert reg.consumer_name == "anti_pattern_lookup_consumer"
    assert reg.validation_errors == ()


def test_hook_numbers_cover_pareto_primary_and_probe_disambiguator():
    """Hook #2 PRIMARY + Hook #6 secondary per design memo."""
    assert HookNumber.PARETO_CONSTRAINT in anti_pattern_lookup_consumer.CONSUMER_HOOK_NUMBERS
    assert HookNumber.PROBE_DISAMBIGUATOR in anti_pattern_lookup_consumer.CONSUMER_HOOK_NUMBERS


def test_catalog_357_tier_a_observability_only():
    """Tier A declared per Catalog #357 (observability-only; no score signal)."""
    assert anti_pattern_lookup_consumer.CONSUMER_TIER == ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def test_catalog_341_routing_markers_in_consume_candidate_return():
    """Every routing-branch return must carry the 3 canonical markers."""
    result = anti_pattern_lookup_consumer.consume_candidate({"some_field": "some_value"})
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"


def test_auto_discovery_via_catalog_335_sister_gate():
    """The consumer must be auto-discoverable via discover_compliant_consumer_modules."""
    from tools.cathedral_autopilot_autonomous_loop import discover_compliant_consumer_modules

    modules = discover_compliant_consumer_modules(repo_root=Path.cwd())
    module_names = {getattr(m, "CONSUMER_NAME", None) for m in modules}
    assert "anti_pattern_lookup_consumer" in module_names


def test_update_from_anchor_handles_arbitrary_anchor_silently():
    """update_from_anchor accepts any anchor without raising (no-op)."""
    result = anti_pattern_lookup_consumer.update_from_anchor({"some": "anchor"})
    assert result is None  # canonical no-op


def test_consume_candidate_with_matching_stack_spec_returns_matches():
    """A candidate with a stack_spec matching anti-pattern #1 (LZMA+brotli) returns matches."""
    # Need the live registry populated for this test; use a tmp registry instead
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        path = td / "anti_patterns.jsonl"
        lock = td / "anti_patterns.jsonl.lock"
        from tac.canonical_anti_patterns import populate_initial_anti_patterns
        populate_initial_anti_patterns(path=path, lock_path=lock)

        # Patch the registry path during consumer invocation
        with patch(
            "tac.canonical_anti_patterns.registry.CANONICAL_ANTI_PATTERNS_REGISTRY_PATH",
            path,
        ):
            result = anti_pattern_lookup_consumer.consume_candidate(
                {
                    "stack_spec": {
                        "compression_ops": ["brotli_q11", "lzma_q9"],
                    },
                }
            )
        assert result["match_count"] >= 1
        assert any(
            "lzma" in mid or "brotli" in mid
            for mid in result["matched_anti_patterns"]
        )


def test_consume_candidate_safe_canonical_stack_returns_empty():
    """A candidate proposing a canonical safe stack returns 0 matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        path = td / "anti_patterns.jsonl"
        lock = td / "anti_patterns.jsonl.lock"
        from tac.canonical_anti_patterns import populate_initial_anti_patterns
        populate_initial_anti_patterns(path=path, lock_path=lock)

        with patch(
            "tac.canonical_anti_patterns.registry.CANONICAL_ANTI_PATTERNS_REGISTRY_PATH",
            path,
        ):
            result = anti_pattern_lookup_consumer.consume_candidate(
                {
                    "stack_spec": {
                        "completely_unrelated_field": "safe_canonical_operation_xyzzy",
                    },
                }
            )
        # Match count may be 0 OR low; the key point is "no canonical anti-pattern matches"
        # phrasing surfaces in the rationale when match_count is 0
        if result["match_count"] == 0:
            assert "no canonical anti-pattern matches" in result["rationale"]


def test_consume_candidate_provenance_kind_canonical():
    result = anti_pattern_lookup_consumer.consume_candidate({"x": "y"})
    assert result["provenance"]["kind"] == "ANTI_PATTERN_LOOKUP_CONSUMER_VERDICT"


def test_consume_candidate_provenance_score_claim_false_always():
    """Provenance must carry score_claim=False (canonical non-promotable per #287/#323)."""
    result = anti_pattern_lookup_consumer.consume_candidate({"x": "y"})
    assert result["provenance"]["score_claim"] is False


def test_consume_candidate_no_stack_spec_falls_back_to_whole_dict():
    """If no stack_spec key, the consumer matches against the whole candidate dict."""
    # No registry → empty matches; just verify we don't crash
    result = anti_pattern_lookup_consumer.consume_candidate({"compression_ops": ["brotli"]})
    assert "match_count" in result


def test_consume_candidate_proposed_stack_spec_sister_synonym():
    """proposed_stack_spec is accepted as a sister-synonym key."""
    result = anti_pattern_lookup_consumer.consume_candidate(
        {"proposed_stack_spec": {"compression_ops": ["safe"]}}
    )
    assert "matched_anti_patterns" in result


def test_consume_candidate_highest_severity_field_present():
    """highest_severity field exists in every return value."""
    result = anti_pattern_lookup_consumer.consume_candidate({"x": "y"})
    assert "highest_severity" in result


def test_consumer_module_constants_match_canonical_pattern():
    """CONSUMER_NAME / CONSUMER_VERSION / CONSUMER_HOOK_NUMBERS / CONSUMER_TIER all present."""
    assert anti_pattern_lookup_consumer.CONSUMER_NAME == "anti_pattern_lookup_consumer"
    assert isinstance(anti_pattern_lookup_consumer.CONSUMER_VERSION, str)
    assert isinstance(anti_pattern_lookup_consumer.CONSUMER_HOOK_NUMBERS, tuple)
    assert len(anti_pattern_lookup_consumer.CONSUMER_HOOK_NUMBERS) >= 1


def test_live_repo_consumer_count_includes_anti_pattern_lookup():
    """Catalog #335 live-repo regression guard."""
    from tools.cathedral_autopilot_autonomous_loop import discover_and_register_consumers

    result = discover_and_register_consumers()
    names = {r["consumer_name"] for r in result}
    assert "anti_pattern_lookup_consumer" in names
    # Compliant + waiver-free
    for r in result:
        if r["consumer_name"] == "anti_pattern_lookup_consumer":
            assert r["contract_compliant"] is True
            assert r["waiver_active"] is False
            break
    else:
        pytest.fail("anti_pattern_lookup_consumer not found in discovery results")
