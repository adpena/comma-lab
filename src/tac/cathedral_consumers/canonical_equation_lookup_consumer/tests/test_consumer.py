# SPDX-License-Identifier: MIT
"""Tests for the canonical_equation_lookup_consumer (Catalog #344 sister)."""
from __future__ import annotations

import importlib

import pytest

from tac.cathedral.consumer_contract import (
    HookNumber,
    validate_consumer_module,
)


MODULE_PATH = "tac.cathedral_consumers.canonical_equation_lookup_consumer"


def _load_module():
    return importlib.import_module(MODULE_PATH)


def test_consumer_satisfies_canonical_contract():
    mod = _load_module()
    res = validate_consumer_module(mod)
    assert res.contract_compliant, f"validation errors: {res.validation_errors}"


def test_consumer_declares_hooks_4_and_5():
    mod = _load_module()
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in mod.CONSUMER_HOOK_NUMBERS


def test_consumer_name_and_version_are_canonical():
    mod = _load_module()
    assert mod.CONSUMER_NAME == "canonical_equation_lookup_consumer"
    assert mod.CONSUMER_VERSION == "0.1.0"


def test_consume_candidate_returns_canonical_keys():
    mod = _load_module()
    out = mod.consume_candidate({"lane_id": "test", "archive_family": "pr101"})
    for required_key in (
        "predicted_delta_adjustment",
        "rationale",
        "axis_tag",
        "promotable",
        "confidence",
    ):
        assert required_key in out
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["promotable"] is False
    assert out["axis_tag"] == "[predicted]"


def test_consume_candidate_matches_equation_consumer_token():
    mod = _load_module()
    out = mod.consume_candidate(
        {"helper": "tac.master_gradient_iterative_refinement"}
    )
    matched = out.get("matched_equations", [])
    matched_ids = {m["equation_id"] for m in matched}
    # Equations whose canonical_consumers mention this helper:
    expected = {
        "per_byte_leverage_uniformly_distributed_v1",
        "per_pair_master_gradient_score_impact_taylor_v1",
        "master_gradient_locality_violation_by_codec_v1",
    }
    assert expected.issubset(matched_ids), f"got {matched_ids}"


def test_consume_candidate_no_match_yields_observability_only():
    mod = _load_module()
    out = mod.consume_candidate({"unrelated_key": "unrelated_value"})
    assert out["matched_equations"] == []
    assert out["predicted_delta_adjustment"] == 0.0


def test_update_from_anchor_is_no_op_at_consumer_level():
    """Consumer's update_from_anchor MUST NOT raise; auto-recalibration is
    operator-triggered per Catalog #287/#323."""
    mod = _load_module()
    mod.update_from_anchor(object())  # any anchor type; consumer ignores


def test_predicted_delta_always_zero_never_promoted():
    """Observability-only contract: predicted_delta_adjustment MUST be 0.0
    regardless of candidate input per CLAUDE.md "Apples-to-apples"."""
    mod = _load_module()
    for candidate in [
        {},
        {"foo": "bar"},
        {"archive_family": "pr101", "expected_band_lower": -0.01},
        {"helper": "tac.mps_diagnostic"},
    ]:
        out = mod.consume_candidate(candidate)
        assert out["predicted_delta_adjustment"] == 0.0
        assert out["promotable"] is False
