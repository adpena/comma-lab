# SPDX-License-Identifier: MIT
"""Tests for the Wyner-Ziv decoder-side PoseNet side-information cathedral consumer.

Per operator task #1496 Wave N+36 routing + CLAUDE.md "Subagent coherence-by-default"
non-negotiable + Catalog #335 paradigm-shift (canonical contract auto-discovery)
+ Catalog #341 Tier A canonical-routing markers.
"""

from __future__ import annotations

import importlib

from tac.cathedral.consumer_contract import (
    HookNumber,
    validate_consumer_module,
)

MODULE_PATH = "tac.cathedral_consumers.wyner_ziv_posenet_side_information_consumer"


def _load_module():
    return importlib.import_module(MODULE_PATH)


# ---------------------------------------------------------------------------
# Catalog #335 canonical Protocol contract regression guards
# ---------------------------------------------------------------------------


def test_consumer_satisfies_canonical_contract() -> None:
    mod = _load_module()
    res = validate_consumer_module(mod)
    assert res.contract_compliant, f"validation errors: {res.validation_errors}"


def test_consumer_name_is_canonical() -> None:
    mod = _load_module()
    assert mod.CONSUMER_NAME == "wyner_ziv_posenet_side_information_consumer"


def test_consumer_version_is_semver_like() -> None:
    mod = _load_module()
    assert mod.CONSUMER_VERSION == "0.1.0"


def test_consumer_declares_all_6_hooks() -> None:
    """Per the consumer docstring: hooks 1-6 ALL active for Wyner-Ziv side-info."""
    mod = _load_module()
    expected_hooks = {
        HookNumber.SENSITIVITY_MAP,
        HookNumber.PARETO_CONSTRAINT,
        HookNumber.BIT_ALLOCATOR,
        HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
        HookNumber.CONTINUAL_LEARNING_POSTERIOR,
        HookNumber.PROBE_DISAMBIGUATOR,
    }
    declared = set(mod.CONSUMER_HOOK_NUMBERS)
    assert declared == expected_hooks


def test_consumer_exposes_callable_surfaces() -> None:
    mod = _load_module()
    assert callable(mod.update_from_anchor)
    assert callable(mod.consume_candidate)


# ---------------------------------------------------------------------------
# Catalog #341 Tier A canonical-routing markers
# ---------------------------------------------------------------------------


def test_consume_candidate_returns_canonical_tier_a_markers_for_matching_candidate() -> None:
    """Tier A: predicted_delta_adjustment=0.0 + promotable=False + axis_tag=[predicted]."""
    mod = _load_module()
    out = mod.consume_candidate(
        {
            "lane_id": "test_wyner_ziv_substrate",
            "substrate_id": "z8_hierarchical_predictive_coding",
            "archive_family": "wyner_ziv_layer",
            "side_info_delivery_mode": "archive_charged",
            "side_info_charged_bytes": 128,
        }
    )
    # Catalog #341 canonical Tier A markers MUST be present + correct.
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["promotable"] is False
    assert out["axis_tag"] == "[predicted]"
    assert out["confidence"] == 0.0
    assert out["decoder_side_info_custody_proven"] is True


def test_consume_candidate_returns_canonical_tier_a_markers_for_non_matching_candidate() -> None:
    """Even when candidate does NOT match, Tier A markers MUST still hold."""
    mod = _load_module()
    out = mod.consume_candidate({"lane_id": "unrelated", "archive_family": "pr101_baseline"})
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["promotable"] is False
    assert out["axis_tag"] == "[predicted]"


# ---------------------------------------------------------------------------
# Domain-match behavior
# ---------------------------------------------------------------------------


def test_match_via_wyner_ziv_token() -> None:
    mod = _load_module()
    out = mod.consume_candidate(
        {
            "lane_id": "wyner_ziv_substrate",
            "side_info_delivery_mode": "archive_charged",
            "side_info_charged_bytes": 64,
        }
    )
    assert out["wyner_ziv_posenet_side_info_match"] is True


def test_match_via_wyner_dash_ziv_token() -> None:
    mod = _load_module()
    out = mod.consume_candidate(
        {
            "lane_id": "lane_wyner-ziv_pipeline",
            "side_info_delivery_mode": "fixed_contest_input",
        }
    )
    assert out["wyner_ziv_posenet_side_info_match"] is True


def test_match_via_z8_hierarchical_predictive_coding_token() -> None:
    mod = _load_module()
    out = mod.consume_candidate(
        {
            "substrate_id": "z8_hierarchical_predictive_coding",
            "archive_bound_side_info": True,
        }
    )
    assert out["wyner_ziv_posenet_side_info_match"] is True


def test_match_via_posenet_conditional_token() -> None:
    mod = _load_module()
    out = mod.consume_candidate(
        {
            "description": "posenet_conditional coding",
            "side_info_delivery_mode": "archive_charged",
        }
    )
    assert out["wyner_ziv_posenet_side_info_match"] is True


def test_match_via_cooperative_receiver_atick_redlich_sister() -> None:
    """Atick-Redlich cooperative-receiver IS a canonical sister surface."""
    mod = _load_module()
    out = mod.consume_candidate(
        {
            "lane_id": "cooperative_receiver_substrate",
            "side_info_delivery_mode": "archive_charged",
        }
    )
    assert out["wyner_ziv_posenet_side_info_match"] is True


def test_wyner_ziv_token_without_custody_fails_closed() -> None:
    mod = _load_module()
    out = mod.consume_candidate({"lane_id": "wyner_ziv_substrate"})
    assert out["wyner_ziv_posenet_side_info_match"] is False
    assert out["decoder_side_info_custody_proven"] is False
    assert "posenet_side_info_archive_custody_or_fixed_input_proof_missing" in out["blockers"]


def test_free_scorer_runtime_side_info_fails_closed() -> None:
    mod = _load_module()
    out = mod.consume_candidate(
        {
            "lane_id": "wyner_ziv_substrate",
            "side_info_delivery_mode": "scorer_runtime_free",
        }
    )
    assert out["wyner_ziv_posenet_side_info_match"] is False
    assert "non_compliant_inflate_time_scorer_side_information_forbidden" in out["blockers"]


def test_no_match_for_unrelated_candidate() -> None:
    mod = _load_module()
    out = mod.consume_candidate({"lane_id": "fec6_pr101", "archive_family": "selector_codec"})
    assert out["wyner_ziv_posenet_side_info_match"] is False


def test_no_match_for_empty_candidate() -> None:
    mod = _load_module()
    out = mod.consume_candidate({})
    assert out["wyner_ziv_posenet_side_info_match"] is False


def test_match_via_nested_mapping_field() -> None:
    """consumer walks nested string/numeric values for match tokens."""
    mod = _load_module()
    out = mod.consume_candidate(
        {
            "metadata": {
                "codec_lineage": "wyner_ziv_layer",
                "score_axis": "cuda",
            },
            "side_info_delivery_mode": "archive_charged",
        }
    )
    assert out["wyner_ziv_posenet_side_info_match"] is True


# ---------------------------------------------------------------------------
# Hook #5 continual-learning posterior
# ---------------------------------------------------------------------------


def test_update_from_anchor_is_no_op_per_canonical_contract() -> None:
    """Per the docstring: structurally NO-OP at the per-anchor surface."""
    mod = _load_module()
    # Should not raise and should not return anything useful.
    result = mod.update_from_anchor({"anchor_id": "test", "residual": 0.5})
    assert result is None


def test_update_from_anchor_tolerates_none_anchor() -> None:
    mod = _load_module()
    # Should not raise on None.
    mod.update_from_anchor(None)


# ---------------------------------------------------------------------------
# Residual summary lookup (canonical equation registry integration)
# ---------------------------------------------------------------------------


def test_consume_candidate_surfaces_matched_equation_id_when_match() -> None:
    mod = _load_module()
    out = mod.consume_candidate(
        {
            "lane_id": "test_wyner_ziv_z8_substrate",
            "side_info_delivery_mode": "archive_charged",
        }
    )
    assert out["wyner_ziv_posenet_side_info_match"] is True
    # After Wave N+36 lands the equation, anchor_count should be >= 1.
    # (Test tolerates registry not yet populated — it's a regression guard
    # that the consumer SURFACES the key, not the absolute count.)
    assert "anchor_count" in out
    assert "is_well_calibrated" in out
    assert "per_axis_residuals" in out


def test_consume_candidate_rationale_carries_predicted_axis_tag_when_no_match() -> None:
    """Per Catalog #287 + #323: every rationale MUST carry an axis tag."""
    mod = _load_module()
    out = mod.consume_candidate({"lane_id": "fec6_unrelated"})
    assert "[predicted]" in out["rationale"]


def test_consume_candidate_rationale_carries_wyner_ziv_1976_citation_when_match() -> None:
    """When matched, the rationale MUST cite the canonical Wyner-Ziv 1976 paradigm."""
    mod = _load_module()
    out = mod.consume_candidate(
        {
            "lane_id": "wyner_ziv_substrate",
            "side_info_delivery_mode": "archive_charged",
        }
    )
    assert "Wyner-Ziv 1976" in out["rationale"]


# ---------------------------------------------------------------------------
# Canonical exports
# ---------------------------------------------------------------------------


def test_consumer_module_exports_canonical_surface() -> None:
    mod = _load_module()
    for required in (
        "CONSUMER_NAME",
        "CONSUMER_VERSION",
        "CONSUMER_HOOK_NUMBERS",
        "consume_candidate",
        "update_from_anchor",
    ):
        assert required in mod.__all__, f"missing canonical export: {required}"
