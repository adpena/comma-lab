# SPDX-License-Identifier: MIT
"""Tests for the wave_n_plus_48_hygiene_lookup_consumer (Catalog #344 sister)."""
from __future__ import annotations

import importlib

import pytest

from tac.cathedral.consumer_contract import (
    HookNumber,
    validate_consumer_module,
)


MODULE_PATH = "tac.cathedral_consumers.wave_n_plus_48_hygiene_lookup_consumer"


def _load_module():
    return importlib.import_module(MODULE_PATH)


# ---------------------------------------------------------------------------
# Catalog #335 canonical consumer contract compliance
# ---------------------------------------------------------------------------


def test_consumer_satisfies_canonical_contract():
    """Per Catalog #335: every cathedral consumer MUST satisfy the canonical
    Protocol contract (CONSUMER_NAME / CONSUMER_VERSION / CONSUMER_HOOK_NUMBERS
    + update_from_anchor + consume_candidate)."""
    mod = _load_module()
    res = validate_consumer_module(mod)
    assert res.contract_compliant, f"validation errors: {res.validation_errors}"


def test_consumer_declares_hooks_4_and_5():
    """Per Catalog #125: this consumer declares hook #4 (cathedral autopilot
    dispatch) + hook #5 (continual-learning posterior)."""
    mod = _load_module()
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in mod.CONSUMER_HOOK_NUMBERS


def test_consumer_name_and_version_are_canonical():
    mod = _load_module()
    assert mod.CONSUMER_NAME == "wave_n_plus_48_hygiene_lookup_consumer"
    assert mod.CONSUMER_VERSION == "0.1.0"


def test_consumer_cites_canonical_equation_id():
    """Sister of Slot O Phase 1: the canonical equation id MUST match the
    registered equation_id in canonical_equations_registry.jsonl row 326."""
    mod = _load_module()
    assert (
        mod.WAVE_N_PLUS_48_EQUATION_ID
        == "wave_n_plus_48_l1_l42_hygiene_ev_decay_predicts_pr95_parity_gap_v1"
    )


# ---------------------------------------------------------------------------
# Catalog #341 Tier A canonical-routing markers compliance
# ---------------------------------------------------------------------------


def test_consume_candidate_returns_canonical_tier_a_markers():
    """Per Catalog #341: every consume_candidate return value MUST carry
    predicted_delta_adjustment=0.0 + promotable=False + axis_tag='[predicted]'
    (Tier A observability-only contract)."""
    mod = _load_module()
    out = mod.consume_candidate({"lane_id": "test", "substrate_family": "pr101"})
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


def test_consume_candidate_emits_wave_n_plus_48_specific_fields():
    """Wave N+48 consumer surfaces hygiene_ev + tier_label + mission hint +
    routing_recommendation + canonical_equation_id in addition to canonical
    Tier A markers."""
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "pr101_family"})
    for wave_n48_key in (
        "matched_family",
        "hygiene_ev",
        "tier_label",
        "mission_contribution_hint",
        "routing_recommendation",
        "canonical_equation_id",
    ):
        assert wave_n48_key in out


def test_predicted_delta_always_zero_never_promoted():
    """Observability-only contract: predicted_delta_adjustment MUST be 0.0
    regardless of candidate input per CLAUDE.md "Apples-to-apples"."""
    mod = _load_module()
    for candidate in [
        {},
        {"foo": "bar"},
        {"substrate_family": "pr101_family"},
        {"substrate_family": "sane_hnerv"},
        {"substrate_family": "time_traveler_l5"},
        {"substrate_family": "fec6"},
        {"substrate_family": "unknown_family_xyz"},
    ]:
        out = mod.consume_candidate(candidate)
        assert out["predicted_delta_adjustment"] == 0.0
        assert out["promotable"] is False
        assert out["axis_tag"] == "[predicted]"


# ---------------------------------------------------------------------------
# Wave N+48 substrate-family scoring matrix (canonical reference data)
# ---------------------------------------------------------------------------


def test_top_2_paired_cuda_eligible_pr101_family_scores_0_80():
    """Per Wave N+48 audit Phase B: PR101_FAMILY achieves HYG-EV 0.80 (TOP-2)."""
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "pr101_family"})
    assert out["matched_family"] == "pr101_family"
    assert out["hygiene_ev"] == 0.80
    assert out["tier_label"] == "TOP_2_PAIRED_CUDA_ELIGIBLE"
    assert out["mission_contribution_hint"] == "frontier_protecting"


def test_top_2_paired_cuda_eligible_frame_exploit_hfv_scores_0_80():
    """Per Wave N+48 audit Phase B: FRAME_EXPLOIT_HFV achieves HYG-EV 0.80 (TOP-2)."""
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "frame_exploit_hfv"})
    assert out["hygiene_ev"] == 0.80
    assert out["tier_label"] == "TOP_2_PAIRED_CUDA_ELIGIBLE"
    assert out["mission_contribution_hint"] == "frontier_breaking_enabler"


def test_mid_tier_pr110_opt_scores_0_71():
    """Per Wave N+48 audit: PR110_OPT mid-tier 0.71."""
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "pr110_opt"})
    assert out["hygiene_ev"] == 0.71
    assert out["tier_label"] == "MID_TIER_CLOSE_REVIEW_BASELINE"


def test_mid_tier_pr103_hnerv_lc_ac_scores_0_68():
    """Per Wave N+48 audit: PR103_HNERV_LC_AC mid-tier 0.68."""
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "pr103_hnerv_lc_ac"})
    assert out["hygiene_ev"] == 0.68
    assert out["tier_label"] == "MID_TIER_CLOSE_REVIEW_BASELINE"


def test_mid_tier_pr95_family_direct_scores_0_59():
    """Per Wave N+48 audit: PR95_FAMILY_DIRECT mid-tier 0.59."""
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "pr95_family_direct"})
    assert out["hygiene_ev"] == 0.59
    assert out["tier_label"] == "MID_TIER_CLOSE_REVIEW_BASELINE"


def test_bottom_tier_sane_hnerv_scores_0_49():
    """Per Wave N+48 audit: sane_hnerv (PR100 lineage) BOTTOM-TIER 0.49 — empirical
    confirmation of operator prediction that 12/12 baseline drops at L1-L42."""
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "sane_hnerv"})
    assert out["hygiene_ev"] == 0.49
    assert out["tier_label"] == "BOTTOM_TIER_L14_L42_BACKFILL_REQUIRED"
    assert out["mission_contribution_hint"] == "apparatus_maintenance"


def test_bottom_tier_pr106_latent_sidecar_scores_0_49():
    """Per Wave N+48 audit: PR106 BOTTOM-TIER 0.49."""
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "pr106_latent_sidecar"})
    assert out["hygiene_ev"] == 0.49
    assert out["tier_label"] == "BOTTOM_TIER_L14_L42_BACKFILL_REQUIRED"


def test_bottom_tier_a1_sister_scores_0_49():
    """Per Wave N+48 audit: A1_SISTER BOTTOM-TIER 0.49."""
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "a1_sister"})
    assert out["hygiene_ev"] == 0.49
    assert out["tier_label"] == "BOTTOM_TIER_L14_L42_BACKFILL_REQUIRED"


def test_low_confidence_high_leverage_time_traveler_z_family_scores_0_05():
    """Per Wave N+48 audit: TIME_TRAVELER_Z_FAMILY HYG-EV 0.05 LOW-confidence-
    HIGH-leverage class-shift (Tier 3 $0 MLX-LOCAL)."""
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "time_traveler_z_family"})
    assert out["hygiene_ev"] == 0.05
    assert out["tier_label"] == "LOW_CONFIDENCE_HIGH_LEVERAGE_MLX_LOCAL"
    assert out["mission_contribution_hint"] == "frontier_breaking_enabler"


def test_fec_bolt_on_stack_scores_0_34():
    """Per Wave N+48 audit: FEC_RATE_FAMILY HYG-EV 0.34 off-the-shelf stacking."""
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "fec_rate_family"})
    assert out["hygiene_ev"] == 0.34
    assert out["tier_label"] == "FEC_BOLT_ON_STACK"


# ---------------------------------------------------------------------------
# Routing recommendations per Wave N+48 Phase C trichotomy
# ---------------------------------------------------------------------------


def test_top_2_routing_recommends_tier_1_paired_cuda_eligible():
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "pr101_family"})
    assert "TIER_1_PAIRED_CUDA_ELIGIBLE" in out["routing_recommendation"]
    assert "canonical-frontier-protected" in out["routing_recommendation"]


def test_mid_tier_routing_recommends_tier_2_queue_eligible():
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "pr110_opt"})
    assert "TIER_2_PAIRED_CUDA_QUEUE_ELIGIBLE" in out["routing_recommendation"]


def test_bottom_tier_routing_recommends_apparatus_maintenance():
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "sane_hnerv"})
    assert "TIER_4_APPARATUS_MAINTENANCE" in out["routing_recommendation"]
    assert "L14-L42" in out["routing_recommendation"]


def test_low_confidence_routing_recommends_mlx_local_fanout():
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "time_traveler_l5"})
    assert "TIER_3_MLX_LOCAL_FANOUT" in out["routing_recommendation"]
    assert "dispatch_enabled=false" in out["routing_recommendation"]


# ---------------------------------------------------------------------------
# Best-effort substring matching (highest hygiene-EV wins on overlap)
# ---------------------------------------------------------------------------


def test_consume_candidate_no_match_yields_observability_only():
    mod = _load_module()
    out = mod.consume_candidate({"unrelated_key": "unrelated_value_xyz"})
    assert out["matched_family"] is None
    assert out["hygiene_ev"] is None
    assert out["tier_label"] is None
    assert out["predicted_delta_adjustment"] == 0.0


def test_consume_candidate_matches_token_in_lane_id():
    """The consumer walks all string values in the candidate dict; substrate
    family tokens can appear in lane_id, archive_family, recipe, etc."""
    mod = _load_module()
    out = mod.consume_candidate(
        {"lane_id": "lane_pr101_family_clone_20260512"}
    )
    assert out["matched_family"] == "pr101_family"
    assert out["hygiene_ev"] == 0.80


def test_consume_candidate_picks_highest_hygiene_ev_on_overlap():
    """When multiple family tokens match (e.g. candidate cites both pr101_family
    and pr106), the highest hygiene-EV match wins (canonical disambiguation)."""
    mod = _load_module()
    out = mod.consume_candidate(
        {"lane_id": "lane_pr101_family_plus_pr106_stack_composition"}
    )
    # PR101_FAMILY (0.80) wins over PR106 (0.49)
    assert out["hygiene_ev"] == 0.80
    assert out["matched_family"] == "pr101_family"


# ---------------------------------------------------------------------------
# Catalog #125 hook contract: update_from_anchor + consume_candidate behavior
# ---------------------------------------------------------------------------


def test_update_from_anchor_is_no_op_at_consumer_level():
    """Consumer's update_from_anchor MUST NOT raise; auto-recalibration is
    operator-triggered per Catalog #287/#323 + Catalog #371 auto-recalibrator."""
    mod = _load_module()
    mod.update_from_anchor(object())  # any anchor type; consumer ignores
    mod.update_from_anchor(None)
    mod.update_from_anchor({"some": "anchor", "dict": "value"})


def test_consume_candidate_rationale_includes_canonical_equation_id():
    """Per Catalog #344: every annotation surfaces the canonical equation id
    so downstream consumers can route to the registry for residual lookup."""
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "pr101_family"})
    assert (
        "wave_n_plus_48_l1_l42_hygiene_ev_decay_predicts_pr95_parity_gap_v1"
        in out["rationale"]
    )


def test_consume_candidate_rationale_includes_axis_tag_marker():
    """Per Catalog #287: rationale string must carry the [predicted] axis tag
    so search/grep tools recognize this as observability-only."""
    mod = _load_module()
    out = mod.consume_candidate({"substrate_family": "pr101_family"})
    assert "[predicted]" in out["rationale"]


# ---------------------------------------------------------------------------
# Catalog #335 auto-discovery regression: module is importable + discoverable
# ---------------------------------------------------------------------------


def test_consumer_module_importable_via_canonical_namespace():
    """Per Catalog #335 cathedral consumer auto-discovery: this module MUST be
    importable via the canonical tac.cathedral_consumers.* namespace so
    auto-discovery loop finds it."""
    import importlib
    mod = importlib.import_module(
        "tac.cathedral_consumers.wave_n_plus_48_hygiene_lookup_consumer"
    )
    assert mod.CONSUMER_NAME == "wave_n_plus_48_hygiene_lookup_consumer"
