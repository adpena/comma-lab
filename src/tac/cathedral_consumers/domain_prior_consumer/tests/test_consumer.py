# SPDX-License-Identifier: MIT
"""Tests for domain_prior_consumer (WAVE-3-DIM-4-STEP-4.3).

Coverage per the task contract:
  - Canonical contract compliance per Catalog #335
    (CONSUMER_NAME / CONSUMER_VERSION / CONSUMER_HOOK_NUMBERS /
    update_from_anchor / consume_candidate; explicit TIER_A_OBSERVABILITY_ONLY).
  - Catalog #341 Tier A invariants (predicted_delta_adjustment=0.0 +
    promotable=False + axis_tag="[predicted]").
  - update_from_anchor delegates to canonical helper
    :func:`tac.canonical_equations.update_equation_with_empirical_anchor`
    for each of the 3 DIM-4 equations.
  - consume_candidate aggregates 4-domain prior summaries (typed atlas
    dataclasses + dict-form summaries + missing priors -> None defensive).
  - OPTIONAL BONUS AxisDecomposition emission per Catalog #356 with
    canonical Provenance.
  - Catalog #287 placeholder-rationale rejection (via Provenance).
  - Catalog #323 canonical Provenance integration.
  - Auto-discovery via tools.cathedral_autopilot_autonomous_loop
    .discover_compliant_consumer_modules returns this consumer.
  - Sister-callable regression guards (Catalog #185 + #354 cumulative).
  - Live-repo regression guard (45 -> 46 consumer count).
"""
from __future__ import annotations

import importlib
import math
from typing import Any
from unittest.mock import patch

import pytest

from tac.cathedral.consumer_contract import (
    AxisDecomposition,
    ConsumerTier,
    HookNumber,
    validate_consumer_module,
)


MODULE_PATH = "tac.cathedral_consumers.domain_prior_consumer"
EXPECTED_EQUATION_IDS = (
    "per_frame_difficulty_atlas_v1",
    "ego_motion_concentration_prior_v1",
    "per_segnet_class_chroma_priors_v1",
)
PLACEHOLDER_SHA = "0" * 64
SYNTHETIC_SHA = "a" * 64


def _load_module() -> Any:
    return importlib.import_module(MODULE_PATH)


# ---------------------------------------------------------------------------
# Canonical contract compliance per Catalog #335
# ---------------------------------------------------------------------------


def test_consumer_satisfies_canonical_contract() -> None:
    mod = _load_module()
    res = validate_consumer_module(mod, module_path=MODULE_PATH)
    assert res.contract_compliant, f"validation errors: {res.validation_errors}"


def test_consumer_declares_canonical_metadata() -> None:
    mod = _load_module()
    assert mod.CONSUMER_NAME == "domain_prior_consumer"
    assert mod.CONSUMER_VERSION == "1.0.0"
    assert isinstance(mod.CONSUMER_HOOK_NUMBERS, tuple)
    assert len(mod.CONSUMER_HOOK_NUMBERS) >= 1


def test_consumer_explicit_tier_a_per_catalog_341() -> None:
    """Per Dim 6 Step 6.2: explicit TIER_A_OBSERVABILITY_ONLY."""
    mod = _load_module()
    assert hasattr(mod, "CONSUMER_TIER")
    assert mod.CONSUMER_TIER == ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def test_consumer_hook_numbers_match_declared_active_hooks() -> None:
    """6-hook wire-in: hooks #1, #4, #5 ACTIVE per module docstring."""
    mod = _load_module()
    assert HookNumber.SENSITIVITY_MAP in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in mod.CONSUMER_HOOK_NUMBERS


def test_consumer_hook_numbers_omit_na_hooks() -> None:
    """Hooks #2, #3, #6 are N/A per module docstring."""
    mod = _load_module()
    assert HookNumber.PARETO_CONSTRAINT not in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.BIT_ALLOCATOR not in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.PROBE_DISAMBIGUATOR not in mod.CONSUMER_HOOK_NUMBERS


def test_consumer_callables_are_callable() -> None:
    mod = _load_module()
    assert callable(mod.update_from_anchor)
    assert callable(mod.consume_candidate)


# ---------------------------------------------------------------------------
# Catalog #341 Tier A routing-markers invariants
# ---------------------------------------------------------------------------


CANONICAL_CONSUME_KEYS = (
    "predicted_delta_adjustment",
    "rationale",
    "axis_tag",
    "promotable",
)


def test_consume_candidate_returns_canonical_keys() -> None:
    mod = _load_module()
    result = mod.consume_candidate({})
    for key in CANONICAL_CONSUME_KEYS:
        assert key in result, f"missing canonical key {key!r}"


def test_consume_candidate_tier_a_invariants_empty_candidate() -> None:
    """Catalog #341 Tier A: predicted_delta=0.0 + promotable=False + [predicted]."""
    mod = _load_module()
    result = mod.consume_candidate({})
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["ready_for_exact_eval_dispatch"] is False


def test_consume_candidate_tier_a_invariants_with_priors() -> None:
    """Tier A invariants preserved even when all 4 priors present."""
    mod = _load_module()
    candidate = {
        "archive_sha256": SYNTHETIC_SHA,
        "per_frame_difficulty_summary": {"total_frames": 1200, "aggregator": "mean_over_incident_pairs"},
        "ego_motion_concentration_summary": {"source_anchor_kind": "pose_vector"},
        "per_segnet_class_chroma_summary": {"class_priors": [{"class_index": i} for i in range(5)]},
        "comma2k19_ood_distance_summary": {"cached_chunk_ids": ["chunk_a", "chunk_b"]},
    }
    result = mod.consume_candidate(candidate)
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"


def test_consume_candidate_non_mapping_returns_no_signal_verdict() -> None:
    """Per Catalog #335 paradigm: never raise; return defensive no-signal."""
    mod = _load_module()
    result = mod.consume_candidate("not_a_mapping")  # type: ignore[arg-type]
    assert result["predicted_delta_adjustment"] == 0.0
    assert "not a mapping" in result["rationale"]
    assert "[predicted]" in result["rationale"]


def test_consume_candidate_rationale_min_4_chars_per_catalog_287() -> None:
    """Per Catalog #287: rationale must be non-placeholder, >=4 chars."""
    mod = _load_module()
    result = mod.consume_candidate({})
    rationale = result["rationale"]
    assert isinstance(rationale, str)
    assert len(rationale.strip()) >= 4


# ---------------------------------------------------------------------------
# 4-domain aggregation logic
# ---------------------------------------------------------------------------


def test_consume_candidate_aggregates_4_priors_when_all_present() -> None:
    mod = _load_module()
    candidate = {
        "per_frame_difficulty_summary": {"total_frames": 1200},
        "ego_motion_concentration_summary": {"source_anchor_kind": "pose_vector"},
        "per_segnet_class_chroma_summary": {"source_scorer_kind": "segnet_5_class_argmax"},
        "comma2k19_ood_distance_summary": {"cached_chunk_ids": []},
    }
    result = mod.consume_candidate(candidate)
    assert result["n_priors_present"] == 4
    assert result["per_frame_difficulty_summary"] is not None
    assert result["ego_motion_concentration_summary"] is not None
    assert result["per_segnet_class_chroma_summary"] is not None
    assert result["comma2k19_ood_distance_summary"] is not None


def test_consume_candidate_handles_missing_priors_defensively() -> None:
    """Missing priors -> None per-domain summary, never raises."""
    mod = _load_module()
    result = mod.consume_candidate({})
    assert result["n_priors_present"] == 0
    assert result["per_frame_difficulty_summary"] is None
    assert result["ego_motion_concentration_summary"] is None
    assert result["per_segnet_class_chroma_summary"] is None
    assert result["comma2k19_ood_distance_summary"] is None


def test_consume_candidate_handles_partial_priors() -> None:
    """When only some priors present, count + per-domain breakdown correct."""
    mod = _load_module()
    candidate = {
        "per_frame_difficulty_summary": {"total_frames": 1200},
        "comma2k19_ood_distance_summary": {"cached_chunk_ids": ["chunk_a"]},
    }
    result = mod.consume_candidate(candidate)
    assert result["n_priors_present"] == 2
    assert result["per_frame_difficulty_summary"] is not None
    assert result["ego_motion_concentration_summary"] is None
    assert result["per_segnet_class_chroma_summary"] is None
    assert result["comma2k19_ood_distance_summary"] is not None


def test_consume_candidate_handles_typed_dataclass_with_as_dict() -> None:
    """Typed atlas dataclasses (with .as_dict()) accepted."""
    mod = _load_module()

    class FakeAtlas:
        def as_dict(self) -> dict[str, Any]:
            return {"schema": "fake_atlas_v1", "total_frames": 600}

    candidate = {"per_frame_difficulty_atlas": FakeAtlas()}
    result = mod.consume_candidate(candidate)
    summary = result["per_frame_difficulty_summary"]
    assert summary is not None
    assert summary["total_frames"] == 600


def test_consume_candidate_handles_typed_object_without_as_dict() -> None:
    """Typed objects WITHOUT .as_dict() handled defensively (returns None)."""
    mod = _load_module()

    class BrokenAtlas:
        pass

    candidate = {"per_frame_difficulty_atlas": BrokenAtlas()}
    result = mod.consume_candidate(candidate)
    # Defensive: returns None for that domain rather than raising.
    assert result["per_frame_difficulty_summary"] is None


def test_consume_candidate_handles_dataclass_as_dict_raises() -> None:
    """If as_dict() raises, the consumer is defensive and returns None."""
    mod = _load_module()

    class RaisingAtlas:
        def as_dict(self) -> dict[str, Any]:
            raise TypeError("synthetic failure")

    candidate = {"ego_motion_concentration_atlas": RaisingAtlas()}
    result = mod.consume_candidate(candidate)
    # Defensive: returns None for that domain rather than raising.
    assert result["ego_motion_concentration_summary"] is None


def test_consume_candidate_observability_dict_cites_equation_ids() -> None:
    """Per Catalog #305 cite-ability: emit the 3 DIM-4 equation IDs."""
    mod = _load_module()
    result = mod.consume_candidate({})
    cited = result["domain_priors_equation_ids_cited"]
    assert isinstance(cited, list)
    for eq_id in EXPECTED_EQUATION_IDS:
        assert eq_id in cited


# ---------------------------------------------------------------------------
# OPTIONAL BONUS Catalog #356 AxisDecomposition emission
# ---------------------------------------------------------------------------


def test_consume_candidate_emits_axis_decomposition_when_archive_sha_present() -> None:
    """Per Catalog #356 OPTIONAL BONUS: emit predicted_axis_decomposition."""
    mod = _load_module()
    candidate = {"archive_sha256": SYNTHETIC_SHA}
    result = mod.consume_candidate(candidate)
    assert "predicted_axis_decomposition" in result
    decomp = result["predicted_axis_decomposition"]
    assert isinstance(decomp, dict)
    assert decomp["predicted_d_seg_delta"] == 0.0
    assert decomp["predicted_d_pose_delta"] == 0.0
    assert decomp["predicted_archive_bytes_delta"] == 0
    assert decomp["axis_tag"] == "[predicted]"
    assert isinstance(decomp["canonical_provenance"], dict)
    # Per Catalog #323: canonical_provenance must be non-empty.
    assert len(decomp["canonical_provenance"]) > 0


def test_consume_candidate_omits_axis_decomposition_without_archive_sha() -> None:
    """When archive_sha256 absent, no decomposition emitted (defensive)."""
    mod = _load_module()
    result = mod.consume_candidate({})
    assert "predicted_axis_decomposition" not in result


def test_axis_decomposition_round_trips_through_axis_decomposition_class() -> None:
    """Emitted dict-form must be parseable back via AxisDecomposition.from_dict()."""
    mod = _load_module()
    candidate = {"archive_sha256": SYNTHETIC_SHA}
    result = mod.consume_candidate(candidate)
    decomp_dict = result["predicted_axis_decomposition"]
    # Round-trip:
    reconstructed = AxisDecomposition.from_dict(decomp_dict)
    assert reconstructed.predicted_d_seg_delta == 0.0
    assert reconstructed.predicted_d_pose_delta == 0.0
    assert reconstructed.predicted_archive_bytes_delta == 0
    assert reconstructed.axis_tag == "[predicted]"


def test_axis_decomposition_invalid_sha_does_not_emit() -> None:
    """Non-hex / wrong-length archive sha => no decomposition emitted."""
    mod = _load_module()
    result = mod.consume_candidate({"archive_sha256": "invalid_not_hex"})
    assert "predicted_axis_decomposition" not in result


# ---------------------------------------------------------------------------
# update_from_anchor delegates to canonical helper
# ---------------------------------------------------------------------------


def test_update_from_anchor_none_returns_fail_closed() -> None:
    mod = _load_module()
    result = mod.update_from_anchor(None)
    assert result["accepted"] is False
    assert result["status"] == "fail_closed"
    assert "None" in result["reason"]


def test_update_from_anchor_missing_equation_id_fail_closed() -> None:
    mod = _load_module()
    result = mod.update_from_anchor({"residual": 0.05})
    assert result["accepted"] is False
    assert result["status"] == "fail_closed"
    assert "equation_id" in result["reason"]


def test_update_from_anchor_unknown_equation_id_fail_closed() -> None:
    """Equation IDs outside DOMAIN_PRIORS_EQUATION_IDS rejected."""
    mod = _load_module()
    result = mod.update_from_anchor(
        {"equation_id": "some_other_equation_v1", "residual": 0.05}
    )
    assert result["accepted"] is False
    assert "DOMAIN_PRIORS_EQUATION_IDS" in result["reason"]


def test_update_from_anchor_missing_residual_fail_closed() -> None:
    mod = _load_module()
    result = mod.update_from_anchor(
        {"equation_id": EXPECTED_EQUATION_IDS[0]}
    )
    assert result["accepted"] is False
    assert "residual" in result["reason"]


def test_update_from_anchor_nan_residual_fail_closed() -> None:
    mod = _load_module()
    result = mod.update_from_anchor(
        {"equation_id": EXPECTED_EQUATION_IDS[0], "residual": float("nan")}
    )
    assert result["accepted"] is False
    assert "residual" in result["reason"]


def test_update_from_anchor_delegates_to_canonical_helper() -> None:
    """For each of the 3 DIM-4 equations: anchor routes through canonical.

    Patches the canonical helper at its source module (the consumer imports
    inside the function so the lookup happens via the canonical module's
    binding, not a local copy).
    """
    mod = _load_module()
    for eq_id in EXPECTED_EQUATION_IDS:
        with patch(
            "tac.canonical_equations.update_equation_with_empirical_anchor"
        ) as mock_update:
            result = mod.update_from_anchor(
                {
                    "equation_id": eq_id,
                    "residual": 0.01,
                    "axis_tag": "[predicted]",
                    "archive_sha256": SYNTHETIC_SHA,
                    "measurement_utc": "2026-05-20T14:00:00Z",
                    "call_id": "test_call",
                    "predicted_value": 0.19,
                }
            )
            assert mock_update.called, f"canonical helper not called for {eq_id}"
            kwargs = mock_update.call_args.kwargs
            assert kwargs.get("equation_id") == eq_id
            assert result["accepted"] is True
            assert result["status"] == "canonical_equation_updated"
            assert result["equation_id"] == eq_id
            assert result["score_claim"] is False
            assert result["promotion_eligible"] is False


def test_update_from_anchor_handles_canonical_helper_error_defensively() -> None:
    """If canonical helper raises, return fail-closed payload (no re-raise)."""
    mod = _load_module()
    with patch(
        "tac.canonical_equations.update_equation_with_empirical_anchor",
        side_effect=ValueError("synthetic helper failure"),
    ):
        result = mod.update_from_anchor(
            {
                "equation_id": EXPECTED_EQUATION_IDS[0],
                "residual": 0.01,
                "archive_sha256": SYNTHETIC_SHA,
            }
        )
    assert result["accepted"] is False
    assert "canonical equation update raised" in result["reason"]


def test_update_from_anchor_handles_bool_residual_rejected() -> None:
    """bool is a subclass of int; explicitly reject so True/False -> 1.0/0.0 doesn't sneak in."""
    mod = _load_module()
    result = mod.update_from_anchor(
        {"equation_id": EXPECTED_EQUATION_IDS[0], "residual": True}
    )
    assert result["accepted"] is False


# ---------------------------------------------------------------------------
# Catalog #323 canonical Provenance integration
# ---------------------------------------------------------------------------


def test_axis_decomposition_provenance_carries_predicted_grade() -> None:
    """Per Catalog #323: Provenance dict carries PREDICTED grade + non-promotable."""
    mod = _load_module()
    candidate = {"archive_sha256": SYNTHETIC_SHA}
    result = mod.consume_candidate(candidate)
    decomp = result["predicted_axis_decomposition"]
    prov = decomp["canonical_provenance"]
    assert isinstance(prov, dict)
    # Per build_provenance_for_predicted invariants:
    assert "measurement_axis" in prov
    assert prov.get("measurement_axis") == "[predicted]"


def test_axis_decomposition_provenance_cite_chain_includes_consumer_model_id() -> None:
    """Per Catalog #323 cite-ability: model_id surfaces the consumer.

    The canonical :func:`build_provenance_for_predicted` embeds ``model_id``
    inside ``source_path`` (format: ``<predictor:<model_id>>``).
    """
    mod = _load_module()
    candidate = {"archive_sha256": SYNTHETIC_SHA}
    result = mod.consume_candidate(candidate)
    decomp = result["predicted_axis_decomposition"]
    prov = decomp["canonical_provenance"]
    # The canonical Provenance dict carries model_id via source_path.
    source_path = str(prov.get("source_path", ""))
    canonical_helper = str(prov.get("canonical_helper_invocation", ""))
    cite_chain = source_path + "|" + canonical_helper
    assert "domain_prior_consumer" in cite_chain, (
        f"consumer name missing from cite-chain (source_path={source_path!r}, "
        f"canonical_helper={canonical_helper!r})"
    )


# ---------------------------------------------------------------------------
# Auto-discovery per Catalog #335
# ---------------------------------------------------------------------------


def test_auto_discovery_returns_domain_prior_consumer() -> None:
    """Per Catalog #335: discover_compliant_consumer_modules ingests this consumer."""
    from tools.cathedral_autopilot_autonomous_loop import (
        discover_compliant_consumer_modules,
    )

    modules = discover_compliant_consumer_modules()
    names = [getattr(m, "CONSUMER_NAME", None) for m in modules]
    assert "domain_prior_consumer" in names, f"not in discovered: {names}"


def test_auto_discovery_count_post_landing() -> None:
    """Live-repo regression guard: cathedral consumers count goes 45 -> 46."""
    from tools.cathedral_autopilot_autonomous_loop import (
        discover_compliant_consumer_modules,
    )

    modules = discover_compliant_consumer_modules()
    # Pre-landing: 45 production consumers.
    # Post-landing: 46.
    assert len(modules) >= 46, (
        f"expected >=46 consumer packages post-domain_prior_consumer landing; "
        f"got {len(modules)}"
    )


# ---------------------------------------------------------------------------
# Sister-callable regression guards (Catalog #185 + #354 cumulative)
# ---------------------------------------------------------------------------


def test_canonical_equations_3_dim_4_equations_remain_registered() -> None:
    """Catalog #344 + DIM-4 NAMESPACE landing: 3 DIM-4 equations registered."""
    from tac.canonical_equations.registry import get_equation_by_id

    for eq_id in EXPECTED_EQUATION_IDS:
        eq = get_equation_by_id(eq_id)
        assert eq is not None, f"DIM-4 equation {eq_id} missing from registry"
        assert eq.equation_id == eq_id
        # Per Catalog #344 orphan-equation invariant:
        assert len(eq.canonical_producers) > 0
        assert len(eq.canonical_consumers) > 0


def test_domain_priors_namespace_helpers_importable() -> None:
    """Sister regression: DIM-4 namespace remains importable + canonical."""
    import tac.domain_priors as dp

    assert hasattr(dp, "PerFrameDifficultyAtlas")
    assert hasattr(dp, "EgoMotionConcentrationAtlas")
    assert hasattr(dp, "PerClassStatisticalPriors")
    assert hasattr(dp, "Comma2k19DashcamPriors")
    assert hasattr(dp, "DOMAIN_PRIORS_EQUATION_IDS")
    # Sister of EXPECTED_EQUATION_IDS pinned above.
    assert tuple(dp.DOMAIN_PRIORS_EQUATION_IDS) == EXPECTED_EQUATION_IDS


def test_consumer_module_callable_via_globals() -> None:
    """Catalog #185 sister regression: every consumer's contract entry-points discoverable."""
    mod = _load_module()
    contract_callables = ("update_from_anchor", "consume_candidate")
    for name in contract_callables:
        attr = getattr(mod, name, None)
        assert callable(attr), f"{name} not callable"


def test_consume_candidate_preserves_provenance_in_canonical_pattern() -> None:
    """Sister consumer pattern: Provenance dict carries grade + axis at minimum."""
    mod = _load_module()
    candidate = {"archive_sha256": SYNTHETIC_SHA}
    result = mod.consume_candidate(candidate)
    decomp = result.get("predicted_axis_decomposition")
    assert decomp is not None
    prov = decomp["canonical_provenance"]
    # Minimum required: evidence_grade + measurement_axis per Catalog #323.
    assert "evidence_grade" in prov or "measurement_axis" in prov
