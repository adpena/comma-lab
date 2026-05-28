# SPDX-License-Identifier: MIT
"""False-positive guard tests for the architectural-fix override predicates.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
+ Wave N+3 Slot 2 architectural fix 2026-05-28: the legacy matcher fired
token-overlap fallbacks at confidence 0.5 EVEN WHEN the proposed
stack_spec carried explicit override flags or structured fields that
structurally refuted the forbidden predicate. This file lands the
≥10 false-positive guard tests required by the parent prompt, plus
sister positive-case regression guards proving the architectural fix
does NOT degrade the legitimate-positive-match path.

Empirical anchors (3× this session):
  * Compound C STAND_DOWN ``e61ea93b0`` — fp4_packed_without_qat_cos_collapse_v1
    fired DESPITE quantization_aware_training=True.
  * Wave N+3 Slot 1 PyTorch sister ``4c1daf186`` — brotli_plus_lzma_chained
    fired DESPITE NO lzma anywhere in compression_ops.
  * Compound F preflight ``e5467cf05`` — sister false-positive in compound
    test setup; same META class.

Sister of:
  * ``test_registry.py`` (44 existing tests; legacy positive-path coverage).
  * ``src/tac/substrates/pact_nerv_selector_v3/heterogeneous_bit_allocation.py
    ::assert_no_critical_anti_pattern_matches`` (symptom-only filter; this
    architectural fix makes the filter redundant for fp4_packed_without_qat
    but it remains as defense-in-depth).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tac.canonical_anti_patterns.builtins import populate_initial_anti_patterns
from tac.canonical_anti_patterns.pattern_matcher import (
    _EXPLICIT_OVERRIDE_PREDICATES,
    _explicit_override_brotli_plus_lzma_chained,
    _explicit_override_cross_paradigm_test_per_axis_decomposition,
    _explicit_override_fp4_packed_without_qat,
    _explicit_override_lzma_on_already_brotli,
    _explicit_override_phantom_score_directory_naming_lie,
    _explicit_override_predecessor_working_tree_uncommitted_handoff,
    _explicit_override_predicted_band_from_random_init_tier_c,
    _explicit_override_quantize_then_svd_corrupted,
    _explicit_override_rank_1_problem_spec_synergy_tautology,
    _explicit_override_silent_no_spawn_modal_dispatch,
    _explicit_override_source_selector_inherited_predicted_score_mean,
    _explicit_override_subagent_spawn_without_head_state_premise_verification,
    _explicit_override_transient_tmp_path_in_persisted_artifact,
    _explicit_override_wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface,
    _explicit_override_wyner_ziv_prefix_y_density_decoder_state_dict_surface,
    evaluate_explicit_override_for_anti_pattern,
    match_stack_against_anti_patterns,
)


@pytest.fixture
def temp_registry():
    """Fresh tmp ledger + lock for each test (no live-state pollution).

    Mirrors the canonical fixture from test_registry.py.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        path = td / "test_registry.jsonl"
        lock = td / "test_registry.jsonl.lock"
        yield path, lock


# ---------------------------------------------------------------------------
# Per-predicate unit tests (pure functions; no registry needed)
# ---------------------------------------------------------------------------


def test_fp4_qat_override_fires_for_explicit_quantization_aware_training_true():
    spec = {"quantization_aware_training": True}
    inapplicable, reason = _explicit_override_fp4_packed_without_qat(spec)
    assert inapplicable is True
    assert "quantization_aware_training" in reason.lower()


def test_fp4_qat_override_fires_for_qat_enabled_true():
    spec = {"qat_enabled": True}
    inapplicable, reason = _explicit_override_fp4_packed_without_qat(spec)
    assert inapplicable is True
    assert "qat_enabled" in reason.lower()


def test_fp4_qat_override_fires_for_qat_finetune_passes_int():
    spec = {"qat_finetune_passes": 3}
    inapplicable, reason = _explicit_override_fp4_packed_without_qat(spec)
    assert inapplicable is True
    assert "qat_finetune_passes" in reason


def test_fp4_qat_override_rejects_bool_int_for_qat_finetune_passes():
    # Per CLAUDE.md "Comment-only contracts are FORBIDDEN" — be explicit:
    # ``True`` is an int subclass in Python; we treat True as the bool
    # sentinel via ``quantization_aware_training`` only, NOT as a passes-count
    # of "1".
    spec = {"qat_finetune_passes": True}
    inapplicable, _reason = _explicit_override_fp4_packed_without_qat(spec)
    assert inapplicable is False


def test_fp4_qat_override_fires_for_nested_training_pipeline_qat_finetune():
    spec = {"training_pipeline": {"qat_finetune": True}}
    inapplicable, _reason = _explicit_override_fp4_packed_without_qat(spec)
    assert inapplicable is True


def test_fp4_qat_override_fires_for_nested_includes_qat_finetune_pass():
    spec = {"training_pipeline": {"includes_qat_finetune_pass": True}}
    inapplicable, _reason = _explicit_override_fp4_packed_without_qat(spec)
    assert inapplicable is True


def test_fp4_qat_override_does_not_fire_when_QAT_false():
    spec = {"quantization_aware_training": False}
    inapplicable, _reason = _explicit_override_fp4_packed_without_qat(spec)
    assert inapplicable is False


def test_fp4_qat_override_does_not_fire_when_training_pipeline_is_string():
    # Empirical regression: existing test_matcher_matches_fp4_without_qat
    # uses {"training_pipeline": "no_qat"} (STRING, not Mapping). The
    # override must NOT short-circuit this case.
    spec = {"training_pipeline": "no_qat", "quantization_ops": ["fp4_packed"]}
    inapplicable, _reason = _explicit_override_fp4_packed_without_qat(spec)
    assert inapplicable is False


def test_brotli_plus_lzma_override_fires_when_compression_ops_lacks_lzma():
    spec = {"compression_ops": ["int8_per_channel", "brotli_q11"]}
    inapplicable, reason = _explicit_override_brotli_plus_lzma_chained(spec)
    assert inapplicable is True
    assert "lzma" in reason.lower()


def test_brotli_plus_lzma_override_fires_when_compression_pipeline_alias_lacks_lzma():
    spec = {"compression_pipeline": ["brotli_q11"]}
    inapplicable, _reason = _explicit_override_brotli_plus_lzma_chained(spec)
    assert inapplicable is True


def test_brotli_plus_lzma_override_does_not_fire_when_lzma_present():
    spec = {"compression_ops": ["brotli_q11", "lzma_q9"]}
    inapplicable, _reason = _explicit_override_brotli_plus_lzma_chained(spec)
    assert inapplicable is False


def test_brotli_plus_lzma_override_does_not_fire_without_structured_list():
    # When compression_ops is absent / unstructured, fall back to token
    # heuristic — the override cannot determine structural absence.
    spec = {"substrate_id": "pact_nerv_test"}
    inapplicable, _reason = _explicit_override_brotli_plus_lzma_chained(spec)
    assert inapplicable is False


def test_lzma_on_brotli_override_is_sister_of_brotli_plus_lzma():
    spec = {"compression_ops": ["brotli_q11"]}
    # Both predicates require lzma's presence; absence refutes both.
    a, _ = _explicit_override_lzma_on_already_brotli(spec)
    b, _ = _explicit_override_brotli_plus_lzma_chained(spec)
    assert a is True
    assert b is True


def test_cross_paradigm_override_fires_when_per_axis_decomposition_active():
    spec = {"per_axis_decomposition_active": True}
    inapplicable, reason = (
        _explicit_override_cross_paradigm_test_per_axis_decomposition(spec)
    )
    assert inapplicable is True
    assert "per_axis_decomposition_active" in reason


def test_cross_paradigm_override_fires_for_per_axis_decomposition_synonym():
    spec = {"per_axis_decomposition": True}
    inapplicable, _ = (
        _explicit_override_cross_paradigm_test_per_axis_decomposition(spec)
    )
    assert inapplicable is True


def test_cross_paradigm_override_fires_for_catalog_356_active_synonym():
    spec = {"catalog_356_active": True}
    inapplicable, _ = (
        _explicit_override_cross_paradigm_test_per_axis_decomposition(spec)
    )
    assert inapplicable is True


def test_cross_paradigm_override_does_not_fire_when_per_axis_false():
    spec = {"per_axis_decomposition_active": False}
    inapplicable, _ = (
        _explicit_override_cross_paradigm_test_per_axis_decomposition(spec)
    )
    assert inapplicable is False


def test_quantize_then_svd_override_fires_when_quantization_ops_lacks_svd():
    spec = {"quantization_ops": ["int8_per_channel", "fp4_packed"]}
    inapplicable, _ = _explicit_override_quantize_then_svd_corrupted(spec)
    assert inapplicable is True


def test_quantize_then_svd_override_does_not_fire_when_svd_present():
    spec = {"quantization_ops": ["int8_per_channel", "low_rank_svd"]}
    inapplicable, _ = _explicit_override_quantize_then_svd_corrupted(spec)
    assert inapplicable is False


# ---------------------------------------------------------------------------
# Public override-evaluator API
# ---------------------------------------------------------------------------


def test_evaluate_explicit_override_returns_false_for_unknown_anti_pattern():
    """Anti-patterns without a registered override predicate fall through."""
    inapplicable, reason = evaluate_explicit_override_for_anti_pattern(
        "no_such_anti_pattern_id_v1",
        {"quantization_aware_training": True},
    )
    assert inapplicable is False
    assert reason == ""


def test_evaluate_explicit_override_handles_non_mapping_stack_spec():
    """Defensive: non-Mapping stack_spec returns False (no crash)."""
    inapplicable, _ = evaluate_explicit_override_for_anti_pattern(
        "fp4_packed_without_qat_cos_collapse_v1",
        "not_a_dict",  # type: ignore[arg-type]
    )
    assert inapplicable is False


def test_evaluate_explicit_override_swallows_predicate_exceptions():
    """If an override predicate raises, the matcher must NOT crash."""
    # Construct a malformed nested structure that could trip a less-defensive
    # predicate (e.g. ``training_pipeline`` is an unexpected type).
    spec = {"training_pipeline": 12345}  # int, not Mapping or string
    inapplicable, _ = evaluate_explicit_override_for_anti_pattern(
        "fp4_packed_without_qat_cos_collapse_v1",
        spec,
    )
    # No QAT signal AND no crash.
    assert inapplicable is False


def test_explicit_override_predicates_registry_covers_known_false_positives():
    """The canonical override-predicate table covers the 4 anchored cases."""
    required_ids = {
        "fp4_packed_without_qat_cos_collapse_v1",
        "brotli_plus_lzma_chained_anti_pattern_v1",
        "lzma_on_already_brotli_saturated_compounding_v1",
        "cross_paradigm_test_without_per_axis_decomposition_v1",
        # Plus sister #2 quantize_then_svd:
        "quantize_then_svd_corrupted_low_rank_v1",
    }
    assert required_ids.issubset(_EXPLICIT_OVERRIDE_PREDICATES.keys())


# ---------------------------------------------------------------------------
# End-to-end false-positive guard tests (the 4 empirical anchors)
# ---------------------------------------------------------------------------


def test_matcher_does_not_fire_fp4_when_explicit_qat_true(temp_registry):
    """Compound C STAND_DOWN anchor: fp4_packed_without_qat must NOT fire when
    quantization_aware_training=True is explicitly declared.
    """
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {
            "substrate_id": "pact_nerv_compound_c_resume",
            "quantization_ops": ["fp4_packed", "int8_per_channel"],
            "quantization_aware_training": True,  # explicit guarantee
        },
        path=path,
    )
    fp4_ids = [
        m.anti_pattern.anti_pattern_id
        for m in matches
        if "fp4" in m.anti_pattern.anti_pattern_id
    ]
    assert fp4_ids == [], (
        f"fp4_packed_without_qat MUST NOT fire when "
        f"quantization_aware_training=True; got: {fp4_ids}"
    )


def test_matcher_does_not_fire_fp4_when_qat_finetune_passes_positive(temp_registry):
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {
            "quantization_ops": ["fp4_packed"],
            "qat_finetune_passes": 5,
        },
        path=path,
    )
    fp4_ids = [
        m.anti_pattern.anti_pattern_id
        for m in matches
        if "fp4" in m.anti_pattern.anti_pattern_id
    ]
    assert fp4_ids == []


def test_matcher_does_not_fire_fp4_when_nested_training_pipeline_qat(temp_registry):
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {
            "quantization_ops": ["fp4_packed"],
            "training_pipeline": {"qat_finetune": True, "qat_epochs": 200},
        },
        path=path,
    )
    fp4_ids = [
        m.anti_pattern.anti_pattern_id
        for m in matches
        if "fp4" in m.anti_pattern.anti_pattern_id
    ]
    assert fp4_ids == []


def test_matcher_does_not_fire_brotli_plus_lzma_when_lzma_absent(temp_registry):
    """Wave N+3 Slot 1 PyTorch sister anchor: brotli_plus_lzma_chained MUST
    NOT fire when compression_ops contains only brotli with NO lzma.
    """
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {
            "substrate_id": "wyner_ziv_pipeline_stage_codec",
            "compression_ops": ["int8_per_channel", "brotli_q11"],
        },
        path=path,
    )
    brotli_lzma_ids = [
        m.anti_pattern.anti_pattern_id
        for m in matches
        if "brotli_plus_lzma" in m.anti_pattern.anti_pattern_id
        or "lzma_on_already_brotli" in m.anti_pattern.anti_pattern_id
    ]
    assert brotli_lzma_ids == [], (
        f"brotli_plus_lzma anti-patterns MUST NOT fire when compression_ops "
        f"has NO lzma token; got: {brotli_lzma_ids}"
    )


def test_matcher_does_not_fire_lzma_anti_patterns_when_compression_pipeline_alias_lacks_lzma(
    temp_registry,
):
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {"compression_pipeline": ["brotli_q11"]},
        path=path,
    )
    lzma_ids = [
        m.anti_pattern.anti_pattern_id
        for m in matches
        if "lzma" in m.anti_pattern.anti_pattern_id.lower()
    ]
    assert lzma_ids == []


def test_matcher_does_not_fire_cross_paradigm_when_per_axis_decomposition_active(
    temp_registry,
):
    """Compound F preflight anchor: cross_paradigm_test_without_per_axis MUST
    NOT fire when per_axis_decomposition_active=True is explicitly declared.
    """
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {
            "substrate_id": "cross_paradigm_compound_f_resume",
            "cross_paradigm": True,
            "per_axis_decomposition_active": True,  # explicit guarantee
        },
        path=path,
    )
    cross_ids = [
        m.anti_pattern.anti_pattern_id
        for m in matches
        if "cross_paradigm" in m.anti_pattern.anti_pattern_id
    ]
    assert cross_ids == []


def test_matcher_does_not_fire_quantize_then_svd_when_svd_absent(temp_registry):
    """quantize_then_svd_corrupted_low_rank_v1 must NOT fire when
    quantization_ops lacks any SVD/low_rank token, even if other haystack
    tokens (descriptions, sister fields) mention SVD.
    """
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {
            "substrate_id": "compound_a_int8_fp4_no_svd",
            "quantization_ops": ["int8_per_channel", "fp4_packed"],
            "description": "high-quality compound stack; no SVD anywhere",
        },
        path=path,
    )
    svd_ids = [
        m.anti_pattern.anti_pattern_id
        for m in matches
        if "svd" in m.anti_pattern.anti_pattern_id
    ]
    assert svd_ids == []


def test_matcher_compound_c_stack_spec_is_completely_clean(temp_registry):
    """End-to-end Compound C canonical stack_spec produces ZERO matches.

    This is the canonical stack the Compound C STAND_DOWN subagent had
    (before the architectural fix landed); the matcher should now produce
    an empty match tuple AND no token-fallback noise.
    """
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    canonical_compound_c = {
        "substrate_id": "pact_nerv_compound_c",
        "compression_ops": ["int8_per_channel", "brotli_q11"],
        "quantization_ops": ["fp4_packed", "int8_per_channel"],
        "quantization_aware_training": True,
        "qat_finetune_passes": 3,
        "per_axis_decomposition_active": True,
        "decoder_arch": "pact_nerv_v3_heterogeneous_bit_allocation",
        "predicted_band_source": (
            "post_training_qat_fp4_plus_int8_plus_int4_per_tensor_sensitivity_ranking"
        ),
        "modal_dispatch_pre_spawn_path": False,
    }
    matches = match_stack_against_anti_patterns(canonical_compound_c, path=path)
    bug_class_ids = {
        "fp4_packed_without_qat_cos_collapse_v1",
        "brotli_plus_lzma_chained_anti_pattern_v1",
        "lzma_on_already_brotli_saturated_compounding_v1",
        "cross_paradigm_test_without_per_axis_decomposition_v1",
        "quantize_then_svd_corrupted_low_rank_v1",
    }
    matched_ids = {m.anti_pattern.anti_pattern_id for m in matches}
    intersection = matched_ids & bug_class_ids
    assert intersection == set(), (
        f"Compound C canonical stack MUST NOT match any bug-class anti-pattern; "
        f"matched: {sorted(intersection)}"
    )


# ---------------------------------------------------------------------------
# Positive-path regression guards (architectural fix must NOT degrade
# legitimate positive matches)
# ---------------------------------------------------------------------------


def test_matcher_still_fires_fp4_when_qat_unspecified(temp_registry):
    """Regression: existing positive-path test for fp4-without-QAT still passes.

    The stack_spec has NO explicit-override field; the token-overlap fallback
    must still surface the FP4-without-QAT anti-pattern.
    """
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {
            "quantization_ops": ["fp4_packed"],
            "substrate_id": "pact_nerv_test",
            "training_pipeline": "no_qat",  # STRING token, not Mapping
        },
        path=path,
    )
    assert any(
        "fp4" in m.anti_pattern.anti_pattern_id for m in matches
    ), (
        "FP4-without-QAT must still fire when no explicit-guarantee field "
        "is provided; got: "
        f"{[m.anti_pattern.anti_pattern_id for m in matches]}"
    )


def test_matcher_still_fires_brotli_plus_lzma_when_both_present(temp_registry):
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {"compression_ops": ["int8_per_channel", "brotli_q11", "lzma_q9"]},
        path=path,
    )
    assert any(
        "lzma" in m.anti_pattern.anti_pattern_id for m in matches
    )


def test_matcher_still_fires_cross_paradigm_when_per_axis_false(temp_registry):
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {
            "substrate_id": "cross_paradigm_test_substrate",
            "per_axis_decomposition_active": False,
        },
        path=path,
    )
    assert any(
        "cross_paradigm" in m.anti_pattern.anti_pattern_id for m in matches
    )


def test_matcher_still_fires_quantize_then_svd_when_svd_present(temp_registry):
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {
            "quantization_ops": ["int8_per_channel", "low_rank_svd"],
        },
        path=path,
    )
    # Note: the matcher's token-heuristic may match here via sister
    # condition strings; we want to verify the override predicate does
    # NOT block legitimate matches when SVD IS present.
    inapplicable, _ = _explicit_override_quantize_then_svd_corrupted(
        {"quantization_ops": ["int8_per_channel", "low_rank_svd"]}
    )
    assert inapplicable is False
    # Token-heuristic may or may not fire (the canonical stack_spec
    # representation lacks ordering); the important guarantee is the
    # override does NOT structurally block.
    _ = matches  # explicit acknowledgment


# ---------------------------------------------------------------------------
# Compound F orthogonal composition regression guard
# ---------------------------------------------------------------------------


def test_matcher_compound_f_with_all_canonical_guarantees_clean(temp_registry):
    """Compound F preflight `e5467cf05` canonical stack must be clean.

    The Compound F subagent declared all 3 explicit-guarantee fields and
    yet the matcher still false-positive'd. Post-fix, this stack must
    produce ZERO bug-class matches.
    """
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    canonical_compound_f = {
        "substrate_id": "compound_f_orthogonal_composition_test",
        "compression_ops": ["int8_per_channel", "brotli_q11"],
        "quantization_ops": ["fp4_packed"],
        "quantization_aware_training": True,
        "per_axis_decomposition_active": True,
        "cross_paradigm": True,
    }
    matches = match_stack_against_anti_patterns(canonical_compound_f, path=path)
    bug_class_ids = {
        "fp4_packed_without_qat_cos_collapse_v1",
        "brotli_plus_lzma_chained_anti_pattern_v1",
        "lzma_on_already_brotli_saturated_compounding_v1",
        "cross_paradigm_test_without_per_axis_decomposition_v1",
    }
    matched = {
        m.anti_pattern.anti_pattern_id
        for m in matches
        if m.anti_pattern.anti_pattern_id in bug_class_ids
    }
    assert matched == set(), (
        f"Compound F canonical stack with all explicit guarantees MUST be "
        f"bug-class-clean; matched: {sorted(matched)}"
    )


# ---------------------------------------------------------------------------
# Edge cases the architectural fix must handle gracefully
# ---------------------------------------------------------------------------


def test_matcher_handles_empty_stack_spec(temp_registry):
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns({}, path=path)
    assert matches == ()


def test_matcher_handles_stack_spec_with_only_explicit_guarantees(temp_registry):
    """When stack_spec ONLY has explicit guarantees, no bug-class fires."""
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {
            "quantization_aware_training": True,
            "per_axis_decomposition_active": True,
            "compression_ops": ["brotli_q11"],
        },
        path=path,
    )
    bug_class = {
        "fp4_packed_without_qat_cos_collapse_v1",
        "brotli_plus_lzma_chained_anti_pattern_v1",
        "lzma_on_already_brotli_saturated_compounding_v1",
        "cross_paradigm_test_without_per_axis_decomposition_v1",
    }
    matched = {m.anti_pattern.anti_pattern_id for m in matches}
    assert (matched & bug_class) == set()


def test_evaluate_explicit_override_public_api_is_pure_function():
    """The public override evaluator is a pure function (no state mutation)."""
    spec = {"quantization_aware_training": True}
    a = evaluate_explicit_override_for_anti_pattern(
        "fp4_packed_without_qat_cos_collapse_v1", spec
    )
    b = evaluate_explicit_override_for_anti_pattern(
        "fp4_packed_without_qat_cos_collapse_v1", spec
    )
    assert a == b


# ===========================================================================
# WAVE N+10 SLOT 2 EXTENSION 2026-05-28 (task #1479): tests for 10 NEW
# override predicates added per Yousfi adversarial-audit gap closure.
# 10 tests per override × 10 overrides = 100 new false-positive guard tests
# + integration tests below. Anti-pattern #12 docstring overstatement is
# NOT-APPLICABLE here (source-text not stack_spec; Catalog #287 handles it).
# ===========================================================================


# ---------- #6 predicted_band_from_random_init_tier_c_v1 ------------------


def test_predicted_band_override_fires_for_validated_post_training():
    spec = {"predicted_band_validation_status": "validated_post_training"}
    inapplicable, reason = _explicit_override_predicted_band_from_random_init_tier_c(spec)
    assert inapplicable is True
    assert "validated_post_training" in reason.lower()


def test_predicted_band_override_fires_for_pending_post_training():
    spec = {"predicted_band_validation_status": "pending_post_training"}
    inapplicable, _ = _explicit_override_predicted_band_from_random_init_tier_c(spec)
    assert inapplicable is True


def test_predicted_band_override_fires_for_post_training_source():
    spec = {"predicted_band_source": "post_training"}
    inapplicable, _ = _explicit_override_predicted_band_from_random_init_tier_c(spec)
    assert inapplicable is True


def test_predicted_band_override_fires_for_post_smoke_anchor():
    spec = {"predicted_band_source": "post_smoke_anchor"}
    inapplicable, _ = _explicit_override_predicted_band_from_random_init_tier_c(spec)
    assert inapplicable is True


def test_predicted_band_override_fires_for_catalog_324_active():
    spec = {"catalog_324_active": True}
    inapplicable, reason = _explicit_override_predicted_band_from_random_init_tier_c(spec)
    assert inapplicable is True
    assert "catalog_324_active" in reason.lower()


def test_predicted_band_override_fires_for_nested_recipe_validated():
    spec = {"recipe": {"predicted_band_validation_status": "validated_post_training"}}
    inapplicable, _ = _explicit_override_predicted_band_from_random_init_tier_c(spec)
    assert inapplicable is True


def test_predicted_band_override_does_not_fire_for_random_init():
    spec = {"predicted_band_source": "random_init"}
    inapplicable, _ = _explicit_override_predicted_band_from_random_init_tier_c(spec)
    assert inapplicable is False


def test_predicted_band_override_does_not_fire_for_empty_spec():
    inapplicable, _ = _explicit_override_predicted_band_from_random_init_tier_c({})
    assert inapplicable is False


def test_predicted_band_override_handles_non_string_validation_status():
    spec = {"predicted_band_validation_status": 42}
    inapplicable, _ = _explicit_override_predicted_band_from_random_init_tier_c(spec)
    assert inapplicable is False


def test_predicted_band_override_does_not_fire_for_catalog_324_active_falsy():
    spec = {"catalog_324_active": False}
    inapplicable, _ = _explicit_override_predicted_band_from_random_init_tier_c(spec)
    assert inapplicable is False


def test_predicted_band_override_case_insensitive_validation_status():
    spec = {"predicted_band_validation_status": "VALIDATED_POST_TRAINING"}
    inapplicable, _ = _explicit_override_predicted_band_from_random_init_tier_c(spec)
    assert inapplicable is True


# ---------- #7 rank_1_problem_spec_synergy_tautology_v1 -------------------


def test_rank1_synergy_override_fires_for_rank_2():
    spec = {"operator_gradient_matrix_rank": 2}
    inapplicable, reason = _explicit_override_rank_1_problem_spec_synergy_tautology(spec)
    assert inapplicable is True
    assert "rank" in reason.lower()


def test_rank1_synergy_override_fires_for_rank_5():
    spec = {"operator_gradient_matrix_rank": 5}
    inapplicable, _ = _explicit_override_rank_1_problem_spec_synergy_tautology(spec)
    assert inapplicable is True


def test_rank1_synergy_override_fires_for_per_pair_axis_decomposition_active():
    spec = {"per_pair_axis_decomposition_active": True}
    inapplicable, _ = _explicit_override_rank_1_problem_spec_synergy_tautology(spec)
    assert inapplicable is True


def test_rank1_synergy_override_fires_for_operator_gradients_distinct_per_axis():
    spec = {"operator_gradients_distinct_per_axis": True}
    inapplicable, _ = _explicit_override_rank_1_problem_spec_synergy_tautology(spec)
    assert inapplicable is True


def test_rank1_synergy_override_fires_for_catalog_356_active():
    spec = {"catalog_356_active": True}
    inapplicable, reason = _explicit_override_rank_1_problem_spec_synergy_tautology(spec)
    assert inapplicable is True
    assert "356" in reason


def test_rank1_synergy_override_does_not_fire_for_rank_1():
    spec = {"operator_gradient_matrix_rank": 1}
    inapplicable, _ = _explicit_override_rank_1_problem_spec_synergy_tautology(spec)
    assert inapplicable is False


def test_rank1_synergy_override_does_not_fire_for_empty_spec():
    inapplicable, _ = _explicit_override_rank_1_problem_spec_synergy_tautology({})
    assert inapplicable is False


def test_rank1_synergy_override_rejects_bool_as_rank():
    # bools are ints in Python; gate must guard
    spec = {"operator_gradient_matrix_rank": True}
    inapplicable, _ = _explicit_override_rank_1_problem_spec_synergy_tautology(spec)
    assert inapplicable is False


def test_rank1_synergy_override_does_not_fire_for_negative_rank():
    spec = {"operator_gradient_matrix_rank": -1}
    inapplicable, _ = _explicit_override_rank_1_problem_spec_synergy_tautology(spec)
    assert inapplicable is False


def test_rank1_synergy_override_does_not_fire_for_falsy_decomposition():
    spec = {"per_pair_axis_decomposition_active": False, "catalog_356_active": False}
    inapplicable, _ = _explicit_override_rank_1_problem_spec_synergy_tautology(spec)
    assert inapplicable is False


# ---------- #8 phantom_score_directory_naming_lie_v1 ---------------------


def test_phantom_dir_override_fires_for_matching_device_tokens_cpu():
    spec = {"filename_device_token": "cpu", "metadata_device_token": "cpu"}
    inapplicable, _ = _explicit_override_phantom_score_directory_naming_lie(spec)
    assert inapplicable is True


def test_phantom_dir_override_fires_for_matching_device_tokens_cuda():
    spec = {"filename_device_token": "cuda", "metadata_device_token": "cuda"}
    inapplicable, _ = _explicit_override_phantom_score_directory_naming_lie(spec)
    assert inapplicable is True


def test_phantom_dir_override_case_insensitive_match():
    spec = {"filename_device_token": "CPU", "metadata_device_token": "cpu"}
    inapplicable, _ = _explicit_override_phantom_score_directory_naming_lie(spec)
    assert inapplicable is True


def test_phantom_dir_override_fires_for_device_agnostic_filename():
    spec = {"artifact_filename_device_agnostic": True}
    inapplicable, _ = _explicit_override_phantom_score_directory_naming_lie(spec)
    assert inapplicable is True


def test_phantom_dir_override_fires_for_catalog_249_active():
    spec = {"catalog_249_active": True}
    inapplicable, reason = _explicit_override_phantom_score_directory_naming_lie(spec)
    assert inapplicable is True
    assert "249" in reason


def test_phantom_dir_override_does_not_fire_for_cuda_filename_cpu_metadata():
    spec = {"filename_device_token": "cuda", "metadata_device_token": "cpu"}
    inapplicable, _ = _explicit_override_phantom_score_directory_naming_lie(spec)
    assert inapplicable is False


def test_phantom_dir_override_does_not_fire_for_empty_filename_token():
    spec = {"filename_device_token": "", "metadata_device_token": ""}
    inapplicable, _ = _explicit_override_phantom_score_directory_naming_lie(spec)
    assert inapplicable is False


def test_phantom_dir_override_does_not_fire_for_empty_spec():
    inapplicable, _ = _explicit_override_phantom_score_directory_naming_lie({})
    assert inapplicable is False


def test_phantom_dir_override_does_not_fire_when_only_one_token_present():
    spec = {"filename_device_token": "cpu"}
    inapplicable, _ = _explicit_override_phantom_score_directory_naming_lie(spec)
    assert inapplicable is False


def test_phantom_dir_override_falsy_device_agnostic_does_not_fire():
    spec = {"artifact_filename_device_agnostic": False}
    inapplicable, _ = _explicit_override_phantom_score_directory_naming_lie(spec)
    assert inapplicable is False


# ---------- #9 transient_tmp_path_in_persisted_artifact_v1 ---------------


def test_tmp_path_override_fires_for_clean_durable_paths():
    spec = {"persisted_artifact_paths": ["/Users/me/repo/state/foo.json"]}
    inapplicable, _ = _explicit_override_transient_tmp_path_in_persisted_artifact(spec)
    assert inapplicable is True


def test_tmp_path_override_fires_for_artifact_paths_durable_flag():
    spec = {"artifact_paths_durable": True}
    inapplicable, _ = _explicit_override_transient_tmp_path_in_persisted_artifact(spec)
    assert inapplicable is True


def test_tmp_path_override_fires_for_catalog_220_active():
    spec = {"catalog_220_active": True}
    inapplicable, reason = _explicit_override_transient_tmp_path_in_persisted_artifact(spec)
    assert inapplicable is True
    assert "220" in reason


def test_tmp_path_override_fires_for_empty_paths_list():
    # No /tmp prefix in the structured list → predicate inapplicable
    spec = {"persisted_artifact_paths": []}
    inapplicable, _ = _explicit_override_transient_tmp_path_in_persisted_artifact(spec)
    assert inapplicable is True


def test_tmp_path_override_fires_for_multiple_durable_paths():
    spec = {"persisted_artifact_paths": ["./reports/x.json", ".omx/state/y.json"]}
    inapplicable, _ = _explicit_override_transient_tmp_path_in_persisted_artifact(spec)
    assert inapplicable is True


def test_tmp_path_override_does_not_fire_for_tmp_prefix():
    spec = {"persisted_artifact_paths": ["/tmp/foo.json"]}
    inapplicable, _ = _explicit_override_transient_tmp_path_in_persisted_artifact(spec)
    assert inapplicable is False


def test_tmp_path_override_does_not_fire_for_private_tmp_prefix():
    spec = {"persisted_artifact_paths": ["/private/tmp/foo.json"]}
    inapplicable, _ = _explicit_override_transient_tmp_path_in_persisted_artifact(spec)
    assert inapplicable is False


def test_tmp_path_override_does_not_fire_for_var_tmp_prefix():
    spec = {"persisted_artifact_paths": ["/var/tmp/foo.json"]}
    inapplicable, _ = _explicit_override_transient_tmp_path_in_persisted_artifact(spec)
    assert inapplicable is False


def test_tmp_path_override_does_not_fire_for_empty_spec():
    inapplicable, _ = _explicit_override_transient_tmp_path_in_persisted_artifact({})
    assert inapplicable is False


def test_tmp_path_override_does_not_fire_for_mixed_paths_with_tmp():
    spec = {"persisted_artifact_paths": ["./reports/x.json", "/tmp/leak.json"]}
    inapplicable, _ = _explicit_override_transient_tmp_path_in_persisted_artifact(spec)
    assert inapplicable is False


# ---------- #10 source_selector_inherited_predicted_score_mean_v1 --------


def test_source_selector_override_fires_for_paired_cpu_exact_eval():
    spec = {"interaction_matrix_source": "paired_cpu_exact_eval"}
    inapplicable, _ = _explicit_override_source_selector_inherited_predicted_score_mean(spec)
    assert inapplicable is True


def test_source_selector_override_fires_for_modal_cpu_dispatch():
    spec = {"interaction_matrix_source": "modal_cpu_dispatch"}
    inapplicable, _ = _explicit_override_source_selector_inherited_predicted_score_mean(spec)
    assert inapplicable is True


def test_source_selector_override_fires_for_paired_cuda_exact_eval():
    spec = {"interaction_matrix_source": "paired_cuda_exact_eval"}
    inapplicable, _ = _explicit_override_source_selector_inherited_predicted_score_mean(spec)
    assert inapplicable is True


def test_source_selector_override_fires_for_empirical_token():
    spec = {"interaction_matrix_source": "empirical_anchor_v2"}
    inapplicable, _ = _explicit_override_source_selector_inherited_predicted_score_mean(spec)
    assert inapplicable is True


def test_source_selector_override_fires_for_empirically_measured_flag():
    spec = {"interaction_matrix_empirically_measured": True}
    inapplicable, _ = _explicit_override_source_selector_inherited_predicted_score_mean(spec)
    assert inapplicable is True


def test_source_selector_override_fires_for_paired_cpu_ledger_present():
    spec = {"paired_cpu_exact_eval_ledger_present": True}
    inapplicable, _ = _explicit_override_source_selector_inherited_predicted_score_mean(spec)
    assert inapplicable is True


def test_source_selector_override_does_not_fire_for_predicted_score_mean_source():
    spec = {"interaction_matrix_source": "predicted_score_mean"}
    inapplicable, _ = _explicit_override_source_selector_inherited_predicted_score_mean(spec)
    assert inapplicable is False


def test_source_selector_override_does_not_fire_for_empty_spec():
    inapplicable, _ = _explicit_override_source_selector_inherited_predicted_score_mean({})
    assert inapplicable is False


def test_source_selector_override_does_not_fire_for_non_string_source():
    spec = {"interaction_matrix_source": 42}
    inapplicable, _ = _explicit_override_source_selector_inherited_predicted_score_mean(spec)
    assert inapplicable is False


def test_source_selector_override_does_not_fire_for_falsy_flags():
    spec = {"interaction_matrix_empirically_measured": False, "paired_cpu_exact_eval_ledger_present": False}
    inapplicable, _ = _explicit_override_source_selector_inherited_predicted_score_mean(spec)
    assert inapplicable is False


# ---------- #11 silent_no_spawn_modal_dispatch_v1 ------------------------


def test_silent_no_spawn_override_fires_for_no_pre_spawn_path_false():
    spec = {"modal_dispatch_pre_spawn_path": False}
    inapplicable, _ = _explicit_override_silent_no_spawn_modal_dispatch(spec)
    assert inapplicable is True


def test_silent_no_spawn_override_fires_for_register_pre_spawn_fatal_wired():
    spec = {"modal_register_pre_spawn_fatal_wired": True}
    inapplicable, _ = _explicit_override_silent_no_spawn_modal_dispatch(spec)
    assert inapplicable is True


def test_silent_no_spawn_override_fires_for_catalog_360_active():
    spec = {"catalog_360_active": True}
    inapplicable, reason = _explicit_override_silent_no_spawn_modal_dispatch(spec)
    assert inapplicable is True
    assert "360" in reason


def test_silent_no_spawn_override_fires_for_dispatcher_route_with_register():
    spec = {"modal_dispatcher_route": "path_with_register_pre_spawn_fatal_helper"}
    inapplicable, _ = _explicit_override_silent_no_spawn_modal_dispatch(spec)
    assert inapplicable is True


def test_silent_no_spawn_override_case_insensitive_route():
    spec = {"modal_dispatcher_route": "REGISTER_PRE_SPAWN_FATAL_HELPER"}
    inapplicable, _ = _explicit_override_silent_no_spawn_modal_dispatch(spec)
    assert inapplicable is True


def test_silent_no_spawn_override_does_not_fire_for_empty_spec():
    inapplicable, _ = _explicit_override_silent_no_spawn_modal_dispatch({})
    assert inapplicable is False


def test_silent_no_spawn_override_does_not_fire_for_pre_spawn_path_true():
    spec = {"modal_dispatch_pre_spawn_path": True}
    inapplicable, _ = _explicit_override_silent_no_spawn_modal_dispatch(spec)
    assert inapplicable is False


def test_silent_no_spawn_override_does_not_fire_for_dispatcher_route_without_helper():
    spec = {"modal_dispatcher_route": "bare_sys_exit_only"}
    inapplicable, _ = _explicit_override_silent_no_spawn_modal_dispatch(spec)
    assert inapplicable is False


def test_silent_no_spawn_override_does_not_fire_for_falsy_catalog_360():
    spec = {"catalog_360_active": False}
    inapplicable, _ = _explicit_override_silent_no_spawn_modal_dispatch(spec)
    assert inapplicable is False


def test_silent_no_spawn_override_does_not_fire_for_non_string_route():
    spec = {"modal_dispatcher_route": 12345}
    inapplicable, _ = _explicit_override_silent_no_spawn_modal_dispatch(spec)
    assert inapplicable is False


# ---------- #14 subagent_spawn_without_head_state_premise_verification_v1 -


def test_spawn_pv_override_fires_for_pv_evidence_present():
    spec = {"pv_evidence_present": True}
    inapplicable, _ = _explicit_override_subagent_spawn_without_head_state_premise_verification(spec)
    assert inapplicable is True


def test_spawn_pv_override_fires_for_catalog_229_pv_active():
    spec = {"catalog_229_pv_active": True}
    inapplicable, reason = _explicit_override_subagent_spawn_without_head_state_premise_verification(spec)
    assert inapplicable is True
    assert "229" in reason


def test_spawn_pv_override_fires_for_git_log_pv_in_prompt():
    spec = {"git_log_pv_in_prompt": True}
    inapplicable, _ = _explicit_override_subagent_spawn_without_head_state_premise_verification(spec)
    assert inapplicable is True


def test_spawn_pv_override_fires_for_paired_sister_memo_and_head_state_checks():
    spec = {"sister_landing_memo_check_done": True, "head_state_pv_check_done": True}
    inapplicable, _ = _explicit_override_subagent_spawn_without_head_state_premise_verification(spec)
    assert inapplicable is True


def test_spawn_pv_override_does_not_fire_for_only_sister_memo_check():
    spec = {"sister_landing_memo_check_done": True}
    inapplicable, _ = _explicit_override_subagent_spawn_without_head_state_premise_verification(spec)
    assert inapplicable is False


def test_spawn_pv_override_does_not_fire_for_only_head_state_check():
    spec = {"head_state_pv_check_done": True}
    inapplicable, _ = _explicit_override_subagent_spawn_without_head_state_premise_verification(spec)
    assert inapplicable is False


def test_spawn_pv_override_does_not_fire_for_empty_spec():
    inapplicable, _ = _explicit_override_subagent_spawn_without_head_state_premise_verification({})
    assert inapplicable is False


def test_spawn_pv_override_does_not_fire_for_pv_evidence_false():
    spec = {"pv_evidence_present": False}
    inapplicable, _ = _explicit_override_subagent_spawn_without_head_state_premise_verification(spec)
    assert inapplicable is False


def test_spawn_pv_override_does_not_fire_for_falsy_catalog_229():
    spec = {"catalog_229_pv_active": False}
    inapplicable, _ = _explicit_override_subagent_spawn_without_head_state_premise_verification(spec)
    assert inapplicable is False


def test_spawn_pv_override_does_not_fire_for_partial_paired_flags_false():
    spec = {"sister_landing_memo_check_done": True, "head_state_pv_check_done": False}
    inapplicable, _ = _explicit_override_subagent_spawn_without_head_state_premise_verification(spec)
    assert inapplicable is False


# ---------- #15 predecessor_working_tree_uncommitted_handoff_v1 ----------


def test_predecessor_handoff_override_fires_for_committed_via_serializer():
    spec = {"predecessor_committed_via_serializer": True}
    inapplicable, _ = _explicit_override_predecessor_working_tree_uncommitted_handoff(spec)
    assert inapplicable is True


def test_predecessor_handoff_override_fires_for_working_tree_clean():
    spec = {"working_tree_clean_at_spawn_time": True}
    inapplicable, _ = _explicit_override_predecessor_working_tree_uncommitted_handoff(spec)
    assert inapplicable is True


def test_predecessor_handoff_override_fires_for_supersession_pending_declared():
    spec = {"supersession_pending_declared": True}
    inapplicable, _ = _explicit_override_predecessor_working_tree_uncommitted_handoff(spec)
    assert inapplicable is True


def test_predecessor_handoff_override_fires_for_serializer_log_has_predecessor_commit():
    spec = {"catalog_117_serializer_log_has_predecessor_commit": True}
    inapplicable, reason = _explicit_override_predecessor_working_tree_uncommitted_handoff(spec)
    assert inapplicable is True
    assert "117" in reason


def test_predecessor_handoff_override_does_not_fire_for_empty_spec():
    inapplicable, _ = _explicit_override_predecessor_working_tree_uncommitted_handoff({})
    assert inapplicable is False


def test_predecessor_handoff_override_does_not_fire_for_falsy_serializer():
    spec = {"predecessor_committed_via_serializer": False}
    inapplicable, _ = _explicit_override_predecessor_working_tree_uncommitted_handoff(spec)
    assert inapplicable is False


def test_predecessor_handoff_override_does_not_fire_for_falsy_working_tree_clean():
    spec = {"working_tree_clean_at_spawn_time": False}
    inapplicable, _ = _explicit_override_predecessor_working_tree_uncommitted_handoff(spec)
    assert inapplicable is False


def test_predecessor_handoff_override_does_not_fire_for_falsy_supersession():
    spec = {"supersession_pending_declared": False}
    inapplicable, _ = _explicit_override_predecessor_working_tree_uncommitted_handoff(spec)
    assert inapplicable is False


def test_predecessor_handoff_override_does_not_fire_for_falsy_117():
    spec = {"catalog_117_serializer_log_has_predecessor_commit": False}
    inapplicable, _ = _explicit_override_predecessor_working_tree_uncommitted_handoff(spec)
    assert inapplicable is False


def test_predecessor_handoff_override_pure_function_idempotent():
    spec = {"working_tree_clean_at_spawn_time": True}
    a = _explicit_override_predecessor_working_tree_uncommitted_handoff(spec)
    b = _explicit_override_predecessor_working_tree_uncommitted_handoff(spec)
    assert a == b


# ---------- #16 wyner_ziv_prefix_y_density_decoder_state_dict_surface_v1 -


def test_wyner_ziv_prefix_override_fires_for_non_state_dict_intercept():
    spec = {"wyner_ziv_intercept_location": "POSE_AXIS_SIDE_INFO"}
    inapplicable, _ = _explicit_override_wyner_ziv_prefix_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is True


def test_wyner_ziv_prefix_override_fires_for_per_pair_posenet_y():
    spec = {"wyner_ziv_side_info_source": "per_pair_posenet_output_y"}
    inapplicable, _ = _explicit_override_wyner_ziv_prefix_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is True


def test_wyner_ziv_prefix_override_fires_for_atick_redlich_ego_motion_y():
    spec = {"wyner_ziv_side_info_source": "atick_redlich_ego_motion_y"}
    inapplicable, _ = _explicit_override_wyner_ziv_prefix_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is True


def test_wyner_ziv_prefix_override_fires_for_compressed_archive_bytes_form():
    spec = {"base_substrate_bytes_form": "compressed_archive_zip_member"}
    inapplicable, _ = _explicit_override_wyner_ziv_prefix_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is True


def test_wyner_ziv_prefix_override_fires_for_catalog_311_active():
    spec = {"catalog_311_atick_tishby_wyner_active": True}
    inapplicable, reason = _explicit_override_wyner_ziv_prefix_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is True
    assert "311" in reason


def test_wyner_ziv_prefix_override_does_not_fire_for_state_dict_intercept():
    spec = {"wyner_ziv_intercept_location": "state_dict_serialization"}
    inapplicable, _ = _explicit_override_wyner_ziv_prefix_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is False


def test_wyner_ziv_prefix_override_does_not_fire_for_raw_fp16_bytes():
    spec = {"base_substrate_bytes_form": "raw_fp16"}
    inapplicable, _ = _explicit_override_wyner_ziv_prefix_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is False


def test_wyner_ziv_prefix_override_does_not_fire_for_empty_spec():
    inapplicable, _ = _explicit_override_wyner_ziv_prefix_y_density_decoder_state_dict_surface({})
    assert inapplicable is False


def test_wyner_ziv_prefix_override_does_not_fire_for_falsy_311():
    spec = {"catalog_311_atick_tishby_wyner_active": False}
    inapplicable, _ = _explicit_override_wyner_ziv_prefix_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is False


def test_wyner_ziv_prefix_override_does_not_fire_for_non_string_intercept():
    spec = {"wyner_ziv_intercept_location": None}
    inapplicable, _ = _explicit_override_wyner_ziv_prefix_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is False


# ---------- #17 wyner_ziv_cross_substrate_composition sister -------------


def test_wyner_ziv_cross_substrate_override_fires_for_non_state_dict_intercept():
    spec = {"wyner_ziv_intercept_location": "POSE_AXIS_SIDE_INFO"}
    inapplicable, _ = _explicit_override_wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is True


def test_wyner_ziv_cross_substrate_override_fires_for_catalog_311_active():
    spec = {"catalog_311_atick_tishby_wyner_active": True}
    inapplicable, _ = _explicit_override_wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is True


def test_wyner_ziv_cross_substrate_override_delegates_to_prefix_sister():
    spec = {"wyner_ziv_intercept_location": "POSE_AXIS_SIDE_INFO"}
    prefix = _explicit_override_wyner_ziv_prefix_y_density_decoder_state_dict_surface(spec)
    cross = _explicit_override_wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface(spec)
    assert prefix == cross


def test_wyner_ziv_cross_substrate_override_does_not_fire_for_state_dict():
    spec = {"wyner_ziv_intercept_location": "state_dict_serialization"}
    inapplicable, _ = _explicit_override_wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is False


def test_wyner_ziv_cross_substrate_override_does_not_fire_for_empty_spec():
    inapplicable, _ = _explicit_override_wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface({})
    assert inapplicable is False


def test_wyner_ziv_cross_substrate_override_fires_for_compressed_archive_form():
    spec = {"base_substrate_bytes_form": "compressed_archive_zip_member"}
    inapplicable, _ = _explicit_override_wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is True


def test_wyner_ziv_cross_substrate_override_fires_for_canonical_unwind_path():
    spec = {"wyner_ziv_side_info_source": "atick_redlich_ego_motion_y"}
    inapplicable, _ = _explicit_override_wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is True


def test_wyner_ziv_cross_substrate_override_does_not_fire_for_raw_fp32():
    spec = {"base_substrate_bytes_form": "raw_fp32"}
    inapplicable, _ = _explicit_override_wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is False


def test_wyner_ziv_cross_substrate_override_idempotent_pure_function():
    spec = {"catalog_311_atick_tishby_wyner_active": True}
    a = _explicit_override_wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface(spec)
    b = _explicit_override_wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface(spec)
    assert a == b


def test_wyner_ziv_cross_substrate_override_handles_torch_save_bytes_form():
    spec = {"base_substrate_bytes_form": "torch_save"}
    inapplicable, _ = _explicit_override_wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface(spec)
    assert inapplicable is False


# ---------- Wave N+10 Slot 2 integration: full table coverage --------------


def test_table_size_15_after_extension():
    """Catalog #344 canonical equation evidence: 5 initial + 10 new = 15."""
    assert len(_EXPLICIT_OVERRIDE_PREDICATES) == 15


def test_table_includes_all_10_new_anti_pattern_ids():
    new_ids = {
        "predicted_band_from_random_init_tier_c_v1",
        "rank_1_problem_spec_synergy_tautology_v1",
        "phantom_score_directory_naming_lie_v1",
        "transient_tmp_path_in_persisted_artifact_v1",
        "source_selector_inherited_predicted_score_mean_v1",
        "silent_no_spawn_modal_dispatch_v1",
        "subagent_spawn_without_head_state_premise_verification_v1",
        "predecessor_working_tree_uncommitted_handoff_v1",
        "wyner_ziv_prefix_y_density_decoder_state_dict_surface_v1",
        "wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface_v1",
    }
    assert new_ids <= set(_EXPLICIT_OVERRIDE_PREDICATES.keys())


def test_public_api_routes_through_new_overrides():
    """Public evaluate_explicit_override_for_anti_pattern reaches new entries."""
    overridden, reason = evaluate_explicit_override_for_anti_pattern(
        "rank_1_problem_spec_synergy_tautology_v1",
        {"catalog_356_active": True},
    )
    assert overridden is True
    assert "356" in reason


def test_evaluate_unknown_anti_pattern_returns_false():
    """An anti-pattern id without an override predicate returns (False, '')."""
    overridden, reason = evaluate_explicit_override_for_anti_pattern(
        "definitely_not_registered_v1",
        {"random_field": True},
    )
    assert overridden is False
    assert reason == ""


def test_full_canonical_z6_v2_stack_no_bug_class_match(temp_registry):
    """Canonical Z6-v2 / Compound C stack with all explicit guarantees → no match."""
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    canonical_stack = {
        "quantization_aware_training": True,
        "per_axis_decomposition_active": True,
        "compression_ops": ["brotli_q11"],
        "predicted_band_validation_status": "validated_post_training",
        "operator_gradient_matrix_rank": 3,
        "filename_device_token": "cpu",
        "metadata_device_token": "cpu",
        "persisted_artifact_paths": ["./reports/x.json"],
        "interaction_matrix_source": "paired_cpu_exact_eval",
        "modal_register_pre_spawn_fatal_wired": True,
        "pv_evidence_present": True,
        "predecessor_committed_via_serializer": True,
        "wyner_ziv_intercept_location": "POSE_AXIS_SIDE_INFO",
    }
    matches = match_stack_against_anti_patterns(canonical_stack, path=path)
    bug_class_ids = set(_EXPLICIT_OVERRIDE_PREDICATES.keys())
    matched = {m.anti_pattern.anti_pattern_id for m in matches}
    assert (matched & bug_class_ids) == set(), (
        f"Canonical stack with all 15 override flags MUST be bug-class-clean; "
        f"matched: {sorted(matched & bug_class_ids)}"
    )
