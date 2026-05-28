# SPDX-License-Identifier: MIT
"""Pattern matcher — given a proposed stack_spec, return matching anti-patterns.

THE critical compounding-consumer API. The cathedral autopilot ranker (via
the auto-discovered ``anti_pattern_lookup_consumer``) calls
:func:`match_stack_against_anti_patterns` on every candidate to surface
applicable canonical anti-patterns + their canonical_unwind_path
recommendations.

Per the canonical design memo §"Mathematical compounding identity":

    NextCycleAttackDirection = argmax_axis (
        PredictedΔS_axis_i × λ_axis_i_tight_from_Dykstra
    ) subject to (
        NOT any (proposed_stack matches AntiPattern_j AND not waived)
    )

This module implements the LEFT side of the constraint — the matcher that
determines whether a proposed stack matches any registered anti-pattern.
Slot 1's Dykstra Pareto polytope solver (in flight) will consume these
matches as ACTIVE polytope constraints in Wave N+2 Layer 5 integration.

Per CLAUDE.md "Beauty, simplicity, and developer experience": the
``stack_spec`` is an open Mapping[str, Any] so consumers can populate as
much or as little detail as they have. The matcher uses best-effort
substring + structural matching against each anti-pattern's
``recurrence_conditions`` + ``forbidden_pattern_predicate``; matched
results are ordered by severity (CRITICAL first; LOW last) so the worst
applicable anti-pattern is surfaced first.

Per Catalog #287/#323/#341: this is an OBSERVABILITY-ONLY surface. The
matcher does NOT promote / dispatch / mutate state. Slot 1's Pareto
solver consumes the match results as constraints; the operator decides
whether to apply the canonical unwind path OR override with a waiver.

ARCHITECTURAL FIX 2026-05-28 (Wave N+3 Slot 2): false-positive matches
fired empirically 3× this session when stack_spec carried explicit
override flags or contradicting compression_ops (Compound C STAND_DOWN
`e61ea93b0`, Wave N+3 Slot 1 PyTorch sister `4c1daf186`, Compound F
preflight `e5467cf05`). Pre-fix matcher relied on token-overlap
heuristics that fired regardless of explicit-guarantee fields like
``quantization_aware_training=True`` or absence of ``lzma`` in
``compression_ops``. The fix introduces a per-anti-pattern OVERRIDE
TABLE that names structural guarantees ("if stack_spec[K]==V then
predicate IS NOT MET — refuse match") + structural contradictions
("if stack_spec[K] is a list and 'lzma' not in it, then 'brotli+lzma
chained' predicate IS NOT MET"). The table is decoupled from the
``AntiPattern`` dataclass so JSONL persisted state stays
back-compatible and downstream consumers see only the cleaner
``match_stack_against_anti_patterns`` return value.

WAVE N+10 SLOT 2 EXTENSION 2026-05-28 (task #1479): per Yousfi adversarial
audit gap surfaced earlier this session, the override table is extended
from 5 entries to 15 entries, closing the override-predicate coverage
gap across initial 12 + Wave N+7/N+9 additions #13/#14/#15/#16 (10 new
entries; anti-pattern #12 docstring overstatement is NOT-APPLICABLE here
because its forbidden state lives in source-text content, not stack_spec
— per-source ``# DOCSTRING_PERCENT_CLAIM_OK`` waiver per Catalog #287
sister discipline handles that surface separately). Each new predicate
encodes the canonical structural-guarantee tokens that downstream
operator-authorize recipes + cathedral autopilot ranker can declare to
prove the forbidden predicate is structurally inapplicable.

Sister of:
  * ``tac.canonical_anti_patterns.registry`` (the registered anti-patterns)
  * ``tac.canonical_anti_patterns.builtins`` (initial 12 anti-patterns)
  * ``src/tac/cathedral_consumers/anti_pattern_lookup_consumer/`` (consumer)
  * Slot 1 Dykstra Pareto polytope solver (Wave N+2 integration target)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tac.canonical_anti_patterns.anti_pattern import (
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    AntiPattern,
)
from tac.canonical_anti_patterns.registry import query_anti_patterns


# Canonical severity ranking for matched-result ordering.
# CRITICAL appears first (worst-applicable-dominates per design memo
# MAX-aggregation identity); LOW appears last.
_SEVERITY_RANK = {
    SEVERITY_CRITICAL: 0,
    SEVERITY_HIGH: 1,
    SEVERITY_MEDIUM: 2,
    SEVERITY_LOW: 3,
}


@dataclass(frozen=True)
class AntiPatternMatch:
    """One matched anti-pattern with the triggering recurrence condition.

    Fields:
        anti_pattern: the canonical AntiPattern that matched.
        recurrence_condition_triggered: which of the anti-pattern's
            recurrence_conditions surfaced in the stack_spec (free-form
            string for operator-readable audit).
        canonical_unwind_path_recommended: the anti-pattern's
            canonical_unwind_path (passed through; surfaced for ease of
            consumer routing).
        match_confidence: float in [0, 1]; rough confidence in the match.
            1.0 = exact structural match (e.g. anti-pattern explicitly names
            a token present in stack_spec). 0.5 = partial substring match.
            Below 0.3 = treated as advisory by the consumer.
    """

    anti_pattern: AntiPattern
    recurrence_condition_triggered: str
    canonical_unwind_path_recommended: str
    match_confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.anti_pattern, AntiPattern):
            raise ValueError(
                "AntiPatternMatch.anti_pattern must be AntiPattern, got "
                f"{type(self.anti_pattern).__name__}"
            )
        if not isinstance(self.recurrence_condition_triggered, str):
            raise ValueError("recurrence_condition_triggered must be a string")
        if not isinstance(self.canonical_unwind_path_recommended, str):
            raise ValueError(
                "canonical_unwind_path_recommended must be a string"
            )
        if not isinstance(self.match_confidence, (int, float)):
            raise ValueError("match_confidence must be numeric")
        if self.match_confidence < 0 or self.match_confidence > 1:
            raise ValueError(
                f"match_confidence={self.match_confidence} must be in [0, 1]"
            )


@dataclass(frozen=True)
class ValidationResult:
    """Result from ``validate_compound_stack_order``.

    Fields:
        is_valid: True iff the operation order honors canonical
            anti-patterns (no ORDER-violations detected).
        violations: tuple of human-readable strings describing each
            detected ORDER-violation.
        suggested_canonical_order: canonical ordering of the input ops
            (if the matcher can derive one) OR empty string.
    """

    is_valid: bool
    violations: tuple[str, ...]
    suggested_canonical_order: str

    def __post_init__(self) -> None:
        if not isinstance(self.is_valid, bool):
            raise ValueError("is_valid must be a bool")
        if not isinstance(self.violations, tuple):
            raise ValueError("violations must be a tuple")
        for i, v in enumerate(self.violations):
            if not isinstance(v, str):
                raise ValueError(f"violations[{i}] must be a string")
        if not isinstance(self.suggested_canonical_order, str):
            raise ValueError("suggested_canonical_order must be a string")


# ---------------------------------------------------------------------------
# ARCHITECTURAL FIX: per-anti-pattern explicit-override predicate table.
# ---------------------------------------------------------------------------
#
# Each entry maps a canonical ``anti_pattern_id`` to a callable
# ``(stack_spec) -> (is_structurally_inapplicable, human_readable_reason)``.
# When ``is_structurally_inapplicable`` is True, the matcher REFUSES to match
# the anti-pattern regardless of any token-overlap heuristic — the stack_spec
# explicitly proves the forbidden predicate IS NOT MET.
#
# The table is intentionally decoupled from the ``AntiPattern`` dataclass:
#   1. JSONL persisted state stays back-compatible (no schema bump).
#   2. The override knowledge is matcher-engineering (how to interpret
#      stack_spec fields), not anti-pattern-engineering (what the forbidden
#      pattern IS).
#   3. New anti-patterns landing in ``builtins.py`` can co-land a sister entry
#      here as their canonical override predicate; the matcher inherits the
#      protection structurally.
#
# Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
# this fix STRUCTURALLY extincts the false-positive bug class at the matcher
# surface; the symptom-only filter at
# ``tac.substrates.pact_nerv_selector_v3.heterogeneous_bit_allocation.
# assert_no_critical_anti_pattern_matches`` is now redundant for the
# fp4_packed_without_qat case (kept as defense-in-depth).
#
# CASES (each backed by an empirical false-positive anchor):
#
# (a) fp4_packed_without_qat_cos_collapse_v1 — anchor: Compound C STAND_DOWN
#     `e61ea93b0`. Predicate: NOT training_pipeline.includes_qat_finetune_pass.
#     Explicit guarantee: stack_spec["quantization_aware_training"] is True
#     OR stack_spec["qat_finetune_passes"] >= 1 OR
#     stack_spec["training_pipeline"]["qat_finetune"] truthy.
#
# (b) brotli_plus_lzma_chained_anti_pattern_v1 — anchor: Wave N+3 Slot 1
#     `4c1daf186`. Predicate: compression_pipeline contains BOTH brotli AND
#     lzma. Explicit contradiction: stack_spec["compression_ops"] is a list
#     AND no token contains "lzma".
#
# (c) lzma_on_already_brotli_saturated_compounding_v1 — sister of (b);
#     same explicit-contradiction predicate (absence of lzma).
#
# (d) cross_paradigm_test_without_per_axis_decomposition_v1 — anchor: Compound
#     F preflight `e5467cf05`. Predicate: cross_paradigm=True AND
#     per_axis_decomposition_active=False. Explicit guarantee:
#     stack_spec["per_axis_decomposition_active"] is True.
#
_OverridePredicate = Callable[[Mapping[str, Any]], tuple[bool, str]]


def _iter_list_like(value: Any) -> tuple[Any, ...]:
    """Return a tuple view of list/tuple values; empty tuple otherwise."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return ()


def _stack_spec_contains_token_in_field(
    stack_spec: Mapping[str, Any],
    field: str,
    token: str,
) -> bool:
    """Return True iff stack_spec[field] is a list-like containing `token`.

    Per-element matching is case-insensitive substring on the str()
    representation of each element. Used by explicit-contradiction
    predicates to determine whether a forbidden token is genuinely
    present in a structured field (NOT just appearing in haystack via
    some sister substring).
    """
    value = stack_spec.get(field)
    if not isinstance(value, (list, tuple)):
        return False
    token_lower = token.lower()
    for element in value:
        if isinstance(element, str) and token_lower in element.lower():
            return True
    return False


def _explicit_override_fp4_packed_without_qat(
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Refuse fp4_packed_without_qat_cos_collapse_v1 match when QAT proven.

    The forbidden predicate is ``NOT training_pipeline.includes_qat_finetune_pass``.
    The match is STRUCTURALLY INAPPLICABLE when the stack_spec explicitly
    declares QAT is active via any of:

      * ``stack_spec["quantization_aware_training"]`` is True
      * ``stack_spec["qat_finetune_passes"]`` is int >= 1
      * ``stack_spec["training_pipeline"]["qat_finetune"]`` is truthy
      * ``stack_spec["training_pipeline"]["includes_qat_finetune_pass"]`` is True
      * ``stack_spec["qat_enabled"]`` is True
    """
    if stack_spec.get("quantization_aware_training") is True:
        return True, (
            "explicit override: stack_spec['quantization_aware_training']=True "
            "structurally satisfies the predicate "
            "'training_pipeline.includes_qat_finetune_pass'"
        )
    if stack_spec.get("qat_enabled") is True:
        return True, (
            "explicit override: stack_spec['qat_enabled']=True structurally "
            "satisfies the QAT-pipeline predicate"
        )
    passes = stack_spec.get("qat_finetune_passes")
    if isinstance(passes, int) and not isinstance(passes, bool) and passes >= 1:
        return True, (
            f"explicit override: stack_spec['qat_finetune_passes']={passes} >= 1 "
            "structurally satisfies the QAT-pipeline predicate"
        )
    training_pipeline = stack_spec.get("training_pipeline")
    if isinstance(training_pipeline, Mapping):
        if training_pipeline.get("qat_finetune") in (True, 1):
            return True, (
                "explicit override: stack_spec['training_pipeline']"
                "['qat_finetune'] is truthy structurally satisfies the "
                "QAT-pipeline predicate"
            )
        if training_pipeline.get("includes_qat_finetune_pass") is True:
            return True, (
                "explicit override: stack_spec['training_pipeline']"
                "['includes_qat_finetune_pass']=True structurally satisfies "
                "the QAT-pipeline predicate"
            )
    return False, ""


def _explicit_override_brotli_plus_lzma_chained(
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Refuse brotli_plus_lzma_chained match when compression_ops lacks lzma.

    The forbidden predicate is ``compression_pipeline contains brotli AND
    compression_pipeline contains lzma``. The match is STRUCTURALLY
    INAPPLICABLE when the stack_spec declares a structured
    ``compression_ops`` (or sister synonym ``compression_pipeline``)
    list AND no element contains the lzma token. This catches the
    Wave N+3 Slot 1 false positive where the proposed stack was
    ``["brotli_q11"]`` (no lzma anywhere) yet the matcher's
    token-fallback heuristic fired due to sister-pattern haystack
    overlap.
    """
    for field in ("compression_ops", "compression_pipeline", "entropy_coders"):
        value = stack_spec.get(field)
        if isinstance(value, (list, tuple)):
            has_lzma = any(
                isinstance(e, str) and "lzma" in e.lower() for e in value
            )
            if not has_lzma:
                return True, (
                    f"explicit override: stack_spec[{field!r}]={list(value)!r} "
                    "is a structured compression-pipeline list with NO lzma "
                    "token; the forbidden 'brotli+lzma chained' predicate is "
                    "structurally not met"
                )
    return False, ""


def _explicit_override_lzma_on_already_brotli(
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Refuse lzma_on_already_brotli_saturated match when compression_ops lacks lzma.

    Sister of (b): the forbidden predicate is ``LZMA AFTER brotli``.
    If structured compression_ops lacks lzma entirely, the predicate is
    structurally not met.
    """
    # Reuse the brotli+lzma override — both predicates require lzma's
    # presence; absence structurally refutes both.
    return _explicit_override_brotli_plus_lzma_chained(stack_spec)


def _explicit_override_cross_paradigm_test_per_axis_decomposition(
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Refuse cross_paradigm_test_without_per_axis_decomposition when active.

    The forbidden predicate is ``cross_paradigm=True AND
    per_axis_decomposition_active=False``. The match is STRUCTURALLY
    INAPPLICABLE when the stack_spec explicitly declares per-axis
    decomposition active via any of:

      * ``stack_spec["per_axis_decomposition_active"]`` is True
      * ``stack_spec["per_axis_decomposition"]`` is True
      * ``stack_spec["catalog_356_active"]`` is True
    """
    if stack_spec.get("per_axis_decomposition_active") is True:
        return True, (
            "explicit override: stack_spec['per_axis_decomposition_active']="
            "True structurally satisfies Catalog #356 per-axis decomposition"
        )
    if stack_spec.get("per_axis_decomposition") is True:
        return True, (
            "explicit override: stack_spec['per_axis_decomposition']=True "
            "structurally satisfies Catalog #356 per-axis decomposition"
        )
    if stack_spec.get("catalog_356_active") is True:
        return True, (
            "explicit override: stack_spec['catalog_356_active']=True "
            "structurally satisfies the Catalog #356 GAP FIX predicate"
        )
    return False, ""


def _explicit_override_quantize_then_svd_corrupted(
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Refuse quantize_then_svd_corrupted_low_rank when structured ops absent.

    The forbidden predicate is ``compound_stack contains svd AND quantize
    AND svd.order > quantize.order``. The match is STRUCTURALLY
    INAPPLICABLE when ``compression_ops`` / ``quantization_ops`` are
    structured lists AND no SVD/low-rank token follows quantization.
    Sister-pattern protection for stacks that mention SVD in haystack
    but not in execution-order list.
    """
    quantization_ops = stack_spec.get("quantization_ops")
    if isinstance(quantization_ops, (list, tuple)):
        has_quant = any(
            isinstance(e, str)
            and any(
                t in e.lower() for t in ("int8", "int4", "fp4", "quantize", "qat")
            )
            for e in quantization_ops
        )
        has_svd = any(
            isinstance(e, str)
            and any(t in e.lower() for t in ("svd", "low_rank", "low-rank"))
            for e in quantization_ops
        )
        if has_quant and not has_svd:
            return True, (
                "explicit override: stack_spec['quantization_ops'] contains "
                "quantization but NO svd/low_rank token; the forbidden "
                "'quantize THEN svd' predicate is structurally not met"
            )
    return False, ""


# ---------------------------------------------------------------------------
# WAVE N+10 SLOT 2 EXTENSION 2026-05-28 (task #1479): cover anti-patterns
# #6-#11 + #13-#16 (10 new override predicates). Anti-pattern #12 docstring
# overstatement is NOT-APPLICABLE here because the forbidden state lives in
# source-text content, not stack_spec; per-source `# DOCSTRING_PERCENT_CLAIM_OK`
# waiver per Catalog #287 sister discipline handles that surface separately.
# ---------------------------------------------------------------------------


def _explicit_override_predicted_band_from_random_init_tier_c(
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Refuse predicted_band_from_random_init_tier_c_v1 match when validated.

    The forbidden predicate is ``recipe.predicted_band_source IN
    {random_init, pre_training} AND recipe.predicted_band_validation_status
    NOT IN {validated_post_training, pending_post_training}``. The match is
    STRUCTURALLY INAPPLICABLE when the stack_spec explicitly declares
    validation status via any of:

      * ``stack_spec["predicted_band_validation_status"]`` in
        {validated_post_training, pending_post_training}
      * ``stack_spec["predicted_band_source"]`` is post_training /
        post_smoke_anchor / post_full_anchor (not pre-training)
      * ``stack_spec["catalog_324_active"]`` is True
      * ``stack_spec["recipe"]["predicted_band_validation_status"]`` in
        {validated_post_training, pending_post_training}

    Empirical anchor: C6 IBPS 22x miss; Catalog #324 STRICT preflight gate.
    """
    validation_status = stack_spec.get("predicted_band_validation_status")
    if isinstance(validation_status, str) and validation_status.lower() in (
        "validated_post_training",
        "pending_post_training",
    ):
        return True, (
            f"explicit override: stack_spec['predicted_band_validation_status']="
            f"{validation_status!r} structurally satisfies Catalog #324 "
            "post-training validation discipline"
        )
    source = stack_spec.get("predicted_band_source")
    if isinstance(source, str):
        source_lower = source.lower()
        if source_lower in (
            "post_training",
            "post_smoke_anchor",
            "post_full_anchor",
            "validated_post_training",
        ):
            return True, (
                f"explicit override: stack_spec['predicted_band_source']="
                f"{source!r} is post-training (not random-init); the forbidden "
                "pre-training predicate is structurally not met"
            )
    if stack_spec.get("catalog_324_active") is True:
        return True, (
            "explicit override: stack_spec['catalog_324_active']=True "
            "structurally satisfies the Catalog #324 STRICT preflight gate"
        )
    recipe = stack_spec.get("recipe")
    if isinstance(recipe, Mapping):
        nested_status = recipe.get("predicted_band_validation_status")
        if isinstance(nested_status, str) and nested_status.lower() in (
            "validated_post_training",
            "pending_post_training",
        ):
            return True, (
                f"explicit override: stack_spec['recipe']"
                f"['predicted_band_validation_status']={nested_status!r} "
                "structurally satisfies Catalog #324 validation discipline"
            )
    return False, ""


def _explicit_override_rank_1_problem_spec_synergy_tautology(
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Refuse rank_1_problem_spec_synergy_tautology_v1 match when rank>1.

    The forbidden predicate is ``problem_spec.operator_gradient_matrix.rank
    == 1 AND downstream_metric == 'synergy'``. The match is STRUCTURALLY
    INAPPLICABLE when stack_spec explicitly declares the gradient matrix
    rank is > 1 OR axis-decomposition is active:

      * ``stack_spec["operator_gradient_matrix_rank"]`` is int >= 2
      * ``stack_spec["per_pair_axis_decomposition_active"]`` is True
      * ``stack_spec["catalog_356_active"]`` is True (per-axis decomposition
        guarantees distinct per-axis operator gradients)
      * ``stack_spec["operator_gradients_distinct_per_axis"]`` is True

    Empirical anchor: paradox half 2 rigor review commit 21014faa7.
    """
    rank = stack_spec.get("operator_gradient_matrix_rank")
    if (
        isinstance(rank, int)
        and not isinstance(rank, bool)
        and rank >= 2
    ):
        return True, (
            f"explicit override: stack_spec['operator_gradient_matrix_rank']="
            f"{rank} >= 2 structurally refutes the rank-1 tautology predicate"
        )
    if stack_spec.get("per_pair_axis_decomposition_active") is True:
        return True, (
            "explicit override: stack_spec['per_pair_axis_decomposition_active']"
            "=True structurally guarantees per-pair distinct operator gradients"
        )
    if stack_spec.get("operator_gradients_distinct_per_axis") is True:
        return True, (
            "explicit override: stack_spec['operator_gradients_distinct_per_axis']"
            "=True structurally refutes the rank-1 tautology predicate"
        )
    if stack_spec.get("catalog_356_active") is True:
        return True, (
            "explicit override: stack_spec['catalog_356_active']=True per-axis "
            "decomposition guarantees distinct per-axis operator gradients "
            "(rank > 1 structurally)"
        )
    return False, ""


def _explicit_override_phantom_score_directory_naming_lie(
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Refuse phantom_score_directory_naming_lie_v1 match when filename matches metadata.

    The forbidden predicate is
    ``artifact.filename.contains_device_token(cuda|cpu|mps) AND
    artifact.metadata.device != artifact.filename.device_token``.
    The match is STRUCTURALLY INAPPLICABLE when stack_spec explicitly
    declares either:

      * ``stack_spec["filename_device_token"] ==
        stack_spec["metadata_device_token"]`` (filename matches metadata)
      * ``stack_spec["artifact_filename_device_agnostic"]`` is True
      * ``stack_spec["catalog_249_active"]`` is True (STRICT gate active)

    Empirical anchor: Z3 v2 FULL Modal A100; Catalog #249 self-protection.
    """
    filename_token = stack_spec.get("filename_device_token")
    metadata_token = stack_spec.get("metadata_device_token")
    if (
        isinstance(filename_token, str)
        and isinstance(metadata_token, str)
        and filename_token.lower() == metadata_token.lower()
        and filename_token.strip()
    ):
        return True, (
            f"explicit override: stack_spec['filename_device_token']="
            f"{filename_token!r} matches metadata_device_token={metadata_token!r} "
            "structurally refutes the phantom-score-directory predicate"
        )
    if stack_spec.get("artifact_filename_device_agnostic") is True:
        return True, (
            "explicit override: stack_spec['artifact_filename_device_agnostic']"
            "=True (no device token in filename) structurally refutes the "
            "phantom-score-directory predicate"
        )
    if stack_spec.get("catalog_249_active") is True:
        return True, (
            "explicit override: stack_spec['catalog_249_active']=True "
            "structurally satisfies the Catalog #249 STRICT preflight gate"
        )
    return False, ""


def _explicit_override_transient_tmp_path_in_persisted_artifact(
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Refuse transient_tmp_path_in_persisted_artifact_v1 match when no /tmp paths.

    The forbidden predicate is
    ``persisted_artifact.body.contains_path_starting_with('/tmp/') AND
    artifact_category IN {lane_registry, dispatch_claim, build_manifest,
    commit_message}``. The match is STRUCTURALLY INAPPLICABLE when the
    stack_spec explicitly declares the persisted artifact paths are
    durable (no /tmp/ prefix):

      * ``stack_spec["persisted_artifact_paths"]`` is a list AND no
        element starts with '/tmp/' / '/private/tmp/' / '/var/tmp/'
      * ``stack_spec["artifact_paths_durable"]`` is True
      * ``stack_spec["catalog_220_active"]`` is True (sister gate active)

    Empirical anchor: lane_pr106_stacked; Catalog #220 sister discipline.
    """
    forbidden_prefixes = ("/tmp/", "/private/tmp/", "/var/tmp/")
    paths = stack_spec.get("persisted_artifact_paths")
    if isinstance(paths, (list, tuple)):
        any_tmp = any(
            isinstance(p, str)
            and any(p.startswith(pfx) for pfx in forbidden_prefixes)
            for p in paths
        )
        if not any_tmp:
            return True, (
                f"explicit override: stack_spec['persisted_artifact_paths']="
                f"{list(paths)!r} is a structured list with NO /tmp / "
                "/private/tmp / /var/tmp prefix; the forbidden transient-path "
                "predicate is structurally not met"
            )
    if stack_spec.get("artifact_paths_durable") is True:
        return True, (
            "explicit override: stack_spec['artifact_paths_durable']=True "
            "structurally refutes the transient-tmp-path predicate"
        )
    if stack_spec.get("catalog_220_active") is True:
        return True, (
            "explicit override: stack_spec['catalog_220_active']=True "
            "sister-gate-active structurally guarantees no /tmp in persisted "
            "artifact"
        )
    return False, ""


def _explicit_override_source_selector_inherited_predicted_score_mean(
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Refuse source_selector_inherited_predicted_score_mean_v1 when paired-CPU backed.

    The forbidden predicate is ``interaction_matrix.populated_from=
    'predicted_score_mean' AND predicted_score_mean.derivation_path.includes
    (source_selector)``. The match is STRUCTURALLY INAPPLICABLE when the
    stack_spec explicitly declares the interaction matrix source is empirical:

      * ``stack_spec["interaction_matrix_source"]`` in
        {paired_cpu_exact_eval, modal_cpu_dispatch, paired_cuda_exact_eval}
      * ``stack_spec["interaction_matrix_empirically_measured"]`` is True
      * ``stack_spec["paired_cpu_exact_eval_ledger_present"]`` is True

    Empirical anchor: DQS1 drop-many BUILD-1 verdict 2026-05-25.
    """
    source = stack_spec.get("interaction_matrix_source")
    if isinstance(source, str):
        source_lower = source.lower()
        empirical_tokens = (
            "paired_cpu_exact_eval",
            "modal_cpu_dispatch",
            "paired_cuda_exact_eval",
            "empirical",
            "exact_eval_ledger",
        )
        if any(tok in source_lower for tok in empirical_tokens):
            return True, (
                f"explicit override: stack_spec['interaction_matrix_source']="
                f"{source!r} is an empirical paired-eval source; the forbidden "
                "predicted_score_mean inheritance predicate is structurally "
                "not met"
            )
    if stack_spec.get("interaction_matrix_empirically_measured") is True:
        return True, (
            "explicit override: stack_spec['interaction_matrix_empirically_"
            "measured']=True structurally refutes the inherited-source predicate"
        )
    if stack_spec.get("paired_cpu_exact_eval_ledger_present") is True:
        return True, (
            "explicit override: stack_spec['paired_cpu_exact_eval_ledger_"
            "present']=True structurally satisfies the canonical empirical "
            "interaction-matrix backing"
        )
    return False, ""


def _explicit_override_silent_no_spawn_modal_dispatch(
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Refuse silent_no_spawn_modal_dispatch_v1 match when register_pre_spawn_fatal wired.

    The forbidden predicate is ``experiments/modal_*.py::main() contains
    sys.exit(...) BEFORE fn.spawn() AND does NOT call
    register_pre_spawn_fatal()``. The match is STRUCTURALLY INAPPLICABLE
    when the stack_spec explicitly declares:

      * ``stack_spec["modal_dispatch_pre_spawn_path"]`` is False (no pre-
        spawn FATAL path possible)
      * ``stack_spec["modal_register_pre_spawn_fatal_wired"]`` is True
        (Catalog #360 helper is called before every sys.exit)
      * ``stack_spec["catalog_360_active"]`` is True
      * ``stack_spec["modal_dispatcher_route"]`` includes "register_pre_spawn_fatal"

    Empirical anchor: STC v2 5th consecutive silent-no-spawn; Catalog #360.
    """
    if stack_spec.get("modal_dispatch_pre_spawn_path") is False:
        return True, (
            "explicit override: stack_spec['modal_dispatch_pre_spawn_path']="
            "False structurally refutes the silent-no-spawn predicate (no "
            "pre-spawn FATAL path exists)"
        )
    if stack_spec.get("modal_register_pre_spawn_fatal_wired") is True:
        return True, (
            "explicit override: stack_spec['modal_register_pre_spawn_fatal_"
            "wired']=True structurally satisfies Catalog #360 (every "
            "pre-spawn FATAL routes through canonical helper)"
        )
    if stack_spec.get("catalog_360_active") is True:
        return True, (
            "explicit override: stack_spec['catalog_360_active']=True "
            "structurally satisfies the Catalog #360 STRICT preflight gate"
        )
    route = stack_spec.get("modal_dispatcher_route")
    if isinstance(route, str) and "register_pre_spawn_fatal" in route.lower():
        return True, (
            f"explicit override: stack_spec['modal_dispatcher_route']="
            f"{route!r} contains register_pre_spawn_fatal token; structurally "
            "satisfies Catalog #360 canonical-helper routing"
        )
    return False, ""


def _explicit_override_subagent_spawn_without_head_state_premise_verification(
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Refuse subagent_spawn_without_head_state_premise_verification_v1 when PV evidence present.

    The forbidden predicate is ``Agent.spawn(...) called WITHOUT preceding
    git log --oneline -30 AND git status AND sister-landing-memo PV per
    Catalog #229``. The match is STRUCTURALLY INAPPLICABLE when the
    stack_spec explicitly declares PV evidence:

      * ``stack_spec["pv_evidence_present"]`` is True (Catalog #373)
      * ``stack_spec["catalog_229_pv_active"]`` is True
      * ``stack_spec["git_log_pv_in_prompt"]`` is True
      * ``stack_spec["sister_landing_memo_check_done"]`` is True
      * ``stack_spec["head_state_pv_check_done"]`` is True

    Empirical anchors: Wave N+5 Slot 1 Compound C STAND_DOWN + Wave N+5
    Slot 2 framework_agnostic STAND_DOWN.
    """
    if stack_spec.get("pv_evidence_present") is True:
        return True, (
            "explicit override: stack_spec['pv_evidence_present']=True "
            "structurally satisfies the Catalog #229 premise-verification-"
            "before-edit discipline (Catalog #373 sister gate)"
        )
    if stack_spec.get("catalog_229_pv_active") is True:
        return True, (
            "explicit override: stack_spec['catalog_229_pv_active']=True "
            "structurally satisfies Catalog #229 PV discipline"
        )
    if stack_spec.get("git_log_pv_in_prompt") is True:
        return True, (
            "explicit override: stack_spec['git_log_pv_in_prompt']=True "
            "structurally satisfies the pre-spawn HEAD-state PV step"
        )
    if (
        stack_spec.get("sister_landing_memo_check_done") is True
        and stack_spec.get("head_state_pv_check_done") is True
    ):
        return True, (
            "explicit override: stack_spec['sister_landing_memo_check_done']"
            "=True AND stack_spec['head_state_pv_check_done']=True together "
            "structurally satisfy Catalog #229 PV"
        )
    return False, ""


def _explicit_override_predecessor_working_tree_uncommitted_handoff(
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Refuse predecessor_working_tree_uncommitted_handoff_v1 when predecessor committed.

    The forbidden predicate is ``subagent SUBAGENT_TERMINATE without
    canonical-serializer commit AND working-tree has predecessor-owned
    edits``. The match is STRUCTURALLY INAPPLICABLE when the stack_spec
    declares the predecessor committed cleanly:

      * ``stack_spec["predecessor_committed_via_serializer"]`` is True
        (Catalog #117 + #157 + #174 canonical commit-serializer trace)
      * ``stack_spec["working_tree_clean_at_spawn_time"]`` is True
      * ``stack_spec["supersession_pending_declared"]`` is True (explicit
        STAND_DOWN supersession declaration)
      * ``stack_spec["catalog_117_serializer_log_has_predecessor_commit"]``
        is True

    Empirical anchor: Wave N+5 Slot 2 framework_agnostic STAND_DOWN.
    """
    if stack_spec.get("predecessor_committed_via_serializer") is True:
        return True, (
            "explicit override: stack_spec['predecessor_committed_via_"
            "serializer']=True structurally satisfies Catalog #117 + #157 + "
            "#174 canonical serializer discipline"
        )
    if stack_spec.get("working_tree_clean_at_spawn_time") is True:
        return True, (
            "explicit override: stack_spec['working_tree_clean_at_spawn_time']"
            "=True structurally refutes the uncommitted-handoff predicate"
        )
    if stack_spec.get("supersession_pending_declared") is True:
        return True, (
            "explicit override: stack_spec['supersession_pending_declared']="
            "True structurally satisfies the canonical STAND_DOWN "
            "supersession-declaration discipline"
        )
    if stack_spec.get("catalog_117_serializer_log_has_predecessor_commit") is True:
        return True, (
            "explicit override: stack_spec['catalog_117_serializer_log_has_"
            "predecessor_commit']=True structurally satisfies Catalog #117 "
            "serializer-log discipline"
        )
    return False, ""


def _explicit_override_wyner_ziv_prefix_y_density_decoder_state_dict_surface(
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Refuse wyner_ziv_prefix_y_density_decoder_state_dict_surface_v1 when surface differs.

    The forbidden predicate is ``wyner_ziv_layer.intercept_location ==
    STATE_DICT_SERIALIZATION AND wyner_ziv_layer.side_info_source IN
    canonical_4_sources AND base_substrate_bytes_form IN {raw_fp16,
    raw_fp32, torch_save}``. The match is STRUCTURALLY INAPPLICABLE when
    stack_spec explicitly declares any of:

      * ``stack_spec["wyner_ziv_intercept_location"]`` not in
        {STATE_DICT_SERIALIZATION, state_dict, raw_weights} (the
        intercept is at a different layer — e.g. POSE_AXIS_SIDE_INFO)
      * ``stack_spec["wyner_ziv_side_info_source"]`` is per-pair PoseNet-
        output Y (per Catalog #311 canonical unwind path)
      * ``stack_spec["base_substrate_bytes_form"]`` is NOT in raw_fp16 /
        raw_fp32 / torch_save (e.g. compressed_archive_zip_member)
      * ``stack_spec["catalog_311_atick_tishby_wyner_active"]`` is True

    Empirical anchor: Wyner-Ziv L1 LONG MLX 600-pair landing commit 6f5eabf30.
    """
    intercept = stack_spec.get("wyner_ziv_intercept_location")
    if isinstance(intercept, str):
        intercept_lower = intercept.lower()
        forbidden_intercepts = ("state_dict_serialization", "state_dict", "raw_weights")
        if intercept_lower and not any(
            t in intercept_lower for t in forbidden_intercepts
        ):
            return True, (
                f"explicit override: stack_spec['wyner_ziv_intercept_location']"
                f"={intercept!r} is not at state_dict_serialization; the "
                "forbidden predicate is structurally not met"
            )
    side_info = stack_spec.get("wyner_ziv_side_info_source")
    if isinstance(side_info, str):
        side_info_lower = side_info.lower()
        canonical_unwind_tokens = (
            "per_pair_posenet_output_y",
            "per_pair_pose_y",
            "atick_redlich_ego_motion_y",
            "posenet_precomputed_per_pair",
        )
        if any(tok in side_info_lower for tok in canonical_unwind_tokens):
            return True, (
                f"explicit override: stack_spec['wyner_ziv_side_info_source']="
                f"{side_info!r} uses canonical Catalog #311 unwind path "
                "(per-pair PoseNet-output Y), not the falsified 4-canonical-"
                "source family"
            )
    bytes_form = stack_spec.get("base_substrate_bytes_form")
    if isinstance(bytes_form, str):
        bytes_form_lower = bytes_form.lower()
        forbidden_forms = ("raw_fp16", "raw_fp32", "torch_save")
        if bytes_form_lower and not any(
            t in bytes_form_lower for t in forbidden_forms
        ):
            return True, (
                f"explicit override: stack_spec['base_substrate_bytes_form']="
                f"{bytes_form!r} is not raw_fp16/fp32/torch_save; the entropy-"
                "flat-surface predicate is structurally not met"
            )
    if stack_spec.get("catalog_311_atick_tishby_wyner_active") is True:
        return True, (
            "explicit override: stack_spec['catalog_311_atick_tishby_wyner_"
            "active']=True structurally satisfies the canonical Catalog #311 "
            "Atick-Tishby-Wyner triple unwind path"
        )
    return False, ""


def _explicit_override_wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface(
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Refuse wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface_v1 sister.

    Sister of the prefix-Y density override; the forbidden predicate is
    the same family (raw_fp16/fp32 entropy-flat surface) but for cross-
    substrate-composition-Y side-info specifically. Reuses the parent
    sister-helper because both predicates fall away under the same
    structural conditions (intercept-at-non-state-dict OR canonical
    Catalog #311 unwind path OR non-raw bytes form).

    Empirical anchor: Wave N+9 Slot 3 cross-substrate-composition Y
    FALSIFIED commit 2cedcee48.
    """
    return _explicit_override_wyner_ziv_prefix_y_density_decoder_state_dict_surface(
        stack_spec
    )


_EXPLICIT_OVERRIDE_PREDICATES: dict[str, _OverridePredicate] = {
    "fp4_packed_without_qat_cos_collapse_v1": (
        _explicit_override_fp4_packed_without_qat
    ),
    "brotli_plus_lzma_chained_anti_pattern_v1": (
        _explicit_override_brotli_plus_lzma_chained
    ),
    "lzma_on_already_brotli_saturated_compounding_v1": (
        _explicit_override_lzma_on_already_brotli
    ),
    "cross_paradigm_test_without_per_axis_decomposition_v1": (
        _explicit_override_cross_paradigm_test_per_axis_decomposition
    ),
    "quantize_then_svd_corrupted_low_rank_v1": (
        _explicit_override_quantize_then_svd_corrupted
    ),
    # WAVE N+10 SLOT 2 EXTENSION 2026-05-28 (task #1479): +10 entries
    # closing the override-coverage gap surfaced in Yousfi adversarial
    # audit. Anti-pattern #12 docstring overstatement is NOT-APPLICABLE
    # at the stack_spec surface (source-text content, not stack_spec).
    "predicted_band_from_random_init_tier_c_v1": (
        _explicit_override_predicted_band_from_random_init_tier_c
    ),
    "rank_1_problem_spec_synergy_tautology_v1": (
        _explicit_override_rank_1_problem_spec_synergy_tautology
    ),
    "phantom_score_directory_naming_lie_v1": (
        _explicit_override_phantom_score_directory_naming_lie
    ),
    "transient_tmp_path_in_persisted_artifact_v1": (
        _explicit_override_transient_tmp_path_in_persisted_artifact
    ),
    "source_selector_inherited_predicted_score_mean_v1": (
        _explicit_override_source_selector_inherited_predicted_score_mean
    ),
    "silent_no_spawn_modal_dispatch_v1": (
        _explicit_override_silent_no_spawn_modal_dispatch
    ),
    "subagent_spawn_without_head_state_premise_verification_v1": (
        _explicit_override_subagent_spawn_without_head_state_premise_verification
    ),
    "predecessor_working_tree_uncommitted_handoff_v1": (
        _explicit_override_predecessor_working_tree_uncommitted_handoff
    ),
    "wyner_ziv_prefix_y_density_decoder_state_dict_surface_v1": (
        _explicit_override_wyner_ziv_prefix_y_density_decoder_state_dict_surface
    ),
    "wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface_v1": (
        _explicit_override_wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface
    ),
}


def evaluate_explicit_override_for_anti_pattern(
    anti_pattern_id: str,
    stack_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    """Public API: evaluate a single anti-pattern's explicit-override predicate.

    Returns ``(is_structurally_inapplicable, human_readable_reason)``.
    When ``is_structurally_inapplicable=True``, the matcher MUST refuse
    to match the anti-pattern regardless of token-overlap confidence.

    Anti-patterns without a registered explicit-override predicate
    return ``(False, "")`` (the matcher falls back to token-heuristic
    matching).

    Per CLAUDE.md "Beauty, simplicity, and developer experience": exposed
    as a public function so downstream consumers (e.g. cathedral autopilot
    diagnostics, operator-facing CLI list_canonical_anti_patterns.py) can
    introspect the override decision without re-implementing the table.
    """
    if not isinstance(stack_spec, Mapping):
        return False, ""
    predicate = _EXPLICIT_OVERRIDE_PREDICATES.get(anti_pattern_id)
    if predicate is None:
        return False, ""
    try:
        return predicate(stack_spec)
    except Exception:  # noqa: BLE001 - defensive; never crash matcher
        return False, ""


# Canonical compounding-order violations the matcher recognizes structurally.
# These mirror anti-patterns #1 / #2 / #4 / #5 in the initial canonical
# population. Each entry: (after_op_tokens, before_op_tokens) where the
# pattern fires when an `after_op_token` appears AFTER a `before_op_token`
# in the ops list (i.e. the FORBIDDEN ordering).
_FORBIDDEN_ORDER_PAIRS: tuple[tuple[tuple[str, ...], tuple[str, ...], str], ...] = (
    (
        ("lzma",),
        ("brotli",),
        "LZMA chained after brotli saturates at ~1.001 ratio "
        "(anti-pattern lzma_on_already_brotli_saturated_compounding_v1)",
    ),
    (
        ("svd", "low_rank"),
        ("quantize", "int8", "fp4", "fp4_packed"),
        "SVD applied AFTER quantization corrupts low-rank residual; "
        "canonical order is SVD FIRST then quantize "
        "(anti-pattern quantize_then_svd_corrupted_low_rank_v1)",
    ),
)


def _stack_spec_to_haystack(stack_spec: Mapping[str, Any]) -> str:
    """Flatten a stack_spec to a single lowercase string for substring matching.

    Walks nested mappings + lists + primitives. Non-string non-numeric
    values are skipped silently.
    """
    parts: list[str] = []

    def _walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                parts.append(str(k).lower())
                _walk(v)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _walk(item)
        elif isinstance(value, (str, int, float, bool)):
            parts.append(str(value).lower())

    _walk(stack_spec)
    return " ".join(parts)


def _confidence_for_condition_match(
    condition: str, haystack: str, predicate: str
) -> float:
    """Estimate match confidence for a single condition→haystack hit.

    1.0 if predicate-text appears verbatim in haystack;
    0.7 if a strong recurrence-condition phrase (>= 6 chars) matches;
    0.5 if a weaker substring match exists.
    """
    cond_lower = condition.lower()
    pred_lower = predicate.lower()
    if pred_lower and pred_lower in haystack:
        return 1.0
    if cond_lower and len(cond_lower) >= 6 and cond_lower in haystack:
        return 0.7
    # Token-level overlap fallback: tokenize the condition and check that
    # at least 2 non-trivial tokens of the condition appear in haystack.
    tokens = [t for t in cond_lower.split() if len(t) >= 4]
    if tokens:
        hits = sum(1 for t in tokens if t in haystack)
        if hits >= 2:
            return 0.5
    return 0.0


def match_stack_against_anti_patterns(
    stack_spec: Mapping[str, Any],
    *,
    path: Path | None = None,
    min_confidence: float = 0.5,
) -> tuple[AntiPatternMatch, ...]:
    """Match a proposed stack_spec against every registered anti-pattern.

    Returns matches ordered by severity (CRITICAL first; LOW last; within
    a severity tier, by descending confidence).

    Args:
        stack_spec: open Mapping describing the proposed compound stack.
            Canonical shape (all optional; the matcher honors explicit
            guarantee fields per the override table when present):

              {
                  "substrate_id": str,
                  "compression_ops": list[str],   # structured execution order
                  "quantization_ops": list[str],  # structured execution order
                  "decoder_arch": str,
                  "per_axis_decomposition_active": bool,   # Catalog #356
                  "predicted_band_source": str,
                  "archive_artifact_path": str | None,
                  "data_source_inheritance": dict[str, str] | None,
                  "modal_dispatch_pre_spawn_path": bool,
                  "quantization_aware_training": bool,    # Catalog #146 QAT
                  "qat_enabled": bool,                    # sister synonym
                  "qat_finetune_passes": int,             # sister synonym
                  "training_pipeline": {"qat_finetune": bool, ...},
                  ...
              }

            ARCHITECTURAL FIX 2026-05-28: when stack_spec carries explicit
            override fields (e.g. ``quantization_aware_training=True``,
            structured ``compression_ops`` list with no ``lzma`` token,
            ``per_axis_decomposition_active=True``), the matcher consults the
            per-anti-pattern override table BEFORE the token-overlap heuristic
            and REFUSES to match anti-patterns whose forbidden predicate is
            structurally proven inapplicable. See
            :func:`evaluate_explicit_override_for_anti_pattern` for the
            canonical override decision API.

        path: optional override for the registry JSONL path (for tests).
        min_confidence: matches below this threshold are dropped (default 0.5).

    Returns:
        Tuple of AntiPatternMatch ordered by severity then confidence.
        Empty tuple if no anti-patterns match.

    Per Catalog #287/#323: observability-only — does NOT mutate state.
    """
    if not isinstance(stack_spec, Mapping):
        # Defensive: an invalid stack_spec returns empty rather than raise
        # so the consumer never crashes the ranker cascade.
        return ()
    haystack = _stack_spec_to_haystack(stack_spec)
    if not haystack:
        return ()

    registered = query_anti_patterns(path=path)
    matches: list[AntiPatternMatch] = []
    for ap in registered:
        # ARCHITECTURAL FIX: consult the explicit-override predicate FIRST.
        # When the stack_spec structurally proves the forbidden predicate is
        # not met, the match is refused regardless of token-heuristic.
        is_inapplicable, _override_reason = evaluate_explicit_override_for_anti_pattern(
            ap.anti_pattern_id, stack_spec
        )
        if is_inapplicable:
            continue

        best_condition: str | None = None
        best_confidence: float = 0.0
        for condition in ap.recurrence_conditions:
            conf = _confidence_for_condition_match(
                condition, haystack, ap.forbidden_pattern_predicate
            )
            if conf > best_confidence:
                best_confidence = conf
                best_condition = condition
        # Also try matching the forbidden predicate directly even when no
        # recurrence_condition surfaced.
        pred_lower = ap.forbidden_pattern_predicate.lower()
        if pred_lower and pred_lower in haystack and best_confidence < 1.0:
            best_confidence = 1.0
            best_condition = (
                best_condition
                or f"forbidden predicate matched: {ap.forbidden_pattern_predicate[:80]}"
            )
        if best_confidence >= min_confidence and best_condition is not None:
            matches.append(
                AntiPatternMatch(
                    anti_pattern=ap,
                    recurrence_condition_triggered=best_condition,
                    canonical_unwind_path_recommended=ap.canonical_unwind_path,
                    match_confidence=float(best_confidence),
                )
            )

    matches.sort(
        key=lambda m: (
            _SEVERITY_RANK.get(m.anti_pattern.severity, 99),
            -m.match_confidence,
        )
    )
    return tuple(matches)


def validate_compound_stack_order(
    ops: list[str],
) -> ValidationResult:
    """Validate a proposed compound stack's operation order.

    Specifically checks for the canonical compounding-order ORDER-violations:

    1. LZMA after brotli (anti-pattern #1).
    2. SVD after quantize (anti-pattern #2).
    3. Brotli + LZMA chained (anti-pattern #4 — sister of #1; broader check).

    Returns a :class:`ValidationResult`. If ``is_valid=False``, the
    ``violations`` tuple enumerates each detected ORDER-violation with
    human-readable context citing the matched anti-pattern.

    Args:
        ops: list of operation tokens in execution order (e.g.
            ``["int8_per_channel", "brotli_q11", "lzma_q9"]``).

    Per the canonical design memo §"Layer 1": this is a sister API to
    ``match_stack_against_anti_patterns`` specifically for the
    compounding-order anti-pattern class. The cathedral autopilot ranker
    + Slot 1 Dykstra solver consume both APIs as ACTIVE constraints in
    the Pareto polytope feasibility set.
    """
    if not isinstance(ops, list):
        return ValidationResult(
            is_valid=False,
            violations=("ops must be a list of operation tokens",),
            suggested_canonical_order="",
        )
    ops_lower = [str(op).lower() for op in ops]
    violations: list[str] = []

    for after_tokens, before_tokens, message in _FORBIDDEN_ORDER_PAIRS:
        before_idx = -1
        after_idx = -1
        for i, op in enumerate(ops_lower):
            if before_idx == -1 and any(tok in op for tok in before_tokens):
                before_idx = i
            elif before_idx != -1 and any(tok in op for tok in after_tokens):
                after_idx = i
                break
        if before_idx != -1 and after_idx != -1 and after_idx > before_idx:
            violations.append(
                f"ops[{after_idx}]={ops[after_idx]!r} appears AFTER "
                f"ops[{before_idx}]={ops[before_idx]!r}: {message}"
            )

    # Sister specific check: brotli + lzma chained anywhere (anti-pattern #4).
    has_brotli = any("brotli" in op for op in ops_lower)
    has_lzma = any("lzma" in op for op in ops_lower)
    if has_brotli and has_lzma:
        # Already covered above when LZMA is after brotli; here we add a
        # sister-violation when they are interleaved in any order (the
        # broader anti-pattern #4 "compounding entropy coders that operate
        # on similar redundancy domains saturate").
        already_flagged = any("lzma" in v.lower() for v in violations)
        if not already_flagged:
            violations.append(
                "ops chain brotli + LZMA together: compounding entropy coders "
                "that operate on similar redundancy domains saturate (anti-"
                "pattern brotli_plus_lzma_chained_anti_pattern_v1)"
            )

    is_valid = not violations
    suggested = ""
    if not is_valid:
        # Best-effort canonical-order suggestion: any quantize-then-svd
        # ORDER-violation is corrected by reordering to svd-FIRST.
        if any("svd" in v.lower() and "quantize" in v.lower() for v in violations):
            suggested = (
                "Canonical compound-stack order: SVD/low-rank FIRST -> "
                "quantization (int8 or FP4) SECOND -> entropy coding "
                "(brotli OR ANS, choose one) THIRD"
            )
        elif any("lzma" in v.lower() and "brotli" in v.lower() for v in violations):
            suggested = (
                "Canonical compound-stack order: choose ONE high-quality "
                "entropy coder (brotli q=11) standalone rather than chaining "
                "LZMA after brotli (which saturates at ~1.001 ratio)"
            )
    return ValidationResult(
        is_valid=is_valid,
        violations=tuple(violations),
        suggested_canonical_order=suggested,
    )


__all__ = [
    "AntiPatternMatch",
    "ValidationResult",
    "evaluate_explicit_override_for_anti_pattern",
    "match_stack_against_anti_patterns",
    "validate_compound_stack_order",
]
