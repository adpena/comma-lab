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

Sister of:
  * ``tac.canonical_anti_patterns.registry`` (the registered anti-patterns)
  * ``tac.canonical_anti_patterns.builtins`` (initial 12 anti-patterns)
  * ``src/tac/cathedral_consumers/anti_pattern_lookup_consumer/`` (consumer)
  * Slot 1 Dykstra Pareto polytope solver (Wave N+2 integration target)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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
            Canonical shape (all optional):

              {
                  "substrate_id": str,
                  "compression_ops": list[str],
                  "quantization_ops": list[str],
                  "decoder_arch": str,
                  "per_axis_decomposition_active": bool,
                  "predicted_band_source": str,
                  "archive_artifact_path": str | None,
                  "data_source_inheritance": dict[str, str] | None,
                  "modal_dispatch_pre_spawn_path": bool,
                  ...
              }

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
    "match_stack_against_anti_patterns",
    "validate_compound_stack_order",
]
