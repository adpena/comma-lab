# SPDX-License-Identifier: MIT
"""tac.canonical_anti_patterns — typed registry of CLASS-level forbidden patterns.

Per operator NON-NEGOTIABLE 2026-05-28 verbatim: *"learning anti-patterns is
upser important too for compounding continual learning, like the canonical
equations bu netgative and a higher layer of abstraction"*.

Sister of ``tac.canonical_equations`` at the NEGATIVE registry surface.
Where canonical equations capture POSITIVE patterns (predicted bands with
empirical anchors), canonical anti-patterns capture NEGATIVE patterns
(forbidden compounding/diagnosis/provenance failures with empirical
falsifications). Together they bound the Pareto polytope feasibility set
that the cathedral autopilot ranker + the per-axis Dykstra solver (Slot 1
Wave N+2 integration target) consume to steer next-cycle attack direction.

Quick start:

    from tac.canonical_anti_patterns import (
        AntiPattern,
        EmpiricalFalsification,
        register_anti_pattern,
        append_empirical_falsification,
        query_anti_patterns,
        query_anti_patterns_by_substrate,
        populate_initial_anti_patterns,
        match_stack_against_anti_patterns,
        validate_compound_stack_order,
    )

    # Inspect the registry
    for ap in query_anti_patterns():
        print(ap.anti_pattern_id, ap.severity, ap.is_actively_recurring)

    # Match a proposed stack against registered anti-patterns
    matches = match_stack_against_anti_patterns({
        "compression_ops": ["int8_per_channel", "brotli_q11", "lzma_q9"],
    })
    for m in matches:
        print(m.anti_pattern.anti_pattern_id, m.canonical_unwind_path_recommended)

Cross-references:
  * CLAUDE.md "FORBIDDEN PATTERNS" section — canonical source of initial population
  * CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable — Pareto polytope constraint
  * CLAUDE.md "Results must become system intelligence" — extincts orphan-prose
  * Catalog #344 — sister POSITIVE registry (canonical equations)
  * Catalog #335 — cathedral consumer auto-discovery
  * Catalog #323 — canonical Provenance umbrella
  * Catalog #125 — 6-hook wire-in non-negotiable (hook #2 Pareto PRIMARY)
  * Catalog #287 — placeholder rationale rejection sister
"""
from __future__ import annotations

from tac.canonical_anti_patterns.anti_pattern import (
    CANONICAL_ANTI_PATTERN_SCHEMA_VERSION,
    INCIDENT_EDGE_CASE_PARTIAL,
    INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION,
    INCIDENT_PARADIGM_LEVEL_FALSIFICATION,
    INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
    PARADIGM_COMPOUNDING_ORDER,
    PARADIGM_DATA_SOURCE,
    PARADIGM_DIAGNOSIS,
    PARADIGM_OBSERVABILITY,
    PARADIGM_PREMATURE_KILL,
    PARADIGM_PROVENANCE,
    PARADIGM_QUANTIZATION,
    PARADIGM_RIGOR_LOSS,
    RECALIBRATE_NEVER_AUTO,
    RECALIBRATE_ON_NEW_FALSIFICATIONS,
    RECALIBRATE_ON_OPERATOR,
    RECALIBRATE_ON_SEVERITY_DRIFT,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_OBSERVED_CRITICAL,
    SEVERITY_OBSERVED_HIGH,
    SEVERITY_OBSERVED_LOW,
    SEVERITY_OBSERVED_MEDIUM,
    VALID_INCIDENT_CLASSIFICATIONS,
    VALID_OBSERVED_SEVERITIES,
    VALID_PARADIGM_CLASSES,
    VALID_RECALIBRATION_TRIGGERS,
    VALID_SEVERITIES,
    AntiPattern,
    EmpiricalFalsification,
    InvalidAntiPatternError,
)
from tac.canonical_anti_patterns.builtins import (
    build_all_initial_anti_patterns,
    populate_initial_anti_patterns,
)
from tac.canonical_anti_patterns.pattern_matcher import (
    AntiPatternMatch,
    ValidationResult,
    evaluate_explicit_override_for_anti_pattern,
    match_stack_against_anti_patterns,
    validate_compound_stack_order,
)
from tac.canonical_anti_patterns.registry import (
    CANONICAL_ANTI_PATTERNS_REGISTRY_LOCK,
    CANONICAL_ANTI_PATTERNS_REGISTRY_PATH,
    EVENT_ANTI_PATTERN_RECALIBRATED,
    EVENT_ANTI_PATTERN_REGISTERED,
    EVENT_FALSIFICATION_APPENDED,
    EVENT_UNWIND_PATH_RATIFIED,
    VALID_EVENT_TYPES,
    AntiPatternRecalibrationReport,
    AntiPatternRegistryCorruptError,
    append_empirical_falsification,
    auto_recalibrate_from_continual_learning_posterior,
    get_anti_pattern_by_id,
    load_anti_patterns_events_lenient,
    load_anti_patterns_strict,
    query_anti_patterns,
    query_anti_patterns_by_substrate,
    query_falsifications_by_paradigm_class,
    query_recurrence_rate_by_severity,
    register_anti_pattern,
)


__all__ = [
    # Schema + contract constants
    "CANONICAL_ANTI_PATTERN_SCHEMA_VERSION",
    "CANONICAL_ANTI_PATTERNS_REGISTRY_LOCK",
    "CANONICAL_ANTI_PATTERNS_REGISTRY_PATH",
    "EVENT_ANTI_PATTERN_RECALIBRATED",
    "EVENT_ANTI_PATTERN_REGISTERED",
    "EVENT_FALSIFICATION_APPENDED",
    "EVENT_UNWIND_PATH_RATIFIED",
    "INCIDENT_EDGE_CASE_PARTIAL",
    "INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION",
    "INCIDENT_PARADIGM_LEVEL_FALSIFICATION",
    "INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE",
    "PARADIGM_COMPOUNDING_ORDER",
    "PARADIGM_DATA_SOURCE",
    "PARADIGM_DIAGNOSIS",
    "PARADIGM_OBSERVABILITY",
    "PARADIGM_PREMATURE_KILL",
    "PARADIGM_PROVENANCE",
    "PARADIGM_QUANTIZATION",
    "PARADIGM_RIGOR_LOSS",
    "RECALIBRATE_NEVER_AUTO",
    "RECALIBRATE_ON_NEW_FALSIFICATIONS",
    "RECALIBRATE_ON_OPERATOR",
    "RECALIBRATE_ON_SEVERITY_DRIFT",
    "SEVERITY_CRITICAL",
    "SEVERITY_HIGH",
    "SEVERITY_LOW",
    "SEVERITY_MEDIUM",
    "SEVERITY_OBSERVED_CRITICAL",
    "SEVERITY_OBSERVED_HIGH",
    "SEVERITY_OBSERVED_LOW",
    "SEVERITY_OBSERVED_MEDIUM",
    "VALID_EVENT_TYPES",
    "VALID_INCIDENT_CLASSIFICATIONS",
    "VALID_OBSERVED_SEVERITIES",
    "VALID_PARADIGM_CLASSES",
    "VALID_RECALIBRATION_TRIGGERS",
    "VALID_SEVERITIES",
    # Dataclasses + errors
    "AntiPattern",
    "AntiPatternMatch",
    "AntiPatternRecalibrationReport",
    "AntiPatternRegistryCorruptError",
    "EmpiricalFalsification",
    "InvalidAntiPatternError",
    "ValidationResult",
    # Registry helpers
    "append_empirical_falsification",
    "auto_recalibrate_from_continual_learning_posterior",
    "build_all_initial_anti_patterns",
    "evaluate_explicit_override_for_anti_pattern",
    "get_anti_pattern_by_id",
    "load_anti_patterns_events_lenient",
    "load_anti_patterns_strict",
    "match_stack_against_anti_patterns",
    "populate_initial_anti_patterns",
    "query_anti_patterns",
    "query_anti_patterns_by_substrate",
    "query_falsifications_by_paradigm_class",
    "query_recurrence_rate_by_severity",
    "register_anti_pattern",
    "validate_compound_stack_order",
]
