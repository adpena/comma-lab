# SPDX-License-Identifier: MIT
"""No-ad-hoc/signal-loss/rediscovery/duplicate/drift invariants (GAP 4).

Per ``feedback_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_canonicalize_and_harden_for_automation_standing_directive_20260528.md``
5 canonical invariants:

| Invariant | Canonical structural enforcement |
|---|---|
| NO AD HOC | Main-thread queue-ranking routes through canonical 3-metric trichotomy (NOT manual per-turn) |
| NO SIGNAL LOSS | Every operator binding correction REGISTERS canonical equation + canonical anti-pattern within same turn per Catalog #344 + #371 |
| NO REDISCOVERY | Canonical anti-pattern registry queryable per Catalog #335 cathedral consumer; rediscovery = registry hit -> structurally refused |
| NO DUPLICATE CODE | Spawn prompts use canonical helper (NOT manual boilerplate duplication) |
| NO DRIFT | Per-turn ranking deterministic per canonical helper output; cap-window discipline enforced via canonical Catalog #371 auto-recalibration |

This helper validates a main-thread decision against the 5 canonical
invariants and emits a typed verdict the cathedral autopilot ranker
consumes per Catalog #335 auto-discovery.

Per ``meta_orchestrator_three_metric_trichotomy_orthogonality_v1`` +
``meta_orchestrator_variance_acceptance_dominance_at_low_probability_high_leverage_v1``
+ ``meta_orchestrator_cap_window_discipline_per_turn_variance_amortization_v1``
canonical equations: each invariant maps to a canonical structural
enforcement surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class InvariantValidationStatus(str, Enum):
    """Canonical verdict status for invariant validation.

    Mirrors Catalog #341 / #357 dual-tier consumer status taxonomy at
    the invariant-validator surface.
    """

    PASS_ALL_INVARIANTS_HONORED = "pass_all_invariants_honored"
    FAIL_ONE_OR_MORE_INVARIANTS_VIOLATED = "fail_one_or_more_invariants_violated"
    UNKNOWN_INSUFFICIENT_EVIDENCE = "unknown_insufficient_evidence"


# Canonical invariant names — 5 per the operator triple-message standing directive.
INVARIANT_NO_AD_HOC = "no_ad_hoc"
INVARIANT_NO_SIGNAL_LOSS = "no_signal_loss"
INVARIANT_NO_REDISCOVERY = "no_rediscovery"
INVARIANT_NO_DUPLICATE_CODE = "no_duplicate_code"
INVARIANT_NO_DRIFT = "no_drift"

VALID_INVARIANTS = frozenset(
    {
        INVARIANT_NO_AD_HOC,
        INVARIANT_NO_SIGNAL_LOSS,
        INVARIANT_NO_REDISCOVERY,
        INVARIANT_NO_DUPLICATE_CODE,
        INVARIANT_NO_DRIFT,
    }
)


@dataclass(frozen=True)
class InvariantValidationVerdict:
    """Typed verdict for the 5 canonical invariants.

    Args:
        status: one of :class:`InvariantValidationStatus`.
        per_invariant_status: mapping of invariant name -> per-invariant
            verdict ("pass" | "fail" | "unknown"). All 5 canonical
            invariants MUST appear as keys (the gate refuses partial
            verdicts).
        per_invariant_rationale: mapping of invariant name -> readable
            rationale (per-invariant explanation for the per-invariant
            verdict).
        violated_invariants: tuple of invariant names that returned
            "fail" (subset of VALID_INVARIANTS).
        rationale: operator-facing readable summary explaining the
            overall verdict + cross-references to canonical apparatus
            surfaces.
        promotable: ALWAYS False per Tier A canonical-routing markers.
        axis_tag: ``"[predicted]"`` per Catalog #287/#341.
    """

    status: InvariantValidationStatus
    per_invariant_status: Mapping[str, str]
    per_invariant_rationale: Mapping[str, str]
    violated_invariants: tuple[str, ...]
    rationale: str
    promotable: bool = False
    axis_tag: str = "[predicted]"

    def __post_init__(self) -> None:
        if not isinstance(self.status, InvariantValidationStatus):
            raise ValueError(
                "status must be an InvariantValidationStatus member"
            )
        if not isinstance(self.per_invariant_status, Mapping):
            raise ValueError("per_invariant_status must be a Mapping")
        if not isinstance(self.per_invariant_rationale, Mapping):
            raise ValueError("per_invariant_rationale must be a Mapping")
        missing = VALID_INVARIANTS - set(self.per_invariant_status.keys())
        if missing:
            raise ValueError(
                f"per_invariant_status missing keys: {sorted(missing)}; "
                f"all 5 canonical invariants must appear (got "
                f"{sorted(self.per_invariant_status.keys())})"
            )
        missing_rat = VALID_INVARIANTS - set(self.per_invariant_rationale.keys())
        if missing_rat:
            raise ValueError(
                f"per_invariant_rationale missing keys: {sorted(missing_rat)}"
            )
        for inv, status in self.per_invariant_status.items():
            if status not in _VALID_PER_INVARIANT_STATUSES:
                raise ValueError(
                    f"per_invariant_status[{inv!r}]={status!r} must be one "
                    f"of {sorted(_VALID_PER_INVARIANT_STATUSES)}"
                )
        if not isinstance(self.violated_invariants, tuple):
            raise ValueError("violated_invariants must be a tuple")
        for v in self.violated_invariants:
            if v not in VALID_INVARIANTS:
                raise ValueError(
                    f"violated_invariants entry {v!r} not in VALID_INVARIANTS"
                )
        # Cross-validation: status consistency.
        derived_violations = tuple(
            sorted(
                inv
                for inv, s in self.per_invariant_status.items()
                if s == "fail"
            )
        )
        if tuple(sorted(self.violated_invariants)) != derived_violations:
            raise ValueError(
                "violated_invariants must match per_invariant_status fail keys; "
                f"got violated={sorted(self.violated_invariants)}, "
                f"derived={list(derived_violations)}"
            )
        if derived_violations:
            if self.status != InvariantValidationStatus.FAIL_ONE_OR_MORE_INVARIANTS_VIOLATED:
                raise ValueError(
                    "status must be FAIL_ONE_OR_MORE_INVARIANTS_VIOLATED when "
                    f"violations exist (got {self.status})"
                )
        elif any(
            s == "unknown" for s in self.per_invariant_status.values()
        ):
            if self.status != InvariantValidationStatus.UNKNOWN_INSUFFICIENT_EVIDENCE:
                raise ValueError(
                    "status must be UNKNOWN_INSUFFICIENT_EVIDENCE when no "
                    "fails but some unknowns (got status="
                    f"{self.status})"
                )
        else:
            if self.status != InvariantValidationStatus.PASS_ALL_INVARIANTS_HONORED:
                raise ValueError(
                    "status must be PASS_ALL_INVARIANTS_HONORED when all "
                    f"pass (got {self.status})"
                )
        if self.promotable is not False:
            raise ValueError(
                "InvariantValidationVerdict.promotable MUST be False per "
                "Tier A canonical-routing markers (Catalog #341)"
            )
        if self.axis_tag != "[predicted]":
            raise ValueError(
                f"axis_tag={self.axis_tag!r} must be '[predicted]'"
            )

    def as_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict per cathedral consumer contract."""
        return {
            "status": self.status.value,
            "per_invariant_status": dict(self.per_invariant_status),
            "per_invariant_rationale": dict(self.per_invariant_rationale),
            "violated_invariants": list(self.violated_invariants),
            "rationale": self.rationale,
            "promotable": self.promotable,
            "axis_tag": self.axis_tag,
        }


_VALID_PER_INVARIANT_STATUSES = frozenset({"pass", "fail", "unknown"})


def _check_no_ad_hoc(
    main_thread_decision: Mapping[str, Any],
) -> tuple[str, str]:
    """NO AD HOC: ranking routes through canonical 3-metric trichotomy."""
    used_canonical = main_thread_decision.get(
        "ranking_via_canonical_three_metric_trichotomy"
    )
    if used_canonical is True:
        return (
            "pass",
            "Decision routes through canonical 3-metric trichotomy "
            "(tac.cathedral_autopilot.three_metric_trichotomy) per "
            "meta_orchestrator_three_metric_trichotomy_orthogonality_v1 "
            "canonical equation.",
        )
    if used_canonical is False:
        return (
            "fail",
            "Decision uses MANUAL ad-hoc per-turn ranking; violates "
            "manual_main_thread_orchestrator_ranking_drift_across_turns_via_ad_hoc_priority_assignment_v1 "
            "canonical anti-pattern. Route through "
            "rank_candidates_via_three_metric_trichotomy.",
        )
    return (
        "unknown",
        "Decision did not declare ranking_via_canonical_three_metric_trichotomy "
        "field; cannot determine NO_AD_HOC compliance.",
    )


def _check_no_signal_loss(
    main_thread_decision: Mapping[str, Any],
    canonical_state: Mapping[str, Any] | None,
) -> tuple[str, str]:
    """NO SIGNAL LOSS: operator corrections register canonical mutations."""
    pending_corrections = main_thread_decision.get(
        "pending_operator_corrections_count"
    )
    same_turn_registrations = main_thread_decision.get(
        "operator_correction_canonical_mutations_registered_this_turn"
    )
    if (
        isinstance(pending_corrections, int)
        and pending_corrections == 0
        and (same_turn_registrations is None or same_turn_registrations >= 0)
    ):
        return (
            "pass",
            "No pending operator corrections; canonical apparatus mutations "
            "have absorbed all binding corrections per "
            "operator_correction_canonical_apparatus_mutation_lag_v1 "
            "canonical anti-pattern unwind.",
        )
    if isinstance(pending_corrections, int) and pending_corrections > 0:
        if isinstance(same_turn_registrations, int) and same_turn_registrations >= pending_corrections:
            return (
                "pass",
                f"{pending_corrections} pending operator corrections all "
                f"absorbed by {same_turn_registrations} canonical mutations "
                "registered this turn (Catalog #344 + #371).",
            )
        return (
            "fail",
            f"{pending_corrections} pending operator corrections; only "
            f"{same_turn_registrations or 0} canonical mutations registered "
            "this turn. Violates operator_correction_canonical_apparatus_mutation_lag_v1 "
            "canonical anti-pattern. Route through "
            "register_operator_binding_correction.",
        )
    return (
        "unknown",
        "Decision did not declare pending_operator_corrections_count; "
        "cannot determine NO_SIGNAL_LOSS compliance.",
    )


def _check_no_rediscovery(
    main_thread_decision: Mapping[str, Any],
    canonical_state: Mapping[str, Any] | None,
) -> tuple[str, str]:
    """NO REDISCOVERY: canonical anti-pattern registry consulted before proposal."""
    consulted = main_thread_decision.get(
        "canonical_anti_pattern_registry_consulted"
    )
    matched_anti_patterns = main_thread_decision.get(
        "matched_anti_pattern_ids", ()
    )
    if consulted is True:
        if matched_anti_patterns:
            acknowledged = main_thread_decision.get(
                "matched_anti_patterns_acknowledged", False
            )
            if acknowledged:
                return (
                    "pass",
                    f"Canonical anti-pattern registry consulted; "
                    f"{len(matched_anti_patterns)} match(es) acknowledged "
                    "with canonical_unwind_path routing per Catalog #344 + #373.",
                )
            return (
                "fail",
                f"Canonical anti-pattern registry consulted but "
                f"{len(matched_anti_patterns)} match(es) NOT acknowledged. "
                "Violates hygiene_ev_vs_frontier_breaking_ev_vs_highest_ev_shortest_wall_clock_conflation_canonical_rediscovery_v1 "
                "canonical anti-pattern (rediscovering known anti-patterns). "
                "Acknowledge via canonical_unwind_path.",
            )
        return (
            "pass",
            "Canonical anti-pattern registry consulted; no matches; "
            "no rediscovery risk per Catalog #344.",
        )
    if consulted is False:
        return (
            "fail",
            "Canonical anti-pattern registry NOT consulted. Violates the "
            "NO REDISCOVERY invariant. Query tac.canonical_anti_patterns.query_anti_patterns.",
        )
    return (
        "unknown",
        "Decision did not declare canonical_anti_pattern_registry_consulted; "
        "cannot determine NO_REDISCOVERY compliance.",
    )


def _check_no_duplicate_code(
    main_thread_decision: Mapping[str, Any],
) -> tuple[str, str]:
    """NO DUPLICATE CODE: spawn prompts use canonical helper (not duplicate boilerplate)."""
    spawn_uses_canonical = main_thread_decision.get(
        "spawn_prompt_uses_canonical_helper"
    )
    duplicates_existing_code = main_thread_decision.get(
        "duplicates_existing_canonical_module"
    )
    if duplicates_existing_code is True:
        return (
            "fail",
            "Decision duplicates an existing canonical module. Violates "
            "spawn_prompt_boilerplate_duplication_across_subagent_waves_v1 "
            "canonical anti-pattern. EXTEND the existing canonical surface "
            "instead of building a parallel package.",
        )
    if spawn_uses_canonical is True:
        return (
            "pass",
            "Spawn prompt uses canonical helper per the no-duplicate-code "
            "invariant. EXTEND existing canonical surfaces instead of "
            "boilerplate duplication.",
        )
    if spawn_uses_canonical is False:
        return (
            "fail",
            "Spawn prompt does NOT use canonical helper; violates the "
            "NO DUPLICATE CODE invariant. Route through canonical "
            "spawn-prompt helper.",
        )
    if duplicates_existing_code is False:
        return (
            "pass",
            "Decision does NOT duplicate existing canonical module; "
            "no_duplicate_code invariant honored (partial evidence: "
            "spawn-prompt-helper field not declared).",
        )
    return (
        "unknown",
        "Decision did not declare spawn_prompt_uses_canonical_helper OR "
        "duplicates_existing_canonical_module; cannot determine "
        "NO_DUPLICATE_CODE compliance.",
    )


def _check_no_drift(
    main_thread_decision: Mapping[str, Any],
) -> tuple[str, str]:
    """NO DRIFT: per-turn ranking deterministic per canonical helper output."""
    deterministic = main_thread_decision.get(
        "per_turn_ranking_deterministic_per_canonical_helper"
    )
    if deterministic is True:
        return (
            "pass",
            "Per-turn ranking deterministic per canonical helper output "
            "(NO drift across turns per "
            "meta_orchestrator_cap_window_discipline_per_turn_variance_amortization_v1 "
            "canonical equation).",
        )
    if deterministic is False:
        return (
            "fail",
            "Per-turn ranking shifts based on landed sister vs canonical "
            "metric. Violates manual_main_thread_orchestrator_ranking_drift_across_turns_via_ad_hoc_priority_assignment_v1 "
            "canonical anti-pattern. Make ranking deterministic per "
            "canonical helper output.",
        )
    return (
        "unknown",
        "Decision did not declare per_turn_ranking_deterministic_per_canonical_helper; "
        "cannot determine NO_DRIFT compliance.",
    )


def validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants(
    main_thread_decision: Mapping[str, Any],
    canonical_state: Mapping[str, Any] | None = None,
) -> InvariantValidationVerdict:
    """Validate a main-thread decision against the 5 canonical invariants.

    Per ``feedback_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_canonicalize_and_harden_for_automation_standing_directive_20260528.md``
    + canonical equations
    ``meta_orchestrator_three_metric_trichotomy_orthogonality_v1`` +
    ``meta_orchestrator_cap_window_discipline_per_turn_variance_amortization_v1``:
    each invariant maps to a canonical structural enforcement surface.

    The helper consumes a ``main_thread_decision`` mapping (typically
    produced by :func:`tac.cathedral_autopilot.select_canonical_next_spawn_for_main_thread`
    + auxiliary main-thread context) and validates the 5 invariants in
    independent checks. The verdict surfaces per-invariant status +
    rationale so the cathedral autopilot ranker can route per-invariant
    canonical_unwind_path recommendations.

    Args:
        main_thread_decision: mapping carrying decision fields:
            - ``ranking_via_canonical_three_metric_trichotomy`` (bool)
            - ``pending_operator_corrections_count`` (int)
            - ``operator_correction_canonical_mutations_registered_this_turn`` (int)
            - ``canonical_anti_pattern_registry_consulted`` (bool)
            - ``matched_anti_pattern_ids`` (tuple[str, ...])
            - ``matched_anti_patterns_acknowledged`` (bool)
            - ``spawn_prompt_uses_canonical_helper`` (bool)
            - ``duplicates_existing_canonical_module`` (bool)
            - ``per_turn_ranking_deterministic_per_canonical_helper`` (bool)
            Fields may be absent; per-invariant verdict will be ``unknown``.
        canonical_state: optional mapping of canonical apparatus state
            for cross-validation; reserved for future Phase 2+ checks.

    Returns:
        :class:`InvariantValidationVerdict` with per-invariant status +
        rationale + violated-invariants list.
    """
    if not isinstance(main_thread_decision, Mapping):
        raise ValueError("main_thread_decision must be a Mapping")
    if canonical_state is not None and not isinstance(canonical_state, Mapping):
        raise ValueError("canonical_state must be a Mapping or None")

    per_invariant_status: dict[str, str] = {}
    per_invariant_rationale: dict[str, str] = {}

    status, rationale = _check_no_ad_hoc(main_thread_decision)
    per_invariant_status[INVARIANT_NO_AD_HOC] = status
    per_invariant_rationale[INVARIANT_NO_AD_HOC] = rationale

    status, rationale = _check_no_signal_loss(main_thread_decision, canonical_state)
    per_invariant_status[INVARIANT_NO_SIGNAL_LOSS] = status
    per_invariant_rationale[INVARIANT_NO_SIGNAL_LOSS] = rationale

    status, rationale = _check_no_rediscovery(main_thread_decision, canonical_state)
    per_invariant_status[INVARIANT_NO_REDISCOVERY] = status
    per_invariant_rationale[INVARIANT_NO_REDISCOVERY] = rationale

    status, rationale = _check_no_duplicate_code(main_thread_decision)
    per_invariant_status[INVARIANT_NO_DUPLICATE_CODE] = status
    per_invariant_rationale[INVARIANT_NO_DUPLICATE_CODE] = rationale

    status, rationale = _check_no_drift(main_thread_decision)
    per_invariant_status[INVARIANT_NO_DRIFT] = status
    per_invariant_rationale[INVARIANT_NO_DRIFT] = rationale

    violated = tuple(
        sorted(
            inv
            for inv, s in per_invariant_status.items()
            if s == "fail"
        )
    )

    if violated:
        overall_status = InvariantValidationStatus.FAIL_ONE_OR_MORE_INVARIANTS_VIOLATED
        rationale_summary = (
            f"FAIL: {len(violated)} of 5 canonical invariants violated: "
            f"{list(violated)}. Per CLAUDE.md 'Bugs must be permanently fixed "
            "AND self-protected against': route through canonical "
            "apparatus surfaces (3-metric trichotomy / canonical equations "
            "registry / anti-pattern registry / canonical helper) per "
            "per-invariant canonical_unwind_path."
        )
    elif any(s == "unknown" for s in per_invariant_status.values()):
        overall_status = InvariantValidationStatus.UNKNOWN_INSUFFICIENT_EVIDENCE
        unknowns = [
            inv for inv, s in per_invariant_status.items() if s == "unknown"
        ]
        rationale_summary = (
            f"UNKNOWN: {len(unknowns)} of 5 canonical invariants lack "
            f"sufficient evidence: {sorted(unknowns)}. Decision must declare "
            "the canonical invariant fields explicitly."
        )
    else:
        overall_status = InvariantValidationStatus.PASS_ALL_INVARIANTS_HONORED
        rationale_summary = (
            "PASS: all 5 canonical invariants honored per "
            "meta_orchestrator_three_metric_trichotomy_orthogonality_v1 + "
            "meta_orchestrator_cap_window_discipline_per_turn_variance_amortization_v1 "
            "canonical equations."
        )

    return InvariantValidationVerdict(
        status=overall_status,
        per_invariant_status=per_invariant_status,
        per_invariant_rationale=per_invariant_rationale,
        violated_invariants=violated,
        rationale=rationale_summary,
    )


__all__ = [
    "INVARIANT_NO_AD_HOC",
    "INVARIANT_NO_DRIFT",
    "INVARIANT_NO_DUPLICATE_CODE",
    "INVARIANT_NO_REDISCOVERY",
    "INVARIANT_NO_SIGNAL_LOSS",
    "InvariantValidationStatus",
    "InvariantValidationVerdict",
    "VALID_INVARIANTS",
    "validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants",
]
