# SPDX-License-Identifier: MIT
"""Cathedral consumer: anti-pattern lookup for compound stack proposal validation.

Per operator NON-NEGOTIABLE 2026-05-28 verbatim: *"learning anti-patterns is
upser important too for compounding continual learning, like the canonical
equations bu netgative and a higher layer of abstraction"* + canonical
anti-patterns design memo (.omx/research/canonical_anti_patterns_registry_design_20260528.md
commit 37b5a0184).

Sister of ``canonical_equation_lookup_consumer`` at the NEGATIVE registry
surface. Where the canonical_equation_lookup_consumer surfaces positive
prediction annotations on candidates, THIS consumer surfaces matched anti-
patterns + their canonical_unwind_path recommendations so the cathedral
autopilot ranker can STEER next-cycle attack direction AWAY from known
recurrences.

Hook coverage per Catalog #125:
  * #2 Pareto constraint — PRIMARY (anti-patterns EXCLUDE polytope feasibility
    regions; Wave N+2 Slot 1 Dykstra solver consumes as ACTIVE constraints)
  * #6 probe-disambiguator — matched anti-pattern + canonical unwind path IS
    the canonical disambiguator between speculative-recurrence vs measured-
    falsification

Tier per Catalog #357: TIER_A_OBSERVABILITY_ONLY (the consumer surfaces
observability-only verdicts; ``predicted_delta_adjustment=0.0`` always).
Routing recommendation IS the contribution; promotion stays gated by
empirically-grounded paired-axis evidence per CLAUDE.md "Submission auth
eval — BOTH CPU AND CUDA" non-negotiable.

Canonical routing markers per Catalog #341:
  * ``predicted_delta_adjustment=0.0``
  * ``promotable=False``
  * ``axis_tag="[predicted]"``

Per CLAUDE.md "Forbidden premature KILL": a matched anti-pattern is NOT a
KILL verdict on the candidate's technique. It surfaces the canonical
unwind path so the operator can choose:
  (a) Apply the canonical unwind path (canonical correct alternative)
  (b) Add ``# ANTI_PATTERN_MATCH_INTENTIONAL_OK:<rationale>`` waiver
  (c) Append empirical falsification that explicitly RATIFIES the
      anti-pattern in this context
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import ConsumerTier, HookNumber


CONSUMER_NAME = "anti_pattern_lookup_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.PARETO_CONSTRAINT,
    HookNumber.PROBE_DISAMBIGUATOR,
)
CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    When a new empirical anchor lands in the canonical continual-learning
    posterior, the canonical anti-patterns registry's
    ``auto_recalibrate_from_continual_learning_posterior`` is the canonical
    refresh path. This consumer is structurally NO-OP here — refresh
    happens via the operator-triggered recalibrator OR via direct
    ``append_empirical_falsification`` calls from the producer surface.

    Per Catalog #287/#323: we do not synthesize falsifications from
    arbitrary posterior anchors; the falsification must be explicitly
    appended via the canonical helper with full Provenance.
    """
    _ = anchor  # explicit acknowledgment; no state to update


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — match candidate's proposed stack against anti-patterns.

    Looks up registered anti-patterns whose ``recurrence_conditions`` or
    ``forbidden_pattern_predicate`` match the candidate's ``stack_spec``
    (best-effort substring + structural match via
    ``match_stack_against_anti_patterns``).

    Returns canonical routing markers per Catalog #341 + the matched
    anti-patterns + their recommended canonical unwind paths.

    The candidate's stack_spec is read from one of:
      * ``candidate["stack_spec"]`` (canonical key)
      * ``candidate["proposed_stack_spec"]`` (sister synonym)
      * the candidate dict itself if neither is present (whole-dict match)

    Returns dict with:
      * ``predicted_delta_adjustment`` = 0.0 (Catalog #341 Tier A)
      * ``promotable`` = False (Catalog #127/#192/#317)
      * ``axis_tag`` = "[predicted]" (Catalog #287/#323)
      * ``rationale`` = human-readable summary
      * ``matched_anti_patterns`` = tuple of matched anti_pattern_id strings
      * ``canonical_unwind_paths_recommended`` = tuple of unwind path strings
      * ``match_count`` = int
      * ``highest_severity`` = severity of first match OR None
      * ``provenance`` = Catalog #323 canonical Provenance dict
    """
    try:
        from tac.canonical_anti_patterns import match_stack_against_anti_patterns
    except ImportError:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": "canonical_anti_patterns registry unavailable [predicted]",
            "axis_tag": "[predicted]",
            "promotable": False,
            "matched_anti_patterns": (),
            "canonical_unwind_paths_recommended": (),
            "match_count": 0,
            "highest_severity": None,
            "provenance": {
                "kind": "ANTI_PATTERN_LOOKUP_CONSUMER_VERDICT",
                "consumer_name": CONSUMER_NAME,
                "consumer_version": CONSUMER_VERSION,
                "score_claim": False,
                "evidence_grade": "predicted",
            },
        }

    # Resolve stack_spec from canonical / sister-synonym / fallback positions
    stack_spec: Mapping[str, Any]
    if isinstance(candidate.get("stack_spec"), Mapping):
        stack_spec = candidate["stack_spec"]
    elif isinstance(candidate.get("proposed_stack_spec"), Mapping):
        stack_spec = candidate["proposed_stack_spec"]
    else:
        stack_spec = candidate  # fall through to whole-candidate match

    try:
        matches = match_stack_against_anti_patterns(stack_spec)
    except Exception:  # noqa: BLE001 - defensive; never crash the ranker
        matches = ()

    matched_ids = tuple(m.anti_pattern.anti_pattern_id for m in matches)
    unwind_paths = tuple(m.canonical_unwind_path_recommended for m in matches)
    highest_sev = matches[0].anti_pattern.severity if matches else None

    if not matches:
        rationale = (
            "no canonical anti-pattern matches this candidate; "
            "observability-only annotation [predicted]"
        )
    else:
        ids_str = ", ".join(matched_ids[:3])
        if len(matched_ids) > 3:
            ids_str += f" + {len(matched_ids) - 3} more"
        rationale = (
            f"{len(matches)} canonical anti-pattern(s) match this candidate's "
            f"proposed stack: {ids_str}. Highest severity: {highest_sev}. "
            f"Consult tools/list_canonical_anti_patterns.py for canonical "
            f"unwind paths. [predicted]"
        )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "matched_anti_patterns": matched_ids,
        "canonical_unwind_paths_recommended": unwind_paths,
        "match_count": len(matches),
        "highest_severity": highest_sev,
        "provenance": {
            "kind": "ANTI_PATTERN_LOOKUP_CONSUMER_VERDICT",
            "consumer_name": CONSUMER_NAME,
            "consumer_version": CONSUMER_VERSION,
            "consumer_tier": str(CONSUMER_TIER.name),
            "score_claim": False,
            "evidence_grade": "predicted",
            "captured_at_utc_source": "tac.cathedral_consumers.anti_pattern_lookup_consumer",
        },
    }
