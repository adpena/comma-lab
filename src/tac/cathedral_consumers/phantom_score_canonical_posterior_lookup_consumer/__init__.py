# SPDX-License-Identifier: MIT
"""Cathedral consumer: phantom-score canonical posterior lookup (Catalog #382 sister).

Per operator BINDING META-META question 2026-05-29 ~13:55Z America/Chicago verbatim:

    "why do we keep having the phantom score artifacts issue? seems like our
     current approach is not optimal to permanent self protecting and fixing
     and also recovering and continuing"

Sister of ``canonical_equation_lookup_consumer`` + ``anti_pattern_lookup_consumer``
at the READ surface (where the equation + anti-pattern consumers surface
prediction + anti-pattern annotations, THIS consumer surfaces phantom-score
verdicts for candidates citing canonical posterior tokens whose latest event
flips to FALSIFIED / KILLED / PHANTOM / INVALIDATED).

Catalog #335 canonical contract auto-discovered. Tier A canonical-routing
markers per Catalog #341 (predicted_delta_adjustment=0.0 + promotable=False
+ axis_tag=[predicted]) — observability-only; no candidate score mutation.

Hook coverage per Catalog #125:
  * #4 cathedral autopilot dispatch — PRIMARY (surfaces phantom-score
    verdict to ranker so candidates citing FALSIFIED tokens are visible)
  * #5 continual-learning posterior — ACTIVE (READS canonical posterior;
    auto-discovered consumer surfaces per-candidate verdict every iteration)
  * #6 probe-disambiguator — ACTIVE (4-state verdict + matched anchor IS
    the canonical disambiguator between current-CLEAN vs phantom-recurrence)
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.cathedral.consumer_contract import ConsumerTier, HookNumber

CONSUMER_NAME = "phantom_score_canonical_posterior_lookup_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)
CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    When a new empirical anchor lands in the canonical continual-learning
    posterior, the canonical posterior read validator's READ-surface query
    surfaces the verdict on next ``consume_candidate`` invocation
    automatically (latest-event-wins semantics; no caching). This consumer
    is structurally NO-OP at update time — refresh happens at READ time.

    Per Catalog #287/#323: we do not synthesize verdicts from arbitrary
    posterior anchors; the verdict is queried from canonical helper APIs.
    """
    _ = anchor  # explicit acknowledgment; no state to update


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — surface phantom-score verdict for candidate.

    Extracts canonical-posterior tokens from candidate's known fields
    (``substrate`` / ``lane_id`` / ``archive_sha`` / ``probe_id`` /
    ``equation_id`` / ``anti_pattern_id`` / etc.) and queries the canonical
    posterior read validator per token.

    Returns canonical routing markers per Catalog #341 (Tier A) + the
    per-token verdicts + aggregate verdict.

    Per Catalog #341: ``predicted_delta_adjustment=0.0`` + ``promotable=False``
    + ``axis_tag="[predicted]"`` always. The consumer is observability-only;
    it does NOT mutate candidate score. The cathedral autopilot ranker can
    consume the surfaced verdict to deprioritize candidates citing phantom
    tokens but the ranker decision is gated by the canonical 3-metric
    trichotomy per Catalog #379.
    """
    try:
        from tac.canonical_posterior_read_validator import (
            validate_memo_claim_against_canonical_posterior,
        )
    except ImportError:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": "canonical_posterior_read_validator unavailable [predicted]",
            "axis_tag": "[predicted]",
            "promotable": False,
            "matched_phantom_tokens": (),
            "match_count": 0,
            "provenance": {
                "kind": "PHANTOM_SCORE_LOOKUP_CONSUMER_VERDICT",
                "consumer_name": CONSUMER_NAME,
                "consumer_version": CONSUMER_VERSION,
                "score_claim": False,
                "evidence_grade": "predicted",
            },
        }

    # Extract candidate tokens to query
    candidate_tokens: list[str] = []
    for field_name in (
        "substrate",
        "lane_id",
        "probe_id",
        "equation_id",
        "anti_pattern_id",
        "archive_sha",
        "archive_sha256",
        "claim_token",
        "candidate_id",
        "candidate_name",
        "name",
        "id",
    ):
        val = candidate.get(field_name)
        if val and isinstance(val, str) and val.strip():
            candidate_tokens.append(val)

    # If no canonical tokens found, return clean observability annotation
    if not candidate_tokens:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "no canonical-posterior token extractable from candidate; "
                "observability-only annotation [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "matched_phantom_tokens": (),
            "match_count": 0,
            "provenance": _build_provenance(),
        }

    # Query validator for each token
    matched_phantom: list[dict[str, Any]] = []
    for token in candidate_tokens:
        try:
            verdict = validate_memo_claim_against_canonical_posterior("", token)
        except Exception:
            continue
        if verdict.is_blocking:
            matched_phantom.append(
                {
                    "claim_token": token,
                    "verdict": verdict.verdict.value,
                    "matched_anchor_id": verdict.matched_anchor_id,
                    "matched_anchor_source": verdict.matched_anchor_source,
                    "matched_anchor_summary": verdict.matched_anchor_summary[:200],
                }
            )

    if not matched_phantom:
        rationale = (
            f"none of {len(candidate_tokens)} candidate token(s) match a "
            "canonical posterior blocking verdict (FALSIFIED / KILLED / "
            "PHANTOM / INVALIDATED); observability-only [predicted]"
        )
    else:
        first_match = matched_phantom[0]
        rationale = (
            f"{len(matched_phantom)} candidate token(s) match canonical "
            f"posterior BLOCKING verdict(s); first match: "
            f"{first_match['claim_token']!r} verdict={first_match['verdict']} "
            f"anchor={first_match['matched_anchor_id']} per Catalog #382 "
            "READ-surface canonical posterior validator [predicted]"
        )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "matched_phantom_tokens": tuple(m["claim_token"] for m in matched_phantom),
        "match_count": len(matched_phantom),
        "matched_phantom_details": tuple(matched_phantom),
        "provenance": _build_provenance(),
    }


def _build_provenance() -> Mapping[str, Any]:
    """Build canonical Provenance per Catalog #323 for consumer output."""
    return {
        "kind": "PHANTOM_SCORE_LOOKUP_CONSUMER_VERDICT",
        "consumer_name": CONSUMER_NAME,
        "consumer_version": CONSUMER_VERSION,
        "consumer_tier": str(CONSUMER_TIER.name),
        "score_claim": False,
        "evidence_grade": "predicted",
        "captured_at_utc_source": (
            "tac.cathedral_consumers.phantom_score_canonical_posterior_lookup_consumer"
        ),
    }
