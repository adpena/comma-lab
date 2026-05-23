# SPDX-License-Identifier: MIT
"""Cathedral consumer for PacketIR candidate queues.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
This closes the producer-to-consumer loop for PR101/FEC6 PacketIR queue
artifacts without promoting queue rows into score authority.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.cathedral.consumer_contract import HookNumber
from tac.packet_compiler.pr101_fec6_candidate_queue import (
    PR101_FEC6_CANDIDATE_QUEUE_SCHEMA,
)

CONSUMER_NAME = "packetir_candidate_queue_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.PROBE_DISAMBIGUATOR,
)

_AUTHORITY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
    "ready_for_operator_probe",
    "ready_for_provider_dispatch",
    "dispatch_attempted",
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 noop: queue rows are byte-custody artifacts."""
    _ = anchor


def _authority_blockers(
    payload: Mapping[str, Any],
    *,
    allow_score_claim: bool,
) -> list[str]:
    blockers: list[str] = []
    for field in _AUTHORITY_FIELDS:
        if field == "score_claim" and allow_score_claim:
            continue
        if payload.get(field) is True:
            blockers.append(f"{field}_overclaimed")
    return blockers


_RUNTIME_PROVEN_BLOCKER = "runtime_byte_consumption_noop_detector_missing"


def consume_candidate(
    candidate: Mapping[str, Any],
    *,
    allow_score_claim: bool = False,
    runtime_consumption_proven_required: bool = True,
) -> Mapping[str, Any]:
    """Consume one PacketIR queue candidate as observability-only signal.

    Per codex adversarial review 2026-05-19: candidate verdicts default to
    ``runtime_consumption_proven_required=True``; when the candidate's blockers
    list still names ``runtime_byte_consumption_noop_detector_missing`` the
    verdict surfaces ``runtime_consumption_proven=False`` so downstream cathedral
    autopilot rerankers cannot promote unproved bytes to eligible.
    """

    blockers = _authority_blockers(
        candidate,
        allow_score_claim=allow_score_claim,
    )
    candidate_blockers = candidate.get("blockers")
    if isinstance(candidate_blockers, list):
        blockers.extend(str(item) for item in candidate_blockers)
    surfaces = candidate.get("consumer_surfaces")
    if not isinstance(surfaces, list) or not surfaces:
        blockers.append("consumer_surfaces_missing")
        surfaces = []
    runtime_proven = candidate.get("runtime_consumption_proven")
    if runtime_consumption_proven_required and runtime_proven is not True:
        if _RUNTIME_PROVEN_BLOCKER not in blockers:
            blockers.append(_RUNTIME_PROVEN_BLOCKER)
        if "runtime_consumption_proven_required_but_false" not in blockers:
            blockers.append("runtime_consumption_proven_required_but_false")
    candidate_id = str(candidate.get("candidate_id") or "unknown_packetir_candidate")
    rationale = (
        "tac.packet_compiler PacketIR candidate queue consumed as "
        "observability-only [predicted] signal; materialization, runtime "
        "byte-consumption proof (Catalog #105 no-op detector), and paired "
        "contest exact eval remain required"
    )
    return {
        "candidate_id": candidate_id,
        "eligible": False,
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0 if blockers else 0.25,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "runtime_consumption_proven": runtime_proven is True,
        "consumer_surfaces": list(surfaces),
        "blockers": blockers,
    }


def consume_queue(
    queue: Mapping[str, Any],
    *,
    runtime_consumption_proven_required: bool = True,
) -> Mapping[str, Any]:
    """Consume a PacketIR queue artifact and return per-candidate verdicts."""

    blockers: list[str] = []
    if queue.get("schema") != PR101_FEC6_CANDIDATE_QUEUE_SCHEMA:
        blockers.append("queue_schema_mismatch")
    blockers.extend(_authority_blockers(queue, allow_score_claim=False))
    queue_blockers = queue.get("blockers")
    if isinstance(queue_blockers, list):
        for entry in queue_blockers:
            text = str(entry)
            if text not in blockers:
                blockers.append(text)
    accounting = queue.get("byte_accounting")
    queue_runtime_proven = False
    if isinstance(accounting, Mapping):
        queue_runtime_proven = accounting.get("runtime_consumption_proven") is True
        if runtime_consumption_proven_required and not queue_runtime_proven:
            if _RUNTIME_PROVEN_BLOCKER not in blockers:
                blockers.append(_RUNTIME_PROVEN_BLOCKER)
            if "runtime_consumption_proven_required_but_false" not in blockers:
                blockers.append("runtime_consumption_proven_required_but_false")
    candidates = queue.get("candidates")
    if not isinstance(candidates, list):
        blockers.append("queue_candidates_missing")
        candidates = []
    verdicts = [
        consume_candidate(
            candidate,
            runtime_consumption_proven_required=runtime_consumption_proven_required,
        )
        for candidate in candidates
        if isinstance(candidate, Mapping)
    ]
    if len(verdicts) != len(candidates):
        blockers.append("queue_contains_non_object_candidate")
    return {
        "schema": "packetir_candidate_queue_consumer_result_v2",
        "consumer_name": CONSUMER_NAME,
        "queue_schema": queue.get("schema"),
        "candidate_count": len(candidates),
        "consumed_candidate_count": len(verdicts),
        "eligible_candidate_count": 0,
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.cathedral_consumers.packetir_candidate_queue_consumer "
            "registered PacketIR queue rows for [predicted] observability only"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0 if blockers else 0.25,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "queue_runtime_consumption_proven": queue_runtime_proven,
        "blockers": blockers,
        "candidate_verdicts": verdicts,
    }


__all__ = [
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "consume_candidate",
    "consume_queue",
    "update_from_anchor",
]
