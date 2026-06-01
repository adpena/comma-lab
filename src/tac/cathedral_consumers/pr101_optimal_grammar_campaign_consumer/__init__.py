# SPDX-License-Identifier: MIT
"""Cathedral consumer for PR101 optimal-grammar campaign summaries.

This consumer closes the loop between the deterministic packet compiler and the
cathedral/autopilot layer.  It consumes
``pr101_optimal_grammar_campaign_summary.v1`` payloads as planning signal:
archive-positive, receiver-compatible summaries can move to local replay gates;
archive-negative or entropy-saturated summaries become durable demotion
posterior instead of repeated format churn.  It never promotes score authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tac.cathedral.consumer_contract import HookNumber
from tac.packet_compiler.pr101_per_tensor_grammar_solver import (
    PR101_GRAMMAR_CAMPAIGN_SUMMARY_SCHEMA,
    contest_rate_term,
)

CONSUMER_NAME = "pr101_optimal_grammar_campaign_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
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

_REPLAY_READY_VERDICT = "grouped_positive_runtime_ready_for_local_replay_gate"
_DEMOTION_VERDICTS = frozenset(
    {
        "current_substrate_grammar_saturated",
        "isolated_gap_not_grouped_positive",
        "grouped_positive_consumed_by_archive_overhead",
        "grouped_positive_archive_materialized_but_receiver_incompatible",
    }
)


def update_from_anchor(anchor: Any) -> None:
    """Hook #5 placeholder: summary payloads already carry planner posterior."""

    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Consume a PR101 grammar campaign summary as false-authority signal.

    The returned row is intentionally Tier-A: it may route local follow-up, but
    predicted score adjustment remains zero until a byte-closed local replay and
    exact contest eval independently clear their gates.
    """

    blockers = _authority_blockers(candidate)
    if candidate.get("schema") != PR101_GRAMMAR_CAMPAIGN_SUMMARY_SCHEMA:
        blockers.append("campaign_summary_schema_mismatch")

    verdict = str(candidate.get("verdict") or "unknown")
    next_action = str(candidate.get("next_action") or "")
    rate_axis = _mapping(candidate.get("rate_axis"))
    artifact_status = _mapping(candidate.get("artifact_status"))
    planner_feedback = _mapping(candidate.get("planner_feedback"))
    candidate_blockers = _string_list(candidate.get("blockers"))
    for blocker in candidate_blockers:
        if blocker not in blockers:
            blockers.append(blocker)

    archive_delta = _optional_int(rate_axis.get("archive_zip_delta_bytes"))
    archive_rate_positive = rate_axis.get("archive_rate_positive") is True
    runtime_compatible = (
        artifact_status.get("runtime_tree_compatible_with_archive_layout") is True
    )
    local_replay_recommended = (
        verdict == _REPLAY_READY_VERDICT
        and archive_rate_positive
        and runtime_compatible
        and not _truthy_authority(candidate)
    )
    receiver_work_justified = (
        planner_feedback.get("receiver_adapter_work_justified") is True
    )
    demotion_recommended = verdict in _DEMOTION_VERDICTS
    if local_replay_recommended:
        planner_action = "run_full_frame_inflate_parity_and_local_replay_gate"
    elif receiver_work_justified:
        planner_action = "build_receiver_adapter_then_reconsume_campaign_summary"
    elif demotion_recommended:
        planner_action = "record_negative_rate_posterior_and_demote_format_churn"
    else:
        planner_action = "complete_grouped_archive_runtime_measurement"

    rate_delta_score_if_components_unchanged = (
        None if archive_delta is None else contest_rate_term(archive_delta)
    )
    return {
        "schema": "pr101_optimal_grammar_campaign_consumer_result.v1",
        "consumer_name": CONSUMER_NAME,
        "campaign_id": str(candidate.get("campaign_id") or ""),
        "source_schema": candidate.get("schema"),
        "verdict": verdict,
        "next_action": next_action,
        "planner_action": planner_action,
        "local_replay_recommended": local_replay_recommended,
        "receiver_adapter_work_justified": receiver_work_justified,
        "demotion_recommended": demotion_recommended,
        "grouped_positive": planner_feedback.get("grouped_positive") is True,
        "grammar_payoff_is_substrate_conditional": (
            planner_feedback.get("grammar_payoff_is_substrate_conditional") is True
        ),
        "candidate_archive_bytes_delta": archive_delta,
        "rate_delta_score_if_components_unchanged": rate_delta_score_if_components_unchanged,
        "predicted_delta_adjustment": 0.0,
        "rationale": _rationale(
            verdict=verdict,
            archive_delta=archive_delta,
            local_replay_recommended=local_replay_recommended,
            demotion_recommended=demotion_recommended,
        ),
        "axis_tag": "[planning-only byte-profile]",
        "promotable": False,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "confidence": 0.0 if blockers else (0.5 if local_replay_recommended else 0.25),
        "blockers": blockers,
    }


def _authority_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for field in _AUTHORITY_FIELDS:
        if payload.get(field) is True:
            blockers.append(f"{field}_overclaimed")
    return blockers


def _truthy_authority(payload: Mapping[str, Any]) -> bool:
    return any(payload.get(field) is True for field in _AUTHORITY_FIELDS)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rationale(
    *,
    verdict: str,
    archive_delta: int | None,
    local_replay_recommended: bool,
    demotion_recommended: bool,
) -> str:
    if local_replay_recommended:
        return (
            "Grammar campaign has archive-positive, receiver-compatible bytes; "
            "route to local full-frame replay before any exact auth dispatch."
        )
    if demotion_recommended:
        return (
            "Grammar campaign is saturated, incompatible, or consumed by legal "
            f"archive overhead (archive_delta={archive_delta}); preserve as "
            "negative rate posterior for future substrate grammar selection."
        )
    return (
        "Grammar campaign is incomplete; finish grouped packet, archive, and "
        "runtime evidence before replay or exact auth consideration."
    )


__all__ = [
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "consume_candidate",
    "update_from_anchor",
]
