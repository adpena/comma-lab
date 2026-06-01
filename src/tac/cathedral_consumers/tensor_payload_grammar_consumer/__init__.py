# SPDX-License-Identifier: MIT
"""Cathedral consumer for generic tensor payload grammar reports.

This consumes ``tensor_payload_grammar_optimizer.v1`` reports emitted by the
substrate-agnostic packet compiler.  It closes the loop between MLX/PyTorch/NPZ
export profiling and the autopilot: saturated tensor payloads become negative
rate posterior, while unsaturated payloads route to receiver/archive binding.

The consumer is deliberately Tier-A observability-only.  It never promotes a
score claim and never dispatches exact auth; it only preserves the rate-axis
signal so future substrates inherit the optimal grammar gate automatically.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tac.archive_byte_profile import contest_rate_term
from tac.cathedral.consumer_contract import HookNumber
from tac.packet_compiler.tensor_payload_grammar_optimizer import (
    TENSOR_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA,
)

CONSUMER_NAME = "tensor_payload_grammar_consumer"
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
_SATURATED_STATUSES = frozenset({"entropy_saturated", "weak_entropy_gap"})
_UNSATURATED_STATUS = "unsaturated_entropy_gap"


def update_from_anchor(anchor: Any) -> None:
    """Hook #5 placeholder: optimizer reports already carry posterior hooks."""

    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Consume a generic tensor grammar report as planning-only rate signal."""

    blockers = _authority_blockers(candidate)
    if candidate.get("schema") != TENSOR_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA:
        blockers.append("tensor_payload_grammar_schema_mismatch")

    byte_accounting = _mapping(candidate.get("byte_accounting"))
    grouped_diagnostic = _mapping(candidate.get("grouped_brotli_order_diagnostic"))
    saturation = _mapping(candidate.get("saturation_diagnostic"))
    planner_feedback = _mapping(candidate.get("planner_feedback"))
    source_manifest = _mapping(candidate.get("source_payload_manifest"))
    report_blockers = _string_list(candidate.get("blockers"))
    for blocker in report_blockers:
        if blocker not in blockers:
            blockers.append(blocker)

    saved = max(0, _safe_int(byte_accounting.get("selected_saved_bytes_vs_baseline")))
    grouped_saved = max(
        0,
        _safe_int(
            grouped_diagnostic.get("grouped_saved_bytes_vs_selected_isolated")
        ),
    )
    grouped_delta = _safe_int(
        grouped_diagnostic.get("grouped_delta_bytes_vs_selected_isolated")
    )
    grouped_saved_vs_identity = max(
        0,
        _safe_int(grouped_diagnostic.get("grouped_saved_bytes_vs_identity")),
    )
    grouped_delta_vs_identity = _safe_int(
        grouped_diagnostic.get("grouped_delta_bytes_vs_identity")
    )
    grouped_bytes = _safe_int(grouped_diagnostic.get("selected_grouped_brotli_bytes"))
    selected_bytes = _safe_int(byte_accounting.get("selected_isolated_tensor_bytes"))
    baseline_bytes = _safe_int(byte_accounting.get("baseline_isolated_tensor_bytes"))
    ratio = _optional_float(byte_accounting.get("selected_over_floor_ratio"))
    status = str(saturation.get("status") or "unknown")
    operation_count = _safe_int(planner_feedback.get("operation_hint_count"))
    rate_positive_count = _safe_int(planner_feedback.get("rate_positive_hint_count"))
    tensor_count = _safe_int(candidate.get("tensor_count") or source_manifest.get("tensor_count"))

    isolated_receiver_work = status == _UNSATURATED_STATUS and saved > 0
    grouped_receiver_work = grouped_saved > 0
    receiver_work_justified = isolated_receiver_work or grouped_receiver_work
    demotion_recommended = status in _SATURATED_STATUSES and not grouped_receiver_work
    if grouped_receiver_work:
        planner_action = "bind_substrate_receiver_and_materialize_grouped_brotli_archive"
    elif receiver_work_justified:
        planner_action = "bind_substrate_receiver_and_materialize_byte_closed_archive"
    elif demotion_recommended:
        planner_action = "record_tensor_payload_saturation_and_demote_format_churn"
    elif operation_count <= 0:
        planner_action = "rerun_tensor_payload_grammar_with_valid_tensor_export"
    else:
        planner_action = "inspect_tensor_entropy_gap_before_receiver_work"

    return {
        "schema": "tensor_payload_grammar_consumer_result.v1",
        "consumer_name": CONSUMER_NAME,
        "campaign_id": str(candidate.get("campaign_id") or ""),
        "source_schema": candidate.get("schema"),
        "source_kind": source_manifest.get("source_kind"),
        "tensor_count": tensor_count,
        "saturation_status": status,
        "selected_over_floor_ratio": ratio,
        "selected_isolated_tensor_bytes": selected_bytes,
        "baseline_isolated_tensor_bytes": baseline_bytes,
        "selected_saved_bytes_vs_baseline": saved,
        "rate_delta_score_if_components_unchanged": contest_rate_term(-saved),
        "grouped_selected_brotli_bytes": grouped_bytes,
        "grouped_saved_bytes_vs_identity": grouped_saved_vs_identity,
        "grouped_delta_bytes_vs_identity": grouped_delta_vs_identity,
        "grouped_saved_bytes_vs_selected_isolated": grouped_saved,
        "grouped_delta_bytes_vs_selected_isolated": grouped_delta,
        "grouped_rate_delta_score_if_components_unchanged": contest_rate_term(
            grouped_delta
        ),
        "operation_hint_count": operation_count,
        "rate_positive_hint_count": rate_positive_count,
        "grouped_receiver_work_justified": grouped_receiver_work,
        "receiver_work_justified": receiver_work_justified,
        "demotion_recommended": demotion_recommended,
        "planner_action": planner_action,
        "predicted_delta_adjustment": 0.0,
        "rationale": _rationale(
            status=status,
            saved=saved,
            grouped_saved=grouped_saved,
            receiver_work_justified=receiver_work_justified,
            demotion_recommended=demotion_recommended,
        ),
        "axis_tag": "[planning-only byte-profile]",
        "promotable": False,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "confidence": 0.0 if blockers else (0.5 if receiver_work_justified else 0.25),
        "blockers": blockers,
    }


def _authority_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for field in _AUTHORITY_FIELDS:
        if payload.get(field) is True:
            blockers.append(f"{field}_overclaimed")
    return blockers


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _rationale(
    *,
    status: str,
    saved: int,
    grouped_saved: int,
    receiver_work_justified: bool,
    demotion_recommended: bool,
) -> str:
    if grouped_saved > 0:
        return (
            "Generic tensor payload grammar found grouped Brotli ordering savings "
            f"({grouped_saved} byte(s) versus selected isolated tensor payloads); "
            "bind a substrate receiver/archive before replay."
        )
    if receiver_work_justified:
        return (
            "Generic tensor payload grammar has an unsaturated entropy gap and "
            f"{saved} isolated byte(s) of planning-only savings; route to a "
            "substrate receiver/archive materializer before replay."
        )
    if demotion_recommended:
        return (
            "Generic tensor payload grammar is saturated or weak-gap "
            f"(status={status}, saved={saved}); preserve as negative rate "
            "posterior and avoid repeated within-substrate format churn."
        )
    return (
        "Generic tensor payload grammar report is incomplete or inconclusive; "
        "finish valid tensor export profiling before receiver work."
    )


__all__ = [
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "consume_candidate",
    "update_from_anchor",
]
