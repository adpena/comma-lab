# SPDX-License-Identifier: MIT
"""Cathedral consumer for the SNeRV/HiNeRV top-priority stack packet."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tac.cathedral.consumer_contract import HookNumber

CONSUMER_NAME = "nerv_top_priority_stack_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Hook #5 placeholder; bridge payloads carry their own blocker context."""

    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Consume the normalized NeRV master-consumer packet as routing signal."""

    blockers = _string_list(candidate.get("blockers"))
    if candidate.get("schema") != "nerv_master_consumer_bridge.v1":
        blockers.append("nerv_master_consumer_bridge_schema_missing")
    normalized = candidate.get("normalized_candidate")
    if not isinstance(normalized, Mapping):
        blockers.append("normalized_candidate_missing")
        normalized = {}
    unit_count = len(_mapping_list(candidate.get("master_consumer_units")))
    route_count = len(_mapping_list(candidate.get("master_consumer_routes")))
    if unit_count == 0:
        blockers.append("master_consumer_units_missing")
    if route_count == 0:
        blockers.append("master_consumer_routes_missing")

    source_parity_blocked = any("official" in blocker for blocker in blockers)
    receiver_blocked = any("receiver" in blocker for blocker in blockers)
    rate_blocked = any(
        token in blocker for blocker in blockers for token in ("byte", "modelsize")
    )
    if source_parity_blocked:
        planner_action = "route_to_source_faithful_snerv_hinerv_parity"
    elif receiver_blocked or rate_blocked:
        planner_action = "route_to_receiver_bytes_and_rate_grammar"
    elif blockers:
        planner_action = "hold_exact_dispatch_and_close_stack_blockers"
    else:
        planner_action = "route_to_next_local_training_smoke"

    return {
        "schema": "nerv_top_priority_stack_consumer_result.v1",
        "consumer_name": CONSUMER_NAME,
        "source_schema": candidate.get("schema"),
        "top_priority_carriers": list(candidate.get("top_priority_carriers") or []),
        "baseline_to_beat": candidate.get("baseline_to_beat"),
        "unit_count": unit_count,
        "route_count": route_count,
        "planner_action": planner_action,
        "source_parity_blocked": source_parity_blocked,
        "receiver_blocked": receiver_blocked,
        "rate_blocked": rate_blocked,
        "predicted_delta_adjustment": 0.0,
        "axis_tag": "[planning/control]",
        "promotable": False,
        "score_claim": False,
        "score_claim_valid": False,
        "frontier_score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "confidence": 0.0 if blockers else 0.25,
        "blockers": _unique(blockers),
    }


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


__all__ = [
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "consume_candidate",
    "update_from_anchor",
]
