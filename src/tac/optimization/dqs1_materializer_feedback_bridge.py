# SPDX-License-Identifier: MIT
"""Bridge family-agnostic materializer feedback into DQS1 follow-up planning."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from tac.optimization.materializer_feedback import materializer_observation_feedback_rows
from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    ordered_unique,
    require_no_truthy_authority_fields,
)

SCHEMA = "dqs1_materializer_feedback_bridge.v1"
TOOL = "tac.optimization.dqs1_materializer_feedback_bridge"

FALSE_AUTHORITY: dict[str, bool] = {
    **PROXY_FALSE_AUTHORITY_FIELDS,
    "score_claim_valid": False,
    "score_claim_eligible": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "exact_cuda_auth_eval": False,
    "contest_cuda_auth_eval": False,
}


def _string_list(values: Sequence[Any] | None) -> list[str]:
    return [str(item) for item in values or [] if str(item)]


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _receiver_satisfied(row: Mapping[str, Any]) -> bool:
    return (
        row.get("receiver_contract_satisfied") is True
        or row.get("inflate_parity_satisfied") is True
    )


def _rate_positive(row: Mapping[str, Any]) -> bool:
    return row.get("rate_positive") is True and _safe_int(row.get("saved_bytes")) > 0


def _target_kind(row: Mapping[str, Any]) -> str:
    return str(row.get("target_kind") or "unknown_materializer_target")


def _demotion_reason(group: Mapping[str, Any]) -> str:
    if group["receiver_positive_rate_saving_count"] > 0:
        return ""
    if group["rate_positive_count"] > 0:
        return "rate_saving_without_receiver_contract"
    if group["receiver_positive_no_delta_count"] > 0:
        return "receiver_contract_satisfied_but_no_archive_delta"
    if group["receiver_negative_count"] > 0:
        return "receiver_contract_unsatisfied"
    return "no_rate_positive_materializer_observation"


def _target_group_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    blockers: dict[str, list[str]] = defaultdict(list)
    actions: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        target = _target_kind(row)
        group = grouped.setdefault(
            target,
            {
                "target_kind": target,
                "observation_count": 0,
                "saved_bytes_sum": 0,
                "rate_positive_count": 0,
                "receiver_positive_rate_saving_count": 0,
                "receiver_positive_no_delta_count": 0,
                "receiver_negative_count": 0,
                "manifest_paths": [],
                **FALSE_AUTHORITY,
            },
        )
        saved = _safe_int(row.get("saved_bytes"))
        receiver_ok = _receiver_satisfied(row)
        rate_positive = _rate_positive(row)
        group["observation_count"] += 1
        group["saved_bytes_sum"] += saved
        if rate_positive:
            group["rate_positive_count"] += 1
        if receiver_ok and rate_positive:
            group["receiver_positive_rate_saving_count"] += 1
        elif receiver_ok:
            group["receiver_positive_no_delta_count"] += 1
        else:
            group["receiver_negative_count"] += 1
        manifest_path = str(row.get("manifest_path") or row.get("source_path") or "")
        if manifest_path:
            group["manifest_paths"].append(manifest_path)
        blockers[target].extend(str(item) for item in row.get("readiness_blockers") or [])
        action = str(row.get("recommended_planner_action") or "").strip()
        if action:
            actions[target].append(action)

    out: list[dict[str, Any]] = []
    for target, group in sorted(grouped.items()):
        reason = _demotion_reason(group)
        group["demote_for_dqs1_followup"] = bool(reason)
        group["demotion_reason"] = reason or None
        group["readiness_blockers"] = ordered_unique(blockers[target])
        group["recommended_planner_actions"] = ordered_unique(actions[target])
        group["manifest_paths"] = ordered_unique(group["manifest_paths"])
        out.append(group)
    return out


def _dqs1_candidate_rows(
    candidate_ids: Sequence[str],
    *,
    candidate_limit: int | None,
    dqs1_observations: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    observation_by_candidate = {
        str(row.get("candidate_id")): row
        for row in dqs1_observations
        if isinstance(row, Mapping) and row.get("candidate_id")
    }
    rows: list[dict[str, Any]] = []
    for rank, candidate_id in enumerate(_string_list(candidate_ids)):
        observation = observation_by_candidate.get(candidate_id, {})
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "planned_rank": rank,
            "source": "dqs1_local_first_queue_selection",
            **FALSE_AUTHORITY,
        }
        if observation:
            require_no_truthy_authority_fields(
                observation,
                context=f"dqs1_materializer_feedback_bridge.{candidate_id}",
            )
            row.update(
                {
                    "source": "dqs1_local_first_harvest_observation",
                    "family": observation.get("family"),
                    "observed_axis": observation.get("observed_axis"),
                    "score_delta_vs_baseline": observation.get(
                        "score_delta_vs_baseline"
                    ),
                    "archive_byte_delta_vs_baseline": observation.get(
                        "archive_byte_delta_vs_baseline"
                    ),
                    "selected_pair_indices": observation.get("selected_pair_indices"),
                    "component_deltas": observation.get("component_deltas"),
                    "source_artifact_path": observation.get("source_artifact_path"),
                }
            )
        rows.append(row)
    limit = candidate_limit if candidate_limit is not None and candidate_limit > 0 else None
    return rows[:limit]


def build_dqs1_materializer_feedback_bridge(
    *,
    materializer_feedback_payloads: Sequence[Mapping[str, Any]] = (),
    materializer_feedback_source_paths: Sequence[str] = (),
    planned_dqs1_candidate_ids: Sequence[str] = (),
    candidate_limit: int | None = None,
    dqs1_observations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any] | None:
    """Return a false-authority bridge from materializer feedback to DQS1 planning.

    The bridge is intentionally advisory. It records when generic archive/member
    transforms have become saturated and points the local-first queue at DQS1
    selector composition without pretending the local signal is contest score
    authority.
    """

    rows: list[dict[str, Any]] = []
    for index, payload in enumerate(materializer_feedback_payloads):
        if not isinstance(payload, Mapping):
            continue
        path = (
            str(materializer_feedback_source_paths[index])
            if index < len(materializer_feedback_source_paths)
            else None
        )
        rows.extend(
            materializer_observation_feedback_rows(
                payload,
                source_path=path,
            )
        )
    if not rows and not dqs1_observations:
        return None

    target_groups = _target_group_rows(rows)
    demoted_targets = [group for group in target_groups if group["demote_for_dqs1_followup"]]
    receiver_positive_targets = [
        group
        for group in target_groups
        if group["receiver_positive_rate_saving_count"] > 0
    ]
    dqs1_candidates = _dqs1_candidate_rows(
        planned_dqs1_candidate_ids,
        candidate_limit=candidate_limit,
        dqs1_observations=dqs1_observations,
    )
    if receiver_positive_targets:
        next_action = "materializer_receiver_positive_followup_before_dqs1_switch"
    elif demoted_targets and dqs1_candidates:
        next_action = "switch_to_dqs1_pairset_composition_followup"
    elif dqs1_candidates:
        next_action = "run_dqs1_pairset_composition_followup"
    else:
        next_action = "collect_more_materializer_or_dqs1_signal"

    return {
        "schema": SCHEMA,
        "producer": TOOL,
        "materializer_feedback_source_paths": _string_list(
            materializer_feedback_source_paths
        ),
        "materializer_observation_count": len(rows),
        "materializer_target_groups": target_groups,
        "demoted_materializer_target_kinds": [
            {
                "target_kind": group["target_kind"],
                "demotion_reason": group["demotion_reason"],
                "observation_count": group["observation_count"],
                "saved_bytes_sum": group["saved_bytes_sum"],
                **FALSE_AUTHORITY,
            }
            for group in demoted_targets
        ],
        "receiver_positive_rate_saving_target_kinds": [
            group["target_kind"] for group in receiver_positive_targets
        ],
        "planned_dqs1_candidate_count": len(dqs1_candidates),
        "planned_dqs1_candidates": dqs1_candidates,
        "recommended_next_action": next_action,
        "allowed_use": "local_dqs1_pairset_replanning_signal_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


__all__ = [
    "FALSE_AUTHORITY",
    "SCHEMA",
    "TOOL",
    "build_dqs1_materializer_feedback_bridge",
]
