# SPDX-License-Identifier: MIT
"""Entropy-stage chain contracts for repair campaign automation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_campaign_replay_bundle import (
    REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA,
)
from tac.optimization.repair_campaign_scorer import (
    REPAIR_CAMPAIGN_OPTIMIZER_DECISION_SCHEMA,
    REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
)

REPAIR_CAMPAIGN_ENTROPY_STAGE_CHAIN_CONTRACT_SCHEMA = "repair_campaign_entropy_stage_chain_contract.v1"
REPAIR_CAMPAIGN_ENTROPY_STAGE_CHAIN_NODE_SCHEMA = "repair_campaign_entropy_stage_chain_node.v1"


class RepairCampaignChainContractError(ValueError):
    """Raised when a repair campaign chain contract cannot be built."""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _repo_rel(path: str | Path | None, *, repo_root: str | Path | None) -> str | None:
    if path is None:
        return None
    value = Path(path)
    if repo_root is None:
        return value.as_posix()
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return value.as_posix()


def _entropy_pipeline(allocation: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = _mapping(allocation.get("entropy_pipeline_position"))
    if direct:
        return direct
    return _mapping(_mapping(allocation.get("multiscale_action_row")).get("entropy_pipeline_position"))


def _stage_index(allocation: Mapping[str, Any]) -> int:
    return _safe_int(_entropy_pipeline(allocation).get("stage_index"), default=999)


def _allocation_sort_key(allocation: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        _stage_index(allocation),
        _safe_int(allocation.get("campaign_rank"), default=999),
        str(allocation.get("typed_response_id") or allocation.get("candidate_id") or ""),
    )


def _stage_record(
    *,
    stage_id: str,
    required: bool,
    satisfied: bool,
    artifact_key: str,
    artifact_schema: str | None = None,
    blockers: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "required": required,
        "satisfied": bool(satisfied) if required else False,
        "artifact_key": artifact_key,
        "artifact_schema": artifact_schema,
        "blockers": ordered_unique(blockers),
    }


def _node_for_allocation(
    allocation: Mapping[str, Any],
    *,
    chain_order: int,
) -> dict[str, Any]:
    typed_response_id = str(allocation.get("typed_response_id") or "").strip()
    pipeline = _entropy_pipeline(allocation)
    lineage = _mapping(allocation.get("repair_materialization_lineage"))
    local_mlx_advisory = str(allocation.get("component_response_axis") or "") == "[macOS-MLX research-signal]"
    replay_bundle_path = str(
        allocation.get("stackability_replay_bundle_path") or allocation.get("deterministic_replay_bundle_path") or ""
    ).strip()
    replay_satisfied = bool(replay_bundle_path)
    archive_ready = lineage.get("byte_closed_candidate_archive_ready") is True
    runtime_proof_ready = lineage.get("archive_bound_runtime_consumption_proof_ready") is True
    component_replay_ready = lineage.get("component_response_replay_manifest_ready") is True
    exact_axis_ready = lineage.get("exact_axis_component_response_ready") is True
    exact_handoff_ready = lineage.get("materialization_ready_for_exact_axis_handoff") is True
    stages = [
        _stage_record(
            stage_id="planner_score_queue",
            required=True,
            satisfied=True,
            artifact_key="repair_campaign_score_report",
            artifact_schema=REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
        ),
        _stage_record(
            stage_id="local_mlx_deterministic_replay_bundle",
            required=local_mlx_advisory,
            satisfied=replay_satisfied,
            artifact_key="stackability_replay_bundle_path",
            artifact_schema=REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA,
            blockers=(
                ["deterministic_local_mlx_replay_bundle_required"]
                if local_mlx_advisory and not replay_satisfied
                else []
            ),
        ),
        _stage_record(
            stage_id="byte_closed_materializer",
            required=True,
            satisfied=archive_ready,
            artifact_key="receiver_consumed_candidate_archive_path",
            blockers=(["byte_closed_candidate_archive_missing_or_unverified"] if not archive_ready else []),
        ),
        _stage_record(
            stage_id="receiver_decode_only_runtime_proof",
            required=True,
            satisfied=runtime_proof_ready,
            artifact_key="runtime_consumption_proof_path",
            blockers=(
                ["archive_bound_runtime_consumption_proof_missing_or_unverified"] if not runtime_proof_ready else []
            ),
        ),
        _stage_record(
            stage_id="component_response_replay",
            required=True,
            satisfied=component_replay_ready,
            artifact_key="component_response_replay_manifest_path",
            blockers=(
                ["component_response_replay_manifest_missing_or_unverified"] if not component_replay_ready else []
            ),
        ),
        _stage_record(
            stage_id="exact_eval_handoff",
            required=True,
            satisfied=exact_axis_ready and exact_handoff_ready,
            artifact_key="exact_axis_component_response_path",
            blockers=(
                ["exact_axis_component_response_artifact_missing_or_unverified"]
                if not (exact_axis_ready and exact_handoff_ready)
                else []
            ),
        ),
    ]
    blockers = ordered_unique(blocker for stage in stages for blocker in _string_list(stage.get("blockers")))
    node = {
        "schema": REPAIR_CAMPAIGN_ENTROPY_STAGE_CHAIN_NODE_SCHEMA,
        "chain_order": chain_order,
        "typed_response_id": typed_response_id or None,
        "candidate_id": allocation.get("candidate_id"),
        "family_id": allocation.get("family_id"),
        "entropy_pipeline_position": dict(pipeline),
        "entropy_pipeline_stage_index": _stage_index(allocation),
        "entropy_position_label": allocation.get("entropy_position_label"),
        "chain_stage_sequence": stages,
        "local_mlx_advisory_row": local_mlx_advisory,
        "deterministic_replay_bundle_required": local_mlx_advisory,
        "deterministic_replay_bundle_satisfied": replay_satisfied,
        "receiver_decode_only_required": True,
        "exact_eval_handoff_ready": exact_axis_ready and exact_handoff_ready,
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_campaign_entropy_stage_chain_node_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        node,
        context=f"repair_campaign_entropy_stage_chain_node:{typed_response_id}",
    )
    return node


def build_repair_campaign_entropy_stage_chain_contract(
    *,
    score_report: Mapping[str, Any],
    score_report_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed chain compiler contract from a repair score report."""

    if score_report.get("schema") != REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA:
        raise RepairCampaignChainContractError("chain contract requires repair_campaign_score_report.v1")
    require_no_truthy_authority_fields(
        score_report,
        context="repair_campaign_entropy_stage_chain_contract_input",
    )
    decision = _mapping(score_report.get("optimizer_decision"))
    if decision.get("schema") != REPAIR_CAMPAIGN_OPTIMIZER_DECISION_SCHEMA:
        raise RepairCampaignChainContractError("score report missing repair_campaign_optimizer_decision.v1")
    allocations = sorted(
        [row for row in decision.get("selected_allocation_rows") or [] if isinstance(row, Mapping)],
        key=_allocation_sort_key,
    )
    nodes = [
        _node_for_allocation(allocation, chain_order=order) for order, allocation in enumerate(allocations, start=1)
    ]
    blockers = ordered_unique(
        [
            "local_mlx_chain_contract_is_not_score_authority",
            "exact_auth_eval_required_before_score_or_promotion_claim",
            *[blocker for node in nodes for blocker in _string_list(node.get("blockers"))],
        ]
    )
    stage_histogram: dict[str, int] = {}
    for node in nodes:
        stage = str(
            _mapping(node.get("entropy_pipeline_position")).get("entropy_position_class")
            or "unknown_entropy_pipeline_position"
        )
        stage_histogram[stage] = stage_histogram.get(stage, 0) + 1
    contract = {
        "schema": REPAIR_CAMPAIGN_ENTROPY_STAGE_CHAIN_CONTRACT_SCHEMA,
        "node_schema": REPAIR_CAMPAIGN_ENTROPY_STAGE_CHAIN_NODE_SCHEMA,
        "source_score_report_path": _repo_rel(score_report_path, repo_root=repo_root),
        "source_score_report_schema": score_report.get("schema"),
        "optimizer_decision_schema": decision.get("schema"),
        "selected_allocation_count": len(allocations),
        "chain_node_count": len(nodes),
        "ordered_entropy_pipeline_stage_histogram": dict(sorted(stage_histogram.items())),
        "ordered_chain_nodes": nodes,
        "required_downstream_queue_artifacts": [
            "repair_campaign_stackability_queue",
            "repair_campaign_byte_closed_materialization_queue",
            "receiver_decode_only_runtime_proof",
            "exact_eval_handoff",
        ],
        "local_mlx_rows_are_advisory_only": True,
        "deterministic_replay_bundle_required_for_local_mlx_rows": True,
        "receiver_decode_only_required": True,
        "exact_eval_handoff_requires_complete_archive_runtime_component_custody": True,
        "all_chain_nodes_exact_handoff_ready": bool(nodes)
        and all(node.get("exact_eval_handoff_ready") is True for node in nodes),
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_campaign_entropy_stage_chain_contract_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        contract,
        context="repair_campaign_entropy_stage_chain_contract",
    )
    return contract


__all__ = [
    "REPAIR_CAMPAIGN_ENTROPY_STAGE_CHAIN_CONTRACT_SCHEMA",
    "REPAIR_CAMPAIGN_ENTROPY_STAGE_CHAIN_NODE_SCHEMA",
    "RepairCampaignChainContractError",
    "build_repair_campaign_entropy_stage_chain_contract",
]
