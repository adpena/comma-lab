# SPDX-License-Identifier: MIT
"""Planner-consumable learning signals from repair stackability replay bundles."""

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
    REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
    REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA,
)
from tac.repo_io import sha256_file

REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA = "repair_campaign_learning_signal.v1"
REPAIR_CAMPAIGN_LOCAL_PLANNING_UPDATE_SCHEMA = (
    "repair_campaign_local_planning_update.v1"
)


class RepairCampaignLearningSignalError(ValueError):
    """Raised when a repair campaign learning signal cannot be built."""


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


def _safe_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _resolve(path: str | Path, *, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, *, repo_root: str | Path) -> str:
    value = Path(path)
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return value.as_posix()


def _file_record(label: str, path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    resolved = _resolve(path, repo_root=repo_root)
    if not resolved.is_file():
        raise RepairCampaignLearningSignalError(f"required artifact missing: {label}={path}")
    return {
        "label": label,
        "path": _repo_rel(resolved, repo_root=repo_root),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def build_repair_campaign_learning_signal(
    *,
    score_report_path: str | Path,
    probe_path: str | Path,
    replay_bundle_path: str | Path,
    score_report: Mapping[str, Any],
    probe: Mapping[str, Any],
    replay_bundle: Mapping[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build a false-authority local learning signal from one repair replay."""

    if score_report.get("schema") != REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA:
        raise RepairCampaignLearningSignalError(
            "score_report must be repair_campaign_score_report.v1"
        )
    if probe.get("schema") != REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA:
        raise RepairCampaignLearningSignalError(
            "probe must be repair_campaign_stackability_probe.v1"
        )
    if replay_bundle.get("schema") != REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA:
        raise RepairCampaignLearningSignalError(
            "replay_bundle must be repair_campaign_stackability_replay_bundle.v1"
        )
    for label, payload in (
        ("score_report", score_report),
        ("probe", probe),
        ("replay_bundle", replay_bundle),
    ):
        require_no_truthy_authority_fields(
            payload,
            context=f"repair_campaign_learning_signal_input:{label}",
        )
    typed_response_id = str(probe.get("typed_response_id") or "").strip()
    if not typed_response_id:
        raise RepairCampaignLearningSignalError("probe missing typed_response_id")
    if str(replay_bundle.get("typed_response_id") or "") != typed_response_id:
        raise RepairCampaignLearningSignalError(
            "replay bundle typed_response_id does not match probe"
        )

    allocation = _mapping(probe.get("optimizer_allocation"))
    score_row = _mapping(probe.get("source_score_row"))
    family_id = str(allocation.get("family_id") or score_row.get("family_id") or "")
    entropy_position = str(probe.get("entropy_position_label") or "")
    expected_improvement = _safe_float(
        probe.get("expected_local_improvement_score_units")
    )
    allocated_bytes = _safe_int(probe.get("allocated_repair_bytes"))
    improvement_per_allocated_byte = (
        expected_improvement / allocated_bytes if allocated_bytes > 0 else 0.0
    )
    hard_blockers = [
        "local_mlx_learning_signal_is_not_score_authority",
        "exact_axis_component_response_required_before_budget_spend",
        "receiver_runtime_materialization_required_before_exact_dispatch",
        "post_materialization_stackability_must_be_remeasured",
    ]
    signal = {
        "schema": REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
        "typed_response_id": typed_response_id,
        "candidate_id": allocation.get("candidate_id") or score_row.get("candidate_id"),
        "family_id": family_id or "unclassified_repair_family",
        "component_response_axis": "[macOS-MLX research-signal]",
        "evidence_grade": "local_mlx_research_signal_only",
        "source_artifacts": [
            _file_record(
                "repair_campaign_score_report",
                score_report_path,
                repo_root=repo_root,
            ),
            _file_record("repair_campaign_stackability_probe", probe_path, repo_root=repo_root),
            _file_record(
                "repair_campaign_stackability_replay_bundle",
                replay_bundle_path,
                repo_root=repo_root,
            ),
        ],
        "replay_identity": {
            "hash_manifest_sha256": replay_bundle.get("hash_manifest_sha256"),
            "source_records_sha256": replay_bundle.get("source_records_sha256"),
            "replay_argv_sha256": replay_bundle.get("replay_argv_sha256"),
            "execution_context_sha256": replay_bundle.get("execution_context_sha256"),
            "environment_sha256": replay_bundle.get("environment_sha256"),
        },
        "local_planning_update": {
            "schema": REPAIR_CAMPAIGN_LOCAL_PLANNING_UPDATE_SCHEMA,
            "posterior_surface": "repair_campaign_stackability_local_mlx_posterior",
            "local_planning_update_ready": probe.get("stackability_ready") is True,
            "recommended_acquisition_policy": (
                "increase_priority_for_exact_axis_component_response_replay"
                if expected_improvement > 0.0
                else "hold_until_additional_local_response_signal"
            ),
            "recommended_stackability_followup": (
                "remeasure_after_parent_rate_materialization_and_sibling_repairs"
            ),
            "planner_feature_vector": {
                "allocated_repair_bytes": allocated_bytes,
                "receiver_closed_rate_credit_bytes": _safe_int(
                    probe.get("receiver_closed_rate_credit_bytes")
                ),
                "remaining_receiver_closed_rate_credit_bytes_after": _safe_int(
                    probe.get("remaining_receiver_closed_rate_credit_bytes_after")
                ),
                "expected_local_improvement_score_units": expected_improvement,
                "improvement_per_allocated_byte": improvement_per_allocated_byte,
                "interaction_penalty": _safe_float(probe.get("interaction_penalty")),
                "entropy_position_label": entropy_position,
                "entropy_position_class": probe.get("entropy_position_class"),
                "targeted_dimensions": _string_list(
                    allocation.get("targeted_dimensions")
                    or score_row.get("targeted_dimensions")
                ),
                "operation_levels": _string_list(
                    allocation.get("operation_levels") or score_row.get("operation_levels")
                ),
            },
            "posterior_update_blockers": hard_blockers,
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "blockers": ordered_unique([*_string_list(probe.get("blockers")), *hard_blockers]),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "local_repair_campaign_planning_and_acquisition_update_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        signal,
        context="repair_campaign_learning_signal",
    )
    return signal


__all__ = [
    "REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA",
    "REPAIR_CAMPAIGN_LOCAL_PLANNING_UPDATE_SCHEMA",
    "RepairCampaignLearningSignalError",
    "build_repair_campaign_learning_signal",
]
