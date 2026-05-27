# SPDX-License-Identifier: MIT
"""Build queue-owned MLX-local probe specs for structural repair cascades."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.scheduler.experiment_queue import QUEUE_SCHEMA, normalize_queue_definition
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_campaign_learning_signal import (
    REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
    REPAIR_CAMPAIGN_LOCAL_PLANNING_UPDATE_SCHEMA,
)
from tac.optimization.repair_campaign_posterior import (
    DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH,
    DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH,
    REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_APPEND_REPORT_SCHEMA,
)
from tac.optimization.repair_campaign_scorer import REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA
from tac.repo_io import json_text, sha256_bytes, sha256_file

REPAIR_CASCADE_MLX_PROBE_QUEUE_METADATA_SCHEMA = (
    "repair_cascade_mlx_probe_queue_metadata.v1"
)
REPAIR_CASCADE_MLX_PROBE_EXPERIMENT_METADATA_SCHEMA = (
    "repair_cascade_mlx_probe_experiment_metadata.v1"
)
REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA = "repair_cascade_mlx_probe_spec.v1"
REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA = "repair_cascade_mlx_probe_result.v1"
REPAIR_CASCADE_MLX_REPAIR_FAMILY_CAMPAIGN_SCHEMA = (
    "repair_cascade_mlx_repair_family_campaign.v1"
)
REPAIR_CASCADE_OPPORTUNITY_ROW_SCHEMA = (
    "frontier_rate_attack_repair_cascade_opportunity_row.v1"
)
REPAIR_BUDGET_WATERFILL_WORK_ORDER_SCHEMA = (
    "frontier_rate_attack_repair_budget_waterfill_work_order.v1"
)

_CANONICAL_MLX_REPAIR_FAMILY_CAMPAIGNS: tuple[dict[str, Any], ...] = (
    {
        "family_id": "posenet_null_bottom_decile",
        "label": "PoseNet-null bottom-decile repair",
        "entropy_position_label": "scorer_entropy_repair_before_selector_codec",
        "entropy_position_class": "scorer_entropy_before_selector_codec",
        "operator_stage": "P19",
        "required_measurements": ("posenet_null_bottom_decile_pair_ids",),
        "required_artifact_keys": ("posenet_null_bottom_decile_pair_ids_path",),
        "materializer_family_id": "posenet_null_bottom_decile",
    },
    {
        "family_id": "segnet_class_region_waterfill",
        "label": "SegNet class-region waterfill",
        "entropy_position_label": "scorer_entropy_repair_before_selector_codec",
        "entropy_position_class": "scorer_entropy_before_selector_codec",
        "operator_stage": "P18",
        "required_measurements": ("segnet_class_region_mask_ids",),
        "required_artifact_keys": ("segnet_class_region_mask_ids_path",),
        "materializer_family_id": "segnet_class_region_waterfill",
    },
    {
        "family_id": "per_region_selector_codec",
        "label": "per-region selector codec",
        "entropy_position_label": "selector_codec_entropy",
        "entropy_position_class": "selector_codec_entropy",
        "operator_stage": "P11",
        "required_measurements": ("selector_payload_bits_per_region",),
        "required_artifact_keys": (
            "selector_payload_bits_per_region_path",
            "runtime_consumption_proof_path",
        ),
        "materializer_family_id": "per_region_selector_codec",
    },
    {
        "family_id": "frame0_k16_palette_asymmetry",
        "label": "frame0 K16 palette asymmetry",
        "entropy_position_label": "before_entropy_coder_distribution_shaping",
        "entropy_position_class": "pre_entropy_distribution_shaping",
        "operator_stage": "K16",
        "required_measurements": ("frame0_k16_palette_response_curve",),
        "required_artifact_keys": ("repair_dynamics_palette_probe_matrix_path",),
        "materializer_family_id": "palette_frame_asymmetry_prior",
    },
    {
        "family_id": "entropy_boundary_probe",
        "label": "pre-coder and coder-boundary entropy probe",
        "entropy_position_label": "at_entropy_coder_integer_codeword_boundary",
        "entropy_position_class": "entropy_coder_boundary",
        "operator_stage": "entropy_boundary",
        "required_measurements": ("entropy_boundary_probe_manifest",),
        "required_artifact_keys": ("entropy_boundary_probe_manifest_path",),
        "materializer_family_id": "entropy_boundary_probe",
    },
)


class RepairCascadeMlxProbeQueueError(ValueError):
    """Raised when a structural repair-cascade probe queue cannot be built."""


def _slug(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(chars).split("_")) or "unknown"


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    resolved = Path(path)
    repo = Path(repo_root)
    try:
        return str(resolved.resolve(strict=False).relative_to(repo.resolve(strict=False)))
    except ValueError:
        return str(resolved)


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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _source_payload_schema(payload: Mapping[str, Any]) -> str:
    return str(payload.get("schema") or "")


def _cascade_rows_from_work_order(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("repair_cascade_opportunity_rows") or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("schema") != REPAIR_CASCADE_OPPORTUNITY_ROW_SCHEMA:
            continue
        rows.append(dict(row))
    return rows


def _cascade_rows_from_score_report(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        family_id = str(row.get("family_id") or "")
        if (
            family_id != "entropy_position_cascade"
            and row.get("source_row_kind") != "repair_cascade_opportunity"
        ):
            continue
        source = _mapping(row.get("source_structural_opportunity"))
        cascade = dict(source) if source else {}
        cascade.setdefault("schema", REPAIR_CASCADE_OPPORTUNITY_ROW_SCHEMA)
        cascade.setdefault("cascade_id", row.get("cascade_id") or row.get("candidate_id"))
        cascade.setdefault("label", row.get("cascade_label") or "structural repair cascade")
        cascade.setdefault("source_relation", row.get("source_relation"))
        cascade.setdefault(
            "pipeline_position",
            row.get("entropy_position_label")
            or "scorer_entropy_repair_before_selector_codec",
        )
        cascade.setdefault("blockers", _string_list(row.get("source_blockers")))
        cascade.setdefault(
            "required_probe_measurements",
            _string_list(row.get("source_missing_artifacts")),
        )
        rows.append(cascade)
    return rows


def repair_cascade_rows_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return structural repair-cascade rows from supported source payloads."""

    schema = _source_payload_schema(payload)
    if schema == REPAIR_BUDGET_WATERFILL_WORK_ORDER_SCHEMA:
        return _cascade_rows_from_work_order(payload)
    if schema == REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA:
        return _cascade_rows_from_score_report(payload)
    raise RepairCascadeMlxProbeQueueError(
        "repair cascade MLX probe queue requires a repair waterfill work order "
        "or repair campaign score report"
    )


def _targeted_positions(cascade: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in cascade.get("targeted_positions") or []
        if isinstance(row, Mapping)
    ]


def _required_probe_measurements(cascade: Mapping[str, Any]) -> list[str]:
    measurements = _string_list(cascade.get("required_probe_measurements"))
    if measurements:
        return ordered_unique(measurements)
    return [
        "posenet_null_bottom_decile_pair_ids",
        "segnet_class_region_mask_ids",
        "selector_payload_bits_per_region",
        "receiver_consumed_runtime_replay_proof",
    ]


def _artifact_status(
    cascade: Mapping[str, Any],
    key: str,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    text = str(cascade.get(key) or "").strip()
    path = _resolve(text, repo_root) if text else None
    exists = bool(path is not None and path.is_file())
    return {
        "key": key,
        "path": text or None,
        "exists": exists,
    }


def _campaign_artifact_status(
    artifact_status: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    by_key = {str(item.get("key") or ""): item for item in artifact_status}
    return [
        {
            "key": key,
            "path": by_key.get(key, {}).get("path"),
            "exists": by_key.get(key, {}).get("exists") is True,
        }
        for key in keys
    ]


def _canonical_repair_family_campaign_rows(
    *,
    cascade: Mapping[str, Any],
    artifact_status: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    targeted_position_ids = ordered_unique(
        str(row.get("position_id") or "")
        for row in _targeted_positions(cascade)
        if str(row.get("position_id") or "").strip()
    )
    rows: list[dict[str, Any]] = []
    for index, campaign in enumerate(_CANONICAL_MLX_REPAIR_FAMILY_CAMPAIGNS, start=1):
        required_artifact_keys = _string_list(campaign.get("required_artifact_keys"))
        campaign_status = _campaign_artifact_status(
            artifact_status,
            required_artifact_keys,
        )
        missing = ordered_unique(
            f"{item['key']}:missing_or_unverified"
            for item in campaign_status
            if item.get("exists") is not True
        )
        row = {
            "schema": REPAIR_CASCADE_MLX_REPAIR_FAMILY_CAMPAIGN_SCHEMA,
            "campaign_order": index,
            "cascade_id": cascade.get("cascade_id"),
            "family_id": campaign["family_id"],
            "materializer_family_id": campaign["materializer_family_id"],
            "label": campaign["label"],
            "operator_stage": campaign["operator_stage"],
            "targeted_position_ids": targeted_position_ids,
            "entropy_position_label": campaign["entropy_position_label"],
            "entropy_position_class": campaign["entropy_position_class"],
            "required_measurements": _string_list(campaign.get("required_measurements")),
            "required_artifact_keys": required_artifact_keys,
            "artifact_status": campaign_status,
            "missing_artifacts": missing,
            "campaign_execution_mode": "local_mlx_advisory_only",
            "component_response_axis": "[macOS-MLX research-signal]",
            "byte_closed_materializer_required": True,
            "receiver_decode_only_proof_required": True,
            "exact_axis_replay_required_before_score_or_budget": True,
            "ready_for_local_mlx_advisory_execution": not missing,
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            "allowed_use": "repair_family_mlx_advisory_campaign_row_only",
            "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            row,
            context=f"repair_cascade_mlx_family_campaign:{campaign['family_id']}",
        )
        rows.append(row)
    return rows


def build_repair_cascade_mlx_probe_spec(
    *,
    source_payload: Mapping[str, Any],
    source_payload_path: str | Path,
    cascade_id: str,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build a false-authority MLX-local probe spec for one cascade row."""

    require_no_truthy_authority_fields(
        source_payload,
        context="repair_cascade_mlx_probe_source_payload",
    )
    rows = repair_cascade_rows_from_payload(source_payload)
    cascade = next(
        (row for row in rows if str(row.get("cascade_id") or "") == cascade_id),
        None,
    )
    if cascade is None:
        raise RepairCascadeMlxProbeQueueError(f"cascade id not found: {cascade_id}")
    measurements = _required_probe_measurements(cascade)
    artifact_keys = ordered_unique(
        [
            "local_mlx_response_path",
            "reference_local_mlx_response_path",
            "posenet_null_bottom_decile_pair_ids_path",
            "segnet_class_region_mask_ids_path",
            "selector_payload_bits_per_region_path",
            "runtime_consumption_proof_path",
            "repair_dynamics_palette_probe_matrix_path",
            "entropy_boundary_probe_manifest_path",
        ]
    )
    artifact_status = [
        _artifact_status(cascade, key, repo_root=repo_root) for key in artifact_keys
    ]
    repair_family_campaign_rows = _canonical_repair_family_campaign_rows(
        cascade=cascade,
        artifact_status=artifact_status,
    )
    missing = ordered_unique(
        [
            f"{item['key']}:missing_or_unverified"
            for item in artifact_status
            if item.get("exists") is not True
        ]
    )
    blockers = ordered_unique(
        [
            *missing,
            *_string_list(cascade.get("blockers")),
            "exact_axis_component_response_required_before_budget_spend",
            "receiver_runtime_materialization_required_before_exact_dispatch",
        ]
    )
    spec = {
        "schema": REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA,
        "source_payload_path": str(source_payload_path),
        "source_payload_schema": source_payload.get("schema"),
        "cascade_id": cascade_id,
        "cascade_label": cascade.get("label"),
        "source_relation": cascade.get("source_relation"),
        "pipeline_position": cascade.get("pipeline_position"),
        "targeted_positions": [dict(row) for row in _targeted_positions(cascade)],
        "required_probe_measurements": measurements,
        "required_local_mlx_artifacts": artifact_keys,
        "local_mlx_artifact_status": artifact_status,
        "repair_family_campaign_schema": (
            REPAIR_CASCADE_MLX_REPAIR_FAMILY_CAMPAIGN_SCHEMA
        ),
        "repair_family_campaign_count": len(repair_family_campaign_rows),
        "repair_family_campaign_rows": repair_family_campaign_rows,
        "campaign_execution_mode": "local_mlx_advisory_only",
        "local_mlx_rows_are_advisory_only": True,
        "missing_local_mlx_artifacts": missing,
        "probe_measurement_plan": [
            {
                "measurement": measurement,
                "axis": "[macOS-MLX research-signal]",
                "authority": "local_repair_planning_only",
                **FALSE_AUTHORITY,
            }
            for measurement in measurements
        ],
        "local_mlx_probe_execution_ready": not missing,
        "recommended_next_action": (
            "run_cascade_mlx_local_component_response_probe"
            if not missing
            else "materialize_missing_cascade_mlx_probe_artifacts"
        ),
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "cascade_mlx_local_probe_spec_for_repair_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        spec,
        context=f"repair_cascade_mlx_probe_spec:{cascade_id}",
    )
    return spec


def _artifact_status_from_record(
    record: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    key = str(record.get("key") or "unknown").strip() or "unknown"
    path_text = str(record.get("path") or "").strip()
    path = _resolve(path_text, repo_root) if path_text else None
    return {
        "key": key,
        "path": path_text or None,
        "exists": bool(path is not None and path.is_file()),
    }


def _safe_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(json_text(payload).encode("utf-8"))


def _file_record(
    label: str,
    path: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    resolved = _resolve(path, repo_root)
    if not resolved.is_file():
        raise RepairCascadeMlxProbeQueueError(f"required artifact missing: {label}={path}")
    return {
        "label": label,
        "path": _repo_rel(resolved, repo_root),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def build_repair_cascade_mlx_probe_result(
    *,
    probe_spec: Mapping[str, Any],
    probe_spec_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Record deterministic readiness for one MLX-local cascade probe spec."""

    if probe_spec.get("schema") != REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA:
        raise RepairCascadeMlxProbeQueueError(
            "repair cascade MLX probe result requires a probe spec"
        )
    require_no_truthy_authority_fields(
        probe_spec,
        context="repair_cascade_mlx_probe_result_spec",
    )
    artifact_status = [
        _artifact_status_from_record(record, repo_root=repo_root)
        for record in probe_spec.get("local_mlx_artifact_status") or []
        if isinstance(record, Mapping)
    ]
    missing = ordered_unique(
        [
            f"{item['key']}:missing_or_unverified"
            for item in artifact_status
            if item.get("exists") is not True
        ]
    )
    spec_blockers = _string_list(probe_spec.get("blockers"))
    repair_family_campaign_rows = [
        dict(row)
        for row in probe_spec.get("repair_family_campaign_rows") or []
        if isinstance(row, Mapping)
    ]
    blockers = ordered_unique(
        [
            *missing,
            *spec_blockers,
            *(
                ["local_mlx_probe_artifacts_missing"]
                if missing
                else ["concrete_mlx_component_response_runner_required"]
            ),
            "exact_axis_component_response_required_before_budget_spend",
            "receiver_runtime_materialization_required_before_exact_dispatch",
        ]
    )
    result = {
        "schema": REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA,
        "probe_spec_path": str(probe_spec_path),
        "probe_spec_schema": probe_spec.get("schema"),
        "cascade_id": probe_spec.get("cascade_id"),
        "cascade_label": probe_spec.get("cascade_label"),
        "source_payload_path": probe_spec.get("source_payload_path"),
        "source_payload_schema": probe_spec.get("source_payload_schema"),
        "pipeline_position": probe_spec.get("pipeline_position"),
        "targeted_positions": list(probe_spec.get("targeted_positions") or []),
        "component_response_axis": "[macOS-MLX research-signal]",
        "local_mlx_artifact_status": artifact_status,
        "missing_local_mlx_artifacts": missing,
        "required_probe_measurements": _string_list(
            probe_spec.get("required_probe_measurements")
        ),
        "probe_measurement_plan": list(probe_spec.get("probe_measurement_plan") or []),
        "repair_family_campaign_schema": probe_spec.get(
            "repair_family_campaign_schema"
        ),
        "repair_family_campaign_count": len(repair_family_campaign_rows),
        "repair_family_campaign_rows": repair_family_campaign_rows,
        "campaign_execution_mode": probe_spec.get("campaign_execution_mode"),
        "local_mlx_rows_are_advisory_only": True,
        "local_mlx_probe_execution_ready": not missing,
        "local_mlx_probe_executed": False,
        "component_response_row_emitted": False,
        "learning_signal_kind": (
            "blocked_repair_cascade_mlx_probe"
            if missing
            else "repair_cascade_mlx_probe_ready_for_component_response_runner"
        ),
        "recommended_next_action": (
            "materialize_missing_cascade_mlx_probe_artifacts"
            if missing
            else "run_concrete_mlx_component_response_probe_and_harvest_response_row"
        ),
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "cascade_mlx_local_probe_readiness_result_for_learning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        result,
        context=f"repair_cascade_mlx_probe_result:{probe_spec.get('cascade_id')}",
    )
    return result


def build_repair_cascade_mlx_learning_signal(
    *,
    probe_result: Mapping[str, Any],
    probe_result_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build a posterior-consumable learning signal from one cascade probe result."""

    if probe_result.get("schema") != REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA:
        raise RepairCascadeMlxProbeQueueError(
            "repair cascade learning signal requires a probe result"
        )
    require_no_truthy_authority_fields(
        probe_result,
        context="repair_cascade_mlx_learning_signal_result",
    )
    source_artifact = _file_record(
        "repair_cascade_mlx_probe_result",
        probe_result_path,
        repo_root=repo_root,
    )
    cascade_id = str(probe_result.get("cascade_id") or "unknown_cascade").strip()
    missing_artifacts = _string_list(probe_result.get("missing_local_mlx_artifacts"))
    blockers = ordered_unique(
        [
            *_string_list(probe_result.get("blockers")),
            "cascade_mlx_learning_signal_is_not_score_authority",
            "exact_axis_component_response_required_before_budget_spend",
            "receiver_runtime_materialization_required_before_exact_dispatch",
        ]
    )
    targeted_positions = [
        dict(row)
        for row in probe_result.get("targeted_positions") or []
        if isinstance(row, Mapping)
    ]
    entropy_surfaces = ordered_unique(
        str(row.get("entropy_surface") or "").strip()
        for row in targeted_positions
        if str(row.get("entropy_surface") or "").strip()
    )
    position_ids = ordered_unique(
        str(row.get("position_id") or "").strip()
        for row in targeted_positions
        if str(row.get("position_id") or "").strip()
    )
    measurement_plan = [
        dict(row)
        for row in probe_result.get("probe_measurement_plan") or []
        if isinstance(row, Mapping)
    ]
    repair_family_campaign_rows = [
        dict(row)
        for row in probe_result.get("repair_family_campaign_rows") or []
        if isinstance(row, Mapping)
    ]
    measurements = _string_list(probe_result.get("required_probe_measurements"))
    ready = probe_result.get("local_mlx_probe_execution_ready") is True
    recommended_policy = (
        "increase_priority_for_exact_axis_component_response_replay"
        if ready
        else "materialize_missing_local_mlx_custody_before_stackability"
    )
    identity = {
        "schema": "repair_cascade_mlx_learning_identity.v1",
        "cascade_id": cascade_id,
        "probe_result_schema": probe_result.get("schema"),
        "source_probe_result_sha256": source_artifact.get("sha256"),
        "pipeline_position": probe_result.get("pipeline_position"),
        "missing_artifacts": missing_artifacts,
        "blockers": blockers,
        "required_probe_measurements": measurements,
        "targeted_positions": targeted_positions,
        "repair_family_campaign_rows": repair_family_campaign_rows,
    }
    signal = {
        "schema": REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
        "learning_signal_kind": probe_result.get("learning_signal_kind"),
        "typed_response_id": f"repair_cascade_mlx_probe:{cascade_id}",
        "candidate_id": cascade_id,
        "family_id": "entropy_position_cascade",
        "component_response_axis": probe_result.get("component_response_axis"),
        "evidence_grade": (
            "local_mlx_cascade_probe_result_ready_for_component_response"
            if ready
            else "blocked_local_mlx_cascade_probe_result"
        ),
        "source_artifacts": [source_artifact],
        "replay_identity": {
            "schema": "repair_cascade_mlx_learning_replay_identity.v1",
            "replay_identity_kind": "cascade_mlx_probe_result_no_component_response",
            "hash_manifest_sha256": _stable_sha256(identity),
            "source_records_sha256": _stable_sha256(
                {
                    "schema": "repair_cascade_mlx_learning_source.v1",
                    "source_artifact": source_artifact,
                }
            ),
            "replay_argv_sha256": None,
            "execution_context_sha256": None,
            "environment_sha256": None,
        },
        "local_planning_update": {
            "schema": REPAIR_CAMPAIGN_LOCAL_PLANNING_UPDATE_SCHEMA,
            "posterior_surface": "repair_campaign_stackability_local_mlx_posterior",
            "local_planning_update_ready": True,
            "recommended_acquisition_policy": recommended_policy,
            "recommended_stackability_followup": probe_result.get(
                "recommended_next_action"
            ),
            "planner_feature_vector": {
                "cascade_probe_execution_ready": ready,
                "component_response_row_emitted": (
                    probe_result.get("component_response_row_emitted") is True
                ),
                "missing_artifact_count": len(missing_artifacts),
                "blocker_count": len(blockers),
                "required_probe_measurement_count": len(measurements),
                "probe_measurement_plan_count": len(measurement_plan),
                "repair_family_campaign_count": len(repair_family_campaign_rows),
                "repair_family_ids": ordered_unique(
                    str(row.get("family_id") or "")
                    for row in repair_family_campaign_rows
                    if str(row.get("family_id") or "").strip()
                ),
                "repair_family_ready_count": sum(
                    1
                    for row in repair_family_campaign_rows
                    if row.get("ready_for_local_mlx_advisory_execution") is True
                ),
                "targeted_position_count": len(targeted_positions),
                "position_ids": position_ids,
                "entropy_surfaces": entropy_surfaces,
                "entropy_position_label": probe_result.get("pipeline_position"),
                "operation_levels": ordered_unique(
                    [
                        "pixel",
                        "region",
                        "boundary",
                        "frame",
                        "pair",
                        "batch",
                        "full_video",
                        "selector_codec",
                    ]
                ),
                "local_mlx_artifact_count": len(
                    list(probe_result.get("local_mlx_artifact_status") or [])
                ),
                "existing_local_mlx_artifact_count": sum(
                    1
                    for row in probe_result.get("local_mlx_artifact_status") or []
                    if isinstance(row, Mapping) and row.get("exists") is True
                ),
                "source_payload_schema_present": bool(
                    probe_result.get("source_payload_schema")
                ),
                "source_payload_path_present": bool(probe_result.get("source_payload_path")),
                "allocated_repair_bytes": 0,
                "expected_local_improvement_score_units": 0.0,
                "improvement_per_allocated_byte": 0.0,
                "requested_repair_bytes": _safe_int(
                    probe_result.get("requested_repair_bytes")
                ),
            },
            "posterior_update_blockers": [
                "cascade_mlx_learning_signal_is_not_score_authority",
                "exact_axis_component_response_required_before_budget_spend",
                "receiver_runtime_materialization_required_before_exact_dispatch",
            ],
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "blockers": blockers,
        "missing_artifacts": missing_artifacts,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "repair_cascade_mlx_acquisition_update_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        signal,
        context=f"repair_cascade_mlx_learning_signal:{cascade_id}",
    )
    return signal


def _cascade_experiment(
    *,
    source_payload: Mapping[str, Any],
    source_payload_path: str | Path,
    cascade: Mapping[str, Any],
    repo_root: str | Path,
    queue_root: Path,
    posterior_path: Path,
    posterior_lock_path: Path,
    priority: int,
) -> dict[str, Any]:
    cascade_id = str(cascade.get("cascade_id") or f"cascade_{priority}").strip()
    slug = _slug(cascade_id)
    spec_path = queue_root / slug / "repair_cascade_mlx_probe_spec.json"
    result_path = queue_root / slug / "repair_cascade_mlx_probe_result.json"
    learning_signal_path = queue_root / slug / "repair_cascade_mlx_learning_signal.json"
    posterior_append_report_path = (
        queue_root / slug / "repair_cascade_mlx_posterior_append_report.json"
    )
    spec_ref = _repo_rel(spec_path, repo_root)
    result_ref = _repo_rel(result_path, repo_root)
    learning_signal_ref = _repo_rel(learning_signal_path, repo_root)
    posterior_append_report_ref = _repo_rel(posterior_append_report_path, repo_root)
    posterior_ref = _repo_rel(posterior_path, repo_root)
    posterior_lock_ref = _repo_rel(posterior_lock_path, repo_root)
    missing = _required_probe_measurements(cascade)
    metadata = {
        "schema": REPAIR_CASCADE_MLX_PROBE_EXPERIMENT_METADATA_SCHEMA,
        "source_payload_path": str(source_payload_path),
        "source_payload_schema": source_payload.get("schema"),
        "cascade_id": cascade_id,
        "cascade_label": cascade.get("label"),
        "source_relation": cascade.get("source_relation"),
        "pipeline_position": cascade.get("pipeline_position"),
        "probe_spec_path": spec_ref,
        "probe_spec_schema": REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA,
        "probe_result_path": result_ref,
        "probe_result_schema": REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA,
        "learning_signal_path": learning_signal_ref,
        "learning_signal_schema": REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
        "posterior_path": posterior_ref,
        "posterior_lock_path": posterior_lock_ref,
        "posterior_append_report_path": posterior_append_report_ref,
        "posterior_append_report_schema": (
            REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_APPEND_REPORT_SCHEMA
        ),
        "required_probe_measurements": missing,
        "repair_family_campaign_schema": (
            REPAIR_CASCADE_MLX_REPAIR_FAMILY_CAMPAIGN_SCHEMA
        ),
        "repair_family_campaign_count": len(_CANONICAL_MLX_REPAIR_FAMILY_CAMPAIGNS),
        "repair_family_campaign_family_ids": [
            str(row["family_id"]) for row in _CANONICAL_MLX_REPAIR_FAMILY_CAMPAIGNS
        ],
        "campaign_execution_mode": "local_mlx_advisory_only",
        "local_mlx_advisory_custody_required": True,
        "component_response_axis": "[macOS-MLX research-signal]",
        "queue_actuation_ready": True,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "queue_owned_cascade_mlx_probe_metadata",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        metadata,
        context=f"repair_cascade_mlx_probe_experiment:{cascade_id}",
    )
    return {
        "id": f"repair_cascade_mlx_probe_{slug}",
        "priority": priority,
        "status": "queued",
        "tags": [
            "frontier-rate-attack",
            "repair-cascade",
            "local-mlx-advisory-custody",
            "no-score-authority",
        ],
        "metadata": metadata,
        "steps": [
            {
                "id": "emit_repair_cascade_mlx_probe_spec",
                "kind": "command",
                "command": [
                    ".venv/bin/python",
                    "tools/build_repair_cascade_mlx_probe_spec.py",
                    "--source-payload",
                    str(source_payload_path),
                    "--cascade-id",
                    cascade_id,
                    "--probe-spec-out",
                    spec_ref,
                    "--overwrite",
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": spec_ref,
                        "key": "schema",
                        "equals": REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA,
                    },
                    {"type": "json_false_authority", "path": spec_ref},
                    {
                        "type": "json_equals",
                        "path": spec_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [spec_ref],
                    "input_artifact_paths": [str(source_payload_path)],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "inspect_repair_cascade_mlx_probe_missing_artifacts",
                "kind": "command",
                "requires": ["emit_repair_cascade_mlx_probe_spec"],
                "command": [".venv/bin/python", "-m", "json.tool", spec_ref],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 60,
                "telemetry": {
                    "input_artifact_paths": [spec_ref],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "record_repair_cascade_mlx_probe_result",
                "kind": "command",
                "requires": ["emit_repair_cascade_mlx_probe_spec"],
                "command": [
                    ".venv/bin/python",
                    "tools/run_repair_cascade_mlx_probe.py",
                    "--probe-spec",
                    spec_ref,
                    "--output",
                    result_ref,
                    "--overwrite",
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": result_ref,
                        "key": "schema",
                        "equals": REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA,
                    },
                    {"type": "json_false_authority", "path": result_ref},
                    {
                        "type": "json_equals",
                        "path": result_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [result_ref],
                    "input_artifact_paths": [spec_ref],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "build_repair_cascade_mlx_learning_signal",
                "kind": "command",
                "requires": ["record_repair_cascade_mlx_probe_result"],
                "command": [
                    ".venv/bin/python",
                    "tools/build_repair_cascade_mlx_learning_signal.py",
                    "--probe-result",
                    result_ref,
                    "--learning-signal-out",
                    learning_signal_ref,
                    "--overwrite",
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": learning_signal_ref,
                        "key": "schema",
                        "equals": REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
                    },
                    {"type": "json_false_authority", "path": learning_signal_ref},
                    {
                        "type": "json_equals",
                        "path": learning_signal_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [learning_signal_ref],
                    "input_artifact_paths": [result_ref],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "append_repair_cascade_mlx_learning_posterior",
                "kind": "command",
                "requires": ["build_repair_cascade_mlx_learning_signal"],
                "command": [
                    ".venv/bin/python",
                    "tools/append_repair_campaign_stackability_posterior.py",
                    "--learning-signal",
                    learning_signal_ref,
                    "--posterior-path",
                    posterior_ref,
                    "--lock-path",
                    posterior_lock_ref,
                    "--report-out",
                    posterior_append_report_ref,
                    "--overwrite",
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": posterior_append_report_ref,
                        "key": "schema",
                        "equals": (
                            REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_APPEND_REPORT_SCHEMA
                        ),
                    },
                    {"type": "json_false_authority", "path": posterior_append_report_ref},
                    {"type": "jsonl_false_authority", "path": posterior_ref},
                    {
                        "type": "json_equals",
                        "path": posterior_append_report_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [posterior_append_report_ref, posterior_ref],
                    "input_artifact_paths": [learning_signal_ref],
                    "include_postcondition_paths": True,
                },
            },
        ],
    }


def _empty_experiment(
    *,
    source_payload_path: str | Path,
) -> dict[str, Any]:
    return {
        "id": "repair_cascade_mlx_probe_no_structural_cascade_rows",
        "priority": 1,
        "status": "frozen",
        "tags": [
            "frontier-rate-attack",
            "repair-cascade",
            "blocked-empty-selection",
            "no-score-authority",
        ],
        "metadata": {
            "schema": REPAIR_CASCADE_MLX_PROBE_EXPERIMENT_METADATA_SCHEMA,
            "source_payload_path": str(source_payload_path),
            "cascade_id": None,
            "queue_actuation_ready": False,
            "queue_actuation_blockers": ["structural_repair_cascade_rows_empty"],
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "steps": [
            {
                "id": "inspect_empty_repair_cascade_selection",
                "kind": "command",
                "command": [
                    ".venv/bin/python",
                    "-c",
                    (
                        "import json; print(json.dumps({"
                        "'schema':'repair_cascade_mlx_probe_blocker.v1',"
                        "'blockers':['structural_repair_cascade_rows_empty'],"
                        "'budget_spend_allowed':False,"
                        "'ready_for_exact_eval_dispatch':False,"
                        "'score_claim':False,"
                        "'promotion_eligible':False,"
                        "'rank_or_kill_eligible':False"
                        "}, sort_keys=True))"
                    ),
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 30,
            }
        ],
    }


def build_repair_cascade_mlx_probe_queue(
    *,
    repo_root: str | Path,
    source_payload: Mapping[str, Any],
    source_payload_path: str | Path,
    results_root: str | Path,
    queue_id: str = "repair_cascade_mlx_probe_queue",
    experiment_limit: int | None = None,
    posterior_path: str | Path = DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH,
    posterior_lock_path: str | Path = (
        DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH
    ),
) -> dict[str, Any]:
    """Build local probe-spec rows for structural repair cascades."""

    require_no_truthy_authority_fields(
        source_payload,
        context="repair_cascade_mlx_probe_queue_input",
    )
    cascade_rows = repair_cascade_rows_from_payload(source_payload)
    if experiment_limit is not None:
        cascade_rows = cascade_rows[: max(0, int(experiment_limit))]
    queue_root = (
        _resolve(results_root, repo_root) / "repair_cascade_mlx_probe_queue" / _slug(queue_id)
    )
    resolved_posterior_path = _resolve(posterior_path, repo_root)
    resolved_posterior_lock_path = _resolve(posterior_lock_path, repo_root)
    experiments = (
        [
            _cascade_experiment(
                source_payload=source_payload,
                source_payload_path=source_payload_path,
                cascade=cascade,
                repo_root=repo_root,
                queue_root=queue_root,
                posterior_path=resolved_posterior_path,
                posterior_lock_path=resolved_posterior_lock_path,
                priority=priority,
            )
            for priority, cascade in enumerate(cascade_rows, start=1)
        ]
        if cascade_rows
        else [_empty_experiment(source_payload_path=source_payload_path)]
    )
    ready_count = sum(
        1
        for experiment in experiments
        if _mapping(experiment.get("metadata")).get("queue_actuation_ready") is True
    )
    blockers = ordered_unique(
        blocker
        for experiment in experiments
        for blocker in _string_list(
            _mapping(experiment.get("metadata")).get("queue_actuation_blockers")
        )
    )
    if not cascade_rows:
        blockers.append("structural_repair_cascade_rows_empty")
    metadata = {
        "schema": REPAIR_CASCADE_MLX_PROBE_QUEUE_METADATA_SCHEMA,
        "source_payload_path": str(source_payload_path),
        "source_payload_schema": source_payload.get("schema"),
        "queue_id": queue_id,
        "structural_repair_cascade_count": len(cascade_rows),
        "experiment_count": len(experiments),
        "ready_experiment_count": ready_count,
        "blocked_experiment_count": len(experiments) - ready_count,
        "results_root": _repo_rel(queue_root, repo_root),
        "posterior_path": _repo_rel(resolved_posterior_path, repo_root),
        "posterior_lock_path": _repo_rel(resolved_posterior_lock_path, repo_root),
        "posterior_append_report_schema": (
            REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_APPEND_REPORT_SCHEMA
        ),
        "repair_family_campaign_schema": (
            REPAIR_CASCADE_MLX_REPAIR_FAMILY_CAMPAIGN_SCHEMA
        ),
        "repair_family_campaign_count_per_cascade": len(
            _CANONICAL_MLX_REPAIR_FAMILY_CAMPAIGNS
        ),
        "queue_actuation_blockers": blockers,
        "local_mlx_advisory_custody_required": True,
        "component_response_axis": "[macOS-MLX research-signal]",
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_cascade_mlx_probe_queue_for_local_planning",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        metadata,
        context="repair_cascade_mlx_probe_queue_metadata",
    )
    return normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {"local_cpu": 1},
            },
            "metadata": metadata,
            "experiments": experiments,
        }
    )


__all__ = [
    "REPAIR_CASCADE_MLX_PROBE_EXPERIMENT_METADATA_SCHEMA",
    "REPAIR_CASCADE_MLX_PROBE_QUEUE_METADATA_SCHEMA",
    "REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA",
    "REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA",
    "REPAIR_CASCADE_MLX_REPAIR_FAMILY_CAMPAIGN_SCHEMA",
    "RepairCascadeMlxProbeQueueError",
    "build_repair_cascade_mlx_learning_signal",
    "build_repair_cascade_mlx_probe_queue",
    "build_repair_cascade_mlx_probe_result",
    "build_repair_cascade_mlx_probe_spec",
    "repair_cascade_rows_from_payload",
]
