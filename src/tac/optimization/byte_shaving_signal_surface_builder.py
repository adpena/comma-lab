# SPDX-License-Identifier: MIT
"""Canonical builder for byte-shaving signal surfaces.

This module is the aggregation layer between the solver stack's scattered
signals and ``tac.optimization.byte_shaving_campaign``. It records where each
signal came from, keeps proxy/local evidence planning-only, and refuses any
input that would smuggle score or dispatch authority into the byte-shaving
planner.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.auth_eval_schema import (
    eval_device,
    eval_metric_summary,
    required_contest_auth_axis_payload_blockers,
)
from tac.optimization.byte_shaving_campaign import (
    FALSE_AUTHORITY,
    SIGNAL_SURFACE_SCHEMA,
    ByteShavingCampaignError,
    build_signal_surface_from_candidate_queue,
    build_signal_surface_from_engineered_correction_targeting,
    build_signal_surface_from_inverse_action_functional,
    build_signal_surface_from_master_gradient_anchor,
    validate_signal_surface,
)
from tac.optimization.mlx_dynamic_sweep_observations import (
    MLXDynamicSweepObservationError,
    load_observation_rows,
)
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.scorer_inverse_decision_surface import (
    build_inverse_scorer_decision_surface,
)
from tac.optimization.scorer_response_dataset import (
    RATE_SCORE_PER_BYTE,
    ScorerResponseDatasetError,
    normalize_legacy_response_dataset_authority,
    scorer_response_planning_value_for_target,
    summarize_rows,
)

DQS1_PAIRSET_ACQUISITION_SCHEMA = "decoder_q_pairset_acquisition.v1"
DQS1_PAIRSET_DROP_PAIR_TARGET_KIND = "dqs1_pairset_drop_pair"
DQS1_PAIRSET_DROP_PAIR_MATERIALIZER = "dqs1_pairset_drop_pair_adapter"
DQS1_PAIRSET_RECEIVER_CONTRACT_KIND = "archive_charged_pairset_runtime_selector"
DQS1_OBSERVATION_ROW_SCHEMA = "mlx_dynamic_sweep_observation.v1"
DQS1_OBSERVATION_SOURCE_SCHEMA = "dqs1_local_first_harvest.v1"
DQS1_OBSERVATION_SWEEP_CONFIG_ID = "dqs1_local_first_macos_cpu_advisory"
DQS1_PAIR_FRAME_OUTCOME_REF_KIND = "dqs1_pair_frame_geometry_outcome_anchor"
DQS1_PAIR_FRAME_OUTCOME_ANCHOR_SCHEMA = "dqs1_pair_frame_geometry_outcome_anchor.v1"
DQS1_CANONICAL_PAIR_COUNT = 600
DQS1_MAX_PAIR_INDEX = DQS1_CANONICAL_PAIR_COUNT - 1


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ByteShavingCampaignError(f"{path}: expected JSON object")
    return payload


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dedupe_unit_id(
    unit: Mapping[str, Any],
    *,
    prefix: str,
    seen: set[str],
) -> dict[str, Any]:
    out = dict(unit)
    raw = str(out.get("unit_id") or out.get("id") or "unit").strip() or "unit"
    candidate = raw
    if candidate in seen:
        candidate = f"{prefix}_{raw}"
    suffix = 2
    while candidate in seen:
        candidate = f"{prefix}_{raw}_{suffix}"
        suffix += 1
    out["unit_id"] = candidate
    seen.add(candidate)
    return out


def _extend_refs(target: list[dict[str, Any]], refs: Iterable[Any]) -> None:
    for item in refs:
        if isinstance(item, Mapping):
            target.append(dict(item))


def _json_payload_ref(path: Path, repo_root: Path, *, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": _repo_rel(path, repo_root),
        "sha256": _file_sha256(path),
        "schema": payload.get("schema") or payload.get("schema_version"),
    }


def _candidate_queue_surface(
    path: Path,
    repo_root: Path,
    *,
    campaign_id: str,
    index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    payload = _load_json_object(path)
    surface = build_signal_surface_from_candidate_queue(
        payload,
        campaign_id=f"{campaign_id}_candidate_queue_{index}",
    )
    refs = {
        "source_signal_refs": [
            {
                **_json_payload_ref(
                    path,
                    repo_root,
                    kind="optimizer_candidate_queue",
                    payload=payload,
                ),
                "row_count": len(_as_list(payload.get("top_k"))),
                "top_k_count": payload.get("top_k_count"),
                "surface_unit_count": len(_as_list(surface.get("units"))),
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "auth_eval_refs": [],
        "mlx_calibration_refs": [],
        "scorer_response_refs": [],
    }
    _extend_refs(refs["source_signal_refs"], _as_list(surface.get("source_signal_refs")))
    _extend_refs(refs["auth_eval_refs"], _as_list(surface.get("auth_eval_refs")))
    _extend_refs(refs["mlx_calibration_refs"], _as_list(surface.get("mlx_calibration_refs")))
    _extend_refs(refs["scorer_response_refs"], _as_list(surface.get("scorer_response_refs")))
    return [
        dict(unit)
        for unit in _as_list(surface.get("units"))
        if isinstance(unit, Mapping)
    ], dict(surface), refs


def _master_gradient_surface(
    repo_root: Path,
    *,
    archive_sha256: str,
    ledger_path: str | Path | None,
    axis: str | None,
    campaign_id: str,
    index: int,
    low_sensitivity_quantile: float,
    max_units: int,
    max_span_bytes: int,
    quality_cost_multiplier: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    surface = build_signal_surface_from_master_gradient_anchor(
        archive_sha256=archive_sha256,
        repo_root=repo_root,
        ledger_path=ledger_path,
        axis=axis,
        campaign_id=f"{campaign_id}_master_gradient_{index}",
        low_sensitivity_quantile=low_sensitivity_quantile,
        max_units=max_units,
        max_span_bytes=max_span_bytes,
        quality_cost_multiplier=quality_cost_multiplier,
    )
    return [
        dict(unit)
        for unit in _as_list(surface.get("units"))
        if isinstance(unit, Mapping)
    ], dict(surface)


def _engineered_correction_targeting_surface(
    path: Path,
    repo_root: Path,
    *,
    campaign_id: str,
    index: int,
    max_targets: int | None,
    default_predicted_quality_score_delta: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json_object(path)
    surface = build_signal_surface_from_engineered_correction_targeting(
        payload,
        campaign_id=f"{campaign_id}_engineered_correction_targeting_{index}",
        max_targets=max_targets,
        default_predicted_quality_score_delta=default_predicted_quality_score_delta,
    )
    ref = {
        **_json_payload_ref(
            path,
            repo_root,
            kind="engineered_correction_targeting",
            payload=payload,
        ),
        "consumer_id": payload.get("consumer_id"),
        "archive_sha256": payload.get("archive_sha256"),
        "measurement_axis": payload.get("measurement_axis"),
        "measurement_hardware": payload.get("measurement_hardware"),
        "targets_per_pair": payload.get("targets_per_pair"),
        "total_targets": payload.get("total_targets"),
        "surface_unit_count": len(_as_list(surface.get("units"))),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return [
        dict(unit)
        for unit in _as_list(surface.get("units"))
        if isinstance(unit, Mapping)
    ], dict(surface), [ref]


def _inverse_action_functional_surface(
    path: Path,
    repo_root: Path,
    *,
    campaign_id: str,
    index: int,
    allow_leaf_cell_candidates: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json_object(path)
    surface = build_signal_surface_from_inverse_action_functional(
        payload,
        campaign_id=f"{campaign_id}_inverse_action_functional_{index}",
        allow_leaf_cell_candidates=allow_leaf_cell_candidates,
    )
    portfolio = _as_mapping(surface.get("water_bucket_materialization_portfolio"))
    ref = {
        **_json_payload_ref(
            path,
            repo_root,
            kind="inverse_steganalysis_action_functional",
            payload=payload,
        ),
        "cell_count": _as_mapping(payload.get("integral_totals")).get("cell_count"),
        "selected_count": _as_mapping(payload.get("water_bucket")).get("selected_count"),
        "surface_unit_count": len(_as_list(surface.get("units"))),
        "water_bucket_materialization_portfolio_schema": portfolio.get("schema"),
        "water_bucket_materialization_portfolio_row_count": portfolio.get(
            "portfolio_row_count"
        ),
        "water_bucket_materialization_actuation_modes": _as_list(
            portfolio.get("actuation_modes")
        ),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return [
        dict(unit)
        for unit in _as_list(surface.get("units"))
        if isinstance(unit, Mapping)
    ], dict(surface), [ref]


def _auth_eval_ref(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    metrics = eval_metric_summary(payload)
    blockers = required_contest_auth_axis_payload_blockers(payload, metrics)
    return {
        **_json_payload_ref(path, repo_root, kind="auth_eval", payload=payload),
        "evidence_grade": payload.get("evidence_grade"),
        "lane_tag": payload.get("lane_tag"),
        "score_axis": payload.get("score_axis"),
        "evidence_semantics": payload.get("evidence_semantics"),
        "device": eval_device(payload),
        "metrics": metrics,
        "strict_contest_auth_axis_blockers": blockers,
        "strict_contest_auth_axis_eligible": not blockers,
        "source_score_claim_present": payload.get("score_claim") is True,
        "source_score_claim_valid_present": payload.get("score_claim_valid") is True,
        "source_promotion_authority_present": payload.get("promotion_eligible") is True,
        "source_rank_or_kill_authority_present": payload.get("rank_or_kill_eligible") is True,
        "consumption_role": "reference_only_for_planning_surface_calibration",
    }


def _mlx_calibration_ref(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    try:
        require_no_truthy_authority_fields(
            payload,
            context="byte_shaving_mlx_calibration_ref",
        )
    except ValueError as exc:
        raise ByteShavingCampaignError(str(exc)) from exc
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    policy = (
        payload.get("decision_policy")
        if isinstance(payload.get("decision_policy"), Mapping)
        else {}
    )
    return {
        **_json_payload_ref(
            path,
            repo_root,
            kind="mlx_score_calibration",
            payload=payload,
        ),
        "row_count": payload.get("row_count"),
        "evidence_grade": payload.get("evidence_grade"),
        "evidence_tag": payload.get("evidence_tag"),
        "calibration_role": payload.get("calibration_role"),
        "summary": dict(summary),
        "decision_policy": dict(policy),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _scorer_response_ref(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    try:
        require_no_truthy_authority_fields(
            payload,
            context="byte_shaving_scorer_response_ref",
        )
        normalized = normalize_legacy_response_dataset_authority(
            payload,
            source_label=_repo_rel(path, repo_root),
        )
        planning_summary = _scorer_response_planning_summary(normalized, path=path)
    except ValueError as exc:
        raise ByteShavingCampaignError(str(exc)) from exc
    except ScorerResponseDatasetError as exc:
        raise ByteShavingCampaignError(str(exc)) from exc
    rows = _as_list(normalized.get("rows"))
    return {
        **_json_payload_ref(
            path,
            repo_root,
            kind="scorer_response_dataset",
            payload=normalized,
        ),
        "row_count": normalized.get("row_count")
        if normalized.get("row_count") is not None
        else len(rows),
        "evidence_grade": normalized.get("evidence_grade"),
        "evidence_tag": normalized.get("evidence_tag"),
        "consumer_routing_schema": normalized.get("consumer_routing_schema"),
        "planning_target_accessor": "scorer_response_planning_value_for_target",
        "planning_summary": planning_summary,
        "authority_normalization": normalized.get("authority_normalization"),
        "mlx_scorer_response_row_count": sum(
            1
            for row in rows
            if isinstance(row, Mapping)
            and row.get("source_schema") == "mlx_scorer_response.v1"
        ),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _scorer_response_planning_summary(
    dataset: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    rows = dataset.get("rows")
    if not isinstance(rows, list):
        raise ScorerResponseDatasetError(f"{path}: scorer-response rows[] missing")
    normalized_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    if len(normalized_rows) != len(rows):
        raise ScorerResponseDatasetError(f"{path}: scorer-response row must be object")
    validated_targets = (
        "delta_vs_baseline_score",
        "scorer_delta_vs_baseline",
        "observed_scorer_gain_vs_baseline",
    )
    for row in normalized_rows:
        label = str(row.get("row_id") or row.get("candidate_id") or path)
        for target in validated_targets:
            scorer_response_planning_value_for_target(row, target, label=label)
    return {
        "schema": "byte_shaving_scorer_response_planning_summary.v1",
        "target_value_accessor": "scorer_response_planning_value_for_target",
        "validated_targets": list(validated_targets),
        **summarize_rows(normalized_rows),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _safe_unit_token(value: Any, *, fallback: str) -> str:
    raw = str(value or fallback).strip() or fallback
    out = "".join(ch if ch.isalnum() else "_" for ch in raw.lower())
    return out.strip("_") or fallback


def _scorer_response_units(
    path: Path,
    repo_root: Path,
    *,
    index: int,
) -> list[dict[str, Any]]:
    payload = _load_json_object(path)
    try:
        require_no_truthy_authority_fields(
            payload,
            context="byte_shaving_scorer_response_units",
        )
        normalized = normalize_legacy_response_dataset_authority(
            payload,
            source_label=_repo_rel(path, repo_root),
        )
    except ValueError as exc:
        raise ByteShavingCampaignError(str(exc)) from exc
    except ScorerResponseDatasetError as exc:
        raise ByteShavingCampaignError(str(exc)) from exc
    rows = normalized.get("rows")
    if not isinstance(rows, list):
        raise ByteShavingCampaignError(f"{path}: scorer-response rows[] missing")

    units: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ByteShavingCampaignError(
                f"{path}: scorer-response row {row_index} must be object"
            )
        row_payload = dict(row)
        label = str(row_payload.get("row_id") or row_payload.get("candidate_id") or path)
        try:
            projected_delta = scorer_response_planning_value_for_target(
                row_payload,
                "delta_vs_baseline_score",
                label=label,
            )
            normalized_gain = scorer_response_planning_value_for_target(
                row_payload,
                "observed_scorer_gain_vs_baseline",
                label=label,
            )
            normalized_margin = scorer_response_planning_value_for_target(
                row_payload,
                "byte_budget_margin_vs_break_even",
                label=label,
            )
        except ScorerResponseDatasetError as exc:
            raise ByteShavingCampaignError(str(exc)) from exc
        if projected_delta is None:
            continue
        added_archive_bytes = row_payload.get("added_archive_bytes")
        saved_bytes = 0
        if isinstance(added_archive_bytes, int | float) and added_archive_bytes < 0:
            saved_bytes = round(abs(float(added_archive_bytes)))
        rate_delta = -RATE_SCORE_PER_BYTE * float(saved_bytes)
        quality_delta = float(projected_delta) - rate_delta
        token = _safe_unit_token(
            row_payload.get("candidate_id") or row_payload.get("row_id"),
            fallback=f"row_{index}_{row_index}",
        )
        operation = {
            "operation_id": "materialize_scorer_response_candidate",
            "operation_family": "materialize_scorer_response_candidate",
            "candidate_saved_bytes": saved_bytes,
            "predicted_quality_score_delta": quality_delta,
            "materializer": row_payload.get("source_path"),
            "params": {
                "source_row_id": row_payload.get("row_id"),
                "source_candidate_id": row_payload.get("candidate_id"),
                "source_path": row_payload.get("source_path"),
                "pair_indices": row_payload.get("pair_indices"),
                "source_pair_window": row_payload.get("source_pair_window"),
            },
            "blockers": [
                "scorer_response_unit_requires_byte_closed_materializer",
                "scorer_response_unit_requires_runtime_consumption_proof",
                "scorer_response_unit_requires_exact_auth_eval_before_score_claim",
            ],
        }
        units.append(
            {
                "unit_id": f"scorer_response_{token}",
                "unit_kind": "scorer_response_row",
                "candidate_saved_bytes": saved_bytes,
                "predicted_quality_score_delta": quality_delta,
                "confidence": row_payload.get("confidence", 0.5),
                "operation_families": ["materialize_scorer_response_candidate"],
                "operations": [operation],
                "source_candidate_id": row_payload.get("candidate_id"),
                "source_paths": [_repo_rel(path, repo_root)],
                "source_index": row_index,
                "score_axis": row_payload.get("axis") or row_payload.get("source_evidence_tag"),
                "evidence_grade": row_payload.get("source_evidence_grade"),
                "evidence_semantics": "scorer_response_normalized_full_video_planning_unit",
                "normalized_full_video_scorer_gain_vs_baseline": normalized_gain,
                "projected_full_video_delta_vs_baseline_score": projected_delta,
                "normalized_full_video_byte_budget_margin_vs_break_even": (
                    normalized_margin
                ),
                "added_archive_bytes": added_archive_bytes,
                "planning_value_accessor": "scorer_response_planning_value_for_target",
                "planning_value_scope": "normalized_full_video",
                "blockers": [
                    "scorer_response_unit_is_planning_only",
                    "requires_materializer_before_candidate_archive",
                    "requires_exact_auth_eval_before_score_claim",
                ],
                **FALSE_AUTHORITY,
            }
        )
    return units


def _safe_id_token(value: Any, *, fallback: str) -> str:
    raw = str(value or fallback).strip() or fallback
    token = "".join(char.lower() if char.isalnum() else "_" for char in raw)
    token = "_".join(part for part in token.split("_") if part)
    return token or fallback


def _pairset_acquisition_surface(
    path: Path,
    repo_root: Path,
    *,
    index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json_object(path)
    if payload.get("schema") != DQS1_PAIRSET_ACQUISITION_SCHEMA:
        raise ByteShavingCampaignError(
            f"{path}: expected schema {DQS1_PAIRSET_ACQUISITION_SCHEMA}"
        )
    try:
        require_no_truthy_authority_fields(
            payload,
            context="byte_shaving_pairset_acquisition",
        )
    except ValueError as exc:
        raise ByteShavingCampaignError(str(exc)) from exc
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ByteShavingCampaignError(f"{path}: candidates[] missing")

    units: list[dict[str, Any]] = []
    dropped_pair_counts: list[int] = []
    selector_kind_counts: dict[str, int] = {}
    repair_budget_rows = 0
    for row_index, item in enumerate(candidates):
        if not isinstance(item, Mapping):
            raise ByteShavingCampaignError(
                f"{path}: candidates[{row_index}] must be object"
            )
        try:
            require_no_truthy_authority_fields(
                item,
                context=f"byte_shaving_pairset_acquisition.candidates[{row_index}]",
            )
        except ValueError as exc:
            raise ByteShavingCampaignError(str(exc)) from exc
        repair = item.get("distortion_repair_budget_from_rate_savings")
        if not isinstance(repair, Mapping) or repair.get("active") is not True:
            continue
        try:
            require_no_truthy_authority_fields(
                repair,
                context=(
                    "byte_shaving_pairset_acquisition."
                    f"candidates[{row_index}].distortion_repair_budget"
                ),
            )
        except ValueError as exc:
            raise ByteShavingCampaignError(str(exc)) from exc
        saved_bytes = repair.get("saved_bytes_vs_source_selector")
        if not isinstance(saved_bytes, int) or isinstance(saved_bytes, bool) or saved_bytes <= 0:
            continue
        operation = item.get("acquisition_operation")
        if not isinstance(operation, Mapping):
            operation = {}
        try:
            require_no_truthy_authority_fields(
                operation,
                context=(
                    "byte_shaving_pairset_acquisition."
                    f"candidates[{row_index}].acquisition_operation"
                ),
            )
        except ValueError as exc:
            raise ByteShavingCampaignError(str(exc)) from exc
        raw_dropped_pairs = _as_list(operation.get("dropped_pair_indices"))
        if not raw_dropped_pairs and operation.get("dropped_pair_index") is not None:
            raw_dropped_pairs = [operation.get("dropped_pair_index")]
        dropped_pairs = _strict_int_list(
            raw_dropped_pairs,
            context=f"{path}: candidates[{row_index}].acquisition_operation.dropped_pairs",
        )
        operation_kind = str(operation.get("op") or "").strip()
        if operation_kind not in {
            "drop_one",
            "drop_two",
            "drop_many",
            "pair_frame_geometry_low_impact_drop_many",
        }:
            continue
        if not dropped_pairs:
            continue
        selected_pairs = _strict_int_list(
            _as_list(item.get("selected_pair_indices")),
            context=f"{path}: candidates[{row_index}].selected_pair_indices",
        )
        acquisition_id = str(item.get("acquisition_id") or item.get("candidate_id") or "")
        if not acquisition_id:
            raise ByteShavingCampaignError(
                f"{path}: candidates[{row_index}] missing acquisition_id"
            )
        selector_kind = str(item.get("selector_kind") or operation.get("op") or "pairset")
        selector_kind_counts[selector_kind] = selector_kind_counts.get(selector_kind, 0) + 1
        dropped_pair_counts.append(len(dropped_pairs))
        repair_budget_rows += 1
        unit_token = _safe_id_token(acquisition_id, fallback=f"pairset_{index}_{row_index}")
        operation_family = "drop_pair"
        unit = {
            "unit_id": f"dqs1_pairset_{unit_token}",
            "unit_kind": "pair",
            "candidate_saved_bytes": saved_bytes,
            "predicted_quality_score_delta": 0.0,
            "confidence": 0.45 if selector_kind.startswith("pair_frame_geometry") else 0.35,
            "operation_families": [operation_family],
            "operations": [
                {
                    "operation_id": f"materialize_{unit_token}",
                    "operation_family": operation_family,
                    "materializer": DQS1_PAIRSET_DROP_PAIR_MATERIALIZER,
                    "target_kind": DQS1_PAIRSET_DROP_PAIR_TARGET_KIND,
                    "candidate_saved_bytes": saved_bytes,
                    "predicted_quality_score_delta": 0.0,
                    "params": {
                        "source_schema": DQS1_PAIRSET_ACQUISITION_SCHEMA,
                        "source_path": _repo_rel(path, repo_root),
                        "acquisition_id": acquisition_id,
                        "selector_kind": selector_kind,
                        "selected_pair_indices": selected_pairs,
                        "dropped_pair_indices": dropped_pairs,
                        "operation_kind": operation_kind,
                        "candidate_payload_bytes": repair.get("candidate_payload_bytes"),
                        "source_selector_payload_bytes": repair.get(
                            "source_selector_payload_bytes"
                        ),
                        "score_budget": repair.get("score_budget"),
                        "segnet_distortion_budget_at_fixed_pose": repair.get(
                            "segnet_distortion_budget_at_fixed_pose"
                        ),
                        "posenet_score_term_budget_at_fixed_seg": repair.get(
                            "posenet_score_term_budget_at_fixed_seg"
                        ),
                        "acquisition_operation": dict(operation),
                    },
                    "receiver_contract_kind": DQS1_PAIRSET_RECEIVER_CONTRACT_KIND,
                    "materializer_contract_kind": DQS1_PAIRSET_DROP_PAIR_TARGET_KIND,
                    "materializer_adapter_registered": True,
                    "materializer_executable": False,
                    "executable_work_ready": False,
                    "materializer_execution_status": (
                        "registered_adapter_blocked_until_context_and_receiver_proof"
                    ),
                    "blockers": [
                        "dqs1_pairset_acquisition_signal_is_planning_only",
                        "requires_local_dqs1_materialization_and_locality_controls",
                        "requires_receiver_runtime_consumption_proof",
                        "requires_exact_auth_eval_before_score_claim",
                    ],
                }
            ],
            "source_candidate_id": acquisition_id,
            "source_paths": [_repo_rel(path, repo_root)],
            "source_index": row_index,
            "score_axis": payload.get("score_axis") or "[planning-only]",
            "evidence_grade": "local_dqs1_pairset_acquisition_planning_signal",
            "evidence_semantics": (
                "dqs1_pairset_rate_saving_repair_budget_planning_unit"
            ),
            "selected_pair_indices": selected_pairs,
            "dropped_pair_indices": dropped_pairs,
            "selector_kind": selector_kind,
            "distortion_repair_budget_from_rate_savings": dict(repair),
            "xray_signal": item.get("source_pair_frame_geometry_request"),
            "master_gradient_signal": {
                "status": operation.get(
                    "master_gradient_status",
                    "pair_frame_or_rank_proxy_until_pair_gradient_binding_lands",
                ),
                "source_operation": operation.get("op"),
            },
            "inverse_scorer_signal": {
                "status": operation.get(
                    "inverse_scorer_status",
                    "receiver_runtime_consumer_required_before_inverse_scorer_authority",
                ),
                "source_operation": operation.get("op"),
            },
            "blockers": [
                "dqs1_pairset_acquisition_unit_is_planning_only",
                "requires_local_dqs1_materialization_and_locality_controls",
                "requires_receiver_runtime_consumption_proof",
                "requires_exact_auth_eval_before_score_claim",
            ],
            **FALSE_AUTHORITY,
        }
        units.append(unit)

    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    ref = {
        **_json_payload_ref(
            path,
            repo_root,
            kind="decoder_q_pairset_acquisition",
            payload=payload,
        ),
        "candidate_count": summary.get("candidate_count"),
        "drop_many_candidate_count": summary.get("drop_many_candidate_count"),
        "pair_frame_geometry_candidate_count": summary.get(
            "pair_frame_geometry_candidate_count"
        ),
        "repair_budget_row_count": repair_budget_rows,
        "surface_unit_count": len(units),
        "selector_kind_counts": dict(sorted(selector_kind_counts.items())),
        "dropped_pair_count_values": sorted(set(dropped_pair_counts)),
        "allowed_use": "byte_shaving_planning_surface_and_local_dqs1_queue_ranking",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return units, payload, [ref]


def _finite_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def _finite_int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _strict_int_list(values: Sequence[Any], *, context: str) -> list[int]:
    out: list[int] = []
    for index, item in enumerate(values):
        if isinstance(item, bool):
            raise ByteShavingCampaignError(f"{context}[{index}]: bool is not an integer id")
        if isinstance(item, int):
            parsed = item
        elif (isinstance(item, float) and item.is_integer()) or (
            isinstance(item, str) and item.strip().lstrip("+-").isdigit()
        ):
            parsed = int(item)
        else:
            parsed = None
        if parsed is None:
            raise ByteShavingCampaignError(
                f"{context}[{index}]: expected integer id, got {item!r}"
            )
        if parsed < 0 or parsed > DQS1_MAX_PAIR_INDEX:
            raise ByteShavingCampaignError(
                f"{context}[{index}]: pair id {parsed} outside canonical range "
                f"0..{DQS1_MAX_PAIR_INDEX}"
            )
        out.append(parsed)
    return out


def _dqs1_observation_is_harvest(row: Mapping[str, Any]) -> bool:
    return (
        row.get("schema") == DQS1_OBSERVATION_ROW_SCHEMA
        and (
            row.get("source_schema") == DQS1_OBSERVATION_SOURCE_SCHEMA
            or row.get("sweep_config_id") == DQS1_OBSERVATION_SWEEP_CONFIG_ID
        )
    )


def _operation_dropped_pairs(operation: Mapping[str, Any], *, context: str) -> list[int]:
    raw = _as_list(operation.get("dropped_pair_indices"))
    if not raw and operation.get("dropped_pair_index") is not None:
        raw = [operation.get("dropped_pair_index")]
    return _strict_int_list(raw, context=context)


def _dqs1_outcome_surface(
    path: Path,
    repo_root: Path,
    *,
    index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        rows = load_observation_rows(path)
    except MLXDynamicSweepObservationError as exc:
        raise ByteShavingCampaignError(str(exc)) from exc

    units: list[dict[str, Any]] = []
    dqs1_row_count = 0
    ignored_non_dqs1_count = 0
    missing_runtime_evidence_count = 0
    byte_saving_row_count = 0
    improved_row_count = 0
    regressed_row_count = 0
    k_gt_one_row_count = 0
    operation_kind_counts: dict[str, int] = {}
    emitted_saved_bytes_sum = 0

    for row_index, row in enumerate(rows):
        if not _dqs1_observation_is_harvest(row):
            ignored_non_dqs1_count += 1
            continue
        dqs1_row_count += 1
        try:
            require_no_truthy_authority_fields(
                row,
                context=f"byte_shaving_dqs1_observation.rows[{row_index}]",
            )
        except ValueError as exc:
            raise ByteShavingCampaignError(str(exc)) from exc
        required_evidence = (
            "archive_sha256",
            "runtime_sha256",
            "raw_output_or_cache_sha256",
            "source_artifact_path",
            "source_artifact_sha256",
            "planner_artifact_path",
            "planner_artifact_sha256",
            "baseline_artifact_path",
            "baseline_artifact_sha256",
        )
        if any(not row.get(field) for field in required_evidence):
            missing_runtime_evidence_count += 1
            continue
        operation = row.get("acquisition_operation")
        if not isinstance(operation, Mapping):
            missing_runtime_evidence_count += 1
            continue
        try:
            require_no_truthy_authority_fields(
                operation,
                context=f"byte_shaving_dqs1_observation.rows[{row_index}].operation",
            )
        except ValueError as exc:
            raise ByteShavingCampaignError(str(exc)) from exc
        operation_kind = str(operation.get("op") or row.get("family") or "").strip()
        operation_kind_counts[operation_kind] = operation_kind_counts.get(operation_kind, 0) + 1
        dropped_pairs = _operation_dropped_pairs(
            operation,
            context=f"{path}: rows[{row_index}].acquisition_operation.dropped_pairs",
        )
        selected_pairs = _strict_int_list(
            _as_list(row.get("selected_pair_indices")),
            context=f"{path}: rows[{row_index}].selected_pair_indices",
        )
        if len(dropped_pairs) > 1:
            k_gt_one_row_count += 1
        archive_delta = _finite_int_or_none(row.get("archive_byte_delta_vs_baseline"))
        score_delta = _finite_float_or_none(row.get("score_delta_vs_baseline"))
        if archive_delta is None or score_delta is None:
            missing_runtime_evidence_count += 1
            continue
        if score_delta < 0.0:
            improved_row_count += 1
        elif score_delta > 0.0:
            regressed_row_count += 1
        if archive_delta >= 0:
            continue
        saved_bytes = abs(archive_delta)
        if saved_bytes <= 0 or not dropped_pairs:
            continue
        byte_saving_row_count += 1
        emitted_saved_bytes_sum += saved_bytes
        rate_delta = -RATE_SCORE_PER_BYTE * float(saved_bytes)
        quality_delta = score_delta - rate_delta
        token = _safe_id_token(
            row.get("candidate_id"),
            fallback=f"dqs1_outcome_{index}_{row_index}",
        )
        component_deltas = (
            dict(row.get("component_deltas"))
            if isinstance(row.get("component_deltas"), Mapping)
            else {}
        )
        outcome_signal = {
            "schema": DQS1_PAIR_FRAME_OUTCOME_ANCHOR_SCHEMA,
            "source_observation_path": _repo_rel(path, repo_root),
            "source_observation_row_index": row_index,
            "candidate_id": row.get("candidate_id"),
            "operation_kind": operation_kind,
            "selected_pair_indices": selected_pairs,
            "dropped_pair_indices": dropped_pairs,
            "observed_axis": row.get("observed_axis"),
            "evidence_tag": row.get("evidence_tag"),
            "score_delta_vs_baseline": score_delta,
            "archive_byte_delta_vs_baseline": archive_delta,
            "component_deltas": component_deltas,
            "archive_sha256": row.get("archive_sha256"),
            "runtime_sha256": row.get("runtime_sha256"),
            "raw_output_or_cache_sha256": row.get("raw_output_or_cache_sha256"),
            "source_artifact_path": row.get("source_artifact_path"),
            "planner_artifact_path": row.get("planner_artifact_path"),
            "baseline_artifact_path": row.get("baseline_artifact_path"),
            "allowed_use": (
                "planning_anchor_for_byte_shaving_campaign_inverse_scorer_"
                "master_gradient_and_bit_allocator_only"
            ),
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
            **FALSE_AUTHORITY,
        }
        units.append(
            {
                "unit_id": f"dqs1_pair_frame_outcome_{token}",
                "unit_kind": "pair",
                "candidate_saved_bytes": saved_bytes,
                "predicted_quality_score_delta": quality_delta,
                "confidence": 0.62 if score_delta <= 0.0 else 0.38,
                "operation_families": ["drop_pair"],
                "operations": [
                    {
                        "operation_id": f"replan_from_{token}",
                        "operation_family": "drop_pair",
                        "materializer": DQS1_PAIRSET_DROP_PAIR_MATERIALIZER,
                        "target_kind": DQS1_PAIRSET_DROP_PAIR_TARGET_KIND,
                        "candidate_saved_bytes": saved_bytes,
                        "predicted_quality_score_delta": quality_delta,
                        "params": {
                            "source_schema": DQS1_OBSERVATION_ROW_SCHEMA,
                            "source_path": _repo_rel(path, repo_root),
                            "candidate_id": row.get("candidate_id"),
                            "operation_kind": operation_kind,
                            "selected_pair_indices": selected_pairs,
                            "dropped_pair_indices": dropped_pairs,
                            "observed_score_delta_vs_baseline": score_delta,
                            "observed_archive_byte_delta_vs_baseline": archive_delta,
                            "component_deltas": component_deltas,
                            "source_operation": dict(operation),
                            "outcome_signal": outcome_signal,
                        },
                        "receiver_contract_kind": DQS1_PAIRSET_RECEIVER_CONTRACT_KIND,
                        "materializer_contract_kind": DQS1_PAIRSET_DROP_PAIR_TARGET_KIND,
                        "materializer_adapter_registered": True,
                        "materializer_executable": False,
                        "executable_work_ready": False,
                        "materializer_execution_status": (
                            "registered_adapter_blocked_until_context_and_receiver_proof"
                        ),
                        "blockers": [
                            "dqs1_outcome_anchor_is_local_advisory_only",
                            "requires_receiver_runtime_consumption_proof",
                            "requires_exact_auth_eval_before_score_claim",
                        ],
                    }
                ],
                "source_candidate_id": row.get("candidate_id"),
                "source_paths": [_repo_rel(path, repo_root)],
                "source_index": row_index,
                "score_axis": row.get("observed_axis"),
                "evidence_grade": row.get("evidence_grade"),
                "evidence_semantics": (
                    "dqs1_pair_frame_geometry_empirical_rate_distortion_anchor"
                ),
                "selected_pair_indices": selected_pairs,
                "dropped_pair_indices": dropped_pairs,
                "dqs1_outcome_signal": outcome_signal,
                "master_gradient_signal": {
                    "status": "empirical_pairset_outcome_anchor_ready",
                    "consumer": "tac.master_gradient_pair_frame_binding",
                    "archive_sha256": row.get("archive_sha256"),
                    "component_deltas": component_deltas,
                    "score_delta_vs_baseline": score_delta,
                    "selected_pair_indices": selected_pairs,
                    "dropped_pair_indices": dropped_pairs,
                    **FALSE_AUTHORITY,
                },
                "inverse_scorer_signal": {
                    "status": "empirical_pairset_outcome_anchor_ready",
                    "consumer": "tac.optimization.scorer_inverse_decision_surface",
                    "operation_kind": operation_kind,
                    "score_delta_vs_baseline": score_delta,
                    **FALSE_AUTHORITY,
                },
                "bit_allocator_signal": {
                    "status": "empirical_pairset_rate_distortion_budget_anchor_ready",
                    "consumer": "tac.optimization.bit_allocator",
                    "saved_bytes": saved_bytes,
                    "quality_delta_score": quality_delta,
                    "component_deltas": component_deltas,
                    **FALSE_AUTHORITY,
                },
                "blockers": [
                    "dqs1_pair_frame_outcome_anchor_is_planning_only",
                    "macos_cpu_advisory_is_not_contest_authority",
                    "requires_receiver_runtime_consumption_proof",
                    "requires_exact_auth_eval_before_score_claim",
                ],
                **FALSE_AUTHORITY,
            }
        )

    ref = {
        **_json_payload_ref(
            path,
            repo_root,
            kind=DQS1_PAIR_FRAME_OUTCOME_REF_KIND,
            payload={"schema": DQS1_OBSERVATION_ROW_SCHEMA},
        ),
        "schema": DQS1_PAIR_FRAME_OUTCOME_ANCHOR_SCHEMA,
        "source_row_schema": DQS1_OBSERVATION_ROW_SCHEMA,
        "row_count": len(rows),
        "dqs1_row_count": dqs1_row_count,
        "ignored_non_dqs1_count": ignored_non_dqs1_count,
        "missing_runtime_evidence_count": missing_runtime_evidence_count,
        "byte_saving_row_count": byte_saving_row_count,
        "emitted_unit_count": len(units),
        "emitted_saved_bytes_sum": emitted_saved_bytes_sum,
        "improved_row_count": improved_row_count,
        "regressed_row_count": regressed_row_count,
        "k_gt_one_row_count": k_gt_one_row_count,
        "operation_kind_counts": dict(sorted(operation_kind_counts.items())),
        "planner_consumers": [
            "byte_shaving_campaign_plan",
            "inverse_steganalysis_action_functional",
            "master_gradient_pair_frame_binding",
            "inverse_scorer_decision_surface",
            "bit_allocator_pair_budget",
        ],
        "allowed_use": (
            "planning_anchor_for_solver_ranking_and_repair_budget_modeling_only"
        ),
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return units, [ref]


def _inverse_scorer_surface(
    path: Path,
    repo_root: Path,
    *,
    max_units: int,
    null_scorer_delta_epsilon: float,
    fragile_scorer_delta_threshold: float,
    allow_native_mlx_window_objective: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json_object(path)
    try:
        surface = build_inverse_scorer_decision_surface(
            payload,
            source_label=_repo_rel(path, repo_root),
            max_units=max_units,
            null_scorer_delta_epsilon=null_scorer_delta_epsilon,
            fragile_scorer_delta_threshold=fragile_scorer_delta_threshold,
            allow_native_mlx_window_objective=allow_native_mlx_window_objective,
        )
    except ScorerResponseDatasetError as exc:
        raise ByteShavingCampaignError(str(exc)) from exc
    ref = {
        **_json_payload_ref(
            path,
            repo_root,
            kind="scorer_inverse_decision_surface",
            payload=surface,
        ),
        "source_row_count": surface["source_row_count"],
        "cell_count": surface["cell_count"],
        "emitted_unit_count": surface["emitted_unit_count"],
        "decision_surface_classes": list(surface["decision_surface_classes"]),
        "allow_native_mlx_window_objective": surface["allow_native_mlx_window_objective"],
        "planning_scope": "compressed_scorer_coordinate",
        "blockers": _as_list(surface.get("blockers")),
        "dispatch_blockers": _as_list(surface.get("dispatch_blockers")),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return ref, [dict(unit) for unit in _as_list(surface.get("units"))]


def _xray_hook_refs(hooks: Sequence[str]) -> list[dict[str, Any]]:
    if not hooks:
        return []
    from tac.xray.registry import specs_by_hook

    refs: list[dict[str, Any]] = []
    for hook in ordered_unique(str(hook) for hook in hooks):
        specs = specs_by_hook(hook)  # type: ignore[arg-type]
        refs.append({
            "kind": "xray_hook_inventory",
            "hook": hook,
            "primitive_count": len(specs),
            "primitives": [
                {
                    "primitive_name": spec.primitive_name,
                    "canonical_module": spec.canonical_module,
                    "category": spec.category,
                    "evidence_grade": spec.evidence_grade,
                    "wire_in_hooks": list(spec.wire_in_hooks),
                    "upstream_memo": spec.upstream_memo,
                }
                for spec in specs
            ],
        })
    return refs


def _canonical_equation_refs(
    *,
    repo_root: Path,
    domains: Sequence[str],
    consumers: Sequence[str],
    registry_path: str | Path | None,
) -> list[dict[str, Any]]:
    if not domains and not consumers:
        return []
    from tac.canonical_equations.registry import (
        query_equations_by_consumer,
        query_equations_by_domain,
    )

    path = None if registry_path is None else _resolve_path(registry_path, repo_root)
    refs_by_id: dict[str, dict[str, Any]] = {}

    def add(eq: Any, *, reason: str, token: str) -> None:
        payload = eq.to_dict()
        eq_id = str(payload["equation_id"])
        ref = refs_by_id.setdefault(
            eq_id,
            {
                "kind": "canonical_equation",
                "equation_id": eq_id,
                "name": payload.get("name"),
                "one_line_summary": payload.get("one_line_summary"),
                "python_callable_module_path": payload.get(
                    "python_callable_module_path"
                ),
                "domain_of_validity": payload.get("domain_of_validity"),
                "units_in": payload.get("units_in"),
                "units_out": payload.get("units_out"),
                "empirical_anchor_count": len(
                    _as_list(payload.get("empirical_anchors"))
                ),
                "predicted_vs_empirical_residual": payload.get(
                    "predicted_vs_empirical_residual"
                ),
                "last_calibration_utc": payload.get("last_calibration_utc"),
                "canonical_consumers": payload.get("canonical_consumers"),
                "canonical_producers": payload.get("canonical_producers"),
                "query_matches": [],
            },
        )
        ref["query_matches"].append({"reason": reason, "token": token})

    for token in ordered_unique(str(item) for item in domains):
        for eq in query_equations_by_domain(token, path=path):
            add(eq, reason="domain", token=token)
    for token in ordered_unique(str(item) for item in consumers):
        for eq in query_equations_by_consumer(token, path=path):
            add(eq, reason="consumer", token=token)
    return sorted(refs_by_id.values(), key=lambda row: str(row["equation_id"]))


def _atom_refs(
    *,
    repo_root: Path,
    atom_ids: Sequence[str],
    min_predicted_impact: float | None,
    ledger_path: str | Path | None,
) -> list[dict[str, Any]]:
    if not atom_ids and min_predicted_impact is None:
        return []
    from tac.atom.ledger import ATOM_LEDGER_PATH, load_atoms_strict

    path = _resolve_path(ledger_path, repo_root) if ledger_path is not None else ATOM_LEDGER_PATH
    rows = load_atoms_strict(path)
    wanted = set(ordered_unique(str(item) for item in atom_ids))
    refs: list[dict[str, Any]] = []
    for row in rows:
        atom = row.get("atom")
        if not isinstance(atom, Mapping):
            continue
        atom_id = str(atom.get("atom_id") or "")
        if wanted and atom_id not in wanted:
            continue
        lo = atom.get("predicted_impact_delta_s_lower")
        if min_predicted_impact is not None and (
            not isinstance(lo, int | float) or float(lo) < min_predicted_impact
        ):
            continue
        try:
            require_no_truthy_authority_fields(
                row,
                context=f"byte_shaving_atom_ref:{atom_id}",
            )
        except ValueError as exc:
            raise ByteShavingCampaignError(str(exc)) from exc
        refs.append({
            "kind": "atom_ledger_row",
            "ledger_path": _repo_rel(path, repo_root),
            "atom_id": atom_id,
            "atom_kind": atom.get("kind"),
            "resolution_path": atom.get("resolution_path"),
            "predicted_impact_delta_s_lower": lo,
            "predicted_impact_delta_s_upper": atom.get(
                "predicted_impact_delta_s_upper"
            ),
            "wired_hooks": _as_list(atom.get("wired_hooks")),
            "observability_surface": _as_list(atom.get("observability_surface")),
            "event_type": row.get("event_type"),
            "written_at_utc": row.get("written_at_utc"),
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        })
    return sorted(refs, key=lambda row: str(row.get("atom_id")))


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_byte_shaving_signal_surface(
    *,
    repo_root: str | Path = ".",
    campaign_id: str = "byte_shaving_signal_surface",
    candidate_queue_paths: Sequence[str | Path] = (),
    pairset_acquisition_paths: Sequence[str | Path] = (),
    dqs1_observation_paths: Sequence[str | Path] = (),
    engineered_correction_targeting_paths: Sequence[str | Path] = (),
    engineered_correction_max_targets: int | None = None,
    engineered_correction_default_predicted_quality_score_delta: float = 0.0,
    inverse_action_functional_paths: Sequence[str | Path] = (),
    allow_inverse_action_leaf_cell_candidates: bool = False,
    master_gradient_archive_sha256s: Sequence[str] = (),
    master_gradient_ledger_path: str | Path | None = None,
    master_gradient_axis: str | None = None,
    master_gradient_low_sensitivity_quantile: float = 0.05,
    master_gradient_max_units: int = 32,
    master_gradient_max_span_bytes: int = 4096,
    master_gradient_quality_cost_multiplier: float = 1.0,
    auth_eval_paths: Sequence[str | Path] = (),
    mlx_calibration_paths: Sequence[str | Path] = (),
    scorer_response_paths: Sequence[str | Path] = (),
    inverse_scorer_response_paths: Sequence[str | Path] = (),
    inverse_scorer_max_units: int = 16,
    inverse_scorer_null_delta_epsilon: float = 1e-6,
    inverse_scorer_fragile_delta_threshold: float = 0.0,
    inverse_scorer_allow_native_mlx_window_objective: bool = False,
    xray_hooks: Sequence[str] = (),
    canonical_equation_domains: Sequence[str] = (),
    canonical_equation_consumers: Sequence[str] = (),
    canonical_equation_registry_path: str | Path | None = None,
    atom_ids: Sequence[str] = (),
    atom_min_predicted_impact: float | None = None,
    atom_ledger_path: str | Path | None = None,
    candidate_id: str | None = None,
    lane_id: str = "byte_shaving_signal_surface",
    frontier_axis: str = "[planning-only]",
) -> dict[str, Any]:
    """Build one validated planning-only byte-shaving surface."""

    repo = Path(repo_root)
    units: list[dict[str, Any]] = []
    seen_unit_ids: set[str] = set()
    source_signal_refs: list[dict[str, Any]] = []
    auth_eval_refs: list[dict[str, Any]] = []
    mlx_calibration_refs: list[dict[str, Any]] = []
    scorer_response_refs: list[dict[str, Any]] = []
    pairset_acquisition_refs: list[dict[str, Any]] = []
    pair_frame_geometry_outcome_refs: list[dict[str, Any]] = []
    inverse_scorer_surface_refs: list[dict[str, Any]] = []
    engineered_correction_refs: list[dict[str, Any]] = []
    inverse_action_functional_refs: list[dict[str, Any]] = []
    inverse_action_materialization_portfolios: list[dict[str, Any]] = []
    surface_blockers: list[str] = []

    for index, raw_path in enumerate(candidate_queue_paths):
        path = _resolve_path(raw_path, repo)
        if not path.is_file():
            raise ByteShavingCampaignError(f"candidate queue not found: {path}")
        queue_units, _surface, refs = _candidate_queue_surface(
            path,
            repo,
            campaign_id=campaign_id,
            index=index,
        )
        for unit in queue_units:
            units.append(
                _dedupe_unit_id(unit, prefix=f"candidate_queue_{index}", seen=seen_unit_ids)
            )
        _extend_refs(source_signal_refs, refs["source_signal_refs"])
        _extend_refs(auth_eval_refs, refs["auth_eval_refs"])
        _extend_refs(mlx_calibration_refs, refs["mlx_calibration_refs"])
        _extend_refs(scorer_response_refs, refs["scorer_response_refs"])

    for index, raw_path in enumerate(pairset_acquisition_paths):
        path = _resolve_path(raw_path, repo)
        if not path.is_file():
            raise ByteShavingCampaignError(f"pairset acquisition JSON not found: {path}")
        pairset_units, _pairset_payload, refs = _pairset_acquisition_surface(
            path,
            repo,
            index=index,
        )
        for unit in pairset_units:
            units.append(
                _dedupe_unit_id(
                    unit,
                    prefix=f"pairset_acquisition_{index}",
                    seen=seen_unit_ids,
                )
            )
        _extend_refs(pairset_acquisition_refs, refs)
        _extend_refs(source_signal_refs, refs)

    for index, raw_path in enumerate(dqs1_observation_paths):
        path = _resolve_path(raw_path, repo)
        if not path.is_file():
            raise ByteShavingCampaignError(f"DQS1 observation JSONL not found: {path}")
        outcome_units, refs = _dqs1_outcome_surface(
            path,
            repo,
            index=index,
        )
        for unit in outcome_units:
            units.append(
                _dedupe_unit_id(
                    unit,
                    prefix=f"dqs1_outcome_{index}",
                    seen=seen_unit_ids,
                )
            )
        _extend_refs(pair_frame_geometry_outcome_refs, refs)
        _extend_refs(source_signal_refs, refs)

    for index, raw_path in enumerate(engineered_correction_targeting_paths):
        path = _resolve_path(raw_path, repo)
        if not path.is_file():
            raise ByteShavingCampaignError(
                f"engineered correction targeting JSON not found: {path}"
            )
        ec_units, ec_surface, refs = _engineered_correction_targeting_surface(
            path,
            repo,
            campaign_id=campaign_id,
            index=index,
            max_targets=engineered_correction_max_targets,
            default_predicted_quality_score_delta=(
                engineered_correction_default_predicted_quality_score_delta
            ),
        )
        for unit in ec_units:
            units.append(
                _dedupe_unit_id(
                    unit,
                    prefix=f"engineered_correction_{index}",
                    seen=seen_unit_ids,
                )
            )
        _extend_refs(source_signal_refs, _as_list(ec_surface.get("source_signal_refs")))
        _extend_refs(engineered_correction_refs, refs)

    for index, raw_path in enumerate(inverse_action_functional_paths):
        path = _resolve_path(raw_path, repo)
        if not path.is_file():
            raise ByteShavingCampaignError(
                f"inverse action functional JSON not found: {path}"
            )
        ia_units, ia_surface, refs = _inverse_action_functional_surface(
            path,
            repo,
            campaign_id=campaign_id,
            index=index,
            allow_leaf_cell_candidates=allow_inverse_action_leaf_cell_candidates,
        )
        for unit in ia_units:
            units.append(
                _dedupe_unit_id(
                    unit,
                    prefix=f"inverse_action_functional_{index}",
                    seen=seen_unit_ids,
                )
            )
        _extend_refs(source_signal_refs, _as_list(ia_surface.get("source_signal_refs")))
        _extend_refs(inverse_action_functional_refs, refs)
        portfolio = ia_surface.get("water_bucket_materialization_portfolio")
        if isinstance(portfolio, Mapping):
            inverse_action_materialization_portfolios.append(
                {
                    **dict(portfolio),
                    "source_path": _repo_rel(path, repo),
                    "source_sha256": _file_sha256(path),
                    "surface_campaign_id": ia_surface.get("campaign_id"),
                }
            )
        surface_blockers.extend(str(item) for item in _as_list(ia_surface.get("blockers")))

    for index, archive_sha256 in enumerate(master_gradient_archive_sha256s):
        mg_units, mg_surface = _master_gradient_surface(
            repo,
            archive_sha256=archive_sha256,
            ledger_path=master_gradient_ledger_path,
            axis=master_gradient_axis,
            campaign_id=campaign_id,
            index=index,
            low_sensitivity_quantile=master_gradient_low_sensitivity_quantile,
            max_units=master_gradient_max_units,
            max_span_bytes=master_gradient_max_span_bytes,
            quality_cost_multiplier=master_gradient_quality_cost_multiplier,
        )
        for unit in mg_units:
            units.append(
                _dedupe_unit_id(unit, prefix=f"master_gradient_{index}", seen=seen_unit_ids)
            )
        _extend_refs(source_signal_refs, _as_list(mg_surface.get("source_signal_refs")))

    for raw_path in auth_eval_paths:
        path = _resolve_path(raw_path, repo)
        if not path.is_file():
            raise ByteShavingCampaignError(f"auth eval JSON not found: {path}")
        auth_eval_refs.append(_auth_eval_ref(path, repo))
    for raw_path in mlx_calibration_paths:
        path = _resolve_path(raw_path, repo)
        if not path.is_file():
            raise ByteShavingCampaignError(f"MLX calibration JSON not found: {path}")
        mlx_calibration_refs.append(_mlx_calibration_ref(path, repo))
    for raw_path in scorer_response_paths:
        path = _resolve_path(raw_path, repo)
        if not path.is_file():
            raise ByteShavingCampaignError(f"scorer-response JSON not found: {path}")
        scorer_response_refs.append(_scorer_response_ref(path, repo))
        for unit in _scorer_response_units(path, repo, index=len(scorer_response_refs) - 1):
            units.append(
                _dedupe_unit_id(
                    unit,
                    prefix=f"scorer_response_{len(scorer_response_refs) - 1}",
                    seen=seen_unit_ids,
                )
            )
    for raw_path in inverse_scorer_response_paths:
        path = _resolve_path(raw_path, repo)
        if not path.is_file():
            raise ByteShavingCampaignError(f"inverse scorer-response JSON not found: {path}")
        inverse_ref, inverse_units = _inverse_scorer_surface(
            path,
            repo,
            max_units=inverse_scorer_max_units,
            null_scorer_delta_epsilon=inverse_scorer_null_delta_epsilon,
            fragile_scorer_delta_threshold=inverse_scorer_fragile_delta_threshold,
            allow_native_mlx_window_objective=(
                inverse_scorer_allow_native_mlx_window_objective
            ),
        )
        inverse_scorer_surface_refs.append(inverse_ref)
        for unit in inverse_units:
            unit["blockers"] = ordered_unique(
                [
                    *[str(item) for item in _as_list(unit.get("blockers"))],
                    *[str(item) for item in _as_list(inverse_ref.get("blockers"))],
                    *[str(item) for item in _as_list(inverse_ref.get("dispatch_blockers"))],
                ]
            )
            units.append(
                _dedupe_unit_id(
                    unit,
                    prefix=f"inverse_scorer_{len(inverse_scorer_surface_refs) - 1}",
                    seen=seen_unit_ids,
                )
            )
        surface_blockers.extend(str(item) for item in _as_list(inverse_ref.get("blockers")))
        surface_blockers.extend(
            str(item) for item in _as_list(inverse_ref.get("dispatch_blockers"))
        )

    payload = {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": campaign_id,
        "candidate_id": candidate_id,
        "lane_id": lane_id,
        "frontier_axis": frontier_axis,
        "source_signal_refs": source_signal_refs,
        "auth_eval_refs": auth_eval_refs,
        "mlx_calibration_refs": mlx_calibration_refs,
        "scorer_response_refs": scorer_response_refs,
        "pairset_acquisition_refs": pairset_acquisition_refs,
        "pair_frame_geometry_outcome_refs": pair_frame_geometry_outcome_refs,
        "inverse_scorer_surface_refs": inverse_scorer_surface_refs,
        "engineered_correction_refs": engineered_correction_refs,
        "inverse_action_functional_refs": inverse_action_functional_refs,
        "inverse_action_materialization_portfolios": (
            inverse_action_materialization_portfolios
        ),
        "xray_refs": _xray_hook_refs(xray_hooks),
        "canonical_equation_refs": _canonical_equation_refs(
            repo_root=repo,
            domains=canonical_equation_domains,
            consumers=canonical_equation_consumers,
            registry_path=canonical_equation_registry_path,
        ),
        "atom_refs": _atom_refs(
            repo_root=repo,
            atom_ids=atom_ids,
            min_predicted_impact=atom_min_predicted_impact,
            ledger_path=atom_ledger_path,
        ),
        "units": units,
        "blockers": [
            "byte_shaving_signal_surface_is_planning_only",
            "source_refs_are_not_score_authority",
            *ordered_unique(surface_blockers),
            "requires_materializer_before_candidate_archive",
            "requires_locality_controls_before_exact_eval",
            "requires_exact_auth_eval_before_score_claim",
        ],
        "evidence_boundary": {
            "planning_only": True,
            "auth_eval_refs_may_be_score_bearing_sources": True,
            "surface_score_authority": False,
            "next_gate": (
                "materialize selected byte-shaving operations, run local "
                "controls, then submit only byte-closed packets to exact auth eval"
            ),
        },
        **FALSE_AUTHORITY,
    }
    validate_signal_surface(payload)
    return payload


__all__ = ["build_byte_shaving_signal_surface"]
