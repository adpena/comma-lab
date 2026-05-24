# SPDX-License-Identifier: MIT
"""Execute local MLX learned-sweep rows and append observations."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.exact_eval_custody import CONTEST_EXACT_SAMPLE_COUNT
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_scorer_response import (
    build_mlx_scorer_response_payload,
    write_mlx_scorer_response_payload,
)
from tac.optimization.mlx_dynamic_learned_sweep import (
    FALSE_AUTHORITY,
    render_mlx_dynamic_learned_sweep_markdown,
)
from tac.optimization.mlx_dynamic_sweep_observations import (
    ROW_SCHEMA as OBSERVATION_ROW_SCHEMA,
)
from tac.optimization.mlx_dynamic_sweep_observations import (
    append_observation_row,
    build_observation_row,
    deduplicate_observation_rows,
    load_observation_rows,
    summarize_observations,
)
from tac.optimization.mlx_effective_spend_triage_selection import (
    ROW_SCHEMA as SELECTION_ROW_SCHEMA,
)
from tac.optimization.mlx_effective_spend_triage_selection import (
    SCHEMA as SELECTION_SCHEMA,
)
from tac.optimization.normalized_objective import RATE_SCORE_PER_BYTE
from tac.repo_io import sha256_bytes, sha256_file

SCHEMA = "mlx_dynamic_learned_sweep_local_actuation.v1"
TOOL = "tac.optimization.mlx_dynamic_learned_sweep_local_actuator"
SUPPORTED_SWEEP_CONFIG_ID = "mlx_local_response"

ResponseBuilder = Callable[..., dict[str, Any]]


class MLXDynamicLearnedSweepLocalActuatorError(ValueError):
    """Raised when a dynamic learned-sweep row cannot be locally actuated."""


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MLXDynamicLearnedSweepLocalActuatorError(f"{path}: expected JSON object")
    return payload


def execute_local_mlx_sweep_rows(
    *,
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    output_dir: Path,
    observation_jsonl: Path | None = None,
    max_rows: int = 1,
    sweep_config_id: str = SUPPORTED_SWEEP_CONFIG_ID,
    optimization_pass_id: str | None = None,
    candidate_ids: Sequence[str] | None = None,
    queue_candidate_ids: Sequence[str] | None = None,
    source_artifact_root: str | Path | None = None,
    device_type: str = "cpu",
    allow_gpu_research_signal: bool = False,
    batch_pairs: int = 1,
    progress_every: int = 0,
    allow_duplicate_observation: bool = False,
    response_builder: ResponseBuilder = build_mlx_scorer_response_payload,
) -> dict[str, Any]:
    """Execute supported local MLX rows and optionally append observations.

    This is an actuator, not a scheduler. It consumes an already-ranked
    ``mlx_dynamic_learned_sweep_plan.v1`` and only handles cache-backed
    ``mlx_local_response`` rows. Other configs remain plan rows until a real
    executor exists for them.
    """

    if plan.get("schema") != "mlx_dynamic_learned_sweep_plan.v1":
        raise MLXDynamicLearnedSweepLocalActuatorError("plan schema mismatch")
    if selection.get("schema") != SELECTION_SCHEMA:
        raise MLXDynamicLearnedSweepLocalActuatorError("selection schema mismatch")
    _require_false_authority(plan, label="plan")
    _require_false_authority(selection, label="selection")
    if max_rows <= 0:
        raise MLXDynamicLearnedSweepLocalActuatorError("max_rows must be positive")
    if sweep_config_id != SUPPORTED_SWEEP_CONFIG_ID:
        raise MLXDynamicLearnedSweepLocalActuatorError(
            f"unsupported sweep_config_id {sweep_config_id!r}; "
            f"only {SUPPORTED_SWEEP_CONFIG_ID!r} has a local MLX executor"
        )
    if batch_pairs <= 0:
        raise MLXDynamicLearnedSweepLocalActuatorError("batch_pairs must be positive")
    candidate_id_filters = _normalize_filter_values(
        candidate_ids,
        label="candidate_ids",
    )
    queue_candidate_id_filters = _normalize_filter_values(
        queue_candidate_ids,
        label="queue_candidate_ids",
    )
    if queue_candidate_id_filters and max_rows < len(queue_candidate_id_filters):
        raise MLXDynamicLearnedSweepLocalActuatorError(
            "max_rows must cover every requested queue_candidate_id"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    selection_by_candidate = _selection_rows_by_candidate(selection)
    selected_rows = _select_ready_rows(
        plan,
        sweep_config_id=sweep_config_id,
        optimization_pass_id=optimization_pass_id,
        candidate_ids=candidate_id_filters,
        queue_candidate_ids=queue_candidate_id_filters,
        max_rows=max_rows,
    )
    if not selected_rows:
        raise MLXDynamicLearnedSweepLocalActuatorError(
            "no ready local sweep rows matched actuator filters"
        )

    executed: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for row in selected_rows:
        candidate_id = _required_text(row, "candidate_id")
        source_row = selection_by_candidate.get(candidate_id)
        if source_row is None:
            raise MLXDynamicLearnedSweepLocalActuatorError(
                f"{candidate_id}: missing source selection row"
            )
        executed_row, observation = _execute_one_row(
            row,
            source_row,
            output_dir=output_dir,
            source_artifact_root=source_artifact_root,
            device_type=device_type,
            allow_gpu_research_signal=allow_gpu_research_signal,
            batch_pairs=batch_pairs,
            progress_every=progress_every,
            response_builder=response_builder,
        )
        if observation_jsonl is not None:
            observation = append_observation_row(
                observation,
                output_path=observation_jsonl,
                allow_duplicate_observation=allow_duplicate_observation,
            )
        executed_row["observation_row"] = observation
        observations.append(observation)
        executed.append(executed_row)

    filter_summary = _executed_filter_summary(
        executed,
        optimization_pass_id=optimization_pass_id,
        candidate_ids=candidate_id_filters,
        queue_candidate_ids=queue_candidate_id_filters,
    )
    return {
        "schema": SCHEMA,
        "producer": TOOL,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "observation_only": True,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "local_mlx_device_used": device_type == "gpu",
        "allowed_use": "local_mlx_learned_sweep_actuation_feedback_only",
        "supported_sweep_config_id": SUPPORTED_SWEEP_CONFIG_ID,
        "device_type": device_type,
        "allow_gpu_research_signal": bool(allow_gpu_research_signal),
        "batch_pairs": int(batch_pairs),
        "optimization_pass_id_filter": optimization_pass_id,
        "candidate_id_filters": candidate_id_filters,
        "queue_candidate_id_filters": queue_candidate_id_filters,
        **filter_summary,
        "observation_jsonl": None if observation_jsonl is None else str(observation_jsonl),
        "executed_row_count": len(executed),
        "observation_row_count": len(observations),
        "observation_row_schema": OBSERVATION_ROW_SCHEMA,
        "observation_summary": summarize_observations(observations),
        "executed_rows": executed,
    }


def replan_after_local_actuation(
    *,
    incumbent_score: float,
    candidate_payloads: Sequence[Mapping[str, Any]],
    source_plan: Mapping[str, Any],
    observation_jsonl_paths: Sequence[Path],
    json_out: Path,
    md_out: Path | None = None,
    top_k: int | None = None,
    per_pass_top_k: int | None = None,
) -> dict[str, Any]:
    """Rerun the learned-sweep planner after local observations append."""

    from tac.optimization.mlx_dynamic_learned_sweep import (
        build_mlx_dynamic_learned_sweep_plan,
        write_json,
    )

    raw_observations: list[dict[str, Any]] = []
    for path in observation_jsonl_paths:
        raw_observations.extend(load_observation_rows(path))
    observations = deduplicate_observation_rows(raw_observations)
    policy = source_plan.get("selection_policy")
    plan = build_mlx_dynamic_learned_sweep_plan(
        incumbent_score=incumbent_score,
        candidate_payloads=candidate_payloads,
        execution_configs=_sequence_dicts(source_plan.get("execution_configs")),
        optimization_passes=_sequence_dicts(source_plan.get("optimization_passes")),
        observations=observations,
        top_k=int(top_k or _mapping_value(policy, "top_k", default=32)),
        per_pass_top_k=(
            per_pass_top_k
            if per_pass_top_k is not None
            else _optional_int(_mapping_value(policy, "per_pass_top_k"))
        ),
        lcb_z=float(_mapping_value(policy, "lcb_z", default=1.0)),
        expected_improvement_weight=float(
            _mapping_value(policy, "expected_improvement_weight", default=1.0)
        ),
        exploration_weight=float(_mapping_value(policy, "exploration_weight", default=1.0)),
        source_artifacts={
            "source_plan": source_plan.get("source_artifacts"),
            "observation_jsonl": [str(path) for path in observation_jsonl_paths],
            "raw_observation_row_count": len(raw_observations),
            "deduplicated_observation_row_count": len(observations),
            "duplicate_observation_row_count": len(raw_observations) - len(observations),
        },
    )
    write_json(json_out, plan)
    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(render_mlx_dynamic_learned_sweep_markdown(plan), encoding="utf-8")
    return plan


def _execute_one_row(
    row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    *,
    output_dir: Path,
    source_artifact_root: str | Path | None,
    device_type: str,
    allow_gpu_research_signal: bool,
    batch_pairs: int,
    progress_every: int,
    response_builder: ResponseBuilder,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_false_authority(row, label="plan row")
    if row.get("ready_for_local_sweep") is not True:
        raise MLXDynamicLearnedSweepLocalActuatorError("plan row is not local-ready")
    if row.get("sweep_config_id") != SUPPORTED_SWEEP_CONFIG_ID:
        raise MLXDynamicLearnedSweepLocalActuatorError(
            f"row sweep_config_id must be {SUPPORTED_SWEEP_CONFIG_ID}"
        )
    candidate_source_path = _resolve_existing_path(
        source_row.get("source_path"),
        source_artifact_root=source_artifact_root,
        label="source_path",
    )
    baseline_source_path = _resolve_existing_path(
        source_row.get("window_baseline_source_path"),
        source_artifact_root=source_artifact_root,
        label="window_baseline_source_path",
    )
    candidate_source = load_json_object(candidate_source_path)
    baseline_source = load_json_object(baseline_source_path)
    sample_budget = int(_required_float(row, "sample_budget"))
    if sample_budget <= 0:
        raise MLXDynamicLearnedSweepLocalActuatorError("sample_budget must be positive")
    start_pair = int(candidate_source.get("start_pair") or _first_pair_index(row))
    max_pairs = min(
        sample_budget,
        int(candidate_source.get("total_cache_pairs") or CONTEST_EXACT_SAMPLE_COUNT)
        - start_pair,
    )
    if max_pairs <= 0:
        raise MLXDynamicLearnedSweepLocalActuatorError("computed max_pairs is not positive")

    safe_id = _safe_id(_required_text(row, "queue_candidate_id"))
    row_dir = output_dir / safe_id
    candidate_components = row_dir / "candidate_components"
    baseline_components = row_dir / "baseline_components"
    candidate_response_path = row_dir / "candidate_response.json"
    baseline_response_path = row_dir / "baseline_response.json"

    candidate_response = _run_response(
        candidate_source,
        output_path=candidate_response_path,
        components_dir=candidate_components,
        source_artifact_root=source_artifact_root,
        start_pair=start_pair,
        max_pairs=max_pairs,
        batch_pairs=batch_pairs,
        device_type=device_type,
        allow_gpu_research_signal=allow_gpu_research_signal,
        progress_every=progress_every,
        response_family=_required_text(row, "family"),
        response_builder=response_builder,
    )
    baseline_response = _run_response(
        baseline_source,
        output_path=baseline_response_path,
        components_dir=baseline_components,
        source_artifact_root=source_artifact_root,
        start_pair=start_pair,
        max_pairs=max_pairs,
        batch_pairs=batch_pairs,
        device_type=device_type,
        allow_gpu_research_signal=allow_gpu_research_signal,
        progress_every=progress_every,
        response_family=f"{_required_text(row, 'family')}_baseline",
        response_builder=response_builder,
    )
    observation = _observation_from_responses(
        row,
        source_row,
        candidate_response=candidate_response,
        baseline_response=baseline_response,
        candidate_response_path=candidate_response_path,
        baseline_response_path=baseline_response_path,
    )
    executed_row = {
        "schema": "mlx_dynamic_learned_sweep_local_actuation_row.v1",
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "observation_only": True,
        "candidate_id": row.get("candidate_id"),
        "queue_candidate_id": row.get("queue_candidate_id"),
        "sweep_config_id": row.get("sweep_config_id"),
        "optimization_pass_id": row.get("optimization_pass_id"),
        "family": row.get("family"),
        "sample_budget": sample_budget,
        "start_pair": start_pair,
        "max_pairs": max_pairs,
        "candidate_response_path": str(candidate_response_path),
        "candidate_response_sha256": sha256_file(candidate_response_path),
        "baseline_response_path": str(baseline_response_path),
        "baseline_response_sha256": sha256_file(baseline_response_path),
    }
    return executed_row, observation


def _run_response(
    source: Mapping[str, Any],
    *,
    output_path: Path,
    components_dir: Path,
    source_artifact_root: str | Path | None,
    start_pair: int,
    max_pairs: int,
    batch_pairs: int,
    device_type: str,
    allow_gpu_research_signal: bool,
    progress_every: int,
    response_family: str,
    response_builder: ResponseBuilder,
) -> dict[str, Any]:
    cache_identity = _mapping(source.get("cache_identity"), label="cache_identity")
    reference = _mapping(cache_identity.get("reference"), label="cache_identity.reference")
    candidate = _mapping(cache_identity.get("candidate"), label="cache_identity.candidate")
    payload = response_builder(
        reference_cache_dir=_resolve_existing_path(
            reference.get("path"),
            source_artifact_root=source_artifact_root,
            label="reference cache path",
        ),
        candidate_cache_dir=_resolve_existing_path(
            candidate.get("path"),
            source_artifact_root=source_artifact_root,
            label="candidate cache path",
        ),
        archive_size_bytes=int(_required_float(source, "archive_size_bytes")),
        repo_root=Path(source_artifact_root or "."),
        batch_pairs=int(batch_pairs),
        device_type=device_type,
        components_dir=components_dir,
        progress_every=int(progress_every),
        start_pair=int(start_pair),
        max_pairs=int(max_pairs),
        allow_gpu_research_signal=allow_gpu_research_signal,
        allow_batch_shape_research_signal=False,
        allow_unaudited_candidate_cache_debug=False,
        allow_local_cpu_advisory_cache_identity=False,
        response_family=response_family,
    )
    write_mlx_scorer_response_payload(payload, output_path)
    return payload


def _observation_from_responses(
    row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    *,
    candidate_response: Mapping[str, Any],
    baseline_response: Mapping[str, Any],
    candidate_response_path: Path,
    baseline_response_path: Path,
) -> dict[str, Any]:
    scale = _normalization_scale(candidate_response, source_row)
    rate_delta = (
        _required_float(candidate_response, "archive_size_bytes")
        - _required_float(baseline_response, "archive_size_bytes")
    ) * RATE_SCORE_PER_BYTE
    seg_delta = (
        100.0 * _required_float(candidate_response, "avg_segnet_dist")
        - 100.0 * _required_float(baseline_response, "avg_segnet_dist")
    ) * scale
    pose_delta = (
        math.sqrt(10.0 * _required_float(candidate_response, "avg_posenet_dist"))
        - math.sqrt(10.0 * _required_float(baseline_response, "avg_posenet_dist"))
    ) * scale
    observed_score_or_delta = rate_delta + seg_delta + pose_delta
    return build_observation_row(
        candidate_id=_required_text(row, "candidate_id"),
        sweep_config_id=_required_text(row, "sweep_config_id"),
        optimization_pass_id=_required_text(row, "optimization_pass_id"),
        family=_required_text(row, "family"),
        observed_axis=EVIDENCE_TAG_MLX,
        evidence_tag=EVIDENCE_TAG_MLX,
        observed_score_or_delta=observed_score_or_delta,
        archive_sha256=_required_sha(candidate_response, "archive_sha256"),
        runtime_sha256=_runtime_identity_sha256(
            row,
            candidate_response=candidate_response,
            baseline_response=baseline_response,
        ),
        raw_output_or_cache_sha256=_cache_identity_sha256(candidate_response),
        component_deltas={
            "segnet_delta": seg_delta,
            "posenet_delta": pose_delta,
            "rate_delta": rate_delta,
            "scorer_delta": seg_delta + pose_delta,
        },
        source_artifact_path=str(candidate_response_path),
        source_artifact_sha256=sha256_file(candidate_response_path),
        extra={
            "evidence_grade": EVIDENCE_GRADE_MLX,
            "selected_pair_indices": row.get("selected_pair_indices"),
            "selected_pair_count": row.get("selected_pair_count"),
            "source_schema": "mlx_dynamic_learned_sweep_local_actuation_row.v1",
            "source_row": {
                "candidate_id": row.get("candidate_id"),
                "queue_candidate_id": row.get("queue_candidate_id"),
                "sweep_config_id": row.get("sweep_config_id"),
                "optimization_pass_id": row.get("optimization_pass_id"),
                **FALSE_AUTHORITY,
            },
            "baseline_artifact_path": str(baseline_response_path),
            "baseline_artifact_sha256": sha256_file(baseline_response_path),
            "component_delta_baseline_policy": (
                "local_mlx_response_candidate_minus_baseline_scaled_to_full_video"
            ),
            "score_delta_vs_baseline": observed_score_or_delta,
            "archive_byte_delta_vs_baseline": (
                _required_float(candidate_response, "archive_size_bytes")
                - _required_float(baseline_response, "archive_size_bytes")
            ),
            "notes": (
                "Executed local MLX scorer-response from fixed cache tensors; "
                "observation is replanning-only and carries no score or dispatch authority."
            ),
        },
    )


def _select_ready_rows(
    plan: Mapping[str, Any],
    *,
    sweep_config_id: str,
    optimization_pass_id: str | None,
    candidate_ids: Sequence[str] | None,
    queue_candidate_ids: Sequence[str] | None,
    max_rows: int,
) -> list[Mapping[str, Any]]:
    rows = plan.get("ranked_sweep_rows")
    if not isinstance(rows, list):
        raise MLXDynamicLearnedSweepLocalActuatorError("plan ranked_sweep_rows[] missing")
    candidate_filter = set(
        _normalize_filter_values(candidate_ids, label="candidate_ids")
    )
    queue_candidate_filter = set(
        _normalize_filter_values(
            queue_candidate_ids,
            label="queue_candidate_ids",
        )
    )
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("schema") != "mlx_dynamic_learned_sweep_row.v1":
            continue
        if row.get("ready_for_local_sweep") is not True:
            continue
        if row.get("sweep_config_id") != sweep_config_id:
            continue
        if optimization_pass_id is not None and row.get("optimization_pass_id") != optimization_pass_id:
            continue
        if candidate_filter and str(row.get("candidate_id") or "") not in candidate_filter:
            continue
        if (
            queue_candidate_filter
            and str(row.get("queue_candidate_id") or "") not in queue_candidate_filter
        ):
            continue
        selected.append(row)
        if len(selected) >= max_rows:
            break
    if queue_candidate_filter:
        selected_queue_ids = {
            str(row.get("queue_candidate_id") or "") for row in selected
        }
        if selected_queue_ids != queue_candidate_filter:
            missing = sorted(queue_candidate_filter - selected_queue_ids)
            raise MLXDynamicLearnedSweepLocalActuatorError(
                "requested queue_candidate_id filters were not all ready: "
                + ", ".join(missing)
            )
    return selected


def _executed_filter_summary(
    executed: Sequence[Mapping[str, Any]],
    *,
    optimization_pass_id: str | None,
    candidate_ids: Sequence[str],
    queue_candidate_ids: Sequence[str],
) -> dict[str, Any]:
    executed_candidate_ids = [
        str(row.get("candidate_id") or "") for row in executed if isinstance(row, Mapping)
    ]
    executed_queue_candidate_ids = [
        str(row.get("queue_candidate_id") or "")
        for row in executed
        if isinstance(row, Mapping)
    ]
    executed_pass_ids = [
        str(row.get("optimization_pass_id") or "")
        for row in executed
        if isinstance(row, Mapping)
    ]
    violations = 0
    if optimization_pass_id is not None:
        violations += sum(pass_id != optimization_pass_id for pass_id in executed_pass_ids)
    if candidate_ids:
        allowed = set(candidate_ids)
        violations += sum(candidate_id not in allowed for candidate_id in executed_candidate_ids)
    if queue_candidate_ids:
        allowed_queue = set(queue_candidate_ids)
        violations += sum(
            queue_candidate_id not in allowed_queue
            for queue_candidate_id in executed_queue_candidate_ids
        )
        violations += len(allowed_queue - set(executed_queue_candidate_ids))
    unique_candidate_ids = sorted(set(executed_candidate_ids))
    unique_queue_candidate_ids = sorted(set(executed_queue_candidate_ids))
    requested_queue_candidate_ids = sorted(set(queue_candidate_ids))
    requested_candidate_ids = sorted(set(candidate_ids))
    return {
        "requested_filter": {
            "schema": "local_sweep_filter.v1",
            "sweep_config_id": SUPPORTED_SWEEP_CONFIG_ID,
            "optimization_pass_id": optimization_pass_id,
            "candidate_ids": list(candidate_ids),
            "queue_candidate_ids": list(queue_candidate_ids),
            **FALSE_AUTHORITY,
        },
        "executed_filter_match": violations == 0,
        "executed_filter_violation_count": violations,
        "executed_candidate_ids": executed_candidate_ids,
        "executed_queue_candidate_ids": executed_queue_candidate_ids,
        "requested_candidate_id_count": len(requested_candidate_ids),
        "requested_queue_candidate_id_count": len(requested_queue_candidate_ids),
        "executed_candidate_id_count": len(executed_candidate_ids),
        "executed_queue_candidate_id_count": len(executed_queue_candidate_ids),
        "executed_candidate_id_set": unique_candidate_ids,
        "executed_queue_candidate_id_set": unique_queue_candidate_ids,
        "executed_unique_candidate_id": (
            unique_candidate_ids[0] if len(unique_candidate_ids) == 1 else None
        ),
        "executed_unique_queue_candidate_id": (
            unique_queue_candidate_ids[0]
            if len(unique_queue_candidate_ids) == 1
            else None
        ),
    }


def _normalize_filter_values(
    values: Sequence[str] | None,
    *,
    label: str,
) -> list[str]:
    if values is None:
        return []
    normalized: list[str] = []
    for index, value in enumerate(values):
        text = str(value).strip()
        if not text:
            raise MLXDynamicLearnedSweepLocalActuatorError(
                f"{label}[{index}] must be non-empty"
            )
        if text not in normalized:
            normalized.append(text)
    return normalized


def _selection_rows_by_candidate(selection: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = selection.get("selected_rows")
    if not isinstance(rows, list):
        raise MLXDynamicLearnedSweepLocalActuatorError("selection selected_rows[] missing")
    out: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise MLXDynamicLearnedSweepLocalActuatorError(
                f"selection row {index} must be object"
            )
        if row.get("schema") != SELECTION_ROW_SCHEMA:
            raise MLXDynamicLearnedSweepLocalActuatorError(
                f"selection row {index} schema mismatch"
            )
        _require_false_authority(row, label=f"selection row {index}")
        out[_required_text(row, "candidate_id")] = row
    return out


def _normalization_scale(
    candidate_response: Mapping[str, Any],
    source_row: Mapping[str, Any],
) -> float:
    source_n = _required_float(candidate_response, "n_samples")
    denominator = _required_float(
        source_row,
        "full_video_denominator",
        default=CONTEST_EXACT_SAMPLE_COUNT,
    )
    if source_n <= 0.0 or denominator <= 0.0 or source_n > denominator:
        raise MLXDynamicLearnedSweepLocalActuatorError(
            "n_samples/full_video_denominator invalid"
        )
    return source_n / denominator


def _runtime_identity_sha256(
    row: Mapping[str, Any],
    *,
    candidate_response: Mapping[str, Any],
    baseline_response: Mapping[str, Any],
) -> str:
    return _hash_payload(
        {
            "schema": "mlx_dynamic_learned_sweep_local_actuator_runtime_identity.v1",
            "tool": TOOL,
            "candidate_id": row.get("candidate_id"),
            "sweep_config_id": row.get("sweep_config_id"),
            "optimization_pass_id": row.get("optimization_pass_id"),
            "candidate_hardware_substrate": candidate_response.get("hardware_substrate"),
            "baseline_hardware_substrate": baseline_response.get("hardware_substrate"),
            "candidate_device_contract": candidate_response.get("device_contract"),
            "baseline_device_contract": baseline_response.get("device_contract"),
        }
    )


def _cache_identity_sha256(response: Mapping[str, Any]) -> str:
    cache = _mapping(response.get("cache_identity"), label="cache_identity")
    candidate = _mapping(cache.get("candidate"), label="cache_identity.candidate")
    return _hash_payload(
        {
            "schema": "mlx_dynamic_learned_sweep_local_actuator_cache_identity.v1",
            "start_pair": response.get("start_pair"),
            "max_pairs": response.get("max_pairs"),
            "n_samples": response.get("n_samples"),
            "archive_sha256": response.get("archive_sha256"),
            "array_sha256": candidate.get("array_sha256"),
            "inflated_outputs_aggregate_sha256": candidate.get(
                "inflated_outputs_aggregate_sha256"
            ),
        }
    )


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )


def _resolve_existing_path(
    value: Any,
    *,
    source_artifact_root: str | Path | None,
    label: str,
) -> Path:
    if value is None:
        raise MLXDynamicLearnedSweepLocalActuatorError(f"{label} is required")
    path = Path(str(value))
    if path.exists():
        return path
    if source_artifact_root is not None:
        rooted = Path(source_artifact_root) / path
        if rooted.exists():
            return rooted
    raise MLXDynamicLearnedSweepLocalActuatorError(f"{label} does not exist: {path}")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MLXDynamicLearnedSweepLocalActuatorError(f"{label} must be an object")
    return value


def _sequence_dicts(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise MLXDynamicLearnedSweepLocalActuatorError("expected list in source plan")
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _mapping_value(value: Any, key: str, *, default: Any = None) -> Any:
    if not isinstance(value, Mapping):
        return default
    return value.get(key, default)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    text = "" if value is None else str(value).strip()
    if not text:
        raise MLXDynamicLearnedSweepLocalActuatorError(f"{key} is required")
    return text


def _required_float(
    row: Mapping[str, Any],
    key: str,
    *,
    default: float | int | None = None,
) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MLXDynamicLearnedSweepLocalActuatorError(f"{key} must be numeric") from exc


def _required_sha(row: Mapping[str, Any], key: str) -> str:
    value = _required_text(row, key).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise MLXDynamicLearnedSweepLocalActuatorError(
            f"{key} must be a 64-character SHA-256 hex digest"
        )
    return value


def _first_pair_index(row: Mapping[str, Any]) -> int:
    pairs = row.get("selected_pair_indices")
    if isinstance(pairs, list) and pairs:
        return int(pairs[0])
    raise MLXDynamicLearnedSweepLocalActuatorError("selected_pair_indices missing")


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in value)


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if key in payload and payload.get(key) is not False:
            raise MLXDynamicLearnedSweepLocalActuatorError(
                f"{label} {key} must be false"
            )


__all__ = [
    "SCHEMA",
    "SUPPORTED_SWEEP_CONFIG_ID",
    "TOOL",
    "MLXDynamicLearnedSweepLocalActuatorError",
    "execute_local_mlx_sweep_rows",
    "load_json_object",
    "replan_after_local_actuation",
]
