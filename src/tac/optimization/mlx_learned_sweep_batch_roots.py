# SPDX-License-Identifier: MIT
"""Row-level batch-root planning for local learned sweeps."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.mlx_dynamic_learned_sweep import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

MLX_LEARNED_SWEEP_BATCH_ROOT_PLAN_SCHEMA = (
    "mlx_dynamic_learned_sweep_autopilot_batch_root_plan.v1"
)
TOOL = "tac.optimization.mlx_learned_sweep_batch_roots"
LEARNED_SWEEP_PLAN_SCHEMA = "mlx_dynamic_learned_sweep_plan.v1"
LEARNED_SWEEP_ROW_SCHEMA = "mlx_dynamic_learned_sweep_row.v1"
DEFAULT_SWEEP_CONFIG_ID = "mlx_local_response"
LOCAL_AUTOPILOT_TELEMETRY_SCHEMA = "mlx_dynamic_learned_sweep_local_autopilot.v1"
EXPERIMENT_QUEUE_WORKER_RESULT_SCHEMA = "experiment_queue_worker_result.v1"
EXPLICIT_FALSE_AUTHORITY_FIELDS: tuple[str, ...] = (
    *tuple(FALSE_AUTHORITY),
    "dispatch_attempted",
    "gpu_launched",
)
LOCAL_FALSE_AUTHORITY: dict[str, bool] = {
    **FALSE_AUTHORITY,
    "dispatch_attempted": False,
    "gpu_launched": False,
}


class MLXLearnedSweepBatchRootError(ValueError):
    """Raised when a learned-sweep batch-root plan would be unsafe or ambiguous."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_id(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in value)
    return out.strip("._-").lower() or "root"


def _as_positive_int(value: int, *, label: str) -> int:
    if isinstance(value, bool) or int(value) < 1:
        raise MLXLearnedSweepBatchRootError(f"{label} must be >= 1")
    return int(value)


def _as_float(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MLXLearnedSweepBatchRootError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise MLXLearnedSweepBatchRootError(f"{label} must be finite")
    return parsed


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for key in EXPLICIT_FALSE_AUTHORITY_FIELDS:
        if payload.get(key) is not False:
            raise MLXLearnedSweepBatchRootError(f"{label}: {key} must be explicit false")
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise MLXLearnedSweepBatchRootError(str(exc)) from exc


def _require_no_truthy_authority(payload: Mapping[str, Any], *, label: str) -> None:
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise MLXLearnedSweepBatchRootError(str(exc)) from exc


def _path_text(value: str | Path) -> str:
    return str(value)


def _derive_observation_jsonl(base: str | Path, run_id: str) -> str:
    path = Path(base)
    safe_run_id = _safe_id(run_id)
    if path.suffix == ".jsonl":
        return str(path.with_name(f"{path.stem}_{safe_run_id}{path.suffix}"))
    return str(path / f"{safe_run_id}.jsonl")


def _row_sort_key(row: Mapping[str, Any]) -> tuple[float, int, float, float, str, str, str]:
    return (
        -_as_float(row.get("acquisition_value"), label="row.acquisition_value"),
        int(row.get("recursive_stage") or 0),
        _as_float(row.get("lower_confidence_score"), label="row.lower_confidence_score"),
        _as_float(row.get("cost_units"), label="row.cost_units"),
        str(row.get("candidate_id") or ""),
        str(row.get("sweep_config_id") or ""),
        str(row.get("optimization_pass_id") or ""),
    )


def _ready_local_rows(
    plan: Mapping[str, Any],
    *,
    sweep_config_id: str,
) -> list[dict[str, Any]]:
    if plan.get("schema") != LEARNED_SWEEP_PLAN_SCHEMA:
        raise MLXLearnedSweepBatchRootError(
            f"plan schema must be {LEARNED_SWEEP_PLAN_SCHEMA}"
        )
    _require_false_authority(plan, label="plan")
    rows = plan.get("ranked_sweep_rows")
    if not isinstance(rows, list):
        raise MLXLearnedSweepBatchRootError("plan ranked_sweep_rows must be a list")

    ready: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise MLXLearnedSweepBatchRootError(
                f"plan ranked_sweep_rows[{index}] must be an object"
            )
        if raw_row.get("schema") != LEARNED_SWEEP_ROW_SCHEMA:
            continue
        if raw_row.get("ready_for_local_sweep") is not True:
            continue
        if raw_row.get("sweep_config_id") != sweep_config_id:
            continue
        if raw_row.get("exact_eval_candidate") is True:
            raise MLXLearnedSweepBatchRootError(
                "plan has exact-eval candidate marked ready for local MLX sweep"
            )
        _require_false_authority(raw_row, label=f"plan ranked_sweep_rows[{index}]")
        ready.append(dict(raw_row))
    if not ready:
        raise MLXLearnedSweepBatchRootError(
            f"plan has no ready {sweep_config_id} rows for automatic batch roots"
        )
    ready.sort(key=_row_sort_key)
    return ready


def _row_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "mlx_dynamic_learned_sweep_batch_root_row_ref.v1",
        "candidate_id": row.get("candidate_id"),
        "queue_candidate_id": row.get("queue_candidate_id"),
        "rank": row.get("rank"),
        "family": row.get("family"),
        "sweep_config_id": row.get("sweep_config_id"),
        "optimization_pass_id": row.get("optimization_pass_id"),
        "recursive_stage": row.get("recursive_stage"),
        "acquisition_value": row.get("acquisition_value"),
        "expected_improvement": row.get("expected_improvement"),
        "learning_value_per_cost": row.get("learning_value_per_cost"),
        "cost_units": row.get("cost_units"),
        "lower_confidence_score": row.get("lower_confidence_score"),
        "selected_pair_count": row.get("selected_pair_count"),
        "selected_pair_indices": row.get("selected_pair_indices"),
        **LOCAL_FALSE_AUTHORITY,
    }


def _row_planned_cost(row: Mapping[str, Any]) -> float:
    return _as_float(row.get("cost_units"), label="row.cost_units")


def _queue_candidate_id(row: Mapping[str, Any]) -> str:
    return str(row.get("queue_candidate_id") or "").strip()


def _runtime_cost_for_row(
    row: Mapping[str, Any],
    runtime_seconds_by_queue_candidate_id: Mapping[str, float] | None,
) -> tuple[float, str]:
    queue_candidate_id = _queue_candidate_id(row)
    if runtime_seconds_by_queue_candidate_id and queue_candidate_id:
        observed = runtime_seconds_by_queue_candidate_id.get(queue_candidate_id)
        if observed is not None:
            return _as_float(
                observed,
                label=f"runtime telemetry {queue_candidate_id}",
            ), "telemetry_seconds_per_queue_candidate"
    return _row_planned_cost(row), "planned_cost_units"


def _row_ref_with_runtime(
    row: Mapping[str, Any],
    runtime_seconds_by_queue_candidate_id: Mapping[str, float] | None,
) -> dict[str, Any]:
    runtime_cost, runtime_source = _runtime_cost_for_row(
        row,
        runtime_seconds_by_queue_candidate_id,
    )
    out = _row_ref(row)
    out["runtime_cost_estimate"] = runtime_cost
    out["runtime_cost_source"] = runtime_source
    return out


def _add_runtime_observation(
    accum: dict[str, list[float]],
    queue_candidate_ids: Sequence[Any],
    *,
    elapsed_seconds: Any,
    label: str,
) -> tuple[int, int]:
    seconds = _as_float(elapsed_seconds, label=f"{label}.elapsed_seconds")
    if seconds <= 0.0:
        raise MLXLearnedSweepBatchRootError(
            f"{label}.elapsed_seconds must be > 0"
        )
    normalized_ids = sorted(
        {
            str(queue_candidate_id).strip()
            for queue_candidate_id in queue_candidate_ids
            if str(queue_candidate_id).strip()
        }
    )
    if not normalized_ids:
        raise MLXLearnedSweepBatchRootError(
            f"{label} must include at least one queue_candidate_id"
        )
    per_candidate_seconds = seconds / len(normalized_ids)
    for queue_candidate_id in normalized_ids:
        accum.setdefault(queue_candidate_id, []).append(per_candidate_seconds)
    return len(normalized_ids), int(len(normalized_ids) > 1)


def _command_queue_candidate_ids(command: Any) -> list[str]:
    if not isinstance(command, list):
        raise MLXLearnedSweepBatchRootError(
            "experiment_queue worker telemetry command must be a list"
        )
    out: list[str] = []
    index = 0
    while index < len(command):
        if command[index] == "--queue-candidate-id":
            if index + 1 >= len(command):
                raise MLXLearnedSweepBatchRootError(
                    "experiment_queue worker telemetry has dangling "
                    "--queue-candidate-id"
                )
            out.append(str(command[index + 1]))
            index += 2
            continue
        index += 1
    return out


def _runtime_telemetry_summary(
    telemetry_payloads: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if not telemetry_payloads:
        return {
            "schema": "mlx_dynamic_learned_sweep_runtime_telemetry_summary.v1",
            "runtime_telemetry_used": False,
            "runtime_telemetry_payload_count": 0,
            "runtime_telemetry_observation_count": 0,
            "runtime_telemetry_key_count": 0,
            "runtime_cost_policy": "planned_cost_units",
            "runtime_telemetry_assignment_policy": "none",
            "runtime_telemetry_even_split_observation_count": 0,
            "runtime_telemetry_source_state_paths": [],
            "runtime_telemetry_source_queue_ids": [],
            "runtime_seconds_by_queue_candidate_id": {},
            "runtime_source_count_by_queue_candidate_id": {},
            **LOCAL_FALSE_AUTHORITY,
        }

    observed_seconds: dict[str, list[float]] = {}
    observation_count = 0
    even_split_observation_count = 0
    schema_counts: Counter[str] = Counter()
    source_state_paths: set[str] = set()
    source_queue_ids: set[str] = set()

    for index, payload in enumerate(telemetry_payloads):
        if not isinstance(payload, Mapping):
            raise MLXLearnedSweepBatchRootError(
                f"runtime_telemetry_payloads[{index}] must be an object"
            )
        label = f"runtime_telemetry_payloads[{index}]"
        _require_no_truthy_authority(payload, label=label)
        source_state_path = str(payload.get("source_state_path") or "").strip()
        if source_state_path:
            source_state_paths.add(source_state_path)
        raw_source_queue_ids = payload.get("source_queue_ids")
        if isinstance(raw_source_queue_ids, list):
            source_queue_ids.update(
                str(queue_id).strip()
                for queue_id in raw_source_queue_ids
                if str(queue_id).strip()
            )
        schema = str(payload.get("schema") or "")
        schema_counts[schema or "missing"] += 1
        if schema == LOCAL_AUTOPILOT_TELEMETRY_SCHEMA:
            _require_false_authority(payload, label=label)
            if payload.get("executed_filter_match") is not True:
                raise MLXLearnedSweepBatchRootError(
                    f"{label}.executed_filter_match must be true"
                )
            if int(payload.get("executed_filter_violation_count") or 0) != 0:
                raise MLXLearnedSweepBatchRootError(
                    f"{label}.executed_filter_violation_count must be 0"
                )
            queue_candidate_ids = payload.get("executed_queue_candidate_id_set")
            if not isinstance(queue_candidate_ids, list):
                raise MLXLearnedSweepBatchRootError(
                    f"{label}.executed_queue_candidate_id_set must be a list"
                )
            observed_count, even_split_count = _add_runtime_observation(
                observed_seconds,
                queue_candidate_ids,
                elapsed_seconds=payload.get("elapsed_seconds"),
                label=label,
            )
            observation_count += observed_count
            even_split_observation_count += even_split_count
            continue

        if schema == EXPERIMENT_QUEUE_WORKER_RESULT_SCHEMA:
            steps = payload.get("step_results")
            if not isinstance(steps, list):
                raise MLXLearnedSweepBatchRootError(
                    f"{label}.step_results must be a list"
                )
            for step_index, raw_step in enumerate(steps):
                if not isinstance(raw_step, Mapping):
                    raise MLXLearnedSweepBatchRootError(
                        f"{label}.step_results[{step_index}] must be an object"
                    )
                step_label = f"{label}.step_results[{step_index}]"
                _require_no_truthy_authority(raw_step, label=step_label)
                if raw_step.get("succeeded") is not True:
                    continue
                if raw_step.get("timed_out") is True:
                    continue
                if raw_step.get("failed_postconditions"):
                    continue
                if raw_step.get("postcondition_errors"):
                    continue
                queue_candidate_ids = _command_queue_candidate_ids(
                    raw_step.get("command")
                )
                if not queue_candidate_ids:
                    continue
                observed_count, even_split_count = _add_runtime_observation(
                    observed_seconds,
                    queue_candidate_ids,
                    elapsed_seconds=raw_step.get("elapsed_seconds"),
                    label=step_label,
                )
                observation_count += observed_count
                even_split_observation_count += even_split_count
            continue

        raise MLXLearnedSweepBatchRootError(
            f"{label}.schema unsupported for runtime telemetry: {schema!r}"
        )

    runtime_seconds_by_id = {
        queue_candidate_id: sum(samples) / len(samples)
        for queue_candidate_id, samples in sorted(observed_seconds.items())
    }
    source_counts_by_id = {
        queue_candidate_id: len(samples)
        for queue_candidate_id, samples in sorted(observed_seconds.items())
    }
    return {
        "schema": "mlx_dynamic_learned_sweep_runtime_telemetry_summary.v1",
        "runtime_telemetry_used": bool(runtime_seconds_by_id),
        "runtime_telemetry_payload_count": len(telemetry_payloads),
        "runtime_telemetry_observation_count": observation_count,
        "runtime_telemetry_key_count": len(runtime_seconds_by_id),
        "runtime_telemetry_schema_counts": dict(sorted(schema_counts.items())),
        "runtime_telemetry_assignment_policy": (
            "elapsed_seconds_even_split_by_queue_candidate"
            if runtime_seconds_by_id
            else "none"
        ),
        "runtime_telemetry_even_split_observation_count": even_split_observation_count,
        "runtime_telemetry_source_state_paths": sorted(source_state_paths),
        "runtime_telemetry_source_queue_ids": sorted(source_queue_ids),
        "runtime_cost_policy": (
            "telemetry_seconds_per_queue_candidate_with_planned_fallback"
            if runtime_seconds_by_id
            else "planned_cost_units"
        ),
        "runtime_seconds_by_queue_candidate_id": runtime_seconds_by_id,
        "runtime_source_count_by_queue_candidate_id": source_counts_by_id,
        **LOCAL_FALSE_AUTHORITY,
    }


def _family_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("family") or "unknown") for row in rows).items()))


def _require_unique_values(
    values: Sequence[str],
    *,
    label: str,
) -> None:
    duplicates = sorted(
        value for value, count in Counter(values).items() if count > 1
    )
    if duplicates:
        raise MLXLearnedSweepBatchRootError(
            f"{label} must be unique for exact row roots: " + ", ".join(duplicates)
        )


def _row_chunks(
    rows: Sequence[dict[str, Any]],
    *,
    rows_per_root: int,
) -> list[list[dict[str, Any]]]:
    return [
        list(rows[index : index + rows_per_root])
        for index in range(0, len(rows), rows_per_root)
    ]


def _adaptive_row_groups(
    rows: Sequence[dict[str, Any]],
    *,
    root_count: int,
    max_rows_per_root: int,
    runtime_seconds_by_queue_candidate_id: Mapping[str, float] | None = None,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    suppressed_nonpositive = 0
    for row in rows:
        acquisition_value = _as_float(
            row.get("acquisition_value"),
            label="row.acquisition_value",
        )
        learning_value = _as_float(
            row.get("learning_value_per_cost"),
            label="row.learning_value_per_cost",
        )
        if acquisition_value <= 0.0 or learning_value <= 0.0:
            suppressed_nonpositive += 1
            continue
        eligible.append(row)

    if not eligible:
        raise MLXLearnedSweepBatchRootError(
            "adaptive batch roots require at least one positive-utility ready row"
        )

    bucket_count = min(root_count, len(eligible))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(bucket_count)]
    bucket_costs = [0.0 for _ in range(bucket_count)]
    for row in eligible[: root_count * max_rows_per_root]:
        open_bucket_indexes = [
            index for index, bucket in enumerate(buckets) if len(bucket) < max_rows_per_root
        ]
        if not open_bucket_indexes:
            break
        runtime_cost, _runtime_source = _runtime_cost_for_row(
            row,
            runtime_seconds_by_queue_candidate_id,
        )
        target = min(
            open_bucket_indexes,
            key=lambda index: (bucket_costs[index], len(buckets[index]), index),
        )
        buckets[target].append(row)
        bucket_costs[target] += runtime_cost

    groups = [bucket for bucket in buckets if bucket]
    group_runtime_costs = [
        sum(
            _runtime_cost_for_row(row, runtime_seconds_by_queue_candidate_id)[0]
            for row in group
        )
        for group in groups
    ]
    group_planned_costs = [
        sum(_row_planned_cost(row) for row in group)
        for group in groups
    ]
    group_observed_counts = [
        sum(
            1
            for row in group
            if _runtime_cost_for_row(row, runtime_seconds_by_queue_candidate_id)[1]
            == "telemetry_seconds_per_queue_candidate"
        )
        for group in groups
    ]
    return groups, {
        "schema": "mlx_dynamic_learned_sweep_adaptive_row_grouping.v1",
        "strategy": (
            "positive_utility_runtime_balanced_waterfill"
            if runtime_seconds_by_queue_candidate_id
            else "positive_utility_cost_balanced_waterfill"
        ),
        "max_rows_per_root": max_rows_per_root,
        "eligible_positive_utility_row_count": len(eligible),
        "suppressed_nonpositive_utility_row_count": suppressed_nonpositive,
        "selected_total_row_count": sum(len(group) for group in groups),
        "root_size_counts": [len(group) for group in groups],
        "root_cost_units": group_planned_costs,
        "root_runtime_cost_estimates": group_runtime_costs,
        "root_runtime_observed_queue_candidate_counts": group_observed_counts,
        "runtime_cost_policy": (
            "telemetry_seconds_per_queue_candidate_with_planned_fallback"
            if runtime_seconds_by_queue_candidate_id
            else "planned_cost_units"
        ),
        **LOCAL_FALSE_AUTHORITY,
    }


def _resolve_adaptive_max_rows_per_root(
    *,
    explicit_rows_per_root: int | None,
    max_new_observations: int | None,
    rows_per_replan: int | None,
    ready_row_count: int,
    requested_roots: int,
) -> int:
    if explicit_rows_per_root is not None:
        limit = _as_positive_int(explicit_rows_per_root, label="rows_per_root")
        if max_new_observations is not None and int(max_new_observations) < limit:
            raise MLXLearnedSweepBatchRootError(
                "max_new_observations must cover rows_per_root for adaptive roots"
            )
        if rows_per_replan is not None and int(rows_per_replan) < limit:
            raise MLXLearnedSweepBatchRootError(
                "rows_per_replan must cover rows_per_root for adaptive roots"
            )
        return limit

    caps: list[int] = []
    if max_new_observations is not None:
        caps.append(_as_positive_int(max_new_observations, label="max_new_observations"))
    if rows_per_replan is not None:
        caps.append(_as_positive_int(rows_per_replan, label="rows_per_replan"))
    if caps:
        return min(caps)
    return max(1, math.ceil(ready_row_count / requested_roots))


def build_mlx_learned_sweep_autopilot_batch_root_plan(
    plan: Mapping[str, Any],
    *,
    plan_path: str | Path,
    selection_path: str | Path,
    candidate_payload_paths: Sequence[str | Path],
    incumbent_score: float,
    output_root: str | Path,
    observation_jsonl: str | Path,
    root_count: int,
    rows_per_root: int | None = 1,
    adaptive_rows_per_root: bool = False,
    run_prefix: str = "pass",
    sweep_config_id: str = DEFAULT_SWEEP_CONFIG_ID,
    max_new_observations: int | None = None,
    rows_per_replan: int | None = None,
    chain_steps: int | None = None,
    device: str | None = None,
    allow_gpu_research_signal: bool | None = None,
    source_artifact_root: str | Path | None = None,
    batch_pairs: int | None = None,
    runtime_telemetry_payloads: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select queue-candidate-specific local roots and return run specs."""

    requested_roots = _as_positive_int(root_count, label="root_count")
    if chain_steps is not None and int(chain_steps) != 1:
        raise MLXLearnedSweepBatchRootError(
            "row-specific auto batch roots require chain_steps=1"
        )
    if not candidate_payload_paths:
        raise MLXLearnedSweepBatchRootError(
            "candidate_payload_paths must be non-empty"
        )
    incumbent = _as_float(incumbent_score, label="incumbent_score")
    ready_rows = _ready_local_rows(plan, sweep_config_id=sweep_config_id)
    runtime_telemetry = _runtime_telemetry_summary(runtime_telemetry_payloads)
    runtime_seconds_by_queue_candidate_id = runtime_telemetry[
        "runtime_seconds_by_queue_candidate_id"
    ]
    grouping_metadata: dict[str, Any]
    if adaptive_rows_per_root:
        max_rows_per_root = _resolve_adaptive_max_rows_per_root(
            explicit_rows_per_root=rows_per_root,
            max_new_observations=max_new_observations,
            rows_per_replan=rows_per_replan,
            ready_row_count=len(ready_rows),
            requested_roots=requested_roots,
        )
        row_groups, grouping_metadata = _adaptive_row_groups(
            ready_rows,
            root_count=requested_roots,
            max_rows_per_root=max_rows_per_root,
            runtime_seconds_by_queue_candidate_id=runtime_seconds_by_queue_candidate_id,
        )
    else:
        fixed_rows_per_root = _as_positive_int(
            1 if rows_per_root is None else rows_per_root,
            label="rows_per_root",
        )
        if (
            max_new_observations is not None
            and int(max_new_observations) < fixed_rows_per_root
        ):
            raise MLXLearnedSweepBatchRootError(
                "max_new_observations must cover rows_per_root for row-specific roots"
            )
        if rows_per_replan is not None and int(rows_per_replan) < fixed_rows_per_root:
            raise MLXLearnedSweepBatchRootError(
                "rows_per_replan must cover rows_per_root for row-specific roots"
            )
        max_rows_per_root = fixed_rows_per_root
        row_groups = _row_chunks(ready_rows, rows_per_root=fixed_rows_per_root)
        grouping_metadata = {
            "schema": "mlx_dynamic_learned_sweep_fixed_row_grouping.v1",
            "strategy": "fixed_sorted_chunks",
            "max_rows_per_root": fixed_rows_per_root,
            "eligible_positive_utility_row_count": None,
            "suppressed_nonpositive_utility_row_count": 0,
            "selected_total_row_count": sum(len(group) for group in row_groups),
            "root_size_counts": [len(group) for group in row_groups],
            "root_cost_units": [
                sum(_row_planned_cost(row) for row in group) for group in row_groups
            ],
            "root_runtime_cost_estimates": [
                sum(
                    _runtime_cost_for_row(row, runtime_seconds_by_queue_candidate_id)[0]
                    for row in group
                )
                for group in row_groups
            ],
            "root_runtime_observed_queue_candidate_counts": [
                sum(
                    1
                    for row in group
                    if _runtime_cost_for_row(
                        row,
                        runtime_seconds_by_queue_candidate_id,
                    )[1]
                    == "telemetry_seconds_per_queue_candidate"
                )
                for group in row_groups
            ],
            "runtime_cost_policy": runtime_telemetry["runtime_cost_policy"],
            **LOCAL_FALSE_AUTHORITY,
        }

    root_summaries: list[dict[str, Any]] = []
    for selected_rows in row_groups:
        acquisition_sum = sum(
            _as_float(row.get("acquisition_value"), label="row.acquisition_value")
            for row in selected_rows
        )
        expected_improvement_sum = sum(
            _as_float(row.get("expected_improvement"), label="row.expected_improvement")
            for row in selected_rows
        )
        learning_value_sum = sum(
            _as_float(
                row.get("learning_value_per_cost"),
                label="row.learning_value_per_cost",
            )
            for row in selected_rows
        )
        cost_sum = sum(
            _as_float(row.get("cost_units"), label="row.cost_units")
            for row in selected_rows
        )
        runtime_cost_pairs = [
            _runtime_cost_for_row(row, runtime_seconds_by_queue_candidate_id)
            for row in selected_rows
        ]
        runtime_cost_sum = sum(cost for cost, _source in runtime_cost_pairs)
        runtime_observed_count = sum(
            1
            for _cost, source in runtime_cost_pairs
            if source == "telemetry_seconds_per_queue_candidate"
        )
        best = selected_rows[0]
        queue_candidate_ids = [
            str(row.get("queue_candidate_id") or "").strip()
            for row in selected_rows
        ]
        if any(not queue_candidate_id for queue_candidate_id in queue_candidate_ids):
            raise MLXLearnedSweepBatchRootError(
                "ready rows require queue_candidate_id"
            )
        _require_unique_values(
            queue_candidate_ids,
            label="root queue_candidate_ids",
        )
        candidate_ids = [
            str(row.get("candidate_id") or "").strip()
            for row in selected_rows
        ]
        if any(not candidate_id for candidate_id in candidate_ids):
            raise MLXLearnedSweepBatchRootError("ready rows require candidate_id")
        optimization_pass_ids = [
            str(row.get("optimization_pass_id") or "").strip()
            for row in selected_rows
        ]
        if any(not pass_id for pass_id in optimization_pass_ids):
            raise MLXLearnedSweepBatchRootError(
                "ready rows require optimization_pass_id"
            )
        unique_optimization_pass_ids = sorted(set(optimization_pass_ids))
        representative_pass_id = str(best.get("optimization_pass_id") or "")
        root_summaries.append(
            {
                "optimization_pass_id": (
                    representative_pass_id
                    if len(unique_optimization_pass_ids) == 1
                    else None
                ),
                "representative_optimization_pass_id": representative_pass_id,
                "optimization_pass_ids": unique_optimization_pass_ids,
                "candidate_ids": sorted(set(candidate_ids)),
                "queue_candidate_ids": queue_candidate_ids,
                "recursive_stage": int(best.get("recursive_stage") or 0),
                "root_utility": acquisition_sum,
                "expected_improvement_sum": expected_improvement_sum,
                "learning_value_per_cost_sum": learning_value_sum,
                "cost_units_sum": cost_sum,
                "runtime_cost_estimate_sum": runtime_cost_sum,
                "runtime_observed_queue_candidate_count": runtime_observed_count,
                "runtime_cost_policy": runtime_telemetry["runtime_cost_policy"],
                "best_lower_confidence_score": _as_float(
                    best.get("lower_confidence_score"),
                    label="best.lower_confidence_score",
                ),
                "ready_row_count": len(ready_rows),
                "selected_row_count": len(selected_rows),
                "rows_per_root_policy": (
                    "adaptive_positive_utility_cost_balanced_waterfill"
                    if adaptive_rows_per_root
                    else "fixed_sorted_chunks"
                ),
                "family_counts": _family_counts(selected_rows),
                "row_refs": [
                    _row_ref_with_runtime(row, runtime_seconds_by_queue_candidate_id)
                    for row in selected_rows
                ],
                **LOCAL_FALSE_AUTHORITY,
            }
        )

    root_summaries.sort(
        key=lambda root: (
            -float(root["root_utility"]),
            int(root["recursive_stage"]),
            float(root["best_lower_confidence_score"]),
            float(root["runtime_cost_estimate_sum"]),
            float(root["cost_units_sum"]),
            str(root["queue_candidate_ids"][0]),
        )
    )
    selected = root_summaries[:requested_roots]
    selected_queue_candidate_ids = [
        queue_candidate_id
        for root in selected
        for queue_candidate_id in root["queue_candidate_ids"]
    ]
    _require_unique_values(
        selected_queue_candidate_ids,
        label="selected queue_candidate_ids",
    )
    selected_candidate_ids = [
        candidate_id for root in selected for candidate_id in root["candidate_ids"]
    ]
    selected_pass_ids = [
        pass_id for root in selected for pass_id in root["optimization_pass_ids"]
    ]

    run_specs: list[dict[str, Any]] = []
    selected_roots: list[dict[str, Any]] = []
    for index, root in enumerate(selected, start=1):
        pass_id = root.get("optimization_pass_id")
        first_queue_candidate_id = str(root["queue_candidate_ids"][0])
        run_id = _safe_id(f"{run_prefix}_{index:04d}_{first_queue_candidate_id}")
        run_spec: dict[str, Any] = {
            "run_id": run_id,
            "plan_path": _path_text(plan_path),
            "selection_path": _path_text(selection_path),
            "candidate_payload_paths": [_path_text(path) for path in candidate_payload_paths],
            "incumbent_score": incumbent,
            "output_root": str(Path(output_root) / run_id),
            "observation_jsonl": _derive_observation_jsonl(observation_jsonl, run_id),
            "queue_candidate_ids": list(root["queue_candidate_ids"]),
            "sweep_config_id": sweep_config_id,
        }
        if pass_id is not None:
            run_spec["optimization_pass_id"] = pass_id
        optional_values = {
            "max_new_observations": max_new_observations,
            "rows_per_replan": rows_per_replan,
            "chain_steps": chain_steps,
            "device": device,
            "allow_gpu_research_signal": allow_gpu_research_signal,
            "source_artifact_root": _path_text(source_artifact_root)
            if source_artifact_root is not None
            else None,
            "batch_pairs": batch_pairs,
        }
        for key, value in optional_values.items():
            if value is not None:
                run_spec[key] = value
        run_specs.append(run_spec)
        selected_roots.append(
            {
                "schema": "mlx_dynamic_learned_sweep_autopilot_batch_root.v1",
                "run_id": run_id,
                "run_spec_index": index,
                "root_kind": "queue_candidate_waterfill",
                "root_granularity": "queue_candidate_set",
                "candidate_specific_filter_supported": True,
                "queue_candidate_filter_supported": True,
                "row_specific_filter_supported": True,
                "queue_candidate_disjoint_guaranteed": True,
                "candidate_disjoint_guaranteed": (
                    len(selected_candidate_ids) == len(set(selected_candidate_ids))
                ),
                "pass_disjoint_guaranteed": (
                    len(selected_pass_ids) == len(set(selected_pass_ids))
                ),
                **root,
            }
        )

    return {
        "schema": MLX_LEARNED_SWEEP_BATCH_ROOT_PLAN_SCHEMA,
        "producer": TOOL,
        "generated_at_utc": _utc_now(),
        **LOCAL_FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "observation_only": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "allowed_use": "compile_local_learned_sweep_autopilot_batch_queue_only",
        "root_kind": "queue_candidate_waterfill",
        "root_granularity": "queue_candidate_set",
        "candidate_specific_filter_supported": True,
        "queue_candidate_filter_supported": True,
        "row_specific_filter_supported": True,
        "queue_candidate_disjoint_guaranteed": (
            len(selected_queue_candidate_ids) == len(set(selected_queue_candidate_ids))
        ),
        "candidate_disjoint_guaranteed": (
            len(selected_candidate_ids) == len(set(selected_candidate_ids))
        ),
        "pass_disjoint_guaranteed": len(selected_pass_ids) == len(set(selected_pass_ids)),
        "requested_root_count": requested_roots,
        "selected_root_count": len(selected_roots),
        "unfilled_root_count": max(0, requested_roots - len(selected_roots)),
        "unfilled_reason": (
            "ready_queue_candidate_rows_exhausted"
            if len(selected_roots) < requested_roots
            else None
        ),
        "rows_per_root": None if adaptive_rows_per_root else max_rows_per_root,
        "rows_per_root_policy": (
            "adaptive_positive_utility_cost_balanced_waterfill"
            if adaptive_rows_per_root
            else "fixed_sorted_chunks"
        ),
        "adaptive_rows_per_root": bool(adaptive_rows_per_root),
        "max_rows_per_root": max_rows_per_root,
        "selected_total_row_count": sum(
            int(root["selected_row_count"]) for root in selected_roots
        ),
        "unselected_ready_row_count": max(
            0,
            len(ready_rows)
            - sum(int(root["selected_row_count"]) for root in selected_roots),
        ),
        "row_grouping": grouping_metadata,
        "runtime_telemetry": runtime_telemetry,
        "runtime_telemetry_used": runtime_telemetry["runtime_telemetry_used"],
        "runtime_telemetry_payload_count": runtime_telemetry[
            "runtime_telemetry_payload_count"
        ],
        "runtime_telemetry_observation_count": runtime_telemetry[
            "runtime_telemetry_observation_count"
        ],
        "runtime_telemetry_key_count": runtime_telemetry[
            "runtime_telemetry_key_count"
        ],
        "runtime_cost_policy": runtime_telemetry["runtime_cost_policy"],
        "sweep_config_id": sweep_config_id,
        "incumbent_score": incumbent,
        "source_artifacts": {
            "plan_path": _path_text(plan_path),
            "selection_path": _path_text(selection_path),
            "candidate_payload_paths": [
                _path_text(path) for path in candidate_payload_paths
            ],
        },
        "selection_policy": {
            "schema": "mlx_dynamic_learned_sweep_autopilot_batch_root_selection_policy.v1",
            "group_by": (
                (
                    "positive_utility_runtime_balanced_buckets"
                    if runtime_telemetry["runtime_telemetry_used"]
                    else "positive_utility_cost_balanced_buckets"
                )
                if adaptive_rows_per_root
                else "queue_candidate_id_chunks"
            ),
            "root_utility": "sum_queue_candidate_acquisition_value",
            "rows_per_root": None if adaptive_rows_per_root else max_rows_per_root,
            "max_rows_per_root": max_rows_per_root,
            "sort": [
                "root_utility_desc",
                "recursive_stage_asc",
                "best_lower_confidence_score_asc",
                "runtime_cost_estimate_sum_asc",
                "cost_units_sum_asc",
                "queue_candidate_id_asc",
            ],
            **LOCAL_FALSE_AUTHORITY,
        },
        "selected_roots": selected_roots,
        "run_specs": run_specs,
    }


__all__ = [
    "DEFAULT_SWEEP_CONFIG_ID",
    "LEARNED_SWEEP_PLAN_SCHEMA",
    "LEARNED_SWEEP_ROW_SCHEMA",
    "MLX_LEARNED_SWEEP_BATCH_ROOT_PLAN_SCHEMA",
    "TOOL",
    "MLXLearnedSweepBatchRootError",
    "build_mlx_learned_sweep_autopilot_batch_root_plan",
]
