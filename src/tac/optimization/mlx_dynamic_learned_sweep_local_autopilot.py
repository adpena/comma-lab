# SPDX-License-Identifier: MIT
"""Bounded local MLX learned-sweep autopilot.

This layer intentionally stays thin: it repeatedly calls the local MLX actuator,
appends observations, and replans. It does not claim scores, promote archives,
or dispatch exact eval work.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.mlx_dynamic_learned_sweep import FALSE_AUTHORITY
from tac.optimization.mlx_dynamic_learned_sweep_local_actuator import (
    SUPPORTED_SWEEP_CONFIG_ID,
    SUPPORTED_SWEEP_CONFIG_IDS,
    MLXDynamicLearnedSweepLocalActuatorError,
    ResponseBuilder,
    execute_local_mlx_sweep_rows,
    replan_after_local_actuation,
)
from tac.optimization.mlx_dynamic_sweep_observations import load_observation_rows
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

SCHEMA = "mlx_dynamic_learned_sweep_local_autopilot.v1"
TOOL = "tac.optimization.mlx_dynamic_learned_sweep_local_autopilot"


class MLXDynamicLearnedSweepLocalAutopilotError(ValueError):
    """Raised when the local learned-sweep autopilot cannot proceed safely."""


def run_local_mlx_sweep_autopilot(
    *,
    initial_plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    candidate_payloads: Sequence[Mapping[str, Any]],
    incumbent_score: float,
    output_dir: Path,
    observation_jsonl: Path,
    max_iterations: int = 1,
    max_new_observations: int = 1,
    rows_per_replan: int = 1,
    sweep_config_id: str = SUPPORTED_SWEEP_CONFIG_ID,
    optimization_pass_id: str | None = None,
    candidate_ids: Sequence[str] | None = None,
    queue_candidate_ids: Sequence[str] | None = None,
    source_artifact_root: str | Path | None = None,
    device_type: str = "cpu",
    allow_gpu_research_signal: bool = False,
    batch_pairs: int = 1,
    progress_every: int = 0,
    max_seconds: float | None = None,
    replan_top_k: int | None = None,
    replan_per_pass_top_k: int | None = None,
    response_builder: ResponseBuilder | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    source_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run bounded local actuation/replan cycles.

    ``max_iterations`` counts actuation/replan cycles. ``rows_per_replan``
    controls how many already-ranked local rows the actuator may execute or
    harvest before a fresh plan is built from the append-only observation
    ledger.
    """

    _validate_inputs(
        initial_plan=initial_plan,
        selection=selection,
        candidate_payloads=candidate_payloads,
        max_iterations=max_iterations,
        max_new_observations=max_new_observations,
        rows_per_replan=rows_per_replan,
        sweep_config_id=sweep_config_id,
        max_seconds=max_seconds,
    )
    queue_candidate_id_filters = _normalize_filter_values(queue_candidate_ids)
    if queue_candidate_id_filters and (
        int(max_iterations) != 1
        or int(max_new_observations) < len(queue_candidate_id_filters)
        or int(rows_per_replan) < len(queue_candidate_id_filters)
    ):
        raise MLXDynamicLearnedSweepLocalAutopilotError(
            "queue_candidate_id filters require one bounded exact-row cycle"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = float(monotonic())
    initial_observation_count = len(load_observation_rows(observation_jsonl))

    current_plan: Mapping[str, Any] = initial_plan
    cycles: list[dict[str, Any]] = []
    observations_added = 0
    stopping_reason = "max_iterations_reached"

    for iteration_index in range(1, int(max_iterations) + 1):
        elapsed = _elapsed_seconds(monotonic, start_time)
        if max_seconds is not None and elapsed >= float(max_seconds):
            stopping_reason = "max_seconds_reached_before_cycle"
            break
        if observations_added >= int(max_new_observations):
            stopping_reason = "max_new_observations_reached"
            break

        remaining_observations = int(max_new_observations) - observations_added
        max_rows = min(int(rows_per_replan), remaining_observations)
        cycle_dir = output_dir / f"cycle_{iteration_index:04d}"
        try:
            summary = execute_local_mlx_sweep_rows(
                plan=current_plan,
                selection=selection,
                output_dir=cycle_dir / "actuation",
                observation_jsonl=observation_jsonl,
                max_rows=max_rows,
                sweep_config_id=sweep_config_id,
                optimization_pass_id=optimization_pass_id,
                candidate_ids=candidate_ids,
                queue_candidate_ids=queue_candidate_ids,
                source_artifact_root=source_artifact_root,
                device_type=device_type,
                allow_gpu_research_signal=allow_gpu_research_signal,
                batch_pairs=batch_pairs,
                progress_every=progress_every,
                response_builder=response_builder
                if response_builder is not None
                else _default_response_builder(),
            )
        except MLXDynamicLearnedSweepLocalActuatorError as exc:
            if "no ready local sweep rows matched actuator filters" in str(exc):
                stopping_reason = "ready_rows_exhausted"
                break
            raise MLXDynamicLearnedSweepLocalAutopilotError(str(exc)) from exc

        cycle_observation_count = int(summary.get("observation_row_count") or 0)
        observations_added += cycle_observation_count
        replan_json_out = cycle_dir / "learned_sweep_plan.after_cycle.json"
        replan_md_out = cycle_dir / "learned_sweep_plan.after_cycle.md"
        current_plan = replan_after_local_actuation(
            incumbent_score=float(incumbent_score),
            candidate_payloads=candidate_payloads,
            source_plan=current_plan,
            observation_jsonl_paths=[observation_jsonl],
            json_out=replan_json_out,
            md_out=replan_md_out,
            top_k=replan_top_k,
            per_pass_top_k=replan_per_pass_top_k,
        )
        cycles.append(
            {
                "schema": "mlx_dynamic_learned_sweep_local_autopilot_cycle.v1",
                **FALSE_AUTHORITY,
                "candidate_generation_only": True,
                "observation_only": True,
                "dispatch_attempted": False,
                "iteration_index": iteration_index,
                "actuation_output_dir": str(cycle_dir / "actuation"),
                "executed_row_count": summary["executed_row_count"],
                "observation_row_count": summary["observation_row_count"],
                "replan_json_out": str(replan_json_out),
                "replan_md_out": str(replan_md_out),
                "replan_ranked_row_count": current_plan["summary"]["ranked_row_count"],
                "replan_local_ready_row_count": current_plan["summary"][
                    "local_ready_row_count"
                ],
                "replan_suppressed_observed_row_count": current_plan["summary"][
                    "suppressed_observed_row_count"
                ],
                "actuation_summary": _compact_actuation_summary(summary),
            }
        )
        if observations_added >= int(max_new_observations):
            stopping_reason = "max_new_observations_reached"
            break
    else:
        stopping_reason = "max_iterations_reached"

    final_observation_count = len(load_observation_rows(observation_jsonl))
    elapsed_total = _elapsed_seconds(monotonic, start_time)
    filter_rollup = _cycle_filter_rollup(cycles)
    return {
        "schema": SCHEMA,
        "producer": TOOL,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "observation_only": True,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "local_mlx_device_used": any(
            cycle["actuation_summary"]["local_mlx_device_used"] for cycle in cycles
        ),
        "local_cpu_advisory_artifact_used": any(
            cycle["actuation_summary"].get("local_cpu_advisory_artifact_used")
            for cycle in cycles
        ),
        "allowed_use": "bounded_local_learned_sweep_feedback_loop_only",
        "supported_sweep_config_id": SUPPORTED_SWEEP_CONFIG_ID,
        "supported_sweep_config_ids": sorted(SUPPORTED_SWEEP_CONFIG_IDS),
        "sweep_config_id": sweep_config_id,
        "optimization_pass_id": optimization_pass_id,
        "candidate_id_filters": _normalize_filter_values(candidate_ids),
        "queue_candidate_id_filters": queue_candidate_id_filters,
        **filter_rollup,
        "device_type": device_type,
        "allow_gpu_research_signal": bool(allow_gpu_research_signal),
        "batch_pairs": int(batch_pairs),
        "max_iterations": int(max_iterations),
        "max_new_observations": int(max_new_observations),
        "rows_per_replan": int(rows_per_replan),
        "max_seconds": max_seconds,
        "elapsed_seconds": elapsed_total,
        "stopping_reason": stopping_reason,
        "cycle_count": len(cycles),
        "executed_row_count": sum(
            int(cycle["executed_row_count"]) for cycle in cycles
        ),
        "new_observation_row_count": observations_added,
        "initial_observation_row_count": initial_observation_count,
        "final_observation_row_count": final_observation_count,
        "observation_jsonl": str(observation_jsonl),
        "final_plan_summary": _final_plan_summary(current_plan),
        "source_artifacts": dict(source_artifacts or {}),
        "cycles": cycles,
    }


def _default_response_builder() -> ResponseBuilder:
    from tac.local_acceleration.mlx_scorer_response import (
        build_mlx_scorer_response_payload,
    )

    return build_mlx_scorer_response_payload


def _validate_inputs(
    *,
    initial_plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    candidate_payloads: Sequence[Mapping[str, Any]],
    max_iterations: int,
    max_new_observations: int,
    rows_per_replan: int,
    sweep_config_id: str,
    max_seconds: float | None,
) -> None:
    if initial_plan.get("schema") != "mlx_dynamic_learned_sweep_plan.v1":
        raise MLXDynamicLearnedSweepLocalAutopilotError("initial_plan schema mismatch")
    if selection.get("schema") != "mlx_effective_spend_triage_candidate_selection.v1":
        raise MLXDynamicLearnedSweepLocalAutopilotError("selection schema mismatch")
    _require_false_authority(initial_plan, label="initial_plan")
    _require_false_authority(selection, label="selection")
    if not candidate_payloads:
        raise MLXDynamicLearnedSweepLocalAutopilotError(
            "candidate_payloads must be non-empty"
        )
    if int(max_iterations) <= 0:
        raise MLXDynamicLearnedSweepLocalAutopilotError(
            "max_iterations must be positive"
        )
    if int(max_new_observations) <= 0:
        raise MLXDynamicLearnedSweepLocalAutopilotError(
            "max_new_observations must be positive"
        )
    if int(rows_per_replan) <= 0:
        raise MLXDynamicLearnedSweepLocalAutopilotError(
            "rows_per_replan must be positive"
        )
    if sweep_config_id not in SUPPORTED_SWEEP_CONFIG_IDS:
        raise MLXDynamicLearnedSweepLocalAutopilotError(
            f"unsupported sweep_config_id {sweep_config_id!r}; "
            "supported local learned-sweep configs are: "
            + ", ".join(sorted(SUPPORTED_SWEEP_CONFIG_IDS))
        )
    if max_seconds is not None and float(max_seconds) <= 0.0:
        raise MLXDynamicLearnedSweepLocalAutopilotError(
            "max_seconds must be positive when supplied"
        )


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if payload.get(key) is not False:
            raise MLXDynamicLearnedSweepLocalAutopilotError(
                f"{label} {key} must be explicit false"
            )
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise MLXDynamicLearnedSweepLocalAutopilotError(str(exc)) from exc


def _elapsed_seconds(monotonic: Callable[[], float], start_time: float) -> float:
    return max(0.0, float(monotonic()) - start_time)


def _compact_actuation_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": summary.get("schema"),
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "observation_only": True,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "local_mlx_device_used": bool(summary.get("local_mlx_device_used")),
        "local_cpu_advisory_artifact_used": bool(
            summary.get("local_cpu_advisory_artifact_used")
        ),
        "device_type": summary.get("device_type"),
        "batch_pairs": summary.get("batch_pairs"),
        "optimization_pass_id_filter": summary.get("optimization_pass_id_filter"),
        "candidate_id_filters": summary.get("candidate_id_filters"),
        "queue_candidate_id_filters": summary.get("queue_candidate_id_filters"),
        "executed_filter_match": summary.get("executed_filter_match"),
        "executed_filter_violation_count": summary.get(
            "executed_filter_violation_count"
        ),
        "requested_candidate_id_count": summary.get("requested_candidate_id_count"),
        "requested_queue_candidate_id_count": summary.get(
            "requested_queue_candidate_id_count"
        ),
        "executed_candidate_id_count": summary.get("executed_candidate_id_count"),
        "executed_queue_candidate_id_count": summary.get(
            "executed_queue_candidate_id_count"
        ),
        "executed_candidate_id_set": summary.get("executed_candidate_id_set"),
        "executed_queue_candidate_id_set": summary.get("executed_queue_candidate_id_set"),
        "executed_unique_candidate_id": summary.get("executed_unique_candidate_id"),
        "executed_unique_queue_candidate_id": summary.get(
            "executed_unique_queue_candidate_id"
        ),
        "executed_row_count": summary.get("executed_row_count"),
        "observation_row_count": summary.get("observation_row_count"),
        "observation_jsonl": summary.get("observation_jsonl"),
        "executed_queue_candidate_ids": [
            row.get("queue_candidate_id")
            for row in summary.get("executed_rows", [])
            if isinstance(row, Mapping)
        ],
    }


def _cycle_filter_rollup(cycles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = [
        cycle.get("actuation_summary")
        for cycle in cycles
        if isinstance(cycle.get("actuation_summary"), Mapping)
    ]
    candidate_ids = _flatten_summary_values(summaries, "executed_candidate_ids")
    queue_candidate_ids = _flatten_summary_values(summaries, "executed_queue_candidate_ids")
    unique_candidate_ids = sorted(set(candidate_ids))
    unique_queue_candidate_ids = sorted(set(queue_candidate_ids))
    return {
        "executed_filter_match": (
            bool(summaries)
            and all(summary.get("executed_filter_match") is True for summary in summaries)
        ),
        "executed_filter_violation_count": sum(
            int(summary.get("executed_filter_violation_count") or 0)
            for summary in summaries
        ),
        "requested_candidate_id_count": sum(
            int(summary.get("requested_candidate_id_count") or 0)
            for summary in summaries
        ),
        "requested_queue_candidate_id_count": sum(
            int(summary.get("requested_queue_candidate_id_count") or 0)
            for summary in summaries
        ),
        "executed_candidate_id_count": len(candidate_ids),
        "executed_queue_candidate_id_count": len(queue_candidate_ids),
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


def _flatten_summary_values(
    summaries: Sequence[Mapping[str, Any]],
    key: str,
) -> list[str]:
    values: list[str] = []
    for summary in summaries:
        raw_values = summary.get(key)
        if not isinstance(raw_values, list):
            continue
        values.extend(str(value) for value in raw_values if value is not None)
    return values


def _final_plan_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    summary = plan.get("summary")
    if not isinstance(summary, Mapping):
        return {
            **FALSE_AUTHORITY,
            "schema": "mlx_dynamic_learned_sweep_local_autopilot_final_plan_summary.v1",
            "summary_available": False,
        }
    return {
        "schema": "mlx_dynamic_learned_sweep_local_autopilot_final_plan_summary.v1",
        **FALSE_AUTHORITY,
        "summary_available": True,
        "ranked_row_count": summary.get("ranked_row_count"),
        "local_ready_row_count": summary.get("local_ready_row_count"),
        "exact_eval_candidate_row_count": summary.get("exact_eval_candidate_row_count"),
        "suppressed_observed_row_count": summary.get("suppressed_observed_row_count"),
        "observation_row_count": summary.get("observation_row_count"),
    }


def _normalize_filter_values(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


__all__ = [
    "SCHEMA",
    "TOOL",
    "MLXDynamicLearnedSweepLocalAutopilotError",
    "run_local_mlx_sweep_autopilot",
]
