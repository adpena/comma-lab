# SPDX-License-Identifier: MIT
"""Compile MLX learned-sweep autopilot runs into experiment_queue work."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.mlx_dynamic_learned_sweep import FALSE_AUTHORITY
from tac.optimization.mlx_dynamic_learned_sweep_local_actuator import (
    SUPPORTED_SWEEP_CONFIG_ID,
    SUPPORTED_SWEEP_CONFIG_IDS,
    SWEEP_CONFIG_ID_MACOS_CPU_ADVISORY,
)
from tac.optimization.mlx_dynamic_learned_sweep_local_autopilot import (
    SCHEMA as MLX_LEARNED_SWEEP_AUTOPILOT_SUMMARY_SCHEMA,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

from .experiment_queue import QUEUE_SCHEMA, ExperimentQueueError, normalize_queue_definition

MLX_LEARNED_SWEEP_AUTOPILOT_QUEUE_SCHEMA = (
    "mlx_dynamic_learned_sweep_autopilot_queue_plan.v1"
)
MLX_LEARNED_SWEEP_AUTOPILOT_BATCH_QUEUE_SCHEMA = (
    "mlx_dynamic_learned_sweep_autopilot_batch_queue_plan.v1"
)
MLX_RUNTIME_TELEMETRY_STATE_DISCOVERY_POLICY_SCHEMA = (
    "mlx_runtime_telemetry_state_discovery_policy.v1"
)
TOOL_NAME = "comma_lab.scheduler.mlx_learned_sweep_autopilot_queue"
AUTOPILOT_TOOL = "tools/run_mlx_dynamic_learned_sweep_autopilot.py"
LEARNED_SWEEP_PLAN_SCHEMA = "mlx_dynamic_learned_sweep_plan.v1"
EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA = (
    "mlx_effective_spend_triage_candidate_selection.v1"
)
_MACOS_CPU_ADVISORY_SOURCE_KEYS = (
    "local_cpu_advisory_source_path",
    "macos_cpu_advisory_source_path",
    "local_cpu_advisory_path",
    "macos_cpu_advisory_path",
    "advisory_eval_path",
)
_MACOS_CPU_ADVISORY_BASELINE_KEYS = (
    "window_baseline_local_cpu_advisory_source_path",
    "window_baseline_macos_cpu_advisory_source_path",
    "window_baseline_local_cpu_advisory_path",
    "window_baseline_macos_cpu_advisory_path",
    "baseline_local_cpu_advisory_path",
    "baseline_macos_cpu_advisory_path",
    "baseline_advisory_eval_path",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentQueueError(f"{label}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ExperimentQueueError(f"{label}: expected JSON object")
    return payload


def _file_record(path: Path, *, repo_root: Path) -> dict[str, Any]:
    from tac.repo_io import sha256_file

    if not path.is_file():
        raise ExperimentQueueError(f"required input artifact missing: {path}")
    return {
        "path": _repo_rel(path, repo_root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _safe_id(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in value)
    return out.strip("._-").lower() or "mlx_learned_sweep_autopilot"


def _positive_int(value: int, *, label: str) -> int:
    if isinstance(value, bool) or int(value) < 1:
        raise ExperimentQueueError(f"{label} must be >= 1")
    return int(value)


def _non_negative_int(value: int, *, label: str) -> int:
    if isinstance(value, bool) or int(value) < 0:
        raise ExperimentQueueError(f"{label} must be non-negative")
    return int(value)


def _optional_positive_float(value: float | None, *, label: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0.0:
        raise ExperimentQueueError(f"{label} must be positive when supplied")
    return parsed


def _optional_text(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ExperimentQueueError(f"{label} must be a non-empty string when supplied")
    return value.strip()


def _required_text(value: Any, *, label: str) -> str:
    text = _optional_text(value, label=label)
    if text is None:
        raise ExperimentQueueError(f"{label} is required")
    return text


def _optional_text_list(value: Any, *, label: str) -> list[str] | None:
    if value is None:
        return None
    values = value if isinstance(value, list) else [value]
    out: list[str] = []
    for index, item in enumerate(values):
        text = str(item).strip()
        if not text:
            raise ExperimentQueueError(f"{label}[{index}] must be non-empty")
        if text not in out:
            out.append(text)
    return out


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if payload.get(key) is not False:
            raise ExperimentQueueError(f"{label}: {key} must be explicit false")
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc


def _validate_plan(
    path: Path,
    *,
    repo_root: Path,
    sweep_config_id: str,
    optimization_pass_id: str | None,
    candidate_ids: Sequence[str] | None,
    queue_candidate_ids: Sequence[str] | None,
) -> dict[str, Any]:
    payload = _load_json_object(path, label="plan")
    if payload.get("schema") != LEARNED_SWEEP_PLAN_SCHEMA:
        raise ExperimentQueueError(
            f"plan schema must be {LEARNED_SWEEP_PLAN_SCHEMA}"
        )
    _require_false_authority(payload, label="plan")
    _require_ready_local_mlx_rows(
        payload,
        sweep_config_id=sweep_config_id,
        optimization_pass_id=optimization_pass_id,
        candidate_ids=candidate_ids,
        queue_candidate_ids=queue_candidate_ids,
    )
    return _file_record(path, repo_root=repo_root)


def _require_ready_local_mlx_rows(
    plan: Mapping[str, Any],
    *,
    sweep_config_id: str,
    optimization_pass_id: str | None,
    candidate_ids: Sequence[str] | None,
    queue_candidate_ids: Sequence[str] | None,
) -> None:
    rows = plan.get("ranked_sweep_rows")
    if not isinstance(rows, list):
        raise ExperimentQueueError("plan ranked_sweep_rows must be a list")
    queue_candidate_filter = set(queue_candidate_ids or [])
    queue_candidate_match_counts = dict.fromkeys(queue_candidate_filter, 0)
    ready_rows = _matching_ready_plan_rows(
        plan,
        sweep_config_id=sweep_config_id,
        optimization_pass_id=optimization_pass_id,
        candidate_ids=candidate_ids,
        queue_candidate_ids=queue_candidate_ids,
    )
    for row in ready_rows:
        if queue_candidate_filter:
            queue_candidate_id = str(row.get("queue_candidate_id") or "")
            queue_candidate_match_counts[queue_candidate_id] += 1
    if not ready_rows:
        raise ExperimentQueueError(
            f"plan has no ready {sweep_config_id} rows for the requested autopilot filters"
        )
    missing_queue_candidate_ids = sorted(
        queue_candidate_id
        for queue_candidate_id, count in queue_candidate_match_counts.items()
        if count == 0
    )
    if missing_queue_candidate_ids:
        raise ExperimentQueueError(
            "requested queue_candidate_id filters were not all ready before execution: "
            + ", ".join(missing_queue_candidate_ids)
        )
    duplicate_queue_candidate_ids = sorted(
        queue_candidate_id
        for queue_candidate_id, count in queue_candidate_match_counts.items()
        if count > 1
    )
    if duplicate_queue_candidate_ids:
        raise ExperimentQueueError(
            "queue_candidate_ids must identify exactly one ready row before execution; "
            "duplicate ready rows for: " + ", ".join(duplicate_queue_candidate_ids)
        )


def _matching_ready_plan_rows(
    plan: Mapping[str, Any],
    *,
    sweep_config_id: str,
    optimization_pass_id: str | None,
    candidate_ids: Sequence[str] | None,
    queue_candidate_ids: Sequence[str] | None,
) -> list[Mapping[str, Any]]:
    rows = plan.get("ranked_sweep_rows")
    if not isinstance(rows, list):
        raise ExperimentQueueError("plan ranked_sweep_rows must be a list")
    candidate_filter = set(candidate_ids or [])
    queue_candidate_filter = set(queue_candidate_ids or [])
    out: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ExperimentQueueError(
                f"plan ranked_sweep_rows[{index}] must be an object"
            )
        if row.get("schema") != "mlx_dynamic_learned_sweep_row.v1":
            continue
        if row.get("ready_for_local_sweep") is not True:
            continue
        if row.get("sweep_config_id") != sweep_config_id:
            continue
        if row.get("exact_eval_candidate") is True:
            raise ExperimentQueueError(
                "plan has exact-eval candidate marked ready for local sweep"
            )
        if (
            optimization_pass_id is not None
            and row.get("optimization_pass_id") != optimization_pass_id
        ):
            continue
        if candidate_filter and str(row.get("candidate_id") or "") not in candidate_filter:
            continue
        if (
            queue_candidate_filter
            and str(row.get("queue_candidate_id") or "") not in queue_candidate_filter
        ):
            continue
        out.append(row)
    return out


def _validate_selection(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = _load_json_object(path, label="selection")
    if payload.get("schema") != EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA:
        raise ExperimentQueueError(
            f"selection schema must be {EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA}"
        )
    _require_false_authority(payload, label="selection")
    return _file_record(path, repo_root=repo_root)


def _validate_macos_cpu_advisory_selection_paths(
    *,
    plan_path: Path,
    selection_path: Path,
    source_artifact_root: Path,
    sweep_config_id: str,
    optimization_pass_id: str | None,
    candidate_ids: Sequence[str],
    queue_candidate_ids: Sequence[str],
) -> dict[str, Any] | None:
    if sweep_config_id != SWEEP_CONFIG_ID_MACOS_CPU_ADVISORY:
        return None
    plan = _load_json_object(plan_path, label="plan")
    selection = _load_json_object(selection_path, label="selection")
    ready_rows = _matching_ready_plan_rows(
        plan,
        sweep_config_id=sweep_config_id,
        optimization_pass_id=optimization_pass_id,
        candidate_ids=candidate_ids,
        queue_candidate_ids=queue_candidate_ids,
    )
    selected_rows = selection.get("selected_rows")
    if not isinstance(selected_rows, list):
        raise ExperimentQueueError("selection selected_rows must be a list")
    selection_by_candidate: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(selected_rows):
        if not isinstance(row, Mapping):
            raise ExperimentQueueError(
                f"selection selected_rows[{index}] must be an object"
            )
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id:
            selection_by_candidate[candidate_id] = row

    validated: list[dict[str, Any]] = []
    for row in ready_rows:
        candidate_id = _required_text(row.get("candidate_id"), label="candidate_id")
        queue_candidate_id = _required_text(
            row.get("queue_candidate_id"),
            label="queue_candidate_id",
        )
        selection_row = selection_by_candidate.get(candidate_id)
        if selection_row is None:
            raise ExperimentQueueError(
                f"{queue_candidate_id}: missing source selection row"
            )
        candidate_path = _resolve_existing_path_from_any_key(
            selection_row,
            keys=_MACOS_CPU_ADVISORY_SOURCE_KEYS,
            source_artifact_root=source_artifact_root,
            label=f"{queue_candidate_id} candidate macOS-CPU advisory path",
        )
        baseline_path = _resolve_existing_path_from_any_key(
            selection_row,
            keys=_MACOS_CPU_ADVISORY_BASELINE_KEYS,
            source_artifact_root=source_artifact_root,
            label=f"{queue_candidate_id} baseline macOS-CPU advisory path",
        )
        validated.append(
            {
                "queue_candidate_id": queue_candidate_id,
                "candidate_id": candidate_id,
                "candidate_advisory_path": str(candidate_path),
                "baseline_advisory_path": str(baseline_path),
            }
        )
    return {
        "schema": "mlx_learned_sweep_macos_cpu_advisory_selection_path_preflight.v1",
        "sweep_config_id": sweep_config_id,
        "validated_row_count": len(validated),
        "validated_rows": validated,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _resolve_existing_path_from_any_key(
    row: Mapping[str, Any],
    *,
    keys: Sequence[str],
    source_artifact_root: Path,
    label: str,
) -> Path:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            resolved = path if path.is_absolute() else source_artifact_root / path
            if not resolved.is_file():
                raise ExperimentQueueError(f"{label} does not exist: {resolved}")
            return resolved
    raise ExperimentQueueError(f"{label} is required")


def _validate_candidate_payloads(
    paths: Sequence[Path],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    if not paths:
        raise ExperimentQueueError("at least one candidate_payload_path is required")
    records: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        payload = _load_json_object(path, label=f"candidate_payload[{index}]")
        try:
            require_no_truthy_authority_fields(
                payload,
                context=f"candidate_payload[{index}]",
            )
        except ValueError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        records.append(_file_record(path, repo_root=repo_root))
    return records


def _runtime_telemetry_policy(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ExperimentQueueError("runtime_telemetry_policy must be an object")
    try:
        require_no_truthy_authority_fields(
            value,
            context="runtime_telemetry_policy",
        )
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    schema = value.get("schema")
    if schema != MLX_RUNTIME_TELEMETRY_STATE_DISCOVERY_POLICY_SCHEMA:
        raise ExperimentQueueError(
            "runtime_telemetry_policy schema must be "
            f"{MLX_RUNTIME_TELEMETRY_STATE_DISCOVERY_POLICY_SCHEMA}"
        )
    policy_id = _optional_text(value.get("policy_id"), label="runtime_telemetry_policy.policy_id")
    if policy_id is None:
        raise ExperimentQueueError("runtime_telemetry_policy.policy_id is required")
    mode = _optional_text(value.get("mode"), label="runtime_telemetry_policy.mode")
    if mode not in {"explicit_states", "auto_discover_compatible_states"}:
        raise ExperimentQueueError(
            "runtime_telemetry_policy.mode must be explicit_states or "
            "auto_discover_compatible_states"
        )
    selected_state_paths = value.get("selected_state_paths")
    if not isinstance(selected_state_paths, list):
        raise ExperimentQueueError(
            "runtime_telemetry_policy.selected_state_paths must be a list"
        )
    discovered_state_paths = value.get("discovered_state_paths", [])
    if not isinstance(discovered_state_paths, list):
        raise ExperimentQueueError(
            "runtime_telemetry_policy.discovered_state_paths must be a list"
        )
    out = dict(value)
    out["policy_id"] = policy_id
    out["mode"] = mode
    out["selected_state_paths"] = [
        _required_text(
            path,
            label=f"runtime_telemetry_policy.selected_state_paths[{index}]",
        )
        for index, path in enumerate(selected_state_paths)
    ]
    out["discovered_state_paths"] = [
        _required_text(
            path,
            label=f"runtime_telemetry_policy.discovered_state_paths[{index}]",
        )
        for index, path in enumerate(discovered_state_paths)
    ]
    out.update(FALSE_AUTHORITY)
    return out


def _append_optional_flag(
    command: list[str],
    flag: str,
    value: str | int | float | None,
) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _autopilot_command(
    *,
    plan_path: Path,
    selection_path: Path,
    candidate_payload_paths: Sequence[Path],
    incumbent_score: float,
    output_dir: Path,
    observation_jsonl: Path,
    summary_json_out: Path,
    repo_root: Path,
    max_iterations: int,
    max_new_observations: int,
    rows_per_replan: int,
    sweep_config_id: str,
    optimization_pass_id: str | None,
    candidate_ids: Sequence[str] | None,
    queue_candidate_ids: Sequence[str] | None,
    source_artifact_root: Path,
    device: str,
    allow_gpu_research_signal: bool,
    batch_pairs: int,
    progress_every: int,
    max_seconds: float | None,
    replan_top_k: int | None,
    replan_per_pass_top_k: int | None,
) -> list[str]:
    command = [
        ".venv/bin/python",
        AUTOPILOT_TOOL,
        "--plan",
        _repo_rel(plan_path, repo_root),
        "--selection",
        _repo_rel(selection_path, repo_root),
    ]
    for path in candidate_payload_paths:
        command.extend(["--candidate-payload", _repo_rel(path, repo_root)])
    command.extend(
        [
            "--incumbent-score",
            str(float(incumbent_score)),
            "--output-dir",
            _repo_rel(output_dir, repo_root),
            "--observation-jsonl",
            _repo_rel(observation_jsonl, repo_root),
            "--summary-json-out",
            _repo_rel(summary_json_out, repo_root),
            "--max-iterations",
            str(max_iterations),
            "--max-new-observations",
            str(max_new_observations),
            "--rows-per-replan",
            str(rows_per_replan),
            "--sweep-config-id",
            sweep_config_id,
            "--source-artifact-root",
            _repo_rel(source_artifact_root, repo_root),
            "--device",
            device,
            "--batch-pairs",
            str(batch_pairs),
            "--progress-every",
            str(progress_every),
        ]
    )
    if optimization_pass_id:
        command.extend(["--optimization-pass-id", optimization_pass_id])
    for candidate_id in candidate_ids or []:
        command.extend(["--candidate-id", candidate_id])
    for queue_candidate_id in queue_candidate_ids or []:
        command.extend(["--queue-candidate-id", queue_candidate_id])
    if allow_gpu_research_signal:
        command.append("--allow-gpu-research-signal")
    _append_optional_flag(command, "--max-seconds", max_seconds)
    _append_optional_flag(command, "--replan-top-k", replan_top_k)
    _append_optional_flag(command, "--replan-per-pass-top-k", replan_per_pass_top_k)
    return command


def _step_id(index: int, *, chain_steps: int) -> str:
    if chain_steps == 1:
        return "run_mlx_learned_sweep_autopilot"
    return f"run_mlx_learned_sweep_autopilot_{index:04d}"


def _step_output_dir(run_root: Path, index: int, *, chain_steps: int) -> Path:
    if chain_steps == 1:
        return run_root
    return run_root / f"step_{index:04d}"


def _step_replan_path(output_dir: Path) -> Path:
    return output_dir / "cycle_0001" / "learned_sweep_plan.after_cycle.json"


def _step_postconditions(
    *,
    summary_path: Path,
    observation_path: Path,
    repo: Path,
    include_replan_path: bool,
    sweep_config_id: str,
    candidate_ids: Sequence[str],
    queue_candidate_ids: Sequence[str],
) -> list[dict[str, Any]]:
    required_equals: dict[str, Any] = {
        "schema": MLX_LEARNED_SWEEP_AUTOPILOT_SUMMARY_SCHEMA,
        "sweep_config_id": sweep_config_id,
        "executed_filter_match": True,
    }
    if len(candidate_ids) == 1:
        required_equals["executed_unique_candidate_id"] = candidate_ids[0]
    if len(queue_candidate_ids) == 1:
        required_equals["executed_unique_queue_candidate_id"] = queue_candidate_ids[0]
    if candidate_ids:
        required_equals["executed_candidate_id_set"] = sorted(set(candidate_ids))
        required_equals["executed_candidate_id_count"] = len(candidate_ids)
    if queue_candidate_ids:
        required_equals["executed_queue_candidate_id_set"] = sorted(
            set(queue_candidate_ids)
        )
        required_equals["executed_queue_candidate_id_count"] = len(queue_candidate_ids)
    postconditions: list[dict[str, Any]] = [
        {
            "type": "path_exists",
            "path": _repo_rel(summary_path, repo),
        },
        {
            "type": "json_equals",
            "path": _repo_rel(summary_path, repo),
            "key": "schema",
            "equals": MLX_LEARNED_SWEEP_AUTOPILOT_SUMMARY_SCHEMA,
        },
        {
            "type": "json_false_authority",
            "path": _repo_rel(summary_path, repo),
        },
        {
            "type": "json_completion_contract",
            "path": _repo_rel(summary_path, repo),
            "required_equals": required_equals,
            "required_false": list(FALSE_AUTHORITY),
            "required_positive_int": [
                "cycle_count",
                "executed_row_count",
                "new_observation_row_count",
                "final_observation_row_count",
            ],
        },
        {
            "type": "path_exists",
            "path": _repo_rel(observation_path, repo),
        },
        {
            "type": "jsonl_false_authority",
            "path": _repo_rel(observation_path, repo),
            "schema_equals": "mlx_dynamic_sweep_observation.v1",
        },
    ]
    if include_replan_path:
        replan_path = _step_replan_path(summary_path.parent)
        postconditions.extend(
            [
                {
                    "type": "path_exists",
                    "path": _repo_rel(replan_path, repo),
                },
                {
                    "type": "json_equals",
                    "path": _repo_rel(replan_path, repo),
                    "key": "schema",
                    "equals": LEARNED_SWEEP_PLAN_SCHEMA,
                },
                {
                    "type": "json_false_authority",
                    "path": _repo_rel(replan_path, repo),
                },
            ]
        )
    return postconditions


def build_mlx_learned_sweep_autopilot_queue(
    *,
    plan_path: str | Path,
    selection_path: str | Path,
    candidate_payload_paths: Sequence[str | Path],
    incumbent_score: float,
    output_root: str | Path,
    observation_jsonl: str | Path,
    queue_id: str,
    repo_root: str | Path,
    lane_id: str = "mlx_dynamic_learned_sweep_local_autopilot",
    local_cpu_concurrency: int = 1,
    local_mlx_concurrency: int = 1,
    timeout_seconds: int = 0,
    max_iterations: int = 1,
    max_new_observations: int = 1,
    rows_per_replan: int = 1,
    chain_steps: int = 1,
    sweep_config_id: str = SUPPORTED_SWEEP_CONFIG_ID,
    optimization_pass_id: str | None = None,
    candidate_ids: Sequence[str] | None = None,
    queue_candidate_ids: Sequence[str] | None = None,
    source_artifact_root: str | Path | None = None,
    device: str = "gpu",
    allow_gpu_research_signal: bool = True,
    batch_pairs: int = 1,
    progress_every: int = 0,
    max_seconds: float | None = None,
    replan_top_k: int | None = None,
    replan_per_pass_top_k: int | None = None,
    runtime_telemetry_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return queue-owned work for one bounded local learned-sweep autopilot run."""

    repo = Path(repo_root)
    if sweep_config_id not in SUPPORTED_SWEEP_CONFIG_IDS:
        raise ExperimentQueueError(
            f"sweep_config_id must be one of {', '.join(sorted(SUPPORTED_SWEEP_CONFIG_IDS))}"
        )
    if device not in {"cpu", "gpu"}:
        raise ExperimentQueueError("device must be cpu or gpu")
    if sweep_config_id == SWEEP_CONFIG_ID_MACOS_CPU_ADVISORY and device != "cpu":
        raise ExperimentQueueError(
            "macos_cpu_advisory sweep queues must use device=cpu"
        )
    if sweep_config_id == SWEEP_CONFIG_ID_MACOS_CPU_ADVISORY and allow_gpu_research_signal:
        raise ExperimentQueueError(
            "macos_cpu_advisory sweep queues cannot allow GPU research signal"
        )
    if device == "cpu" and allow_gpu_research_signal:
        raise ExperimentQueueError(
            "allow_gpu_research_signal must be false for cpu autopilot queues"
        )
    if device == "gpu" and not allow_gpu_research_signal:
        raise ExperimentQueueError(
            "gpu autopilot queues require --allow-gpu-research-signal"
        )
    local_cpu_concurrency = _positive_int(
        local_cpu_concurrency,
        label="local_cpu_concurrency",
    )
    local_mlx_concurrency = _positive_int(
        local_mlx_concurrency,
        label="local_mlx_concurrency",
    )
    timeout_seconds = _non_negative_int(timeout_seconds, label="timeout_seconds")
    max_iterations = _positive_int(max_iterations, label="max_iterations")
    max_new_observations = _positive_int(
        max_new_observations,
        label="max_new_observations",
    )
    rows_per_replan = _positive_int(rows_per_replan, label="rows_per_replan")
    chain_steps = _positive_int(chain_steps, label="chain_steps")
    if chain_steps > 1 and max_iterations != 1:
        raise ExperimentQueueError(
            "chain_steps > 1 requires max_iterations=1 so each step has a stable "
            "single-cycle replan path"
        )
    batch_pairs = _positive_int(batch_pairs, label="batch_pairs")
    progress_every = _non_negative_int(progress_every, label="progress_every")
    max_seconds = _optional_positive_float(max_seconds, label="max_seconds")
    if replan_top_k is not None:
        replan_top_k = _positive_int(replan_top_k, label="replan_top_k")
    if replan_per_pass_top_k is not None:
        replan_per_pass_top_k = _positive_int(
            replan_per_pass_top_k,
            label="replan_per_pass_top_k",
        )
    candidate_id_filters = _optional_text_list(candidate_ids, label="candidate_ids") or []
    queue_candidate_id_filters = (
        _optional_text_list(queue_candidate_ids, label="queue_candidate_ids") or []
    )
    if queue_candidate_id_filters and (
        chain_steps != 1
        or max_iterations != 1
        or max_new_observations < len(queue_candidate_id_filters)
        or rows_per_replan < len(queue_candidate_id_filters)
    ):
        raise ExperimentQueueError(
            "queue_candidate_ids require chain_steps=1, max_iterations=1, "
            "and row caps covering the filter set"
        )

    plan = _resolve_path(plan_path, repo_root=repo)
    selection = _resolve_path(selection_path, repo_root=repo)
    candidate_payloads = [
        _resolve_path(path, repo_root=repo) for path in candidate_payload_paths
    ]
    output_dir = _resolve_path(output_root, repo_root=repo) / _safe_id(queue_id)
    observation_path = _resolve_path(observation_jsonl, repo_root=repo)
    source_root = _resolve_path(source_artifact_root or repo, repo_root=repo)
    runtime_policy = _runtime_telemetry_policy(runtime_telemetry_policy)
    macos_cpu_advisory_preflight = _validate_macos_cpu_advisory_selection_paths(
        plan_path=plan,
        selection_path=selection,
        source_artifact_root=source_root,
        sweep_config_id=sweep_config_id,
        optimization_pass_id=optimization_pass_id,
        candidate_ids=candidate_id_filters,
        queue_candidate_ids=queue_candidate_id_filters,
    )

    source_records = {
        "plan": _validate_plan(
            plan,
            repo_root=repo,
            sweep_config_id=sweep_config_id,
            optimization_pass_id=optimization_pass_id,
            candidate_ids=candidate_id_filters,
            queue_candidate_ids=queue_candidate_id_filters,
        ),
        "selection": _validate_selection(selection, repo_root=repo),
        "candidate_payloads": _validate_candidate_payloads(
            candidate_payloads,
            repo_root=repo,
        ),
        "observation_jsonl": {
            "path": _repo_rel(observation_path, repo),
            "sha256": None,
            "bytes": None,
            "may_be_created_or_appended": True,
        }
        if not observation_path.exists()
        else _file_record(observation_path, repo_root=repo),
    }
    if macos_cpu_advisory_preflight is not None:
        source_records["macos_cpu_advisory_selection_path_preflight"] = (
            macos_cpu_advisory_preflight
        )
    resource_kind = "local_mlx" if device == "gpu" else "local_cpu"
    experiment_id = f"mlx_learned_sweep_autopilot_{_safe_id(queue_id)}"
    queue_metadata = {
        "schema": MLX_LEARNED_SWEEP_AUTOPILOT_QUEUE_SCHEMA,
        "tool": TOOL_NAME,
        "generated_at_utc": _utc_now(),
        "lane_id": lane_id,
        "source_artifacts": source_records,
        "autopilot_output_dir": _repo_rel(output_dir, repo),
        "observation_jsonl": _repo_rel(observation_path, repo),
        "chain_steps": chain_steps,
        "max_iterations_per_step": max_iterations,
        "max_new_observations_per_step": max_new_observations,
        "rows_per_replan": rows_per_replan,
        "sweep_config_id": sweep_config_id,
        "optimization_pass_id_filter": optimization_pass_id,
        "candidate_id_filters": candidate_id_filters,
        "queue_candidate_id_filters": queue_candidate_id_filters,
        "candidate_generation_only": True,
        "observation_only": True,
        "requires_exact_eval_before_promotion": True,
        **FALSE_AUTHORITY,
    }
    if runtime_policy is not None:
        queue_metadata["runtime_telemetry_policy"] = runtime_policy
    steps: list[dict[str, Any]] = []
    current_plan = plan
    for index in range(1, chain_steps + 1):
        step_id = _step_id(index, chain_steps=chain_steps)
        step_output_dir = _step_output_dir(output_dir, index, chain_steps=chain_steps)
        summary_path = step_output_dir / "local_mlx_autopilot_summary.json"
        command = _autopilot_command(
            plan_path=current_plan,
            selection_path=selection,
            candidate_payload_paths=candidate_payloads,
            incumbent_score=incumbent_score,
            output_dir=step_output_dir,
            observation_jsonl=observation_path,
            summary_json_out=summary_path,
            repo_root=repo,
            max_iterations=max_iterations,
            max_new_observations=max_new_observations,
            rows_per_replan=rows_per_replan,
            sweep_config_id=sweep_config_id,
            optimization_pass_id=optimization_pass_id,
            candidate_ids=candidate_id_filters,
            queue_candidate_ids=queue_candidate_id_filters,
            source_artifact_root=source_root,
            device=device,
            allow_gpu_research_signal=allow_gpu_research_signal,
            batch_pairs=batch_pairs,
            progress_every=progress_every,
            max_seconds=max_seconds,
            replan_top_k=replan_top_k,
            replan_per_pass_top_k=replan_per_pass_top_k,
        )
        step = {
            "id": step_id,
            "kind": "command",
            "command": command,
            "requires": []
            if index == 1
            else [_step_id(index - 1, chain_steps=chain_steps)],
            "resources": {"kind": resource_kind},
            "timeout_seconds": timeout_seconds,
            "postconditions": _step_postconditions(
                summary_path=summary_path,
                observation_path=observation_path,
                repo=repo,
                include_replan_path=chain_steps > 1,
                sweep_config_id=sweep_config_id,
                candidate_ids=candidate_id_filters,
                queue_candidate_ids=queue_candidate_id_filters,
            ),
            "telemetry": {
                "schema": MLX_LEARNED_SWEEP_AUTOPILOT_QUEUE_SCHEMA,
                "artifact_paths": [
                    _repo_rel(step_output_dir, repo),
                    _repo_rel(summary_path, repo),
                    _repo_rel(observation_path, repo),
                ],
                "input_artifact_paths": [
                    _repo_rel(current_plan, repo),
                    source_records["selection"]["path"],
                    *[
                        record["path"]
                        for record in source_records["candidate_payloads"]
                    ],
                ],
                "recursive": True,
            },
        }
        steps.append(step)
        current_plan = _step_replan_path(step_output_dir)
    return normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {
                    "local_cpu": local_cpu_concurrency,
                    "local_mlx": local_mlx_concurrency,
                },
            },
            "experiments": [
                {
                    "id": experiment_id,
                    "lane_id": lane_id,
                    "priority": 10,
                    "metadata": queue_metadata,
                    "steps": steps,
                }
            ],
        }
    )


def _spec_value(
    spec: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    return spec[key] if key in spec and spec.get(key) is not None else default


def _required_spec_value(spec: Mapping[str, Any], key: str, *, index: int) -> Any:
    value = spec.get(key)
    if value is None:
        raise ExperimentQueueError(f"run_specs[{index}].{key} is required")
    return value


def _spec_path_list(spec: Mapping[str, Any], key: str, default: Any) -> list[Any]:
    value = _spec_value(spec, key, default)
    if not isinstance(value, list) or not value:
        raise ExperimentQueueError(f"run spec {key} must be a non-empty list")
    return list(value)


def build_mlx_learned_sweep_autopilot_batch_queue(
    run_specs: Sequence[Mapping[str, Any]],
    *,
    queue_id: str,
    repo_root: str | Path,
    lane_id: str = "mlx_dynamic_learned_sweep_local_autopilot",
    local_cpu_concurrency: int = 1,
    local_mlx_concurrency: int = 1,
    timeout_seconds: int = 0,
    max_iterations: int = 1,
    max_new_observations: int = 1,
    rows_per_replan: int = 1,
    chain_steps: int = 1,
    sweep_config_id: str = SUPPORTED_SWEEP_CONFIG_ID,
    optimization_pass_id: str | None = None,
    candidate_ids: Sequence[str] | None = None,
    queue_candidate_ids: Sequence[str] | None = None,
    source_artifact_root: str | Path | None = None,
    device: str = "gpu",
    allow_gpu_research_signal: bool = True,
    batch_pairs: int = 1,
    progress_every: int = 0,
    max_seconds: float | None = None,
    replan_top_k: int | None = None,
    replan_per_pass_top_k: int | None = None,
    runtime_telemetry_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one queue with multiple independent MLX autopilot chain roots."""

    if not run_specs:
        raise ExperimentQueueError("at least one autopilot run spec is required")
    seen_run_ids: set[str] = set()
    experiments: list[dict[str, Any]] = []
    for index, raw_spec in enumerate(run_specs, start=1):
        if not isinstance(raw_spec, Mapping):
            raise ExperimentQueueError(f"run_specs[{index - 1}] must be an object")
        run_id = _optional_text(
            raw_spec.get("run_id"),
            label=f"run_specs[{index - 1}].run_id",
        ) or f"run_{index:04d}"
        safe_run_id = _safe_id(run_id)
        if safe_run_id in seen_run_ids:
            raise ExperimentQueueError(f"duplicate autopilot run_id: {run_id}")
        seen_run_ids.add(safe_run_id)
        sub_queue = build_mlx_learned_sweep_autopilot_queue(
            plan_path=_required_spec_value(raw_spec, "plan_path", index=index - 1),
            selection_path=_required_spec_value(
                raw_spec,
                "selection_path",
                index=index - 1,
            ),
            candidate_payload_paths=_spec_path_list(
                raw_spec,
                "candidate_payload_paths",
                None,
            ),
            incumbent_score=float(
                _required_spec_value(raw_spec, "incumbent_score", index=index - 1)
            ),
            output_root=_required_spec_value(raw_spec, "output_root", index=index - 1),
            observation_jsonl=_required_spec_value(
                raw_spec,
                "observation_jsonl",
                index=index - 1,
            ),
            queue_id=f"{queue_id}_{safe_run_id}",
            repo_root=repo_root,
            lane_id=str(_spec_value(raw_spec, "lane_id", lane_id)),
            local_cpu_concurrency=local_cpu_concurrency,
            local_mlx_concurrency=local_mlx_concurrency,
            timeout_seconds=int(_spec_value(raw_spec, "timeout_seconds", timeout_seconds)),
            max_iterations=int(_spec_value(raw_spec, "max_iterations", max_iterations)),
            max_new_observations=int(
                _spec_value(raw_spec, "max_new_observations", max_new_observations)
            ),
            rows_per_replan=int(_spec_value(raw_spec, "rows_per_replan", rows_per_replan)),
            chain_steps=int(_spec_value(raw_spec, "chain_steps", chain_steps)),
            sweep_config_id=str(
                _spec_value(raw_spec, "sweep_config_id", sweep_config_id)
            ),
            optimization_pass_id=_optional_text(
                _spec_value(raw_spec, "optimization_pass_id", optimization_pass_id),
                label=f"run_specs[{index - 1}].optimization_pass_id",
            ),
            candidate_ids=_optional_text_list(
                _spec_value(raw_spec, "candidate_ids", candidate_ids),
                label=f"run_specs[{index - 1}].candidate_ids",
            ),
            queue_candidate_ids=_optional_text_list(
                _spec_value(raw_spec, "queue_candidate_ids", queue_candidate_ids),
                label=f"run_specs[{index - 1}].queue_candidate_ids",
            ),
            source_artifact_root=_spec_value(
                raw_spec,
                "source_artifact_root",
                source_artifact_root,
            ),
            device=str(_spec_value(raw_spec, "device", device)),
            allow_gpu_research_signal=bool(
                _spec_value(
                    raw_spec,
                    "allow_gpu_research_signal",
                    allow_gpu_research_signal,
                )
            ),
            batch_pairs=int(_spec_value(raw_spec, "batch_pairs", batch_pairs)),
            progress_every=int(_spec_value(raw_spec, "progress_every", progress_every)),
            max_seconds=_spec_value(raw_spec, "max_seconds", max_seconds),
            replan_top_k=_spec_value(raw_spec, "replan_top_k", replan_top_k),
            replan_per_pass_top_k=_spec_value(
                raw_spec,
                "replan_per_pass_top_k",
                replan_per_pass_top_k,
            ),
            runtime_telemetry_policy=_spec_value(
                raw_spec,
                "runtime_telemetry_policy",
                runtime_telemetry_policy,
            ),
        )
        experiment = dict(sub_queue["experiments"][0])
        metadata = dict(experiment.get("metadata") or {})
        metadata.update(
            {
                "batch_schema": MLX_LEARNED_SWEEP_AUTOPILOT_BATCH_QUEUE_SCHEMA,
                "batch_parent_queue_id": queue_id,
                "batch_run_id": run_id,
                "batch_run_index": index,
                **FALSE_AUTHORITY,
            }
        )
        experiment["metadata"] = metadata
        experiment["priority"] = int(raw_spec.get("priority", 10 + index))
        experiments.append(experiment)

    return normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {
                    "local_cpu": _positive_int(
                        local_cpu_concurrency,
                        label="local_cpu_concurrency",
                    ),
                    "local_mlx": _positive_int(
                        local_mlx_concurrency,
                        label="local_mlx_concurrency",
                    ),
                },
            },
            "experiments": experiments,
        }
    )


__all__ = [
    "MLX_LEARNED_SWEEP_AUTOPILOT_BATCH_QUEUE_SCHEMA",
    "MLX_LEARNED_SWEEP_AUTOPILOT_QUEUE_SCHEMA",
    "MLX_RUNTIME_TELEMETRY_STATE_DISCOVERY_POLICY_SCHEMA",
    "TOOL_NAME",
    "build_mlx_learned_sweep_autopilot_batch_queue",
    "build_mlx_learned_sweep_autopilot_queue",
]
