#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile a bounded MLX learned-sweep autopilot run into experiment_queue work."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import ExperimentQueueError  # noqa: E402
from comma_lab.scheduler.mlx_learned_sweep_autopilot_queue import (  # noqa: E402
    MLX_RUNTIME_TELEMETRY_STATE_DISCOVERY_POLICY_SCHEMA,
    build_mlx_learned_sweep_autopilot_batch_queue,
    build_mlx_learned_sweep_autopilot_queue,
)
from tac.optimization.mlx_learned_sweep_batch_roots import (  # noqa: E402
    MLXLearnedSweepBatchRootError,
    build_mlx_learned_sweep_autopilot_batch_root_plan,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402

MLX_AUTOPILOT_STEP_PREFIX = "run_mlx_learned_sweep_autopilot"
MLX_AUTOPILOT_TOOL = "tools/run_mlx_dynamic_learned_sweep_autopilot.py"
MLX_LEARNED_SWEEP_QUEUE_ID_PREFIX = "mlx_learned_sweep_"
DEFAULT_SWEEP_CONFIG_ID = "mlx_local_response"
MACOS_CPU_ADVISORY_SWEEP_CONFIG_ID = "macos_cpu_advisory"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-spec",
        type=Path,
        help=(
            "JSON object with runs[]/run_specs[] for multiple independent "
            "autopilot chain roots"
        ),
    )
    parser.add_argument(
        "--auto-batch-from-plan",
        action="store_true",
        help=(
            "derive pass-level batch roots from --plan acquisition signal instead "
            "of reading a manual --batch-spec"
        ),
    )
    parser.add_argument(
        "--auto-batch-root-count",
        type=int,
        help=(
            "number of pass-level roots to select; defaults to "
            "--local-mlx-concurrency"
        ),
    )
    parser.add_argument(
        "--auto-batch-root-plan-output",
        type=Path,
        help="optional JSON artifact describing selected roots and run_specs",
    )
    parser.add_argument(
        "--auto-batch-run-prefix",
        default="pass",
        help="run_id prefix for automatically selected pass-level roots",
    )
    parser.add_argument(
        "--auto-batch-rows-per-root",
        type=int,
        help=(
            "fixed rows per selected auto-batch root; in adaptive mode this is "
            "the per-root cap"
        ),
    )
    parser.add_argument(
        "--auto-batch-adaptive-rows-per-root",
        action="store_true",
        help=(
            "water-fill positive-utility ready rows across selected roots, "
            "balancing cost_units instead of using fixed sorted chunks"
        ),
    )
    parser.add_argument(
        "--auto-batch-runtime-telemetry",
        action="append",
        default=[],
        type=Path,
        help=(
            "JSON runtime telemetry artifact for adaptive auto-batch roots. "
            "May repeat. Supports local MLX autopilot summaries and "
            "experiment_queue worker results; advisory only."
        ),
    )
    parser.add_argument(
        "--auto-batch-runtime-telemetry-state",
        action="append",
        default=[],
        type=Path,
        help=(
            "SQLite experiment_queue state to mine for completed step runtime "
            "telemetry. May repeat. Converts step_succeeded/step_failed events "
            "into advisory experiment_queue_worker_result.v1 payloads."
        ),
    )
    parser.add_argument(
        "--auto-batch-discover-runtime-telemetry-states",
        action="store_true",
        help=(
            "automatically mine compatible prior MLX learned-sweep "
            "experiment_queue SQLite states from --auto-batch-runtime-telemetry-state-dir"
        ),
    )
    parser.add_argument(
        "--auto-batch-runtime-telemetry-state-dir",
        default=Path(".omx/state"),
        type=Path,
        help="directory searched when auto-discovering runtime telemetry states",
    )
    parser.add_argument(
        "--auto-batch-runtime-telemetry-state-pattern",
        action="append",
        default=[],
        help=(
            "glob pattern used inside --auto-batch-runtime-telemetry-state-dir "
            "during discovery; defaults to experiment_queue_mlx_learned_sweep*.sqlite"
        ),
    )
    parser.add_argument(
        "--auto-batch-runtime-telemetry-state-limit",
        default=3,
        type=int,
        help="maximum discovered compatible telemetry states to use; 0 means no limit",
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--candidate-payload", action="append", default=[], type=Path)
    parser.add_argument("--incumbent-score", type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument(
        "--lane-id",
        default="mlx_dynamic_learned_sweep_local_autopilot",
    )
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--observation-jsonl", type=Path)
    parser.add_argument("--local-cpu-concurrency", default=1, type=int)
    parser.add_argument("--local-mlx-concurrency", default=1, type=int)
    parser.add_argument("--timeout-seconds", default=0, type=int)
    parser.add_argument("--max-iterations", default=1, type=int)
    parser.add_argument("--max-new-observations", default=1, type=int)
    parser.add_argument("--rows-per-replan", default=1, type=int)
    parser.add_argument(
        "--sweep-config-id",
        default=DEFAULT_SWEEP_CONFIG_ID,
        help=(
            "learned-sweep config to execute; defaults to mlx_local_response. "
            "macos_cpu_advisory consumes explicit local advisory artifacts."
        ),
    )
    parser.add_argument(
        "--chain-steps",
        default=1,
        type=int,
        help=(
            "emit dependent one-cycle autopilot queue steps; values >1 require "
            "--max-iterations 1"
        ),
    )
    parser.add_argument("--optimization-pass-id")
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="Restrict execution to candidate_id. May repeat.",
    )
    parser.add_argument(
        "--queue-candidate-id",
        action="append",
        default=[],
        help="Restrict execution to queue_candidate_id. May repeat.",
    )
    parser.add_argument("--source-artifact-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--allow-gpu-research-signal", action="store_true")
    parser.add_argument("--batch-pairs", default=1, type=int)
    parser.add_argument("--progress-every", default=0, type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--replan-top-k", type=int)
    parser.add_argument("--replan-per-pass-top-k", type=int)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256")
    return parser.parse_args(argv)


def _load_batch_specs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        runs = payload
    elif isinstance(payload, dict):
        runs = payload.get("runs", payload.get("run_specs"))
    else:
        raise ExperimentQueueError("--batch-spec must be a JSON object or list")
    if not isinstance(runs, list) or not runs:
        raise ExperimentQueueError("--batch-spec requires a non-empty runs[] list")
    if not all(isinstance(run, dict) for run in runs):
        raise ExperimentQueueError("--batch-spec runs[] must contain objects")
    return runs


def _require_single_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name, value in (
            ("--plan", args.plan),
            ("--selection", args.selection),
            ("--candidate-payload", args.candidate_payload),
            ("--incumbent-score", args.incumbent_score),
            ("--output-root", args.output_root),
            ("--observation-jsonl", args.observation_jsonl),
        )
        if value is None or value == []
    ]
    if missing:
        raise ExperimentQueueError(
            "single-run queue build requires " + ", ".join(missing)
        )


def _resolve_cli_path(path: Path, *, repo_root: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else repo_root / path


def _load_plan_payload(path: Path, *, repo_root: Path) -> dict:
    resolved = _resolve_cli_path(path, repo_root=repo_root)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExperimentQueueError("--plan must be a JSON object")
    return payload


def _ready_queue_candidate_ids_from_plan(
    plan: dict,
    *,
    sweep_config_id: str,
) -> set[str]:
    rows = plan.get("ranked_sweep_rows")
    if not isinstance(rows, list):
        return set()
    out: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("schema") != "mlx_dynamic_learned_sweep_row.v1":
            continue
        if row.get("ready_for_local_sweep") is not True:
            continue
        if row.get("sweep_config_id") != sweep_config_id:
            continue
        queue_candidate_id = str(row.get("queue_candidate_id") or "").strip()
        if queue_candidate_id:
            out.add(queue_candidate_id)
    return out


def _load_runtime_telemetry_payloads(
    paths: list[Path],
    *,
    repo_root: Path,
) -> list[dict]:
    payloads: list[dict] = []
    for raw_path in paths:
        resolved = _resolve_cli_path(raw_path, repo_root=repo_root)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ExperimentQueueError(
                "--auto-batch-runtime-telemetry must point to JSON objects"
            )
        payloads.append(payload)
    return payloads


def _load_runtime_telemetry_payload_from_state(
    path: Path,
    *,
    repo_root: Path,
) -> dict:
    resolved = _resolve_cli_path(path, repo_root=repo_root)
    if not resolved.exists():
        raise ExperimentQueueError(
            f"--auto-batch-runtime-telemetry-state not found: {resolved}"
        )
    step_results: list[dict] = []
    try:
        with sqlite3.connect(resolved) as conn:
            conn.row_factory = sqlite3.Row
            resource_kind_by_step: dict[tuple[str, str, str], str] = {}
            table_names = {
                str(row["name"])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if "step_state" in table_names:
                step_columns = {
                    str(row["name"])
                    for row in conn.execute("PRAGMA table_info(step_state)").fetchall()
                }
                if "resource_kind" in step_columns:
                    for state_row in conn.execute(
                        """
                        SELECT queue_id, experiment_id, step_id, resource_kind
                        FROM step_state
                        WHERE resource_kind IS NOT NULL
                        """
                    ).fetchall():
                        resource_kind_by_step[
                            (
                                str(state_row["queue_id"]),
                                str(state_row["experiment_id"]),
                                str(state_row["step_id"]),
                            )
                        ] = str(state_row["resource_kind"])
            rows = conn.execute(
                """
                SELECT queue_id, experiment_id, step_id, event_type, payload_json
                FROM queue_events
                WHERE event_type IN ('step_succeeded', 'step_failed')
                ORDER BY id
                """
            ).fetchall()
    except sqlite3.Error as exc:
        raise ExperimentQueueError(
            f"{resolved}: invalid experiment_queue state: {exc}"
        ) from exc

    queue_ids: set[str] = set()
    for index, row in enumerate(rows):
        queue_ids.add(str(row["queue_id"]))
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError as exc:
            raise ExperimentQueueError(
                f"{resolved}: queue_events[{index}] has invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise ExperimentQueueError(
                f"{resolved}: queue_events[{index}] payload must be a JSON object"
            )
        payload.setdefault("succeeded", row["event_type"] == "step_succeeded")
        payload.setdefault("queue_id", row["queue_id"])
        payload.setdefault("experiment_id", row["experiment_id"])
        payload.setdefault("step_id", row["step_id"])
        if not payload.get("resource_kind"):
            resource_kind = resource_kind_by_step.get(
                (
                    str(row["queue_id"]),
                    str(row["experiment_id"]),
                    str(row["step_id"]),
                )
            )
            if resource_kind:
                payload["resource_kind"] = resource_kind
        step_results.append(payload)

    return {
        "schema": "experiment_queue_worker_result.v1",
        "source_kind": "experiment_queue_sqlite_state",
        "source_state_path": str(resolved),
        "source_queue_ids": sorted(queue_ids),
        "step_results": step_results,
    }


def _load_runtime_telemetry_payloads_from_states(
    paths: list[Path],
    *,
    repo_root: Path,
) -> list[dict]:
    return [
        _load_runtime_telemetry_payload_from_state(path, repo_root=repo_root)
        for path in paths
    ]


def _command_queue_candidate_ids(command: object) -> set[str]:
    if not isinstance(command, list):
        return set()
    out: set[str] = set()
    index = 0
    while index < len(command):
        if command[index] == "--queue-candidate-id" and index + 1 < len(command):
            queue_candidate_id = str(command[index + 1]).strip()
            if queue_candidate_id:
                out.add(queue_candidate_id)
            index += 2
            continue
        index += 1
    return out


def _command_invokes_mlx_autopilot_tool(command: object) -> bool:
    return isinstance(command, list) and MLX_AUTOPILOT_TOOL in {
        str(item) for item in command
    }


def _resource_kind_for_sweep_config(sweep_config_id: str) -> str:
    return (
        "local_cpu"
        if sweep_config_id == MACOS_CPU_ADVISORY_SWEEP_CONFIG_ID
        else "local_mlx"
    )


def _state_payload_discovery_eligible(
    payload: dict,
    *,
    resource_kind: str,
) -> bool:
    if payload.get("schema") != "experiment_queue_worker_result.v1":
        return False
    source_queue_ids = payload.get("source_queue_ids")
    if not isinstance(source_queue_ids, list):
        return False
    if not any(
        str(queue_id).startswith(MLX_LEARNED_SWEEP_QUEUE_ID_PREFIX)
        for queue_id in source_queue_ids
    ):
        return False
    steps = payload.get("step_results")
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("succeeded") is not True:
            continue
        if step.get("resource_kind") != resource_kind:
            continue
        if not str(step.get("step_id") or "").startswith(MLX_AUTOPILOT_STEP_PREFIX):
            continue
        if not _command_invokes_mlx_autopilot_tool(step.get("command")):
            continue
        telemetry = step.get("telemetry")
        if not isinstance(telemetry, dict):
            continue
        if telemetry.get("schema") != "experiment_queue_step_telemetry.v1":
            continue
        if int(telemetry.get("artifact_record_count") or 0) <= 0:
            continue
        return True
    return False


def _payload_compatible_queue_candidate_ids(
    payload: dict,
    *,
    ready_queue_candidate_ids: set[str],
    resource_kind: str,
) -> set[str]:
    if payload.get("schema") != "experiment_queue_worker_result.v1":
        return set()
    steps = payload.get("step_results")
    if not isinstance(steps, list):
        return set()
    out: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("resource_kind") != resource_kind:
            continue
        if not str(step.get("step_id") or "").startswith(MLX_AUTOPILOT_STEP_PREFIX):
            continue
        if not _command_invokes_mlx_autopilot_tool(step.get("command")):
            continue
        if step.get("succeeded") is not True:
            continue
        if step.get("timed_out") is True:
            continue
        if step.get("failed_postconditions"):
            continue
        if step.get("postcondition_errors"):
            continue
        out.update(
            _command_queue_candidate_ids(step.get("command"))
            & ready_queue_candidate_ids
        )
    return out


def _unique_paths(paths: list[Path], *, repo_root: Path) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = _resolve_cli_path(path, repo_root=repo_root).resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def _runtime_state_policy_payload(
    *,
    args: argparse.Namespace,
    runtime_state_paths: list[Path],
    discovered_runtime_state_paths: list[Path],
    resource_kind: str,
) -> dict | None:
    if not runtime_state_paths and not args.auto_batch_discover_runtime_telemetry_states:
        return None
    return {
        "schema": MLX_RUNTIME_TELEMETRY_STATE_DISCOVERY_POLICY_SCHEMA,
        "policy_id": "mlx_learned_sweep_runtime_telemetry_state_discovery",
        "mode": (
            "auto_discover_compatible_states"
            if args.auto_batch_discover_runtime_telemetry_states
            else "explicit_states"
        ),
        "state_dir": str(args.auto_batch_runtime_telemetry_state_dir),
        "state_patterns": (
            args.auto_batch_runtime_telemetry_state_pattern
            or ["experiment_queue_mlx_learned_sweep*.sqlite"]
        ),
        "state_limit": int(args.auto_batch_runtime_telemetry_state_limit),
        "selected_state_paths": [str(path) for path in runtime_state_paths],
        "discovered_state_paths": [
            str(path) for path in discovered_runtime_state_paths
        ],
        "compatibility_filter": {
            "schema": "mlx_runtime_telemetry_state_compatibility_filter.v1",
            "ready_queue_candidate_source": (
                f"current_plan_ready_{args.sweep_config_id}_rows"
            ),
            "accepted_step_event_types": ["step_succeeded"],
            "requires_succeeded": True,
            "requires_source_queue_id_prefix": MLX_LEARNED_SWEEP_QUEUE_ID_PREFIX,
            "requires_step_id_prefix": MLX_AUTOPILOT_STEP_PREFIX,
            "requires_command_tool": MLX_AUTOPILOT_TOOL,
            "requires_resource_kind": resource_kind,
            "requires_artifact_telemetry_schema": "experiment_queue_step_telemetry.v1",
            "requires_positive_artifact_record_count": True,
            "rejects_timed_out": True,
            "rejects_failed_postconditions": True,
            "rejects_postcondition_errors": True,
        },
        "allowed_use": "local_runtime_balanced_batch_planning_only",
        "sweep_config_id": args.sweep_config_id,
        "candidate_generation_only": True,
        "observation_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }


def _discover_runtime_telemetry_state_paths(
    *,
    repo_root: Path,
    state_dir: Path,
    patterns: list[str],
    limit: int,
    plan_payload: dict,
    sweep_config_id: str,
) -> list[Path]:
    if limit < 0:
        raise ExperimentQueueError(
            "--auto-batch-runtime-telemetry-state-limit must be >= 0"
        )
    ready_queue_candidate_ids = _ready_queue_candidate_ids_from_plan(
        plan_payload,
        sweep_config_id=sweep_config_id,
    )
    if not ready_queue_candidate_ids:
        return []
    resolved_dir = _resolve_cli_path(state_dir, repo_root=repo_root)
    if not resolved_dir.is_dir():
        raise ExperimentQueueError(
            "--auto-batch-runtime-telemetry-state-dir not found: "
            f"{resolved_dir}"
        )
    active_patterns = patterns or ["experiment_queue_mlx_learned_sweep*.sqlite"]
    candidate_paths: list[Path] = []
    for pattern in active_patterns:
        if not pattern.strip():
            raise ExperimentQueueError(
                "--auto-batch-runtime-telemetry-state-pattern must be non-empty"
            )
        candidate_paths.extend(resolved_dir.glob(pattern))
    selected: list[tuple[float, str, Path]] = []
    for path in sorted(set(candidate_paths)):
        if not path.is_file() or path.is_symlink():
            continue
        payload = _load_runtime_telemetry_payload_from_state(path, repo_root=repo_root)
        resource_kind = _resource_kind_for_sweep_config(sweep_config_id)
        if not _state_payload_discovery_eligible(
            payload,
            resource_kind=resource_kind,
        ):
            continue
        compatible = _payload_compatible_queue_candidate_ids(
            payload,
            ready_queue_candidate_ids=ready_queue_candidate_ids,
            resource_kind=resource_kind,
        )
        if not compatible:
            continue
        stat = path.stat()
        selected.append((stat.st_mtime, str(path), path))
    selected.sort(key=lambda item: (-item[0], item[1]))
    paths = [path for _mtime, _text, path in selected]
    return paths if limit == 0 else paths[:limit]


def _require_auto_batch_args(args: argparse.Namespace) -> None:
    if args.batch_spec is not None:
        raise ExperimentQueueError(
            "--auto-batch-from-plan cannot be combined with --batch-spec"
        )
    _require_single_args(args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root_plan: dict | None = None
    discovered_runtime_state_paths: list[Path] = []
    try:
        if (
            args.auto_batch_discover_runtime_telemetry_states
            and not args.auto_batch_from_plan
        ):
            raise ExperimentQueueError(
                "--auto-batch-discover-runtime-telemetry-states requires "
                "--auto-batch-from-plan"
            )
        if args.auto_batch_from_plan:
            _require_auto_batch_args(args)
            plan_payload = _load_plan_payload(args.plan, repo_root=args.repo_root)
            resource_kind = _resource_kind_for_sweep_config(args.sweep_config_id)
            if args.auto_batch_discover_runtime_telemetry_states:
                discovered_runtime_state_paths = _discover_runtime_telemetry_state_paths(
                    repo_root=args.repo_root,
                    state_dir=args.auto_batch_runtime_telemetry_state_dir,
                    patterns=args.auto_batch_runtime_telemetry_state_pattern,
                    limit=args.auto_batch_runtime_telemetry_state_limit,
                    plan_payload=plan_payload,
                    sweep_config_id=args.sweep_config_id,
                )
            runtime_state_paths = _unique_paths(
                list(args.auto_batch_runtime_telemetry_state)
                + discovered_runtime_state_paths,
                repo_root=args.repo_root,
            )
            runtime_telemetry_policy = _runtime_state_policy_payload(
                args=args,
                runtime_state_paths=runtime_state_paths,
                discovered_runtime_state_paths=discovered_runtime_state_paths,
                resource_kind=resource_kind,
            )
            root_plan = build_mlx_learned_sweep_autopilot_batch_root_plan(
                plan_payload,
                plan_path=args.plan,
                selection_path=args.selection,
                candidate_payload_paths=args.candidate_payload,
                incumbent_score=args.incumbent_score,
                output_root=args.output_root,
                observation_jsonl=args.observation_jsonl,
                root_count=(
                    args.auto_batch_root_count
                    if args.auto_batch_root_count is not None
                    else (
                        args.local_cpu_concurrency
                        if args.sweep_config_id == MACOS_CPU_ADVISORY_SWEEP_CONFIG_ID
                        else args.local_mlx_concurrency
                    )
                ),
                rows_per_root=(
                    args.auto_batch_rows_per_root
                    if args.auto_batch_rows_per_root is not None
                    else args.max_new_observations
                ),
                adaptive_rows_per_root=args.auto_batch_adaptive_rows_per_root,
                run_prefix=args.auto_batch_run_prefix,
                sweep_config_id=args.sweep_config_id,
                max_new_observations=args.max_new_observations,
                rows_per_replan=args.rows_per_replan,
                chain_steps=args.chain_steps,
                device=args.device,
                allow_gpu_research_signal=args.allow_gpu_research_signal,
                source_artifact_root=args.source_artifact_root,
                batch_pairs=args.batch_pairs,
                runtime_telemetry_payloads=_load_runtime_telemetry_payloads(
                    args.auto_batch_runtime_telemetry,
                    repo_root=args.repo_root,
                )
                + _load_runtime_telemetry_payloads_from_states(
                    runtime_state_paths,
                    repo_root=args.repo_root,
                ),
            )
            if runtime_telemetry_policy is not None:
                root_plan["runtime_telemetry_state_policy"] = runtime_telemetry_policy
            if args.auto_batch_root_plan_output is not None:
                write_json_artifact(
                    args.auto_batch_root_plan_output,
                    root_plan,
                    allow_overwrite=args.allow_overwrite,
                )
            queue = build_mlx_learned_sweep_autopilot_batch_queue(
                [
                    {
                        **run_spec,
                        "runtime_telemetry_policy": runtime_telemetry_policy,
                    }
                    for run_spec in root_plan["run_specs"]
                ],
                queue_id=args.queue_id,
                repo_root=args.repo_root,
                lane_id=args.lane_id,
                local_cpu_concurrency=args.local_cpu_concurrency,
                local_mlx_concurrency=args.local_mlx_concurrency,
                timeout_seconds=args.timeout_seconds,
                max_iterations=args.max_iterations,
                max_new_observations=args.max_new_observations,
                rows_per_replan=args.rows_per_replan,
                chain_steps=args.chain_steps,
                sweep_config_id=args.sweep_config_id,
                source_artifact_root=args.source_artifact_root,
                device=args.device,
                allow_gpu_research_signal=args.allow_gpu_research_signal,
                batch_pairs=args.batch_pairs,
                progress_every=args.progress_every,
                max_seconds=args.max_seconds,
                replan_top_k=args.replan_top_k,
                replan_per_pass_top_k=args.replan_per_pass_top_k,
            )
        elif args.batch_spec is not None:
            queue = build_mlx_learned_sweep_autopilot_batch_queue(
                _load_batch_specs(args.batch_spec),
                queue_id=args.queue_id,
                repo_root=args.repo_root,
                lane_id=args.lane_id,
                local_cpu_concurrency=args.local_cpu_concurrency,
                local_mlx_concurrency=args.local_mlx_concurrency,
                timeout_seconds=args.timeout_seconds,
                max_iterations=args.max_iterations,
                max_new_observations=args.max_new_observations,
                rows_per_replan=args.rows_per_replan,
                chain_steps=args.chain_steps,
                sweep_config_id=args.sweep_config_id,
                optimization_pass_id=args.optimization_pass_id,
                candidate_ids=args.candidate_id or None,
                queue_candidate_ids=args.queue_candidate_id or None,
                source_artifact_root=args.source_artifact_root,
                device=args.device,
                allow_gpu_research_signal=args.allow_gpu_research_signal,
                batch_pairs=args.batch_pairs,
                progress_every=args.progress_every,
                max_seconds=args.max_seconds,
                replan_top_k=args.replan_top_k,
                replan_per_pass_top_k=args.replan_per_pass_top_k,
            )
        else:
            _require_single_args(args)
            queue = build_mlx_learned_sweep_autopilot_queue(
                plan_path=args.plan,
                selection_path=args.selection,
                candidate_payload_paths=args.candidate_payload,
                incumbent_score=args.incumbent_score,
                output_root=args.output_root,
                observation_jsonl=args.observation_jsonl,
                queue_id=args.queue_id,
                repo_root=args.repo_root,
                lane_id=args.lane_id,
                local_cpu_concurrency=args.local_cpu_concurrency,
                local_mlx_concurrency=args.local_mlx_concurrency,
                timeout_seconds=args.timeout_seconds,
                max_iterations=args.max_iterations,
                max_new_observations=args.max_new_observations,
                rows_per_replan=args.rows_per_replan,
                chain_steps=args.chain_steps,
                sweep_config_id=args.sweep_config_id,
                optimization_pass_id=args.optimization_pass_id,
                candidate_ids=args.candidate_id or None,
                queue_candidate_ids=args.queue_candidate_id or None,
                source_artifact_root=args.source_artifact_root,
                device=args.device,
                allow_gpu_research_signal=args.allow_gpu_research_signal,
                batch_pairs=args.batch_pairs,
                progress_every=args.progress_every,
                max_seconds=args.max_seconds,
                replan_top_k=args.replan_top_k,
                replan_per_pass_top_k=args.replan_per_pass_top_k,
            )
        write_json_artifact(
            args.output,
            queue,
            allow_overwrite=args.allow_overwrite,
            expected_existing_sha256=args.expected_output_sha256,
        )
    except (
        ArtifactWriteError,
        ExperimentQueueError,
        MLXLearnedSweepBatchRootError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(
        json.dumps(
            {
                "output": str(args.output),
                "queue_id": queue["queue_id"],
                "experiment_count": len(queue["experiments"]),
                "step_count": sum(
                    len(experiment["steps"]) for experiment in queue["experiments"]
                ),
                "score_claim": False,
                "score_claim_valid": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
                "auto_batch_root_plan_output": (
                    str(args.auto_batch_root_plan_output)
                    if args.auto_batch_root_plan_output is not None
                    else None
                ),
                "auto_batch_selected_root_count": (
                    root_plan.get("selected_root_count")
                    if isinstance(root_plan, dict)
                    else None
                ),
                "auto_batch_runtime_telemetry_used": (
                    root_plan.get("runtime_telemetry_used")
                    if isinstance(root_plan, dict)
                    else None
                ),
                "auto_batch_runtime_telemetry_key_count": (
                    root_plan.get("runtime_telemetry_key_count")
                    if isinstance(root_plan, dict)
                    else None
                ),
                "auto_batch_runtime_cost_policy": (
                    root_plan.get("runtime_cost_policy")
                    if isinstance(root_plan, dict)
                    else None
                ),
                "auto_batch_runtime_telemetry_state_count": (
                    len(
                        _unique_paths(
                            list(args.auto_batch_runtime_telemetry_state)
                            + discovered_runtime_state_paths,
                            repo_root=args.repo_root,
                        )
                    )
                    if isinstance(root_plan, dict)
                    else None
                ),
                "auto_batch_runtime_telemetry_state_discovery_enabled": (
                    bool(args.auto_batch_discover_runtime_telemetry_states)
                    if isinstance(root_plan, dict)
                    else None
                ),
                "auto_batch_runtime_telemetry_state_discovered_count": (
                    len(discovered_runtime_state_paths)
                    if isinstance(root_plan, dict)
                    else None
                ),
                "auto_batch_runtime_telemetry_state_discovered_paths": (
                    [str(path) for path in discovered_runtime_state_paths]
                    if isinstance(root_plan, dict)
                    else None
                ),
                "auto_batch_runtime_telemetry_policy_schema": (
                    root_plan.get("runtime_telemetry_state_policy", {}).get("schema")
                    if isinstance(root_plan, dict)
                    else None
                ),
                "auto_batch_runtime_telemetry_policy_id": (
                    root_plan.get("runtime_telemetry_state_policy", {}).get(
                        "policy_id"
                    )
                    if isinstance(root_plan, dict)
                    else None
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
