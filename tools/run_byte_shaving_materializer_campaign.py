#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build and run a byte-shaving materializer campaign queue.

This is the one-command control surface for the local materializer loop:
compile the campaign, emit executable materializer rows, append per-row
harvest/exact-readiness/paused-dispatch-plan follow-ups, initialize the
canonical experiment queue state, run a bounded worker, and write live
observation/performance artifacts. It never performs paid dispatch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import default_state_path, load_queue_definition  # noqa: E402
from comma_lab.scheduler.staircase_dag import (  # noqa: E402
    build_staircase_dag_from_experiment_queue,
    experiment_queue_status_map,
    parse_resource_pool_spec,
    plan_staircase_dispatch,
)
from comma_lab.scheduler.storage_preflight import (  # noqa: E402
    validate_scheduler_storage_preflight_config,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402

RUN_SCHEMA = "byte_shaving_materializer_campaign_run.v1"


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _display_path(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve(strict=False).relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _run(command: list[str], *, check: bool = True) -> CommandResult:
    started = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    result = CommandResult(
        command=command,
        returncode=int(proc.returncode),
        stdout=proc.stdout,
        stderr=proc.stderr,
        elapsed_seconds=time.monotonic() - started,
    )
    if check and proc.returncode != 0:
        raise SystemExit(
            f"command failed ({proc.returncode}): {' '.join(command)}\n{proc.stderr}"
        )
    return result


def _json_from_stdout(result: CommandResult) -> dict[str, Any] | None:
    text = result.stdout.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _require_json_stdout(
    result: CommandResult,
    *,
    label: str,
    allow_nonzero: bool = False,
) -> dict[str, Any]:
    if result.returncode != 0 and not allow_nonzero:
        raise SystemExit(
            f"{label} failed ({result.returncode}): {result.stderr or result.stdout}"
        )
    payload = _json_from_stdout(result)
    if payload is None:
        raise SystemExit(f"{label} did not emit a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        write_json_artifact(path, payload)
    except ArtifactWriteError as exc:
        raise SystemExit(str(exc)) from exc


def _parse_resource_concurrency(values: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        if "=" not in raw:
            raise SystemExit("--materializer-resource-concurrency entries must be KIND=LIMIT")
        key, value = raw.split("=", 1)
        if not key.strip():
            raise SystemExit("--materializer-resource-concurrency KIND must be non-empty")
        try:
            limit = int(value)
        except ValueError as exc:
            raise SystemExit(
                f"--materializer-resource-concurrency has non-integer limit: {raw!r}"
            ) from exc
        if limit < 1:
            raise SystemExit(
                f"--materializer-resource-concurrency limit must be >= 1: {raw!r}"
            )
        out.extend(["--materializer-resource-concurrency", f"{key.strip()}={limit}"])
    return out


def _parse_remote_repo_roots(values: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        machine, sep, root = raw.partition("=")
        if not sep or not machine.strip() or not root.strip():
            raise SystemExit("--staircase-ssh-remote-repo-root entries must be MACHINE_ID=PATH")
        out.extend(["--remote-repo-root", f"{machine.strip()}={root.strip()}"])
    return out


def _parse_artifact_path_maps(values: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        local, sep, remote = raw.partition("=")
        if not sep or not local.strip() or not remote.strip():
            raise SystemExit("--staircase-ssh-artifact-path-map entries must be LOCAL_PREFIX=REMOTE_PREFIX")
        out.extend(["--artifact-path-map", f"{local.strip()}={remote.strip()}"])
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--materializer-contexts", type=Path, default=None)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="artifact directory; defaults to .omx/research/byte_shaving_materializer_campaign_<UTC>",
    )
    parser.add_argument("--queue-id", default="byte_shaving_materializer_local_proof_chain")
    parser.add_argument("--lane-id", default=None)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--materializer-execution-limit", type=int, default=None)
    parser.add_argument("--materializer-execution-timeout-seconds", type=int, default=0)
    parser.add_argument(
        "--materializer-resource-concurrency",
        action="append",
        default=[],
        metavar="KIND=LIMIT",
    )
    parser.add_argument("--local-cpu-concurrency", default="auto")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--idle-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-idle-cycles", type=int, default=1)
    parser.add_argument("--tail-lines", type=int, default=20)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--include-storage-preflight", action="store_true")
    parser.add_argument("--results-root", default="experiments/results")
    parser.add_argument("--storage-tier", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--storage-workload-subdir", default=None)
    parser.add_argument("--storage-expected-workload-root", default=None)
    parser.add_argument("--storage-reserve-free-gb", type=float, default=40.0)
    parser.add_argument("--storage-expected-bytes", type=int, default=0)
    parser.add_argument("--proactive-cleanup-root", action="append", default=[])
    parser.add_argument("--proactive-cleanup-action", choices=("move", "delete"), default="move")
    parser.add_argument("--proactive-cleanup-min-bytes", default="1")
    parser.add_argument("--proactive-cleanup-cold-store-root", action="append", default=[])
    parser.add_argument("--proactive-cleanup-cold-store-reserve-gb", type=float, default=40.0)
    parser.add_argument("--exact-readiness-require-ready", action="store_true")
    parser.add_argument("--exact-eval-dispatch-require-authorized", action="store_true")
    parser.add_argument("--exact-eval-dispatch-provider", choices=("lightning", "vastai"), default="lightning")
    parser.add_argument("--exact-eval-dispatch-label-prefix", default="materializer_exact_eval")
    parser.add_argument("--exact-eval-dispatch-estimated-cost-per-dispatch", type=float, default=0.30)
    parser.add_argument("--exact-eval-dispatch-max-total-cost", type=float, default=5.00)
    parser.add_argument(
        "--emit-staircase-plan",
        action="store_true",
        help="emit staircase_dag.v1 and staircase_dispatch_plan.v1 from the generated queue",
    )
    parser.add_argument(
        "--staircase-resource-pool",
        action="append",
        default=[],
        metavar="SPEC",
        help="resource pool spec accepted by tools/plan_staircase_dag.py",
    )
    parser.add_argument("--staircase-max-nodes", type=int, default=None)
    parser.add_argument("--staircase-allow-cloud", action="store_true")
    parser.add_argument("--staircase-diversity-bucket-limit", type=int, default=None)
    parser.add_argument(
        "--staircase-ssh-dry-run",
        action="store_true",
        help="run tools/run_staircase_ssh_executor.py without --execute against the emitted plan",
    )
    parser.add_argument(
        "--staircase-ssh-execute",
        action="store_true",
        help="run tools/run_staircase_ssh_executor.py --execute against the emitted plan",
    )
    parser.add_argument("--staircase-ssh-max-steps", type=int, default=1)
    parser.add_argument("--staircase-ssh-machine-id", default=None)
    parser.add_argument(
        "--staircase-ssh-remote-repo-root",
        action="append",
        default=[],
        metavar="MACHINE=PATH",
    )
    parser.add_argument(
        "--staircase-ssh-require-artifact-mobility",
        action="store_true",
        help="require SSH artifact pullback/shared-storage visibility",
    )
    parser.add_argument(
        "--staircase-ssh-artifact-path-map",
        action="append",
        default=[],
        metavar="LOCAL_PREFIX=REMOTE_PREFIX",
    )
    parser.add_argument("--staircase-ssh-artifact-shared-path-rationale", default=None)
    parser.add_argument("--staircase-ssh-allow-dirty-remote-git", action="store_true")
    parser.add_argument("--staircase-ssh-dirty-remote-git-rationale", default=None)
    parser.add_argument("--staircase-ssh-rsync-binary", default="rsync")
    parser.add_argument("--staircase-ssh-artifact-pull-timeout-seconds", type=int, default=300)
    parser.add_argument("--staircase-ssh-allow-future-executor", action="store_true")
    return parser.parse_args(argv)


def _build_queue_command(args: argparse.Namespace, *, run_dir: Path) -> list[str]:
    if args.include_storage_preflight:
        try:
            validate_scheduler_storage_preflight_config(
                proactive_cleanup_execute=True,
                proactive_cleanup_action=args.proactive_cleanup_action,
                proactive_cleanup_cold_store_roots=tuple(
                    args.proactive_cleanup_cold_store_root
                ),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    materialization = run_dir / "materialization.json"
    portfolio = run_dir / "portfolio.json"
    action_summary = run_dir / "action_summary.json"
    backlog = run_dir / "materializer_backlog.json"
    work_queue = run_dir / "materializer_work_queue.json"
    execution_queue = run_dir / "materializer_execution_queue.json"
    command = [
        sys.executable,
        "tools/build_byte_shaving_campaign_queue.py",
        "--repo-root",
        REPO_ROOT.as_posix(),
        "--plan",
        _display_path(_resolve(args.plan)),
        "--materialization-out",
        _display_path(materialization),
        "--portfolio-out",
        _display_path(portfolio),
        "--action-summary-out",
        _display_path(action_summary),
        "--materializer-backlog-out",
        _display_path(backlog),
        "--materializer-work-queue-out",
        _display_path(work_queue),
        "--materializer-execution-queue-out",
        _display_path(execution_queue),
        "--materializer-execution-queue-id",
        args.queue_id,
        "--candidate-limit",
        str(args.candidate_limit),
        "--local-cpu-concurrency",
        str(args.local_cpu_concurrency),
        "--results-root",
        str(args.results_root),
        "--include-materializer-exact-readiness-followup",
        "--materializer-exact-eval-dispatch-provider",
        args.exact_eval_dispatch_provider,
        "--materializer-exact-eval-dispatch-label-prefix",
        args.exact_eval_dispatch_label_prefix,
        "--materializer-exact-eval-dispatch-estimated-cost-per-dispatch",
        str(args.exact_eval_dispatch_estimated_cost_per_dispatch),
        "--materializer-exact-eval-dispatch-max-total-cost",
        str(args.exact_eval_dispatch_max_total_cost),
    ]
    if args.materializer_contexts is not None:
        command.extend(["--materializer-contexts", _display_path(_resolve(args.materializer_contexts))])
    if args.lane_id:
        command.extend(["--materializer-execution-lane-id", str(args.lane_id)])
    if args.materializer_execution_limit is not None:
        command.extend(["--materializer-execution-limit", str(args.materializer_execution_limit)])
    if args.materializer_execution_timeout_seconds:
        command.extend([
            "--materializer-execution-timeout-seconds",
            str(args.materializer_execution_timeout_seconds),
        ])
    command.extend(_parse_resource_concurrency(args.materializer_resource_concurrency))
    if args.overwrite_output:
        command.append("--overwrite-output")
    if args.exact_readiness_require_ready:
        command.append("--materializer-exact-readiness-followup-require-ready")
    if args.exact_eval_dispatch_require_authorized:
        command.append("--materializer-exact-eval-dispatch-require-authorized")
    if args.include_storage_preflight:
        command.append("--include-materializer-scheduler-preflight")
        command.extend(["--materializer-scheduler-storage-reserve-free-gb", str(args.storage_reserve_free_gb)])
        command.extend(["--materializer-scheduler-storage-expected-bytes", str(args.storage_expected_bytes)])
        command.extend(["--materializer-scheduler-proactive-cleanup-action", args.proactive_cleanup_action])
        command.extend(["--materializer-scheduler-proactive-cleanup-min-bytes", str(args.proactive_cleanup_min_bytes)])
        command.extend([
            "--materializer-scheduler-proactive-cleanup-cold-store-reserve-gb",
            str(args.proactive_cleanup_cold_store_reserve_gb),
        ])
        command.append("--materializer-scheduler-proactive-cleanup-execute")
        if args.storage_workload_subdir:
            command.extend(["--materializer-scheduler-storage-workload-subdir", args.storage_workload_subdir])
        if args.storage_expected_workload_root:
            command.extend([
                "--materializer-scheduler-storage-expected-workload-root",
                args.storage_expected_workload_root,
            ])
        for tier in args.storage_tier:
            command.extend(["--materializer-scheduler-storage-tier", tier])
        for root in args.proactive_cleanup_root:
            command.extend(["--materializer-scheduler-proactive-cleanup-root", root])
        for root in args.proactive_cleanup_cold_store_root:
            command.extend(["--materializer-scheduler-proactive-cleanup-cold-store-root", root])
    return command


def _build_staircase_artifacts(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    execution_queue: Path,
    state_path: Path,
    queue: dict[str, Any],
) -> dict[str, Any]:
    resource_pools = [
        parse_resource_pool_spec(spec)
        for spec in args.staircase_resource_pool
    ] or None
    status_map = experiment_queue_status_map(
        queue_path=execution_queue,
        repo_root=REPO_ROOT,
        state_path=state_path,
    )
    dag = build_staircase_dag_from_experiment_queue(
        queue,
        dag_id=f"{queue['queue_id']}_staircase",
        source_path=_display_path(execution_queue),
        resource_pools=resource_pools,
    )
    plan = plan_staircase_dispatch(
        dag,
        status_map=status_map,
        max_nodes=args.staircase_max_nodes,
        allow_cloud=args.staircase_allow_cloud,
        diversity_bucket_limit=args.staircase_diversity_bucket_limit,
    )
    dag_path = run_dir / "staircase_dag.json"
    plan_path = run_dir / "staircase_dispatch_plan.json"
    _write_json(dag_path, dag)
    _write_json(plan_path, plan)
    return {
        "dag_path": _display_path(dag_path),
        "dispatch_plan_path": _display_path(plan_path),
        "dag_hash": dag.get("dag_hash"),
        "plan_hash": plan.get("plan_hash"),
        "selected_count": plan.get("selected_count"),
        "blocked_count": plan.get("blocked_count"),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _ssh_executor_command(
    args: argparse.Namespace,
    *,
    execution_queue: Path,
    state_path: Path,
    staircase_plan_path: Path,
    run_dir: Path,
    execute: bool,
) -> list[str]:
    command = [
        sys.executable,
        "tools/run_staircase_ssh_executor.py",
        "--plan",
        _display_path(staircase_plan_path),
        "--queue",
        _display_path(execution_queue),
        "--state",
        _display_path(state_path),
        "--output",
        _display_path(
            run_dir
            / (
                "staircase_ssh_executor_execute.json"
                if execute
                else "staircase_ssh_executor_dry_run.json"
            )
        ),
    ]
    if execute:
        command.append("--execute")
    command.extend(["--max-steps", str(args.staircase_ssh_max_steps)])
    if args.staircase_ssh_machine_id:
        command.extend(["--machine-id", str(args.staircase_ssh_machine_id)])
    if args.staircase_ssh_allow_future_executor:
        command.append("--allow-future-executor")
    if args.staircase_ssh_allow_dirty_remote_git:
        command.append("--allow-dirty-remote-git")
    if args.staircase_ssh_dirty_remote_git_rationale:
        command.extend([
            "--dirty-remote-git-rationale",
            str(args.staircase_ssh_dirty_remote_git_rationale),
        ])
    command.extend(_parse_remote_repo_roots(args.staircase_ssh_remote_repo_root))
    if execute or args.staircase_ssh_require_artifact_mobility:
        command.append("--require-artifact-mobility")
    command.extend(_parse_artifact_path_maps(args.staircase_ssh_artifact_path_map))
    if args.staircase_ssh_artifact_shared_path_rationale:
        command.extend([
            "--artifact-shared-path-rationale",
            str(args.staircase_ssh_artifact_shared_path_rationale),
        ])
    command.extend(["--rsync-binary", str(args.staircase_ssh_rsync_binary)])
    command.extend([
        "--artifact-pull-timeout-seconds",
        str(args.staircase_ssh_artifact_pull_timeout_seconds),
    ])
    return command


def _ssh_executor_dry_run_command(
    args: argparse.Namespace,
    *,
    execution_queue: Path,
    state_path: Path,
    staircase_plan_path: Path,
    run_dir: Path,
) -> list[str]:
    return _ssh_executor_command(
        args,
        execution_queue=execution_queue,
        state_path=state_path,
        staircase_plan_path=staircase_plan_path,
        run_dir=run_dir,
        execute=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.candidate_limit < 1:
        raise SystemExit("--candidate-limit must be >= 1")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")
    if args.staircase_ssh_max_steps < 1:
        raise SystemExit("--staircase-ssh-max-steps must be >= 1")
    if args.staircase_ssh_execute and args.execute:
        raise SystemExit(
            "--staircase-ssh-execute and top-level --execute cannot target the same queue run"
        )
    if args.staircase_ssh_artifact_pull_timeout_seconds < 1:
        raise SystemExit("--staircase-ssh-artifact-pull-timeout-seconds must be >= 1")
    if args.staircase_ssh_dirty_remote_git_rationale and not args.staircase_ssh_allow_dirty_remote_git:
        raise SystemExit(
            "--staircase-ssh-dirty-remote-git-rationale requires "
            "--staircase-ssh-allow-dirty-remote-git"
        )
    if args.staircase_ssh_execute and not (
        args.staircase_ssh_artifact_path_map
        or args.staircase_ssh_artifact_shared_path_rationale
    ):
        raise SystemExit(
            "--staircase-ssh-execute requires --staircase-ssh-artifact-path-map "
            "or --staircase-ssh-artifact-shared-path-rationale"
        )
    if args.staircase_ssh_artifact_path_map and args.staircase_ssh_artifact_shared_path_rationale:
        raise SystemExit(
            "--staircase-ssh-artifact-path-map and "
            "--staircase-ssh-artifact-shared-path-rationale are mutually exclusive"
        )
    run_dir = _resolve(args.run_dir) if args.run_dir is not None else (
        REPO_ROOT / ".omx" / "research" / f"byte_shaving_materializer_campaign_{_utc_stamp()}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    execution_queue = run_dir / "materializer_execution_queue.json"

    commands: list[CommandResult] = []
    build_result = _run(_build_queue_command(args, run_dir=run_dir))
    commands.append(build_result)
    queue = load_queue_definition(execution_queue)
    state_path = default_state_path(REPO_ROOT, queue["queue_id"])

    for command in (
        [sys.executable, "tools/experiment_queue.py", "--queue", _display_path(execution_queue), "validate"],
        [sys.executable, "tools/experiment_queue.py", "--queue", _display_path(execution_queue), "init"],
    ):
        commands.append(_run(command))

    staircase_artifacts: dict[str, Any] | None = None
    ssh_executor_dry_run: dict[str, Any] | None = None
    ssh_executor_execute: dict[str, Any] | None = None
    ssh_execute_result: CommandResult | None = None
    if args.emit_staircase_plan or args.staircase_ssh_dry_run or args.staircase_ssh_execute:
        staircase_artifacts = _build_staircase_artifacts(
            args,
            run_dir=run_dir,
            execution_queue=execution_queue,
            state_path=state_path,
            queue=queue,
        )
        if args.staircase_ssh_dry_run:
            ssh_result = _run(
                _ssh_executor_dry_run_command(
                    args,
                    execution_queue=execution_queue,
                    state_path=state_path,
                    staircase_plan_path=run_dir / "staircase_dispatch_plan.json",
                    run_dir=run_dir,
                ),
                check=False,
            )
            commands.append(ssh_result)
            ssh_executor_dry_run = _require_json_stdout(
                ssh_result,
                label="staircase SSH executor dry-run",
            )
        if args.staircase_ssh_execute:
            ssh_execute_result = _run(
                _ssh_executor_command(
                    args,
                    execution_queue=execution_queue,
                    state_path=state_path,
                    staircase_plan_path=run_dir / "staircase_dispatch_plan.json",
                    run_dir=run_dir,
                    execute=True,
                ),
                check=False,
            )
            commands.append(ssh_execute_result)
            ssh_executor_execute = _require_json_stdout(
                ssh_execute_result,
                label="staircase SSH executor execute",
                allow_nonzero=True,
            )

    worker_command = [
        sys.executable,
        "tools/experiment_queue.py",
        "--queue",
        _display_path(execution_queue),
        "run-worker",
        "--max-steps",
        str(args.max_steps),
        "--max-parallel",
        str(args.max_parallel),
        "--idle-sleep-seconds",
        str(args.idle_sleep_seconds),
        "--max-idle-cycles",
        str(args.max_idle_cycles),
    ]
    if args.max_experiments is not None:
        worker_command.extend(["--max-experiments", str(args.max_experiments)])
    if args.execute:
        worker_command.append("--execute")
    worker_result = _run(worker_command)
    commands.append(worker_result)

    observe_result = _run([
        sys.executable,
        "tools/experiment_queue.py",
        "--queue",
        _display_path(execution_queue),
        "observe",
        "--tail-lines",
        str(args.tail_lines),
        "--format",
        "json",
    ])
    commands.append(observe_result)
    performance_result = _run([
        sys.executable,
        "tools/experiment_queue.py",
        "--queue",
        _display_path(execution_queue),
        "performance",
    ], check=False)
    commands.append(performance_result)

    payload = {
        "schema": RUN_SCHEMA,
        "run_dir": _display_path(run_dir),
        "plan": _display_path(_resolve(args.plan)),
        "queue_path": _display_path(execution_queue),
        "state_path": _display_path(state_path),
        "execute": bool(args.execute),
        "staircase_ssh_execute": bool(args.staircase_ssh_execute),
        "queue_id": queue["queue_id"],
        "experiment_count": len(queue["experiments"]),
        "build": _json_from_stdout(build_result),
        "worker": _json_from_stdout(worker_result),
        "staircase": staircase_artifacts,
        "ssh_executor_dry_run": ssh_executor_dry_run,
        "ssh_executor_execute": ssh_executor_execute,
        "observation": _json_from_stdout(observe_result),
        "performance": _json_from_stdout(performance_result),
        "commands": [result.to_dict() for result in commands],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    summary_path = run_dir / "materializer_campaign_run.json"
    _write_json(summary_path, payload)
    payload["summary_path"] = _display_path(summary_path)
    _json_print(payload)
    ssh_execute_returncode = ssh_execute_result.returncode if ssh_execute_result is not None else 0
    return 2 if worker_result.returncode != 0 or ssh_execute_returncode != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
