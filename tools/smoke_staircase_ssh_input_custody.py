#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise SSH input-artifact custody without network or score authority.

This smoke uses the production staircase SSH executor with a deterministic
fake transport. It proves the queue-owned path for directory input pushes,
recursive manifests, output pullback, local postconditions, and false-authority
guards without contacting SSH, launching paid work, or claiming a score.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import (  # noqa: E402
    connect_state,
    initialize_queue_state,
    normalize_queue_definition,
)
from comma_lab.scheduler.ssh_experiment_queue_executor import (  # noqa: E402
    run_staircase_ssh_executor,
)
from comma_lab.scheduler.staircase_dag import (  # noqa: E402
    build_staircase_dag_from_experiment_queue,
    experiment_queue_status_map,
    plan_staircase_dispatch,
)
from tac.optimization.proxy_candidate_contract import apply_proxy_evidence_boundary  # noqa: E402
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402

SMOKE_SCHEMA = "staircase_ssh_input_custody_smoke.v1"
OUTPUT_SCHEMA = "staircase_ssh_input_custody_smoke_output.v1"


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


def _write_json(path: Path, payload: object) -> None:
    try:
        write_json_artifact(path, payload, allow_overwrite=True)
    except ArtifactWriteError as exc:
        raise SystemExit(str(exc)) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "artifact directory; defaults to "
            ".omx/research/staircase_ssh_input_custody_smoke_<UTC>"
        ),
    )
    parser.add_argument("--queue-id", default="staircase_ssh_input_custody_smoke")
    parser.add_argument("--machine-id", default="ssh_input_custody_smokebox")
    parser.add_argument("--ssh-target", default="smoke@sshbox")
    parser.add_argument("--remote-repo-root", default="/remote/pact-smoke")
    parser.add_argument("--remote-artifact-root", default="/remote/pact-smoke/artifacts")
    parser.add_argument("--max-steps", type=int, default=1)
    return parser.parse_args(argv)


def _make_input_bundle(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "surface.json").write_text(
        json.dumps(
            {
                "schema": "staircase_ssh_input_custody_smoke_input.v1",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "values": [1, 2, 3],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    nested = input_dir / "nested"
    nested.mkdir(exist_ok=True)
    (nested / "payload.bin").write_bytes(b"ssh-input-custody-smoke\n")


def _make_queue(
    *,
    queue_id: str,
    input_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    command = [
        "python",
        "-c",
        (
            "import json, pathlib; "
            f"path = pathlib.Path({output_path.as_posix()!r}); "
            "path.parent.mkdir(parents=True, exist_ok=True); "
            "path.write_text(json.dumps({"
            f"'schema': {OUTPUT_SCHEMA!r}, "
            "'score_claim': False, "
            "'promotion_eligible': False, "
            "'rank_or_kill_eligible': False, "
            "'ready_for_exact_eval_dispatch': False"
            "}))"
        ),
    ]
    return normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": queue_id,
            "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
            "experiments": [
                {
                    "id": "directory_input_custody_smoke",
                    "status": "queued",
                    "priority": 1,
                    "steps": [
                        {
                            "id": "remote_materialize",
                            "kind": "command",
                            "command": command,
                            "resources": {"kind": "local_cpu"},
                            "timeout_seconds": 60,
                            "telemetry": {
                                "input_artifact_paths": [input_dir.as_posix()],
                                "pullback_artifact_paths": [output_path.as_posix()],
                                "artifact_paths": [output_path.as_posix()],
                            },
                            "artifact_mobility": {
                                "schema": "experiment_queue_artifact_mobility.v1",
                                "mode": "pullback",
                                "required": True,
                            },
                            "postconditions": [
                                {
                                    "type": "path_exists",
                                    "path": output_path.as_posix(),
                                },
                                {
                                    "type": "json_false_authority",
                                    "path": output_path.as_posix(),
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    )


def _make_plan(
    *,
    queue: dict[str, Any],
    queue_path: Path,
    state_path: Path,
    machine_id: str,
    ssh_target: str,
    remote_repo_root: str,
) -> dict[str, Any]:
    dag = build_staircase_dag_from_experiment_queue(
        queue,
        dag_id=f"{queue['queue_id']}_staircase",
        source_path=_display_path(queue_path),
        resource_pools=[
            {
                "id": machine_id,
                "label": "SSH input custody smoke worker",
                "slots": {"local_cpu": 1},
                "memory_gb": 4,
                "disk_gb": 4,
                "tags": ["ssh", "smoke", "no-network"],
                "executor": "ssh_experiment_queue",
                "ssh_target": ssh_target,
                "remote_repo_root": remote_repo_root,
            }
        ],
    )
    status_map = experiment_queue_status_map(
        queue_path=queue_path,
        repo_root=REPO_ROOT,
        state_path=state_path,
    )
    return plan_staircase_dispatch(dag, status_map=status_map, max_nodes=1)


def _is_preflight_only(remote_script: str) -> bool:
    return (
        "git rev-parse HEAD" in remote_script
        and " && python " not in remote_script
        and " && .venv/bin/python " not in remote_script
    )


class FakeSshTransport:
    def __init__(
        self,
        *,
        ssh_target: str,
        output_path: Path,
        input_dir: Path,
    ) -> None:
        self.ssh_target = ssh_target
        self.output_path = output_path
        self.input_dir = input_dir
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        tool = Path(str(argv[0])).name
        remote_script = str(argv[-1])
        if tool == "ssh" and _is_preflight_only(remote_script):
            return subprocess.CompletedProcess(argv, 0, stdout="smoke preflight ok\n")
        if tool == "ssh" and remote_script.startswith("mkdir -p "):
            return subprocess.CompletedProcess(argv, 0, stdout="smoke mkdir ok\n")
        if tool == "rsync" and str(argv[-1]).startswith(f"{self.ssh_target}:"):
            return subprocess.CompletedProcess(argv, 0, stdout="smoke input push ok\n")
        if tool == "rsync":
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(
                json.dumps(
                    {
                        "schema": OUTPUT_SCHEMA,
                        "input_bundle": self.input_dir.as_posix(),
                        "transport": "fake_ssh_no_network",
                        "score_claim": False,
                        "score_claim_valid": False,
                        "score_claim_eligible": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "promotable": False,
                        "ready_for_exact_eval_dispatch": False,
                        "field_selection_ready_for_exact_eval_dispatch": False,
                        "dispatch_attempted": False,
                        "exact_cuda_auth_eval": False,
                        "contest_cuda_auth_eval": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(argv, 0, stdout="smoke pullback ok\n")
        return subprocess.CompletedProcess(argv, 0, stdout="smoke remote command ok\n")


def _directory_push_used_delete(calls: list[list[str]], *, ssh_target: str) -> bool:
    for call in calls:
        if Path(str(call[0])).name != "rsync":
            continue
        if not str(call[-1]).startswith(f"{ssh_target}:"):
            continue
        if "--delete" in call:
            return True
    return False


def _input_manifest_from_result(result: dict[str, Any]) -> dict[str, Any] | None:
    for task_result in result.get("task_results", []):
        if not isinstance(task_result, dict):
            continue
        event = task_result.get("event")
        if not isinstance(event, dict):
            event = task_result
        input_mobility = event.get("input_mobility") if isinstance(event, dict) else None
        if not isinstance(input_mobility, dict):
            continue
        pushes = input_mobility.get("pushes")
        if not isinstance(pushes, list) or not pushes:
            continue
        first = pushes[0]
        if isinstance(first, dict):
            return {
                "local_manifest": first.get("local_manifest"),
                "local_manifest_sha256": first.get("local_manifest_sha256"),
                "remote_path": first.get("remote_path"),
            }
    return None


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")
    run_dir = _resolve(args.run_dir) if args.run_dir else (
        REPO_ROOT / ".omx" / "research" / f"staircase_ssh_input_custody_smoke_{_utc_stamp()}"
    )
    input_dir = run_dir / "inputs" / "directory_bundle"
    output_path = run_dir / "outputs" / "remote_result.json"
    queue_path = run_dir / "experiment_queue.json"
    plan_path = run_dir / "staircase_dispatch_plan.json"
    state_path = run_dir / "queue_state.sqlite"
    report_path = run_dir / "smoke_report.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    _make_input_bundle(input_dir)
    queue = _make_queue(queue_id=args.queue_id, input_dir=input_dir, output_path=output_path)
    _write_json(queue_path, queue)
    with connect_state(state_path) as conn:
        initialize_queue_state(conn, queue)
    plan = _make_plan(
        queue=queue,
        queue_path=queue_path,
        state_path=state_path,
        machine_id=args.machine_id,
        ssh_target=args.ssh_target,
        remote_repo_root=args.remote_repo_root,
    )
    _write_json(plan_path, plan)

    transport = FakeSshTransport(
        ssh_target=args.ssh_target,
        output_path=output_path,
        input_dir=input_dir,
    )
    result = run_staircase_ssh_executor(
        plan,
        queue,
        state_path=state_path,
        repo_root=REPO_ROOT,
        execute=True,
        max_steps=args.max_steps,
        machine_id=args.machine_id,
        allow_noncanonical_state=True,
        artifact_path_maps={run_dir.as_posix(): args.remote_artifact_root},
        require_artifact_mobility=True,
        runner=transport,
    )
    output_payload = json.loads(output_path.read_text(encoding="utf-8")) if output_path.is_file() else None
    observations = {
        "success_count": int(result.get("success_count") or 0),
        "failure_count": int(result.get("failure_count") or 0),
        "directory_push_used_delete": _directory_push_used_delete(
            transport.calls,
            ssh_target=args.ssh_target,
        ),
        "output_artifact_exists": output_path.is_file(),
        "output_false_authority": isinstance(output_payload, dict)
        and output_payload.get("score_claim") is False
        and output_payload.get("promotion_eligible") is False
        and output_payload.get("rank_or_kill_eligible") is False
        and output_payload.get("ready_for_exact_eval_dispatch") is False,
        "network_attempted": False,
        "paid_dispatch_attempted": False,
    }
    report = apply_proxy_evidence_boundary(
        {
            "schema": SMOKE_SCHEMA,
            "run_dir": _display_path(run_dir),
            "queue_path": _display_path(queue_path),
            "plan_path": _display_path(plan_path),
            "state_path": _display_path(state_path),
            "report_path": _display_path(report_path),
            "input_dir": _display_path(input_dir),
            "output_artifact_path": _display_path(output_path),
            "queue_id": queue["queue_id"],
            "machine_id": args.machine_id,
            "transport": "fake_ssh_no_network",
            "ssh_target": args.ssh_target,
            "remote_repo_root": args.remote_repo_root,
            "remote_artifact_root": args.remote_artifact_root,
            "observations": observations,
            "input_manifest": _input_manifest_from_result(result),
            "executor_result": result,
            "transport_calls": transport.calls,
        },
        dispatch_blockers=[
            "local_fake_transport_no_network",
            "non_authoritative_custody_smoke_only",
            "exact_eval_not_attempted",
        ],
    )
    _write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    payload = run_smoke(parse_args(argv))
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload["observations"]["success_count"] == 1 else 2


if __name__ == "__main__":
    raise SystemExit(main())
