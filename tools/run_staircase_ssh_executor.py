#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run SSH-backed staircase tasks while experiment_queue.v1 remains authoritative."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import (  # noqa: E402
    ExperimentQueueError,
    default_state_path,
    load_queue_definition,
)
from comma_lab.scheduler.ssh_experiment_queue_executor import (  # noqa: E402
    run_staircase_ssh_executor,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402

_PLACEHOLDER_RATIONALES = {"", "n/a", "na", "none", "test", "true", "yes", "because"}


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExperimentQueueError(f"{path}: expected JSON object")
    return payload


def _remote_repo_roots(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        machine, sep, root = value.partition("=")
        if not sep or not machine.strip() or not root.strip():
            raise ExperimentQueueError("--remote-repo-root must be MACHINE_ID=PATH")
        out[machine.strip()] = root.strip()
    return out


def _artifact_path_maps(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        local, sep, remote = value.partition("=")
        if not sep or not local.strip() or not remote.strip():
            raise ExperimentQueueError("--artifact-path-map must be LOCAL_PREFIX=REMOTE_PREFIX")
        out[local.strip()] = remote.strip()
    return out


def _rationale_text(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if len(text) < 12 or text.lower() in _PLACEHOLDER_RATIONALES:
        raise ExperimentQueueError(f"{label} must be a specific non-placeholder rationale")
    return text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="staircase_dispatch_plan.v1 JSON")
    parser.add_argument("--queue", required=True, help="experiment_queue.v1 JSON/YAML-compatible file")
    parser.add_argument("--state", default=None, help="SQLite state path; defaults to canonical queue state")
    parser.add_argument("--output", default=None, help="write JSON result artifact")
    parser.add_argument("--execute", action="store_true", help="actually claim and launch remote work")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--machine-id", default=None)
    parser.add_argument(
        "--remote-repo-root",
        action="append",
        default=[],
        metavar="MACHINE=PATH",
        help="remote repo root for a machine id; required unless machine metadata carries it",
    )
    parser.add_argument(
        "--allow-future-executor",
        action="store_true",
        help="allow machine.executor=ssh_experiment_queue_future during staged rollout",
    )
    parser.add_argument(
        "--allow-dirty-remote-git",
        action="store_true",
        help="skip remote git diff cleanliness checks; remote HEAD must still match",
    )
    parser.add_argument("--dirty-remote-git-rationale", default=None)
    parser.add_argument("--ssh-binary", default="ssh")
    parser.add_argument("--ssh-connect-timeout-seconds", type=int, default=10)
    parser.add_argument("--remote-preflight-timeout-seconds", type=int, default=20)
    parser.add_argument("--log-root", default=None)
    parser.add_argument(
        "--require-artifact-mobility",
        action="store_true",
        help="require pullback/shared-storage visibility for local postcondition artifacts",
    )
    parser.add_argument(
        "--artifact-path-map",
        action="append",
        default=[],
        metavar="LOCAL_PREFIX=REMOTE_PREFIX",
        help="map a local artifact prefix to the corresponding remote prefix for rsync pullback",
    )
    parser.add_argument(
        "--artifact-shared-path-rationale",
        default=None,
        help="specific rationale when remote artifacts are already visible locally via shared storage",
    )
    parser.add_argument("--rsync-binary", default="rsync")
    parser.add_argument("--artifact-pull-timeout-seconds", type=int, default=300)
    parser.add_argument("--noncanonical-state-rationale", default=None)
    parser.add_argument("--orphaned-state-rationale", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = _load_json(args.plan)
        queue = load_queue_definition(args.queue)
        state = Path(args.state) if args.state else default_state_path(REPO_ROOT, queue["queue_id"])
        noncanonical_rationale = _rationale_text(
            args.noncanonical_state_rationale,
            label="--noncanonical-state-rationale",
        )
        orphaned_rationale = _rationale_text(
            args.orphaned_state_rationale,
            label="--orphaned-state-rationale",
        )
        dirty_remote_git_rationale = _rationale_text(
            args.dirty_remote_git_rationale,
            label="--dirty-remote-git-rationale",
        )
        artifact_shared_path_rationale = _rationale_text(
            args.artifact_shared_path_rationale,
            label="--artifact-shared-path-rationale",
        )
        if artifact_shared_path_rationale is not None and args.artifact_path_map:
            raise ExperimentQueueError(
                "--artifact-shared-path-rationale and --artifact-path-map are mutually exclusive"
            )
        if args.allow_dirty_remote_git and dirty_remote_git_rationale is None:
            raise ExperimentQueueError(
                "--allow-dirty-remote-git requires --dirty-remote-git-rationale"
            )
        if dirty_remote_git_rationale is not None and not args.allow_dirty_remote_git:
            raise ExperimentQueueError(
                "--dirty-remote-git-rationale requires --allow-dirty-remote-git"
            )
        if args.artifact_pull_timeout_seconds < 1:
            raise ExperimentQueueError("--artifact-pull-timeout-seconds must be >= 1")
        result = run_staircase_ssh_executor(
            plan,
            queue,
            state_path=state,
            repo_root=REPO_ROOT,
            execute=args.execute,
            max_steps=args.max_steps,
            machine_id=args.machine_id,
            remote_repo_roots=_remote_repo_roots(args.remote_repo_root),
            allow_future_executor=args.allow_future_executor,
            allow_noncanonical_state=noncanonical_rationale is not None,
            allow_orphaned_state=orphaned_rationale is not None,
            require_clean_remote_git=not args.allow_dirty_remote_git,
            dirty_remote_git_rationale=dirty_remote_git_rationale,
            ssh_binary=args.ssh_binary,
            ssh_connect_timeout_seconds=args.ssh_connect_timeout_seconds,
            remote_preflight_timeout_seconds=args.remote_preflight_timeout_seconds,
            log_root=args.log_root,
            artifact_path_maps=_artifact_path_maps(args.artifact_path_map),
            require_artifact_mobility=args.require_artifact_mobility,
            artifact_shared_path_rationale=artifact_shared_path_rationale,
            rsync_binary=args.rsync_binary,
            artifact_pull_timeout_seconds=args.artifact_pull_timeout_seconds,
        )
        if args.output:
            try:
                write_json_artifact(args.output, result)
            except ArtifactWriteError as exc:
                raise ExperimentQueueError(str(exc)) from exc
        _json_print(result)
        if not args.execute:
            return 0
        failed_or_refused = (
            int(result.get("failure_count") or 0)
            + int(result.get("claim_refused_count") or 0)
            + int(result.get("blocked_count") or 0)
        )
        return 0 if failed_or_refused == 0 else 2
    except ExperimentQueueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
