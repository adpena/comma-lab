#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build or plan a planning-only staircase DAG.

This emits executor-ready task specs for local workers or a future Dask
cluster, while preserving the repository's false-authority contract. It never
executes commands and never promotes scores.
"""

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

from comma_lab.scheduler.experiment_queue import ExperimentQueueError, load_queue_definition  # noqa: E402
from comma_lab.scheduler.staircase_dag import (  # noqa: E402
    DEFAULT_MACHINE_PRESETS,
    build_storage_plan_payload,
    build_staircase_dag_from_experiment_queue,
    experiment_queue_status_map,
    load_staircase_dag,
    local_lab_resource_pools,
    parse_resource_pool_spec,
    plan_staircase_dispatch,
    write_staircase_dag,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _write_json(path: str | Path, payload: object) -> None:
    try:
        write_json_artifact(path, payload)
    except ArtifactWriteError as exc:
        raise ExperimentQueueError(str(exc)) from exc


def _resource_pools(args: argparse.Namespace) -> list[dict]:
    pools = [parse_resource_pool_spec(spec) for spec in args.resource_pool]
    if pools:
        return pools
    if args.local_lab_presets:
        return local_lab_resource_pools()
    return []


def _storage_plan(args: argparse.Namespace) -> dict | None:
    if not getattr(args, "storage_waterfall", False):
        return None
    return build_storage_plan_payload(
        repo_root=REPO_ROOT,
        storage_tiers=args.storage_tier,
        workload_subdir=args.storage_workload_subdir,
        requested_bytes=args.storage_expected_bytes,
        min_free_bytes=args.storage_min_free_bytes,
        reserve_free_gb=args.storage_reserve_gb,
        allow_local_disk=args.allow_local_storage_tier,
        create=args.create_storage_dirs,
    )


def cmd_machine_presets(_args: argparse.Namespace) -> int:
    _json_print(
        {
            "schema": "staircase_machine_presets.v1",
            "resource_pools": list(DEFAULT_MACHINE_PRESETS),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    )
    return 0


def cmd_from_queue(args: argparse.Namespace) -> int:
    queue = load_queue_definition(args.queue)
    dag = build_staircase_dag_from_experiment_queue(
        queue,
        dag_id=args.dag_id,
        source_path=args.queue,
        resource_pools=_resource_pools(args) or None,
        storage_plan=_storage_plan(args),
    )
    if args.output:
        write_staircase_dag(args.output, dag)
    _json_print(dag)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    if args.dag:
        dag = load_staircase_dag(args.dag)
    elif args.queue:
        queue = load_queue_definition(args.queue)
        dag = build_staircase_dag_from_experiment_queue(
            queue,
            dag_id=args.dag_id,
            source_path=args.queue,
            resource_pools=_resource_pools(args) or None,
            storage_plan=_storage_plan(args),
        )
    else:
        raise ExperimentQueueError("plan requires --dag or --queue")

    status_map = {}
    if args.queue and args.use_queue_state:
        state = Path(args.state) if args.state else None
        status_map = experiment_queue_status_map(
            queue_path=args.queue,
            repo_root=REPO_ROOT,
            state_path=state,
        )
    elif args.status_json:
        payload = json.loads(Path(args.status_json).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ExperimentQueueError("--status-json must contain an object")
        status_map = {str(key): str(value) for key, value in payload.items()}

    plan = plan_staircase_dispatch(
        dag,
        status_map=status_map,
        max_nodes=args.max_nodes,
        allow_cloud=args.allow_cloud,
        diversity_bucket_limit=args.diversity_bucket_limit,
    )
    if args.output:
        _write_json(args.output, plan)
    _json_print(plan)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    presets = sub.add_parser("machine-presets", help="print built-in local lab resource pools")
    presets.set_defaults(func=cmd_machine_presets)

    from_queue = sub.add_parser("from-queue", help="convert an experiment queue to a staircase DAG")
    from_queue.add_argument("--queue", required=True)
    from_queue.add_argument("--dag-id", default=None)
    from_queue.add_argument("--output", default=None)
    from_queue.add_argument("--local-lab-presets", action="store_true")
    from_queue.add_argument(
        "--resource-pool",
        action="append",
        default=[],
        help="id:local_cpu=4,local_mlx=1,memory_gb=128,disk_gb=80,tags=darwin+mlx",
    )
    _add_storage_args(from_queue)
    from_queue.set_defaults(func=cmd_from_queue)

    plan = sub.add_parser("plan", help="select ready DAG nodes for execution")
    plan.add_argument("--dag", default=None)
    plan.add_argument("--queue", default=None)
    plan.add_argument("--dag-id", default=None)
    plan.add_argument("--state", default=None)
    plan.add_argument("--use-queue-state", action="store_true")
    plan.add_argument("--status-json", default=None)
    plan.add_argument("--output", default=None)
    plan.add_argument("--max-nodes", type=int, default=None)
    plan.add_argument("--allow-cloud", action="store_true")
    plan.add_argument("--diversity-bucket-limit", type=int, default=None)
    plan.add_argument("--local-lab-presets", action="store_true")
    plan.add_argument(
        "--resource-pool",
        action="append",
        default=[],
        help="id:local_cpu=4,local_mlx=1,memory_gb=128,disk_gb=80,tags=darwin+mlx",
    )
    _add_storage_args(plan)
    plan.set_defaults(func=cmd_plan)
    return parser.parse_args(argv)


def _add_storage_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--storage-waterfall",
        action="store_true",
        help="attach a storage-tier plan to the DAG and block dispatch if no tier is eligible",
    )
    parser.add_argument("--storage-tier", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--storage-workload-subdir", default="experiments/results")
    parser.add_argument("--storage-reserve-gb", type=float, default=40.0)
    parser.add_argument("--storage-expected-bytes", type=int, default=0)
    parser.add_argument("--storage-min-free-bytes", type=int, default=0)
    parser.add_argument("--allow-local-storage-tier", action="store_true")
    parser.add_argument("--create-storage-dirs", action="store_true")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return int(args.func(args))
    except ExperimentQueueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
