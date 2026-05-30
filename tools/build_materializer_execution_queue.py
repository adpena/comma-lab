#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile a byte_shaving_materializer_work_queue.v1 into experiment_queue.v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.byte_shaving_campaign_queue import (  # noqa: E402
    MATERIALIZER_WORK_QUEUE_SCHEMA,
    build_materializer_execution_queue,
)
from comma_lab.scheduler.experiment_queue import (  # noqa: E402
    ExperimentQueueError,
    default_state_path,
)
from tac.optimization.byte_shaving_campaign import FALSE_AUTHORITY  # noqa: E402
from tac.repo_io import ArtifactWriteError, sha256_file, write_json_artifact  # noqa: E402

RESULT_SCHEMA = "materializer_execution_queue_build_result.v1"
AUTO_LOCAL_CPU_CONCURRENCY = "auto"


def _repo_rel(path: str | Path, repo_root: Path) -> str:
    value = Path(path)
    try:
        return value.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return payload


def _auto_local_cpu_concurrency() -> int:
    count = os.cpu_count()
    return count if count is not None and count > 0 else 1


def _parse_local_cpu_concurrency(value: str) -> int:
    text = str(value).strip().lower()
    if text == AUTO_LOCAL_CPU_CONCURRENCY:
        return _auto_local_cpu_concurrency()
    try:
        parsed = int(text)
    except ValueError as exc:
        raise SystemExit(
            "--local-cpu-concurrency must be a positive integer or 'auto'"
        ) from exc
    if parsed < 1:
        raise SystemExit("--local-cpu-concurrency must be >= 1 or 'auto'")
    return parsed


def _parse_resource_concurrency(values: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for raw in values:
        if "=" not in raw:
            raise SystemExit("--resource-concurrency entries must be KIND=LIMIT")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit("--resource-concurrency KIND must be non-empty")
        try:
            limit = int(value)
        except ValueError as exc:
            raise SystemExit(
                f"--resource-concurrency has non-integer limit: {raw!r}"
            ) from exc
        if limit < 1:
            raise SystemExit(f"--resource-concurrency limit must be >= 1: {raw!r}")
        parsed[key] = limit
    return parsed


def _default_queue_id(work_queue_path: Path, repo_root: Path) -> str:
    rel = _repo_rel(work_queue_path, repo_root)
    digest = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:10]
    stem = "".join(
        char.lower() if char.isalnum() else "_" for char in work_queue_path.stem
    )
    stem = "_".join(part for part in stem.split("_") if part)[:38] or "materializer"
    return f"materializer_exec_{stem}_{digest}"


def _experiment_ids(queue: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for experiment in queue.get("experiments") or []:
        if isinstance(experiment, Mapping) and isinstance(experiment.get("id"), str):
            out.append(str(experiment["id"]))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-queue", type=Path, required=True)
    parser.add_argument("--queue-out", type=Path, required=True)
    parser.add_argument("--queue-id", default=None)
    parser.add_argument("--lane-id", default=None)
    parser.add_argument("--source-state", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--local-cpu-concurrency",
        default=AUTO_LOCAL_CPU_CONCURRENCY,
        help="local_cpu resource cap; use 'auto' for os.cpu_count().",
    )
    parser.add_argument(
        "--resource-concurrency",
        action="append",
        default=[],
        metavar="KIND=LIMIT",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument(
        "--include-exact-readiness-followup",
        action="store_true",
        help="append local harvest/exact-readiness/paused-dispatch-plan follow-up steps.",
    )
    parser.add_argument(
        "--exact-readiness-followup-require-ready",
        action="store_true",
        help="fail generated follow-up if no exact-ready row is emitted.",
    )
    parser.add_argument(
        "--require-renderer-payload-dfl1-parity-followup",
        action="store_true",
    )
    parser.add_argument(
        "--exact-eval-dispatch-require-authorized",
        action="store_true",
    )
    parser.add_argument(
        "--exact-eval-dispatch-provider",
        choices=("lightning", "vastai"),
        default="lightning",
    )
    parser.add_argument("--exact-eval-dispatch-label-prefix", default="materializer_exact_eval")
    parser.add_argument("--exact-eval-dispatch-estimated-cost-per-dispatch", type=float, default=0.30)
    parser.add_argument("--exact-eval-dispatch-max-total-cost", type=float, default=5.00)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--expected-queue-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo_root
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.timeout_seconds < 0:
        raise SystemExit("--timeout-seconds must be non-negative")
    if not args.work_queue.is_file():
        raise SystemExit(f"--work-queue does not exist: {args.work_queue}")
    work_queue = _load_json(args.work_queue)
    if work_queue.get("schema") != MATERIALIZER_WORK_QUEUE_SCHEMA:
        raise SystemExit(f"expected schema {MATERIALIZER_WORK_QUEUE_SCHEMA}")
    queue_id = args.queue_id or _default_queue_id(args.work_queue, repo)
    try:
        queue = build_materializer_execution_queue(
            work_queue,
            queue_id=queue_id,
            repo_root=repo,
            lane_id=args.lane_id,
            source_work_queue_path=args.work_queue,
            source_state_path=args.source_state,
            local_cpu_concurrency=_parse_local_cpu_concurrency(
                args.local_cpu_concurrency
            ),
            resource_concurrency=_parse_resource_concurrency(
                args.resource_concurrency
            ),
            step_timeout_seconds=args.timeout_seconds,
            limit=args.limit,
            include_exact_readiness_followup=args.include_exact_readiness_followup,
            require_renderer_payload_dfl1_parity_followup=(
                args.require_renderer_payload_dfl1_parity_followup
            ),
            exact_readiness_followup_require_ready=(
                args.exact_readiness_followup_require_ready
            ),
            exact_eval_dispatch_require_authorized=(
                args.exact_eval_dispatch_require_authorized
            ),
            exact_eval_dispatch_provider=args.exact_eval_dispatch_provider,
            exact_eval_dispatch_label_prefix=args.exact_eval_dispatch_label_prefix,
            exact_eval_dispatch_estimated_cost_per_dispatch=(
                args.exact_eval_dispatch_estimated_cost_per_dispatch
            ),
            exact_eval_dispatch_max_total_cost=args.exact_eval_dispatch_max_total_cost,
        )
    except ExperimentQueueError as exc:
        raise SystemExit(str(exc)) from exc

    expected_sha = args.expected_queue_sha256
    if expected_sha is None and args.queue_out.is_file() and not args.overwrite_output:
        expected_sha = sha256_file(args.queue_out)
    try:
        artifact = write_json_artifact(
            args.queue_out,
            queue,
            allow_overwrite=args.overwrite_output or expected_sha is not None,
            expected_existing_sha256=expected_sha,
        )
    except ArtifactWriteError as exc:
        raise SystemExit(str(exc)) from exc

    result = {
        "schema": RESULT_SCHEMA,
        "source_work_queue_path": _repo_rel(args.work_queue, repo),
        "source_work_queue_schema": MATERIALIZER_WORK_QUEUE_SCHEMA,
        "queue_out": artifact.path,
        "queue_id": queue_id,
        "queue_schema": queue.get("schema"),
        "state_path": _repo_rel(default_state_path(repo, queue_id), repo),
        "experiment_count": len(queue.get("experiments") or []),
        "experiment_ids": _experiment_ids(queue),
        "include_exact_readiness_followup": bool(args.include_exact_readiness_followup),
        "allowed_use": "local_materializer_execution_queue_build_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_exact_eval_authority",
        "artifact": {
            "path": artifact.path,
            "bytes": artifact.bytes_written,
            "sha256": artifact.sha256,
        },
        **FALSE_AUTHORITY,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
