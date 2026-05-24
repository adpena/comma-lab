#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile MLX scorer-response execution plans into experiment_queue work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import ExperimentQueueError  # noqa: E402
from comma_lab.scheduler.mlx_execution_queue import (  # noqa: E402
    build_mlx_scorer_response_execution_queue,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--queue-id", default="mlx_scorer_response_local_substrate")
    parser.add_argument("--lane-id", default="mlx_scorer_response_local_substrate")
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--local-cpu-concurrency", default=1, type=int)
    parser.add_argument("--local-mlx-concurrency", default=1, type=int)
    parser.add_argument("--timeout-seconds", default=0, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        queue = build_mlx_scorer_response_execution_queue(
            [_load_json(path) for path in args.plan],
            queue_id=args.queue_id,
            repo_root=args.repo_root,
            lane_id=args.lane_id,
            local_cpu_concurrency=args.local_cpu_concurrency,
            local_mlx_concurrency=args.local_mlx_concurrency,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
        )
        write_json_artifact(
            args.output,
            queue,
            allow_overwrite=args.allow_overwrite,
            expected_existing_sha256=args.expected_output_sha256,
        )
    except (ArtifactWriteError, ExperimentQueueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(
        json.dumps(
            {
                "output": str(args.output),
                "queue_id": queue["queue_id"],
                "experiment_count": len(queue["experiments"]),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
