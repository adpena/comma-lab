#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an experiment_queue.v1 file from local MLX drift-scope search plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from comma_lab.scheduler.experiment_queue import ExperimentQueueError  # noqa: E402
from comma_lab.scheduler.mlx_drift_scope_queue import (  # noqa: E402
    build_mlx_drift_scope_search_queue,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def _load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExperimentQueueError(f"{path}: expected JSON object")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--lane-id", default="local_mlx_drift_scope_search")
    parser.add_argument("--local-cpu-concurrency", type=int, default=1)
    parser.add_argument("--local-mlx-concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue = build_mlx_drift_scope_search_queue(
        [_load_plan(path) for path in args.plan],
        queue_id=args.queue_id,
        repo_root=args.repo_root,
        lane_id=args.lane_id,
        local_cpu_concurrency=args.local_cpu_concurrency,
        local_mlx_concurrency=args.local_mlx_concurrency,
        timeout_seconds=args.timeout_seconds,
        limit=args.limit,
    )
    try:
        write_json_artifact(
            args.output,
            queue,
            allow_overwrite=args.allow_overwrite,
            expected_existing_sha256=args.expected_output_sha256,
        )
    except ArtifactWriteError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
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


if __name__ == "__main__":
    raise SystemExit(main())
