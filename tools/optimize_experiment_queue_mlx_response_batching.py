#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Rewrite MLX scorer-response queue steps into batch-process execution."""

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

from comma_lab.scheduler.experiment_queue import normalize_queue_definition  # noqa: E402
from comma_lab.scheduler.experiment_queue_rewriters import (  # noqa: E402
    batch_mlx_scorer_response_steps,
)
from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-jobs-per-batch", type=int, default=4)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        queue = json.loads(args.queue.read_text(encoding="utf-8"))
        if not isinstance(queue, dict):
            raise ValueError(f"queue must be a JSON object: {args.queue}")
        updated = batch_mlx_scorer_response_steps(
            queue,
            max_jobs_per_batch=args.max_jobs_per_batch,
            reason=args.reason,
        )
        normalized = normalize_queue_definition(updated)
        expected_sha = sha256_file(args.output) if args.output.is_file() and args.overwrite else None
        write = write_json_artifact(
            args.output,
            normalized,
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=expected_sha,
        )
    except (ArtifactWriteError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: MLX response batching queue optimization failed: {exc}", file=sys.stderr)
        return 2
    migration = normalized.get("metadata", {}).get("queue_migrations", [])[-1]
    print(
        json_text(
            {
                "schema": "experiment_queue_mlx_response_batching_cli_result.v1",
                "output": str(args.output),
                "bytes_written": write.bytes_written,
                "sha256": write.sha256,
                "batch_experiment_count": migration.get("batch_experiment_count"),
                "changed_command_count": migration.get("changed_command_count"),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
