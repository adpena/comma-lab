#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an experiment queue for deterministic MLX replay verification."""

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

from comma_lab.scheduler.experiment_queue import ExperimentQueueError  # noqa: E402
from comma_lab.scheduler.mlx_master_gradient_replay_queue import (  # noqa: E402
    MLX_MASTER_GRADIENT_REPLAY_QUEUE_SCHEMA,
    build_mlx_master_gradient_replay_queue,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-bundle", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--local-mlx-concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--python-executable", default=".venv/bin/python")
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--append-manifest", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        queue = build_mlx_master_gradient_replay_queue(
            replay_bundle_paths=args.replay_bundle,
            output_root=args.output_root,
            queue_id=args.queue_id,
            repo_root=args.repo_root,
            local_mlx_concurrency=args.local_mlx_concurrency,
            timeout_seconds=args.timeout_seconds,
            strict=not args.no_strict,
            append_manifest=args.append_manifest,
            python_executable=args.python_executable,
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
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema": MLX_MASTER_GRADIENT_REPLAY_QUEUE_SCHEMA,
                "queue_id": queue["queue_id"],
                "step_count": sum(
                    len(experiment["steps"]) for experiment in queue["experiments"]
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
