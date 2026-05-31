#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Normalize production MLX scorer-response steps to singleton batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue_rewriters import (  # noqa: E402
    normalize_mlx_response_singleton_batches,
)
from tac.repo_io import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite output queue: {args.output}")
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    updated = normalize_mlx_response_singleton_batches(queue, reason=args.reason)
    write_json(args.output, updated)
    migration = (updated.get("metadata") or {}).get("queue_migrations", [])[-1]
    print(
        json.dumps(
            {
                "schema": "experiment_queue_mlx_response_batch_normalizer_result.v1",
                "queue": str(args.queue),
                "output": str(args.output),
                "changed_command_count": migration.get("changed_command_count", 0),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

