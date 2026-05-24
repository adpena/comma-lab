#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest succeeded local-training queue outputs into optimizer candidates."""

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
from comma_lab.scheduler.local_training_harvest import (  # noqa: E402
    LocalTrainingHarvestError,
    harvest_local_training_optimizer_candidates,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        queue = load_queue_definition(args.queue)
        state = args.state or default_state_path(args.repo_root, queue["queue_id"])
        harvested = harvest_local_training_optimizer_candidates(
            queue,
            state_path=state,
            repo_root=args.repo_root,
            top_k=args.top_k,
        )
        write_json_artifact(
            args.output,
            harvested,
            allow_overwrite=args.allow_overwrite,
            expected_existing_sha256=args.expected_output_sha256,
        )
    except (
        ArtifactWriteError,
        ExperimentQueueError,
        LocalTrainingHarvestError,
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
                "schema": harvested["schema"],
                "n_candidates": harvested["n_candidates"],
                "dispatch_ready_count": harvested["dispatch_ready_count"],
                "harvested_representation_manifest_count": harvested["harvest"][
                    "harvested_representation_manifest_count"
                ],
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
