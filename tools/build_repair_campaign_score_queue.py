#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the default repair-campaign scoring queue from waterfill work orders."""

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

from comma_lab.scheduler.repair_campaign_score_queue import (  # noqa: E402
    RepairCampaignScoreQueueError,
    build_repair_campaign_score_queue,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-budget-waterfill-queue", required=True, type=Path)
    parser.add_argument("--score-queue-out", required=True, type=Path)
    parser.add_argument(
        "--results-root",
        default=Path("experiments/results"),
        type=Path,
        help="Root used for generated score report paths.",
    )
    parser.add_argument(
        "--queue-id",
        default="repair_campaign_score_queue",
        help="Queue id for the generated scorer queue.",
    )
    parser.add_argument(
        "--experiment-limit",
        type=int,
        default=None,
        help="Optional maximum number of waterfill rows to convert.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        waterfill_queue_path = _resolve(args.repair_budget_waterfill_queue)
        waterfill_queue = json.loads(waterfill_queue_path.read_text(encoding="utf-8"))
        if not isinstance(waterfill_queue, dict):
            raise RepairCampaignScoreQueueError("waterfill queue must be a JSON object")
        score_queue = build_repair_campaign_score_queue(
            repo_root=REPO_ROOT,
            repair_budget_waterfill_queue=waterfill_queue,
            repair_budget_waterfill_queue_path=args.repair_budget_waterfill_queue,
            results_root=args.results_root,
            queue_id=args.queue_id,
            experiment_limit=args.experiment_limit,
        )
        score_queue_out = _resolve(args.score_queue_out)
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if score_queue_out.exists() and args.overwrite:
            existing_text = score_queue_out.read_text(encoding="utf-8")
            next_text = json_text(score_queue)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(score_queue_out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                score_queue_out,
                score_queue,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
    except (
        ArtifactWriteError,
        OSError,
        RepairCampaignScoreQueueError,
        ValueError,
    ) as exc:
        print(f"FATAL: repair campaign score queue build failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "repair_campaign_score_queue_cli_result.v1",
                "repair_budget_waterfill_queue": str(
                    args.repair_budget_waterfill_queue
                ),
                "score_queue_out": str(args.score_queue_out),
                "queue_id": score_queue["queue_id"],
                "experiment_count": len(score_queue["experiments"]),
                "ready_experiment_count": score_queue["metadata"][
                    "ready_experiment_count"
                ],
                "blocked_experiment_count": score_queue["metadata"][
                    "blocked_experiment_count"
                ],
                "bytes_written": (
                    write_result.bytes_written if write_result is not None else 0
                ),
                "skipped_identical_existing_artifact": (
                    skipped_identical_existing_artifact
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
