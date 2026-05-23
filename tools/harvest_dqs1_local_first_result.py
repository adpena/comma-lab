#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest and optionally reroute the DQS1 local-first queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.dqs1_local_first_harvest import (  # noqa: E402
    DEFAULT_QUEUE_PATH,
    ExperimentQueueError,
    build_dqs1_harvest_result,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default=DEFAULT_QUEUE_PATH, help="DQS1 queue YAML/JSON path")
    parser.add_argument(
        "--timestamp",
        default=None,
        help="UTC-ish run id for append-only harvest/request artifacts",
    )
    parser.add_argument(
        "--harvest-out",
        type=Path,
        default=None,
        help="optional harvest JSON output path",
    )
    parser.add_argument(
        "--exact-auth-request-out",
        type=Path,
        default=None,
        help="optional exact-auth request JSON output path for positive eureka only",
    )
    parser.add_argument(
        "--reroute-observe-only",
        action="store_true",
        help="when eureka is observe_only, rebuild the queue to the next safe candidate",
    )
    parser.add_argument(
        "--write-queue",
        action="store_true",
        help="write the rerouted queue back to --queue; requires --reroute-observe-only",
    )
    parser.add_argument(
        "--action-summary",
        default="latest",
        help="action_summary.json path, or latest",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="DQS1 results root override used while rerouting",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.write_queue and not args.reroute_observe_only:
        print("--write-queue requires --reroute-observe-only", file=sys.stderr)
        return 2
    try:
        kwargs = {
            "queue_path": args.queue,
            "repo_root": REPO_ROOT,
            "timestamp": args.timestamp,
            "reroute_observe_only": args.reroute_observe_only,
            "output_queue_path": args.queue if args.write_queue else None,
            "action_summary": args.action_summary,
        }
        if args.results_root is not None:
            kwargs["results_root"] = args.results_root
        result = build_dqs1_harvest_result(**kwargs)
    except ExperimentQueueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    if args.harvest_out is not None:
        _write_json(args.harvest_out, result.harvest_record)
    if result.exact_auth_request is not None and args.exact_auth_request_out is not None:
        _write_json(args.exact_auth_request_out, result.exact_auth_request)

    summary = {
        "candidate_id": result.harvest_record["candidate_id"],
        "recommended_action": result.harvest_record["recommended_action"],
        "eureka_trigger": result.harvest_record["eureka_trigger"],
        "local_score": result.harvest_record["local_score"],
        "conservative_projected_contest_score": result.harvest_record[
            "conservative_projected_contest_score"
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "exact_auth_request": result.exact_auth_request is not None,
        "rerouted_queue": result.rerouted_queue is not None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
