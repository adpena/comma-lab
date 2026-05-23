#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest and optionally reroute the DQS1 local-first queue."""

from __future__ import annotations

import argparse
import hashlib
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
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    try:
        write_json_artifact(path, payload)
    except ArtifactWriteError as exc:
        raise FileExistsError(str(exc)) from exc


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _artifact_token(value: object) -> str:
    text = str(value or "unknown")
    return "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in text)


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
    if args.write_queue and args.harvest_out is None:
        print("--write-queue requires --harvest-out for append-only custody", file=sys.stderr)
        return 2
    queue_path = Path(args.queue)
    if not queue_path.is_absolute():
        queue_path = REPO_ROOT / queue_path
    prior_queue_text = queue_path.read_text(encoding="utf-8") if queue_path.is_file() else ""
    prior_queue_sha256 = _sha256_text(prior_queue_text) if prior_queue_text else None
    try:
        kwargs = {
            "queue_path": args.queue,
            "repo_root": REPO_ROOT,
            "timestamp": args.timestamp,
            "reroute_observe_only": args.reroute_observe_only,
            "output_queue_path": args.queue if args.write_queue else None,
            "expected_output_queue_sha256": prior_queue_sha256 if args.write_queue else None,
            "action_summary": args.action_summary,
        }
        if args.results_root is not None:
            kwargs["results_root"] = args.results_root
        result = build_dqs1_harvest_result(**kwargs)
    except ExperimentQueueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    try:
        if args.harvest_out is not None:
            _write_json(args.harvest_out, result.harvest_record)
        if result.exact_auth_request is not None:
            request_out = args.exact_auth_request_out
            if request_out is None:
                request_out = (
                    REPO_ROOT
                    / ".omx/research"
                    / (
                        "exact_auth_anchor_request_"
                        f"{_artifact_token(result.harvest_record['candidate_id'])}_"
                        f"{_artifact_token(result.harvest_record['harvested_at_utc'])}.json"
                    )
                )
            _write_json(request_out, result.exact_auth_request)
        if args.write_queue and result.rerouted_queue is not None:
            new_queue_text = queue_path.read_text(encoding="utf-8")
            ledger_path = (
                REPO_ROOT
                / ".omx/research"
                / (
                    "dqs1_local_first_queue_reroute_"
                    f"{_artifact_token(result.harvest_record['candidate_id'])}_"
                    f"{_artifact_token(result.harvest_record['harvested_at_utc'])}.json"
                )
            )
            _write_json(
                ledger_path,
                {
                    "schema": "dqs1_local_first_queue_reroute_record.v1",
                    "candidate_id": result.harvest_record["candidate_id"],
                    "harvest_record_path": str(args.harvest_out),
                    "queue_path": args.queue,
                    "previous_queue_sha256": prior_queue_sha256,
                    "new_queue_sha256": _sha256_text(new_queue_text),
                    "rerouted_queue_experiment_ids": [
                        str(experiment.get("id"))
                        for experiment in result.rerouted_queue.get("experiments", [])
                        if isinstance(experiment, dict)
                    ],
                    "source_eureka_signal_path": result.harvest_record["eureka_signal_path"],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
            )
    except FileExistsError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

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
