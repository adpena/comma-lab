#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile byte-shaving campaign plans into DQS1 local-first queue inputs."""
from __future__ import annotations

import argparse
import json
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
    compile_dqs1_byte_shaving_campaign,
)
from comma_lab.scheduler.dqs1_local_first_queue import (  # noqa: E402
    DEFAULT_QUEUE_ID,
    DEFAULT_RESULTS_ROOT,
    build_queue_from_action_summary,
)
from comma_lab.scheduler.experiment_queue import ExperimentQueueError  # noqa: E402
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return payload


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    allow_overwrite: bool = False,
    expected_existing_sha256: str | None = None,
) -> None:
    try:
        write_json_artifact(
            path,
            payload,
            allow_overwrite=allow_overwrite,
            expected_existing_sha256=expected_existing_sha256,
        )
    except ArtifactWriteError as exc:
        raise SystemExit(str(exc)) from exc


def _parse_pair_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    out: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            out.append(int(item))
        except ValueError as exc:
            raise SystemExit(f"--base-pair-indices contains a non-integer: {item!r}") from exc
    if not out:
        raise SystemExit("--base-pair-indices must contain at least one integer")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--materialization-out", type=Path, required=True)
    parser.add_argument("--portfolio-out", type=Path, required=True)
    parser.add_argument("--action-summary-out", type=Path, required=True)
    parser.add_argument("--queue-out", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--base-pair-indices",
        default=None,
        help="comma-separated DQS1 base pairset; required unless the plan carries dqs1_base_pair_indices",
    )
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--queue-candidate-limit", type=int, default=2)
    parser.add_argument("--allow-partial-materialization", action="store_true")
    parser.add_argument("--partial-materialization-rationale", default=None)
    parser.add_argument("--queue-id", default=DEFAULT_QUEUE_ID)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--completed-results-root", action="append", default=[])
    parser.add_argument("--local-cpu-concurrency", type=int, default=2)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--expected-materialization-sha256")
    parser.add_argument("--expected-portfolio-sha256")
    parser.add_argument("--expected-action-summary-sha256")
    parser.add_argument("--expected-queue-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.candidate_limit < 1:
        raise SystemExit("--candidate-limit must be >= 1")
    if args.queue_candidate_limit < 1:
        raise SystemExit("--queue-candidate-limit must be >= 1")
    if args.local_cpu_concurrency < 1:
        raise SystemExit("--local-cpu-concurrency must be >= 1")
    if args.allow_partial_materialization and not str(
        args.partial_materialization_rationale or ""
    ).strip():
        raise SystemExit(
            "--allow-partial-materialization requires --partial-materialization-rationale"
        )
    if not args.plan.is_file():
        raise SystemExit(f"--plan does not exist: {args.plan}")

    plan = _load_json(args.plan)
    try:
        compiled = compile_dqs1_byte_shaving_campaign(
            plan,
            repo_root=args.repo_root,
            plan_path=args.plan,
            base_pair_indices=_parse_pair_indices(args.base_pair_indices),
            candidate_limit=args.candidate_limit,
            portfolio_json=str(args.portfolio_out),
            allow_partial_materialization=bool(args.allow_partial_materialization),
            partial_materialization_rationale=args.partial_materialization_rationale,
        )
    except ExperimentQueueError as exc:
        raise SystemExit(str(exc)) from exc

    _write_json(
        args.materialization_out,
        compiled,
        allow_overwrite=bool(args.overwrite_output),
        expected_existing_sha256=args.expected_materialization_sha256,
    )
    _write_json(
        args.portfolio_out,
        compiled["portfolio"],
        allow_overwrite=bool(args.overwrite_output),
        expected_existing_sha256=args.expected_portfolio_sha256,
    )
    _write_json(
        args.action_summary_out,
        compiled["action_summary"],
        allow_overwrite=bool(args.overwrite_output),
        expected_existing_sha256=args.expected_action_summary_sha256,
    )

    queue_payload = None
    if args.queue_out is not None:
        if int(compiled["blocked_row_count"]) > 0 and not args.allow_partial_materialization:
            raise SystemExit(
                "blocked byte-shaving materialization rows present; refusing partial "
                "queue build without --allow-partial-materialization and rationale"
            )
        if int(compiled["executable_row_count"]) <= 0:
            raise SystemExit("no executable DQS1 byte-shaving rows; refusing to build queue")
        if int(compiled["queueable_row_count"]) <= 0:
            raise SystemExit("no queueable DQS1 byte-shaving rows; refusing to build queue")
        queue_result = build_queue_from_action_summary(
            args.action_summary_out,
            repo_root=args.repo_root,
            results_root=args.results_root,
            queue_id=args.queue_id,
            completed_results_roots=tuple(args.completed_results_root),
            candidate_limit=args.queue_candidate_limit,
            local_cpu_concurrency=args.local_cpu_concurrency,
        )
        _write_json(
            args.queue_out,
            queue_result.queue,
            allow_overwrite=bool(args.overwrite_output),
            expected_existing_sha256=args.expected_queue_sha256,
        )
        queue_payload = {
            "queue_out": str(args.queue_out),
            "queue_id": queue_result.queue["queue_id"],
            "experiment_count": len(queue_result.queue["experiments"]),
            "selected_candidate_ids": [
                selection.candidate_id for selection in queue_result.selections
            ],
        }

    print(
        json.dumps(
            {
                "schema": "byte_shaving_campaign_queue_build_result.v1",
                "materialization_out": str(args.materialization_out),
                "portfolio_out": str(args.portfolio_out),
                "action_summary_out": str(args.action_summary_out),
                "executable_row_count": compiled["executable_row_count"],
                "blocked_row_count": compiled["blocked_row_count"],
                "queue": queue_payload,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
