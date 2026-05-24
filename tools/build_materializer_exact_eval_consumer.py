#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a paused dry-run queue from materializer exact-ready artifacts."""

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
from comma_lab.scheduler.materializer_exact_eval_consumer import (  # noqa: E402
    build_materializer_exact_eval_consumer_queue,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bridge-report",
        action="append",
        default=[],
        help="materializer_chain_exact_readiness_bridge_report.v1 artifact",
    )
    parser.add_argument(
        "--exact-ready-queue",
        action="append",
        default=[],
        help="optimizer_candidate_exact_eval_ready_queue_v1 artifact",
    )
    parser.add_argument("--consumer-report-out", required=True)
    parser.add_argument("--experiment-queue-out", required=True)
    parser.add_argument(
        "--queue-id",
        default="materializer_exact_eval_consumer_queue",
    )
    parser.add_argument("--provider", default="lightning", choices=["lightning", "vastai"])
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--estimated-cost-per-dispatch", type=float, default=0.30)
    parser.add_argument("--max-total-cost", type=float, default=5.00)
    parser.add_argument("--label-prefix", default="materializer_exact_eval_consumer")
    parser.add_argument("--agent", default="codex")
    parser.add_argument("--dispatch-claims-path", default=None)
    parser.add_argument("--active-floor-archive-bytes", type=int, default=None)
    parser.add_argument("--active-floor-score", type=float, default=None)
    parser.add_argument(
        "--allow-above-active-floor-dispatch",
        action="store_true",
        help="allow dispatch above active floor when paired with an operator override",
    )
    parser.add_argument("--operator-override-reason", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_materializer_exact_eval_consumer_queue(
            repo_root=REPO_ROOT,
            bridge_report_paths=args.bridge_report,
            exact_ready_queue_paths=args.exact_ready_queue,
            experiment_queue_id=args.queue_id,
            provider=args.provider,
            max_concurrency=args.max_concurrency,
            estimated_cost_per_dispatch=args.estimated_cost_per_dispatch,
            max_total_cost=args.max_total_cost,
            label_prefix=args.label_prefix,
            agent=args.agent,
            dispatch_claims_path=args.dispatch_claims_path,
            active_floor_archive_bytes=args.active_floor_archive_bytes,
            active_floor_score=args.active_floor_score,
            allow_above_active_floor_dispatch=args.allow_above_active_floor_dispatch,
            operator_override_reason=args.operator_override_reason,
        )
        write_json(
            args.consumer_report_out,
            result["report"],
            overwrite=bool(args.overwrite),
        )
        write_json(
            args.experiment_queue_out,
            result["experiment_queue"],
            overwrite=bool(args.overwrite),
        )
        print(
            json.dumps(
                {
                    "schema": "materializer_exact_eval_consumer_cli_result.v1",
                    "consumer_report_out": str(Path(args.consumer_report_out)),
                    "experiment_queue_out": str(Path(args.experiment_queue_out)),
                    "authorized_candidate_count": result["report"][
                        "authorized_candidate_count"
                    ],
                    "blocked_candidate_count": result["report"][
                        "blocked_candidate_count"
                    ],
                    "experiment_count": result["report"]["experiment_count"],
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
    except (ExperimentQueueError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
