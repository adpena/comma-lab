#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a claim-then-dispatch experiment queue from materializer exact-ready rows.

The default output is dry-run only. Use ``--dispatch-mode execute`` together
with ``--allow-paid-dispatch-queue`` to emit real dispatch steps; even then the
generated queue runs a lane-claim step before each dispatch step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from comma_lab.scheduler.materializer_exact_eval_dispatch_plan import (  # noqa: E402
    build_materializer_exact_eval_dispatch_plan,
    write_json,
)
from tac.optimizer.exact_readiness import (  # noqa: E402
    ACTIVE_FLOOR_ARCHIVE_BYTES,
    ACTIVE_FLOOR_SCORE,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--bridge-report", type=Path, default=None)
    parser.add_argument("--exact-ready-queue", type=Path, action="append", default=[])
    parser.add_argument("--dispatch-plan-out", type=Path, required=True)
    parser.add_argument("--experiment-queue-out", type=Path, required=True)
    parser.add_argument(
        "--experiment-queue-id",
        default="materializer_exact_eval_dispatch_queue",
    )
    parser.add_argument("--provider", choices=["lightning", "vastai"], default="lightning")
    parser.add_argument(
        "--dispatch-mode",
        choices=["dry_run", "execute"],
        default="dry_run",
    )
    parser.add_argument(
        "--allow-paid-dispatch-queue",
        action="store_true",
        help="Required with --dispatch-mode execute.",
    )
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--estimated-cost-per-dispatch", type=float, default=0.30)
    parser.add_argument("--max-total-cost", type=float, default=5.00)
    parser.add_argument("--label-prefix", default="materializer_exact_eval")
    parser.add_argument("--agent", default="codex")
    parser.add_argument("--dispatch-claims-path", type=Path, default=None)
    parser.add_argument(
        "--active-floor-archive-bytes",
        type=int,
        default=ACTIVE_FLOOR_ARCHIVE_BYTES,
    )
    parser.add_argument(
        "--active-floor-score",
        type=float,
        default=ACTIVE_FLOOR_SCORE,
    )
    parser.add_argument("--disable-active-floor-archive-bytes", action="store_true")
    parser.add_argument("--disable-active-floor-score", action="store_true")
    parser.add_argument("--allow-above-active-floor-dispatch", action="store_true")
    parser.add_argument("--operator-override-reason", default=None)
    parser.add_argument(
        "--require-authorized",
        action="store_true",
        help="Exit nonzero if no candidate is authorized into the dispatch queue.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting generated JSON outputs. Defaults fail closed.",
    )
    args = parser.parse_args(argv)

    if args.bridge_report is None and not args.exact_ready_queue:
        raise SystemExit("provide --bridge-report or at least one --exact-ready-queue")
    if (
        args.disable_active_floor_archive_bytes or args.disable_active_floor_score
    ) and not args.operator_override_reason:
        raise SystemExit(
            "disabling active floors requires --operator-override-reason"
        )
    active_floor_archive_bytes = (
        None
        if args.disable_active_floor_archive_bytes
        else args.active_floor_archive_bytes
    )
    active_floor_score = (
        None if args.disable_active_floor_score else args.active_floor_score
    )

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=args.repo_root,
        bridge_report_path=args.bridge_report,
        exact_ready_queue_paths=args.exact_ready_queue,
        experiment_queue_id=args.experiment_queue_id,
        dispatch_mode=args.dispatch_mode,
        allow_paid_dispatch_queue=args.allow_paid_dispatch_queue,
        provider=args.provider,
        max_concurrency=args.max_concurrency,
        estimated_cost_per_dispatch=args.estimated_cost_per_dispatch,
        max_total_cost=args.max_total_cost,
        label_prefix=args.label_prefix,
        agent=args.agent,
        dispatch_claims_path=args.dispatch_claims_path,
        active_floor_archive_bytes=active_floor_archive_bytes,
        active_floor_score=active_floor_score,
        allow_above_active_floor_dispatch=args.allow_above_active_floor_dispatch,
        operator_override_reason=args.operator_override_reason,
    )
    write_json(args.dispatch_plan_out, result["plan"], overwrite=args.overwrite)
    write_json(
        args.experiment_queue_out,
        result["experiment_queue"],
        overwrite=args.overwrite,
    )
    plan = result["plan"]
    print(
        f"planned materializer exact-eval dispatch rows: "
        f"authorized={plan['authorized_candidate_count']} "
        f"blocked={plan['blocked_candidate_count']} "
        f"mode={plan['dispatch_mode']} queue={args.experiment_queue_out}"
    )
    if args.require_authorized and (
        int(plan["authorized_candidate_count"]) < 1 or plan.get("plan_blockers")
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
