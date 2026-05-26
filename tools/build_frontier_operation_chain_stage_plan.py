#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a queue-owned staged plan from a multisurface chain work order."""

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
from comma_lab.scheduler.frontier_rate_attack_feedback import (  # noqa: E402
    FrontierRateAttackFeedbackError,
    build_frontier_operation_chain_compiler_stage_plan,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation-chain-compiler-work-orders",
        required=True,
        type=Path,
    )
    parser.add_argument("--source-operation-id", required=True)
    parser.add_argument("--stage-plan-out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        work_orders_path = args.operation_chain_compiler_work_orders
        if not work_orders_path.is_absolute():
            work_orders_path = REPO_ROOT / work_orders_path
        work_orders = json.loads(work_orders_path.read_text(encoding="utf-8"))
        if not isinstance(work_orders, dict):
            raise FrontierRateAttackFeedbackError(
                "operation chain compiler work orders must be a JSON object"
            )
        stage_plan = build_frontier_operation_chain_compiler_stage_plan(
            operation_chain_compiler_work_orders=work_orders,
            source_operation_id=args.source_operation_id,
        )
        stage_plan_out = args.stage_plan_out
        if not stage_plan_out.is_absolute():
            stage_plan_out = REPO_ROOT / stage_plan_out
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if stage_plan_out.exists() and args.overwrite:
            existing_text = stage_plan_out.read_text(encoding="utf-8")
            next_text = json_text(stage_plan)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(stage_plan_out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                stage_plan_out,
                stage_plan,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
    except (
        ArtifactWriteError,
        ExperimentQueueError,
        FrontierRateAttackFeedbackError,
        OSError,
        ValueError,
    ) as exc:
        print(f"FATAL: operation chain stage plan failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "frontier_rate_attack_operation_chain_stage_plan_cli_result.v1",
                "source_operation_id": args.source_operation_id,
                "stage_plan_out": str(args.stage_plan_out),
                "bytes_written": (
                    write_result.bytes_written if write_result is not None else 0
                ),
                "skipped_identical_existing_artifact": (
                    skipped_identical_existing_artifact
                ),
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
