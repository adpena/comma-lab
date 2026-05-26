#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a queue-owned autonomous many-op chain work order."""

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
    build_frontier_autonomous_chain_work_order,
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
        "--autonomous-chain-optimization",
        required=True,
        type=Path,
    )
    parser.add_argument("--chain-id", required=True)
    parser.add_argument("--work-order-out", required=True, type=Path)
    parser.add_argument("--child-queue-artifact-path", action="append", default=[])
    parser.add_argument("--missing-queue-artifact-key", action="append", default=[])
    parser.add_argument("--queue-actuation-ready", action="store_true")
    parser.add_argument("--post-repair-refresh-planned", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source_path = _resolve_repo_path(args.autonomous_chain_optimization)
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise FrontierRateAttackFeedbackError(
                "autonomous chain optimization must be a JSON object"
            )
        work_order = build_frontier_autonomous_chain_work_order(
            autonomous_chain_optimization=payload,
            chain_id=args.chain_id,
            child_queue_artifact_paths=tuple(args.child_queue_artifact_path),
            missing_queue_artifact_keys=tuple(args.missing_queue_artifact_key),
            queue_actuation_ready=bool(args.queue_actuation_ready),
            post_repair_refresh_planned=bool(args.post_repair_refresh_planned),
        )
        work_order_out = _resolve_repo_path(args.work_order_out)
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if work_order_out.exists() and args.overwrite:
            existing_text = work_order_out.read_text(encoding="utf-8")
            next_text = json_text(work_order)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(work_order_out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                work_order_out,
                work_order,
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
        print(f"FATAL: autonomous chain work order failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "frontier_rate_attack_autonomous_chain_work_order_cli_result.v1",
                "chain_id": args.chain_id,
                "work_order_out": str(args.work_order_out),
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
