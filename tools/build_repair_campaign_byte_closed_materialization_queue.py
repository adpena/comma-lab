#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a byte-closed materialization queue from a repair score report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    _TOOL_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _TOOL_DIR.parent
    for _path in (str(_REPO_ROOT), str(_TOOL_DIR)):
        if _path not in sys.path:
            sys.path.insert(0, _path)
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.repair_campaign_materialization_queue import (  # noqa: E402
    RepairCampaignMaterializationQueueError,
    build_repair_campaign_byte_closed_materialization_queue,
)
from tac.optimization.repair_campaign_posterior import (  # noqa: E402
    DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH,
    DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-report", required=True, type=Path)
    parser.add_argument("--work-order", required=True, type=Path)
    parser.add_argument("--materialization-queue-out", required=True, type=Path)
    parser.add_argument(
        "--results-root",
        default=Path("experiments/results"),
        type=Path,
        help="Root used for generated materialization queue artifacts.",
    )
    parser.add_argument(
        "--queue-id",
        default="repair_campaign_byte_closed_materialization_queue",
        help="Queue id for the generated materialization queue.",
    )
    parser.add_argument("--experiment-limit", type=int, default=None)
    parser.add_argument("--materializer-work-queue", type=Path)
    parser.add_argument("--materializer-execution-queue", type=Path)
    parser.add_argument("--repair-palette-mode", action="append", default=[])
    parser.add_argument(
        "--posterior-path",
        type=Path,
        default=DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH,
    )
    parser.add_argument(
        "--posterior-lock-path",
        type=Path,
        default=DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        score_report_path = _resolve(args.score_report)
        score_report = json.loads(score_report_path.read_text(encoding="utf-8"))
        if not isinstance(score_report, dict):
            raise RepairCampaignMaterializationQueueError(
                "score report must be a JSON object"
            )
        queue = build_repair_campaign_byte_closed_materialization_queue(
            repo_root=REPO_ROOT,
            score_report=score_report,
            score_report_path=args.score_report,
            work_order_path=args.work_order,
            results_root=args.results_root,
            queue_id=args.queue_id,
            experiment_limit=args.experiment_limit,
            materializer_work_queue=args.materializer_work_queue,
            materializer_execution_queue=args.materializer_execution_queue,
            repair_palette_modes=tuple(args.repair_palette_mode),
            posterior_path=args.posterior_path,
            posterior_lock_path=args.posterior_lock_path,
        )
        queue_out = _resolve(args.materialization_queue_out)
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if queue_out.exists() and args.overwrite:
            existing_text = queue_out.read_text(encoding="utf-8")
            next_text = json_text(queue)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(queue_out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                queue_out,
                queue,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
    except (
        ArtifactWriteError,
        OSError,
        RepairCampaignMaterializationQueueError,
        ValueError,
    ) as exc:
        print(
            f"FATAL: repair materialization queue build failed: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        json_text(
            {
                "schema": "repair_campaign_byte_closed_materialization_queue_cli_result.v1",
                "score_report": str(args.score_report),
                "work_order": str(args.work_order),
                "materialization_queue_out": str(args.materialization_queue_out),
                "queue_id": queue["queue_id"],
                "posterior_path": str(args.posterior_path),
                "experiment_count": len(queue["experiments"]),
                "ready_experiment_count": queue["metadata"][
                    "ready_experiment_count"
                ],
                "blocked_experiment_count": queue["metadata"][
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
