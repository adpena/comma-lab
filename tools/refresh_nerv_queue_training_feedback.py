#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Refresh planner feedback from queue-owned NeRV training telemetry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import (  # noqa: E402
    connect_state,
    default_state_path,
    initialize_queue_state,
    load_queue_definition,
    queue_summary,
)
from tac.analysis.nerv_queue_training_feedback_refresh import (  # noqa: E402
    refresh_nerv_queue_training_feedback,
    write_nerv_queue_training_feedback_refresh,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--include-status",
        action="append",
        default=None,
        help="Queue step status to harvest. Repeatable. Default: running.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    queue = load_queue_definition(args.queue)
    state = args.state or default_state_path(REPO_ROOT, queue["queue_id"])
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        summary = queue_summary(conn, queue, repo_root=REPO_ROOT)
    report = refresh_nerv_queue_training_feedback(
        queue=queue,
        queue_path=args.queue,
        queue_summary=summary,
        output_dir=args.output_dir,
        include_statuses=tuple(args.include_status or ("running",)),
    )
    write_result = write_nerv_queue_training_feedback_refresh(
        report=report,
        output_json=args.output_json,
        output_jsonl=args.output_jsonl,
        output_md=args.output_md,
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "queue_id": report["queue_id"],
                "refreshed_row_count": report["refreshed_row_count"],
                "skipped_count": report["skipped_count"],
                "output_json": write_result["report_path"],
                "output_jsonl": write_result.get("jsonl_path"),
                "output_md": write_result.get("markdown_path"),
                "score_claim": report["score_claim"],
                "ready_for_exact_eval_dispatch": report[
                    "ready_for_exact_eval_dispatch"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
