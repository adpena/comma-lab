#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the false-authority HiNeRV distortion-stabilization DAG gate."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.hinerv_distortion_stabilization_queue import (  # noqa: E402
    DEFAULT_LANE_ID,
    DEFAULT_MIN_FREE_BYTES,
    build_hinerv_distortion_stabilization_queue,
    load_json_with_source_identity,
    render_hinerv_distortion_stabilization_queue_markdown,
)
from tac.repo_io import write_json_artifact, write_text_artifact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-feedback-row",
        action="append",
        default=[],
        type=Path,
        help="HiNeRV nerv_candidate_feedback_row.v1 JSON. Repeatable.",
    )
    parser.add_argument(
        "--checkpoint-export-report",
        action="append",
        default=[],
        type=Path,
        help="hinerv_checkpoint_archive_export.v1 JSON. Repeatable.",
    )
    parser.add_argument(
        "--waterfill-report",
        action="append",
        default=[],
        type=Path,
        help="hinerv_archive_ladder_waterfill.v1 JSON. Repeatable.",
    )
    parser.add_argument(
        "--replay-actuator-report",
        action="append",
        default=[],
        type=Path,
        help="hinerv_archive_ladder_replay_actuator.v1 JSON. Repeatable.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    parser.add_argument("--queue-id", default="hinerv_distortion_stabilization_queue.v1")
    parser.add_argument("--min-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    parser.add_argument(
        "--allow-local-output",
        action="store_true",
        help="Allow non-SSD output root. Intended only for tests.",
    )
    args = parser.parse_args(argv)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = (
        args.output_root
        or Path("/Volumes/VertigoDataTier/pact")
        / f"hinerv_distortion_stabilization_queue_{stamp}"
    )
    output_json = (
        args.output_json or output_root / "hinerv_distortion_stabilization_queue.json"
    )
    output_md = args.output_md or output_root / "hinerv_distortion_stabilization_queue.md"
    report = build_hinerv_distortion_stabilization_queue(
        candidate_feedback_rows=[
            load_json_with_source_identity(path) for path in args.candidate_feedback_row
        ],
        checkpoint_export_reports=[
            load_json_with_source_identity(path) for path in args.checkpoint_export_report
        ],
        waterfill_reports=[
            load_json_with_source_identity(path) for path in args.waterfill_report
        ],
        replay_actuator_reports=[
            load_json_with_source_identity(path) for path in args.replay_actuator_report
        ],
        output_root=output_root,
        lane_id=str(args.lane_id),
        queue_id=str(args.queue_id),
        min_free_bytes=int(args.min_free_bytes),
        allow_local_output=bool(args.allow_local_output),
    )
    json_result = write_json_artifact(output_json, report)
    md_result = write_text_artifact(
        output_md,
        render_hinerv_distortion_stabilization_queue_markdown(report),
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "lane_id": report["lane_id"],
                "output_json": json_result.path,
                "output_json_sha256": json_result.sha256,
                "output_md": md_result.path,
                "output_md_sha256": md_result.sha256,
                "dag_node_count": report["dag_node_count"],
                "blocked_dag_node_count": report["blocked_dag_node_count"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
