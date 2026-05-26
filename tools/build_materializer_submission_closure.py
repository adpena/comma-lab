#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a byte-closed submission/runtime closure for a materializer candidate."""

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

from tac.optimizer.materializer_submission_closure import (  # noqa: E402
    MaterializerSubmissionClosureError,
    build_materializer_submission_runtime_closure,
)
from tac.repo_io import json_text  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-queue", required=True, type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--source-runtime-dir", type=Path)
    parser.add_argument("--submission-dir-out", required=True, type=Path)
    parser.add_argument("--closed-source-queue-out", required=True, type=Path)
    parser.add_argument("--closure-report-out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_materializer_submission_runtime_closure(
            repo_root=REPO_ROOT,
            source_queue_path=args.source_queue,
            candidate_id=args.candidate_id,
            source_runtime_dir=args.source_runtime_dir,
            submission_dir_out=args.submission_dir_out,
            closed_source_queue_out=args.closed_source_queue_out,
            closure_report_out=args.closure_report_out,
            overwrite=args.overwrite,
        )
    except (
        MaterializerSubmissionClosureError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"FATAL: materializer submission closure failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "materializer_submission_closure_cli_result.v1",
                "closure_report_out": str(args.closure_report_out),
                "closed_source_queue_out": str(args.closed_source_queue_out),
                "submission_dir_out": str(args.submission_dir_out),
                "candidate_id": report.get("candidate_id"),
                "archive_bytes": report.get("archive_bytes"),
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
