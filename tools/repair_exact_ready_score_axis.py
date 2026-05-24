#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Copy-repair legacy exact-ready queues that only lack score axis metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimizer.exact_ready_audit import discover_exact_ready_queues  # noqa: E402
from tac.optimizer.exact_ready_axis_repair import (  # noqa: E402
    DEFAULT_SCORE_AXIS,
    repair_exact_ready_score_axis_queues,
)
from tac.repo_io import json_text, write_json  # noqa: E402

DEFAULT_SCAN_ROOTS = (Path("experiments/results"), Path(".omx/research"))
DEFAULT_REPAIRED_OUTPUT_MARKER = ".omx/research/exact_ready_score_axis_repair_"


def _is_repair_output_path(path: Path, *, repo_root: Path) -> bool:
    try:
        rel = path if not path.is_absolute() else path.relative_to(repo_root)
    except ValueError:
        rel = path
    return DEFAULT_REPAIRED_OUTPUT_MARKER in rel.as_posix()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--queue", type=Path, action="append", default=[])
    parser.add_argument("--scan-root", type=Path, action="append", default=None)
    parser.add_argument(
        "--pattern",
        action="append",
        default=["**/exact_ready_queue.json", "**/*exact_ready_queue.json"],
    )
    parser.add_argument(
        "--dispatch-claims-path",
        type=Path,
        default=Path(".omx/state/active_lane_dispatch_claims.md"),
    )
    parser.add_argument("--score-axis", default=DEFAULT_SCORE_AXIS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument(
        "--include-repaired-output-queues",
        action="store_true",
        help="Include prior exact_ready_score_axis_repair_* outputs in default scans.",
    )
    parser.add_argument(
        "--write-repaired-queues",
        action="store_true",
        help="Actually write copied repaired queues. Without this, emit a plan only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    queue_paths = list(args.queue)
    if not queue_paths:
        queue_paths = discover_exact_ready_queues(
            repo_root=repo_root,
            scan_root=args.scan_root or DEFAULT_SCAN_ROOTS,
            patterns=args.pattern,
        )
        if not args.include_repaired_output_queues:
            queue_paths = [
                path
                for path in queue_paths
                if not _is_repair_output_path(path, repo_root=repo_root)
            ]
    missing = [
        path
        for path in queue_paths
        if not (path if path.is_absolute() else repo_root / path).is_file()
    ]
    if missing:
        parser.error("queue path(s) missing: " + ", ".join(path.as_posix() for path in missing))

    try:
        report = repair_exact_ready_score_axis_queues(
            queue_paths,
            repo_root=repo_root,
            out_dir=args.out_dir,
            dispatch_claims_path=args.dispatch_claims_path,
            score_axis=args.score_axis,
            write_repaired_queues=args.write_repaired_queues,
            overwrite=args.overwrite,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.report_out is not None:
        report_path = args.report_out if args.report_out.is_absolute() else repo_root / args.report_out
        write_json(report_path, report)
    else:
        sys.stdout.write(json_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
