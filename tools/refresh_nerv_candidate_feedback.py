#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Refresh NeRV candidate feedback rows from existing runner reports.

This is a backfill/harvest tool for reports produced before the MLX
acquisition-vs-local-replay split was encoded. It recomputes MLX prefilter
coverage from file-backed profile paths, repairs stale prefilter blockers in a
feedback-only copy of the report, and writes false-authority feedback artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.analysis.nerv_candidate_feedback import (  # noqa: E402
    write_refreshed_nerv_candidate_feedback_files,
)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"runner report must be a JSON object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runner-report",
        required=True,
        type=Path,
        help="Existing compact_renderer_mlx_spine_runner_report.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for refreshed feedback artifacts. Defaults to "
            "<runner-report-dir>/refreshed_candidate_feedback."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used to resolve relative MLX profile paths.",
    )
    parser.add_argument(
        "--mlx-profile",
        action="append",
        default=[],
        type=Path,
        help=(
            "Additional MLX profile path. May be passed multiple times; report "
            "embedded mlx_profile_paths are also consumed."
        ),
    )
    parser.add_argument(
        "--required-pairs",
        type=int,
        default=600,
        help="Pair count required for full-video coverage.",
    )
    parser.add_argument(
        "--max-mlx-score-for-local-replay",
        type=float,
        default=0.5,
        help=(
            "Hard local replay threshold. Batched profiles can count for "
            "acquisition coverage but still fail replay unlock."
        ),
    )
    args = parser.parse_args(argv)

    report_path = args.runner_report.expanduser().resolve(strict=False)
    if not report_path.is_file():
        raise SystemExit(f"runner report missing: {report_path}")
    output_dir = (
        args.output_dir.expanduser().resolve(strict=False)
        if args.output_dir is not None
        else report_path.parent / "refreshed_candidate_feedback"
    )
    result = write_refreshed_nerv_candidate_feedback_files(
        runner_report=_load_json_object(report_path),
        output_dir=output_dir,
        repo_root=args.repo_root,
        source_report_path=report_path,
        mlx_profile_paths=tuple(args.mlx_profile),
        required_pairs=int(args.required_pairs),
        max_mlx_score_for_local_replay=float(args.max_mlx_score_for_local_replay),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
