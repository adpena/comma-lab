#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a P19 PoseNet-null pair-selection artifact from selector codes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.scorer_region_waterfill import (  # noqa: E402
    ScorerRegionWaterfillError,
    build_p19_posenet_null_pairs,
)
from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-submission-dir", required=True, type=Path)
    parser.add_argument("--pose-null-modes-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--null-fraction", type=float, default=0.10)
    parser.add_argument("--exclude-identity", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_p19_posenet_null_pairs(
            repo_root=REPO_ROOT,
            source_submission_dir=args.source_submission_dir,
            pose_null_modes_artifact=args.pose_null_modes_artifact,
            null_fraction=args.null_fraction,
            include_identity=not args.exclude_identity,
        )
        output = _resolve(args.output)
        expected_existing_sha256 = (
            sha256_file(output) if output.is_file() and args.overwrite else None
        )
        write = write_json_artifact(
            output,
            payload,
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=expected_existing_sha256,
        )
    except (
        ArtifactWriteError,
        OSError,
        ScorerRegionWaterfillError,
        ValueError,
    ) as exc:
        print(f"FATAL: P19 PoseNet-null pair build failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "p19_posenet_null_pair_detection_cli_result.v1",
                "output": str(args.output),
                "bytes_written": write.bytes_written,
                "selected_pair_count": payload["selected_pair_count"],
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
