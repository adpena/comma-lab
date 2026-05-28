#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a frame-1 region waterfill receiver patch from P18/P19 rows."""

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
    build_frame1_region_waterfill_runtime_patch,
)
from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402


def _rgb_delta(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--rgb-delta must have three comma-separated ints")
    return (parts[0], parts[1], parts[2])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-submission-dir", required=True, type=Path)
    parser.add_argument("--segnet-region-waterfill", required=True, type=Path)
    parser.add_argument("--output-submission-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--candidate-archive", type=Path)
    parser.add_argument("--candidate-archive-source")
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--regions-per-pair", type=int, default=1)
    parser.add_argument("--rgb-delta", type=_rgb_delta, default=(-1, -1, -1))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_frame1_region_waterfill_runtime_patch(
            repo_root=REPO_ROOT,
            source_submission_dir=args.source_submission_dir,
            segnet_region_waterfill=args.segnet_region_waterfill,
            output_submission_dir=args.output_submission_dir,
            candidate_archive=args.candidate_archive,
            candidate_archive_source=args.candidate_archive_source,
            max_pairs=args.max_pairs,
            regions_per_pair=args.regions_per_pair,
            rgb_delta=args.rgb_delta,
            overwrite=args.overwrite,
        )
        output = _resolve(args.output_manifest)
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
        print(f"FATAL: frame1 region waterfill runtime patch failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "frame1_region_waterfill_runtime_patch_cli_result.v1",
                "output_manifest": str(args.output_manifest),
                "bytes_written": write.bytes_written,
                "output_submission_dir": payload["output_submission_dir"],
                "patched_pair_count": payload["patched_pair_count"],
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
