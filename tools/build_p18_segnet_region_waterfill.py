#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a P18 SegNet-region waterfill artifact for P19-selected pairs."""

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
    build_p18_segnet_region_waterfill,
)
from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posenet-null-pairs", required=True, type=Path)
    parser.add_argument("--segnet-softmax-16", required=True, type=Path)
    parser.add_argument("--segnet-softmax-256", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-regions-per-pair", type=int, default=4)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_p18_segnet_region_waterfill(
            repo_root=REPO_ROOT,
            posenet_null_pairs=args.posenet_null_pairs,
            segnet_softmax_16=args.segnet_softmax_16,
            segnet_softmax_256=args.segnet_softmax_256,
            top_regions_per_pair=args.top_regions_per_pair,
            image_width=args.image_width,
            image_height=args.image_height,
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
        print(f"FATAL: P18 SegNet-region waterfill build failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "p18_segnet_region_waterfill_cli_result.v1",
                "output": str(args.output),
                "bytes_written": write.bytes_written,
                "selected_pair_count": payload["selected_pair_count"],
                "top_regions_per_pair": payload["top_regions_per_pair"],
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
