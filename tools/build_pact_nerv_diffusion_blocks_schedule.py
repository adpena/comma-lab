#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a fail-closed Pact-NeRV DiffusionBlocks schedule manifest."""

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

from comma_lab.scheduler.pact_nerv_diffusion_blocks_queue import (  # noqa: E402
    PactNervDiffusionBlocksQueueError,
    build_pact_nerv_diffusion_blocks_schedule,
)
from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-count", type=int, default=3)
    parser.add_argument(
        "--difficulty-mass-source",
        default="scorer_region_waterfill_and_master_gradient",
    )
    parser.add_argument("--overlap-fraction", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        schedule = build_pact_nerv_diffusion_blocks_schedule(
            block_count=args.block_count,
            difficulty_mass_source=args.difficulty_mass_source,
            overlap_fraction=args.overlap_fraction,
        )
        output = _resolve(args.output)
        expected_existing_sha256 = (
            sha256_file(output) if output.is_file() and args.overwrite else None
        )
        write = write_json_artifact(
            output,
            schedule,
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=expected_existing_sha256,
        )
    except (ArtifactWriteError, OSError, PactNervDiffusionBlocksQueueError) as exc:
        print(f"FATAL: pact-nerv DiffusionBlocks schedule failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "pact_nerv_diffusion_blocks_schedule_cli_result.v1",
                "output": str(args.output),
                "bytes_written": write.bytes_written,
                "block_count": schedule["block_count"],
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
