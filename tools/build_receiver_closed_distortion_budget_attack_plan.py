#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Promote rate-only savings into a receiver-closed distortion-budget plan."""

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
    build_receiver_closed_distortion_budget_attack_plan,
)
from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-report", required=True, type=Path)
    parser.add_argument("--posenet-null-pairs", required=True, type=Path)
    parser.add_argument("--segnet-region-waterfill", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--score-bytes-denominator", type=int, default=37_545_489)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_receiver_closed_distortion_budget_attack_plan(
            repo_root=REPO_ROOT,
            chain_report=args.chain_report,
            posenet_null_pairs=args.posenet_null_pairs,
            segnet_region_waterfill=args.segnet_region_waterfill,
            score_bytes_denominator=args.score_bytes_denominator,
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
        print(f"FATAL: distortion-budget attack plan failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "receiver_closed_distortion_budget_attack_plan_cli_result.v1",
                "output": str(args.output),
                "bytes_written": write.bytes_written,
                "rate_saved_bytes": payload["rate_saved_bytes"],
                "budget_pair_count": payload["budget_pair_count"],
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
