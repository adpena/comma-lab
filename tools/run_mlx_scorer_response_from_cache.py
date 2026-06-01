#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run an MLX scorer response directly from scorer-input caches."""

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

from tac.local_acceleration.mlx_scorer_response import (  # noqa: E402
    build_mlx_scorer_response_payload,
    write_mlx_scorer_response_payload,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache-dir", required=True, type=Path)
    parser.add_argument("--candidate-cache-dir", required=True, type=Path)
    archive_group = parser.add_mutually_exclusive_group(required=True)
    archive_group.add_argument("--archive", type=Path)
    archive_group.add_argument("--archive-size-bytes", type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        help=(
            "Explicit upstream scorer snapshot directory. Defaults to "
            "<repo-root>/upstream. Use this for SSD worktrees whose code checkout "
            "is separate from the canonical upstream snapshot."
        ),
    )
    parser.add_argument("--batch-pairs", type=int, default=1)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--allow-gpu-research-signal", action="store_true")
    parser.add_argument("--allow-batch-shape-research-signal", action="store_true")
    parser.add_argument("--allow-unaudited-candidate-cache-debug", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--components-dir", type=Path)
    parser.add_argument("--response-family")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    archive_size = (
        args.archive.resolve().stat().st_size
        if args.archive is not None
        else int(args.archive_size_bytes)
    )
    payload = build_mlx_scorer_response_payload(
        reference_cache_dir=args.reference_cache_dir,
        candidate_cache_dir=args.candidate_cache_dir,
        archive_size_bytes=archive_size,
        repo_root=args.repo_root,
        upstream_dir=args.upstream_dir,
        batch_pairs=args.batch_pairs,
        device_type=args.device,
        components_dir=args.components_dir,
        progress_every=args.progress_every,
        start_pair=args.start_pair,
        max_pairs=args.max_pairs,
        allow_gpu_research_signal=args.allow_gpu_research_signal,
        allow_batch_shape_research_signal=args.allow_batch_shape_research_signal,
        allow_unaudited_candidate_cache_debug=args.allow_unaudited_candidate_cache_debug,
        response_family=args.response_family,
    )
    payload["source_cache_run"] = {
        "schema": "mlx_scorer_response_from_cache_run.v1",
        "archive_size_bytes": archive_size,
        "candidate_cache_dir": str(args.candidate_cache_dir),
        "reference_cache_dir": str(args.reference_cache_dir),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    write_mlx_scorer_response_payload(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_samples": payload["n_samples"],
                "canonical_score": payload["canonical_score"],
                "archive_size_bytes": archive_size,
                "device": args.device,
                "score_claim": payload["score_claim"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
