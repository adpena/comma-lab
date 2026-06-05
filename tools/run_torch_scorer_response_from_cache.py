#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run upstream PyTorch CPU scorer response directly from scorer-input caches."""

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

from tac.local_acceleration.torch_scorer_response import (  # noqa: E402
    build_torch_cpu_scorer_response_payload,
    write_torch_cpu_scorer_response_payload,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache-dir", required=True, type=Path)
    parser.add_argument("--candidate-cache-dir", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--archive", type=Path)
    group.add_argument("--archive-size-bytes", type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--upstream-dir", type=Path)
    parser.add_argument("--batch-pairs", type=int, default=1)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--components-dir", type=Path)
    parser.add_argument("--response-family")
    parser.add_argument(
        "--allow-unaudited-candidate-cache-debug",
        action="store_true",
        help=(
            "Permit local diagnostic response scoring from a candidate cache "
            "without auth-eval identity custody. This remains false-authority."
        ),
    )
    parser.add_argument(
        "--allow-local-cpu-advisory-cache-identity",
        action="store_true",
        help=(
            "Permit matching local CPU advisory cache identity. This remains "
            "non-promotional and cannot replace exact eval."
        ),
    )
    parser.add_argument(
        "--cache-integrity-mode",
        choices=("strict", "manifest"),
        default="manifest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        archive_size_bytes = (
            int(args.archive.stat().st_size)
            if args.archive is not None
            else int(args.archive_size_bytes)
        )
        payload = build_torch_cpu_scorer_response_payload(
            reference_cache_dir=args.reference_cache_dir,
            candidate_cache_dir=args.candidate_cache_dir,
            archive_size_bytes=archive_size_bytes,
            repo_root=args.repo_root,
            upstream_dir=args.upstream_dir,
            batch_pairs=args.batch_pairs,
            start_pair=args.start_pair,
            max_pairs=args.max_pairs,
            components_dir=args.components_dir,
            progress_every=args.progress_every,
            response_family=args.response_family,
            allow_unaudited_candidate_cache_debug=(
                args.allow_unaudited_candidate_cache_debug
            ),
            allow_local_cpu_advisory_cache_identity=(
                args.allow_local_cpu_advisory_cache_identity
            ),
            cache_integrity_mode=args.cache_integrity_mode,
        )
    except (OSError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    write_torch_cpu_scorer_response_payload(payload, args.output)
    print(
        json.dumps(
            {
                "archive_size_bytes": payload["archive_size_bytes"],
                "canonical_score": payload["canonical_score"],
                "avg_posenet_dist": payload["avg_posenet_dist"],
                "avg_segnet_dist": payload["avg_segnet_dist"],
                "n_samples": payload["n_samples"],
                "output": str(args.output),
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
