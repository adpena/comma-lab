#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run MLX scorer responses using archive bytes from a local CPU advisory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.local_acceleration.mlx_scorer_response import (  # noqa: E402
    build_mlx_scorer_response_payload,
    write_mlx_scorer_response_payload,
)

_AUTHORITY_FALSE_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "promotable",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-cpu-advisory", required=True, type=Path)
    parser.add_argument("--reference-cache-dir", required=True, type=Path)
    parser.add_argument("--candidate-cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--batch-pairs", type=int, default=1)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--allow-gpu-research-signal", action="store_true")
    parser.add_argument("--allow-batch-shape-research-signal", action="store_true")
    parser.add_argument("--allow-local-cpu-advisory-cache-identity", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--components-dir", type=Path, default=None)
    parser.add_argument("--response-family")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    advisory = _load_json_object(args.local_cpu_advisory)
    archive_size = _archive_size_from_local_advisory(advisory, args.local_cpu_advisory)
    payload = build_mlx_scorer_response_payload(
        reference_cache_dir=args.reference_cache_dir,
        candidate_cache_dir=args.candidate_cache_dir,
        archive_size_bytes=archive_size,
        repo_root=args.repo_root,
        batch_pairs=args.batch_pairs,
        device_type=args.device,
        components_dir=args.components_dir,
        progress_every=args.progress_every,
        start_pair=args.start_pair,
        max_pairs=args.max_pairs,
        allow_gpu_research_signal=args.allow_gpu_research_signal,
        allow_batch_shape_research_signal=args.allow_batch_shape_research_signal,
        allow_local_cpu_advisory_cache_identity=args.allow_local_cpu_advisory_cache_identity,
        response_family=args.response_family,
    )
    payload["source_local_cpu_advisory"] = {
        "path": str(args.local_cpu_advisory),
        "archive_size_bytes": archive_size,
        "score_axis": advisory.get("score_axis"),
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
                "promotable": payload["promotable"],
            },
            sort_keys=True,
        )
    )
    return 0


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return payload


def _archive_size_from_local_advisory(payload: dict[str, Any], path: Path) -> int:
    if payload.get("score_axis") != "cpu_advisory":
        raise SystemExit(f"{path}: score_axis must be cpu_advisory")
    if payload.get("evidence_semantics") != "non_contest_cpu_auth_eval_advisory":
        raise SystemExit(f"{path}: evidence_semantics must be non_contest_cpu_auth_eval_advisory")
    for field in _AUTHORITY_FALSE_FIELDS:
        if payload.get(field) is not False:
            raise SystemExit(f"{path}: {field} must be exactly false")
    archive_size = payload.get("archive_size_bytes")
    if isinstance(archive_size, bool) or not isinstance(archive_size, int) or archive_size <= 0:
        raise SystemExit(f"{path}: archive_size_bytes must be a positive integer")
    return archive_size


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, NotImplementedError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
