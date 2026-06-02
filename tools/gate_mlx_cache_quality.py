#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed quality gate for MLX scorer input caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.mlx_cache_quality_gate import (  # noqa: E402
    MLX_CACHE_QUALITY_GATE_SCHEMA,
    write_mlx_cache_quality_gate,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cache-dir", required=True, type=Path)
    parser.add_argument("--reference-cache-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--sample-pairs", type=int, default=16)
    parser.add_argument("--min-segnet-std", type=float, default=1.0)
    parser.add_argument("--min-segnet-dynamic-range", type=float, default=16.0)
    parser.add_argument(
        "--max-segnet-mae-vs-reference-for-fit-gate",
        type=float,
        default=64.0,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = write_mlx_cache_quality_gate(
        output_json=args.output_json,
        candidate_cache_dir=args.candidate_cache_dir,
        reference_cache_dir=args.reference_cache_dir,
        sample_pairs=int(args.sample_pairs),
        min_segnet_std=float(args.min_segnet_std),
        min_segnet_dynamic_range=float(args.min_segnet_dynamic_range),
        max_segnet_mae_vs_reference_for_fit_gate=float(
            args.max_segnet_mae_vs_reference_for_fit_gate
        ),
    )
    print(
        json.dumps(
            {
                "schema": MLX_CACHE_QUALITY_GATE_SCHEMA,
                "report_path": args.output_json.expanduser()
                .resolve(strict=False)
                .as_posix(),
                "verdict": report["verdict"],
                "candidate_cache_nondegenerate": report[
                    "candidate_cache_nondegenerate"
                ],
                "fit_gate_passed": report["fit_gate_passed"],
                "score_claim": report["score_claim"],
                "ready_for_exact_eval_dispatch": report[
                    "ready_for_exact_eval_dispatch"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
