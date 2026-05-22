#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a decision-quality calibration table for MLX scorer responses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.local_acceleration.mlx_score_calibration import (
    build_mlx_score_calibration_manifest,
    write_mlx_score_calibration_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="JSON object with a rows array, or a JSON array of row objects.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("rows")
            run_id = args.run_id or payload.get("run_id")
        else:
            rows = payload
            run_id = args.run_id
        if not isinstance(rows, list):
            raise ValueError("--input must contain a rows array")
        manifest = build_mlx_score_calibration_manifest(
            rows,
            repo_root=args.repo_root,
            run_id=None if run_id is None else str(run_id),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    write_mlx_score_calibration_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "row_count": manifest["row_count"],
                "mlx_cpu_rank_inversions": manifest["summary"].get(
                    "mlx_cpu_rank_inversions"
                ),
                "mlx_cuda_rank_inversions": manifest["summary"].get(
                    "mlx_cuda_rank_inversions"
                ),
                "score_claim": manifest["score_claim"],
                "promotion_eligible": manifest["promotion_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
