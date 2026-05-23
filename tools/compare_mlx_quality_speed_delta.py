#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a false-authority MLX quality/speed delta manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.local_acceleration.mlx_quality_speed_delta import (
    build_quality_speed_delta_manifest,
    load_json_object,
    write_quality_speed_delta_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--mlx-response", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-summary", type=Path, default=None)
    parser.add_argument("--frontier-score", type=float, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--calibration-safety-factor", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        calibration = (
            load_json_object(args.calibration_summary)
            if args.calibration_summary is not None
            else None
        )
        manifest = build_quality_speed_delta_manifest(
            anchor_payload=load_json_object(args.anchor),
            mlx_payloads=[load_json_object(path) for path in args.mlx_response],
            anchor_path=args.anchor,
            mlx_paths=args.mlx_response,
            calibration_payload=calibration,
            calibration_path=args.calibration_summary,
            run_id=args.run_id,
            frontier_score=args.frontier_score,
            calibration_safety_factor=args.calibration_safety_factor,
        )
    except (OSError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    write_quality_speed_delta_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "row_count": manifest["summary"]["row_count"],
                "all_rows_blocked_for_spend_triage": manifest["summary"][
                    "all_rows_blocked_for_spend_triage"
                ],
                "smallest_abs_score_delta_row": manifest["summary"][
                    "smallest_abs_score_delta_row"
                ],
                "score_claim": manifest["score_claim"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
