#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare local MLX scorer signal against byte-closed auth-eval ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.local_acceleration.mlx_scorer_fidelity import (
    MLXScorerFidelityThresholds,
    build_mlx_scorer_training_signal_fidelity_manifest,
    load_json_object,
    write_fidelity_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-payload", required=True, type=Path)
    parser.add_argument("--auth-eval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-score-abs-delta", type=float, default=1.0e-3)
    parser.add_argument("--max-seg-contribution-abs-delta", type=float, default=5.0e-4)
    parser.add_argument("--max-pose-contribution-abs-delta", type=float, default=5.0e-4)
    parser.add_argument("--max-rate-contribution-abs-delta", type=float, default=0.0)
    parser.add_argument(
        "--expected-n-samples",
        type=int,
        default=600,
        help="Expected sample count; pass 0 to disable for subset calibration.",
    )
    parser.add_argument(
        "--allow-missing-inflated-output-identity",
        action="store_true",
        help="Allow archive-SHA-only calibration when inflated aggregate custody is unavailable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    thresholds = MLXScorerFidelityThresholds(
        max_score_abs_delta=args.max_score_abs_delta,
        max_seg_contribution_abs_delta=args.max_seg_contribution_abs_delta,
        max_pose_contribution_abs_delta=args.max_pose_contribution_abs_delta,
        max_rate_contribution_abs_delta=args.max_rate_contribution_abs_delta,
        expected_n_samples=None if args.expected_n_samples == 0 else args.expected_n_samples,
        require_inflated_output_identity=not args.allow_missing_inflated_output_identity,
    )
    manifest = build_mlx_scorer_training_signal_fidelity_manifest(
        load_json_object(args.mlx_payload),
        load_json_object(args.auth_eval),
        thresholds=thresholds,
        run_id=args.run_id,
    )
    write_fidelity_manifest(manifest, args.output)
    print(json.dumps({"passed": manifest["passed"], "verdict": manifest["verdict"]}, sort_keys=True))
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
