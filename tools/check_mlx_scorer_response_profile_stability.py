#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check local MLX scorer-response profile stability across batches/devices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.local_acceleration.mlx_profile_stability import (
    MLXProfileStabilityThresholds,
    build_profile_stability_manifest,
    load_json_object,
    write_profile_stability_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--baseline-device", default=None)
    parser.add_argument("--baseline-batch-pairs", type=int, default=None)
    parser.add_argument("--max-score-abs-delta", type=float, default=1.0e-5)
    parser.add_argument("--max-posenet-avg-abs-delta", type=float, default=1.0e-8)
    parser.add_argument("--max-segnet-avg-abs-delta", type=float, default=1.0e-8)
    parser.add_argument(
        "--require-component-sha-match",
        action="store_true",
        help="Treat component array SHA drift as a blocker instead of a warning.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = build_profile_stability_manifest(
        load_json_object(args.profile),
        thresholds=MLXProfileStabilityThresholds(
            max_score_abs_delta=args.max_score_abs_delta,
            max_posenet_avg_abs_delta=args.max_posenet_avg_abs_delta,
            max_segnet_avg_abs_delta=args.max_segnet_avg_abs_delta,
            require_component_sha_match=args.require_component_sha_match,
        ),
        baseline_device=args.baseline_device,
        baseline_batch_pairs=args.baseline_batch_pairs,
        run_id=args.run_id,
    )
    write_profile_stability_manifest(manifest, args.output)
    print(json.dumps({"passed": manifest["passed"], "verdict": manifest["verdict"]}, sort_keys=True))
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
