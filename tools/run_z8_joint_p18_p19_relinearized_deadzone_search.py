#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run bounded relinearized Z8 joint P18/P19 coefficient dead-zone search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.substrates.z8_hierarchical_predictive_coding.joint_coefficient_waterfill import (
    Z8JointCoefficientRelinearizationSearchConfig,
    load_joint_p18_p19_surface_file,
    materialize_joint_p18_p19_relinearized_deadzone_search,
)


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-bin", required=True, type=Path)
    parser.add_argument(
        "--surface",
        required=True,
        action="append",
        type=Path,
        help="Fresh joint surface per iteration; repeat for relinearization.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--joint-weight-quantiles",
        type=_float_tuple,
        default=_float_tuple("0.20,0.35,0.50"),
    )
    parser.add_argument(
        "--coefficient-deadzone-quantiles",
        type=_float_tuple,
        default=_float_tuple("0.25,0.50,0.75"),
    )
    parser.add_argument(
        "--quantization-steps",
        type=_float_tuple,
        default=_float_tuple("0.0039215686,0.0078431373,0.0156862745"),
    )
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-cumulative-mse", type=float, default=None)
    parser.add_argument("--rate-weight", type=float, default=25.0)
    parser.add_argument("--distortion-weight", type=float, default=10_000.0)
    parser.add_argument("--interaction-penalty-weight", type=float, default=10_000.0)
    parser.add_argument("--allow-reused-surface", action="store_true")
    parser.add_argument(
        "--allow-stale-surfaces",
        action="store_true",
        help=(
            "Allow surfaces whose embedded linearization_archive_sha does not match "
            "the current archive. Exact full-video relinearization keeps this off."
        ),
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--allow-without-pose-null-mask", action="store_true")
    parser.add_argument(
        "--allow-broadcast-surface",
        action="store_true",
        help="Allow exploratory pair-broadcast surfaces; exact acquisition keeps this off.",
    )
    parser.add_argument("--no-archive-zip", action="store_true")
    parser.add_argument("--emit-receiver-proof", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    surfaces = [load_joint_p18_p19_surface_file(path) for path in args.surface]
    config = Z8JointCoefficientRelinearizationSearchConfig(
        joint_weight_quantiles=args.joint_weight_quantiles,
        coefficient_deadzone_quantiles=args.coefficient_deadzone_quantiles,
        quantization_steps=args.quantization_steps,
        max_iterations=args.max_iterations,
        max_cumulative_mse=args.max_cumulative_mse,
        rate_weight=args.rate_weight,
        distortion_weight=args.distortion_weight,
        interaction_penalty_weight=args.interaction_penalty_weight,
        require_fresh_surface_per_iteration=not args.allow_reused_surface,
        pose_null_required=not args.allow_without_pose_null_mask,
        max_pairs=args.max_pairs,
        emit_archive_zip=not args.no_archive_zip,
        emit_receiver_proof=args.emit_receiver_proof,
        require_full_video_surface_coverage=not args.allow_broadcast_surface,
        require_surface_archive_freshness=not args.allow_stale_surfaces,
    )
    manifest = materialize_joint_p18_p19_relinearized_deadzone_search(
        args.archive_bin.read_bytes(),
        args.output_dir,
        surfaces=surfaces,
        config=config,
        repo_root=args.repo_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
