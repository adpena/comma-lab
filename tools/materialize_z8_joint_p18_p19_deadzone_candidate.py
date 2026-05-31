#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a Z8 joint P18/P19 coefficient dead-zone archive candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.substrates.z8_hierarchical_predictive_coding.joint_coefficient_waterfill import (
    Z8JointCoefficientWaterfillConfig,
    load_joint_p18_p19_surface_file,
    materialize_joint_p18_p19_deadzone_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-bin", required=True, type=Path)
    parser.add_argument("--surface", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--joint-weight-quantile", type=float, default=0.35)
    parser.add_argument("--coefficient-deadzone-quantile", type=float, default=0.50)
    parser.add_argument("--quantization-step", type=float, default=1.0 / 255.0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--allow-without-pose-null-mask", action="store_true")
    parser.add_argument("--no-archive-zip", action="store_true")
    parser.add_argument("--emit-receiver-proof", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    joint_weight, safe_mask = load_joint_p18_p19_surface_file(args.surface)
    config = Z8JointCoefficientWaterfillConfig(
        joint_weight_quantile=args.joint_weight_quantile,
        coefficient_deadzone_quantile=args.coefficient_deadzone_quantile,
        quantization_step=args.quantization_step,
        pose_null_required=not args.allow_without_pose_null_mask,
        max_pairs=args.max_pairs,
        emit_archive_zip=not args.no_archive_zip,
        emit_receiver_proof=args.emit_receiver_proof,
    )
    manifest = materialize_joint_p18_p19_deadzone_candidate(
        args.archive_bin.read_bytes(),
        args.output_dir,
        joint_weight=joint_weight,
        rate_attack_deadzone_mask=safe_mask,
        config=config,
        repo_root=args.repo_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
