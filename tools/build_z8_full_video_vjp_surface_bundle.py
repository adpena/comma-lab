#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a queue-owned Z8 full-video VJP plan or surface bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.substrates.z8_hierarchical_predictive_coding.full_video_vjp_acquisition import (
    Z8FullVideoMlxVjpShardConfig,
    Z8FullVideoVjpAcquisitionConfig,
    assemble_z8_full_video_vjp_surface_bundle,
    build_z8_full_video_mlx_vjp_surface_shard,
    load_z8_full_video_vjp_surface_shard_file,
    write_z8_full_video_vjp_acquisition_plan,
    write_z8_full_video_vjp_surface_bundle,
    write_z8_full_video_vjp_surface_shard,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-bin", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--shard",
        action="append",
        type=Path,
        default=[],
        help=(
            "Archive-pinned VJP shard NPZ/JSON. If omitted, only the shard plan "
            "is written."
        ),
    )
    parser.add_argument("--target-mode", default="contest_video_overfit")
    parser.add_argument("--pair-chunk-size", type=int, default=64)
    parser.add_argument("--parallel-workers", type=int, default=None)
    parser.add_argument("--corpus-manifest-path", default=None)
    parser.add_argument("--disable-minibatch-probes", action="store_true")
    parser.add_argument("--allow-partial-production-probe-surface", action="store_true")
    parser.add_argument(
        "--reference-pairs-npy",
        type=Path,
        help="Full-video reference pairs, shape (pairs,2,H,W,3). Emits one MLX VJP shard.",
    )
    parser.add_argument(
        "--candidate-pairs-npy",
        type=Path,
        help="Full-video candidate pairs, shape (pairs,2,H,W,3). Emits one MLX VJP shard.",
    )
    parser.add_argument("--pair-start", type=int, default=None)
    parser.add_argument("--pair-end", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--full-video-d-pose", type=float, default=None)
    parser.add_argument("--rgb-value-range", type=float, default=255.0)
    parser.add_argument("--scorer-hw", default="384,512")
    parser.add_argument(
        "--pose-axis-count",
        type=int,
        default=6,
        help="Number of PoseNet output axes to VJP for true P19; default is contest first-six pose axes.",
    )
    parser.add_argument(
        "--pose-inverse-variance",
        default="1,1,1,1,1,1",
        help=(
            "Comma-separated inverse-variance weights for the Mahalanobis P19 "
            "norm. Identity matches upstream first-six pose MSE."
        ),
    )
    parser.add_argument("--seg-margin-delta", type=float, default=1.0)
    parser.add_argument("--pose-null-threshold", type=float, default=1e-8)
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=Path("upstream"),
        help="Upstream scorer model directory used when emitting MLX VJP shards.",
    )
    return parser


def _parse_hw(text: str) -> tuple[int, int]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--scorer-hw must be formatted as H,W")
    return int(parts[0]), int(parts[1])


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise ValueError("--pose-inverse-variance must contain at least one value")
    values = tuple(float(part) for part in parts)
    if any(value <= 0.0 for value in values):
        raise ValueError("--pose-inverse-variance entries must be positive")
    return values


def main() -> int:
    args = build_parser().parse_args()
    archive_bytes = args.archive_bin.read_bytes()
    config = Z8FullVideoVjpAcquisitionConfig(
        target_mode=args.target_mode,
        pair_chunk_size=args.pair_chunk_size,
        parallel_workers=args.parallel_workers,
        corpus_manifest_path=args.corpus_manifest_path,
        allow_minibatch_probe_between_full_passes=not args.disable_minibatch_probes,
        allow_partial_production_probe_surface=args.allow_partial_production_probe_surface,
    )
    if args.reference_pairs_npy or args.candidate_pairs_npy:
        if args.shard:
            raise SystemExit("--shard cannot be combined with --reference-pairs-npy/--candidate-pairs-npy")
        if not args.reference_pairs_npy or not args.candidate_pairs_npy:
            raise SystemExit("--reference-pairs-npy and --candidate-pairs-npy must be provided together")
        if args.pair_start is None or args.pair_end is None:
            raise SystemExit("--pair-start and --pair-end are required for MLX shard emission")
        if args.full_video_d_pose is None:
            raise SystemExit("--full-video-d-pose is required for exact full-video pose-term scaling")
        import numpy as np

        from tac.local_acceleration.mlx_scorer_adapters import (
            load_mlx_distortion_scorer_adapter_from_upstream,
        )

        reference_pairs = np.load(args.reference_pairs_npy)
        candidate_pairs = np.load(args.candidate_pairs_npy)
        mlx_scorer = load_mlx_distortion_scorer_adapter_from_upstream(args.upstream_dir, device="cpu")
        shard = build_z8_full_video_mlx_vjp_surface_shard(
            archive_bytes,
            reference_pairs_rgb=reference_pairs,
            candidate_pairs_rgb=candidate_pairs,
            mlx_scorer=mlx_scorer,
            config=Z8FullVideoMlxVjpShardConfig(
                shard_index=int(args.shard_index),
                pair_start=int(args.pair_start),
                pair_end=int(args.pair_end),
                full_video_pair_count=int(candidate_pairs.shape[0]),
                full_video_d_pose=float(args.full_video_d_pose),
                target_mode=args.target_mode,
                rgb_value_range=float(args.rgb_value_range),
                scorer_hw=_parse_hw(args.scorer_hw),
                pose_axis_count=int(args.pose_axis_count),
                pose_inverse_variance=_parse_float_tuple(args.pose_inverse_variance),
                seg_margin_delta=float(args.seg_margin_delta),
                pose_null_threshold=float(args.pose_null_threshold),
            ),
        )
        artifact = write_z8_full_video_vjp_surface_shard(shard, args.output_dir)
    elif not args.shard:
        artifact = write_z8_full_video_vjp_acquisition_plan(
            archive_bytes,
            args.output_dir,
            config=config,
        )
    else:
        shard_surfaces = [
            load_z8_full_video_vjp_surface_shard_file(path)
            for path in args.shard
        ]
        bundle = assemble_z8_full_video_vjp_surface_bundle(
            archive_bytes,
            shard_surfaces=shard_surfaces,
            config=config,
        )
        artifact = write_z8_full_video_vjp_surface_bundle(bundle, args.output_dir)
    print(json.dumps(artifact, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
