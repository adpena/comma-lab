#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build NumPy scorer-input cache from an inflated contest .raw file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.local_acceleration.mlx_preprocess import (
    count_video_pairs,
    load_raw_video_memmap,
    non_overlapping_pair_indices,
    write_scorer_input_cache_from_raw_file,
    write_scorer_input_cache_from_video_file,
    write_scorer_input_cache_hash_manifest_from_raw_file,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw", type=Path, help="Inflated RGB .raw file.")
    source.add_argument("--video", type=Path, help="Upstream-format ground-truth video file.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--inflated-outputs-aggregate-sha256")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--allow-large-tensor-cache",
        action="store_true",
        help=(
            "Allow eager full .npy tensor cache writes for large raw surfaces. "
            "Without this flag, large surfaces fail closed; use --hash-only for "
            "auth-axis identity artifacts."
        ),
    )
    parser.add_argument(
        "--large-cache-pair-threshold",
        type=int,
        default=64,
        help="Maximum eager full-cache pair count allowed without --allow-large-tensor-cache.",
    )
    parser.add_argument(
        "--hash-only",
        action="store_true",
        help="Write only manifest.json with streamed scorer-input array hashes.",
    )
    parser.add_argument("--batch-pairs", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.max_pairs is not None and args.max_pairs < 1:
        raise SystemExit("--max-pairs must be >= 1")
    if args.batch_pairs < 1:
        raise SystemExit("--batch-pairs must be >= 1")
    if args.hash_only:
        if args.video is not None:
            raise SystemExit("--hash-only currently supports --raw only")
        if args.max_pairs is not None:
            raise SystemExit("--hash-only does not support --max-pairs; hash the full auth surface")
        manifest = write_scorer_input_cache_hash_manifest_from_raw_file(
            args.raw,
            args.output_dir / "manifest.json",
            archive_sha256=args.archive_sha256,
            inflated_outputs_aggregate_sha256=args.inflated_outputs_aggregate_sha256,
            batch_pairs=args.batch_pairs,
        )
    else:
        _refuse_unacknowledged_large_tensor_cache(
            raw_path=args.raw,
            video_path=args.video,
            max_pairs=args.max_pairs,
            threshold=args.large_cache_pair_threshold,
            allow_large_tensor_cache=args.allow_large_tensor_cache,
        )
        if args.raw is not None:
            manifest = write_scorer_input_cache_from_raw_file(
                args.raw,
                args.output_dir,
                archive_sha256=args.archive_sha256,
                inflated_outputs_aggregate_sha256=args.inflated_outputs_aggregate_sha256,
                max_pairs=args.max_pairs,
            )
        else:
            manifest = write_scorer_input_cache_from_video_file(
                args.video,
                args.output_dir,
                archive_sha256=args.archive_sha256,
                inflated_outputs_aggregate_sha256=args.inflated_outputs_aggregate_sha256,
                max_pairs=args.max_pairs,
                batch_pairs=args.batch_pairs,
            )
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "manifest.json"),
                "pair_count": manifest["pair_count"],
                "segnet_last_rgb_shape": manifest["segnet_last_rgb_shape"],
                "posenet_yuv6_pair_shape": manifest["posenet_yuv6_pair_shape"],
            },
            sort_keys=True,
        )
    )
    return 0


def _refuse_unacknowledged_large_tensor_cache(
    *,
    raw_path: Path | None,
    video_path: Path | None,
    max_pairs: int | None,
    threshold: int,
    allow_large_tensor_cache: bool,
) -> None:
    if threshold < 1:
        raise SystemExit("--large-cache-pair-threshold must be >= 1")
    if raw_path is not None:
        raw = load_raw_video_memmap(raw_path)
        total_pairs = len(non_overlapping_pair_indices(raw.shape[0]))
        requested_pairs = min(total_pairs, int(max_pairs)) if max_pairs is not None else total_pairs
    elif video_path is not None:
        requested_pairs = int(max_pairs) if max_pairs is not None else count_video_pairs(video_path)
    else:
        raise SystemExit("one of --raw or --video is required")
    if requested_pairs > threshold and not allow_large_tensor_cache:
        raise SystemExit(
            "refusing eager full MLX scorer-input tensor cache for "
            f"{requested_pairs} pairs (> threshold {threshold}); use --hash-only "
            "for auth-axis identity or pass --allow-large-tensor-cache explicitly"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
