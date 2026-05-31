#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a queue-owned Z8 full-video VJP plan or surface bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.substrates.z8_hierarchical_predictive_coding.full_video_vjp_acquisition import (
    Z8FullVideoVjpAcquisitionConfig,
    assemble_z8_full_video_vjp_surface_bundle,
    load_z8_full_video_vjp_surface_shard_file,
    write_z8_full_video_vjp_acquisition_plan,
    write_z8_full_video_vjp_surface_bundle,
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
    return parser


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
    if not args.shard:
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
