#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build NumPy scorer-input cache from an inflated contest .raw file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.local_acceleration.mlx_preprocess import (
    write_scorer_input_cache_from_raw_file,
    write_scorer_input_cache_hash_manifest_from_raw_file,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path, help="Inflated RGB .raw file.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--inflated-outputs-aggregate-sha256")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--hash-only",
        action="store_true",
        help="Write only manifest.json with streamed scorer-input array hashes.",
    )
    parser.add_argument("--batch-pairs", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.hash_only:
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
        manifest = write_scorer_input_cache_from_raw_file(
            args.raw,
            args.output_dir,
            archive_sha256=args.archive_sha256,
            inflated_outputs_aggregate_sha256=args.inflated_outputs_aggregate_sha256,
            max_pairs=args.max_pairs,
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
