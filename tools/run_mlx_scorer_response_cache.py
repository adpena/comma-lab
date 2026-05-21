#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run MLX PoseNet/SegNet responses from fixed scorer-input caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.local_acceleration.mlx_scorer_response import (
    build_mlx_scorer_response_payload,
    write_mlx_scorer_response_payload,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache-dir", required=True, type=Path)
    parser.add_argument("--candidate-cache-dir", required=True, type=Path)
    parser.add_argument("--archive-size-bytes", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--batch-pairs", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--components-dir",
        type=Path,
        default=None,
        help="Optional directory for per-pair PoseNet/SegNet distortion .npy arrays.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = build_mlx_scorer_response_payload(
        reference_cache_dir=args.reference_cache_dir,
        candidate_cache_dir=args.candidate_cache_dir,
        archive_size_bytes=args.archive_size_bytes,
        repo_root=args.repo_root,
        batch_pairs=args.batch_pairs,
        device_type=args.device,
        components_dir=args.components_dir,
    )
    write_mlx_scorer_response_payload(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_samples": payload["n_samples"],
                "avg_posenet_dist": payload["avg_posenet_dist"],
                "avg_segnet_dist": payload["avg_segnet_dist"],
                "canonical_score": payload["canonical_score"],
                "score_claim": payload["score_claim"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
