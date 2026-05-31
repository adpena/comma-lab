#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay a byte-closed Z8 candidate with the full-video local MLX scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from tac.local_acceleration.mlx_scorer_adapters import (
    load_mlx_distortion_scorer_adapter_from_upstream,
)
from tac.substrates.z8_hierarchical_predictive_coding.full_video_vjp_acquisition import (
    compute_full_video_mlx_distortion_replay,
    reconstruct_z8_archive_pairs_rgb255,
)


def _parse_hw(text: str) -> tuple[int, int]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--scorer-hw must be formatted as H,W")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("--scorer-hw dimensions must be positive")
    return height, width


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-archive-bin", required=True, type=Path)
    parser.add_argument("--reference-pairs-npy", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument("--pair-chunk-size", type=int, default=64)
    parser.add_argument("--rgb-value-range", type=float, default=255.0)
    parser.add_argument("--scorer-hw", type=_parse_hw, default=(384, 512))
    parser.add_argument(
        "--archive-zip",
        type=Path,
        default=None,
        help="Byte-closed archive.zip to charge for rate. Defaults to payload bytes.",
    )
    parser.add_argument(
        "--archive-rate-bytes",
        type=int,
        default=None,
        help="Explicit charged bytes. Mutually exclusive with --archive-zip.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="MLX scorer adapter device. Keep cpu unless the adapter explicitly supports more.",
    )
    return parser


def _resolve_rate_bytes(args: argparse.Namespace) -> tuple[int, str, str | None]:
    if args.archive_zip is not None and args.archive_rate_bytes is not None:
        raise ValueError("--archive-zip and --archive-rate-bytes are mutually exclusive")
    if args.archive_zip is not None:
        return (
            int(args.archive_zip.stat().st_size),
            "byte_closed_archive_zip",
            _sha256_file(args.archive_zip),
        )
    if args.archive_rate_bytes is not None:
        if args.archive_rate_bytes < 0:
            raise ValueError("--archive-rate-bytes must be >= 0")
        return int(args.archive_rate_bytes), "explicit_archive_rate_bytes", None
    return (
        int(args.candidate_archive_bin.stat().st_size),
        "z8hpc1_payload_bytes",
        None,
    )


def main() -> int:
    args = build_parser().parse_args()
    candidate_archive = args.candidate_archive_bin.read_bytes()
    reference_pairs = np.load(args.reference_pairs_npy)
    candidate_pairs = reconstruct_z8_archive_pairs_rgb255(candidate_archive)
    archive_rate_bytes, archive_rate_bytes_source, archive_zip_sha256 = _resolve_rate_bytes(args)
    mlx_scorer = load_mlx_distortion_scorer_adapter_from_upstream(
        args.upstream_dir,
        device=str(args.device),
    )
    report = compute_full_video_mlx_distortion_replay(
        reference_pairs_rgb=reference_pairs,
        candidate_pairs_rgb=candidate_pairs,
        mlx_scorer=mlx_scorer,
        archive_rate_bytes=archive_rate_bytes,
        archive_rate_bytes_source=archive_rate_bytes_source,
        rgb_value_range=float(args.rgb_value_range),
        scorer_hw=args.scorer_hw,
        pair_chunk_size=int(args.pair_chunk_size),
    )
    report.update(
        {
            "candidate_archive_path": args.candidate_archive_bin.as_posix(),
            "candidate_archive_sha256": hashlib.sha256(candidate_archive).hexdigest(),
            "reference_pairs_npy_path": args.reference_pairs_npy.as_posix(),
            "reference_pairs_npy_sha256": _sha256_file(args.reference_pairs_npy),
            "byte_closed_archive_zip_path": (
                args.archive_zip.as_posix() if args.archive_zip is not None else None
            ),
            "byte_closed_archive_zip_sha256": archive_zip_sha256,
            "upstream_dir": args.upstream_dir.as_posix(),
            "scorer_hw": [int(args.scorer_hw[0]), int(args.scorer_hw[1])],
            "pair_chunk_size": int(args.pair_chunk_size),
        }
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"replay_z8_full_video_mlx_candidate failed: {exc}", file=sys.stderr)
        raise
