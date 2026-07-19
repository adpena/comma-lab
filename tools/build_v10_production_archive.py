#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a scorer-free V10 production ``archive.zip`` from charged y planes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from tac.witness_dsl.v10_production_receiver import (
    ProductionReceiverError,
    build_production_archive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--y-npy", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--camera-height", type=int, required=True)
    parser.add_argument("--camera-width", type=int, required=True)
    parser.add_argument(
        "--y-codec",
        choices=("raw-uint8-y", "brotli-y", "witness-y-stub"),
        default="raw-uint8-y",
    )
    parser.add_argument("--quotient-residual-npy", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        y_planes = np.load(args.y_npy, allow_pickle=False)
        residual = (
            None if args.quotient_residual_npy is None else np.load(args.quotient_residual_npy, allow_pickle=False)
        )
        result = build_production_archive(
            y_planes,
            archive_path=args.archive_dir / "archive.zip",
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            y_codec_id=args.y_codec,
            quotient_residual=residual,
            manifest_path=args.manifest,
        )
    except (OSError, ValueError, ProductionReceiverError) as exc:
        raise SystemExit(f"V10 production archive build refused: {exc}") from exc
    print(
        json.dumps(
            {
                "archive_path": str(result.archive_path),
                "archive_bytes": result.archive_bytes,
                "archive_sha256": result.archive_sha256,
                "packet_bytes": result.packet_bytes,
                "packet_sha256": result.packet_sha256,
                "manifest_path": str(result.manifest_path),
                "launch_ready": False,
                "score_claim": False,
                "promotion_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
