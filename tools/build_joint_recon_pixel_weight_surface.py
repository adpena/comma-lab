#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a joint P18/P19 recon_pixel_weight artifact for MLX carriers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.optimization.recon_pixel_weight_surface import (
    JointReconPixelWeightConfig,
    write_joint_p18_p19_recon_pixel_weight_artifact,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--source-video-path",
        default=Path("upstream/videos/0.mkv"),
        type=Path,
    )
    parser.add_argument("--upstream-dir", default=Path("upstream"), type=Path)
    parser.add_argument("--num-pairs", default=2, type=int)
    parser.add_argument("--pair-chunk-size", default=2, type=int)
    parser.add_argument("--scorer-device", default="cpu")
    parser.add_argument("--d-pose-operating-point", default=3.4e-5, type=float)
    parser.add_argument("--seg-weight", default=100.0, type=float)
    parser.add_argument("--pose-axis-count", default=6, type=int)
    parser.add_argument(
        "--pose-inverse-variance",
        default="1,1,1,1,1,1",
        help="Comma-separated positive inverse-variance weights.",
    )
    parser.add_argument("--seg-margin-delta", default=1.0, type=float)
    parser.add_argument("--weight-floor-fraction", default=0.05, type=float)
    parser.add_argument(
        "--normalize",
        default="mean",
        choices=("mean", "none"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    pose_inverse_variance = tuple(
        float(item.strip())
        for item in str(args.pose_inverse_variance).split(",")
        if item.strip()
    )
    config = JointReconPixelWeightConfig(
        num_pairs=int(args.num_pairs),
        pair_chunk_size=int(args.pair_chunk_size),
        d_pose_operating_point=float(args.d_pose_operating_point),
        seg_weight=float(args.seg_weight),
        pose_axis_count=int(args.pose_axis_count),
        pose_inverse_variance=pose_inverse_variance,
        seg_margin_delta=float(args.seg_margin_delta),
        weight_floor_fraction=float(args.weight_floor_fraction),
        normalize=str(args.normalize),
    )
    manifest = write_joint_p18_p19_recon_pixel_weight_artifact(
        output_dir=args.output_dir,
        source_video_path=args.source_video_path,
        upstream_dir=args.upstream_dir,
        config=config,
        scorer_device=str(args.scorer_device),
        allow_overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "manifest_path": manifest["manifest_path"],
                "weight_path": manifest["weight_path"],
                "weight_sha256": manifest["weight_sha256"],
                "weight_bytes": manifest["weight_bytes"],
                "training_consumption_recommended": manifest["metadata"][
                    "training_consumption_recommended"
                ],
                "blockers": manifest["metadata"]["blockers"],
                "score_claim": manifest["score_claim"],
                "ready_for_exact_eval_dispatch": manifest[
                    "ready_for_exact_eval_dispatch"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
