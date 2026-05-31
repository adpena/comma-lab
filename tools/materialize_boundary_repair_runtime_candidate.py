#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize an archive-bound SegNet boundary repair runtime candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.boundary_repair_runtime_materializer import (  # noqa: E402
    BoundaryRepairMaterializerError,
    materialize_boundary_repair_runtime_candidate,
)
from tac.repo_io import json_text  # noqa: E402


def _shape(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("shape must contain comma-separated integers")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--surface", type=Path)
    parser.add_argument("--base-submission-dir", required=True, type=Path)
    parser.add_argument("--base-archive", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=("source_pixel_patch", "masked_local_median"),
    )
    parser.add_argument("--source-raw", type=Path)
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--video-name", default="0.mkv")
    parser.add_argument("--raw-shape", type=_shape, default=(1200, 874, 1164, 3))
    parser.add_argument("--grid-shape", type=_shape, default=(384, 512))
    parser.add_argument("--max-grid-pixels", type=int, default=2048)
    parser.add_argument("--max-raw-points", type=int, default=16384)
    parser.add_argument("--postfilter-radius", type=int, default=1)
    parser.add_argument("--expected-receiver-output-bytes", type=int)
    parser.add_argument("--retain-receiver-output", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = materialize_boundary_repair_runtime_candidate(
            bridge_path=args.bridge,
            surface_path=args.surface,
            base_submission_dir=args.base_submission_dir,
            base_archive_path=args.base_archive,
            output_dir=args.output_dir,
            repo_root=REPO_ROOT,
            strategy=args.strategy,
            candidate_id=args.candidate_id,
            source_raw_path=args.source_raw,
            source_video_path=args.source_video,
            video_name=args.video_name,
            raw_shape=args.raw_shape,
            grid_shape=args.grid_shape,
            max_grid_pixels=args.max_grid_pixels,
            max_raw_points=args.max_raw_points,
            postfilter_radius=args.postfilter_radius,
            expected_receiver_output_bytes=args.expected_receiver_output_bytes,
            retain_receiver_output=args.retain_receiver_output,
            timeout_seconds=args.timeout_seconds,
        )
    except (
        BoundaryRepairMaterializerError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"FATAL: boundary repair materialization failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "boundary_repair_runtime_materializer_cli_result.v1",
                "candidate_id": manifest["candidate_id"],
                "strategy": manifest["strategy"],
                "manifest_path": str(
                    args.output_dir / "boundary_repair_materializer_manifest.json"
                ),
                "candidate_archive": manifest["candidate_archive"],
                "receiver_contract_satisfied": manifest["receiver_contract_satisfied"],
                "runtime_consumption_proof_ready": manifest[
                    "runtime_consumption_proof_ready"
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
