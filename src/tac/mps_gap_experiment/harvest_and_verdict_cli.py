# SPDX-License-Identifier: MIT
"""CLI wrapper around the split-device harvest helpers.

Two subcommands:

* ``reference`` — runs :func:`compute_local_mps_reference_components` on
  Apple Silicon Mac MPS hardware (typically invoked by the harness's
  Phase 1.5 fallback when the in-training auto-capture failed silently).
* ``diff`` — runs :func:`diff_components_and_classify_verdict` on the
  pre-captured LOCAL MPS + REMOTE CUDA component JSONs (typically invoked
  by the harness's Phase 5 after the Modal harvest landed).

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192/#317: every artifact
this CLI produces is research-signal-only, NEVER promotable, NEVER a
contest-axis score claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.mps_gap_experiment.harvest_and_verdict import (
    compute_local_mps_reference_components,
    diff_components_and_classify_verdict,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    ref = subparsers.add_parser(
        "reference",
        help="capture the LOCAL MPS reference components from a trained checkpoint",
    )
    ref.add_argument("--checkpoint", type=Path, required=True)
    ref.add_argument("--frame-cache", type=Path, required=True)
    ref.add_argument("--output-dir", type=Path, required=True)
    ref.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    ref.add_argument(
        "--include-scorer",
        action="store_true",
        help="include SegNet + PoseNet mean-output rows",
    )
    ref.add_argument(
        "--upstream-dir",
        type=Path,
        default=Path("upstream"),
        help="required when --include-scorer is set",
    )

    diff = subparsers.add_parser(
        "diff",
        help="diff a pre-captured local + target component pair into gap_results.json",
    )
    diff.add_argument(
        "--local",
        type=Path,
        required=True,
        help="path to local_mps_components.json from compute_local_mps_reference_components",
    )
    diff.add_argument(
        "--target",
        type=Path,
        required=True,
        help="path to target_cuda_components.json harvested from Modal",
    )
    diff.add_argument(
        "--output",
        type=Path,
        required=True,
        help="canonical gap_results.json output path",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "reference":
        components_path = compute_local_mps_reference_components(
            checkpoint_path=args.checkpoint,
            frame_cache_path=args.frame_cache,
            output_dir=args.output_dir,
            device=args.device,
            include_scorer_components=args.include_scorer,
            upstream_dir=args.upstream_dir if args.include_scorer else None,
        )
        print(
            f"[mps_gap_experiment] LOCAL MPS reference components written to "
            f"{components_path} [MPS-research-signal]"
        )
        return 0
    if args.cmd == "diff":
        manifest = diff_components_and_classify_verdict(
            local_mps_components_path=args.local,
            target_cuda_components_path=args.target,
            output_path=args.output,
        )
        print(
            f"[mps_gap_experiment] split-device gap manifest written to "
            f"{args.output} (verdict={manifest.verdict}, "
            f"gap_relative_aggregate={manifest.gap_relative_aggregate:.6f}) "
            f"[MPS-research-signal vs diagnostic-CUDA Modal A10G]"
        )
        return 0
    parser.error(f"unknown subcommand: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
