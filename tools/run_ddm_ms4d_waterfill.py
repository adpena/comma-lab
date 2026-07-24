#!/usr/bin/env python3
"""Run the MS4D post-admission tolerance-waterfill gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for local_path in (str(SRC), str(REPO)):
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

from tac.optimization.ddm_ms4d_waterfill_admission import (  # noqa: E402
    build_post_admission_refusal,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=REPO
        / ".omx/research/ddm_ms4d_direct_metric_completion_20260724T155932Z/BUNDLE-COMPLETE.json",
    )
    parser.add_argument(
        "--rd1-duals",
        type=Path,
        default=REPO
        / (
            ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
            "typed_dimension_duals_effective_quantum.json"
        ),
    )
    parser.add_argument(
        "--rd1-frontier",
        type=Path,
        default=REPO
        / (
            ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
            "typed_R_D_frontier_rows_v5.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO
        / ".omx/research/ddm_ms4d_direct_metric_completion_20260724T155932Z",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    receipt, path = build_post_admission_refusal(
        bundle_path=args.bundle,
        rd1_duals_path=args.rd1_duals,
        rd1_frontier_path=args.rd1_frontier,
        output_root=args.output_root,
        repository_root=REPO,
    )
    print(path)
    print(receipt["verdict"])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
