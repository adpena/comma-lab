#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run/resume the local-only DDM MS4D direct scorer-metric completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.optimization.ddm_ms4d_direct_completion import materialize

REPO = Path(__file__).resolve().parents[1]
RUN_ID = "ddm_ms4d_direct_metric_completion_20260724T155932Z"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / ".omx/research/configs/ddm_ms4d_direct_metric_completion_20260724.json",
    )
    parser.add_argument(
        "--bulk-output",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact") / RUN_ID,
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=REPO / ".omx/research" / RUN_ID,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = materialize(
        args.config,
        bulk_output=args.bulk_output,
        receipt_output=args.receipt_output,
        repository_root=REPO,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
