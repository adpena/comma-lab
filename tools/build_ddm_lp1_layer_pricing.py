#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize the SHA-bound DDM #669(b+c) layer-pricing receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.optimization.ddm_lp1_layer_pricing import materialize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = materialize(args.config, args.output_dir)
    summary = result["c1_corrected_waterfill"]
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "corrected_measured_allocated_bytes": summary[
                    "corrected_measured_allocated_bytes"
                ],
                "unallocated_headroom_bytes": summary["unallocated_headroom_bytes"],
                "sense_rows": result["costate_sense"]["row_count"],
                "score_claim": result["score_claim"],
                "pointer_moved": result["pointer_moved"],
                "main_review_required": result["main_review_required"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
