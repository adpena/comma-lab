#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize the bounded DDM #669c 25-row solved-value pricing receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.optimization.ddm_dm1_solved_value_pricing import materialize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = materialize(args.config, args.output_dir)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "row_count": result["row_count"],
                "joint_bytes": result["joint_shared_context"]["exact_counted_bytes"],
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
