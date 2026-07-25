#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize the SHA-bound DDM LA1 #669(b+c) real-coder receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.optimization.ddm_la1_layer_assignment_context_pricing import materialize


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
                "verdict": result["verdict"],
                "context_mass_fraction": result["residual_vs_context"]["context_winning_typed_mass_fraction"],
                "la1_bytes": result["layer_assignment"]["rehomed_real_coder_bytes"],
                "net_delta_vs_130789": result["coordination"]["net_composed_delta_vs_130789_bytes"],
                "score_claim": result["score_claim"],
                "pointer_moved": result["pointer_moved"],
                "main_review_required": result["main_landing_review_required"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
