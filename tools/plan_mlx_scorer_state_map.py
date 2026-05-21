#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit the upstream scorer PyTorch-to-MLX state-dict layout map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.local_acceleration.mlx_scorer_state_map import (
    build_upstream_scorer_state_map,
    write_state_map,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Load upstream safetensors on CPU before emitting the map.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = build_upstream_scorer_state_map(
        repo_root=args.repo_root,
        load_weights=args.load_weights,
    )
    if args.output is not None:
        write_state_map(payload, args.output)
    print(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "weights_loaded": payload["weights_loaded"],
                "summary": payload["summary"],
                "output": str(args.output) if args.output is not None else None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
