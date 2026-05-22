#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit MLX-native rgb_to_yuv6 parity against upstream frame_utils."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.local_acceleration.mlx_yuv6_primitive_parity import (  # noqa: E402
    DEFAULT_EPSILON,
    build_mlx_yuv6_primitive_parity_manifest,
    deterministic_rgb_fixture,
    json_text,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--input-npy", type=Path, help="Optional NCHW RGB float/uint array.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--run-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rgb = (
            np.load(args.input_npy)
            if args.input_npy is not None
            else deterministic_rgb_fixture(
                seed=args.seed,
                batch=args.batch,
                height=args.height,
                width=args.width,
            )
        )
        manifest = build_mlx_yuv6_primitive_parity_manifest(
            rgb_chw=rgb,
            repo_root=args.repo_root,
            epsilon=args.epsilon,
            run_id=args.run_id,
        )
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json_text(manifest), encoding="utf-8")
    print(
        json_text(
            {
                "output": str(args.output),
                "passed": manifest["passed"],
                "verdict": manifest["verdict"],
                "max_abs_delta": manifest["deltas"]["max_abs_delta"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
