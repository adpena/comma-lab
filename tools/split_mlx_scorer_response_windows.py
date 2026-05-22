#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Split one singleton MLX scorer-response payload into per-window rows."""

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

from tac.local_acceleration.mlx_response_windows import (  # noqa: E402
    MLXResponseWindowSplitError,
    load_distortion_components_from_response,
    split_mlx_scorer_response_windows,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--index-out", required=True, type=Path)
    parser.add_argument("--components-dir", type=Path)
    parser.add_argument("--window-pairs", type=int, default=1)
    parser.add_argument("--stride-pairs", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--prefix", default="window")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        payload = json.loads(args.response.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise MLXResponseWindowSplitError("response must be a JSON object")
        pose, seg = load_distortion_components_from_response(payload)
        index = split_mlx_scorer_response_windows(
            response_payload=payload,
            posenet_distortion=pose,
            segnet_distortion=seg,
            output_dir=args.output_dir,
            window_pairs=args.window_pairs,
            stride_pairs=args.stride_pairs,
            max_windows=args.max_windows,
            prefix=args.prefix,
            components_dir=args.components_dir,
        )
    except (OSError, json.JSONDecodeError, ValueError, MLXResponseWindowSplitError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    args.index_out.parent.mkdir(parents=True, exist_ok=True)
    args.index_out.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "index_out": str(args.index_out),
                "output_dir": str(args.output_dir),
                "score_claim": False,
                "window_count": index["window_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
