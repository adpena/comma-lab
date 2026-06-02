#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile SNeRV SNAR1 binary/package byte attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.snerv_binary_profile import (  # noqa: E402
    DEFAULT_FRONTIER_BYTES,
    build_snerv_binary_profile,
    write_snerv_binary_profile,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="SNeRV archive.zip containing 0.bin or raw .snar packet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to stdout only.",
    )
    parser.add_argument(
        "--frontier-bytes",
        type=int,
        default=DEFAULT_FRONTIER_BYTES,
        help="Reference frontier archive bytes for rate-only comparison.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output is not None:
        profile = write_snerv_binary_profile(
            input_path=args.input_path,
            output_path=args.output,
            frontier_bytes=args.frontier_bytes,
        )
    else:
        profile = build_snerv_binary_profile(
            input_path=args.input_path,
            frontier_bytes=args.frontier_bytes,
        )
    print(json.dumps(profile, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
