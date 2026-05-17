#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Scan canonical state and report the best contest anchor per axis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.frontier_scan import (  # noqa: E402
    build_frontier_scan_payload,
    render_frontier_scan_json,
    render_frontier_scan_text,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--check-drift",
        action="store_true",
        help="Exit 1 if reports/latest.md is worse than canonical state.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_frontier_scan_payload(args.repo_root.resolve())
    if args.format == "json":
        print(render_frontier_scan_json(payload))
    else:
        print(render_frontier_scan_text(payload))
    return 1 if args.check_drift and payload.get("drift") else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
