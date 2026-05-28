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
    refresh_frontier_citation_surfaces,
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
    parser.add_argument(
        "--refresh-citation-surfaces",
        action="store_true",
        help=(
            "Update generated frontier citation blocks in reports/latest.md, "
            ".omx/state/current_focus.md, and .omx/state/next_experiments.md."
        ),
    )
    parser.add_argument(
        "--checked-at-utc",
        default=None,
        help="UTC timestamp to stamp into generated citation blocks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    refresh_payload = None
    if args.refresh_citation_surfaces:
        refresh_payload = refresh_frontier_citation_surfaces(
            repo_root,
            checked_at_utc=args.checked_at_utc,
        )
        payload = refresh_payload["after_payload"]
    else:
        payload = build_frontier_scan_payload(repo_root)
    if args.format == "json":
        if refresh_payload is not None:
            refresh_payload = dict(refresh_payload)
            refresh_payload.pop("after_payload", None)
            payload = {**payload, "citation_surface_refresh": refresh_payload}
        print(render_frontier_scan_json(payload))
    else:
        print(render_frontier_scan_text(payload))
        if refresh_payload is not None:
            changed = refresh_payload.get("changed")
            if isinstance(changed, dict):
                changed_paths = [
                    path
                    for path, row in sorted(changed.items())
                    if isinstance(row, dict) and row.get("changed")
                ]
                rendered = ", ".join(changed_paths) if changed_paths else "none"
                print(f"\nCitation surfaces refreshed: {rendered}")
    return 1 if args.check_drift and payload.get("drift") else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
