#!/usr/bin/env python3
"""Build a planning-only HNeRV section repack target table."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_section_repack import build_section_repack_plan, render_markdown  # noqa: E402
from tac.repo_io import json_text, read_json  # noqa: E402

DEFAULT_SCORECARD = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "public_hnerv_frontier_payload_profiles_20260504_codex"
    / "scorecard.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--label", action="append", help="Restrict to one label; repeatable.")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scorecard = read_json(args.scorecard)
    plan = build_section_repack_plan(scorecard, labels=args.label)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(plan), encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(plan), encoding="utf-8")
    if not args.json_out and not args.md_out:
        if args.format == "markdown":
            print(render_markdown(plan), end="")
        else:
            print(json_text(plan), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
