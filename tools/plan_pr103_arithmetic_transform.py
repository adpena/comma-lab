#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan one fail-closed PR103 arithmetic transform target."""

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

from tac.pr103_arithmetic_transform_plan import (  # noqa: E402
    DEFAULT_STRATEGY,
    Pr103ArithmeticTransformPlanError,
    build_pr103_arithmetic_transform_plan,
    render_markdown,
)
from tac.repo_io import json_text, write_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-manifest", required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--target-label", help="Exact target stream label.")
    group.add_argument("--target-rank", type=int, help="One-based target rank.")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--fail-if-blocked",
        action="store_true",
        help="Exit 1 when the plan is not archive-preflight ready.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        plan = build_pr103_arithmetic_transform_plan(
            schema_manifest=args.schema_manifest,
            target_label=args.target_label,
            target_rank=args.target_rank,
            strategy=args.strategy,
            repo_root=REPO_ROOT,
        )
    except (OSError, Pr103ArithmeticTransformPlanError) as exc:
        print(f"FATAL: PR103 arithmetic transform plan failed: {exc}", file=sys.stderr)
        return 2

    plan = attach_tool_run_manifest(
        plan,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[args.schema_manifest],
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    if args.json_out is not None:
        write_json(args.json_out, plan)
    else:
        print(json_text(plan), end="")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(plan), encoding="utf-8")
    if args.fail_if_blocked and plan.get("ready_for_archive_preflight") is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
