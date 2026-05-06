#!/usr/bin/env python3
"""Audit reverse_engineering/ for clean custody and promotion decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.reverse_engineering import (  # noqa: E402
    audit_reverse_engineering_tree,
    blocking_records,
    load_release_resolution_rules,
    release_blocking_records,
    render_json,
    render_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--reverse-root", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on raw/manual-review blockers.")
    parser.add_argument(
        "--release-strict",
        action="store_true",
        help="Exit nonzero on any unresolved promotion, ledger, externalization, or manual-review disposition.",
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        help="Machine-readable release manifest that resolves release-strict dispositions.",
    )
    parser.add_argument("--summary", action="store_true", help="Print only compact counts to stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    reverse_root = None if args.reverse_root is None else args.reverse_root.resolve()
    records = audit_reverse_engineering_tree(repo_root, reverse_root=reverse_root)
    release_rules = load_release_resolution_rules(args.release_manifest)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            render_json(records, release_strict=args.release_strict, release_rules=release_rules),
            encoding="utf-8",
        )
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(records), encoding="utf-8")
    blockers = (
        release_blocking_records(records, release_rules=release_rules)
        if args.release_strict
        else blocking_records(records)
    )
    if args.summary:
        print(
            "reverse-engineering audit: "
            f"files={len(records)} blockers={len(blockers)}"
        )
    elif not args.json_out and not args.md_out:
        print(render_markdown(records), end="")
    if (args.strict or args.release_strict) and blockers:
        mode = "release-strict" if args.release_strict else "strict"
        print(f"reverse-engineering audit found {len(blockers)} {mode} blocker(s)")
        for record in blockers[:20]:
            print(f"- {record.relpath}: {record.disposition}: {record.reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
