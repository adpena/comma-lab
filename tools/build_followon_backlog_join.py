#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the registry-first follow-on backlog join report."""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)


def _parse_date(value: str | None) -> _dt.date | None:
    if value is None:
        return None
    try:
        return _dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", type=_parse_date, default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--max-dispositions", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    return parser


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    from tac.followon_backlog_join import (
        build_followon_backlog_join,
        render_markdown,
        write_json,
    )

    report = build_followon_backlog_join(
        repo_root=REPO,
        since=args.since,
        cache_ttl_s=None if args.no_cache else 6 * 3600.0,
        max_dispositions=args.max_dispositions,
    )
    json_path = _resolve(args.output_json)
    write_json(json_path, report)
    if args.output_md is not None:
        md_path = _resolve(args.output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = md_path.with_name(f".{md_path.name}.tmp")
        tmp.write_text(render_markdown(report), encoding="utf-8")
        tmp.replace(md_path)
    print(
        "[build_followon_backlog_join] OK: "
        f"{len(report['dispositions'])} dispositions, "
        f"{report['summaries']['unowned_queued_rows']} unowned queued rows -> {json_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
