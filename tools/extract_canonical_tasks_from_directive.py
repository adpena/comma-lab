#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Extract ITEM sections from codex routing directives into canonical status."""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

ITEM_HEADER_RE = re.compile(r"^#{2,4}\s+ITEM\s+(\d+)\s*(?:[-—:]\s*)?(.*)$")
COST_RE = re.compile(r"predicted(?:_|\s+)cost(?:_|\s+)usd[^0-9-]*([-+]?\d+(?:\.\d+)?)", re.I)
DELTA_RE = re.compile(
    r"predicted(?:_|\s+)(?:delta|Δ)S(?:_|\s+)?(?:band)?[^-\d[]*"
    r"\[?\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]?",
    re.I,
)


def _expand_directives(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = glob.glob(value)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(value))
    return sorted({(REPO / path).resolve() if not path.is_absolute() else path.resolve() for path in paths})


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_float_match(regex: re.Pattern[str], text: str) -> float | None:
    match = regex.search(text)
    return None if match is None else float(match.group(1))


def _parse_delta_band(text: str) -> tuple[float, float] | None:
    match = DELTA_RE.search(text)
    if match is None:
        return None
    return (float(match.group(1)), float(match.group(2)))


def extract_tasks_from_directive(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    headers: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        match = ITEM_HEADER_RE.match(line.strip())
        if not match:
            continue
        item_num = int(match.group(1))
        title = match.group(2).strip() or f"ITEM {item_num}"
        headers.append((idx, item_num, title))
    tasks: list[dict[str, object]] = []
    for pos, (start_idx, item_num, title) in enumerate(headers):
        end_idx = headers[pos + 1][0] if pos + 1 < len(headers) else len(lines)
        section = "\n".join(lines[start_idx:end_idx])
        item_id = f"ITEM_{item_num}"
        from tac.canonical_task_status import task_id_for_memo_item

        tasks.append(
            {
                "task_id": task_id_for_memo_item(_relative(path), item_id),
                "source_design_memo": _relative(path),
                "item_id": item_id,
                "title": title,
                "predicted_cost_usd": _parse_float_match(COST_RE, section),
                "predicted_delta_s_band": _parse_delta_band(section),
            }
        )
    return tasks


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directive",
        nargs="+",
        default=[".omx/research/codex_routing_directive_*.md"],
        help="Directive paths or globs.",
    )
    parser.add_argument("--register-all", action="store_true")
    parser.add_argument("--owner", default="codex")
    parser.add_argument("--actor", default="codex_session_local")
    parser.add_argument("--session-id", default="codex_session_local")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from tac.canonical_task_status import register_task

    tasks: list[dict[str, object]] = []
    for directive in _expand_directives(args.directive):
        if not directive.exists():
            continue
        tasks.extend(extract_tasks_from_directive(directive))
    registered = []
    if args.register_all:
        for task in tasks:
            row = register_task(
                str(task["task_id"]),
                str(task["source_design_memo"]),
                str(task["title"]),
                args.owner,
                task["predicted_cost_usd"],  # type: ignore[arg-type]
                task["predicted_delta_s_band"],  # type: ignore[arg-type]
                actor=args.actor,
                session_id=args.session_id,
                repo_root=REPO,
                notes="registered_from_directive_extractor",
            )
            registered.append(row.to_json_obj())
    if args.json:
        import json

        print(json.dumps(registered if args.register_all else tasks, indent=2, sort_keys=True))
    else:
        print(f"extracted={len(tasks)} registered={len(registered)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

