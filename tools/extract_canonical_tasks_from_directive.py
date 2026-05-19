#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Extract task sections from codex routing directives into canonical status."""

from __future__ import annotations

import argparse
import glob
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

ITEM_HEADER_RE = re.compile(r"^#{2,6}\s+ITEM\s+(\d+)\s*(?:[-—:]\s*)?(.*)$", re.I)
WIRE_IN_HEADER_RE = re.compile(
    r"^#{2,6}\s+WIRE-IN\s+#?(\d+)\s*(?:[-—:]\s*)?(.*)$",
    re.I,
)
BUILD_HEADER_RE = re.compile(r"^#{2,6}\s+BUILD\s+#?(\d+)\s*(?:[-—:]\s*)?(.*)$", re.I)
OP_HEADER_RE = re.compile(
    r"^#{2,6}\s+OP[-\s]?(\d+)(?:\s+FIRST)?\s*(?:[-—:]\s*)?(.*)$",
    re.I,
)
CLUSTER_HEADER_RE = re.compile(
    r"^#{2,6}\s+CLUSTER\s+([A-Z](?:\.\d+|\d*)?)\s*(?:[-—:]\s*)?(.*)$",
    re.I,
)
SUBCLUSTER_HEADER_RE = re.compile(
    r"^\*{0,2}SUB-CLUSTER\s+([A-Z](?:\.\d+|\d*)?)\s*(?:[-—:]\s*)?(.*?)(?:\*{2})?:?\s*$",
    re.I,
)
COST_RE = re.compile(r"predicted(?:_|\s+)cost(?:_|\s+)usd[^0-9-]*([-+]?\d+(?:\.\d+)?)", re.I)
DELTA_RE = re.compile(
    r"predicted(?:_|\s+)(?:delta|Δ)S(?:_|\s+)?(?:band)?[^-\d[]*"
    r"\[?\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]?",
    re.I,
)


@dataclass(frozen=True)
class _DirectiveHeader:
    line_idx: int
    item_id: str
    title: str
    kind: str
    cluster_key: str | None = None


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


def _clean_title(value: str, fallback: str) -> str:
    title = value.strip()
    while title.startswith("*"):
        title = title[1:].strip()
    while title.endswith(":") or title.endswith("*"):
        title = title[:-1].strip()
    return title or fallback


def _normalize_cluster_label(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _match_directive_header(line: str, idx: int) -> _DirectiveHeader | None:
    stripped = line.strip()
    match = ITEM_HEADER_RE.match(stripped)
    if match:
        item_num = int(match.group(1))
        item_id = f"ITEM_{item_num}"
        return _DirectiveHeader(
            idx,
            item_id,
            _clean_title(match.group(2), f"ITEM {item_num}"),
            "item",
        )
    match = WIRE_IN_HEADER_RE.match(stripped)
    if match:
        item_num = int(match.group(1))
        item_id = f"WIRE_IN_{item_num}"
        return _DirectiveHeader(
            idx,
            item_id,
            _clean_title(match.group(2), f"Wire-in #{item_num}"),
            "wire_in",
        )
    match = BUILD_HEADER_RE.match(stripped)
    if match:
        item_num = int(match.group(1))
        item_id = f"BUILD_{item_num}"
        return _DirectiveHeader(
            idx,
            item_id,
            _clean_title(match.group(2), f"Build #{item_num}"),
            "build",
        )
    match = OP_HEADER_RE.match(stripped)
    if match:
        item_num = int(match.group(1))
        item_id = f"OP_{item_num}"
        return _DirectiveHeader(
            idx,
            item_id,
            _clean_title(match.group(2), f"OP-{item_num}"),
            "op",
        )
    match = CLUSTER_HEADER_RE.match(stripped)
    if match:
        cluster_key = _normalize_cluster_label(match.group(1))
        item_id = f"CLUSTER_{cluster_key}"
        return _DirectiveHeader(
            idx,
            item_id,
            _clean_title(match.group(2), f"CLUSTER {match.group(1).upper()}"),
            "cluster",
            cluster_key=cluster_key,
        )
    match = SUBCLUSTER_HEADER_RE.match(stripped)
    if match:
        cluster_key = _normalize_cluster_label(match.group(1))
        item_id = f"CLUSTER_{cluster_key}"
        return _DirectiveHeader(
            idx,
            item_id,
            _clean_title(match.group(2), f"Sub-cluster {match.group(1).upper()}"),
            "subcluster",
            cluster_key=cluster_key,
        )
    return None


def _parent_cluster_keys_with_children(headers: list[_DirectiveHeader]) -> set[str]:
    skipped: set[str] = set()
    cluster_positions = [
        (pos, header)
        for pos, header in enumerate(headers)
        if header.kind == "cluster" and header.cluster_key is not None
    ]
    for pos, header in cluster_positions:
        assert header.cluster_key is not None
        next_cluster_idx = len(headers)
        for next_pos in range(pos + 1, len(headers)):
            if headers[next_pos].kind == "cluster":
                next_cluster_idx = next_pos
                break
        child_prefix = header.cluster_key
        has_child = any(
            candidate.kind == "subcluster"
            and candidate.cluster_key is not None
            and candidate.cluster_key.startswith(child_prefix)
            and candidate.cluster_key != child_prefix
            for candidate in headers[pos + 1 : next_cluster_idx]
        )
        if has_child:
            skipped.add(header.cluster_key)
    return skipped


def _filter_task_headers(headers: list[_DirectiveHeader]) -> list[_DirectiveHeader]:
    parent_keys = _parent_cluster_keys_with_children(headers)
    return [
        header
        for header in headers
        if not (header.kind == "cluster" and header.cluster_key in parent_keys)
    ]


def extract_tasks_from_directive(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    raw_headers: list[_DirectiveHeader] = []
    for idx, line in enumerate(lines):
        header = _match_directive_header(line, idx)
        if header is not None:
            raw_headers.append(header)
    headers = _filter_task_headers(raw_headers)
    tasks: list[dict[str, object]] = []
    for pos, header in enumerate(headers):
        start_idx = header.line_idx
        end_idx = headers[pos + 1].line_idx if pos + 1 < len(headers) else len(lines)
        section = "\n".join(lines[start_idx:end_idx])
        from tac.canonical_task_status import task_id_for_memo_item

        tasks.append(
            {
                "task_id": task_id_for_memo_item(_relative(path), header.item_id),
                "source_design_memo": _relative(path),
                "item_id": header.item_id,
                "title": header.title,
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
