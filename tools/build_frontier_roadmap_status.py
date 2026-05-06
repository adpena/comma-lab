#!/usr/bin/env python3
"""Build a live-safe roadmap status from the cross-paradigm frontier queue.

This is an operator artifact. It does not build archives, dispatch jobs, or
claim scores. Its purpose is to make the next tranche safe on a shared ``main``
by joining the static frontier inventory with the current worktree state.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text  # noqa: E402
from tools.build_cross_paradigm_frontier_inventory import build_inventory  # noqa: E402

SCHEMA_VERSION = 1


def git_dirty_paths(repo_root: Path) -> list[str]:
    """Return sorted dirty paths from ``git status --porcelain=v1 -z``."""

    proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=False,
    )
    raw = proc.stdout
    if not raw:
        return []
    entries = raw.split(b"\0")
    paths: list[str] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        text = entry.decode("utf-8", errors="replace")
        status = text[:2]
        path = text[3:]
        if (status[0] == "R" or status[0] == "C") and index < len(entries) and entries[index]:
            path = entries[index].decode("utf-8", errors="replace")
            index += 1
        if path:
            paths.append(path)
    return sorted(set(paths))


def _path_matches(candidate: str, dirty_path: str) -> bool:
    return (
        dirty_path == candidate
        or dirty_path.startswith(candidate.rstrip("/") + "/")
        or candidate.startswith(dirty_path.rstrip("/") + "/")
    )


def dirty_paths_for_row(row: dict[str, Any], dirty_paths: list[str]) -> list[str]:
    """Return dirty paths that intersect a frontier row's code/evidence paths."""

    watched = [*row.get("code_paths", []), *row.get("evidence_paths", [])]
    matches = []
    for dirty_path in dirty_paths:
        if any(_path_matches(str(path), dirty_path) for path in watched):
            matches.append(dirty_path)
    return sorted(set(matches))


def _readiness_stage(row: dict[str, Any], dirty_matches: list[str]) -> str:
    if dirty_matches:
        return "blocked_by_dirty_worktree"
    if row.get("score_snapshot"):
        return "exact_evidence_present_review_before_promotion"
    action = str(row.get("action_class") or "")
    if "exact_eval" in action or "promote" in action:
        return "needs_lane_claim_and_exact_cuda"
    if "build" in action or "prove" in action or "replace" in action:
        return "needs_byte_closed_candidate_or_fixture"
    return "needs_research_or_contract_hardening"


def build_roadmap_status(
    *,
    repo_root: Path,
    dirty_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Join static frontier inventory with live dirty-worktree blockers."""

    inventory = build_inventory(repo_root=repo_root)
    live_dirty_paths = git_dirty_paths(repo_root) if dirty_paths is None else sorted(set(dirty_paths))
    rows = []
    stage_counts: Counter[str] = Counter()
    dirty_blocked_count = 0
    for row in inventory["frontier_action_queue"]:
        source_row = next(item for item in inventory["rows"] if item["key"] == row["key"])
        dirty_matches = dirty_paths_for_row(source_row, live_dirty_paths)
        stage = _readiness_stage(source_row, dirty_matches)
        stage_counts[stage] += 1
        dirty_blocked_count += int(bool(dirty_matches))
        rows.append(
            {
                "key": row["key"],
                "title": source_row["title"],
                "priority_tier": row["priority_tier"],
                "action_class": row["action_class"],
                "role": row["role"],
                "status": row["status"],
                "evidence_grade": source_row["evidence_grade"],
                "stackability": source_row["stackability"],
                "replacement_potential": source_row["replacement_potential"],
                "readiness_stage": stage,
                "dirty_path_blockers": dirty_matches,
                "safe_to_touch_now": not dirty_matches,
                "score_claim": False,
                "dispatch_attempted": False,
                "ready_for_exact_eval_dispatch": False,
                "next_patch": row["next_patch"],
                "blockers": row["blockers"],
                "missing_code_path_count": source_row["path_audit"]["code"]["missing_count"],
                "missing_evidence_path_count": source_row["path_audit"]["evidence"]["missing_count"],
            }
        )
    next_unblocked = [
        row["key"]
        for row in rows
        if row["safe_to_touch_now"]
    ][:5]
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/build_frontier_roadmap_status.py",
        "inventory_tool": inventory["tool"],
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "row_count": len(rows),
        "dirty_path_count": len(live_dirty_paths),
        "dirty_paths": live_dirty_paths,
        "dirty_blocked_row_count": dirty_blocked_count,
        "stage_counts": dict(sorted(stage_counts.items())),
        "next_unblocked_keys": next_unblocked,
        "rows": rows,
        "dispatch_blockers": [
            "roadmap_status_only",
            "requires_candidate_specific_archive_manifest",
            "requires_lane_dispatch_claim",
            "requires_exact_cuda_auth_eval",
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Frontier Roadmap Status",
        "",
        "Live-safe operator roadmap. It does not claim scores or dispatch work.",
        "",
        f"- row_count: `{payload['row_count']}`",
        f"- dirty_path_count: `{payload['dirty_path_count']}`",
        f"- dirty_blocked_row_count: `{payload['dirty_blocked_row_count']}`",
        f"- next_unblocked_keys: `{', '.join(payload['next_unblocked_keys'])}`",
        "",
        "| key | tier | role | stage | safe | action | evidence | blockers | next patch |",
        "|---|---:|---|---|---|---|---|---:|---|",
    ]
    for row in payload["rows"]:
        blocker_count = (
            len(row["dirty_path_blockers"])
            + int(row["missing_code_path_count"])
            + int(row["missing_evidence_path_count"])
            + len(row["blockers"])
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_md(row['key'])}`",
                    str(row["priority_tier"]),
                    f"`{_md(row['role'])}`",
                    f"`{_md(row['readiness_stage'])}`",
                    "`yes`" if row["safe_to_touch_now"] else "`no`",
                    f"`{_md(row['action_class'])}`",
                    _md(row["evidence_grade"]),
                    str(blocker_count),
                    _md(row["next_patch"]),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _md(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_roadmap_status(repo_root=args.repo_root)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(payload), encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    if args.json_out is None and args.md_out is None:
        sys.stdout.write(json_text(payload) if args.format == "json" else render_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
