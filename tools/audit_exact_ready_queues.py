#!/usr/bin/env python3
"""Audit generated exact-ready queues against terminal lane-claim evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimizer.exact_ready_audit import (  # noqa: E402
    audit_exact_ready_queues,
    discover_exact_ready_queues,
)
from tac.optimizer.exact_readiness import ACTIVE_FLOOR_SCORE  # noqa: E402


def _markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Exact-Ready Queue Terminal-Evidence Audit",
        "",
        f"- passed: `{str(payload['passed']).lower()}`",
        f"- queue_count: `{payload['queue_count']}`",
        f"- stale_ready_row_count: `{payload['stale_ready_row_count']}`",
        "",
    ]
    for queue in payload["queues"]:  # type: ignore[index]
        rows = queue["stale_ready_rows"]
        if not rows:
            continue
        lines.append(f"## {queue['queue_path']}")
        lines.append("")
        for row in rows:
            lines.append(
                "- "
                f"candidate `{row.get('candidate_id')}` "
                f"lane `{row.get('lane_id')}` "
                f"archive `{row.get('archive_sha256')}`"
            )
            for blocker in row["blockers"]:
                lines.append(f"  - `{blocker}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--queue",
        type=Path,
        action="append",
        default=[],
        help="Exact-ready queue JSON to audit. May be repeated.",
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=Path("experiments/results"),
        help="Root scanned when --queue is omitted. Default: experiments/results.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=["**/exact_ready_queue.json", "**/*exact_ready_queue.json"],
        help="Glob pattern under --scan-root. May be repeated.",
    )
    parser.add_argument(
        "--dispatch-claims-path",
        type=Path,
        default=REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md",
    )
    parser.add_argument("--active-floor-score", type=float, default=ACTIVE_FLOOR_SCORE)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Always exit 0 after writing the audit report.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    queue_paths = list(args.queue)
    if not queue_paths:
        queue_paths = discover_exact_ready_queues(
            repo_root=repo_root,
            scan_root=args.scan_root,
            patterns=args.pattern,
        )
    missing = [path for path in queue_paths if not (path if path.is_absolute() else repo_root / path).is_file()]
    if missing:
        parser.error("queue path(s) missing: " + ", ".join(path.as_posix() for path in missing))

    resolved_queues = [path if path.is_absolute() else repo_root / path for path in queue_paths]
    claims_path = (
        args.dispatch_claims_path
        if args.dispatch_claims_path.is_absolute()
        else repo_root / args.dispatch_claims_path
    )
    payload = audit_exact_ready_queues(
        resolved_queues,
        repo_root=repo_root,
        dispatch_claims_path=claims_path,
        active_floor_score=args.active_floor_score,
    )
    text = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else _markdown(payload)
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if not payload["passed"] and not args.warn_only:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
