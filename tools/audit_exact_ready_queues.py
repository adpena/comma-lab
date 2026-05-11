#!/usr/bin/env python3
"""Audit generated exact-ready queues against terminal lane-claim evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimizer.exact_ready_audit import (  # noqa: E402
    apply_suppression_manifest,
    audit_exact_ready_queues,
    build_suppression_manifest,
    discover_exact_ready_queues,
    load_suppression_manifest,
)
from tac.optimizer.exact_readiness import ACTIVE_FLOOR_SCORE  # noqa: E402
from tac.repo_io import json_text, write_json  # noqa: E402

DEFAULT_SCAN_ROOTS = (Path("experiments/results"), Path(".omx/research"))


def _markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Exact-Ready Queue Terminal-Evidence Audit",
        "",
        f"- passed: `{str(payload['passed']).lower()}`",
        f"- queue_count: `{payload['queue_count']}`",
        f"- stale_ready_row_count: `{payload['stale_ready_row_count']}`",
    ]
    if "raw_stale_ready_row_count" in payload:
        lines.append(f"- raw_stale_ready_row_count: `{payload['raw_stale_ready_row_count']}`")
    if "suppressed_ready_row_count" in payload:
        lines.append(f"- suppressed_ready_row_count: `{payload['suppressed_ready_row_count']}`")
    if payload.get("suppression_manifest_path"):
        lines.append(f"- suppression_manifest_path: `{payload['suppression_manifest_path']}`")
    lines.append("")
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
    for queue in payload["queues"]:  # type: ignore[index]
        rows = queue.get("suppressed_ready_rows", [])
        if not rows:
            continue
        lines.append(f"## Suppressed/retracted: {queue['queue_path']}")
        lines.append("")
        for row in rows:
            suppression = row.get("suppression", {})
            lines.append(
                "- "
                f"candidate `{row.get('candidate_id')}` "
                f"lane `{row.get('lane_id')}` "
                f"archive `{row.get('archive_sha256')}` "
                f"classification `{suppression.get('classification')}`"
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
        action="append",
        default=None,
        help=(
            "Root scanned when --queue is omitted. May be repeated. "
            "Default: experiments/results and .omx/research."
        ),
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
    parser.add_argument(
        "--active-floor-score",
        type=float,
        default=ACTIVE_FLOOR_SCORE,
        help=(
            "Active score frontier for terminal/exact-score routing; the flag name "
            "is kept for compatibility and does not set the rate-only byte floor."
        ),
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--suppression-manifest",
        type=Path,
        help="Read a durable suppression/retraction manifest and report only unresolved stale rows.",
    )
    parser.add_argument(
        "--write-suppression-manifest",
        type=Path,
        help="Write a suppression/retraction manifest for the current stale rows.",
    )
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
            scan_root=args.scan_root or DEFAULT_SCAN_ROOTS,
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
    suppression_manifest = None
    suppression_manifest_path = None
    if args.write_suppression_manifest is not None:
        suppression_manifest_path = (
            args.write_suppression_manifest
            if args.write_suppression_manifest.is_absolute()
            else repo_root / args.write_suppression_manifest
        )
        suppression_manifest = build_suppression_manifest(payload)
        write_json(suppression_manifest_path, suppression_manifest)
    if args.suppression_manifest is not None:
        suppression_manifest_path = (
            args.suppression_manifest
            if args.suppression_manifest.is_absolute()
            else repo_root / args.suppression_manifest
        )
        try:
            suppression_manifest = load_suppression_manifest(suppression_manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"suppression manifest invalid: {exc}")
    if suppression_manifest is not None and suppression_manifest_path is not None:
        payload = apply_suppression_manifest(
            payload,
            manifest=suppression_manifest,
            manifest_path=suppression_manifest_path,
            repo_root=repo_root,
        )
    text = json_text(payload) if args.format == "json" else _markdown(payload)
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
