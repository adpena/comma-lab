#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Select existing archive anchors by public leaderboard CPU axis."""

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
    build_cpu_axis_optimal_payload,
    collect_all_anchors,
    render_frontier_scan_json,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--current-frontier-cpu",
        type=float,
        default=None,
        help="Optional CPU frontier override. Defaults to canonical best CPU anchor.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help="Exit nonzero when no existing CPU anchor improves the current frontier.",
    )
    return parser.parse_args(argv)


def _render_text(payload: dict[str, object]) -> str:
    lines = [
        "G1 CPU-axis optimal archive selector",
        "====================================",
        f"axis: {payload.get('axis')}",
        f"evidence_grade: {payload.get('evidence_grade')}",
        f"current_frontier_cpu: {payload.get('current_frontier_cpu')}",
    ]
    overall = payload.get("overall_cpu_optimal")
    if isinstance(overall, dict):
        extra = overall.get("extra")
        lane_id = extra.get("lane_id") if isinstance(extra, dict) else None
        lines.extend(
            [
                f"overall_cpu_optimal_score: {overall.get('score')}",
                f"overall_cpu_optimal_sha256: {overall.get('archive_sha256')}",
                f"overall_cpu_optimal_lane_id: {lane_id}",
                f"delta_vs_current_frontier: {payload.get('delta_vs_current_frontier')}",
                "verdict: "
                + (
                    "improvement_found"
                    if bool(payload.get("improvement_found"))
                    else "current_frontier_remains_cpu_optimal"
                ),
            ]
        )
    else:
        lines.append("overall_cpu_optimal: <none>")
    per_family = payload.get("per_family_optimal")
    if isinstance(per_family, dict) and per_family:
        lines.append("")
        lines.append("per_metadata_bucket_optimal:")
        for family, row in sorted(per_family.items()):
            if not isinstance(row, dict):
                continue
            extra = row.get("extra")
            lane_id = extra.get("lane_id") if isinstance(extra, dict) else None
            lines.append(
                f"  {family}: score={row.get('score')} "
                f"sha={str(row.get('archive_sha256') or '')[:12]} lane={lane_id}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = args.repo_root.resolve()
    payload = build_cpu_axis_optimal_payload(
        collect_all_anchors(repo_root),
        current_frontier_cpu=args.current_frontier_cpu,
    )
    if args.json:
        print(render_frontier_scan_json(payload))
    else:
        print(_render_text(payload))
    if args.require_improvement and not payload.get("improvement_found"):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
