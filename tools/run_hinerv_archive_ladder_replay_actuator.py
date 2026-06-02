#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan or execute HiNeRV archive-ladder replay commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.hinerv_archive_ladder_replay_actuator import (  # noqa: E402
    DEFAULT_REPLAY_TIMEOUT_SECONDS,
    HINERV_ARCHIVE_LADDER_REPLAY_ACTUATOR_SCHEMA,
    build_hinerv_archive_ladder_replay_actuator_report,
    render_hinerv_archive_ladder_replay_actuator_markdown,
)
from tac.repo_io import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waterfill-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", default=None, type=Path)
    parser.add_argument("--row-id", action="append", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Load existing per-row replay JSONs without executing commands.",
    )
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument(
        "--timeout-seconds",
        default=DEFAULT_REPLAY_TIMEOUT_SECONDS,
        type=int,
    )
    parser.add_argument(
        "--replay-output-root",
        default=None,
        type=Path,
        help="Fresh SSD root for bulky per-row replay output dirs.",
    )
    parser.add_argument(
        "--artifact-tag",
        default=None,
        help="Optional tag used to rewrite per-row replay JSON/Markdown paths.",
    )
    parser.add_argument(
        "--allow-non-ssd-output",
        action="store_true",
        help="Permit non-/Volumes replay output dirs. Intended for tiny tests only.",
    )
    args = parser.parse_args(argv)

    waterfill = json.loads(args.waterfill_json.read_text(encoding="utf-8"))
    report = build_hinerv_archive_ladder_replay_actuator_report(
        waterfill,
        row_ids=args.row_id,
        execute=bool(args.execute),
        cwd=args.repo_root,
        timeout_seconds=int(args.timeout_seconds),
        replay_output_root=args.replay_output_root,
        artifact_tag=args.artifact_tag,
        load_existing=bool(args.load_existing),
        allow_non_ssd_output=bool(args.allow_non_ssd_output),
    )
    output = args.output_json.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = output.as_posix()
    write_json(output, report)
    if args.output_md is not None:
        md_output = args.output_md.expanduser().resolve(strict=False)
        md_output.parent.mkdir(parents=True, exist_ok=True)
        report["markdown_report_path"] = md_output.as_posix()
        md_output.write_text(
            render_hinerv_archive_ladder_replay_actuator_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps(_summary(report), sort_keys=True))
    return 0


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": HINERV_ARCHIVE_LADDER_REPLAY_ACTUATOR_SCHEMA,
        "report_path": report.get("report_path"),
        "execution_requested": report["execution_requested"],
        "load_existing_requested": report["load_existing_requested"],
        "row_count": report["row_count"],
        "executed_row_count": report["executed_row_count"],
        "loaded_replay_report_count": report["loaded_replay_report_count"],
        "receiver_proof_ready_row_count": report["receiver_proof_ready_row_count"],
        "archive_bytes_by_row_id": report["archive_bytes_by_row_id"],
        "score_claim": report["score_claim"],
        "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
        "blockers": report["blockers"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
