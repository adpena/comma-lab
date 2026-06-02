#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile SNeRV waterfill actions into explicit mixed decoder modes."""

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

from tac.analysis.snerv_waterfill_mode_assignment import (  # noqa: E402
    SNERV_WATERFILL_MODE_ASSIGNMENT_SCHEMA,
    build_snerv_waterfill_mode_assignment,
    load_snerv_waterfill_mode_assignment_source,
    render_snerv_waterfill_mode_assignment_markdown,
)
from tac.repo_io import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waterfill-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", default=None, type=Path)
    parser.add_argument("--candidate-id", default=None)
    args = parser.parse_args(argv)

    source = load_snerv_waterfill_mode_assignment_source(args.waterfill_json)
    report = build_snerv_waterfill_mode_assignment(
        source,
        candidate_id=args.candidate_id,
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
            render_snerv_waterfill_mode_assignment_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps(_summary(report), sort_keys=True))
    return 0


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SNERV_WATERFILL_MODE_ASSIGNMENT_SCHEMA,
        "report_path": report.get("report_path"),
        "row_count": report["row_count"],
        "local_advisory_probe_ready_row_count": report[
            "local_advisory_probe_ready_row_count"
        ],
        "receiver_mode_export_ready_row_count": report[
            "receiver_mode_export_ready_row_count"
        ],
        "score_claim": report["score_claim"],
        "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
        "blockers": report["blockers"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
