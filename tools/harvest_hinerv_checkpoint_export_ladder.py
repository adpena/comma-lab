#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Wrap trained HiNeRV checkpoint exports as archive-size ladder rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.hinerv_archive_size_ladder import (  # noqa: E402
    HINERV_ARCHIVE_SIZE_LADDER_SCHEMA,
    build_hinerv_archive_size_ladder_from_checkpoint_exports,
    render_hinerv_archive_size_ladder_markdown,
)
from tac.repo_io import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-export-json",
        action="append",
        required=True,
        type=Path,
        help="hinerv_checkpoint_archive_export.v1 JSON. Repeatable.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--num-pairs", type=int)
    args = parser.parse_args(argv)

    exports = [_load(path) for path in args.checkpoint_export_json]
    output = args.output_json.expanduser().resolve(strict=False)
    report = build_hinerv_archive_size_ladder_from_checkpoint_exports(
        exports,
        report_path=output,
        num_pairs=args.num_pairs,
    )
    report["tool_invocation"] = {
        "schema": "hinerv_checkpoint_export_ladder_tool_invocation.v1",
        "tool": "tools/harvest_hinerv_checkpoint_export_ladder.py",
        "argv": list(sys.argv[1:] if argv is None else argv),
        "checkpoint_export_json": [
            path.expanduser().resolve(strict=False).as_posix()
            for path in args.checkpoint_export_json
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    if args.output_md is not None:
        md_output = args.output_md.expanduser().resolve(strict=False)
        md_output.parent.mkdir(parents=True, exist_ok=True)
        report["markdown_report_path"] = md_output.as_posix()
        md_output.write_text(
            render_hinerv_archive_size_ladder_markdown(report),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "schema": HINERV_ARCHIVE_SIZE_LADDER_SCHEMA,
                "report_path": output.as_posix(),
                "row_count": report["row_count"],
                "archive_bytes": {
                    row["row_id"]: row["archive_bytes"]
                    for row in report["archive_rows"]
                },
                "blockers": report["blockers"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _load(path: Path) -> dict:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected JSON object")
    return payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
