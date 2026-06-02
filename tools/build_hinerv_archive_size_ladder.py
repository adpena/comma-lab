#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export a measured false-authority HiNeRV archive-size ladder."""

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

from tac.analysis.hinerv_archive_size_ladder import (  # noqa: E402
    HINERV_ARCHIVE_SIZE_LADDER_SCHEMA,
    build_hinerv_archive_size_ladder,
    render_hinerv_archive_size_ladder_markdown,
)
from tac.repo_io import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", default=None, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--num-pairs", default=600, type=int)
    parser.add_argument("--row-id", action="append", default=None)
    parser.add_argument("--decoder-codec", default="int8_mixed")
    parser.add_argument("--emit-receiver-proof", action="store_true")
    parser.add_argument("--retain-receiver-proof-output", action="store_true")
    args = parser.parse_args(argv)

    report = build_hinerv_archive_size_ladder(
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        num_pairs=int(args.num_pairs),
        row_ids=args.row_id,
        decoder_codec=str(args.decoder_codec),
        emit_receiver_proof=bool(args.emit_receiver_proof),
        retain_receiver_proof_output=bool(args.retain_receiver_proof_output),
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
            render_hinerv_archive_size_ladder_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps(_summary(report), sort_keys=True))
    return 0


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": HINERV_ARCHIVE_SIZE_LADDER_SCHEMA,
        "report_path": report.get("report_path"),
        "output_dir": report["output_dir"],
        "row_count": report["row_count"],
        "archive_bytes": {
            row["row_id"]: row["archive_bytes"] for row in report["archive_rows"]
        },
        "score_claim": report["score_claim"],
        "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
