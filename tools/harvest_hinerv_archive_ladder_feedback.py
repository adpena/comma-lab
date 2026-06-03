#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest HiNeRV archive-ladder rate evidence as candidate feedback rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_candidate_feedback import (  # noqa: E402
    HINERV_ARCHIVE_LADDER_FEEDBACK_SCHEMA,
    build_hinerv_archive_ladder_feedback_report,
)
from tac.repo_io import read_json, write_json_artifact  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Optional append-only row ledger for --candidate-feedback-source.",
    )
    parser.add_argument("--expected-existing-output-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ladder_path = _resolve(args.ladder_json)
    output_path = _resolve(args.output_json)
    report = build_hinerv_archive_ladder_feedback_report(
        archive_ladder_report=read_json(ladder_path),
        source_report_path=ladder_path,
    )
    result = write_json_artifact(
        output_path,
        report,
        allow_overwrite=args.expected_existing_output_sha256 is not None,
        expected_existing_sha256=args.expected_existing_output_sha256,
    )
    jsonl_path = None
    if args.output_jsonl is not None:
        jsonl_path = _resolve(args.output_jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for row in report["rows"]:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "schema": HINERV_ARCHIVE_LADDER_FEEDBACK_SCHEMA,
                "output_json": Path(result.path).as_posix(),
                "output_json_sha256": result.sha256,
                "output_jsonl": None if jsonl_path is None else jsonl_path.as_posix(),
                "row_count": report["row_count"],
                "receiver_proof_attached_row_count": sum(
                    1 for row in report["rows"] if row.get("receiver_proof_attached") is True
                ),
                "feedback_ready_row_count": sum(
                    1 for row in report["rows"] if row.get("feedback_ready") is True
                ),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
                "blockers": report["blockers"],
            },
            sort_keys=True,
        )
    )
    return 0


def _resolve(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False)
    return (REPO_ROOT / expanded).resolve(strict=False)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
