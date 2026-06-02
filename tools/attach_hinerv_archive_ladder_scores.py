#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Attach measured scorer rows to a HiNeRV archive-size ladder."""

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

from tac.analysis.hinerv_archive_size_ladder import (  # noqa: E402
    attach_hinerv_archive_ladder_score_rows,
)
from tac.repo_io import read_json, write_json_artifact  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder-json", type=Path, required=True)
    parser.add_argument(
        "--score-json",
        type=Path,
        action="append",
        required=True,
        help=(
            "Measured scorer artifact containing score_rows/archive_rows/rows. "
            "Repeatable; row_id must match HiNeRV ladder row_id."
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--allow-partial-video", action="store_true")
    parser.add_argument("--expected-existing-output-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ladder_path = _resolve(args.ladder_json)
    output_path = _resolve(args.output_json)
    ladder = read_json(ladder_path)
    score_paths = [_resolve(path) for path in args.score_json]
    score_payloads = [read_json(path) for path in score_paths]
    attached = attach_hinerv_archive_ladder_score_rows(
        ladder,
        score_payloads,
        score_source_path=",".join(path.as_posix() for path in score_paths),
        require_full_video=not bool(args.allow_partial_video),
    )
    result = write_json_artifact(
        output_path,
        attached,
        allow_overwrite=args.expected_existing_output_sha256 is not None,
        expected_existing_sha256=args.expected_existing_output_sha256,
    )
    print(
        json.dumps(
            {
                "schema": attached["schema"],
                "output_json": Path(result.path).as_posix(),
                "bytes": result.bytes_written,
                "sha256": result.sha256,
                "matched_archive_row_count": attached["score_attachment"][
                    "matched_archive_row_count"
                ],
                "matched_full_video_row_count": attached["score_attachment"][
                    "matched_full_video_row_count"
                ],
                "section_value_row_count": len(attached.get("section_value_rows", [])),
                "byte_price_decision_counts": attached["byte_price_plan"][
                    "decision_counts"
                ],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
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
