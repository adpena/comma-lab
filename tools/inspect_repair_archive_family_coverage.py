#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inspect archive-family coverage for repair/final-rate adapter routing."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    _TOOL_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _TOOL_DIR.parent
    for _path in (str(_REPO_ROOT), str(_TOOL_DIR)):
        if _path not in sys.path:
            sys.path.insert(0, _path)
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.archive_family_fingerprint import (  # noqa: E402
    build_archive_family_coverage_report,
)
from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", default=[], type=Path)
    parser.add_argument("--archive-glob", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-archives", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _archive_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = [_resolve(path) for path in args.archive]
    for pattern in args.archive_glob:
        full_pattern = str(_resolve(Path(pattern)))
        paths.extend(Path(path) for path in glob.glob(full_pattern, recursive=True))
    unique = sorted({path for path in paths if path.is_file()})
    if args.max_archives and args.max_archives > 0:
        unique = unique[: args.max_archives]
    return unique


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    archives = _archive_paths(args)
    if not archives:
        print("FATAL: no archive paths matched", file=sys.stderr)
        return 2
    report = build_archive_family_coverage_report(archives, repo_root=REPO_ROOT)
    output = _resolve(args.output)
    expected_existing_sha256 = sha256_file(output) if output.exists() and args.overwrite else None
    try:
        write_result = write_json_artifact(
            output,
            report,
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=expected_existing_sha256,
        )
    except ArtifactWriteError as exc:
        print(f"FATAL: archive family coverage write failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "repair_archive_family_coverage_cli_result.v1",
                "output": str(args.output),
                "output_bytes": write_result.bytes_written,
                "archive_count": report["archive_count"],
                "family_counts": report["family_counts"],
                "implemented_score_affecting_family_counts": report[
                    "implemented_score_affecting_family_counts"
                ],
                "unsupported_score_affecting_family_counts": report[
                    "unsupported_score_affecting_family_counts"
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
