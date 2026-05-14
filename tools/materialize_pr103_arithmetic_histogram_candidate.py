#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a fail-closed PR103 arithmetic histogram candidate archive."""

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

from tac.pr103_arithmetic_transform_plan import (  # noqa: E402
    Pr103ArithmeticTransformPlanError,
    materialize_pr103_arithmetic_histogram_candidate,
    render_candidate_markdown,
)
from tac.repo_io import json_text, write_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-manifest", required=True, type=Path)
    parser.add_argument(
        "--beam-probe-report",
        required=True,
        action="append",
        type=Path,
        help=(
            "Tracked beam-search JSON report. Repeat to compose streams. "
            "Used as greedy source unless --global-combo-report is supplied."
        ),
    )
    parser.add_argument(
        "--global-combo-report",
        type=Path,
        help="Optional global-combo JSON report whose moves_by_label selects candidate moves.",
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        help="Override source archive path. Defaults to source_archive.path in the manifest.",
    )
    parser.add_argument("--member-name", help="Override output ZIP member name.")
    parser.add_argument("--output-archive", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--fail-if-blocked",
        action="store_true",
        help="Exit 1 when the candidate is not archive-preflight ready.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    input_paths = [
        args.schema_manifest,
        *args.beam_probe_report,
        *([args.global_combo_report] if args.global_combo_report is not None else []),
    ]
    if args.source_archive is not None:
        input_paths.append(args.source_archive)
    try:
        report = materialize_pr103_arithmetic_histogram_candidate(
            schema_manifest=args.schema_manifest,
            beam_probe_reports=args.beam_probe_report,
            output_archive=args.output_archive,
            source_archive=args.source_archive,
            global_combo_report=args.global_combo_report,
            member_name=args.member_name,
            repo_root=REPO_ROOT,
        )
    except (OSError, Pr103ArithmeticTransformPlanError) as exc:
        print(f"FATAL: PR103 histogram candidate materialization failed: {exc}", file=sys.stderr)
        return 2

    report = attach_tool_run_manifest(
        report,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=input_paths,
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    if args.json_out is not None:
        write_json(args.json_out, report)
    else:
        print(json_text(report), end="")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_candidate_markdown(report), encoding="utf-8")
    if args.fail_if_blocked and report.get("ready_for_archive_preflight") is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
