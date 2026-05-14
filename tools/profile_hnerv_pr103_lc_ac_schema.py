#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile PR103 hnerv_lc_ac arithmetic schema and fail-closed readiness."""

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

from tac.hnerv_pr103_lc_ac_schema import (  # noqa: E402
    HnervPr103LcAcSchemaError,
    build_pr103_lc_ac_schema_manifest,
    render_markdown,
)
from tac.repo_io import json_text, write_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--source-label", default="PR103 hnerv_lc_ac")
    parser.add_argument("--exact-adjudication-log", type=Path)
    parser.add_argument("--replay-fidelity-json", type=Path)
    parser.add_argument(
        "--candidate-archive",
        type=Path,
        help="Optional future byte-different archive; this tool still will not authorize dispatch.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--fail-if-blocked",
        action="store_true",
        help="Exit 1 when schema review is blocked.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    input_paths = [
        args.source_archive,
        *([args.exact_adjudication_log] if args.exact_adjudication_log else []),
        *([args.replay_fidelity_json] if args.replay_fidelity_json else []),
        *([args.candidate_archive] if args.candidate_archive else []),
    ]
    try:
        manifest = build_pr103_lc_ac_schema_manifest(
            source_archive=args.source_archive,
            source_label=args.source_label,
            exact_adjudication_log=args.exact_adjudication_log,
            replay_fidelity_json=args.replay_fidelity_json,
            candidate_archive=args.candidate_archive,
            repo_root=REPO_ROOT,
        )
    except (OSError, HnervPr103LcAcSchemaError) as exc:
        print(f"FATAL: PR103 lc_ac schema profile failed: {exc}", file=sys.stderr)
        return 2

    manifest = attach_tool_run_manifest(
        manifest,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=input_paths,
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    if args.json_out is not None:
        write_json(args.json_out, manifest)
    else:
        print(json_text(manifest), end="")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(manifest), encoding="utf-8")
    if args.fail_if_blocked and manifest.get("ready_for_schema_review") is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
