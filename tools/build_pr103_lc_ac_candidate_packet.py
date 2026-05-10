#!/usr/bin/env python3
"""Build a local PR103 LC-AC candidate packet for compliance checks."""

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

from tac.pr103_lc_ac_runtime_adapter import (  # noqa: E402
    Pr103RuntimeAdapterError,
    build_pr103_lc_ac_candidate_packet,
)
from tac.repo_io import json_text, write_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-adapter-manifest", required=True, type=Path)
    parser.add_argument("--packet-dir", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--fail-if-blocked",
        action="store_true",
        help="Exit 1 when the packet is not exact-eval ready.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        report = build_pr103_lc_ac_candidate_packet(
            runtime_adapter_manifest=args.runtime_adapter_manifest,
            packet_dir=args.packet_dir,
            repo_root=REPO_ROOT,
            force=args.force,
        )
    except (OSError, Pr103RuntimeAdapterError) as exc:
        print(f"FATAL: PR103 candidate packet build failed: {exc}", file=sys.stderr)
        return 2
    report = attach_tool_run_manifest(
        report,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[args.runtime_adapter_manifest],
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    if args.json_out is not None:
        write_json(args.json_out, report)
    else:
        print(json_text(report), end="")
    if args.fail_if_blocked and report.get("ready_for_exact_eval_dispatch") is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
