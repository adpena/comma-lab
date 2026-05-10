#!/usr/bin/env python3
"""Probe same-runtime PR103 LC-AC rendered-frame parity."""

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
    probe_pr103_lc_ac_frame_parity,
)
from tac.repo_io import json_text, write_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime-py", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--candidate-runtime-py", required=True, type=Path)
    parser.add_argument("--candidate-archive", required=True, type=Path)
    parser.add_argument(
        "--pair-index",
        action="append",
        type=int,
        default=[],
        help="Pair index to render and hash. Repeat for multiple pairs.",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Render every pair. This is expensive for PR103 and still not an auth eval.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    pair_indices = args.pair_index
    if args.all_pairs:
        pair_indices = list(range(600))
    if not pair_indices:
        print("FATAL: pass --pair-index at least once, or --all-pairs", file=sys.stderr)
        return 2
    try:
        report = probe_pr103_lc_ac_frame_parity(
            source_runtime_py=args.source_runtime_py,
            source_archive=args.source_archive,
            candidate_runtime_py=args.candidate_runtime_py,
            candidate_archive=args.candidate_archive,
            pair_indices=pair_indices,
            device=args.device,
            repo_root=REPO_ROOT,
        )
    except (OSError, Pr103RuntimeAdapterError) as exc:
        print(f"FATAL: PR103 frame parity probe failed: {exc}", file=sys.stderr)
        return 2
    report = attach_tool_run_manifest(
        report,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[
            args.source_runtime_py,
            args.source_archive,
            args.candidate_runtime_py,
            args.candidate_archive,
        ],
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    if args.json_out is not None:
        write_json(args.json_out, report)
    else:
        print(json_text(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
