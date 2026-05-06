#!/usr/bin/env python3
"""Audit PR91/HPM1 replay readiness without dispatching scorer work."""

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

from tac.pr91_hpm1_codec import DEFAULT_PR91_ARCHIVE, DEFAULT_PR91_RUNTIME_SOURCE_DIR  # noqa: E402
from tac.pr91_hpm1_readiness import audit_pr91_hpm1_readiness  # noqa: E402
from tac.repo_io import json_text, read_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_PR91_ARCHIVE)
    parser.add_argument("--runtime-source-dir", type=Path, default=DEFAULT_PR91_RUNTIME_SOURCE_DIR)
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    payload = audit_pr91_hpm1_readiness(
        archive=args.archive,
        runtime_source_dir=args.runtime_source_dir,
        parity_report=read_json(args.parity_report) if args.parity_report else None,
    )
    input_paths = [path for path in (args.archive, args.parity_report) if path and path.is_file()]
    payload = attach_tool_run_manifest(
        payload,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=input_paths,
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    text = json_text(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
