#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build materializer context rows for final-byte operation work queues."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.final_byte_operation_contexts import (  # noqa: E402
    build_final_byte_operation_contexts,
)
from tac.repo_io import read_json, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backlog", type=Path, required=True)
    parser.add_argument("--artifact-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--default-output-root", type=Path)
    parser.add_argument(
        "--fail-if-blocked",
        action="store_true",
        help="Exit nonzero if generated contexts still have blockers.",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_final_byte_operation_contexts(
        read_json(args.backlog),
        artifact_map=read_json(args.artifact_map),
        repo_root=args.repo_root,
        default_output_root=args.default_output_root,
    )
    write_json_artifact(args.output, payload, allow_overwrite=args.allow_overwrite)
    if args.fail_if_blocked and int(payload.get("blocked_context_count") or 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
