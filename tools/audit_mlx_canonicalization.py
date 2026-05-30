#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit MLX files for canonical helper routing or explicit unique waivers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.local_acceleration.mlx_canonicalization_audit import (  # noqa: E402
    DEFAULT_SCAN_ROOTS,
    MlxCanonicalizationAuditError,
    build_mlx_canonicalization_audit,
    summarize_mlx_canonicalization_audit,
)
from tac.repo_io import json_text, write_text_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-root",
        type=Path,
        action="append",
        default=[],
        help="File or directory to scan. Defaults to TAC MLX/substrate/tool roots.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when files require canonical-helper routing review.",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include tests in the audit. Default scans production/tool surfaces.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    roots = args.scan_root or [Path(path) for path in DEFAULT_SCAN_ROOTS]
    try:
        payload = build_mlx_canonicalization_audit(
            repo_root=REPO_ROOT,
            scan_roots=roots,
            exclude_path_parts=() if args.include_tests else None,
        )
    except (OSError, MlxCanonicalizationAuditError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    md = summarize_mlx_canonicalization_audit(payload)
    if args.json_out:
        write_text_artifact(args.json_out, json_text(payload))
    if args.md_out:
        write_text_artifact(args.md_out, md)
    if not args.json_out and not args.md_out:
        print(json_text(payload).rstrip())
    if args.strict and payload.get("mlx_canonicalization_ready") is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
