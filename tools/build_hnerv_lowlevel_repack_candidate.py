#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a byte-proved HNeRV low-level brotli-repack candidate."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_lowlevel_packer import (  # noqa: E402
    REPACKABLE_SECTIONS,
    build_lowlevel_brotli_repack_candidate,
)
from tac.hnerv_frontier_defaults import HNERV_ACTIVE_SCORECARD  # noqa: E402
from tac.repo_io import json_text, read_json  # noqa: E402

DEFAULT_SCORECARD = HNERV_ACTIVE_SCORECARD
DEFAULT_JOBS = max(1, min(os.cpu_count() or 1, 8))


def parse_lgwins(values: list[str] | None) -> list[int | None]:
    if not values:
        return [None, 18, 20, 22, 24]
    out: list[int | None] = []
    for value in values:
        if value.lower() in {"none", "default"}:
            out.append(None)
        else:
            out.append(int(value))
    return out


def parse_lgblocks(values: list[str] | None) -> list[int | None]:
    if not values:
        return [None]
    out: list[int | None] = []
    for value in values:
        if value.lower() in {"none", "default"}:
            out.append(None)
        else:
            out.append(int(value))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-section",
        action="append",
        choices=REPACKABLE_SECTIONS,
        help="Packed brotli section to recode; repeatable. Defaults to both repackable sections.",
    )
    parser.add_argument("--quality", action="append", type=int, help="Brotli quality; repeatable.")
    parser.add_argument("--lgwin", action="append", help="Brotli lgwin or 'default'; repeatable.")
    parser.add_argument("--lgblock", action="append", help="Brotli lgblock or 'default'; repeatable.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=(
            "Maximum brotli recode attempts to run concurrently. "
            f"Default: min(CPU count, 8) = {DEFAULT_JOBS}; use --jobs 1 for serial compatibility."
        ),
    )
    parser.add_argument(
        "--allow-rate-regression",
        action="store_true",
        help="Emit changed candidates even when they are not byte-smaller.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--fail-if-blocked",
        action="store_true",
        help=(
            "Exit nonzero unless the candidate is ready for archive preflight. "
            "This does not mean exact-eval dispatch readiness."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_lowlevel_brotli_repack_candidate(
        source_archive=args.source_archive,
        scorecard=read_json(args.scorecard),
        source_label=args.source_label,
        output_dir=args.output_dir,
        target_sections=args.target_section or REPACKABLE_SECTIONS,
        qualities=args.quality or [9, 10, 11],
        lgwins=parse_lgwins(args.lgwin),
        lgblocks=parse_lgblocks(args.lgblock),
        allow_rate_regression=args.allow_rate_regression,
        jobs=args.jobs,
    )
    text = json_text(result)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 1 if args.fail_if_blocked and not result.get("ready_for_archive_preflight") else 0


if __name__ == "__main__":
    raise SystemExit(main())
