#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a fail-closed HDM3 HNeRV archive candidate."""

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

from tac.hnerv_hdm3_archive_candidate import (  # noqa: E402
    HnervHdm3ArchiveCandidateError,
    build_hdm3_archive_candidate,
)
from tac.hnerv_lowlevel_packer import HnervLowlevelPackError  # noqa: E402
from tac.repo_io import json_text, write_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-label", required=True)
    parser.add_argument(
        "--decoder-recode-variant",
        choices=("hdm3", "hdm4", "hdm6", "hdm7", "hdm8"),
        default="hdm3",
        help="Lossless decoder-section recode to materialize.",
    )
    parser.add_argument(
        "--allow-rate-regression",
        action="store_true",
        help="Materialize the archive even when the selected decoder section is not smaller.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--fail-if-blocked",
        action="store_true",
        help="Exit 1 if the archive-build gate is blocked.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        manifest = build_hdm3_archive_candidate(
            source_archive=args.source_archive,
            output_dir=args.output_dir,
            source_label=args.source_label,
            decoder_recode_variant=args.decoder_recode_variant,
            allow_rate_regression=args.allow_rate_regression,
            repo_root=REPO_ROOT,
        )
    except (HnervHdm3ArchiveCandidateError, HnervLowlevelPackError) as exc:
        print(f"FATAL: HDM3 archive candidate input rejected: {exc}", file=sys.stderr)
        return 2

    manifest = attach_tool_run_manifest(
        manifest,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[args.source_archive],
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    if args.json_out is not None:
        write_json(args.json_out, manifest)
    else:
        print(json_text(manifest), end="")
    if args.fail_if_blocked and manifest.get("ready_for_archive_preflight") is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
