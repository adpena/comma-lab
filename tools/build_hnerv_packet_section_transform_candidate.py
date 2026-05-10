#!/usr/bin/env python3
"""Build a no-score HNeRV packet-section transform candidate."""

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

from tac.hnerv_lowlevel_packer import REPACKABLE_SECTIONS  # noqa: E402
from tac.packet_section_transform import (  # noqa: E402
    BrotliRecodeSectionTransform,
    CompositePacketSectionTransform,
    compile_hnerv_pr106_section_transform_candidate,
)
from tac.repo_io import json_text  # noqa: E402

DEFAULT_JOBS = max(1, min(os.cpu_count() or 1, 8))


def parse_lgwins(values: list[str] | None) -> list[int | None]:
    if not values:
        return [None, 18, 20, 22, 24]
    return [_parse_optional_int(value, name="lgwin") for value in values]


def parse_lgblocks(values: list[str] | None) -> list[int | None]:
    if not values:
        return [None]
    return [_parse_optional_int(value, name="lgblock") for value in values]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--target-section",
        action="append",
        choices=REPACKABLE_SECTIONS,
        required=True,
        help="Packed PR106 brotli section to recode; repeat for disjoint sections.",
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
        help="Emit changed section payloads even when they are not byte-smaller.",
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
    transform = CompositePacketSectionTransform(
        tuple(
            BrotliRecodeSectionTransform(
                target_section=section,
                qualities=tuple(args.quality or [9, 10, 11]),
                lgwins=tuple(parse_lgwins(args.lgwin)),
                lgblocks=tuple(parse_lgblocks(args.lgblock)),
                jobs=args.jobs,
                allow_rate_regression=args.allow_rate_regression,
            )
            for section in dict.fromkeys(args.target_section)
        )
    )
    result = compile_hnerv_pr106_section_transform_candidate(
        source_archive=args.source_archive,
        label=args.label,
        transform=transform,
        output_archive=args.output_archive,
        repo_root=REPO_ROOT,
    )
    text = json_text(result)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 1 if args.fail_if_blocked and not result.get("ready_for_archive_preflight") else 0


def _parse_optional_int(value: str, *, name: str) -> int | None:
    if value.lower() in {"none", "default"}:
        return None
    try:
        return int(value)
    except ValueError:
        raise SystemExit(f"{name} must be an integer or 'default': {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
