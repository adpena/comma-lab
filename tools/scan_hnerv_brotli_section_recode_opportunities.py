#!/usr/bin/env python3
"""Scan public HNeRV packet sections for no-score Brotli recode opportunities."""

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

from tac.analysis.hnerv_packet_sections import PARSER_AUTO, PARSER_CHOICES  # noqa: E402
from tac.packet_section_transform import scan_hnerv_brotli_recode_opportunities  # noqa: E402
from tac.repo_io import json_text  # noqa: E402

DEFAULT_JOBS = max(1, min(os.cpu_count() or 1, 8))
DEFAULT_ARCHIVES = (
    (
        "PR101",
        REPO_ROOT / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip",
        PARSER_AUTO,
    ),
    (
        "PR103",
        REPO_ROOT / "experiments/results/public_pr103_intake_20260504_codex/archive.zip",
        PARSER_AUTO,
    ),
    (
        "PR106",
        REPO_ROOT / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip",
        PARSER_AUTO,
    ),
)


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
    parser.add_argument(
        "--archive",
        action="append",
        help=(
            "Archive path, or LABEL=PATH. Repeatable. Defaults to local "
            "PR101/PR103/PR106 public archives."
        ),
    )
    parser.add_argument(
        "--parser",
        choices=PARSER_CHOICES,
        default=PARSER_AUTO,
        help="Parser for all --archive inputs. Defaults to auto.",
    )
    parser.add_argument("--quality", action="append", type=int, help="Brotli quality; repeatable.")
    parser.add_argument("--lgwin", action="append", help="Brotli lgwin or 'default'; repeatable.")
    parser.add_argument("--lgblock", action="append", help="Brotli lgblock or 'default'; repeatable.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=(
            "Maximum Brotli recode attempts to run concurrently per section. "
            f"Default: min(CPU count, 8) = {DEFAULT_JOBS}; use --jobs 1 for serial compatibility."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--fail-if-blocked",
        action="store_true",
        help="Exit nonzero if the scan itself has archive/parser blockers.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = scan_hnerv_brotli_recode_opportunities(
        _archive_specs(args.archive, parser=args.parser),
        qualities=tuple(args.quality or [9, 10, 11]),
        lgwins=tuple(parse_lgwins(args.lgwin)),
        lgblocks=tuple(parse_lgblocks(args.lgblock)),
        jobs=args.jobs,
        repo_root=REPO_ROOT,
    )
    text = json_text(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 1 if args.fail_if_blocked and payload.get("blockers") else 0


def _archive_specs(values: list[str] | None, *, parser: str) -> list[tuple[str, Path, str]]:
    if not values:
        return [(label, path, item_parser) for label, path, item_parser in DEFAULT_ARCHIVES]
    specs: list[tuple[str, Path, str]] = []
    for value in values:
        if "=" in value:
            label, raw_path = value.split("=", 1)
            if not label:
                raise SystemExit(f"archive label is empty: {value!r}")
        else:
            raw_path = value
            label = Path(raw_path).stem
        specs.append((label, Path(raw_path), parser))
    return specs


def _parse_optional_int(value: str, *, name: str) -> int | None:
    if value.lower() in {"none", "default"}:
        return None
    try:
        return int(value)
    except ValueError:
        raise SystemExit(f"{name} must be an integer or 'default': {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
