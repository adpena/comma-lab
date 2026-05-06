#!/usr/bin/env python3
"""Profile lossless structural recodes for a PR106-style HNeRV decoder."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_decoder_recode import build_structural_recode_profile  # noqa: E402
from tac.hnerv_lowlevel_packer import (  # noqa: E402
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
)
from tac.repo_io import json_text  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = read_strict_single_member_zip(args.source_archive)
    packed = parse_ff_packed_brotli_hnerv(source.payload)
    profile = build_structural_recode_profile(
        packed,
        source_label=args.source_label,
        source_archive_sha256=source.archive_sha256,
    )
    text = json_text(profile)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
