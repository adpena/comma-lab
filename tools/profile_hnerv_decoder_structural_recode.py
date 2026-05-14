#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
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
from tac.hnerv_lowlevel_packer import read_packed_archive_view  # noqa: E402
from tac.repo_io import json_text  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--include-hdm5-search",
        action="store_true",
        help="Run the explicit planning-only HDM5 order/partition search.",
    )
    parser.add_argument(
        "--hdm5-max-parts",
        type=int,
        default=8,
        help="Maximum contiguous q-Brotli chunks considered by the HDM5 DP search.",
    )
    parser.add_argument(
        "--hdm5-workers",
        type=int,
        default=None,
        help="Thread workers for HDM5 segment compression; default caps at local CPU/8.",
    )
    parser.add_argument(
        "--hdm5-top-k",
        type=int,
        default=16,
        help="Number of HDM5 planning candidates retained in the report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    view = read_packed_archive_view(args.source_archive)
    profile = build_structural_recode_profile(
        view.packed,
        source_label=args.source_label,
        source_archive_sha256=view.archive.archive_sha256,
        include_hdm5_search=args.include_hdm5_search,
        hdm5_max_parts=args.hdm5_max_parts,
        hdm5_workers=args.hdm5_workers,
        hdm5_top_k=args.hdm5_top_k,
    )
    profile["source_payload_kind"] = view.payload_kind
    profile["source_decoder_section_name"] = (
        "inner_decoder_packed_brotli"
        if view.payload_kind == "pr106_sidecar_wrapper"
        else "decoder_packed_brotli"
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
