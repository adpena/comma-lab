#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit bounded Brotli saturation for an HNeRV decoder section."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_brotli_saturation import (  # noqa: E402
    DEFAULT_LGBLOCKS,
    DEFAULT_LGWINS,
    DEFAULT_MODES,
    DEFAULT_QUALITIES,
    build_hnerv_decoder_brotli_saturation_audit,
    render_markdown,
)
from tac.hnerv_frontier_defaults import (  # noqa: E402
    HNERV_ACTIVE_ENTROPY_RANKING,
    HNERV_ACTIVE_SCORECARD,
)
from tac.repo_io import json_text, read_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402

DEFAULT_RESULT_ROOT = REPO_ROOT / "experiments" / "results"
DEFAULT_SCORECARD = HNERV_ACTIVE_SCORECARD
DEFAULT_ENTROPY_RANKING = HNERV_ACTIVE_ENTROPY_RANKING
DEFAULT_JOBS = max(1, min(os.cpu_count() or 1, 8))


def parse_optional_ints(values: list[str] | None, default: tuple[int | None, ...]) -> list[int | None]:
    if not values:
        return list(default)
    out: list[int | None] = []
    for value in values:
        if value.lower() in {"default", "none"}:
            out.append(None)
        else:
            out.append(int(value))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--entropy-ranking", type=Path, default=DEFAULT_ENTROPY_RANKING)
    parser.add_argument("--quality", action="append", type=int)
    parser.add_argument("--lgwin", action="append")
    parser.add_argument("--lgblock", action="append")
    parser.add_argument("--mode", action="append", choices=DEFAULT_MODES)
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=f"Maximum concurrent Brotli attempts. Default: {DEFAULT_JOBS}.",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    input_paths = [args.source_archive, args.scorecard, args.entropy_ranking]
    try:
        manifest = build_hnerv_decoder_brotli_saturation_audit(
            source_archive=args.source_archive,
            source_label=args.source_label,
            scorecard=read_json(args.scorecard),
            entropy_ranking=read_json(args.entropy_ranking),
            qualities=args.quality or DEFAULT_QUALITIES,
            lgwins=parse_optional_ints(args.lgwin, DEFAULT_LGWINS),
            lgblocks=parse_optional_ints(args.lgblock, DEFAULT_LGBLOCKS),
            modes=args.mode or DEFAULT_MODES,
            jobs=args.jobs,
        )
    except (OSError, ValueError) as exc:
        print(f"FATAL: HNeRV Brotli saturation audit failed: {exc}", file=sys.stderr)
        return 2

    manifest = attach_tool_run_manifest(
        manifest,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=input_paths,
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json_text(manifest), encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(manifest), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
