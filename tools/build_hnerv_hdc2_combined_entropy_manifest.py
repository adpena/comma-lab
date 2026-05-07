#!/usr/bin/env python3
"""Build a planning-only HDC2 combined entropy reduction manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_hdc2_combined_entropy import (  # noqa: E402
    Hdc2CombinedEntropyError,
    build_hdc2_combined_entropy_manifest,
    render_markdown,
)
from tac.repo_io import json_text, read_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-ranking", type=Path, required=True)
    parser.add_argument("--entropy-audit", type=Path, required=True)
    parser.add_argument("--hdc2-work-product", type=Path, required=True)
    parser.add_argument("--active-floor-label")
    parser.add_argument("--active-floor-archive-bytes", type=int)
    parser.add_argument("--active-floor-archive-sha256")
    parser.add_argument("--active-floor-score", type=float)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        manifest = build_hdc2_combined_entropy_manifest(
            read_json(args.frontier_ranking),
            read_json(args.entropy_audit),
            read_json(args.hdc2_work_product),
            active_floor=_active_floor_from_args(args),
        )
    except (OSError, Hdc2CombinedEntropyError) as exc:
        print(f"FATAL: HDC2 combined entropy manifest input rejected: {exc}", file=sys.stderr)
        return 2
    manifest = attach_tool_run_manifest(
        manifest,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[args.frontier_ranking, args.entropy_audit, args.hdc2_work_product],
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json_text(manifest), encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(manifest), encoding="utf-8")
    return 0


def _active_floor_from_args(args: argparse.Namespace) -> dict[str, object] | None:
    if args.active_floor_archive_bytes is None:
        return None
    return {
        "label": args.active_floor_label or "",
        "archive_bytes": args.active_floor_archive_bytes,
        "archive_sha256": args.active_floor_archive_sha256 or "",
        "score": args.active_floor_score,
    }


if __name__ == "__main__":
    raise SystemExit(main())
