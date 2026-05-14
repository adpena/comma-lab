#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Select the next static-ready HNeRV entropy frontier candidate."""

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

from tac.hnerv_entropy_frontier_selector import (  # noqa: E402
    ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES,
    HnervEntropyFrontierSelectorError,
    build_hnerv_entropy_frontier_selection,
    render_markdown,
)
from tac.repo_io import json_text  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Candidate manifest to rank. Repeatable.",
    )
    parser.add_argument(
        "--active-candidate",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Already queued/running candidate to record and exclude. Repeatable.",
    )
    parser.add_argument(
        "--active-rate-only-floor-archive-bytes",
        type=int,
        default=ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES,
        help=(
            "Current rate-only exact-eval spend byte floor. Rate-only candidates at or "
            "above this floor are blocked unless they declare a scorer-changing stack path."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--fail-if-none",
        action="store_true",
        help="Exit 1 when no exact-evaluable-after-lane-claim candidate is selected.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        candidates = [_parse_labeled_path(item) for item in args.candidate]
        active = [_parse_labeled_path(item) for item in args.active_candidate]
        manifest = build_hnerv_entropy_frontier_selection(
            candidates,
            active_candidates=active,
            active_rate_only_floor_archive_bytes=args.active_rate_only_floor_archive_bytes,
            repo_root=REPO_ROOT,
        )
    except HnervEntropyFrontierSelectorError as exc:
        print(f"FATAL: HNeRV entropy frontier selector input rejected: {exc}", file=sys.stderr)
        return 2

    input_paths = [path for _label, path in candidates + active]
    manifest = attach_tool_run_manifest(
        manifest,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=input_paths,
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(manifest), encoding="utf-8")
    else:
        print(json_text(manifest), end="")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(manifest), encoding="utf-8")
    if args.fail_if_none and manifest.get("selected_next_candidate") is None:
        return 1
    return 0


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    label, sep, path = value.partition("=")
    if not sep or not label or not path:
        raise HnervEntropyFrontierSelectorError(
            f"labeled path must have form LABEL=PATH, got {value!r}"
        )
    return label, Path(path)


if __name__ == "__main__":
    raise SystemExit(main())
