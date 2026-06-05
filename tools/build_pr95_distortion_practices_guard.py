#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a PR95 distortion-practices guard for a NeRV row/verdict/queue."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.pr95_distortion_practices_guard import (  # noqa: E402
    build_pr95_distortion_practices_payload_guard,
    render_pr95_distortion_practices_markdown,
)
from tac.repo_io import read_json, write_json_artifact, write_text_artifact  # noqa: E402


def _default_output_json() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/pr95_distortion_practices_guard_{stamp}_codex.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--fail-if-blocked",
        action="store_true",
        help="Exit 1 when any source or row practice blocker is present.",
    )
    parser.add_argument("--expected-output-json-sha256")
    parser.add_argument("--expected-output-md-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source = args.input_json if args.input_json.is_absolute() else REPO_ROOT / args.input_json
    payload = read_json(source)
    if not isinstance(payload, dict):
        raise TypeError(f"{source}: expected JSON object")
    guard = build_pr95_distortion_practices_payload_guard(payload, repo_root=REPO_ROOT)
    output_json = args.output_json or _default_output_json()
    if not output_json.is_absolute():
        output_json = REPO_ROOT / output_json
    json_result = write_json_artifact(
        output_json,
        guard,
        allow_overwrite=args.expected_output_json_sha256 is not None,
        expected_existing_sha256=args.expected_output_json_sha256,
    )
    md_result = None
    if args.output_md:
        output_md = args.output_md if args.output_md.is_absolute() else REPO_ROOT / args.output_md
        md_result = write_text_artifact(
            output_md,
            render_pr95_distortion_practices_markdown(guard),
            allow_overwrite=args.expected_output_md_sha256 is not None,
            expected_existing_sha256=args.expected_output_md_sha256,
        )
    print(
        json.dumps(
            {
                "schema": guard["schema"],
                "launch_allowed": guard["launch_allowed"],
                "candidate_row_count": guard["candidate_row_count"],
                "blocker_count": len(guard["blockers"]),
                "score_claim": guard["score_claim"],
                "ready_for_exact_eval_dispatch": guard["ready_for_exact_eval_dispatch"],
                "output_json": json_result.path,
                "output_md": None if md_result is None else md_result.path,
            },
            sort_keys=True,
        )
    )
    return 1 if args.fail_if_blocked and guard["blockers"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
