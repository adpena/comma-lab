#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Consume a NeRV long-training campaign plan/queue into a Cathedral verdict."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.cathedral_consumers.nerv_long_training_campaign_consumer import (  # noqa: E402
    consume_candidate,
    render_markdown,
)
from tac.repo_io import read_json, write_json_artifact, write_text_artifact  # noqa: E402


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/nerv_long_training_campaign_consumer_{stamp}_codex.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-plan-or-queue", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing outputs only with expected sha flags.",
    )
    parser.add_argument("--expected-existing-json-sha256")
    parser.add_argument("--expected-existing-md-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source = (
        args.campaign_plan_or_queue
        if args.campaign_plan_or_queue.is_absolute()
        else REPO_ROOT / args.campaign_plan_or_queue
    )
    payload = read_json(source)
    if not isinstance(payload, dict):
        raise TypeError(f"{source}: expected JSON object")
    verdict = dict(consume_candidate(payload))

    output_json = args.output_json or _default_out()
    if not output_json.is_absolute():
        output_json = REPO_ROOT / output_json
    json_result = write_json_artifact(
        output_json,
        verdict,
        allow_overwrite=args.allow_overwrite,
        expected_existing_sha256=args.expected_existing_json_sha256,
    )

    md_result = None
    if args.output_md:
        output_md = args.output_md if args.output_md.is_absolute() else REPO_ROOT / args.output_md
        md_result = write_text_artifact(
            output_md,
            render_markdown(verdict),
            allow_overwrite=args.allow_overwrite,
            expected_existing_sha256=args.expected_existing_md_sha256,
        )

    print("[NeRV long-training campaign consumer] false-authority")
    print(f"  source_schema: {verdict['source_schema']}")
    print(f"  planner_action: {verdict['planner_action']}")
    print(f"  local_mlx_ready: {verdict['ready_local_mlx_experiment_count']}")
    print(f"  exact_auth_recommended: {verdict['exact_auth_recommended']}")
    print(f"  blockers: {len(verdict['blockers'])}")
    print(
        f"  wrote {json_result.path} "
        f"({json_result.bytes_written} bytes sha256={json_result.sha256})"
    )
    if md_result is not None:
        print(
            f"  wrote {md_result.path} "
            f"({md_result.bytes_written} bytes sha256={md_result.sha256})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
