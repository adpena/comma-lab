#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build a fail-closed NeRV rate/allocator planner queue."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_rate_allocator_queue import (  # noqa: E402
    build_nerv_rate_allocator_work_queue,
)
from tac.repo_io import write_json_artifact  # noqa: E402


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/nerv_rate_allocator_queue_{stamp}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate-bridge", type=Path, required=True)
    parser.add_argument(
        "--section-value-json",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional section-value/profile artifact to price through the "
            "compact NeRV byte-price controller. Repeatable."
        ),
    )
    parser.add_argument("--queue-id", default="nerv_rate_allocator_work_queue")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing --out only with --expected-existing-sha256.",
    )
    parser.add_argument("--expected-existing-sha256")
    return parser


def _load(path: Path) -> dict:
    source = path if path.is_absolute() else REPO_ROOT / path
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{source}: expected JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = build_nerv_rate_allocator_work_queue(
        rate_bridge=_load(args.rate_bridge),
        section_value_artifacts=[_load(path) for path in args.section_value_json],
        queue_id=args.queue_id,
    )
    out_path = args.out or _default_out()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    result = write_json_artifact(
        out_path,
        payload,
        allow_overwrite=args.allow_overwrite,
        expected_existing_sha256=args.expected_existing_sha256,
    )
    print("[NeRV rate/allocator queue] false-authority")
    print(f"  verdict: {payload['verdict']}")
    print(f"  queue_rows: {payload['queue_row_count']}")
    print(f"  blocked_queue_rows: {payload['blocked_queue_row_count']}")
    print(f"  local_planning_ready_rows: {payload['local_planning_ready_row_count']}")
    print(f"  section_admission_rows: {payload['section_admission_queue_row_count']}")
    print(f"  blockers: {len(payload['blockers'])}")
    print(f"  wrote {result.path} ({result.bytes_written} bytes sha256={result.sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
