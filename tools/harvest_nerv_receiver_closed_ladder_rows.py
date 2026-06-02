#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest receiver-closed NeRV ladder rows from measured JSON artifacts."""

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

from tac.analysis.nerv_receiver_closed_ladder_row_harvest import (  # noqa: E402
    ReceiverRowSource,
    build_nerv_receiver_closed_ladder_row_harvest,
)
from tac.auth_eval_schema import FULL_CONTEST_SAMPLE_COUNT  # noqa: E402
from tac.repo_io import write_json_artifact  # noqa: E402


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/nerv_receiver_closed_ladder_row_harvest_{stamp}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-json",
        type=Path,
        action="append",
        required=True,
        help="Measured scorer/replay JSON artifact to harvest. Repeatable.",
    )
    parser.add_argument("--carrier-id", required=True)
    parser.add_argument("--full-pair-count", type=int, default=FULL_CONTEST_SAMPLE_COUNT)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing --out only with --expected-existing-sha256.",
    )
    parser.add_argument("--expected-existing-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    sources = [_load_source(path) for path in args.source_json]
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        sources,
        carrier_id=str(args.carrier_id),
        full_pair_count=int(args.full_pair_count),
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
    print("[NeRV receiver-closed ladder row harvest] false-authority")
    print(f"  status: {payload['status']}")
    print(f"  sources: {payload['source_count']}")
    print(f"  harvested_rows: {payload['harvested_row_count']}")
    print(f"  full_scope_rows: {payload['full_scope_row_count']}")
    print(f"  receiver_proof_rows: {payload['receiver_proof_row_count']}")
    print(f"  ladder_candidate_rows: {payload['ladder_candidate_row_count']}")
    print(f"  blockers: {len(payload['blockers'])}")
    print(f"  wrote {result.path} ({result.bytes_written} bytes sha256={result.sha256})")
    return 0


def _load_source(path: Path) -> ReceiverRowSource:
    source = path if path.is_absolute() else REPO_ROOT / path
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{source}: expected JSON object")
    return ReceiverRowSource(payload=payload, path=source.as_posix())


if __name__ == "__main__":
    raise SystemExit(main())
