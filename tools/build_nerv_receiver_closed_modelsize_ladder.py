#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build a fail-closed receiver-closed NeRV modelsize/fc_dim ladder."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_receiver_closed_modelsize_ladder import (  # noqa: E402
    build_nerv_receiver_closed_modelsize_ladder,
)
from tac.repo_io import write_json_artifact  # noqa: E402


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/nerv_receiver_closed_modelsize_ladder_{stamp}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-json", type=Path, required=True)
    parser.add_argument(
        "--rows-key",
        default="modelsize_budget_rows",
        help=(
            "Rows key inside --rows-json; falls back to "
            "rows/candidates/variant_rows/curve_rows."
        ),
    )
    parser.add_argument("--carrier-id", required=True)
    parser.add_argument("--baseline-id", default="pr95_hnerv")
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
    rows = _load_rows(args.rows_json, rows_key=str(args.rows_key))
    source = args.rows_json if args.rows_json.is_absolute() else REPO_ROOT / args.rows_json
    payload = build_nerv_receiver_closed_modelsize_ladder(
        rows,
        carrier_id=str(args.carrier_id),
        baseline_id=str(args.baseline_id),
        source_artifact_path=source.as_posix(),
        repo_root=REPO_ROOT,
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
    print("[NeRV receiver-closed modelsize ladder] false-authority")
    print(f"  status: {payload['status']}")
    print(f"  rows: {payload['row_count']}")
    print(f"  budget_rows: {payload['budget_row_count']}")
    print(f"  receiver_closed_rows: {payload['receiver_closed_row_count']}")
    print(f"  selected_bytes: {payload['receiver_closed_selected_archive_bytes']}")
    print(f"  blockers: {len(payload['blockers'])}")
    print(f"  wrote {result.path} ({result.bytes_written} bytes sha256={result.sha256})")
    return 0


def _load_rows(path: Path, *, rows_key: str) -> list[Mapping[str, Any]]:
    source = path if path.is_absolute() else REPO_ROOT / path
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get(rows_key)
        if rows is None:
            rows = payload.get("rows")
        if rows is None:
            rows = payload.get("candidates")
        if rows is None:
            rows = payload.get("variant_rows")
        if rows is None:
            rows = payload.get("curve_rows")
    else:
        raise SystemExit(f"{source}: expected JSON object or array")
    if not isinstance(rows, list):
        raise SystemExit(
            f"{source}: no list rows under {rows_key}/rows/candidates/curve_rows"
        )
    if not all(isinstance(row, dict) for row in rows):
        raise SystemExit(f"{source}: rows must all be JSON objects")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
