#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the fail-closed SNeRV/HiNeRV trained-row emission contract."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_ladder_row_emission_contract import (  # noqa: E402
    build_nerv_ladder_row_emission_contract,
)
from tac.repo_io import write_json_artifact  # noqa: E402


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/nerv_ladder_row_emission_contract_{stamp}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        action="append",
        choices=("snerv", "hinerv", "hi_nerv"),
        help="Carrier family to include. Repeatable; defaults to SNeRV and HiNeRV.",
    )
    parser.add_argument(
        "--source-parity-json",
        type=Path,
        help="Optional source-parity contract JSON artifact.",
    )
    parser.add_argument(
        "--row-harvest-json",
        type=Path,
        action="append",
        default=[],
        help="Receiver-closed ladder-row harvest artifact. Repeatable.",
    )
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
    source_parity = (
        _load_json(args.source_parity_json) if args.source_parity_json is not None else None
    )
    harvests = [_load_harvest(path) for path in args.row_harvest_json]
    payload = build_nerv_ladder_row_emission_contract(
        families=tuple(args.family or ("snerv", "hinerv")),
        source_parity_contract=source_parity,
        row_harvests=harvests,
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
    print("[NeRV ladder-row emission contract] false-authority")
    print(f"  families: {','.join(payload['families'])}")
    print(
        "  ready_for_trained_ladder_row_emission: "
        f"{payload['ready_for_trained_ladder_row_emission']}"
    )
    print(f"  row_harvests: {len(payload['row_harvest_summaries'])}")
    print(f"  blockers: {len(payload['blockers'])}")
    print(f"  wrote {result.path} ({result.bytes_written} bytes sha256={result.sha256})")
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    source = path if path.is_absolute() else REPO_ROOT / path
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{source}: expected JSON object")
    return payload


def _load_harvest(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    source = path if path.is_absolute() else REPO_ROOT / path
    payload = dict(payload)
    payload.setdefault("source_artifact_path", source.as_posix())
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
