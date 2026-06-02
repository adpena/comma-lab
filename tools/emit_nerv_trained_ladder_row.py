#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit a fail-closed trained SNeRV/HiNeRV ladder row payload."""

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

from tac.analysis.nerv_trained_ladder_row_emitter import (  # noqa: E402
    build_nerv_trained_ladder_row_payload,
)
from tac.repo_io import write_json_artifact  # noqa: E402


def _default_out(family: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/{family}_trained_ladder_row_{stamp}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("snerv", "hinerv", "hi_nerv"))
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument("--trainer-json", type=Path)
    parser.add_argument(
        "--trainer-row-key",
        default="rows",
        help="Optional list key inside --trainer-json for selecting one row.",
    )
    parser.add_argument("--trainer-row-index", type=int)
    parser.add_argument("--receiver-proof-json", type=Path)
    parser.add_argument("--scorer-json", type=Path)
    parser.add_argument("--official-controls-json", type=Path)
    parser.add_argument("--row-id")
    parser.add_argument("--n-pairs", type=int)
    parser.add_argument("--modelsize-mparams", type=float)
    parser.add_argument("--fc-dim", type=int)
    parser.add_argument("--full-pair-count", type=int, default=600)
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
    trainer = _load_selected_json(
        args.trainer_json,
        row_key=str(args.trainer_row_key),
        row_index=args.trainer_row_index,
    )
    payload = build_nerv_trained_ladder_row_payload(
        family=str(args.family),
        archive_path=args.archive_path,
        trainer_metadata=trainer,
        receiver_proof=_load_json(args.receiver_proof_json),
        scorer_eval=_load_json(args.scorer_json),
        official_controls=_load_json(args.official_controls_json),
        row_id=args.row_id,
        n_pairs=args.n_pairs,
        modelsize_mparams=args.modelsize_mparams,
        fc_dim=args.fc_dim,
        full_pair_count=int(args.full_pair_count),
        repo_root=REPO_ROOT,
    )
    out_path = args.out or _default_out(str(args.family))
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    result = write_json_artifact(
        out_path,
        payload,
        allow_overwrite=args.allow_overwrite,
        expected_existing_sha256=args.expected_existing_sha256,
    )
    row = payload["rows"][0]
    print("[NeRV trained ladder row emitter] false-authority")
    print(f"  family: {payload['family']}")
    print(f"  status: {payload['status']}")
    print(f"  ready_for_receiver_closed_ladder_harvest: {payload['ready_for_receiver_closed_ladder_harvest']}")
    print(f"  archive_bytes: {row.get('archive_bytes')}")
    print(f"  blockers: {len(payload['blockers'])}")
    print(f"  wrote {result.path} ({result.bytes_written} bytes sha256={result.sha256})")
    return 0


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = path if path.is_absolute() else REPO_ROOT / path
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{source}: expected JSON object")
    return payload


def _load_selected_json(
    path: Path | None,
    *,
    row_key: str,
    row_index: int | None,
) -> dict[str, Any] | None:
    payload = _load_json(path)
    if payload is None:
        return None
    if row_index is None:
        return payload
    rows = payload.get(row_key)
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: --trainer-row-key {row_key!r} is not a list")
    if row_index < 0 or row_index >= len(rows):
        raise SystemExit(f"{path}: --trainer-row-index {row_index} out of range")
    row = rows[row_index]
    if not isinstance(row, dict):
        raise SystemExit(f"{path}: selected trainer row is not an object")
    merged = {key: value for key, value in payload.items() if key != row_key}
    merged.update(row)
    merged.setdefault("source_trainer_row_index", int(row_index))
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
