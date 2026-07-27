#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile or validate a strict Einstein--Kolmogorov R-D custody receipt."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for candidate in (REPO, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tac.optimization.einstein_kolmogorov_frontier import (  # noqa: E402
    CANONICAL_POINTER_SOURCE,
    MAX_RECEIPT_BYTES,
    FrontierRefusal,
    canonical_json,
    compile_frontier,
    preflight_u3,
    validate_receipt,
    write_checkpoint,
)

DEFAULT_EVIDENCE_ROOT = Path("/Volumes/VertigoDataTier/pact/evidence/einstein_kolmogorov_20260721")
SSD_VOLUME_ROOT = Path("/Volumes/VertigoDataTier")
MAX_CANDIDATE_INPUTS = 256
MAX_TOTAL_INPUT_BYTES = 256 * 1024 * 1024


def _json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size > MAX_RECEIPT_BYTES:
        raise FrontierRefusal(f"JSON_METADATA_MISSING_OR_TOO_LARGE: {path}")
    try:
        with path.open("rb") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrontierRefusal(f"UNREADABLE_JSON: {path}") from exc


def _candidate_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        loaded = _json(path)
        if isinstance(loaded, list):
            rows.extend(loaded)
        elif isinstance(loaded, dict):
            rows.append(loaded)
        else:
            raise FrontierRefusal(f"INVALID_CANDIDATE_JSON: {path}")
    return rows


def _mapping_json(path: Path, label: str) -> dict[str, Any]:
    loaded = _json(path)
    if not isinstance(loaded, dict):
        raise FrontierRefusal(f"INVALID_{label}_JSON_ROOT:{path}")
    return loaded


def _file_ref(path: Path, *, stored_path: str | None = None) -> dict[str, object]:
    if not path.is_file():
        raise FrontierRefusal(f"CUSTODY_FILE_MISSING:{path}")
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path) if stored_path is None else stored_path,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _stored_path(path: Path) -> str:
    """Keep repo-local input paths portable while preserving external paths."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO.resolve()))
    except ValueError:
        return str(resolved)


def _require_ssd_output(path: Path, input_paths: list[Path]) -> None:
    if not SSD_VOLUME_ROOT.is_dir() or not os.path.ismount(SSD_VOLUME_ROOT):
        raise FrontierRefusal(f"SSD_VOLUME_NOT_MOUNTED:{SSD_VOLUME_ROOT}")
    resolved = path.resolve()
    root = DEFAULT_EVIDENCE_ROOT.resolve()
    if root not in (resolved, *resolved.parents):
        raise FrontierRefusal(f"OUTPUT_MUST_BE_UNDER_SSD_EVIDENCE_ROOT: {root}")
    if len(input_paths) > MAX_CANDIDATE_INPUTS + 3:
        raise FrontierRefusal("TOO_MANY_INPUT_METADATA_FILES")
    total_input_bytes = sum(item.stat().st_size for item in input_paths if item.is_file())
    if total_input_bytes > MAX_TOTAL_INPUT_BYTES:
        raise FrontierRefusal(f"TOTAL_INPUT_METADATA_TOO_LARGE:{total_input_bytes}")
    free = shutil.disk_usage(root if root.exists() else root.parent).free
    required_free = 64 * 1024 * 1024 + 4 * total_input_bytes
    if free < required_free:
        raise FrontierRefusal(f"SSD_FREE_SPACE_PREFLIGHT_FAILED:{free}<{required_free}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--candidate", action="append", type=Path, required=True)
    compile_parser.add_argument("--u3-row", type=Path)
    compile_parser.add_argument("--sibling-arms", type=Path)
    compile_parser.add_argument("--pointer-source", type=Path, default=Path(CANONICAL_POINTER_SOURCE))
    compile_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--receipt", type=Path, required=True)
    u3_parser = subparsers.add_parser("preflight-u3")
    u3_parser.add_argument("--u3-row", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == "validate":
            validate_receipt(_mapping_json(args.receipt, "RECEIPT"))
            return 0
        if args.mode == "preflight-u3":
            report = preflight_u3(_mapping_json(args.u3_row, "U3"))
            print(
                json.dumps(
                    {"status": report.status, "predicate_table": report.predicate_table, "blockers": report.blockers},
                    sort_keys=True,
                )
            )
            return 0 if report.status == "U3_RECEIVER_TUPLE_READY" else 2
        pointer_path = args.pointer_source if args.pointer_source.is_absolute() else REPO / args.pointer_source
        if len(args.candidate) > MAX_CANDIDATE_INPUTS:
            raise FrontierRefusal(f"TOO_MANY_CANDIDATE_INPUTS:{len(args.candidate)}>{MAX_CANDIDATE_INPUTS}")
        metadata_paths = [*args.candidate, pointer_path]
        metadata_paths.extend(path for path in (args.u3_row, args.sibling_arms) if path is not None)
        _require_ssd_output(args.output, metadata_paths)
        receipt = compile_frontier(
            _candidate_rows(args.candidate),
            u3_row=None if args.u3_row is None else _mapping_json(args.u3_row, "U3"),
            sibling_arms=None if args.sibling_arms is None else _mapping_json(args.sibling_arms, "SIBLING_ARMS"),
            input_manifests=[
                _file_ref(path, stored_path=_stored_path(path)) for path in metadata_paths if path != pointer_path
            ],
            pointer_source=_file_ref(pointer_path, stored_path=CANONICAL_POINTER_SOURCE),
        )
        if len(canonical_json(receipt.as_dict())) + 1 > MAX_RECEIPT_BYTES:
            raise FrontierRefusal("FRONTIER_RECEIPT_EXCEEDS_VALIDATION_LIMIT")
        write_checkpoint(args.output, receipt)
        print(json.dumps(receipt.as_dict(), sort_keys=True))
        return 0
    except FrontierRefusal as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
