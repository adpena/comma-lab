#!/usr/bin/env python3
"""Audit a categorical compression candidate manifest before exact eval."""

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

from tac.categorical_candidate_readiness import audit_categorical_candidate_manifest  # noqa: E402
from tac.repo_io import json_text, read_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def _candidate_archive_path(candidate_json: Path, payload: dict) -> Path | None:
    archive = payload.get("candidate_archive")
    if not isinstance(archive, dict):
        return None
    raw_path = archive.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    local = candidate_json.parent / path
    if local.exists():
        return local
    return REPO_ROOT / path


def _archive_member_manifest_path(candidate_json: Path, payload: dict) -> Path | None:
    manifest = payload.get("archive_member_manifest")
    if not isinstance(manifest, dict):
        return None
    raw_path = manifest.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    local = candidate_json.parent / path
    if local.exists():
        return local
    return REPO_ROOT / path


def _hpm1_structural_inventory_path(candidate_json: Path, payload: dict) -> Path | None:
    inventory = payload.get("hpm1_structural_decode_inventory")
    if not isinstance(inventory, dict):
        return None
    raw_path = inventory.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    local = candidate_json.parent / path
    if local.exists():
        return local
    return REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    source_payload = read_json(args.candidate_json)
    if not isinstance(source_payload, dict):
        raise SystemExit("candidate JSON must contain an object")
    payload = audit_categorical_candidate_manifest(
        source_payload,
        repo_root=REPO_ROOT,
        manifest_dir=args.candidate_json.parent,
    )
    input_paths = [args.candidate_json]
    archive_path = _candidate_archive_path(args.candidate_json, source_payload)
    if archive_path is not None and archive_path.exists():
        input_paths.append(archive_path)
    archive_member_manifest_path = _archive_member_manifest_path(args.candidate_json, source_payload)
    if archive_member_manifest_path is not None and archive_member_manifest_path.exists():
        input_paths.append(archive_member_manifest_path)
    hpm1_structural_inventory_path = _hpm1_structural_inventory_path(
        args.candidate_json,
        source_payload,
    )
    if hpm1_structural_inventory_path is not None and hpm1_structural_inventory_path.exists():
        input_paths.append(hpm1_structural_inventory_path)
    payload = attach_tool_run_manifest(
        payload,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=input_paths,
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    text = json_text(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
