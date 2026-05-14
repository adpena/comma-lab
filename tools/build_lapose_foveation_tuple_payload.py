#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a planning-only LFV1 payload from LA-POSE foveation atom rows."""

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

from tac.analysis.lapose_foveation_payload import (  # noqa: E402
    build_lapose_foveation_tuple_payload_artifact,
)
from tac.repo_io import json_text, read_json, sha256_file  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--payload-out", type=Path, required=True)
    parser.add_argument("--atom-id", action="append", default=[])
    parser.add_argument("--max-atoms", type=int)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    manifest = read_json(args.manifest_json)
    if not isinstance(manifest, dict):
        raise SystemExit("--manifest-json must contain a JSON object")
    payload = build_lapose_foveation_tuple_payload_artifact(
        manifest,
        payload_path=args.payload_out,
        repo_root=REPO_ROOT,
        selected_atom_ids=args.atom_id,
        max_atoms=args.max_atoms,
        source_manifest_path=args.manifest_json,
        source_manifest_sha256=sha256_file(args.manifest_json),
    )
    payload = attach_tool_run_manifest(
        payload,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[args.manifest_json],
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
