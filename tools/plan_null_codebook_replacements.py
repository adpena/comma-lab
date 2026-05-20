#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan archive-charged seed replacements for null-byte probe spans.

The tool is planning-only: it reads a null-byte probe summary/indices pair,
optionally hashes the charged ZIP inner member, and writes a fail-closed
candidate plan. It does not rewrite ``archive.zip`` or claim score movement.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.procedural_codebook_generator import (  # noqa: E402
    build_null_seed_replacement_plan,
    render_null_seed_replacement_markdown,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _read_inner_member(archive_zip: Path, member_name: str | None) -> tuple[str, bytes]:
    with zipfile.ZipFile(archive_zip) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if member_name is None:
            if len(infos) != 1:
                raise ValueError(
                    f"{archive_zip} has {len(infos)} file members; pass --inner-member-name"
                )
            info = infos[0]
        else:
            matches = [info for info in infos if info.filename == member_name]
            if len(matches) != 1:
                raise ValueError(
                    f"{archive_zip} expected exactly one member named {member_name!r}"
                )
            info = matches[0]
        return info.filename, zf.read(info)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fail-closed null-space seed replacement plan from "
            "tools/probe_null_byte_master_gradient.py outputs."
        )
    )
    parser.add_argument("--null-summary", type=Path, required=True)
    parser.add_argument("--null-indices", type=Path, required=True)
    parser.add_argument("--archive-zip", type=Path)
    parser.add_argument("--inner-member-name")
    parser.add_argument("--seed-bytes", type=int, default=8)
    parser.add_argument("--runtime-header-bytes", type=int, default=0)
    parser.add_argument("--min-run-length", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    null_summary = _load_json(args.null_summary)
    null_indices = np.load(args.null_indices)
    inner_bytes = None
    inner_member_name = args.inner_member_name
    if args.archive_zip is not None:
        inner_member_name, inner_bytes = _read_inner_member(
            args.archive_zip, args.inner_member_name
        )

    plan = build_null_seed_replacement_plan(
        null_summary=null_summary,
        null_indices=null_indices,
        inner_bytes=inner_bytes,
        seed_bytes=args.seed_bytes,
        runtime_header_bytes=args.runtime_header_bytes,
        min_run_length=args.min_run_length,
        max_candidates=args.max_candidates,
    )
    plan["input_paths"] = {
        "null_summary": str(args.null_summary),
        "null_indices": str(args.null_indices),
        "archive_zip": str(args.archive_zip) if args.archive_zip is not None else None,
        "inner_member_name": inner_member_name,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(
        render_null_seed_replacement_markdown(plan),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": plan["schema"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "candidate_count": plan["summary"]["candidate_count"],
                "best_net_saved_inner_bytes": plan["summary"][
                    "best_net_saved_inner_bytes"
                ],
                "score_claim": plan["score_claim"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
