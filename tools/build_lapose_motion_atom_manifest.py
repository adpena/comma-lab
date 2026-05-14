#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a planning-only LA-POSE motion atom manifest."""

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

from tac.analysis.lapose_motion_atoms import (  # noqa: E402
    build_motion_atom_manifest,
    records_from_json_payload,
)
from tac.repo_io import json_text, read_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--base-pose-dist", type=float, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target-average-degree", type=float, default=2.0)
    parser.add_argument("--max-atoms", type=int)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    manifest = build_motion_atom_manifest(
        records_from_json_payload(read_json(args.records_json)),
        base_pose_dist=args.base_pose_dist,
        source=args.source,
        target_average_degree=args.target_average_degree,
        max_atoms=args.max_atoms,
    )
    manifest = attach_tool_run_manifest(
        manifest,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[args.records_json],
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    text = json_text(manifest)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
