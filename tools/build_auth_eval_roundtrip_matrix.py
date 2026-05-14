#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a contest/digital-diagnostic auth-eval target matrix for a packet."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.auth_eval_roundtrip_matrix import (  # noqa: E402
    AuthEvalRoundtripInput,
    build_auth_eval_roundtrip_matrix,
)
from tac.repo_io import read_json, write_json  # noqa: E402


def _load_contest_auth_eval_module() -> Any:
    path = REPO_ROOT / "experiments/contest_auth_eval.py"
    spec = importlib.util.spec_from_file_location("contest_auth_eval_runtime_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import runtime manifest helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--submission-dir", type=Path)
    parser.add_argument("--inflate-sh", default="inflate.sh")
    parser.add_argument("--packet-manifest", type=Path)
    parser.add_argument("--label", default="candidate")
    parser.add_argument(
        "--output-root",
        default="experiments/results/auth_eval_roundtrip_matrix",
    )
    parser.add_argument("--lane-id", default="auth_eval_roundtrip_matrix")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--no-diagnostics", action="store_true")
    return parser.parse_args(argv)


def _candidate_from_args(args: argparse.Namespace) -> AuthEvalRoundtripInput:
    if args.packet_manifest:
        payload = read_json(args.packet_manifest)
        archive = payload.get("archive") if isinstance(payload.get("archive"), dict) else {}
        runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
        return AuthEvalRoundtripInput(
            archive=str(args.archive or archive.get("path")),
            submission_dir=str(args.submission_dir or runtime.get("path")),
            inflate_sh=str(args.inflate_sh or "inflate.sh"),
            label=args.label,
            output_root=str(args.output_root),
            lane_id=str(args.lane_id),
        )
    if args.archive is None or args.submission_dir is None:
        raise SystemExit("FATAL: pass --packet-manifest or both --archive and --submission-dir")
    return AuthEvalRoundtripInput(
        archive=args.archive.as_posix(),
        submission_dir=args.submission_dir.as_posix(),
        inflate_sh=args.inflate_sh,
        label=args.label,
        output_root=str(args.output_root),
        lane_id=str(args.lane_id),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidate = _candidate_from_args(args)
    runtime_dir = Path(candidate.submission_dir)
    inflate_path = runtime_dir / candidate.inflate_sh
    if not inflate_path.is_file():
        raise SystemExit(f"FATAL: inflate.sh not found: {inflate_path}")
    contest_auth_eval = _load_contest_auth_eval_module()
    runtime_manifest = contest_auth_eval._runtime_dependency_manifest(
        inflate_path,
        REPO_ROOT / "upstream",
        repo_root=REPO_ROOT,
    )
    matrix = build_auth_eval_roundtrip_matrix(
        candidate=candidate,
        runtime_manifest=runtime_manifest,
        repo_root=REPO_ROOT,
        include_diagnostics=not args.no_diagnostics,
    )
    write_json(args.json_out, matrix)
    print(json.dumps(matrix, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
