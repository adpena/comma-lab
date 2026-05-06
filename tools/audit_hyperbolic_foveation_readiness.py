#!/usr/bin/env python3
"""Audit charged hyperbolic foveation parameters for non-score readiness."""

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

from tac.foveation_readiness import audit_foveation_params  # noqa: E402
from tac.repo_io import json_text  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foveation-params-bin", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--expected-image-size", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--source-archive-sha256")
    parser.add_argument("--candidate-archive", type=Path)
    parser.add_argument("--runtime-consumer", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    payload = audit_foveation_params(
        args.foveation_params_bin,
        repo_root=REPO_ROOT,
        expected_frames=args.expected_frames,
        expected_image_size=tuple(args.expected_image_size) if args.expected_image_size else None,
        source_archive_sha256=args.source_archive_sha256,
        candidate_archive=args.candidate_archive,
        runtime_consumer=args.runtime_consumer,
    )
    input_paths = [args.foveation_params_bin]
    if args.candidate_archive is not None:
        input_paths.append(args.candidate_archive)
    if args.runtime_consumer is not None:
        input_paths.append(args.runtime_consumer)
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
