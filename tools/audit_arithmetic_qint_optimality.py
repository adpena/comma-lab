#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile AQv1 qint coding against its zero-order entropy floor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.arithmetic_qint_codec import profile_arithmetic_container, profile_qints_arithmetic  # noqa: E402
from tac.repo_io import json_text  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--qints-npy", type=Path)
    source.add_argument("--aqv1-bin", type=Path)
    parser.add_argument("--num-symbols", type=int, default=3)
    parser.add_argument("--offset", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    if args.qints_npy:
        qints = np.load(args.qints_npy)
        payload = profile_qints_arithmetic(
            qints,
            num_symbols=args.num_symbols,
            offset=args.offset,
        )
        input_paths = [args.qints_npy]
    else:
        payload = profile_arithmetic_container(args.aqv1_bin.read_bytes())
        input_paths = [args.aqv1_bin]
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
