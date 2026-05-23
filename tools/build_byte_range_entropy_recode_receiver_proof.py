#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build byte-range entropy-recode receiver proof from a PR103 runtime adapter."""

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

from tac.optimization.byte_range_entropy_recode_materializer import (  # noqa: E402
    ByteRangeEntropyRecodeMaterializerError,
    build_byte_range_entropy_recode_receiver_proof,
)
from tac.repo_io import json_text, write_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-adapter-manifest", required=True, type=Path)
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        help=(
            "Optional explicit PR103 candidate manifest. Defaults to the path "
            "recorded in the runtime adapter manifest."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--fail-if-not-ready",
        action="store_true",
        help="Exit 1 when the receiver proof is not runtime-ready.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    input_paths = [args.runtime_adapter_manifest]
    if args.candidate_manifest is not None:
        input_paths.append(args.candidate_manifest)
    try:
        proof = build_byte_range_entropy_recode_receiver_proof(
            runtime_adapter_manifest=args.runtime_adapter_manifest,
            candidate_manifest=args.candidate_manifest,
            repo_root=REPO_ROOT,
        )
    except (OSError, ByteRangeEntropyRecodeMaterializerError) as exc:
        print(f"FATAL: byte-range entropy receiver proof failed: {exc}", file=sys.stderr)
        return 2

    proof = attach_tool_run_manifest(
        proof,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=input_paths,
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    if args.json_out is not None:
        write_json(args.json_out, proof)
    else:
        print(json_text(proof), end="")
    if args.fail_if_not_ready and proof.get("ready_for_exact_eval_runtime") is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
