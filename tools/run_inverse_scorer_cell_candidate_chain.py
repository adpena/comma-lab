#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the local inverse-scorer IAS1 candidate proof chain."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.inverse_scorer_cell_chain import (  # noqa: E402
    CHAIN_MANIFEST_NAME,
    InverseScorerCellChainError,
    build_inverse_scorer_cell_candidate_chain,
)
from tac.repo_io import json_text, sha256_file, write_json_artifact  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-archive-template", type=Path, required=True)
    parser.add_argument("--inverse-action-functional", type=Path, required=True)
    parser.add_argument("--raw-contest-video-digest", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atom-id", action="append", default=[])
    parser.add_argument("--selected-limit", type=int, default=None)
    parser.add_argument("--min-free-bytes", type=int, default=0)
    parser.add_argument("--source-inflate-output-dir", type=Path)
    parser.add_argument("--candidate-inflate-output-dir", type=Path)
    parser.add_argument("--inflate-runtime-dir", type=Path)
    parser.add_argument("--source-archive-for-parity", type=Path)
    parser.add_argument("--inflate-timeout-seconds", type=int, default=3600)
    parser.add_argument("--inflate-work-dir", type=Path)
    parser.add_argument("--keep-inflate-work-dir", action="store_true")
    parser.add_argument("--fail-if-receiver-blocked", action="store_true")
    parser.add_argument("--fail-if-inflate-parity-blocked", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    input_paths = [args.candidate_archive_template, args.inverse_action_functional]
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    try:
        chain = build_inverse_scorer_cell_candidate_chain(
            raw_contest_video_digest=args.raw_contest_video_digest,
            candidate_archive_template=args.candidate_archive_template,
            inverse_action_functional=args.inverse_action_functional,
            output_dir=output_dir,
            atom_ids=tuple(args.atom_id),
            selected_limit=args.selected_limit,
            repo_root=REPO_ROOT,
            min_free_bytes=args.min_free_bytes,
            source_inflate_output_dir=args.source_inflate_output_dir,
            candidate_inflate_output_dir=args.candidate_inflate_output_dir,
            inflate_runtime_dir=args.inflate_runtime_dir,
            source_archive_for_parity=args.source_archive_for_parity,
            inflate_timeout_seconds=args.inflate_timeout_seconds,
            inflate_work_dir=args.inflate_work_dir,
            keep_inflate_work_dir=args.keep_inflate_work_dir,
        )
    except (OSError, InverseScorerCellChainError) as exc:
        print(f"FATAL: inverse-scorer cell candidate chain failed: {exc}", file=sys.stderr)
        return 2

    chain_manifest = output_dir / CHAIN_MANIFEST_NAME
    existing_sha = sha256_file(chain_manifest)
    chain = attach_tool_run_manifest(
        chain,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=input_paths,
        repo_root=REPO_ROOT,
        output_path=chain_manifest,
    )
    write_json_artifact(
        chain_manifest,
        chain,
        allow_overwrite=True,
        expected_existing_sha256=existing_sha,
    )
    print(json_text(chain), end="")
    if args.fail_if_receiver_blocked and chain.get("receiver_contract_satisfied") is not True:
        return 1
    if args.fail_if_inflate_parity_blocked and chain.get("inflate_parity_satisfied") is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
