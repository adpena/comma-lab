#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the local byte-range entropy-recode proof chain."""

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

from tac.optimization.byte_range_entropy_recode_chain import (  # noqa: E402
    ByteRangeEntropyRecodeChainError,
    build_byte_range_entropy_recode_chain,
)
from tac.pr103_arithmetic_transform_plan import RETUNABLE_BROTLI_SECTIONS  # noqa: E402
from tac.repo_io import json_text, sha256_file, write_json_artifact  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-manifest", required=True, type=Path)
    parser.add_argument(
        "--beam-probe-report",
        required=True,
        action="append",
        type=Path,
        help="Tracked PR103 beam-search report. Repeat to compose streams.",
    )
    parser.add_argument("--source-runtime-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--global-combo-report",
        type=Path,
        help="Optional global-combo report selecting candidate moves.",
    )
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--member-name")
    parser.add_argument(
        "--retune-brotli-section",
        action="append",
        choices=sorted(RETUNABLE_BROTLI_SECTIONS),
        default=[],
    )
    parser.add_argument(
        "--min-free-bytes",
        type=int,
        default=0,
        help="Additional free-byte floor required before manifest writes.",
    )
    parser.add_argument(
        "--fail-if-receiver-blocked",
        action="store_true",
        help="Exit 1 unless the runtime receiver contract is satisfied.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    input_paths = [
        args.schema_manifest,
        *args.beam_probe_report,
        *_runtime_input_files(args.source_runtime_dir),
        *([args.global_combo_report] if args.global_combo_report is not None else []),
        *([args.source_archive] if args.source_archive is not None else []),
    ]
    try:
        chain = build_byte_range_entropy_recode_chain(
            schema_manifest=args.schema_manifest,
            beam_probe_reports=args.beam_probe_report,
            source_runtime_dir=args.source_runtime_dir,
            output_dir=args.output_dir,
            source_archive=args.source_archive,
            global_combo_report=args.global_combo_report,
            member_name=args.member_name,
            repo_root=REPO_ROOT,
            retune_brotli_sections=args.retune_brotli_section,
            min_free_bytes=args.min_free_bytes,
        )
    except (OSError, ByteRangeEntropyRecodeChainError) as exc:
        print(f"FATAL: byte-range entropy recode chain failed: {exc}", file=sys.stderr)
        return 2

    chain_manifest = args.output_dir / "byte_range_entropy_recode_chain_manifest.json"
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
    return 0


def _runtime_input_files(runtime_dir: Path) -> list[Path]:
    if not runtime_dir.is_dir():
        return [runtime_dir]
    return [
        path
        for path in sorted(runtime_dir.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.suffix not in {".pyc", ".pyo"} and path.name != ".DS_Store"
    ]


if __name__ == "__main__":
    raise SystemExit(main())
