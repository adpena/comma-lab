#!/usr/bin/env python3
"""Atomically materialize one explicit nondominated G119 row for public eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.witness_dsl.taskspace_g110_release_materializer_v1 import (
    capture_public_runtime_v1,
    materialize_g110_release_v1,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    subparsers.add_parser(
        "runtime-id",
        help="print the current closed G110 public-runtime identity",
    )
    materialize = subparsers.add_parser(
        "materialize",
        help="publish archive.zip, the public runtime, and a sealed receipt",
    )
    materialize.add_argument(
        "--joint-ledger",
        type=Path,
        required=True,
        help="physical g119_post_g105_joint_axes.json",
    )
    materialize.add_argument(
        "--expected-joint-ledger-file-sha256",
        required=True,
        help="SHA-256 of the exact physical G119 ledger file",
    )
    materialize.add_argument(
        "--joint-row-sha256",
        required=True,
        help="explicit nondominated G119 row identity; never inferred as BEST",
    )
    materialize.add_argument(
        "--expected-runtime-tree-sha256",
        required=True,
        help="tree identity printed by the runtime-id operation",
    )
    materialize.add_argument(
        "--resume-from",
        type=Path,
        required=True,
        help="absolute SSD submission directory; cold start and resume use the same path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation == "runtime-id":
        try:
            snapshot = capture_public_runtime_v1(
                repo_root=REPO_ROOT,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"G110_RELEASE_REFUSED: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "runtime_root": str(snapshot.root),
                    "runtime_tree_sha256": snapshot.tree_sha256,
                    "files": list(snapshot.files),
                },
                sort_keys=True,
            )
        )
        return 0
    command = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    try:
        result = materialize_g110_release_v1(
            joint_ledger_path=args.joint_ledger,
            expected_joint_ledger_file_sha256=(
                args.expected_joint_ledger_file_sha256
            ),
            joint_row_sha256=args.joint_row_sha256,
            expected_runtime_tree_sha256=(
                args.expected_runtime_tree_sha256
            ),
            output_root=args.resume_from,
            command=command,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"G110_RELEASE_REFUSED: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "submission_dir": str(result.submission_dir),
                "archive_path": str(result.archive_path),
                "archive_bytes": result.archive_bytes,
                "archive_sha256": result.archive_sha256,
                "runtime_tree_sha256": result.runtime_tree_sha256,
                "release_receipt_path": str(
                    result.release_receipt_path
                ),
                "release_receipt_file_sha256": (
                    result.release_receipt_file_sha256
                ),
                "release_receipt_body_sha256": (
                    result.release_receipt_body_sha256
                ),
                "upstream_evaluate_py_run": False,
                "score_claim": False,
                "pointer_moved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
