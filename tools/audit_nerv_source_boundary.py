#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit NeRV eval-time source for uncharged learned payload leakage."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_source_boundary_audit import (  # noqa: E402
    audit_nerv_source_boundary,
)
from tac.repo_io import write_json_artifact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_json = args.output_json or (
        Path("/Volumes/VertigoDataTier/pact")
        / f"nerv_source_boundary_audit_{_stamp()}"
        / "source_boundary_audit.json"
    )
    payload = audit_nerv_source_boundary(
        source_paths=args.source_path,
        archive_zip=args.archive_zip,
        mode=args.mode,
        large_source_bytes=args.large_source_bytes,
        large_literal_bytes=args.large_literal_bytes,
    )
    result = write_json_artifact(output_json, payload)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "output_json": result.path,
                "output_json_sha256": result.sha256,
                "mode": payload["mode"],
                "source_boundary_clean": payload["source_boundary_clean"],
                "ready_for_witness_compile": payload["ready_for_witness_compile"],
                "long_training_gate_satisfied": payload[
                    "long_training_gate_satisfied"
                ],
                "blocker_count": len(payload["blockers"]),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0 if payload["source_boundary_clean"] else 2


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-path",
        action="append",
        required=True,
        type=Path,
        help="Eval-time source file or directory to audit. Repeatable.",
    )
    parser.add_argument("--archive-zip", type=Path)
    parser.add_argument(
        "--mode",
        choices=["conservative", "aggressive"],
        default="conservative",
    )
    parser.add_argument("--large-source-bytes", type=int, default=64_000)
    parser.add_argument("--large-literal-bytes", type=int, default=16_384)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(sys.argv[1:] if argv is None else argv)


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
