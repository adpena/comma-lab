#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a repair-campaign work order from real byte-closed archives."""

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

from tac.optimization.repair_archive_candidate_intake import (  # noqa: E402
    RepairArchiveCandidateIntakeError,
    build_repair_campaign_work_order_from_archives,
)
from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", required=True, type=Path)
    parser.add_argument("--source-label", action="append", default=[])
    parser.add_argument("--training-artifact", action="append", default=[], type=Path)
    parser.add_argument("--equivalence-gate", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-order-out", required=True, type=Path)
    parser.add_argument("--chain-id", default="real_archive_repair_campaign")
    parser.add_argument("--receiver-closed-saved-bytes-floor", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _expected(path: Path, *, overwrite: bool) -> str | None:
    return sha256_file(path) if overwrite and path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        work_order = build_repair_campaign_work_order_from_archives(
            archive_paths=args.archive,
            output_dir=args.output_dir,
            repo_root=REPO_ROOT,
            source_labels=args.source_label,
            training_artifact_paths=args.training_artifact,
            equivalence_gate_paths=args.equivalence_gate,
            chain_id=args.chain_id,
            receiver_closed_saved_bytes_floor=args.receiver_closed_saved_bytes_floor,
            overwrite=args.overwrite,
        )
        write_json_artifact(
            args.work_order_out,
            work_order,
            allow_overwrite=args.overwrite,
            expected_existing_sha256=_expected(args.work_order_out, overwrite=args.overwrite),
        )
    except (OSError, ArtifactWriteError, RepairArchiveCandidateIntakeError, ValueError) as exc:
        print(f"FATAL: repair archive candidate intake failed: {exc}", file=sys.stderr)
        return 2
    print(json_text(work_order), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
