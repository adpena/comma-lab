#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Refresh the canonical DuckDB read-model from existing Pact state files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--db-path", type=Path, default=Path(".omx/state/canonical.duckdb"))
    parser.add_argument(
        "--tables",
        nargs="+",
        default=["all"],
        help="Tables to refresh, or 'all'.",
    )
    parser.add_argument(
        "--push-hf-canonical-task-status",
        action="store_true",
        help=(
            "After refresh, export canonical_task_status history/latest parquet "
            "and optionally push to Hugging Face Datasets."
        ),
    )
    parser.add_argument(
        "--hf-dataset-id",
        default="adpena/pact-canonical-task-status",
        help="HF dataset id for --push-hf-canonical-task-status.",
    )
    parser.add_argument(
        "--hf-export-dir",
        type=Path,
        default=Path(".omx/state/canonical_duckdb_hf_exports/canonical_task_status"),
        help="Local export directory for canonical_task_status HF parquet + manifest.",
    )
    parser.add_argument(
        "--hf-public",
        action="store_true",
        help=(
            "Refused for canonical_task_status raw export. Public disclosure "
            "requires a separate sanitized projection."
        ),
    )
    parser.add_argument(
        "--operator-approved",
        action="store_true",
        help="Required with --fire-hf-push to upload derived state remotely.",
    )
    parser.add_argument(
        "--fire-hf-push",
        action="store_true",
        help=(
            "Actually upload to HF. Without this flag the command is a local dry run "
            "that writes parquet + manifest and fires no network upload."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from tac.canonical_duckdb import (
        CANONICAL_TABLES,
        push_canonical_task_status_to_hf,
        refresh_all_tables,
        refresh_table,
    )

    repo_root = args.repo_root.resolve()
    db_path = args.db_path
    if args.push_hf_canonical_task_status and args.hf_public:
        raise SystemExit(
            "canonical_task_status HF export is private-only; public disclosure "
            "requires a sanitized projection and hygiene audit"
        )
    if args.tables == ["all"]:
        result = refresh_all_tables(repo_root, db_path=db_path)
    else:
        unknown = sorted(set(args.tables) - set(CANONICAL_TABLES))
        if unknown:
            raise SystemExit(f"unknown canonical DuckDB tables: {', '.join(unknown)}")
        result = {
            table: refresh_table(table, repo_root, db_path=db_path)
            for table in args.tables
        }
    if args.push_hf_canonical_task_status:
        if args.tables != ["all"] and "canonical_task_status" not in args.tables:
            result["canonical_task_status"] = refresh_table(
                "canonical_task_status",
                repo_root,
                db_path=db_path,
            )
        result["hf_canonical_task_status"] = push_canonical_task_status_to_hf(
            db_path,
            args.hf_dataset_id,
            export_dir=args.hf_export_dir,
            private=True,
            operator_approved=args.operator_approved,
            dry_run=not args.fire_hf_push,
            repo_root=repo_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
