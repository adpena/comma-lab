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
    parser.add_argument(
        "--multi-granularity-archive-sha256",
        help="Optional archive SHA filter for bounded multi_granularity_sensitivity refresh.",
    )
    parser.add_argument(
        "--multi-granularity-class-source",
        type=Path,
        help="Class-weight/source JSON required to materialize per-class ITEM_8 rows.",
    )
    parser.add_argument(
        "--multi-granularity-byte-offsets",
        help="Comma-separated byte offsets for bounded ITEM_8 materialization.",
    )
    parser.add_argument(
        "--multi-granularity-top-k-bytes",
        type=int,
        help="Materialize only the top-K bytes by per-pair absolute sensitivity.",
    )
    parser.add_argument(
        "--multi-granularity-max-rows",
        type=int,
        default=1_000_000,
        help="Fail closed if ITEM_8 materialization would emit more rows.",
    )
    return parser.parse_args(argv)


def _parse_int_csv(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return [int(part) for part in parts]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from tac.canonical_duckdb import (
        CANONICAL_TABLES,
        push_canonical_task_status_to_hf,
        refresh_all_tables,
        refresh_table,
    )
    from tac.canonical_duckdb.per_byte_sensitivity_ext import (
        refresh_multi_granularity_sensitivity_table,
        refresh_per_byte_sensitivity_table,
    )

    multi_granularity_options = {
        "archive_sha256": args.multi_granularity_archive_sha256,
        "class_source_path": args.multi_granularity_class_source,
        "byte_offsets": _parse_int_csv(args.multi_granularity_byte_offsets),
        "top_k_bytes": args.multi_granularity_top_k_bytes,
        "max_rows": args.multi_granularity_max_rows,
    }
    extension_refreshers = {
        "multi_granularity_sensitivity": lambda root, *, db_path: refresh_multi_granularity_sensitivity_table(
            root,
            db_path=db_path,
            **multi_granularity_options,
        ),
        "per_byte_sensitivity": refresh_per_byte_sensitivity_table,
    }

    repo_root = args.repo_root.resolve()
    db_path = args.db_path
    if args.push_hf_canonical_task_status and args.hf_public:
        raise SystemExit(
            "canonical_task_status HF export is private-only; public disclosure "
            "requires a sanitized projection and hygiene audit"
        )
    if args.tables == ["all"]:
        result = refresh_all_tables(repo_root, db_path=db_path)
        result.update(
            {
                table: refresher(repo_root, db_path=db_path)
                for table, refresher in extension_refreshers.items()
            }
        )
    else:
        known_tables = set(CANONICAL_TABLES) | set(extension_refreshers)
        unknown = sorted(set(args.tables) - known_tables)
        if unknown:
            raise SystemExit(f"unknown canonical DuckDB tables: {', '.join(unknown)}")
        result = {
            table: extension_refreshers[table](repo_root, db_path=db_path)
            if table in extension_refreshers
            else refresh_table(table, repo_root, db_path=db_path)
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
