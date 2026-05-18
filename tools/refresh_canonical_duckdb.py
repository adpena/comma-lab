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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from tac.canonical_duckdb import CANONICAL_TABLES, refresh_all_tables, refresh_table

    repo_root = args.repo_root.resolve()
    db_path = args.db_path
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
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
