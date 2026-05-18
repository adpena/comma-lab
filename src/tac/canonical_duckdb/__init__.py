# SPDX-License-Identifier: MIT
"""Canonical DuckDB read-model over the existing fcntl-locked JSONL/JSON/MD canonicals.

[verified-against: .omx/research/empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518.md]
[verified-against: Catalog #128 fcntl-locked sister discipline]
[verified-against: Catalog #131 no-bare-writes-to-shared-state]
[verified-against: Catalog #138 strict-load-discipline]
[verified-against: Catalog #213 HF download canonical cache]
[verified-against: Catalog #245 modal_call_id_ledger 4-layer canonical pattern]
[verified-against: Catalog #265 canonical-contract tokens]
[verified-against: Catalog #287 evidence-tag discipline]
[verified-against: Catalog #305 observability surface]
[verified-against: Catalog #323 canonical Provenance contract]

DuckDB is a CONSUMER (read-model) NOT a SOURCE OF TRUTH. The fcntl-locked JSONL/JSON
canonicals remain authoritative per CLAUDE.md "Operator gates must be wired and used".
DuckDB is refreshed on operator demand via `tools/refresh_canonical_duckdb.py`.

Catalog # citations: this module is the canonical surface for the per-X codec
planner + canonical DuckDB unification per
.omx/research/empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518.md
operator directive 2026-05-18.

Public API (Section 2.1 of the design memo):

- `BOOTSTRAP_SQL`: the canonical 10-table schema CREATE TABLE statements
- `CANONICAL_TABLES`: tuple of canonical table names
- `connect(db_path, *, read_only=False)`: open + bootstrap a DuckDB connection
- `refresh_all_tables(repo_root, *, db_path=DEFAULT_DB_PATH)`: full refresh
- `refresh_table(table, repo_root, *, db_path=DEFAULT_DB_PATH)`: per-table refresh
- `push_to_hf(table, hub_id, *, private=True, operator_approved=False)`: HF Datasets push
- `audit_table_provenance(table, *, db_path)`: Provenance audit per Catalog #323
"""
from __future__ import annotations

from pathlib import Path

from tac.canonical_duckdb.backfill import (
    REFRESH_DISPATCH,
    refresh_all_tables,
    refresh_table,
)
from tac.canonical_duckdb.hf_push import push_to_hf
from tac.canonical_duckdb.query import (
    CANONICAL_QUERY_SCRIPTS,
    audit_table_provenance,
    run_canonical_query,
)
from tac.canonical_duckdb.schema import (
    BOOTSTRAP_SQL,
    CANONICAL_TABLES,
    connect,
)

DEFAULT_DB_PATH = Path(".omx/state/canonical.duckdb")
"""Default canonical DuckDB path."""

CANONICAL_LOCK_PATH = Path(".omx/state/.canonical_duckdb.lock")
"""Sister lock path per Catalog #131 sister discipline."""


__all__ = [
    "BOOTSTRAP_SQL",
    "CANONICAL_LOCK_PATH",
    "CANONICAL_QUERY_SCRIPTS",
    "CANONICAL_TABLES",
    "DEFAULT_DB_PATH",
    "REFRESH_DISPATCH",
    "audit_table_provenance",
    "connect",
    "push_to_hf",
    "refresh_all_tables",
    "refresh_table",
    "run_canonical_query",
]
