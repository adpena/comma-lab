# SPDX-License-Identifier: MIT
"""Query helpers for the canonical DuckDB read-model.

[verified-against: .omx/research/empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518.md]
[verified-against: Catalog #245 modal_call_id_ledger 4-layer canonical pattern]
[verified-against: Catalog #265 canonical-contract tokens]
[verified-against: Catalog #323 canonical Provenance contract]

Catalog # citations: queries are read-only views over derived tables. They do
not alter source-of-truth JSON/JSONL/Markdown ledgers.
"""
from __future__ import annotations

from pathlib import Path

from tac.canonical_duckdb.schema import CANONICAL_TABLES, connect

EXTENSION_TABLE_SOURCES: dict[str, str] = {
    "multi_granularity_sensitivity": (
        "per_pair master_gradient anchor + explicit class source sidecar"
    ),
    "multi_granularity_sensitivity_runs": (
        "per_pair master_gradient anchor + explicit class source sidecar"
    ),
    "per_byte_sensitivity": "master_gradient_anchors.jsonl + .npy sidecars",
}
"""Additive DuckDB extension tables that remain derived read-models."""

CANONICAL_QUERY_SCRIPTS: dict[str, str] = {
    "research_memos_by_lane": """
        SELECT lane_id, COUNT(*) AS memo_count, SUM(bytes) AS total_bytes
        FROM research_memos
        WHERE lane_id IS NOT NULL AND lane_id != ''
        GROUP BY lane_id
        ORDER BY memo_count DESC, lane_id
    """,
    "lanes_with_research_memos": """
        SELECT l.lane_id, l.name, l.level, COUNT(r.path) AS memo_count
        FROM lanes l
        LEFT JOIN research_memos r ON r.lane_id = l.lane_id
        GROUP BY l.lane_id, l.name, l.level
        ORDER BY memo_count DESC, l.lane_id
    """,
    "state_file_inventory": """
        SELECT 'json' AS kind, COUNT(*) AS file_count, SUM(bytes) AS total_bytes
        FROM state_json_files
        UNION ALL
        SELECT 'jsonl' AS kind, COUNT(*) AS file_count, SUM(bytes) AS total_bytes
        FROM state_jsonl_files
    """,
    "canonical_task_status_by_memo": """
        SELECT
            t.source_design_memo,
            COALESCE(r.title, t.source_design_memo) AS memo_title,
            COUNT(*) AS task_count,
            SUM(CASE WHEN t.status = 'pending' THEN 1 ELSE 0 END) AS pending_count,
            SUM(CASE WHEN t.status = 'in_progress' THEN 1 ELSE 0 END) AS in_progress_count,
            SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) AS completed_count,
            SUM(CASE WHEN t.status = 'blocked' THEN 1 ELSE 0 END) AS blocked_count,
            SUM(CASE WHEN t.test_status = 'green' THEN 1 ELSE 0 END) AS green_test_count,
            MIN(t.event_timestamp_utc) AS oldest_event_timestamp_utc,
            MAX(t.event_timestamp_utc) AS newest_event_timestamp_utc
        FROM canonical_task_status_latest t
        LEFT JOIN research_memos r ON r.path = t.source_design_memo
        GROUP BY t.source_design_memo, memo_title
        ORDER BY pending_count DESC, in_progress_count DESC, task_count DESC, t.source_design_memo
    """,
    "canonical_task_status_pending_with_memo": """
        SELECT
            t.task_id,
            t.title,
            t.status,
            t.owner,
            t.source_design_memo,
            COALESCE(r.title, t.source_design_memo) AS memo_title,
            r.lane_id,
            r.research_only,
            t.predicted_cost_usd,
            t.predicted_delta_s_lower,
            t.predicted_delta_s_upper,
            t.started_at_utc,
            t.event_timestamp_utc
        FROM canonical_task_status_latest t
        LEFT JOIN research_memos r ON r.path = t.source_design_memo
        WHERE t.status IN ('pending', 'in_progress', 'blocked')
        ORDER BY
            CASE t.status
                WHEN 'in_progress' THEN 0
                WHEN 'blocked' THEN 1
                ELSE 2
            END,
            t.predicted_delta_s_lower ASC NULLS LAST,
            t.predicted_cost_usd ASC NULLS LAST,
            t.event_timestamp_utc ASC,
            t.task_id
    """,
}
"""Named read-only SQL queries exposed by the package."""


def run_canonical_query(
    query_name: str,
    *,
    db_path: str | Path,
) -> list[dict]:
    """Run one named canonical query and return row dictionaries."""
    sql = CANONICAL_QUERY_SCRIPTS.get(query_name)
    if sql is None:
        raise ValueError(f"unknown canonical DuckDB query: {query_name}")
    con = connect(db_path, read_only=True)
    try:
        return con.execute(sql).fetchdf().to_dict(orient="records")
    finally:
        con.close()


def audit_table_provenance(
    table: str,
    *,
    db_path: str | Path,
) -> dict:
    """Return a lightweight provenance audit for one derived table."""
    if table not in CANONICAL_TABLES and table not in EXTENSION_TABLE_SOURCES:
        raise ValueError(f"unknown canonical DuckDB table: {table}")
    con = connect(db_path, read_only=True)
    try:
        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        columns = [row[1] for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
    finally:
        con.close()
    return {
        "table": table,
        "row_count": int(row_count),
        "columns": list(columns),
        "source_of_truth": EXTENSION_TABLE_SOURCES.get(table, "json_jsonl_markdown"),
        "duckdb_is_source_of_truth": False,
    }


__all__ = [
    "CANONICAL_QUERY_SCRIPTS",
    "EXTENSION_TABLE_SOURCES",
    "audit_table_provenance",
    "run_canonical_query",
]
