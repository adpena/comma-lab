# SPDX-License-Identifier: MIT
"""Schema and connection helpers for the canonical DuckDB read-model.

[verified-against: .omx/research/empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518.md]
[verified-against: Catalog #128 fcntl-locked sister discipline]
[verified-against: Catalog #131 no-bare-writes-to-shared-state]
[verified-against: Catalog #245 modal_call_id_ledger 4-layer canonical pattern]
[verified-against: Catalog #265 canonical-contract tokens]

Catalog # citations: this module deliberately makes DuckDB a read-model over
existing JSON/JSONL/Markdown canonicals. It does not replace those stores.
"""
from __future__ import annotations

from pathlib import Path


CANONICAL_TABLES: tuple[str, ...] = (
    "lanes",
    "council_deliberations",
    "probe_outcomes",
    "dispatch_claims",
    "modal_call_ids",
    "master_gradient_anchors",
    "continual_learning_posterior",
    "research_memos",
    "state_json_files",
    "state_jsonl_files",
)
"""Canonical table names from the DuckDB unification design memo."""


BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS lanes (
    lane_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    phase DOUBLE,
    level INTEGER,
    gates JSON,
    notes TEXT,
    research_only BOOLEAN,
    refreshed_at_utc TIMESTAMP,
    source_path VARCHAR
);

CREATE TABLE IF NOT EXISTS council_deliberations (
    row_id VARCHAR PRIMARY KEY,
    source_path VARCHAR,
    payload JSON,
    refreshed_at_utc TIMESTAMP
);

CREATE TABLE IF NOT EXISTS probe_outcomes (
    row_id VARCHAR PRIMARY KEY,
    source_path VARCHAR,
    payload JSON,
    refreshed_at_utc TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dispatch_claims (
    row_id VARCHAR PRIMARY KEY,
    source_path VARCHAR,
    raw_line TEXT,
    refreshed_at_utc TIMESTAMP
);

CREATE TABLE IF NOT EXISTS modal_call_ids (
    row_id VARCHAR PRIMARY KEY,
    source_path VARCHAR,
    payload JSON,
    refreshed_at_utc TIMESTAMP
);

CREATE TABLE IF NOT EXISTS master_gradient_anchors (
    row_id VARCHAR PRIMARY KEY,
    source_path VARCHAR,
    payload JSON,
    refreshed_at_utc TIMESTAMP
);

CREATE TABLE IF NOT EXISTS continual_learning_posterior (
    row_id VARCHAR PRIMARY KEY,
    source_path VARCHAR,
    payload JSON,
    refreshed_at_utc TIMESTAMP
);

CREATE TABLE IF NOT EXISTS research_memos (
    path VARCHAR PRIMARY KEY,
    title VARCHAR,
    date_utc VARCHAR,
    lane_id VARCHAR,
    research_only BOOLEAN,
    score_claim BOOLEAN,
    promotion_eligible BOOLEAN,
    sha256 VARCHAR,
    bytes BIGINT,
    frontmatter JSON,
    refreshed_at_utc TIMESTAMP
);

CREATE TABLE IF NOT EXISTS state_json_files (
    path VARCHAR PRIMARY KEY,
    sha256 VARCHAR,
    bytes BIGINT,
    payload JSON,
    refreshed_at_utc TIMESTAMP
);

CREATE TABLE IF NOT EXISTS state_jsonl_files (
    path VARCHAR PRIMARY KEY,
    sha256 VARCHAR,
    bytes BIGINT,
    row_count BIGINT,
    refreshed_at_utc TIMESTAMP
);
"""
"""DuckDB bootstrap SQL."""


def connect(db_path: str | Path, *, read_only: bool = False):
    """Open a DuckDB connection and bootstrap schema unless read-only.

    Parameters
    ----------
    db_path:
        Target DuckDB file.
    read_only:
        Passed to DuckDB. When true, schema bootstrap is skipped so read-only
        consumers do not attempt writes.
    """
    import duckdb

    path = Path(db_path)
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        con.execute(BOOTSTRAP_SQL)
    return con


__all__ = ["BOOTSTRAP_SQL", "CANONICAL_TABLES", "connect"]
