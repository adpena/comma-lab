# SPDX-License-Identifier: MIT
"""HF export helper for canonical DuckDB read-model tables.

[verified-against: .omx/research/empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518.md]
[verified-against: Catalog #213 HF download canonical cache]
[verified-against: Catalog #265 canonical-contract tokens]
[verified-against: Catalog #323 canonical Provenance contract]

Catalog # citations: pushing derived state to HF is operator-gated and private
by default. This helper refuses to upload without explicit operator approval.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from tac.canonical_duckdb.query import audit_table_provenance
from tac.canonical_duckdb.schema import CANONICAL_TABLES, connect


def push_to_hf(
    table: str,
    hub_id: str,
    *,
    db_path: str | Path,
    export_dir: str | Path,
    token: Optional[str] = None,
    private: bool = True,
    operator_approved: bool = False,
) -> dict:
    """Export a canonical table to parquet and push it to HF Datasets.

    The function is intentionally fail-closed:
    - `operator_approved=True` is required.
    - public pushes require `private=False` plus operator approval.
    - table provenance is returned with the upload URL.
    """
    if table not in CANONICAL_TABLES:
        raise ValueError(f"unknown canonical DuckDB table: {table}")
    if not operator_approved:
        raise PermissionError(
            "canonical DuckDB HF push requires operator_approved=True; "
            "this helper never uploads derived state by accident"
        )

    from huggingface_hub import HfApi, get_token

    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{table}.parquet"

    con = connect(db_path, read_only=True)
    try:
        con.execute(f"COPY (SELECT * FROM {table}) TO ? (FORMAT PARQUET)", [str(parquet_path)])
    finally:
        con.close()

    resolved_token = token or get_token()
    if not resolved_token:
        raise RuntimeError("No HF token available for canonical DuckDB push")

    api = HfApi(token=resolved_token)
    api.create_repo(repo_id=hub_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo=f"{table}.parquet",
        repo_id=hub_id,
        repo_type="dataset",
        commit_message=f"canonical_duckdb: refresh {table}",
    )
    return {
        "table": table,
        "hub_url": f"https://huggingface.co/datasets/{hub_id}",
        "private": private,
        "parquet_path": str(parquet_path),
        "provenance": audit_table_provenance(table, db_path=db_path),
    }


__all__ = ["push_to_hf"]
