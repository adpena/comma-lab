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

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any

from tac.canonical_duckdb.query import audit_table_provenance
from tac.canonical_duckdb.schema import CANONICAL_TABLES, connect

DEFAULT_CANONICAL_TASK_STATUS_HF_DATASET_ID = "adpena/pact-canonical-task-status"


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_manifest(path: Path, *, path_in_repo: str) -> dict:
    return {
        "path_in_repo": path_in_repo,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _local_path_for_repo_path(export_dir: Path, path_in_repo: str) -> Path:
    return export_dir / Path(path_in_repo)


def _write_export_manifest(manifest_path: Path, manifest: dict) -> None:
    payload = {key: value for key, value in manifest.items() if key != "manifest"}
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = _file_manifest(
        manifest_path,
        path_in_repo="metadata/canonical_task_status_hf_manifest.json",
    )


def _write_canonical_task_status_dataset_card(card_path: Path, *, hub_id: str | None) -> None:
    hub_line = f"\nHub id: `{hub_id}`\n" if hub_id else ""
    card_path.write_text(
        "---\n"
        "license: mit\n"
        "pretty_name: Pact canonical task status read model\n"
        "---\n"
        "\n"
        "# Pact Canonical Task Status Read Model\n"
        "\n"
        "This private dataset is an operator-custody read model for Pact task-status "
        "observability. The source of truth remains the append-only ledger at "
        "`.omx/state/canonical_task_status.jsonl`; DuckDB and these parquet files "
        "are derived consumers, not authority surfaces.\n"
        "\n"
        "The raw export is private-only. Public disclosure requires a separate "
        "sanitized projection and release-hygiene audit.\n"
        f"{hub_line}"
        "\n"
        "Files:\n"
        "- `data/canonical_task_status.parquet`: full append-only event history.\n"
        "- `data/canonical_task_status_latest.parquet`: latest row per task id.\n"
        "- `metadata/canonical_task_status_hf_manifest.json`: hashes, row counts, "
        "source-ledger watermark, and privacy status.\n",
        encoding="utf-8",
    )


def _canonical_task_status_source_metadata(repo_root: str | Path) -> dict:
    from tac.canonical_task_status.loader import (
        ledger_path,
        load_canonical_task_status_strict,
    )

    root = Path(repo_root)
    path = ledger_path(root)
    rows = load_canonical_task_status_strict(root)
    return {
        "source_ledger_path": ".omx/state/canonical_task_status.jsonl",
        "source_ledger_bytes": path.stat().st_size if path.exists() else 0,
        "source_ledger_sha256": _sha256_file(path) if path.exists() else None,
        "source_ledger_rows": len(rows),
        "source_ledger_latest_event_timestamp_utc": (
            max((row.event_timestamp_utc for row in rows), default=None)
        ),
    }


def _assert_canonical_task_status_duckdb_fresh(
    duckdb_path: str | Path,
    source_metadata: dict,
) -> None:
    con = connect(duckdb_path, read_only=True)
    try:
        row_count = con.execute("SELECT COUNT(*) FROM canonical_task_status").fetchone()[0]
        latest_event = con.execute(
            "SELECT MAX(event_timestamp_utc) FROM canonical_task_status"
        ).fetchone()[0]
    finally:
        con.close()
    if latest_event is None:
        latest_event_text = None
    else:
        if latest_event.tzinfo is None:
            latest_event = latest_event.replace(tzinfo=_dt.UTC)
        else:
            latest_event = latest_event.astimezone(_dt.UTC)
        latest_event_text = latest_event.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if int(row_count) != int(source_metadata["source_ledger_rows"]):
        raise RuntimeError(
            "canonical_task_status DuckDB read-model is stale: "
            f"duckdb_rows={row_count} source_rows={source_metadata['source_ledger_rows']}"
        )
    if latest_event_text != source_metadata["source_ledger_latest_event_timestamp_utc"]:
        raise RuntimeError(
            "canonical_task_status DuckDB read-model latest event mismatch: "
            f"duckdb_latest={latest_event_text} "
            f"source_latest={source_metadata['source_ledger_latest_event_timestamp_utc']}"
        )


def _ensure_hf_repo_private(api: Any, *, repo_id: str) -> str | None:
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
    api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=True)
    info = api.repo_info(repo_id, repo_type="dataset")
    if not bool(info.private):
        raise RuntimeError(
            f"HF visibility mismatch for {repo_id}: expected private=True, "
            f"observed {info.private}"
        )
    return getattr(info, "sha", None)


def export_canonical_task_status_hf_dataset(
    duckdb_path: str | Path,
    *,
    export_dir: str | Path,
    repo_root: str | Path | None = None,
    source_metadata: dict | None = None,
) -> dict:
    """Export canonical task status history + latest view for HF Datasets.

    The JSONL ledger remains the source of truth. This export is a read-model
    snapshot designed for private HF Dataset publication and operator SQL use.
    """
    out_dir = Path(export_dir)
    data_dir = out_dir / "data"
    meta_dir = out_dir / "metadata"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "README.md"
    _write_canonical_task_status_dataset_card(card_path, hub_id=None)
    if source_metadata is None and repo_root is not None:
        source_metadata = _canonical_task_status_source_metadata(repo_root)
    if source_metadata is not None:
        _assert_canonical_task_status_duckdb_fresh(duckdb_path, source_metadata)

    history_path = data_dir / "canonical_task_status.parquet"
    latest_path = data_dir / "canonical_task_status_latest.parquet"

    con = connect(duckdb_path, read_only=True)
    try:
        con.execute(
            "COPY (SELECT * FROM canonical_task_status ORDER BY event_timestamp_utc, task_id) "
            "TO ? (FORMAT PARQUET)",
            [str(history_path)],
        )
        con.execute(
            "COPY (SELECT * FROM canonical_task_status_latest ORDER BY task_id) "
            "TO ? (FORMAT PARQUET)",
            [str(latest_path)],
        )
        history_rows = con.execute("SELECT COUNT(*) FROM canonical_task_status").fetchone()[0]
        latest_rows = con.execute("SELECT COUNT(*) FROM canonical_task_status_latest").fetchone()[0]
    finally:
        con.close()

    duckdb_file = Path(duckdb_path)
    try:
        duckdb_display_path = (
            duckdb_file.resolve().relative_to(Path(repo_root).resolve()).as_posix()
            if repo_root is not None
            else duckdb_file.as_posix()
        )
    except ValueError:
        duckdb_display_path = duckdb_file.name
    manifest = {
        "schema_version": "canonical_task_status_hf_export_v1_20260518",
        "generated_at_utc": _utc_now(),
        "dataset_kind": "canonical_task_status_read_model",
        "source_of_truth": ".omx/state/canonical_task_status.jsonl",
        "duckdb_is_source_of_truth": False,
        "duckdb_path": duckdb_display_path,
        "duckdb_sha256": _sha256_file(duckdb_file) if duckdb_file.exists() else None,
        "history_rows": int(history_rows),
        "latest_rows": int(latest_rows),
        "private_only": True,
        "provenance": audit_table_provenance("canonical_task_status", db_path=duckdb_path),
        "source": source_metadata,
        "files": [
            _file_manifest(history_path, path_in_repo="data/canonical_task_status.parquet"),
            _file_manifest(latest_path, path_in_repo="data/canonical_task_status_latest.parquet"),
        ],
        "dataset_card": _file_manifest(card_path, path_in_repo="README.md"),
    }
    manifest_path = meta_dir / "canonical_task_status_hf_manifest.json"
    _write_export_manifest(manifest_path, manifest)
    return manifest


def push_canonical_task_status_to_hf(
    duckdb_path: str | Path,
    hf_dataset_id: str = DEFAULT_CANONICAL_TASK_STATUS_HF_DATASET_ID,
    *,
    export_dir: str | Path | None = None,
    token: str | None = None,
    private: bool = True,
    operator_approved: bool = False,
    dry_run: bool = True,
    repo_root: str | Path | None = None,
    refresh_before_export: bool = True,
    hf_api: Any | None = None,
) -> dict:
    """Export and optionally push canonical task status to a private HF Dataset.

    The default is a dry run that writes parquet + a manifest and fires no
    network upload. A real upload requires both ``dry_run=False`` and
    ``operator_approved=True`` so private operator state is never published by
    an accidental session-end refresh.
    """
    if not private:
        raise PermissionError(
            "canonical_task_status raw HF export is private-only; public disclosure "
            "requires a separate sanitized projection and hygiene audit"
        )
    if repo_root is not None and refresh_before_export:
        from tac.canonical_duckdb.backfill import refresh_table

        refresh_table("canonical_task_status", Path(repo_root), db_path=duckdb_path)
    source_metadata = (
        _canonical_task_status_source_metadata(repo_root)
        if repo_root is not None
        else None
    )
    target_dir = (
        Path(export_dir)
        if export_dir is not None
        else Path(duckdb_path).parent / "canonical_duckdb_hf_exports" / "canonical_task_status"
    )
    manifest = export_canonical_task_status_hf_dataset(
        duckdb_path,
        export_dir=target_dir,
        repo_root=repo_root,
        source_metadata=source_metadata,
    )
    manifest.update(
        {
            "hub_id": hf_dataset_id,
            "hub_url": f"https://huggingface.co/datasets/{hf_dataset_id}",
            "private": private,
            "operator_approved": operator_approved,
            "remote_push_fired": False,
        }
    )
    _write_canonical_task_status_dataset_card(
        _local_path_for_repo_path(target_dir, manifest["dataset_card"]["path_in_repo"]),
        hub_id=hf_dataset_id,
    )
    manifest["dataset_card"] = _file_manifest(
        _local_path_for_repo_path(target_dir, "README.md"),
        path_in_repo="README.md",
    )

    if dry_run:
        manifest["status"] = (
            "dry_run_operator_approved"
            if operator_approved
            else "dry_run_pending_operator_approval"
        )
        _write_export_manifest(
            _local_path_for_repo_path(target_dir, manifest["manifest"]["path_in_repo"]),
            manifest,
        )
        return manifest
    if not operator_approved:
        raise PermissionError(
            "canonical_task_status HF push requires operator_approved=True "
            "when dry_run=False"
        )

    from huggingface_hub import HfApi, get_token

    resolved_token = token or get_token()
    if hf_api is None and not resolved_token:
        raise RuntimeError("No HF token available for canonical_task_status HF push")

    api = hf_api or HfApi(token=resolved_token)
    manifest["hf_repo_sha_before_upload"] = _ensure_hf_repo_private(api, repo_id=hf_dataset_id)
    for file_row in [*manifest["files"], manifest["dataset_card"]]:
        local_path = _local_path_for_repo_path(target_dir, file_row["path_in_repo"])
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=file_row["path_in_repo"],
            repo_id=hf_dataset_id,
            repo_type="dataset",
            commit_message="canonical_duckdb: refresh canonical_task_status",
        )
    manifest["status"] = "pushed"
    manifest["remote_push_fired"] = True
    info_after = api.repo_info(hf_dataset_id, repo_type="dataset")
    if not bool(info_after.private):
        raise RuntimeError(
            f"HF visibility mismatch after upload for {hf_dataset_id}: "
            f"expected private=True, observed {info_after.private}"
        )
    manifest["hf_repo_private_verified_after_upload"] = True
    manifest["hf_repo_sha_after_data_upload"] = getattr(info_after, "sha", None)
    _write_export_manifest(
        _local_path_for_repo_path(target_dir, manifest["manifest"]["path_in_repo"]),
        manifest,
    )
    api.upload_file(
        path_or_fileobj=str(_local_path_for_repo_path(target_dir, manifest["manifest"]["path_in_repo"])),
        path_in_repo=manifest["manifest"]["path_in_repo"],
        repo_id=hf_dataset_id,
        repo_type="dataset",
        commit_message="canonical_duckdb: refresh canonical_task_status manifest",
    )
    return manifest


def push_to_hf(
    table: str,
    hub_id: str,
    *,
    db_path: str | Path,
    export_dir: str | Path,
    token: str | None = None,
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


__all__ = [
    "DEFAULT_CANONICAL_TASK_STATUS_HF_DATASET_ID",
    "export_canonical_task_status_hf_dataset",
    "push_canonical_task_status_to_hf",
    "push_to_hf",
]
