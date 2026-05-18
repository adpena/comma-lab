# SPDX-License-Identifier: MIT
"""Per-byte sensitivity extension for the canonical DuckDB read-model.

[verified-against: .omx/research/empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518.md Section 2.2 Table 5]
[verified-against: Catalog #265 canonical-contract tokens]
[verified-against: Catalog #287 evidence-tag discipline]
[verified-against: Catalog #305 observability surface]

This module adds the canonical `per_byte_sensitivity` DERIVED table to the
sister-landed canonical DuckDB schema. Sister `tac.canonical_duckdb` provides
the JSON-payload tables (master_gradient_anchors etc.); this extension adds
the typed per-byte rows that the per-X codec planner consumes.

The table schema is:
    per_byte_sensitivity (
        archive_sha256 VARCHAR,
        byte_idx INTEGER,
        grad_seg FLOAT,
        grad_pose FLOAT,
        grad_rate FLOAT,
        sensitivity_l1 FLOAT,            -- |grad_seg|+|grad_pose|+|grad_rate|
        sensitivity_quantile_rank DOUBLE, -- 0.0 = top byte, 1.0 = bottom byte
        sensitivity_class VARCHAR,       -- top_2pct / top_5pct / top_20pct / tail
        source_measurement_axis VARCHAR,
        source_measurement_hardware VARCHAR,
        source_measurement_method VARCHAR,
        source_measurement_utc VARCHAR,
        evidence_grade VARCHAR,
        source_anchor_authoritative BOOLEAN,
        promotion_eligible BOOLEAN,
        PRIMARY KEY (archive_sha256, byte_idx)
    )

Backfill consumes `.omx/state/master_gradient_anchors.jsonl` + `.npy` sidecars
in the canonical fcntl-locked pattern per Catalog #128 / #131 / #245 sister.

Quantile class boundaries match the Fields-Medal subagent's `sensitivity_mask_aware_quantizr_v1`
design (top 2% / next 3% [cumulative 5%] / next 15% [cumulative 20%] / tail).
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

from tac.canonical_duckdb.backfill import (
    DEFAULT_DB_PATH,
    canonical_duckdb_lock,
)
from tac.canonical_duckdb.schema import connect
from tac.master_gradient import (
    is_authoritative_axis_anchor,
    is_authoritative_contest_axis_anchor,
)


PER_BYTE_SENSITIVITY_BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS per_byte_sensitivity (
    archive_sha256 VARCHAR,
    byte_idx INTEGER,
    grad_seg FLOAT,
    grad_pose FLOAT,
    grad_rate FLOAT,
    sensitivity_l1 FLOAT,
    sensitivity_quantile_rank DOUBLE,
    sensitivity_class VARCHAR,
    PRIMARY KEY (archive_sha256, byte_idx)
);

CREATE INDEX IF NOT EXISTS idx_per_byte_archive ON per_byte_sensitivity (archive_sha256);
CREATE INDEX IF NOT EXISTS idx_per_byte_class ON per_byte_sensitivity (sensitivity_class);
"""

QUANTILE_BOUNDARIES: tuple[float, ...] = (0.02, 0.05, 0.20)
"""Cumulative-from-top quantile boundaries matching sensitivity_mask_aware_quantizr_v1.

(top 2% / top 5% [cumulative] / top 20% [cumulative]; tail = remaining 80%)
"""

CLASS_NAMES: tuple[str, ...] = ("top_2pct", "top_5pct", "top_20pct", "tail")
"""Canonical class names for sensitivity classification."""


def bootstrap_per_byte_sensitivity_table(con) -> None:
    """Create the per_byte_sensitivity table + indices (idempotent)."""
    con.execute(PER_BYTE_SENSITIVITY_BOOTSTRAP_SQL)
    for ddl in (
        "ALTER TABLE per_byte_sensitivity ADD COLUMN IF NOT EXISTS source_measurement_axis VARCHAR",
        "ALTER TABLE per_byte_sensitivity ADD COLUMN IF NOT EXISTS source_measurement_hardware VARCHAR",
        "ALTER TABLE per_byte_sensitivity ADD COLUMN IF NOT EXISTS source_measurement_method VARCHAR",
        "ALTER TABLE per_byte_sensitivity ADD COLUMN IF NOT EXISTS source_measurement_utc VARCHAR",
        "ALTER TABLE per_byte_sensitivity ADD COLUMN IF NOT EXISTS evidence_grade VARCHAR",
        "ALTER TABLE per_byte_sensitivity ADD COLUMN IF NOT EXISTS source_anchor_authoritative BOOLEAN",
        "ALTER TABLE per_byte_sensitivity ADD COLUMN IF NOT EXISTS promotion_eligible BOOLEAN",
    ):
        con.execute(ddl)


def compute_sensitivity_classes(
    sensitivity: "np.ndarray",
    *,
    boundaries: tuple[float, ...] = QUANTILE_BOUNDARIES,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Compute (rank_array, class_array) given per-byte sensitivity values.

    Args:
        sensitivity: shape (N,) per-byte L1 sensitivity.
        boundaries: cumulative quantile cutoffs from top, e.g. (0.02, 0.05, 0.20).
        class_names: matching class names (len = len(boundaries) + 1).

    Returns:
        (rank_array, class_array):
        - rank_array: shape (N,) where rank[i] in [0, 1] = position in descending sensitivity (0.0 = highest)
        - class_array: shape (N,) dtype object; class name per byte

    Raises:
        ValueError: if boundaries + class_names lengths mismatch.
    """
    if np is None:  # pragma: no cover
        raise RuntimeError("numpy required for sensitivity classification")
    if len(class_names) != len(boundaries) + 1:
        raise ValueError(
            f"len(class_names)={len(class_names)} must equal "
            f"len(boundaries)+1={len(boundaries)+1}"
        )
    n = sensitivity.shape[0]
    order = np.argsort(-sensitivity)  # descending
    rank_arr = np.empty(n, dtype=np.float64)
    rank_arr[order] = np.arange(n) / n if n > 0 else 0.0

    cls_arr = np.full(n, class_names[-1], dtype=object)  # default to tail
    prev = 0
    for boundary, name in zip(boundaries, class_names[:-1]):
        cur = int(n * boundary)
        cls_arr[order[prev:cur]] = name
        prev = cur
    return rank_arr, cls_arr


def refresh_per_byte_sensitivity(
    con,
    repo_root: Path,
    refreshed_at: str | None = None,
) -> int:
    """Backfill per_byte_sensitivity from master_gradient_anchors JSONL + .npy sidecars.

    Returns total row count inserted across all archives.
    """
    if np is None:  # pragma: no cover
        return 0
    bootstrap_per_byte_sensitivity_table(con)

    # Load master gradient anchors from the canonical JSONL source-of-truth
    # (NOT from the DuckDB master_gradient_anchors table because that table's
    # schema is sister's JSON-payload style; we read source directly for robustness)
    jsonl_path = repo_root / ".omx/state/master_gradient_anchors.jsonl"
    if not jsonl_path.exists():
        return 0

    anchors = []
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(r, dict):
            continue
        if not is_authoritative_axis_anchor(r):
            continue
        if r.get("gradient_tensor_kind", "aggregate_per_byte_v1") != "aggregate_per_byte_v1":
            continue
        sha = r.get("archive_sha256")
        npy_path_rel = r.get("gradient_array_path")
        n_bytes = r.get("n_bytes")
        if not (sha and npy_path_rel and n_bytes):
            continue
        anchors.append(
            (
                sha,
                npy_path_rel,
                int(n_bytes),
                str(r.get("measurement_axis") or ""),
                str(r.get("measurement_hardware") or ""),
                str(r.get("measurement_method") or ""),
                str(r.get("measurement_utc") or ""),
                bool(is_authoritative_contest_axis_anchor(r)),
            )
        )

    if not anchors:
        return 0

    # Refresh: drop + reinsert all per-archive rows for known anchors
    archive_shas = [a[0] for a in anchors]
    placeholders = ",".join("?" * len(archive_shas))
    con.execute(
        f"DELETE FROM per_byte_sensitivity WHERE archive_sha256 IN ({placeholders})",
        archive_shas,
    )

    total = 0
    for (
        archive_sha,
        npy_path_rel,
        declared_n,
        measurement_axis,
        measurement_hardware,
        measurement_method,
        measurement_utc,
        source_anchor_authoritative,
    ) in anchors:
        npy_path = repo_root / npy_path_rel
        if not npy_path.exists():
            continue
        try:
            arr = np.load(npy_path)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] != 3:
            continue
        n = arr.shape[0]
        sens = np.abs(arr).sum(axis=1).astype(np.float32)
        rank_arr, cls_arr = compute_sensitivity_classes(sens)

        # Build rows
        rows = [
            (
                archive_sha,
                int(i),
                float(arr[i, 0]),
                float(arr[i, 1]),
                float(arr[i, 2]),
                float(sens[i]),
                float(rank_arr[i]),
                str(cls_arr[i]),
                measurement_axis,
                measurement_hardware,
                measurement_method,
                measurement_utc,
                "diagnostic_from_contest_authoritative_source"
                if source_anchor_authoritative
                else "diagnostic",
                bool(source_anchor_authoritative),
                False,
            )
            for i in range(n)
        ]
        con.executemany(
            """
            INSERT INTO per_byte_sensitivity (
                archive_sha256,
                byte_idx,
                grad_seg,
                grad_pose,
                grad_rate,
                sensitivity_l1,
                sensitivity_quantile_rank,
                sensitivity_class,
                source_measurement_axis,
                source_measurement_hardware,
                source_measurement_method,
                source_measurement_utc,
                evidence_grade,
                source_anchor_authoritative,
                promotion_eligible
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        total += n

    return total


def refresh_per_byte_sensitivity_table(
    repo_root: Path | str,
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict:
    """Operator-facing entry: refresh per_byte_sensitivity under canonical lock."""
    root = Path(repo_root)
    db = Path(db_path)
    if not db.is_absolute():
        db = root / db
    refreshed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    with canonical_duckdb_lock(root):
        con = connect(db)
        try:
            row_count = refresh_per_byte_sensitivity(con, root, refreshed_at)
        finally:
            con.close()
    return {
        "table": "per_byte_sensitivity",
        "row_count": row_count,
        "refreshed_at_utc": refreshed_at,
        "db_path": str(db),
        "source_of_truth": "master_gradient_anchors.jsonl + .npy sidecars",
    }


__all__ = [
    "CLASS_NAMES",
    "PER_BYTE_SENSITIVITY_BOOTSTRAP_SQL",
    "QUANTILE_BOUNDARIES",
    "bootstrap_per_byte_sensitivity_table",
    "compute_sensitivity_classes",
    "refresh_per_byte_sensitivity",
    "refresh_per_byte_sensitivity_table",
]
