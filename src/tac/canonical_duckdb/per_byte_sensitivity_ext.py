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
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
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
    PER_PAIR_GRADIENT_TENSOR_KIND,
    SCORE_AXIS_LABELS,
    OperatingPoint,
    compute_marginal_coefficients,
    is_authoritative_axis_anchor,
    is_authoritative_contest_axis_anchor,
    load_anchors_lenient,
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

MULTI_GRANULARITY_SENSITIVITY_BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS multi_granularity_sensitivity_runs (
    run_id TEXT PRIMARY KEY,
    archive_sha256 TEXT NOT NULL,
    source_anchor_utc TIMESTAMP,
    gradient_tensor_kind TEXT NOT NULL,
    gradient_array_path TEXT NOT NULL,
    gradient_array_sha256 TEXT,
    class_source_path TEXT,
    class_source_sha256 TEXT,
    class_basis TEXT NOT NULL,
    source_measurement_axis TEXT NOT NULL,
    evidence_grade TEXT NOT NULL,
    source_anchor_authoritative BOOLEAN NOT NULL,
    score_claim BOOLEAN NOT NULL DEFAULT FALSE,
    promotion_eligible BOOLEAN NOT NULL DEFAULT FALSE,
    ready_for_exact_eval_dispatch BOOLEAN NOT NULL DEFAULT FALSE,
    row_count BIGINT NOT NULL,
    blocker_reason TEXT
);

CREATE TABLE IF NOT EXISTS multi_granularity_sensitivity (
    run_id TEXT NOT NULL,
    archive_sha256 TEXT NOT NULL,
    pair_id INTEGER NOT NULL,
    byte_offset INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    axis TEXT NOT NULL,
    raw_gradient_fp64 DOUBLE NOT NULL,
    marginal_coefficient_fp64 DOUBLE NOT NULL,
    class_weight_fp64 DOUBLE NOT NULL,
    signed_score_delta_unit_fp64 DOUBLE NOT NULL,
    sensitivity_fp64 DOUBLE NOT NULL,
    derived_at_utc TIMESTAMP NOT NULL,
    PRIMARY KEY (run_id, pair_id, byte_offset, class_id, axis)
);

CREATE INDEX IF NOT EXISTS idx_mgs_archive_axis
    ON multi_granularity_sensitivity (archive_sha256, axis);
CREATE INDEX IF NOT EXISTS idx_mgs_archive_pair
    ON multi_granularity_sensitivity (archive_sha256, pair_id);
CREATE INDEX IF NOT EXISTS idx_mgs_archive_class
    ON multi_granularity_sensitivity (archive_sha256, class_id);
CREATE INDEX IF NOT EXISTS idx_mgs_run ON multi_granularity_sensitivity (run_id);

CREATE OR REPLACE VIEW multi_granularity_sensitivity_with_runs AS
SELECT
    s.*,
    r.gradient_tensor_kind,
    r.gradient_array_path,
    r.gradient_array_sha256,
    r.class_source_path,
    r.class_source_sha256,
    r.class_basis,
    r.source_measurement_axis,
    r.evidence_grade,
    r.source_anchor_authoritative,
    r.score_claim,
    r.promotion_eligible,
    r.ready_for_exact_eval_dispatch,
    r.blocker_reason
FROM multi_granularity_sensitivity s
JOIN multi_granularity_sensitivity_runs r USING (run_id);
"""

QUANTILE_BOUNDARIES: tuple[float, ...] = (0.02, 0.05, 0.20)
"""Cumulative-from-top quantile boundaries matching sensitivity_mask_aware_quantizr_v1.

(top 2% / top 5% [cumulative] / top 20% [cumulative]; tail = remaining 80%)
"""

CLASS_NAMES: tuple[str, ...] = ("top_2pct", "top_5pct", "top_20pct", "tail")
"""Canonical class names for sensitivity classification."""

MULTI_GRANULARITY_AXES: tuple[str, ...] = SCORE_AXIS_LABELS
"""Canonical objective axes for pair x byte x class sensitivity cells."""

SEGNET_CLASS_IDS: tuple[int, ...] = (0, 1, 2, 3, 4)
"""SegNet class ids in the contest scorer."""


@dataclass(frozen=True)
class MultiGranularitySensitivityRun:
    """Provenance row for one multi-granularity tensor materialization."""

    run_id: str
    archive_sha256: str
    gradient_tensor_kind: str
    gradient_array_path: str
    class_basis: str
    source_measurement_axis: str
    evidence_grade: str
    source_anchor_authoritative: bool
    row_count: int
    source_anchor_utc: datetime | str | None = None
    gradient_array_sha256: str | None = None
    class_source_path: str | None = None
    class_source_sha256: str | None = None
    blocker_reason: str | None = None
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def as_insert_tuple(self) -> tuple:
        """Return a DuckDB-ready insert tuple after authority validation."""
        run = validate_multi_granularity_sensitivity_run(self)
        return (
            run.run_id,
            run.archive_sha256,
            run.source_anchor_utc,
            run.gradient_tensor_kind,
            run.gradient_array_path,
            run.gradient_array_sha256,
            run.class_source_path,
            run.class_source_sha256,
            run.class_basis,
            run.source_measurement_axis,
            run.evidence_grade,
            run.source_anchor_authoritative,
            run.score_claim,
            run.promotion_eligible,
            run.ready_for_exact_eval_dispatch,
            run.row_count,
            run.blocker_reason,
        )


@dataclass(frozen=True)
class MultiGranularitySensitivityRow:
    """Typed row for the pair x byte x class x axis sensitivity tensor."""

    run_id: str
    archive_sha256: str
    pair_id: int
    byte_offset: int
    class_id: int
    axis: str
    raw_gradient_fp64: float
    marginal_coefficient_fp64: float
    class_weight_fp64: float
    signed_score_delta_unit_fp64: float | None = None
    sensitivity_fp64: float | None = None
    derived_at_utc: datetime | str | None = None

    def as_insert_tuple(self, default_derived_at_utc: datetime | str) -> tuple:
        """Return a DuckDB-ready insert tuple after fail-closed validation."""
        row = validate_multi_granularity_sensitivity_row(
            {
                "run_id": self.run_id,
                "archive_sha256": self.archive_sha256,
                "pair_id": self.pair_id,
                "byte_offset": self.byte_offset,
                "class_id": self.class_id,
                "axis": self.axis,
                "raw_gradient_fp64": self.raw_gradient_fp64,
                "marginal_coefficient_fp64": self.marginal_coefficient_fp64,
                "class_weight_fp64": self.class_weight_fp64,
                "signed_score_delta_unit_fp64": self.signed_score_delta_unit_fp64,
                "sensitivity_fp64": self.sensitivity_fp64,
                "derived_at_utc": self.derived_at_utc or default_derived_at_utc,
            }
        )
        return (
            row.run_id,
            row.archive_sha256,
            row.pair_id,
            row.byte_offset,
            row.class_id,
            row.axis,
            row.raw_gradient_fp64,
            row.marginal_coefficient_fp64,
            row.class_weight_fp64,
            row.signed_score_delta_unit_fp64,
            row.sensitivity_fp64,
            row.derived_at_utc,
        )


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


def bootstrap_multi_granularity_sensitivity_table(con) -> None:
    """Create the multi-granularity sensitivity tensor table (idempotent)."""
    con.execute(MULTI_GRANULARITY_SENSITIVITY_BOOTSTRAP_SQL)


def bootstrap_sensitivity_extension_tables(con) -> None:
    """Create all sensitivity extension tables owned by this module."""
    bootstrap_per_byte_sensitivity_table(con)
    bootstrap_multi_granularity_sensitivity_table(con)


def _parse_nonnegative_int(value: object, field_name: str) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return parsed


def _normalize_derived_at(value: object) -> datetime | str:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("derived_at_utc must be a non-empty timestamp")


def _parse_finite_float(value: object, field_name: str) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _normalize_optional_timestamp(value: object) -> datetime | str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("source_anchor_utc must be null or a non-empty timestamp")


def validate_multi_granularity_sensitivity_run(
    run: MultiGranularitySensitivityRun | dict,
) -> MultiGranularitySensitivityRun:
    """Validate one run/provenance row and fail closed on score authority."""
    if isinstance(run, MultiGranularitySensitivityRun):
        data = {
            "run_id": run.run_id,
            "archive_sha256": run.archive_sha256,
            "source_anchor_utc": run.source_anchor_utc,
            "gradient_tensor_kind": run.gradient_tensor_kind,
            "gradient_array_path": run.gradient_array_path,
            "gradient_array_sha256": run.gradient_array_sha256,
            "class_source_path": run.class_source_path,
            "class_source_sha256": run.class_source_sha256,
            "class_basis": run.class_basis,
            "source_measurement_axis": run.source_measurement_axis,
            "evidence_grade": run.evidence_grade,
            "source_anchor_authoritative": run.source_anchor_authoritative,
            "score_claim": run.score_claim,
            "promotion_eligible": run.promotion_eligible,
            "ready_for_exact_eval_dispatch": run.ready_for_exact_eval_dispatch,
            "row_count": run.row_count,
            "blocker_reason": run.blocker_reason,
        }
    else:
        data = run

    run_id = str(data.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("run_id must be non-empty")
    archive_sha = str(data.get("archive_sha256") or "").strip()
    if not archive_sha:
        raise ValueError("archive_sha256 must be non-empty")
    gradient_tensor_kind = str(data.get("gradient_tensor_kind") or "").strip()
    if gradient_tensor_kind != PER_PAIR_GRADIENT_TENSOR_KIND:
        raise ValueError(
            f"gradient_tensor_kind must be {PER_PAIR_GRADIENT_TENSOR_KIND!r}"
        )
    gradient_array_path = str(data.get("gradient_array_path") or "").strip()
    if not gradient_array_path:
        raise ValueError("gradient_array_path must be non-empty")
    class_basis = str(data.get("class_basis") or "").strip()
    if not class_basis:
        raise ValueError("class_basis must be non-empty")
    measurement_axis = str(data.get("source_measurement_axis") or "").strip()
    if not measurement_axis:
        raise ValueError("source_measurement_axis must be non-empty")
    evidence_grade = str(data.get("evidence_grade") or "").strip()
    if not evidence_grade:
        raise ValueError("evidence_grade must be non-empty")
    row_count = _parse_nonnegative_int(data.get("row_count"), "row_count")
    blocker_reason = (
        str(data["blocker_reason"]).strip()
        if data.get("blocker_reason")
        else None
    )
    source_anchor_authoritative = bool(data.get("source_anchor_authoritative"))
    if source_anchor_authoritative:
        if measurement_axis not in ("[contest-CPU]", "[contest-CUDA]"):
            raise ValueError(
                "source_anchor_authoritative requires a contest measurement axis"
            )
        if evidence_grade != "diagnostic_from_contest_authoritative_source":
            raise ValueError(
                "source_anchor_authoritative requires authoritative evidence_grade"
            )
        if blocker_reason is not None:
            raise ValueError("source_anchor_authoritative runs cannot carry blockers")
    for field_name in (
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        if bool(data.get(field_name, False)):
            raise ValueError(f"{field_name} must be false for planning-only tensor rows")

    return MultiGranularitySensitivityRun(
        run_id=run_id,
        archive_sha256=archive_sha,
        source_anchor_utc=_normalize_optional_timestamp(data.get("source_anchor_utc")),
        gradient_tensor_kind=gradient_tensor_kind,
        gradient_array_path=gradient_array_path,
        gradient_array_sha256=(
            str(data["gradient_array_sha256"]).strip()
            if data.get("gradient_array_sha256")
            else None
        ),
        class_source_path=(
            str(data["class_source_path"]).strip()
            if data.get("class_source_path")
            else None
        ),
        class_source_sha256=(
            str(data["class_source_sha256"]).strip()
            if data.get("class_source_sha256")
            else None
        ),
        class_basis=class_basis,
        source_measurement_axis=measurement_axis,
        evidence_grade=evidence_grade,
        source_anchor_authoritative=source_anchor_authoritative,
        score_claim=False,
        promotion_eligible=False,
        ready_for_exact_eval_dispatch=False,
        row_count=row_count,
        blocker_reason=blocker_reason,
    )


def validate_multi_granularity_sensitivity_row(
    row: MultiGranularitySensitivityRow | dict,
) -> MultiGranularitySensitivityRow:
    """Validate one multi-granularity tensor row without inventing score authority."""
    if isinstance(row, MultiGranularitySensitivityRow):
        data = {
            "run_id": row.run_id,
            "archive_sha256": row.archive_sha256,
            "pair_id": row.pair_id,
            "byte_offset": row.byte_offset,
            "class_id": row.class_id,
            "axis": row.axis,
            "raw_gradient_fp64": row.raw_gradient_fp64,
            "marginal_coefficient_fp64": row.marginal_coefficient_fp64,
            "class_weight_fp64": row.class_weight_fp64,
            "signed_score_delta_unit_fp64": row.signed_score_delta_unit_fp64,
            "sensitivity_fp64": row.sensitivity_fp64,
            "derived_at_utc": row.derived_at_utc,
        }
    else:
        data = row

    run_id = str(data.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("run_id must be non-empty")
    archive_sha = str(data.get("archive_sha256") or "").strip()
    if not archive_sha:
        raise ValueError("archive_sha256 must be non-empty")
    pair_id = _parse_nonnegative_int(data.get("pair_id"), "pair_id")
    byte_offset = _parse_nonnegative_int(data.get("byte_offset"), "byte_offset")
    class_id = _parse_nonnegative_int(data.get("class_id"), "class_id")
    if class_id not in SEGNET_CLASS_IDS:
        raise ValueError(f"class_id must be one of {SEGNET_CLASS_IDS}")
    axis = str(data.get("axis") or "").strip()
    if axis not in MULTI_GRANULARITY_AXES:
        raise ValueError(f"axis must be one of {MULTI_GRANULARITY_AXES}")
    raw_gradient = _parse_finite_float(data.get("raw_gradient_fp64"), "raw_gradient_fp64")
    marginal = _parse_finite_float(
        data.get("marginal_coefficient_fp64"), "marginal_coefficient_fp64"
    )
    class_weight = _parse_finite_float(data.get("class_weight_fp64"), "class_weight_fp64")
    signed = data.get("signed_score_delta_unit_fp64")
    if signed is None:
        signed_score_delta_unit = raw_gradient * marginal * class_weight
    else:
        signed_score_delta_unit = _parse_finite_float(
            signed,
            "signed_score_delta_unit_fp64",
        )
    sensitivity = data.get("sensitivity_fp64")
    if sensitivity is None:
        sensitivity_fp64 = abs(signed_score_delta_unit)
    else:
        sensitivity_fp64 = _parse_finite_float(sensitivity, "sensitivity_fp64")

    return MultiGranularitySensitivityRow(
        run_id=run_id,
        archive_sha256=archive_sha,
        pair_id=pair_id,
        byte_offset=byte_offset,
        class_id=class_id,
        axis=axis,
        raw_gradient_fp64=raw_gradient,
        marginal_coefficient_fp64=marginal,
        class_weight_fp64=class_weight,
        signed_score_delta_unit_fp64=signed_score_delta_unit,
        sensitivity_fp64=sensitivity_fp64,
        derived_at_utc=_normalize_derived_at(data.get("derived_at_utc")),
    )


def upsert_multi_granularity_sensitivity_run(
    con,
    run: MultiGranularitySensitivityRun | dict,
    rows: list[MultiGranularitySensitivityRow | dict],
    *,
    derived_at_utc: datetime | str | None = None,
) -> int:
    """Insert one provenance run and its latest planning-only tensor cells.

    Rows intentionally cannot carry score, promotion, or dispatch authority.
    The table is a derived read-model for routing and allocator consumers.
    """
    bootstrap_multi_granularity_sensitivity_table(con)
    validated_run = validate_multi_granularity_sensitivity_run(run)
    if validated_run.row_count != len(rows):
        raise ValueError(
            f"run.row_count={validated_run.row_count} must equal len(rows)={len(rows)}"
        )
    default_derived_at = derived_at_utc or datetime.now(UTC).replace(microsecond=0)
    insert_rows = []
    for row in rows:
        if isinstance(row, MultiGranularitySensitivityRow):
            validated_row = validate_multi_granularity_sensitivity_row(row)
        else:
            row = {**row}
            row.setdefault("run_id", validated_run.run_id)
            row.setdefault("archive_sha256", validated_run.archive_sha256)
            row.setdefault("derived_at_utc", default_derived_at)
            validated_row = validate_multi_granularity_sensitivity_row(row)
        if validated_row.run_id != validated_run.run_id:
            raise ValueError("row run_id must match run.run_id")
        if validated_row.archive_sha256 != validated_run.archive_sha256:
            raise ValueError("row archive_sha256 must match run.archive_sha256")
        insert_rows.append(validated_row.as_insert_tuple(default_derived_at))
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "DELETE FROM multi_granularity_sensitivity WHERE run_id = ?",
            [validated_run.run_id],
        )
        con.execute(
            """
            INSERT OR REPLACE INTO multi_granularity_sensitivity_runs (
                run_id,
                archive_sha256,
                source_anchor_utc,
                gradient_tensor_kind,
                gradient_array_path,
                gradient_array_sha256,
                class_source_path,
                class_source_sha256,
                class_basis,
                source_measurement_axis,
                evidence_grade,
                source_anchor_authoritative,
                score_claim,
                promotion_eligible,
                ready_for_exact_eval_dispatch,
                row_count,
                blocker_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            validated_run.as_insert_tuple(),
        )
        if insert_rows:
            con.executemany(
                """
                INSERT OR REPLACE INTO multi_granularity_sensitivity (
                    run_id,
                    archive_sha256,
                    pair_id,
                    byte_offset,
                    class_id,
                    axis,
                    raw_gradient_fp64,
                    marginal_coefficient_fp64,
                    class_weight_fp64,
                    signed_score_delta_unit_fp64,
                    sensitivity_fp64,
                    derived_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_rows,
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(insert_rows)


def upsert_multi_granularity_sensitivity_rows(
    con,
    rows: list[MultiGranularitySensitivityRow | dict],
    *,
    run: MultiGranularitySensitivityRun | dict,
    derived_at_utc: datetime | str | None = None,
) -> int:
    """Compatibility wrapper requiring explicit run/provenance metadata."""
    return upsert_multi_granularity_sensitivity_run(
        con,
        run,
        rows,
        derived_at_utc=derived_at_utc,
    )


def query_multi_granularity_sensitivity_top_cells(
    con,
    archive_sha256: str,
    *,
    axis: str | None = None,
    pair_id: int | None = None,
    class_id: int | None = None,
    include_all_runs: bool = False,
    limit: int = 100,
) -> list[dict]:
    """Return strongest absolute-sensitivity cells with adjacent provenance."""
    bootstrap_multi_granularity_sensitivity_table(con)
    archive_sha = str(archive_sha256 or "").strip()
    if not archive_sha:
        raise ValueError("archive_sha256 must be non-empty")
    if limit <= 0:
        raise ValueError("limit must be positive")
    clauses = ["archive_sha256 = ?"]
    params: list[object] = [archive_sha]
    if not include_all_runs:
        clauses.append(
            """
            run_id = (
                SELECT run_id
                FROM multi_granularity_sensitivity_runs
                WHERE archive_sha256 = ?
                  AND blocker_reason IS NULL
                ORDER BY source_anchor_utc DESC NULLS LAST, run_id DESC
                LIMIT 1
            )
            """
        )
        params.append(archive_sha)
    if axis is not None:
        axis = axis.strip()
        if axis not in MULTI_GRANULARITY_AXES:
            raise ValueError(f"axis must be one of {MULTI_GRANULARITY_AXES}")
        clauses.append("axis = ?")
        params.append(axis)
    if pair_id is not None:
        clauses.append("pair_id = ?")
        params.append(_parse_nonnegative_int(pair_id, "pair_id"))
    if class_id is not None:
        cid = _parse_nonnegative_int(class_id, "class_id")
        if cid not in SEGNET_CLASS_IDS:
            raise ValueError(f"class_id must be one of {SEGNET_CLASS_IDS}")
        clauses.append("class_id = ?")
        params.append(cid)
    params.append(int(limit))
    rows = con.execute(
        f"""
        SELECT
            run_id,
            archive_sha256,
            pair_id,
            byte_offset,
            class_id,
            axis,
            raw_gradient_fp64,
            marginal_coefficient_fp64,
            class_weight_fp64,
            signed_score_delta_unit_fp64,
            sensitivity_fp64,
            derived_at_utc,
            class_basis,
            source_measurement_axis,
            evidence_grade,
            source_anchor_authoritative,
            score_claim,
            promotion_eligible,
            ready_for_exact_eval_dispatch,
            blocker_reason
        FROM multi_granularity_sensitivity_with_runs
        WHERE {" AND ".join(clauses)}
        ORDER BY ABS(sensitivity_fp64) DESC,
                 pair_id ASC,
                 byte_offset ASC,
                 class_id ASC,
                 axis ASC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        {
            "run_id": r[0],
            "archive_sha256": r[1],
            "pair_id": int(r[2]),
            "byte_offset": int(r[3]),
            "class_id": int(r[4]),
            "axis": r[5],
            "raw_gradient_fp64": float(r[6]),
            "marginal_coefficient_fp64": float(r[7]),
            "class_weight_fp64": float(r[8]),
            "signed_score_delta_unit_fp64": float(r[9]),
            "sensitivity_fp64": float(r[10]),
            "derived_at_utc": r[11],
            "class_basis": r[12],
            "source_measurement_axis": r[13],
            "evidence_grade": r[14],
            "source_anchor_authoritative": bool(r[15]),
            "score_claim": bool(r[16]),
            "promotion_eligible": bool(r[17]),
            "ready_for_exact_eval_dispatch": bool(r[18]),
            "blocker_reason": r[19],
        }
        for r in rows
    ]


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _latest_per_pair_anchor(
    repo_root: Path,
    archive_sha256: str | None,
) -> dict | None:
    ledger = repo_root / ".omx/state/master_gradient_anchors.jsonl"
    rows = load_anchors_lenient(ledger)
    candidates = [
        row
        for row in rows
        if row.get("gradient_tensor_kind") == PER_PAIR_GRADIENT_TENSOR_KIND
        and is_authoritative_axis_anchor(row)
    ]
    if archive_sha256 is not None:
        candidates = [
            row for row in candidates if row.get("archive_sha256") == archive_sha256
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row.get("measurement_utc", ""))


def _run_id_for_anchor(
    anchor: dict,
    *,
    class_source_path: Path | None,
    byte_offsets: tuple[int, ...],
) -> str:
    seed = json.dumps(
        {
            "archive_sha256": anchor.get("archive_sha256"),
            "gradient_array_path": anchor.get("gradient_array_path"),
            "measurement_utc": anchor.get("measurement_utc"),
            "class_source_path": str(class_source_path) if class_source_path else None,
            "byte_offsets": list(byte_offsets),
        },
        sort_keys=True,
    )
    return f"mgs_{str(anchor.get('archive_sha256') or '')[:12]}_{sha256(seed.encode()).hexdigest()[:16]}"


def _load_class_weights(class_source_path: Path) -> tuple[str, dict[int, float]]:
    payload = json.loads(class_source_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("class source must be a JSON object")
    basis = str(payload.get("class_basis") or payload.get("basis") or "").strip()
    if not basis:
        raise ValueError("class source must include class_basis")
    weights_obj = payload.get("class_weights", payload.get("weights"))
    if isinstance(weights_obj, list):
        if len(weights_obj) != len(SEGNET_CLASS_IDS):
            raise ValueError("class_weights list must have 5 SegNet entries")
        weights = {idx: _parse_finite_float(value, f"class_weights[{idx}]")
                   for idx, value in enumerate(weights_obj)}
    elif isinstance(weights_obj, dict):
        weights = {
            int(key): _parse_finite_float(value, f"class_weights[{key}]")
            for key, value in weights_obj.items()
        }
    else:
        raise ValueError("class source must include class_weights")
    missing = set(SEGNET_CLASS_IDS) - set(weights)
    extra = set(weights) - set(SEGNET_CLASS_IDS)
    if missing or extra:
        raise ValueError(
            f"class_weights must cover exactly {SEGNET_CLASS_IDS}; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    return basis, weights


def _operating_point_from_anchor(anchor: dict) -> OperatingPoint:
    op = anchor.get("operating_point")
    if not isinstance(op, dict):
        raise ValueError("per-pair anchor missing operating_point")
    return OperatingPoint(
        d_seg=float(op["d_seg"]),
        d_pose=float(op["d_pose"]),
        rate=float(op.get("rate", op.get("R", 0.0))),
        score=float(op["score"]),
    )


def _record_multi_granularity_blocker_run(
    con,
    anchor: dict,
    *,
    blocker_reason: str,
    run_id: str,
    class_source_path: Path | None,
    class_source_sha: str | None = None,
) -> None:
    run = MultiGranularitySensitivityRun(
        run_id=run_id,
        archive_sha256=str(anchor.get("archive_sha256") or ""),
        source_anchor_utc=anchor.get("measurement_utc") or None,
        gradient_tensor_kind=str(anchor.get("gradient_tensor_kind") or ""),
        gradient_array_path=str(anchor.get("gradient_array_path") or ""),
        gradient_array_sha256=None,
        class_source_path=str(class_source_path) if class_source_path else None,
        class_source_sha256=class_source_sha,
        class_basis="blocked_no_class_basis",
        source_measurement_axis=str(anchor.get("measurement_axis") or ""),
        evidence_grade="blocked_planning_only",
        source_anchor_authoritative=bool(is_authoritative_contest_axis_anchor(anchor)),
        row_count=0,
        blocker_reason=blocker_reason,
    )
    upsert_multi_granularity_sensitivity_run(con, run, [])


def _select_byte_offsets(
    arr: np.ndarray,
    *,
    byte_offsets: list[int] | tuple[int, ...] | None,
    top_k_bytes: int | None,
) -> tuple[int, ...]:
    if byte_offsets is not None:
        offsets = tuple(_parse_nonnegative_int(v, "byte_offset") for v in byte_offsets)
    elif top_k_bytes is not None:
        if top_k_bytes <= 0:
            raise ValueError("top_k_bytes must be positive")
        sensitivity_by_byte = np.abs(arr).sum(axis=(1, 2))
        offsets = tuple(int(v) for v in np.argsort(-sensitivity_by_byte)[:top_k_bytes])
    else:
        raise ValueError("byte_offsets or top_k_bytes is required to bound refresh")
    n_bytes = int(arr.shape[0])
    if any(offset >= n_bytes for offset in offsets):
        raise ValueError(f"byte_offset must be < n_bytes={n_bytes}")
    return tuple(dict.fromkeys(offsets))


def refresh_multi_granularity_sensitivity(
    con,
    repo_root: Path,
    *,
    archive_sha256: str | None = None,
    class_source_path: Path | str | None = None,
    byte_offsets: list[int] | tuple[int, ...] | None = None,
    top_k_bytes: int | None = None,
    max_rows: int = 1_000_000,
    refreshed_at: str | None = None,
) -> int:
    """Materialize bounded pair x byte x class x axis sensitivity cells.

    Only ``per_pair_per_byte_v1`` anchors are accepted. Class-specific rows
    require an explicit class-source JSON; without one, the function records a
    blocker run and emits zero rows rather than duplicating invented structure.
    """
    if np is None:  # pragma: no cover
        return 0
    bootstrap_multi_granularity_sensitivity_table(con)
    anchor = _latest_per_pair_anchor(repo_root, archive_sha256)
    if anchor is None:
        return 0

    placeholder_offsets: tuple[int, ...] = tuple(
        _parse_nonnegative_int(v, "byte_offset") for v in (byte_offsets or ())
    )
    class_path = (
        _resolve_repo_path(repo_root, str(class_source_path))
        if class_source_path is not None
        else None
    )
    run_id = _run_id_for_anchor(
        anchor,
        class_source_path=class_path,
        byte_offsets=placeholder_offsets,
    )
    if class_path is None:
        _record_multi_granularity_blocker_run(
            con,
            anchor,
            blocker_reason="class_source_required_for_class_specific_sensitivity",
            run_id=run_id,
            class_source_path=None,
        )
        return 0
    if not class_path.exists():
        _record_multi_granularity_blocker_run(
            con,
            anchor,
            blocker_reason="class_source_path_missing",
            run_id=run_id,
            class_source_path=class_path,
        )
        return 0

    class_source_sha = _file_sha256(class_path)
    class_basis, class_weights = _load_class_weights(class_path)
    gradient_path = _resolve_repo_path(repo_root, str(anchor.get("gradient_array_path") or ""))
    if not gradient_path.exists():
        _record_multi_granularity_blocker_run(
            con,
            anchor,
            blocker_reason="gradient_array_path_missing",
            run_id=run_id,
            class_source_path=class_path,
            class_source_sha=class_source_sha,
        )
        return 0
    arr = np.load(gradient_path)
    declared_n_bytes = int(anchor.get("n_bytes") or -1)
    declared_n_pairs = int(anchor.get("n_pairs") or -1)
    expected = (declared_n_bytes, declared_n_pairs, len(MULTI_GRANULARITY_AXES))
    if arr.shape != expected:
        raise ValueError(f"per-pair gradient shape {arr.shape} != declared {expected}")
    selected_offsets = _select_byte_offsets(
        arr,
        byte_offsets=byte_offsets,
        top_k_bytes=top_k_bytes,
    )
    projected_rows = (
        len(selected_offsets)
        * declared_n_pairs
        * len(SEGNET_CLASS_IDS)
        * len(MULTI_GRANULARITY_AXES)
    )
    if projected_rows > max_rows:
        raise ValueError(
            f"multi_granularity_sensitivity refresh would emit {projected_rows} rows; "
            f"max_rows={max_rows}"
        )
    op = _operating_point_from_anchor(anchor)
    coeffs = compute_marginal_coefficients(op)
    derived_at = refreshed_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    rows: list[MultiGranularitySensitivityRow] = []
    for byte_offset in selected_offsets:
        for pair_id in range(declared_n_pairs):
            for class_id in SEGNET_CLASS_IDS:
                class_weight = class_weights[class_id]
                for axis_idx, axis in enumerate(MULTI_GRANULARITY_AXES):
                    raw = float(arr[byte_offset, pair_id, axis_idx])
                    signed = raw * float(coeffs[axis_idx]) * class_weight
                    rows.append(
                        MultiGranularitySensitivityRow(
                            run_id=run_id,
                            archive_sha256=str(anchor["archive_sha256"]),
                            pair_id=pair_id,
                            byte_offset=byte_offset,
                            class_id=class_id,
                            axis=axis,
                            raw_gradient_fp64=raw,
                            marginal_coefficient_fp64=float(coeffs[axis_idx]),
                            class_weight_fp64=class_weight,
                            signed_score_delta_unit_fp64=signed,
                            sensitivity_fp64=abs(signed),
                            derived_at_utc=derived_at,
                        )
                    )
    run = MultiGranularitySensitivityRun(
        run_id=run_id,
        archive_sha256=str(anchor["archive_sha256"]),
        source_anchor_utc=anchor.get("measurement_utc") or None,
        gradient_tensor_kind=PER_PAIR_GRADIENT_TENSOR_KIND,
        gradient_array_path=str(anchor.get("gradient_array_path") or ""),
        gradient_array_sha256=_file_sha256(gradient_path),
        class_source_path=str(class_path),
        class_source_sha256=class_source_sha,
        class_basis=class_basis,
        source_measurement_axis=str(anchor.get("measurement_axis") or ""),
        evidence_grade="diagnostic_from_contest_authoritative_source"
        if is_authoritative_contest_axis_anchor(anchor)
        else "diagnostic",
        source_anchor_authoritative=bool(is_authoritative_contest_axis_anchor(anchor)),
        row_count=len(rows),
    )
    return upsert_multi_granularity_sensitivity_run(
        con,
        run,
        rows,
        derived_at_utc=derived_at,
    )


def compute_sensitivity_classes(
    sensitivity: np.ndarray,
    *,
    boundaries: tuple[float, ...] = QUANTILE_BOUNDARIES,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
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
    for boundary, name in zip(boundaries, class_names[:-1], strict=True):
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
        _declared_n,
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


def refresh_multi_granularity_sensitivity_table(
    repo_root: Path | str,
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    archive_sha256: str | None = None,
    class_source_path: Path | str | None = None,
    byte_offsets: list[int] | tuple[int, ...] | None = None,
    top_k_bytes: int | None = None,
    max_rows: int = 1_000_000,
) -> dict:
    """Operator-facing bounded refresh for the multi-granularity tensor.

    Without a class source and byte bound this bootstraps/counts only, avoiding
    accidental dense archive x pair x class materialization.
    """
    root = Path(repo_root)
    db = Path(db_path)
    if not db.is_absolute():
        db = root / db
    refreshed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    with canonical_duckdb_lock(root):
        con = connect(db)
        try:
            if class_source_path is not None or byte_offsets is not None or top_k_bytes is not None:
                row_count = refresh_multi_granularity_sensitivity(
                    con,
                    root,
                    archive_sha256=archive_sha256,
                    class_source_path=class_source_path,
                    byte_offsets=byte_offsets,
                    top_k_bytes=top_k_bytes,
                    max_rows=max_rows,
                    refreshed_at=refreshed_at,
                )
            else:
                bootstrap_multi_granularity_sensitivity_table(con)
                row_count = con.execute(
                    "SELECT COUNT(*) FROM multi_granularity_sensitivity"
                ).fetchone()[0]
            row_count = con.execute(
                "SELECT COUNT(*) FROM multi_granularity_sensitivity"
            ).fetchone()[0]
        finally:
            con.close()
    return {
        "table": "multi_granularity_sensitivity",
        "row_count": int(row_count),
        "refreshed_at_utc": refreshed_at,
        "db_path": str(db),
        "source_of_truth": "per_pair master_gradient anchor + explicit class source sidecar",
        "bounded_refresh": bool(
            class_source_path is not None or byte_offsets is not None or top_k_bytes is not None
        ),
    }


__all__ = [
    "CLASS_NAMES",
    "MULTI_GRANULARITY_AXES",
    "MULTI_GRANULARITY_SENSITIVITY_BOOTSTRAP_SQL",
    "PER_BYTE_SENSITIVITY_BOOTSTRAP_SQL",
    "QUANTILE_BOUNDARIES",
    "SEGNET_CLASS_IDS",
    "MultiGranularitySensitivityRow",
    "MultiGranularitySensitivityRun",
    "bootstrap_multi_granularity_sensitivity_table",
    "bootstrap_per_byte_sensitivity_table",
    "bootstrap_sensitivity_extension_tables",
    "compute_sensitivity_classes",
    "query_multi_granularity_sensitivity_top_cells",
    "refresh_multi_granularity_sensitivity",
    "refresh_multi_granularity_sensitivity_table",
    "refresh_per_byte_sensitivity",
    "refresh_per_byte_sensitivity_table",
    "upsert_multi_granularity_sensitivity_rows",
    "upsert_multi_granularity_sensitivity_run",
    "validate_multi_granularity_sensitivity_row",
    "validate_multi_granularity_sensitivity_run",
]
