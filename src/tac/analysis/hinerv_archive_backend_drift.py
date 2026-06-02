# SPDX-License-Identifier: MIT
"""Compare HiNeRV archive replay bytes across local execution backends.

This is a local acceleration primitive, not score authority.  It answers one
small but important engineering question: can the MLX/Metal path reproduce the
receiver-closed archive-byte ladder closely enough to be used for fast local
iteration before contest CPU/CUDA replay?
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from tac.analysis.hinerv_archive_ladder_replay_actuator import (
    HINERV_ARCHIVE_LADDER_REPLAY_ACTUATOR_SCHEMA,
)
from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES
from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    CONTEST_BYTE_PRICE_SCORE,
    FALSE_AUTHORITY,
)

HINERV_ARCHIVE_BACKEND_DRIFT_SCHEMA = "hinerv_archive_backend_drift.v1"
HINERV_ARCHIVE_BACKEND_DRIFT_AUTHORITY = (
    "false_authority_local_backend_byte_drift_no_scorer_claim"
)
DEFAULT_MAX_ABS_BYTE_DELTA = 1024


class HinervArchiveBackendDriftError(ValueError):
    """Raised when backend drift inputs cannot be interpreted."""


def build_hinerv_archive_backend_drift_report(
    reference_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    *,
    reference_label: str = "reference",
    candidate_label: str = "candidate",
    max_abs_byte_delta: int = DEFAULT_MAX_ABS_BYTE_DELTA,
) -> dict[str, Any]:
    """Build a false-authority backend drift report for HiNeRV replay ladders."""

    if max_abs_byte_delta < 0:
        raise HinervArchiveBackendDriftError("max_abs_byte_delta must be non-negative")
    _require_replay_schema(reference_report, label=reference_label)
    _require_replay_schema(candidate_report, label=candidate_label)

    reference_rows = _rows_by_id(reference_report)
    candidate_rows = _rows_by_id(candidate_report)
    row_ids = sorted(set(reference_rows) | set(candidate_rows))
    rows = [
        _compare_row(
            row_id,
            reference_rows.get(row_id),
            candidate_rows.get(row_id),
            reference_label=reference_label,
            candidate_label=candidate_label,
            max_abs_byte_delta=max_abs_byte_delta,
        )
        for row_id in row_ids
    ]
    row_blockers = [
        blocker for row in rows for blocker in row.get("blockers", ()) if blocker
    ]
    matched = [row for row in rows if row.get("matched") is True]
    bytes_ready = [
        row for row in matched if row.get("byte_delta") is not None
    ]
    max_abs_observed = max(
        (int(row["abs_byte_delta"]) for row in bytes_ready),
        default=None,
    )
    sum_byte_delta = sum(int(row["byte_delta"]) for row in bytes_ready)
    candidate_receiver_ready = sum(
        1 for row in rows if row.get("candidate_receiver_proof_ready") is True
    )
    reference_receiver_ready = sum(
        1 for row in rows if row.get("reference_receiver_proof_ready") is True
    )
    within_tolerance = (
        bool(rows)
        and len(bytes_ready) == len(rows)
        and max_abs_observed is not None
        and max_abs_observed <= max_abs_byte_delta
    )
    local_velocity_ready = (
        within_tolerance
        and candidate_receiver_ready == len(rows)
        and reference_receiver_ready == len(rows)
    )
    blockers = [
        "contest_cpu_cuda_exact_eval_not_executed",
        "hinerv_archive_backend_drift_false_authority_no_nonrate_score",
        "hinerv_archive_backend_drift_not_promotion_or_rank_authority",
        *row_blockers,
    ]
    if not rows:
        blockers.append("hinerv_archive_backend_drift_rows_missing")
    if not within_tolerance and rows:
        blockers.append("hinerv_archive_backend_drift_exceeds_or_lacks_byte_tolerance")
    if local_velocity_ready:
        blockers.append("hinerv_archive_backend_drift_local_dev_velocity_only")
    else:
        blockers.append("hinerv_archive_backend_drift_local_dev_velocity_blocked")

    return {
        "schema": HINERV_ARCHIVE_BACKEND_DRIFT_SCHEMA,
        "authority": HINERV_ARCHIVE_BACKEND_DRIFT_AUTHORITY,
        "axis_tag": "[macOS-local:false-authority]",
        "family": "hi_nerv",
        "reference_label": str(reference_label),
        "candidate_label": str(candidate_label),
        "source_schema": HINERV_ARCHIVE_LADDER_REPLAY_ACTUATOR_SCHEMA,
        "reference_report_path": reference_report.get("report_path"),
        "candidate_report_path": candidate_report.get("report_path"),
        "reference_execution_requested": bool(
            reference_report.get("execution_requested")
        ),
        "candidate_execution_requested": bool(
            candidate_report.get("execution_requested")
        ),
        "reference_load_existing_requested": bool(
            reference_report.get("load_existing_requested")
        ),
        "candidate_load_existing_requested": bool(
            candidate_report.get("load_existing_requested")
        ),
        "reference_archive_bytes_by_row_id": dict(
            reference_report.get("archive_bytes_by_row_id") or {}
        ),
        "candidate_archive_bytes_by_row_id": dict(
            candidate_report.get("archive_bytes_by_row_id") or {}
        ),
        "contest_byte_price_score_per_byte": float(CONTEST_BYTE_PRICE_SCORE),
        "original_video_bytes": int(ORIGINAL_VIDEO_BYTES),
        "max_abs_byte_delta_allowed": int(max_abs_byte_delta),
        "row_count": len(rows),
        "matched_row_count": len(matched),
        "byte_ready_row_count": len(bytes_ready),
        "reference_receiver_proof_ready_row_count": reference_receiver_ready,
        "candidate_receiver_proof_ready_row_count": candidate_receiver_ready,
        "max_abs_byte_delta_observed": max_abs_observed,
        "sum_byte_delta_candidate_minus_reference": int(sum_byte_delta),
        "sum_rate_score_delta_candidate_minus_reference": float(
            sum_byte_delta * CONTEST_BYTE_PRICE_SCORE
        ),
        "max_abs_rate_score_delta_observed": (
            None
            if max_abs_observed is None
            else float(max_abs_observed * CONTEST_BYTE_PRICE_SCORE)
        ),
        "within_byte_drift_tolerance": bool(within_tolerance),
        "local_dev_velocity_ready": bool(local_velocity_ready),
        "ready_backend_for_local_iteration": (
            str(candidate_label) if local_velocity_ready else None
        ),
        "reference_archive_export_backend_counts": _backend_counts(
            reference_report,
            reference_rows,
        ),
        "candidate_archive_export_backend_counts": _backend_counts(
            candidate_report,
            candidate_rows,
        ),
        "status": (
            "local_backend_drift_within_tolerance_false_authority"
            if local_velocity_ready
            else "local_backend_drift_blocked_false_authority"
        ),
        "rows": rows,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def render_hinerv_archive_backend_drift_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact operator-facing backend drift summary."""

    lines = [
        "# HiNeRV archive backend drift",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        f"Reference: `{report.get('reference_label')}`",
        f"Candidate: `{report.get('candidate_label')}`",
        f"Local dev velocity ready: `{report.get('local_dev_velocity_ready')}`",
        f"Max abs byte drift: `{report.get('max_abs_byte_delta_observed')}`",
        f"Sum rate-score drift: `{report.get('sum_rate_score_delta_candidate_minus_reference')}`",
        "",
        "| row | ref bytes | cand bytes | delta | abs delta | rate-score delta | cand proof |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows") or ():
        lines.append(
            "| {row_id} | {ref_bytes} | {cand_bytes} | {delta} | {abs_delta} | {rate_delta} | {proof} |".format(
                row_id=row.get("row_id"),
                ref_bytes=_display_int(row.get("reference_archive_bytes")),
                cand_bytes=_display_int(row.get("candidate_archive_bytes")),
                delta=_display_int(row.get("byte_delta")),
                abs_delta=_display_int(row.get("abs_byte_delta")),
                rate_delta=row.get("rate_score_delta_candidate_minus_reference"),
                proof=row.get("candidate_receiver_proof_ready"),
            )
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in report.get("blockers") or ():
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def _require_replay_schema(report: Mapping[str, Any], *, label: str) -> None:
    if report.get("schema") != HINERV_ARCHIVE_LADDER_REPLAY_ACTUATOR_SCHEMA:
        raise HinervArchiveBackendDriftError(
            f"{label} must be {HINERV_ARCHIVE_LADDER_REPLAY_ACTUATOR_SCHEMA}"
        )


def _rows_by_id(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for raw in report.get("rows") or ():
        if not isinstance(raw, Mapping):
            continue
        row_id = raw.get("row_id")
        if row_id is None:
            continue
        rows[str(row_id)] = dict(raw)
    return rows


def _compare_row(
    row_id: str,
    reference_row: Mapping[str, Any] | None,
    candidate_row: Mapping[str, Any] | None,
    *,
    reference_label: str,
    candidate_label: str,
    max_abs_byte_delta: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    ref_bytes = _int_or_none(
        None if reference_row is None else reference_row.get("archive_bytes")
    )
    cand_bytes = _int_or_none(
        None if candidate_row is None else candidate_row.get("archive_bytes")
    )
    if reference_row is None:
        blockers.append("hinerv_archive_backend_drift_reference_row_missing")
    if candidate_row is None:
        blockers.append("hinerv_archive_backend_drift_candidate_row_missing")
    if reference_row is not None and ref_bytes is None:
        blockers.append("hinerv_archive_backend_drift_reference_archive_bytes_missing")
    if candidate_row is not None and cand_bytes is None:
        blockers.append("hinerv_archive_backend_drift_candidate_archive_bytes_missing")

    byte_delta = None if ref_bytes is None or cand_bytes is None else cand_bytes - ref_bytes
    abs_delta = None if byte_delta is None else abs(byte_delta)
    if abs_delta is not None and abs_delta > max_abs_byte_delta:
        blockers.append("hinerv_archive_backend_drift_row_exceeds_byte_tolerance")

    ref_proof = (
        False
        if reference_row is None
        else reference_row.get("receiver_proof_ready") is True
    )
    cand_proof = (
        False
        if candidate_row is None
        else candidate_row.get("receiver_proof_ready") is True
    )
    if reference_row is not None and not ref_proof:
        blockers.append("hinerv_archive_backend_drift_reference_receiver_proof_missing")
    if candidate_row is not None and not cand_proof:
        blockers.append("hinerv_archive_backend_drift_candidate_receiver_proof_missing")

    return {
        "row_id": row_id,
        "reference_label": str(reference_label),
        "candidate_label": str(candidate_label),
        "matched": reference_row is not None and candidate_row is not None,
        "reference_archive_bytes": ref_bytes,
        "candidate_archive_bytes": cand_bytes,
        "byte_delta": byte_delta,
        "abs_byte_delta": abs_delta,
        "rate_score_delta_candidate_minus_reference": (
            None
            if byte_delta is None
            else float(byte_delta * CONTEST_BYTE_PRICE_SCORE)
        ),
        "abs_rate_score_delta": (
            None if abs_delta is None else float(abs_delta * CONTEST_BYTE_PRICE_SCORE)
        ),
        "within_byte_drift_tolerance": (
            abs_delta is not None and abs_delta <= max_abs_byte_delta
        ),
        "reference_receiver_proof_ready": ref_proof,
        "candidate_receiver_proof_ready": cand_proof,
        "reference_archive_sha256": (
            None if reference_row is None else reference_row.get("archive_sha256")
        ),
        "candidate_archive_sha256": (
            None if candidate_row is None else candidate_row.get("archive_sha256")
        ),
        "reference_archive_path": (
            None if reference_row is None else reference_row.get("archive_path")
        ),
        "candidate_archive_path": (
            None if candidate_row is None else candidate_row.get("archive_path")
        ),
        "reference_status": (
            None if reference_row is None else reference_row.get("status")
        ),
        "candidate_status": (
            None if candidate_row is None else candidate_row.get("status")
        ),
        "reference_row_blockers": list(
            () if reference_row is None else reference_row.get("blockers") or ()
        ),
        "candidate_row_blockers": list(
            () if candidate_row is None else candidate_row.get("blockers") or ()
        ),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _merge_backend_counts(values: Iterable[Any]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for value in values:
        if not isinstance(value, Mapping):
            continue
        for key, raw_count in value.items():
            count = _int_or_none(raw_count) or 0
            merged[str(key)] = merged.get(str(key), 0) + count
    return merged


def _backend_counts(
    report: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    report_counts = report.get("archive_export_backend_counts")
    if isinstance(report_counts, Mapping) and report_counts:
        return _merge_backend_counts((report_counts,))
    return _merge_backend_counts(
        row.get("archive_export_backend_counts") for row in rows.values()
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _display_int(value: Any) -> str:
    parsed = _int_or_none(value)
    return "" if parsed is None else str(parsed)


def _ordered_unique(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


__all__ = [
    "DEFAULT_MAX_ABS_BYTE_DELTA",
    "HINERV_ARCHIVE_BACKEND_DRIFT_SCHEMA",
    "HinervArchiveBackendDriftError",
    "build_hinerv_archive_backend_drift_report",
    "render_hinerv_archive_backend_drift_markdown",
]
