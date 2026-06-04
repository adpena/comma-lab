# SPDX-License-Identifier: MIT
"""Adapt SNeRV checkpoint-export LF accounting into the LF codec control route."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

SCHEMA = "snerv_lf_payload_codec_sweep.v1"
SOURCE_SCHEMA = "snerv_checkpoint_archive_export.v1"
AXIS_TAG = "[planning/control:false-authority]"


def build_snerv_lf_payload_codec_report_from_checkpoint_export(
    export_report: Mapping[str, Any],
    *,
    source_artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a sweep-compatible LF report from a real checkpoint export.

    This is accounting, not a recode claim: byte deltas are zero because the LF
    section is already present in the exported packet.  The useful signal is the
    receiver-visible section size, schema, mode histogram, and proof/blocker
    status that existing inventory/queue surfaces can consume.
    """

    if export_report.get("schema") != SOURCE_SCHEMA:
        raise ValueError(
            f"expected {SOURCE_SCHEMA}, got {export_report.get('schema')!r}"
        )
    lf_report = _lf_report(export_report)
    report_path = str(
        source_artifact_path
        or export_report.get("report_path")
        or export_report.get("source_artifact_path")
        or ""
    )
    section_bytes = _int_or_none(lf_report.get("section_bytes"))
    packet_bytes = _int_or_none(lf_report.get("packet_bytes"))
    selected_bytes = section_bytes if section_bytes is not None else packet_bytes
    mode_histogram = dict(lf_report.get("mode_histogram") or {})
    selected_mode = str(export_report.get("lf_payload_codec") or "checkpoint_export")
    packet_schema = str(lf_report.get("schema") or "")
    blockers = _blockers(export_report, lf_report)
    plane_rows = _plane_rows(lf_report)
    source_packet_metadata = dict(export_report.get("packet_metadata_summary") or {})
    if "n_pairs" not in source_packet_metadata:
        source_packet_metadata["n_pairs"] = _nested(
            export_report, ("modelsize_candidate", "num_pairs")
        )

    section_value_row = {
        "row_id": "snerv_checkpoint_export_lf_payload_accounting",
        "section_id": "snerv_lf_payload",
        "family": "snerv",
        "scope": "lf_payload_checkpoint_export_accounting",
        "row_kind": "existing_section_accounting",
        "candidate_mode": selected_mode,
        "baseline_mode": selected_mode,
        "baseline_payload_bytes": selected_bytes,
        "payload_bytes": selected_bytes,
        "byte_delta": 0,
        "delta_nonrate_score": 0.0,
        "axis_tag": AXIS_TAG,
        "receiver_proof_status": (
            "receiver_proof_passed"
            if export_report.get("receiver_proof_passed") is True
            else "receiver_proof_missing_or_failed"
        ),
        "full_video_coverage": bool(
            export_report.get("local_mlx_prefilter_written") is True
        ),
        "blockers": list(blockers),
        **FALSE_AUTHORITY,
    }
    return {
        "schema": SCHEMA,
        "source_schema": SOURCE_SCHEMA,
        "source_kind": "snerv_checkpoint_archive_export",
        "status": "snerv_checkpoint_export_lf_payload_accounting_false_authority",
        "axis_tag": AXIS_TAG,
        "report_path": report_path or None,
        "source_artifact_path": report_path or None,
        "source_artifact_bytes": export_report.get("source_artifact_bytes"),
        "source_artifact_sha256": export_report.get("source_artifact_sha256"),
        "source": {
            "kind": "snar1_packet",
            "path": export_report.get("packet_path"),
            "bytes": export_report.get("packet_bytes"),
            "sha256": export_report.get("packet_sha256"),
            "metadata": source_packet_metadata,
        },
        "plane_count": lf_report.get("plane_count"),
        "plane_shapes": [row.get("shape") for row in plane_rows if row.get("shape")],
        "raw_i64_bytes": lf_report.get("raw_i64_bytes"),
        "baseline_mode": selected_mode,
        "baseline_payload_bytes": selected_bytes,
        "selected_rate_only_row": {
            "mode": selected_mode,
            "payload_bytes": selected_bytes,
            "packet_schema": packet_schema or None,
            "mode_histogram": mode_histogram,
            "wrapper_histogram": dict(lf_report.get("wrapper_histogram") or {}),
            "blockers": list(blockers),
        },
        "rows": [
            {
                "mode": selected_mode,
                "payload_bytes": selected_bytes,
                "packet_schema": packet_schema or None,
                "mode_histogram": mode_histogram,
                "blockers": list(blockers),
            }
        ],
        "section_value_rows": [section_value_row],
        "byte_price_plan": {
            "schema": "compact_nerv_byte_price_controller.v1",
            "source_schema": SOURCE_SCHEMA,
            "decision_rows": [
                {
                    "row_id": section_value_row["row_id"],
                    "section_id": section_value_row["section_id"],
                    "row_kind": section_value_row["row_kind"],
                    "decision": "observe",
                    "economic_decision": "no_mutation_accounting_only",
                    "byte_delta": 0,
                    "delta_nonrate_score": 0.0,
                    "source": {"candidate_mode": selected_mode},
                    "blockers": list(blockers),
                    **FALSE_AUTHORITY,
                }
            ],
            **FALSE_AUTHORITY,
        },
        "blockers": list(blockers),
        **FALSE_AUTHORITY,
    }


def _lf_report(export_report: Mapping[str, Any]) -> dict[str, Any]:
    full = export_report.get("packet_section_reports")
    if isinstance(full, Mapping) and isinstance(full.get("lf_payload_codec_report"), Mapping):
        return dict(full["lf_payload_codec_report"])
    summary = export_report.get("packet_section_report_summary")
    if isinstance(summary, Mapping) and isinstance(
        summary.get("lf_payload_codec_report"),
        Mapping,
    ):
        return dict(summary["lf_payload_codec_report"])
    raise ValueError("checkpoint export is missing lf_payload_codec_report")


def _blockers(
    export_report: Mapping[str, Any],
    lf_report: Mapping[str, Any],
) -> tuple[str, ...]:
    blockers: list[str] = []
    blockers.extend(str(value) for value in export_report.get("blockers") or ())
    blockers.extend(str(value) for value in lf_report.get("blockers") or ())
    if lf_report.get("report_status") != "receiver_visible_lf_payload_accounting_verified":
        blockers.append("snerv_checkpoint_export_lf_payload_accounting_not_verified")
    if export_report.get("receiver_proof_passed") is not True:
        blockers.append("receiver_proof_not_passed")
    if export_report.get("local_mlx_prefilter_written") is not True:
        blockers.append("full_video_scorer_replay_not_executed")
    blockers.append("checkpoint_export_lf_accounting_not_recode_candidate")
    blockers.append("paired_contest_cpu_cuda_auth_eval_missing")
    return tuple(dict.fromkeys(value for value in blockers if value))


def _plane_rows(lf_report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = lf_report.get("plane_rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _nested(root: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = root
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["build_snerv_lf_payload_codec_report_from_checkpoint_export"]
