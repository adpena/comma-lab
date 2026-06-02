# SPDX-License-Identifier: MIT
"""Bind HiNeRV archive-size ladder rows to decoder-weight waterfill plans."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.analysis.nerv_decoder_weight_waterfill import (
    DEFAULT_ACTION_BITS,
    NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
    NervDecoderWeightWaterfillError,
    build_nerv_decoder_weight_waterfill_plan,
    load_state_npz_from_manifest,
)
from tac.repo_io import sha256_file
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    build_nerv_byte_price_plan,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

HINERV_ARCHIVE_LADDER_WATERFILL_SCHEMA = "hinerv_archive_ladder_waterfill.v1"


class HinervArchiveLadderWaterfillError(ValueError):
    """Raised when a HiNeRV ladder waterfill input is malformed."""


def build_hinerv_archive_ladder_waterfill(
    archive_ladder_report: Mapping[str, Any],
    *,
    saliency_by_row_id: Mapping[str, Mapping[str, float]] | None = None,
    global_saliency_by_name: Mapping[str, float] | None = None,
    action_bits: Sequence[int] = DEFAULT_ACTION_BITS,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Build decoder-weight waterfill plans for every archive-ladder row.

    The input rows must carry ``state_npz_manifest_path`` from
    ``export_hi_nerv_mlx_archive``. Missing manifests, missing NPZ artifacts, or
    SHA mismatches are blockers, not guessed paths.
    """

    if archive_ladder_report.get("schema") != "hinerv_archive_size_ladder.v1":
        raise HinervArchiveLadderWaterfillError(
            "expected hinerv_archive_size_ladder.v1 report"
        )
    row_saliency = saliency_by_row_id or {}
    global_saliency = global_saliency_by_name or {}
    num_pairs = int(archive_ladder_report.get("num_pairs") or 0)
    full_video_coverage = num_pairs == 600
    rows = []
    section_value_rows = []
    blockers = [
        "contest_cpu_cuda_exact_eval_not_executed",
        "decoder_weight_saliency_replay_required_for_authority",
    ]
    for archive_row in archive_ladder_report.get("archive_rows") or ():
        if not isinstance(archive_row, Mapping):
            continue
        row_id = str(archive_row.get("row_id") or f"row_{len(rows):04d}")
        row_result = _waterfill_for_archive_row(
            archive_row,
            row_id=row_id,
            saliency_by_name={
                **global_saliency,
                **dict(row_saliency.get(row_id) or {}),
            },
            action_bits=action_bits,
            full_video_coverage=full_video_coverage,
            candidate_id=candidate_id or str(archive_ladder_report.get("candidate_id") or ""),
        )
        rows.append(row_result)
        blockers.extend(row_result["blockers"])
        plan = row_result.get("waterfill_plan")
        if isinstance(plan, Mapping):
            for section in plan.get("section_value_rows") or ():
                if isinstance(section, Mapping):
                    section_value_rows.append(
                        {
                            **dict(section),
                            "row_id": f"{row_id}:{section.get('row_id')}",
                            "section_id": f"{row_id}:{section.get('section_id')}",
                            "archive_ladder_row_id": row_id,
                        }
                    )
    report = {
        "schema": HINERV_ARCHIVE_LADDER_WATERFILL_SCHEMA,
        "source_schema": archive_ladder_report.get("schema"),
        "family": "hi_nerv",
        "axis_tag": "[planning/control]",
        "authority": "false_authority_hinerv_ladder_decoder_waterfill_no_score_claim",
        "candidate_id": candidate_id or archive_ladder_report.get("candidate_id"),
        "num_pairs": num_pairs,
        "full_video_coverage": full_video_coverage,
        "archive_ladder_report_path": archive_ladder_report.get("report_path"),
        "row_count": len(rows),
        "rows": rows,
        "section_value_rows": section_value_rows,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }
    report["byte_price_plan"] = build_nerv_byte_price_plan(report)
    return report


def render_hinerv_archive_ladder_waterfill_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a compact Markdown summary."""

    lines = [
        "# HiNeRV archive ladder decoder waterfill",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        "",
        "| row | groups | byte delta | blocker count |",
        "|---|---:|---:|---:|",
    ]
    for row in report.get("rows") or ():
        summary = row.get("waterfill_summary") or {}
        lines.append(
            "| {row_id} | {groups} | {delta} | {blockers} |".format(
                row_id=row.get("row_id"),
                groups=summary.get("group_count", 0),
                delta=summary.get("total_selected_byte_delta", 0),
                blockers=len(row.get("blockers") or ()),
            )
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in report.get("blockers") or ():
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def _waterfill_for_archive_row(
    archive_row: Mapping[str, Any],
    *,
    row_id: str,
    saliency_by_name: Mapping[str, float],
    action_bits: Sequence[int],
    full_video_coverage: bool,
    candidate_id: str,
) -> dict[str, Any]:
    blockers = []
    manifest_path = Path(str(archive_row.get("state_npz_manifest_path") or ""))
    manifest = {}
    state_dict = None
    if not manifest_path.is_file():
        blockers.append("state_npz_manifest_missing")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            state_dict = load_state_npz_from_manifest(manifest_path)
        except NervDecoderWeightWaterfillError as exc:
            blockers.append(_manifest_error_blocker(str(exc)))
    artifact_path = Path(str(manifest.get("artifact_path") or ""))
    if artifact_path and not artifact_path.is_absolute():
        artifact_path = manifest_path.parent / artifact_path
    if manifest and not artifact_path.is_file():
        blockers.append("state_npz_artifact_missing")
    expected_sha = str(manifest.get("artifact_sha256") or "")
    actual_sha = None
    if artifact_path.is_file():
        actual_sha = sha256_file(artifact_path)
        if expected_sha and actual_sha != expected_sha:
            blockers.append("state_npz_artifact_sha256_mismatch")
    if blockers:
        return {
            "row_id": row_id,
            "archive_bytes": archive_row.get("archive_bytes"),
            "archive_sha256": archive_row.get("archive_sha256"),
            "state_npz_manifest_path": str(manifest_path) if str(manifest_path) else None,
            "state_npz_artifact_path": str(artifact_path) if str(artifact_path) else None,
            "state_npz_artifact_sha256": actual_sha,
            "waterfill_plan": None,
            "waterfill_summary": None,
            "blockers": _ordered_unique(blockers),
            **FALSE_AUTHORITY,
        }
    plan = build_nerv_decoder_weight_waterfill_plan(
        state_dict or {},
        saliency_by_name=saliency_by_name,
        family="hi_nerv",
        candidate_id=f"{candidate_id}:{row_id}" if candidate_id else row_id,
        action_bits=action_bits,
        full_video_coverage=bool(full_video_coverage),
        receiver_proof_status=(
            "runtime_consumption_proof_ready"
            if archive_row.get("runtime_consumption_proof_ready") is True
            else "missing"
        ),
        archive_sha256=str(archive_row.get("archive_sha256") or ""),
    )
    return {
        "row_id": row_id,
        "archive_bytes": int(archive_row.get("archive_bytes") or 0),
        "archive_sha256": archive_row.get("archive_sha256"),
        "state_npz_manifest_path": manifest_path.as_posix(),
        "state_npz_artifact_path": artifact_path.as_posix(),
        "state_npz_artifact_sha256": actual_sha,
        "state_npz_manifest_sha256": expected_sha,
        "waterfill_plan_schema": NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
        "waterfill_summary": {
            "group_count": plan["group_count"],
            "total_baseline_fp32_bytes": plan["total_baseline_fp32_bytes"],
            "total_selected_estimated_bytes": plan["total_selected_estimated_bytes"],
            "total_selected_byte_delta": plan["total_selected_byte_delta"],
        },
        "waterfill_plan": plan,
        "blockers": _ordered_unique([*plan["blockers"]]),
        **FALSE_AUTHORITY,
    }


def _manifest_error_blocker(message: str) -> str:
    if "sha256 mismatch" in message:
        return "state_npz_artifact_sha256_mismatch"
    if "schema" in message:
        return "state_npz_manifest_schema_unexpected"
    if "not consumption-recommended" in message:
        return "state_npz_manifest_not_consumption_recommended"
    if "artifact_path" in message:
        return "state_npz_manifest_missing_artifact_path"
    if "artifact does not exist" in message:
        return "state_npz_artifact_missing"
    return "state_npz_manifest_unreadable"


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


__all__ = [
    "HINERV_ARCHIVE_LADDER_WATERFILL_SCHEMA",
    "HinervArchiveLadderWaterfillError",
    "build_hinerv_archive_ladder_waterfill",
    "render_hinerv_archive_ladder_waterfill_markdown",
]
