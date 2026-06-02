# SPDX-License-Identifier: MIT
"""Harvestable feedback rows for NeRV candidate curriculum runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "nerv_candidate_feedback_row.v1"
LEDGER_SCHEMA = "nerv_candidate_byte_feedback_ledger.v1"

FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def _sha256_file(path: Path) -> str | None:
    import hashlib

    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_nerv_candidate_feedback_row(
    *,
    runner_report: Mapping[str, Any],
    source_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one false-authority feedback row from a runner report."""

    selection = dict(runner_report.get("modelsize_candidate_selection") or {})
    curriculum = dict(
        runner_report.get("candidate_curriculum_plan")
        or selection.get("candidate_curriculum_plan")
        or {}
    )
    byte_feedback = dict(curriculum.get("byte_oracle_logging") or {})
    pr95_binding = dict(
        curriculum.get("pr95_stack_binding")
        or selection.get("pr95_stack_binding")
        or {}
    )
    prelaunch_gate = dict(
        curriculum.get("long_campaign_prelaunch_gate")
        or selection.get("long_campaign_prelaunch_gate")
        or {}
    )
    candidate = selection.get("candidate")
    candidate_row = dict(candidate) if isinstance(candidate, Mapping) else {}
    source_path = (
        Path(source_report_path).expanduser().resolve(strict=False)
        if source_report_path
        else None
    )
    local_replay = runner_report.get("local_cpu_replay_summary")
    local_replay_gate = dict(runner_report.get("local_cpu_replay_gate") or {})
    mlx_prefilter = dict(runner_report.get("mlx_prefilter_coverage") or {})
    snerv_profile = dict(runner_report.get("snerv_binary_profile") or {})
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_report_path": source_path.as_posix() if source_path else None,
        "source_report_sha256": _sha256_file(source_path) if source_path else None,
        "mode": runner_report.get("mode"),
        "family": runner_report.get("execute_family"),
        "candidate_id": candidate_row.get("candidate_id") or curriculum.get("candidate_id"),
        "candidate_conditioned": bool(curriculum.get("candidate_conditioned")),
        "candidate_num_pairs": byte_feedback.get("candidate_num_pairs"),
        "measured_num_pairs": byte_feedback.get("measured_num_pairs"),
        "feedback_scope": byte_feedback.get("feedback_scope"),
        "scope_matches_candidate": bool(byte_feedback.get("scope_matches_candidate")),
        "feedback_ready": bool(byte_feedback.get("feedback_ready")),
        "hard_byte_ceiling": byte_feedback.get("hard_byte_ceiling"),
        "nominal_total_payload_bytes": byte_feedback.get("nominal_total_payload_bytes"),
        "measured_payload_bytes": byte_feedback.get("measured_payload_bytes"),
        "measured_archive_bytes": byte_feedback.get("measured_archive_bytes"),
        "measured_minus_nominal_bytes": byte_feedback.get(
            "measured_minus_nominal_bytes"
        ),
        "archive_path": runner_report.get("archive_path"),
        "archive_bytes": runner_report.get("archive_bytes"),
        "archive_sha256": runner_report.get("archive_sha256"),
        "snerv_binary_profile_path": snerv_profile.get("profile_path"),
        "snerv_binary_profile_written": bool(snerv_profile.get("profile_written")),
        "snerv_binary_profile_verdict": snerv_profile.get("verdict"),
        "snerv_binary_profile_charged_archive_bytes": snerv_profile.get(
            "charged_archive_bytes"
        ),
        "snerv_binary_profile_snar1_packet_bytes": snerv_profile.get(
            "snar1_packet_bytes"
        ),
        "snerv_binary_profile_lf_payload_bytes": snerv_profile.get(
            "lf_payload_bytes"
        ),
        "snerv_binary_profile_lf_payload_fraction_of_packet": snerv_profile.get(
            "lf_payload_fraction_of_packet"
        ),
        "snerv_binary_profile_lf_payload_bytes_per_coeff": snerv_profile.get(
            "lf_payload_bytes_per_coeff"
        ),
        "snerv_binary_profile_blockers": list(snerv_profile.get("blockers") or []),
        "pr95_stack_binding_schema": pr95_binding.get("schema"),
        "pr95_stack_binding_satisfied_count": pr95_binding.get("satisfied_count"),
        "pr95_stack_binding_missing_count": pr95_binding.get("missing_count"),
        "pr95_stack_binding_complete": pr95_binding.get("complete"),
        "pr95_stack_binding_blockers": list(pr95_binding.get("blockers") or []),
        "long_campaign_prelaunch_gate_schema": prelaunch_gate.get("schema"),
        "long_campaign_prelaunch_launch_allowed": prelaunch_gate.get(
            "launch_allowed"
        ),
        "long_campaign_prelaunch_blockers": list(prelaunch_gate.get("blockers") or []),
        "receiver_proof_report_paths": list(
            runner_report.get("receiver_proof_report_paths") or []
        ),
        "local_cpu_replay_summary_present": isinstance(local_replay, Mapping),
        "local_cpu_replay_score_estimate": (
            local_replay.get("local_score_estimate")
            if isinstance(local_replay, Mapping)
            else None
        ),
        "local_cpu_replay_gate_requested": local_replay_gate.get("requested"),
        "local_cpu_replay_gate_default_enabled_for_full_coverage": (
            local_replay_gate.get("default_enabled_for_full_coverage")
        ),
        "local_cpu_replay_gate_has_full_video_mlx_prefilter": (
            local_replay_gate.get("has_full_video_mlx_prefilter")
        ),
        "local_cpu_replay_gate_local_replay_mlx_prefilter_passed": (
            local_replay_gate.get("local_replay_mlx_prefilter_passed")
        ),
        "local_cpu_replay_gate_coverage_valid_for_replay": (
            local_replay_gate.get("coverage_valid_for_replay")
        ),
        "local_cpu_replay_gate_executed": local_replay_gate.get("executed"),
        "mlx_prefilter_profile_count": mlx_prefilter.get("profile_count"),
        "mlx_prefilter_has_full_video": mlx_prefilter.get(
            "has_full_video_mlx_prefilter"
        ),
        "mlx_prefilter_local_replay_passed": mlx_prefilter.get(
            "local_replay_mlx_prefilter_passed"
        ),
        "mlx_prefilter_best_full_video_mlx_score": mlx_prefilter.get(
            "best_full_video_mlx_score"
        ),
        "mlx_prefilter_full_video_profile_paths": list(
            mlx_prefilter.get("full_video_profile_paths") or []
        ),
        "mlx_prefilter_local_replay_profile_paths": list(
            mlx_prefilter.get("local_replay_profile_paths") or []
        ),
        "mlx_prefilter_blockers": list(mlx_prefilter.get("blockers") or []),
        "blockers": list(runner_report.get("blockers") or []),
        **FALSE_AUTHORITY,
    }


def write_nerv_candidate_feedback_files(
    *,
    runner_report: Mapping[str, Any],
    output_dir: str | Path,
    source_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write JSON and JSONL feedback artifacts under ``output_dir``."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    row = build_nerv_candidate_feedback_row(
        runner_report=runner_report,
        source_report_path=source_report_path,
    )
    row_path = out / "nerv_candidate_byte_feedback_row.json"
    ledger_path = out / "nerv_candidate_byte_feedback.jsonl"
    row_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "schema": LEDGER_SCHEMA,
        "row": row,
        "row_path": row_path.as_posix(),
        "ledger_path": ledger_path.as_posix(),
        "append_only": True,
        **FALSE_AUTHORITY,
    }


__all__ = [
    "LEDGER_SCHEMA",
    "SCHEMA",
    "build_nerv_candidate_feedback_row",
    "write_nerv_candidate_feedback_files",
]
