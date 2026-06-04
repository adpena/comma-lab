# SPDX-License-Identifier: MIT
"""Convert SNeRV upstream eval gates into planner feedback rows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.repo_io import read_json, sha256_file, write_json

NERV_CANDIDATE_FEEDBACK_ROW_SCHEMA = "nerv_candidate_feedback_row.v1"
SNERV_UPSTREAM_EVAL_FEEDBACK_KIND = "upstream_eval_gate"
SNERV_UPSTREAM_EVAL_FEEDBACK_SCOPE = "full600_upstream_cpu_eval"
SNERV_UPSTREAM_EVAL_CONTEXT_CANDIDATE_ID = "snerv_upstream_data_only_snsa2"
SNERV_UPSTREAM_EVAL_FRONTIER_RELEVANCE_MAX_SCORE = 1.0

FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
}


def build_snerv_upstream_eval_candidate_feedback(
    *,
    gate_report: Mapping[str, Any],
    gate_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a false-authority planner feedback row from an upstream eval gate."""

    if gate_report.get("schema") != "snerv_upstream_eval_gate.v1":
        raise ValueError("not a SNeRV upstream eval gate report")
    evaluation = _mapping(gate_report.get("evaluation"))
    source_bundle = _mapping(gate_report.get("source_bundle_json"))
    receiver_proof = _mapping(source_bundle.get("receiver_proof"))
    archive = _mapping(evaluation.get("archive_zip"))
    parsed = _mapping(evaluation.get("parsed_report"))
    blockers = _dedupe(
        [
            *(str(blocker) for blocker in gate_report.get("blockers") or [] if blocker),
            *(str(blocker) for blocker in evaluation.get("blockers") or [] if blocker),
        ]
    )
    if evaluation.get("returncode") != 0:
        blockers.append("snerv_upstream_eval_gate_failed")
    final_score = _float_or_none(parsed.get("final_score"))
    if final_score is None:
        blockers.append("snerv_upstream_eval_gate_score_missing")
    elif final_score > SNERV_UPSTREAM_EVAL_FRONTIER_RELEVANCE_MAX_SCORE:
        blockers.append("snerv_upstream_eval_gate_score_bad")
    if (
        receiver_proof.get("runtime_consumption_proof_passed") is not True
        or receiver_proof.get("receiver_contract_satisfied") is not True
    ):
        blockers.append("snerv_upstream_eval_gate_receiver_proof_missing")

    proof_path = _existing_file(receiver_proof.get("proof_path"))
    report_path = _existing_file(
        evaluation.get("report_copy_path") or evaluation.get("report_path")
    )
    gate_path = _existing_file(gate_report_path)
    archive_path = _existing_file(archive.get("path"))
    receiver_proof_attached = bool(proof_path is not None)

    row: dict[str, Any] = {
        "schema": NERV_CANDIDATE_FEEDBACK_ROW_SCHEMA,
        "feedback_kind": SNERV_UPSTREAM_EVAL_FEEDBACK_KIND,
        "feedback_scope": SNERV_UPSTREAM_EVAL_FEEDBACK_SCOPE,
        "feedback_ready": False,
        "launch_control_feedback_ready": False,
        "family": "snerv",
        "candidate_id": SNERV_UPSTREAM_EVAL_CONTEXT_CANDIDATE_ID,
        "candidate_num_pairs": 600,
        "measured_num_pairs": 600,
        "scope_matches_candidate": False,
        "context_only": True,
        "family_scope_matches_target": True,
        "receiver_proof_attached": receiver_proof_attached,
        "receiver_proof_path": proof_path.as_posix() if proof_path else None,
        "receiver_proof_sha256": sha256_file(proof_path) if proof_path else None,
        "full_video_local_prefilter_attached": False,
        "local_cpu_replay_gate_attached": False,
        "measured_payload_bytes": None,
        "measured_archive_bytes": _int_or_none(archive.get("bytes")),
        "archive_path": archive_path.as_posix() if archive_path else archive.get("path"),
        "archive_sha256": archive.get("sha256"),
        "source_report_path": report_path.as_posix() if report_path else None,
        "source_report_sha256": sha256_file(report_path) if report_path else None,
        "upstream_eval_gate_path": gate_path.as_posix() if gate_path else None,
        "upstream_eval_gate_sha256": sha256_file(gate_path) if gate_path else None,
        "axis_tag": evaluation.get("axis_tag"),
        "upstream_eval_gate": {
            "schema": str(gate_report.get("schema")),
            "returncode": evaluation.get("returncode"),
            "device": evaluation.get("device"),
            "wall_seconds": evaluation.get("wall_seconds"),
            "inflated_dir_cleanup": evaluation.get("inflated_dir_cleanup"),
            "inflated_dir_retained": evaluation.get("inflated_dir_retained"),
            "inflated_outputs_manifest": evaluation.get("inflated_outputs_manifest"),
            "parsed_report": dict(parsed),
        },
        "upstream_eval_score": final_score,
        "upstream_eval_pose": _float_or_none(parsed.get("pose")),
        "upstream_eval_seg": _float_or_none(parsed.get("seg")),
        "upstream_eval_rate": _float_or_none(parsed.get("rate")),
        "recommended_launch_mutations": [
            "block_snerv_data_only_archive_as_launch_candidate_due_to_scorer_quality",
            "require_snerv_representation_change_before_more_same_long_training",
        ],
        "direct_feedback_blockers": _dedupe(blockers),
        "feedback_reuse_policy": (
            "family_upstream_eval_context_only_no_archive_receiver_replay_or_launch_authority"
        ),
        **FALSE_AUTHORITY,
    }
    return {key: value for key, value in row.items() if value is not None}


def write_snerv_upstream_eval_candidate_feedback(
    *,
    gate_report: Mapping[str, Any] | None = None,
    gate_report_path: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    """Write a planner feedback row for an existing or in-memory gate report."""

    report = gate_report if gate_report is not None else read_json(gate_report_path)
    row = build_snerv_upstream_eval_candidate_feedback(
        gate_report=report,
        gate_report_path=gate_report_path,
    )
    write_json(output_json, row)
    return row


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _existing_file(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser().resolve(strict=False)
    return path if path.is_file() else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values: list[str]) -> list[str]:
    return sorted(dict.fromkeys(value for value in values if value))
