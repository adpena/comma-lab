# SPDX-License-Identifier: MIT
"""Queue handoff for SNeRV LF payload over-ceiling reroutes.

Measured LF payload bytes are useful only if they change the next queue action.
This module turns disabled SNeRV long-training rows into explicit LF
representation-change work orders, while preserving the no-score/no-dispatch
authority boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    CONTEST_BYTE_PRICE_SCORE,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

SCHEMA = "snerv_lf_over_ceiling_reroute_queue.v1"
ROW_SCHEMA = "snerv_lf_over_ceiling_reroute_row.v1"
AXIS_TAG = "[planning/control:false-authority]"
DEFAULT_QUEUE_ID = "snerv_lf_over_ceiling_reroute_queue.v1"

QUEUE_FALSE_AUTHORITY = {
    **FALSE_AUTHORITY,
    "frontier_score_claim": False,
    "production_hardened_claim": False,
    "local_mlx_long_training_allowed": False,
    "dispatch_allowed": False,
    "exact_or_full_video_cuda_allowed": False,
}

_OVER_CEILING_BLOCKERS = {
    "snerv_lf_payload_rate_axis_over_ceiling_until_representation_changes",
    "snerv_hard_byte_ceiling_not_receiver_satisfied_for_long_training",
    "snerv_lf_payload_recode_still_over_hard_byte_ceiling",
    "snerv_lf_recode_selected_mode_still_over_byte_waterline",
    "snerv_modelsize_auto_calibrated_byte_cap_over_ceiling",
    "snerv_nominal_payload_far_over_ceiling_refuse_long_training",
    "snerv_receiver_proven_archive_over_hard_byte_ceiling_observed_demote_only",
    "snerv_receiver_proven_archive_over_hard_byte_ceiling",
}


class SnervLfOverCeilingRerouteQueueError(ValueError):
    """Raised when the SNeRV LF reroute queue cannot be built."""


def build_snerv_lf_over_ceiling_reroute_queue(
    *,
    campaign_rows: Sequence[Mapping[str, Any]],
    measured_lf_payload_sources: Sequence[Mapping[str, Any]] = (),
    measured_lf_payload_paths: Sequence[str | Path] = (),
    snar_header_grammar_profiles: Sequence[Mapping[str, Any]] = (),
    snar_header_minimization_reports: Sequence[Mapping[str, Any]] = (),
    output_root: str | Path,
    queue_id: str = DEFAULT_QUEUE_ID,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Build a false-authority queue for LF over-ceiling reroutes.

    The queue does not launch long training.  It only describes local packet
    recode/prototype follow-up when a SNeRV campaign row is disabled because
    measured or predicted LF bytes keep the archive over its hard ceiling.
    """

    if not queue_id:
        raise SnervLfOverCeilingRerouteQueueError("queue_id must be non-empty")
    generated = generated_utc or datetime.now(UTC).isoformat()
    evidence = _lf_evidence_rows(
        measured_lf_payload_sources=measured_lf_payload_sources,
        measured_lf_payload_paths=measured_lf_payload_paths,
    )
    header_profiles = _header_profile_index(snar_header_grammar_profiles)
    header_minimization_source_count = len(tuple(snar_header_minimization_reports))
    header_minimizations = _header_minimization_index(snar_header_minimization_reports)
    rows: list[dict[str, Any]] = []
    for campaign_row in campaign_rows:
        if not _needs_lf_reroute(campaign_row):
            continue
        selected = _select_evidence_for_row(campaign_row, evidence)
        rows.extend(
            _queue_rows_for_campaign_row(
                campaign_row=campaign_row,
                evidence=selected,
                header_profiles=header_profiles,
                header_minimizations=header_minimizations,
                output_root=Path(output_root),
            )
        )
    rows.sort(
        key=lambda row: (
            int(row.get("priority") or 999),
            str(row.get("source_campaign_row_id") or ""),
            str(row.get("queue_row_id") or ""),
        )
    )
    blocked_rows = [row for row in rows if row["blocked"]]
    recode_ready_rows = [
        row for row in rows if row["work_order_type"] == "lossless_lf_recode_probe"
        and not row["blocked"]
    ]
    executable_rows = [row for row in rows if row.get("command_argv") and not row["blocked"]]
    return {
        "schema": SCHEMA,
        "queue_id": str(queue_id),
        "generated_utc": generated,
        "axis_tag": AXIS_TAG,
        "queue_kind": "planner_queue_not_training_queue",
        "allowed_use": (
            "local SNeRV LF packet recode, representation-change prototype "
            "selection, and blocker routing"
        ),
        "forbidden_use": (
            "long-training launch, score claim, rank/kill decision, promotion, "
            "exact eval dispatch, or public frontier authority"
        ),
        "source_campaign_row_count": len(tuple(campaign_rows)),
        "measured_lf_payload_source_count": len(evidence),
        "snar_header_grammar_profile_count": len(header_profiles),
        "snar_header_minimization_report_count": header_minimization_source_count,
        "queue_rows": rows,
        "queue_row_count": len(rows),
        "blocked_queue_row_count": len(blocked_rows),
        "local_executable_command_row_count": len(executable_rows),
        "local_recode_command_row_count": len(recode_ready_rows),
        "representation_candidate_row_count": sum(
            1 for row in rows if row["work_order_type"] == "lf_representation_change_candidate"
        ),
        "blocking_queue_row_ids": [row["queue_row_id"] for row in blocked_rows],
        "activation_policy": {
            "planner_rows_may_be_ranked": True,
            "local_packet_recode_probe_allowed": True,
            "local_mlx_long_training_allowed": False,
            "exact_or_full_video_cuda_allowed": False,
            "dispatch_allowed": False,
            "requires_before_any_long_training_row_can_be_reenabled": [
                "measured_lf_payload_bytes",
                "receiver_visible_representation_change",
                "byte_charged_candidate_packet",
                "receiver_replay_proof",
                "full_video_local_prefilter_or_explicit_side_smoke_label",
            ],
        },
        "blockers": _dedupe(
            [
                "snerv_lf_over_ceiling_reroute_queue_false_authority",
                "long_training_reenable_requires_receiver_visible_byte_savings",
                *[
                    blocker
                    for row in rows
                    for blocker in row.get("blockers", ())
                ],
            ]
        ),
        **QUEUE_FALSE_AUTHORITY,
    }


def _queue_rows_for_campaign_row(
    *,
    campaign_row: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
    header_profiles: Mapping[str, Mapping[str, Any]],
    header_minimizations: Mapping[str, Mapping[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    recode_evidence = _recode_admission_evidence_from_campaign_row(
        campaign_row,
        header_profiles=header_profiles,
        header_minimizations=header_minimizations,
    )
    if recode_evidence is not None:
        return _queue_rows_for_recode_admission(
            campaign_row=campaign_row,
            evidence=recode_evidence,
            output_root=output_root,
        )

    if evidence is None:
        return [
            _base_row(
                campaign_row=campaign_row,
                evidence=None,
                representation_candidate_id="snerv_measured_lf_payload_report_required",
                work_order_type="lf_reroute_blocker",
                planner_action="attach_measured_lf_payload_byte_report_before_long_training",
                priority=5,
                blocked=True,
                blockers=[
                    "snerv_measured_lf_payload_report_missing",
                    "snerv_lf_representation_change_candidate_blocked_until_measured_lf_bytes",
                ],
                command_argv=[],
                output_root=output_root,
            )
        ]

    required = _required_lf_savings_bytes(campaign_row, evidence)
    lf_bytes = _positive_int(evidence.get("lf_payload_bytes"))
    rows: list[dict[str, Any]] = []
    rows.append(
        _lossless_recode_probe_row(
            campaign_row=campaign_row,
            evidence=evidence,
            output_root=output_root,
            required_lf_savings_bytes=required,
            lf_payload_bytes=lf_bytes,
        )
    )
    rows.append(
        _representation_candidate_row(
            campaign_row=campaign_row,
            evidence=evidence,
            output_root=output_root,
            representation_candidate_id="snerv_lf_temporal_tub_gate_receiver_visible",
            planner_action="build_byte_charged_receiver_visible_lf_tub_temporal_gate",
            required_lf_savings_bytes=required,
            lf_payload_bytes=lf_bytes,
            blockers=[
                "snerv_lf_tub_temporal_gate_not_implemented",
                "snerv_lf_tub_temporal_gate_learned_bytes_not_charged",
                "snerv_lf_tub_temporal_gate_receiver_replay_proof_missing",
            ],
        )
    )
    rows.append(
        _representation_candidate_row(
            campaign_row=campaign_row,
            evidence=evidence,
            output_root=output_root,
            representation_candidate_id="snerv_lf_resolution_or_quantization_change",
            planner_action="change_lf_resolution_or_quantization_then_receiver_replay",
            required_lf_savings_bytes=required,
            lf_payload_bytes=lf_bytes,
            blockers=[
                "snerv_lf_resolution_quantization_candidate_missing",
                "snerv_lf_representation_change_component_deltas_missing",
                "snerv_lf_representation_change_receiver_replay_proof_missing",
            ],
        )
    )
    return rows


def _queue_rows_for_recode_admission(
    *,
    campaign_row: Mapping[str, Any],
    evidence: Mapping[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    over_waterline = _positive_int(evidence.get("post_recode_over_waterline_bytes"))
    candidate_lf_bytes = _positive_int(evidence.get("candidate_lf_payload_bytes"))
    candidate_header_bytes = _positive_int(evidence.get("candidate_packet_header_bytes"))
    result_blockers = list(evidence.get("blockers") or ())
    if over_waterline is not None and over_waterline > 0:
        result_blockers.extend(
            [
                "snerv_lf_recode_selected_mode_still_over_byte_waterline",
                "snerv_lf_payload_recode_still_over_hard_byte_ceiling",
                "snerv_post_recode_packet_still_over_hard_byte_ceiling",
            ]
        )
        if candidate_lf_bytes is not None and over_waterline > candidate_lf_bytes:
            result_blockers.append(
                "snerv_post_recode_overrun_exceeds_remaining_lf_payload_bytes"
            )
        if candidate_header_bytes is not None and candidate_header_bytes >= over_waterline:
            result_blockers.extend(
                [
                    "snerv_post_recode_overrun_dominated_by_packet_header_bytes",
                    "snerv_snar_packet_header_grammar_rewrite_required",
                ]
            )
    else:
        result_blockers.extend(
            [
                "snerv_lf_recode_packet_waterline_crossed_but_archive_packaging_missing",
                "snerv_lf_recode_full_video_replay_missing",
            ]
        )

    rows = [
        _base_row(
            campaign_row=campaign_row,
            evidence=evidence,
            representation_candidate_id="snerv_lf_recode_admitted_result",
            work_order_type="lf_recode_admission_result",
            planner_action=(
                "preserve_lossless_lf_recode_candidate_then_route_remaining_packet_overrun"
            ),
            priority=8,
            blocked=True,
            blockers=result_blockers,
            command_argv=[],
            output_root=output_root,
            required_lf_savings_bytes=over_waterline,
            lf_payload_bytes=candidate_lf_bytes,
        )
    ]
    if over_waterline is not None and over_waterline > 0:
        if _header_minimization_satisfies_waterline(evidence):
            rows.append(
                _header_minimization_result_row(
                    campaign_row=campaign_row,
                    evidence=evidence,
                    output_root=output_root,
                    required_savings_bytes=over_waterline,
                )
            )
            representation_precedence_blockers = [
                "snerv_snar_header_minimization_result_precedes_lf_representation_change"
            ]
        elif _header_rewrite_should_precede_lf_representation(evidence):
            rows.append(
                _header_rewrite_work_order_row(
                    campaign_row=campaign_row,
                    evidence=evidence,
                    output_root=output_root,
                    required_savings_bytes=over_waterline,
                )
            )
            representation_precedence_blockers = [
                "snerv_snar_header_grammar_rewrite_precedes_lf_representation_change"
            ]
        else:
            representation_precedence_blockers = []
        rows.append(
            _representation_candidate_row(
                campaign_row=campaign_row,
                evidence=evidence,
                output_root=output_root,
                representation_candidate_id="snerv_lf_temporal_tub_gate_receiver_visible",
                planner_action="build_byte_charged_receiver_visible_lf_tub_temporal_gate",
                required_lf_savings_bytes=over_waterline,
                lf_payload_bytes=candidate_lf_bytes,
                blockers=[
                    *representation_precedence_blockers,
                    "snerv_lf_tub_temporal_gate_not_implemented",
                    "snerv_lf_tub_temporal_gate_learned_bytes_not_charged",
                    "snerv_lf_tub_temporal_gate_receiver_replay_proof_missing",
                ],
            )
        )
        rows.append(
            _representation_candidate_row(
                campaign_row=campaign_row,
                evidence=evidence,
                output_root=output_root,
                representation_candidate_id="snerv_lf_resolution_or_quantization_change",
                planner_action="change_lf_resolution_or_quantization_then_receiver_replay",
                required_lf_savings_bytes=over_waterline,
                lf_payload_bytes=candidate_lf_bytes,
                blockers=[
                    *representation_precedence_blockers,
                    "snerv_lf_resolution_quantization_candidate_missing",
                    "snerv_lf_representation_change_component_deltas_missing",
                    "snerv_lf_representation_change_receiver_replay_proof_missing",
                ],
            )
        )
    return rows


def _header_minimization_result_row(
    *,
    campaign_row: Mapping[str, Any],
    evidence: Mapping[str, Any],
    output_root: Path,
    required_savings_bytes: int,
) -> dict[str, Any]:
    return _base_row(
        campaign_row=campaign_row,
        evidence=evidence,
        representation_candidate_id="snerv_snar_header_minimization_result",
        work_order_type="snar_header_minimization_result",
        planner_action="preserve_minimized_snar1_packet_then_run_full_video_replay_and_admission",
        priority=11,
        blocked=True,
        blockers=[
            "snerv_snar_header_minimized_packet_false_authority",
            "snerv_snar_header_minimized_packet_recode_admission_not_rerun",
            "snerv_snar_header_minimized_packet_full_video_replay_missing",
            "paired_contest_cpu_cuda_auth_eval_missing",
        ],
        command_argv=[],
        output_root=output_root,
        required_lf_savings_bytes=required_savings_bytes,
        lf_payload_bytes=_positive_int(evidence.get("candidate_lf_payload_bytes")),
    )


def _header_rewrite_work_order_row(
    *,
    campaign_row: Mapping[str, Any],
    evidence: Mapping[str, Any],
    output_root: Path,
    required_savings_bytes: int,
) -> dict[str, Any]:
    packet_path = str(
        evidence.get("candidate_packet_path")
        or evidence.get("packet_path")
        or ""
    ).strip()
    token = _campaign_candidate_token(campaign_row, "snar_header_minimized")
    out_dir = output_root / token
    blockers: list[str] = []
    if not packet_path:
        blockers.append("snerv_snar_header_grammar_rewrite_packet_missing")
    ceiling = _positive_int(campaign_row.get("hard_byte_ceiling"))
    command = (
        []
        if not packet_path
        else [
            "uv",
            "run",
            "python",
            "tools/minimize_snerv_snar_header.py",
            "--packet",
            packet_path,
            "--output-packet",
            (out_dir / "candidate.minimized.snar").as_posix(),
            "--output-archive-zip",
            (out_dir / "archive.zip").as_posix(),
            "--output-json",
            (out_dir / "snerv_snar_header_minimization.json").as_posix(),
            *(
                []
                if ceiling is None
                else ["--hard-byte-ceiling", str(ceiling)]
            ),
        ]
    )
    return _base_row(
        campaign_row=campaign_row,
        evidence=evidence,
        representation_candidate_id="snerv_snar_header_grammar_rewrite_materialization",
        work_order_type="snar_header_grammar_rewrite_materialization",
        planner_action="run_receiver_proven_snar1_header_prune_then_rerun_recode_admission",
        priority=12,
        blocked=not bool(packet_path),
        blockers=blockers,
        command_argv=command,
        output_root=output_root,
        required_lf_savings_bytes=required_savings_bytes,
        lf_payload_bytes=_positive_int(evidence.get("candidate_lf_payload_bytes")),
    )


def _header_rewrite_should_precede_lf_representation(
    evidence: Mapping[str, Any],
) -> bool:
    profile = evidence.get("snar_header_grammar_profile")
    if not isinstance(profile, Mapping):
        return False
    if _positive_int(profile.get("header_bytes")) is None:
        return False
    ceiling_rows = profile.get("hard_byte_ceiling_rows")
    if not isinstance(ceiling_rows, Sequence):
        return False
    return any(
        isinstance(row, Mapping)
        and row.get("header_bytes_can_cover_overrun") is True
        and _positive_int(row.get("packet_over_ceiling_bytes")) is not None
        for row in ceiling_rows
    )


def _header_minimization_satisfies_waterline(evidence: Mapping[str, Any]) -> bool:
    summary = evidence.get("snar_header_minimization_report")
    if not isinstance(summary, Mapping):
        return False
    if summary.get("receiver_contract_satisfied") is not True:
        return False
    rows = summary.get("hard_byte_ceiling_rows")
    if not isinstance(rows, Sequence):
        return False
    return any(
        isinstance(row, Mapping)
        and row.get("candidate_packet_under_ceiling") is True
        and row.get("candidate_archive_zip_under_ceiling") is not False
        for row in rows
    )


def _lossless_recode_probe_row(
    *,
    campaign_row: Mapping[str, Any],
    evidence: Mapping[str, Any],
    output_root: Path,
    required_lf_savings_bytes: int | None,
    lf_payload_bytes: int | None,
) -> dict[str, Any]:
    mode = _lossless_recode_mode(evidence)
    packet_path = str(evidence.get("packet_path") or "").strip()
    blockers: list[str] = []
    if not packet_path:
        blockers.append("snerv_lf_recode_packet_path_missing")
    if not mode:
        blockers.append("snerv_lf_recode_mode_missing")
    if required_lf_savings_bytes is None:
        blockers.append("snerv_over_ceiling_required_savings_missing")
    elif lf_payload_bytes is not None and required_lf_savings_bytes > lf_payload_bytes:
        blockers.append("snerv_required_savings_exceeds_measured_lf_payload_bytes")
    token = _campaign_candidate_token(campaign_row, "lossless_lf_recode")
    out_dir = output_root / token
    command = (
        []
        if blockers
        else [
            "uv",
            "run",
            "python",
            "tools/recode_snerv_lf_payload_archive.py",
            "--packet",
            packet_path,
            "--mode",
            mode,
            "--output-packet",
            (out_dir / "candidate.snar").as_posix(),
            "--output-json",
            (out_dir / "snerv_lf_payload_archive_recode.json").as_posix(),
            "--output-md",
            (out_dir / "snerv_lf_payload_archive_recode.md").as_posix(),
        ]
    )
    return _base_row(
        campaign_row=campaign_row,
        evidence=evidence,
        representation_candidate_id="snerv_lossless_lf_recode_probe",
        work_order_type="lossless_lf_recode_probe",
        planner_action="run_receiver_visible_lossless_lf_recode_probe",
        priority=10,
        blocked=bool(blockers),
        blockers=blockers,
        command_argv=command,
        output_root=output_root,
        required_lf_savings_bytes=required_lf_savings_bytes,
        lf_payload_bytes=lf_payload_bytes,
    )


def _representation_candidate_row(
    *,
    campaign_row: Mapping[str, Any],
    evidence: Mapping[str, Any],
    output_root: Path,
    representation_candidate_id: str,
    planner_action: str,
    required_lf_savings_bytes: int | None,
    lf_payload_bytes: int | None,
    blockers: Sequence[str],
) -> dict[str, Any]:
    candidate_blockers = list(blockers)
    if required_lf_savings_bytes is None:
        candidate_blockers.append("snerv_over_ceiling_required_savings_missing")
    if lf_payload_bytes is None:
        candidate_blockers.append("snerv_measured_lf_payload_bytes_missing")
    elif required_lf_savings_bytes is not None and required_lf_savings_bytes > lf_payload_bytes:
        candidate_blockers.append("snerv_required_savings_exceeds_measured_lf_payload_bytes")
    return _base_row(
        campaign_row=campaign_row,
        evidence=evidence,
        representation_candidate_id=representation_candidate_id,
        work_order_type="lf_representation_change_candidate",
        planner_action=planner_action,
        priority=20,
        blocked=True,
        blockers=candidate_blockers,
        command_argv=[],
        output_root=output_root,
        required_lf_savings_bytes=required_lf_savings_bytes,
        lf_payload_bytes=lf_payload_bytes,
    )


def _base_row(
    *,
    campaign_row: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
    representation_candidate_id: str,
    work_order_type: str,
    planner_action: str,
    priority: int,
    blocked: bool,
    blockers: Sequence[str],
    command_argv: Sequence[str],
    output_root: Path,
    required_lf_savings_bytes: int | None = None,
    lf_payload_bytes: int | None = None,
) -> dict[str, Any]:
    row_id = str(campaign_row.get("row_id") or campaign_row.get("candidate_id") or "snerv")
    row_token = _campaign_candidate_token(campaign_row, representation_candidate_id)
    evidence_ref = dict(evidence or {})
    lf_fraction = (
        None
        if required_lf_savings_bytes is None or not lf_payload_bytes
        else float(required_lf_savings_bytes) / float(lf_payload_bytes)
    )
    return {
        "schema": ROW_SCHEMA,
        "queue_row_id": f"snerv_lf_reroute_{row_token}",
        "source_campaign_row_id": row_id,
        "candidate_id": campaign_row.get("candidate_id"),
        "representation_candidate_id": representation_candidate_id,
        "work_order_type": work_order_type,
        "planner_action": planner_action,
        "priority": int(priority),
        "status": "blocked_until_prerequisite_evidence" if blocked else "local_packet_recode_ready_no_training",
        "blocked": bool(blocked),
        "blockers": _dedupe(blockers),
        "source_campaign_status": {
            "implementation_status": campaign_row.get("implementation_status"),
            "local_mlx_launch_command_ready": bool(
                campaign_row.get("local_mlx_launch_command_ready")
            ),
            "experiment_queue_status": _nested(
                campaign_row, ("experiment_queue_entry", "status")
            ),
            "queue_launch_blockers": list(
                _nested(
                    campaign_row,
                    ("experiment_queue_entry", "launch_authority_contract", "queue_launch_blockers"),
                )
                or ()
            ),
            "campaign_blockers": list(campaign_row.get("blockers") or ()),
        },
        "hard_byte_ceiling": _positive_int(campaign_row.get("hard_byte_ceiling")),
        "measured_archive_bytes": _positive_int(
            evidence_ref.get("archive_bytes")
            or _nested(campaign_row, ("modelsize_byte_cap_preflight", "predicted_archive_bytes"))
        ),
        "measured_packet_bytes": _positive_int(evidence_ref.get("packet_bytes")),
        "source_lf_payload_bytes": _positive_int(evidence_ref.get("source_lf_payload_bytes")),
        "candidate_lf_payload_bytes": _positive_int(evidence_ref.get("candidate_lf_payload_bytes")),
        "candidate_packet_bytes": _positive_int(evidence_ref.get("candidate_packet_bytes")),
        "candidate_packet_header_bytes": _positive_int(
            evidence_ref.get("candidate_packet_header_bytes")
        ),
        "source_packet_path": evidence_ref.get("source_packet_path"),
        "candidate_packet_path": evidence_ref.get("candidate_packet_path"),
        "measured_lf_payload_bytes": lf_payload_bytes,
        "post_recode_over_waterline_bytes": _positive_int(
            evidence_ref.get("post_recode_over_waterline_bytes")
        ),
        "waterline_crossed_by_recode": evidence_ref.get("waterline_crossed_by_recode"),
        "lossless_lf_recode_already_admitted": bool(
            evidence_ref.get("lossless_lf_recode_already_admitted")
        ),
        "snar_header_grammar_profile_attached": bool(
            evidence_ref.get("snar_header_grammar_profile_attached")
        ),
        "snar_header_grammar_profile": evidence_ref.get("snar_header_grammar_profile"),
        "snar_header_minimization_report_attached": bool(
            evidence_ref.get("snar_header_minimization_report_attached")
        ),
        "snar_header_minimization_report": evidence_ref.get(
            "snar_header_minimization_report"
        ),
        "required_lf_savings_bytes": required_lf_savings_bytes,
        "required_lf_savings_rate_score": (
            None
            if required_lf_savings_bytes is None
            else float(required_lf_savings_bytes * CONTEST_BYTE_PRICE_SCORE)
        ),
        "required_lf_savings_fraction": lf_fraction,
        "lf_payload_can_cover_required_savings": (
            None
            if required_lf_savings_bytes is None or lf_payload_bytes is None
            else bool(required_lf_savings_bytes <= lf_payload_bytes)
        ),
        "evidence": evidence_ref or None,
        "target_consumers": [
            "nerv_long_training_campaign_plan",
            "nerv_rate_allocator_queue",
            "bit_allocator",
            "cathedral_autopilot",
        ],
        "command_argv": list(command_argv),
        "output_root": output_root.as_posix(),
        "local_mlx_long_training_allowed": False,
        "dispatch_allowed": False,
        "exact_or_full_video_cuda_allowed": False,
        **QUEUE_FALSE_AUTHORITY,
    }


def _campaign_candidate_token(
    campaign_row: Mapping[str, Any],
    suffix: str,
) -> str:
    row_id = str(campaign_row.get("row_id") or "snerv")
    candidate_id = str(campaign_row.get("candidate_id") or "candidate")
    return _stable_safe_token(f"{row_id}_{candidate_id}_{suffix}")


def _needs_lf_reroute(row: Mapping[str, Any]) -> bool:
    if row.get("family") != "snerv":
        return False
    blockers = {str(blocker) for blocker in row.get("blockers") or ()}
    if blockers.intersection(_OVER_CEILING_BLOCKERS):
        return True
    if row.get("hard_byte_ceiling_satisfied_for_long_training") is False:
        return True
    return bool(
        row.get("local_mlx_launch_command_ready") is False
        and row.get("candidate_nominal_under_ceiling") is False
    )


def _recode_admission_evidence_from_campaign_row(
    campaign_row: Mapping[str, Any],
    *,
    header_profiles: Mapping[str, Mapping[str, Any]],
    header_minimizations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    plan = campaign_row.get("snerv_lf_payload_recode_admission_plan")
    if not isinstance(plan, Mapping):
        return None
    selected = plan.get("selected_row")
    if not isinstance(selected, Mapping):
        return None
    if selected.get("local_planner_admitted") is not True:
        return None
    source_lf = _positive_int(selected.get("lf_source_bytes"))
    candidate_lf = _positive_int(selected.get("lf_candidate_bytes"))
    if source_lf is None and candidate_lf is None:
        return None
    source_report_path = selected.get("source_report_path")
    candidate_packet_sha256 = selected.get("candidate_packet_sha256")
    header_profile = _matching_header_profile(
        packet_sha256=candidate_packet_sha256,
        header_profiles=header_profiles,
    )
    header_minimization = _matching_header_minimization(
        packet_sha256=candidate_packet_sha256,
        header_minimizations=header_minimizations,
    )
    blockers = [
        str(blocker)
        for blocker in (
            list(plan.get("blockers") or ())
            + list(selected.get("local_admission_blockers") or ())
            + list(selected.get("promotion_blockers") or ())
        )
        if str(blocker)
    ]
    return {
        "schema": "snerv_lf_payload_byte_evidence.v1",
        "source_schema": plan.get("schema"),
        "source_path": source_report_path,
        "source_report_sha256": selected.get("source_report_sha256"),
        "candidate_id": campaign_row.get("candidate_id") or plan.get("candidate_id"),
        "mode": selected.get("mode"),
        "recommended_lossless_recode_mode": selected.get("mode"),
        "source_lf_payload_bytes": source_lf,
        "lf_payload_bytes": candidate_lf,
        "candidate_lf_payload_bytes": candidate_lf,
        "lf_payload_byte_delta": _int_or_none(selected.get("lf_payload_byte_delta")),
        "source_packet_bytes": _positive_int(selected.get("source_packet_bytes")),
        "packet_bytes": _positive_int(selected.get("candidate_packet_bytes")),
        "candidate_packet_bytes": _positive_int(selected.get("candidate_packet_bytes")),
        "candidate_packet_header_bytes": _positive_int(
            selected.get("candidate_packet_header_bytes")
            or _nested(header_profile or {}, ("header", "bytes"))
        ),
        "source_packet_path": selected.get("source_packet_path"),
        "candidate_packet_path": selected.get("candidate_packet_path"),
        "snar_header_grammar_profile_attached": header_profile is not None,
        "snar_header_grammar_profile": _header_profile_summary(header_profile),
        "snar_header_minimization_report_attached": header_minimization is not None,
        "snar_header_minimization_report": _header_minimization_summary(
            header_minimization
        ),
        "packet_byte_delta": _int_or_none(selected.get("packet_byte_delta")),
        "post_recode_over_waterline_bytes": _positive_int(
            selected.get("post_recode_over_waterline_bytes")
        ),
        "waterline_crossed_by_recode": selected.get("waterline_crossed_by_recode"),
        "lossless_lf_recode_already_admitted": True,
        "receiver_contract_satisfied": selected.get("receiver_contract_satisfied") is True,
        "receiver_proof_status": selected.get("receiver_frame_proof_status"),
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }


def _header_profile_index(
    profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for profile in profiles:
        if not isinstance(profile, Mapping):
            continue
        if str(profile.get("schema") or "") != "snerv_snar_header_grammar_profile.v1":
            continue
        for sha in (
            _nested(profile, ("packet", "sha256")),
            _nested(profile, ("input", "sha256")),
        ):
            text = str(sha or "").strip()
            if text:
                out.setdefault(text, profile)
    return out


def _header_minimization_index(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for report in reports:
        if not isinstance(report, Mapping):
            continue
        if str(report.get("schema") or "") != "snerv_snar_header_minimization.v1":
            continue
        for sha in (
            _nested(report, ("source_packet", "sha256")),
            _nested(report, ("candidate_packet", "sha256")),
        ):
            text = str(sha or "").strip()
            if text:
                out.setdefault(text, report)
    return out


def _matching_header_profile(
    *,
    packet_sha256: Any,
    header_profiles: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    sha = str(packet_sha256 or "").strip()
    if not sha:
        return None
    return header_profiles.get(sha)


def _matching_header_minimization(
    *,
    packet_sha256: Any,
    header_minimizations: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    sha = str(packet_sha256 or "").strip()
    if not sha:
        return None
    return header_minimizations.get(sha)


def _header_profile_summary(profile: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(profile, Mapping):
        return None
    return {
        "schema": profile.get("schema"),
        "source_path": _nested(profile, ("input", "path")),
        "packet_sha256": _nested(profile, ("packet", "sha256")),
        "packet_bytes": _positive_int(_nested(profile, ("packet", "bytes"))),
        "header_bytes": _positive_int(_nested(profile, ("header", "bytes"))),
        "metadata_json_bytes": _positive_int(
            _nested(profile, ("header", "metadata_json_bytes"))
        ),
        "section_total_bytes": _positive_int(
            _nested(profile, ("payload", "section_total_bytes"))
        ),
        "top_metadata_contributors": list(
            _nested(profile, ("header", "metadata_top_contributor_rows")) or ()
        )[:8],
        "hard_byte_ceiling_rows": list(profile.get("hard_byte_ceiling_rows") or ()),
        "next_actions": list(profile.get("next_actions") or ()),
        "blockers": list(profile.get("blockers") or ()),
        **FALSE_AUTHORITY,
    }


def _header_minimization_summary(report: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(report, Mapping):
        return None
    return {
        "schema": report.get("schema"),
        "source_packet_path": _nested(report, ("source_packet", "path")),
        "source_packet_sha256": _nested(report, ("source_packet", "sha256")),
        "source_packet_bytes": _positive_int(_nested(report, ("source_packet", "bytes"))),
        "candidate_packet_path": _nested(report, ("candidate_packet", "path")),
        "candidate_packet_sha256": _nested(report, ("candidate_packet", "sha256")),
        "candidate_packet_bytes": _positive_int(_nested(report, ("candidate_packet", "bytes"))),
        "candidate_archive_zip_path": _nested(report, ("candidate_archive_zip", "path")),
        "candidate_archive_zip_sha256": _nested(report, ("candidate_archive_zip", "sha256")),
        "candidate_archive_zip_bytes": _positive_int(
            _nested(report, ("candidate_archive_zip", "bytes"))
        ),
        "packet_byte_delta": _int_or_none(report.get("packet_byte_delta")),
        "header_byte_delta": _int_or_none(report.get("header_byte_delta")),
        "receiver_contract_satisfied": report.get("receiver_contract_satisfied") is True,
        "hard_byte_ceiling_rows": list(report.get("hard_byte_ceiling_rows") or ()),
        "blockers": list(report.get("blockers") or ()),
        "next_actions": list(report.get("next_actions") or ()),
        **FALSE_AUTHORITY,
    }


def _required_lf_savings_bytes(
    campaign_row: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> int | None:
    for value in (
        _nested(campaign_row, ("candidate", "_modelsize_feedback_archive_over_hard_byte_ceiling_bytes")),
        _nested(campaign_row, ("modelsize_byte_cap_preflight", "matching_calibrated_archive_overrun_bytes_max")),
        _positive_overage(evidence.get("archive_bytes"), campaign_row.get("hard_byte_ceiling")),
        _positive_overage(
            _nested(campaign_row, ("modelsize_byte_cap_preflight", "predicted_archive_bytes")),
            campaign_row.get("hard_byte_ceiling"),
        ),
        _positive_overage(campaign_row.get("candidate_nominal_total_payload_bytes"), campaign_row.get("hard_byte_ceiling")),
    ):
        amount = _positive_int(value)
        if amount is not None and amount > 0:
            return amount
    return None


def _positive_overage(value: Any, ceiling: Any) -> int | None:
    measured = _positive_int(value)
    limit = _positive_int(ceiling)
    if measured is None or limit is None:
        return None
    return max(int(measured) - int(limit), 0)


def _lf_evidence_rows(
    *,
    measured_lf_payload_sources: Sequence[Mapping[str, Any]],
    measured_lf_payload_paths: Sequence[str | Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in measured_lf_payload_sources:
        rows.extend(_evidence_from_payload(source, source_path=None))
    for raw_path in measured_lf_payload_paths:
        path = Path(raw_path).expanduser().resolve(strict=False)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.extend(_evidence_from_payload(payload, source_path=path.as_posix()))
    deduped: dict[tuple[str | None, str | None, str | None], dict[str, Any]] = {}
    for row in rows:
        key = (
            _none_or_str(row.get("candidate_id")),
            _none_or_str(row.get("source_path")),
            _none_or_str(row.get("packet_sha256")),
        )
        deduped.setdefault(key, row)
    return list(deduped.values())


def _evidence_from_payload(
    payload: Mapping[str, Any],
    *,
    source_path: str | None,
) -> list[dict[str, Any]]:
    schema = str(payload.get("schema") or "")
    if schema == "snerv_checkpoint_archive_export.v1":
        row = _checkpoint_export_evidence(payload, source_path=source_path)
        return [] if row is None else [row]
    if schema == "snerv_lf_payload_archive_recode.v1":
        return [_recode_evidence(payload, source_path=source_path)]
    if schema == "snerv_lf_payload_codec_sweep.v1":
        row = _codec_sweep_evidence(payload, source_path=source_path)
        return [] if row is None else [row]
    if schema == "snerv_lf_payload_recode_admission_plan.v1":
        selected = payload.get("selected_row")
        return [] if not isinstance(selected, Mapping) else [
            _admission_evidence(payload, selected, source_path=source_path)
        ]
    return []


def _checkpoint_export_evidence(
    payload: Mapping[str, Any],
    *,
    source_path: str | None,
) -> dict[str, Any] | None:
    section_bytes = payload.get("packet_section_bytes")
    if not isinstance(section_bytes, Mapping):
        return None
    lf_bytes = _positive_int(section_bytes.get("lf_payload"))
    if lf_bytes is None:
        return None
    candidate = payload.get("modelsize_candidate")
    candidate_id = payload.get("candidate_id")
    if candidate_id is None and isinstance(candidate, Mapping):
        candidate_id = candidate.get("candidate_id")
    return {
        "schema": "snerv_lf_payload_byte_evidence.v1",
        "source_schema": payload.get("schema"),
        "source_path": source_path or payload.get("report_path"),
        "candidate_id": candidate_id,
        "mode": payload.get("lf_payload_codec"),
        "recommended_lossless_recode_mode": "auto",
        "lf_payload_bytes": lf_bytes,
        "archive_bytes": _positive_int(payload.get("archive_bytes")),
        "packet_bytes": _positive_int(payload.get("packet_bytes")),
        "packet_path": payload.get("packet_path"),
        "packet_sha256": payload.get("packet_sha256"),
        "archive_path": payload.get("archive_path"),
        "archive_sha256": payload.get("archive_sha256"),
        "receiver_contract_satisfied": payload.get("receiver_contract_satisfied") is True,
        "receiver_proof_passed": payload.get("receiver_proof_passed") is True,
        "blockers": list(payload.get("blockers") or ()),
        **FALSE_AUTHORITY,
    }


def _recode_evidence(
    payload: Mapping[str, Any],
    *,
    source_path: str | None,
) -> dict[str, Any]:
    lf_payload = payload.get("lf_payload") if isinstance(payload.get("lf_payload"), Mapping) else {}
    source_packet = payload.get("source_packet") if isinstance(payload.get("source_packet"), Mapping) else {}
    candidate_packet = payload.get("candidate_packet") if isinstance(payload.get("candidate_packet"), Mapping) else {}
    return {
        "schema": "snerv_lf_payload_byte_evidence.v1",
        "source_schema": payload.get("schema"),
        "source_path": source_path or payload.get("report_path"),
        "candidate_id": payload.get("candidate_id"),
        "mode": payload.get("mode"),
        "recommended_lossless_recode_mode": payload.get("mode") or "auto",
        "lf_payload_bytes": _positive_int(lf_payload.get("source_bytes")),
        "candidate_lf_payload_bytes": _positive_int(lf_payload.get("candidate_bytes")),
        "lf_payload_byte_delta": _int_or_none(lf_payload.get("byte_delta")),
        "packet_bytes": _positive_int(source_packet.get("bytes")),
        "candidate_packet_bytes": _positive_int(candidate_packet.get("bytes")),
        "packet_byte_delta": _int_or_none(payload.get("packet_byte_delta")),
        "packet_path": source_packet.get("path"),
        "candidate_packet_path": candidate_packet.get("path"),
        "packet_sha256": source_packet.get("sha256"),
        "receiver_contract_satisfied": payload.get("receiver_contract_satisfied") is True,
        "blockers": list(payload.get("blockers") or ()),
        **FALSE_AUTHORITY,
    }


def _codec_sweep_evidence(
    payload: Mapping[str, Any],
    *,
    source_path: str | None,
) -> dict[str, Any] | None:
    source = payload.get("source") if isinstance(payload.get("source"), Mapping) else {}
    selected = payload.get("selected_rate_only_row")
    selected = selected if isinstance(selected, Mapping) else {}
    lf_bytes = _positive_int(selected.get("payload_bytes")) or _positive_int(
        payload.get("baseline_payload_bytes")
    )
    if lf_bytes is None:
        return None
    return {
        "schema": "snerv_lf_payload_byte_evidence.v1",
        "source_schema": payload.get("schema"),
        "source_path": source_path or payload.get("report_path") or payload.get("source_artifact_path"),
        "candidate_id": payload.get("candidate_id"),
        "mode": selected.get("mode") or payload.get("baseline_mode"),
        "recommended_lossless_recode_mode": selected.get("mode") or payload.get("baseline_mode") or "auto",
        "lf_payload_bytes": lf_bytes,
        "packet_bytes": _positive_int(source.get("bytes")),
        "packet_path": source.get("path"),
        "packet_sha256": source.get("sha256") or source.get("packet_sha256"),
        "receiver_contract_satisfied": True,
        "blockers": list(payload.get("blockers") or ()),
        **FALSE_AUTHORITY,
    }


def _admission_evidence(
    payload: Mapping[str, Any],
    selected: Mapping[str, Any],
    *,
    source_path: str | None,
) -> dict[str, Any]:
    return {
        "schema": "snerv_lf_payload_byte_evidence.v1",
        "source_schema": payload.get("schema"),
        "source_path": source_path,
        "candidate_id": selected.get("candidate_id") or payload.get("candidate_id"),
        "mode": selected.get("mode") or payload.get("selected_mode"),
        "recommended_lossless_recode_mode": selected.get("mode") or payload.get("selected_mode") or "auto",
        "lf_payload_bytes": _positive_int(selected.get("lf_source_bytes")),
        "candidate_lf_payload_bytes": _positive_int(selected.get("lf_candidate_bytes")),
        "lf_payload_byte_delta": _int_or_none(selected.get("lf_payload_byte_delta")),
        "packet_bytes": _positive_int(selected.get("source_packet_bytes")),
        "candidate_packet_bytes": _positive_int(selected.get("candidate_packet_bytes")),
        "packet_byte_delta": _int_or_none(selected.get("packet_byte_delta")),
        "packet_path": selected.get("source_packet_path"),
        "candidate_packet_path": selected.get("candidate_packet_path"),
        "packet_sha256": selected.get("source_packet_sha256"),
        "receiver_contract_satisfied": selected.get("receiver_contract_satisfied") is True,
        "blockers": list(payload.get("blockers") or ()),
        **FALSE_AUTHORITY,
    }


def _select_evidence_for_row(
    campaign_row: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not evidence_rows:
        return None
    candidate_id = str(campaign_row.get("candidate_id") or "")
    matching = [
        row
        for row in evidence_rows
        if str(row.get("candidate_id") or "") == candidate_id
    ]
    pool = matching or list(evidence_rows)
    return max(
        pool,
        key=lambda row: (
            int(row.get("lf_payload_bytes") or 0),
            int(row.get("archive_bytes") or 0),
            str(row.get("source_path") or ""),
        ),
    )


def _lossless_recode_mode(evidence: Mapping[str, Any]) -> str:
    mode = str(evidence.get("recommended_lossless_recode_mode") or "").strip()
    if mode:
        return mode
    mode = str(evidence.get("mode") or "").strip()
    return mode or "auto"


def _safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    return token.strip("_")[:160] or "snerv"


def _stable_safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    token = token.strip("_") or "snerv"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    if len(token) <= 147:
        return f"{token}_{digest}"
    return f"{token[:147].rstrip('_')}_{digest}"


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _nested(root: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = root
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "DEFAULT_QUEUE_ID",
    "SCHEMA",
    "build_snerv_lf_over_ceiling_reroute_queue",
]
