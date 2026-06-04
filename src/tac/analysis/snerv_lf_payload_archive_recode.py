# SPDX-License-Identifier: MIT
"""Lossless full-packet SNeRV LF payload recoding.

The LF codec sweep proves that alternative integer-stream grammars can encode
the same LF planes.  This module closes the next custody gap: swap only the
``lf_payload`` section inside a real SNAR1 packet, then prove the receiver sees
the same decoded LF state while every other section stays byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    CONTEST_BYTE_PRICE_SCORE,
)
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    build_nerv_byte_price_plan,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    DecodedSnervArchive,
    SnervArchiveError,
    encode_lf_quant_payload,
    inspect_lf_quant_payload_header,
    pack_snerv_archive,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SnervFrameCode,
    decode_frame,
    dequantize_lf,
)

SCHEMA = "snerv_lf_payload_archive_recode.v1"
ADMISSION_SCHEMA = "snerv_lf_payload_recode_admission_plan.v1"
AXIS_TAG = "[receiver-proof:false-authority]"
ADMISSION_AXIS_TAG = "[planning/control:false-authority]"
DEFAULT_FRAME_PROOF_MAX_OUTPUT_BYTES = 256 * 1024 * 1024
UNCHANGED_SECTIONS = ("metadata_payload", "decoder_payload", "step_map_packet")


class SnervLfPayloadArchiveRecodeError(ValueError):
    """Raised when a SNeRV archive LF recode is invalid."""


def build_snerv_lf_payload_archive_recode(
    source_packet: bytes,
    *,
    mode: str,
    source_packet_path: str | None = None,
    frame_proof_max_output_bytes: int = DEFAULT_FRAME_PROOF_MAX_OUTPUT_BYTES,
    force_frame_proof: bool = False,
) -> tuple[dict[str, Any], bytes]:
    """Return a losslessly recoded SNAR1 packet plus proof report.

    The candidate packet is not a score or promotion surface.  It is a
    receiver-custody artifact that proves the selected LF codec can live inside
    the full packet grammar without mutating the decoded signal.
    """

    source_blob = bytes(source_packet)
    if not source_blob:
        raise SnervLfPayloadArchiveRecodeError("source_packet must be non-empty")
    if not str(mode).strip():
        raise SnervLfPayloadArchiveRecodeError("mode must be non-empty")

    source = unpack_snerv_archive(source_blob)
    source_lf_planes = source.decode_lf_quant_planes()
    candidate_lf_payload = encode_lf_quant_payload(source_lf_planes, codec=mode)
    candidate_packet = pack_snerv_archive(
        metadata_payload=source.sections["metadata_payload"],
        lf_payload=candidate_lf_payload,
        decoder_payload=source.sections["decoder_payload"],
        step_map_packet=source.sections["step_map_packet"],
        metadata=source.metadata,
    )
    candidate = unpack_snerv_archive(candidate_packet.packet)
    candidate_lf_planes = candidate.decode_lf_quant_planes()

    lf_exact, lf_hash_rows = _lf_plane_equality(source_lf_planes, candidate_lf_planes)
    unchanged = {
        section: (
            source.sections[section] == candidate.sections[section]
            and _sha256(source.sections[section])
            == _sha256(candidate.sections[section])
        )
        for section in UNCHANGED_SECTIONS
    }
    frame_proof = _streaming_frame_equality_proof(
        source,
        candidate,
        max_output_bytes=int(frame_proof_max_output_bytes),
        force=bool(force_frame_proof),
    )
    frame_status = str(frame_proof["status"])
    receiver_contract = bool(
        lf_exact
        and all(unchanged.values())
        and frame_status not in {"failed", "error"}
    )
    source_lf_bytes = len(source.sections["lf_payload"])
    candidate_lf_bytes = len(candidate.sections["lf_payload"])
    packet_byte_delta = int(candidate_packet.total_bytes - len(source_blob))
    lf_byte_delta = int(candidate_lf_bytes - source_lf_bytes)
    blockers = _blockers(
        receiver_contract=receiver_contract,
        frame_status=frame_status,
        unchanged=unchanged,
    )
    report = {
        "schema": SCHEMA,
        "axis_tag": AXIS_TAG,
        "family": "snerv",
        "operation": "lossless_lf_payload_recode_inside_snar1_packet",
        "mode": str(mode),
        "source_packet": {
            "path": source_packet_path,
            "bytes": len(source_blob),
            "sha256": _sha256(source_blob),
            "decoded_packet_sha256": source.packet_sha256,
        },
        "candidate_packet": {
            "bytes": candidate_packet.total_bytes,
            "sha256": _sha256(candidate_packet.packet),
            "decoded_packet_sha256": candidate.packet_sha256,
            "header_bytes": candidate_packet.header_bytes,
        },
        "packet_byte_delta": packet_byte_delta,
        "packet_rate_score_delta": float(packet_byte_delta * CONTEST_BYTE_PRICE_SCORE),
        "lf_payload": {
            "source_bytes": source_lf_bytes,
            "candidate_bytes": candidate_lf_bytes,
            "byte_delta": lf_byte_delta,
            "rate_score_delta": float(lf_byte_delta * CONTEST_BYTE_PRICE_SCORE),
            "source_sha256": _sha256(source.sections["lf_payload"]),
            "candidate_sha256": _sha256(candidate.sections["lf_payload"]),
            "source_header": inspect_lf_quant_payload_header(
                source.sections["lf_payload"]
            ),
            "candidate_header": inspect_lf_quant_payload_header(
                candidate.sections["lf_payload"]
            ),
        },
        "section_bytes": {
            "source": {k: len(v) for k, v in source.sections.items()},
            "candidate": dict(candidate_packet.section_bytes),
        },
        "section_sha256": {
            "source": {k: _sha256(v) for k, v in source.sections.items()},
            "candidate": dict(candidate_packet.section_sha256),
        },
        "unchanged_sections_exact": unchanged,
        "lf_plane_count": len(source_lf_planes),
        "lf_planes_exact_equal": lf_exact,
        "lf_plane_hash_rows": lf_hash_rows,
        "receiver_frame_equality_proof": frame_proof,
        "receiver_contract_satisfied": receiver_contract,
        "runtime_consumption_proof_ready": receiver_contract,
        "score_claim": False,
        "frontier_score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }
    return report, candidate_packet.packet


def render_snerv_lf_payload_archive_recode_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a compact operator-readable recode report."""

    source = report.get("source_packet", {}) if isinstance(report, Mapping) else {}
    candidate = report.get("candidate_packet", {}) if isinstance(report, Mapping) else {}
    lf = report.get("lf_payload", {}) if isinstance(report, Mapping) else {}
    frame = (
        report.get("receiver_frame_equality_proof", {})
        if isinstance(report, Mapping)
        else {}
    )
    blockers = [str(v) for v in report.get("blockers", [])] if isinstance(report, Mapping) else []
    lines = [
        "# SNeRV LF Payload Archive Recode",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- axis: `{report.get('axis_tag')}`",
        f"- mode: `{report.get('mode')}`",
        f"- source packet: `{source.get('bytes')}` bytes `{source.get('sha256')}`",
        f"- candidate packet: `{candidate.get('bytes')}` bytes `{candidate.get('sha256')}`",
        f"- packet byte delta: `{report.get('packet_byte_delta')}`",
        f"- LF byte delta: `{lf.get('byte_delta')}`",
        f"- LF planes exact: `{report.get('lf_planes_exact_equal')}`",
        f"- unchanged sections exact: `{report.get('unchanged_sections_exact')}`",
        f"- receiver frame proof: `{frame.get('status')}`",
        f"- receiver contract satisfied: `{report.get('receiver_contract_satisfied')}`",
        "",
        "## Blockers",
    ]
    lines.extend(f"- `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def build_snerv_lf_payload_recode_admission_plan(
    recode_reports: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    hard_byte_ceiling: int | None = None,
    candidate_id: str | None = None,
    full_video_coverage: bool = False,
) -> dict[str, Any]:
    """Consume receiver-backed LF recode reports as a byte-price plan.

    The LF recoder proves that a codec can be placed inside a real SNAR1 packet.
    This planner makes that proof actionable without granting contest authority:
    it prices the packet delta using the official contest byte price, orders
    receiver-proven modes by waterline pressure, and emits the next local action.
    """

    reports = _recode_report_sequence(recode_reports)
    if not reports:
        raise SnervLfPayloadArchiveRecodeError(
            "recode_reports must contain at least one report"
        )
    ceiling = None if hard_byte_ceiling is None else int(hard_byte_ceiling)
    rows = [
        _admission_row(
            report,
            source_index=index,
            hard_byte_ceiling=ceiling,
            candidate_id=candidate_id,
        )
        for index, report in enumerate(reports)
    ]
    section_value_rows = [
        _admission_section_value_row(row, full_video_coverage=bool(full_video_coverage))
        for row in rows
    ]
    byte_price_plan = build_nerv_byte_price_plan(
        {
            "schema": ADMISSION_SCHEMA,
            "family": "snerv",
            "candidate_id": candidate_id,
            "axis_tag": ADMISSION_AXIS_TAG,
            "section_value_rows": section_value_rows,
            "blockers": [
                "snerv_lf_recode_admission_plan_false_authority",
                "paired_contest_cpu_cuda_auth_eval_missing",
            ],
            **FALSE_AUTHORITY,
        },
        candidate_id=candidate_id,
        baseline_id="source_snar1_packet",
    )
    selected = _select_admission_row(rows)
    selected_mode = None if selected is None else selected["mode"]
    selected_over = (
        None if selected is None else selected["post_recode_over_waterline_bytes"]
    )
    local_blockers = _ordered_unique(
        [
            blocker
            for row in rows
            for blocker in row.get("local_admission_blockers", ())
        ]
    )
    blockers = _ordered_unique(
        [
            "snerv_lf_recode_admission_plan_false_authority",
            *(
                []
                if any(row["local_planner_admitted"] for row in rows)
                else ["snerv_lf_recode_no_receiver_proven_byte_saving_mode"]
            ),
            *(
                []
                if selected is None
                or selected_over is None
                or int(selected_over) <= 0
                else ["snerv_lf_recode_selected_mode_still_over_byte_waterline"]
            ),
            "not_packaged_as_contest_archive_zip",
            "full_video_scorer_replay_missing",
            "paired_contest_cpu_cuda_auth_eval_missing",
            *local_blockers,
        ]
    )
    return {
        "schema": ADMISSION_SCHEMA,
        "axis_tag": ADMISSION_AXIS_TAG,
        "family": "snerv",
        "candidate_id": candidate_id,
        "hard_byte_ceiling": ceiling,
        "rate_score_per_byte": float(CONTEST_BYTE_PRICE_SCORE),
        "input_report_count": len(reports),
        "full_video_coverage": bool(full_video_coverage),
        "admission_rows": rows,
        "section_value_rows": section_value_rows,
        "byte_price_plan": byte_price_plan,
        "selected_mode": selected_mode,
        "selected_row": selected,
        "local_planner_admitted": selected is not None,
        "waterline_satisfied_after_selected_recode": bool(
            selected is not None
            and (selected_over is None or int(selected_over) <= 0)
        ),
        "verdict": _admission_verdict(selected),
        "next_actions": _admission_next_actions(selected),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def render_snerv_lf_payload_recode_admission_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a compact operator-readable LF recode admission report."""

    lines = [
        "# SNeRV LF Recode Admission",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Selected mode: `{report.get('selected_mode')}`",
        f"Verdict: `{report.get('verdict')}`",
        f"Local planner admitted: `{report.get('local_planner_admitted')}`",
        "",
        "| mode | packet delta | rate delta | over waterline | decision |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report.get("admission_rows", ()):
        if not isinstance(row, Mapping):
            continue
        over = row.get("post_recode_over_waterline_bytes")
        lines.append(
            "| {mode} | {delta} | {rate:.9f} | {over} | {decision} |".format(
                mode=row.get("mode"),
                delta=int(row.get("packet_byte_delta") or 0),
                rate=float(row.get("packet_rate_score_delta") or 0.0),
                over="n/a" if over is None else int(over),
                decision=row.get("admission_decision"),
            )
        )
    lines.extend(["", "## Blockers", ""])
    blockers = list(report.get("blockers") or ())
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _recode_report_sequence(
    reports: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(reports, Mapping):
        if reports.get("schema") == ADMISSION_SCHEMA:
            rows = reports.get("source_reports") or reports.get("reports")
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                return [row for row in rows if isinstance(row, Mapping)]
        return [reports]
    return [row for row in reports if isinstance(row, Mapping)]


def _admission_row(
    report: Mapping[str, Any],
    *,
    source_index: int,
    hard_byte_ceiling: int | None,
    candidate_id: str | None,
) -> dict[str, Any]:
    if report.get("schema") != SCHEMA:
        return _invalid_admission_row(
            report,
            source_index=source_index,
            blocker=f"snerv_lf_recode_source_schema_invalid:{report.get('schema')}",
            hard_byte_ceiling=hard_byte_ceiling,
            candidate_id=candidate_id,
        )
    target_candidate_id = _candidate_id_or_none(candidate_id)
    source_report_candidate_id = _candidate_id_or_none(report.get("candidate_id"))
    source_packet = _mapping(report.get("source_packet"))
    candidate_packet = _mapping(report.get("candidate_packet"))
    lf_payload = _mapping(report.get("lf_payload"))
    source_packet_bytes = _positive_int(source_packet.get("bytes"))
    candidate_packet_bytes = _positive_int(candidate_packet.get("bytes"))
    candidate_packet_header_bytes = _positive_int(candidate_packet.get("header_bytes"))
    packet_delta = _int_or_none(report.get("packet_byte_delta"))
    if packet_delta is None and source_packet_bytes is not None and candidate_packet_bytes is not None:
        packet_delta = int(candidate_packet_bytes - source_packet_bytes)
    lf_source_bytes = _positive_int(lf_payload.get("source_bytes"))
    lf_candidate_bytes = _positive_int(lf_payload.get("candidate_bytes"))
    lf_delta = _int_or_none(lf_payload.get("byte_delta"))
    if lf_delta is None and lf_source_bytes is not None and lf_candidate_bytes is not None:
        lf_delta = int(lf_candidate_bytes - lf_source_bytes)
    source_headroom = (
        None
        if hard_byte_ceiling is None or source_packet_bytes is None
        else int(hard_byte_ceiling - source_packet_bytes)
    )
    candidate_headroom = (
        None
        if hard_byte_ceiling is None or candidate_packet_bytes is None
        else int(hard_byte_ceiling - candidate_packet_bytes)
    )
    over_waterline = (
        None if candidate_headroom is None else max(-int(candidate_headroom), 0)
    )
    receiver_contract = bool(report.get("receiver_contract_satisfied") is True)
    local_blockers = []
    if not receiver_contract:
        local_blockers.append("snerv_lf_recode_receiver_contract_not_satisfied")
    if (
        target_candidate_id is not None
        and source_report_candidate_id is not None
        and source_report_candidate_id != target_candidate_id
    ):
        local_blockers.append("snerv_lf_recode_candidate_id_mismatch")
    if source_packet_bytes is None:
        local_blockers.append("snerv_lf_recode_source_packet_bytes_missing")
    if candidate_packet_bytes is None:
        local_blockers.append("snerv_lf_recode_candidate_packet_bytes_missing")
    if packet_delta is None:
        local_blockers.append("snerv_lf_recode_packet_byte_delta_missing")
    elif packet_delta >= 0:
        local_blockers.append("snerv_lf_recode_not_byte_saving")
    mode = str(report.get("mode") or "unknown")
    local_admitted = not local_blockers
    waterline_crossed = bool(
        local_admitted
        and source_headroom is not None
        and candidate_headroom is not None
        and int(source_headroom) < 0
        and int(candidate_headroom) >= 0
    )
    if not local_admitted:
        decision = "block_lf_recode_admission"
        ablation = "blocked_until_receiver_contract_and_byte_savings"
    elif waterline_crossed:
        decision = "admit_lossless_lf_recode_crosses_byte_waterline"
        ablation = "no_lf_ablation_required_after_recode_waterline"
    elif over_waterline is not None and int(over_waterline) > 0:
        decision = "admit_lossless_lf_recode_but_route_remaining_lf_ablation"
        ablation = "ablate_or_receiver_generate_remaining_lf_payload_bytes"
    else:
        decision = "admit_lossless_lf_recode_for_local_archive_rebuild"
        ablation = "no_lf_ablation_required_by_this_waterline"
    packet_rate_delta = (
        None
        if packet_delta is None
        else float(int(packet_delta) * CONTEST_BYTE_PRICE_SCORE)
    )
    lf_rate_delta = (
        None if lf_delta is None else float(int(lf_delta) * CONTEST_BYTE_PRICE_SCORE)
    )
    return {
        "schema": "snerv_lf_payload_recode_admission_row.v1",
        "source_index": int(source_index),
        "row_id": f"snerv_lf_recode_{mode}",
        "family": "snerv",
        "candidate_id": target_candidate_id or source_report_candidate_id,
        "source_report_candidate_id": source_report_candidate_id,
        "source_report_schema": str(report.get("schema") or ""),
        "source_report_path": _source_report_path(report),
        "source_report_sha256": _source_report_sha256(report),
        "source_report_producer": _source_report_producer(report),
        "mode": mode,
        "source_packet_bytes": source_packet_bytes,
        "candidate_packet_bytes": candidate_packet_bytes,
        "candidate_packet_header_bytes": candidate_packet_header_bytes,
        "packet_byte_delta": packet_delta,
        "packet_rate_score_delta": packet_rate_delta,
        "lf_source_bytes": lf_source_bytes,
        "lf_candidate_bytes": lf_candidate_bytes,
        "lf_payload_byte_delta": lf_delta,
        "lf_payload_rate_score_delta": lf_rate_delta,
        "hard_byte_ceiling": hard_byte_ceiling,
        "source_packet_headroom_bytes": source_headroom,
        "candidate_packet_headroom_bytes": candidate_headroom,
        "post_recode_over_waterline_bytes": over_waterline,
        "post_recode_over_waterline_rate_score": (
            None
            if over_waterline is None
            else float(int(over_waterline) * CONTEST_BYTE_PRICE_SCORE)
        ),
        "waterfill_credit_bytes": (
            0 if packet_delta is None else max(-int(packet_delta), 0)
        ),
        "waterfill_credit_rate_score": (
            None
            if packet_delta is None
            else float(max(-int(packet_delta), 0) * CONTEST_BYTE_PRICE_SCORE)
        ),
        "waterline_crossed_by_recode": waterline_crossed,
        "receiver_contract_satisfied": receiver_contract,
        "receiver_frame_proof_status": _receiver_frame_status(report),
        "local_planner_admitted": local_admitted,
        "admission_decision": decision,
        "ablation_decision": ablation,
        "local_admission_blockers": _ordered_unique(local_blockers),
        "promotion_blockers": [
            "not_packaged_as_contest_archive_zip",
            "full_video_scorer_replay_missing",
            "paired_contest_cpu_cuda_auth_eval_missing",
        ],
        "source_packet_sha256": source_packet.get("sha256"),
        "candidate_packet_sha256": candidate_packet.get("sha256"),
        "source_report_blockers": [
            str(blocker) for blocker in report.get("blockers") or ()
        ],
        **FALSE_AUTHORITY,
    }


def _invalid_admission_row(
    report: Mapping[str, Any],
    *,
    source_index: int,
    blocker: str,
    hard_byte_ceiling: int | None,
    candidate_id: str | None,
) -> dict[str, Any]:
    return {
        "schema": "snerv_lf_payload_recode_admission_row.v1",
        "source_index": int(source_index),
        "row_id": f"snerv_lf_recode_invalid_{source_index}",
        "family": "snerv",
        "candidate_id": _candidate_id_or_none(candidate_id)
        or _candidate_id_or_none(report.get("candidate_id")),
        "source_report_candidate_id": _candidate_id_or_none(report.get("candidate_id")),
        "source_report_schema": str(report.get("schema") or ""),
        "source_report_path": _source_report_path(report),
        "source_report_sha256": _source_report_sha256(report),
        "source_report_producer": _source_report_producer(report),
        "mode": str(report.get("mode") or "unknown"),
        "source_packet_bytes": None,
        "candidate_packet_bytes": None,
        "packet_byte_delta": None,
        "packet_rate_score_delta": None,
        "lf_payload_byte_delta": None,
        "hard_byte_ceiling": hard_byte_ceiling,
        "post_recode_over_waterline_bytes": None,
        "receiver_contract_satisfied": False,
        "local_planner_admitted": False,
        "admission_decision": "block_lf_recode_admission",
        "ablation_decision": "blocked_until_valid_recode_report",
        "local_admission_blockers": [blocker],
        "promotion_blockers": [
            "not_packaged_as_contest_archive_zip",
            "full_video_scorer_replay_missing",
            "paired_contest_cpu_cuda_auth_eval_missing",
        ],
        **FALSE_AUTHORITY,
    }


def _admission_section_value_row(
    row: Mapping[str, Any],
    *,
    full_video_coverage: bool,
) -> dict[str, Any]:
    blockers = [
        *[str(v) for v in row.get("local_admission_blockers") or ()],
        "snerv_lf_recode_exact_axis_replay_missing",
    ]
    if not full_video_coverage:
        blockers.append("snerv_lf_recode_full_video_coverage_missing")
    packet_delta = _int_or_none(row.get("packet_byte_delta"))
    return {
        "row_id": row.get("row_id"),
        "section_id": "snerv_lf_payload",
        "family": "snerv",
        "candidate_id": row.get("candidate_id"),
        "scope": "receiver_proven_lf_payload_recode_inside_snar1_packet",
        "row_kind": "existing_section_cut",
        "candidate_mode": row.get("mode"),
        "section_bytes": row.get("lf_source_bytes"),
        "payload_bytes": row.get("lf_candidate_bytes"),
        "byte_delta": packet_delta,
        "delta_nonrate_score": 0.0 if row.get("local_planner_admitted") else None,
        "delta_rate_score": row.get("packet_rate_score_delta"),
        "archive_sha256": row.get("candidate_packet_sha256"),
        "baseline_packet_sha256": row.get("source_packet_sha256"),
        "source_report_path": row.get("source_report_path"),
        "source_report_sha256": row.get("source_report_sha256"),
        "source_report_producer": row.get("source_report_producer"),
        "axis_tag": ADMISSION_AXIS_TAG,
        "receiver_proof_status": (
            "runtime_consumption_proof_ready"
            if row.get("receiver_contract_satisfied") is True
            else "missing"
        ),
        "full_video_coverage": bool(full_video_coverage),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _select_admission_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    admitted = [dict(row) for row in rows if row.get("local_planner_admitted") is True]
    if not admitted:
        return None

    def key(row: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
        over = row.get("post_recode_over_waterline_bytes")
        over_value = 0 if over is None else int(over)
        packet_bytes = int(row.get("candidate_packet_bytes") or 2**63 - 1)
        credit = int(row.get("waterfill_credit_bytes") or 0)
        return (
            0 if over is None or over_value <= 0 else 1,
            over_value,
            packet_bytes,
            -credit,
            str(row.get("mode") or ""),
        )

    return sorted(admitted, key=key)[0]


def _admission_verdict(selected: Mapping[str, Any] | None) -> str:
    if selected is None:
        return "NO_ADMISSIBLE_LF_RECODE__RERUN_RECEIVER_PROVEN_CODEC_SWEEP"
    decision = str(selected.get("admission_decision") or "")
    if "crosses_byte_waterline" in decision:
        return "ADMIT_LF_RECODE__CROSSES_BYTE_WATERLINE__FALSE_AUTHORITY"
    if _post_recode_overrun_is_header_dominated(selected):
        return "ADMIT_LF_RECODE__POST_RECODE_PACKET_HEADER_GRAMMAR_DOMINATES"
    if "remaining_lf_ablation" in decision:
        return "ADMIT_LF_RECODE__REMAINING_LF_BYTES_REQUIRE_ABLATION"
    return "ADMIT_LF_RECODE__BYTE_PRICED_RECEIVER_PROVEN__FALSE_AUTHORITY"


def _admission_next_actions(selected: Mapping[str, Any] | None) -> list[str]:
    if selected is None:
        return [
            "rerun_snerv_lf_payload_archive_recode_on_receiver_proven_snar_packet",
            "refuse_long_training_admission_from_orphaned_lf_codec_profiler_rows",
        ]
    mode = str(selected.get("mode") or "unknown")
    if int(selected.get("post_recode_over_waterline_bytes") or 0) > 0:
        if _post_recode_overrun_is_header_dominated(selected):
            return [
                f"preserve_snerv_lossless_lf_payload_codec:{mode}",
                "attack_snerv_snar_packet_header_grammar_or_packaging_overhead",
                "rerun_receiver_proof_and_byte_price_admission_after_packet_header_rewrite",
            ]
        return [
            f"route_snerv_native_export_lf_payload_codec:{mode}",
            "attack_remaining_lf_payload_with_temporal_delta_generation_or_coarser_lf_ablation",
            "rerun_receiver_proof_and_byte_price_admission_after_ablation",
        ]
    return [
        f"route_snerv_native_export_lf_payload_codec:{mode}",
        "materialize_recode_candidate_archive_zip_then_receiver_proof",
        "run_full600_local_cpu_replay_before_any_exact_axis_dispatch",
    ]


def _post_recode_overrun_is_header_dominated(selected: Mapping[str, Any]) -> bool:
    over = _positive_int(selected.get("post_recode_over_waterline_bytes"))
    header = _positive_int(selected.get("candidate_packet_header_bytes"))
    return bool(over is not None and header is not None and header >= over)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _receiver_frame_status(report: Mapping[str, Any]) -> str:
    proof = report.get("receiver_frame_equality_proof")
    if isinstance(proof, Mapping):
        status = str(proof.get("status") or "")
        if status:
            return status
    return "missing"


def _candidate_id_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _source_report_path(report: Mapping[str, Any]) -> str | None:
    for key in (
        "source_report_path",
        "report_path",
        "_source_report_path",
        "_report_path",
        "path",
    ):
        value = report.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _source_report_producer(report: Mapping[str, Any]) -> str:
    for key in ("producer", "tool", "generated_by", "operation"):
        value = report.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "snerv_lf_payload_archive_recode"


def _source_report_sha256(report: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _jsonable_mapping(report),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jsonable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(str(k) for k in value):
        out[key] = _jsonable_value(value.get(key))
    return out


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return {
            "ndarray_shape": [int(v) for v in value.shape],
            "ndarray_dtype": str(value.dtype),
            "ndarray_sha256": _sha256(np.ascontiguousarray(value).tobytes()),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _lf_plane_equality(
    source: list[np.ndarray],
    candidate: list[np.ndarray],
) -> tuple[bool, list[dict[str, Any]]]:
    if len(source) != len(candidate):
        return False, [
            {
                "plane_index": -1,
                "source_count": len(source),
                "candidate_count": len(candidate),
                "exact_equal": False,
            }
        ]
    rows = []
    all_equal = True
    for idx, (a, b) in enumerate(zip(source, candidate, strict=True)):
        a_arr = np.asarray(a, dtype="<i8")
        b_arr = np.asarray(b, dtype="<i8")
        exact = bool(a_arr.shape == b_arr.shape and np.array_equal(a_arr, b_arr))
        all_equal = all_equal and exact
        rows.append(
            {
                "plane_index": idx,
                "shape": [int(v) for v in a_arr.shape],
                "source_sha256": _sha256(a_arr.tobytes()),
                "candidate_sha256": _sha256(b_arr.tobytes()),
                "exact_equal": exact,
            }
        )
    return all_equal, rows


def _streaming_frame_equality_proof(
    source: DecodedSnervArchive,
    candidate: DecodedSnervArchive,
    *,
    max_output_bytes: int,
    force: bool,
) -> dict[str, Any]:
    try:
        estimate = _estimated_receiver_output_bytes(source)
        if not force and estimate is not None and estimate > int(max_output_bytes):
            return {
                "status": "skipped_by_output_byte_guard",
                "estimated_output_bytes": estimate,
                "max_output_bytes": int(max_output_bytes),
                "exactness_basis": (
                    "lf_planes_exact_and_metadata_decoder_step_map_sections_unchanged"
                ),
            }
        source_hash, candidate_hash, compared, max_abs = _stream_frame_hash_compare(
            source,
            candidate,
        )
        exact = source_hash == candidate_hash and max_abs == 0.0
        return {
            "status": "proven_exact" if exact else "failed",
            "estimated_output_bytes": estimate,
            "max_output_bytes": int(max_output_bytes),
            "compared_plane_count": compared,
            "source_frame_sha256": source_hash,
            "candidate_frame_sha256": candidate_hash,
            "max_abs_diff": max_abs,
            "exact_equal": exact,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "estimated_output_bytes": _estimated_receiver_output_bytes(source),
            "max_output_bytes": int(max_output_bytes),
        }


def _stream_frame_hash_compare(
    source: DecodedSnervArchive,
    candidate: DecodedSnervArchive,
) -> tuple[str, str, int, float]:
    source_parts = _receiver_decode_parts(source)
    candidate_parts = _receiver_decode_parts(candidate)
    if source_parts["orig_hw"] != candidate_parts["orig_hw"]:
        raise SnervArchiveError("candidate orig_hw differs from source")
    if source_parts["levels"] != candidate_parts["levels"]:
        raise SnervArchiveError("candidate levels differ from source")
    if source_parts["wavelet"] != candidate_parts["wavelet"]:
        raise SnervArchiveError("candidate wavelet differs from source")

    source_hash = hashlib.sha256()
    candidate_hash = hashlib.sha256()
    compared = 0
    max_abs = 0.0
    for idx, (source_code, candidate_code) in enumerate(
        zip(source_parts["codes"], candidate_parts["codes"], strict=True)
    ):
        source_lf_sequence = None
        candidate_lf_sequence = None
        sequence_index = None
        if source_parts["temporal_group_count"] > 1:
            group = idx % source_parts["temporal_group_count"]
            source_lf_sequence = source_parts["decoded_lfs"][group :: source_parts["temporal_group_count"]]
            candidate_lf_sequence = candidate_parts["decoded_lfs"][group :: candidate_parts["temporal_group_count"]]
            sequence_index = idx // source_parts["temporal_group_count"]
        source_frame = np.clip(
            decode_frame(
                source_code,
                source_parts["decoder"],
                lf_sequence=source_lf_sequence,
                sequence_index=sequence_index,
            ),
            0.0,
            255.0,
        ).astype("<f4", copy=False)
        candidate_frame = np.clip(
            decode_frame(
                candidate_code,
                candidate_parts["decoder"],
                lf_sequence=candidate_lf_sequence,
                sequence_index=sequence_index,
            ),
            0.0,
            255.0,
        ).astype("<f4", copy=False)
        if source_frame.shape != candidate_frame.shape:
            raise SnervArchiveError(
                f"candidate frame {idx} shape {candidate_frame.shape} != source {source_frame.shape}"
            )
        diff = float(np.max(np.abs(source_frame - candidate_frame)))
        max_abs = max(max_abs, diff)
        source_hash.update(source_frame.tobytes())
        candidate_hash.update(candidate_frame.tobytes())
        compared += 1
    return source_hash.hexdigest(), candidate_hash.hexdigest(), compared, max_abs


def _receiver_decode_parts(decoded: DecodedSnervArchive) -> dict[str, Any]:
    metadata = decoded.metadata
    levels = _metadata_int(metadata, "levels", minimum=1)
    wavelet = _metadata_str(metadata, "wavelet")
    orig_hw = _metadata_hw(metadata)
    lf_planes = decoded.decode_lf_quant_planes()
    zeros = decoded.decode_lf_zero_points()
    step_maps = decoded.decode_step_maps()
    decoder = decoded.decode_decoder()
    if not (len(lf_planes) == len(zeros) == len(step_maps)):
        raise SnervArchiveError("receiver replay state count mismatch")
    codes: list[SnervFrameCode] = []
    decoded_lfs: list[np.ndarray] = []
    for idx, (q, zero, steps) in enumerate(zip(lf_planes, zeros, step_maps, strict=True)):
        if q.shape != steps.shape:
            raise SnervArchiveError(
                f"receiver replay plane {idx} LF shape {q.shape} != step shape {steps.shape}"
            )
        code = SnervFrameCode(
            lf_quant=q,
            lf_scale=1.0,
            lf_zero=float(zero),
            lf_shape=tuple(int(v) for v in q.shape),
            levels=levels,
            wavelet=wavelet,
            orig_hw=orig_hw,
            per_element_steps=steps,
        )
        codes.append(code)
        decoded_lfs.append(dequantize_lf(q, 1.0, float(zero), per_element_steps=steps))

    temporal_group_count = 1
    if int(decoder.model_size.temporal_context) > 0:
        temporal_group_count = _metadata_int(metadata, "channels", default=1, minimum=1)
    return {
        "metadata": metadata,
        "levels": levels,
        "wavelet": wavelet,
        "orig_hw": orig_hw,
        "decoder": decoder,
        "codes": codes,
        "decoded_lfs": decoded_lfs,
        "temporal_group_count": temporal_group_count,
    }


def _estimated_receiver_output_bytes(decoded: DecodedSnervArchive) -> int | None:
    try:
        h, w = _metadata_hw(decoded.metadata)
        n_pairs = _metadata_int(decoded.metadata, "n_pairs", default=0, minimum=0)
        frames_per_pair = _metadata_int(
            decoded.metadata,
            "frames_per_pair",
            default=0,
            minimum=0,
        )
        channels = _metadata_int(decoded.metadata, "channels", default=0, minimum=0)
        if n_pairs and frames_per_pair and channels:
            plane_count = n_pairs * frames_per_pair * channels
        else:
            plane_count = len(decoded.decode_lf_quant_planes())
        return int(plane_count * h * w * np.dtype("<f4").itemsize)
    except Exception:
        return None


def _blockers(
    *,
    receiver_contract: bool,
    frame_status: str,
    unchanged: Mapping[str, bool],
) -> list[str]:
    blockers = []
    if not receiver_contract:
        blockers.append("snerv_lf_payload_archive_recode_receiver_contract_failed")
    if not all(unchanged.values()):
        blockers.append("snerv_lf_payload_archive_recode_mutated_non_lf_section")
    if frame_status == "skipped_by_output_byte_guard":
        blockers.append("receiver_frame_streaming_proof_skipped_by_output_byte_guard")
    if frame_status in {"failed", "error"}:
        blockers.append("receiver_frame_equality_proof_failed")
    blockers.extend(
        [
            "not_packaged_as_contest_archive_zip",
            "full_video_scorer_replay_missing",
            "paired_contest_cpu_cuda_auth_eval_missing",
        ]
    )
    return _ordered_unique(blockers)


def _metadata_int(
    metadata: Mapping[str, Any],
    key: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
) -> int:
    if key not in metadata:
        if default is None:
            raise SnervArchiveError(f"SNeRV archive metadata missing {key!r}")
        value = int(default)
    else:
        value = int(metadata[key])
    if minimum is not None and value < minimum:
        raise SnervArchiveError(f"SNeRV archive metadata {key!r} below {minimum}")
    return value


def _metadata_str(metadata: Mapping[str, Any], key: str) -> str:
    value = str(metadata.get(key) or "")
    if not value:
        raise SnervArchiveError(f"SNeRV archive metadata missing {key!r}")
    return value


def _metadata_hw(metadata: Mapping[str, Any]) -> tuple[int, int]:
    value = metadata.get("orig_hw") or metadata.get("carrier_hw")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SnervArchiveError("SNeRV archive metadata missing orig_hw")
    return (int(value[0]), int(value[1]))


def _ordered_unique(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(bytes(blob)).hexdigest()


__all__ = [
    "ADMISSION_AXIS_TAG",
    "ADMISSION_SCHEMA",
    "AXIS_TAG",
    "DEFAULT_FRAME_PROOF_MAX_OUTPUT_BYTES",
    "SCHEMA",
    "SnervLfPayloadArchiveRecodeError",
    "build_snerv_lf_payload_archive_recode",
    "build_snerv_lf_payload_recode_admission_plan",
    "render_snerv_lf_payload_archive_recode_markdown",
    "render_snerv_lf_payload_recode_admission_markdown",
]
