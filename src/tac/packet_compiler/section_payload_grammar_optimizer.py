# SPDX-License-Identifier: MIT
"""Generic section-payload grammar optimizer.

This is the substrate-agnostic sibling of the PR101 per-tensor grammar solver:
given named byte sections from any archive/export grammar, measure the shared
coder portfolio, select the smallest exact codec per section, and emit a
fail-closed planner surface.  It deliberately reuses the PR101 codec backend so
future substrates inherit the same tested byte portfolio instead of growing
new one-off entropy probes.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any

import brotli

from tac.archive_byte_profile import contest_rate_term
from tac.packet_compiler.pr101_per_tensor_grammar_solver import (
    DEFAULT_CODERS,
    FALSE_AUTHORITY_FIELDS,
    CoderName,
    empirical_shannon_floor_bytes,
    measure_payload_coder_candidate,
    payload_saturation_diagnostic,
)

SECTION_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA = "section_payload_grammar_optimizer.v1"
SECTION_PAYLOAD_GRAMMAR_CANDIDATE_SCHEMA = "section_payload_grammar_candidate.v1"
SECTION_PAYLOAD_GRAMMAR_QUEUE_SCHEMA = "optimizer_candidate_queue_v1"
SECTION_PAYLOAD_SOURCE_MANIFEST_SCHEMA = "section_payload_source_manifest.v1"


def sections_from_single_member_zip_archive(
    archive_zip: bytes | bytearray | memoryview,
    *,
    spans: Sequence[Mapping[str, Any]] | None = None,
    member_name: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract named payload sections from a single-member ZIP archive.

    The returned section rows are directly consumable by
    :func:`solve_section_payload_grammar`.  This keeps archive profiling
    deterministic and in-memory: no extracted scratch tree is needed, and the
    manifest records enough archive/member SHA and span data to reproduce the
    exact section bytes.
    """

    archive_bytes = bytes(archive_zip)
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise ValueError(f"expected a single-member ZIP archive, got {len(infos)}")
        info = infos[0]
        if member_name is not None and info.filename != member_name:
            raise ValueError(
                f"ZIP member mismatch: expected {member_name!r}, got {info.filename!r}"
            )
        member_bytes = zf.read(info.filename)
    member_sha = hashlib.sha256(member_bytes).hexdigest()
    span_rows = _normalize_spans(spans, member_len=len(member_bytes), member_name=info.filename)
    sections = [
        {
            "name": row["name"],
            "payload": member_bytes[int(row["start"]) : int(row["end"])],
        }
        for row in span_rows
    ]
    section_manifest_rows = [
        {
            **{key: value for key, value in row.items() if key != "payload"},
            "sha256": hashlib.sha256(section["payload"]).hexdigest(),
            "bytes": len(section["payload"]),
        }
        for row, section in zip(span_rows, sections, strict=True)
    ]
    overhead_bytes = len(archive_bytes) - int(info.compress_size)
    blockers: list[str] = []
    if info.compress_type != zipfile.ZIP_STORED:
        blockers.append("zip_member_not_stored_payload_bytes_are_decompressed_view")
    if not _spans_cover_contiguously(span_rows, member_len=len(member_bytes)):
        blockers.append("section_spans_do_not_cover_member_contiguously")
    manifest = {
        "schema": SECTION_PAYLOAD_SOURCE_MANIFEST_SCHEMA,
        "source_kind": "single_member_zip_archive",
        "archive_zip_bytes": len(archive_bytes),
        "archive_zip_sha256": archive_sha,
        "zip_member_count": 1,
        "zip_member_name": info.filename,
        "zip_member_file_size": int(info.file_size),
        "zip_member_compress_size": int(info.compress_size),
        "zip_member_compress_type": int(info.compress_type),
        "zip_member_is_stored": info.compress_type == zipfile.ZIP_STORED,
        "zip_overhead_bytes": overhead_bytes,
        "zip_overhead_scope": "archive_zip_bytes_minus_member_compress_size",
        "member_payload_bytes": len(member_bytes),
        "member_payload_sha256": member_sha,
        "section_count": len(section_manifest_rows),
        "sections": section_manifest_rows,
        "blockers": blockers,
    }
    return sections, manifest


def measure_section_coder_candidates(
    payload: bytes | bytearray | memoryview,
    *,
    section_name: str,
    section_index: int = 0,
    coders: Sequence[CoderName] = DEFAULT_CODERS,
    brotli_quality: int = 11,
) -> list[dict[str, Any]]:
    """Measure exact codec candidates for one archive section payload."""

    name = _section_name(section_name)
    data = bytes(payload)
    floor = empirical_shannon_floor_bytes(data)
    rows: list[dict[str, Any]] = []
    for coder in coders:
        measured = measure_payload_coder_candidate(
            data,
            coder=coder,
            brotli_quality=brotli_quality,
            brotli_lgwin_sweep=False,
        )
        status = str(measured["status"])
        runtime_status = _runtime_status_for_coder(coder, status)
        blockers = _candidate_blockers(status=status, runtime_status=runtime_status)
        charged = int(measured["charged_bytes"])
        rows.append(
            {
                "schema": SECTION_PAYLOAD_GRAMMAR_CANDIDATE_SCHEMA,
                "section_index": int(section_index),
                "section_name": name,
                "payload_bytes": len(data),
                "coder": coder,
                "coder_params": measured["coder_params"],
                "charged_bytes": charged,
                "codec_payload_bytes": int(measured["codec_payload_bytes"]),
                "side_info_bytes": int(measured["side_info_bytes"]),
                "empirical_shannon_floor_bytes": floor,
                "coded_over_floor_ratio": None if floor <= 0.0 else charged / floor,
                "codec_roundtrip_exact": bool(measured["roundtrip_exact"]),
                "roundtrip_exact": bool(measured["roundtrip_exact"]),
                "status": status,
                "runtime_consumption_status": runtime_status,
                "byte_accounting_scope": "isolated_section_payload_not_archive_authority",
                "axis_tag": "[planning-only byte-profile]",
                "blockers": blockers,
                **FALSE_AUTHORITY_FIELDS,
            }
        )
    rows.sort(
        key=lambda row: (
            row["status"] != "ok",
            row["roundtrip_exact"] is False,
            int(row["charged_bytes"]),
            str(row["coder"]),
        )
    )
    return rows


def select_best_section_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the smallest exact candidate from one section's candidates."""

    exact = [
        dict(row)
        for row in candidates
        if row.get("status") == "ok" and bool(row.get("roundtrip_exact"))
    ]
    if not exact:
        raise ValueError("no exact section grammar candidate was produced")
    exact.sort(key=lambda row: (int(row["charged_bytes"]), str(row["coder"])))
    selected = dict(exact[0])
    selected["selected"] = True
    return selected


def solve_section_payload_grammar(
    sections: Mapping[str, bytes] | Sequence[Mapping[str, Any]],
    *,
    coders: Sequence[CoderName] = DEFAULT_CODERS,
    brotli_quality: int = 11,
    baseline_coder: CoderName = "brotli",
    campaign_id: str = "section_payload_grammar",
    source_payload_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve independent per-section codec selection for arbitrary sections.

    ``sections`` may be a mapping of ``name -> bytes`` or a sequence of objects
    with ``name`` and ``payload`` keys.  The output is packet-compiler
    intelligence, not a submission artifact; receiver proof and byte-closed
    archive replay remain separate gates.
    """

    normalized = _normalize_sections(sections)
    rows: list[dict[str, Any]] = []
    selected_total = 0
    baseline_total = 0
    floor_total = 0.0
    for index, (name, payload) in enumerate(normalized):
        candidates = measure_section_coder_candidates(
            payload,
            section_name=name,
            section_index=index,
            coders=coders,
            brotli_quality=brotli_quality,
        )
        selected = select_best_section_candidate(candidates)
        baseline = _baseline_candidate(candidates, baseline_coder=baseline_coder)
        selected_total += int(selected["charged_bytes"])
        baseline_total += int(baseline["charged_bytes"])
        floor_total += float(selected["empirical_shannon_floor_bytes"])
        rows.append(
            {
                "schema": "section_payload_grammar_row.v1",
                "section_index": index,
                "section_name": name,
                "payload_bytes": len(payload),
                "selected": selected,
                "baseline": baseline,
                "top_candidates": candidates[:8],
            }
        )

    ratio = None if floor_total <= 0.0 else selected_total / floor_total
    order_diagnostic = _grouped_brotli_order_diagnostic(
        normalized,
        brotli_quality=brotli_quality,
    )
    blockers = [
        "section_codec_choices_not_bound_to_receiver",
        "byte_closed_archive_not_materialized",
        "runtime_consumption_proof_missing",
        "full_frame_inflate_parity_missing",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if source_payload_manifest is not None:
        blockers.extend(str(item) for item in source_payload_manifest.get("blockers", []))
    return {
        "schema": SECTION_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA,
        "campaign_id": campaign_id,
        "source_payload_manifest": None
        if source_payload_manifest is None
        else dict(source_payload_manifest),
        "section_count": len(rows),
        "coders": list(coders),
        "baseline_coder": baseline_coder,
        "brotli_quality": int(brotli_quality),
        "byte_accounting": {
            "selected_isolated_section_bytes": int(selected_total),
            "baseline_isolated_section_bytes": int(baseline_total),
            "selected_saved_bytes_vs_baseline": int(baseline_total - selected_total),
            "empirical_shannon_floor_bytes": float(floor_total),
            "selected_over_floor_ratio": ratio,
            "isolated_section_rate_term_not_archive_authority": contest_rate_term(
                selected_total
            ),
            "isolated_savings_rate_term_not_archive_authority": contest_rate_term(
                baseline_total - selected_total
            ),
        },
        "grouped_brotli_order_diagnostic": order_diagnostic,
        "saturation_diagnostic": payload_saturation_diagnostic(ratio),
        "planner_feedback": _planner_feedback(rows, order_diagnostic=order_diagnostic),
        "rows": rows,
        "blockers": sorted(dict.fromkeys(blockers)),
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "contest_rate_bytes_authority": False,
            "ready_for_exact_eval_dispatch": False,
            "reason": (
                "section codec choices are reusable packet-compiler signals "
                "until a receiver consumes them and a byte-closed archive replay passes"
            ),
        },
        "axis_tag": "[planning-only byte-profile]",
        **FALSE_AUTHORITY_FIELDS,
    }


def build_section_payload_optimizer_queue(
    report: Mapping[str, Any],
    *,
    campaign_id: str | None = None,
) -> dict[str, Any]:
    """Convert a generic section grammar report into optimizer-candidate rows."""

    if report.get("schema") != SECTION_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA:
        raise ValueError("expected section_payload_grammar_optimizer.v1 report")
    rows = report.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("section grammar report missing rows")
    cid = campaign_id or str(report.get("campaign_id") or "section_payload_grammar")
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        selected = row.get("selected")
        baseline = row.get("baseline")
        if not isinstance(selected, Mapping) or not isinstance(baseline, Mapping):
            continue
        saved = int(baseline.get("charged_bytes") or 0) - int(
            selected.get("charged_bytes") or 0
        )
        section_name = str(row.get("section_name"))
        candidate = {
            "schema": "optimizer_candidate_queue_row_v1",
            "candidate_id": f"{cid}:section:{_safe_id(section_name)}",
            "candidate_kind": "planning_only_section_payload_grammar",
            "status": "blocked_planning_signal_only",
            "target_kind": "archive_section_payload_grammar",
            "operation_family": "section_payload_coder_selection",
            "operation_families": ["section_payload_coder_selection"],
            "operation_id": "section_payload_coder_selection",
            "operation_params": {
                "section_index": row.get("section_index"),
                "section_name": section_name,
                "coder": selected.get("coder"),
                "coder_params": selected.get("coder_params") or {},
                "baseline_coder": baseline.get("coder"),
                "payload_bytes": row.get("payload_bytes"),
            },
            "selected_operations": [
                {
                    "operation_family": "section_payload_coder_selection",
                    "section_name": section_name,
                    "selected_coder": selected.get("coder"),
                    "baseline_coder": baseline.get("coder"),
                    "candidate_saved_bytes": max(0, saved),
                }
            ],
            "candidate_saved_bytes": max(0, saved),
            "saved_bytes_scope": "isolated_section_payload_only_not_archive_authority",
            "predicted_delta_bytes": -saved,
            "predicted_delta_bytes_scope": "isolated_section_payload_only_not_archive_authority",
            "runtime_consumption_status": selected.get("runtime_consumption_status"),
            "consumer_payload": {
                "selected_section_codec": dict(selected),
                "baseline_section_codec": dict(baseline),
                "byte_accounting_scope": "isolated_section_payload_not_archive_authority",
            },
            "blockers": [
                "section_codec_choice_not_bound_to_receiver",
                "byte_closed_archive_not_materialized",
                "runtime_consumption_proof_missing",
                "full_frame_inflate_parity_missing",
            ],
            "consumer_surfaces": [
                "tac.optimization.byte_shaving_campaign.build_signal_surface_from_candidate_queue",
                "tac.cathedral_consumers.packetir_candidate_queue_consumer.consume_queue",
            ],
            "axis_tag": "[planning-only byte-profile]",
            **FALSE_AUTHORITY_FIELDS,
        }
        candidates.append(candidate)
    return {
        "schema": SECTION_PAYLOAD_GRAMMAR_QUEUE_SCHEMA,
        "campaign_id": cid,
        "source_schema": report.get("schema"),
        "producer": "tac.packet_compiler.section_payload_grammar_optimizer",
        "proof_scope": "planning_only_section_payload_grammar_no_dispatch",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "top_k": [
            row for row in candidates if int(row.get("candidate_saved_bytes") or 0) > 0
        ],
        "blockers": [
            "section_codec_choices_not_bound_to_receiver",
            "byte_closed_archive_not_materialized",
            "runtime_consumption_proof_missing",
            "full_frame_inflate_parity_missing",
        ],
        "consumer_surfaces": [
            "tac.optimization.byte_shaving_campaign.build_signal_surface_from_candidate_queue",
        ],
        "axis_tag": "[planning-only byte-profile]",
        **FALSE_AUTHORITY_FIELDS,
    }


def _normalize_sections(
    sections: Mapping[str, bytes] | Sequence[Mapping[str, Any]],
) -> list[tuple[str, bytes]]:
    if isinstance(sections, Mapping):
        items = [(str(name), bytes(payload)) for name, payload in sections.items()]
    elif isinstance(sections, Sequence) and not isinstance(sections, (str, bytes)):
        items = []
        for index, row in enumerate(sections):
            if not isinstance(row, Mapping):
                raise ValueError(f"sections[{index}] must be an object")
            name = _section_name(str(row.get("name") or row.get("section_name") or ""))
            payload = row.get("payload")
            if not isinstance(payload, (bytes, bytearray, memoryview)):
                raise ValueError(f"sections[{index}].payload must be bytes-like")
            items.append((name, bytes(payload)))
    else:
        raise ValueError("sections must be a mapping or sequence of section objects")
    if not items:
        raise ValueError("at least one section is required")
    seen: set[str] = set()
    normalized: list[tuple[str, bytes]] = []
    for name, payload in items:
        clean = _section_name(name)
        if clean in seen:
            raise ValueError(f"duplicate section name: {clean!r}")
        seen.add(clean)
        normalized.append((clean, bytes(payload)))
    return normalized


def _normalize_spans(
    spans: Sequence[Mapping[str, Any]] | None,
    *,
    member_len: int,
    member_name: str,
) -> list[dict[str, Any]]:
    if spans is None:
        return [
            {
                "name": f"member:{member_name}",
                "start": 0,
                "end": member_len,
                "length": member_len,
            }
        ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, span in enumerate(spans):
        if not isinstance(span, Mapping):
            raise ValueError(f"spans[{index}] must be an object")
        name = _section_name(str(span.get("name") or span.get("section_name") or ""))
        if name in seen:
            raise ValueError(f"duplicate section span name: {name!r}")
        seen.add(name)
        start = _nonnegative_int(span.get("start"), f"spans[{index}].start")
        if "end" in span and span.get("end") is not None:
            end = _nonnegative_int(span.get("end"), f"spans[{index}].end")
            length = end - start
        else:
            length = _nonnegative_int(span.get("length"), f"spans[{index}].length")
            end = start + length
        if length < 0:
            raise ValueError(f"spans[{index}] end must be >= start")
        if end > member_len:
            raise ValueError(
                f"spans[{index}] exceeds ZIP member length: end={end} member_len={member_len}"
            )
        rows.append({"name": name, "start": start, "end": end, "length": length})
    return rows


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return parsed


def _spans_cover_contiguously(
    spans: Sequence[Mapping[str, Any]],
    *,
    member_len: int,
) -> bool:
    cursor = 0
    for span in sorted(spans, key=lambda row: int(row["start"])):
        if int(span["start"]) != cursor:
            return False
        cursor = int(span["end"])
    return cursor == member_len


def _section_name(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("section name must be non-empty")
    return text


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_") or "section"


def _baseline_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    baseline_coder: CoderName,
) -> dict[str, Any]:
    exact = [
        dict(row)
        for row in candidates
        if row.get("coder") == baseline_coder
        and row.get("status") == "ok"
        and bool(row.get("roundtrip_exact"))
    ]
    if exact:
        exact.sort(key=lambda row: int(row["charged_bytes"]))
        return exact[0]
    return select_best_section_candidate(candidates)


def _runtime_status_for_coder(coder: CoderName, status: str) -> str:
    if status != "ok":
        return "codec_candidate_not_usable"
    if coder == "brotli":
        return "brotli_decode_required"
    return "new_receiver_adapter_required"


def _candidate_blockers(*, status: str, runtime_status: str) -> list[str]:
    blockers = [
        "isolated_section_measurement_not_archive_authority",
        "byte_closed_archive_not_materialized",
        "runtime_consumption_proof_missing",
        "full_frame_inflate_parity_missing",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if status != "ok":
        blockers.append("codec_candidate_not_usable")
    if runtime_status == "new_receiver_adapter_required":
        blockers.append("receiver_adapter_not_emitted")
    return blockers


def _grouped_brotli_order_diagnostic(
    sections: Sequence[tuple[str, bytes]],
    *,
    brotli_quality: int,
) -> dict[str, Any]:
    candidates = [
        ("identity", list(range(len(sections)))),
        (
            "size_desc",
            sorted(range(len(sections)), key=lambda idx: (-len(sections[idx][1]), idx)),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for label, order in candidates:
        payload = b"".join(sections[idx][1] for idx in order)
        compressed = brotli.compress(payload, quality=int(brotli_quality))
        rows.append(
            {
                "schema": "section_payload_grouped_brotli_order_candidate.v1",
                "order_label": label,
                "section_order": [sections[idx][0] for idx in order],
                "raw_bytes": len(payload),
                "compressed_bytes": len(compressed),
                "roundtrip_exact": brotli.decompress(compressed) == payload,
            }
        )
    rows.sort(key=lambda row: (int(row["compressed_bytes"]), str(row["order_label"])))
    selected = rows[0]
    return {
        "schema": "section_payload_grouped_brotli_order_diagnostic.v1",
        "selected_order_label": selected["order_label"],
        "selected_grouped_brotli_bytes": selected["compressed_bytes"],
        "candidate_count": len(rows),
        "candidates": rows,
        "byte_accounting_scope": "single_stream_grouped_brotli_diagnostic_not_archive_authority",
    }


def _planner_feedback(
    rows: Sequence[Mapping[str, Any]],
    *,
    order_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    hints: list[dict[str, Any]] = []
    for row in rows:
        selected = row.get("selected")
        baseline = row.get("baseline")
        if not isinstance(selected, Mapping) or not isinstance(baseline, Mapping):
            continue
        saved = int(baseline.get("charged_bytes") or 0) - int(
            selected.get("charged_bytes") or 0
        )
        hints.append(
            {
                "schema": "section_payload_grammar_operation_hint.v1",
                "operation_family": "section_payload_coder_selection",
                "section_index": row.get("section_index"),
                "section_name": row.get("section_name"),
                "selected_coder": selected.get("coder"),
                "baseline_coder": baseline.get("coder"),
                "isolated_byte_delta_vs_baseline": -saved,
                "runtime_consumption_status": selected.get(
                    "runtime_consumption_status"
                ),
            }
        )
    return {
        "schema": "section_payload_grammar_planner_feedback.v1",
        "operation_hint_count": len(hints),
        "operation_hints": hints,
        "grouped_brotli_order_hint": {
            "selected_order_label": order_diagnostic.get("selected_order_label"),
            "selected_grouped_brotli_bytes": order_diagnostic.get(
                "selected_grouped_brotli_bytes"
            ),
        },
        "posterior_update_hooks": [
            "section_entropy_gap_by_payload_family",
            "section_coder_selection_by_archive_grammar",
            "grouped_brotli_order_by_payload_histogram",
        ],
        "consumer_surfaces": [
            "tac.optimization.byte_shaving_campaign.build_signal_surface_from_candidate_queue",
            "tac.cathedral_consumers.packetir_candidate_queue_consumer.consume_queue",
        ],
    }


__all__ = [
    "SECTION_PAYLOAD_GRAMMAR_CANDIDATE_SCHEMA",
    "SECTION_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA",
    "SECTION_PAYLOAD_GRAMMAR_QUEUE_SCHEMA",
    "SECTION_PAYLOAD_SOURCE_MANIFEST_SCHEMA",
    "build_section_payload_optimizer_queue",
    "measure_section_coder_candidates",
    "sections_from_single_member_zip_archive",
    "select_best_section_candidate",
    "solve_section_payload_grammar",
]
