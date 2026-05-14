# SPDX-License-Identifier: MIT
"""Audit entropy-codec gaps without building candidate archives.

This module compares observed byte streams against three local rate floors:

* zero-order Shannon entropy;
* optimal static Huffman prefix-code payload length;
* local AQv1/AQc1-style static arithmetic container lower bounds.

It is a planning/audit surface only. It does not encode payloads, mutate
archives, load scorers, dispatch jobs, or claim score movement.
"""
from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

SCHEMA_VERSION = 2
TOOL_NAME = "tac.optimization.entropy_codec_gap_audit"
SCORE_EVIDENCE_GRADE = "invalid"
DEFAULT_EVIDENCE_GRADE = "empirical"

AQV1_FIXED_HEADER_BYTES = 28
AQV1_DENSE_FREQ_BYTES_PER_SYMBOL = 4
AQC1_FIXED_HEADER_BYTES = 30
AQC1_SPARSE_FREQ_BYTES_PER_SYMBOL = 6

DISPATCH_BLOCKERS = [
    "planning_only_entropy_codec_gap_audit",
    "requires_byte_equivalent_codec_transform",
    "requires_roundtrip_decode_validation",
    "requires_archive_manifest_preflight",
    "requires_runtime_parity_proof",
    "requires_lane_dispatch_claim_before_gpu",
    "requires_exact_cuda_auth_eval",
]

FAIL_CLOSED_CRITERIA = [
    "refuse_if_counts_missing_or_nonpositive",
    "refuse_if_actual_bytes_missing_or_nonpositive",
    "refuse_if_count_record_sequence_is_malformed",
    "refuse_if_known_overhead_accounting_exceeds_actual_bytes",
    "refuse_if_huffman_lengths_not_prefix_floor",
    "refuse_if_aq_floor_used_with_alphabet_size_less_than_two",
    "refuse_if_roundtrip_decode_validation_missing",
]

ENTROPY_OVERHEAD_TARGET_ACTIONS = {
    "known_payload_entropy_gap": {
        "target_action": "prototype_byte_equivalent_entropy_coder_for_encoded_payload",
        "required_next_artifact": "roundtrip_payload_recode_manifest",
    },
    "known_model_overhead": {
        "target_action": "reduce_or_share_static_model_context_metadata",
        "required_next_artifact": "byte_accounted_model_overhead_reduction_manifest",
    },
    "known_container_overhead": {
        "target_action": "collapse_container_framing_or_merge_streams",
        "required_next_artifact": "container_overhead_diff_manifest",
    },
    "static_arithmetic_container_gap": {
        "target_action": "prototype_static_arithmetic_container_and_roundtrip_decode",
        "required_next_artifact": "byte_equivalent_static_arithmetic_archive_diff",
    },
}
_TARGET_KIND_PRIORITY = {
    "known_payload_entropy_gap": 0,
    "known_model_overhead": 1,
    "known_container_overhead": 2,
    "static_arithmetic_container_gap": 3,
}
BYTE_EQUIVALENCE_BLOCKERS = [
    "missing_source_archive_manifest",
    "missing_source_stream_sha256_and_byte_range",
    "missing_candidate_stream_sha256_and_byte_range",
    "missing_decoded_output_byte_equivalence_report",
    "missing_roundtrip_decode_validation_manifest",
    "missing_candidate_archive_manifest",
    "missing_runtime_tree_parity_manifest",
]
COMMON_EXACT_NEXT_ARTIFACT_REQUIREMENTS = [
    "source_archive_manifest_with_archive_sha256_bytes_and_runtime_tree_sha256",
    "source_stream_section_sha256_byte_range_and_symbol_count",
    "candidate_stream_section_sha256_byte_range_and_byte_count",
    "old_new_decoded_output_sha256_equality_report",
    "roundtrip_decode_validation_manifest",
    "candidate_archive_manifest_with_member_sha256s",
    "strict_pre_submission_compliance_json",
    "meta_lagrangian_atom_json_with_byte_delta_and_interaction_assumptions",
]
TARGET_KIND_ARTIFACT_REQUIREMENTS = {
    "known_payload_entropy_gap": [
        "byte_equivalent_payload_entropy_recode_manifest",
        "entropy_coder_decoder_contract",
    ],
    "known_model_overhead": [
        "byte_accounted_static_model_context_reduction_manifest",
        "old_new_model_context_table_diff",
    ],
    "known_container_overhead": [
        "container_header_or_stream_merge_diff_manifest",
        "old_new_zip_member_and_payload_layout_diff",
    ],
    "static_arithmetic_container_gap": [
        "static_arithmetic_container_roundtrip_manifest",
        "old_new_arithmetic_payload_floor_accounting",
    ],
}
META_LAGRANGIAN_REQUIRED_EXPORT_FIELDS = [
    "atom_id",
    "family",
    "family_group",
    "pareto_scope",
    "byte_delta",
    "expected_seg_dist_delta",
    "expected_pose_dist_delta",
    "confidence",
    "evidence_grade",
    "raw_equal",
    "interaction_assumptions",
    "archive_manifest_path",
    "archive_manifest_sha256",
]
_TARGET_KIND_FAMILY_SUFFIX = {
    "known_payload_entropy_gap": "payload_entropy_recode",
    "known_model_overhead": "entropy_model_overhead_recode",
    "known_container_overhead": "container_overhead_recode",
    "static_arithmetic_container_gap": "static_arithmetic_recode",
}


class EntropyCodecGapAuditError(ValueError):
    """Raised when an entropy-codec audit input is malformed."""


def build_entropy_codec_gap_audit(
    streams: Sequence[Mapping[str, Any]],
    *,
    source_label: str = "",
    evidence_grade: str = DEFAULT_EVIDENCE_GRADE,
) -> dict[str, Any]:
    """Build a deterministic manifest of entropy-codec floor gaps."""

    if not isinstance(streams, Sequence) or isinstance(streams, (str, bytes, bytearray)):
        raise EntropyCodecGapAuditError("streams must be a sequence of objects")
    if not streams:
        raise EntropyCodecGapAuditError("streams must be nonempty")

    rows = [
        _build_stream_row(
            stream,
            index=index,
            default_evidence_grade=evidence_grade,
        )
        for index, stream in enumerate(streams)
    ]
    labels = [str(row["label"]) for row in rows]
    if len(labels) != len(set(labels)):
        raise EntropyCodecGapAuditError("stream labels must be unique")
    rows.sort(key=lambda row: str(row["label"]))

    total_actual_bytes = sum(int(row["actual_bytes"]) for row in rows)
    total_symbol_count = sum(int(row["symbol_count"]) for row in rows)
    total_entropy_bits = sum(float(row["entropy_floor_bits"]) for row in rows)
    total_huffman_bits = sum(float(row["huffman_payload_bits"]) for row in rows)
    total_huffman_bits_exact = sum(int(row["huffman_payload_bits_exact"]) for row in rows)
    total_streamwise_entropy_bytes_ceil = sum(
        int(row["entropy_floor_bytes_ceil"]) for row in rows
    )
    total_huffman_payload_bytes_ceil = sum(
        int(row["huffman_payload_bytes_ceil"]) for row in rows
    )
    aq_floor_rows = [
        int(row["best_static_arithmetic_container_floor_bytes"])
        for row in rows
        if row["best_static_arithmetic_container_floor_bytes"] is not None
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "planning_only": True,
        "score_claim": False,
        "score_evidence_grade": SCORE_EVIDENCE_GRADE,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "fail_closed_criteria": list(FAIL_CLOSED_CRITERIA),
        "source_label": str(source_label),
        "evidence_grade": str(evidence_grade),
        "stream_count": len(rows),
        "total_symbol_count": total_symbol_count,
        "total_actual_bytes": total_actual_bytes,
        "total_entropy_floor_bits": _round_float(total_entropy_bits),
        "total_entropy_floor_bytes": _round_float(total_entropy_bits / 8.0),
        "total_entropy_floor_bytes_ceil": _ceil_bytes(total_entropy_bits),
        "total_streamwise_entropy_floor_bytes_ceil": total_streamwise_entropy_bytes_ceil,
        "total_gap_to_entropy_floor_bytes": _round_float(
            total_actual_bytes - total_entropy_bits / 8.0
        ),
        "total_huffman_payload_bits": _round_float(total_huffman_bits),
        "total_huffman_payload_bits_exact": total_huffman_bits_exact,
        "total_huffman_payload_bytes": _round_float(total_huffman_bits / 8.0),
        "total_huffman_payload_bytes_ceil": total_huffman_payload_bytes_ceil,
        "total_huffman_gap_over_entropy_bits": _round_float(
            total_huffman_bits - total_entropy_bits
        ),
        "total_best_static_arithmetic_container_floor_bytes": (
            sum(aq_floor_rows) if len(aq_floor_rows) == len(rows) else None
        ),
        "known_overhead_accounting": _known_overhead_accounting(rows),
        "streams": rows,
        "entropy_overhead_target_ranking": _entropy_overhead_target_ranking(rows),
        "opportunity_ranking": _opportunity_ranking(rows),
    }


def render_markdown(manifest: Mapping[str, Any]) -> str:
    """Render an entropy-codec gap audit as deterministic markdown."""

    streams = manifest.get("streams")
    if not isinstance(streams, list):
        raise EntropyCodecGapAuditError("manifest streams must be a list")

    lines = [
        "# Entropy Codec Gap Audit",
        "",
        f"- planning_only: `{_bool_text(manifest.get('planning_only') is True)}`",
        f"- score_claim: `{_bool_text(manifest.get('score_claim') is True)}`",
        f"- ready_for_exact_eval_dispatch: `{_bool_text(manifest.get('ready_for_exact_eval_dispatch') is True)}`",
        f"- total_actual_bytes: `{manifest.get('total_actual_bytes')}`",
        f"- total_entropy_floor_bytes: `{manifest.get('total_entropy_floor_bytes')}`",
        f"- total_huffman_payload_bytes: `{manifest.get('total_huffman_payload_bytes')}`",
        f"- total_best_static_arithmetic_container_floor_bytes: `{manifest.get('total_best_static_arithmetic_container_floor_bytes')}`",
        "",
        "| stream | actual bytes | H bits/sym | Huffman bits/sym | AQ floor bytes | best AQ kind | gap to AQ floor |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in streams:
        if not isinstance(row, Mapping):
            raise EntropyCodecGapAuditError("manifest stream rows must be objects")
        lines.append(
            "| {label} | {actual} | {entropy} | {huffman} | {aq_floor} | `{aq_kind}` | {gap} |".format(
                label=row.get("label"),
                actual=row.get("actual_bytes"),
                entropy=row.get("entropy_bits_per_symbol"),
                huffman=row.get("huffman_bits_per_symbol"),
                aq_floor=row.get("best_static_arithmetic_container_floor_bytes"),
                aq_kind=row.get("best_static_arithmetic_container_kind"),
                gap=row.get("gap_to_best_static_arithmetic_container_floor_bytes"),
            )
        )
    overhead = manifest.get("known_overhead_accounting")
    if isinstance(overhead, Mapping) and int(overhead.get("streams_with_known_accounting") or 0):
        lines.extend(
            [
                "",
                "## Known Overhead Accounting",
                "",
                f"- streams_with_known_accounting: `{overhead.get('streams_with_known_accounting')}`",
                f"- total_known_model_overhead_bytes: `{overhead.get('total_known_model_overhead_bytes')}`",
                f"- total_known_container_overhead_bytes: `{overhead.get('total_known_container_overhead_bytes')}`",
                f"- total_known_payload_gap_to_entropy_floor_bytes: `{overhead.get('total_known_payload_gap_to_entropy_floor_bytes')}`",
                "",
                "| stream | known overhead bytes | model bytes | container bytes | payload gap to H floor | unattributed bytes |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in overhead.get("largest_known_overhead_streams") or []:
            if not isinstance(row, Mapping):
                raise EntropyCodecGapAuditError("known overhead ranking rows must be objects")
            lines.append(
                "| {label} | {known} | {model} | {container} | {payload_gap} | {unattributed} |".format(
                    label=row.get("label"),
                    known=row.get("known_overhead_bytes"),
                    model=row.get("known_model_overhead_bytes"),
                    container=row.get("known_container_overhead_bytes"),
                    payload_gap=row.get("known_payload_gap_to_entropy_floor_bytes"),
                    unattributed=row.get("known_unattributed_bytes"),
                )
            )
    targets = manifest.get("entropy_overhead_target_ranking")
    if isinstance(targets, list) and targets:
        lines.extend(
            [
                "",
                "## Entropy-Overhead Target Ranking",
                "",
                "| rank | stream | target kind | target bytes | next artifact | ready for exact eval |",
                "|---:|---|---|---:|---|---|",
            ]
        )
        for row in targets:
            if not isinstance(row, Mapping):
                raise EntropyCodecGapAuditError("entropy-overhead target rows must be objects")
            lines.append(
                "| {rank} | {label} | `{kind}` | {bytes} | `{artifact}` | `{ready}` |".format(
                    rank=row.get("rank"),
                    label=row.get("label"),
                    kind=row.get("target_kind"),
                    bytes=row.get("target_bytes"),
                    artifact=row.get("required_next_artifact"),
                    ready=_bool_text(row.get("ready_for_exact_eval_dispatch") is True),
                )
            )
        lines.extend(
            [
                "",
                "| rank | stream | artifact requirements | byte-equivalence blockers | meta atom export |",
                "|---:|---|---:|---:|---|",
            ]
        )
        for row in targets:
            if not isinstance(row, Mapping):
                raise EntropyCodecGapAuditError("entropy-overhead target rows must be objects")
            meta_export = row.get("meta_lagrangian_atom_export")
            meta_atom = ""
            if isinstance(meta_export, Mapping):
                atom_template = meta_export.get("atom_template")
                if isinstance(atom_template, Mapping):
                    meta_atom = str(atom_template.get("atom_id") or "")
            lines.append(
                "| {rank} | {label} | {requirements} | {blockers} | `{meta_atom}` |".format(
                    rank=row.get("rank"),
                    label=row.get("label"),
                    requirements=len(row.get("exact_next_artifact_requirements") or []),
                    blockers=len(row.get("byte_equivalence_blockers") or []),
                    meta_atom=meta_atom,
                )
            )
    lines.append("")
    return "\n".join(lines)


def _build_stream_row(
    stream: Mapping[str, Any],
    *,
    index: int,
    default_evidence_grade: str,
) -> dict[str, Any]:
    if not isinstance(stream, Mapping):
        raise EntropyCodecGapAuditError(f"stream[{index}] must be an object")
    label = _required_label(stream.get("label"), f"stream[{index}].label")
    actual_bytes = _as_positive_int(
        stream.get("actual_bytes", stream.get("bytes_charged")),
        f"{label}.actual_bytes",
    )
    known_encoded_payload_bytes = _optional_nonnegative_int(
        stream.get("encoded_payload_bytes", stream.get("payload_bytes")),
        f"{label}.encoded_payload_bytes",
    )
    known_model_overhead_bytes = _optional_nonnegative_int(
        stream.get("model_overhead_bytes"),
        f"{label}.model_overhead_bytes",
    )
    known_container_overhead_bytes = _optional_nonnegative_int(
        stream.get("container_overhead_bytes"),
        f"{label}.container_overhead_bytes",
    )
    _validate_known_accounting(
        actual_bytes=actual_bytes,
        label=label,
        encoded_payload_bytes=known_encoded_payload_bytes,
        model_overhead_bytes=known_model_overhead_bytes,
        container_overhead_bytes=known_container_overhead_bytes,
    )
    counts = _normalize_counts(stream.get("symbol_counts", stream.get("counts")), f"{label}.symbol_counts")
    entropy = _entropy_floor(counts)
    huffman = _huffman_floor(counts)
    aq = _aq_container_floor(counts, entropy_bits=float(entropy["_entropy_floor_bits_raw"]))

    best_aq_bytes = aq["best_static_arithmetic_container_floor_bytes"]
    gap_to_best_aq = (
        _round_float(actual_bytes - int(best_aq_bytes))
        if best_aq_bytes is not None
        else None
    )
    entropy_bytes = float(entropy["entropy_floor_bits"]) / 8.0

    return {
        "label": label,
        "source": str(stream.get("source") or ""),
        "codec_surface": str(stream.get("codec_surface") or ""),
        "evidence_grade": str(stream.get("evidence_grade") or default_evidence_grade),
        "planning_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "fail_closed_criteria": list(FAIL_CLOSED_CRITERIA),
        "actual_bytes": actual_bytes,
        "symbol_count": entropy["symbol_count"],
        "alphabet_size": entropy["alphabet_size"],
        "positive_symbol_count": entropy["positive_symbol_count"],
        "entropy_bits_per_symbol": entropy["entropy_bits_per_symbol"],
        "entropy_floor_bits": entropy["entropy_floor_bits"],
        "entropy_floor_bytes": entropy["entropy_floor_bytes"],
        "entropy_floor_bytes_ceil": entropy["entropy_floor_bytes_ceil"],
        "gap_to_entropy_floor_bytes": _round_float(actual_bytes - entropy_bytes),
        **_known_accounting_row(
            actual_bytes=actual_bytes,
            entropy_floor_bytes=float(entropy["entropy_floor_bytes"]),
            encoded_payload_bytes=known_encoded_payload_bytes,
            model_overhead_bytes=known_model_overhead_bytes,
            container_overhead_bytes=known_container_overhead_bytes,
        ),
        "huffman_code_lengths": huffman["code_lengths"],
        "huffman_bits_per_symbol": huffman["huffman_bits_per_symbol"],
        "huffman_payload_bits": huffman["huffman_payload_bits"],
        "huffman_payload_bits_exact": huffman["huffman_payload_bits_exact"],
        "huffman_payload_bytes": huffman["huffman_payload_bytes"],
        "huffman_payload_bytes_ceil": huffman["huffman_payload_bytes_ceil"],
        "huffman_gap_over_entropy_bits": _round_float(
            float(huffman["huffman_payload_bits"]) - float(entropy["entropy_floor_bits"])
        ),
        "huffman_degenerate_zero_bit_single_symbol": huffman[
            "degenerate_zero_bit_single_symbol"
        ],
        **aq,
        "gap_to_best_static_arithmetic_container_floor_bytes": gap_to_best_aq,
        "symbol_counts": counts,
    }


def _entropy_floor(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = sum(int(row["count"]) for row in records)
    entropy_bits_per_symbol = 0.0
    positive = 0
    for row in records:
        count = int(row["count"])
        if count <= 0:
            continue
        positive += 1
        probability = count / total
        entropy_bits_per_symbol -= probability * math.log2(probability)
    bits = total * entropy_bits_per_symbol
    return {
        "symbol_count": total,
        "alphabet_size": len(records),
        "positive_symbol_count": positive,
        "_entropy_floor_bits_raw": bits,
        "entropy_bits_per_symbol": _round_float(entropy_bits_per_symbol),
        "entropy_floor_bits": _round_float(bits),
        "entropy_floor_bytes": _round_float(bits / 8.0),
        "entropy_floor_bytes_ceil": _ceil_bytes(bits),
    }


def _huffman_floor(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positive = [
        (str(row["symbol"]), int(row["count"]))
        for row in records
        if int(row["count"]) > 0
    ]
    if not positive:
        raise EntropyCodecGapAuditError("positive count set must be nonempty")
    if len(positive) == 1:
        symbol, count = positive[0]
        return {
            "code_lengths": {symbol: 0},
            "huffman_bits_per_symbol": 0.0,
            "huffman_payload_bits": 0.0,
            "huffman_payload_bits_exact": 0,
            "huffman_payload_bytes": 0.0,
            "huffman_payload_bytes_ceil": 0,
            "degenerate_zero_bit_single_symbol": True,
        }

    heap: list[tuple[int, int, tuple[str, ...]]] = []
    for order, (symbol, count) in enumerate(sorted(positive)):
        heapq.heappush(heap, (count, order, (symbol,)))
    lengths = {symbol: 0 for symbol, _count in positive}
    next_order = len(heap)
    while len(heap) > 1:
        count_a, _order_a, symbols_a = heapq.heappop(heap)
        count_b, _order_b, symbols_b = heapq.heappop(heap)
        for symbol in symbols_a:
            lengths[symbol] += 1
        for symbol in symbols_b:
            lengths[symbol] += 1
        heapq.heappush(
            heap,
            (
                count_a + count_b,
                next_order,
                tuple(sorted((*symbols_a, *symbols_b))),
            ),
        )
        next_order += 1

    count_by_symbol = dict(positive)
    total = sum(count_by_symbol.values())
    payload_bits = sum(count_by_symbol[symbol] * lengths[symbol] for symbol in lengths)
    return {
        "code_lengths": dict(sorted(lengths.items())),
        "huffman_bits_per_symbol": _round_float(payload_bits / total),
        "huffman_payload_bits": _round_float(payload_bits),
        "huffman_payload_bits_exact": int(payload_bits),
        "huffman_payload_bytes": _round_float(payload_bits / 8.0),
        "huffman_payload_bytes_ceil": _ceil_bytes(payload_bits),
        "degenerate_zero_bit_single_symbol": False,
    }


def _aq_container_floor(
    records: Sequence[Mapping[str, Any]],
    *,
    entropy_bits: float,
) -> dict[str, Any]:
    alphabet_size = len(records)
    positive_symbol_count = sum(1 for row in records if int(row["count"]) > 0)
    entropy_payload_bytes = _ceil_bytes(entropy_bits)
    arithmetic_payload_floor_bytes = max(1, entropy_payload_bytes)
    if alphabet_size < 2:
        return {
            "aq_floor_applicable": False,
            "static_arithmetic_entropy_payload_floor_bytes": entropy_payload_bytes,
            "static_arithmetic_payload_floor_bytes": None,
            "aqv1_static_model_floor_bytes": None,
            "aqc1_sparse_model_floor_bytes": None,
            "best_static_arithmetic_container_kind": "",
            "best_static_arithmetic_container_floor_bytes": None,
        }
    aqv1 = (
        AQV1_FIXED_HEADER_BYTES
        + AQV1_DENSE_FREQ_BYTES_PER_SYMBOL * alphabet_size
        + arithmetic_payload_floor_bytes
    )
    aqc1 = (
        AQC1_FIXED_HEADER_BYTES
        + AQC1_SPARSE_FREQ_BYTES_PER_SYMBOL * positive_symbol_count
        + arithmetic_payload_floor_bytes
    )
    if aqc1 < aqv1:
        best_kind = "AQc1"
        best_bytes = aqc1
    else:
        best_kind = "AQv1"
        best_bytes = aqv1
    return {
        "aq_floor_applicable": True,
        "static_arithmetic_entropy_payload_floor_bytes": entropy_payload_bytes,
        "static_arithmetic_payload_floor_bytes": arithmetic_payload_floor_bytes,
        "aqv1_static_model_floor_bytes": aqv1,
        "aqc1_sparse_model_floor_bytes": aqc1,
        "best_static_arithmetic_container_kind": best_kind,
        "best_static_arithmetic_container_floor_bytes": best_bytes,
    }


def _normalize_counts(value: Any, context: str) -> list[dict[str, Any]]:
    if value is None:
        raise EntropyCodecGapAuditError(f"{context} missing")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if isinstance(value, Mapping):
        if not value:
            raise EntropyCodecGapAuditError(f"{context} must be nonempty")
        items = sorted(value.items(), key=lambda item: str(item[0]))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise EntropyCodecGapAuditError(f"{context} must be nonempty")
        if all(isinstance(item, Mapping) for item in value):
            items = []
            for index, item in enumerate(value):
                if "symbol" not in item or "count" not in item:
                    raise EntropyCodecGapAuditError(
                        f"{context}[{index}] must contain symbol and count"
                    )
                items.append((item["symbol"], item["count"]))
            items.sort(key=lambda item: str(item[0]))
        else:
            items = [(str(index), count) for index, count in enumerate(value)]
    else:
        raise EntropyCodecGapAuditError(f"{context} must be an object or sequence")
    for symbol_raw, count_raw in items:
        symbol = _required_label(symbol_raw, f"{context}.symbol")
        if symbol in seen:
            raise EntropyCodecGapAuditError(
                f"{context} has duplicate symbol after string conversion: {symbol}"
            )
        seen.add(symbol)
        rows.append(
            {
                "symbol": symbol,
                "count": _as_nonnegative_int(count_raw, f"{context}.{symbol}"),
            }
        )
    if sum(int(row["count"]) for row in rows) <= 0:
        raise EntropyCodecGapAuditError(f"{context} must have positive total")
    return rows


def _known_accounting_row(
    *,
    actual_bytes: int,
    entropy_floor_bytes: float,
    encoded_payload_bytes: int | None,
    model_overhead_bytes: int | None,
    container_overhead_bytes: int | None,
) -> dict[str, Any]:
    known_parts = [
        value
        for value in (
            encoded_payload_bytes,
            model_overhead_bytes,
            container_overhead_bytes,
        )
        if value is not None
    ]
    known_accounting_complete = bool(known_parts)
    known_payload = encoded_payload_bytes if encoded_payload_bytes is not None else None
    known_model = model_overhead_bytes if model_overhead_bytes is not None else None
    known_container = (
        container_overhead_bytes if container_overhead_bytes is not None else None
    )
    known_overhead = int(known_model or 0) + int(known_container or 0)
    known_total = int(sum(known_parts)) if known_parts else None
    known_unattributed = actual_bytes - int(known_total) if known_total is not None else None
    payload_gap = (
        _round_float(float(known_payload) - entropy_floor_bytes)
        if known_payload is not None
        else None
    )
    return {
        "known_encoded_payload_bytes": known_payload,
        "known_model_overhead_bytes": known_model,
        "known_container_overhead_bytes": known_container,
        "known_overhead_bytes": known_overhead if known_accounting_complete else None,
        "known_unattributed_bytes": known_unattributed,
        "known_overhead_accounting_complete": known_accounting_complete
        and known_unattributed == 0,
        "known_payload_gap_to_entropy_floor_bytes": payload_gap,
    }


def _known_overhead_accounting(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    known_rows = [
        row
        for row in rows
        if row.get("known_encoded_payload_bytes") is not None
        or row.get("known_model_overhead_bytes") is not None
        or row.get("known_container_overhead_bytes") is not None
    ]
    ranked = []
    for row in known_rows:
        ranked.append(
            {
                "label": row["label"],
                "actual_bytes": row["actual_bytes"],
                "entropy_floor_bytes": row["entropy_floor_bytes"],
                "known_encoded_payload_bytes": row["known_encoded_payload_bytes"],
                "known_model_overhead_bytes": row["known_model_overhead_bytes"],
                "known_container_overhead_bytes": row["known_container_overhead_bytes"],
                "known_overhead_bytes": row["known_overhead_bytes"],
                "known_unattributed_bytes": row["known_unattributed_bytes"],
                "known_overhead_accounting_complete": row[
                    "known_overhead_accounting_complete"
                ],
                "known_payload_gap_to_entropy_floor_bytes": row[
                    "known_payload_gap_to_entropy_floor_bytes"
                ],
                "ready_for_exact_eval_dispatch": False,
                "dispatch_blockers": list(DISPATCH_BLOCKERS),
            }
        )
    ranked.sort(
        key=lambda row: (
            -int(row["known_overhead_bytes"] or 0),
            -float(row["known_payload_gap_to_entropy_floor_bytes"] or -1e18),
            str(row["label"]),
        )
    )
    return {
        "streams_with_known_accounting": len(known_rows),
        "complete_stream_accounting_count": sum(
            1 for row in known_rows if row.get("known_overhead_accounting_complete") is True
        ),
        "total_known_encoded_payload_bytes": sum(
            int(row.get("known_encoded_payload_bytes") or 0) for row in known_rows
        ),
        "total_known_model_overhead_bytes": sum(
            int(row.get("known_model_overhead_bytes") or 0) for row in known_rows
        ),
        "total_known_container_overhead_bytes": sum(
            int(row.get("known_container_overhead_bytes") or 0) for row in known_rows
        ),
        "total_known_overhead_bytes": sum(
            int(row.get("known_overhead_bytes") or 0) for row in known_rows
        ),
        "total_known_unattributed_bytes": sum(
            int(row.get("known_unattributed_bytes") or 0) for row in known_rows
        ),
        "total_known_payload_gap_to_entropy_floor_bytes": _round_float(
            sum(
                float(row.get("known_payload_gap_to_entropy_floor_bytes") or 0.0)
                for row in known_rows
            )
        ),
        "largest_known_overhead_streams": ranked,
    }


def _entropy_overhead_target_ranking(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for row in rows:
        has_known_accounting = (
            row.get("known_encoded_payload_bytes") is not None
            or row.get("known_model_overhead_bytes") is not None
            or row.get("known_container_overhead_bytes") is not None
        )
        if has_known_accounting:
            _append_entropy_overhead_target(
                ranked,
                row,
                target_kind="known_payload_entropy_gap",
                target_bytes=row.get("known_payload_gap_to_entropy_floor_bytes"),
                target_bytes_field="known_payload_gap_to_entropy_floor_bytes",
                accounting_source="known_overhead_accounting",
            )
            _append_entropy_overhead_target(
                ranked,
                row,
                target_kind="known_model_overhead",
                target_bytes=row.get("known_model_overhead_bytes"),
                target_bytes_field="known_model_overhead_bytes",
                accounting_source="known_overhead_accounting",
            )
            _append_entropy_overhead_target(
                ranked,
                row,
                target_kind="known_container_overhead",
                target_bytes=row.get("known_container_overhead_bytes"),
                target_bytes_field="known_container_overhead_bytes",
                accounting_source="known_overhead_accounting",
            )
            continue
        _append_entropy_overhead_target(
            ranked,
            row,
            target_kind="static_arithmetic_container_gap",
            target_bytes=row.get("gap_to_best_static_arithmetic_container_floor_bytes"),
            target_bytes_field="gap_to_best_static_arithmetic_container_floor_bytes",
            accounting_source="static_arithmetic_floor",
        )
    ranked.sort(
        key=lambda row: (
            -float(row["target_bytes"]),
            _TARGET_KIND_PRIORITY[str(row["target_kind"])],
            str(row["label"]),
            str(row["target_kind"]),
        )
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _append_entropy_overhead_target(
    ranked: list[dict[str, Any]],
    row: Mapping[str, Any],
    *,
    target_kind: str,
    target_bytes: Any,
    target_bytes_field: str,
    accounting_source: str,
) -> None:
    positive_bytes = _positive_target_bytes(target_bytes)
    if positive_bytes is None:
        return
    action = ENTROPY_OVERHEAD_TARGET_ACTIONS[target_kind]
    ranked.append(
        {
            "label": row["label"],
            "source": row["source"],
            "codec_surface": row["codec_surface"],
            "target_kind": target_kind,
            "target_bytes": positive_bytes,
            "target_bytes_field": target_bytes_field,
            "accounting_source": accounting_source,
            "target_action": action["target_action"],
            "required_next_artifact": action["required_next_artifact"],
            "exact_next_artifact_requirements": _exact_next_artifact_requirements(
                target_kind
            ),
            "byte_equivalence_blockers": list(BYTE_EQUIVALENCE_BLOCKERS),
            "actual_bytes": row["actual_bytes"],
            "entropy_floor_bytes": row["entropy_floor_bytes"],
            "best_static_arithmetic_container_kind": row["best_static_arithmetic_container_kind"],
            "best_static_arithmetic_container_floor_bytes": row["best_static_arithmetic_container_floor_bytes"],
            "known_overhead_accounting_complete": row["known_overhead_accounting_complete"],
            "readiness_stage": "planning_target_requires_byte_equivalent_artifacts",
            "planning_only": True,
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_byte_closed_candidate_build": False,
            "ready_for_meta_lagrangian_atom_export": False,
            "ready_for_archive_preflight": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_blockers": list(DISPATCH_BLOCKERS),
            "fail_closed_criteria": list(FAIL_CLOSED_CRITERIA),
            "meta_lagrangian_atom_export": _meta_lagrangian_atom_export(
                row,
                target_kind=target_kind,
                target_bytes=positive_bytes,
                target_bytes_field=target_bytes_field,
            ),
        }
    )


def _exact_next_artifact_requirements(target_kind: str) -> list[str]:
    action = ENTROPY_OVERHEAD_TARGET_ACTIONS[target_kind]
    return _unique_ordered_strings(
        [
            action["required_next_artifact"],
            *TARGET_KIND_ARTIFACT_REQUIREMENTS[target_kind],
            *COMMON_EXACT_NEXT_ARTIFACT_REQUIREMENTS,
        ]
    )


def _meta_lagrangian_atom_export(
    row: Mapping[str, Any],
    *,
    target_kind: str,
    target_bytes: int | float,
    target_bytes_field: str,
) -> dict[str, Any]:
    label_fragment = _atom_id_fragment(str(row["label"]))
    family_prefix = _target_family_prefix(row)
    family = f"{family_prefix}_{_TARGET_KIND_FAMILY_SUFFIX[target_kind]}"
    family_group = f"{family_prefix}_rate_equivalent_recode"
    target_amount = float(target_bytes)
    ledger_byte_delta = -math.floor(target_amount)
    export_blockers = [
        "planning_target_not_byte_closed_candidate",
        *BYTE_EQUIVALENCE_BLOCKERS,
        "missing_archive_manifest_path",
        "missing_archive_manifest_sha256",
    ]
    if ledger_byte_delta == 0:
        export_blockers.append("target_bytes_less_than_one_meta_lagrangian_byte")
    return {
        "schema": "meta_lagrangian_atom_export_v1",
        "export_ready": False,
        "ready_for_meta_lagrangian_atom_export": False,
        "export_blockers": _unique_ordered_strings(export_blockers),
        "required_fields_before_export": list(META_LAGRANGIAN_REQUIRED_EXPORT_FIELDS),
        "atom_template": {
            "atom_id": f"{label_fragment}:{target_kind}",
            "family": family,
            "family_group": family_group,
            "pareto_scope": f"{family_group}:{label_fragment}",
            "conflicts_with_families": [],
            "conflicts_with_atoms": [],
            "byte_delta": ledger_byte_delta,
            "estimated_byte_delta": _round_float(-target_amount),
            "target_bytes": target_bytes,
            "target_bytes_field": target_bytes_field,
            "expected_seg_dist_delta": 0.0,
            "expected_pose_dist_delta": 0.0,
            "confidence": 0.0,
            "evidence_grade": "invalid_planning_target_until_byte_equivalent_candidate",
            "raw_equal": False,
            "score_claim": False,
            "dispatchable": False,
            "ready_for_exact_eval_dispatch": False,
            "interaction_assumptions": [
                "rate_only_decoded_output_equivalence_required"
            ],
            "hard_pair_support": [],
            "pair_support": [],
            "class_support": [],
            "geometry_priors": [],
            "openpilot_priors": [],
            "evidence_source_path": str(row.get("source") or ""),
            "source_archive_sha256": "",
            "archive_manifest_path": "",
            "archive_manifest_sha256": "",
        },
    }


def _target_family_prefix(row: Mapping[str, Any]) -> str:
    haystack = " ".join(
        str(row.get(key) or "") for key in ("label", "source", "codec_surface")
    ).lower()
    return "hnerv" if "hnerv" in haystack else "entropy_codec"


def _atom_id_fragment(value: str) -> str:
    pieces = []
    for char in value.lower():
        pieces.append(char if char.isalnum() else "_")
    fragment = "_".join(part for part in "".join(pieces).split("_") if part)
    return fragment or "stream"


def _opportunity_ranking(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for row in rows:
        ranked.append(
            {
                "label": row["label"],
                "actual_bytes": row["actual_bytes"],
                "gap_to_entropy_floor_bytes": row["gap_to_entropy_floor_bytes"],
                "gap_to_best_static_arithmetic_container_floor_bytes": row[
                    "gap_to_best_static_arithmetic_container_floor_bytes"
                ],
                "best_static_arithmetic_container_kind": row[
                    "best_static_arithmetic_container_kind"
                ],
                "ready_for_exact_eval_dispatch": False,
                "dispatch_blockers": list(DISPATCH_BLOCKERS),
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row["gap_to_best_static_arithmetic_container_floor_bytes"] or -1e18),
            -float(row["gap_to_entropy_floor_bytes"]),
            str(row["label"]),
        )
    )
    return ranked


def _required_label(value: Any, context: str) -> str:
    label = str(value or "").strip()
    if not label:
        raise EntropyCodecGapAuditError(f"{context} must be nonempty")
    if any(ord(char) > 127 for char in label):
        raise EntropyCodecGapAuditError(f"{context} must be ASCII")
    return label


def _as_nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise EntropyCodecGapAuditError(f"{context} must be a non-negative integer")
    if value < 0:
        raise EntropyCodecGapAuditError(f"{context} must be a non-negative integer")
    return int(value)


def _optional_nonnegative_int(value: Any, context: str) -> int | None:
    if value is None:
        return None
    return _as_nonnegative_int(value, context)


def _as_positive_int(value: Any, context: str) -> int:
    integer = _as_nonnegative_int(value, context)
    if integer <= 0:
        raise EntropyCodecGapAuditError(f"{context} must be a positive integer")
    return integer


def _validate_known_accounting(
    *,
    actual_bytes: int,
    label: str,
    encoded_payload_bytes: int | None,
    model_overhead_bytes: int | None,
    container_overhead_bytes: int | None,
) -> None:
    known_total = sum(
        int(value)
        for value in (
            encoded_payload_bytes,
            model_overhead_bytes,
            container_overhead_bytes,
        )
        if value is not None
    )
    if known_total > actual_bytes:
        raise EntropyCodecGapAuditError(
            f"{label}.known_overhead_accounting exceeds actual_bytes"
        )


def _ceil_bytes(bits: float) -> int:
    return math.ceil(max(0.0, float(bits)) / 8.0 - 1e-12)


def _round_float(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if abs(rounded) < 5e-13 else rounded


def _positive_target_bytes(value: Any) -> int | float | None:
    if value is None:
        return None
    amount = _round_float(float(value))
    if amount <= 0:
        return None
    return int(amount) if float(amount).is_integer() else amount


def _unique_ordered_strings(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "AQC1_FIXED_HEADER_BYTES",
    "AQC1_SPARSE_FREQ_BYTES_PER_SYMBOL",
    "AQV1_DENSE_FREQ_BYTES_PER_SYMBOL",
    "AQV1_FIXED_HEADER_BYTES",
    "BYTE_EQUIVALENCE_BLOCKERS",
    "COMMON_EXACT_NEXT_ARTIFACT_REQUIREMENTS",
    "DEFAULT_EVIDENCE_GRADE",
    "DISPATCH_BLOCKERS",
    "ENTROPY_OVERHEAD_TARGET_ACTIONS",
    "FAIL_CLOSED_CRITERIA",
    "META_LAGRANGIAN_REQUIRED_EXPORT_FIELDS",
    "SCHEMA_VERSION",
    "SCORE_EVIDENCE_GRADE",
    "TARGET_KIND_ARTIFACT_REQUIREMENTS",
    "TOOL_NAME",
    "EntropyCodecGapAuditError",
    "build_entropy_codec_gap_audit",
    "render_markdown",
]
