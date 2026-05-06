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

SCHEMA_VERSION = 1
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
    "refuse_if_huffman_lengths_not_prefix_floor",
    "refuse_if_aq_floor_used_with_alphabet_size_less_than_two",
    "refuse_if_roundtrip_decode_validation_missing",
]


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
        "streams": rows,
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


def _as_positive_int(value: Any, context: str) -> int:
    integer = _as_nonnegative_int(value, context)
    if integer <= 0:
        raise EntropyCodecGapAuditError(f"{context} must be a positive integer")
    return integer


def _ceil_bytes(bits: float) -> int:
    return math.ceil(max(0.0, float(bits)) / 8.0 - 1e-12)


def _round_float(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if abs(rounded) < 5e-13 else rounded


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "AQC1_FIXED_HEADER_BYTES",
    "AQC1_SPARSE_FREQ_BYTES_PER_SYMBOL",
    "AQV1_DENSE_FREQ_BYTES_PER_SYMBOL",
    "AQV1_FIXED_HEADER_BYTES",
    "DEFAULT_EVIDENCE_GRADE",
    "DISPATCH_BLOCKERS",
    "FAIL_CLOSED_CRITERIA",
    "SCHEMA_VERSION",
    "SCORE_EVIDENCE_GRADE",
    "TOOL_NAME",
    "EntropyCodecGapAuditError",
    "build_entropy_codec_gap_audit",
    "render_markdown",
]
