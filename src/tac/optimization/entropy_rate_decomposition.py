"""Planning-only entropy-rate decomposition for contest byte streams.

This module estimates entropy floors from observed symbol counts and optional
conditional count groups. It is a rate-planning surface only: it never builds
archives, loads scorers, dispatches GPUs, or claims contest score movement.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 1
TOOL_NAME = "tac.optimization.entropy_rate_decomposition"
DEFAULT_EVIDENCE_GRADE = "empirical"
SCORE_EVIDENCE_GRADE = "invalid"
DISPATCH_BLOCKERS = [
    "planning_only_entropy_rate_decomposition",
    "requires_byte_equivalent_codec_transform",
    "requires_archive_manifest_preflight",
    "requires_runtime_parity_proof",
    "requires_lane_dispatch_claim_before_gpu",
    "requires_exact_cuda_auth_eval",
]
STREAM_KIND_ACTIONS = {
    "hnerv": "target decoder and latent payload recoding before archive rebuild",
    "categorical": "fit context model or arithmetic coder against symbol stream",
    "pose": "test delta or context-conditioned pose residual coding",
}
STREAM_KIND_PRIORITY = {
    "hnerv": 0,
    "categorical": 1,
    "pose": 2,
}


class EntropyRateDecompositionError(ValueError):
    """Raised when entropy-rate inputs are not well-formed."""


def entropy_bits_per_symbol(counts: Mapping[Any, Any] | Sequence[Any]) -> float:
    """Return empirical Shannon entropy in bits per symbol for ``counts``."""

    records = _normalize_counts(counts, "counts")
    return _entropy_bits_per_symbol_from_records(records)


def build_entropy_rate_decomposition(
    streams: Sequence[Mapping[str, Any]],
    *,
    source_label: str = "",
    evidence_grade: str = DEFAULT_EVIDENCE_GRADE,
) -> dict[str, Any]:
    """Return a deterministic planning manifest for stream entropy gaps."""

    if not isinstance(streams, Sequence) or isinstance(streams, (str, bytes, bytearray)):
        raise EntropyRateDecompositionError("streams must be a sequence of objects")
    if not streams:
        raise EntropyRateDecompositionError("streams must be nonempty")

    rows: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for index, stream in enumerate(streams):
        if not isinstance(stream, Mapping):
            raise EntropyRateDecompositionError(f"stream[{index}] must be an object")
        row = _build_stream_row(stream, default_evidence_grade=evidence_grade)
        label = str(row["label"])
        if label in seen_labels:
            raise EntropyRateDecompositionError(f"duplicate stream label: {label}")
        seen_labels.add(label)
        rows.append(row)

    rows.sort(
        key=lambda row: (
            STREAM_KIND_PRIORITY.get(str(row["stream_kind"]), 99),
            str(row["stream_kind"]),
            str(row["label"]),
        )
    )

    total_actual_bytes = sum(int(row["actual_bytes"]) for row in rows)
    total_symbol_count = sum(int(row["symbol_count"]) for row in rows)
    total_entropy_bits = sum(float(row["entropy_floor_bits"]) for row in rows)
    total_best_conditional_bits = sum(float(row["best_entropy_floor_bits"]) for row in rows)
    total_entropy_bytes = total_entropy_bits / 8.0
    total_best_conditional_bytes = total_best_conditional_bits / 8.0
    evidence_grades = sorted({str(row["evidence_grade"]) for row in rows})

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
        "source_label": str(source_label),
        "evidence_grade": DEFAULT_EVIDENCE_GRADE,
        "stream_evidence_grades": evidence_grades,
        "stream_count": len(rows),
        "total_symbol_count": total_symbol_count,
        "total_actual_bytes": total_actual_bytes,
        "total_entropy_floor_bits": _round_float(total_entropy_bits),
        "total_entropy_floor_bytes": _round_float(total_entropy_bytes),
        "total_entropy_floor_bytes_ceil": _ceil_bytes(total_entropy_bits),
        "total_gap_to_entropy_floor_bytes": _round_float(total_actual_bytes - total_entropy_bytes),
        "total_best_conditional_entropy_floor_bits": _round_float(total_best_conditional_bits),
        "total_best_conditional_entropy_floor_bytes": _round_float(total_best_conditional_bytes),
        "total_best_conditional_entropy_floor_bytes_ceil": _ceil_bytes(total_best_conditional_bits),
        "total_gap_to_best_conditional_floor_bytes": _round_float(
            total_actual_bytes - total_best_conditional_bytes
        ),
        "streams": rows,
        "opportunity_ranking": _opportunity_ranking(rows),
    }


def render_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a deterministic markdown summary for a decomposition manifest."""

    streams = manifest.get("streams")
    if not isinstance(streams, list):
        raise EntropyRateDecompositionError("manifest streams must be a list")

    lines = [
        "# Entropy-Rate Decomposition",
        "",
        f"- planning_only: `{_bool_text(manifest.get('planning_only') is True)}`",
        f"- score_claim: `{_bool_text(manifest.get('score_claim') is True)}`",
        f"- ready_for_exact_eval_dispatch: `{_bool_text(manifest.get('ready_for_exact_eval_dispatch') is True)}`",
        f"- evidence_grade: `{manifest.get('evidence_grade')}`",
        f"- score_evidence_grade: `{manifest.get('score_evidence_grade')}`",
        f"- total_actual_bytes: `{manifest.get('total_actual_bytes')}`",
        f"- total_entropy_floor_bytes: `{manifest.get('total_entropy_floor_bytes')}`",
        f"- total_gap_to_entropy_floor_bytes: `{manifest.get('total_gap_to_entropy_floor_bytes')}`",
        "",
        "| stream | kind | actual bytes | entropy floor bytes | gap bytes | best conditional | blockers |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    blockers = ",".join(str(item) for item in manifest.get("dispatch_blockers") or [])
    for row in streams:
        if not isinstance(row, Mapping):
            raise EntropyRateDecompositionError("manifest stream rows must be objects")
        best_model = row.get("best_conditional_model_label") or ""
        lines.append(
            "| {label} | `{kind}` | {actual} | {floor} | {gap} | `{best}` | {blockers} |".format(
                label=row.get("label"),
                kind=row.get("stream_kind"),
                actual=row.get("actual_bytes"),
                floor=row.get("entropy_floor_bytes"),
                gap=row.get("gap_to_entropy_floor_bytes"),
                best=best_model,
                blockers=blockers,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _build_stream_row(
    stream: Mapping[str, Any],
    *,
    default_evidence_grade: str,
) -> dict[str, Any]:
    label = _required_label(stream.get("label"), "stream label")
    stream_kind = _required_label(
        stream.get("stream_kind", stream.get("kind", "unknown")),
        f"{label}.stream_kind",
    ).lower()
    actual_bytes = _as_nonnegative_int(stream.get("actual_bytes"), f"{label}.actual_bytes")
    counts = _normalize_counts(
        stream.get("symbol_counts", stream.get("counts")),
        f"{label}.symbol_counts",
    )
    count_by_symbol = {str(row["symbol"]): int(row["count"]) for row in counts}
    entropy = _entropy_floor(counts)
    conditional_models = _conditional_models(
        stream.get("conditional_groups"),
        base_counts=count_by_symbol,
        stream_label=label,
    )
    best_model = min(
        conditional_models,
        key=lambda model: (
            float(model["conditional_entropy_floor_bits"]),
            str(model["label"]),
        ),
        default=None,
    )
    best_bits = (
        float(best_model["conditional_entropy_floor_bits"])
        if best_model is not None
        else float(entropy["entropy_floor_bits"])
    )
    best_bytes = best_bits / 8.0
    entropy_bytes = float(entropy["entropy_floor_bits"]) / 8.0
    evidence_grade = str(stream.get("evidence_grade") or default_evidence_grade)
    if not evidence_grade:
        raise EntropyRateDecompositionError(f"{label}.evidence_grade must be nonempty")

    return {
        "label": label,
        "stream_kind": stream_kind,
        "source": str(stream.get("source") or ""),
        "evidence_grade": evidence_grade,
        "planning_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "actual_bytes": actual_bytes,
        "symbol_count": int(entropy["symbol_count"]),
        "alphabet_size": int(entropy["alphabet_size"]),
        "positive_symbol_count": int(entropy["positive_symbol_count"]),
        "actual_bytes_per_symbol": _round_float(actual_bytes / int(entropy["symbol_count"])),
        "entropy_bits_per_symbol": entropy["entropy_bits_per_symbol"],
        "entropy_floor_bits": entropy["entropy_floor_bits"],
        "entropy_floor_bytes": entropy["entropy_floor_bytes"],
        "entropy_floor_bytes_ceil": entropy["entropy_floor_bytes_ceil"],
        "gap_to_entropy_floor_bytes": _round_float(actual_bytes - entropy_bytes),
        "gap_to_entropy_floor_bytes_ceil": actual_bytes - int(entropy["entropy_floor_bytes_ceil"]),
        "conditional_model_count": len(conditional_models),
        "conditional_models": conditional_models,
        "best_conditional_model_label": best_model["label"] if best_model is not None else "",
        "best_entropy_floor_bits": _round_float(best_bits),
        "best_entropy_floor_bytes": _round_float(best_bytes),
        "best_entropy_floor_bytes_ceil": _ceil_bytes(best_bits),
        "gap_to_best_conditional_floor_bytes": _round_float(actual_bytes - best_bytes),
        "conditional_gain_over_unconditional_bytes": _round_float(entropy_bytes - best_bytes),
        "recommended_next_action": STREAM_KIND_ACTIONS.get(
            stream_kind,
            "decompose stream grammar before codec action",
        ),
        "symbol_counts": counts,
    }


def _conditional_models(
    value: Any,
    *,
    base_counts: Mapping[str, int],
    stream_label: str,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        raise EntropyRateDecompositionError(f"{stream_label}.conditional_groups must be an object")
    models: list[dict[str, Any]] = []
    total_symbols = sum(int(count) for count in base_counts.values())
    for model_label_raw, groups_raw in sorted(value.items(), key=lambda item: str(item[0])):
        model_label = _required_label(model_label_raw, f"{stream_label}.conditional_model_label")
        if not isinstance(groups_raw, Mapping) or not groups_raw:
            raise EntropyRateDecompositionError(f"{stream_label}.{model_label} groups must be a nonempty object")
        aggregate: Counter[str] = Counter()
        group_rows: list[dict[str, Any]] = []
        conditional_bits = 0.0
        for group_label_raw, group_counts_raw in sorted(groups_raw.items(), key=lambda item: str(item[0])):
            group_label = _required_label(group_label_raw, f"{stream_label}.{model_label}.group_label")
            group_counts = _normalize_counts(
                group_counts_raw,
                f"{stream_label}.{model_label}.{group_label}.counts",
            )
            for row in group_counts:
                aggregate[str(row["symbol"])] += int(row["count"])
            group_entropy = _entropy_floor(group_counts)
            conditional_bits += float(group_entropy["entropy_floor_bits"])
            group_rows.append(
                {
                    "label": group_label,
                    "symbol_count": group_entropy["symbol_count"],
                    "entropy_bits_per_symbol": group_entropy["entropy_bits_per_symbol"],
                    "entropy_floor_bits": group_entropy["entropy_floor_bits"],
                    "entropy_floor_bytes": group_entropy["entropy_floor_bytes"],
                    "symbol_counts": group_counts,
                }
            )
        if dict(sorted(aggregate.items())) != dict(sorted(base_counts.items())):
            raise EntropyRateDecompositionError(
                f"{stream_label}.{model_label} conditional counts must sum to base symbol counts"
            )
        models.append(
            {
                "label": model_label,
                "group_count": len(group_rows),
                "symbol_count": total_symbols,
                "conditional_entropy_bits_per_symbol": _round_float(conditional_bits / total_symbols),
                "conditional_entropy_floor_bits": _round_float(conditional_bits),
                "conditional_entropy_floor_bytes": _round_float(conditional_bits / 8.0),
                "conditional_entropy_floor_bytes_ceil": _ceil_bytes(conditional_bits),
                "planning_only": True,
                "score_claim": False,
                "dispatch_attempted": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_blockers": list(DISPATCH_BLOCKERS),
                "groups": group_rows,
            }
        )
    return models


def _entropy_floor(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    symbol_count = sum(int(row["count"]) for row in records)
    bits_per_symbol = _entropy_bits_per_symbol_from_records(records)
    bits = symbol_count * bits_per_symbol
    return {
        "symbol_count": symbol_count,
        "alphabet_size": len(records),
        "positive_symbol_count": sum(1 for row in records if int(row["count"]) > 0),
        "entropy_bits_per_symbol": _round_float(bits_per_symbol),
        "entropy_floor_bits": _round_float(bits),
        "entropy_floor_bytes": _round_float(bits / 8.0),
        "entropy_floor_bytes_ceil": _ceil_bytes(bits),
    }


def _entropy_bits_per_symbol_from_records(records: Sequence[Mapping[str, Any]]) -> float:
    total = sum(int(row["count"]) for row in records)
    if total <= 0:
        raise EntropyRateDecompositionError("counts must have positive total")
    entropy = 0.0
    for row in records:
        count = int(row["count"])
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def _normalize_counts(value: Any, context: str) -> list[dict[str, Any]]:
    if value is None:
        raise EntropyRateDecompositionError(f"{context} missing")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if isinstance(value, Mapping):
        if not value:
            raise EntropyRateDecompositionError(f"{context} must be nonempty")
        iterable = sorted(value.items(), key=lambda item: str(item[0]))
        for symbol_raw, count_raw in iterable:
            symbol = _required_label(symbol_raw, f"{context}.symbol")
            if symbol in seen:
                raise EntropyRateDecompositionError(f"{context} has duplicate symbol after string conversion: {symbol}")
            seen.add(symbol)
            rows.append({"symbol": symbol, "count": _as_nonnegative_int(count_raw, f"{context}.{symbol}")})
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise EntropyRateDecompositionError(f"{context} must be nonempty")
        for index, count_raw in enumerate(value):
            rows.append({"symbol": str(index), "count": _as_nonnegative_int(count_raw, f"{context}.{index}")})
    else:
        raise EntropyRateDecompositionError(f"{context} must be an object or sequence")

    if sum(int(row["count"]) for row in rows) <= 0:
        raise EntropyRateDecompositionError(f"{context} must have positive total")
    return rows


def _opportunity_ranking(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for row in rows:
        ranked.append(
            {
                "label": row["label"],
                "stream_kind": row["stream_kind"],
                "actual_bytes": row["actual_bytes"],
                "gap_to_entropy_floor_bytes": row["gap_to_entropy_floor_bytes"],
                "gap_to_best_conditional_floor_bytes": row["gap_to_best_conditional_floor_bytes"],
                "conditional_gain_over_unconditional_bytes": row["conditional_gain_over_unconditional_bytes"],
                "recommended_next_action": row["recommended_next_action"],
                "ready_for_exact_eval_dispatch": False,
                "dispatch_blockers": list(DISPATCH_BLOCKERS),
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row["gap_to_best_conditional_floor_bytes"]),
            -float(row["conditional_gain_over_unconditional_bytes"]),
            str(row["stream_kind"]),
            str(row["label"]),
        )
    )
    return ranked


def _as_nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EntropyRateDecompositionError(f"{context} must be a non-negative integer")
    if value < 0:
        raise EntropyRateDecompositionError(f"{context} must be a non-negative integer")
    return int(value)


def _required_label(value: Any, context: str) -> str:
    label = str(value or "").strip()
    if not label:
        raise EntropyRateDecompositionError(f"{context} must be nonempty")
    if any(ord(char) > 127 for char in label):
        raise EntropyRateDecompositionError(f"{context} must be ASCII")
    return label


def _ceil_bytes(bits: float) -> int:
    return math.ceil(max(0.0, float(bits)) / 8.0 - 1e-12)


def _round_float(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if abs(rounded) < 5e-13 else rounded


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "DEFAULT_EVIDENCE_GRADE",
    "DISPATCH_BLOCKERS",
    "SCHEMA_VERSION",
    "SCORE_EVIDENCE_GRADE",
    "TOOL_NAME",
    "EntropyRateDecompositionError",
    "build_entropy_rate_decomposition",
    "entropy_bits_per_symbol",
    "render_markdown",
]
