# SPDX-License-Identifier: MIT
"""Planning-only DQS1 decoder-q pair-set acquisition."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

from tac.optimization.decoder_q_selective_runtime_packet import (
    CONTEST_RATE_DENOMINATOR_BYTES,
    FALSE_AUTHORITY,
    FEC6_PAIR_COUNT,
    affected_frames_for_pairs,
    choose_dqs1_pair_encoding,
)

SCHEMA = "decoder_q_pairset_acquisition.v1"
CANDIDATE_SCHEMA = "decoder_q_pairset_acquisition_candidate.v1"
SOURCE_SCHEMA = "decoder_q_selective_selector_pareto.v1"
TOOL = "tac.optimization.decoder_q_pairset_acquisition"

FALSE_ACQUISITION_AUTHORITY: dict[str, bool] = {
    **FALSE_AUTHORITY,
    "dispatch_attempted": False,
}


class DecoderQPairsetAcquisitionError(ValueError):
    """Raised when pair-set acquisition planning would lose custody or authority."""


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(payload), encoding="utf-8")


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DecoderQPairsetAcquisitionError(f"{path}: expected JSON object")
    return payload


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if payload.get(key) is not False:
            raise DecoderQPairsetAcquisitionError(f"{label} {key} must be explicit false")


def _require_acquisition_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for key in FALSE_ACQUISITION_AUTHORITY:
        if payload.get(key) is not False:
            raise DecoderQPairsetAcquisitionError(f"{label} {key} must be explicit false")


def _as_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DecoderQPairsetAcquisitionError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise DecoderQPairsetAcquisitionError(f"{label} must be finite")
    return result


def _as_optional_float(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return _as_float(value, label=label)


def _as_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise DecoderQPairsetAcquisitionError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DecoderQPairsetAcquisitionError(f"{label} must be an integer") from exc
    if result != value and not (isinstance(value, str) and str(result) == value):
        raise DecoderQPairsetAcquisitionError(f"{label} must be integral")
    return result


def _canonical_pair_indices(pair_indices: Sequence[Any], *, label: str) -> list[int]:
    pairs = [
        _as_int(pair, label=f"{label}[{index}]")
        for index, pair in enumerate(pair_indices)
    ]
    if not pairs:
        raise DecoderQPairsetAcquisitionError(f"{label} must not be empty")
    if len(set(pairs)) != len(pairs):
        raise DecoderQPairsetAcquisitionError(f"{label} contains duplicates")
    for pair in pairs:
        if not 0 <= pair < FEC6_PAIR_COUNT:
            raise DecoderQPairsetAcquisitionError(f"{label} pair index out of range: {pair}")
    return sorted(pairs)


def _rank_order(raw: Any, *, fallback_pairs: Sequence[int], label: str) -> list[int]:
    if not isinstance(raw, list):
        return list(fallback_pairs)
    pairs = [
        _as_int(pair, label=f"{label}[{index}]")
        for index, pair in enumerate(raw)
    ]
    if len(set(pairs)) != len(pairs) or set(pairs) != set(fallback_pairs):
        return list(fallback_pairs)
    for pair in pairs:
        if not 0 <= pair < FEC6_PAIR_COUNT:
            return list(fallback_pairs)
    return pairs


def _default_positive_ints(max_count: int) -> list[int]:
    coarse = (1, 2, 4, 8, 12, 16, 24, 32, max_count)
    dense_tail = tuple(range(max(1, min(max_count, 32) - 6), min(max_count, 32) + 1, 2))
    near_full = tuple(range(max(1, max_count - 2), max_count + 1))
    return sorted({value for value in (*coarse, *dense_tail, *near_full) if 0 < value <= max_count})


def _parse_positive_ints(values: Sequence[int] | None, *, max_count: int) -> list[int]:
    if max_count <= 0:
        raise DecoderQPairsetAcquisitionError("positive integer selection requires max_count > 0")
    if values is None:
        return _default_positive_ints(max_count)
    out = sorted({int(value) for value in values if 0 < int(value) <= max_count})
    if not out:
        raise DecoderQPairsetAcquisitionError("at least one positive selection size is required")
    return out


def _payload_stats(pair_indices: Sequence[int], *, frame_policy: str) -> dict[str, Any]:
    pairs = _canonical_pair_indices(pair_indices, label="selected_pair_indices")
    encoding = choose_dqs1_pair_encoding(pairs)
    selected = encoding["selected"]
    payload_bytes = int(selected["descriptor_bytes"])
    return {
        "pair_indices": pairs,
        "selected_pair_count": len(pairs),
        "affected_frame_indices": affected_frames_for_pairs(pairs, frame_policy=frame_policy),
        "payload_bytes": payload_bytes,
        "pair_encoding": selected["pair_encoding"],
        "pair_index_payload_bytes": int(selected["pair_index_payload_bytes"]),
        "pair_encoding_candidates": encoding["candidates"],
        "rate_delta": 25.0 * payload_bytes / CONTEST_RATE_DENOMINATOR_BYTES,
    }


def _predicted_score_mean(row: Mapping[str, Any], *, label: str) -> float | None:
    if row.get("predicted_score_mean") is not None:
        return _as_optional_float(row.get("predicted_score_mean"), label=f"{label} predicted_score_mean")
    estimate = row.get("exact_cpu_calibrated_estimate")
    if isinstance(estimate, Mapping) and estimate.get("predicted_score") is not None:
        return _as_optional_float(
            estimate.get("predicted_score"),
            label=f"{label} exact_cpu_calibrated_estimate.predicted_score",
        )
    return None


def _selector_rows(selector_pareto: Mapping[str, Any]) -> list[dict[str, Any]]:
    if selector_pareto.get("schema") != SOURCE_SCHEMA:
        raise DecoderQPairsetAcquisitionError("selector pareto schema mismatch")
    _require_false_authority(selector_pareto, label="selector pareto")
    rows = selector_pareto.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise DecoderQPairsetAcquisitionError("selector pareto candidates[] missing")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DecoderQPairsetAcquisitionError(f"selector candidate {index} must be object")
        _require_false_authority(row, label=f"selector candidate {index}")
        if row.get("schema") != "decoder_q_selective_selector_candidate.v1":
            raise DecoderQPairsetAcquisitionError(f"selector candidate {index} schema mismatch")
        selector_id = str(row.get("selector_id") or f"selector_{index:04d}")
        if selector_id in seen_ids:
            raise DecoderQPairsetAcquisitionError(f"duplicate selector_id: {selector_id}")
        seen_ids.add(selector_id)
        raw_pairs = row.get("selected_pair_indices")
        if not isinstance(raw_pairs, list):
            raise DecoderQPairsetAcquisitionError(f"{selector_id} selected_pair_indices missing")
        pairs = _canonical_pair_indices(raw_pairs, label=f"{selector_id} selected_pair_indices")
        exact_cpu_calibrated_estimate = row.get("exact_cpu_calibrated_estimate")
        if isinstance(exact_cpu_calibrated_estimate, Mapping):
            _require_false_authority(
                exact_cpu_calibrated_estimate,
                label=f"{selector_id} exact_cpu_calibrated_estimate",
            )
            exact_cpu_calibrated_estimate = dict(exact_cpu_calibrated_estimate)
        else:
            exact_cpu_calibrated_estimate = None
        normalized.append(
            {
                "source_index": index,
                "selector_id": selector_id,
                "selector_kind": str(row.get("selector_kind") or "unknown_selector"),
                "selector_rank": _as_int(row.get("selector_rank", index + 1), label=f"{selector_id} selector_rank"),
                "selected_pair_indices": pairs,
                "rank_order_pair_indices": _rank_order(
                    row.get("rank_order_pair_indices"),
                    fallback_pairs=pairs,
                    label=f"{selector_id} rank_order_pair_indices",
                ),
                "source_payload_bytes": (
                    _as_int(row.get("payload_bytes"), label=f"{selector_id} payload_bytes")
                    if row.get("payload_bytes") is not None
                    else None
                ),
                "predicted_score_mean": _predicted_score_mean(row, label=selector_id),
                "exact_cpu_calibrated_estimate": exact_cpu_calibrated_estimate,
            }
        )
    return normalized


def _recommended_selector_id(selector_pareto: Mapping[str, Any]) -> str | None:
    summary = selector_pareto.get("summary")
    if isinstance(summary, Mapping) and summary.get("recommended_selector_id"):
        return str(summary["recommended_selector_id"])
    return None


def _rank_selectors(rows: Sequence[dict[str, Any]], selector_pareto: Mapping[str, Any]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[int, float, int, str]:
        predicted = row.get("predicted_score_mean")
        return (
            int(row["selector_rank"]),
            float(predicted) if predicted is not None else math.inf,
            int(row["source_payload_bytes"] or 0),
            str(row["selector_id"]),
        )

    ranked = sorted(rows, key=sort_key)
    recommended = _recommended_selector_id(selector_pareto)
    if recommended is None:
        return ranked
    best = [row for row in ranked if row["selector_id"] == recommended]
    if not best:
        return ranked
    return best + [row for row in ranked if row["selector_id"] != recommended]


def _slug(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in text)
    return "_".join(part for part in cleaned.split("_") if part)[:48] or "selector"


def _diversity_score(pair_indices: Sequence[int]) -> float:
    pairs = sorted({int(pair) for pair in pair_indices})
    if len(pairs) <= 1:
        return 0.0
    gaps = [b - a for a, b in pairwise(pairs)]
    min_gap = min(gaps) / float(FEC6_PAIR_COUNT - 1)
    pairwise_total = 0
    pairwise_count = 0
    for left, right in combinations(pairs, 2):
        pairwise_total += right - left
        pairwise_count += 1
    mean_pairwise = pairwise_total / float(pairwise_count * (FEC6_PAIR_COUNT - 1))
    return 0.35 * min_gap + 0.65 * mean_pairwise


def _spaced_subset(universe: Sequence[int], count: int) -> list[int]:
    pairs = sorted({int(pair) for pair in universe})
    if count <= 0 or count > len(pairs):
        raise DecoderQPairsetAcquisitionError("spaced subset count outside universe")
    if count == 1:
        return [pairs[len(pairs) // 2]]
    positions = [
        round(index * (len(pairs) - 1) / (count - 1))
        for index in range(count)
    ]
    selected = [pairs[position] for position in positions]
    selected_set = set(selected)
    if len(selected_set) < count:
        for pair in pairs:
            if pair not in selected_set:
                selected.append(pair)
                selected_set.add(pair)
            if len(selected_set) == count:
                break
    return sorted(selected_set)


def _acquisition_score(
    *,
    predicted_score_mean: float | None,
    payload_bytes: int,
    diversity_score: float,
    diversity_weight: float,
) -> float:
    rate_delta = 25.0 * payload_bytes / CONTEST_RATE_DENOMINATOR_BYTES
    predicted_component = -predicted_score_mean if predicted_score_mean is not None else 0.0
    return predicted_component - rate_delta + diversity_weight * diversity_score


def _candidate_row(
    *,
    acquisition_id: str,
    selector_kind: str,
    pair_indices: Sequence[int],
    source_selector: Mapping[str, Any],
    frame_policy: str,
    diversity_weight: float,
    source_selector_ids: Sequence[str] | None = None,
    operation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats = _payload_stats(pair_indices, frame_policy=frame_policy)
    predicted_score_mean = source_selector.get("predicted_score_mean")
    predicted = float(predicted_score_mean) if predicted_score_mean is not None else None
    diversity = _diversity_score(stats["pair_indices"])
    source_selector_pairs = list(source_selector.get("selected_pair_indices") or [])
    candidate_matches_source_selector = (
        tuple(stats["pair_indices"]) == tuple(_canonical_pair_indices(
            source_selector_pairs,
            label=f"{source_selector['selector_id']} selected_pair_indices",
        ))
        if source_selector_pairs
        else False
    )
    row: dict[str, Any] = {
        "schema": CANDIDATE_SCHEMA,
        "selector_id": acquisition_id,
        "acquisition_id": acquisition_id,
        "selector_kind": selector_kind,
        "source_selector_id": source_selector["selector_id"],
        "source_selector_kind": source_selector["selector_kind"],
        "source_selector_ids": list(source_selector_ids or [str(source_selector["selector_id"])]),
        "selected_pair_indices": stats["pair_indices"],
        "selected_pair_count": stats["selected_pair_count"],
        "affected_frame_count": len(stats["affected_frame_indices"]),
        "payload_bytes": stats["payload_bytes"],
        "payload_byte_estimate": stats["payload_bytes"],
        "payload_byte_estimate_kind": "dqs1_descriptor_bytes_from_pair_index_encoding",
        "pair_encoding": stats["pair_encoding"],
        "pair_index_payload_bytes": stats["pair_index_payload_bytes"],
        "pair_encoding_candidates": stats["pair_encoding_candidates"],
        "rate_delta": stats["rate_delta"],
        "diversity_score": diversity,
        "acquisition_score": _acquisition_score(
            predicted_score_mean=predicted,
            payload_bytes=int(stats["payload_bytes"]),
            diversity_score=diversity,
            diversity_weight=diversity_weight,
        ),
        "acquisition_score_orientation": "higher_is_better",
        "candidate_generation_only": True,
        "local_only": True,
        "allowed_use": "local_pairset_acquisition_planning_only_no_dispatch_authority",
        **FALSE_ACQUISITION_AUTHORITY,
    }
    if operation is not None:
        row["acquisition_operation"] = operation
    if predicted is not None:
        row["predicted_score_mean"] = predicted
        row["predicted_score_source"] = (
            "source_selector_candidate_specific_non_authoritative"
            if candidate_matches_source_selector
            else "source_selector_inherited_non_authoritative"
        )
        row["predicted_score_scope"] = (
            "candidate_specific"
            if candidate_matches_source_selector
            else "source_selector_scope_not_child_candidate"
        )
    exact_estimate = source_selector.get("exact_cpu_calibrated_estimate")
    if isinstance(exact_estimate, Mapping):
        if candidate_matches_source_selector:
            row["exact_cpu_calibrated_estimate"] = dict(exact_estimate)
            row["exact_cpu_calibrated_estimate_scope"] = "candidate_specific"
        else:
            row["source_selector_exact_cpu_calibrated_estimate"] = dict(exact_estimate)
            row["source_selector_exact_cpu_calibrated_estimate_scope"] = (
                "source_selector_scope_not_child_candidate"
            )
    _require_acquisition_false_authority(row, label=acquisition_id)
    return row


def _bounded_drop_two_pairs(rank_order_pairs: Sequence[int], *, limit: int) -> list[tuple[int, int]]:
    if limit <= 0:
        return []
    tail_first = list(reversed(rank_order_pairs))
    out: list[tuple[int, int]] = []
    for left, right in combinations(tail_first, 2):
        out.append((left, right))
        if len(out) >= limit:
            break
    return out


def build_decoder_q_pairset_acquisition_plan(
    selector_pareto: dict[str, Any],
    *,
    frame_policy: str = "pair_all_frames",
    prefix_ks: Sequence[int] | None = None,
    diversity_ks: Sequence[int] | None = None,
    include_drop_one: bool = True,
    max_drop_two: int = 128,
    max_swap_in: int = 32,
    diversity_weight: float = 0.15,
) -> dict[str, Any]:
    """Build local-only DQS1 pair-set acquisition candidates from selector Pareto rows."""

    selectors = _rank_selectors(_selector_rows(selector_pareto), selector_pareto)
    best = selectors[0]
    best_order = list(best["rank_order_pair_indices"])
    best_pair_set = set(best_order)
    if not best_order:
        raise DecoderQPairsetAcquisitionError("best selector has no selected pairs")
    diversity_weight = _as_float(diversity_weight, label="diversity_weight")
    if diversity_weight < 0.0:
        raise DecoderQPairsetAcquisitionError("diversity_weight must be non-negative")

    rows: list[dict[str, Any]] = []
    seen_pair_sets: set[tuple[int, ...]] = set()

    def add_row(
        acquisition_id: str,
        selector_kind: str,
        pair_indices: Sequence[int],
        source_selector: Mapping[str, Any],
        *,
        source_selector_ids: Sequence[str] | None = None,
        operation: dict[str, Any] | None = None,
    ) -> None:
        key = tuple(_canonical_pair_indices(pair_indices, label=f"{acquisition_id} selected_pair_indices"))
        if key in seen_pair_sets:
            return
        seen_pair_sets.add(key)
        rows.append(
            _candidate_row(
                acquisition_id=acquisition_id,
                selector_kind=selector_kind,
                pair_indices=key,
                source_selector=source_selector,
                frame_policy=frame_policy,
                diversity_weight=diversity_weight,
                source_selector_ids=source_selector_ids,
                operation=operation,
            )
        )

    for k in _parse_positive_ints(prefix_ks, max_count=len(best_order)):
        add_row(
            f"pairset_prefix_k{k:03d}",
            "prefix_variant",
            best_order[:k],
            best,
            operation={"op": "prefix", "k": k},
        )

    if include_drop_one and len(best_order) > 1:
        for rank, pair in enumerate(best_order, start=1):
            selected = [candidate for candidate in best_order if candidate != pair]
            add_row(
                f"pairset_drop_one_rank{rank:03d}_pair{pair:04d}",
                "drop_one_from_best",
                selected,
                best,
                operation={"op": "drop_one", "dropped_pair_index": pair, "dropped_pair_rank": rank},
            )

    if len(best_order) > 2:
        rank_by_pair = {pair: rank for rank, pair in enumerate(best_order, start=1)}
        for left, right in _bounded_drop_two_pairs(best_order, limit=max_drop_two):
            selected = [candidate for candidate in best_order if candidate not in {left, right}]
            left_rank = rank_by_pair[left]
            right_rank = rank_by_pair[right]
            add_row(
                f"pairset_drop_two_r{left_rank:03d}_{right_rank:03d}_p{left:04d}_{right:04d}",
                "drop_two_from_best",
                selected,
                best,
                operation={
                    "op": "drop_two",
                    "dropped_pair_indices": sorted([left, right]),
                    "dropped_pair_ranks": sorted([left_rank, right_rank]),
                },
            )

    swap_count = 0
    if max_swap_in > 0 and len(best_order) > 1:
        drop_pair = best_order[-1]
        drop_rank = len(best_order)
        for selector in selectors[1:]:
            alternatives = [
                pair
                for pair in selector["rank_order_pair_indices"]
                if int(pair) not in best_pair_set
            ]
            for alt_pair in alternatives:
                selected = [pair for pair in best_order if pair != drop_pair] + [int(alt_pair)]
                add_row(
                    (
                        f"pairset_swap_{_slug(str(selector['selector_id']))}"
                        f"_in{int(alt_pair):04d}_drop{drop_pair:04d}"
                    ),
                    "swap_in_alternative",
                    selected,
                    selector,
                    source_selector_ids=[str(best["selector_id"]), str(selector["selector_id"])],
                    operation={
                        "op": "swap_in",
                        "dropped_pair_index": drop_pair,
                        "dropped_pair_rank": drop_rank,
                        "inserted_pair_index": int(alt_pair),
                    },
                )
                swap_count += 1
                if swap_count >= max_swap_in:
                    break
            if swap_count >= max_swap_in:
                break

    universe = sorted({pair for selector in selectors for pair in selector["selected_pair_indices"]})
    diversity_max = min(len(best_order), len(universe))
    if diversity_max > 0:
        for k in _parse_positive_ints(diversity_ks, max_count=diversity_max):
            selected = _spaced_subset(universe, k)
            add_row(
                f"pairset_diversity_k{k:03d}",
                "diversity_spaced",
                selected,
                best,
                source_selector_ids=[str(selector["selector_id"]) for selector in selectors],
                operation={"op": "diversity_spaced", "k": k, "universe_pair_count": len(universe)},
            )

    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["acquisition_score"]),
            int(row["payload_bytes"]),
            str(row["selector_id"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["acquisition_rank"] = rank
        row["rank_kind"] = "sorted_local_acquisition_rank"

    summary = {
        "candidate_count": len(ranked),
        "recommended_acquisition_id": ranked[0]["acquisition_id"] if ranked else None,
        "prefix_candidate_count": sum(1 for row in ranked if row["selector_kind"] == "prefix_variant"),
        "drop_one_candidate_count": sum(1 for row in ranked if row["selector_kind"] == "drop_one_from_best"),
        "drop_two_candidate_count": sum(1 for row in ranked if row["selector_kind"] == "drop_two_from_best"),
        "swap_in_candidate_count": sum(1 for row in ranked if row["selector_kind"] == "swap_in_alternative"),
        "diversity_candidate_count": sum(1 for row in ranked if row["selector_kind"] == "diversity_spaced"),
    }

    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": TOOL,
        "source_schema": SOURCE_SCHEMA,
        "evidence_grade": "planning-only local pair-set acquisition",
        "allowed_use": "local_candidate_pairset_planning_only_requires_materialization_and_exact_eval",
        "candidate_generation_only": True,
        "local_only": True,
        "frame_policy": frame_policy,
        "selection_policy": {
            "best_selector_id": best["selector_id"],
            "best_selector_kind": best["selector_kind"],
            "prefix_ks": _parse_positive_ints(prefix_ks, max_count=len(best_order)),
            "diversity_ks": _parse_positive_ints(diversity_ks, max_count=diversity_max) if diversity_max else [],
            "default_k_policy": (
                "coarse_global_sweep_plus_dense_tail_for_observation_response_interpolation"
            ),
            "include_drop_one": include_drop_one,
            "max_drop_two": max_drop_two,
            "max_swap_in": max_swap_in,
            "diversity_weight": diversity_weight,
            "ranking": "acquisition_score_desc_payload_bytes_asc_selector_id_asc",
        },
        "source_selector_summary": {
            "selector_candidate_count": len(selectors),
            "best_selected_pair_count": len(best_order),
            "best_selected_pair_indices": sorted(best_order),
        },
        "summary": summary,
        "candidates": ranked,
        **FALSE_ACQUISITION_AUTHORITY,
    }


build_pairset_acquisition_plan = build_decoder_q_pairset_acquisition_plan


__all__ = [
    "CANDIDATE_SCHEMA",
    "FALSE_ACQUISITION_AUTHORITY",
    "SCHEMA",
    "SOURCE_SCHEMA",
    "TOOL",
    "DecoderQPairsetAcquisitionError",
    "build_decoder_q_pairset_acquisition_plan",
    "build_pairset_acquisition_plan",
    "dumps_json",
    "load_json_object",
    "write_json",
]
