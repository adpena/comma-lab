# SPDX-License-Identifier: MIT
"""Planning-only DQS1 decoder-q pair-set acquisition."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
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
from tac.optimization.dqs1_materializer_feedback_bridge import (
    DQS1_OBSERVATION_SOURCE_SCHEMA,
    DQS1_OBSERVATION_SWEEP_CONFIG_ID,
)
from tac.optimization.pair_frame_scorer_geometry_lattice import (
    REQUEST_SCHEMA as PAIR_FRAME_GEOMETRY_REQUEST_SCHEMA,
)
from tac.optimization.pair_frame_scorer_geometry_lattice import (
    SCHEMA as PAIR_FRAME_GEOMETRY_LATTICE_SCHEMA,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

SCHEMA = "decoder_q_pairset_acquisition.v1"
CANDIDATE_SCHEMA = "decoder_q_pairset_acquisition_candidate.v1"
EUREKA_EXPANSION_SCHEMA = "decoder_q_pairset_acquisition_eureka_expansion.v1"
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
    source_payload_bytes = source_selector.get("source_payload_bytes")
    source_payload = (
        int(source_payload_bytes)
        if isinstance(source_payload_bytes, int) and not isinstance(source_payload_bytes, bool)
        else None
    )
    payload_delta = (
        None if source_payload is None else int(stats["payload_bytes"]) - source_payload
    )
    rate_score_delta = (
        None
        if payload_delta is None
        else 25.0 * payload_delta / CONTEST_RATE_DENOMINATOR_BYTES
    )
    saved_bytes = None if payload_delta is None else max(0, -payload_delta)
    repair_score_budget = (
        None if rate_score_delta is None else max(0.0, -rate_score_delta)
    )
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
        "source_selector_payload_bytes": source_payload,
        "payload_bytes_delta_vs_source_selector": payload_delta,
        "pair_encoding": stats["pair_encoding"],
        "pair_index_payload_bytes": stats["pair_index_payload_bytes"],
        "pair_encoding_candidates": stats["pair_encoding_candidates"],
        "rate_delta": stats["rate_delta"],
        "rate_score_delta_vs_source_selector": rate_score_delta,
        "distortion_repair_budget_from_rate_savings": {
            "schema": "decoder_q_pairset_rate_saved_distortion_repair_budget.v1",
            "active": bool(repair_score_budget and repair_score_budget > 0.0),
            "source_selector_payload_bytes": source_payload,
            "candidate_payload_bytes": int(stats["payload_bytes"]),
            "saved_bytes_vs_source_selector": saved_bytes,
            "score_budget": repair_score_budget,
            "segnet_distortion_budget_at_fixed_pose": (
                None if repair_score_budget is None else repair_score_budget / 100.0
            ),
            "posenet_score_term_budget_at_fixed_seg": repair_score_budget,
            "allowed_use": (
                "planning_only_rate_savings_budget_for_segnet_posenet_repair"
            ),
            "forbidden_use": "score_claim_or_distortion_authority",
            **FALSE_ACQUISITION_AUTHORITY,
        },
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


def _parse_drop_many_counts(
    values: Sequence[int] | None,
    *,
    max_count: int,
    eureka_active: bool,
) -> list[int]:
    if max_count <= 2:
        return []
    default = (3, 4, 6, 8) if eureka_active else ()
    raw_values = values if values is not None else default
    out = sorted({int(value) for value in raw_values if 2 < int(value) <= max_count})
    if values is not None and not out:
        raise DecoderQPairsetAcquisitionError(
            "drop_many_counts must include at least one count in 3..max_count"
        )
    return out


def _drop_many_id(
    *,
    dropped_pairs: Sequence[int],
    rank_by_pair: Mapping[int, int],
) -> str:
    pairs = sorted(int(pair) for pair in dropped_pairs)
    ranks = sorted(int(rank_by_pair[int(pair)]) for pair in pairs)
    prefix = f"pairset_drop_many_k{len(pairs):03d}"
    if len(pairs) <= 4:
        rank_part = "_".join(f"{rank:03d}" for rank in ranks)
        pair_part = "_".join(f"{pair:04d}" for pair in pairs)
        return f"{prefix}_r{rank_part}_p{pair_part}"
    digest = sha256(
        ",".join(str(pair) for pair in pairs).encode("utf-8")
    ).hexdigest()[:10]
    return f"{prefix}_h{digest}"


def _spaced_from_ordered(values: Sequence[int], count: int) -> tuple[int, ...]:
    if count <= 0 or count > len(values):
        return ()
    if count == 1:
        return (int(values[len(values) // 2]),)
    positions = [
        round(index * (len(values) - 1) / (count - 1))
        for index in range(count)
    ]
    selected = [int(values[position]) for position in positions]
    seen = set(selected)
    if len(seen) < count:
        for value in values:
            if int(value) in seen:
                continue
            selected.append(int(value))
            seen.add(int(value))
            if len(seen) == count:
                break
    return tuple(sorted(seen))


def _bounded_drop_many_sets(
    rank_order_pairs: Sequence[int],
    *,
    drop_counts: Sequence[int],
    limit: int,
) -> list[tuple[int, ...]]:
    if limit <= 0:
        return []
    tail_first = list(reversed([int(pair) for pair in rank_order_pairs]))
    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def add(values: Sequence[int]) -> None:
        if len(out) >= limit:
            return
        key = tuple(sorted({int(value) for value in values}))
        if len(key) != len(values) or key in seen:
            return
        seen.add(key)
        out.append(key)

    for drop_count in drop_counts:
        if drop_count <= 2 or drop_count >= len(tail_first):
            continue
        add(tail_first[:drop_count])
        tail_window = tail_first[: min(len(tail_first), max(drop_count * 3, drop_count + 4))]
        add(_spaced_from_ordered(tail_window, drop_count))
        for start in range(1, max(1, min(len(tail_first) - drop_count + 1, drop_count + 3))):
            add(tail_first[start : start + drop_count])
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
    return out


def _eureka_planner_hint_ids(eureka_planning: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(eureka_planning, Mapping):
        return []
    try:
        require_no_truthy_authority_fields(
            eureka_planning,
            context="decoder_q_pairset_acquisition.eureka_planning",
        )
    except ValueError as exc:
        raise DecoderQPairsetAcquisitionError(str(exc)) from exc
    hints = eureka_planning.get("planner_hints")
    if not isinstance(hints, list):
        return []
    hint_ids: list[str] = []
    for index, hint in enumerate(hints):
        if not isinstance(hint, Mapping):
            continue
        try:
            require_no_truthy_authority_fields(
                hint,
                context=f"decoder_q_pairset_acquisition.eureka_hint[{index}]",
            )
        except ValueError as exc:
            raise DecoderQPairsetAcquisitionError(str(exc)) from exc
        hint_id = str(hint.get("hint_id") or "")
        if hint_id:
            hint_ids.append(hint_id)
    return hint_ids


def _eureka_pairset_profile(
    eureka_planning: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not isinstance(eureka_planning, Mapping):
        return {}
    hints = eureka_planning.get("planner_hints")
    if not isinstance(hints, list):
        return {}
    for hint in hints:
        if not isinstance(hint, Mapping):
            continue
        profile = hint.get("pairset_acquisition_profile")
        if isinstance(profile, Mapping):
            return profile
    profile = eureka_planning.get("pairset_acquisition_profile")
    return profile if isinstance(profile, Mapping) else {}


def _profile_int(
    profile: Mapping[str, Any],
    key: str,
) -> int | None:
    value = profile.get(key)
    if value is None:
        return None
    return _as_int(value, label=f"eureka_pairset_profile.{key}")


def _profile_int_list(
    profile: Mapping[str, Any],
    key: str,
) -> list[int] | None:
    value = profile.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise DecoderQPairsetAcquisitionError(
            f"eureka_pairset_profile.{key} must be a list"
        )
    return [_as_int(item, label=f"eureka_pairset_profile.{key}[{index}]") for index, item in enumerate(value)]


def _observed_candidate_ids_from_dqs1_observations(
    observations: Sequence[Mapping[str, Any]] | None,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(observations or ()):
        if not isinstance(row, Mapping):
            continue
        try:
            require_no_truthy_authority_fields(
                row,
                context=f"decoder_q_pairset_acquisition.observation[{index}]",
            )
        except ValueError as exc:
            raise DecoderQPairsetAcquisitionError(str(exc)) from exc
        if (
            row.get("source_schema") != DQS1_OBSERVATION_SOURCE_SCHEMA
            or row.get("sweep_config_id") != DQS1_OBSERVATION_SWEEP_CONFIG_ID
        ):
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id and candidate_id not in seen:
            out.append(candidate_id)
            seen.add(candidate_id)
    return out


def _pair_frame_geometry_requests(
    lattice: Mapping[str, Any] | None,
    *,
    best_pair_set: set[int],
) -> list[dict[str, Any]]:
    if lattice is None:
        return []
    if lattice.get("schema") != PAIR_FRAME_GEOMETRY_LATTICE_SCHEMA:
        raise DecoderQPairsetAcquisitionError("pair-frame geometry lattice schema mismatch")
    try:
        require_no_truthy_authority_fields(
            lattice,
            context="decoder_q_pairset_acquisition.pair_frame_geometry_lattice",
        )
    except ValueError as exc:
        raise DecoderQPairsetAcquisitionError(str(exc)) from exc
    raw_requests = lattice.get("queue_executable_pairset_drop_requests")
    if not isinstance(raw_requests, list):
        raise DecoderQPairsetAcquisitionError(
            "pair-frame geometry lattice missing queue_executable_pairset_drop_requests[]"
        )
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, request in enumerate(raw_requests):
        if not isinstance(request, Mapping):
            raise DecoderQPairsetAcquisitionError(
                f"pair-frame geometry request {index} must be object"
            )
        try:
            require_no_truthy_authority_fields(
                request,
                context=f"decoder_q_pairset_acquisition.pair_frame_geometry_request[{index}]",
            )
        except ValueError as exc:
            raise DecoderQPairsetAcquisitionError(str(exc)) from exc
        if request.get("schema") != PAIR_FRAME_GEOMETRY_REQUEST_SCHEMA:
            raise DecoderQPairsetAcquisitionError(
                f"pair-frame geometry request {index} schema mismatch"
            )
        if request.get("queue_executable") is not True:
            raise DecoderQPairsetAcquisitionError(
                f"pair-frame geometry request {index} is not queue executable"
            )
        candidate_id = str(request.get("candidate_id") or "")
        if not candidate_id:
            raise DecoderQPairsetAcquisitionError(
                f"pair-frame geometry request {index} missing candidate_id"
            )
        if candidate_id in seen_ids:
            raise DecoderQPairsetAcquisitionError(
                f"duplicate pair-frame geometry candidate_id: {candidate_id}"
            )
        seen_ids.add(candidate_id)
        selected = _canonical_pair_indices(
            request.get("selected_pair_indices") or (),
            label=f"{candidate_id} selected_pair_indices",
        )
        selected_set = set(selected)
        if not selected_set.issubset(best_pair_set):
            raise DecoderQPairsetAcquisitionError(
                f"{candidate_id} selected_pair_indices leave the source selector universe"
            )
        dropped = _canonical_pair_indices(
            request.get("dropped_pair_indices") or (),
            label=f"{candidate_id} dropped_pair_indices",
        )
        if not set(dropped).issubset(best_pair_set) or selected_set & set(dropped):
            raise DecoderQPairsetAcquisitionError(
                f"{candidate_id} dropped_pair_indices inconsistent with source selector"
            )
        out.append(dict(request))
    return out


def build_decoder_q_pairset_acquisition_plan(
    selector_pareto: dict[str, Any],
    *,
    frame_policy: str = "pair_all_frames",
    prefix_ks: Sequence[int] | None = None,
    diversity_ks: Sequence[int] | None = None,
    include_drop_one: bool = True,
    max_drop_two: int = 128,
    drop_many_counts: Sequence[int] | None = None,
    max_drop_many: int | None = None,
    max_swap_in: int = 32,
    diversity_weight: float = 0.15,
    dqs1_observations: Sequence[Mapping[str, Any]] | None = None,
    include_observed_candidates: bool = False,
    eureka_planning: Mapping[str, Any] | None = None,
    pair_frame_geometry_lattice: Mapping[str, Any] | None = None,
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
    if isinstance(max_drop_two, bool) or max_drop_two < 0:
        raise DecoderQPairsetAcquisitionError("max_drop_two must be non-negative")
    if isinstance(max_swap_in, bool) or max_swap_in < 0:
        raise DecoderQPairsetAcquisitionError("max_swap_in must be non-negative")
    hint_ids = _eureka_planner_hint_ids(eureka_planning)
    eureka_expand_beyond_drop_two = (
        "dqs1_expand_beyond_drop_two_near_boundary" in set(hint_ids)
    )
    profile = _eureka_pairset_profile(eureka_planning)
    profile_max_drop_many = _profile_int(profile, "max_drop_many")
    profile_drop_many_counts = _profile_int_list(profile, "drop_many_counts")
    effective_max_drop_many = (
        max_drop_many
        if max_drop_many is not None
        else (
            profile_max_drop_many
            if profile_max_drop_many is not None
            else (96 if eureka_expand_beyond_drop_two else 0)
        )
    )
    if isinstance(effective_max_drop_many, bool) or effective_max_drop_many < 0:
        raise DecoderQPairsetAcquisitionError("max_drop_many must be non-negative")
    effective_drop_many_counts = _parse_drop_many_counts(
        drop_many_counts if drop_many_counts is not None else profile_drop_many_counts,
        max_count=max(0, len(best_order) - 1),
        eureka_active=eureka_expand_beyond_drop_two,
    )
    geometry_requests = _pair_frame_geometry_requests(
        pair_frame_geometry_lattice,
        best_pair_set=best_pair_set,
    )

    rows: list[dict[str, Any]] = []
    seen_pair_sets: set[tuple[int, ...]] = set()
    observed_candidate_ids = _observed_candidate_ids_from_dqs1_observations(
        dqs1_observations
    )
    observed_candidate_id_set = set(observed_candidate_ids)

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

        for dropped in _bounded_drop_many_sets(
            best_order,
            drop_counts=effective_drop_many_counts,
            limit=effective_max_drop_many,
        ):
            drop_set = set(dropped)
            selected = [candidate for candidate in best_order if candidate not in drop_set]
            dropped_ranks = sorted(rank_by_pair[pair] for pair in dropped)
            add_row(
                _drop_many_id(dropped_pairs=dropped, rank_by_pair=rank_by_pair),
                "drop_many_beam_pairwise_interaction_waterfill",
                selected,
                best,
                operation={
                    "op": "drop_many",
                    "drop_count": len(dropped),
                    "dropped_pair_indices": sorted(dropped),
                    "dropped_pair_ranks": dropped_ranks,
                    "generation_policy": (
                        "bounded_tail_beam_plus_spaced_tail_waterfill"
                    ),
                    "eureka_planner_hint_ids": hint_ids,
                    "rate_distortion_policy": (
                        "rate_gain_probe_with_distortion_authority_false"
                    ),
                    "master_gradient_status": (
                        "rank_order_proxy_until_pair_gradient_binding_lands"
                    ),
                    "inverse_scorer_status": (
                        "planner_consumer_requested_not_score_authority"
                    ),
                },
            )

        rank_by_pair = {pair: rank for rank, pair in enumerate(best_order, start=1)}
        for request in geometry_requests:
            dropped = sorted(int(pair) for pair in request["dropped_pair_indices"])
            add_row(
                str(request["candidate_id"]),
                str(request.get("selector_kind") or "pair_frame_geometry_low_impact_drop_many"),
                request["selected_pair_indices"],
                best,
                operation={
                    "op": "pair_frame_geometry_low_impact_drop_many",
                    "dropped_pair_indices": dropped,
                    "dropped_pair_ranks": [rank_by_pair[pair] for pair in dropped],
                    "source_lattice_schema": PAIR_FRAME_GEOMETRY_LATTICE_SCHEMA,
                    "source_request_schema": PAIR_FRAME_GEOMETRY_REQUEST_SCHEMA,
                    "generation_policy": request.get("generation_policy"),
                    "geometry_coverage": request.get("geometry_coverage"),
                    "queue_executable": True,
                    "queue_family": request.get("queue_family"),
                    "rate_distortion_policy": (
                        "geometry_low_impact_drop_probe_with_repair_budget_false_authority"
                    ),
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

    unfiltered_candidate_count = len(rows)
    suppressed_observed_candidate_ids: list[str] = []
    if include_observed_candidates:
        filtered_rows = rows
    else:
        filtered_rows = []
        for row in rows:
            acquisition_id = str(row.get("acquisition_id") or row.get("selector_id") or "")
            if acquisition_id in observed_candidate_id_set:
                suppressed_observed_candidate_ids.append(acquisition_id)
                continue
            filtered_rows.append(row)

    ranked = sorted(
        filtered_rows,
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
        "unfiltered_candidate_count": unfiltered_candidate_count,
        "candidate_count": len(ranked),
        "observed_dqs1_candidate_count": len(observed_candidate_ids),
        "observed_dqs1_candidate_ids": observed_candidate_ids,
        "suppressed_observed_candidate_count": len(suppressed_observed_candidate_ids),
        "suppressed_observed_candidate_ids": suppressed_observed_candidate_ids,
        "include_observed_candidates": bool(include_observed_candidates),
        "recommended_acquisition_id": ranked[0]["acquisition_id"] if ranked else None,
        "prefix_candidate_count": sum(1 for row in ranked if row["selector_kind"] == "prefix_variant"),
        "drop_one_candidate_count": sum(1 for row in ranked if row["selector_kind"] == "drop_one_from_best"),
        "drop_two_candidate_count": sum(1 for row in ranked if row["selector_kind"] == "drop_two_from_best"),
        "swap_in_candidate_count": sum(1 for row in ranked if row["selector_kind"] == "swap_in_alternative"),
        "diversity_candidate_count": sum(1 for row in ranked if row["selector_kind"] == "diversity_spaced"),
        "drop_many_candidate_count": sum(
            1
            for row in ranked
            if row["selector_kind"] == "drop_many_beam_pairwise_interaction_waterfill"
        ),
        "pair_frame_geometry_candidate_count": sum(
            1
            for row in ranked
            if row["selector_kind"] == "pair_frame_geometry_low_impact_drop_many"
        ),
    }

    eureka_blocked_family_requests = [
        {
            "family": "within_selected_set_mask_feather_probe",
            "blocker": (
                "requires receiver/materializer support for "
                "non-pair-drop mask semantics"
            ),
            **FALSE_ACQUISITION_AUTHORITY,
        },
        {
            "family": "inverse_scorer_null_direction_masked_variant",
            "blocker": (
                "requires inverse-scorer action cell to runtime "
                "materializer binding"
            ),
            **FALSE_ACQUISITION_AUTHORITY,
        },
    ]
    if not geometry_requests:
        eureka_blocked_family_requests.insert(
            0,
            {
                "family": "global_low_impact_full_pair_drop_probe",
                "blocker": (
                    "requires pair-frame scorer-geometry lattice binding "
                    "before full-board pair/frame drops are queue-executable"
                ),
                **FALSE_ACQUISITION_AUTHORITY,
            },
        )

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
            "max_drop_many": effective_max_drop_many,
            "drop_many_counts": effective_drop_many_counts,
            "max_swap_in": max_swap_in,
            "diversity_weight": diversity_weight,
            "eureka_expansion": {
                "schema": EUREKA_EXPANSION_SCHEMA,
                "active": eureka_expand_beyond_drop_two,
                "planner_hint_ids": hint_ids,
                "drop_many_candidate_generation_active": (
                    eureka_expand_beyond_drop_two
                    and effective_max_drop_many > 0
                    and bool(effective_drop_many_counts)
                ),
                "levels_considered": [
                    "bit",
                    "byte",
                    "packet_member",
                    "tensor_channel",
                    "pixel",
                    "region",
                    "boundary",
                    "frame",
                    "pair",
                    "batch",
                    "full_video",
                    "scorer_axis",
                    "receiver_runtime",
                ],
                "executable_family_this_pass": (
                    "dqs1_pairset_drop_many_local_first"
                    if eureka_expand_beyond_drop_two
                    else None
                ),
                "executable_families_this_pass": [
                    *(
                        ["dqs1_pairset_drop_many_local_first"]
                        if eureka_expand_beyond_drop_two
                        else []
                    ),
                    *(
                        ["global_low_impact_full_pair_drop_probe"]
                        if geometry_requests
                        else []
                    ),
                ],
                "blocked_family_requests": eureka_blocked_family_requests,
                **FALSE_ACQUISITION_AUTHORITY,
            },
            "pair_frame_geometry_lattice": {
                "schema": "decoder_q_pairset_pair_frame_geometry_lattice_binding.v1",
                "active": bool(geometry_requests),
                "source_schema": (
                    pair_frame_geometry_lattice.get("schema")
                    if isinstance(pair_frame_geometry_lattice, Mapping)
                    else None
                ),
                "queue_executable_request_count": len(geometry_requests),
                "candidate_count": summary["pair_frame_geometry_candidate_count"],
                "allowed_use": "local_dqs1_pairset_drop_start_generation_only",
                **FALSE_ACQUISITION_AUTHORITY,
            },
            "observation_skip": {
                "schema": "dqs1_pairset_acquisition_observation_skip.v1",
                "active": bool(observed_candidate_ids)
                and not bool(include_observed_candidates),
                "observed_candidate_count": len(observed_candidate_ids),
                "suppressed_candidate_count": len(suppressed_observed_candidate_ids),
                "include_observed_candidates": bool(include_observed_candidates),
                "allowed_use": "local_pairset_acquisition_rerun_suppression_only",
                **FALSE_ACQUISITION_AUTHORITY,
            },
            "ranking": "acquisition_score_desc_payload_bytes_asc_selector_id_asc",
        },
        "source_selector_summary": {
            "selector_candidate_count": len(selectors),
            "best_selected_pair_count": len(best_order),
            "best_rank_order_pair_indices": list(best_order),
            "best_selected_pair_indices": sorted(best_order),
        },
        "summary": summary,
        "candidates": ranked,
        **FALSE_ACQUISITION_AUTHORITY,
    }


build_pairset_acquisition_plan = build_decoder_q_pairset_acquisition_plan


__all__ = [
    "CANDIDATE_SCHEMA",
    "EUREKA_EXPANSION_SCHEMA",
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
