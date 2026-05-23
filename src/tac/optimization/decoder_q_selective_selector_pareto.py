# SPDX-License-Identifier: MIT
"""Pareto planning for compact DQS1 selective decoder-q pair sets."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.optimization.decoder_q_selective_runtime_packet import (
    BRIDGE_SCHEMA,
    CONTEST_RATE_DENOMINATOR_BYTES,
    FALSE_AUTHORITY,
    FEC6_PAIR_COUNT,
    affected_frames_for_pairs,
    choose_dqs1_pair_encoding,
)

SCHEMA = "decoder_q_selective_selector_pareto.v1"
TOOL = "tac.optimization.decoder_q_selective_selector_pareto"


class DecoderQSelectiveSelectorParetoError(ValueError):
    """Raised when selector Pareto planning would lose custody or authority."""


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(payload), encoding="utf-8")


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DecoderQSelectiveSelectorParetoError(f"{path}: expected JSON object")
    return payload


def _require_false_authority(payload: dict[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if payload.get(key) is not False:
            raise DecoderQSelectiveSelectorParetoError(
                f"{label} {key} must be explicit false"
            )


def _as_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DecoderQSelectiveSelectorParetoError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise DecoderQSelectiveSelectorParetoError(f"{label} must be finite")
    return result


def _as_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise DecoderQSelectiveSelectorParetoError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DecoderQSelectiveSelectorParetoError(f"{label} must be an integer") from exc
    if result != value and not (isinstance(value, str) and str(result) == value):
        raise DecoderQSelectiveSelectorParetoError(f"{label} must be integral")
    return result


def _bridge_units(bridge_plan: dict[str, Any]) -> list[dict[str, Any]]:
    if bridge_plan.get("schema") != BRIDGE_SCHEMA:
        raise DecoderQSelectiveSelectorParetoError("bridge plan schema mismatch")
    _require_false_authority(bridge_plan, label="bridge plan")
    if bridge_plan.get("candidate_generation_only") is not True:
        raise DecoderQSelectiveSelectorParetoError("bridge plan must be candidate_generation_only")
    units = bridge_plan.get("work_units")
    if not isinstance(units, list) or not units:
        raise DecoderQSelectiveSelectorParetoError("bridge plan work_units[] missing")
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, unit in enumerate(units):
        if not isinstance(unit, dict):
            raise DecoderQSelectiveSelectorParetoError(f"work unit {index} must be object")
        _require_false_authority(unit, label=f"work unit {index}")
        window = unit.get("pair_window")
        if not isinstance(window, list) or len(window) != 2:
            raise DecoderQSelectiveSelectorParetoError(f"work unit {index} pair_window invalid")
        pair = _as_int(window[0], label=f"work unit {index} pair_window[0]")
        if _as_int(window[1], label=f"work unit {index} pair_window[1]") != pair + 1:
            raise DecoderQSelectiveSelectorParetoError(
                "selector Pareto v1 requires singleton pair windows"
            )
        if not 0 <= pair < FEC6_PAIR_COUNT:
            raise DecoderQSelectiveSelectorParetoError(
                f"pair index out of FEC6 range: {pair}"
            )
        if pair in seen:
            raise DecoderQSelectiveSelectorParetoError(f"duplicate pair window: {pair}")
        seen.add(pair)
        denominator = _as_int(
            unit.get("full_video_denominator"),
            label=f"work unit {index} full_video_denominator",
        )
        if denominator != FEC6_PAIR_COUNT:
            raise DecoderQSelectiveSelectorParetoError(
                f"work unit {index} full_video_denominator must be {FEC6_PAIR_COUNT}"
            )
        normalized.append(
            {
                "rank": _as_int(unit.get("rank", index + 1), label=f"work unit {index} rank"),
                "pair_index": pair,
                "observed_mlx_window_gain": _as_float(
                    unit.get("observed_mlx_window_gain"),
                    label=f"work unit {index} observed_mlx_window_gain",
                ),
                "normalized_full_video_gain": _as_float(
                    unit.get("normalized_full_video_gain"),
                    label=f"work unit {index} normalized_full_video_gain",
                ),
                "full_video_denominator": denominator,
                "unit_id": unit.get("unit_id"),
            }
        )
    return sorted(normalized, key=lambda row: (int(row["rank"]), int(row["pair_index"])))


def _parse_positive_ints(values: Sequence[int] | None, *, max_count: int) -> list[int]:
    if values is None:
        values = (1, 2, 4, 8, 12, 16, 20, 24, 28, max_count)
    out = sorted({int(value) for value in values if 0 < int(value) <= max_count})
    if not out:
        raise DecoderQSelectiveSelectorParetoError("at least one prefix k is required")
    return out


def _payload_stats(pair_indices: Sequence[int], *, frame_policy: str) -> dict[str, Any]:
    pairs = sorted({int(pair) for pair in pair_indices})
    if len(pairs) != len(pair_indices):
        raise DecoderQSelectiveSelectorParetoError("pair set contains duplicates")
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


def _calibration(
    *,
    base_score: float | None,
    reference_score: float | None,
    reference_rate_delta: float,
    reference_normalized_gain: float,
    reference_window_gain: float,
) -> dict[str, Any] | None:
    if base_score is None or reference_score is None:
        return None
    base = _as_float(base_score, label="base score")
    reference = _as_float(reference_score, label="reference score")
    gain = base + float(reference_rate_delta) - reference
    scale = gain / reference_normalized_gain if reference_normalized_gain > 0.0 else None
    return {
        "schema": "decoder_q_selective_selector_exact_cpu_top32_calibration.v1",
        "base_score": base,
        "reference_score": reference,
        "reference_rate_delta": reference_rate_delta,
        "reference_observed_mlx_window_gain_sum": reference_window_gain,
        "reference_normalized_full_video_gain_sum": reference_normalized_gain,
        "reference_observed_mlx_gain_sum": reference_normalized_gain,
        "exact_component_gain": gain,
        "component_gain_per_normalized_full_video_gain": scale,
        "component_gain_per_mlx_gain": scale,
        "score_axis": "contest_cpu",
        "evidence_grade": "contest-CPU",
        "allowed_use": "non_authoritative_selector_ranking_only",
        **FALSE_AUTHORITY,
    }


def _candidate_row(
    *,
    selector_id: str,
    selector_kind: str,
    pair_indices: Sequence[int],
    rank_order_pairs: Sequence[int],
    gain_by_pair: dict[int, float],
    window_gain_by_pair: dict[int, float],
    frame_policy: str,
    calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    stats = _payload_stats(pair_indices, frame_policy=frame_policy)
    normalized_gain = sum(gain_by_pair[pair] for pair in stats["pair_indices"])
    window_gain = sum(window_gain_by_pair[pair] for pair in stats["pair_indices"])
    row: dict[str, Any] = {
        "schema": "decoder_q_selective_selector_candidate.v1",
        "selector_id": selector_id,
        "selector_kind": selector_kind,
        "rank_order_pair_indices": list(rank_order_pairs),
        "selected_pair_indices": stats["pair_indices"],
        "selected_pair_count": stats["selected_pair_count"],
        "affected_frame_count": len(stats["affected_frame_indices"]),
        "payload_bytes": stats["payload_bytes"],
        "pair_encoding": stats["pair_encoding"],
        "pair_index_payload_bytes": stats["pair_index_payload_bytes"],
        "rate_delta": stats["rate_delta"],
        "non_authoritative_mlx_window_gain_sum": window_gain,
        "non_authoritative_normalized_full_video_gain_sum": normalized_gain,
        "non_authoritative_mlx_gain_sum": normalized_gain,
        "net_normalized_full_video_gain_after_rate_non_authoritative": (
            normalized_gain - stats["rate_delta"]
        ),
        "net_mlx_gain_after_rate_non_authoritative": (
            normalized_gain - stats["rate_delta"]
        ),
        "full_video_denominator": FEC6_PAIR_COUNT,
        "pair_encoding_candidates": stats["pair_encoding_candidates"],
        **FALSE_AUTHORITY,
    }
    if calibration is not None and calibration.get("component_gain_per_mlx_gain") is not None:
        scale = float(calibration["component_gain_per_mlx_gain"])
        predicted_component_gain = normalized_gain * scale
        predicted_score = float(calibration["base_score"]) - predicted_component_gain + stats["rate_delta"]
        row["exact_cpu_calibrated_estimate"] = {
            "schema": "decoder_q_selective_selector_exact_cpu_calibrated_estimate.v1",
            "predicted_component_gain": predicted_component_gain,
            "predicted_score": predicted_score,
            "predicted_delta_vs_base": predicted_score - float(calibration["base_score"]),
            "calibration_score_axis": calibration["score_axis"],
            "allowed_use": "non_authoritative_selector_ranking_only",
            **FALSE_AUTHORITY,
        }
    return row


def _dominance_score(row: dict[str, Any]) -> float:
    estimate = row.get("exact_cpu_calibrated_estimate")
    if isinstance(estimate, dict):
        return float(estimate["predicted_score"])
    return -float(row["net_mlx_gain_after_rate_non_authoritative"])


def _annotate_pareto_frontier(rows: list[dict[str, Any]]) -> None:
    """Mark non-dominated candidates for the planning objective.

    The objective is two-dimensional: minimize archive payload bytes and
    minimize the non-authoritative planning score. With no exact-CPU
    calibration, the score is the negated MLX net-gain proxy so lower is still
    better. This metadata keeps the artifact honest: sorted ranks are a review
    convenience, while ``pareto_frontier`` is the actual dominance relation.
    """

    for row in rows:
        dominated_by: str | None = None
        row_payload = int(row["payload_bytes"])
        row_score = _dominance_score(row)
        for other in rows:
            if other is row:
                continue
            other_payload = int(other["payload_bytes"])
            other_score = _dominance_score(other)
            if (
                other_payload <= row_payload
                and other_score <= row_score
                and (other_payload < row_payload or other_score < row_score)
            ):
                dominated_by = str(other["selector_id"])
                break
        row["pareto_frontier"] = dominated_by is None
        row["dominated_by_selector_id"] = dominated_by
        row["dominance_objective"] = {
            "schema": "decoder_q_selective_selector_dominance_objective.v1",
            "minimize": ["payload_bytes", "planning_score"],
            "planning_score": row_score,
            "score_source": (
                "exact_cpu_calibrated_estimate.predicted_score"
                if isinstance(row.get("exact_cpu_calibrated_estimate"), dict)
                else "-net_mlx_gain_after_rate_non_authoritative"
            ),
            **FALSE_AUTHORITY,
        }


def build_selector_pareto_plan(
    bridge_plan: dict[str, Any],
    *,
    frame_policy: str = "pair_all_frames",
    prefix_ks: Sequence[int] | None = None,
    include_drop_one: bool = True,
    include_singletons: bool = True,
    base_score: float | None = None,
    reference_score: float | None = None,
) -> dict[str, Any]:
    """Build non-authoritative compact selector candidates from a bridge plan."""

    units = _bridge_units(bridge_plan)
    ranked_pairs = [int(unit["pair_index"]) for unit in units]
    gain_by_pair = {
        int(unit["pair_index"]): float(unit["normalized_full_video_gain"])
        for unit in units
    }
    window_gain_by_pair = {
        int(unit["pair_index"]): float(unit["observed_mlx_window_gain"])
        for unit in units
    }
    prefix_values = _parse_positive_ints(prefix_ks, max_count=len(units))
    full_pair_set = ranked_pairs
    reference_stats = _payload_stats(full_pair_set, frame_policy=frame_policy)
    reference_gain = sum(gain_by_pair[pair] for pair in reference_stats["pair_indices"])
    reference_window_gain = sum(
        window_gain_by_pair[pair] for pair in reference_stats["pair_indices"]
    )
    calibration = _calibration(
        base_score=base_score,
        reference_score=reference_score,
        reference_rate_delta=float(reference_stats["rate_delta"]),
        reference_normalized_gain=reference_gain,
        reference_window_gain=reference_window_gain,
    )

    rows: list[dict[str, Any]] = []
    seen_pair_sets: set[tuple[int, ...]] = set()

    def add_row(selector_id: str, selector_kind: str, selected_rank_order: Sequence[int]) -> None:
        key = tuple(sorted(int(pair) for pair in selected_rank_order))
        if key in seen_pair_sets:
            return
        seen_pair_sets.add(key)
        rows.append(
            _candidate_row(
                selector_id=selector_id,
                selector_kind=selector_kind,
                pair_indices=key,
                rank_order_pairs=list(selected_rank_order),
                gain_by_pair=gain_by_pair,
                window_gain_by_pair=window_gain_by_pair,
                frame_policy=frame_policy,
                calibration=calibration,
            )
        )

    for k in prefix_values:
        add_row(f"prefix_k{k:03d}", "top_rank_prefix", ranked_pairs[:k])

    if include_singletons:
        for unit in units:
            pair = int(unit["pair_index"])
            add_row(f"singleton_rank{int(unit['rank']):03d}_pair{pair:04d}", "singleton", [pair])

    if include_drop_one and len(ranked_pairs) > 1:
        for unit in units:
            pair = int(unit["pair_index"])
            selected = [candidate for candidate in ranked_pairs if candidate != pair]
            add_row(f"drop_rank{int(unit['rank']):03d}_pair{pair:04d}", "full_minus_one", selected)

    def score_key(row: dict[str, Any]) -> tuple[float, int, str]:
        estimate = row.get("exact_cpu_calibrated_estimate")
        if isinstance(estimate, dict):
            return (
                float(estimate["predicted_score"]),
                int(row["payload_bytes"]),
                str(row["selector_id"]),
            )
        return (
            -float(row["net_mlx_gain_after_rate_non_authoritative"]),
            int(row["payload_bytes"]),
            str(row["selector_id"]),
        )

    _annotate_pareto_frontier(rows)
    ranked = sorted(rows, key=score_key)
    frontier_rank = 0
    for rank, row in enumerate(ranked, start=1):
        row["selector_rank"] = rank
        row["rank_kind"] = "sorted_planning_rank"
        if row["pareto_frontier"]:
            frontier_rank += 1
            row["pareto_rank"] = frontier_rank
        else:
            row["pareto_rank"] = None

    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": TOOL,
        "evidence_grade": "macOS-MLX research signal with optional exact-CPU calibration",
        "allowed_use": "selector_planning_only_requires_materialization_and_exact_eval",
        "frame_policy": frame_policy,
        "bridge_plan_summary": {
            "work_unit_count": len(units),
            "top_rank_pair": ranked_pairs[0],
            "last_rank_pair": ranked_pairs[-1],
            "full_payload_bytes": reference_stats["payload_bytes"],
            "full_rate_delta": reference_stats["rate_delta"],
            "full_non_authoritative_mlx_window_gain_sum": reference_window_gain,
            "full_non_authoritative_normalized_full_video_gain_sum": reference_gain,
            "full_non_authoritative_mlx_gain_sum": reference_gain,
        },
        "exact_cpu_calibration": calibration,
        "summary": {
            "candidate_count": len(ranked),
            "pareto_frontier_candidate_count": sum(
                1 for row in ranked if row["pareto_frontier"]
            ),
            "prefix_candidate_count": sum(1 for row in ranked if row["selector_kind"] == "top_rank_prefix"),
            "singleton_candidate_count": sum(1 for row in ranked if row["selector_kind"] == "singleton"),
            "drop_one_candidate_count": sum(1 for row in ranked if row["selector_kind"] == "full_minus_one"),
            "recommended_selector_id": ranked[0]["selector_id"] if ranked else None,
        },
        "candidates": ranked,
        **FALSE_AUTHORITY,
    }


__all__ = [
    "SCHEMA",
    "TOOL",
    "DecoderQSelectiveSelectorParetoError",
    "build_selector_pareto_plan",
    "dumps_json",
    "load_json_object",
    "write_json",
]
