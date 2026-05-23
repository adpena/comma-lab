# SPDX-License-Identifier: MIT
"""Generalized planning surface for shaving excess bytes after training.

The planner answers the recurring contest question: how many units should be
changed, from where, and by which operation family? It consumes grammar-aware
unit signals from X-ray, master-gradient, atoms, local proxy sweeps, or trained
candidate manifests and emits a bounded sweep ladder. It never promotes a score
claim; every row remains planning-only until byte-closed archive/runtime and
exact auth-eval gates pass.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.archive_byte_profile import CONTEST_ORIGINAL_BYTES
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)

SIGNAL_SURFACE_SCHEMA = "byte_shaving_signal_surface.v1"
PLAN_SCHEMA = "byte_shaving_campaign_plan.v1"
TOOL_NAME = "tools/plan_byte_shaving_campaign.py"
RATE_MULTIPLIER = 25.0

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}

BASE_BLOCKERS: tuple[str, ...] = (
    "byte_shaving_plan_is_planning_only",
    "requires_byte_closed_materialization_before_dispatch",
    "requires_runtime_consumption_proof_before_exact_eval",
    "requires_same_runtime_locality_or_inflate_parity_check",
    "requires_exact_auth_eval_before_score_claim",
)

UNIT_KINDS: frozenset[str] = frozenset(
    {
        "pair",
        "frame",
        "byte_range",
        "archive_section",
        "tensor",
        "packet_member",
        "scorer_response_row",
    }
)

DEFAULT_OPERATION_FAMILIES: dict[str, tuple[str, ...]] = {
    "pair": (
        "drop_pair",
        "substitute_pair",
        "lower_pair_precision",
        "proceduralize_pair_residual",
    ),
    "frame": (
        "drop_frame",
        "substitute_frame",
        "temporal_predict_frame",
        "lower_frame_precision",
    ),
    "byte_range": (
        "entropy_recode",
        "null_remove_or_seed",
        "delta_encode",
        "literal_elide",
    ),
    "archive_section": (
        "section_header_elide",
        "section_entropy_recode",
        "section_reorder",
        "section_proceduralize",
    ),
    "tensor": (
        "quantize_tensor",
        "prune_tensor",
        "factorize_tensor",
        "shared_codebook_tensor",
    ),
    "packet_member": (
        "zip_header_elide",
        "member_recompress",
        "member_reorder",
        "member_merge",
    ),
    "scorer_response_row": (
        "materialize_scorer_response_candidate",
        "probe_followup_neighbor",
    ),
}


class ByteShavingCampaignError(ValueError):
    """Raised when byte-shaving signal would be malformed or over-authoritative."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_repo_path(path_value: Any, repo_root: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rate_delta_for_saved_bytes(saved_bytes: int) -> float:
    return -RATE_MULTIPLIER * float(saved_bytes) / float(CONTEST_ORIGINAL_BYTES)


def _quality_cost(unit: Mapping[str, Any]) -> float:
    explicit_delta = _finite_float(
        unit.get("predicted_quality_score_delta")
        if unit.get("predicted_quality_score_delta") is not None
        else unit.get("quality_delta_score")
        if unit.get("quality_delta_score") is not None
        else unit.get("predicted_non_rate_delta_score")
    )
    if explicit_delta is not None:
        return explicit_delta
    explicit = _finite_float(unit.get("predicted_quality_score_cost"))
    if explicit is not None:
        return max(0.0, explicit)
    seg_cost = _finite_float(unit.get("predicted_seg_score_cost")) or 0.0
    pose_cost = _finite_float(unit.get("predicted_pose_score_cost")) or 0.0
    master_gradient_cost = _finite_float(unit.get("master_gradient_score_cost")) or 0.0
    return max(0.0, seg_cost + pose_cost + master_gradient_cost)


def _quality_cost_with_fallback(
    operation: Mapping[str, Any],
    unit: Mapping[str, Any],
) -> float:
    explicit_delta = _finite_float(
        operation.get("predicted_quality_score_delta")
        if operation.get("predicted_quality_score_delta") is not None
        else operation.get("quality_delta_score")
        if operation.get("quality_delta_score") is not None
        else operation.get("predicted_non_rate_delta_score")
    )
    if explicit_delta is not None:
        return explicit_delta
    explicit = _finite_float(operation.get("predicted_quality_score_cost"))
    if explicit is not None:
        return max(0.0, explicit)
    seg_cost = _finite_float(operation.get("predicted_seg_score_cost"))
    pose_cost = _finite_float(operation.get("predicted_pose_score_cost"))
    master_gradient_cost = _finite_float(operation.get("master_gradient_score_cost"))
    if seg_cost is not None or pose_cost is not None or master_gradient_cost is not None:
        return max(0.0, (seg_cost or 0.0) + (pose_cost or 0.0) + (master_gradient_cost or 0.0))
    return _quality_cost(unit)


def _confidence(unit: Mapping[str, Any]) -> float:
    raw = _finite_float(unit.get("confidence"))
    if raw is None:
        return 0.5
    return min(1.0, max(0.0, raw))


def _unit_kind(unit: Mapping[str, Any]) -> str:
    kind = str(unit.get("unit_kind") or unit.get("kind") or "").strip()
    return kind if kind in UNIT_KINDS else "byte_range"


def _operation_families(unit: Mapping[str, Any]) -> list[str]:
    explicit = ordered_unique(str(item) for item in _as_list(unit.get("operation_families")))
    if explicit:
        return explicit
    return list(DEFAULT_OPERATION_FAMILIES[_unit_kind(unit)])


def _operation_saved_bytes(operation: Mapping[str, Any], unit_saved_bytes: int) -> int:
    saved = _finite_int(
        operation.get("candidate_saved_bytes")
        if operation.get("candidate_saved_bytes") is not None
        else operation.get("saved_bytes")
        if operation.get("saved_bytes") is not None
        else operation.get("bytes")
    )
    if saved is None or saved <= 0:
        return unit_saved_bytes
    return saved


def _operation_confidence(operation: Mapping[str, Any], unit: Mapping[str, Any]) -> float:
    raw = _finite_float(operation.get("confidence"))
    if raw is None:
        return _confidence(unit)
    return min(1.0, max(0.0, raw))


def _operation_candidates(unit: Mapping[str, Any], unit_saved_bytes: int) -> list[dict[str, Any]]:
    explicit_operations = [item for item in _as_list(unit.get("operations")) if isinstance(item, Mapping)]
    raw_operations: list[Mapping[str, Any]] = explicit_operations or [
        {"operation_family": family} for family in _operation_families(unit)
    ]

    out: list[dict[str, Any]] = []
    for index, operation in enumerate(raw_operations):
        family = str(
            operation.get("operation_family")
            or operation.get("family")
            or operation.get("operation")
            or _operation_families(unit)[0]
        )
        saved_bytes = _operation_saved_bytes(operation, unit_saved_bytes)
        quality_cost = _quality_cost_with_fallback(operation, unit)
        rate_delta = _rate_delta_for_saved_bytes(saved_bytes)
        expected_delta = rate_delta + quality_cost
        gain = -expected_delta
        confidence = _operation_confidence(operation, unit)
        out.append({
            "operation_id": str(
                operation.get("operation_id")
                or operation.get("id")
                or f"{family}_{index}"
            ),
            "operation_family": family,
            "candidate_saved_bytes": saved_bytes,
            "rate_delta_score": rate_delta,
            "quality_cost_score": quality_cost,
            "expected_delta_score": expected_delta,
            "expected_score_gain": gain,
            "confidence": confidence,
            "confidence_adjusted_gain": gain * confidence,
            "gain_per_byte": gain / float(saved_bytes) if saved_bytes > 0 else 0.0,
            "materializer": operation.get("materializer"),
            "operation_params": dict(_mapping(operation.get("params"))),
            "blockers": ordered_unique(
                str(item) for item in _as_list(operation.get("blockers"))
            ),
        })
    return sorted(
        out,
        key=lambda row: (
            -float(row["confidence_adjusted_gain"]),
            -float(row["expected_score_gain"]),
            float(row["quality_cost_score"]),
            -int(row["candidate_saved_bytes"]),
            str(row["operation_family"]),
            str(row["operation_id"]),
        ),
    )


def validate_signal_surface(payload: Mapping[str, Any]) -> None:
    """Validate the signal surface and preserve false-authority semantics."""

    if payload.get("schema") != SIGNAL_SURFACE_SCHEMA:
        raise ByteShavingCampaignError(f"expected schema {SIGNAL_SURFACE_SCHEMA}")
    try:
        require_no_truthy_authority_fields(payload, context="byte_shaving_signal_surface")
    except ValueError as exc:
        raise ByteShavingCampaignError(str(exc)) from exc
    for key, expected in FALSE_AUTHORITY.items():
        if key in payload and payload.get(key) is not expected:
            raise ByteShavingCampaignError(f"{key} must be explicit false")
    units = payload.get("units")
    if not isinstance(units, list) or not units:
        raise ByteShavingCampaignError("units must be a non-empty list")
    for index, item in enumerate(units):
        if not isinstance(item, Mapping):
            raise ByteShavingCampaignError(f"units[{index}] must be an object")
        if not str(item.get("unit_id") or item.get("id") or "").strip():
            raise ByteShavingCampaignError(f"units[{index}].unit_id is required")
        saved = _finite_int(
            item.get("candidate_saved_bytes")
            if item.get("candidate_saved_bytes") is not None
            else item.get("saved_bytes")
            if item.get("saved_bytes") is not None
            else item.get("bytes")
        )
        allows_zero_saved = _unit_kind(item) == "scorer_response_row"
        if saved is None or saved < 0 or (saved == 0 and not allows_zero_saved):
            raise ByteShavingCampaignError(
                f"units[{index}] requires positive candidate_saved_bytes/saved_bytes/bytes"
            )


def normalize_unit_signal(unit: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one pair/frame/byte/section/tensor opportunity."""

    saved_bytes = _finite_int(
        unit.get("candidate_saved_bytes")
        if unit.get("candidate_saved_bytes") is not None
        else unit.get("saved_bytes")
        if unit.get("saved_bytes") is not None
        else unit.get("bytes")
    )
    allows_zero_saved = _unit_kind(unit) == "scorer_response_row"
    if saved_bytes is None or saved_bytes < 0 or (saved_bytes == 0 and not allows_zero_saved):
        raise ByteShavingCampaignError("unit requires positive saved bytes")
    operation_candidates = _operation_candidates(unit, saved_bytes)
    best_operation = operation_candidates[0]
    quality_cost = float(best_operation["quality_cost_score"])
    rate_delta = float(best_operation["rate_delta_score"])
    expected_delta = float(best_operation["expected_delta_score"])
    gain = float(best_operation["expected_score_gain"])
    confidence = float(best_operation["confidence"])
    density = float(best_operation["gain_per_byte"])
    adjusted_gain = float(best_operation["confidence_adjusted_gain"])
    return {
        "unit_id": str(unit.get("unit_id") or unit.get("id")),
        "unit_kind": _unit_kind(unit),
        "source_index": unit.get("source_index"),
        "source_span": unit.get("source_span"),
        "candidate_saved_bytes": int(best_operation["candidate_saved_bytes"]),
        "rate_delta_score": rate_delta,
        "quality_cost_score": quality_cost,
        "expected_delta_score": expected_delta,
        "expected_score_gain": gain,
        "confidence": confidence,
        "confidence_adjusted_gain": adjusted_gain,
        "gain_per_byte": density,
        "recommended_operation_id": best_operation["operation_id"],
        "recommended_operation_family": best_operation["operation_family"],
        "recommended_operation_materializer": best_operation["materializer"],
        "recommended_operation_params": best_operation["operation_params"],
        "operation_families": ordered_unique(
            str(row["operation_family"]) for row in operation_candidates
        ),
        "operation_candidates": operation_candidates,
        "score_axis": unit.get("score_axis") or unit.get("dominant_axis"),
        "evidence_grade": unit.get("evidence_grade"),
        "evidence_semantics": unit.get("evidence_semantics"),
        "source_paths": _as_list(unit.get("source_paths")),
        "source_candidate_id": unit.get("source_candidate_id"),
        "candidate_archive_sha256": unit.get("candidate_archive_sha256"),
        "candidate_archive_bytes": unit.get("candidate_archive_bytes"),
        "local_axis": unit.get("local_axis"),
        "target_axis": unit.get("target_axis"),
        "local_score": unit.get("local_score"),
        "projected_contest_score": unit.get("projected_contest_score"),
        "conservative_projected_contest_score": unit.get(
            "conservative_projected_contest_score"
        ),
        "xray_signal": unit.get("xray_signal"),
        "master_gradient_signal": unit.get("master_gradient_signal"),
        "canonical_equation_provenance": unit.get("canonical_equation_provenance"),
        "atom_ids": ordered_unique(str(item) for item in _as_list(unit.get("atom_ids"))),
        "candidate_trust_region_blockers": ordered_unique(
            str(item) for item in _as_list(unit.get("candidate_trust_region_blockers"))
        ),
        "paired_control_required": bool(unit.get("paired_control_required", True)),
        "blockers": ordered_unique(
            [
                *[str(item) for item in _as_list(unit.get("blockers"))],
                *[
                    str(item)
                    for item in _as_list(best_operation.get("blockers"))
                ],
            ]
        ),
    }


def _rank_units(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [normalize_unit_signal(unit) for unit in units]
    return sorted(
        normalized,
        key=lambda row: (
            -float(row["confidence_adjusted_gain"]),
            -float(row["expected_score_gain"]),
            float(row["quality_cost_score"]),
            -int(row["candidate_saved_bytes"]),
            str(row["unit_id"]),
        ),
    )


def _conflict_sets(payload: Mapping[str, Any]) -> list[set[str]]:
    conflicts: list[set[str]] = []
    for item in _as_list(payload.get("conflicts")):
        raw_ids: Any = (
            item.get("unit_ids") or item.get("units")
            if isinstance(item, Mapping)
            else item
        )
        ids = {str(value) for value in _as_list(raw_ids) if str(value)}
        if len(ids) >= 2:
            conflicts.append(ids)
    return conflicts


def _violates_conflict(unit_ids: set[str], conflicts: Sequence[set[str]]) -> bool:
    return any(len(unit_ids & conflict) >= 2 for conflict in conflicts)


def _interaction_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(_as_list(payload.get("interactions"))):
        if not isinstance(item, Mapping):
            continue
        unit_ids = ordered_unique(str(value) for value in _as_list(item.get("unit_ids") or item.get("units")))
        if len(unit_ids) < 2:
            continue
        rows.append({
            "interaction_id": str(item.get("interaction_id") or item.get("id") or f"interaction_{index}"),
            "unit_ids": unit_ids,
            "operation_families": ordered_unique(
                str(value)
                for value in _as_list(item.get("operation_families") or item.get("operations"))
            ),
            "delta_score": _finite_float(item.get("delta_score")) or 0.0,
            "quality_cost_delta_score": _finite_float(item.get("quality_cost_delta_score")) or 0.0,
            "extra_saved_bytes": _finite_int(item.get("extra_saved_bytes")) or 0,
            "shared_overhead_bytes": _finite_int(item.get("shared_overhead_bytes")) or 0,
            "rationale": item.get("rationale"),
        })
    return rows


def _selection_from_unit(unit: Mapping[str, Any], operation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "unit_id": str(unit["unit_id"]),
        "unit_kind": str(unit["unit_kind"]),
        "operation_id": str(operation["operation_id"]),
        "operation_family": str(operation["operation_family"]),
        "candidate_saved_bytes": int(operation["candidate_saved_bytes"]),
        "quality_cost_score": float(operation["quality_cost_score"]),
        "confidence": float(operation["confidence"]),
        "materializer": operation.get("materializer"),
        "params": operation.get("operation_params") or {},
    }


def _active_interactions(
    selections: Sequence[Mapping[str, Any]],
    interactions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected_unit_ids = {str(selection["unit_id"]) for selection in selections}
    selected_families = {str(selection["operation_family"]) for selection in selections}
    active: list[dict[str, Any]] = []
    for interaction in interactions:
        required_units = {str(value) for value in _as_list(interaction.get("unit_ids"))}
        if not required_units or not required_units.issubset(selected_unit_ids):
            continue
        required_families = {
            str(value) for value in _as_list(interaction.get("operation_families"))
        }
        if required_families and not required_families.issubset(selected_families):
            continue
        active.append(dict(interaction))
    return active


def _combo_row(
    selections: Sequence[Mapping[str, Any]],
    *,
    interactions: Sequence[Mapping[str, Any]],
    combo_id: str,
) -> dict[str, Any]:
    active = _active_interactions(selections, interactions)
    base_saved = sum(int(selection["candidate_saved_bytes"]) for selection in selections)
    extra_saved = sum(int(interaction.get("extra_saved_bytes") or 0) for interaction in active)
    overhead = sum(int(interaction.get("shared_overhead_bytes") or 0) for interaction in active)
    total_saved = max(0, base_saved + extra_saved - overhead)
    quality_cost = sum(float(selection["quality_cost_score"]) for selection in selections) + sum(
        float(interaction.get("quality_cost_delta_score") or 0.0)
        for interaction in active
    )
    direct_delta = sum(float(interaction.get("delta_score") or 0.0) for interaction in active)
    rate_delta = _rate_delta_for_saved_bytes(total_saved)
    expected_delta = rate_delta + quality_cost + direct_delta
    confidence_values = [float(selection["confidence"]) for selection in selections]
    confidence = min(confidence_values) if confidence_values else 0.0
    return {
        "combo_id": combo_id,
        "unit_count": len(selections),
        "candidate_saved_bytes": total_saved,
        "base_saved_bytes": base_saved,
        "interaction_extra_saved_bytes": extra_saved,
        "interaction_shared_overhead_bytes": overhead,
        "rate_delta_score": rate_delta,
        "quality_cost_score": quality_cost,
        "interaction_delta_score": direct_delta,
        "expected_delta_score": expected_delta,
        "expected_score_gain": -expected_delta,
        "confidence": confidence,
        "confidence_adjusted_gain": -expected_delta * confidence,
        "selected_unit_ids": [str(selection["unit_id"]) for selection in selections],
        "selected_operations": [
            {
                "unit_id": str(selection["unit_id"]),
                "operation_id": str(selection["operation_id"]),
                "operation_family": str(selection["operation_family"]),
                "materializer": selection.get("materializer"),
                "params": selection.get("params") or {},
            }
            for selection in selections
        ],
        "active_interactions": active,
        "operation_families": ordered_unique(
            str(selection["operation_family"]) for selection in selections
        ),
        "dispatch_blockers": list(BASE_BLOCKERS),
        **FALSE_AUTHORITY,
    }


def _state_sort_key(row: Mapping[str, Any]) -> tuple[float, float, int, str]:
    return (
        -float(row["confidence_adjusted_gain"]),
        -float(row["expected_score_gain"]),
        -int(row["candidate_saved_bytes"]),
        ",".join(str(value) for value in _as_list(row.get("selected_unit_ids"))),
    )


def _combo_ladder(
    ranked_units: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    *,
    max_units_considered: int,
    max_ops_per_unit: int,
    beam_width: int,
    max_combos: int,
) -> list[dict[str, Any]]:
    """Bounded beam search over unit/operation combinations with interactions."""

    conflicts = _conflict_sets(payload)
    interactions = _interaction_rows(payload)
    states: list[list[dict[str, Any]]] = [[]]
    completed: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    for unit in ranked_units[:max_units_considered]:
        next_states = list(states)
        unit_operations = [
            item
            for item in _as_list(unit.get("operation_candidates"))
            if isinstance(item, Mapping)
        ][:max_ops_per_unit]
        for state in states:
            for operation in unit_operations:
                selection = _selection_from_unit(unit, operation)
                unit_ids = {str(item["unit_id"]) for item in [*state, selection]}
                if _violates_conflict(unit_ids, conflicts):
                    continue
                candidate_state = [*state, selection]
                if len(candidate_state) >= 2:
                    key = tuple(
                        sorted(
                            (str(item["unit_id"]), str(item["operation_id"]))
                            for item in candidate_state
                        )
                    )
                    row = _combo_row(
                        candidate_state,
                        interactions=interactions,
                        combo_id=f"combo_{len(completed) + 1:04d}",
                    )
                    prior = completed.get(key)
                    if prior is None or _state_sort_key(row) < _state_sort_key(prior):
                        completed[key] = row
                next_states.append(candidate_state)
        ranked_states = sorted(
            [
                state
                for state in next_states
                if not _violates_conflict(
                    {str(item["unit_id"]) for item in state},
                    conflicts,
                )
            ],
            key=lambda state: _state_sort_key(
                _combo_row(
                    state,
                    interactions=interactions,
                    combo_id="state",
                )
            )
            if state
            else (0.0, 0.0, 0, ""),
        )
        states = ranked_states[:beam_width]
    return sorted(completed.values(), key=_state_sort_key)[:max_combos]


def _sweep_ladder_counts(n_units: int, *, max_k: int | None) -> list[int]:
    if n_units <= 0:
        return []
    upper = min(n_units, max_k if max_k is not None else n_units)
    base = [1, 2, 3, 4, 5, 8, 13, 21, 32, 48, 64, 96, 128]
    counts = {k for k in base if 1 <= k <= upper}
    counts.update({max(1, upper // 4), max(1, upper // 2), upper})
    return sorted(counts)


def _conflict_violation_ids(unit_ids: Sequence[str], conflicts: Sequence[set[str]]) -> list[list[str]]:
    selected = set(unit_ids)
    return [
        sorted(selected & conflict)
        for conflict in conflicts
        if len(selected & conflict) >= 2
    ]


def _prefix_rows(
    ranked_units: Sequence[Mapping[str, Any]],
    counts: Iterable[int],
    *,
    conflicts: Sequence[set[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for count in counts:
        selected = list(ranked_units[:count])
        selected_unit_ids = [str(row["unit_id"]) for row in selected]
        conflict_violations = _conflict_violation_ids(selected_unit_ids, conflicts)
        saved_bytes = int(sum(int(row["candidate_saved_bytes"]) for row in selected))
        rate_delta = float(sum(float(row["rate_delta_score"]) for row in selected))
        quality_cost = float(sum(float(row["quality_cost_score"]) for row in selected))
        expected_delta = rate_delta + quality_cost
        blockers = list(BASE_BLOCKERS)
        if conflict_violations:
            blockers.append("prefix_selection_violates_conflict_sets")
        rows.append({
            "sweep_id": f"top_{count:04d}",
            "unit_count": count,
            "candidate_saved_bytes": saved_bytes,
            "rate_delta_score": rate_delta,
            "quality_cost_score": quality_cost,
            "expected_delta_score": expected_delta,
            "expected_score_gain": -expected_delta,
            "operation_families": ordered_unique(
                str(row.get("recommended_operation_family") or family)
                for row in selected
                for family in _as_list(row.get("operation_families"))
            ),
            "selected_operations": [
                {
                    "unit_id": str(row["unit_id"]),
                    "operation_id": str(row["recommended_operation_id"]),
                    "operation_family": str(row["recommended_operation_family"]),
                    "materializer": row.get("recommended_operation_materializer"),
                    "params": row.get("recommended_operation_params") or {},
                }
                for row in selected
            ],
            "selected_unit_ids": selected_unit_ids,
            "conflict_violations": conflict_violations,
            "dispatch_blockers": blockers,
            **FALSE_AUTHORITY,
        })
    return rows


def _best_nonpositive_prefix(prefix_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    viable = [
        row
        for row in prefix_rows
        if float(row.get("expected_delta_score") or 0.0) < 0.0
        and not row.get("conflict_violations")
    ]
    if not viable:
        return None
    return max(
        viable,
        key=lambda row: (
            float(row["expected_score_gain"]),
            int(row["candidate_saved_bytes"]),
            -int(row["unit_count"]),
        ),
    )


def build_byte_shaving_campaign_plan(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None = None,
    repo_root: Path | None = None,
    max_k: int | None = None,
) -> dict[str, Any]:
    """Build a ranked, bounded sweep plan from one signal surface."""

    validate_signal_surface(payload)
    repo_root = repo_root or Path.cwd()
    ranked = _rank_units([item for item in _as_list(payload.get("units")) if isinstance(item, Mapping)])
    conflicts = _conflict_sets(payload)
    prefix_rows = _prefix_rows(
        ranked,
        _sweep_ladder_counts(len(ranked), max_k=max_k),
        conflicts=conflicts,
    )
    combo_rows = _combo_ladder(
        ranked,
        payload,
        max_units_considered=int(payload.get("max_combo_units_considered") or 32),
        max_ops_per_unit=int(payload.get("max_combo_ops_per_unit") or 3),
        beam_width=int(payload.get("combo_beam_width") or 64),
        max_combos=int(payload.get("max_combo_count") or 32),
    )
    best = _best_nonpositive_prefix(prefix_rows)
    best_combo = _best_nonpositive_prefix(combo_rows)
    source_paths = (
        [_repo_rel(source_path, repo_root)] if source_path is not None else []
    )
    plan = {
        "schema": PLAN_SCHEMA,
        "tool": TOOL_NAME,
        "generated_at_utc": _utc_now(),
        "campaign_id": str(payload.get("campaign_id") or "byte_shaving_campaign"),
        "candidate_id": payload.get("candidate_id"),
        "lane_id": payload.get("lane_id") or "byte_shaving_campaign",
        "source_paths": source_paths,
        "source_signal_refs": _as_list(payload.get("source_signal_refs")),
        "auth_eval_refs": _as_list(payload.get("auth_eval_refs")),
        "mlx_calibration_refs": _as_list(payload.get("mlx_calibration_refs")),
        "scorer_response_refs": _as_list(payload.get("scorer_response_refs")),
        "frontier_axis": payload.get("frontier_axis") or "[planning-only]",
        "ranked_units": ranked,
        "sweep_ladder": prefix_rows,
        "recommended_prefix": dict(best) if best is not None else None,
        "combination_ladder": combo_rows,
        "recommended_combination": dict(best_combo) if best_combo is not None else None,
        "operation_menu": DEFAULT_OPERATION_FAMILIES,
        "combination_policy": {
            "max_units_considered": int(payload.get("max_combo_units_considered") or 32),
            "max_ops_per_unit": int(payload.get("max_combo_ops_per_unit") or 3),
            "beam_width": int(payload.get("combo_beam_width") or 64),
            "max_combo_count": int(payload.get("max_combo_count") or 32),
            "interactions_mode": (
                "direct delta_score plus extra_saved_bytes/shared_overhead_bytes "
                "for scorer-axis and runtime externalities"
            ),
            "conflicts_mode": "unit conflict sets cannot co-occur in one combo",
        },
        "dispatch_blockers": ordered_unique(
            [*BASE_BLOCKERS, *[str(item) for item in _as_list(payload.get("blockers"))]]
        ),
        "evidence_boundary": {
            "planning_only": True,
            "rate_delta_formula": f"-25 * saved_bytes / {CONTEST_ORIGINAL_BYTES}",
            "quality_cost_source": (
                "predicted_quality_score_cost or seg/pose/master-gradient score costs"
            ),
            "next_gate": (
                "materialize selected operation family, run locality/inflate "
                "controls, then exact auth eval before any score claim"
            ),
        },
        **FALSE_AUTHORITY,
    }
    return apply_proxy_evidence_boundary(plan, dispatch_blockers=plan["dispatch_blockers"])


def build_signal_surface_from_candidate_queue(
    queue_payload: Mapping[str, Any],
    *,
    campaign_id: str = "candidate_queue_byte_shaving_surface",
) -> dict[str, Any]:
    """Extract coarse byte-shaving units from existing optimizer queue rows."""

    if queue_payload.get("schema") != "optimizer_candidate_queue_v1":
        raise ByteShavingCampaignError("expected optimizer_candidate_queue_v1")
    try:
        require_no_truthy_authority_fields(
            queue_payload,
            context="optimizer_candidate_queue_byte_shaving_surface",
        )
    except ValueError as exc:
        raise ByteShavingCampaignError(str(exc)) from exc
    units: list[dict[str, Any]] = []
    for row in _as_list(queue_payload.get("top_k")):
        if not isinstance(row, Mapping):
            continue
        try:
            require_no_truthy_authority_fields(
                row,
                context="optimizer_candidate_queue_byte_shaving_row",
            )
        except ValueError as exc:
            raise ByteShavingCampaignError(str(exc)) from exc
        bytes_value = _finite_int(
            row.get("candidate_saved_bytes")
            or row.get("saved_bytes")
            or row.get("archive_bytes_saved")
            or row.get("source_saved_bytes")
        )
        if bytes_value is None or bytes_value <= 0:
            continue
        units.append({
            "unit_id": str(row.get("candidate_id") or f"queue_row_{len(units)}"),
            "unit_kind": row.get("unit_kind") or "archive_section",
            "candidate_saved_bytes": bytes_value,
            "predicted_quality_score_cost": _finite_float(
                row.get("predicted_quality_score_cost")
                or row.get("quality_cost_score")
            )
            or 0.0,
            "confidence": _finite_float(row.get("confidence")) or 0.5,
            "operation_families": row.get("operation_families") or [],
            "score_axis": row.get("score_axis") or row.get("dominant_axis"),
            "evidence_grade": row.get("evidence_grade"),
            "evidence_semantics": row.get("evidence_semantics"),
            "source_paths": _as_list(row.get("source_paths")),
            "source_candidate_id": row.get("source_candidate_id"),
            "candidate_archive_sha256": row.get("candidate_archive_sha256")
            or row.get("archive_sha256"),
            "candidate_archive_bytes": row.get("candidate_archive_bytes")
            or row.get("archive_bytes"),
            "candidate_trust_region_blockers": _as_list(
                row.get("candidate_trust_region_blockers")
            ),
            "local_axis": row.get("local_axis"),
            "target_axis": row.get("target_axis"),
            "local_score": row.get("local_score"),
            "projected_contest_score": row.get("projected_contest_score"),
            "conservative_projected_contest_score": row.get(
                "conservative_projected_contest_score"
            ),
            "master_gradient_signal": row.get("master_gradient_signal")
            or row.get("master_gradient_provenance"),
            "canonical_equation_provenance": row.get("canonical_equation_provenance"),
            "atom_ids": _as_list(row.get("atom_ids")),
            "blockers": row.get("dispatch_blockers") or [],
        })
    if not units:
        raise ByteShavingCampaignError("optimizer queue has no rows with saved-byte estimates")
    return {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": campaign_id,
        "source_signal_refs": _as_list(queue_payload.get("source_signal_refs")),
        "auth_eval_refs": _as_list(queue_payload.get("auth_eval_refs")),
        "mlx_calibration_refs": _as_list(queue_payload.get("mlx_calibration_refs")),
        "scorer_response_refs": _as_list(queue_payload.get("scorer_response_refs")),
        "units": units,
        **FALSE_AUTHORITY,
    }


def _contiguous_spans(indices: Sequence[int], *, max_span_bytes: int) -> list[tuple[int, int]]:
    if not indices:
        return []
    spans: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for raw in indices[1:]:
        idx = int(raw)
        if idx == prev + 1 and idx - start < max_span_bytes:
            prev = idx
            continue
        spans.append((start, prev + 1))
        start = prev = idx
    spans.append((start, prev + 1))
    return spans


def build_signal_surface_from_master_gradient_anchor(
    *,
    archive_sha256: str,
    repo_root: Path | str = ".",
    ledger_path: Path | str | None = None,
    axis: str | None = None,
    campaign_id: str = "master_gradient_byte_shaving_surface",
    low_sensitivity_quantile: float = 0.05,
    max_units: int = 32,
    max_span_bytes: int = 4096,
    quality_cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Build byte-range planning units from a canonical master-gradient anchor.

    This bridge intentionally emits planning-only byte-range opportunities, not
    packet mutations. Master-gradient coordinates must still be mapped through
    archive grammar/materializers, then checked with locality controls and exact
    auth eval before any score claim.
    """

    if not archive_sha256:
        raise ByteShavingCampaignError("archive_sha256 is required")
    if not (0.0 < low_sensitivity_quantile <= 1.0):
        raise ByteShavingCampaignError("low_sensitivity_quantile must be in (0, 1]")
    if max_units < 1:
        raise ByteShavingCampaignError("max_units must be >= 1")
    if max_span_bytes < 1:
        raise ByteShavingCampaignError("max_span_bytes must be >= 1")
    if quality_cost_multiplier < 0.0 or not math.isfinite(quality_cost_multiplier):
        raise ByteShavingCampaignError("quality_cost_multiplier must be finite and >= 0")

    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ByteShavingCampaignError("numpy is required for master-gradient surfaces") from exc

    from tac.master_gradient import (  # local import keeps generic planner light
        latest_anchor_for_archive,
    )

    repo = Path(repo_root)
    ledger = repo / ".omx/state/master_gradient_anchors.jsonl" if ledger_path is None else Path(ledger_path)
    if ledger is not None and not ledger.is_absolute():
        ledger = repo / ledger
    anchor = latest_anchor_for_archive(archive_sha256, path=ledger, axis=axis)
    if anchor is None:
        axis_text = f" axis={axis!r}" if axis is not None else ""
        raise ByteShavingCampaignError(
            f"no usable master-gradient planning anchor for archive_sha256={archive_sha256!r}{axis_text}"
        )
    npy_path = _resolve_repo_path(anchor.get("gradient_array_path"), repo)
    if npy_path is None or not npy_path.is_file():
        raise ByteShavingCampaignError(
            f"master-gradient anchor missing readable gradient_array_path: {anchor.get('gradient_array_path')!r}"
        )
    arr = np.load(npy_path)
    if arr.ndim == 2 and arr.shape[1] == 3:
        sensitivity = np.abs(arr).sum(axis=1).astype(np.float64)
        tensor_kind = "aggregate_per_byte_v1"
    elif arr.ndim == 3 and arr.shape[2] == 3:
        sensitivity = np.abs(arr).sum(axis=2).mean(axis=1).astype(np.float64)
        tensor_kind = "per_pair_per_byte_v1_mean_abs"
    else:
        raise ByteShavingCampaignError(
            f"master-gradient array must have shape (N, 3) or (N, P, 3); got {arr.shape}"
        )
    if sensitivity.size == 0:
        raise ByteShavingCampaignError("master-gradient array is empty")
    declared_n = _finite_int(anchor.get("n_bytes"))
    if declared_n is not None and declared_n != int(sensitivity.size):
        raise ByteShavingCampaignError(
            f"anchor n_bytes={declared_n} but gradient array has {int(sensitivity.size)} bytes"
        )

    threshold = float(np.quantile(sensitivity, low_sensitivity_quantile))
    candidate_indices = np.flatnonzero(sensitivity <= threshold).astype(int).tolist()
    spans = _contiguous_spans(candidate_indices, max_span_bytes=max_span_bytes)
    span_rows: list[dict[str, Any]] = []
    for start, end in spans:
        span_sensitivity = sensitivity[start:end]
        if span_sensitivity.size == 0:
            continue
        sensitivity_sum = float(span_sensitivity.sum())
        length = end - start
        quality_cost = max(0.0, sensitivity_sum * quality_cost_multiplier)
        span_rows.append({
            "start": start,
            "end": end,
            "length": length,
            "mean_sensitivity": float(span_sensitivity.mean()),
            "max_sensitivity": float(span_sensitivity.max()),
            "sensitivity_sum": sensitivity_sum,
            "quality_cost": quality_cost,
        })
    span_rows.sort(
        key=lambda row: (
            float(row["quality_cost"]),
            float(row["mean_sensitivity"]),
            -int(row["length"]),
            int(row["start"]),
        )
    )
    blockers = [
        "master_gradient_byte_ranges_are_planning_coordinates_only",
        "requires_archive_grammar_mapping_before_materialization",
        "master_gradient_quality_cost_is_first_order_proxy",
        "requires_locality_control_before_exact_auth_eval",
        "requires_exact_auth_eval_before_score_claim",
    ]
    units: list[dict[str, Any]] = []
    for row in span_rows[:max_units]:
        start = int(row["start"])
        end = int(row["end"])
        saved_bytes = int(row["length"])
        quality_cost = float(row["quality_cost"])
        mean_sensitivity = float(row["mean_sensitivity"])
        units.append({
            "unit_id": f"mg_byte_span_{start:07d}_{end:07d}",
            "unit_kind": "byte_range",
            "source_index": start,
            "source_span": {"start": start, "end_exclusive": end},
            "candidate_saved_bytes": saved_bytes,
            "predicted_quality_score_cost": quality_cost,
            "confidence": 0.55 if mean_sensitivity == 0.0 else 0.35,
            "operation_families": [
                "null_remove_or_seed",
                "entropy_recode",
                "delta_encode",
            ],
            "operations": [
                {
                    "operation_id": "null_remove_or_seed",
                    "operation_family": "null_remove_or_seed",
                    "candidate_saved_bytes": saved_bytes,
                    "predicted_quality_score_cost": quality_cost,
                    "blockers": blockers,
                },
                {
                    "operation_id": "entropy_recode",
                    "operation_family": "entropy_recode",
                    "candidate_saved_bytes": saved_bytes,
                    "predicted_quality_score_cost": quality_cost,
                    "blockers": blockers,
                },
                {
                    "operation_id": "delta_encode",
                    "operation_family": "delta_encode",
                    "candidate_saved_bytes": max(1, saved_bytes // 2),
                    "predicted_quality_score_cost": quality_cost * 0.5,
                    "blockers": blockers,
                },
            ],
            "master_gradient_score_cost": quality_cost,
            "master_gradient_signal": {
                "archive_sha256": archive_sha256,
                "measurement_axis": anchor.get("measurement_axis"),
                "measurement_hardware": anchor.get("measurement_hardware"),
                "measurement_call_id": anchor.get("measurement_call_id"),
                "gradient_tensor_kind": anchor.get("gradient_tensor_kind") or tensor_kind,
                "gradient_array_path": _repo_rel(npy_path, repo),
                "low_sensitivity_quantile": low_sensitivity_quantile,
                "threshold": threshold,
                "span_sensitivity_sum": float(row["sensitivity_sum"]),
                "span_mean_sensitivity": mean_sensitivity,
                "span_max_sensitivity": float(row["max_sensitivity"]),
                "quality_cost_multiplier": quality_cost_multiplier,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "score_axis": anchor.get("measurement_axis"),
            "blockers": blockers,
        })
    if not units:
        raise ByteShavingCampaignError("master-gradient anchor produced no byte-range units")
    return {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": campaign_id,
        "candidate_id": f"{archive_sha256[:12]}_master_gradient_low_sensitivity",
        "lane_id": "master_gradient_byte_shaving_planning",
        "frontier_axis": anchor.get("measurement_axis") or "[planning-only]",
        "source_signal_refs": [
            {
                "kind": "master_gradient_anchor",
                "archive_sha256": archive_sha256,
                "measurement_axis": anchor.get("measurement_axis"),
                "measurement_hardware": anchor.get("measurement_hardware"),
                "measurement_call_id": anchor.get("measurement_call_id"),
                "gradient_array_path": _repo_rel(npy_path, repo),
                "ledger_path": _repo_rel(ledger, repo),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "surface_parameters": {
            "low_sensitivity_quantile": low_sensitivity_quantile,
            "max_units": max_units,
            "max_span_bytes": max_span_bytes,
            "quality_cost_multiplier": quality_cost_multiplier,
            "sensitivity_threshold": threshold,
            "candidate_span_count_before_cap": len(span_rows),
        },
        "units": units,
        **FALSE_AUTHORITY,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "BASE_BLOCKERS",
    "FALSE_AUTHORITY",
    "PLAN_SCHEMA",
    "SIGNAL_SURFACE_SCHEMA",
    "ByteShavingCampaignError",
    "build_byte_shaving_campaign_plan",
    "build_signal_surface_from_candidate_queue",
    "build_signal_surface_from_master_gradient_anchor",
    "normalize_unit_signal",
    "validate_signal_surface",
]
