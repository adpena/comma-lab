# SPDX-License-Identifier: MIT
"""DQS1 drop-many beam search with pairwise interactions and repair budget.

The helpers in this module are local planning primitives.  They make the
``drop_many_beam_pairwise_interaction_waterfill`` selector executable enough for
queue-owned acquisition, while all outputs remain false-authority until measured
component replay and exact auth evidence exist.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

DEFAULT_WIDTH_K = 8
DEFAULT_DEPTH_D = 4


@dataclasses.dataclass(frozen=True)
class PairCandidate:
    """Single pair-level action considered by the drop-many beam."""

    pair_index: int
    rate_score_delta_vs_source_selector: float
    predicted_score_mean: float
    payload_bytes_delta_vs_source_selector: int
    distortion_repair_budget_score: float


@dataclasses.dataclass(frozen=True)
class BeamSearchConfig:
    """Beam search controls.

    ``target_depths`` lets callers ask for only K-drop rows while the search can
    still use a shared expansion routine.
    """

    width_k: int = DEFAULT_WIDTH_K
    depth_d: int = DEFAULT_DEPTH_D
    target_depths: tuple[int, ...] = ()
    early_stop_when_no_negative_delta: bool = True
    random_seed: int = 0


@dataclasses.dataclass(frozen=True)
class DykstraFeasibilityConfig:
    """Conservative rate/SegNet/PoseNet feasibility envelope."""

    rate_min_bytes_saved: int = 1
    segnet_max_score_units: float = 5.0e-4
    posenet_max_score_units: float = 5.0e-4
    max_iterations: int = 8
    convergence_eps: float = 1.0e-6


@dataclasses.dataclass(frozen=True)
class WaterfillConfig:
    """How freed rate budget is split between component repair axes."""

    segnet_repair_fraction: float = 0.5
    posenet_repair_fraction: float = 0.5
    planning_credit_fraction: float = 1.0


@dataclasses.dataclass(frozen=True)
class BeamCandidate:
    """Predicted drop tuple from the local-only beam search."""

    drop_tuple: tuple[int, ...]
    depth: int
    delta_s_independent: float
    delta_s_interaction: float
    delta_s_waterfill_budget_consumed: float
    delta_s_total: float
    dykstra_feasible: bool
    canonical_provenance: Mapping[str, Any]
    axis_decomposition: Mapping[str, Any]


def build_pairwise_interaction_matrix(
    candidates: Sequence[PairCandidate],
    *,
    interaction_values: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return a sparse pairwise interaction matrix payload.

    Callers may pass sparse values keyed as ``"left,right"``.  Missing entries
    are treated as zero, which is the conservative no-synergy baseline.
    """

    if not candidates:
        raise ValueError("candidates must be non-empty")
    pair_indices = sorted({int(candidate.pair_index) for candidate in candidates})
    p_max = max(pair_indices) + 1
    sparse: dict[str, float] = {}
    for key, value in dict(interaction_values or {}).items():
        parsed = _parse_interaction_key(key)
        if parsed is None:
            continue
        left, right = parsed
        if left == right:
            continue
        numeric = _finite_float_or_zero(value)
        if numeric == 0.0:
            continue
        lo, hi = sorted((left, right))
        sparse[f"{lo},{hi}"] = numeric
    return {
        "schema": "dqs1_drop_many_pairwise_interaction_matrix.v1",
        "p_max": p_max,
        "pair_indices": pair_indices,
        "interaction_matrix_sparse_lower_triangle": sparse,
        "matrix_metadata": {
            "kind": "sparse_pairwise_interaction_matrix",
            "missing_entry_policy": "zero_interaction_conservative_baseline",
            "allowed_use": "local_drop_many_beam_planning_only",
            "forbidden_use": "score_claim_or_dispatch_authority",
            **FALSE_AUTHORITY,
        },
        **FALSE_AUTHORITY,
    }


def dykstra_alternating_projection_feasibility(
    drop_tuple: tuple[int, ...],
    candidates: Sequence[PairCandidate],
    *,
    config: DykstraFeasibilityConfig = DykstraFeasibilityConfig(),  # noqa: B008
    waterfill_config: WaterfillConfig = WaterfillConfig(),  # noqa: B008
) -> bool:
    """Return whether the drop tuple satisfies the local feasibility envelope."""

    return dykstra_feasibility_details(
        drop_tuple,
        candidates,
        config=config,
        waterfill_config=waterfill_config,
    )["feasible"] is True


def dykstra_feasibility_details(
    drop_tuple: tuple[int, ...],
    candidates: Sequence[PairCandidate],
    *,
    config: DykstraFeasibilityConfig = DykstraFeasibilityConfig(),  # noqa: B008
    waterfill_config: WaterfillConfig = WaterfillConfig(),  # noqa: B008
) -> dict[str, Any]:
    """Return fail-closed feasibility details for provenance and tests."""

    candidate_by_pair = _candidate_by_pair(candidates)
    selected = [candidate_by_pair[pair] for pair in drop_tuple if pair in candidate_by_pair]
    saved_bytes = sum(
        max(0, -int(candidate.payload_bytes_delta_vs_source_selector))
        for candidate in selected
    )
    score_budget = sum(
        max(0.0, float(candidate.distortion_repair_budget_score))
        for candidate in selected
    )
    segnet_budget = score_budget * _validated_fraction(
        waterfill_config.segnet_repair_fraction,
        label="segnet_repair_fraction",
    )
    posenet_budget = score_budget * _validated_fraction(
        waterfill_config.posenet_repair_fraction,
        label="posenet_repair_fraction",
    )
    blockers: list[str] = []
    if not drop_tuple:
        blockers.append("empty_drop_tuple")
    if saved_bytes < int(config.rate_min_bytes_saved):
        blockers.append("rate_saving_polytope_not_satisfied")
    if segnet_budget > float(config.segnet_max_score_units):
        blockers.append("segnet_budget_polytope_exceeded")
    if posenet_budget > float(config.posenet_max_score_units):
        blockers.append("posenet_budget_polytope_exceeded")
    return {
        "schema": "dqs1_drop_many_dykstra_feasibility.v1",
        "feasible": not blockers,
        "drop_tuple": list(drop_tuple),
        "saved_bytes": saved_bytes,
        "score_budget": score_budget,
        "segnet_budget_score_units": segnet_budget,
        "posenet_budget_score_units": posenet_budget,
        "max_iterations": int(config.max_iterations),
        "convergence_eps": float(config.convergence_eps),
        "residual": 0.0 if not blockers else None,
        "blockers": blockers,
        "allowed_use": "local_feasibility_planning_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def waterfill_budget_consumed(
    drop_tuple: tuple[int, ...],
    candidates: Sequence[PairCandidate],
    *,
    config: WaterfillConfig = WaterfillConfig(),  # noqa: B008
) -> float:
    """Return planning-only repair budget credit for a drop tuple."""

    candidate_by_pair = _candidate_by_pair(candidates)
    axis_fraction = _validated_fraction(
        config.segnet_repair_fraction,
        label="segnet_repair_fraction",
    ) + _validated_fraction(
        config.posenet_repair_fraction,
        label="posenet_repair_fraction",
    )
    credit_fraction = _validated_fraction(
        config.planning_credit_fraction,
        label="planning_credit_fraction",
    )
    budget = sum(
        max(0.0, float(candidate_by_pair[pair].distortion_repair_budget_score))
        for pair in drop_tuple
        if pair in candidate_by_pair
    )
    return budget * min(axis_fraction, 1.0) * credit_fraction


def beam_search_drop_many(
    candidates: Sequence[PairCandidate],
    interaction_matrix: Mapping[str, Any],
    *,
    config: BeamSearchConfig = BeamSearchConfig(),  # noqa: B008
    dykstra_config: DykstraFeasibilityConfig = DykstraFeasibilityConfig(),  # noqa: B008
    waterfill_config: WaterfillConfig = WaterfillConfig(),  # noqa: B008
) -> list[BeamCandidate]:
    """Run bounded local-only beam search over drop-many tuples."""

    if not candidates:
        raise ValueError("candidates must be non-empty")
    if config.width_k <= 0 or config.depth_d <= 0:
        raise ValueError("width_k and depth_d must be positive")
    candidate_by_pair = _candidate_by_pair(candidates)
    if not candidate_by_pair:
        raise ValueError("candidates must include at least one unique pair")
    target_depths = {
        int(depth)
        for depth in config.target_depths
        if int(depth) > 0 and int(depth) <= int(config.depth_d)
    }
    accepted: list[BeamCandidate] = []
    beam: list[tuple[int, ...]] = [()]
    ordered_pairs = _ordered_candidate_pairs(candidates)
    for depth in range(1, int(config.depth_d) + 1):
        expanded: dict[tuple[int, ...], BeamCandidate] = {}
        for base_tuple in beam:
            for pair in ordered_pairs:
                if pair in base_tuple:
                    continue
                drop_tuple = tuple(sorted((*base_tuple, pair)))
                if len(drop_tuple) != depth or drop_tuple in expanded:
                    continue
                beam_candidate = _score_drop_tuple(
                    drop_tuple,
                    candidate_by_pair=candidate_by_pair,
                    interaction_matrix=interaction_matrix,
                    dykstra_config=dykstra_config,
                    waterfill_config=waterfill_config,
                )
                expanded[drop_tuple] = beam_candidate
        ranked = sorted(
            expanded.values(),
            key=lambda row: (
                float(row.delta_s_total),
                float(row.delta_s_independent),
                row.drop_tuple,
            ),
        )
        if not ranked:
            break
        if not target_depths or depth in target_depths:
            accepted.extend(row for row in ranked if row.dykstra_feasible)
        beam = [row.drop_tuple for row in ranked[: int(config.width_k)]]
        if (
            config.early_stop_when_no_negative_delta
            and ranked[0].delta_s_total >= 0.0
        ):
            break
    return sorted(
        accepted,
        key=lambda row: (
            float(row.delta_s_total),
            float(row.delta_s_independent),
            row.drop_tuple,
        ),
    )[: int(config.width_k)]


def beam_candidate_to_json(row: BeamCandidate) -> dict[str, Any]:
    """Return a deterministic JSON-friendly beam candidate."""

    return {
        "schema": "dqs1_drop_many_beam_candidate.v1",
        "drop_tuple": list(row.drop_tuple),
        "depth": row.depth,
        "delta_s_independent": row.delta_s_independent,
        "delta_s_interaction": row.delta_s_interaction,
        "delta_s_waterfill_budget_consumed": row.delta_s_waterfill_budget_consumed,
        "delta_s_total": row.delta_s_total,
        "dykstra_feasible": row.dykstra_feasible,
        "canonical_provenance": dict(row.canonical_provenance),
        "axis_decomposition": dict(row.axis_decomposition),
        **FALSE_AUTHORITY,
    }


def _score_drop_tuple(
    drop_tuple: tuple[int, ...],
    *,
    candidate_by_pair: Mapping[int, PairCandidate],
    interaction_matrix: Mapping[str, Any],
    dykstra_config: DykstraFeasibilityConfig,
    waterfill_config: WaterfillConfig,
) -> BeamCandidate:
    selected = [candidate_by_pair[pair] for pair in drop_tuple]
    delta_independent = sum(
        float(candidate.rate_score_delta_vs_source_selector)
        for candidate in selected
    )
    delta_interaction = sum(
        _interaction_value(interaction_matrix, left, right)
        for index, left in enumerate(drop_tuple)
        for right in drop_tuple[index + 1 :]
    )
    delta_waterfill = waterfill_budget_consumed(
        drop_tuple,
        list(candidate_by_pair.values()),
        config=waterfill_config,
    )
    feasible_details = dykstra_feasibility_details(
        drop_tuple,
        list(candidate_by_pair.values()),
        config=dykstra_config,
        waterfill_config=waterfill_config,
    )
    total_saved_bytes = sum(
        max(0, -int(candidate.payload_bytes_delta_vs_source_selector))
        for candidate in selected
    )
    delta_total = delta_independent + delta_interaction - delta_waterfill
    provenance = {
        "schema": "dqs1_drop_many_beam_provenance.v1",
        "equation_candidate": "dqs1_drop_many_pairwise_interaction_beam_search_v1",
        "interaction_matrix_schema": interaction_matrix.get("schema"),
        "dykstra_feasibility": feasible_details,
        "allowed_use": "local_drop_many_beam_planning_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    axis_decomposition = {
        "schema": "dqs1_drop_many_beam_axis_decomposition.v1",
        "rate_delta_score_units": delta_independent,
        "interaction_delta_score_units": delta_interaction,
        "repair_budget_credit_score_units": delta_waterfill,
        "archive_saved_bytes": total_saved_bytes,
        "segnet_delta_score_units": 0.0,
        "posenet_delta_score_units": 0.0,
        "component_response_status": (
            "not_measured_component_replay_required_before_budget_spend"
        ),
        **FALSE_AUTHORITY,
    }
    return BeamCandidate(
        drop_tuple=drop_tuple,
        depth=len(drop_tuple),
        delta_s_independent=delta_independent,
        delta_s_interaction=delta_interaction,
        delta_s_waterfill_budget_consumed=delta_waterfill,
        delta_s_total=delta_total,
        dykstra_feasible=feasible_details["feasible"] is True,
        canonical_provenance=provenance,
        axis_decomposition=axis_decomposition,
    )


def _candidate_by_pair(candidates: Sequence[PairCandidate]) -> dict[int, PairCandidate]:
    out: dict[int, PairCandidate] = {}
    for candidate in candidates:
        pair = int(candidate.pair_index)
        if pair in out:
            continue
        out[pair] = candidate
    return out


def _ordered_candidate_pairs(candidates: Sequence[PairCandidate]) -> list[int]:
    return [
        int(candidate.pair_index)
        for candidate in sorted(
            _candidate_by_pair(candidates).values(),
            key=lambda row: (
                float(row.rate_score_delta_vs_source_selector)
                - max(0.0, float(row.distortion_repair_budget_score)),
                int(row.payload_bytes_delta_vs_source_selector),
                int(row.pair_index),
            ),
        )
    ]


def _interaction_value(
    interaction_matrix: Mapping[str, Any],
    left: int,
    right: int,
) -> float:
    if left == right:
        return 0.0
    lo, hi = sorted((int(left), int(right)))
    sparse = interaction_matrix.get("interaction_matrix_sparse_lower_triangle")
    if isinstance(sparse, Mapping):
        for key in (f"{lo},{hi}", f"{lo}:{hi}", f"{lo}|{hi}"):
            if key in sparse:
                return _finite_float_or_zero(sparse[key])
    dense = interaction_matrix.get("interaction_matrix")
    if isinstance(dense, Sequence) and not isinstance(dense, str | bytes):
        try:
            row = dense[lo]
            if isinstance(row, Sequence) and not isinstance(row, str | bytes):
                return _finite_float_or_zero(row[hi])
        except (IndexError, TypeError):
            return 0.0
    return 0.0


def _parse_interaction_key(key: str) -> tuple[int, int] | None:
    for sep in (",", ":", "|"):
        if sep not in str(key):
            continue
        left, right = str(key).split(sep, 1)
        try:
            return int(left), int(right)
        except ValueError:
            return None
    return None


def _validated_fraction(value: float, *, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return numeric


def _finite_float_or_zero(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0
