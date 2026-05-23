# SPDX-License-Identifier: MIT
"""Planning-only inverse-steganalysis acquisition surface.

The rows here are scorer-in-the-loop search signal, not score authority. They
encode multiscale atoms plus local/proxy calibration observations so schedulers
can rank next probes by expected score gain per second, GB, and resource kind.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tac.optimization.candidate_evidence_contract import is_sha256_hex
from tac.optimization.proxy_candidate_contract import (
    CONSUMER_PAYLOAD_FORBIDDEN_TRUE_AUTHORITY_FIELDS,
    PROXY_FALSE_AUTHORITY_FIELDS,
    apply_proxy_evidence_boundary,
    ordered_unique,
    truthy_authority_field_violations,
)

SCHEMA = "inverse_steganalysis_acquisition_plan.v1"
ATOM_SCHEMA = "inverse_steganalysis_atom.v1"
OBSERVATION_SCHEMA = "inverse_steganalysis_observation.v1"
PRIORITY_SCHEMA = "inverse_steganalysis_acquisition_priority.v1"
ACTION_FUNCTIONAL_SCHEMA = "inverse_steganalysis_discrete_action_functional.v1"
ACTION_CELL_SCHEMA = "inverse_steganalysis_action_cell.v1"
TOOL = "tac.optimization.inverse_steganalysis_acquisition"
CONTEST_RATE_SCORE_PER_BYTE = 25.0 / 50_000_000.0

ALLOWED_SCALES = frozenset(
    {
        "candidate",
        "frame",
        "pair",
        "frame_pair",
        "region",
        "frequency",
        "byte",
        "component",
        "coherence",
        "region_frequency",
        "byte_range",
        "multiscale",
    }
)
SCOPE_AXES = frozenset(
    {
        "bytes",
        "pixels",
        "regions",
        "boundaries",
        "frames",
        "pairs",
        "batches",
        "full_video",
    }
)
SCALE_TO_SCOPE_AXIS = {
    "byte": "bytes",
    "byte_range": "bytes",
    "region": "regions",
    "region_frequency": "regions",
    "frame": "frames",
    "frame_pair": "pairs",
    "pair": "pairs",
    "frequency": "pixels",
    "component": "full_video",
    "coherence": "batches",
    "candidate": "full_video",
    "multiscale": "full_video",
}
RUNTIME_IDENTITY_KEYS = frozenset(
    {
        "runtime_tree_sha256",
        "runtime_manifest_sha256",
        "runtime_sha256",
        "inflate_runtime_sha256",
        "scorer_runtime_sha256",
        "scorer_version",
        "runtime_contract_sha256",
    }
)
CACHE_IDENTITY_KEYS = frozenset(
    {
        "cache_sha256",
        "cache_key",
        "input_cache_sha256",
        "raw_sha256",
        "inflated_outputs_aggregate_sha256",
        "array_sha256",
        "candidate_cache_array_sha256",
        "reference_cache_array_sha256",
        "pair_indices_sha256",
    }
)
AUTHORITY_FIELDS = tuple(
    dict.fromkeys(
        (
            *CONSUMER_PAYLOAD_FORBIDDEN_TRUE_AUTHORITY_FIELDS,
            *PROXY_FALSE_AUTHORITY_FIELDS.keys(),
        )
    )
)
RESOURCE_MULTIPLIERS: dict[str, float] = {
    "local_mlx": 1.0,
    "macos_mlx": 1.0,
    "macos_mlx_research_signal": 1.0,
    "local_cpu": 1.25,
    "macos_cpu_advisory": 1.25,
    "local_io_heavy": 1.5,
    "local_gpu": 2.0,
    "remote_cpu": 3.0,
    "modal_t4": 4.0,
    "remote_gpu": 5.0,
    "contest_exact_eval": 8.0,
}
DEFAULT_RESOURCE_MULTIPLIER = 2.0
MIN_ELAPSED_SECONDS = 1.0
MIN_ARTIFACT_GB = 1.0e-6


class InverseSteganalysisAcquisitionError(ValueError):
    """Raised when inverse-steganalysis planning rows are malformed."""


@dataclass(frozen=True)
class AcquisitionPriorityTerms:
    """Resource-normalized score-lowering priority terms."""

    predicted_score_gain: float
    expected_score_gain: float
    uncertainty_bonus: float
    calibration_penalty: float
    elapsed_seconds: float
    artifact_bytes: int
    artifact_gb: float
    resource_kind: str
    resource_multiplier: float
    score_gain_per_second: float
    score_gain_per_gb: float
    acquisition_priority: float

    def to_dict(self) -> dict[str, Any]:
        return {"schema": PRIORITY_SCHEMA, **self.__dict__}


def normalize_inverse_steganalysis_atom(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a multiscale inverse-scorer atom and force proxy authority."""

    _reject_truthy_authority(row, label="inverse-steganalysis atom")
    scale = _scale(row.get("scale"))
    effects = _predicted_effects(row)
    action_terms = _action_surface_terms(row, effects)
    out = {
        "schema": ATOM_SCHEMA,
        "atom_id": _text(row.get("atom_id"), "atom_id"),
        "candidate_id": _text(row.get("candidate_id"), "candidate_id"),
        "scale": scale,
        "scope_axis": _scope_axis(row.get("scope_axis"), scale),
        "parent_unit_id": _optional_text(row.get("parent_unit_id")),
        "frame_range": _range(row.get("frame_range"), "frame_range"),
        "pair_indices": _int_list(row.get("pair_indices"), "pair_indices"),
        "region_bbox": _bbox(row.get("region_bbox")),
        "frequency_band": _optional_text(row.get("frequency_band")),
        "byte_range": _range(row.get("byte_range"), "byte_range"),
        "component": _text(row.get("component"), "component"),
        "coherence_group": _optional_text(row.get("coherence_group")),
        "sparsity_prior": _float(row.get("sparsity_prior", 0.0), "sparsity_prior", minimum=0.0),
        **effects,
        **action_terms,
        "uncertainty": _float(
            row.get("uncertainty", row.get("prediction_uncertainty", 0.0)),
            "uncertainty",
            minimum=0.0,
        ),
        "calibration_error": _float(row.get("calibration_error", 0.0), "calibration_error", minimum=0.0),
        "elapsed_seconds": _float_or_none(row.get("elapsed_seconds"), "elapsed_seconds", minimum=0.0, exclusive=True),
        "artifact_bytes": _int(row.get("artifact_bytes", row.get("source_artifact_bytes", 0)), "artifact_bytes", minimum=0),
        "resource_kind": _resource(row.get("resource_kind", "local_cpu")),
        "candidate_generation_only": True,
        "planning_only": True,
        "allowed_use": "planning_rank_for_candidate_generation_or_exact_eval_followup",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        "tool": TOOL,
    }
    return _false_authority(
        out,
        "inverse_steganalysis_atom_is_not_score_authority",
        "requires_byte_closed_archive_and_exact_auth_eval_before_promotion",
    )


def normalize_inverse_steganalysis_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a local/proxy scorer observation used by acquisition ranking."""

    _reject_truthy_authority(row, label="inverse-steganalysis observation")
    candidate_id = _text(row.get("candidate_id"), "candidate_id")
    axis = _text(_first(row.get("axis"), row.get("score_axis")), "axis")
    out = {
        "schema": OBSERVATION_SCHEMA,
        "observation_id": _optional_text(row.get("observation_id")) or f"obs_{_slug(candidate_id)}_{_slug(axis)}",
        "candidate_id": candidate_id,
        "axis": axis,
        "axis_normalized": _token(axis),
        "source_path": _optional_text(row.get("source_path")),
        "runtime_identity": _identity(row.get("runtime_identity"), RUNTIME_IDENTITY_KEYS, "runtime_identity"),
        "cache_identity": _identity(row.get("cache_identity"), CACHE_IDENTITY_KEYS, "cache_identity"),
        "observed_score_gain": _float_or_none(
            _first(
                row.get("observed_score_gain"),
                row.get("observed_scorer_gain_vs_baseline"),
                row.get("normalized_full_video_scorer_gain_vs_baseline"),
            ),
            "observed_score_gain",
            minimum=0.0,
        ),
        "calibration_error": _float(
            _first(
                row.get("calibration_error"),
                row.get("absolute_calibration_error"),
                row.get("calibration_uncertainty_score"),
                0.0,
            ),
            "calibration_error",
            minimum=0.0,
        ),
        "elapsed_seconds": _float_or_none(row.get("elapsed_seconds"), "elapsed_seconds", minimum=0.0, exclusive=True),
        "artifact_bytes": _int(_first(row.get("artifact_bytes"), row.get("source_artifact_bytes"), 0), "artifact_bytes", minimum=0),
        "resource_kind": _resource(row.get("resource_kind", "local_cpu")),
        "candidate_generation_only": True,
        "planning_only": True,
        "allowed_use": "local_or_proxy_acquisition_ranking_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        "tool": TOOL,
    }
    return _false_authority(
        out,
        "inverse_steganalysis_observation_is_not_score_authority",
        "local_or_proxy_observation_requires_exact_auth_eval_before_promotion",
    )


def compute_acquisition_priority(
    atom: Mapping[str, Any],
    observation: Mapping[str, Any] | None = None,
    *,
    uncertainty_weight: float = 0.25,
) -> dict[str, Any]:
    """Compute score-gain priority normalized by time, artifact bytes, resource."""

    atom_row = (
        dict(atom)
        if atom.get("schema") == ATOM_SCHEMA
        else normalize_inverse_steganalysis_atom(atom)
    )
    obs_row = None
    if observation is not None:
        obs_row = (
            dict(observation)
            if observation.get("schema") == OBSERVATION_SCHEMA
            else normalize_inverse_steganalysis_observation(observation)
        )
    if not math.isfinite(uncertainty_weight) or uncertainty_weight < 0.0:
        raise InverseSteganalysisAcquisitionError("uncertainty_weight must be finite and non-negative")

    predicted_gain = _float(atom_row.get("predicted_score_gain"), "predicted_score_gain", minimum=0.0)
    first_order = _float(atom_row.get("first_order_marginal_effect"), "first_order_marginal_effect")
    second_order = _float(atom_row.get("second_order_interaction_effect"), "second_order_interaction_effect")
    fragility_penalty = _float(atom_row.get("fragility_penalty"), "fragility_penalty", minimum=0.0)
    observed_gain = _float_or_none(
        None if obs_row is None else obs_row.get("observed_score_gain"),
        "observed_score_gain",
        minimum=0.0,
    )
    action_gain = max(0.0, first_order + second_order)
    base_gain = max(predicted_gain, action_gain) if observed_gain is None else observed_gain
    uncertainty_bonus = _float(atom_row.get("uncertainty"), "uncertainty", minimum=0.0) * uncertainty_weight
    calibration_penalty = _float(
        _first(
            None if obs_row is None else obs_row.get("calibration_error"),
            atom_row.get("calibration_error"),
            0.0,
        ),
        "calibration_error",
        minimum=0.0,
    )
    elapsed_seconds = max(
        _float(
            _first(
                None if obs_row is None else obs_row.get("elapsed_seconds"),
                atom_row.get("elapsed_seconds"),
                MIN_ELAPSED_SECONDS,
            ),
            "elapsed_seconds",
            minimum=0.0,
            exclusive=True,
        ),
        MIN_ELAPSED_SECONDS,
    )
    artifact_bytes = _int(
        _first(
            None if obs_row is None else obs_row.get("artifact_bytes"),
            atom_row.get("artifact_bytes"),
            0,
        ),
        "artifact_bytes",
        minimum=0,
    )
    resource_kind = _resource(
        _first(
            None if obs_row is None else obs_row.get("resource_kind"),
            atom_row.get("resource_kind"),
            "local_cpu",
        )
    )
    resource_multiplier = RESOURCE_MULTIPLIERS.get(resource_kind, DEFAULT_RESOURCE_MULTIPLIER)
    expected_gain = max(0.0, base_gain - calibration_penalty - fragility_penalty) + uncertainty_bonus
    artifact_gb = max(artifact_bytes / 1_000_000_000.0, MIN_ARTIFACT_GB)
    terms = AcquisitionPriorityTerms(
        predicted_score_gain=predicted_gain,
        expected_score_gain=expected_gain,
        uncertainty_bonus=uncertainty_bonus,
        calibration_penalty=calibration_penalty,
        elapsed_seconds=elapsed_seconds,
        artifact_bytes=artifact_bytes,
        artifact_gb=artifact_gb,
        resource_kind=resource_kind,
        resource_multiplier=resource_multiplier,
        score_gain_per_second=expected_gain / elapsed_seconds,
        score_gain_per_gb=expected_gain / artifact_gb,
        acquisition_priority=(expected_gain / elapsed_seconds) / (resource_multiplier * (1.0 + artifact_gb)),
    )
    return terms.to_dict()


def build_inverse_steganalysis_acquisition_plan(
    atoms: Iterable[Mapping[str, Any]],
    *,
    observations: Iterable[Mapping[str, Any]] = (),
    top_k: int | None = None,
) -> dict[str, Any]:
    """Rank atoms using optional local/proxy calibration observations."""

    obs_rows = [normalize_inverse_steganalysis_observation(row) for row in observations]
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for obs in obs_rows:
        by_candidate.setdefault(str(obs["candidate_id"]), []).append(obs)

    ranked: list[dict[str, Any]] = []
    for index, raw_atom in enumerate(atoms):
        atom = normalize_inverse_steganalysis_atom(raw_atom)
        candidate_obs = by_candidate.get(str(atom["candidate_id"]), [])
        best_obs = _best_observation(atom, candidate_obs)
        ranked.append(
            _false_authority(
                {
                    **atom,
                    "source_atom_index": index,
                    "best_observation_id": None if best_obs is None else best_obs["observation_id"],
                    "observation_count": len(candidate_obs),
                    "priority": compute_acquisition_priority(atom, best_obs),
                },
                "ranked_inverse_steganalysis_row_is_planning_only",
                "requires_exact_eval_before_score_or_promotion_claim",
            )
        )

    ranked.sort(
        key=lambda row: (
            -float(row["priority"]["acquisition_priority"]),
            -float(row["priority"]["expected_score_gain"]),
            float(row["priority"]["elapsed_seconds"]),
            str(row["atom_id"]),
        )
    )
    if top_k is not None:
        if top_k < 1:
            raise InverseSteganalysisAcquisitionError("top_k must be positive")
        ranked = ranked[:top_k]
    for rank, row in enumerate(ranked, start=1):
        row["acquisition_rank"] = rank

    return _false_authority(
        {
            "schema": SCHEMA,
            "tool": TOOL,
            "candidate_generation_only": True,
            "planning_only": True,
            "authority": "false_authority_proxy_acquisition_only",
            "ranked_atoms": ranked,
            "summary": {
                "atom_count": len(ranked),
                "observation_count": len(obs_rows),
                "top_atom_id": None if not ranked else ranked[0]["atom_id"],
                "top_candidate_id": None if not ranked else ranked[0]["candidate_id"],
            },
        },
        "inverse_steganalysis_acquisition_plan_is_not_score_authority",
        "planner_rank_requires_exact_eval_before_promotion_or_rank_kill",
    )


def build_discrete_scorer_action_functional(
    atoms: Iterable[Mapping[str, Any]],
    *,
    observations: Iterable[Mapping[str, Any]] = (),
    total_byte_budget: int | None = None,
    lambda_rate: float = CONTEST_RATE_SCORE_PER_BYTE,
) -> dict[str, Any]:
    """Approximate hydrated auth eval as a coupled discrete action surface.

    The returned rows are a Riemann-sum style planning model over byte, pixel,
    region, frame, pair, batch, and full-video cells.  They carry local first-
    order score marginals, second-order synergy/antagonism terms, discontinuity
    barriers, and a rate shadow price so deterministic materializers can choose
    the next water bucket without treating proxy evidence as score authority.
    """

    if not math.isfinite(lambda_rate) or lambda_rate < 0.0:
        raise InverseSteganalysisAcquisitionError("lambda_rate must be finite and non-negative")
    if total_byte_budget is not None and total_byte_budget < 1:
        raise InverseSteganalysisAcquisitionError("total_byte_budget must be positive")

    obs_rows = [normalize_inverse_steganalysis_observation(row) for row in observations]
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for obs in obs_rows:
        by_candidate.setdefault(str(obs["candidate_id"]), []).append(obs)

    cells: list[dict[str, Any]] = []
    total_first_order = 0.0
    total_second_order = 0.0
    total_synergy = 0.0
    total_antagonism = 0.0
    total_fragility_penalty = 0.0
    total_expected_gain = 0.0
    blocked_cells = 0
    for index, raw_atom in enumerate(atoms):
        atom = normalize_inverse_steganalysis_atom(raw_atom)
        best_obs = _best_observation(atom, by_candidate.get(str(atom["candidate_id"]), []))
        priority = compute_acquisition_priority(atom, best_obs)
        measure = _cell_measure(atom)
        first_order = _float(atom["first_order_marginal_effect"], "first_order_marginal_effect")
        second_order = _float(atom["second_order_interaction_effect"], "second_order_interaction_effect")
        synergy = _float(atom["synergy_effect"], "synergy_effect", minimum=0.0)
        antagonism = _float(atom["antagonism_effect"], "antagonism_effect", minimum=0.0)
        fragility = _float(atom["fragility_penalty"], "fragility_penalty", minimum=0.0)
        expected_gain = _float(priority["expected_score_gain"], "expected_score_gain", minimum=0.0)
        byte_cost = int(measure["water_fill_cost_bytes"])
        marginal_utility = expected_gain / float(byte_cost)
        residual = marginal_utility - lambda_rate
        guard = dict(atom["discontinuity_guard"])
        blocked = bool(guard.get("blocked"))
        if blocked:
            blocked_cells += 1
        total_first_order += first_order
        total_second_order += second_order
        total_synergy += synergy
        total_antagonism += antagonism
        total_fragility_penalty += fragility
        total_expected_gain += expected_gain
        cells.append(
            _false_authority(
                {
                    "schema": ACTION_CELL_SCHEMA,
                    "cell_index": index,
                    "atom_id": atom["atom_id"],
                    "candidate_id": atom["candidate_id"],
                    "scale": atom["scale"],
                    "scope_axis": atom["scope_axis"],
                    "component": atom["component"],
                    "measure": measure,
                    "first_order_marginal_effect": first_order,
                    "second_order_interaction_effect": second_order,
                    "synergy_effect": synergy,
                    "antagonism_effect": antagonism,
                    "fragility_penalty": fragility,
                    "expected_score_gain": expected_gain,
                    "lambda_rate": lambda_rate,
                    "marginal_utility_per_byte": marginal_utility,
                    "euler_lagrange_residual": residual,
                    "water_bucket_selectable": residual > 0.0 and not blocked,
                    "discontinuity_guard": guard,
                    "best_observation_id": None if best_obs is None else best_obs["observation_id"],
                    "priority": priority,
                },
                "inverse_steganalysis_action_cell_is_planning_only",
                "requires_materialized_archive_and_exact_auth_eval_before_score_claim",
            )
        )

    water_bucket = _water_bucket_fill(cells, total_byte_budget=total_byte_budget)
    cells.sort(
        key=lambda row: (
            -float(row["euler_lagrange_residual"]),
            -float(row["expected_score_gain"]),
            int(row["measure"]["water_fill_cost_bytes"]),
            str(row["atom_id"]),
        )
    )
    return _false_authority(
        {
            "schema": ACTION_FUNCTIONAL_SCHEMA,
            "tool": TOOL,
            "candidate_generation_only": True,
            "planning_only": True,
            "authority": "false_authority_discrete_action_surface_only",
            "math_model": {
                "representation": "discrete_riemann_sum_with_second_order_interactions",
                "coordinates": [
                    "bytes",
                    "pixels",
                    "regions",
                    "boundaries",
                    "frames",
                    "pairs",
                    "batches",
                    "full_video",
                    "scorer_component",
                ],
                "objective_terms": [
                    "segnet_error_field",
                    "posenet_geometry_field",
                    "rate_shadow_price",
                    "second_order_synergy_antagonism_kernel",
                    "discontinuity_barrier",
                    "calibration_residual",
                ],
                "stationarity_rule": "select positive euler_lagrange_residual cells under byte budget and guard barriers",
                "lambda_rate": lambda_rate,
            },
            "integral_totals": {
                "cell_count": len(cells),
                "blocked_cell_count": blocked_cells,
                "first_order_marginal_effect_sum": total_first_order,
                "second_order_interaction_effect_sum": total_second_order,
                "synergy_effect_sum": total_synergy,
                "antagonism_effect_sum": total_antagonism,
                "fragility_penalty_sum": total_fragility_penalty,
                "expected_score_gain_sum": total_expected_gain,
                "net_action_gain_after_fragility": max(
                    0.0,
                    total_first_order
                    + total_second_order
                    - total_fragility_penalty,
                ),
            },
            "water_bucket": water_bucket,
            "cells": cells,
        },
        "inverse_steganalysis_discrete_action_functional_is_not_score_authority",
        "requires_byte_closed_candidate_generation_before_dispatch",
        "requires_exact_auth_eval_before_promotion_or_rank_kill",
    )


def _best_observation(atom: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not observations:
        return None

    def key(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
        priority = compute_acquisition_priority(atom, row)
        return (
            float(priority["acquisition_priority"]),
            float(priority["expected_score_gain"]),
            -float(priority["elapsed_seconds"]),
            str(row.get("observation_id")),
        )

    return dict(max(observations, key=key))


def _false_authority(row: Mapping[str, Any], *blockers: str) -> dict[str, Any]:
    return apply_proxy_evidence_boundary(
        dict(row),
        dispatch_blockers=ordered_unique(
            ("inverse_steganalysis_acquisition_false_authority_only", *blockers)
        ),
    )


def _reject_truthy_authority(row: Mapping[str, Any], *, label: str) -> None:
    violations = truthy_authority_field_violations(row, fields=AUTHORITY_FIELDS)
    if violations:
        raise InverseSteganalysisAcquisitionError(
            f"{label}: forbidden truthy authority fields: {', '.join(violations)}"
        )


def _predicted_effects(row: Mapping[str, Any]) -> dict[str, float]:
    seg = _float(row.get("predicted_segnet_gain", row.get("predicted_segnet_score_gain", 0.0)), "predicted_segnet_gain", minimum=0.0)
    pose = _float(row.get("predicted_posenet_gain", row.get("predicted_posenet_score_gain", 0.0)), "predicted_posenet_gain", minimum=0.0)
    rate_gain = _float(row.get("predicted_rate_gain", row.get("predicted_rate_score_gain", 0.0)), "predicted_rate_gain", minimum=0.0)
    rate_cost = _float(row.get("predicted_rate_cost", row.get("predicted_rate_score_cost", 0.0)), "predicted_rate_cost", minimum=0.0)
    explicit = _float_or_none(row.get("predicted_score_gain"), "predicted_score_gain", minimum=0.0)
    delta = _float_or_none(row.get("predicted_delta_vs_baseline_score"), "predicted_delta_vs_baseline_score")
    return {
        "predicted_segnet_gain": seg,
        "predicted_posenet_gain": pose,
        "predicted_rate_gain": rate_gain,
        "predicted_rate_cost": rate_cost,
        "predicted_score_gain": explicit if explicit is not None else (max(0.0, -delta) if delta is not None else max(0.0, seg + pose + rate_gain - rate_cost)),
    }


def _action_surface_terms(
    row: Mapping[str, Any],
    effects: Mapping[str, float],
) -> dict[str, Any]:
    first_order = _float(
        row.get("first_order_marginal_effect", effects["predicted_score_gain"]),
        "first_order_marginal_effect",
    )
    second_order = _float(
        row.get(
            "second_order_interaction_effect",
            row.get("synergy_effect", row.get("antagonism_effect", 0.0)),
        ),
        "second_order_interaction_effect",
    )
    discontinuity_risk = _float(
        row.get("discontinuity_risk", row.get("fragility_risk", 0.0)),
        "discontinuity_risk",
        minimum=0.0,
    )
    fragility_penalty = _float(
        row.get("fragility_penalty", discontinuity_risk * abs(first_order + second_order)),
        "fragility_penalty",
        minimum=0.0,
    )
    guard_threshold = _float_or_none(
        row.get("discontinuity_threshold"),
        "discontinuity_threshold",
        minimum=0.0,
    )
    guard_blocked = guard_threshold is not None and discontinuity_risk > guard_threshold
    return {
        "first_order_marginal_effect": first_order,
        "second_order_interaction_effect": second_order,
        "interaction_kind": _interaction_kind(second_order),
        "synergy_effect": max(0.0, second_order),
        "antagonism_effect": max(0.0, -second_order),
        "discontinuity_risk": discontinuity_risk,
        "fragility_penalty": fragility_penalty,
        "discontinuity_guard": {
            "schema": "inverse_steganalysis_discontinuity_guard.v1",
            "risk": discontinuity_risk,
            "threshold": guard_threshold,
            "blocked": guard_blocked,
            "blocker": "discontinuity_risk_exceeds_threshold" if guard_blocked else None,
        },
    }


def _cell_measure(atom: Mapping[str, Any]) -> dict[str, Any]:
    byte_span = _span(atom.get("byte_range"))
    frame_span = _span(atom.get("frame_range"))
    pair_count = len(atom.get("pair_indices") or []) if isinstance(atom.get("pair_indices"), list) else 0
    region_area = _region_area(atom.get("region_bbox"))
    component_count = 1 if atom.get("component") else 0
    water_fill_cost_bytes = max(1, byte_span or _int(atom.get("artifact_bytes", 0), "artifact_bytes", minimum=0))
    return {
        "schema": "inverse_steganalysis_action_cell_measure.v1",
        "byte_span": byte_span,
        "frame_span": frame_span,
        "pair_count": pair_count,
        "region_area": region_area,
        "component_count": component_count,
        "water_fill_cost_bytes": water_fill_cost_bytes,
    }


def _water_bucket_fill(
    cells: Sequence[Mapping[str, Any]],
    *,
    total_byte_budget: int | None,
) -> dict[str, Any]:
    ordered = sorted(
        (dict(cell) for cell in cells if bool(cell.get("water_bucket_selectable"))),
        key=lambda row: (
            -float(row["euler_lagrange_residual"]),
            -float(row["expected_score_gain"]),
            int(row["measure"]["water_fill_cost_bytes"]),
            str(row["atom_id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    used_bytes = 0
    expected_gain = 0.0
    for row in ordered:
        cost = int(row["measure"]["water_fill_cost_bytes"])
        if total_byte_budget is not None and used_bytes + cost > total_byte_budget:
            continue
        selected.append(
            {
                "atom_id": row["atom_id"],
                "candidate_id": row["candidate_id"],
                "scope_axis": row["scope_axis"],
                "component": row["component"],
                "water_fill_cost_bytes": cost,
                "expected_score_gain": row["expected_score_gain"],
                "euler_lagrange_residual": row["euler_lagrange_residual"],
            }
        )
        used_bytes += cost
        expected_gain += float(row["expected_score_gain"])
    return {
        "schema": "inverse_steganalysis_water_bucket_plan.v1",
        "total_byte_budget": total_byte_budget,
        "selected_count": len(selected),
        "selected_water_fill_cost_bytes": used_bytes,
        "selected_expected_score_gain": expected_gain,
        "selected_cells": selected,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _span(value: Any) -> int:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 2:
        return 0
    start = _int(value[0], "range[0]", minimum=0)
    end = _int(value[1], "range[1]", minimum=0)
    return max(0, end - start)


def _region_area(value: Any) -> float:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 4:
        return 0.0
    x_min = _float(value[0], "region_bbox[0]")
    y_min = _float(value[1], "region_bbox[1]")
    x_max = _float(value[2], "region_bbox[2]")
    y_max = _float(value[3], "region_bbox[3]")
    return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


def action_surface_terms(atom: Mapping[str, Any]) -> dict[str, Any]:
    """Return the domain-math/action terms from a normalized or raw atom."""

    row = (
        dict(atom)
        if atom.get("schema") == ATOM_SCHEMA
        else normalize_inverse_steganalysis_atom(atom)
    )
    return {
        "scope_axis": row["scope_axis"],
        "first_order_marginal_effect": row["first_order_marginal_effect"],
        "second_order_interaction_effect": row["second_order_interaction_effect"],
        "interaction_kind": row["interaction_kind"],
        "synergy_effect": row["synergy_effect"],
        "antagonism_effect": row["antagonism_effect"],
        "discontinuity_risk": row["discontinuity_risk"],
        "fragility_penalty": row["fragility_penalty"],
        "discontinuity_guard": dict(row["discontinuity_guard"]),
    }


def _interaction_kind(value: float) -> str:
    if value > 0.0:
        return "synergy"
    if value < 0.0:
        return "antagonism"
    return "neutral"


def _identity(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise InverseSteganalysisAcquisitionError(f"{label} must be a non-empty object")
    out = dict(value)
    if not any(_has_value(out.get(key)) for key in keys):
        raise InverseSteganalysisAcquisitionError(f"{label} must include one of: {', '.join(sorted(keys))}")
    for key, item in out.items():
        if key.endswith("sha256") and item is not None:
            _sha256_value(item, f"{label}.{key}")
    return out


def _sha256_value(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _sha256_value(item, f"{label}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _sha256_value(item, f"{label}[{index}]")
        return
    if not is_sha256_hex(value):
        raise InverseSteganalysisAcquisitionError(f"{label} must be sha256 hex")


def _has_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_has_value(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_has_value(item) for item in value)
    return value not in (None, "")


def _scale(value: Any) -> str:
    scale = _token(_text(value, "scale"))
    if scale not in ALLOWED_SCALES:
        raise InverseSteganalysisAcquisitionError(f"scale must be one of {sorted(ALLOWED_SCALES)}, got {scale!r}")
    return scale


def _scope_axis(value: Any, scale: str) -> str:
    scope = _token(value) if value is not None else SCALE_TO_SCOPE_AXIS[scale]
    if scope not in SCOPE_AXES:
        raise InverseSteganalysisAcquisitionError(f"scope_axis must be one of {sorted(SCOPE_AXES)}, got {scope!r}")
    return scope


def _resource(value: Any) -> str:
    return _token(_text(value, "resource_kind")) or "local_cpu"


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().strip("[]").lower()).strip("_")


def _slug(value: Any) -> str:
    return _token(value)[:64] or "unknown"


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text(value: Any, label: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise InverseSteganalysisAcquisitionError(f"{label} must be a non-empty string")
    return text


def _range(value: Any, label: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 2:
        raise InverseSteganalysisAcquisitionError(f"{label} must be [start, end]")
    start = _int(value[0], f"{label}[0]", minimum=0)
    end = _int(value[1], f"{label}[1]", minimum=0)
    if end < start:
        raise InverseSteganalysisAcquisitionError(f"{label} end must be >= start")
    return [start, end]


def _int_list(value: Any, label: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise InverseSteganalysisAcquisitionError(f"{label} must be a list")
    out = [_int(item, f"{label}[{index}]", minimum=0) for index, item in enumerate(value)]
    if len(set(out)) != len(out):
        raise InverseSteganalysisAcquisitionError(f"{label} contains duplicates")
    return out


def _bbox(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 4:
        raise InverseSteganalysisAcquisitionError("region_bbox must be [x0, y0, x1, y1]")
    bbox = [_float(coord, f"region_bbox[{index}]") for index, coord in enumerate(value)]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise InverseSteganalysisAcquisitionError("region_bbox max coordinates must exceed min")
    return bbox


def _int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise InverseSteganalysisAcquisitionError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise InverseSteganalysisAcquisitionError(f"{label} must be an integer") from exc
    if minimum is not None and result < minimum:
        raise InverseSteganalysisAcquisitionError(f"{label} must be >= {minimum}")
    return result


def _float_or_none(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    exclusive: bool = False,
) -> float | None:
    if value is None:
        return None
    return _float(value, label, minimum=minimum, exclusive=exclusive)


def _float(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    exclusive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise InverseSteganalysisAcquisitionError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InverseSteganalysisAcquisitionError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise InverseSteganalysisAcquisitionError(f"{label} must be finite")
    if minimum is not None and (result <= minimum if exclusive else result < minimum):
        op = ">" if exclusive else ">="
        raise InverseSteganalysisAcquisitionError(f"{label} must be {op} {minimum}")
    return result


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


__all__ = [
    "ACTION_CELL_SCHEMA",
    "ACTION_FUNCTIONAL_SCHEMA",
    "ALLOWED_SCALES",
    "ATOM_SCHEMA",
    "CONTEST_RATE_SCORE_PER_BYTE",
    "OBSERVATION_SCHEMA",
    "PRIORITY_SCHEMA",
    "RESOURCE_MULTIPLIERS",
    "SCHEMA",
    "SCOPE_AXES",
    "TOOL",
    "AcquisitionPriorityTerms",
    "InverseSteganalysisAcquisitionError",
    "action_surface_terms",
    "build_discrete_scorer_action_functional",
    "build_inverse_steganalysis_acquisition_plan",
    "compute_acquisition_priority",
    "normalize_inverse_steganalysis_atom",
    "normalize_inverse_steganalysis_observation",
]
