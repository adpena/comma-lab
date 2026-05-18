# SPDX-License-Identifier: MIT
"""Planning-only field-equation layer for cross-paradigm archive atoms.

The planner treats candidate archive edits as small perturbation atoms around
the current exact frontier. It does not evaluate archives and never promotes a
score. Its job is to put atoms from masks, poses, HNeRV weights, sidechannels,
entropy models, foveation fields, and categorical labels into one deterministic
variational/KKT/Volterra planning surface.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.optimization.meta_lagrangian_allocator import rate_score_delta
from tac.optimization.bayesian_experimental_design import expected_improvement_minimize
from tac.optimization.research_basis import (
    research_basis_ids_for_family,
    research_basis_manifest,
)

SCHEMA_VERSION = 2
TOOL = "tac.optimization.field_equation_planner.build_field_equation_plan"
DEFAULT_CONSTRAINTS = {
    "max_byte_delta": 0,
    "max_expected_seg_dist_delta": 0.0,
    "max_expected_pose_dist_delta": 0.0,
    "min_confidence": 0.0,
    "require_pareto_frontier": True,
    "lambda_byte_violation": 1.0,
    "lambda_seg_violation": 100.0,
    "lambda_pose_violation": 10.0,
    "lambda_confidence_violation": 1.0,
    "lambda_rankable_violation": 1_000.0,
    "lambda_byte_closure_violation": 100.0,
    "lambda_proxy_violation": 50.0,
    "lambda_pareto_violation": 25.0,
    "lambda_pareto_dominated_violation": 5.0,
    "lambda_kkt_readiness_violation": 10.0,
}


class FieldEquationPlannerError(ValueError):
    """Raised when field-equation planner inputs are malformed."""


def _unique_ordered_strings(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _as_float(value: Any, *, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FieldEquationPlannerError(f"{key} must be numeric")
    return float(value)


def _as_bool(value: Any, *, key: str) -> bool:
    if not isinstance(value, bool):
        raise FieldEquationPlannerError(f"{key} must be boolean")
    return value


def _constraints(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_CONSTRAINTS)
    for key, value in (overrides or {}).items():
        if key not in out:
            raise FieldEquationPlannerError(f"unknown constraint: {key}")
        out[key] = value
    for key in (
        "max_byte_delta",
        "max_expected_seg_dist_delta",
        "max_expected_pose_dist_delta",
        "min_confidence",
        "lambda_byte_violation",
        "lambda_seg_violation",
        "lambda_pose_violation",
        "lambda_confidence_violation",
        "lambda_rankable_violation",
        "lambda_byte_closure_violation",
        "lambda_proxy_violation",
        "lambda_pareto_violation",
        "lambda_pareto_dominated_violation",
        "lambda_kkt_readiness_violation",
    ):
        out[key] = _as_float(out[key], key=key)
    out["require_pareto_frontier"] = _as_bool(
        out["require_pareto_frontier"],
        key="require_pareto_frontier",
    )
    return out


def _rows_from_ledger(atom_ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = atom_ledger.get("rows")
    if not isinstance(rows, list):
        raise FieldEquationPlannerError("atom ledger must contain rows list")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def frechet_derivatives(row: Mapping[str, Any]) -> dict[str, float]:
    """Return first-order derivatives with respect to atom amplitude epsilon."""

    byte_delta = _as_float(row.get("byte_delta"), key="byte_delta")
    seg_delta = _as_float(row.get("expected_seg_dist_delta"), key="expected_seg_dist_delta")
    pose_delta = _as_float(row.get("expected_pose_dist_delta"), key="expected_pose_dist_delta")
    score_delta = _as_float(row.get("expected_total_score_delta"), key="expected_total_score_delta")
    return {
        "d_score_d_epsilon": round(score_delta, 12),
        "d_seg_dist_d_epsilon": round(seg_delta, 12),
        "d_pose_dist_d_epsilon": round(pose_delta, 12),
        "d_bytes_d_epsilon": round(byte_delta, 12),
        "d_rate_score_d_epsilon": round(rate_score_delta(byte_delta), 12),
    }


def _violation(value: float, limit: float) -> float:
    return max(0.0, value - limit)


def _confidence_violation(confidence: float, minimum: float) -> float:
    return max(0.0, minimum - confidence)


def _float_or_default(value: Any, *, default: float, key: str) -> float:
    if value is None:
        return default
    return _as_float(value, key=key)


def _constraint_residuals(row: Mapping[str, Any], constraints: Mapping[str, Any]) -> dict[str, float]:
    byte_delta = _as_float(row.get("byte_delta"), key="byte_delta")
    seg_delta = _as_float(row.get("expected_seg_dist_delta"), key="expected_seg_dist_delta")
    pose_delta = _as_float(row.get("expected_pose_dist_delta"), key="expected_pose_dist_delta")
    confidence = _as_float(row.get("confidence"), key="confidence")
    return {
        "byte_violation": round(_violation(byte_delta, float(constraints["max_byte_delta"])), 12),
        "seg_violation": round(
            _violation(seg_delta, float(constraints["max_expected_seg_dist_delta"])),
            12,
        ),
        "pose_violation": round(
            _violation(pose_delta, float(constraints["max_expected_pose_dist_delta"])),
            12,
        ),
        "confidence_violation": round(
            _confidence_violation(confidence, float(constraints["min_confidence"])),
            12,
        ),
    }


def variational_action_delta(row: Mapping[str, Any], constraints: Mapping[str, Any]) -> float:
    """Return planning action delta with KKT-style constraint penalties."""

    residuals = _constraint_residuals(row, constraints)
    action = _as_float(row.get("expected_total_score_delta"), key="expected_total_score_delta")
    action += float(constraints["lambda_byte_violation"]) * residuals["byte_violation"]
    action += float(constraints["lambda_seg_violation"]) * residuals["seg_violation"]
    action += float(constraints["lambda_pose_violation"]) * residuals["pose_violation"]
    action += float(constraints["lambda_confidence_violation"]) * residuals["confidence_violation"]
    action += sum(_selection_constraint_penalties(row, constraints).values())
    return round(action, 12)


def _selection_constraint_penalties(row: Mapping[str, Any], constraints: Mapping[str, Any]) -> dict[str, float]:
    terms: dict[str, float] = {}
    if row.get("rankable") is not True:
        terms["non_rankable_atom"] = float(constraints["lambda_rankable_violation"])
    if row.get("byte_closed_archive_manifest_attached") is not True:
        terms["missing_byte_closed_archive_manifest"] = float(
            constraints["lambda_byte_closure_violation"]
        )
    if row.get("proxy_row") is True:
        terms["proxy_row"] = float(constraints["lambda_proxy_violation"])
    if constraints["require_pareto_frontier"]:
        if row.get("pareto_eligible") is not True:
            terms["pareto_ineligible_atom"] = float(constraints["lambda_pareto_violation"])
        elif row.get("pareto_frontier") is not True:
            terms["pareto_dominated_atom"] = float(
                constraints["lambda_pareto_dominated_violation"]
            )
    if row.get("kkt_ready_for_field_planning") is not True:
        terms["kkt_not_ready_for_field_planning"] = float(
            constraints["lambda_kkt_readiness_violation"]
        )
    return terms


def _kkt_report(row: Mapping[str, Any], constraints: Mapping[str, Any]) -> dict[str, Any]:
    residuals = _constraint_residuals(row, constraints)
    selection_penalties = _selection_constraint_penalties(row, constraints)
    action = variational_action_delta(row, constraints)
    stationarity = action
    blockers: list[str] = list(row.get("kkt_blockers") or [])
    if row.get("rankable") is not True:
        blockers.append("atom_not_rankable")
    if constraints["require_pareto_frontier"]:
        if row.get("pareto_eligible") is False:
            blockers.append("pareto_ineligible_atom")
        elif row.get("pareto_frontier") is False:
            blockers.append("pareto_dominated_atom")
    for key, value in residuals.items():
        if value > 0:
            blockers.append(key)
    if row.get("byte_closed_archive_manifest_attached") is not True:
        blockers.append("missing_byte_closed_archive_manifest")
    if row.get("proxy_row") is True:
        blockers.append("proxy_evidence_not_kkt_ready")
    if not list(row.get("interaction_assumptions") or []):
        blockers.append("missing_interaction_assumptions")
    blockers = _unique_ordered_strings(blockers)
    return {
        "stationarity_residual": round(stationarity, 12),
        "constraint_residuals": residuals,
        "selection_penalty_terms": {
            key: round(value, 12) for key, value in sorted(selection_penalties.items())
        },
        "selection_penalty_score_delta": round(sum(selection_penalties.values()), 12),
        "locally_descending": bool(stationarity < 0 and not blockers),
        "kkt_blockers": blockers,
    }


def _description_length(row: Mapping[str, Any]) -> dict[str, Any]:
    byte_delta = _as_float(row.get("byte_delta"), key="byte_delta")
    model_delta = row.get("model_byte_delta", 0)
    data_delta = row.get("data_byte_delta", byte_delta)
    if isinstance(model_delta, bool) or not isinstance(model_delta, int | float):
        model_delta = 0
    if isinstance(data_delta, bool) or not isinstance(data_delta, int | float):
        data_delta = byte_delta
    total = float(model_delta) + float(data_delta)
    return {
        "model_byte_delta": round(float(model_delta), 12),
        "data_byte_delta": round(float(data_delta), 12),
        "description_length_delta_bytes": round(total, 12),
        "description_length_rate_score_delta": round(rate_score_delta(total), 12),
    }


def field_row(row: Mapping[str, Any], constraints: Mapping[str, Any]) -> dict[str, Any]:
    atom_id = str(row.get("atom_id") or "")
    if not atom_id:
        raise FieldEquationPlannerError("atom row missing atom_id")
    dispatch_blockers = list(row.get("dispatch_blockers") or [])
    kkt = _kkt_report(row, constraints)
    for blocker in kkt["kkt_blockers"]:
        if blocker not in dispatch_blockers:
            dispatch_blockers.append(blocker)
    dispatch_blockers.extend(
        blocker
        for blocker in (
            "field_equation_planning_only",
            "research_basis_is_not_score_evidence",
            "requires_exact_cuda_auth_eval",
            "requires_stack_interaction_review",
        )
        if blocker not in dispatch_blockers
    )
    basis_ids = _research_basis_ids(row)
    return {
        "atom_id": atom_id,
        "family": str(row.get("family") or "unknown"),
        "family_group": str(row.get("family_group") or row.get("family") or "unknown"),
        "pareto_scope": str(row.get("pareto_scope") or row.get("family_group") or "unknown"),
        "conflicts_with_families": sorted({str(item) for item in row.get("conflicts_with_families") or [] if str(item)}),
        "conflicts_with_atoms": sorted({str(item) for item in row.get("conflicts_with_atoms") or [] if str(item)}),
        "pareto_eligible": bool(row.get("pareto_eligible")),
        "pareto_frontier": bool(row.get("pareto_frontier")),
        "pareto_dominated_by": list(row.get("pareto_dominated_by") or []),
        "rankable": bool(row.get("rankable")),
        "confidence": _as_float(row.get("confidence"), key="confidence"),
        "evidence_grade": str(row.get("evidence_grade") or "prediction"),
        "byte_closed_archive_manifest_attached": bool(
            row.get("byte_closed_archive_manifest_attached")
        ),
        "archive_manifest_path": str(row.get("archive_manifest_path") or ""),
        "archive_manifest_sha256": str(row.get("archive_manifest_sha256") or ""),
        "research_basis_ids": basis_ids,
        "proxy_row": bool(row.get("proxy_row")),
        "expected_information_gain_nats": round(
            _float_or_default(
                row.get("expected_information_gain_nats"),
                default=0.0,
                key="expected_information_gain_nats",
            ),
            12,
        ),
        "expected_score_variance": round(
            _float_or_default(
                row.get("expected_score_variance"),
                default=0.0,
                key="expected_score_variance",
            ),
            12,
        ),
        "observation_noise_variance": round(
            _float_or_default(
                row.get("observation_noise_variance"),
                default=0.0,
                key="observation_noise_variance",
            ),
            12,
        ),
        "expected_uncertainty_reduction": dict(row.get("expected_uncertainty_reduction") or {}),
        "kkt_ready_for_field_planning": bool(row.get("kkt_ready_for_field_planning")),
        "frechet_derivatives": frechet_derivatives(row),
        "variational_action_delta": variational_action_delta(row, constraints),
        "meta_lagrangian_selection_score_delta": round(
            _float_or_default(
                row.get("selection_score_delta"),
                default=_as_float(row.get("expected_total_score_delta"), key="expected_total_score_delta"),
                key="selection_score_delta",
            ),
            12,
        ),
        "meta_lagrangian_selection_penalty_terms": dict(row.get("selection_penalty_terms") or {}),
        "kkt": kkt,
        "mdl": _description_length(row),
        "interaction_assumptions": list(row.get("interaction_assumptions") or []),
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatchable": False,
        "dispatch_blockers": dispatch_blockers,
    }


def _research_basis_ids(row: Mapping[str, Any]) -> list[str]:
    explicit = [
        str(item)
        for item in row.get("research_basis_ids") or []
        if str(item)
    ]
    inferred = research_basis_ids_for_family(
        str(row.get("family") or ""),
        str(row.get("family_group") or ""),
        str(row.get("pareto_scope") or ""),
        str(row.get("paradigm") or ""),
    )
    out: list[str] = []
    for basis_id in [*explicit, *inferred]:
        if basis_id not in out:
            out.append(basis_id)
    return out


def _interaction_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def _row_family_tokens(row: Mapping[str, Any]) -> set[str]:
    return {
        text
        for value in (
            row.get("family"),
            row.get("family_group"),
            row.get("pareto_scope"),
        )
        if value is not None
        for text in (str(value),)
        if text
    }


def _declared_conflict_blockers(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    atom_a = str(a.get("atom_id") or "")
    atom_b = str(b.get("atom_id") or "")
    if atom_b in {
        text
        for item in a.get("conflicts_with_atoms") or []
        if item is not None
        for text in (str(item),)
        if text
    }:
        blockers.append("declared_atom_conflict")
    if atom_a in {
        text
        for item in b.get("conflicts_with_atoms") or []
        if item is not None
        for text in (str(item),)
        if text
    }:
        blockers.append("declared_atom_conflict")
    a_family_conflicts = {
        text
        for item in a.get("conflicts_with_families") or []
        if item is not None
        for text in (str(item),)
        if text
    }
    b_family_conflicts = {
        text
        for item in b.get("conflicts_with_families") or []
        if item is not None
        for text in (str(item),)
        if text
    }
    if a_family_conflicts & _row_family_tokens(b):
        blockers.append("declared_family_conflict")
    if b_family_conflicts & _row_family_tokens(a):
        blockers.append("declared_family_conflict")
    return _unique_ordered_strings(blockers)


def _stack_pair_compatibility(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    for row in (a, b):
        if row.get("rankable") is not True:
            blockers.append("atom_not_rankable")
        if row.get("byte_closed_archive_manifest_attached") is not True:
            blockers.append("missing_byte_closed_archive_manifest")
        if row.get("pareto_eligible") is not True:
            blockers.append("pareto_ineligible_atom")
        elif row.get("pareto_frontier") is not True:
            blockers.append("pareto_dominated_atom")
        if row.get("kkt_ready_for_field_planning") is not True:
            blockers.append("kkt_not_ready_for_field_planning")
        if row.get("proxy_row") is True:
            blockers.append("proxy_evidence_not_stack_compatible")
    blockers.extend(_declared_conflict_blockers(a, b))
    blockers = _unique_ordered_strings(blockers)
    return {
        "compatible_for_stack_planning": not blockers,
        "compatibility_blockers": blockers,
    }


def volterra_interaction_rows(
    field_rows: Iterable[Mapping[str, Any]],
    interactions: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build second-order pair rows from supplied interaction deltas."""

    by_id = {str(row["atom_id"]): row for row in field_rows}
    out = []
    seen: set[tuple[str, str]] = set()
    for item in interactions or []:
        atom_a = str(item.get("atom_a") or "")
        atom_b = str(item.get("atom_b") or "")
        if atom_a not in by_id or atom_b not in by_id or atom_a == atom_b:
            continue
        key = _interaction_key(atom_a, atom_b)
        if key in seen:
            continue
        seen.add(key)
        first_a = by_id[atom_a]["frechet_derivatives"]
        first_b = by_id[atom_b]["frechet_derivatives"]
        second_score = _as_float(item.get("score_delta", 0.0), key="score_delta")
        second_seg = _as_float(item.get("seg_dist_delta", 0.0), key="seg_dist_delta")
        second_pose = _as_float(item.get("pose_dist_delta", 0.0), key="pose_dist_delta")
        second_bytes = _as_float(item.get("byte_delta", 0.0), key="byte_delta")
        compatibility = _stack_pair_compatibility(by_id[atom_a], by_id[atom_b])
        total_score = (
            float(first_a["d_score_d_epsilon"])
            + float(first_b["d_score_d_epsilon"])
            + second_score
        )
        out.append(
            {
                "interaction_id": f"volterra:{key[0]}+{key[1]}",
                "atom_ids": [key[0], key[1]],
                "volterra_order": 2,
                "first_order_score_delta": round(
                    float(first_a["d_score_d_epsilon"]) + float(first_b["d_score_d_epsilon"]),
                    12,
                ),
                "second_order_score_delta": round(second_score, 12),
                "combined_score_delta": round(total_score, 12),
                "combined_seg_dist_delta": round(
                    float(first_a["d_seg_dist_d_epsilon"])
                    + float(first_b["d_seg_dist_d_epsilon"])
                    + second_seg,
                    12,
                ),
                "combined_pose_dist_delta": round(
                    float(first_a["d_pose_dist_d_epsilon"])
                    + float(first_b["d_pose_dist_d_epsilon"])
                    + second_pose,
                    12,
                ),
                "combined_byte_delta": round(
                    float(first_a["d_bytes_d_epsilon"])
                    + float(first_b["d_bytes_d_epsilon"])
                    + second_bytes,
                    12,
                ),
                "interaction_assumption": str(item.get("assumption") or "measured_or_planned_pair_delta"),
                **compatibility,
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_blockers": [
                    "volterra_interaction_planning_only",
                    *compatibility["compatibility_blockers"],
                    "requires_exact_stacked_archive_cuda_eval",
                ],
            }
        )
    out.sort(key=lambda row: (float(row["combined_score_delta"]), row["interaction_id"]))
    return out


def _field_acquisition_row(
    row: Mapping[str, Any],
    *,
    base_score: float | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    score_delta = float(row["frechet_derivatives"]["d_score_d_epsilon"])
    predicted_score_mean: float | None = None
    expected_improvement = 0.0
    if base_score is None:
        blockers.append("missing_base_score_for_expected_improvement")
    else:
        predicted_score_mean = float(base_score) + score_delta
        expected_improvement = expected_improvement_minimize(
            predicted_score_mean,
            float(row["expected_score_variance"]),
            float(base_score),
        )
    if row.get("rankable") is not True:
        blockers.append("atom_not_rankable")
    if row.get("pareto_eligible") is not True:
        blockers.append("pareto_ineligible_atom")
    elif row.get("pareto_frontier") is not True:
        blockers.append("pareto_dominated_atom")
    if row.get("kkt_ready_for_field_planning") is not True:
        blockers.append("kkt_not_ready_for_field_planning")
    if row.get("byte_closed_archive_manifest_attached") is not True:
        blockers.append("missing_byte_closed_archive_manifest")
    if row.get("proxy_row") is True:
        blockers.append("proxy_evidence_not_dispatch_design_ready")
    if row["kkt"]["kkt_blockers"]:
        blockers.append("kkt_blockers_present")
    action_bonus = max(0.0, -float(row["variational_action_delta"]))
    information_gain = float(row["expected_information_gain_nats"])
    acquisition_value = expected_improvement + information_gain + action_bonus
    return {
        "atom_id": str(row["atom_id"]),
        "family": str(row["family"]),
        "pareto_scope": str(row["pareto_scope"]),
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "predicted_score_mean": (
            round(predicted_score_mean, 12)
            if predicted_score_mean is not None
            else None
        ),
        "expected_improvement": round(expected_improvement, 12),
        "expected_information_gain_nats": round(information_gain, 12),
        "descending_action_bonus": round(action_bonus, 12),
        "acquisition_value": round(acquisition_value, 12),
        "acquisition_formula": "EI_minimize(base_score + d_score) + EIG_nats + max(0, -variational_action_delta)",
        "design_ready": not blockers,
        "dispatch_design_blockers": _unique_ordered_strings(blockers),
    }


def field_acquisition_ranking(
    field_rows: Iterable[Mapping[str, Any]],
    *,
    base_score: float | None,
) -> dict[str, Any]:
    """Return planning-only Bayesian/KKT acquisition ranking for field rows."""

    rows = [
        _field_acquisition_row(row, base_score=base_score)
        for row in field_rows
    ]
    rows.sort(
        key=lambda row: (
            not bool(row["design_ready"]),
            -float(row["acquisition_value"]),
            -float(row["expected_improvement"]),
            -float(row["expected_information_gain_nats"]),
            str(row["atom_id"]),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    blocker_counts: Counter[str] = Counter()
    for row in rows:
        blocker_counts.update(row["dispatch_design_blockers"])
    return {
        "schema": "field_acquisition_ranking_v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "base_score": base_score,
        "objective": (
            "Choose the next exact CUDA observation by expected improvement, "
            "expected information gain, and variational descent after KKT/Pareto/"
            "byte-closure blockers are explicit."
        ),
        "design_ready_count": sum(1 for row in rows if row["design_ready"]),
        "row_count": len(rows),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "rows": rows,
    }


def build_field_equation_plan(
    atom_ledger: Mapping[str, Any],
    *,
    source: str,
    constraints: Mapping[str, Any] | None = None,
    interactions: Iterable[Mapping[str, Any]] | None = None,
    base_score: float | None = None,
    research_basis_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Convert a meta-Lagrangian atom ledger into a field-equation plan."""

    resolved_constraints = _constraints(constraints)
    rows = [field_row(row, resolved_constraints) for row in _rows_from_ledger(atom_ledger)]
    rows.sort(
        key=lambda row: (
            not bool(row["rankable"]),
            not bool(row["pareto_eligible"]),
            not bool(row["pareto_frontier"]),
            not bool(row["kkt_ready_for_field_planning"]),
            float(row["variational_action_delta"]),
            str(row["family_group"]),
            str(row["pareto_scope"]),
            str(row["atom_id"]),
        )
    )
    interaction_rows = volterra_interaction_rows(rows, interactions)
    all_basis_ids: list[str] = []
    for basis_id in research_basis_ids or []:
        if basis_id not in all_basis_ids:
            all_basis_ids.append(str(basis_id))
    for row in rows:
        for basis_id in row["research_basis_ids"]:
            if basis_id not in all_basis_ids:
                all_basis_ids.append(basis_id)
    descending = [row for row in rows if row["kkt"]["locally_descending"]]
    kkt_blocker_counts: Counter[str] = Counter()
    for row in rows:
        kkt_blocker_counts.update(row["kkt"]["kkt_blockers"])
    volterra_blocker_counts: Counter[str] = Counter()
    for row in interaction_rows:
        volterra_blocker_counts.update(row["compatibility_blockers"])
    independent_score_delta = round(
        sum(float(row["frechet_derivatives"]["d_score_d_epsilon"]) for row in descending),
        12,
    )
    compatible_interaction_rows = [
        row for row in interaction_rows if row["compatible_for_stack_planning"]
    ]
    acquisition = field_acquisition_ranking(rows, base_score=base_score)
    best_pair_delta = (
        float(compatible_interaction_rows[0]["combined_score_delta"])
        if compatible_interaction_rows
        else 0.0
    )
    floor_estimate: dict[str, Any] = {
        "evidence_grade": "derivation",
        "score_claim": False,
        "assumption": "independent_locally_descending_atoms_plus_best_compatible_nonconflicting_pair_interaction",
        "independent_score_delta": independent_score_delta,
        "best_pair_combined_score_delta": round(best_pair_delta, 12),
        "base_score": base_score,
        "optimistic_floor_score": None,
        "blockers": [
            "nonconvex_proxy_bound",
            "missing_exact_stacked_archive_eval",
            "missing_higher_order_interaction_terms",
        ],
    }
    if base_score is not None:
        floor_estimate["optimistic_floor_score"] = round(
            float(base_score) + min(independent_score_delta, best_pair_delta),
            12,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "source": source,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "objective": (
            "J = expected_total_score_delta + lambda_byte*g_byte + "
            "lambda_seg*g_seg + lambda_pose*g_pose + lambda_confidence*g_confidence"
        ),
        "constraints": resolved_constraints,
        "research_basis": research_basis_manifest(all_basis_ids or None),
        "atom_count": len(rows),
        "pareto_eligible_count": sum(1 for row in rows if row["pareto_eligible"]),
        "kkt_ready_for_field_planning_count": sum(
            1 for row in rows if row["kkt_ready_for_field_planning"]
        ),
        "proxy_row_count": sum(1 for row in rows if row["proxy_row"]),
        "kkt_blocker_counts": dict(sorted(kkt_blocker_counts.items())),
        "locally_descending_count": len(descending),
        "volterra_interaction_count": len(interaction_rows),
        "compatible_volterra_interaction_count": len(compatible_interaction_rows),
        "blocked_volterra_interaction_count": len(interaction_rows) - len(compatible_interaction_rows),
        "volterra_blocker_counts": dict(sorted(volterra_blocker_counts.items())),
        "field_acquisition_ranking": acquisition,
        "theoretical_floor_estimate": floor_estimate,
        "trainable_surrogate": {
            "target": "minimize_variational_action_delta",
            "equation": (
                "J_theta = score_delta_theta + lambda_rate*byte_violation_theta + "
                "lambda_seg*seg_violation_theta + lambda_pose*pose_violation_theta + "
                "lambda_conf*confidence_violation_theta"
            ),
            "gradient_fields": [
                "d_score_d_epsilon",
                "d_seg_dist_d_epsilon",
                "d_pose_dist_d_epsilon",
                "d_bytes_d_epsilon",
            ],
            "trainable_parameters": [
                "atom_amplitude_epsilon",
                "selector_logits",
                "byte_allocator_lambdas",
                "interaction_kernel",
            ],
            "determinism_contract": (
                "training may choose archive atoms, but emitted archive bytes and "
                "inflate behavior must be deterministic and byte-closed"
            ),
        },
        "rows": rows,
        "volterra_interactions": interaction_rows,
        "dispatch_blockers": [
            "field_equation_planning_only",
            "requires_byte_closed_candidate_archive",
            "requires_exact_cuda_auth_eval",
            "requires_empirical_interaction_calibration",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# GAP-3 closure: per-pair Lagrangian dual variable consumption
# (MEDIUM gap closure wave 2026-05-17 — `lane_medium_gap_closure_3_optimization_
# modules_per_pair_consumption_20260517`).
#
# Per the comprehensive wiring + integration pass identifying that this planner
# is AGGREGATE-ONLY at the constraint-multiplier surface: the just-landed
# OptimalPerPairTreatmentPlan (sister `a2ab518a` Lagrangian-dual planner)
# exposes the dual variables that already SOLVE the per-pair budget under
# archive/compute/inflate KKT constraints. Binding those into the field-
# equation planner's lambda_* knobs eliminates a manual re-derivation step.
#
# The contract: when an OptimalPlan exists for the archive, lift the planner's
# dual variables (lambda_archive, lambda_compute, lambda_inflate) onto the
# planner's `lambda_*_violation` constraint multipliers and the per-pair nu
# values onto a typed per-pair constraint envelope. When no plan exists, the
# planner uses the existing default constraints (legacy behavior preserved).
#
# Per CLAUDE.md "Apples-to-apples evidence discipline" + "Meta-Lagrangian/Pareto
# solver" non-negotiable: the bound dual variables are PREDICTIONS, never
# score claims. Downstream consumers must adjudicate any field-equation
# verdict against contest CUDA + CPU evidence before promotion.
# ─────────────────────────────────────────────────────────────────────────────


PER_PAIR_LAGRANGIAN_DUAL_SCHEMA = "tac_field_equation_planner_per_pair_lagrangian_dual_v1"

# ─────────────────────────────────────────────────────────────────────────────
# Sidecar emission for forward-compat BUCKET C autopilot consumer (sister #817
# + #818 wave producer-consumer loop closure 2026-05-17 per #818 op-routable #2;
# closed in `lane_817_sidecar_emission_wire_in_bucket_c_loop_closure_20260517`).
#
# This is the 3rd producer in the loop closure (bonus coverage). The current
# `tools/cathedral_autopilot_autonomous_loop.py` does NOT yet have a consumer
# for the field_equation_per_pair_lagrangian sidecar — but emitting the sidecar
# is forward-compat for the next autopilot extension wave AND lets downstream
# audit tooling query the bound Lagrangian dual envelope per-archive without
# re-running the planner.
#
# Sidecar path:
#   .omx/state/master_gradient_consumers/field_equation_per_pair_lagrangian_<sha[:12]>_<utc>.json
# Schema: field_equation_per_pair_lagrangian_sidecar_v1
# ─────────────────────────────────────────────────────────────────────────────

_FIELD_EQ_SIDECAR_DIR_PARTS: tuple[str, ...] = (
    ".omx",
    "state",
    "master_gradient_consumers",
)
FIELD_EQUATION_PER_PAIR_LAGRANGIAN_SIDECAR_SCHEMA = (
    "field_equation_per_pair_lagrangian_sidecar_v1"
)


def _field_eq_sidecar_dir() -> Path:
    """Resolve the canonical .omx/state/master_gradient_consumers/ directory."""
    repo_root = Path(__file__).resolve().parents[3]
    sidecar_dir = repo_root.joinpath(*_FIELD_EQ_SIDECAR_DIR_PARTS)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    return sidecar_dir


def _persist_field_equation_per_pair_lagrangian_sidecar(
    envelope: Mapping[str, Any],
) -> Path:
    """Persist a field_equation_per_pair_lagrangian sidecar.

    Per Catalog #131 (fcntl-locked writes to shared state) + Catalog #245
    (atomic-write-no-tmp-leakage). The envelope is a typed dict (NOT a
    dataclass) so the entire envelope is included at the top level of the
    sidecar payload + key `lambda_archive` / `lambda_compute` / `lambda_inflate`
    / `nu_per_pair` / `bound_constraints` mirrored to TOP LEVEL for direct read.
    """
    sidecar_dir = _field_eq_sidecar_dir()
    archive_sha256 = str(envelope.get("archive_sha256", ""))
    if not archive_sha256:
        raise FieldEquationPlannerError(
            "envelope missing archive_sha256; cannot persist sidecar"
        )
    sha_short = archive_sha256[:12]
    utc = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "Z"
    sidecar_path = (
        sidecar_dir
        / f"field_equation_per_pair_lagrangian_{sha_short}_{utc}.json"
    )
    lock_path = (
        sidecar_dir / f".field_equation_per_pair_lagrangian_{sha_short}.lock"
    )

    # Tuple → list for JSON compatibility
    nu_per_pair_value = envelope.get("nu_per_pair", ())
    if isinstance(nu_per_pair_value, tuple):
        nu_per_pair_value = list(nu_per_pair_value)

    payload: dict[str, Any] = {
        "sidecar_schema": FIELD_EQUATION_PER_PAIR_LAGRANGIAN_SIDECAR_SCHEMA,
        "envelope_schema": envelope.get("schema"),
        "cascade_path_used": envelope.get("cascade_path_used"),
        "archive_sha256": archive_sha256,
        "lambda_archive": envelope.get("lambda_archive"),
        "lambda_compute": envelope.get("lambda_compute"),
        "lambda_inflate": envelope.get("lambda_inflate"),
        "nu_per_pair": nu_per_pair_value,
        "n_pairs": envelope.get("n_pairs", 0),
        "kkt_residual": envelope.get("kkt_residual"),
        "is_pareto_feasible": envelope.get("is_pareto_feasible"),
        "predicted_score_delta": envelope.get("predicted_score_delta"),
        "bound_constraints": dict(envelope.get("bound_constraints", {})),
        "optimal_plan_consumed": bool(envelope.get("optimal_plan_consumed", False)),
        "rationale": envelope.get("rationale", ""),
        "evidence_grade": envelope.get("evidence_grade", ""),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "written_at_utc": utc,
    }

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        try:
            tmp_path = sidecar_path.with_suffix(
                f".tmp.{uuid.uuid4().hex[:12]}"
            )
            text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
            tmp_path.write_text(text, encoding="utf-8")
            with open(tmp_path, "rb") as tf:
                os.fsync(tf.fileno())
            os.replace(tmp_path, sidecar_path)
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)

    return sidecar_path


def consume_per_pair_lagrangian_duals(
    *,
    archive_sha256: str,
    optimal_plan: Any | None = None,
    auto_load: bool = True,
    persist_sidecar: bool = True,
) -> dict[str, Any]:
    """Bind per-pair Lagrangian dual variables to the field-equation planner.

    Per Catalog #125 hook #2 (Pareto constraint hook): plugs the Lagrangian-
    dual planner's λ multipliers + per-pair ν envelope into the field-equation
    planner's constraint multiplier surface. The result is a typed dict that
    the planner's downstream consumers (autopilot ranking, archive composer)
    can ingest by schema version.

    Cascade:
      1. If ``optimal_plan`` is supplied → consume its lambda_* + nu_per_pair
         directly into the typed envelope.
      2. Else if ``auto_load=True`` → load the most-recent OptimalPlan sidecar
         for the archive via
         `tac.master_gradient_consumers.load_optimal_plan_for_archive`.
      3. Else return a fallback envelope with default constraints (no per-pair
         dual binding; legacy planner behavior preserved).

    Per CLAUDE.md "Apples-to-apples evidence discipline" the returned envelope
    carries ``score_claim=False`` and ``evidence_grade='[predicted; field-
    equation planner per-pair Lagrangian dual v1]'``.

    Parameters
    ----------
    archive_sha256
        64-char hex sha of the target archive bytes. Used to look up the
        canonical optimal-plan sidecar.
    optimal_plan
        Optional `OptimalPerPairTreatmentPlan` dataclass OR sidecar dict.
        When None and ``auto_load=True``, loaded via the canonical helper.
    auto_load
        When True (default), missing optimal_plan is auto-loaded from canonical
        sidecar.

    Returns
    -------
    Typed envelope dict with the bound dual variables + constraint surface +
    cascade evidence. Schema = ``PER_PAIR_LAGRANGIAN_DUAL_SCHEMA``.

    Raises
    ------
    FieldEquationPlannerError on invalid archive sha format.
    """
    if (
        not isinstance(archive_sha256, str)
        or len(archive_sha256) < 12
        or any(c not in "0123456789abcdefABCDEF" for c in archive_sha256)
    ):
        raise FieldEquationPlannerError(
            f"archive_sha256 must be a 12+ char hex string; got {archive_sha256!r}"
        )

    # ── Cascade: auto-load if not supplied ─────────────────────────────────
    if optimal_plan is None and auto_load:
        try:
            from tac.master_gradient_consumers import load_optimal_plan_for_archive
            optimal_plan = load_optimal_plan_for_archive(archive_sha256)
        except (ImportError, ValueError, FileNotFoundError, OSError):
            optimal_plan = None

    if optimal_plan is None:
        # Fallback envelope: no per-pair dual binding; planner uses defaults
        _fallback_envelope = {
            "schema": PER_PAIR_LAGRANGIAN_DUAL_SCHEMA,
            "cascade_path_used": "default_constraints_fallback",
            "archive_sha256": archive_sha256,
            "lambda_archive": None,
            "lambda_compute": None,
            "lambda_inflate": None,
            "nu_per_pair": (),
            "n_pairs": 0,
            "kkt_residual": None,
            "feasibility_certificate": None,
            "is_pareto_feasible": None,
            "predicted_score_delta": None,
            "bound_constraints": dict(DEFAULT_CONSTRAINTS),
            "optimal_plan_consumed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "evidence_grade": "[predicted; field-equation planner per-pair Lagrangian dual v1]",
            "rationale": (
                "[predicted; field-equation planner per-pair Lagrangian dual v1] "
                "No OptimalPerPairTreatmentPlan available; falling back to "
                "default constraints (legacy planner behavior)."
            ),
        }
        if persist_sidecar:
            _persist_field_equation_per_pair_lagrangian_sidecar(_fallback_envelope)
        return _fallback_envelope

    # ── Extract dual variables from dataclass OR dict surface ──────────────
    if hasattr(optimal_plan, "lambda_archive"):
        lambda_archive = float(optimal_plan.lambda_archive)
        lambda_compute = float(optimal_plan.lambda_compute)
        lambda_inflate = float(optimal_plan.lambda_inflate)
        nu_per_pair = tuple(float(x) for x in optimal_plan.nu_per_pair)
        kkt_residual = float(optimal_plan.kkt_residual)
        feasibility_certificate = dict(optimal_plan.feasibility_certificate)
        is_pareto_feasible = bool(optimal_plan.is_pareto_feasible)
        predicted_score_delta = float(optimal_plan.predicted_score_delta)
    elif isinstance(optimal_plan, Mapping):
        lambda_archive = float(optimal_plan.get("lambda_archive", 0.0))
        lambda_compute = float(optimal_plan.get("lambda_compute", 0.0))
        lambda_inflate = float(optimal_plan.get("lambda_inflate", 0.0))
        nu_per_pair_raw = optimal_plan.get("nu_per_pair", ())
        nu_per_pair = tuple(float(x) for x in nu_per_pair_raw)
        kkt_residual = float(optimal_plan.get("kkt_residual", 0.0))
        feasibility_certificate = dict(optimal_plan.get("feasibility_certificate", {}))
        is_pareto_feasible = bool(optimal_plan.get("is_pareto_feasible", False))
        predicted_score_delta = float(optimal_plan.get("predicted_score_delta", 0.0))
    else:
        raise FieldEquationPlannerError(
            f"optimal_plan must be OptimalPerPairTreatmentPlan or Mapping; got {type(optimal_plan).__name__}"
        )

    # ── Bind dual variables onto the field-equation planner's lambda_* knobs ──
    bound_constraints = dict(DEFAULT_CONSTRAINTS)
    bound_constraints["lambda_byte_violation"] = max(lambda_archive, 0.0)
    # Per CLAUDE.md "Meta-Lagrangian/Pareto solver": the planner's lambda_seg
    # and lambda_pose multipliers are NOT touched here — they encode the
    # per-axis scorer-distortion weights which are operating-point-specific.
    # Lambda_byte / lambda_proxy / lambda_kkt_readiness ARE bound from the
    # canonical archive / compute / inflate duals.
    bound_constraints["lambda_proxy_violation"] = max(lambda_compute * 50.0, 0.0)
    bound_constraints["lambda_kkt_readiness_violation"] = max(lambda_inflate * 10.0, 0.0)

    _bound_envelope = {
        "schema": PER_PAIR_LAGRANGIAN_DUAL_SCHEMA,
        "cascade_path_used": "optimal_plan_binding",
        "archive_sha256": archive_sha256,
        "lambda_archive": lambda_archive,
        "lambda_compute": lambda_compute,
        "lambda_inflate": lambda_inflate,
        "nu_per_pair": nu_per_pair,
        "n_pairs": len(nu_per_pair),
        "kkt_residual": kkt_residual,
        "feasibility_certificate": feasibility_certificate,
        "is_pareto_feasible": is_pareto_feasible,
        "predicted_score_delta": predicted_score_delta,
        "bound_constraints": bound_constraints,
        "optimal_plan_consumed": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[predicted; field-equation planner per-pair Lagrangian dual v1]",
        "rationale": (
            f"[predicted; field-equation planner per-pair Lagrangian dual v1] "
            f"OptimalPerPairTreatmentPlan bound: lambda_archive={lambda_archive:.4f}, "
            f"lambda_compute={lambda_compute:.4f}, lambda_inflate={lambda_inflate:.4f}; "
            f"nu_per_pair n={len(nu_per_pair)}; kkt_residual={kkt_residual:.6f}; "
            f"is_pareto_feasible={is_pareto_feasible}."
        ),
    }
    if persist_sidecar:
        _persist_field_equation_per_pair_lagrangian_sidecar(_bound_envelope)
    return _bound_envelope


__all__ = [
    "DEFAULT_CONSTRAINTS",
    "SCHEMA_VERSION",
    "TOOL",
    "PER_PAIR_LAGRANGIAN_DUAL_SCHEMA",
    "FIELD_EQUATION_PER_PAIR_LAGRANGIAN_SIDECAR_SCHEMA",
    "FieldEquationPlannerError",
    "build_field_equation_plan",
    "consume_per_pair_lagrangian_duals",
    "field_row",
    "field_acquisition_ranking",
    "frechet_derivatives",
    "variational_action_delta",
    "volterra_interaction_rows",
]
