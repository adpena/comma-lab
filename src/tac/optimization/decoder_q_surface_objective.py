# SPDX-License-Identifier: MIT
"""Surface-guided objective helpers for decoder-q waterbucket planning."""

from __future__ import annotations

import math
from typing import Any

DECODER_Q_SURFACE_OBJECTIVE_SCHEMA = "decoder_q_surface_objective.v1"
DECODER_Q_RESPONSE_SURFACE_SCHEMA = "decoder_q_response_surface_plan.v1"

_FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "promotion_eligible",
    "ready_for_exact_eval_dispatch",
    "rank_or_kill_eligible",
    "promotable",
)


class DecoderQSurfaceObjectiveError(ValueError):
    """Raised when a decoder-q response surface is not usable as an objective."""


def build_surface_objective(response_surface: dict[str, Any]) -> dict[str, Any]:
    """Convert a response surface into a non-authoritative waterbucket objective."""

    if not isinstance(response_surface, dict):
        raise DecoderQSurfaceObjectiveError("response_surface must be a JSON object")
    if response_surface.get("schema") != DECODER_Q_RESPONSE_SURFACE_SCHEMA:
        raise DecoderQSurfaceObjectiveError("response_surface schema mismatch")
    for field in _FALSE_AUTHORITY_FIELDS:
        if response_surface.get(field) is not False:
            raise DecoderQSurfaceObjectiveError(f"response_surface {field} must be false")
    summary = response_surface.get("summary")
    if not isinstance(summary, dict):
        raise DecoderQSurfaceObjectiveError("response_surface summary missing")

    preserve_gain = _finite_float(summary.get("preserve_gain_sum")) or 0.0
    suppress_harm = _finite_float(summary.get("suppress_harm_sum")) or 0.0
    matched_count = int(_finite_float(summary.get("matched_count")) or 0)
    if matched_count <= 0:
        raise DecoderQSurfaceObjectiveError("response_surface matched_count must be positive")
    axis_counts = summary.get("axis_dominance_counts")
    if not isinstance(axis_counts, dict):
        axis_counts = {}
    dominant_axis = _dominant_axis(axis_counts)
    strategy = (
        "suppress_or_invert_regressions_first"
        if suppress_harm > preserve_gain
        else "preserve_improvements_first"
    )
    return {
        "schema": DECODER_Q_SURFACE_OBJECTIVE_SCHEMA,
        "producer": "tac.optimization.decoder_q_surface_objective",
        "source_schema": response_surface.get("schema"),
        "strategy": strategy,
        "dominant_axis": dominant_axis,
        "matched_count": matched_count,
        "preserve_candidate_effect_count": int(_finite_float(summary.get("preserve_candidate_effect_count")) or 0),
        "suppress_or_invert_candidate_effect_count": int(_finite_float(summary.get("suppress_or_invert_candidate_effect_count")) or 0),
        "preserve_gain_sum": preserve_gain,
        "suppress_harm_sum": suppress_harm,
        "harm_to_gain_ratio": suppress_harm / max(preserve_gain, 1.0e-12),
        "axis_dominance_counts": axis_counts,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def sort_atoms_for_surface_objective(
    atoms: list[dict[str, Any]],
    objective: dict[str, Any],
    *,
    preferred_direction: str,
) -> list[dict[str, Any]]:
    """Return atoms annotated and ranked by the surface objective."""

    if objective.get("schema") != DECODER_Q_SURFACE_OBJECTIVE_SCHEMA:
        raise DecoderQSurfaceObjectiveError("surface objective schema mismatch")
    annotated = [
        annotate_atom(atom, objective, preferred_direction=preferred_direction)
        for atom in atoms
    ]
    annotated.sort(
        key=lambda atom: (
            -float(atom["surface_objective"]["proxy_priority"]),
            abs(int(atom["mutation"]["delta"])),
            str(atom["mutation"]["tensor_name"]),
            int(atom["mutation"]["q_offset"]),
            int(atom["mutation"]["delta"]),
        )
    )
    return annotated


def annotate_atom(
    atom: dict[str, Any],
    objective: dict[str, Any],
    *,
    preferred_direction: str,
) -> dict[str, Any]:
    """Attach objective metadata to one atom without changing its mutation."""

    dominant_axis = str(objective.get("dominant_axis") or "seg")
    target_mass = _finite_float(atom.get("target_mass")) or 0.0
    axis_mass = atom.get("axis_mass") if isinstance(atom.get("axis_mass"), dict) else {}
    dominant_axis_mass = _finite_float(axis_mass.get(dominant_axis)) or 0.0
    total_axis_mass = sum(
        _finite_float(axis_mass.get(axis)) or 0.0
        for axis in ("seg", "pose", "rate")
    )
    dominant_axis_share = (
        dominant_axis_mass / total_axis_mass
        if total_axis_mass > 0.0
        else 0.0
    )
    if preferred_direction == "suppress":
        direction_weight = _finite_float(objective.get("harm_to_gain_ratio")) or 1.0
    else:
        direction_weight = 1.0
    proxy_priority = target_mass * (1.0 + dominant_axis_share) * direction_weight
    out = dict(atom)
    out["surface_objective"] = {
        "schema": DECODER_Q_SURFACE_OBJECTIVE_SCHEMA,
        "strategy": objective.get("strategy"),
        "preferred_direction": preferred_direction,
        "dominant_axis": dominant_axis,
        "dominant_axis_mass": dominant_axis_mass,
        "dominant_axis_share": dominant_axis_share,
        "target_mass": target_mass,
        "proxy_priority": proxy_priority,
        "score_claim": False,
    }
    return out


def summarize_candidate_surface_proxy(
    atoms: list[dict[str, Any]],
    objective: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Summarize objective proxy mass for a materialized candidate."""

    if objective is None:
        return None
    dominant_axis = str(objective.get("dominant_axis") or "seg")
    target_mass_sum = 0.0
    dominant_axis_mass_sum = 0.0
    proxy_priority_sum = 0.0
    for atom in atoms:
        surface = atom.get("surface_objective")
        if isinstance(surface, dict):
            target_mass_sum += _finite_float(surface.get("target_mass")) or 0.0
            dominant_axis_mass_sum += _finite_float(surface.get("dominant_axis_mass")) or 0.0
            proxy_priority_sum += _finite_float(surface.get("proxy_priority")) or 0.0
        else:
            target_mass_sum += _finite_float(atom.get("target_mass")) or 0.0
            axis_mass = atom.get("axis_mass") if isinstance(atom.get("axis_mass"), dict) else {}
            dominant_axis_mass_sum += _finite_float(axis_mass.get(dominant_axis)) or 0.0
    return {
        "schema": DECODER_Q_SURFACE_OBJECTIVE_SCHEMA,
        "strategy": objective.get("strategy"),
        "dominant_axis": dominant_axis,
        "target_mass_sum": target_mass_sum,
        "dominant_axis_mass_sum": dominant_axis_mass_sum,
        "proxy_priority_sum": proxy_priority_sum,
        "preserve_gain_sum": objective.get("preserve_gain_sum"),
        "suppress_harm_sum": objective.get("suppress_harm_sum"),
        "score_claim": False,
    }


def _dominant_axis(axis_counts: dict[str, Any]) -> str:
    if not axis_counts:
        return "seg"
    return max(
        axis_counts,
        key=lambda axis: int(_finite_float(axis_counts.get(axis)) or 0),
    )


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


__all__ = [
    "DECODER_Q_SURFACE_OBJECTIVE_SCHEMA",
    "DecoderQSurfaceObjectiveError",
    "annotate_atom",
    "build_surface_objective",
    "sort_atoms_for_surface_objective",
    "summarize_candidate_surface_proxy",
]
