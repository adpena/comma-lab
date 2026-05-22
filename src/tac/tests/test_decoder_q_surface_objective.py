from __future__ import annotations

import math

from tac.optimization.decoder_q_surface_objective import (
    build_surface_objective,
    sort_atoms_for_surface_objective,
    summarize_candidate_surface_proxy,
)


def _surface() -> dict:
    return {
        "schema": "decoder_q_response_surface_plan.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "summary": {
            "matched_count": 300,
            "preserve_candidate_effect_count": 76,
            "suppress_or_invert_candidate_effect_count": 224,
            "preserve_gain_sum": 0.02,
            "suppress_harm_sum": 0.10,
            "axis_dominance_counts": {"seg": 216, "pose": 84},
        },
    }


def test_surface_objective_prefers_suppress_when_harm_dominates() -> None:
    objective = build_surface_objective(_surface())

    assert objective["score_claim"] is False
    assert objective["strategy"] == "suppress_or_invert_regressions_first"
    assert objective["dominant_axis"] == "seg"
    assert math.isclose(objective["harm_to_gain_ratio"], 5.0)


def test_surface_objective_sorts_atoms_by_dominant_axis_priority() -> None:
    objective = build_surface_objective(_surface())
    atoms = [
        {
            "candidate_id": "pose-heavy",
            "mutation": {"tensor_name": "a", "q_offset": 1, "delta": -1},
            "target_mass": 10.0,
            "axis_mass": {"seg": 1.0, "pose": 9.0, "rate": 0.0},
        },
        {
            "candidate_id": "seg-heavy",
            "mutation": {"tensor_name": "b", "q_offset": 2, "delta": -1},
            "target_mass": 10.0,
            "axis_mass": {"seg": 9.0, "pose": 1.0, "rate": 0.0},
        },
    ]

    ranked = sort_atoms_for_surface_objective(
        atoms,
        objective,
        preferred_direction="suppress",
    )
    proxy = summarize_candidate_surface_proxy(ranked[:1], objective)

    assert ranked[0]["candidate_id"] == "seg-heavy"
    assert ranked[0]["surface_objective"]["preferred_direction"] == "suppress"
    assert proxy is not None
    assert proxy["score_claim"] is False
    assert proxy["dominant_axis"] == "seg"
    assert proxy["dominant_axis_mass_sum"] == 9.0
