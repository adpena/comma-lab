from __future__ import annotations

import math

import pytest

from tac.procedural_codebook_generator.seed_budget_allocation import (
    RATE_SCORE_PER_BYTE,
    SeedBudgetAllocationError,
    allocate_seed_budget_from_frame_sensitivity,
)


def _decomposition() -> dict:
    return {
        "schema": "per_frame_decomposition_segnet_per_frame_posenet_per_pair_v1",
        "topology": "non_overlapping",
        "n_pairs": 2,
        "n_frames": 4,
        "top_frames": [
            {
                "rank": 1,
                "frame_index": 1,
                "total_l1": 3.0,
                "seg_l1": 2.0,
                "pose_l1": 1.0,
                "rate_l1": 0.0,
            },
            {
                "rank": 2,
                "frame_index": 3,
                "total_l1": 1.0,
                "seg_l1": 0.5,
                "pose_l1": 0.5,
                "rate_l1": 0.0,
            },
        ],
    }


def test_allocate_seed_budget_from_frame_sensitivity_allocates_by_frame_l1() -> None:
    out = allocate_seed_budget_from_frame_sensitivity(
        procedural_candidate={
            "n_codebook_bytes": 4096,
            "affected_frame_indices": [1, 3],
        },
        per_frame_decomposition=_decomposition(),
        default_seed_bytes=16,
    )

    assert out["schema"] == "procedural_codebook_seed_budget_allocation_v1"
    assert out["allocation_status"] == "allocated"
    assert out["score_claim"] is False
    assert out["promotion_eligible"] is False
    assert out["ready_for_exact_eval_dispatch"] is False
    assert out["recommended_k_seed_bytes"] == 16
    assert out["allocation"][0]["frame_index"] == 1
    assert out["allocation"][0]["seed_budget_hint_bytes"] == 12
    assert out["allocation"][1]["frame_index"] == 3
    assert out["allocation"][1]["seed_budget_hint_bytes"] == 4


def test_allocate_seed_budget_lists_formula_consistent_candidate_deltas() -> None:
    out = allocate_seed_budget_from_frame_sensitivity(
        procedural_candidate={
            "n_codebook_bytes": 4096,
            "frame_indices": [1],
        },
        per_frame_decomposition=_decomposition(),
        seed_budget_candidates=(16, 32),
        default_seed_bytes=32,
    )

    by_k = {row["k_seed_bytes"]: row for row in out["seed_budget_candidates"]}
    assert math.isclose(
        by_k[16]["predicted_delta_s"],
        -RATE_SCORE_PER_BYTE * (4096 - 16),
    )
    assert math.isclose(
        by_k[32]["predicted_delta_s"],
        -RATE_SCORE_PER_BYTE * (4096 - 32),
    )


def test_allocate_seed_budget_requires_explicit_frame_scope() -> None:
    out = allocate_seed_budget_from_frame_sensitivity(
        procedural_candidate={"n_codebook_bytes": 4096},
        per_frame_decomposition=_decomposition(),
    )

    assert out["allocation_status"] == "missing_frame_scope"
    assert out["recommended_k_seed_bytes"] is None
    assert out["score_claim"] is False


def test_allocate_seed_budget_rejects_invalid_inputs() -> None:
    with pytest.raises(SeedBudgetAllocationError):
        allocate_seed_budget_from_frame_sensitivity(
            procedural_candidate={"n_codebook_bytes": 0, "frame_indices": [1]},
            per_frame_decomposition=_decomposition(),
        )

    with pytest.raises(SeedBudgetAllocationError):
        allocate_seed_budget_from_frame_sensitivity(
            procedural_candidate={"n_codebook_bytes": 4096, "frame_indices": [-1]},
            per_frame_decomposition=_decomposition(),
        )
