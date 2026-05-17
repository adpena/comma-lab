# SPDX-License-Identifier: MIT
"""Tests for ``tac.optimization.bit_allocator_end_to_end.allocate_per_pair_bits``.

MEDIUM gap closure wave 2026-05-17 — `lane_medium_gap_closure_3_optimization_
modules_per_pair_consumption_20260517`. Closes GAP-2 from the comprehensive
wiring + integration audit by exercising the per-pair master gradient cascade:
OptimalPerPairTreatmentPlan → Wyner-Ziv composition → aggregate fallback.

Per CLAUDE.md "Apples-to-apples evidence discipline": every outcome carries
`[predicted; bit allocator per-pair v1]` and `score_claim=False`.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.bit_allocator_end_to_end import (
    PER_PAIR_BIT_ALLOCATION_SCHEMA,
    PER_PAIR_BUDGET_BASELINE_BYTES,
    PER_PAIR_BUDGET_MAX_CEILING_BYTES,
    PER_PAIR_BUDGET_MIN_FLOOR_BYTES,
    PerPairBitAllocationOutcome,
    allocate_per_pair_bits,
)


def test_aggregate_fallback_path_when_no_inputs() -> None:
    """No optimal plan + no gradient + no reweight → aggregate fallback."""
    result = allocate_per_pair_bits(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=100,
        auto_load=False,
    )
    assert result.cascade_path_used == "aggregate_fallback"
    assert result.total_allocated_bytes == 100
    assert result.optimal_plan_consumed is False
    assert result.sensitivity_reweight_consumed is False
    assert result.per_pair_gradient_consumed is False
    assert result.score_claim is False
    assert result.evidence_grade == "[predicted; bit allocator per-pair v1]"
    assert result.schema == PER_PAIR_BIT_ALLOCATION_SCHEMA


def test_optimal_plan_path_consumes_per_pair_assignments() -> None:
    """OptimalPlan supplied → uses predicted_delta_rate_bytes per pair."""

    # Build a synthetic OptimalPlan-like dict (matches sidecar contract)
    plan_dict = {
        "plan": [
            {"pair_index": 0, "predicted_delta_rate_bytes": 50},
            {"pair_index": 1, "predicted_delta_rate_bytes": 30},
            {"pair_index": 2, "predicted_delta_rate_bytes": 20},
        ],
    }
    result = allocate_per_pair_bits(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=200,
        optimal_plan=plan_dict,
        auto_load=False,
    )
    assert result.cascade_path_used == "optimal_plan"
    assert result.total_allocated_bytes == 100  # 50 + 30 + 20
    assert result.optimal_plan_consumed is True
    assert result.n_bytes_classified_pair_specific == 3
    # Per-byte allocation should index by pair_index in this proxy
    assert result.per_byte_bit_allocation[0] == 50
    assert result.per_byte_bit_allocation[1] == 30
    assert result.per_byte_bit_allocation[2] == 20


def test_optimal_plan_respects_budget_ceiling() -> None:
    """OptimalPlan that exceeds budget is capped at total_bit_budget."""
    plan_dict = {
        "plan": [
            {"pair_index": 0, "predicted_delta_rate_bytes": 1000},
            {"pair_index": 1, "predicted_delta_rate_bytes": 1000},
        ],
    }
    result = allocate_per_pair_bits(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=500,
        optimal_plan=plan_dict,
        auto_load=False,
    )
    assert result.cascade_path_used == "optimal_plan"
    assert result.total_allocated_bytes == 500  # capped at budget
    assert result.total_allocated_bytes <= result.total_bit_budget


def test_wyner_ziv_composition_classifies_shared_vs_pair_specific() -> None:
    """Per-pair gradient + sensitivity reweight → Wyner-Ziv composition."""
    rng = np.random.default_rng(seed=42)
    n_bytes = 6
    n_pairs = 4
    per_pair = rng.standard_normal((n_bytes, n_pairs, 3)).astype(np.float64)

    # Mark bytes 0,1 as shared-prior (downweighted), 2,3 as pair-specific
    # (upweighted), 4,5 as mixed.
    sensitivity_reweight = {
        0: 0.1,  # shared-prior
        1: 0.1,
        2: 2.0,  # pair-specific
        3: 2.0,
        4: 1.0,  # mixed
        5: 1.0,
    }
    result = allocate_per_pair_bits(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=1000,
        per_pair_gradient=per_pair,
        sensitivity_reweight=sensitivity_reweight,
        auto_load=False,
    )
    assert result.cascade_path_used == "wyner_ziv_composition"
    assert result.per_pair_gradient_consumed is True
    assert result.sensitivity_reweight_consumed is True
    assert result.n_bytes_classified_shared_prior == 2
    assert result.n_bytes_classified_pair_specific == 2
    assert result.n_bytes_classified_mixed == 2

    # Shared-prior bytes (0, 1) get MIN floor
    assert result.per_byte_bit_allocation[0] == PER_PAIR_BUDGET_MIN_FLOOR_BYTES
    assert result.per_byte_bit_allocation[1] == PER_PAIR_BUDGET_MIN_FLOOR_BYTES
    # Pair-specific bytes (2, 3) get >= baseline
    assert result.per_byte_bit_allocation[2] >= PER_PAIR_BUDGET_BASELINE_BYTES
    assert result.per_byte_bit_allocation[3] >= PER_PAIR_BUDGET_BASELINE_BYTES
    # Mixed bytes (4, 5) get baseline
    assert result.per_byte_bit_allocation[4] == PER_PAIR_BUDGET_BASELINE_BYTES


def test_wyner_ziv_path_respects_total_budget() -> None:
    """Wyner-Ziv composition stops at total_bit_budget; never overshoots."""
    rng = np.random.default_rng(seed=42)
    n_bytes = 100
    n_pairs = 4
    per_pair = rng.standard_normal((n_bytes, n_pairs, 3)).astype(np.float64)
    sensitivity_reweight = {i: 2.0 for i in range(n_bytes)}  # all pair-specific
    result = allocate_per_pair_bits(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=50,
        per_pair_gradient=per_pair,
        sensitivity_reweight=sensitivity_reweight,
        auto_load=False,
    )
    assert result.cascade_path_used == "wyner_ziv_composition"
    assert result.total_allocated_bytes <= 50


def test_invalid_archive_sha_rejected() -> None:
    """Sub-12-char or non-hex sha rejected with ValueError."""
    with pytest.raises(ValueError, match="archive_sha256 must be"):
        allocate_per_pair_bits(
            archive_sha256="abc",
            total_bit_budget=100,
        )
    with pytest.raises(ValueError, match="archive_sha256 must be"):
        allocate_per_pair_bits(
            archive_sha256="not-a-hex-sha",
            total_bit_budget=100,
        )


def test_negative_budget_rejected() -> None:
    """Negative budget rejected with ValueError."""
    with pytest.raises(ValueError, match="total_bit_budget must be non-negative"):
        allocate_per_pair_bits(
            archive_sha256="deadbeef1234567890abcdef",
            total_bit_budget=-1,
        )


def test_zero_budget_returns_empty_allocation() -> None:
    """Zero budget → empty allocation map."""
    result = allocate_per_pair_bits(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=0,
        auto_load=False,
    )
    assert result.total_allocated_bytes == 0
    assert len(result.per_byte_bit_allocation) == 0


def test_per_pair_gradient_shape_validation() -> None:
    """Wrong-shape per-pair gradient raises ValueError."""
    bad_gradient = np.zeros((10, 4, 2), dtype=np.float64)  # last dim != 3
    sensitivity_reweight = {0: 1.0}
    with pytest.raises(ValueError, match="shape \\(N_bytes, N_pairs, 3\\)"):
        allocate_per_pair_bits(
            archive_sha256="deadbeef1234567890abcdef",
            total_bit_budget=100,
            per_pair_gradient=bad_gradient,
            sensitivity_reweight=sensitivity_reweight,
            auto_load=False,
        )


def test_optimal_plan_path_skips_zero_delta_assignments() -> None:
    """Per-pair assignments with predicted_delta_rate_bytes <= 0 skipped."""
    plan_dict = {
        "plan": [
            {"pair_index": 0, "predicted_delta_rate_bytes": 0},
            {"pair_index": 1, "predicted_delta_rate_bytes": -5},
            {"pair_index": 2, "predicted_delta_rate_bytes": 30},
        ],
    }
    result = allocate_per_pair_bits(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=200,
        optimal_plan=plan_dict,
        auto_load=False,
    )
    assert result.total_allocated_bytes == 30
    assert 0 not in result.per_byte_bit_allocation
    assert 1 not in result.per_byte_bit_allocation
    assert result.per_byte_bit_allocation[2] == 30


def test_cascade_prefers_optimal_plan_over_wyner_ziv() -> None:
    """OptimalPlan supplied → wins over per-pair-gradient + sensitivity."""
    rng = np.random.default_rng(seed=42)
    per_pair = rng.standard_normal((10, 4, 3)).astype(np.float64)
    sensitivity_reweight = {i: 2.0 for i in range(10)}
    plan_dict = {"plan": [{"pair_index": 0, "predicted_delta_rate_bytes": 25}]}
    result = allocate_per_pair_bits(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=100,
        per_pair_gradient=per_pair,
        sensitivity_reweight=sensitivity_reweight,
        optimal_plan=plan_dict,
        auto_load=False,
    )
    assert result.cascade_path_used == "optimal_plan"
    assert result.sensitivity_reweight_consumed is False
    assert result.per_pair_gradient_consumed is False


def test_outcome_dataclass_invariants_score_claim_false() -> None:
    """Every outcome carries score_claim=False per Apples-to-apples discipline."""
    result = allocate_per_pair_bits(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=100,
        auto_load=False,
    )
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.ready_for_exact_eval_dispatch is False
    assert "[predicted; bit allocator per-pair v1]" in result.evidence_grade


def test_rationale_string_explains_cascade_path() -> None:
    """Rationale string mentions cascade path and evidence tag."""
    result = allocate_per_pair_bits(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=100,
        auto_load=False,
    )
    assert "[predicted; bit allocator per-pair v1]" in result.rationale
    assert "aggregate_fallback" in result.rationale.lower() or "Aggregate" in result.rationale
