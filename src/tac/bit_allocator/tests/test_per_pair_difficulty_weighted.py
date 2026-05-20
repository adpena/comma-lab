# SPDX-License-Identifier: MIT
"""Focused tests for :mod:`tac.bit_allocator.per_pair_difficulty_weighted`."""
from __future__ import annotations

import math

import pytest

from tac.bit_allocator import (
    AllocationStrategy,
    BitAllocationResult,
    BitAllocationStrategyError,
    allocate_bits_per_pair,
)
from tac.bit_allocator.per_pair_difficulty_weighted import (
    CANONICAL_MODEL_ID,
)
from tac.provenance import (
    ProvenanceEvidenceGrade,
    ProvenanceKind,
)


# ---- Strategy math correctness -----------------------------------------


def test_uniform_strategy_distributes_equally() -> None:
    result = allocate_bits_per_pair(
        total_bits=900,
        difficulty_per_pair={0: 99.0, 1: 1.0, 2: 50.0},
        strategy=AllocationStrategy.UNIFORM,
    )
    assert result.bits_per_pair == {0: 300, 1: 300, 2: 300}
    assert result.strategy is AllocationStrategy.UNIFORM


def test_linear_strategy_proportional_to_difficulty() -> None:
    result = allocate_bits_per_pair(
        total_bits=1000,
        difficulty_per_pair={0: 1.0, 1: 4.0, 2: 5.0},
        strategy=AllocationStrategy.LINEAR,
    )
    assert result.bits_per_pair == {0: 100, 1: 400, 2: 500}


def test_sqrt_strategy_proportional_to_sqrt_difficulty() -> None:
    # diffs (1, 4, 0.5) -> sqrt = (1.0, 2.0, 0.707) -> total ~= 3.707
    # ratios ~= (0.270, 0.539, 0.191); 1024 * = (276.4, 552.5, 195.1)
    result = allocate_bits_per_pair(
        total_bits=1024,
        difficulty_per_pair={0: 1.0, 1: 4.0, 2: 0.5},
        strategy=AllocationStrategy.SQRT,
    )
    assert sum(result.bits_per_pair.values()) == 1024
    assert result.bits_per_pair[1] > result.bits_per_pair[0] > result.bits_per_pair[2]
    # Sanity: bits_p ≈ total * sqrt(diff_p) / sum(sqrt(diffs))
    total_sqrt = 1.0 + 2.0 + math.sqrt(0.5)
    expected_1 = round(1024 * 2.0 / total_sqrt)
    assert abs(result.bits_per_pair[1] - expected_1) <= 1


# ---- Sum/coverage invariants -------------------------------------------


def test_sum_of_bits_equals_total_bits_always() -> None:
    for strategy in AllocationStrategy:
        for total in (0, 1, 100, 17, 999_999):
            result = allocate_bits_per_pair(
                total_bits=total,
                difficulty_per_pair={i: float(i + 1) for i in range(7)},
                strategy=strategy,
            )
            assert sum(result.bits_per_pair.values()) == total, (
                f"strategy={strategy.value} total={total}"
            )


def test_total_bits_zero_yields_all_zero_allocations() -> None:
    result = allocate_bits_per_pair(
        total_bits=0,
        difficulty_per_pair={0: 1.0, 1: 4.0, 2: 0.5},
        strategy=AllocationStrategy.LINEAR,
    )
    assert all(v == 0 for v in result.bits_per_pair.values())


def test_n_pairs_matches_input_keys() -> None:
    diffs = {i: float(i) for i in range(13)}
    result = allocate_bits_per_pair(total_bits=1000, difficulty_per_pair=diffs)
    assert result.n_pairs == 13
    assert set(result.bits_per_pair.keys()) == set(diffs.keys())


# ---- Determinism -------------------------------------------------------


def test_allocation_is_deterministic_across_calls() -> None:
    diffs = {7: 3.14, 0: 1.41, 5: 2.71, 1: 1.61, 3: 0.5}
    a = allocate_bits_per_pair(
        total_bits=523,
        difficulty_per_pair=diffs,
        strategy=AllocationStrategy.SQRT,
        captured_at_utc="2026-05-19T00:00:00+00:00",
    )
    b = allocate_bits_per_pair(
        total_bits=523,
        difficulty_per_pair=diffs,
        strategy=AllocationStrategy.SQRT,
        captured_at_utc="2026-05-19T00:00:00+00:00",
    )
    assert a.bits_per_pair == b.bits_per_pair
    assert a.provenance.source_sha256 == b.provenance.source_sha256


def test_input_order_does_not_affect_output() -> None:
    diffs_forward = {0: 1.0, 1: 2.0, 2: 3.0}
    diffs_reverse = {2: 3.0, 1: 2.0, 0: 1.0}
    a = allocate_bits_per_pair(
        total_bits=600,
        difficulty_per_pair=diffs_forward,
        strategy=AllocationStrategy.LINEAR,
        captured_at_utc="2026-05-19T00:00:00+00:00",
    )
    b = allocate_bits_per_pair(
        total_bits=600,
        difficulty_per_pair=diffs_reverse,
        strategy=AllocationStrategy.LINEAR,
        captured_at_utc="2026-05-19T00:00:00+00:00",
    )
    assert a.bits_per_pair == b.bits_per_pair
    assert a.provenance.source_sha256 == b.provenance.source_sha256


# ---- Strategy enum + string coercion -----------------------------------


def test_string_strategy_coerces_to_enum() -> None:
    result = allocate_bits_per_pair(
        total_bits=100,
        difficulty_per_pair={0: 1.0, 1: 1.0},
        strategy="uniform",
    )
    assert result.strategy is AllocationStrategy.UNIFORM


def test_unknown_strategy_string_raises() -> None:
    with pytest.raises(BitAllocationStrategyError, match="unknown strategy"):
        allocate_bits_per_pair(
            total_bits=100,
            difficulty_per_pair={0: 1.0},
            strategy="cubic_root_of_log",
        )


def test_non_enum_non_str_strategy_raises() -> None:
    with pytest.raises(BitAllocationStrategyError, match="strategy must be"):
        allocate_bits_per_pair(
            total_bits=100,
            difficulty_per_pair={0: 1.0},
            strategy=42,  # type: ignore[arg-type]
        )


# ---- Input validation --------------------------------------------------


def test_empty_difficulty_mapping_raises() -> None:
    with pytest.raises(BitAllocationStrategyError, match="at least one pair"):
        allocate_bits_per_pair(total_bits=100, difficulty_per_pair={})


def test_negative_total_bits_raises() -> None:
    with pytest.raises(BitAllocationStrategyError, match="non-negative"):
        allocate_bits_per_pair(
            total_bits=-5, difficulty_per_pair={0: 1.0}
        )


def test_negative_difficulty_raises() -> None:
    with pytest.raises(BitAllocationStrategyError, match="non-negative"):
        allocate_bits_per_pair(
            total_bits=10, difficulty_per_pair={0: -1.0}
        )


def test_non_finite_difficulty_raises() -> None:
    with pytest.raises(BitAllocationStrategyError, match="finite"):
        allocate_bits_per_pair(
            total_bits=10, difficulty_per_pair={0: math.inf}
        )


def test_non_int_pair_key_raises() -> None:
    with pytest.raises(BitAllocationStrategyError, match="pair_index must be int"):
        allocate_bits_per_pair(
            total_bits=10,
            difficulty_per_pair={"zero": 1.0},  # type: ignore[dict-item]
        )


def test_bool_pair_key_rejected() -> None:
    # In Python, bool is a subclass of int; explicitly reject for clarity.
    with pytest.raises(BitAllocationStrategyError, match="pair_index must be int"):
        allocate_bits_per_pair(
            total_bits=10,
            difficulty_per_pair={True: 1.0},  # type: ignore[dict-item]
        )


def test_bool_difficulty_rejected() -> None:
    with pytest.raises(BitAllocationStrategyError, match="numeric, got bool"):
        allocate_bits_per_pair(
            total_bits=10,
            difficulty_per_pair={0: True},  # type: ignore[dict-item]
        )


def test_non_int_total_bits_raises() -> None:
    with pytest.raises(BitAllocationStrategyError, match="total_bits must be int"):
        allocate_bits_per_pair(
            total_bits=1024.0,  # type: ignore[arg-type]
            difficulty_per_pair={0: 1.0},
        )


# ---- Provenance + Catalog #323 invariants ------------------------------


def test_provenance_is_predicted_grade() -> None:
    result = allocate_bits_per_pair(
        total_bits=100,
        difficulty_per_pair={0: 1.0, 1: 2.0},
    )
    assert result.provenance.evidence_grade == ProvenanceEvidenceGrade.PREDICTED
    assert result.provenance.artifact_kind == ProvenanceKind.PREDICTED_FROM_MODEL
    assert result.provenance.promotion_eligible is False
    assert result.provenance.score_claim_valid is False


def test_provenance_model_id_canonical() -> None:
    result = allocate_bits_per_pair(
        total_bits=100,
        difficulty_per_pair={0: 1.0, 1: 2.0},
    )
    # The builder wraps model_id as <predictor:...> for source_path.
    assert CANONICAL_MODEL_ID in result.provenance.source_path
    assert "predictor" in result.provenance.source_path


def test_archive_sha256_threaded_into_inputs_hash() -> None:
    a = allocate_bits_per_pair(
        total_bits=100,
        difficulty_per_pair={0: 1.0, 1: 2.0},
        archive_sha256="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        captured_at_utc="2026-05-19T00:00:00+00:00",
    )
    b = allocate_bits_per_pair(
        total_bits=100,
        difficulty_per_pair={0: 1.0, 1: 2.0},
        archive_sha256=None,
        captured_at_utc="2026-05-19T00:00:00+00:00",
    )
    # Same bits (deterministic), different provenance sha (archive sha threaded).
    assert a.bits_per_pair == b.bits_per_pair
    assert a.provenance.source_sha256 != b.provenance.source_sha256
    assert a.notes["archive_sha256_prefix"] == "6bae0201fb08"


# ---- BitAllocationResult invariants (Catalog #323) ---------------------


def test_result_must_carry_predicted_axis_tag() -> None:
    result = allocate_bits_per_pair(
        total_bits=100, difficulty_per_pair={0: 1.0, 1: 1.0}
    )
    assert result.axis_tag == "[predicted]"
    assert result.score_claim is False
    assert result.promotion_eligible is False


def test_result_invariant_refuses_score_claim_true() -> None:
    result = allocate_bits_per_pair(
        total_bits=100, difficulty_per_pair={0: 1.0, 1: 1.0}
    )
    with pytest.raises(BitAllocationStrategyError, match="score_claim must be False"):
        BitAllocationResult(
            bits_per_pair=dict(result.bits_per_pair),
            strategy=result.strategy,
            total_bits=result.total_bits,
            n_pairs=result.n_pairs,
            provenance=result.provenance,
            score_claim=True,  # forbidden
        )


def test_result_invariant_refuses_promotion_eligible_true() -> None:
    result = allocate_bits_per_pair(
        total_bits=100, difficulty_per_pair={0: 1.0, 1: 1.0}
    )
    with pytest.raises(
        BitAllocationStrategyError, match="promotion_eligible must be False"
    ):
        BitAllocationResult(
            bits_per_pair=dict(result.bits_per_pair),
            strategy=result.strategy,
            total_bits=result.total_bits,
            n_pairs=result.n_pairs,
            provenance=result.provenance,
            promotion_eligible=True,  # forbidden
        )


def test_result_invariant_refuses_wrong_axis_tag() -> None:
    result = allocate_bits_per_pair(
        total_bits=100, difficulty_per_pair={0: 1.0, 1: 1.0}
    )
    with pytest.raises(
        BitAllocationStrategyError, match="axis_tag must be"
    ):
        BitAllocationResult(
            bits_per_pair=dict(result.bits_per_pair),
            strategy=result.strategy,
            total_bits=result.total_bits,
            n_pairs=result.n_pairs,
            provenance=result.provenance,
            axis_tag="[contest-CUDA]",  # forbidden — promotes to score claim
        )


def test_as_dict_round_trips_serializable() -> None:
    import json

    result = allocate_bits_per_pair(
        total_bits=128,
        difficulty_per_pair={0: 1.0, 1: 2.0, 2: 4.0},
        strategy=AllocationStrategy.SQRT,
    )
    payload = result.as_dict()
    # Must be JSON-serializable (canonical manifest emission requirement).
    blob = json.dumps(payload, sort_keys=True)
    parsed = json.loads(blob)
    assert parsed["score_claim"] is False
    assert parsed["axis_tag"] == "[predicted]"
    assert parsed["strategy"] == "sqrt"
    assert set(parsed["bits_per_pair"].keys()) == {"0", "1", "2"}
