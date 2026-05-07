"""Tests for tac.score_geometry_floor_explorer — what-if floor reduction techniques."""
from __future__ import annotations

import pytest

from tac.score_geometry_floor_explorer import (
    baseline_floor_summary,
    explore_architecture_shrink,
    explore_lossy_quantization,
    explore_sparsity,
    explore_water_filling,
)

# Synthetic baseline for hermetic tests
SCHEMA = (("w1", (100, 100)), ("w2", (50, 50)))  # 12,500 elements
N_QUANT = 127
EMPIRICAL_BITS = {"w1": 5.0, "w2": 4.5}  # somewhat skewed
ARCHIVE_OVERHEAD = 1000


def test_baseline_floor_makes_sense() -> None:
    """Baseline empirical floor is below uniform floor."""
    report = baseline_floor_summary(
        schema=SCHEMA,
        baseline_n_quant=N_QUANT,
        baseline_per_tensor_bits=EMPIRICAL_BITS,
        archive_overhead_bytes=ARCHIVE_OVERHEAD,
    )
    assert report.total_bytes_empirical_floor is not None
    assert report.total_bytes_empirical_floor < report.total_bytes_uniform_floor


def test_lossy_quantization_reduces_floor() -> None:
    """Reducing n_quant reduces both uniform AND empirical floors."""
    result = explore_lossy_quantization(
        schema=SCHEMA,
        baseline_n_quant=N_QUANT,
        baseline_per_tensor_bits=EMPIRICAL_BITS,
        target_n_quant=63,
        archive_overhead_bytes=ARCHIVE_OVERHEAD,
    )
    baseline = baseline_floor_summary(
        schema=SCHEMA,
        baseline_n_quant=N_QUANT,
        baseline_per_tensor_bits=EMPIRICAL_BITS,
        archive_overhead_bytes=ARCHIVE_OVERHEAD,
    )
    assert result.bytes_floor < baseline.total_bytes_empirical_floor
    assert result.distortion_risk in {"lossy_low", "lossy_high"}


def test_lossy_quantization_rejects_no_op() -> None:
    """target_n_quant >= baseline raises (no-op)."""
    with pytest.raises(ValueError, match="no-op"):
        explore_lossy_quantization(
            schema=SCHEMA,
            baseline_n_quant=N_QUANT,
            baseline_per_tensor_bits=EMPIRICAL_BITS,
            target_n_quant=N_QUANT,
            archive_overhead_bytes=ARCHIVE_OVERHEAD,
        )


def test_sparsity_reduces_floor() -> None:
    """50% sparsity should ~halve content bits (minus overhead)."""
    result = explore_sparsity(
        schema=SCHEMA,
        baseline_n_quant=N_QUANT,
        baseline_per_tensor_bits=EMPIRICAL_BITS,
        sparsity_fraction=0.5,
        archive_overhead_bytes=ARCHIVE_OVERHEAD,
    )
    baseline = baseline_floor_summary(
        schema=SCHEMA,
        baseline_n_quant=N_QUANT,
        baseline_per_tensor_bits=EMPIRICAL_BITS,
        archive_overhead_bytes=ARCHIVE_OVERHEAD,
    )
    # 50% sparse → ~50% content reduction
    assert result.bytes_floor < baseline.total_bytes_empirical_floor
    # Specifically: should be in [40%, 60%] of baseline content
    baseline_content = baseline.total_bytes_empirical_floor - ARCHIVE_OVERHEAD
    new_content = result.bytes_floor - ARCHIVE_OVERHEAD
    assert 0.4 < new_content / baseline_content < 0.6


def test_sparsity_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="sparsity_fraction"):
        explore_sparsity(
            schema=SCHEMA,
            baseline_n_quant=N_QUANT,
            baseline_per_tensor_bits=EMPIRICAL_BITS,
            sparsity_fraction=1.5,
            archive_overhead_bytes=ARCHIVE_OVERHEAD,
        )


def test_architecture_shrink_reduces_floor_proportionally() -> None:
    """Shrinking elements by 0.5 should ~halve content bits."""
    result = explore_architecture_shrink(
        schema=SCHEMA,
        baseline_n_quant=N_QUANT,
        baseline_per_tensor_bits=EMPIRICAL_BITS,
        element_multiplier=0.5,
        archive_overhead_bytes=ARCHIVE_OVERHEAD,
    )
    baseline = baseline_floor_summary(
        schema=SCHEMA,
        baseline_n_quant=N_QUANT,
        baseline_per_tensor_bits=EMPIRICAL_BITS,
        archive_overhead_bytes=ARCHIVE_OVERHEAD,
    )
    # 0.5× elements → ~0.5× content
    baseline_content = baseline.total_bytes_empirical_floor - ARCHIVE_OVERHEAD
    new_content = result.bytes_floor - ARCHIVE_OVERHEAD
    assert 0.45 < new_content / baseline_content < 0.55


def test_water_filling_respects_baseline_cap() -> None:
    """Water-filling with budget > total entropy capacity caps at baseline (no over-allocation)."""
    sensitivities = {"w1": 1.0, "w2": 0.1}  # w1 much more sensitive
    # Big budget — should saturate to baseline since we cap each tensor
    big_budget = 1_000_000
    result = explore_water_filling(
        schema=SCHEMA,
        baseline_n_quant=N_QUANT,
        baseline_per_tensor_bits=EMPIRICAL_BITS,
        sensitivities=sensitivities,
        total_bit_budget=big_budget,
        archive_overhead_bytes=ARCHIVE_OVERHEAD,
    )
    baseline = baseline_floor_summary(
        schema=SCHEMA,
        baseline_n_quant=N_QUANT,
        baseline_per_tensor_bits=EMPIRICAL_BITS,
        archive_overhead_bytes=ARCHIVE_OVERHEAD,
    )
    # Should not exceed baseline
    assert result.bytes_floor <= baseline.total_bytes_empirical_floor + 10  # ceil noise


def test_water_filling_with_tight_budget_below_baseline() -> None:
    """Tight budget → floor below baseline."""
    sensitivities = {"w1": 1.0, "w2": 1.0}
    # Tight budget: total available bits much less than baseline content
    baseline = baseline_floor_summary(
        schema=SCHEMA,
        baseline_n_quant=N_QUANT,
        baseline_per_tensor_bits=EMPIRICAL_BITS,
        archive_overhead_bytes=ARCHIVE_OVERHEAD,
    )
    baseline_content_bits = (baseline.total_bytes_empirical_floor - ARCHIVE_OVERHEAD) * 8
    tight = baseline_content_bits // 2
    result = explore_water_filling(
        schema=SCHEMA,
        baseline_n_quant=N_QUANT,
        baseline_per_tensor_bits=EMPIRICAL_BITS,
        sensitivities=sensitivities,
        total_bit_budget=tight,
        archive_overhead_bytes=ARCHIVE_OVERHEAD,
    )
    assert result.bytes_floor < baseline.total_bytes_empirical_floor


def test_water_filling_rejects_zero_budget() -> None:
    with pytest.raises(ValueError, match="total_bit_budget"):
        explore_water_filling(
            schema=SCHEMA,
            baseline_n_quant=N_QUANT,
            baseline_per_tensor_bits=EMPIRICAL_BITS,
            sensitivities={"w1": 1.0, "w2": 1.0},
            total_bit_budget=0,
            archive_overhead_bytes=ARCHIVE_OVERHEAD,
        )


def test_water_filling_rejects_zero_sensitivity_total() -> None:
    with pytest.raises(ValueError, match="sensitivities zero"):
        explore_water_filling(
            schema=SCHEMA,
            baseline_n_quant=N_QUANT,
            baseline_per_tensor_bits=EMPIRICAL_BITS,
            sensitivities={"w1": 0.0, "w2": 0.0},
            total_bit_budget=1000,
            archive_overhead_bytes=ARCHIVE_OVERHEAD,
        )
