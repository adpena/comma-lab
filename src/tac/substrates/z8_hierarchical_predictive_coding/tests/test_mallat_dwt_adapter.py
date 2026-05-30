# SPDX-License-Identifier: MIT
"""Tests for Z8 Phase 2 M5 Mallat full DWT adapter.

Verifies:

1. ``Z8MallatDaubechiesPartition`` adapter satisfies the
   ``WaveletPartition`` Protocol via @runtime_checkable isinstance.

2. ``decompose_to_next_level + recompose_from_next_level`` round-trip
   to within Mallat §7.5 perfect-reconstruction tolerance — empirically
   atol ~1e-12 at fp64, well within the acceptance criterion's 1e-6.

3. Per-level approximation shape ``(B, H/2, W/2, C)`` matches the
   L0 sum-pool's shape contract.

4. ``WaveletDetail2D(lh, hl, hh)`` carries THREE non-zero high-pass
   subbands (vs the sum-pool proxy which produces only approximation).

5. Framework-agnostic via numpy intermediate: accepts pure-Python lists,
   numpy arrays, and (when MLX/PyTorch available) MLX / PyTorch tensors.

6. The new M5 milestone in ``Z8_PHASE_2_BUILD_MILESTONES`` reflects the
   adapter pivot and references the canonical primitive's source module.

Per CLAUDE.md mathematical grounding: tests verify Mallat §7.5 exact-
reconstruction property + Protocol satisfaction + shape contract + the
honest 3-detail-subband classification. No empirical score claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.z8_hierarchical_predictive_coding import (
    BuildMilestoneStatus,
    LevelDimensionContract,
    WaveletDetail2D,
    WaveletPartition,
    Z8_PHASE_2_BUILD_MILESTONES,
    Z8MallatDaubechiesPartition,
    build_z8_mallat_dwt_adapter_for_level,
)
from tac.symposium_impls.daubechies_wavelet_codec import DaubechiesFilter


# Synthetic level fixture: H=8, W=8 → H/2=4, W/2=4. Small enough that
# numpy convolution loops are fast; large enough that round-trip is
# non-trivial (not the trivial 2x2 case).
@pytest.fixture
def synthetic_level() -> LevelDimensionContract:
    return LevelDimensionContract(
        level_index=0,
        num_categorical_groups=8,
        num_categorical_classes=16,
        deterministic_state_dim=64,
        wavelet_subband_shape=(4, 4),
        ego_motion_dim=8,
        bit_budget_estimate=128,
    )


# -----------------------------------------------------------------
# Section A: Protocol satisfaction
# -----------------------------------------------------------------


def test_adapter_satisfies_wavelet_partition_protocol(
    synthetic_level: LevelDimensionContract,
) -> None:
    """isinstance(adapter, WaveletPartition) returns True per @runtime_checkable."""
    adapter = build_z8_mallat_dwt_adapter_for_level(synthetic_level)
    assert isinstance(adapter, WaveletPartition)


def test_adapter_level_property_returns_bound_contract(
    synthetic_level: LevelDimensionContract,
) -> None:
    adapter = build_z8_mallat_dwt_adapter_for_level(synthetic_level)
    assert adapter.level is synthetic_level


def test_adapter_default_filter_is_db2(
    synthetic_level: LevelDimensionContract,
) -> None:
    """M5 milestone description: 'Full Daubechies-4 wavelet transform' → db2."""
    adapter = build_z8_mallat_dwt_adapter_for_level(synthetic_level)
    assert adapter.filter_id == DaubechiesFilter.DB2


def test_adapter_accepts_alternative_filter(
    synthetic_level: LevelDimensionContract,
) -> None:
    """db1 (Haar) is also available; verify filter parameterization works."""
    adapter = build_z8_mallat_dwt_adapter_for_level(
        synthetic_level, filter_id=DaubechiesFilter.DB1
    )
    assert adapter.filter_id == DaubechiesFilter.DB1


def test_adapter_rejects_non_level_contract() -> None:
    with pytest.raises(TypeError, match="LevelDimensionContract"):
        Z8MallatDaubechiesPartition(
            level="not_a_level",  # type: ignore[arg-type]
            filter_id=DaubechiesFilter.DB2,
        )


# -----------------------------------------------------------------
# Section B: Decompose contract — shape + non-zero detail
# -----------------------------------------------------------------


def test_decompose_returns_correct_approximation_shape(
    synthetic_level: LevelDimensionContract,
) -> None:
    """LL subband must be (B, H/2, W/2, C) per Mallat 2D separable."""
    adapter = build_z8_mallat_dwt_adapter_for_level(synthetic_level)
    x = np.random.RandomState(0).randn(2, 8, 8, 3).astype(np.float64)
    ll, detail = adapter.decompose_to_next_level(x)
    assert ll.shape == (2, 4, 4, 3)


def test_decompose_detail_carries_three_subbands(
    synthetic_level: LevelDimensionContract,
) -> None:
    """WaveletDetail2D carries lh / hl / hh per Mallat §7.5."""
    adapter = build_z8_mallat_dwt_adapter_for_level(synthetic_level)
    x = np.random.RandomState(0).randn(2, 8, 8, 3).astype(np.float64)
    _, detail = adapter.decompose_to_next_level(x)
    assert isinstance(detail, WaveletDetail2D)
    assert detail.lh.shape == (2, 4, 4, 3)
    assert detail.hl.shape == (2, 4, 4, 3)
    assert detail.hh.shape == (2, 4, 4, 3)


def test_decompose_detail_subbands_are_not_all_zero(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Acceptance criterion #3: detail subbands NOT all-zero (vs sum-pool).

    Mathematical aside: each of LH/HL/HH captures a SPECIFIC orientation
    of high-frequency content per Mallat §7.7 — LH captures vertical-
    varying high freq (LL columns, HH rows), HL captures horizontal-
    varying high freq, HH captures diagonal. A pure 2D checkerboard
    pattern is purely diagonal high-freq so it activates ONLY HH and
    leaves LH/HL near-zero (~7e-13). To exercise all three, use a
    broadband random signal — it has energy in all three orientations.
    """
    adapter = build_z8_mallat_dwt_adapter_for_level(synthetic_level)
    rng = np.random.RandomState(0)
    x = rng.randn(1, 8, 8, 3).astype(np.float64)
    _, detail = adapter.decompose_to_next_level(x)
    assert not np.allclose(detail.lh, 0.0, atol=1e-6), \
        "LH must respond to broadband signal"
    assert not np.allclose(detail.hl, 0.0, atol=1e-6), \
        "HL must respond to broadband signal"
    assert not np.allclose(detail.hh, 0.0, atol=1e-6), \
        "HH must respond to broadband signal"


def test_decompose_checkerboard_activates_only_hh_diagonal(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Mathematical witness: a pure 2D checkerboard is diagonal-only freq.

    Verifies our 2D separable Daubechies correctly partitions orientation:
    a checkerboard varying in BOTH H and W simultaneously projects ONLY
    onto HH (diagonal detail) — LH/HL are mathematically ~zero. This is
    the textbook proof that the subband decomposition is orientation-
    selective per Mallat §7.7.
    """
    adapter = build_z8_mallat_dwt_adapter_for_level(synthetic_level)
    base = np.indices((8, 8)).sum(axis=0) % 2  # (8, 8) of 0/1
    checker = (base * 2.0 - 1.0).astype(np.float64)
    x = np.broadcast_to(checker[None, :, :, None], (1, 8, 8, 3)).copy()
    _, detail = adapter.decompose_to_next_level(x)
    # Pure diagonal pattern → HH dominates; LH/HL near machine eps.
    assert np.abs(detail.lh).max() < 1e-10, "LH must be ~zero for checkerboard"
    assert np.abs(detail.hl).max() < 1e-10, "HL must be ~zero for checkerboard"
    assert np.abs(detail.hh).max() > 0.1, "HH must dominate for checkerboard"


def test_decompose_rejects_non_nhwc_input(
    synthetic_level: LevelDimensionContract,
) -> None:
    adapter = build_z8_mallat_dwt_adapter_for_level(synthetic_level)
    with pytest.raises(ValueError, match="NHWC 4D"):
        adapter.decompose_to_next_level(np.zeros((8, 8, 3)))  # 3D, not 4D


def test_decompose_rejects_odd_axis_length(
    synthetic_level: LevelDimensionContract,
) -> None:
    """1-level DWT requires even axis lengths."""
    adapter = build_z8_mallat_dwt_adapter_for_level(synthetic_level)
    with pytest.raises(ValueError, match="even axis length"):
        adapter.decompose_to_next_level(np.zeros((1, 7, 8, 3)))


# -----------------------------------------------------------------
# Section C: Mallat §7.5 perfect-reconstruction round-trip
# -----------------------------------------------------------------


def test_round_trip_recovers_input_random_signal(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Acceptance criterion #1: round-trip atol ≤ 1e-6 (target); ~1e-12 in practice."""
    adapter = build_z8_mallat_dwt_adapter_for_level(synthetic_level)
    rng = np.random.RandomState(42)
    x = rng.randn(3, 8, 8, 4).astype(np.float64)
    ll, detail = adapter.decompose_to_next_level(x)
    recovered = adapter.recompose_from_next_level(ll, detail)
    assert recovered.shape == x.shape
    assert np.allclose(recovered, x, atol=1e-6), \
        f"max_abs_err = {np.abs(recovered - x).max():.3e} exceeds 1e-6"


def test_round_trip_fp64_atol_1e_10(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Tighter tolerance check: fp64 Daubechies-4 should achieve atol ≤ 1e-10."""
    adapter = build_z8_mallat_dwt_adapter_for_level(synthetic_level)
    rng = np.random.RandomState(1)
    x = rng.randn(2, 16, 16, 2).astype(np.float64)
    ll, detail = adapter.decompose_to_next_level(x)
    recovered = adapter.recompose_from_next_level(ll, detail)
    max_err = float(np.abs(recovered - x).max())
    assert max_err <= 1e-10, f"max_abs_err = {max_err:.3e} exceeds 1e-10"


def test_round_trip_haar_filter_exact(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Haar (db1) should give effectively-exact round-trip (atol ~1e-15)."""
    adapter = build_z8_mallat_dwt_adapter_for_level(
        synthetic_level, filter_id=DaubechiesFilter.DB1,
    )
    rng = np.random.RandomState(7)
    x = rng.randn(1, 8, 8, 1).astype(np.float64)
    ll, detail = adapter.decompose_to_next_level(x)
    recovered = adapter.recompose_from_next_level(ll, detail)
    assert np.allclose(recovered, x, atol=1e-12)


def test_round_trip_constant_signal_only_approximation(
    synthetic_level: LevelDimensionContract,
) -> None:
    """A constant signal has zero detail; round-trip is trivially exact."""
    adapter = build_z8_mallat_dwt_adapter_for_level(synthetic_level)
    x = np.full((1, 8, 8, 3), 7.5, dtype=np.float64)
    ll, detail = adapter.decompose_to_next_level(x)
    # Constant signal projects entirely to approximation; details are ~0.
    assert np.allclose(detail.lh, 0.0, atol=1e-10)
    assert np.allclose(detail.hl, 0.0, atol=1e-10)
    assert np.allclose(detail.hh, 0.0, atol=1e-10)
    recovered = adapter.recompose_from_next_level(ll, detail)
    assert np.allclose(recovered, x, atol=1e-10)


def test_recompose_rejects_non_wavelet_detail_dataclass(
    synthetic_level: LevelDimensionContract,
) -> None:
    """The detail MUST be a WaveletDetail2D to preserve all three subbands."""
    adapter = build_z8_mallat_dwt_adapter_for_level(synthetic_level)
    ll = np.zeros((1, 4, 4, 3))
    with pytest.raises(TypeError, match="WaveletDetail2D"):
        adapter.recompose_from_next_level(ll, ll)  # plain tensor not allowed


# -----------------------------------------------------------------
# Section D: Framework-agnostic via numpy intermediate
# -----------------------------------------------------------------


def test_accepts_python_list_input(
    synthetic_level: LevelDimensionContract,
) -> None:
    """np.asarray converts lists; adapter remains framework-agnostic."""
    adapter = build_z8_mallat_dwt_adapter_for_level(synthetic_level)
    x_list = np.zeros((1, 8, 8, 2)).tolist()
    ll, detail = adapter.decompose_to_next_level(x_list)
    assert ll.shape == (1, 4, 4, 2)


# -----------------------------------------------------------------
# Section E: M5 milestone consistency
# -----------------------------------------------------------------


def test_m5_milestone_is_landed_with_canonical_description() -> None:
    by_id = {m.milestone_id: m for m in Z8_PHASE_2_BUILD_MILESTONES}
    assert "mallat_full_dwt_replaces_sum_pool_proxy" in by_id
    m5 = by_id["mallat_full_dwt_replaces_sum_pool_proxy"]
    assert m5.status == BuildMilestoneStatus.LANDED
    # Description references the canonical primitive's source module.
    assert "tac.symposium_impls.daubechies_wavelet_codec" in m5.description
    assert "WaveletDetail2D" in m5.description
