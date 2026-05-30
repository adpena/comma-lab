# SPDX-License-Identifier: MIT
"""Tests for Z8 Phase 2 M7 — canonical scorer-sensitivity-map helper.

Verifies the three honest production paths per the module docstring:

1. **Path A** (uniform_sensitivity_map_for_level) returns correctly-shaped
   all-ones tensor; satisfies the M8 Protocol invariant from
   ``binding_contract.py:467-470`` that uniform sensitivity reduces to
   standard L2 reconstruction loss.

2. **Path B** (empirical_sensitivity_map_from_slot_ggg) raises
   ``EmpiricalSensitivityMapNotYetLandedError`` with actionable
   reactivation criteria; honest deferral per CLAUDE.md "Forbidden
   premature KILL without research exhaustion".

3. **Path C** (yousfi_uniward_finite_difference_sensitivity_map) raises
   ``EmpiricalSensitivityMapNotYetLandedError`` with actionable
   reactivation criteria; honest deferral.

Per Catalog #287 evidence-tag discipline: no docstring overstatement;
every test claim is paired with adjacent source/observed evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.z8_hierarchical_predictive_coding import (
    BuildMilestoneStatus,
    EmpiricalSensitivityMapNotYetLandedError,
    LevelDimensionContract,
    ScorerSensitivityMapSource,
    Z8_PHASE_2_BUILD_MILESTONES,
    Z8ScorerSensitivityMap,
    build_z8_scorer_sensitivity_map_for_level,
    empirical_sensitivity_map_from_slot_ggg,
    uniform_sensitivity_map_for_level,
    yousfi_uniward_finite_difference_sensitivity_map,
)


# Synthetic level fixture mirrors the M4/M5 test pattern: H=192, W=256
# at level 0 (Z8 canonical full-resolution per-level wavelet subband
# shape per Z8HierarchicalConfig default).
@pytest.fixture
def synthetic_level() -> LevelDimensionContract:
    return LevelDimensionContract(
        level_index=0,
        num_categorical_groups=8,
        num_categorical_classes=16,
        deterministic_state_dim=64,
        wavelet_subband_shape=(192, 256),
        ego_motion_dim=8,
        bit_budget_estimate=128,
    )


# Sister coarser-level fixture for the per-level shape contract test.
@pytest.fixture
def synthetic_level_coarse() -> LevelDimensionContract:
    return LevelDimensionContract(
        level_index=2,
        num_categorical_groups=8,
        num_categorical_classes=16,
        deterministic_state_dim=64,
        wavelet_subband_shape=(48, 64),
        ego_motion_dim=8,
        bit_budget_estimate=32,
    )


# -----------------------------------------------------------------
# Section A: Path A (uniform_sensitivity_map_for_level) shape + M8 invariant
# -----------------------------------------------------------------


def test_uniform_returns_correct_shape(synthetic_level: LevelDimensionContract) -> None:
    """Default (B=1, C=3) returns (1, 3, 192, 256)."""
    arr = uniform_sensitivity_map_for_level(synthetic_level)
    assert arr.shape == (1, 3, 192, 256)


def test_uniform_returns_all_ones(synthetic_level: LevelDimensionContract) -> None:
    """The M8 Protocol invariant: sensitivity_map == 1 everywhere."""
    arr = uniform_sensitivity_map_for_level(synthetic_level)
    assert np.all(arr == 1.0)


def test_uniform_reduces_to_l2_invariant(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Mathematical witness: uniform * (reconstruction - target)^2 == L2 loss.

    This proves the M8 Protocol invariant from binding_contract.py:467-470:
    "integral over uniform sensitivity reduces to standard L2."
    """
    arr = uniform_sensitivity_map_for_level(synthetic_level)
    rng = np.random.RandomState(0)
    target = rng.randn(1, 3, 192, 256).astype(np.float32)
    reconstruction = rng.randn(1, 3, 192, 256).astype(np.float32)
    # M8 per_level_loss canonical form:
    weighted_l2 = (arr * (reconstruction - target) ** 2).sum()
    standard_l2 = ((reconstruction - target) ** 2).sum()
    # The invariant: uniform sensitivity reduces to standard L2 exactly.
    assert np.isclose(weighted_l2, standard_l2)


def test_uniform_honors_custom_batch_and_channels(
    synthetic_level: LevelDimensionContract,
) -> None:
    arr = uniform_sensitivity_map_for_level(
        synthetic_level, batch_size=4, num_channels=1, dtype=np.float64
    )
    assert arr.shape == (4, 1, 192, 256)
    assert arr.dtype == np.float64


def test_uniform_per_level_shape_halves_at_coarser_level(
    synthetic_level_coarse: LevelDimensionContract,
) -> None:
    """Coarser level has smaller (H, W) per Mallat pyramid contract."""
    arr = uniform_sensitivity_map_for_level(synthetic_level_coarse)
    assert arr.shape == (1, 3, 48, 64)


def test_uniform_rejects_non_level_contract() -> None:
    with pytest.raises(TypeError, match="LevelDimensionContract"):
        uniform_sensitivity_map_for_level("not_a_level")  # type: ignore[arg-type]


def test_uniform_rejects_non_positive_batch(
    synthetic_level: LevelDimensionContract,
) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        uniform_sensitivity_map_for_level(synthetic_level, batch_size=0)


def test_uniform_rejects_non_positive_channels(
    synthetic_level: LevelDimensionContract,
) -> None:
    with pytest.raises(ValueError, match="num_channels must be positive"):
        uniform_sensitivity_map_for_level(synthetic_level, num_channels=-1)


# -----------------------------------------------------------------
# Section B: Path B / Path C honest stubs raise with reactivation criteria
# -----------------------------------------------------------------


def test_path_b_raises_not_yet_landed(synthetic_level: LevelDimensionContract) -> None:
    """Slot GGG stub raises with actionable reactivation message."""
    with pytest.raises(EmpiricalSensitivityMapNotYetLandedError) as exc_info:
        empirical_sensitivity_map_from_slot_ggg(synthetic_level)
    msg = str(exc_info.value)
    assert "Path B" in msg
    assert "DEFERRED-pending-research" in msg
    assert "Slot GGG" in msg
    assert "reactivation criteria" in msg
    assert "Path A" in msg  # Operator-actionable fallback hint


def test_path_c_raises_not_yet_landed(synthetic_level: LevelDimensionContract) -> None:
    """Yousfi UNIWARD-analog stub raises with actionable reactivation message."""
    with pytest.raises(EmpiricalSensitivityMapNotYetLandedError) as exc_info:
        yousfi_uniward_finite_difference_sensitivity_map(synthetic_level)
    msg = str(exc_info.value)
    assert "Path C" in msg
    assert "DEFERRED-pending-paid-GPU" in msg
    assert "patch_upstream_yuv6_globally" in msg
    assert "eval_roundtrip=True" in msg
    assert "Path A" in msg  # Operator-actionable fallback hint


def test_not_yet_landed_is_notimplementederror_subclass() -> None:
    """Per Catalog #307 + module docstring: stub deferral is honest
    NotImplementedError, not opaque RuntimeError."""
    assert issubclass(
        EmpiricalSensitivityMapNotYetLandedError, NotImplementedError
    )


# -----------------------------------------------------------------
# Section C: Z8ScorerSensitivityMap dispatcher
# -----------------------------------------------------------------


def test_dispatcher_default_source_is_uniform() -> None:
    sm = Z8ScorerSensitivityMap()
    assert sm.source is ScorerSensitivityMapSource.UNIFORM


def test_dispatcher_uniform_returns_correct_tensor(
    synthetic_level: LevelDimensionContract,
) -> None:
    sm = Z8ScorerSensitivityMap(ScorerSensitivityMapSource.UNIFORM)
    arr = sm.get_for_level(synthetic_level)
    assert arr.shape == (1, 3, 192, 256)
    assert np.all(arr == 1.0)


def test_dispatcher_slot_ggg_raises(
    synthetic_level: LevelDimensionContract,
) -> None:
    sm = Z8ScorerSensitivityMap(ScorerSensitivityMapSource.EMPIRICAL_SLOT_GGG)
    with pytest.raises(EmpiricalSensitivityMapNotYetLandedError):
        sm.get_for_level(synthetic_level)


def test_dispatcher_finite_difference_raises(
    synthetic_level: LevelDimensionContract,
) -> None:
    sm = Z8ScorerSensitivityMap(ScorerSensitivityMapSource.FINITE_DIFFERENCE_UNIWARD)
    with pytest.raises(EmpiricalSensitivityMapNotYetLandedError):
        sm.get_for_level(synthetic_level)


def test_dispatcher_rejects_non_enum_source() -> None:
    with pytest.raises(TypeError, match="ScorerSensitivityMapSource"):
        Z8ScorerSensitivityMap("uniform")  # type: ignore[arg-type]


def test_convenience_builder_default_uniform(
    synthetic_level: LevelDimensionContract,
) -> None:
    """build_z8_scorer_sensitivity_map_for_level defaults to UNIFORM."""
    arr = build_z8_scorer_sensitivity_map_for_level(synthetic_level)
    assert arr.shape == (1, 3, 192, 256)
    assert np.all(arr == 1.0)


def test_convenience_builder_honors_source(
    synthetic_level: LevelDimensionContract,
) -> None:
    with pytest.raises(EmpiricalSensitivityMapNotYetLandedError):
        build_z8_scorer_sensitivity_map_for_level(
            synthetic_level,
            source=ScorerSensitivityMapSource.EMPIRICAL_SLOT_GGG,
        )


# -----------------------------------------------------------------
# Section D: M7 milestone consistency
# -----------------------------------------------------------------


def test_m7_milestone_is_landed_with_honest_path_documentation() -> None:
    by_id = {m.milestone_id: m for m in Z8_PHASE_2_BUILD_MILESTONES}
    assert "empirical_scorer_sensitivity_map_v1_landed" in by_id
    m7 = by_id["empirical_scorer_sensitivity_map_v1_landed"]
    assert m7.status == BuildMilestoneStatus.LANDED
    # The description must reflect the honest 3-path landing per the module
    # docstring (not the original "first v1 empirical" framing that was
    # premature-verified before the master_gradient ledger domain-mismatch
    # was empirically discovered).
    assert "Path A" in m7.description
    assert "Path B" in m7.description
    assert "Path C" in m7.description
    assert "LANDED" in m7.description
    assert "M8" in m7.description
