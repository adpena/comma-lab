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
    ResolutionMismatchAwaitingPerLevelProjectionError,
    ScorerSensitivityMapSource,
    Z8_PHASE_2_BUILD_MILESTONES,
    Z8ScorerSensitivityMap,
    build_z8_scorer_sensitivity_map_for_level,
    empirical_sensitivity_map_from_master_gradient,
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
    # The description must reflect the honest 4-path landing per the module
    # docstring (Phase A baseline + Path B2 master-gradient operational +
    # Path B + Path C honest stubs).
    assert "Path A" in m7.description
    assert "Path B" in m7.description
    assert "Path C" in m7.description
    assert "Path B2" in m7.description  # Phase D wire-in (2026-05-30)
    assert "LANDED" in m7.description
    assert "M8" in m7.description


# -----------------------------------------------------------------
# Section E: Phase D Path B2 (empirical_sensitivity_map_from_master_gradient)
# Yousfi-ordered wire-in 2026-05-30 consuming Phase A canonical extract_M_pixel
# + broadcast_sensitivity_map_to_channels per the M8 Protocol docstring
# verbatim: "The map itself comes from tac.master_gradient empirical
# per-pixel sensitivity measurement."
# -----------------------------------------------------------------


def _make_matching_resolution_contest_tensor(
    level: LevelDimensionContract,
    *,
    n_pairs: int = 4,
    n_axes: int = 3,
    seed: int = 0,
):
    """Build a synthetic ContestGradientTensor whose (H, W) matches the level.

    Phase A canonical helper signature requires a polymorphic
    ContestGradientTensor or InflatedGradientTensor. Phase D's identity-
    resolution requirement means the gradient's native (H, W) must equal
    level.wavelet_subband_shape; this fixture builds exactly that.
    """
    import hashlib
    from pathlib import Path

    from tac.master_gradient_comparison.multi_granularity import (
        ContestGradientTensor,
        OperatingPoint,
    )

    H, W = level.wavelet_subband_shape
    rng = np.random.RandomState(seed)
    arr = rng.randn(n_pairs, n_axes, H, W).astype(np.float32)
    sha = hashlib.sha256(arr.tobytes()).hexdigest()
    tmp = (
        Path(".omx/state/master_gradient_comparison")
        / f"test_phase_d_contest_{sha[:8]}.npy"
    )
    tmp.parent.mkdir(parents=True, exist_ok=True)
    np.save(tmp, arr)
    return ContestGradientTensor(
        array_path=str(tmp),
        array_sha256=sha,
        n_pairs=n_pairs,
        height=H,
        width=W,
        contest_video_sha256="0" * 64,
        scorer_seg_sha256="1" * 64,
        scorer_pose_sha256="2" * 64,
        operating_point=OperatingPoint(
            d_seg=0.067, d_pose=3.4e-5, rate=0.119, score=0.192
        ),
        captured_at_utc="2026-05-30T00:00:00Z",
        measurement_axis="[predicted]",
    )


def test_path_b2_returns_correct_shape(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Path B2 produces (B, C, H, W) matching M8 Protocol contract."""
    gt = _make_matching_resolution_contest_tensor(synthetic_level)
    arr = empirical_sensitivity_map_from_master_gradient(
        synthetic_level, gt, batch_size=1, num_channels=3
    )
    assert arr.shape == (1, 3, 192, 256)


def test_path_b2_is_non_negative(
    synthetic_level: LevelDimensionContract,
) -> None:
    """L2 / L1 / max reductions of master-gradient are all >= 0."""
    gt = _make_matching_resolution_contest_tensor(synthetic_level, seed=1)
    for reduction in ("l2_norm", "l1_norm", "max"):
        arr = empirical_sensitivity_map_from_master_gradient(
            synthetic_level, gt, reduction=reduction
        )
        assert np.all(arr >= 0), f"reduction={reduction} produced negative values"


def test_path_b2_channel_uniformity_invariant(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Fridrich per-spatial-location cost: all channels at (h,w) same weight."""
    gt = _make_matching_resolution_contest_tensor(synthetic_level, seed=2)
    arr = empirical_sensitivity_map_from_master_gradient(
        synthetic_level, gt, batch_size=1, num_channels=3
    )
    # All channels at each (h, w) must be identical per Fridrich convention
    np.testing.assert_array_equal(arr[0, 0], arr[0, 1])
    np.testing.assert_array_equal(arr[0, 1], arr[0, 2])


def test_path_b2_raises_on_resolution_mismatch(
    synthetic_level_coarse: LevelDimensionContract,
) -> None:
    """Phase D identity-resolution invariant: mismatch raises with Phase C hint."""
    # Build tensor at native (192, 256) but pass to coarse level (48, 64)
    fine_level = LevelDimensionContract(
        level_index=0,
        num_categorical_groups=8,
        num_categorical_classes=16,
        deterministic_state_dim=64,
        wavelet_subband_shape=(192, 256),
        ego_motion_dim=8,
        bit_budget_estimate=128,
    )
    gt = _make_matching_resolution_contest_tensor(fine_level, seed=3)
    with pytest.raises(
        ResolutionMismatchAwaitingPerLevelProjectionError
    ) as exc_info:
        empirical_sensitivity_map_from_master_gradient(
            synthetic_level_coarse, gt
        )
    msg = str(exc_info.value)
    # Operator-actionable error must name both shapes + Phase C unblocker
    assert "192" in msg and "256" in msg
    assert "48" in msg and "64" in msg
    assert "Phase C" in msg
    assert "decompose_M_contest_per_level" in msg


def test_path_b2_rejects_invalid_reduction(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Forwards extract_M_pixel's reduction-validation; helper-specific error
    propagates (tac.master_gradient_comparison raises
    MultiGranularityComparisonError, not ValueError — we surface it unchanged
    per the canonical Phase A helper contract)."""
    from tac.master_gradient_comparison.multi_granularity import (
        MultiGranularityComparisonError,
    )

    gt = _make_matching_resolution_contest_tensor(synthetic_level, seed=4)
    with pytest.raises(MultiGranularityComparisonError, match="reduction"):
        empirical_sensitivity_map_from_master_gradient(
            synthetic_level, gt, reduction="bogus_reduction"
        )


def test_path_b2_m8_invariant_holds_under_uniform_gradient(
    synthetic_level: LevelDimensionContract,
) -> None:
    """M8 Protocol invariant: when gradient is constant across (h,w), Path B2
    output is spatially uniform and M8's weighted L2 == per-channel rescaled L2.

    This is the canonical sanity check that empirical sensitivity converges
    to L2 in the limit of uniform scorer response — Yousfi's 'empty prior'.
    """
    import hashlib
    from pathlib import Path

    from tac.master_gradient_comparison.multi_granularity import (
        ContestGradientTensor,
        OperatingPoint,
    )

    H, W = synthetic_level.wavelet_subband_shape
    # Constant gradient = 1.0 across every (pair, axis, h, w)
    arr_grad = np.ones((4, 3, H, W), dtype=np.float32)
    sha = hashlib.sha256(arr_grad.tobytes()).hexdigest()
    tmp = (
        Path(".omx/state/master_gradient_comparison")
        / f"test_phase_d_uniform_{sha[:8]}.npy"
    )
    tmp.parent.mkdir(parents=True, exist_ok=True)
    np.save(tmp, arr_grad)
    gt = ContestGradientTensor(
        array_path=str(tmp),
        array_sha256=sha,
        n_pairs=4,
        height=H,
        width=W,
        contest_video_sha256="0" * 64,
        scorer_seg_sha256="1" * 64,
        scorer_pose_sha256="2" * 64,
        operating_point=OperatingPoint(
            d_seg=0.067, d_pose=3.4e-5, rate=0.119, score=0.192
        ),
        captured_at_utc="2026-05-30T00:00:00Z",
        measurement_axis="[predicted]",
    )
    arr = empirical_sensitivity_map_from_master_gradient(
        synthetic_level, gt, reduction="l2_norm"
    )
    # L2 of [1, 1, 1] = sqrt(3); broadcast uniformly across all pixels
    expected = float(np.sqrt(3.0))
    np.testing.assert_allclose(arr, expected, rtol=1e-5)


# -----------------------------------------------------------------
# Section F: Dispatcher routing for ScorerSensitivityMapSource.EMPIRICAL_FROM_MASTER_GRADIENT
# -----------------------------------------------------------------


def test_dispatcher_enum_has_path_b2_member() -> None:
    """Phase D added EMPIRICAL_FROM_MASTER_GRADIENT to the enum (4 members)."""
    assert (
        ScorerSensitivityMapSource.EMPIRICAL_FROM_MASTER_GRADIENT.value
        == "empirical_from_master_gradient"
    )
    members = {m.name for m in ScorerSensitivityMapSource}
    assert members == {
        "UNIFORM",
        "EMPIRICAL_SLOT_GGG",
        "FINITE_DIFFERENCE_UNIWARD",
        "EMPIRICAL_FROM_MASTER_GRADIENT",
    }


def test_dispatcher_path_b2_routes_with_gradient_tensor(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Dispatcher with source=EMPIRICAL_FROM_MASTER_GRADIENT forwards gradient."""
    gt = _make_matching_resolution_contest_tensor(synthetic_level, seed=5)
    sm = Z8ScorerSensitivityMap(
        ScorerSensitivityMapSource.EMPIRICAL_FROM_MASTER_GRADIENT
    )
    arr = sm.get_for_level(synthetic_level, gradient_tensor=gt)
    assert arr.shape == (1, 3, 192, 256)
    assert np.all(arr >= 0)


def test_dispatcher_path_b2_raises_without_gradient_tensor(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Path B2 dispatch without gradient_tensor must raise with actionable hint."""
    sm = Z8ScorerSensitivityMap(
        ScorerSensitivityMapSource.EMPIRICAL_FROM_MASTER_GRADIENT
    )
    with pytest.raises(ValueError) as exc_info:
        sm.get_for_level(synthetic_level)  # No gradient_tensor
    msg = str(exc_info.value)
    assert "EMPIRICAL_FROM_MASTER_GRADIENT" in msg
    assert "gradient_tensor" in msg
    assert "Path A" in msg  # Operator-actionable fallback hint


def test_dispatcher_uniform_ignores_gradient_tensor(
    synthetic_level: LevelDimensionContract,
) -> None:
    """UNIFORM path ignores gradient_tensor (Phase A unchanged by Phase D)."""
    gt = _make_matching_resolution_contest_tensor(synthetic_level, seed=6)
    sm = Z8ScorerSensitivityMap(ScorerSensitivityMapSource.UNIFORM)
    arr = sm.get_for_level(synthetic_level, gradient_tensor=gt)
    # Still returns all-ones per Path A
    assert np.all(arr == 1.0)


def test_convenience_builder_path_b2_routes(
    synthetic_level: LevelDimensionContract,
) -> None:
    """build_z8_scorer_sensitivity_map_for_level forwards Phase D kwargs."""
    gt = _make_matching_resolution_contest_tensor(synthetic_level, seed=7)
    arr = build_z8_scorer_sensitivity_map_for_level(
        synthetic_level,
        source=ScorerSensitivityMapSource.EMPIRICAL_FROM_MASTER_GRADIENT,
        gradient_tensor=gt,
        reduction="l1_norm",
    )
    assert arr.shape == (1, 3, 192, 256)
    assert np.all(arr >= 0)


def test_existing_path_b_stub_still_raises_after_phase_d(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Phase D adds Path B2 WITHOUT changing the Path B (slot_ggg) stub."""
    sm = Z8ScorerSensitivityMap(ScorerSensitivityMapSource.EMPIRICAL_SLOT_GGG)
    with pytest.raises(EmpiricalSensitivityMapNotYetLandedError):
        sm.get_for_level(synthetic_level)


def test_existing_path_c_stub_still_raises_after_phase_d(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Phase D adds Path B2 WITHOUT changing the Path C (UNIWARD-FD) stub."""
    sm = Z8ScorerSensitivityMap(
        ScorerSensitivityMapSource.FINITE_DIFFERENCE_UNIWARD
    )
    with pytest.raises(EmpiricalSensitivityMapNotYetLandedError):
        sm.get_for_level(synthetic_level)


def test_resolution_mismatch_error_is_notimplementederror_subclass() -> None:
    """Per CLAUDE.md 'Forbidden premature KILL': honest deferral via NotImplementedError."""
    assert issubclass(
        ResolutionMismatchAwaitingPerLevelProjectionError, NotImplementedError
    )
