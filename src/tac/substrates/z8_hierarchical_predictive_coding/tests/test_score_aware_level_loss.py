# SPDX-License-Identifier: MIT
"""Tests for Z8 Phase 2 M8 — canonical Yousfi-grounded score-aware per-level loss.

Verifies :class:`tac.substrates.z8_hierarchical_predictive_coding.loss.ScoreAwareLevelLossImpl`
satisfies the
:class:`tac.substrates.z8_hierarchical_predictive_coding.binding_contract.ScoreAwareLevelLoss`
Protocol from ``binding_contract.py:419-472`` plus the canonical M8
invariants:

1. **Protocol satisfaction** — ``isinstance(...)`` against the
   ``@runtime_checkable`` Protocol returns True.

2. **Uniform sensitivity reduces to L2** — the canonical Protocol invariant
   from ``binding_contract.py:467-470``. With all-ones sensitivity the
   per-level loss equals standard per-pixel L2 mean.

3. **Non-uniform sensitivity reweights per-pixel contribution** — high-
   sensitivity regions dominate the loss when reconstruction error is
   concentrated there; the Slot GGG empirical-anchor sister test.

4. **Shape contract** — (B, C, H, W) at multiple resolutions; mismatched
   shapes raise.

5. **Non-negative sensitivity invariant** — negative weights produce a
   non-loss and are rejected per the Protocol docstring.

6. **Integration with M7** — ``Z8ScorerSensitivityMap.get_for_level(...)``
   produces M8-consumable sensitivity tensors.

7. **Integration with M7 Path B2 (Phase C)** — per-level Mallat dyadic
   projection from ``decompose_M_contest_per_level`` produces M8-consumable
   downsampled sensitivity tensors.

Per Catalog #287 evidence-tag discipline: no docstring overstatement; every
numerical claim is paired with adjacent source/observed evidence.
Per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE": these tests are
implementation-correctness tests of the loss kernel, NOT score-claim
witnesses; no [contest-CUDA] / [contest-CPU] tagging required.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.z8_hierarchical_predictive_coding import (
    LevelDimensionContract,
    ScoreAwareLevelLoss,
    ScorerSensitivityMapSource,
    Z8ScorerSensitivityMap,
    uniform_sensitivity_map_for_level,
)
from tac.substrates.z8_hierarchical_predictive_coding.loss import (
    InvalidSensitivityMapError,
    ScoreAwareLevelLossImpl,
    build_score_aware_level_loss_for_level,
)


# Synthetic level fixture mirrors the M7 test pattern (H=192, W=256
# at level 0; canonical Z8 full-resolution per-level wavelet subband shape).
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
# Section A: Protocol satisfaction (Catalog #335 sister discipline)
# -----------------------------------------------------------------


def test_satisfies_score_aware_level_loss_protocol() -> None:
    """The implementation satisfies the @runtime_checkable Protocol.

    Per binding_contract.py:419 ``ScoreAwareLevelLoss`` is decorated
    @runtime_checkable so isinstance returns True iff the structural
    Protocol contract is satisfied (i.e. the per_level_loss method
    signature matches).
    """
    impl = ScoreAwareLevelLossImpl()
    assert isinstance(impl, ScoreAwareLevelLoss)


def test_per_level_loss_method_callable() -> None:
    """per_level_loss is callable with (reconstruction, target, sensitivity_map)."""
    impl = ScoreAwareLevelLossImpl()
    assert callable(impl.per_level_loss)


def test_default_construction_uses_canonical_defaults() -> None:
    """Defaults: norm='l2' (Fridrich UNIWARD canonical) + reduction='mean'."""
    impl = ScoreAwareLevelLossImpl()
    assert impl.norm == "l2"
    assert impl.reduction == "mean"
    assert impl.validate_non_negative_sensitivity is True


def test_dataclass_is_frozen() -> None:
    """Frozen so instances are hashable + safe to share across levels."""
    impl = ScoreAwareLevelLossImpl()
    with pytest.raises((AttributeError, Exception)):
        impl.norm = "l1"  # type: ignore[misc]


# -----------------------------------------------------------------
# Section B: Construction-time validation (Catalog #287 explicit-input discipline)
# -----------------------------------------------------------------


def test_invalid_norm_rejected() -> None:
    """norm must be one of ('l2', 'l1')."""
    with pytest.raises(ValueError, match="norm must be one of"):
        ScoreAwareLevelLossImpl(norm="linfty")  # type: ignore[arg-type]


def test_invalid_reduction_rejected() -> None:
    """reduction must be one of ('mean', 'sum')."""
    with pytest.raises(ValueError, match="reduction must be one of"):
        ScoreAwareLevelLossImpl(reduction="none")  # type: ignore[arg-type]


def test_l1_norm_accepted() -> None:
    """Sister formulation: 'l1' = absolute-error."""
    impl = ScoreAwareLevelLossImpl(norm="l1")
    assert impl.norm == "l1"


def test_sum_reduction_accepted() -> None:
    """Sister reduction: 'sum' returns the raw weighted sum (no /N)."""
    impl = ScoreAwareLevelLossImpl(reduction="sum")
    assert impl.reduction == "sum"


# -----------------------------------------------------------------
# Section C: Canonical M8 Protocol invariant — uniform sensitivity reduces to L2
# -----------------------------------------------------------------


def test_uniform_sensitivity_reduces_to_l2_mean(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Canonical M8 invariant from binding_contract.py:467-470.

    With sensitivity_map == 1 everywhere, per_level_loss must equal the
    standard L2 mean reconstruction loss.
    """
    rng = np.random.default_rng(seed=42)
    H, W = synthetic_level.wavelet_subband_shape
    reconstruction = rng.standard_normal((1, 3, H, W)).astype(np.float32)
    target = rng.standard_normal((1, 3, H, W)).astype(np.float32)
    sensitivity = uniform_sensitivity_map_for_level(synthetic_level)

    impl = ScoreAwareLevelLossImpl()
    loss = impl.per_level_loss(reconstruction, target, sensitivity)

    # Canonical reference L2 mean: mean((recon - target)^2) over all dims.
    expected_l2 = np.mean((reconstruction - target) ** 2)
    np.testing.assert_allclose(float(loss), float(expected_l2), rtol=1e-6)


def test_uniform_sensitivity_reduces_to_l1_mean(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Sister invariant for norm='l1': uniform sensitivity reduces to L1 mean."""
    rng = np.random.default_rng(seed=42)
    H, W = synthetic_level.wavelet_subband_shape
    reconstruction = rng.standard_normal((1, 3, H, W)).astype(np.float32)
    target = rng.standard_normal((1, 3, H, W)).astype(np.float32)
    sensitivity = uniform_sensitivity_map_for_level(synthetic_level)

    impl = ScoreAwareLevelLossImpl(norm="l1")
    loss = impl.per_level_loss(reconstruction, target, sensitivity)

    expected_l1 = np.mean(np.abs(reconstruction - target))
    np.testing.assert_allclose(float(loss), float(expected_l1), rtol=1e-6)


def test_uniform_sensitivity_sum_reduction_equals_sum_of_squared_error(
    synthetic_level: LevelDimensionContract,
) -> None:
    """reduction='sum' returns the raw sum of squared error (no /N)."""
    rng = np.random.default_rng(seed=42)
    H, W = synthetic_level.wavelet_subband_shape
    reconstruction = rng.standard_normal((1, 3, H, W)).astype(np.float32)
    target = rng.standard_normal((1, 3, H, W)).astype(np.float32)
    sensitivity = uniform_sensitivity_map_for_level(synthetic_level)

    impl = ScoreAwareLevelLossImpl(reduction="sum")
    loss = impl.per_level_loss(reconstruction, target, sensitivity)

    expected_sum = np.sum((reconstruction - target) ** 2)
    np.testing.assert_allclose(float(loss), float(expected_sum), rtol=1e-6)


# -----------------------------------------------------------------
# Section D: Non-uniform sensitivity reweights per-pixel contribution
# (Slot GGG empirical-anchor sister tests)
# -----------------------------------------------------------------


def test_non_uniform_sensitivity_reweights_per_pixel(
    synthetic_level: LevelDimensionContract,
) -> None:
    """High-sensitivity region with same total error >> uniform with same error.

    The canonical Yousfi-grounded behavior: when reconstruction error
    falls in a high-sensitivity region, the weighted loss is LARGER than
    the same error in a uniform-sensitivity region. This is the empirical
    receipt that the loss IS doing Yousfi reweighting, not just generic L2.
    """
    H, W = synthetic_level.wavelet_subband_shape
    # Reconstruction differs from target by 1.0 in a single pixel quadrant
    # (top-left H/2 x W/2 region), zero elsewhere.
    reconstruction = np.zeros((1, 3, H, W), dtype=np.float32)
    target = np.zeros((1, 3, H, W), dtype=np.float32)
    error_region = np.s_[:, :, : H // 2, : W // 2]
    reconstruction[error_region] = 1.0  # squared error = 1.0 in this region

    # Sensitivity map A: uniform 1.0 everywhere (baseline).
    sensitivity_uniform = np.ones((1, 3, H, W), dtype=np.float32)
    # Sensitivity map B: 10.0 in the error region, 1.0 elsewhere.
    sensitivity_high_in_error = np.ones((1, 3, H, W), dtype=np.float32)
    sensitivity_high_in_error[error_region] = 10.0

    impl = ScoreAwareLevelLossImpl(reduction="sum")
    loss_uniform = float(
        impl.per_level_loss(reconstruction, target, sensitivity_uniform)
    )
    loss_high = float(
        impl.per_level_loss(reconstruction, target, sensitivity_high_in_error)
    )

    # The high-sensitivity-in-error map produces a strictly larger loss
    # because the same error contributes 10x in the error region.
    assert loss_high > loss_uniform, (
        f"High-sensitivity-in-error loss ({loss_high}) must exceed "
        f"uniform loss ({loss_uniform}) — this is the canonical "
        f"Yousfi-grounded reweighting behavior."
    )

    # Concrete arithmetic: error region has H/2 * W/2 * 3 pixels with
    # squared_error=1.0; rest has zero error. Loss_uniform = (H/2)*(W/2)*3*1.
    # Loss_high = (H/2)*(W/2)*3*10. Ratio should be exactly 10.0.
    expected_ratio = 10.0
    actual_ratio = loss_high / loss_uniform
    np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-6)


def test_zero_sensitivity_region_drops_error_contribution(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Sister of Slot GGG SegNet-null finding: error in zero-sensitivity region contributes 0."""
    H, W = synthetic_level.wavelet_subband_shape
    reconstruction = np.zeros((1, 3, H, W), dtype=np.float32)
    target = np.zeros((1, 3, H, W), dtype=np.float32)
    reconstruction[:, :, : H // 2, : W // 2] = 1.0  # error in top-left

    # Sensitivity is 0.0 in the error region, 1.0 elsewhere (scorer-blind
    # in the error region — the canonical SegNet-null analog).
    sensitivity = np.ones((1, 3, H, W), dtype=np.float32)
    sensitivity[:, :, : H // 2, : W // 2] = 0.0

    impl = ScoreAwareLevelLossImpl(reduction="sum")
    loss = float(impl.per_level_loss(reconstruction, target, sensitivity))

    # Error region contribution: 1.0 * 0.0 = 0.0 everywhere; non-error
    # region contribution: 0.0 * 1.0 = 0.0 everywhere. Total = 0.0.
    np.testing.assert_allclose(loss, 0.0, atol=1e-7)


# -----------------------------------------------------------------
# Section E: Shape contract (Protocol invariant from binding_contract.py:459-461)
# -----------------------------------------------------------------


@pytest.mark.parametrize("H,W", [(8, 12), (48, 64), (192, 256), (1, 1)])
def test_shape_contract_at_multiple_resolutions(H: int, W: int) -> None:
    """The loss handles arbitrary (B, C, H, W) shapes."""
    reconstruction = np.zeros((2, 3, H, W), dtype=np.float32)
    target = np.zeros((2, 3, H, W), dtype=np.float32)
    sensitivity = np.ones((2, 3, H, W), dtype=np.float32)

    impl = ScoreAwareLevelLossImpl()
    loss = impl.per_level_loss(reconstruction, target, sensitivity)
    assert float(loss) == 0.0  # zero error -> zero loss regardless of shape


def test_shape_mismatch_between_reconstruction_and_target_raises() -> None:
    """Reconstruction and target shapes MUST match per Protocol contract."""
    reconstruction = np.zeros((1, 3, 8, 12), dtype=np.float32)
    target = np.zeros((1, 3, 8, 16), dtype=np.float32)  # different W
    sensitivity = np.ones((1, 3, 8, 12), dtype=np.float32)

    impl = ScoreAwareLevelLossImpl()
    with pytest.raises(ValueError, match="identical shapes"):
        impl.per_level_loss(reconstruction, target, sensitivity)


def test_broadcast_compatible_sensitivity_accepted() -> None:
    """Sensitivity (1, 1, H, W) broadcasts across (B, C, H, W) per Protocol.

    The Fridrich UNIWARD canonical convention is channel-uniform per-
    spatial-location maps; sensitivity_shape=(1, 1, H, W) is valid and
    broadcasts across batch + channels at multiplication time.
    """
    reconstruction = np.zeros((2, 3, 8, 12), dtype=np.float32)
    target = np.zeros((2, 3, 8, 12), dtype=np.float32)
    sensitivity = np.ones((1, 1, 8, 12), dtype=np.float32)  # broadcast

    impl = ScoreAwareLevelLossImpl()
    loss = impl.per_level_loss(reconstruction, target, sensitivity)
    assert float(loss) == 0.0


# -----------------------------------------------------------------
# Section F: Non-negative sensitivity invariant (Protocol docstring 462-464)
# -----------------------------------------------------------------


def test_negative_sensitivity_rejected_by_default() -> None:
    """Default validation rejects negative sensitivity per the Protocol invariant."""
    reconstruction = np.zeros((1, 3, 8, 8), dtype=np.float32)
    target = np.zeros((1, 3, 8, 8), dtype=np.float32)
    sensitivity = np.ones((1, 3, 8, 8), dtype=np.float32)
    sensitivity[0, 0, 0, 0] = -1.0  # one negative entry

    impl = ScoreAwareLevelLossImpl()
    with pytest.raises(InvalidSensitivityMapError, match="non-negative"):
        impl.per_level_loss(reconstruction, target, sensitivity)


def test_negative_sensitivity_passes_when_validation_disabled() -> None:
    """Disabling validation is the hot-path escape hatch."""
    reconstruction = np.zeros((1, 3, 8, 8), dtype=np.float32)
    target = np.ones((1, 3, 8, 8), dtype=np.float32)  # squared_error = 1.0
    sensitivity = -1.0 * np.ones((1, 3, 8, 8), dtype=np.float32)

    impl = ScoreAwareLevelLossImpl(validate_non_negative_sensitivity=False)
    loss = float(impl.per_level_loss(reconstruction, target, sensitivity))
    # Per-pixel: -1.0 * 1.0 = -1.0; mean over all pixels = -1.0.
    np.testing.assert_allclose(loss, -1.0, rtol=1e-6)


def test_zero_sensitivity_accepted() -> None:
    """Zero sensitivity is non-negative and produces zero loss."""
    reconstruction = np.ones((1, 3, 8, 8), dtype=np.float32)
    target = np.zeros((1, 3, 8, 8), dtype=np.float32)
    sensitivity = np.zeros((1, 3, 8, 8), dtype=np.float32)

    impl = ScoreAwareLevelLossImpl()
    loss = float(impl.per_level_loss(reconstruction, target, sensitivity))
    assert loss == 0.0


# -----------------------------------------------------------------
# Section G: Integration with M7 Z8ScorerSensitivityMap (Path A UNIFORM)
# -----------------------------------------------------------------


def test_integration_with_m7_uniform_path(
    synthetic_level: LevelDimensionContract,
) -> None:
    """M7 Path A (UNIFORM) produces a sensitivity tensor M8 consumes correctly.

    End-to-end witness: M7 dispatcher produces the sensitivity tensor, M8
    consumes it, and the result is the standard L2 mean (the canonical
    Protocol invariant).
    """
    rng = np.random.default_rng(seed=123)
    H, W = synthetic_level.wavelet_subband_shape
    reconstruction = rng.standard_normal((1, 3, H, W)).astype(np.float32)
    target = rng.standard_normal((1, 3, H, W)).astype(np.float32)

    # M7 produces the canonical sensitivity tensor.
    m7 = Z8ScorerSensitivityMap(source=ScorerSensitivityMapSource.UNIFORM)
    sensitivity = m7.get_for_level(synthetic_level)

    # M8 consumes it.
    impl = ScoreAwareLevelLossImpl()
    loss = impl.per_level_loss(reconstruction, target, sensitivity)

    expected_l2 = np.mean((reconstruction - target) ** 2)
    np.testing.assert_allclose(float(loss), float(expected_l2), rtol=1e-6)


def _make_matching_resolution_contest_tensor(
    level: LevelDimensionContract,
    seed: int = 0,
    n_pairs: int = 1,
    n_axes: int = 3,
):
    """Build a synthetic ContestGradientTensor at the level's resolution.

    Mirrors the sister fixture in
    ``tests/test_scorer_sensitivity_map.py`` (line 280). ContestGradientTensor
    requires ``array_path`` + ``array_sha256`` so we persist the synthetic
    array to a deterministic local cache path under
    ``.omx/state/master_gradient_comparison/``.
    """
    import hashlib
    from pathlib import Path

    from tac.master_gradient_comparison.multi_granularity import (
        ContestGradientTensor,
        OperatingPoint,
    )

    H, W = level.wavelet_subband_shape
    rng = np.random.RandomState(seed)
    arr = rng.uniform(0.1, 2.0, size=(n_pairs, n_axes, H, W)).astype(np.float32)
    sha = hashlib.sha256(arr.tobytes()).hexdigest()
    tmp = (
        Path(".omx/state/master_gradient_comparison")
        / f"test_z8_phase_e_loss_contest_{sha[:8]}.npy"
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


def test_integration_with_m7_path_b2_identity_resolution(
    synthetic_level_coarse: LevelDimensionContract,
) -> None:
    """M7 Path B2 (EMPIRICAL_FROM_MASTER_GRADIENT) identity-resolution path.

    Constructs a synthetic ContestGradientTensor whose native shape matches
    the level's wavelet_subband_shape (identity-resolution Phase D wire-in;
    auto_project_to_level=False is sufficient), then verifies M8 consumes
    the resulting (B, C, H, W) sensitivity tensor correctly.

    This is the canonical end-to-end Yousfi-grounded path: master gradient
    → extract_M_pixel → broadcast → M8 weighted loss.
    """
    H, W = synthetic_level_coarse.wavelet_subband_shape  # (48, 64)

    gradient_tensor = _make_matching_resolution_contest_tensor(
        synthetic_level_coarse
    )

    m7 = Z8ScorerSensitivityMap(
        source=ScorerSensitivityMapSource.EMPIRICAL_FROM_MASTER_GRADIENT
    )
    sensitivity = m7.get_for_level(
        synthetic_level_coarse,
        gradient_tensor=gradient_tensor,
        reduction="l2_norm",
    )

    # Verify the sensitivity tensor satisfies M8's shape contract.
    assert sensitivity.shape == (1, 3, H, W)
    # Sister non-negativity invariant: l2_norm output is always >= 0.
    assert np.all(sensitivity >= 0)

    # M8 consumes the M7 Path B2 sensitivity tensor + a synthetic
    # reconstruction/target pair. End-to-end witness that the Yousfi-
    # grounded path produces a finite scalar loss.
    rng2 = np.random.default_rng(seed=11)
    reconstruction = rng2.standard_normal((1, 3, H, W)).astype(np.float32)
    target = rng2.standard_normal((1, 3, H, W)).astype(np.float32)

    impl = ScoreAwareLevelLossImpl()
    loss = impl.per_level_loss(reconstruction, target, sensitivity)
    assert np.isfinite(float(loss))
    # Loss is non-negative because squared_error * non_negative_sensitivity >= 0.
    assert float(loss) >= 0


def test_integration_with_m7_path_b2_phase_c_dyadic_projection(
    synthetic_level_coarse: LevelDimensionContract,
) -> None:
    """M7 Path B2 with auto_project_to_level=True (Phase C dyadic projection).

    The canonical Mallat dyadic projection from native gradient resolution
    to per-level resolution; closes the cascade Phase A → Phase C → Phase D
    → M8 end-to-end at non-identity resolution.

    Uses a coarser-than-native level so the Phase C projection actually
    downsamples (the canonical use case for hierarchical levels deeper than
    the gradient's native resolution).
    """
    H_level, W_level = synthetic_level_coarse.wavelet_subband_shape  # (48, 64)
    # Build a gradient at 2x the level resolution so Phase C downsamples by 2.
    import hashlib
    from pathlib import Path

    from tac.master_gradient_comparison.multi_granularity import (
        ContestGradientTensor,
        OperatingPoint,
    )

    H_grad, W_grad = H_level * 2, W_level * 2
    rng = np.random.RandomState(seed=42)
    arr = rng.uniform(0.1, 2.0, size=(1, 3, H_grad, W_grad)).astype(np.float32)
    sha = hashlib.sha256(arr.tobytes()).hexdigest()
    tmp = (
        Path(".omx/state/master_gradient_comparison")
        / f"test_z8_phase_e_phase_c_{sha[:8]}.npy"
    )
    tmp.parent.mkdir(parents=True, exist_ok=True)
    np.save(tmp, arr)
    gradient_tensor = ContestGradientTensor(
        array_path=str(tmp),
        array_sha256=sha,
        n_pairs=1,
        height=H_grad,
        width=W_grad,
        contest_video_sha256="0" * 64,
        scorer_seg_sha256="1" * 64,
        scorer_pose_sha256="2" * 64,
        operating_point=OperatingPoint(
            d_seg=0.067, d_pose=3.4e-5, rate=0.119, score=0.192
        ),
        captured_at_utc="2026-05-30T00:00:00Z",
        measurement_axis="[predicted]",
    )

    m7 = Z8ScorerSensitivityMap(
        source=ScorerSensitivityMapSource.EMPIRICAL_FROM_MASTER_GRADIENT
    )
    sensitivity = m7.get_for_level(
        synthetic_level_coarse,
        gradient_tensor=gradient_tensor,
        reduction="l2_norm",
        auto_project_to_level=True,
        level_projection_reduction="mean",
    )

    # After dyadic projection sensitivity matches the level's resolution.
    assert sensitivity.shape == (1, 3, H_level, W_level)
    assert np.all(sensitivity >= 0)

    rng2 = np.random.default_rng(seed=11)
    reconstruction = rng2.standard_normal((1, 3, H_level, W_level)).astype(np.float32)
    target = rng2.standard_normal((1, 3, H_level, W_level)).astype(np.float32)

    impl = ScoreAwareLevelLossImpl()
    loss = impl.per_level_loss(reconstruction, target, sensitivity)
    assert np.isfinite(float(loss))
    assert float(loss) >= 0


# -----------------------------------------------------------------
# Section H: build_score_aware_level_loss_for_level builder
# -----------------------------------------------------------------


def test_builder_returns_protocol_satisfying_instance(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Builder returns a Protocol-satisfying instance."""
    impl = build_score_aware_level_loss_for_level(synthetic_level)
    assert isinstance(impl, ScoreAwareLevelLoss)


def test_builder_forwards_norm_and_reduction(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Builder forwards norm + reduction kwargs."""
    impl = build_score_aware_level_loss_for_level(
        synthetic_level, norm="l1", reduction="sum"
    )
    assert impl.norm == "l1"
    assert impl.reduction == "sum"


def test_builder_rejects_non_level() -> None:
    """Builder validates level is a LevelDimensionContract."""
    with pytest.raises(TypeError, match="LevelDimensionContract"):
        build_score_aware_level_loss_for_level("not_a_level")  # type: ignore[arg-type]


def test_builder_at_coarse_level(
    synthetic_level_coarse: LevelDimensionContract,
) -> None:
    """Builder works at non-zero level indices."""
    impl = build_score_aware_level_loss_for_level(synthetic_level_coarse)
    assert isinstance(impl, ScoreAwareLevelLossImpl)


# -----------------------------------------------------------------
# Section I: torch tensor framework-agnostic compatibility
# -----------------------------------------------------------------


def test_torch_tensor_path_works_when_torch_available() -> None:
    """When torch is available, ScoreAwareLevelLossImpl handles torch tensors.

    Framework-agnostic via duck-typed element-wise operations (- * ** abs
    .mean()). The same code path serves numpy + torch + mlx.
    """
    pytest.importorskip("torch")
    import torch

    rng = np.random.default_rng(seed=99)
    reconstruction_np = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    target_np = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    sensitivity_np = np.ones((1, 3, 8, 8), dtype=np.float32)

    # Convert to torch tensors.
    reconstruction = torch.from_numpy(reconstruction_np)
    target = torch.from_numpy(target_np)
    sensitivity = torch.from_numpy(sensitivity_np)

    impl = ScoreAwareLevelLossImpl()
    loss = impl.per_level_loss(reconstruction, target, sensitivity)

    # The result should be a torch scalar tensor (autograd-compatible).
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar

    # Numerically matches the canonical L2 mean.
    expected_l2 = np.mean((reconstruction_np - target_np) ** 2)
    np.testing.assert_allclose(float(loss.item()), float(expected_l2), rtol=1e-6)


def test_torch_autograd_flows_through_loss() -> None:
    """torch tensors with requires_grad propagate gradient through the loss.

    Critical for the M8-in-trainer use case: the trainer's optimizer must
    be able to backprop through ScoreAwareLevelLossImpl.
    """
    pytest.importorskip("torch")
    import torch

    reconstruction = torch.randn(1, 3, 4, 4, requires_grad=True)
    target = torch.zeros(1, 3, 4, 4)
    sensitivity = torch.ones(1, 3, 4, 4)

    impl = ScoreAwareLevelLossImpl()
    loss = impl.per_level_loss(reconstruction, target, sensitivity)
    loss.backward()

    # Gradient should flow back to reconstruction.
    assert reconstruction.grad is not None
    # Gradient of mean((recon - 0)^2) w.r.t. recon = 2 * recon / N.
    N = float(reconstruction.numel())
    expected_grad = 2.0 * reconstruction.detach() / N
    torch.testing.assert_close(reconstruction.grad, expected_grad, rtol=1e-6, atol=1e-7)


# -----------------------------------------------------------------
# Section J: __all__ export hygiene
# -----------------------------------------------------------------


def test_loss_module_exports_canonical_public_api() -> None:
    """The loss module exports ScoreAwareLevelLossImpl + builder + error."""
    from tac.substrates.z8_hierarchical_predictive_coding import loss as _loss

    assert "ScoreAwareLevelLossImpl" in _loss.__all__
    assert "build_score_aware_level_loss_for_level" in _loss.__all__
    assert "InvalidSensitivityMapError" in _loss.__all__


def test_loss_module_reexported_from_package_init() -> None:
    """Package __init__ re-exports ScoreAwareLevelLossImpl + sister."""
    from tac.substrates import z8_hierarchical_predictive_coding as _pkg

    assert hasattr(_pkg, "ScoreAwareLevelLossImpl")
    assert hasattr(_pkg, "build_score_aware_level_loss_for_level")
    assert hasattr(_pkg, "InvalidSensitivityMapError")
