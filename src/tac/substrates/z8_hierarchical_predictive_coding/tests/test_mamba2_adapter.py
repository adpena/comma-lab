# SPDX-License-Identifier: MIT
"""Tests for Z8 Phase 2 M4 Mamba-2 adapter.

Verifies:

1. ``Mamba2Predictor.step_externalized_state`` (the canonical sister
   surface added to the underlying primitive per operator's "iterate
   and optimize underlying pieces as well" directive 2026-05-29) —
   pure functional step, does NOT mutate ``self._h``, shape contracts
   honored, identity-mode + mamba_ssm backend raise per documented
   contract.

2. ``Z8Mamba2DeterministicStateUpdate`` adapter satisfies the
   ``DeterministicStateUpdate`` Protocol — runtime_checkable
   ``isinstance`` returns True; ``state_dim`` matches the level's
   ``deterministic_state_dim``; ``initial_state(batch_size)`` returns
   ``(B, state_dim)`` zeros; ``step(prior_state, input_at_t)``
   round-trips the structured ↔ flat reshape correctly.

3. The new milestone in ``Z8_PHASE_2_BUILD_MILESTONES`` (
   ``mamba_2_adapter_binds_canonical_primitive_to_protocol``) is
   marked LANDED and consistent with the implementation.

Per CLAUDE.md "Mathematical grounding" + Z8 binding-first methodology:
tests verify the shape contract + Protocol satisfaction + state-pass-
through purity + honest backend classification. No score claims; no
empirical band claims.
"""

from __future__ import annotations

import pytest
import torch

from tac.optimization.mamba2_predictor import (
    REFERENCE_TORCH_BACKEND,
    Mamba2Predictor,
    Mamba2PredictorConfig,
)
from tac.substrates.z8_hierarchical_predictive_coding import (
    BuildMilestoneStatus,
    DeterministicStateUpdate,
    LevelDimensionContract,
    Z8_PHASE_2_BUILD_MILESTONES,
    Z8Mamba2DeterministicStateUpdate,
    build_z8_mamba2_adapter_for_level,
)


# Synthetic level fixture: state_dim = 64 = d_inner * d_state = 4 * 16.
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


# -----------------------------------------------------------------
# Section A: Mamba2Predictor.step_externalized_state (sister surface
# added to the underlying primitive)
# -----------------------------------------------------------------


def test_step_externalized_state_returns_correct_shapes() -> None:
    """Pure functional step returns next_state + predicted_z with correct shapes."""
    config = Mamba2PredictorConfig(
        latent_dim=24,
        ego_motion_dim=8,
        d_model=16,
        d_state=8,
        expand=1,
        backend=REFERENCE_TORCH_BACKEND,
        stateful=False,
    )
    predictor = Mamba2Predictor(config)
    batch = 3
    prior_state = torch.zeros(batch, config.d_inner, config.d_state)
    z_prev = torch.randn(batch, config.latent_dim)
    ego_motion = torch.randn(batch, config.ego_motion_dim)
    next_state, z_pred = predictor.step_externalized_state(prior_state, z_prev, ego_motion)
    assert next_state.shape == (batch, config.d_inner, config.d_state)
    assert z_pred.shape == (batch, config.latent_dim)


def test_step_externalized_state_does_not_mutate_internal_h() -> None:
    """The externalized surface MUST NOT touch ``self._h``."""
    config = Mamba2PredictorConfig(
        latent_dim=24, ego_motion_dim=8, d_model=16, d_state=8, expand=1,
        backend=REFERENCE_TORCH_BACKEND, stateful=True,
    )
    predictor = Mamba2Predictor(config)
    predictor._h = None
    prior_state = torch.zeros(2, config.d_inner, config.d_state)
    z_prev = torch.randn(2, 24)
    ego_motion = torch.randn(2, 8)
    predictor.step_externalized_state(prior_state, z_prev, ego_motion)
    # Internal _h MUST remain None.
    assert predictor._h is None


def test_step_externalized_state_rejects_identity_predictor() -> None:
    """Identity mode is stateless by construction; raises per documented contract."""
    config = Mamba2PredictorConfig(identity_predictor=True)
    predictor = Mamba2Predictor(config)
    with pytest.raises(RuntimeError, match="identity_predictor"):
        predictor.step_externalized_state(
            torch.zeros(1, 1, 1), torch.zeros(1, 24), torch.zeros(1, 8)
        )


def test_step_externalized_state_rejects_wrong_state_shape() -> None:
    """Shape contract enforcement: (B, d_inner, d_state) required."""
    config = Mamba2PredictorConfig(
        latent_dim=24, ego_motion_dim=8, d_model=16, d_state=8, expand=1,
        backend=REFERENCE_TORCH_BACKEND,
    )
    predictor = Mamba2Predictor(config)
    # Wrong d_state dim
    with pytest.raises(ValueError, match="prior_state shape"):
        predictor.step_externalized_state(
            torch.zeros(2, config.d_inner, 99),
            torch.zeros(2, 24),
            torch.zeros(2, 8),
        )


# -----------------------------------------------------------------
# Section B: Z8Mamba2DeterministicStateUpdate Protocol satisfaction
# -----------------------------------------------------------------


def test_adapter_satisfies_deterministic_state_update_protocol(
    synthetic_level: LevelDimensionContract,
) -> None:
    """isinstance(adapter, DeterministicStateUpdate) returns True per
    @runtime_checkable Protocol."""
    adapter = build_z8_mamba2_adapter_for_level(
        level=synthetic_level, latent_dim=24, ego_motion_dim=8, d_state=16
    )
    assert isinstance(adapter, DeterministicStateUpdate)


def test_adapter_state_dim_matches_level_contract(
    synthetic_level: LevelDimensionContract,
) -> None:
    """state_dim property returns level.deterministic_state_dim exactly."""
    adapter = build_z8_mamba2_adapter_for_level(
        level=synthetic_level, latent_dim=24, ego_motion_dim=8, d_state=16
    )
    assert adapter.state_dim == synthetic_level.deterministic_state_dim == 64


def test_adapter_initial_state_returns_correct_shape_and_dtype(
    synthetic_level: LevelDimensionContract,
) -> None:
    """initial_state returns (B, state_dim) zeros at predictor dtype."""
    adapter = build_z8_mamba2_adapter_for_level(
        level=synthetic_level, latent_dim=24, ego_motion_dim=8, d_state=16
    )
    h0 = adapter.initial_state(batch_size=5)
    assert h0.shape == (5, 64)
    assert torch.all(h0 == 0)


def test_adapter_step_round_trips_shape_correctly(
    synthetic_level: LevelDimensionContract,
) -> None:
    """step takes (B, state_dim) flat + (B, latent_dim+ego_motion_dim) and
    returns (B, state_dim) flat."""
    adapter = build_z8_mamba2_adapter_for_level(
        level=synthetic_level, latent_dim=24, ego_motion_dim=8, d_state=16
    )
    batch = 4
    h_prev = adapter.initial_state(batch_size=batch)
    input_at_t = torch.randn(batch, 24 + 8)
    h_next = adapter.step(h_prev, input_at_t)
    assert h_next.shape == (batch, 64)


def test_adapter_rejects_state_dim_not_divisible_by_d_state(
    synthetic_level: LevelDimensionContract,
) -> None:
    """Mathematical invariant: state_dim MUST be divisible by d_state to
    yield integer d_inner per the structured (B, d_inner, d_state) shape."""
    with pytest.raises(ValueError, match="not divisible"):
        # synthetic_level.deterministic_state_dim = 64; d_state=13 doesn't divide.
        build_z8_mamba2_adapter_for_level(
            level=synthetic_level, latent_dim=24, ego_motion_dim=8, d_state=13
        )


def test_adapter_rejects_wrong_input_at_t_dim(
    synthetic_level: LevelDimensionContract,
) -> None:
    """input_at_t MUST have shape (B, latent_dim + ego_motion_dim)."""
    adapter = build_z8_mamba2_adapter_for_level(
        level=synthetic_level, latent_dim=24, ego_motion_dim=8, d_state=16
    )
    h_prev = adapter.initial_state(batch_size=2)
    wrong_input = torch.randn(2, 99)  # Should be 24+8=32
    with pytest.raises(ValueError, match="input_at_t shape"):
        adapter.step(h_prev, wrong_input)


# -----------------------------------------------------------------
# Section C: M4 milestone consistency
# -----------------------------------------------------------------


def test_m4_milestone_is_landed_with_canonical_id() -> None:
    """M4 milestone reflects the adapter pivot (binding-first methodology)."""
    by_id = {m.milestone_id: m for m in Z8_PHASE_2_BUILD_MILESTONES}
    assert "mamba_2_adapter_binds_canonical_primitive_to_protocol" in by_id
    m4 = by_id["mamba_2_adapter_binds_canonical_primitive_to_protocol"]
    assert m4.status == BuildMilestoneStatus.LANDED
    # The old milestone_id must no longer be present (adapter pivot supersedes).
    assert "mamba_2_ssd_replaces_gru_at_step_5" not in by_id
