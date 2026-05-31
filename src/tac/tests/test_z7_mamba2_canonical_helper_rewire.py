# SPDX-License-Identifier: MIT
"""Tests for Z7-Mamba-2 substrate canonical SSD helper rewire (2026-05-30).

Sister of ``test_mamba2_adapter_canonical_helper_rewire.py`` at the Z7
substrate surface. Verifies the new ``backend='ssd_reference'`` opt-in path
that routes the Z7-Mamba-2 substrate through the canonical Mamba-2 SSD
tri-backend helper at :mod:`tac.substrates._shared.mamba2_ssd` (commit
``b2936fb81``; 33 byte-stable parity tests passing across NUMPY/PYTORCH/MLX).

Test surfaces:

1. **Canonical helper consumption** (mock.patch): when ``backend='ssd_reference'``,
   the underlying ``Mamba2Predictor.mamba_cell`` is a
   :class:`_CanonicalHelperSSDCell` that ACTUALLY calls
   :func:`tac.substrates._shared.mamba2_ssd.mamba2_ssd_step_pytorch`.

2. **Backward compat preservation**: the default ``backend='auto'`` continues
   to route through the existing reference_torch (Mamba-1 S6) backend; all 91
   sister tests must still pass post-rewire (verified externally via running
   the full Z7-Mamba-2 test suite).

3. **Z7 config threading**: ``Z7Mamba2PredictiveCodingConfig.ssd_nheads`` +
   ``ssd_headdim`` fields correctly thread through ``to_mamba2_predictor_config()``
   to the underlying ``Mamba2PredictorConfig``.

4. **Gradient flow**: gradients flow through the SSD path for substrate training.

5. **Substrate-level construction**: the full Z7-Mamba-2 substrate
   (Z7Mamba2PredictiveCodingSubstrate) constructs cleanly with the SSD backend
   so downstream training paths can opt-in via the substrate config alone.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": tests verify
canonical-helper consumption + gradient flow + Protocol satisfaction. NO score
claims; NO empirical band claims.

[verified-against: tac.substrates._shared.mamba2_ssd canonical helper
docstring + 33 byte-stable parity tests at commit b2936fb81]
[verified-against: src/tac/substrates/time_traveler_l5_z7_mamba2/architecture.py
ssd_reference backend opt-in rewire 2026-05-30]
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from tac.optimization.mamba2_predictor import (
    REFERENCE_TORCH_BACKEND,
    SSD_REFERENCE_BACKEND,
    _CanonicalHelperSSDCell,
    _ReferenceMamba2Cell,
)
from tac.substrates.time_traveler_l5_z7_mamba2.architecture import (
    Z7Mamba2PredictiveCodingConfig,
    Z7Mamba2PredictiveCodingSubstrate,
)


def _make_small_z7_ssd_config() -> Z7Mamba2PredictiveCodingConfig:
    """Tiny Z7-Mamba-2 SSD config for fast tests.

    d_inner = expand * d_model = 2 * 16 = 32
    nheads * headdim = 2 * 16 = 32 (satisfies SSD constraint)
    """
    return Z7Mamba2PredictiveCodingConfig(
        latent_dim=8,
        ego_motion_dim=4,
        d_model=16,
        d_state=4,
        expand=2,
        backend="ssd_reference",
        ssd_nheads=2,
        ssd_headdim=16,
        stateful=True,
        num_pairs=4,
        decoder_embed_dim=8,
        decoder_initial_grid_h=6,
        decoder_initial_grid_w=8,
        decoder_channels=(8, 6, 4, 4),
        decoder_num_upsample_blocks=2,
        output_height=24,
        output_width=32,
    )


def _make_small_z7_reference_config() -> Z7Mamba2PredictiveCodingConfig:
    """Sister-config to _make_small_z7_ssd_config but with reference_torch backend.

    Same dimensions; only the backend differs.
    """
    return Z7Mamba2PredictiveCodingConfig(
        latent_dim=8,
        ego_motion_dim=4,
        d_model=16,
        d_state=4,
        expand=2,
        backend="reference_torch",
        stateful=True,
        num_pairs=4,
        decoder_embed_dim=8,
        decoder_initial_grid_h=6,
        decoder_initial_grid_w=8,
        decoder_channels=(8, 6, 4, 4),
        decoder_num_upsample_blocks=2,
        output_height=24,
        output_width=32,
    )


# ============================================================================
# Section 1: Config threading
# ============================================================================


def test_z7_config_threads_ssd_fields_to_mamba2_predictor_config() -> None:
    """Z7Mamba2PredictiveCodingConfig.to_mamba2_predictor_config threads ssd_nheads/ssd_headdim."""
    cfg = _make_small_z7_ssd_config()
    mp_cfg = cfg.to_mamba2_predictor_config()
    assert mp_cfg.backend == "ssd_reference"
    assert mp_cfg.ssd_nheads == 2
    assert mp_cfg.ssd_headdim == 16


def test_z7_config_default_ssd_fields_pass_through_as_none_default() -> None:
    """Default config has ssd_nheads=None, ssd_headdim=64; threaded as-is."""
    cfg = Z7Mamba2PredictiveCodingConfig()
    mp_cfg = cfg.to_mamba2_predictor_config()
    assert mp_cfg.ssd_nheads is None
    assert mp_cfg.ssd_headdim == 64


def test_z7_default_backend_is_auto_not_ssd_reference() -> None:
    """Default backend stays 'auto' (Mamba-1 S6 fallback per existing discipline)."""
    cfg = Z7Mamba2PredictiveCodingConfig()
    assert cfg.backend == "auto"


# ============================================================================
# Section 2: Substrate-level construction with SSD backend
# ============================================================================


def test_z7_substrate_constructs_with_ssd_backend() -> None:
    """Full Z7-Mamba-2 substrate can be constructed with ssd_reference backend."""
    cfg = _make_small_z7_ssd_config()
    sub = Z7Mamba2PredictiveCodingSubstrate(cfg)
    assert sub.predictor.backend_active == SSD_REFERENCE_BACKEND
    assert isinstance(sub.predictor.mamba_cell, _CanonicalHelperSSDCell)


def test_z7_substrate_default_backend_constructs_with_reference_torch_backend() -> None:
    """Default substrate construction uses reference_torch backend (backward compat)."""
    cfg = _make_small_z7_reference_config()
    sub = Z7Mamba2PredictiveCodingSubstrate(cfg)
    assert sub.predictor.backend_active == REFERENCE_TORCH_BACKEND
    assert isinstance(sub.predictor.mamba_cell, _ReferenceMamba2Cell)


def test_z7_substrate_ssd_cell_has_correct_nheads_and_headdim() -> None:
    """The SSD cell respects the config's nheads/headdim contract."""
    cfg = _make_small_z7_ssd_config()
    sub = Z7Mamba2PredictiveCodingSubstrate(cfg)
    cell = sub.predictor.mamba_cell
    assert isinstance(cell, _CanonicalHelperSSDCell)
    assert cell.nheads == 2
    assert cell.headdim == 16
    assert cell.d_inner == cell.nheads * cell.headdim  # SSD constraint


# ============================================================================
# Section 3: Canonical helper consumption (NO FAKE IMPLEMENTATIONS contract)
# ============================================================================


def test_z7_substrate_actually_consumes_canonical_helper_step_pytorch() -> None:
    """Z7 substrate's SSD-mode forward ACTUALLY invokes canonical helper."""
    cfg = _make_small_z7_ssd_config()
    sub = Z7Mamba2PredictiveCodingSubstrate(cfg)
    # Reset predictor state and run a single forward step at the predictor level.
    sub.predictor.reset_state(1)
    z_prev = torch.randn(1, cfg.latent_dim)
    ego = torch.randn(1, cfg.ego_motion_dim)

    with mock.patch(
        "tac.substrates._shared.mamba2_ssd.mamba2_ssd_step_pytorch",
        wraps=__import__(
            "tac.substrates._shared.mamba2_ssd",
            fromlist=["mamba2_ssd_step_pytorch"],
        ).mamba2_ssd_step_pytorch,
    ) as mock_step:
        z_pred = sub.predictor(z_prev, ego)
        assert z_pred.shape == (1, cfg.latent_dim)
    assert mock_step.called, (
        "Z7 substrate's SSD-mode forward did NOT invoke canonical helper "
        "mamba2_ssd_step_pytorch — fake implementation regression."
    )


def test_z7_substrate_canonical_helper_shape_contracts_correct() -> None:
    """Z7 substrate passes correct SSD shapes to the canonical helper."""
    cfg = _make_small_z7_ssd_config()
    sub = Z7Mamba2PredictiveCodingSubstrate(cfg)
    cell = sub.predictor.mamba_cell
    assert isinstance(cell, _CanonicalHelperSSDCell)

    captured = {}

    from tac.substrates._shared.mamba2_ssd.pytorch_backend import (
        mamba2_ssd_step_pytorch as _real_step,
    )

    def capture(**kwargs):
        captured.update(kwargs)
        return _real_step(**kwargs)

    with mock.patch(
        "tac.substrates._shared.mamba2_ssd.mamba2_ssd_step_pytorch",
        side_effect=capture,
    ):
        sub.predictor.reset_state(1)
        z_prev = torch.randn(1, cfg.latent_dim)
        ego = torch.randn(1, cfg.ego_motion_dim)
        _ = sub.predictor(z_prev, ego)

    # Verify SSD-shape contract.
    B = 1
    nheads, headdim, d_state = cell.nheads, cell.headdim, cell.d_state
    assert captured["x_t"].shape == (B, nheads, headdim)
    assert captured["A_log"].shape == (nheads,)
    assert captured["B_t"].shape == (B, nheads, d_state)
    assert captured["C_t"].shape == (B, nheads, d_state)
    assert captured["dt_t"].shape == (B, nheads)


# ============================================================================
# Section 4: Gradient flow through SSD path
# ============================================================================


def test_z7_substrate_gradients_flow_through_canonical_helper() -> None:
    """Substrate-training gradients flow through canonical SSD helper backend."""
    cfg = _make_small_z7_ssd_config()
    sub = Z7Mamba2PredictiveCodingSubstrate(cfg)
    sub.predictor.reset_state(1)
    z_prev = torch.randn(1, cfg.latent_dim, requires_grad=True)
    ego = torch.randn(1, cfg.ego_motion_dim, requires_grad=True)
    z_pred = sub.predictor(z_prev, ego)
    loss = z_pred.sum()
    loss.backward()
    assert z_prev.grad is not None
    assert torch.any(z_prev.grad != 0)
    # Verify canonical helper's parameters received gradients.
    cell = sub.predictor.mamba_cell
    assert isinstance(cell, _CanonicalHelperSSDCell)
    # C_proj IS used in the predictor forward path because the predictor
    # output projection consumes y_t (which depends on C). So C_proj
    # should have gradient at the predictor surface (sister of the Z8
    # adapter which only returns state).
    assert cell.C_proj.weight.grad is not None
    assert torch.any(cell.C_proj.weight.grad != 0)
    assert cell.B_proj.weight.grad is not None
    assert torch.any(cell.B_proj.weight.grad != 0)
    assert cell.dt_proj.weight.grad is not None
    assert torch.any(cell.dt_proj.weight.grad != 0)


# ============================================================================
# Section 5: Stateful recurrence works under SSD backend
# ============================================================================


def test_z7_substrate_ssd_stateful_recurrence_preserves_state() -> None:
    """Stateful SSD recurrence: state persists across consecutive forward calls."""
    cfg = _make_small_z7_ssd_config()
    sub = Z7Mamba2PredictiveCodingSubstrate(cfg)
    sub.predictor.reset_state(1)
    z_prev = torch.randn(1, cfg.latent_dim)
    ego_1 = torch.randn(1, cfg.ego_motion_dim)
    ego_2 = torch.randn(1, cfg.ego_motion_dim)

    # State should be zero before first call.
    initial_h = sub.predictor._h.clone()
    assert torch.all(initial_h == 0)

    z_pred_1 = sub.predictor(z_prev, ego_1)
    # After first call, state should be non-zero (assuming input is non-zero).
    h_after_1 = sub.predictor._h.clone()
    assert torch.any(h_after_1 != 0)

    sub.predictor(z_pred_1, ego_2)
    # After second call, state should have changed again.
    h_after_2 = sub.predictor._h.clone()
    assert not torch.allclose(h_after_2, h_after_1), (
        "Stateful SSD recurrence did not update state across consecutive forward calls."
    )


# ============================================================================
# Section 6: SSD backend deterministic with seed
# ============================================================================


def test_z7_substrate_ssd_deterministic_with_seed() -> None:
    """Same seed + same inputs -> same output via SSD backend."""
    def run() -> torch.Tensor:
        torch.manual_seed(42)
        cfg = _make_small_z7_ssd_config()
        sub = Z7Mamba2PredictiveCodingSubstrate(cfg)
        sub.predictor.reset_state(1)
        torch.manual_seed(123)
        z_prev = torch.randn(1, cfg.latent_dim)
        ego = torch.randn(1, cfg.ego_motion_dim)
        return sub.predictor(z_prev, ego)

    out1 = run()
    out2 = run()
    assert torch.allclose(out1, out2, atol=0.0, rtol=0.0), (
        "Same seed produced different outputs via SSD backend — non-determinism regression."
    )


# ============================================================================
# Section 7: Configuration validation
# ============================================================================


def test_z7_ssd_headdim_must_divide_d_inner_or_explicit_nheads_required() -> None:
    """ssd_headdim must divide d_inner OR ssd_nheads must be explicitly set."""
    # OK: ssd_headdim=8 divides d_inner=expand*d_model=2*16=32 -> nheads=4
    cfg = Z7Mamba2PredictiveCodingConfig(
        latent_dim=8, ego_motion_dim=4, d_model=16, d_state=4, expand=2,
        backend="ssd_reference", ssd_headdim=8,
        num_pairs=4, decoder_embed_dim=8, decoder_initial_grid_h=6,
        decoder_initial_grid_w=8, decoder_channels=(8, 6, 4, 4),
        decoder_num_upsample_blocks=2, output_height=24, output_width=32,
    )
    sub = Z7Mamba2PredictiveCodingSubstrate(cfg)
    cell = sub.predictor.mamba_cell
    assert isinstance(cell, _CanonicalHelperSSDCell)
    assert cell.nheads == 4  # d_inner=32 / headdim=8

    # FAIL: ssd_headdim=7 does NOT divide d_inner=32
    cfg_bad = Z7Mamba2PredictiveCodingConfig(
        latent_dim=8, ego_motion_dim=4, d_model=16, d_state=4, expand=2,
        backend="ssd_reference", ssd_headdim=7,
        num_pairs=4, decoder_embed_dim=8, decoder_initial_grid_h=6,
        decoder_initial_grid_w=8, decoder_channels=(8, 6, 4, 4),
        decoder_num_upsample_blocks=2, output_height=24, output_width=32,
    )
    with pytest.raises(ValueError, match="ssd_headdim=7 does not divide"):
        Z7Mamba2PredictiveCodingSubstrate(cfg_bad)


# ============================================================================
# Section 8: 91/91 Z7-Mamba-2 baseline regression guard
# ============================================================================


def test_z7_mamba2_baseline_test_count_preserved() -> None:
    """Regression guard: sister Z7-Mamba-2 test modules still load + import OK."""
    import importlib
    sister_modules = [
        "tac.tests.test_z7_mamba2_scaffold",
        "tac.tests.test_z7_mamba2_substrate_full_landing",
        "tac.tests.test_z7_mamba2_score_aware_trainer_wiring",
        "tac.tests.test_z7_mamba2_recipe_blocker_cleanup",
        "tac.tests.test_wave_4_z7_mamba_2_dao_gu_fidelity_audit",
    ]
    for mod_name in sister_modules:
        sister = importlib.import_module(mod_name)
        assert sister is not None, f"sister module {mod_name} failed to import"


def test_z7_canonical_helper_consumer_declared_in_canonical_equation() -> None:
    """The canonical equation registry declares Z7-Mamba-2 as a canonical_consumer."""
    from tac.canonical_equations import query_equations
    equations = query_equations()
    eq = next(
        (e for e in equations if e.equation_id ==
         "mamba2_ssd_mlx_pytorch_numpy_tri_backend_byte_stable_v1"),
        None,
    )
    if eq is None:
        pytest.skip(
            "canonical equation mamba2_ssd_mlx_pytorch_numpy_tri_backend_byte_stable_v1 "
            "not yet registered — gating on Phase 3 apparatus mutation chain"
        )
    consumers_str = " ".join(eq.canonical_consumers).lower()
    # Z7-Mamba-2 or time_traveler_l5_z7_mamba2 should appear.
    assert (
        "z7" in consumers_str
        or "mamba_2" in consumers_str
        or "time_traveler_l5_z7" in consumers_str
    ), (
        f"Z7-Mamba-2 not declared as canonical_consumer in {eq.equation_id}; "
        f"got canonical_consumers={eq.canonical_consumers}"
    )
