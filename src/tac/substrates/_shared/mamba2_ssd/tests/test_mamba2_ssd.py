# SPDX-License-Identifier: MIT
"""Canonical tri-backend (numpy/PyTorch/MLX) byte-stable parity tests for
:mod:`tac.substrates._shared.mamba2_ssd` per CLAUDE.md "Submission auth eval
— BOTH CPU AND CUDA" non-negotiable + Slot 1303 T3 GRAND COUNCIL MLX-vs-PyTorch
drift symposium + Catalog #1265 contest-equivalence gate sister + Catalog
#1297 Z6PCWM1 byte-stable MLX-vs-PyTorch parity sister.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" non-negotiable HIGHEST EMPHASIS:
these tests verify BEHAVIOR not constants — every test computes actual SSD
recurrence on real (small but non-trivial) input and asserts byte-stable
parity across all available backends within the canonical drift band
(max_abs < 3e-5 per Slot 1303 verdict + Slot 1255 PR95-MLX-PYTORCH-DRIFT-
MITIGATION-ENGINEERING discipline).

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + 5-axis
taxonomy: this test suite makes NO score claim; it is a behavioral
correctness anchor, not a score-axis empirical anchor.

Per Catalog #287 documented adaptation: tests use synthetic deterministic
random tensors (seeded) NOT real upstream/videos/0.mkv frames because the
test target is the SSD primitive's mathematical correctness; the scorer-
sensitivity contest-video integration is downstream (Z7/Z8 consumer scope).

Test taxonomy:
  * Section A: per-backend shape contracts + dataclass invariants
  * Section B: numpy<->PyTorch byte-stable parity (mandatory; PyTorch always
    available in this repo)
  * Section C: numpy<->MLX byte-stable parity (Darwin ARM64 only; skip on
    non-MLX hosts)
  * Section D: PyTorch<->MLX byte-stable parity (Darwin ARM64 only)
  * Section E: deterministic output (same input -> same output across N runs)
  * Section F: state externalization round-trip (init -> step -> step matches
    forward_sequence)
  * Section G: gibberish-bug regression per state-spaces/mamba issue #669
    (MLX-LM Mamba-2 correctness bottleneck on Codestral)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tac.substrates._shared.mamba2_ssd import (
    Mamba2SSDConfig,
    Mamba2SSDMLXState,
    Mamba2SSDNumpyState,
    Mamba2SSDPyTorchState,
    MLX_AVAILABLE,
    compute_mamba2_ssd_forward_sequence,
    mamba2_ssd_forward_sequence_mlx,
    mamba2_ssd_forward_sequence_numpy,
    mamba2_ssd_forward_sequence_pytorch,
    mamba2_ssd_init_state_mlx,
    mamba2_ssd_init_state_numpy,
    mamba2_ssd_init_state_pytorch,
    mamba2_ssd_step_mlx,
    mamba2_ssd_step_numpy,
    mamba2_ssd_step_pytorch,
)
from tac.framework_agnostic.backend import Backend


# Canonical drift tolerance per Slot 1303 verdict + Slot 1255 discipline.
# MLX float32 ~3e-5 vs numpy/PyTorch float32 drift is canonical; the gate
# refuses anything outside this band.
MAX_ABS_PARITY_FLOAT32 = 3e-5
MAX_ABS_PARITY_FLOAT64 = 1e-10


def _make_synthetic_inputs_numpy(
    *,
    batch: int = 2,
    seq_len: int = 8,
    nheads: int = 2,
    headdim: int = 4,
    d_state: int = 8,
    seed: int = 0xC0DE0001,
    dtype: np.dtype = np.float32,
) -> dict:
    """Build a complete deterministic synthetic input bundle (numpy)."""
    rng = np.random.default_rng(seed)
    x_seq = rng.standard_normal((batch, seq_len, nheads, headdim)).astype(dtype)
    A_log = rng.standard_normal((nheads,)).astype(dtype)
    B_seq = rng.standard_normal((batch, seq_len, nheads, d_state)).astype(dtype) * 0.1
    C_seq = rng.standard_normal((batch, seq_len, nheads, d_state)).astype(dtype) * 0.1
    # dt MUST be post-softplus (positive) per the contract.
    dt_seq = np.log(1.0 + np.exp(
        rng.standard_normal((batch, seq_len, nheads)).astype(dtype) - 4.0
    ))
    D = rng.standard_normal((nheads, headdim)).astype(dtype) * 0.05
    return {
        "x_seq": x_seq,
        "A_log": A_log,
        "B_seq": B_seq,
        "C_seq": C_seq,
        "dt_seq": dt_seq,
        "D": D,
    }


def _numpy_to_torch(bundle: dict) -> dict:
    """Convert numpy bundle to PyTorch tensors with matching dtype."""
    return {k: torch.from_numpy(v.copy()) for k, v in bundle.items()}


def _numpy_to_mlx(bundle: dict) -> dict:
    """Convert numpy bundle to MLX tensors with matching dtype."""
    if not MLX_AVAILABLE:
        raise pytest.skip.Exception("MLX not available on this host")
    import mlx.core as mx
    return {k: mx.array(v) for k, v in bundle.items()}


# =============================================================================
# Section A: per-backend shape contracts + dataclass invariants
# =============================================================================

class TestMamba2SSDConfig:
    """Canonical Mamba2SSDConfig dataclass invariants."""

    def test_default_config_values(self):
        cfg = Mamba2SSDConfig()
        assert cfg.nheads == 2
        assert cfg.headdim == 64
        assert cfg.d_state == 16
        assert cfg.with_skip_connection is True
        assert cfg.d_inner == 128  # 2 * 64

    def test_custom_config_d_inner(self):
        cfg = Mamba2SSDConfig(nheads=4, headdim=32)
        assert cfg.d_inner == 128  # 4 * 32

    def test_config_rejects_zero_nheads(self):
        with pytest.raises(ValueError, match="nheads must be > 0"):
            Mamba2SSDConfig(nheads=0)

    def test_config_rejects_negative_headdim(self):
        with pytest.raises(ValueError, match="headdim must be > 0"):
            Mamba2SSDConfig(headdim=-1)

    def test_config_rejects_zero_d_state(self):
        with pytest.raises(ValueError, match="d_state must be > 0"):
            Mamba2SSDConfig(d_state=0)


class TestNumpyStateContract:
    """Mamba2SSDNumpyState shape + dtype invariants."""

    def test_init_state_shape(self):
        state = mamba2_ssd_init_state_numpy(
            batch_size=2, nheads=2, headdim=4, d_state=8,
        )
        assert isinstance(state, Mamba2SSDNumpyState)
        assert state.h.shape == (2, 2, 4, 8)
        assert state.h.dtype == np.float32
        assert np.all(state.h == 0.0)

    def test_init_state_rejects_zero_batch(self):
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            mamba2_ssd_init_state_numpy(batch_size=0, nheads=2, headdim=4, d_state=8)

    def test_state_rejects_3d_h(self):
        with pytest.raises(ValueError, match="must have shape"):
            Mamba2SSDNumpyState(h=np.zeros((2, 2, 4), dtype=np.float32))

    def test_state_rejects_int_h(self):
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            Mamba2SSDNumpyState(h=np.zeros((2, 2, 4, 8), dtype=np.int32))


class TestPyTorchStateContract:
    """Mamba2SSDPyTorchState shape invariants."""

    def test_init_state_shape(self):
        state = mamba2_ssd_init_state_pytorch(
            batch_size=2, nheads=2, headdim=4, d_state=8,
        )
        assert isinstance(state, Mamba2SSDPyTorchState)
        assert tuple(state.h.shape) == (2, 2, 4, 8)
        assert state.h.dtype == torch.float32
        assert torch.all(state.h == 0.0)

    def test_state_rejects_non_tensor(self):
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            Mamba2SSDPyTorchState(h=np.zeros((2, 2, 4, 8)))


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available on this host")
class TestMLXStateContract:
    """Mamba2SSDMLXState shape invariants (Darwin ARM64 only)."""

    def test_init_state_shape(self):
        state = mamba2_ssd_init_state_mlx(
            batch_size=2, nheads=2, headdim=4, d_state=8,
        )
        assert isinstance(state, Mamba2SSDMLXState)
        assert state.h.shape == (2, 2, 4, 8)


# =============================================================================
# Section B: numpy<->PyTorch byte-stable parity (mandatory)
# =============================================================================

class TestNumpyPyTorchParity:
    """numpy<->PyTorch byte-stable parity per Slot 1303 drift discipline.

    PyTorch is the canonical contest-resolution framework per CLAUDE.md
    "Submission auth eval — BOTH CPU AND CUDA"; numpy is the mathematical
    truth backend. Byte-stable parity within float32 numerical tolerance
    is the canonical correctness invariant.
    """

    def test_single_step_parity(self):
        """Single SSD step numpy == PyTorch within float32 tolerance."""
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=1, nheads=2, headdim=4, d_state=8)
        # numpy
        np_state = mamba2_ssd_init_state_numpy(
            batch_size=2, nheads=2, headdim=4, d_state=8,
        )
        np_state, y_np = mamba2_ssd_step_numpy(
            state=np_state,
            x_t=bundle["x_seq"][:, 0],
            A_log=bundle["A_log"],
            B_t=bundle["B_seq"][:, 0],
            C_t=bundle["C_seq"][:, 0],
            dt_t=bundle["dt_seq"][:, 0],
            D=bundle["D"],
        )
        # PyTorch
        torch_bundle = _numpy_to_torch(bundle)
        torch_state = mamba2_ssd_init_state_pytorch(
            batch_size=2, nheads=2, headdim=4, d_state=8,
        )
        torch_state, y_torch = mamba2_ssd_step_pytorch(
            state=torch_state,
            x_t=torch_bundle["x_seq"][:, 0],
            A_log=torch_bundle["A_log"],
            B_t=torch_bundle["B_seq"][:, 0],
            C_t=torch_bundle["C_seq"][:, 0],
            dt_t=torch_bundle["dt_seq"][:, 0],
            D=torch_bundle["D"],
        )
        # Compare
        max_abs_y = np.max(np.abs(y_np - y_torch.numpy()))
        max_abs_h = np.max(np.abs(np_state.h - torch_state.h.numpy()))
        assert max_abs_y < MAX_ABS_PARITY_FLOAT32, (
            f"numpy<->PyTorch y_t drift {max_abs_y:.3e} > {MAX_ABS_PARITY_FLOAT32:.3e}"
        )
        assert max_abs_h < MAX_ABS_PARITY_FLOAT32, (
            f"numpy<->PyTorch h_t drift {max_abs_h:.3e} > {MAX_ABS_PARITY_FLOAT32:.3e}"
        )

    def test_full_sequence_parity_seq_len_8(self):
        """Length-8 sequence numpy == PyTorch within float32 tolerance."""
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=8, nheads=2, headdim=4, d_state=8)
        _, y_np = mamba2_ssd_forward_sequence_numpy(**bundle)
        torch_bundle = _numpy_to_torch(bundle)
        _, y_torch = mamba2_ssd_forward_sequence_pytorch(**torch_bundle)
        max_abs = np.max(np.abs(y_np - y_torch.numpy()))
        assert max_abs < MAX_ABS_PARITY_FLOAT32, (
            f"L=8 numpy<->PyTorch drift {max_abs:.3e} > {MAX_ABS_PARITY_FLOAT32:.3e}"
        )

    def test_full_sequence_parity_contest_scale_seq_len_600(self):
        """Contest-scale L=600 sequence numpy == PyTorch within float32 tolerance.

        Sister of Z7-Mamba-2 substrate scale per parent design memo §7.
        """
        bundle = _make_synthetic_inputs_numpy(
            batch=1, seq_len=600, nheads=2, headdim=4, d_state=8,
        )
        _, y_np = mamba2_ssd_forward_sequence_numpy(**bundle)
        torch_bundle = _numpy_to_torch(bundle)
        _, y_torch = mamba2_ssd_forward_sequence_pytorch(**torch_bundle)
        max_abs = np.max(np.abs(y_np - y_torch.numpy()))
        # L=600 may accumulate drift via the recurrence; use slightly relaxed
        # tolerance per Slot 1303 verdict (drift scales with sequence length).
        relaxed_tol = MAX_ABS_PARITY_FLOAT32 * 5.0  # 1.5e-4 for L=600
        assert max_abs < relaxed_tol, (
            f"L=600 numpy<->PyTorch drift {max_abs:.3e} > {relaxed_tol:.3e}"
        )

    def test_parity_without_skip_connection(self):
        """Parity holds when D=None (no skip connection)."""
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=8)
        bundle["D"] = None
        _, y_np = mamba2_ssd_forward_sequence_numpy(**bundle)
        torch_bundle = _numpy_to_torch({k: v for k, v in bundle.items() if v is not None})
        torch_bundle["D"] = None
        _, y_torch = mamba2_ssd_forward_sequence_pytorch(**torch_bundle)
        max_abs = np.max(np.abs(y_np - y_torch.numpy()))
        assert max_abs < MAX_ABS_PARITY_FLOAT32

    def test_pytorch_gradient_flows_through_recurrence(self):
        """PyTorch backend preserves gradient flow through recurrence.

        Critical for training-time use per CLAUDE.md "Phase 1 score-aware
        loss requires differentiable scorer preprocess" + HNeRV parity L8.
        """
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=4)
        torch_bundle = _numpy_to_torch(bundle)
        # Mark inputs as requires_grad
        for k in ("x_seq", "A_log", "B_seq", "C_seq", "dt_seq"):
            torch_bundle[k] = torch_bundle[k].clone().requires_grad_(True)
        _, y_torch = mamba2_ssd_forward_sequence_pytorch(**torch_bundle)
        loss = y_torch.sum()
        loss.backward()
        # Each requires_grad input should have a non-None gradient
        for k in ("x_seq", "A_log", "B_seq", "C_seq", "dt_seq"):
            assert torch_bundle[k].grad is not None, f"{k}.grad is None"
            assert not torch.all(torch_bundle[k].grad == 0.0), (
                f"{k}.grad is all zeros (gradient broken)"
            )


# =============================================================================
# Section C: numpy<->MLX byte-stable parity (Darwin ARM64 only)
# =============================================================================

@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available on this host")
class TestNumpyMLXParity:
    """numpy<->MLX byte-stable parity per Slot 1303 drift discipline.

    Per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th
    standing directive: MLX is the canonical M5 Max training surface.
    Byte-stable parity vs numpy reference is the canonical correctness
    invariant for MLX-LOCAL fast candidate generation.
    """

    def test_single_step_parity(self):
        """Single SSD step numpy == MLX within float32 tolerance."""
        import mlx.core as mx
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=1, nheads=2, headdim=4, d_state=8)
        np_state = mamba2_ssd_init_state_numpy(
            batch_size=2, nheads=2, headdim=4, d_state=8,
        )
        np_state, y_np = mamba2_ssd_step_numpy(
            state=np_state,
            x_t=bundle["x_seq"][:, 0],
            A_log=bundle["A_log"],
            B_t=bundle["B_seq"][:, 0],
            C_t=bundle["C_seq"][:, 0],
            dt_t=bundle["dt_seq"][:, 0],
            D=bundle["D"],
        )
        mlx_bundle = _numpy_to_mlx(bundle)
        mlx_state = mamba2_ssd_init_state_mlx(
            batch_size=2, nheads=2, headdim=4, d_state=8,
        )
        mlx_state, y_mlx = mamba2_ssd_step_mlx(
            state=mlx_state,
            x_t=mlx_bundle["x_seq"][:, 0],
            A_log=mlx_bundle["A_log"],
            B_t=mlx_bundle["B_seq"][:, 0],
            C_t=mlx_bundle["C_seq"][:, 0],
            dt_t=mlx_bundle["dt_seq"][:, 0],
            D=mlx_bundle["D"],
        )
        mx.eval(y_mlx, mlx_state.h)
        max_abs_y = np.max(np.abs(y_np - np.asarray(y_mlx)))
        max_abs_h = np.max(np.abs(np_state.h - np.asarray(mlx_state.h)))
        assert max_abs_y < MAX_ABS_PARITY_FLOAT32, (
            f"numpy<->MLX y_t drift {max_abs_y:.3e} > {MAX_ABS_PARITY_FLOAT32:.3e}"
        )
        assert max_abs_h < MAX_ABS_PARITY_FLOAT32, (
            f"numpy<->MLX h_t drift {max_abs_h:.3e} > {MAX_ABS_PARITY_FLOAT32:.3e}"
        )

    def test_full_sequence_parity_seq_len_8(self):
        """Length-8 sequence numpy == MLX within float32 tolerance."""
        import mlx.core as mx
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=8)
        _, y_np = mamba2_ssd_forward_sequence_numpy(**bundle)
        mlx_bundle = _numpy_to_mlx(bundle)
        _, y_mlx = mamba2_ssd_forward_sequence_mlx(**mlx_bundle)
        mx.eval(y_mlx)
        max_abs = np.max(np.abs(y_np - np.asarray(y_mlx)))
        assert max_abs < MAX_ABS_PARITY_FLOAT32, (
            f"L=8 numpy<->MLX drift {max_abs:.3e} > {MAX_ABS_PARITY_FLOAT32:.3e}"
        )

    def test_full_sequence_parity_contest_scale_seq_len_600(self):
        """Contest-scale L=600 sequence numpy == MLX within drift band.

        Per Slot 1303 verdict: drift accumulates with sequence length; the
        L=600 band is relaxed to 5x base tolerance per the canonical drift
        discipline.
        """
        import mlx.core as mx
        bundle = _make_synthetic_inputs_numpy(
            batch=1, seq_len=600, nheads=2, headdim=4, d_state=8,
        )
        _, y_np = mamba2_ssd_forward_sequence_numpy(**bundle)
        mlx_bundle = _numpy_to_mlx(bundle)
        _, y_mlx = mamba2_ssd_forward_sequence_mlx(**mlx_bundle)
        mx.eval(y_mlx)
        max_abs = np.max(np.abs(y_np - np.asarray(y_mlx)))
        relaxed_tol = MAX_ABS_PARITY_FLOAT32 * 5.0
        assert max_abs < relaxed_tol, (
            f"L=600 numpy<->MLX drift {max_abs:.3e} > {relaxed_tol:.3e}"
        )


# =============================================================================
# Section D: PyTorch<->MLX byte-stable parity (Darwin ARM64 only)
# =============================================================================

@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available on this host")
class TestPyTorchMLXParity:
    """PyTorch<->MLX byte-stable parity sister of Catalog #1297 Z6PCWM1."""

    def test_full_sequence_parity_seq_len_8(self):
        import mlx.core as mx
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=8)
        torch_bundle = _numpy_to_torch(bundle)
        _, y_torch = mamba2_ssd_forward_sequence_pytorch(**torch_bundle)
        mlx_bundle = _numpy_to_mlx(bundle)
        _, y_mlx = mamba2_ssd_forward_sequence_mlx(**mlx_bundle)
        mx.eval(y_mlx)
        max_abs = np.max(np.abs(y_torch.numpy() - np.asarray(y_mlx)))
        # MLX<->PyTorch may have slightly higher drift than numpy reference
        # (both are float32 with separate kernel implementations); use 2x
        # baseline tolerance per Slot 1303 PR95-MLX-PYTORCH-DRIFT discipline.
        relaxed_tol = MAX_ABS_PARITY_FLOAT32 * 2.0
        assert max_abs < relaxed_tol, (
            f"PyTorch<->MLX drift {max_abs:.3e} > {relaxed_tol:.3e}"
        )


# =============================================================================
# Section E: deterministic output (same input -> same output across N runs)
# =============================================================================

class TestDeterminism:
    """Same input -> same output across 10 runs per backend.

    Per CLAUDE.md "Deterministic packet compiler" non-negotiable: the
    canonical helper MUST produce deterministic output for byte-stable
    contest compliance. fp32 deterministic across backends per Slot 1303
    + CLAUDE.md "MLX portable-local-substrate authority".
    """

    def test_numpy_deterministic_10_runs(self):
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=16)
        _, y_first = mamba2_ssd_forward_sequence_numpy(**bundle)
        for _ in range(9):
            _, y_next = mamba2_ssd_forward_sequence_numpy(**bundle)
            assert np.array_equal(y_first, y_next), (
                "numpy backend non-deterministic across 10 runs"
            )

    def test_pytorch_deterministic_10_runs(self):
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=16)
        torch_bundle = _numpy_to_torch(bundle)
        _, y_first = mamba2_ssd_forward_sequence_pytorch(**torch_bundle)
        for _ in range(9):
            _, y_next = mamba2_ssd_forward_sequence_pytorch(**torch_bundle)
            assert torch.equal(y_first, y_next), (
                "PyTorch backend non-deterministic across 10 runs"
            )

    @pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available on this host")
    def test_mlx_deterministic_10_runs(self):
        import mlx.core as mx
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=16)
        mlx_bundle = _numpy_to_mlx(bundle)
        _, y_first = mamba2_ssd_forward_sequence_mlx(**mlx_bundle)
        mx.eval(y_first)
        y_first_np = np.asarray(y_first)
        for _ in range(9):
            _, y_next = mamba2_ssd_forward_sequence_mlx(**mlx_bundle)
            mx.eval(y_next)
            assert np.array_equal(y_first_np, np.asarray(y_next)), (
                "MLX backend non-deterministic across 10 runs"
            )


# =============================================================================
# Section F: state externalization round-trip
# =============================================================================

class TestStateExternalization:
    """Externalized state: init -> step -> step matches forward_sequence.

    Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
    L4 inflate runtime budget: externalized state must permit checkpointable
    recurrence for Z8 binding-contract per parent design memo.
    """

    def test_numpy_externalized_state_matches_forward(self):
        """Stepping externally L times == forward_sequence over L."""
        L = 5
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=L)
        # Path 1: forward_sequence
        _, y_full = mamba2_ssd_forward_sequence_numpy(**bundle)
        # Path 2: externalized step loop
        state = mamba2_ssd_init_state_numpy(
            batch_size=2, nheads=2, headdim=4, d_state=8,
        )
        y_external = np.zeros_like(y_full)
        for t in range(L):
            state, y_t = mamba2_ssd_step_numpy(
                state=state,
                x_t=bundle["x_seq"][:, t],
                A_log=bundle["A_log"],
                B_t=bundle["B_seq"][:, t],
                C_t=bundle["C_seq"][:, t],
                dt_t=bundle["dt_seq"][:, t],
                D=bundle["D"],
            )
            y_external[:, t] = y_t
        assert np.array_equal(y_full, y_external), (
            "Externalized step loop != forward_sequence (numpy)"
        )

    def test_pytorch_externalized_state_matches_forward(self):
        L = 5
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=L)
        torch_bundle = _numpy_to_torch(bundle)
        _, y_full = mamba2_ssd_forward_sequence_pytorch(**torch_bundle)
        state = mamba2_ssd_init_state_pytorch(
            batch_size=2, nheads=2, headdim=4, d_state=8,
        )
        y_external_list = []
        for t in range(L):
            state, y_t = mamba2_ssd_step_pytorch(
                state=state,
                x_t=torch_bundle["x_seq"][:, t],
                A_log=torch_bundle["A_log"],
                B_t=torch_bundle["B_seq"][:, t],
                C_t=torch_bundle["C_seq"][:, t],
                dt_t=torch_bundle["dt_seq"][:, t],
                D=torch_bundle["D"],
            )
            y_external_list.append(y_t)
        y_external = torch.stack(y_external_list, dim=1)
        assert torch.equal(y_full, y_external), (
            "Externalized step loop != forward_sequence (PyTorch)"
        )


# =============================================================================
# Section G: gibberish-bug regression per state-spaces/mamba issue #669
# =============================================================================

class TestGibberishBugRegression:
    """Regression test for state-spaces/mamba #669 MLX-LM Mamba-2
    correctness bottleneck on Codestral (gibberish output).

    The known external bug is that MLX-LM's Mamba-2 implementation produces
    gibberish on the Codestral language model. Our impl is contest scope,
    NOT MLX-LM scope, so we cannot directly reproduce; instead we test
    structural invariants that the gibberish bug would violate:

      1. Output magnitudes are bounded (no explosion)
      2. Output is not all-zero (no collapse)
      3. Output varies across timesteps (no constant output)
      4. Different inputs produce different outputs (function not constant)
    """

    def test_output_magnitudes_bounded(self):
        """No NaN/inf in output across 100-step sequence."""
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=100)
        _, y = mamba2_ssd_forward_sequence_numpy(**bundle)
        assert np.all(np.isfinite(y)), "Output contains NaN/inf (gibberish symptom)"
        max_abs = np.max(np.abs(y))
        assert max_abs < 1e3, f"Output magnitude {max_abs:.3e} exploded (gibberish symptom)"

    def test_output_not_all_zero(self):
        """Non-trivial input produces non-zero output (no collapse)."""
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=8)
        _, y = mamba2_ssd_forward_sequence_numpy(**bundle)
        assert not np.all(y == 0.0), "Output is all-zero (collapse / gibberish symptom)"

    def test_output_varies_across_timesteps(self):
        """y[t] != y[t+1] for non-stationary input (function is not constant in t)."""
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=8)
        _, y = mamba2_ssd_forward_sequence_numpy(**bundle)
        # Compare consecutive timesteps; at least one pair must differ.
        diffs = [np.max(np.abs(y[:, t] - y[:, t + 1])) for t in range(7)]
        assert max(diffs) > 1e-6, (
            "Output constant across timesteps (function-is-constant gibberish symptom)"
        )

    def test_different_inputs_produce_different_outputs(self):
        """Function is not constant — different x_seq produces different y_seq."""
        bundle_a = _make_synthetic_inputs_numpy(batch=1, seq_len=8, seed=0xC0DE0001)
        bundle_b = _make_synthetic_inputs_numpy(batch=1, seq_len=8, seed=0xC0DE0002)
        # Force same A_log, B_seq, C_seq, dt_seq, D — only x differs.
        bundle_b["A_log"] = bundle_a["A_log"]
        bundle_b["B_seq"] = bundle_a["B_seq"]
        bundle_b["C_seq"] = bundle_a["C_seq"]
        bundle_b["dt_seq"] = bundle_a["dt_seq"]
        bundle_b["D"] = bundle_a["D"]
        _, y_a = mamba2_ssd_forward_sequence_numpy(**bundle_a)
        _, y_b = mamba2_ssd_forward_sequence_numpy(**bundle_b)
        diff = np.max(np.abs(y_a - y_b))
        assert diff > 1e-3, (
            f"Different x_seq produced identical y_seq (diff={diff:.3e}; "
            "function-is-constant gibberish symptom)"
        )


# =============================================================================
# Section H: tri-backend dispatch
# =============================================================================

class TestTriBackendDispatch:
    """compute_mamba2_ssd_forward_sequence dispatch via Backend enum."""

    def test_explicit_numpy_backend(self):
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=4)
        _, y = compute_mamba2_ssd_forward_sequence(**bundle, backend=Backend.NUMPY)
        assert isinstance(y, np.ndarray)
        assert y.shape == (2, 4, 2, 4)

    def test_explicit_pytorch_backend(self):
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=4)
        torch_bundle = _numpy_to_torch(bundle)
        _, y = compute_mamba2_ssd_forward_sequence(**torch_bundle, backend=Backend.PYTORCH)
        assert isinstance(y, torch.Tensor)
        assert tuple(y.shape) == (2, 4, 2, 4)

    @pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available on this host")
    def test_explicit_mlx_backend(self):
        import mlx.core as mx
        bundle = _make_synthetic_inputs_numpy(batch=2, seq_len=4)
        mlx_bundle = _numpy_to_mlx(bundle)
        _, y = compute_mamba2_ssd_forward_sequence(**mlx_bundle, backend=Backend.MLX)
        mx.eval(y)
        assert y.shape == (2, 4, 2, 4)
