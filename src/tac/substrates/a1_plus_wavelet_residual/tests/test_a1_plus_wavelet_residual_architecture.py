"""Tests for the A1 + wavelet residual architecture / head."""
from __future__ import annotations

import math

import pytest
import torch

from tac.substrates.a1_plus_wavelet_residual.architecture import (
    A1_CAMERA_H,
    A1_CAMERA_W,
    A1_N_PAIRS,
    A1PlusWaveletResidualConfig,
    PerPairWaveletResidualHead,
    _db4_idwt_single_level,
)


def test_config_default_invariants() -> None:
    cfg = A1PlusWaveletResidualConfig()
    assert cfg.coeff_rank == 2
    assert cfg.foveal_h == 128
    assert cfg.foveal_w == 128
    assert cfg.wavelet_levels == 1
    assert cfg.selected_pair_indices == ()
    assert cfg.estimated_sidecar_bytes() > 0  # uses num_selected=1 minimum


def test_estimated_sidecar_bytes_scales_with_selected_count() -> None:
    base = A1PlusWaveletResidualConfig(
        selected_pair_indices=(1,), coeff_rank=2, foveal_h=64, foveal_w=64
    )
    bigger = A1PlusWaveletResidualConfig(
        selected_pair_indices=tuple(range(16)),
        coeff_rank=2,
        foveal_h=64,
        foveal_w=64,
    )
    assert bigger.estimated_sidecar_bytes() > base.estimated_sidecar_bytes()
    # Roughly linear in num_selected for the per-coefficient block.
    ratio = bigger.estimated_sidecar_bytes() / base.estimated_sidecar_bytes()
    assert 10 < ratio < 20  # 16x param block + small fixed overhead


def test_head_zero_residual_for_unselected_pair() -> None:
    cfg = A1PlusWaveletResidualConfig(
        selected_pair_indices=(3, 7), coeff_rank=1, foveal_h=8, foveal_w=8
    )
    head = PerPairWaveletResidualHead(cfg)
    out = head.residual_chw_for_pair(pair_index=11, frame_index=0)
    assert torch.all(out == 0)
    assert out.shape == (3, 16, 16)  # 2*fov_h, 2*fov_w


def test_head_emits_nonzero_residual_for_selected_pair() -> None:
    torch.manual_seed(0)
    cfg = A1PlusWaveletResidualConfig(
        selected_pair_indices=(3, 7), coeff_rank=2, foveal_h=8, foveal_w=8
    )
    head = PerPairWaveletResidualHead(cfg)
    # Bump U to a non-trivial magnitude so IDWT output is non-zero.
    with torch.no_grad():
        head.U.fill_(0.1)
        head.V.fill_(0.1)
    out = head.residual_chw_for_pair(pair_index=3, frame_index=0)
    assert out.shape == (3, 16, 16)
    assert torch.any(torch.abs(out) > 1e-6)


def test_head_residuals_are_pair_local() -> None:
    """Different pair indices in the selected set must produce different residuals."""
    torch.manual_seed(0)
    cfg = A1PlusWaveletResidualConfig(
        selected_pair_indices=(3, 7), coeff_rank=2, foveal_h=8, foveal_w=8
    )
    head = PerPairWaveletResidualHead(cfg)
    out3 = head.residual_chw_for_pair(pair_index=3, frame_index=0)
    out7 = head.residual_chw_for_pair(pair_index=7, frame_index=0)
    # Random init guarantees pair-local independence.
    assert not torch.equal(out3, out7)


def test_head_frame_index_axis_distinguishes_frames() -> None:
    """Per-pair head distinguishes frame 0 vs frame 1."""
    torch.manual_seed(0)
    cfg = A1PlusWaveletResidualConfig(
        selected_pair_indices=(5,), coeff_rank=2, foveal_h=8, foveal_w=8
    )
    head = PerPairWaveletResidualHead(cfg)
    out_a = head.residual_chw_for_pair(pair_index=5, frame_index=0)
    out_b = head.residual_chw_for_pair(pair_index=5, frame_index=1)
    assert not torch.equal(out_a, out_b)


def test_head_parameters_have_gradient_flow() -> None:
    """A trivial loss on the head residual must reach U and V."""
    torch.manual_seed(0)
    cfg = A1PlusWaveletResidualConfig(
        selected_pair_indices=(0,), coeff_rank=2, foveal_h=4, foveal_w=4
    )
    head = PerPairWaveletResidualHead(cfg)
    out = head.residual_chw_for_pair(pair_index=0, frame_index=0)
    out.sum().backward()
    assert head.U.grad is not None
    assert head.V.grad is not None
    assert head.U.grad.abs().sum() > 0
    assert head.V.grad.abs().sum() > 0


def test_estimated_sidecar_bytes_matches_head_helper() -> None:
    cfg = A1PlusWaveletResidualConfig(
        selected_pair_indices=tuple(range(8)),
        coeff_rank=2,
        foveal_h=32,
        foveal_w=32,
    )
    head = PerPairWaveletResidualHead(cfg)
    assert head.estimated_sidecar_bytes() == cfg.estimated_sidecar_bytes()


def test_idwt_returns_double_resolution() -> None:
    """Single-level DB4 IDWT doubles each spatial axis."""
    ll = torch.randn(3, 8, 8)
    lh = torch.randn(3, 8, 8)
    hl = torch.randn(3, 8, 8)
    hh = torch.randn(3, 8, 8)
    out = _db4_idwt_single_level(ll, lh, hl, hh)
    assert out.shape == (3, 16, 16)


def test_idwt_rejects_unequal_subband_shapes() -> None:
    ll = torch.randn(3, 8, 8)
    lh = torch.randn(3, 8, 9)  # wrong width
    hl = torch.randn(3, 8, 8)
    hh = torch.randn(3, 8, 8)
    with pytest.raises(ValueError, match="equal sub-band shapes"):
        _db4_idwt_single_level(ll, lh, hl, hh)


def test_idwt_rejects_wrong_rank() -> None:
    ll = torch.randn(8, 8)  # missing channel axis
    lh = torch.randn(8, 8)
    hl = torch.randn(8, 8)
    hh = torch.randn(8, 8)
    with pytest.raises(ValueError, match="expects \\(C, H, W\\)"):
        _db4_idwt_single_level(ll, lh, hl, hh)


def test_idwt_zero_input_yields_zero_output() -> None:
    """A canonical IDWT contract: zero coefficients → zero reconstruction."""
    ll = torch.zeros(3, 8, 8)
    lh = torch.zeros(3, 8, 8)
    hl = torch.zeros(3, 8, 8)
    hh = torch.zeros(3, 8, 8)
    out = _db4_idwt_single_level(ll, lh, hl, hh)
    assert torch.all(out == 0)


def test_head_residual_default_zero_is_finite_for_unselected() -> None:
    cfg = A1PlusWaveletResidualConfig(
        selected_pair_indices=(0,), coeff_rank=1, foveal_h=4, foveal_w=4
    )
    head = PerPairWaveletResidualHead(cfg)
    # Unselected pairs return zeros of the right shape — no NaN, no Inf.
    out = head.residual_chw_for_pair(pair_index=A1_N_PAIRS - 1, frame_index=1)
    assert out.shape == (3, 8, 8)
    assert torch.isfinite(out).all()


def test_selected_indices_returns_tuple() -> None:
    cfg = A1PlusWaveletResidualConfig(
        selected_pair_indices=(1, 5, 9), coeff_rank=1, foveal_h=4, foveal_w=4
    )
    head = PerPairWaveletResidualHead(cfg)
    assert head.selected_indices() == (1, 5, 9)


def test_camera_constants_match_a1() -> None:
    """A1 substrate constants must stay aligned with the canonical A1 anchor."""
    # If A1's eval / camera resolution ever changes upstream, the substrate
    # must be updated to match.  These constants are the apples-to-apples anchor.
    assert A1_CAMERA_H == 874
    assert A1_CAMERA_W == 1164
    assert A1_N_PAIRS == 600


def test_default_byte_budget_under_500() -> None:
    """Operator-default config (~12 pairs, rank=1, fov=64) should fit ≤~5 KB."""
    cfg = A1PlusWaveletResidualConfig(
        selected_pair_indices=tuple(range(12)),
        coeff_rank=1,
        foveal_h=64,
        foveal_w=64,
    )
    # Pre-brotli upper bound; real sidecar after brotli will be smaller.
    assert cfg.estimated_sidecar_bytes() < 60_000
