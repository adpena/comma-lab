# SPDX-License-Identifier: MIT
"""Property tests for dct_quant_loss (Fridrich council #1, 2026-04-26).

The DCT-quant loss is a JPEG-Q-table-weighted DCT-domain residual loss. It
penalises low-frequency residual energy ~6× more than high-frequency residual
energy, encouraging the renderer to hide its inevitable approximation error
in DCT directions that CNN scorers cannot detect (UNIWARD analog).

These tests verify:
  - Mathematical correctness (zero residual, DC-only, HF-only, orthonormality)
  - Shape contract and gradient flow
  - Linear scaling with weight argument
  - eval_roundtrip compatibility
  - Input validation
"""
from __future__ import annotations


import pytest
import torch

from tac.fridrich_losses import (
    JPEG_LUMA_Q_TABLE,
    _build_dct8_matrix,
    dct_quant_loss,
)


# ---- Helpers ---- #

def _hwc_pair(h: int = 64, w: int = 64) -> torch.Tensor:
    """Create a (1, 2, H, W, 3) HWC RGB pair of random pixel values [0, 255]."""
    return torch.rand(1, 2, h, w, 3) * 255.0


def _const_pair(value: float, h: int = 64, w: int = 64) -> torch.Tensor:
    """Create a (1, 2, H, W, 3) HWC RGB pair of constant value."""
    return torch.full((1, 2, h, w, 3), value, dtype=torch.float32)


# ---- Mathematical correctness ---- #

class TestZeroResidual:
    """Identical rendered and gt -> zero loss."""

    def test_random_identical(self):
        gt = _hwc_pair()
        loss = dct_quant_loss(gt.clone(), gt)
        assert loss.item() == pytest.approx(0.0, abs=1e-12)

    def test_constant_identical(self):
        gt = _const_pair(127.5)
        loss = dct_quant_loss(gt.clone(), gt)
        assert loss.item() == pytest.approx(0.0, abs=1e-12)


class TestDcOnly:
    """Constant-offset residual lives entirely in the (0,0) DCT bin."""

    def test_dc_offset_matches_predicted_energy(self):
        """A constant offset c in luma -> coeffs = c·N (DC bin only).

        For an orthonormal 8x8 DCT-II of a constant block X[i,j]=c, the
        only nonzero coefficient is Y[0,0] = c · 8 (since the DC basis
        vector is 1/√8 along each axis -> 1/8 over the block, and the
        block sums to 64c).

        Loss = mean over all coeffs (across all blocks) of (Y[i,j] / Q[i,j])².
        Per block, only Y[0,0] is nonzero -> per-block sum = (8c/16)² = (c/2)².
        Per block has 64 coefficients -> mean per block = (c/2)² / 64.
        Mean over all blocks = same value (all blocks identical).
        """
        h, w = 16, 16
        # Use small constant so float math is exact; channel_mode='luma'
        # so c=10 in all RGB channels gives luma c=10 too.
        c = 10.0
        rendered = _const_pair(c, h, w)
        gt = _const_pair(0.0, h, w)
        loss = dct_quant_loss(rendered, gt, channel_mode="luma")

        expected = (c / 2.0) ** 2 / 64.0
        assert loss.item() == pytest.approx(expected, rel=1e-5)

    def test_dc_weight_one_over_sixteen(self):
        """Verify the (0,0) bin weight in the JPEG luma Q-table is 1/16."""
        assert JPEG_LUMA_Q_TABLE[0][0] == 16


class TestHighFreqDominated:
    """A pure high-frequency residual produces much smaller loss than a
    pure low-frequency one of equal energy — that's the whole point.
    """

    def test_hf_loss_smaller_than_lf_loss(self):
        h, w = 16, 16
        gt = _const_pair(0.0, h, w)

        # Low-freq residual: a slow horizontal ramp (mostly DC + low AC).
        x = torch.linspace(-10.0, 10.0, w).view(1, 1, 1, w, 1).expand(1, 2, h, w, 3)
        rendered_lf = x.contiguous()

        # High-freq residual: alternating ±10 chess pattern at the FINEST
        # 8x8-block scale (i.e. flips every pixel — lives in (7,7) bin).
        ii = torch.arange(h).view(1, 1, h, 1, 1)
        jj = torch.arange(w).view(1, 1, 1, w, 1)
        chess = ((ii + jj) % 2 * 2 - 1).float() * 10.0
        rendered_hf = chess.expand(1, 2, h, w, 3).contiguous()

        loss_lf = dct_quant_loss(rendered_lf, gt).item()
        loss_hf = dct_quant_loss(rendered_hf, gt).item()

        # The HF residual has the SAME pixel-energy variance as the LF one
        # (both ±10), but should be penalised much less because (7,7) bin
        # weight is 1/99 vs DC bin 1/16. Ratio ≈ (16/99)² ≈ 0.026.
        assert loss_hf < loss_lf * 0.1, (
            f"HF loss {loss_hf:.6f} should be << LF loss {loss_lf:.6f}"
        )

    def test_hf_pattern_dominates_bin_77(self):
        """The chess pattern at finest scale concentrates DCT energy in the
        ODD AC bins (the {1,3,5,7} × {1,3,5,7} subgrid), peaking at (7,7).

        Note: ±1 sign-checker is NOT the (7,7) basis — that would require a
        real-cosine eigenfunction, not sign alternation. But the chess
        pattern is symmetric in i+j parity so all even-index DCT coefficients
        are zero, and the (7,7) bin carries the largest single coefficient
        (~6.57 vs next-largest 2.31 at (5,7)/(7,5)).

        The verification: this loss must match the result of explicitly
        applying the DCT and Q-table multiplication to the residual block.
        """
        h, w = 8, 8
        gt = _const_pair(0.0, h, w)
        ii_p = torch.arange(h).view(1, 1, h, 1, 1)
        jj_p = torch.arange(w).view(1, 1, 1, w, 1)
        chess = ((ii_p + jj_p) % 2 * 2 - 1).float()  # ±1
        rendered = chess.expand(1, 2, h, w, 3).contiguous()
        loss = dct_quant_loss(rendered, gt).item()

        # Reference: explicit 2D DCT of the residual block, weighted by 1/Q,
        # squared and averaged over the 64 coefficients (one block here).
        from tac.fridrich_losses import _build_dct8_matrix
        M = _build_dct8_matrix(dtype=torch.float64)
        residual = chess.squeeze().double()  # (8, 8) luma residual
        # luma = 0.299+0.587+0.114 = 1.0 -> chess unchanged
        Y = M @ residual @ M.t()
        Q = torch.tensor(JPEG_LUMA_Q_TABLE, dtype=torch.float64)
        weighted = Y / Q
        expected = (weighted * weighted).mean().item()
        assert loss == pytest.approx(expected, rel=1e-5), (
            f"loss {loss:.10f} != expected {expected:.10f}"
        )

        # And confirm the (7,7) bin is the largest contributor — the
        # original purpose of this test (HF energy goes to the corner the
        # Q-table down-weights most).
        bin_contributions = (Y / Q).pow(2)
        assert bin_contributions[7, 7].item() == bin_contributions.max().item()


# ---- Orthonormality ---- #

class TestDctMatrixOrthonormal:
    """The 8×8 DCT-II matrix M must be orthonormal: M @ Mᵀ = I."""

    def test_orthonormal_float32(self):
        M = _build_dct8_matrix(dtype=torch.float32)
        identity = torch.eye(8, dtype=torch.float32)
        torch.testing.assert_close(M @ M.t(), identity, atol=1e-5, rtol=1e-5)

    def test_orthonormal_float64(self):
        M = _build_dct8_matrix(dtype=torch.float64)
        identity = torch.eye(8, dtype=torch.float64)
        torch.testing.assert_close(M @ M.t(), identity, atol=1e-12, rtol=1e-12)

    def test_dct_inverse_roundtrip(self):
        """Round-trip a random 8x8 block through DCT then iDCT."""
        torch.manual_seed(0)
        M = _build_dct8_matrix(dtype=torch.float64)
        x = torch.randn(8, 8, dtype=torch.float64) * 100.0
        y = M @ x @ M.t()             # forward 2D DCT
        x_recon = M.t() @ y @ M       # inverse 2D DCT
        torch.testing.assert_close(x, x_recon, atol=1e-10, rtol=1e-10)


# ---- Shape contract and gradient flow ---- #

class TestShapeContract:
    """Eval-resolution input -> scalar loss; grad flows back."""

    def test_scorer_resolution_returns_scalar(self):
        """Input (1, 2, 384, 512, 3) -> scalar loss."""
        torch.manual_seed(1)
        rendered = torch.rand(1, 2, 384, 512, 3) * 255.0
        gt = torch.rand(1, 2, 384, 512, 3) * 255.0
        loss = dct_quant_loss(rendered, gt)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)

    def test_gradient_flows_back(self):
        torch.manual_seed(2)
        rendered = (torch.rand(1, 2, 64, 96, 3) * 255.0).requires_grad_(True)
        gt = torch.rand(1, 2, 64, 96, 3) * 255.0
        loss = dct_quant_loss(rendered, gt)
        loss.backward()
        assert rendered.grad is not None
        assert rendered.grad.shape == rendered.shape
        # Gradient must be nonzero somewhere — it's a quadratic in the
        # residual, so unless the residual is exactly zero (it isn't) the
        # gradient is nonzero.
        assert rendered.grad.abs().max().item() > 0

    def test_non_multiple_of_8_dimensions_padded(self):
        """H, W not multiples of 8 should be replicate-padded automatically."""
        rendered = torch.rand(1, 2, 100, 130, 3) * 255.0
        gt = torch.rand(1, 2, 100, 130, 3) * 255.0
        loss = dct_quant_loss(rendered, gt)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)
        assert loss.item() > 0  # nonzero residual, nonzero loss

    def test_batch_size_greater_than_1(self):
        rendered = torch.rand(3, 2, 32, 32, 3) * 255.0
        gt = torch.rand(3, 2, 32, 32, 3) * 255.0
        loss = dct_quant_loss(rendered, gt)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)


class TestRgbMode:
    """channel_mode='rgb' should sum residual over all 3 channels."""

    def test_rgb_mode_returns_scalar(self):
        rendered = torch.rand(1, 2, 32, 32, 3) * 255.0
        gt = torch.rand(1, 2, 32, 32, 3) * 255.0
        loss = dct_quant_loss(rendered, gt, channel_mode="rgb")
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)

    def test_rgb_mode_zero_residual(self):
        gt = _hwc_pair(32, 32)
        loss = dct_quant_loss(gt.clone(), gt, channel_mode="rgb")
        assert loss.item() == pytest.approx(0.0, abs=1e-12)


# ---- Weight scaling ---- #

class TestWeightScaling:
    """weight=k -> exactly k× the weight=1.0 loss."""

    def test_weight_2x(self):
        torch.manual_seed(3)
        rendered = torch.rand(1, 2, 32, 32, 3) * 255.0
        gt = torch.rand(1, 2, 32, 32, 3) * 255.0
        loss_1 = dct_quant_loss(rendered, gt, weight=1.0).item()
        loss_2 = dct_quant_loss(rendered, gt, weight=2.0).item()
        assert loss_2 == pytest.approx(2.0 * loss_1, rel=1e-6)

    def test_weight_zero(self):
        rendered = torch.rand(1, 2, 32, 32, 3) * 255.0
        gt = torch.rand(1, 2, 32, 32, 3) * 255.0
        loss = dct_quant_loss(rendered, gt, weight=0.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-12)


# ---- eval_roundtrip safety ---- #

class TestEvalRoundtripSafety:
    """The loss must work after the rendered tensor has been through
    simulate_eval_roundtrip (the H=384->874->384 resize chain).
    """

    def test_after_simulate_eval_roundtrip(self):
        from tac.renderer import simulate_eval_roundtrip

        torch.manual_seed(4)
        # Renderer outputs at scorer resolution (384, 512); roundtrip is
        # 384 -> 874 -> 384 in the contest evaluator. Track gradients on the
        # rendered tensor end-to-end (matches training-loop behaviour where
        # the renderer output is autograd-tracked).
        rendered_chw = (torch.rand(2, 3, 384, 512) * 255.0).requires_grad_(True)
        gt_chw = torch.rand(2, 3, 384, 512) * 255.0
        # eval_roundtrip applies upscale -> uint8 STE -> downscale (and
        # optional noise injection — only when the input requires_grad).
        rendered_rt_chw = simulate_eval_roundtrip(
            rendered_chw, target_h=874, target_w=1164, noise_std=0.5,
        )
        gt_rt_chw = simulate_eval_roundtrip(
            gt_chw, target_h=874, target_w=1164, noise_std=0.0,
        )

        # Re-pack to (B=1, T=2, H, W, 3) HWC pair format.
        rendered_pair = rendered_rt_chw.permute(0, 2, 3, 1).reshape(
            1, 2, 384, 512, 3,
        )
        gt_pair = gt_rt_chw.permute(0, 2, 3, 1).reshape(1, 2, 384, 512, 3)

        loss = dct_quant_loss(rendered_pair, gt_pair)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)
        assert loss.item() > 0  # post-roundtrip residual is nonzero

        # Gradient must flow through the entire roundtrip + DCT chain back
        # to the original (pre-roundtrip) renderer-output tensor.
        loss.backward()
        assert rendered_chw.grad is not None
        assert rendered_chw.grad.shape == rendered_chw.shape
        assert torch.isfinite(rendered_chw.grad).all()
        assert rendered_chw.grad.abs().max().item() > 0


# ---- Input validation ---- #

class TestInputValidation:
    """Bad inputs should raise ValueError, never silently produce garbage."""

    def test_shape_mismatch_raises(self):
        rendered = torch.rand(1, 2, 32, 32, 3)
        gt = torch.rand(1, 2, 64, 64, 3)
        with pytest.raises(ValueError, match="shape"):
            dct_quant_loss(rendered, gt)

    def test_wrong_ndim_raises(self):
        rendered = torch.rand(1, 32, 32, 3)  # missing T dim
        gt = torch.rand(1, 32, 32, 3)
        with pytest.raises(ValueError, match=r"\(B, T, H, W, 3\)"):
            dct_quant_loss(rendered, gt)

    def test_wrong_channel_count_raises(self):
        rendered = torch.rand(1, 2, 32, 32, 4)  # RGBA
        gt = torch.rand(1, 2, 32, 32, 4)
        with pytest.raises(ValueError, match=r"\(B, T, H, W, 3\)"):
            dct_quant_loss(rendered, gt)

    def test_invalid_channel_mode_raises(self):
        rendered = _hwc_pair(32, 32)
        gt = _hwc_pair(32, 32)
        with pytest.raises(ValueError, match="channel_mode"):
            dct_quant_loss(rendered, gt, channel_mode="hsv")
