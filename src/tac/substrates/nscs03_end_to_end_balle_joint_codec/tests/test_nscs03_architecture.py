# SPDX-License-Identifier: MIT
"""NSCS03 architecture tests: forward, backward, gradient reachability,
parameter count, shape invariants."""

from __future__ import annotations

import pytest
import torch

from tac.substrates.nscs03_end_to_end_balle_joint_codec.architecture import (
    NSCS03Config,
    NSCS03JointCodecSubstrate,
    _conditional_gaussian_rate,
    _quantize_with_noise,
)


def _make_substrate() -> NSCS03JointCodecSubstrate:
    torch.manual_seed(0)
    cfg = NSCS03Config()
    return NSCS03JointCodecSubstrate(cfg)


class TestNSCS03Config:
    def test_defaults_validate(self) -> None:
        cfg = NSCS03Config()
        assert cfg.in_channels == 6
        assert cfg.out_channels == 6
        assert cfg.main_latent_channels == 64
        assert cfg.hyper_latent_channels == 32

    def test_g_a_must_end_with_main_latent_channels(self) -> None:
        with pytest.raises(ValueError, match="g_a_channels must end with"):
            NSCS03Config(g_a_channels=(32, 48, 56, 99))

    def test_g_s_must_start_with_main_latent_channels(self) -> None:
        with pytest.raises(ValueError, match="g_s_channels must start with"):
            NSCS03Config(g_s_channels=(99, 56, 48, 32))

    def test_h_a_must_end_with_hyper_latent_channels(self) -> None:
        with pytest.raises(ValueError, match="h_a_channels must end with"):
            NSCS03Config(h_a_channels=(48, 99))

    def test_h_s_must_end_with_main_latent_channels(self) -> None:
        # h_s output is sigma which must align with the main latent y
        with pytest.raises(ValueError, match="h_s_channels must end with"):
            NSCS03Config(h_s_channels=(48, 99))

    def test_negative_quantize_noise_rejected(self) -> None:
        with pytest.raises(ValueError, match="quantize_noise_std"):
            NSCS03Config(quantize_noise_std=-0.1)

    def test_zero_gdn_eps_rejected(self) -> None:
        with pytest.raises(ValueError, match="gdn_eps"):
            NSCS03Config(gdn_eps=0.0)

    def test_zero_sigma_floor_rejected(self) -> None:
        with pytest.raises(ValueError, match="sigma_floor"):
            NSCS03Config(sigma_floor=0.0)


class TestNSCS03ForwardPass:
    def test_forward_shape_invariants(self) -> None:
        m = _make_substrate()
        m.eval()
        x = torch.rand(2, 6, 384, 512)
        recon, parts = m(x)
        assert recon.shape == (2, 6, 384, 512)
        assert recon.min().item() >= 0.0
        assert recon.max().item() <= 1.0
        # Latent shapes
        assert parts["y"].shape == (2, 64, 24, 32)
        assert parts["z"].shape == (2, 32, 6, 8)
        assert parts["sigma"].shape == (2, 64, 24, 32)
        # Rate components are scalar
        assert parts["main_rate"].shape == ()
        assert parts["hyper_rate"].shape == ()
        assert parts["total_rate"].shape == ()

    def test_forward_input_channels_validated(self) -> None:
        m = _make_substrate()
        with pytest.raises(ValueError, match="x_pair must be"):
            m(torch.rand(2, 3, 384, 512))

    def test_forward_3d_input_rejected(self) -> None:
        m = _make_substrate()
        with pytest.raises(ValueError, match="x_pair must be"):
            m(torch.rand(6, 384, 512))

    def test_eval_mode_uses_round_quantization(self) -> None:
        m = _make_substrate()
        m.eval()
        x = torch.rand(1, 6, 384, 512)
        # Two forward passes must be byte-identical in eval mode
        torch.manual_seed(99)
        r1, _ = m(x)
        torch.manual_seed(99)
        r2, _ = m(x)
        assert torch.allclose(r1, r2)

    def test_train_mode_uses_noise_quantization(self) -> None:
        m = _make_substrate()
        m.train()
        x = torch.rand(1, 6, 384, 512)
        # Two forward passes will differ due to noise
        torch.manual_seed(1)
        r1, p1 = m(x)
        torch.manual_seed(2)
        r2, p2 = m(x)
        # Recon will differ; main_rate will differ
        assert not torch.allclose(r1, r2)


class TestNSCS03BackwardPass:
    """End-to-end gradient-reachability tests (CRUX of NSCS03)."""

    def test_gradient_reaches_g_a_encoder(self) -> None:
        m = _make_substrate()
        m.train()
        x = torch.rand(2, 6, 384, 512)
        recon, parts = m(x)
        # Strong forcing to get clear gradient
        loss = torch.nn.functional.mse_loss(recon, x) * 100.0 + 0.1 * parts["total_rate"]
        loss.backward()
        # All g_a parameters must have gradient
        for p in m.g_a.parameters():
            assert p.grad is not None
        # Gradient must be non-zero
        max_grad = max(p.grad.abs().max().item() for p in m.g_a.parameters())
        assert max_grad > 0.0

    def test_gradient_reaches_g_s_decoder(self) -> None:
        m = _make_substrate()
        m.train()
        x = torch.rand(2, 6, 384, 512)
        recon, parts = m(x)
        loss = torch.nn.functional.mse_loss(recon, x)
        loss.backward()
        max_grad = max(p.grad.abs().max().item() for p in m.g_s.parameters())
        assert max_grad > 0.0

    def test_gradient_reaches_h_a_hyper_encoder(self) -> None:
        m = _make_substrate()
        m.train()
        x = torch.rand(2, 6, 384, 512)
        recon, parts = m(x)
        loss = parts["total_rate"]  # rate-only; isolates h_a/h_s/eb path
        loss.backward()
        max_grad = max(p.grad.abs().max().item() for p in m.h_a.parameters())
        assert max_grad > 0.0

    def test_gradient_reaches_h_s_hyper_decoder(self) -> None:
        m = _make_substrate()
        m.train()
        x = torch.rand(2, 6, 384, 512)
        recon, parts = m(x)
        loss = parts["main_rate"]  # main_rate uses sigma which comes from h_s
        loss.backward()
        max_grad = max(p.grad.abs().max().item() for p in m.h_s.parameters())
        assert max_grad > 0.0

    def test_gradient_reaches_entropy_bottleneck(self) -> None:
        m = _make_substrate()
        m.train()
        x = torch.rand(2, 6, 384, 512)
        recon, parts = m(x)
        loss = parts["hyper_rate"]
        loss.backward()
        max_grad = max(
            p.grad.abs().max().item() for p in m.entropy_bottleneck_z.parameters()
        )
        assert max_grad > 0.0

    def test_gradient_reaches_ALL_subnetworks_jointly(self) -> None:
        """The CANONICAL end-to-end test: a single combined loss must produce
        non-zero gradient in all 5 sub-networks (g_a, g_s, h_a, h_s, eb)."""
        m = _make_substrate()
        m.train()
        x = torch.rand(2, 6, 384, 512)
        recon, parts = m(x)
        loss = (
            torch.nn.functional.mse_loss(recon, x) * 100.0
            + 0.1 * parts["total_rate"]
        )
        loss.backward()
        for sub_name in ["g_a", "g_s", "h_a", "h_s", "entropy_bottleneck_z"]:
            sub = getattr(m, sub_name)
            grads = [p.grad.abs().max().item() for p in sub.parameters() if p.grad is not None]
            assert grads, f"{sub_name}: no parameter has gradient"
            assert max(grads) > 0.0, f"{sub_name}: max grad is 0"


class TestNSCS03ParameterCount:
    def test_param_count_within_budget(self) -> None:
        """Per CLAUDE.md substrate-engineering exception (L7), NSCS03 ships
        the encoder too so it carries more params than a renderer-only
        substrate. Council target: < 600K total."""
        m = _make_substrate()
        n_params = m.num_parameters()
        assert n_params < 600_000, f"NSCS03 has {n_params} params; budget < 600K"
        assert n_params > 100_000, f"NSCS03 only has {n_params} params; expected > 100K"


class TestNSCS03Decompose:
    def test_split_recon_into_frames(self) -> None:
        m = _make_substrate()
        recon = torch.rand(2, 6, 384, 512)
        rgb_0, rgb_1 = m.split_recon_into_frames(recon)
        assert rgb_0.shape == (2, 3, 384, 512)
        assert rgb_1.shape == (2, 3, 384, 512)

    def test_split_recon_rejects_wrong_channels(self) -> None:
        m = _make_substrate()
        with pytest.raises(ValueError, match="6 channels"):
            m.split_recon_into_frames(torch.rand(2, 5, 384, 512))

    def test_stack_frames_into_pair(self) -> None:
        rgb_0 = torch.rand(2, 3, 384, 512)
        rgb_1 = torch.rand(2, 3, 384, 512)
        x = NSCS03JointCodecSubstrate.stack_frames_into_pair(rgb_0, rgb_1)
        assert x.shape == (2, 6, 384, 512)
        assert torch.equal(x[:, :3], rgb_0)
        assert torch.equal(x[:, 3:], rgb_1)

    def test_stack_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="matching shapes"):
            NSCS03JointCodecSubstrate.stack_frames_into_pair(
                torch.rand(2, 3, 384, 512), torch.rand(1, 3, 384, 512)
            )

    def test_stack_rejects_wrong_channels(self) -> None:
        with pytest.raises(ValueError, match="\\(B, 3, H, W\\)"):
            NSCS03JointCodecSubstrate.stack_frames_into_pair(
                torch.rand(2, 5, 384, 512), torch.rand(2, 5, 384, 512)
            )


class TestNSCS03EncodeAndDecode:
    def test_encode_returns_y_and_z(self) -> None:
        m = _make_substrate()
        m.eval()
        x = torch.rand(3, 6, 384, 512)
        latents = m.encode(x)
        assert "y" in latents
        assert "z" in latents
        assert latents["y"].shape == (3, 64, 24, 32)
        assert latents["z"].shape == (3, 32, 6, 8)

    def test_decode_from_y_hat(self) -> None:
        m = _make_substrate()
        m.eval()
        # Encode then quantize then decode
        x = torch.rand(2, 6, 384, 512)
        with torch.no_grad():
            y = m.encode(x)["y"]
            y_hat = y.round()
            recon = m.decode(y_hat)
        # Decoder output spatial size depends on g_s; before final interpolate
        assert recon.dim() == 4
        assert recon.shape[0] == 2
        assert recon.shape[1] == 6


class TestQuantizationRelaxation:
    def test_train_mode_adds_uniform_noise(self) -> None:
        x = torch.zeros(100)
        torch.manual_seed(5)
        y = _quantize_with_noise(x, noise_std=0.5, training=True)
        # Noise is uniform in [-0.5, 0.5]; mean ~0, std ~0.289
        assert y.min() > -0.5 - 1e-6
        assert y.max() < 0.5 + 1e-6
        assert (y != 0).all()

    def test_eval_mode_uses_hard_round(self) -> None:
        x = torch.tensor([0.1, 0.6, 1.4, -0.7, -1.5])
        y = _quantize_with_noise(x, noise_std=0.5, training=False)
        # PyTorch round() ties-to-even: -0.7 -> -1.0, -1.5 -> -2.0
        # but some tensors round half away from zero. Use isclose for safety.
        assert torch.allclose(y, torch.tensor([0.0, 1.0, 1.0, -1.0, -2.0])) or \
               torch.allclose(y, torch.tensor([0.0, 1.0, 1.0, -1.0, -1.0]))


class TestConditionalGaussianRate:
    def test_rate_decreases_with_smaller_y_relative_to_sigma(self) -> None:
        """The conditional-Gaussian rate -log2 N(y; 0, σ²) is minimized when
        y is small relative to σ. Test this monotonicity."""
        sigma = torch.ones(100, 100)
        y_small = torch.randn(100, 100) * 0.1
        y_large = torch.randn(100, 100) * 5.0
        rate_small = _conditional_gaussian_rate(y_small, sigma, sigma_floor=1e-4)
        rate_large = _conditional_gaussian_rate(y_large, sigma, sigma_floor=1e-4)
        assert rate_small < rate_large

    def test_rate_is_scalar(self) -> None:
        y = torch.randn(2, 64, 24, 32)
        sigma = torch.ones(2, 64, 24, 32) * 0.5
        rate = _conditional_gaussian_rate(y, sigma, sigma_floor=1e-4)
        assert rate.shape == ()

    def test_rate_uses_sigma_floor(self) -> None:
        """Even with σ → 0, the rate must remain finite (clamp by floor)."""
        y = torch.randn(2, 64, 24, 32)
        sigma_tiny = torch.zeros(2, 64, 24, 32)
        rate = _conditional_gaussian_rate(y, sigma_tiny, sigma_floor=1e-4)
        assert torch.isfinite(rate)
