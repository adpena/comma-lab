"""Unit tests for tac.paradigm_delta_epsilon_zeta.decoder_128k."""
from __future__ import annotations

import pytest
import torch

from tac.paradigm_delta_epsilon_zeta.decoder_128k import (
    DECODER_128K_PARAM_BAND,
    Decoder128K,
    Decoder128KConfig,
    build_decoder_128k,
    decoder_128k_param_count,
)


def test_default_config_param_count_in_band():
    n = decoder_128k_param_count(Decoder128KConfig())
    low, high = DECODER_128K_PARAM_BAND
    assert low <= n <= high, f"{n} outside band [{low}, {high}]"


def test_build_decoder_returns_module():
    decoder = build_decoder_128k()
    assert isinstance(decoder, Decoder128K)


def test_forward_output_shape():
    decoder = build_decoder_128k()
    z = torch.randn(4, 28)
    out = decoder(z)
    assert out.shape == (4, 2, 3, 384, 512)


def test_forward_output_in_pixel_range():
    decoder = build_decoder_128k()
    z = torch.randn(2, 28)
    out = decoder(z)
    # sigmoid * 255 → [0, 255]
    assert out.min().item() >= 0.0
    assert out.max().item() <= 255.0


def test_forward_rejects_wrong_latent_dim():
    decoder = build_decoder_128k()
    z = torch.randn(2, 30)  # wrong dim (default is 28)
    with pytest.raises(ValueError, match="expected"):
        decoder(z)


def test_forward_rejects_wrong_rank():
    decoder = build_decoder_128k()
    z = torch.randn(2, 28, 3)  # wrong rank
    with pytest.raises(ValueError, match="expected"):
        decoder(z)


def test_build_rejects_out_of_band_params_when_strict():
    cfg = Decoder128KConfig(base_channels=80)  # way over budget
    with pytest.raises(RuntimeError, match="expected band"):
        build_decoder_128k(cfg, enforce_param_band=True)


def test_build_allows_out_of_band_when_strict_disabled():
    cfg = Decoder128KConfig(base_channels=80)
    decoder = build_decoder_128k(cfg, enforce_param_band=False)
    n = sum(p.numel() for p in decoder.parameters())
    high = DECODER_128K_PARAM_BAND[1]
    assert n > high  # explicitly exercises the override


def test_config_validates_pixelshuffle_geometry():
    with pytest.raises(ValueError, match="PixelShuffle"):
        Decoder128KConfig(base_h=4, base_w=4)


def test_config_validates_channel_shrink_length():
    with pytest.raises(ValueError, match="channel_shrink"):
        Decoder128KConfig(channel_shrink=(1.0, 0.5))


def test_forward_is_deterministic():
    torch.manual_seed(42)
    decoder1 = build_decoder_128k()
    torch.manual_seed(42)
    decoder2 = build_decoder_128k()
    z = torch.randn(2, 28)
    o1 = decoder1(z)
    o2 = decoder2(z)
    torch.testing.assert_close(o1, o2)


def test_forward_supports_backprop():
    decoder = build_decoder_128k()
    z = torch.randn(2, 28, requires_grad=True)
    out = decoder(z)
    loss = out.mean()
    loss.backward()
    assert z.grad is not None
    # Decoder params receive gradient too.
    grads = [
        p.grad for p in decoder.parameters() if p.requires_grad and p.grad is not None
    ]
    assert len(grads) > 0


def test_forward_separate_frame_heads():
    """Each frame must come from its dedicated rgb head, not a shared one."""
    decoder = build_decoder_128k()
    z = torch.randn(2, 28)
    out = decoder(z)
    f0, f1 = out[:, 0], out[:, 1]
    # Frames must DIFFER from each other in general (separate heads).
    assert not torch.allclose(f0, f1)
