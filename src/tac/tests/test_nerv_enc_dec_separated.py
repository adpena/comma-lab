"""Tests for NeRV-Enc/Dec separated bolt-on."""
from __future__ import annotations

import functools

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.nerv_enc_dec_separated import (
    NeRVEncoder,
    NeRVEncoderConfig,
    encode_pair_batch,
    joint_train_step_with_decoder,
)


# ── Config ───────────────────────────────────────────────────────────────


def test_config_default():
    cfg = NeRVEncoderConfig()
    assert cfg.latent_dim == 16
    assert cfg.n_stages == 4
    assert cfg.frames_per_pair == 2


def test_config_rejects_zero_latent():
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        NeRVEncoderConfig(latent_dim=0)


def test_config_rejects_zero_base_channels():
    with pytest.raises(ValueError, match="base_channels must be positive"):
        NeRVEncoderConfig(base_channels=0)


def test_config_rejects_zero_stages():
    with pytest.raises(ValueError, match="n_stages must be"):
        NeRVEncoderConfig(n_stages=0)


def test_config_rejects_three_frames_per_pair():
    with pytest.raises(ValueError, match="frames_per_pair=2"):
        NeRVEncoderConfig(frames_per_pair=3)


# ── NeRVEncoder ──────────────────────────────────────────────────────────


def test_encoder_forward_shape_default():
    cfg = NeRVEncoderConfig()
    enc = NeRVEncoder(cfg)
    pair = torch.randint(0, 256, (2, 2, 3, 384, 512), dtype=torch.uint8).float()
    out = enc(pair)
    assert out.shape == (2, cfg.latent_dim)


def test_encoder_forward_shape_smaller():
    cfg = NeRVEncoderConfig(latent_dim=4, base_channels=4, n_stages=2)
    enc = NeRVEncoder(cfg)
    pair = torch.randint(0, 256, (3, 2, 3, 16, 16), dtype=torch.uint8).float()
    out = enc(pair)
    assert out.shape == (3, cfg.latent_dim)


def test_encoder_rejects_wrong_dim_count():
    cfg = NeRVEncoderConfig()
    enc = NeRVEncoder(cfg)
    bad = torch.randn(2, 3, 384, 512)
    with pytest.raises(ValueError, match="NeRVEncoder expected 5-D"):
        enc(bad)


def test_encoder_rejects_wrong_frames_per_pair():
    cfg = NeRVEncoderConfig()
    enc = NeRVEncoder(cfg)
    bad = torch.randint(0, 256, (2, 3, 3, 384, 512), dtype=torch.uint8).float()
    with pytest.raises(ValueError, match="frames_per_pair"):
        enc(bad)


def test_encoder_rejects_wrong_channel_count():
    cfg = NeRVEncoderConfig()
    enc = NeRVEncoder(cfg)
    bad = torch.randint(0, 256, (2, 2, 4, 384, 512), dtype=torch.uint8).float()
    with pytest.raises(ValueError, match="3 channels"):
        enc(bad)


def test_encoder_grad_attached():
    cfg = NeRVEncoderConfig(latent_dim=4, base_channels=4, n_stages=2)
    enc = NeRVEncoder(cfg)
    pair = torch.randint(0, 256, (1, 2, 3, 16, 16), dtype=torch.uint8).float()
    out = enc(pair)
    out.sum().backward()
    has_grad = any(p.grad is not None for p in enc.parameters())
    assert has_grad


# ── encode_pair_batch ────────────────────────────────────────────────────


def test_encode_pair_batch_resamples_to_eval_size():
    cfg = NeRVEncoderConfig(latent_dim=4, base_channels=4, n_stages=2,
                             eval_size=(16, 16))
    enc = NeRVEncoder(cfg)
    # Input at camera resolution larger than eval_size.
    pair = torch.randint(0, 256, (2, 2, 3, 64, 64), dtype=torch.uint8)
    out = encode_pair_batch(encoder=enc, frame_pairs_uint8=pair)
    assert out.shape == (2, cfg.latent_dim)


def test_encode_pair_batch_no_resample_when_size_matches():
    cfg = NeRVEncoderConfig(latent_dim=4, base_channels=4, n_stages=2,
                             eval_size=(16, 16))
    enc = NeRVEncoder(cfg)
    pair = torch.randint(0, 256, (2, 2, 3, 16, 16), dtype=torch.uint8)
    out = encode_pair_batch(encoder=enc, frame_pairs_uint8=pair)
    assert out.shape == (2, cfg.latent_dim)


def test_encode_pair_batch_rejects_non_5d():
    cfg = NeRVEncoderConfig()
    enc = NeRVEncoder(cfg)
    bad = torch.randn(2, 3, 384, 512)
    with pytest.raises(ValueError, match="encode_pair_batch expected 5-D"):
        encode_pair_batch(encoder=enc, frame_pairs_uint8=bad)


# ── joint_train_step_with_decoder ────────────────────────────────────────


def test_joint_train_step_uses_encoder_latents_via_lookup():
    """Verify the on-the-fly latent table returns encoder latents at the
    requested pair indices."""
    cfg = NeRVEncoderConfig(latent_dim=4, base_channels=4, n_stages=2,
                             eval_size=(16, 16))
    enc = NeRVEncoder(cfg)

    # Fake decoder train_step that just returns the latents seen by the table.
    captured = {}

    def _fake_train_step(*, latent_table, pair_indices, gt_pairs_uint8, **kwargs):
        latents = latent_table(pair_indices)
        captured["latents"] = latents
        captured["pair_indices"] = pair_indices
        return {"loss": latents.sum()}

    pair = torch.randint(0, 256, (3, 2, 3, 16, 16), dtype=torch.uint8)
    pair_idx = torch.tensor([10, 20, 30])
    result = joint_train_step_with_decoder(
        encoder=enc,
        decoder=enc,  # decoder unused in fake; arg present for interface symmetry
        frame_pairs_uint8=pair,
        decoder_train_step_fn=_fake_train_step,
        pair_indices=pair_idx,
    )
    assert "loss" in result
    assert captured["latents"].shape == (3, cfg.latent_dim)
    assert torch.equal(captured["pair_indices"], pair_idx)


def test_joint_train_step_rejects_pair_indices_size_mismatch():
    cfg = NeRVEncoderConfig(latent_dim=4, base_channels=4, n_stages=2,
                             eval_size=(16, 16))
    enc = NeRVEncoder(cfg)
    pair = torch.randint(0, 256, (3, 2, 3, 16, 16), dtype=torch.uint8)
    pair_idx = torch.tensor([0, 1])  # size 2, batch 3
    with pytest.raises(ValueError, match="encoder batch"):
        joint_train_step_with_decoder(
            encoder=enc, decoder=enc,
            frame_pairs_uint8=pair,
            decoder_train_step_fn=lambda **_: {"loss": torch.tensor(0.0)},
            pair_indices=pair_idx,
        )


def test_joint_train_step_on_the_fly_table_rejects_unknown_index():
    """If the substrate's train_step asks for a pair_index NOT in the encoder
    batch, the on-the-fly latent table raises (catches a bug class where the
    outer loop and the encoder batch get out of sync)."""
    cfg = NeRVEncoderConfig(latent_dim=4, base_channels=4, n_stages=2,
                             eval_size=(16, 16))
    enc = NeRVEncoder(cfg)
    pair = torch.randint(0, 256, (2, 2, 3, 16, 16), dtype=torch.uint8)
    pair_idx = torch.tensor([10, 20])

    def _bad_train_step(*, latent_table, pair_indices, **_):
        # Asks for index 99 which is NOT in the encoder batch.
        return {"loss": latent_table(torch.tensor([99])).sum()}

    with pytest.raises(ValueError, match="not in encoder batch"):
        joint_train_step_with_decoder(
            encoder=enc, decoder=enc,
            frame_pairs_uint8=pair,
            decoder_train_step_fn=_bad_train_step,
            pair_indices=pair_idx,
        )


def test_encoder_composes_with_lane_12_v2_decoder():
    """End-to-end: NeRVEncoder → Lane 12-v2 NeRV decoder forward.

    Confirms the bolt-on's output matches the decoder's expected input shape.
    """
    from tac.lane_12_v2_nerv_as_renderer import (
        Lane12V2NeRVConfig,
        Lane12V2NeRVRenderer,
    )

    enc_cfg = NeRVEncoderConfig(latent_dim=8, base_channels=4, n_stages=2,
                                 eval_size=(16, 16))
    enc = NeRVEncoder(enc_cfg)
    dec_cfg = Lane12V2NeRVConfig(latent_dim=8, base_channels=8, n_pairs=4,
                                  cuda_required=False)
    dec = Lane12V2NeRVRenderer(dec_cfg)

    pair = torch.randint(0, 256, (2, 2, 3, 16, 16), dtype=torch.uint8)
    z = encode_pair_batch(encoder=enc, frame_pairs_uint8=pair)
    out = dec(z)
    assert out.shape == (2, 2, 3, 384, 512)
