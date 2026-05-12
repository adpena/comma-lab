"""Tests for FFNeRV-as-renderer substrate."""
from __future__ import annotations

import math
import struct

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.ffnerv_as_renderer import (
    ARCHIVE_GRAMMAR_FFNERV,
    FFNERV_FORMAT_ID,
    FFNERV_FORMAT_VERSION,
    FFNERV_MAGIC,
    FFNeRVConfig,
    FFNeRVLatentTable,
    FFNeRVRenderer,
    FourierFeatureEncoding,
    _make_synthetic_pair_batch_for_smoke,
    _quantize_per_tensor_int8_with_fp16_scale,
    default_pose_surrogate,
    default_seg_surrogate,
    export_ffnerv_to_archive,
    train_step_ffnerv,
)


def test_config_default():
    cfg = FFNeRVConfig()
    assert cfg.latent_dim == 16
    assert cfg.n_frequencies == 16
    # encoded_dim = latent_dim * (1 + 2*n_freq)
    assert cfg.encoded_dim == 16 * (1 + 2 * 16)


def test_config_rejects_zero_latent():
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        FFNeRVConfig(latent_dim=0)


def test_config_rejects_zero_freq():
    with pytest.raises(ValueError, match="n_frequencies must be positive"):
        FFNeRVConfig(n_frequencies=0)


def test_config_rejects_inverted_freq_range():
    with pytest.raises(ValueError, match="log_freq_max must exceed"):
        FFNeRVConfig(log_freq_min=2.0, log_freq_max=1.0)


def test_config_rejects_non_six_stages():
    with pytest.raises(ValueError, match="Phase A pinned at n_stages=6"):
        FFNeRVConfig(n_stages=4)


# ── Fourier encoding ─────────────────────────────────────────────────────


def test_fourier_encoding_shape():
    enc = FourierFeatureEncoding(latent_dim=4, n_frequencies=8)
    z = torch.randn(2, 4)
    out = enc(z)
    assert out.shape == (2, 4 + 4 * 16)


def test_fourier_encoding_includes_raw_values():
    enc = FourierFeatureEncoding(latent_dim=4, n_frequencies=8)
    z = torch.randn(2, 4)
    out = enc(z)
    # First 4 cols should be raw z
    assert torch.allclose(out[:, :4], z)


def test_fourier_encoding_frequencies_log_spaced():
    enc = FourierFeatureEncoding(latent_dim=4, n_frequencies=4,
                                  log_freq_min=0.0, log_freq_max=3.0)
    expected = 2.0 ** torch.linspace(0, 3, 4)
    assert torch.allclose(enc.frequencies, expected)


def test_fourier_encoding_rejects_wrong_latent_dim():
    enc = FourierFeatureEncoding(latent_dim=4, n_frequencies=8)
    bad = torch.randn(2, 5)
    with pytest.raises(ValueError, match="FourierFeatureEncoding expected"):
        enc(bad)


def test_fourier_encoding_sin_cos_orthogonality_at_zero():
    enc = FourierFeatureEncoding(latent_dim=2, n_frequencies=4)
    z = torch.zeros(1, 2)
    out = enc(z)
    # raw=0, sin(0)=0, cos(0)=1
    raw = out[:, :2]
    sin_part = out[:, 2:2 + 2 * 4]
    cos_part = out[:, 2 + 2 * 4:]
    assert torch.allclose(raw, torch.zeros(1, 2))
    assert torch.allclose(sin_part, torch.zeros(1, 8))
    assert torch.allclose(cos_part, torch.ones(1, 8))


# ── Renderer ─────────────────────────────────────────────────────────────


def test_renderer_forward_default_shape():
    cfg = FFNeRVConfig()
    r = FFNeRVRenderer(cfg)
    z = torch.randn(2, cfg.latent_dim) * 0.01
    out = r(z)
    assert out.shape == (2, 2, 3, 384, 512)


def test_renderer_forward_smaller_config_works():
    # Custom small config — n_stages MUST stay 6 per Phase A; reduce base_channels.
    cfg = FFNeRVConfig(latent_dim=4, n_frequencies=4, base_channels=8)
    r = FFNeRVRenderer(cfg)
    z = torch.randn(1, cfg.latent_dim) * 0.01
    out = r(z)
    assert out.shape == (1, 2, 3, 384, 512)


def test_renderer_rejects_wrong_latent_shape():
    cfg = FFNeRVConfig()
    r = FFNeRVRenderer(cfg)
    bad = torch.randn(2, 99)
    with pytest.raises(ValueError, match="forward expected"):
        r(bad)


def test_renderer_output_in_zero_to_255_range():
    cfg = FFNeRVConfig(base_channels=12)
    r = FFNeRVRenderer(cfg)
    z = torch.randn(1, cfg.latent_dim) * 0.05
    out = r(z)
    assert (out >= 0).all() and (out <= 255.0).all()


def test_renderer_schema_non_empty_well_formed():
    cfg = FFNeRVConfig(base_channels=8)
    r = FFNeRVRenderer(cfg)
    schema = r.schema
    assert len(schema) > 0
    sd = r.state_dict()
    for key, shape in schema:
        assert key in sd
        assert tuple(sd[key].shape) == shape


def test_renderer_fourier_buffer_present():
    cfg = FFNeRVConfig(base_channels=8)
    r = FFNeRVRenderer(cfg)
    assert "fourier.frequencies" in dict(r.named_buffers())


# ── Latent table ─────────────────────────────────────────────────────────


def test_latent_table_init_and_forward():
    table = FFNeRVLatentTable(n_pairs=4, latent_dim=16)
    out = table(torch.tensor([0, 2, 1]))
    assert out.shape == (3, 16)


# ── train_step ───────────────────────────────────────────────────────────


class _DummyScorerSeg(nn.Module):
    def preprocess_input(self, x):
        if x.dim() == 5:
            x = x[:, -1]
        return F.interpolate(x.float(), size=(64, 64), mode="bilinear", align_corners=False)

    def forward(self, x):
        return torch.randn_like(x[:, :1]).repeat(1, 5, 1, 1)


class _DummyScorerPose(nn.Module):
    def preprocess_input(self, x):
        if x.dim() == 5:
            B, F_pp, C, H, W = x.shape
            x = x.reshape(B, F_pp * C, H, W)
        return F.interpolate(x.float(), size=(32, 32), mode="bilinear", align_corners=False)

    def forward(self, x):
        return x.flatten(1)[:, :12]


def test_train_step_ffnerv_returns_loss_dict():
    cfg = FFNeRVConfig(latent_dim=4, n_frequencies=4, base_channels=8, n_pairs=4,
                       cuda_required=False)
    r = FFNeRVRenderer(cfg)
    table = FFNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    pair_idx = torch.tensor([0, 1])
    gt = torch.randint(0, 256, (2, 2, 3, 64, 64), dtype=torch.uint8)
    result = train_step_ffnerv(
        renderer=r, latent_table=table, pair_indices=pair_idx, gt_pairs_uint8=gt,
        scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
        seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
        lambda_seg=100.0, lambda_pose=288.675,
    )
    for key in ("loss", "loss_seg", "loss_pose", "loss_seg_unweighted", "loss_pose_unweighted"):
        assert key in result and torch.is_tensor(result[key])


def test_train_step_ffnerv_refuses_eval_roundtrip_false():
    cfg = FFNeRVConfig(latent_dim=4, n_frequencies=4, base_channels=8, n_pairs=4,
                       cuda_required=False)
    r = FFNeRVRenderer(cfg)
    table = FFNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    with pytest.raises(ValueError, match="eval_roundtrip=False is forbidden"):
        train_step_ffnerv(
            renderer=r, latent_table=table,
            pair_indices=torch.tensor([0]),
            gt_pairs_uint8=torch.randint(0, 256, (1, 2, 3, 64, 64), dtype=torch.uint8),
            scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
            seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
            lambda_seg=1.0, lambda_pose=1.0, eval_roundtrip=False,
        )


def test_train_step_ffnerv_grad_attached():
    cfg = FFNeRVConfig(latent_dim=4, n_frequencies=4, base_channels=8, n_pairs=4,
                       cuda_required=False)
    r = FFNeRVRenderer(cfg)
    table = FFNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    pair_idx = torch.tensor([0])
    gt = torch.randint(0, 256, (1, 2, 3, 64, 64), dtype=torch.uint8)
    result = train_step_ffnerv(
        renderer=r, latent_table=table, pair_indices=pair_idx, gt_pairs_uint8=gt,
        scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
        seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
        lambda_seg=1.0, lambda_pose=1.0,
    )
    assert torch.isfinite(result["loss"]).all()
    assert result["loss"].requires_grad


# ── Quantization + archive ───────────────────────────────────────────────


def test_quantize_per_tensor_int8_round_trip():
    t = torch.randn(8, 4) * 0.1
    q, scale = _quantize_per_tensor_int8_with_fp16_scale(t)
    rec = q.float() * scale.float()
    assert torch.allclose(rec, t, atol=0.01, rtol=0.05)


def test_export_ffnerv_archive_returns_sha_and_writes_file(tmp_path):
    cfg = FFNeRVConfig(latent_dim=4, n_frequencies=4, base_channels=8, n_pairs=4,
                       cuda_required=False)
    r = FFNeRVRenderer(cfg)
    table = FFNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    out = tmp_path / "0.bin"
    sha = export_ffnerv_to_archive(renderer=r, latent_table=table, output_path=out)
    assert isinstance(sha, str) and len(sha) == 64
    assert out.exists()


def test_export_ffnerv_header_well_formed(tmp_path):
    cfg = FFNeRVConfig(latent_dim=4, n_frequencies=4, base_channels=8, n_pairs=4,
                       cuda_required=False)
    r = FFNeRVRenderer(cfg)
    table = FFNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    out = tmp_path / "0.bin"
    export_ffnerv_to_archive(renderer=r, latent_table=table, output_path=out)
    blob = out.read_bytes()
    assert blob[:4] == FFNERV_MAGIC
    (ver,) = struct.unpack_from("<H", blob, 4)
    assert ver == FFNERV_FORMAT_VERSION
    (fid,) = struct.unpack_from("<H", blob, 6)
    assert fid == FFNERV_FORMAT_ID
    (latent_dim,) = struct.unpack_from("<H", blob, 8)
    assert latent_dim == cfg.latent_dim
    (n_pairs,) = struct.unpack_from("<H", blob, 10)
    assert n_pairs == cfg.n_pairs
    (n_freq,) = struct.unpack_from("<H", blob, 12)
    assert n_freq == cfg.n_frequencies


def test_export_ffnerv_deterministic(tmp_path):
    cfg = FFNeRVConfig(latent_dim=4, n_frequencies=4, base_channels=8, n_pairs=4,
                       cuda_required=False)
    torch.manual_seed(0)
    r = FFNeRVRenderer(cfg)
    table = FFNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    out_a = tmp_path / "a.bin"
    out_b = tmp_path / "b.bin"
    sha_a = export_ffnerv_to_archive(renderer=r, latent_table=table, output_path=out_a)
    sha_b = export_ffnerv_to_archive(renderer=r, latent_table=table, output_path=out_b)
    assert sha_a == sha_b


def test_export_ffnerv_rejects_latent_shape_mismatch(tmp_path):
    cfg = FFNeRVConfig(latent_dim=4, n_frequencies=4, base_channels=8, n_pairs=4,
                       cuda_required=False)
    r = FFNeRVRenderer(cfg)
    bad = FFNeRVLatentTable(n_pairs=99, latent_dim=4)
    with pytest.raises(ValueError, match="latent_table shape"):
        export_ffnerv_to_archive(renderer=r, latent_table=bad, output_path=tmp_path / "x.bin")


# ── Smoke + grammar ──────────────────────────────────────────────────────


def test_make_synthetic_pair_batch_smoke_shape():
    pi, gt = _make_synthetic_pair_batch_for_smoke(
        batch_size=2, latent_dim=4, eval_size=(384, 512), n_pairs=4, seed=0,
    )
    assert pi.shape == (2,)
    assert gt.shape == (2, 2, 3, 874, 1164)


def test_archive_grammar_ffnerv_well_formed():
    g = ARCHIVE_GRAMMAR_FFNERV
    assert g["format_id"] == FFNERV_FORMAT_ID
    assert g["magic"] == FFNERV_MAGIC.decode("ascii")
    section_names = [s["name"] for s in g["sections"]]
    assert "freq_table" in section_names
    assert section_names[0] == "header"
