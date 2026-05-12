"""Tests for DSNeRV-as-renderer (diffusion-supervised) substrate."""
from __future__ import annotations

import math
import struct

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.dsnerv_as_renderer import (
    ARCHIVE_GRAMMAR_DSNERV,
    DSNERV_FORMAT_ID,
    DSNERV_FORMAT_VERSION,
    DSNERV_MAGIC,
    DSNeRVConfig,
    DSNeRVLatentTable,
    DSNeRVRenderer,
    NoiseSchedule,
    _make_synthetic_pair_batch_for_smoke,
    _quantize_per_tensor_int8_with_fp16_scale,
    default_pose_surrogate,
    default_seg_surrogate,
    export_dsnerv_to_archive,
    train_step_dsnerv,
)


# ── Config ───────────────────────────────────────────────────────────────


def test_config_default():
    cfg = DSNeRVConfig()
    assert cfg.n_diffusion_steps == 10
    assert cfg.noise_schedule == "cosine"
    assert cfg.sigma_max == 1.0


def test_config_rejects_zero_diffusion_steps():
    with pytest.raises(ValueError, match="n_diffusion_steps must be positive"):
        DSNeRVConfig(n_diffusion_steps=0)


def test_config_rejects_unknown_schedule():
    with pytest.raises(ValueError, match="noise_schedule must be"):
        DSNeRVConfig(noise_schedule="exponential")


def test_config_rejects_zero_sigma_max():
    with pytest.raises(ValueError, match="sigma_max must be positive"):
        DSNeRVConfig(sigma_max=0.0)


def test_config_rejects_zero_latent():
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        DSNeRVConfig(latent_dim=0)


def test_config_rejects_non_six_stages():
    with pytest.raises(ValueError, match="Phase A pinned at n_stages=6"):
        DSNeRVConfig(n_stages=4)


# ── NoiseSchedule ─────────────────────────────────────────────────────────


def test_noise_schedule_linear_monotonic_increasing():
    s = NoiseSchedule(n_steps=10, sigma_max=1.0, schedule="linear")
    sigmas = s.sigmas.tolist()
    assert sigmas == sorted(sigmas)
    assert s.sigma_at(1) < s.sigma_at(10)
    assert s.sigma_at(10) == pytest.approx(1.0)


def test_noise_schedule_cosine_monotonic_increasing():
    s = NoiseSchedule(n_steps=10, sigma_max=1.0, schedule="cosine")
    sigmas = s.sigmas.tolist()
    assert sigmas == sorted(sigmas)
    # Cosine starts slowly and grows
    assert s.sigma_at(1) < s.sigma_at(5) < s.sigma_at(10)


def test_noise_schedule_sigma_at_out_of_range_raises():
    s = NoiseSchedule(n_steps=10, sigma_max=1.0)
    with pytest.raises(ValueError, match="step out of range"):
        s.sigma_at(0)
    with pytest.raises(ValueError, match="step out of range"):
        s.sigma_at(11)


def test_noise_schedule_sample_step_in_range():
    s = NoiseSchedule(n_steps=10, sigma_max=1.0)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        t = s.sample_step(generator=g)
        assert 1 <= t <= 10


def test_noise_schedule_rejects_invalid_init():
    with pytest.raises(ValueError):
        NoiseSchedule(n_steps=0, sigma_max=1.0)
    with pytest.raises(ValueError):
        NoiseSchedule(n_steps=10, sigma_max=0.0)
    with pytest.raises(ValueError):
        NoiseSchedule(n_steps=10, sigma_max=1.0, schedule="bogus")


# ── Renderer ─────────────────────────────────────────────────────────────


def test_renderer_forward_default_shape():
    cfg = DSNeRVConfig()
    r = DSNeRVRenderer(cfg)
    z = torch.randn(2, cfg.latent_dim) * 0.01
    out = r(z)
    assert out.shape == (2, 2, 3, 384, 512)


def test_renderer_forward_smaller_config_works():
    cfg = DSNeRVConfig(latent_dim=4, base_channels=8)
    r = DSNeRVRenderer(cfg)
    z = torch.randn(1, cfg.latent_dim) * 0.01
    out = r(z)
    assert out.shape == (1, 2, 3, 384, 512)


def test_renderer_rejects_wrong_latent_shape():
    cfg = DSNeRVConfig(base_channels=8)
    r = DSNeRVRenderer(cfg)
    with pytest.raises(ValueError, match="forward expected"):
        r(torch.randn(2, 99))


def test_renderer_output_in_zero_to_255_range():
    cfg = DSNeRVConfig(base_channels=8)
    r = DSNeRVRenderer(cfg)
    z = torch.randn(1, cfg.latent_dim) * 0.05
    out = r(z)
    assert (out >= 0).all() and (out <= 255.0).all()


def test_renderer_schema_well_formed():
    cfg = DSNeRVConfig(base_channels=8)
    r = DSNeRVRenderer(cfg)
    sd = r.state_dict()
    for key, shape in r.schema:
        assert key in sd and tuple(sd[key].shape) == shape


# ── Latent table ─────────────────────────────────────────────────────────


def test_latent_table_init_and_forward():
    table = DSNeRVLatentTable(n_pairs=4, latent_dim=16)
    out = table(torch.tensor([0, 2, 1]))
    assert out.shape == (3, 16)


# ── train_step diffusion-supervised ──────────────────────────────────────


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


def test_train_step_dsnerv_returns_loss_dict_with_diffusion_meta():
    cfg = DSNeRVConfig(latent_dim=4, base_channels=8, n_pairs=4, cuda_required=False)
    r = DSNeRVRenderer(cfg)
    table = DSNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    sched = NoiseSchedule(n_steps=cfg.n_diffusion_steps, sigma_max=cfg.sigma_max)
    pair_idx = torch.tensor([0, 1])
    gt = torch.randint(0, 256, (2, 2, 3, 64, 64), dtype=torch.uint8)
    result = train_step_dsnerv(
        renderer=r, latent_table=table, pair_indices=pair_idx, gt_pairs_uint8=gt,
        scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
        seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
        lambda_seg=100.0, lambda_pose=288.675,
        noise_schedule=sched,
        diffusion_step=5,
    )
    for key in ("loss", "loss_seg", "loss_pose", "diffusion_step", "diffusion_sigma"):
        assert key in result
    assert result["diffusion_step"] == 5
    assert result["diffusion_sigma"] > 0.0


def test_train_step_dsnerv_refuses_eval_roundtrip_false():
    cfg = DSNeRVConfig(latent_dim=4, base_channels=8, n_pairs=4, cuda_required=False)
    r = DSNeRVRenderer(cfg)
    table = DSNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    sched = NoiseSchedule(n_steps=cfg.n_diffusion_steps, sigma_max=cfg.sigma_max)
    with pytest.raises(ValueError, match="eval_roundtrip=False is forbidden"):
        train_step_dsnerv(
            renderer=r, latent_table=table,
            pair_indices=torch.tensor([0]),
            gt_pairs_uint8=torch.randint(0, 256, (1, 2, 3, 64, 64), dtype=torch.uint8),
            scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
            seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
            lambda_seg=1.0, lambda_pose=1.0,
            noise_schedule=sched,
            eval_roundtrip=False,
        )


def test_train_step_dsnerv_grad_attached_through_noise():
    cfg = DSNeRVConfig(latent_dim=4, base_channels=8, n_pairs=4, cuda_required=False)
    r = DSNeRVRenderer(cfg)
    table = DSNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    sched = NoiseSchedule(n_steps=cfg.n_diffusion_steps, sigma_max=cfg.sigma_max)
    result = train_step_dsnerv(
        renderer=r, latent_table=table,
        pair_indices=torch.tensor([0]),
        gt_pairs_uint8=torch.randint(0, 256, (1, 2, 3, 64, 64), dtype=torch.uint8),
        scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
        seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
        lambda_seg=1.0, lambda_pose=1.0,
        noise_schedule=sched,
    )
    assert torch.isfinite(result["loss"]).all()
    assert result["loss"].requires_grad


def test_train_step_dsnerv_random_step_when_none_provided():
    cfg = DSNeRVConfig(latent_dim=4, base_channels=8, n_pairs=4, cuda_required=False)
    r = DSNeRVRenderer(cfg)
    table = DSNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    sched = NoiseSchedule(n_steps=cfg.n_diffusion_steps, sigma_max=cfg.sigma_max)
    g = torch.Generator().manual_seed(0)
    result = train_step_dsnerv(
        renderer=r, latent_table=table,
        pair_indices=torch.tensor([0]),
        gt_pairs_uint8=torch.randint(0, 256, (1, 2, 3, 64, 64), dtype=torch.uint8),
        scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
        seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
        lambda_seg=1.0, lambda_pose=1.0,
        noise_schedule=sched,
        generator=g,
    )
    assert 1 <= result["diffusion_step"] <= cfg.n_diffusion_steps


# ── Quantization + archive ───────────────────────────────────────────────


def test_quantize_per_tensor_int8_round_trip():
    t = torch.randn(8, 4) * 0.1
    q, scale = _quantize_per_tensor_int8_with_fp16_scale(t)
    rec = q.float() * scale.float()
    assert torch.allclose(rec, t, atol=0.01, rtol=0.05)


def test_export_dsnerv_archive_returns_sha(tmp_path):
    cfg = DSNeRVConfig(latent_dim=4, base_channels=8, n_pairs=4, cuda_required=False)
    r = DSNeRVRenderer(cfg)
    table = DSNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    out = tmp_path / "0.bin"
    sha = export_dsnerv_to_archive(renderer=r, latent_table=table, output_path=out)
    assert isinstance(sha, str) and len(sha) == 64
    assert out.exists()


def test_export_dsnerv_header_well_formed(tmp_path):
    cfg = DSNeRVConfig(latent_dim=4, base_channels=8, n_pairs=4, cuda_required=False)
    r = DSNeRVRenderer(cfg)
    table = DSNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    out = tmp_path / "0.bin"
    export_dsnerv_to_archive(renderer=r, latent_table=table, output_path=out)
    blob = out.read_bytes()
    assert blob[:4] == DSNERV_MAGIC
    (ver,) = struct.unpack_from("<H", blob, 4)
    assert ver == DSNERV_FORMAT_VERSION
    (fid,) = struct.unpack_from("<H", blob, 6)
    assert fid == DSNERV_FORMAT_ID


def test_export_dsnerv_deterministic(tmp_path):
    cfg = DSNeRVConfig(latent_dim=4, base_channels=8, n_pairs=4, cuda_required=False)
    torch.manual_seed(0)
    r = DSNeRVRenderer(cfg)
    table = DSNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    sha_a = export_dsnerv_to_archive(renderer=r, latent_table=table, output_path=tmp_path / "a.bin")
    sha_b = export_dsnerv_to_archive(renderer=r, latent_table=table, output_path=tmp_path / "b.bin")
    assert sha_a == sha_b


# ── Smoke + grammar ──────────────────────────────────────────────────────


def test_make_synthetic_pair_batch_smoke_shape():
    pi, gt = _make_synthetic_pair_batch_for_smoke(
        batch_size=2, latent_dim=4, eval_size=(384, 512), n_pairs=4, seed=0,
    )
    assert pi.shape == (2,)
    assert gt.shape == (2, 2, 3, 874, 1164)


def test_archive_grammar_dsnerv_well_formed():
    g = ARCHIVE_GRAMMAR_DSNERV
    assert g["format_id"] == DSNERV_FORMAT_ID
    assert g["magic"] == DSNERV_MAGIC.decode("ascii")
    section_names = [s["name"] for s in g["sections"]]
    assert section_names[0] == "header"
    assert "decoder_blob" in section_names
    assert "latent_blob" in section_names
