# SPDX-License-Identifier: MIT
"""Tests for HiNeRV-as-renderer (hierarchical) substrate."""
from __future__ import annotations

import struct

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.hinerv_as_renderer import (
    ARCHIVE_GRAMMAR_HINERV,
    HINERV_FORMAT_ID,
    HINERV_FORMAT_VERSION,
    HINERV_MAGIC,
    HiNeRVConfig,
    HiNeRVLatentTable,
    HiNeRVRenderer,
    _make_synthetic_pair_batch_for_smoke,
    _quantize_per_tensor_int8_with_fp16_scale,
    default_pose_surrogate,
    default_seg_surrogate,
    export_hinerv_to_archive,
    train_step_hinerv,
)


# ── Config ───────────────────────────────────────────────────────────────


def test_config_default():
    cfg = HiNeRVConfig()
    assert cfg.n_levels == 3
    assert cfg.eval_size == (384, 512)
    assert cfg.base_h * (2 ** (cfg.n_levels - 1)) == cfg.eval_size[0]


def test_config_rejects_zero_latent():
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        HiNeRVConfig(latent_dim=0)


def test_config_rejects_one_level():
    with pytest.raises(ValueError, match="n_levels must be >=2"):
        HiNeRVConfig(n_levels=1)


def test_config_rejects_aux_weight_length_mismatch():
    with pytest.raises(ValueError, match="aux_loss_weights len"):
        HiNeRVConfig(n_levels=3, aux_loss_weights=(0.5, 1.0))


def test_config_rejects_zero_final_aux_weight():
    with pytest.raises(ValueError, match="final-level loss weight"):
        HiNeRVConfig(n_levels=3, aux_loss_weights=(0.25, 0.5, 0.0))


def test_config_rejects_eval_size_mismatch():
    with pytest.raises(ValueError, match="eval_size .* != base"):
        HiNeRVConfig(base_h=24, base_w=32, n_levels=3, eval_size=(100, 100))


def test_config_rejects_two_frames():
    with pytest.raises(ValueError, match="frames_per_pair=2"):
        HiNeRVConfig(frames_per_pair=4)


# ── Renderer forward ─────────────────────────────────────────────────────


def test_renderer_forward_default_shape():
    cfg = HiNeRVConfig()
    r = HiNeRVRenderer(cfg)
    z = torch.randn(2, cfg.latent_dim) * 0.01
    out = r(z)
    assert out.shape == (2, 2, 3, 384, 512)


def test_renderer_forward_with_aux_returns_per_stage_list():
    # Use a small config where the per-stage shapes are easy to assert.
    cfg = HiNeRVConfig(
        latent_dim=4, base_channels=8, n_levels=3,
        base_h=8, base_w=12, eval_size=(32, 48),
        aux_loss_weights=(0.25, 0.5, 1.0), n_pairs=4, cuda_required=False,
    )
    r = HiNeRVRenderer(cfg)
    z = torch.randn(2, cfg.latent_dim) * 0.01
    final, all_rgbs = r(z, forward_with_aux=True)
    assert len(all_rgbs) == cfg.n_levels
    # Stage 0 at base resolution
    assert all_rgbs[0].shape == (2, 2, 3, 8, 12)
    # Stage 1 at 2x base
    assert all_rgbs[1].shape == (2, 2, 3, 16, 24)
    # Stage 2 at 4x base = (32, 48)
    assert all_rgbs[2].shape == (2, 2, 3, 32, 48)


def test_renderer_forward_smaller_config_works():
    cfg = HiNeRVConfig(
        latent_dim=4, base_channels=8, n_levels=2,
        base_h=8, base_w=8,
        eval_size=(16, 16),
        aux_loss_weights=(0.5, 1.0),
    )
    r = HiNeRVRenderer(cfg)
    z = torch.randn(1, cfg.latent_dim) * 0.01
    out = r(z)
    assert out.shape == (1, 2, 3, 16, 16)


def test_renderer_rejects_wrong_latent_shape():
    cfg = HiNeRVConfig()
    r = HiNeRVRenderer(cfg)
    with pytest.raises(ValueError, match="forward expected"):
        r(torch.randn(2, 99))


def test_renderer_output_in_zero_to_255_range():
    cfg = HiNeRVConfig()
    r = HiNeRVRenderer(cfg)
    z = torch.randn(1, cfg.latent_dim) * 0.05
    out = r(z)
    assert (out >= 0).all() and (out <= 255.0).all()


def test_renderer_schema_well_formed():
    cfg = HiNeRVConfig()
    r = HiNeRVRenderer(cfg)
    sd = r.state_dict()
    schema = r.schema
    assert len(schema) > 0
    for key, shape in schema:
        assert key in sd and tuple(sd[key].shape) == shape


def test_renderer_schema_includes_per_stage_keys():
    cfg = HiNeRVConfig()
    r = HiNeRVRenderer(cfg)
    keys = [k for k, _ in r.schema]
    assert "stages.0.stem.weight" in keys
    assert "stages.1.prev_proj.weight" in keys  # i>0 has prev_proj
    assert "stages.2.rgb_0.weight" in keys
    assert "stages.0.prev_proj.weight" not in keys  # i=0 has no prev_proj


# ── Latent table ─────────────────────────────────────────────────────────


def test_latent_table_init_and_forward():
    table = HiNeRVLatentTable(n_pairs=4, latent_dim=16)
    out = table(torch.tensor([0, 2, 1]))
    assert out.shape == (3, 16)


# ── train_step (multi-scale) ─────────────────────────────────────────────


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


def _small_cfg():
    return HiNeRVConfig(
        latent_dim=4, base_channels=8, n_levels=2,
        base_h=8, base_w=8, eval_size=(16, 16), n_pairs=4,
        aux_loss_weights=(0.5, 1.0), cuda_required=False,
    )


def test_train_step_hinerv_returns_full_loss_dict():
    cfg = _small_cfg()
    r = HiNeRVRenderer(cfg)
    table = HiNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    pair_idx = torch.tensor([0, 1])
    gt = torch.randint(0, 256, (2, 2, 3, 32, 32), dtype=torch.uint8)
    result = train_step_hinerv(
        renderer=r, latent_table=table, pair_indices=pair_idx, gt_pairs_uint8=gt,
        scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
        seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
        lambda_seg=100.0, lambda_pose=288.675,
    )
    for key in ("loss", "loss_seg", "loss_pose", "loss_score", "loss_aux", "aux_breakdown"):
        assert key in result


def test_train_step_hinerv_refuses_eval_roundtrip_false():
    cfg = _small_cfg()
    r = HiNeRVRenderer(cfg)
    table = HiNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    with pytest.raises(ValueError, match="eval_roundtrip=False is forbidden"):
        train_step_hinerv(
            renderer=r, latent_table=table,
            pair_indices=torch.tensor([0]),
            gt_pairs_uint8=torch.randint(0, 256, (1, 2, 3, 32, 32), dtype=torch.uint8),
            scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
            seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
            lambda_seg=1.0, lambda_pose=1.0,
            eval_roundtrip=False,
        )


def test_train_step_hinerv_grad_flows_through_aux_loss():
    cfg = _small_cfg()
    r = HiNeRVRenderer(cfg)
    table = HiNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    result = train_step_hinerv(
        renderer=r, latent_table=table,
        pair_indices=torch.tensor([0]),
        gt_pairs_uint8=torch.randint(0, 256, (1, 2, 3, 32, 32), dtype=torch.uint8),
        scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
        seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
        lambda_seg=1.0, lambda_pose=1.0,
    )
    assert torch.isfinite(result["loss"]).all()
    assert result["loss"].requires_grad
    assert result["loss_aux"].requires_grad


def test_train_step_hinerv_aux_breakdown_length_matches_n_levels_minus_one():
    cfg = _small_cfg()
    r = HiNeRVRenderer(cfg)
    table = HiNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    result = train_step_hinerv(
        renderer=r, latent_table=table,
        pair_indices=torch.tensor([0]),
        gt_pairs_uint8=torch.randint(0, 256, (1, 2, 3, 32, 32), dtype=torch.uint8),
        scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
        seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
        lambda_seg=1.0, lambda_pose=1.0,
    )
    # aux_loss_weights[:-1] has n_levels-1 entries
    assert len(result["aux_breakdown"]) == cfg.n_levels - 1


# ── Quantization + archive ───────────────────────────────────────────────


def test_quantize_per_tensor_int8_round_trip():
    t = torch.randn(8, 4) * 0.1
    q, scale = _quantize_per_tensor_int8_with_fp16_scale(t)
    rec = q.float() * scale.float()
    assert torch.allclose(rec, t, atol=0.01, rtol=0.05)


def test_export_hinerv_archive_returns_sha(tmp_path):
    cfg = _small_cfg()
    r = HiNeRVRenderer(cfg)
    table = HiNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    out = tmp_path / "0.bin"
    sha = export_hinerv_to_archive(renderer=r, latent_table=table, output_path=out)
    assert isinstance(sha, str) and len(sha) == 64


def test_export_hinerv_header_well_formed(tmp_path):
    cfg = _small_cfg()
    r = HiNeRVRenderer(cfg)
    table = HiNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    out = tmp_path / "0.bin"
    export_hinerv_to_archive(renderer=r, latent_table=table, output_path=out)
    blob = out.read_bytes()
    assert blob[:4] == HINERV_MAGIC
    (ver,) = struct.unpack_from("<H", blob, 4)
    assert ver == HINERV_FORMAT_VERSION
    (fid,) = struct.unpack_from("<H", blob, 6)
    assert fid == HINERV_FORMAT_ID
    (n_levels,) = struct.unpack_from("<H", blob, 14)
    assert n_levels == cfg.n_levels


def test_export_hinerv_deterministic(tmp_path):
    cfg = _small_cfg()
    torch.manual_seed(0)
    r = HiNeRVRenderer(cfg)
    table = HiNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    sha_a = export_hinerv_to_archive(renderer=r, latent_table=table, output_path=tmp_path / "a.bin")
    sha_b = export_hinerv_to_archive(renderer=r, latent_table=table, output_path=tmp_path / "b.bin")
    assert sha_a == sha_b


# ── Smoke + grammar ──────────────────────────────────────────────────────


def test_make_synthetic_pair_batch_smoke_shape():
    pi, gt = _make_synthetic_pair_batch_for_smoke(
        batch_size=2, latent_dim=4, eval_size=(384, 512), n_pairs=4, seed=0,
    )
    assert pi.shape == (2,)
    assert gt.shape == (2, 2, 3, 874, 1164)


def test_archive_grammar_hinerv_well_formed():
    g = ARCHIVE_GRAMMAR_HINERV
    assert g["format_id"] == HINERV_FORMAT_ID
    section_names = [s["name"] for s in g["sections"]]
    assert "decoder_blob" in section_names
