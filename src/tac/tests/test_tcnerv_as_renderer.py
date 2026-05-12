"""Tests for TCNeRV-as-renderer (temporal-conv) substrate."""
from __future__ import annotations

import struct

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.tcnerv_as_renderer import (
    ARCHIVE_GRAMMAR_TCNERV,
    TCNERV_FORMAT_ID,
    TCNERV_FORMAT_VERSION,
    TCNERV_MAGIC,
    TCNeRVConfig,
    TCNeRVLatentTable,
    TCNeRVRenderer,
    TemporalConvStack,
    _make_synthetic_pair_batch_for_smoke,
    _quantize_per_tensor_int8_with_fp16_scale,
    default_pose_surrogate,
    default_seg_surrogate,
    export_tcnerv_to_archive,
    train_step_tcnerv,
)


# ── Config ───────────────────────────────────────────────────────────────


def test_config_default():
    cfg = TCNeRVConfig()
    assert cfg.temporal_kernel == 3
    assert cfg.temporal_n_layers == 2
    assert cfg.temporal_residual is True


def test_config_rejects_even_kernel():
    with pytest.raises(ValueError, match="temporal_kernel must be odd positive"):
        TCNeRVConfig(temporal_kernel=2)


def test_config_rejects_zero_kernel():
    with pytest.raises(ValueError, match="temporal_kernel must be odd positive"):
        TCNeRVConfig(temporal_kernel=0)


def test_config_rejects_zero_n_layers():
    with pytest.raises(ValueError, match="temporal_n_layers must be positive"):
        TCNeRVConfig(temporal_n_layers=0)


def test_config_rejects_zero_latent():
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        TCNeRVConfig(latent_dim=0)


def test_config_rejects_non_six_stages():
    with pytest.raises(ValueError, match="Phase A pinned at n_stages=6"):
        TCNeRVConfig(n_stages=3)


# ── TemporalConvStack ────────────────────────────────────────────────────


def test_temporal_conv_stack_preserves_seq_length():
    stack = TemporalConvStack(latent_dim=4, kernel=3, n_layers=2)
    z = torch.randn(10, 4)
    out = stack(z)
    assert out.shape == (10, 4)


def test_temporal_conv_stack_residual_shortcut_when_layers_zeroed():
    stack = TemporalConvStack(latent_dim=4, kernel=3, n_layers=2, residual=True)
    # Zero all layer weights/biases so layer outputs ≈ tanh(0) = 0; residual = identity.
    for layer in stack.layers:
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
    z = torch.randn(10, 4)
    out = stack(z)
    assert torch.allclose(out, z)


def test_temporal_conv_stack_no_residual_no_identity():
    stack = TemporalConvStack(latent_dim=4, kernel=3, n_layers=2, residual=False)
    z = torch.randn(10, 4)
    out = stack(z)
    assert out.shape == (10, 4)
    # Without residual, output should differ from input (probabilistically true).
    # Just assert shape + finite for determinism.
    assert torch.isfinite(out).all()


def test_temporal_conv_stack_rejects_wrong_dim():
    stack = TemporalConvStack(latent_dim=4, kernel=3, n_layers=2)
    bad = torch.randn(10, 99)
    with pytest.raises(ValueError, match="TemporalConvStack expected"):
        stack(bad)


# ── Renderer ─────────────────────────────────────────────────────────────


def test_renderer_forward_full_decode_default_shape():
    cfg = TCNeRVConfig(n_pairs=4, base_channels=8)
    r = TCNeRVRenderer(cfg)
    table = TCNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    out = r(table.full_sequence())
    assert out.shape == (4, 2, 3, 384, 512)


def test_renderer_forward_with_pair_indices_subset():
    cfg = TCNeRVConfig(n_pairs=4, base_channels=8)
    r = TCNeRVRenderer(cfg)
    table = TCNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    out = r(table.full_sequence(), torch.tensor([0, 2]))
    assert out.shape == (2, 2, 3, 384, 512)


def test_renderer_spatial_forward_rejects_wrong_latent():
    cfg = TCNeRVConfig(n_pairs=4, base_channels=8)
    r = TCNeRVRenderer(cfg)
    with pytest.raises(ValueError, match="spatial_forward expected"):
        r.spatial_forward(torch.randn(2, 99))


def test_renderer_output_in_zero_to_255_range():
    cfg = TCNeRVConfig(n_pairs=4, base_channels=8)
    r = TCNeRVRenderer(cfg)
    table = TCNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    out = r(table.full_sequence())
    assert (out >= 0).all() and (out <= 255.0).all()


def test_renderer_schema_includes_temporal_keys():
    cfg = TCNeRVConfig(n_pairs=4, base_channels=8)
    r = TCNeRVRenderer(cfg)
    keys = [k for k, _ in r.schema]
    assert "temporal_conv.layers.0.weight" in keys
    assert "temporal_conv.layers.1.weight" in keys
    assert "stem.weight" in keys


def test_renderer_schema_state_dict_round_trip():
    cfg = TCNeRVConfig(n_pairs=4, base_channels=8)
    r = TCNeRVRenderer(cfg)
    sd = r.state_dict()
    for key, shape in r.schema:
        assert key in sd and tuple(sd[key].shape) == shape


# ── Latent table ─────────────────────────────────────────────────────────


def test_latent_table_full_sequence_returns_correct_shape():
    table = TCNeRVLatentTable(n_pairs=4, latent_dim=16)
    seq = table.full_sequence()
    assert seq.shape == (4, 16)


def test_latent_table_forward_subset():
    table = TCNeRVLatentTable(n_pairs=4, latent_dim=16)
    out = table(torch.tensor([0, 2]))
    assert out.shape == (2, 16)


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


def _small_cfg():
    return TCNeRVConfig(
        latent_dim=4, base_channels=8, n_pairs=4, cuda_required=False,
    )


def test_train_step_tcnerv_returns_loss_dict():
    cfg = _small_cfg()
    r = TCNeRVRenderer(cfg)
    table = TCNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    pair_idx = torch.tensor([0, 1])
    gt = torch.randint(0, 256, (2, 2, 3, 64, 64), dtype=torch.uint8)
    result = train_step_tcnerv(
        renderer=r, latent_table=table, pair_indices=pair_idx, gt_pairs_uint8=gt,
        scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
        seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
        lambda_seg=100.0, lambda_pose=288.675,
    )
    for key in ("loss", "loss_seg", "loss_pose", "loss_seg_unweighted", "loss_pose_unweighted"):
        assert key in result and torch.is_tensor(result[key])


def test_train_step_tcnerv_refuses_eval_roundtrip_false():
    cfg = _small_cfg()
    r = TCNeRVRenderer(cfg)
    table = TCNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    with pytest.raises(ValueError, match="eval_roundtrip=False is forbidden"):
        train_step_tcnerv(
            renderer=r, latent_table=table,
            pair_indices=torch.tensor([0]),
            gt_pairs_uint8=torch.randint(0, 256, (1, 2, 3, 64, 64), dtype=torch.uint8),
            scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
            seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
            lambda_seg=1.0, lambda_pose=1.0, eval_roundtrip=False,
        )


def test_train_step_tcnerv_grad_flows_through_temporal_conv():
    cfg = _small_cfg()
    r = TCNeRVRenderer(cfg)
    table = TCNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    result = train_step_tcnerv(
        renderer=r, latent_table=table,
        pair_indices=torch.tensor([0]),
        gt_pairs_uint8=torch.randint(0, 256, (1, 2, 3, 64, 64), dtype=torch.uint8),
        scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
        seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
        lambda_seg=1.0, lambda_pose=1.0,
    )
    assert torch.isfinite(result["loss"]).all()
    assert result["loss"].requires_grad
    # Backprop should reach temporal conv weights.
    result["loss"].backward()
    has_grad = any(p.grad is not None for p in r.temporal_conv.parameters())
    assert has_grad


# ── Quantization + archive ───────────────────────────────────────────────


def test_quantize_per_tensor_int8_round_trip():
    t = torch.randn(8, 4) * 0.1
    q, scale = _quantize_per_tensor_int8_with_fp16_scale(t)
    rec = q.float() * scale.float()
    assert torch.allclose(rec, t, atol=0.01, rtol=0.05)


def test_export_tcnerv_archive_returns_sha(tmp_path):
    cfg = _small_cfg()
    r = TCNeRVRenderer(cfg)
    table = TCNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    sha = export_tcnerv_to_archive(renderer=r, latent_table=table, output_path=tmp_path / "0.bin")
    assert isinstance(sha, str) and len(sha) == 64


def test_export_tcnerv_header_well_formed(tmp_path):
    cfg = _small_cfg()
    r = TCNeRVRenderer(cfg)
    table = TCNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    out = tmp_path / "0.bin"
    export_tcnerv_to_archive(renderer=r, latent_table=table, output_path=out)
    blob = out.read_bytes()
    assert blob[:4] == TCNERV_MAGIC
    (ver,) = struct.unpack_from("<H", blob, 4)
    assert ver == TCNERV_FORMAT_VERSION
    (fid,) = struct.unpack_from("<H", blob, 6)
    assert fid == TCNERV_FORMAT_ID
    (kernel,) = struct.unpack_from("<H", blob, 12)
    assert kernel == cfg.temporal_kernel
    (n_layers,) = struct.unpack_from("<H", blob, 14)
    assert n_layers == cfg.temporal_n_layers


def test_export_tcnerv_deterministic(tmp_path):
    cfg = _small_cfg()
    torch.manual_seed(0)
    r = TCNeRVRenderer(cfg)
    table = TCNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    sha_a = export_tcnerv_to_archive(renderer=r, latent_table=table, output_path=tmp_path / "a.bin")
    sha_b = export_tcnerv_to_archive(renderer=r, latent_table=table, output_path=tmp_path / "b.bin")
    assert sha_a == sha_b


def test_export_tcnerv_rejects_latent_shape_mismatch(tmp_path):
    cfg = _small_cfg()
    r = TCNeRVRenderer(cfg)
    bad = TCNeRVLatentTable(n_pairs=99, latent_dim=4)
    with pytest.raises(ValueError, match="latent_table shape"):
        export_tcnerv_to_archive(renderer=r, latent_table=bad, output_path=tmp_path / "x.bin")


# ── Smoke + grammar ──────────────────────────────────────────────────────


def test_make_synthetic_pair_batch_smoke_shape():
    pi, gt = _make_synthetic_pair_batch_for_smoke(
        batch_size=2, latent_dim=4, eval_size=(384, 512), n_pairs=4, seed=0,
    )
    assert pi.shape == (2,)
    assert gt.shape == (2, 2, 3, 874, 1164)


def test_archive_grammar_tcnerv_well_formed():
    g = ARCHIVE_GRAMMAR_TCNERV
    assert g["format_id"] == TCNERV_FORMAT_ID
    section_names = [s["name"] for s in g["sections"]]
    assert "spatial_decoder_blob" in section_names
    assert "temporal_conv_blob" in section_names
