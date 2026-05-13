"""Tests for BlockNeRV-as-renderer substrate (Phase A scaffold).

Mirrors the structure of test_lane_12_v2_nerv_as_renderer.py per CLAUDE.md
HNeRV parity discipline. All tests run on CPU; CUDA-required is the trainer
default but the substrate scaffold is CUDA-or-CPU agnostic.
"""
from __future__ import annotations

import struct

import pytest
import torch
import torch.nn as nn

from tac.blocknerv_as_renderer import (
    ARCHIVE_GRAMMAR_BLOCKNERV,
    BLOCKNERV_FORMAT_ID,
    BLOCKNERV_FORMAT_VERSION,
    BLOCKNERV_MAGIC,
    BlockNeRVConfig,
    BlockNeRVLatentTable,
    BlockNeRVRenderer,
    _make_synthetic_pair_batch_for_smoke,
    _quantize_per_tensor_int8_with_fp16_scale,
    default_pose_surrogate,
    default_seg_surrogate,
    export_blocknerv_to_archive,
    train_step_blocknerv,
)


# ── Config ───────────────────────────────────────────────────────────────


def test_config_default_constructs():
    cfg = BlockNeRVConfig()
    assert cfg.latent_dim == 16
    assert cfg.tile_h == 64 and cfg.tile_w == 64
    assert cfg.tile_rows == 6 and cfg.tile_cols == 8
    assert cfg.eval_size == (384, 512)
    assert cfg.frames_per_pair == 2


def test_config_rejects_nonpositive_latent_dim():
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        BlockNeRVConfig(latent_dim=0)


def test_config_rejects_eval_size_mismatch():
    with pytest.raises(ValueError, match="eval_size .* != tile_rows"):
        BlockNeRVConfig(eval_size=(100, 100))


def test_config_rejects_indivisible_tile_dims():
    # tile_h must be divisible by 2**n_stages (default n_stages=3 → divisor 8)
    with pytest.raises(ValueError, match="not divisible by"):
        BlockNeRVConfig(
            tile_h=63, tile_w=64,
            tile_rows=6, tile_cols=8,
            eval_size=(63 * 6, 64 * 8),
        )


def test_config_rejects_non_two_frames_per_pair():
    with pytest.raises(ValueError, match="frames_per_pair=2"):
        BlockNeRVConfig(frames_per_pair=3)


def test_config_rejects_zero_coord_embed_dim():
    with pytest.raises(ValueError, match="coord_embed_dim must be positive"):
        BlockNeRVConfig(coord_embed_dim=0)


# ── Renderer forward ─────────────────────────────────────────────────────


def test_renderer_forward_shape_default_config():
    cfg = BlockNeRVConfig()
    r = BlockNeRVRenderer(cfg)
    z = torch.randn(2, cfg.latent_dim) * 0.01
    out = r(z)
    assert out.shape == (2, 2, 3, 384, 512)


def test_renderer_forward_shape_smaller_config():
    cfg = BlockNeRVConfig(
        latent_dim=8, tile_h=8, tile_w=8, tile_rows=2, tile_cols=2,
        coord_embed_dim=4, base_channels=8, n_stages=1,
        eval_size=(16, 16),
    )
    r = BlockNeRVRenderer(cfg)
    z = torch.randn(3, cfg.latent_dim) * 0.01
    out = r(z)
    assert out.shape == (3, 2, 3, 16, 16)


def test_renderer_forward_rejects_wrong_latent_shape():
    cfg = BlockNeRVConfig()
    r = BlockNeRVRenderer(cfg)
    bad = torch.randn(2, 99)
    with pytest.raises(ValueError, match="forward expected"):
        r(bad)


def test_renderer_forward_rejects_wrong_dim_count():
    cfg = BlockNeRVConfig()
    r = BlockNeRVRenderer(cfg)
    bad = torch.randn(2, 3, 16)
    with pytest.raises(ValueError):
        r(bad)


def test_renderer_output_in_zero_to_255_range():
    cfg = BlockNeRVConfig()
    r = BlockNeRVRenderer(cfg)
    z = torch.randn(1, cfg.latent_dim) * 0.05
    out = r(z)
    assert (out >= 0).all() and (out <= 255.0).all()


def test_renderer_schema_is_nonempty_and_well_formed():
    cfg = BlockNeRVConfig()
    r = BlockNeRVRenderer(cfg)
    schema = r.schema
    assert len(schema) > 0
    for key, shape in schema:
        assert isinstance(key, str) and isinstance(shape, tuple)
        assert all(isinstance(d, int) and d > 0 for d in shape)


def test_renderer_schema_round_trip_state_dict_match():
    cfg = BlockNeRVConfig()
    r = BlockNeRVRenderer(cfg)
    sd = r.state_dict()
    for key, shape in r.schema:
        assert key in sd, f"schema key {key} not in state_dict"
        assert tuple(sd[key].shape) == shape


def test_renderer_tile_coord_table_has_correct_shape():
    cfg = BlockNeRVConfig()
    r = BlockNeRVRenderer(cfg)
    n_tiles = cfg.tile_rows * cfg.tile_cols
    assert r.tile_coord.shape == (n_tiles, cfg.coord_embed_dim)


# ── Latent table ─────────────────────────────────────────────────────────


def test_latent_table_init_and_forward():
    table = BlockNeRVLatentTable(n_pairs=4, latent_dim=16)
    idx = torch.tensor([0, 2, 1])
    out = table(idx)
    assert out.shape == (3, 16)


# ── Train step (uses dummy scorers) ──────────────────────────────────────


class _DummyScorerSeg(nn.Module):
    def preprocess_input(self, x):
        # x is (B, F, C, H, W) — return last frame at small resolution.
        if x.dim() == 5:
            x = x[:, -1]
        return F.interpolate(x.float(), size=(64, 64), mode="bilinear", align_corners=False)

    def forward(self, x):
        # 5-class logits, downsampled spatial.
        return torch.randn_like(x[:, :1]).repeat(1, 5, 1, 1)


class _DummyScorerPose(nn.Module):
    def preprocess_input(self, x):
        if x.dim() == 5:
            B, F_pp, C, H, W = x.shape
            x = x.reshape(B, F_pp * C, H, W)
        return F.interpolate(x.float(), size=(32, 32), mode="bilinear", align_corners=False)

    def forward(self, x):
        # 12-dim pose; first 6 used by surrogate.
        x = x.flatten(1)[:, :12]
        return x


import torch.nn.functional as F


def test_train_step_blocknerv_returns_loss_dict():
    cfg = BlockNeRVConfig(
        latent_dim=4, tile_h=8, tile_w=8, tile_rows=2, tile_cols=2,
        coord_embed_dim=2, base_channels=4, n_stages=1,
        eval_size=(16, 16), n_pairs=4,
        cuda_required=False,
    )
    r = BlockNeRVRenderer(cfg)
    table = BlockNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    pair_idx = torch.tensor([0, 1])
    gt = torch.randint(0, 256, (2, 2, 3, 32, 32), dtype=torch.uint8)
    scorer_seg = _DummyScorerSeg()
    scorer_pose = _DummyScorerPose()
    result = train_step_blocknerv(
        renderer=r,
        latent_table=table,
        pair_indices=pair_idx,
        gt_pairs_uint8=gt,
        scorer_seg=scorer_seg,
        scorer_pose=scorer_pose,
        seg_surrogate=default_seg_surrogate,
        pose_surrogate=default_pose_surrogate,
        lambda_seg=100.0,
        lambda_pose=288.675,
    )
    for key in ("loss", "loss_seg", "loss_pose", "loss_seg_unweighted", "loss_pose_unweighted"):
        assert key in result
        assert torch.is_tensor(result[key])


def test_train_step_blocknerv_refuses_eval_roundtrip_false():
    cfg = BlockNeRVConfig(
        latent_dim=4, tile_h=8, tile_w=8, tile_rows=2, tile_cols=2,
        coord_embed_dim=2, base_channels=4, n_stages=1,
        eval_size=(16, 16), n_pairs=4,
        cuda_required=False,
    )
    r = BlockNeRVRenderer(cfg)
    table = BlockNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    pair_idx = torch.tensor([0])
    gt = torch.randint(0, 256, (1, 2, 3, 32, 32), dtype=torch.uint8)
    with pytest.raises(ValueError, match="eval_roundtrip=False is forbidden"):
        train_step_blocknerv(
            renderer=r, latent_table=table, pair_indices=pair_idx, gt_pairs_uint8=gt,
            scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
            seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
            lambda_seg=1.0, lambda_pose=1.0,
            eval_roundtrip=False,
        )


def test_train_step_blocknerv_loss_is_finite_and_grad_attached():
    cfg = BlockNeRVConfig(
        latent_dim=4, tile_h=8, tile_w=8, tile_rows=2, tile_cols=2,
        coord_embed_dim=2, base_channels=4, n_stages=1,
        eval_size=(16, 16), n_pairs=4,
        cuda_required=False,
    )
    r = BlockNeRVRenderer(cfg)
    table = BlockNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    pair_idx = torch.tensor([0])
    gt = torch.randint(0, 256, (1, 2, 3, 32, 32), dtype=torch.uint8)
    result = train_step_blocknerv(
        renderer=r, latent_table=table, pair_indices=pair_idx, gt_pairs_uint8=gt,
        scorer_seg=_DummyScorerSeg(), scorer_pose=_DummyScorerPose(),
        seg_surrogate=default_seg_surrogate, pose_surrogate=default_pose_surrogate,
        lambda_seg=1.0, lambda_pose=1.0,
    )
    assert torch.isfinite(result["loss"]).all()
    assert result["loss"].requires_grad


# ── Quantization ─────────────────────────────────────────────────────────


def test_quantize_per_tensor_int8_round_trip_close():
    t = torch.randn(10, 8) * 0.1
    q, scale = _quantize_per_tensor_int8_with_fp16_scale(t)
    assert q.dtype == torch.int8
    assert scale.dtype == torch.float16
    rec = q.float() * scale.float()
    assert torch.allclose(rec, t, atol=0.01, rtol=0.05)


def test_quantize_per_tensor_int8_handles_zero_tensor():
    t = torch.zeros(5, 3)
    q, scale = _quantize_per_tensor_int8_with_fp16_scale(t)
    # Zero tensor → quantization should not raise; scale ~ 1e-8/127.
    assert q.abs().max().item() == 0
    assert scale.dtype == torch.float16


# ── Archive export ───────────────────────────────────────────────────────


def test_export_blocknerv_to_archive_returns_sha256(tmp_path):
    cfg = BlockNeRVConfig(
        latent_dim=4, tile_h=8, tile_w=8, tile_rows=2, tile_cols=2,
        coord_embed_dim=2, base_channels=4, n_stages=1,
        eval_size=(16, 16), n_pairs=4,
        cuda_required=False,
    )
    r = BlockNeRVRenderer(cfg)
    table = BlockNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    out = tmp_path / "0.bin"
    sha = export_blocknerv_to_archive(
        renderer=r, latent_table=table, output_path=out,
    )
    assert isinstance(sha, str) and len(sha) == 64
    assert out.exists() and out.stat().st_size > 0


def test_export_blocknerv_to_archive_header_well_formed(tmp_path):
    cfg = BlockNeRVConfig(
        latent_dim=4, tile_h=8, tile_w=8, tile_rows=2, tile_cols=2,
        coord_embed_dim=2, base_channels=4, n_stages=1,
        eval_size=(16, 16), n_pairs=4,
        cuda_required=False,
    )
    r = BlockNeRVRenderer(cfg)
    table = BlockNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    out = tmp_path / "0.bin"
    export_blocknerv_to_archive(renderer=r, latent_table=table, output_path=out)
    blob = out.read_bytes()
    assert blob[:4] == BLOCKNERV_MAGIC
    (ver,) = struct.unpack_from("<H", blob, 4)
    assert ver == BLOCKNERV_FORMAT_VERSION
    (fid,) = struct.unpack_from("<H", blob, 6)
    assert fid == BLOCKNERV_FORMAT_ID
    (latent_dim,) = struct.unpack_from("<H", blob, 8)
    assert latent_dim == cfg.latent_dim


def test_export_blocknerv_to_archive_deterministic(tmp_path):
    """Same weights → same archive bytes (no_op_detector evidence)."""
    cfg = BlockNeRVConfig(
        latent_dim=4, tile_h=8, tile_w=8, tile_rows=2, tile_cols=2,
        coord_embed_dim=2, base_channels=4, n_stages=1,
        eval_size=(16, 16), n_pairs=4,
        cuda_required=False,
    )
    torch.manual_seed(0)
    r = BlockNeRVRenderer(cfg)
    table = BlockNeRVLatentTable(cfg.n_pairs, cfg.latent_dim)
    out_a = tmp_path / "a.bin"
    out_b = tmp_path / "b.bin"
    sha_a = export_blocknerv_to_archive(renderer=r, latent_table=table, output_path=out_a)
    sha_b = export_blocknerv_to_archive(renderer=r, latent_table=table, output_path=out_b)
    assert sha_a == sha_b
    assert out_a.read_bytes() == out_b.read_bytes()


def test_export_blocknerv_to_archive_rejects_latent_shape_mismatch(tmp_path):
    cfg = BlockNeRVConfig(
        latent_dim=4, tile_h=8, tile_w=8, tile_rows=2, tile_cols=2,
        coord_embed_dim=2, base_channels=4, n_stages=1,
        eval_size=(16, 16), n_pairs=4,
        cuda_required=False,
    )
    r = BlockNeRVRenderer(cfg)
    bad_table = BlockNeRVLatentTable(n_pairs=99, latent_dim=4)
    with pytest.raises(ValueError, match="latent_table shape"):
        export_blocknerv_to_archive(renderer=r, latent_table=bad_table, output_path=tmp_path / "x.bin")


# ── Smoke synthetic helper ──────────────────────────────────────────────


def test_make_synthetic_pair_batch_smoke_shape():
    pi, gt = _make_synthetic_pair_batch_for_smoke(
        batch_size=2, latent_dim=4, eval_size=(16, 16), n_pairs=4, seed=0,
    )
    assert pi.shape == (2,)
    assert gt.shape == (2, 2, 3, 874, 1164)
    assert gt.dtype == torch.uint8


# ── Archive grammar manifest ────────────────────────────────────────────


def test_archive_grammar_blocknerv_well_formed():
    grammar = ARCHIVE_GRAMMAR_BLOCKNERV
    assert grammar["format_version"] == BLOCKNERV_FORMAT_VERSION
    assert grammar["format_id"] == BLOCKNERV_FORMAT_ID
    assert grammar["magic"] == BLOCKNERV_MAGIC.decode("ascii")
    section_names = [s["name"] for s in grammar["sections"]]
    assert section_names == [
        "header",
        "tile_decoder_blob",
        "tile_coord_table",
        "scale_table",
        "latent_blob",
        "sidecar_blob",
    ]
