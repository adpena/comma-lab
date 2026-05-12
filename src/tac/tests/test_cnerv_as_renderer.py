"""Tests for CNeRV-as-renderer substrate (NeRV-family completion)."""
from __future__ import annotations

import struct

import pytest
import torch

from tac.cnerv_as_renderer import (
    ARCHIVE_GRAMMAR_CNERV,
    CNERV_FORMAT_ID,
    CNERV_FORMAT_VERSION,
    CNERV_MAGIC,
    CNeRVConfig,
    CNeRVLatentTable,
    CNeRVRenderer,
    _ConvStem,
    _make_synthetic_pair_batch_for_smoke,
    _quantize_per_tensor_int8_with_fp16_scale,
    default_pose_surrogate,
    default_seg_surrogate,
    export_cnerv_to_archive,
    train_step_cnerv,
)


def test_format_constants():
    assert CNERV_MAGIC == b"CNRV"
    assert CNERV_FORMAT_ID == 0x67
    assert CNERV_FORMAT_VERSION == 1


def test_archive_grammar_blueprint():
    g = ARCHIVE_GRAMMAR_CNERV
    assert g["format_id"] == 0x67
    assert g["magic"] == "CNRV"
    section_names = [s["name"] for s in g["sections"]]
    assert section_names == ["header", "decoder_blob", "scale_table", "latent_blob", "sidecar_blob"]


def test_config_default():
    cfg = CNeRVConfig()
    assert cfg.latent_dim == 16
    assert cfg.base_channels == 36


def test_config_rejects_zero_latent():
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        CNeRVConfig(latent_dim=0)


def test_config_rejects_zero_base_channels():
    with pytest.raises(ValueError, match="base_channels must be positive"):
        CNeRVConfig(base_channels=0)


def test_config_rejects_non_six_stages():
    with pytest.raises(ValueError, match="Phase A pinned at n_stages=6"):
        CNeRVConfig(n_stages=4)


def test_conv_stem_shape():
    stem = _ConvStem(latent_dim=16, base_channels=8, base_h=6, base_w=8)
    z = torch.randn(2, 16)
    out = stem(z)
    assert out.shape == (2, 8, 6, 8)


def test_conv_stem_rejects_wrong_latent():
    stem = _ConvStem(latent_dim=16, base_channels=8, base_h=6, base_w=8)
    with pytest.raises(ValueError):
        stem(torch.randn(2, 32))


def test_conv_stem_pos_bias_param_count():
    """Conv stem must have far fewer params than equivalent Linear stem."""
    stem = _ConvStem(latent_dim=16, base_channels=36, base_h=6, base_w=8)
    n_stem_params = sum(p.numel() for p in stem.parameters())
    # Equivalent Linear would be 16 * (36 * 6 * 8) = 27,648 weight params
    # Conv stem has 16 * 6 * 8 (pos_bias) + 36 * 16 * 1 * 1 (lift) + 36 (bias) = 1,380
    assert n_stem_params < 5000


def test_renderer_forward_shape():
    cfg = CNeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = CNeRVRenderer(cfg)
    z = torch.randn(2, 16)
    out = renderer(z)
    assert out.shape == (2, 2, 3, 384, 512)


def test_renderer_output_in_pixel_range():
    cfg = CNeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = CNeRVRenderer(cfg)
    z = torch.randn(2, 16)
    out = renderer(z)
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_renderer_rejects_wrong_latent():
    cfg = CNeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = CNeRVRenderer(cfg)
    with pytest.raises(ValueError):
        renderer(torch.randn(2, 32))


def test_renderer_schema_includes_conv_stem():
    cfg = CNeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = CNeRVRenderer(cfg)
    keys = [k for k, _ in renderer.schema]
    assert "conv_stem.pos_bias" in keys
    assert "conv_stem.lift.weight" in keys
    assert "conv_stem.lift.bias" in keys


def test_renderer_no_dense_linear_stem():
    """CNeRV's distinguishing feature: NO Linear stem."""
    cfg = CNeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = CNeRVRenderer(cfg)
    # Verify no `stem` attribute (which would be the Linear).
    assert not hasattr(renderer, "stem")


def test_latent_table_shape():
    table = CNeRVLatentTable(n_pairs=600, latent_dim=16)
    out = table(torch.tensor([0, 5]))
    assert out.shape == (2, 16)


def test_train_step_returns_canonical_dict():
    cfg = CNeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = CNeRVRenderer(cfg)
    table = CNeRVLatentTable(n_pairs=4, latent_dim=16)

    class _Scorer(torch.nn.Module):
        def preprocess_input(self, x):
            return x.float() if hasattr(x, "float") else x

        def forward(self, x):
            return x.reshape(x.shape[0], -1, 1, 1)[:, :5]

    scorer = _Scorer()
    out = train_step_cnerv(
        renderer=renderer, latent_table=table,
        pair_indices=torch.tensor([0, 1]),
        gt_pairs_uint8=torch.randint(0, 256, (2, 2, 3, 96, 128), dtype=torch.uint8),
        scorer_seg=scorer, scorer_pose=scorer,
        seg_surrogate=lambda p, t: ((p - t) ** 2).mean(),
        pose_surrogate=lambda p, t: ((p - t) ** 2).mean(),
        lambda_seg=1.0, lambda_pose=1.0,
    )
    assert "loss" in out


def test_train_step_eval_roundtrip_must_be_true():
    cfg = CNeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = CNeRVRenderer(cfg)
    table = CNeRVLatentTable(n_pairs=4, latent_dim=16)
    with pytest.raises(ValueError, match="eval_roundtrip=False is forbidden"):
        train_step_cnerv(
            renderer=renderer, latent_table=table,
            pair_indices=torch.tensor([0]),
            gt_pairs_uint8=torch.randint(0, 256, (1, 2, 3, 96, 128), dtype=torch.uint8),
            scorer_seg=torch.nn.Identity(), scorer_pose=torch.nn.Identity(),
            seg_surrogate=lambda a, b: torch.tensor(0.0),
            pose_surrogate=lambda a, b: torch.tensor(0.0),
            lambda_seg=1.0, lambda_pose=1.0, eval_roundtrip=False,
        )


def test_quantize_per_tensor_int8():
    t = torch.randn(10) * 5.0
    q, scale = _quantize_per_tensor_int8_with_fp16_scale(t)
    assert q.dtype == torch.int8


def test_export_writes_valid_header(tmp_path):
    cfg = CNeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = CNeRVRenderer(cfg)
    table = CNeRVLatentTable(n_pairs=4, latent_dim=16)
    out_path = tmp_path / "0.bin"
    sha = export_cnerv_to_archive(renderer=renderer, latent_table=table, output_path=out_path)
    data = out_path.read_bytes()
    assert data[:4] == CNERV_MAGIC
    assert struct.unpack_from("<H", data, 6)[0] == CNERV_FORMAT_ID
    assert len(sha) == 64


def test_export_archive_size_reasonable(tmp_path):
    cfg = CNeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = CNeRVRenderer(cfg)
    table = CNeRVLatentTable(n_pairs=4, latent_dim=16)
    out_path = tmp_path / "0.bin"
    export_cnerv_to_archive(renderer=renderer, latent_table=table, output_path=out_path)
    assert out_path.stat().st_size < 100_000


def test_synthetic_pair_batch_smoke():
    pi, gt = _make_synthetic_pair_batch_for_smoke(
        batch_size=2, latent_dim=16, eval_size=(384, 512), n_pairs=4, seed=20,
    )
    assert gt.shape == (2, 2, 3, 874, 1164)


def test_default_seg_surrogate_scalar():
    loss = default_seg_surrogate(torch.randn(2, 5, 8, 12), torch.randn(2, 5, 8, 12))
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_default_pose_surrogate_uses_first_six():
    p = torch.randn(2, 12); t = torch.randn(2, 12)
    loss = default_pose_surrogate(p, t)
    p2 = p.clone(); p2[:, 6:] = 99.0
    assert torch.allclose(loss, default_pose_surrogate(p2, t))
