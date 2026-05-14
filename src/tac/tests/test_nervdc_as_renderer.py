# SPDX-License-Identifier: MIT
"""Tests for NeRVdc-as-renderer substrate (NeRV-family completion)."""
from __future__ import annotations

import struct

import pytest
import torch

from tac.nervdc_as_renderer import (
    ARCHIVE_GRAMMAR_NERVDC,
    NERVDC_FORMAT_ID,
    NERVDC_FORMAT_VERSION,
    NERVDC_MAGIC,
    NeRVdcConfig,
    NeRVdcLatentTable,
    NeRVdcRenderer,
    _CondSummary,
    _make_synthetic_pair_batch_for_smoke,
    _quantize_per_tensor_int8_with_fp16_scale,
    default_pose_surrogate,
    default_seg_surrogate,
    export_nervdc_to_archive,
    train_step_nervdc,
)


def test_format_constants():
    assert NERVDC_MAGIC == b"NRVc"
    assert NERVDC_FORMAT_ID == 0x66
    assert NERVDC_FORMAT_VERSION == 1


def test_archive_grammar_blueprint():
    g = ARCHIVE_GRAMMAR_NERVDC
    assert g["format_id"] == 0x66
    assert g["magic"] == "NRVc"
    section_names = [s["name"] for s in g["sections"]]
    assert "header" in section_names
    assert "decoder_blob" in section_names
    assert "latent_blob" in section_names


def test_config_default():
    cfg = NeRVdcConfig()
    assert cfg.latent_dim == 16
    assert cfg.cond_dim == 8


def test_config_rejects_zero_cond_dim():
    with pytest.raises(ValueError, match="cond_dim must be positive"):
        NeRVdcConfig(cond_dim=0)


def test_config_rejects_zero_latent():
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        NeRVdcConfig(latent_dim=0)


def test_config_rejects_non_six_stages():
    with pytest.raises(ValueError, match="Phase A pinned at n_stages=6"):
        NeRVdcConfig(n_stages=4)


def test_cond_summary_shape():
    cs = _CondSummary(cond_dim=8)
    prev = torch.randint(0, 256, (2, 3, 96, 128), dtype=torch.uint8)
    out = cs(prev)
    assert out.shape == (2, 8)


def test_renderer_forward_with_no_prev():
    cfg = NeRVdcConfig(latent_dim=16, cond_dim=8, base_channels=8, n_pairs=4)
    renderer = NeRVdcRenderer(cfg)
    z = torch.randn(2, 16)
    out = renderer(z, prev_frame_uint8=None)  # zero conditioning
    assert out.shape == (2, 2, 3, 384, 512)


def test_renderer_forward_with_prev():
    cfg = NeRVdcConfig(latent_dim=16, cond_dim=8, base_channels=8, n_pairs=4)
    renderer = NeRVdcRenderer(cfg)
    z = torch.randn(2, 16)
    prev = torch.randint(0, 256, (2, 3, 96, 128), dtype=torch.uint8)
    out = renderer(z, prev_frame_uint8=prev)
    assert out.shape == (2, 2, 3, 384, 512)


def test_renderer_rejects_wrong_prev_shape():
    cfg = NeRVdcConfig(latent_dim=16, cond_dim=8, base_channels=8, n_pairs=4)
    renderer = NeRVdcRenderer(cfg)
    z = torch.randn(2, 16)
    bad = torch.randint(0, 256, (2, 6, 96, 128), dtype=torch.uint8)  # 6 channels
    with pytest.raises(ValueError):
        renderer(z, prev_frame_uint8=bad)


def test_renderer_rejects_wrong_latent():
    cfg = NeRVdcConfig(latent_dim=16, cond_dim=8, base_channels=8, n_pairs=4)
    renderer = NeRVdcRenderer(cfg)
    bad = torch.randn(2, 32)
    with pytest.raises(ValueError):
        renderer(bad)


def test_renderer_output_in_pixel_range():
    cfg = NeRVdcConfig(latent_dim=16, cond_dim=8, base_channels=8, n_pairs=4)
    renderer = NeRVdcRenderer(cfg)
    z = torch.randn(2, 16)
    out = renderer(z, prev_frame_uint8=None)
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_renderer_schema_includes_cond_summary():
    cfg = NeRVdcConfig(latent_dim=16, cond_dim=8, base_channels=8, n_pairs=4)
    renderer = NeRVdcRenderer(cfg)
    keys = [k for k, _ in renderer.schema]
    assert "cond_summary.head.weight" in keys
    assert "cond_summary.head.bias" in keys


def test_latent_table_shape():
    table = NeRVdcLatentTable(n_pairs=600, latent_dim=16)
    out = table(torch.tensor([0, 5]))
    assert out.shape == (2, 16)


def test_train_step_returns_canonical_dict():
    cfg = NeRVdcConfig(latent_dim=16, cond_dim=8, base_channels=8, n_pairs=4)
    renderer = NeRVdcRenderer(cfg)
    table = NeRVdcLatentTable(n_pairs=4, latent_dim=16)

    class _Scorer(torch.nn.Module):
        def preprocess_input(self, x):
            return x.float() if hasattr(x, "float") else x

        def forward(self, x):
            return x.reshape(x.shape[0], -1, 1, 1)[:, :5]

    scorer = _Scorer()
    out = train_step_nervdc(
        renderer=renderer, latent_table=table,
        pair_indices=torch.tensor([0, 1]),
        gt_pairs_uint8=torch.randint(0, 256, (2, 2, 3, 96, 128), dtype=torch.uint8),
        prev_frames_uint8=None,
        scorer_seg=scorer, scorer_pose=scorer,
        seg_surrogate=lambda p, t: ((p - t) ** 2).mean(),
        pose_surrogate=lambda p, t: ((p - t) ** 2).mean(),
        lambda_seg=1.0, lambda_pose=1.0,
    )
    assert "loss" in out and "loss_seg" in out and "loss_pose" in out


def test_train_step_eval_roundtrip_must_be_true():
    cfg = NeRVdcConfig(latent_dim=16, cond_dim=8, base_channels=8, n_pairs=4)
    renderer = NeRVdcRenderer(cfg)
    table = NeRVdcLatentTable(n_pairs=4, latent_dim=16)
    with pytest.raises(ValueError, match="eval_roundtrip=False is forbidden"):
        train_step_nervdc(
            renderer=renderer, latent_table=table,
            pair_indices=torch.tensor([0]),
            gt_pairs_uint8=torch.randint(0, 256, (1, 2, 3, 96, 128), dtype=torch.uint8),
            prev_frames_uint8=None,
            scorer_seg=torch.nn.Identity(), scorer_pose=torch.nn.Identity(),
            seg_surrogate=lambda a, b: torch.tensor(0.0),
            pose_surrogate=lambda a, b: torch.tensor(0.0),
            lambda_seg=1.0, lambda_pose=1.0, eval_roundtrip=False,
        )


def test_quantize_per_tensor_int8():
    t = torch.randn(10) * 5.0
    q, scale = _quantize_per_tensor_int8_with_fp16_scale(t)
    assert q.dtype == torch.int8 and scale.dtype == torch.float16


def test_export_writes_valid_header(tmp_path):
    cfg = NeRVdcConfig(latent_dim=16, cond_dim=8, base_channels=8, n_pairs=4)
    renderer = NeRVdcRenderer(cfg)
    table = NeRVdcLatentTable(n_pairs=4, latent_dim=16)
    out_path = tmp_path / "0.bin"
    sha = export_nervdc_to_archive(
        renderer=renderer, latent_table=table, output_path=out_path,
    )
    data = out_path.read_bytes()
    assert data[:4] == NERVDC_MAGIC
    assert struct.unpack_from("<H", data, 6)[0] == NERVDC_FORMAT_ID
    assert len(sha) == 64


def test_export_archive_size_reasonable(tmp_path):
    cfg = NeRVdcConfig(latent_dim=16, cond_dim=8, base_channels=8, n_pairs=4)
    renderer = NeRVdcRenderer(cfg)
    table = NeRVdcLatentTable(n_pairs=4, latent_dim=16)
    out_path = tmp_path / "0.bin"
    export_nervdc_to_archive(renderer=renderer, latent_table=table, output_path=out_path)
    assert out_path.stat().st_size < 100_000


def test_export_reads_all_header_fields(tmp_path):
    cfg = NeRVdcConfig(latent_dim=16, cond_dim=8, base_channels=8, n_pairs=4)
    renderer = NeRVdcRenderer(cfg)
    table = NeRVdcLatentTable(n_pairs=4, latent_dim=16)
    out_path = tmp_path / "0.bin"
    export_nervdc_to_archive(renderer=renderer, latent_table=table, output_path=out_path)
    data = out_path.read_bytes()
    latent_dim = struct.unpack_from("<H", data, 8)[0]
    cond_dim = struct.unpack_from("<H", data, 10)[0]
    n_pairs = struct.unpack_from("<H", data, 12)[0]
    base_channels = struct.unpack_from("<H", data, 14)[0]
    assert latent_dim == 16
    assert cond_dim == 8
    assert n_pairs == 4
    assert base_channels == 8


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
