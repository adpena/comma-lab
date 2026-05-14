# SPDX-License-Identifier: MIT
"""Tests for E-NeRV-as-renderer substrate (NeRV-family completion)."""
from __future__ import annotations

import struct

import pytest
import torch

from tac.e_nerv_as_renderer import (
    ARCHIVE_GRAMMAR_E_NERV,
    E_NERV_FORMAT_ID,
    E_NERV_FORMAT_VERSION,
    E_NERV_MAGIC,
    ENeRVConfig,
    ENeRVEncoder,
    ENeRVLatentTable,
    ENeRVRenderer,
    _make_synthetic_pair_batch_for_smoke,
    _quantize_per_tensor_int8_with_fp16_scale,
    default_pose_surrogate,
    default_seg_surrogate,
    export_e_nerv_to_archive,
    train_step_e_nerv,
)


# ── Constants / format ────────────────────────────────────────────────────


def test_format_constants():
    assert E_NERV_MAGIC == b"ENRV"
    assert E_NERV_FORMAT_ID == 0x65
    assert E_NERV_FORMAT_VERSION == 1


def test_archive_grammar_blueprint():
    g = ARCHIVE_GRAMMAR_E_NERV
    assert g["format_id"] == 0x65
    assert g["magic"] == "ENRV"
    section_names = [s["name"] for s in g["sections"]]
    assert section_names == ["header", "decoder_blob", "scale_table", "latent_blob", "sidecar_blob"]


# ── Config ───────────────────────────────────────────────────────────────


def test_config_default_values():
    cfg = ENeRVConfig()
    assert cfg.latent_dim == 16
    assert cfg.encoder_base_channels == 32
    assert cfg.base_channels == 36


def test_config_rejects_zero_latent():
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        ENeRVConfig(latent_dim=0)


def test_config_rejects_zero_encoder_channels():
    with pytest.raises(ValueError, match="encoder_base_channels must be positive"):
        ENeRVConfig(encoder_base_channels=0)


def test_config_rejects_non_six_stages():
    with pytest.raises(ValueError, match="Phase A pinned at n_stages=6"):
        ENeRVConfig(n_stages=4)


def test_config_rejects_invalid_encoder_stages():
    with pytest.raises(ValueError, match="encoder_n_stages must be >= 2"):
        ENeRVConfig(encoder_n_stages=1)


# ── Encoder ─────────────────────────────────────────────────────────────


def test_encoder_forward_shape():
    cfg = ENeRVConfig(latent_dim=16, encoder_base_channels=8, base_channels=8, n_pairs=4)
    enc = ENeRVEncoder(cfg)
    x = torch.randint(0, 256, (2, 2, 3, 96, 128), dtype=torch.uint8)
    z = enc(x)
    assert z.shape == (2, 16)


def test_encoder_rejects_wrong_channel_count():
    cfg = ENeRVConfig(latent_dim=16, encoder_base_channels=8, base_channels=8, n_pairs=4)
    enc = ENeRVEncoder(cfg)
    bad = torch.randint(0, 256, (2, 3, 3, 96, 128), dtype=torch.uint8)  # T=3 instead of 2
    with pytest.raises(ValueError):
        enc(bad)


def test_encoder_rejects_wrong_dim():
    cfg = ENeRVConfig(latent_dim=16, encoder_base_channels=8, base_channels=8, n_pairs=4)
    enc = ENeRVEncoder(cfg)
    bad = torch.randint(0, 256, (2, 3, 96, 128), dtype=torch.uint8)  # 4D not 5D
    with pytest.raises(ValueError):
        enc(bad)


def test_encoder_compress_time_only_param_count():
    """Encoder is COMPRESS-TIME ONLY; should be reasonably small."""
    cfg = ENeRVConfig(encoder_base_channels=32, base_channels=36, n_pairs=600)
    enc = ENeRVEncoder(cfg)
    n_params = sum(p.numel() for p in enc.parameters())
    assert n_params < 5_000_000


# ── Renderer ─────────────────────────────────────────────────────────────


def test_renderer_forward_shape():
    cfg = ENeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = ENeRVRenderer(cfg)
    z = torch.randn(2, 16)
    out = renderer(z)
    assert out.shape == (2, 2, 3, 384, 512)


def test_renderer_output_in_pixel_range():
    cfg = ENeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = ENeRVRenderer(cfg)
    z = torch.randn(2, 16)
    out = renderer(z)
    assert out.min() >= 0.0
    assert out.max() <= 255.0


def test_renderer_rejects_wrong_latent_dim():
    cfg = ENeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = ENeRVRenderer(cfg)
    bad = torch.randn(2, 32)
    with pytest.raises(ValueError):
        renderer(bad)


def test_renderer_schema_is_orderable():
    cfg = ENeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = ENeRVRenderer(cfg)
    schema = renderer.schema
    keys = [k for k, _ in schema]
    assert keys[0] == "stem.weight"
    assert "rgb_0.weight" in keys
    assert "rgb_1.weight" in keys


# ── Latent table ─────────────────────────────────────────────────────────


def test_latent_table_shape():
    table = ENeRVLatentTable(n_pairs=600, latent_dim=16)
    z = table(torch.tensor([0, 5, 100]))
    assert z.shape == (3, 16)


def test_latent_table_populate_from_encoder():
    cfg = ENeRVConfig(latent_dim=16, encoder_base_channels=8, base_channels=8, n_pairs=4)
    enc = ENeRVEncoder(cfg)
    table = ENeRVLatentTable(n_pairs=4, latent_dim=16)
    pairs = torch.randint(0, 256, (4, 2, 3, 96, 128), dtype=torch.uint8)
    table.populate_from_encoder(enc, pairs)
    # The table embedding should now hold encoder outputs (not the random init).
    enc.eval()
    with torch.no_grad():
        expected = enc(pairs[:1])
    assert torch.allclose(table.embedding.weight[:1], expected, atol=1e-4)


def test_latent_table_populate_rejects_wrong_n():
    cfg = ENeRVConfig(latent_dim=16, encoder_base_channels=8, base_channels=8, n_pairs=4)
    enc = ENeRVEncoder(cfg)
    table = ENeRVLatentTable(n_pairs=4, latent_dim=16)
    bad_pairs = torch.randint(0, 256, (5, 2, 3, 96, 128), dtype=torch.uint8)  # 5 != 4
    with pytest.raises(ValueError):
        table.populate_from_encoder(enc, bad_pairs)


# ── train_step ─────────────────────────────────────────────────────────


def test_train_step_returns_canonical_dict():
    cfg = ENeRVConfig(latent_dim=16, encoder_base_channels=8, base_channels=8, n_pairs=4)
    enc = ENeRVEncoder(cfg)
    renderer = ENeRVRenderer(cfg)
    table = ENeRVLatentTable(n_pairs=4, latent_dim=16)
    pair_indices = torch.tensor([0, 1])
    gt_pairs = torch.randint(0, 256, (2, 2, 3, 96, 128), dtype=torch.uint8)

    # Mock scorers as identity-ish modules.
    class _Scorer(torch.nn.Module):
        def preprocess_input(self, x):
            return x.float() if hasattr(x, "float") else x

        def forward(self, x):
            return x.reshape(x.shape[0], -1, 1, 1)[:, :5]

    scorer = _Scorer()

    def seg_surrogate(p, t):
        return ((p - t) ** 2).mean()

    def pose_surrogate(p, t):
        return ((p - t) ** 2).mean()

    out = train_step_e_nerv(
        encoder=enc, renderer=renderer, latent_table=table,
        pair_indices=pair_indices, gt_pairs_uint8=gt_pairs,
        scorer_seg=scorer, scorer_pose=scorer,
        seg_surrogate=seg_surrogate, pose_surrogate=pose_surrogate,
        lambda_seg=1.0, lambda_pose=1.0,
    )
    assert "loss" in out
    assert "loss_seg" in out
    assert "loss_pose" in out


def test_train_step_eval_roundtrip_must_be_true():
    cfg = ENeRVConfig(latent_dim=16, encoder_base_channels=8, base_channels=8, n_pairs=4)
    enc = ENeRVEncoder(cfg)
    renderer = ENeRVRenderer(cfg)
    table = ENeRVLatentTable(n_pairs=4, latent_dim=16)
    with pytest.raises(ValueError, match="eval_roundtrip=False is forbidden"):
        train_step_e_nerv(
            encoder=enc, renderer=renderer, latent_table=table,
            pair_indices=torch.tensor([0]),
            gt_pairs_uint8=torch.randint(0, 256, (1, 2, 3, 96, 128), dtype=torch.uint8),
            scorer_seg=torch.nn.Identity(), scorer_pose=torch.nn.Identity(),
            seg_surrogate=lambda a, b: torch.tensor(0.0),
            pose_surrogate=lambda a, b: torch.tensor(0.0),
            lambda_seg=1.0, lambda_pose=1.0, eval_roundtrip=False,
        )


# ── Quantization + export ─────────────────────────────────────────────


def test_quantize_per_tensor_int8():
    t = torch.randn(10) * 5.0
    q, scale = _quantize_per_tensor_int8_with_fp16_scale(t)
    assert q.dtype == torch.int8
    assert scale.dtype == torch.float16
    assert q.min() >= -128 and q.max() <= 127


def test_export_writes_valid_header(tmp_path):
    cfg = ENeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = ENeRVRenderer(cfg)
    table = ENeRVLatentTable(n_pairs=4, latent_dim=16)
    out_path = tmp_path / "0.bin"
    sha = export_e_nerv_to_archive(
        renderer=renderer, latent_table=table, output_path=out_path,
    )
    data = out_path.read_bytes()
    assert data[:4] == E_NERV_MAGIC
    fid = struct.unpack_from("<H", data, 6)[0]
    assert fid == E_NERV_FORMAT_ID
    assert len(sha) == 64


def test_export_archive_size_reasonable(tmp_path):
    cfg = ENeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = ENeRVRenderer(cfg)
    table = ENeRVLatentTable(n_pairs=4, latent_dim=16)
    out_path = tmp_path / "0.bin"
    export_e_nerv_to_archive(renderer=renderer, latent_table=table, output_path=out_path)
    # Smoke config produces small archive (4 pairs, 8-channel).
    assert out_path.stat().st_size < 100_000


def test_export_rejects_shape_mismatch(tmp_path):
    cfg = ENeRVConfig(latent_dim=16, base_channels=8, n_pairs=4)
    renderer = ENeRVRenderer(cfg)
    bad_table = ENeRVLatentTable(n_pairs=8, latent_dim=16)  # mismatched n_pairs
    out_path = tmp_path / "0.bin"
    with pytest.raises(ValueError, match="latent_table shape"):
        export_e_nerv_to_archive(renderer=renderer, latent_table=bad_table, output_path=out_path)


# ── Smoke helpers / surrogates ───────────────────────────────────────


def test_synthetic_pair_batch_for_smoke():
    pi, gt = _make_synthetic_pair_batch_for_smoke(
        batch_size=2, latent_dim=16, eval_size=(384, 512), n_pairs=4, seed=20,
    )
    assert pi.shape == (2,)
    assert gt.shape == (2, 2, 3, 874, 1164)
    assert gt.dtype == torch.uint8


def test_default_seg_surrogate_returns_scalar():
    p = torch.randn(2, 5, 8, 12)
    t = torch.randn(2, 5, 8, 12)
    loss = default_seg_surrogate(p, t)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_default_pose_surrogate_uses_first_six():
    p = torch.randn(2, 12)
    t = torch.randn(2, 12)
    loss = default_pose_surrogate(p, t)
    assert loss.dim() == 0
    # MSE on first 6 only — verify by perturbing dims 6+.
    p_perturb = p.clone()
    p_perturb[:, 6:] = 99.0
    loss_perturbed = default_pose_surrogate(p_perturb, t)
    assert torch.allclose(loss, loss_perturbed)
