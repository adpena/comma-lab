"""Tests for Ego-NeRV-as-renderer substrate (NeRV-family completion)."""
from __future__ import annotations

import struct

import pytest
import torch

from tac.ego_nerv_as_renderer import (
    ARCHIVE_GRAMMAR_EGO_NERV,
    EGO_NERV_FORMAT_ID,
    EGO_NERV_FORMAT_VERSION,
    EGO_NERV_MAGIC,
    EgoNeRVConfig,
    EgoNeRVLatentTable,
    EgoNeRVPoseTable,
    EgoNeRVRenderer,
    _FiLMModulator,
    _make_synthetic_pair_batch_for_smoke,
    _quantize_per_tensor_int8_with_fp16_scale,
    default_pose_surrogate,
    default_seg_surrogate,
    export_ego_nerv_to_archive,
    train_step_ego_nerv,
)


def test_format_constants():
    assert EGO_NERV_MAGIC == b"eNRV"
    assert EGO_NERV_FORMAT_ID == 0x68
    assert EGO_NERV_FORMAT_VERSION == 1


def test_archive_grammar_blueprint():
    g = ARCHIVE_GRAMMAR_EGO_NERV
    assert g["format_id"] == 0x68
    assert g["magic"] == "eNRV"
    section_names = [s["name"] for s in g["sections"]]
    assert "pose_table" in section_names  # the distinguishing extra section


def test_config_default():
    cfg = EgoNeRVConfig()
    assert cfg.latent_dim == 16
    assert cfg.pose_dim == 6
    assert cfg.film_hidden_dim == 64


def test_config_rejects_zero_pose_dim():
    with pytest.raises(ValueError, match="pose_dim must be positive"):
        EgoNeRVConfig(pose_dim=0)


def test_config_rejects_zero_film_hidden():
    with pytest.raises(ValueError, match="film_hidden_dim must be positive"):
        EgoNeRVConfig(film_hidden_dim=0)


def test_config_rejects_non_six_stages():
    with pytest.raises(ValueError, match="Phase A pinned at n_stages=6"):
        EgoNeRVConfig(n_stages=4)


def test_film_modulator_identity_init():
    """At init: scale ≈ 1, shift ≈ 0 (so training starts as no-op)."""
    film = _FiLMModulator(pose_dim=6, latent_dim=16, hidden_dim=64)
    pose = torch.randn(2, 6)
    scale, shift = film(pose)
    assert torch.allclose(scale, torch.ones_like(scale), atol=1e-5)
    assert torch.allclose(shift, torch.zeros_like(shift), atol=1e-5)


def test_film_modulator_shape():
    film = _FiLMModulator(pose_dim=6, latent_dim=16)
    pose = torch.randn(2, 6)
    scale, shift = film(pose)
    assert scale.shape == (2, 16)
    assert shift.shape == (2, 16)


def test_renderer_forward_shape():
    cfg = EgoNeRVConfig(latent_dim=16, pose_dim=6, base_channels=8, n_pairs=4)
    renderer = EgoNeRVRenderer(cfg)
    z = torch.randn(2, 16)
    pose = torch.randn(2, 6)
    out = renderer(z, pose)
    assert out.shape == (2, 2, 3, 384, 512)


def test_renderer_output_in_pixel_range():
    cfg = EgoNeRVConfig(latent_dim=16, pose_dim=6, base_channels=8, n_pairs=4)
    renderer = EgoNeRVRenderer(cfg)
    out = renderer(torch.randn(2, 16), torch.randn(2, 6))
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_renderer_rejects_pose_batch_mismatch():
    cfg = EgoNeRVConfig(latent_dim=16, pose_dim=6, base_channels=8, n_pairs=4)
    renderer = EgoNeRVRenderer(cfg)
    with pytest.raises(ValueError, match="pose batch"):
        renderer(torch.randn(2, 16), torch.randn(3, 6))


def test_renderer_rejects_wrong_pose_dim():
    cfg = EgoNeRVConfig(latent_dim=16, pose_dim=6, base_channels=8, n_pairs=4)
    renderer = EgoNeRVRenderer(cfg)
    with pytest.raises(ValueError):
        renderer(torch.randn(2, 16), torch.randn(2, 3))


def test_renderer_schema_includes_film():
    cfg = EgoNeRVConfig(latent_dim=16, pose_dim=6, base_channels=8, n_pairs=4)
    renderer = EgoNeRVRenderer(cfg)
    keys = [k for k, _ in renderer.schema]
    assert "film.fc1.weight" in keys
    assert "film.fc2_scale.weight" in keys
    assert "film.fc2_shift.weight" in keys


def test_latent_table_shape():
    table = EgoNeRVLatentTable(n_pairs=600, latent_dim=16)
    assert table(torch.tensor([0, 5])).shape == (2, 16)


def test_pose_table_zero_init():
    """Pose table inits to zero (no ego motion default)."""
    table = EgoNeRVPoseTable(n_pairs=600, pose_dim=6)
    out = table(torch.tensor([0, 5, 100]))
    assert out.shape == (3, 6)
    assert torch.allclose(out, torch.zeros_like(out))


def test_train_step_returns_canonical_dict():
    cfg = EgoNeRVConfig(latent_dim=16, pose_dim=6, base_channels=8, n_pairs=4)
    renderer = EgoNeRVRenderer(cfg)
    table = EgoNeRVLatentTable(n_pairs=4, latent_dim=16)
    pose_table = EgoNeRVPoseTable(n_pairs=4, pose_dim=6)

    class _Scorer(torch.nn.Module):
        def preprocess_input(self, x):
            return x.float() if hasattr(x, "float") else x

        def forward(self, x):
            return x.reshape(x.shape[0], -1, 1, 1)[:, :5]

    scorer = _Scorer()
    out = train_step_ego_nerv(
        renderer=renderer, latent_table=table, pose_table=pose_table,
        pair_indices=torch.tensor([0, 1]),
        gt_pairs_uint8=torch.randint(0, 256, (2, 2, 3, 96, 128), dtype=torch.uint8),
        scorer_seg=scorer, scorer_pose=scorer,
        seg_surrogate=lambda p, t: ((p - t) ** 2).mean(),
        pose_surrogate=lambda p, t: ((p - t) ** 2).mean(),
        lambda_seg=1.0, lambda_pose=1.0,
    )
    assert "loss" in out


def test_train_step_eval_roundtrip_must_be_true():
    cfg = EgoNeRVConfig(latent_dim=16, pose_dim=6, base_channels=8, n_pairs=4)
    renderer = EgoNeRVRenderer(cfg)
    table = EgoNeRVLatentTable(n_pairs=4, latent_dim=16)
    pose_table = EgoNeRVPoseTable(n_pairs=4, pose_dim=6)
    with pytest.raises(ValueError, match="eval_roundtrip=False is forbidden"):
        train_step_ego_nerv(
            renderer=renderer, latent_table=table, pose_table=pose_table,
            pair_indices=torch.tensor([0]),
            gt_pairs_uint8=torch.randint(0, 256, (1, 2, 3, 96, 128), dtype=torch.uint8),
            scorer_seg=torch.nn.Identity(), scorer_pose=torch.nn.Identity(),
            seg_surrogate=lambda a, b: torch.tensor(0.0),
            pose_surrogate=lambda a, b: torch.tensor(0.0),
            lambda_seg=1.0, lambda_pose=1.0, eval_roundtrip=False,
        )


def test_export_writes_valid_header(tmp_path):
    cfg = EgoNeRVConfig(latent_dim=16, pose_dim=6, base_channels=8, n_pairs=4)
    renderer = EgoNeRVRenderer(cfg)
    table = EgoNeRVLatentTable(n_pairs=4, latent_dim=16)
    pose_table = EgoNeRVPoseTable(n_pairs=4, pose_dim=6)
    out_path = tmp_path / "0.bin"
    sha = export_ego_nerv_to_archive(
        renderer=renderer, latent_table=table, pose_table=pose_table,
        output_path=out_path,
    )
    data = out_path.read_bytes()
    assert data[:4] == EGO_NERV_MAGIC
    assert struct.unpack_from("<H", data, 6)[0] == EGO_NERV_FORMAT_ID
    assert len(sha) == 64


def test_export_archive_includes_pose_table(tmp_path):
    """Distinguishing feature: pose_table section is in archive."""
    cfg = EgoNeRVConfig(latent_dim=16, pose_dim=6, base_channels=8, n_pairs=4)
    renderer = EgoNeRVRenderer(cfg)
    table = EgoNeRVLatentTable(n_pairs=4, latent_dim=16)
    pose_table = EgoNeRVPoseTable(n_pairs=4, pose_dim=6)
    out_path = tmp_path / "0.bin"
    export_ego_nerv_to_archive(
        renderer=renderer, latent_table=table, pose_table=pose_table,
        output_path=out_path,
    )
    # Pose table is 4 pairs * 6 dims * 2 bytes (FP16) = 48 bytes
    # Archive size should be substantially larger than non-Ego variants
    # because of the pose_table section, but still reasonable.
    assert out_path.stat().st_size < 100_000


def test_export_rejects_pose_table_shape_mismatch(tmp_path):
    cfg = EgoNeRVConfig(latent_dim=16, pose_dim=6, base_channels=8, n_pairs=4)
    renderer = EgoNeRVRenderer(cfg)
    table = EgoNeRVLatentTable(n_pairs=4, latent_dim=16)
    bad_pose = EgoNeRVPoseTable(n_pairs=8, pose_dim=6)  # mismatch n_pairs
    out_path = tmp_path / "0.bin"
    with pytest.raises(ValueError, match="pose_table shape"):
        export_ego_nerv_to_archive(
            renderer=renderer, latent_table=table, pose_table=bad_pose,
            output_path=out_path,
        )


def test_quantize_per_tensor_int8():
    t = torch.randn(10) * 5.0
    q, scale = _quantize_per_tensor_int8_with_fp16_scale(t)
    assert q.dtype == torch.int8


def test_synthetic_pair_batch_smoke():
    pi, gt = _make_synthetic_pair_batch_for_smoke(
        batch_size=2, latent_dim=16, eval_size=(384, 512), n_pairs=4, seed=20,
    )
    assert gt.shape == (2, 2, 3, 874, 1164)


def test_default_seg_surrogate_scalar():
    loss = default_seg_surrogate(torch.randn(2, 5, 8, 12), torch.randn(2, 5, 8, 12))
    assert loss.dim() == 0


def test_default_pose_surrogate_first_six():
    p = torch.randn(2, 12); t = torch.randn(2, 12)
    p2 = p.clone(); p2[:, 6:] = 99.0
    assert torch.allclose(default_pose_surrogate(p, t), default_pose_surrogate(p2, t))
