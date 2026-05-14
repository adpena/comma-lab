"""Tests for the L2 SAR coherent pose-pair substrate (SARC).

Covers (a) architecture instantiation + parameter-count predictions, (b) SAR
sparse rFFT roundtrip behavior + temporal-smoothness response, (c) archive
byte-grammar emit/parse byte-for-byte roundtrip, (d) inflate one-video
roundtrip producing the contest raw layout, (e) score-aware loss
differentiability + canonical scorer-preprocess routing.

The score-aware loss tests use the real-scorer test kit shared with sister
substrates (per Catalog #164 the canonical loss helper enforces the
preprocess_input contract).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tac.substrates._shared.inflate_runtime import CAMERA_HW
from tac.substrates._shared.score_aware_loss_real_scorer_test_kit import (
    assert_loss_runs_on_real_segnet,
)
from tac.substrates.sar_coherent_pose_pairs import (
    EVAL_HW,
    NUM_PAIRS,
    PER_PAIR_RESIDUAL_TARGET_BYTES,
    POSE_DIM,
    SARC_MAGIC,
    SARC_SCHEMA_VERSION,
    SARCoherentArchive,
    SARCoherentConfig,
    SARCoherentLossWeights,
    SARCoherentPoseCodec,
    SARCoherentRenderer,
    SARCoherentScoreAwareLoss,
    SARCoherentSubstrate,
    dequantize_per_pair_residual,
    pack_archive,
    parse_archive,
    quantize_per_pair_residual_int8,
)
from tac.substrates.sar_coherent_pose_pairs.archive import (
    SARC_HEADER_SIZE,
    decode_pose_codec_bytes,
    encode_pose_codec_bytes,
)
from tac.substrates.sar_coherent_pose_pairs.inflate import (
    inflate_one_video,
    main_cli,
)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


def test_config_defaults_are_within_budget() -> None:
    cfg = SARCoherentConfig()
    assert cfg.num_pairs == NUM_PAIRS
    assert cfg.pose_dim == POSE_DIM
    assert cfg.output_height == EVAL_HW[0]
    assert cfg.output_width == EVAL_HW[1]
    assert cfg.per_pair_residual_bytes == PER_PAIR_RESIDUAL_TARGET_BYTES
    assert 0.0 < cfg.sar_topk_keep_fraction <= 1.0


def test_config_validation_refuses_invalid_pose_dim() -> None:
    with pytest.raises(ValueError, match="pose_dim must be 6"):
        SARCoherentConfig(pose_dim=4)


def test_config_validation_refuses_zero_residual_bytes() -> None:
    with pytest.raises(ValueError, match="per_pair_residual_bytes"):
        SARCoherentConfig(per_pair_residual_bytes=0)


def test_config_validation_refuses_invalid_topk_fraction() -> None:
    with pytest.raises(ValueError, match="sar_topk_keep_fraction"):
        SARCoherentConfig(sar_topk_keep_fraction=0.0)
    with pytest.raises(ValueError, match="sar_topk_keep_fraction"):
        SARCoherentConfig(sar_topk_keep_fraction=1.5)


def test_substrate_param_count_matches_predicted() -> None:
    cfg = SARCoherentConfig(num_pairs=8, output_height=16, output_width=24)
    sub = SARCoherentSubstrate(cfg)
    actual_renderer = sum(p.numel() for p in sub.renderer.parameters())
    predicted = cfg.predict_renderer_param_count()
    assert actual_renderer == predicted


def test_substrate_renders_pair_with_expected_shape() -> None:
    cfg = SARCoherentConfig(num_pairs=4, output_height=12, output_width=16, hidden_dim=16, num_hidden_layers=2)
    sub = SARCoherentSubstrate(cfg)
    rgb_0, rgb_1 = sub.render_pair(0)
    assert rgb_0.shape == (1, 3, 12, 16)
    assert rgb_1.shape == (1, 3, 12, 16)
    assert torch.all((rgb_0 >= 0.0) & (rgb_0 <= 1.0))
    assert torch.all((rgb_1 >= 0.0) & (rgb_1 <= 1.0))


def test_substrate_render_pair_index_bounds_checked() -> None:
    cfg = SARCoherentConfig(num_pairs=4, output_height=12, output_width=16, hidden_dim=16, num_hidden_layers=2)
    sub = SARCoherentSubstrate(cfg)
    with pytest.raises(IndexError):
        sub.render_pair(99)


# ---------------------------------------------------------------------------
# SAR coherent pose codec
# ---------------------------------------------------------------------------


def test_pose_codec_topk_count_matches_fraction() -> None:
    cfg = SARCoherentConfig(sar_topk_keep_fraction=0.10)
    expected_K = max(1, int(round(0.10 * (NUM_PAIRS // 2 + 1))))
    assert cfg.sar_topk() == expected_K


def test_pose_codec_sparse_rfft_roundtrip_is_lossy_but_bounded() -> None:
    cfg = SARCoherentConfig(num_pairs=64)
    codec = SARCoherentPoseCodec(cfg)
    # Inject a temporally-smooth signal (pure low-frequency sinusoid).
    t = torch.linspace(0.0, 1.0, cfg.num_pairs).unsqueeze(-1).expand(-1, cfg.pose_dim)
    with torch.no_grad():
        codec.pose_deltas.copy_(0.5 * torch.sin(2 * 3.14159 * 2.0 * t))
    sparse, indices = codec.encode_sparse_rfft()
    recovered = codec.decode_from_sparse_rfft(sparse)
    rms = (codec.pose_deltas - recovered).pow(2).mean().sqrt().item()
    # Smooth signal should reconstruct with low RMS error.
    assert rms < 0.1, f"smooth signal RMS too high: {rms}"


def test_pose_codec_topk_indices_shape() -> None:
    cfg = SARCoherentConfig(num_pairs=32)
    codec = SARCoherentPoseCodec(cfg)
    sparse, indices = codec.encode_sparse_rfft()
    K = cfg.sar_topk()
    assert indices.shape == (K, cfg.pose_dim)
    n_rfft_bins = cfg.num_pairs // 2 + 1
    assert sparse.shape == (n_rfft_bins, cfg.pose_dim)


def test_pose_codec_int16_byte_estimate_is_closed_form() -> None:
    cfg = SARCoherentConfig(num_pairs=NUM_PAIRS, sar_topk_keep_fraction=0.10)
    codec = SARCoherentPoseCodec(cfg)
    K = cfg.sar_topk()
    expected = K * cfg.pose_dim * 6 + 8
    assert codec.estimate_int16_bytes() == expected


# ---------------------------------------------------------------------------
# Pose codec bytes encode/decode roundtrip
# ---------------------------------------------------------------------------


def test_pose_codec_bytes_roundtrip_byte_perfect() -> None:
    cfg = SARCoherentConfig(num_pairs=32)
    codec = SARCoherentPoseCodec(cfg)
    with torch.no_grad():
        codec.pose_deltas.copy_(0.3 * torch.randn_like(codec.pose_deltas))
    sparse, indices = codec.encode_sparse_rfft()
    encoded = encode_pose_codec_bytes(sparse, indices, int16_scale=cfg.sar_int16_scale)
    expected_len = cfg.sar_topk() * cfg.pose_dim * 6
    assert len(encoded) == expected_len
    n_rfft_bins = cfg.num_pairs // 2 + 1
    decoded_sparse = decode_pose_codec_bytes(
        encoded,
        n_rfft_bins=n_rfft_bins,
        pose_dim=cfg.pose_dim,
        int16_scale=cfg.sar_int16_scale,
    )
    # Compare nonzero positions match within int16-quant precision.
    diff = (decoded_sparse - sparse).abs().max().item()
    assert diff < 1.0 / cfg.sar_int16_scale * 2.0, f"int16 quant drift too high: {diff}"


def test_pose_codec_bytes_decode_refuses_corrupt_length() -> None:
    with pytest.raises(ValueError, match="not divisible by 6"):
        decode_pose_codec_bytes(b"\x00\x01\x02", n_rfft_bins=10, pose_dim=6, int16_scale=256.0)


# ---------------------------------------------------------------------------
# Per-pair residual quantize/dequantize
# ---------------------------------------------------------------------------


def test_per_pair_residual_quant_dequant_is_int8_bounded() -> None:
    res = torch.randn(8, PER_PAIR_RESIDUAL_TARGET_BYTES) * 0.5
    q = quantize_per_pair_residual_int8(res, scale=64.0)
    assert q.dtype == np.int8
    assert q.shape == (8, PER_PAIR_RESIDUAL_TARGET_BYTES)
    dq = dequantize_per_pair_residual(q, scale=64.0)
    # int8 quantization at scale=64 → step size = 1/64 ≈ 0.0156.
    assert (res - dq).abs().max().item() < 1.0 / 64.0


def test_per_pair_residual_quant_refuses_non_2d() -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        quantize_per_pair_residual_int8(torch.randn(10), scale=64.0)


# ---------------------------------------------------------------------------
# Archive byte-grammar emit/parse roundtrip
# ---------------------------------------------------------------------------


def test_archive_header_size_invariant() -> None:
    assert SARC_HEADER_SIZE == 35


def test_archive_pack_parse_roundtrip_byte_perfect() -> None:
    cfg = SARCoherentConfig(num_pairs=8, output_height=12, output_width=16, hidden_dim=16, num_hidden_layers=2)
    sub = SARCoherentSubstrate(cfg)
    sparse, indices = sub.pose_codec.encode_sparse_rfft()
    pose_bytes = encode_pose_codec_bytes(sparse, indices, int16_scale=cfg.sar_int16_scale)
    residual = quantize_per_pair_residual_int8(
        torch.randn(cfg.num_pairs, cfg.per_pair_residual_bytes) * 0.1,
        scale=64.0,
    )
    meta = {
        "int8_scale": 64.0,
        "sar_int16_scale": cfg.sar_int16_scale,
        "first_omega": cfg.first_omega,
        "hidden_omega": cfg.hidden_omega,
        "coord_feature_freqs": cfg.coord_feature_freqs,
        "sar_topk_keep_fraction": cfg.sar_topk_keep_fraction,
    }
    blob = pack_archive(
        renderer_state_dict=sub.state_dict(),
        pose_codec_bytes=pose_bytes,
        per_pair_residual=residual,
        meta=meta,
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        pose_dim=cfg.pose_dim,
        pose_code_dim=cfg.pose_code_dim,
        per_pair_residual_bytes=cfg.per_pair_residual_bytes,
        sar_topk=cfg.sar_topk(),
    )
    # Magic + version sanity.
    assert blob[:4] == SARC_MAGIC
    assert blob[4] == SARC_SCHEMA_VERSION

    parsed = parse_archive(blob)
    assert parsed.num_pairs == cfg.num_pairs
    assert parsed.hidden_dim == cfg.hidden_dim
    assert parsed.pose_dim == cfg.pose_dim
    assert parsed.per_pair_residual_bytes == cfg.per_pair_residual_bytes
    assert parsed.sar_topk == cfg.sar_topk()
    np.testing.assert_array_equal(parsed.per_pair_residual, residual)
    assert parsed.meta["int8_scale"] == 64.0


def test_archive_parse_refuses_bad_magic() -> None:
    with pytest.raises(ValueError, match="bad magic"):
        parse_archive(b"BAD\x00" + b"\x01" + b"\x00" * (SARC_HEADER_SIZE - 5))


def test_archive_parse_refuses_short_blob() -> None:
    with pytest.raises(ValueError, match="archive too short"):
        parse_archive(b"\x00" * 5)


def test_archive_pack_refuses_residual_shape_mismatch() -> None:
    cfg = SARCoherentConfig(num_pairs=8, output_height=12, output_width=16)
    sub = SARCoherentSubstrate(cfg)
    bad_residual = quantize_per_pair_residual_int8(
        torch.randn(99, cfg.per_pair_residual_bytes) * 0.1, scale=64.0
    )
    with pytest.raises(ValueError, match="per_pair_residual shape"):
        pack_archive(
            renderer_state_dict=sub.state_dict(),
            pose_codec_bytes=b"",
            per_pair_residual=bad_residual,
            meta={},
            num_pairs=cfg.num_pairs,
            hidden_dim=cfg.hidden_dim,
            num_hidden_layers=cfg.num_hidden_layers,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
            pose_dim=cfg.pose_dim,
            pose_code_dim=cfg.pose_code_dim,
            per_pair_residual_bytes=cfg.per_pair_residual_bytes,
            sar_topk=cfg.sar_topk(),
        )


def test_archive_pack_bytes_are_deterministic_across_calls() -> None:
    """Two pack calls with identical inputs produce byte-identical archives."""
    cfg = SARCoherentConfig(num_pairs=4, output_height=8, output_width=12, hidden_dim=8, num_hidden_layers=2)
    torch.manual_seed(0)
    sub = SARCoherentSubstrate(cfg)
    sd = sub.state_dict()
    sparse, indices = sub.pose_codec.encode_sparse_rfft()
    pose_bytes = encode_pose_codec_bytes(sparse, indices, int16_scale=cfg.sar_int16_scale)
    residual = quantize_per_pair_residual_int8(
        torch.zeros(cfg.num_pairs, cfg.per_pair_residual_bytes), scale=64.0
    )
    meta = {"int8_scale": 64.0, "sar_int16_scale": cfg.sar_int16_scale}
    common = dict(
        renderer_state_dict=sd,
        pose_codec_bytes=pose_bytes,
        per_pair_residual=residual,
        meta=meta,
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        pose_dim=cfg.pose_dim,
        pose_code_dim=cfg.pose_code_dim,
        per_pair_residual_bytes=cfg.per_pair_residual_bytes,
        sar_topk=cfg.sar_topk(),
    )
    blob_a = pack_archive(**common)
    blob_b = pack_archive(**common)
    assert blob_a == blob_b
    parsed_a = parse_archive(blob_a)
    parsed_b = parse_archive(blob_b)
    np.testing.assert_array_equal(parsed_a.per_pair_residual, parsed_b.per_pair_residual)
    assert parsed_a.meta == parsed_b.meta
    for key in parsed_a.renderer_state_dict.keys():
        assert torch.allclose(
            parsed_a.renderer_state_dict[key],
            parsed_b.renderer_state_dict[key],
        ), f"state_dict[{key}] must roundtrip identically"


# ---------------------------------------------------------------------------
# Inflate one-video roundtrip (CPU; no scorer)
# ---------------------------------------------------------------------------


def _build_tiny_archive_bytes(tmp_dir: Path, *, num_pairs: int = 2) -> bytes:
    cfg = SARCoherentConfig(
        num_pairs=num_pairs,
        output_height=16,
        output_width=24,
        hidden_dim=16,
        num_hidden_layers=2,
    )
    sub = SARCoherentSubstrate(cfg)
    sparse, indices = sub.pose_codec.encode_sparse_rfft()
    pose_bytes = encode_pose_codec_bytes(sparse, indices, int16_scale=cfg.sar_int16_scale)
    residual = quantize_per_pair_residual_int8(
        torch.zeros(cfg.num_pairs, cfg.per_pair_residual_bytes), scale=64.0
    )
    meta = {
        "int8_scale": 64.0,
        "sar_int16_scale": cfg.sar_int16_scale,
        "first_omega": cfg.first_omega,
        "hidden_omega": cfg.hidden_omega,
        "coord_feature_freqs": cfg.coord_feature_freqs,
        "sar_topk_keep_fraction": cfg.sar_topk_keep_fraction,
    }
    return pack_archive(
        renderer_state_dict=sub.state_dict(),
        pose_codec_bytes=pose_bytes,
        per_pair_residual=residual,
        meta=meta,
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        pose_dim=cfg.pose_dim,
        pose_code_dim=cfg.pose_code_dim,
        per_pair_residual_bytes=cfg.per_pair_residual_bytes,
        sar_topk=cfg.sar_topk(),
    )


def test_inflate_one_video_writes_contest_raw_layout(tmp_path: Path) -> None:
    blob = _build_tiny_archive_bytes(tmp_path, num_pairs=2)
    out_raw = tmp_path / "0.raw"
    n = inflate_one_video(blob, out_raw, device="cpu")
    # 2 pairs × 2 frames/pair = 4 frames, each (874, 1164, 3) uint8.
    assert n == 4
    expected_size = 4 * CAMERA_HW[0] * CAMERA_HW[1] * 3
    assert out_raw.stat().st_size == expected_size


def test_inflate_one_video_refuses_non_existent_archive_fields(tmp_path: Path) -> None:
    blob = _build_tiny_archive_bytes(tmp_path, num_pairs=2)
    # Corrupt the magic.
    corrupt = b"XXXX" + blob[4:]
    with pytest.raises(ValueError, match="bad magic"):
        inflate_one_video(corrupt, tmp_path / "x.raw", device="cpu")


def test_inflate_main_cli_falls_back_to_x_path(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    blob = _build_tiny_archive_bytes(tmp_path, num_pairs=1)
    (archive_dir / "x").write_bytes(blob)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    file_list = tmp_path / "files.txt"
    file_list.write_text("v.mkv\n", encoding="utf-8")

    import sys

    saved_argv = sys.argv
    try:
        sys.argv = ["inflate.py", str(archive_dir), str(output_dir), str(file_list)]
        rc = main_cli()
    finally:
        sys.argv = saved_argv
    assert rc == 0
    assert (output_dir / "v.raw").is_file()


def test_inflate_main_cli_refuses_ambiguous_archive_members(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    blob = _build_tiny_archive_bytes(tmp_path, num_pairs=1)
    (archive_dir / "0.bin").write_bytes(blob)
    (archive_dir / "x").write_bytes(blob)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    file_list = tmp_path / "files.txt"
    file_list.write_text("v.mkv\n", encoding="utf-8")

    import sys

    saved_argv = sys.argv
    try:
        sys.argv = ["inflate.py", str(archive_dir), str(output_dir), str(file_list)]
        with pytest.raises(ValueError, match="ambiguous archive members"):
            main_cli()
    finally:
        sys.argv = saved_argv


# ---------------------------------------------------------------------------
# Score-aware loss differentiability + canonical scorer-preprocess routing
# ---------------------------------------------------------------------------


def _build_loss_factory(seg_scorer, pose_scorer):
    return SARCoherentScoreAwareLoss(
        seg_scorer=seg_scorer,
        pose_scorer=pose_scorer,
        weights=SARCoherentLossWeights(),
    )


def _invoke_loss(loss_fn, ctx):
    return loss_fn(
        ctx["rgb_0"],
        ctx["rgb_1"],
        ctx["gt_0"],
        ctx["gt_1"],
        ctx["bytes_proxy"],
        apply_eval_roundtrip=True,
        noise_std=0.0,
    )


def test_score_aware_loss_runs_on_real_segnet():
    """Defends against re-introduction of the WWW4 5D-vs-4D shape mismatch."""
    assert_loss_runs_on_real_segnet(
        loss_factory=_build_loss_factory,
        invoke_loss=_invoke_loss,
    )


def test_score_aware_loss_routes_through_canonical_score_pair_components(monkeypatch):
    """Loss MUST route through ``score_pair_components`` (Catalog #164)."""
    import tac.differentiable_eval_roundtrip as eval_roundtrip
    import tac.substrates.sar_coherent_pose_pairs.score_aware_loss as soa_mod

    calls: dict[str, object] = {"roundtrip_count": 0}

    def fake_roundtrip(x: torch.Tensor) -> torch.Tensor:
        calls["roundtrip_count"] = int(calls["roundtrip_count"]) + 1
        return x

    def fake_score_pair_components(**kwargs):
        calls["score_pair_components_kwargs"] = kwargs
        return torch.tensor(0.25), torch.tensor(0.04)

    monkeypatch.setattr(
        eval_roundtrip,
        "apply_eval_roundtrip_during_training",
        fake_roundtrip,
    )
    monkeypatch.setattr(soa_mod, "score_pair_components", fake_score_pair_components)

    seg = object()
    pose = object()
    loss_fn = soa_mod.SARCoherentScoreAwareLoss(
        seg_scorer=seg,
        pose_scorer=pose,
        weights=soa_mod.SARCoherentLossWeights(),
    )
    rgb_0 = torch.zeros(1, 3, 4, 4)
    rgb_1 = torch.ones(1, 3, 4, 4)
    gt_0 = torch.full((1, 3, 4, 4), 2.0)
    gt_1 = torch.full((1, 3, 4, 4), 3.0)
    bytes_proxy = torch.tensor(37_545_489.0)

    loss, parts = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bytes_proxy,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    assert calls["roundtrip_count"] == 2
    kwargs = calls["score_pair_components_kwargs"]
    assert kwargs["seg_scorer"] is seg
    assert kwargs["pose_scorer"] is pose
    assert kwargs["rgb_0_rt"] is rgb_0
    assert kwargs["rgb_1_rt"] is rgb_1
    assert kwargs["gt_rgb_0"] is gt_0
    assert kwargs["gt_rgb_1"] is gt_1
    # Loss formula: rate + 100*0.25 + sqrt(10)*sqrt(0.04) = 25 + 25 + ~0.6325
    expected_rate = torch.tensor(25.0)
    expected_seg = torch.tensor(100.0 * 0.25)
    import math
    expected_pose = torch.tensor(math.sqrt(10.0)) * torch.sqrt(torch.tensor(0.04))
    expected_total = expected_rate + expected_seg + expected_pose
    assert torch.allclose(loss, expected_total, atol=1e-5)
    assert "rate_term" in parts and "seg_term" in parts and "pose_term" in parts


def test_score_aware_loss_refuses_eval_roundtrip_false():
    loss = SARCoherentScoreAwareLoss(
        seg_scorer=object(),
        pose_scorer=object(),
        weights=SARCoherentLossWeights(),
    )
    with pytest.raises(ValueError, match="eval_roundtrip"):
        loss(
            torch.zeros(1, 3, 4, 4),
            torch.zeros(1, 3, 4, 4),
            torch.zeros(1, 3, 4, 4),
            torch.zeros(1, 3, 4, 4),
            torch.tensor(1000.0),
            apply_eval_roundtrip=False,
        )


def test_score_aware_loss_temporal_term_penalizes_high_freq_tail(monkeypatch):
    """High-freq pose deltas should add nonzero temporal_term to the loss."""
    import tac.differentiable_eval_roundtrip as eval_roundtrip
    import tac.substrates.sar_coherent_pose_pairs.score_aware_loss as soa_mod

    monkeypatch.setattr(
        eval_roundtrip, "apply_eval_roundtrip_during_training", lambda x: x
    )
    monkeypatch.setattr(
        soa_mod, "score_pair_components",
        lambda **k: (torch.tensor(0.0), torch.tensor(0.0)),
    )

    loss_fn = SARCoherentScoreAwareLoss(
        seg_scorer=object(),
        pose_scorer=object(),
        weights=SARCoherentLossWeights(),
    )
    rgb = torch.zeros(1, 3, 4, 4)
    proxy = torch.tensor(50_000.0)
    pose_deltas = torch.zeros(64, POSE_DIM)
    pose_deltas[::2] = 1.0  # alternating step → all-frequency content.
    _, parts_with = loss_fn(
        rgb, rgb, rgb, rgb, proxy,
        pose_deltas_dense=pose_deltas, apply_eval_roundtrip=True, noise_std=0.0,
    )
    _, parts_without = loss_fn(
        rgb, rgb, rgb, rgb, proxy,
        pose_deltas_dense=None, apply_eval_roundtrip=True, noise_std=0.0,
    )
    assert float(parts_with["temporal_term"]) > 0.0
    assert float(parts_without["temporal_term"]) == 0.0


# ---------------------------------------------------------------------------
# Inflate runtime canonical device-selection path (Catalog #N from BUG-FIX-WAVE)
# ---------------------------------------------------------------------------


def test_inflate_uses_canonical_select_inflate_device(monkeypatch, tmp_path: Path) -> None:
    """Inflate MUST route device selection through ``select_inflate_device``."""
    import tac.substrates.sar_coherent_pose_pairs.inflate as inflate_mod

    sentinel = []
    real_fn = inflate_mod.select_inflate_device

    def _wrap(*args, **kwargs):
        sentinel.append(True)
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(inflate_mod, "select_inflate_device", _wrap)
    blob = _build_tiny_archive_bytes(tmp_path, num_pairs=2)
    out_raw = tmp_path / "ok.raw"
    inflate_one_video(blob, out_raw, device="cpu")
    assert sentinel, "select_inflate_device MUST be invoked by inflate runtime"
