# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_proof for hi_nerv.

Proves the encode/decode contract of the HIV1 monolithic 0.bin grammar and
the substrate's forward-pass parity under fp16 + per-scale int16-quant
roundtrip across the 3-scale latent pyramid.
"""

from __future__ import annotations

import torch

from tac.substrates.hi_nerv.architecture import (
    HINERV_OFFICIAL_FEATURE_GRID_CONVNEXT_PROOF,
    LATENT_STATE_KEYS,
    ConvNeXtBlock,
    HierarchicalFeatureGrid,
    HinervConfig,
    HinervSubstrate,
    trilinear_upsample,
)
from tac.substrates.hi_nerv.archive import (
    HIV1_HEADER_SIZE,
    HIV1_MAGIC,
    HIV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
    repack_archive_decoder_codec,
    split_archive_sections,
)
from tac.substrates.hi_nerv.bitstream import (
    measure_hi_nerv_decoder_bitstream_roundtrip,
    prepare_hi_nerv_decoder_bitstream_state,
)
from tac.substrates.hi_nerv.inflate import build_model_from_archive


def _smoke_cfg() -> HinervConfig:
    return HinervConfig(
        latent_dim_coarse=4,
        latent_dim_mid=6,
        latent_dim_fine=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def _smoke_meta(cfg: HinervConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "mid_injection_block_index": cfg.mid_injection_block_index,
        "fine_injection_block_index": cfg.fine_injection_block_index,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "use_hierarchical_feature_grid": cfg.use_hierarchical_feature_grid,
        "use_convnext_blocks": cfg.use_convnext_blocks,
        "local_grid_levels": cfg.local_grid_levels,
        "local_grid_channels": cfg.local_grid_channels,
        "convnext_mlp_ratio": cfg.convnext_mlp_ratio,
        "convnext_kernel_size": cfg.convnext_kernel_size,
    }


# ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract
def test_archive_pack_then_parse_roundtrip_recovers_tensors():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = HinervSubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }
    lc = sd["latents_coarse"].clone()
    lm = sd["latents_mid"].clone()
    lf = sd["latents_fine"].clone()

    blob = pack_archive(decoder_sd, lc, lm, lf, _smoke_meta(cfg))
    arc = parse_archive(blob)

    assert arc.schema_version == HIV1_SCHEMA_VERSION
    assert blob[:4] == HIV1_MAGIC
    assert set(arc.decoder_state_dict.keys()) == set(decoder_sd.keys())

    assert arc.latents_coarse.shape == lc.shape
    assert arc.latents_mid.shape == lm.shape
    assert arc.latents_fine.shape == lf.shape

    for lat, ref in (
        (arc.latents_coarse, lc),
        (arc.latents_mid, lm),
        (arc.latents_fine, lf),
    ):
        quant_range = max(float(ref.max() - ref.min()), 1e-12)
        step = quant_range / 65534.0
        assert torch.allclose(lat, ref, atol=step * 2.0)


def test_header_size_invariant_is_33_bytes():
    assert HIV1_HEADER_SIZE == 33


def test_parse_archive_rejects_short_blob():
    try:
        parse_archive(b"\x00" * 8)
    except ValueError as exc:
        assert "too short" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on short blob")


def test_parse_archive_rejects_wrong_magic():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = HinervSubstrate(cfg)
    decoder_sd = {
        k: v for k, v in model.state_dict().items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }
    sd = model.state_dict()
    blob = bytearray(
        pack_archive(
            decoder_sd,
            sd["latents_coarse"].clone(),
            sd["latents_mid"].clone(),
            sd["latents_fine"].clone(),
            _smoke_meta(cfg),
        )
    )
    blob[:4] = b"XXXX"
    try:
        parse_archive(bytes(blob))
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bad magic")


def test_forward_pass_after_roundtrip_matches_original_within_tolerance():
    cfg = _smoke_cfg()
    torch.manual_seed(7)
    model = HinervSubstrate(cfg).eval()

    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, rgb_1_a = model(idx)

    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }
    blob = pack_archive(
        decoder_sd,
        sd["latents_coarse"].clone(),
        sd["latents_mid"].clone(),
        sd["latents_fine"].clone(),
        _smoke_meta(cfg),
    )
    arc = parse_archive(blob)

    rebuilt = HinervSubstrate(cfg).eval()
    rebuilt.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        rebuilt.latents_coarse.copy_(arc.latents_coarse.to(rebuilt.latents_coarse.dtype))
        rebuilt.latents_mid.copy_(arc.latents_mid.to(rebuilt.latents_mid.dtype))
        rebuilt.latents_fine.copy_(arc.latents_fine.to(rebuilt.latents_fine.dtype))
        rgb_0_b, rgb_1_b = rebuilt(idx)

    assert torch.allclose(rgb_0_a, rgb_0_b, atol=5e-2)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=5e-2)


def test_receiver_rejects_missing_decoder_weight_before_rendering():
    cfg = _smoke_cfg()
    torch.manual_seed(23)
    model = HinervSubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }
    decoder_sd.pop("head_rgb_1.bias")

    blob = pack_archive(
        decoder_sd,
        sd["latents_coarse"].clone(),
        sd["latents_mid"].clone(),
        sd["latents_fine"].clone(),
        _smoke_meta(cfg),
    )

    try:
        build_model_from_archive(blob)
    except ValueError as exc:
        assert "hi_nerv_archive_decoder_state invalid" in str(exc)
        assert "head_rgb_1.bias" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected receiver to reject missing decoder weight")


def test_receiver_rejects_unexpected_decoder_weight_before_rendering():
    cfg = _smoke_cfg()
    torch.manual_seed(24)
    model = HinervSubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }
    decoder_sd["extra.weight"] = torch.zeros(1)

    blob = pack_archive(
        decoder_sd,
        sd["latents_coarse"].clone(),
        sd["latents_mid"].clone(),
        sd["latents_fine"].clone(),
        _smoke_meta(cfg),
    )

    try:
        build_model_from_archive(blob)
    except ValueError as exc:
        assert "hi_nerv_archive_decoder_state invalid" in str(exc)
        assert "extra.weight" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected receiver to reject unexpected decoder weight")


def test_receiver_rejects_shape_corrupt_decoder_weight_before_rendering():
    cfg = _smoke_cfg()
    torch.manual_seed(25)
    model = HinervSubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {
        k: v.clone() for k, v in sd.items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }
    decoder_sd["head_rgb_0.bias"] = decoder_sd["head_rgb_0.bias"][:2].clone()

    blob = pack_archive(
        decoder_sd,
        sd["latents_coarse"].clone(),
        sd["latents_mid"].clone(),
        sd["latents_fine"].clone(),
        _smoke_meta(cfg),
    )

    try:
        build_model_from_archive(blob)
    except ValueError as exc:
        assert "shape_mismatch" in str(exc)
        assert "head_rgb_0.bias" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected receiver to reject corrupt decoder shape")


def test_receiver_loads_complete_archive_state_strictly(monkeypatch):
    cfg = _smoke_cfg()
    torch.manual_seed(26)
    model = HinervSubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }
    blob = pack_archive(
        decoder_sd,
        sd["latents_coarse"].clone(),
        sd["latents_mid"].clone(),
        sd["latents_fine"].clone(),
        _smoke_meta(cfg),
    )

    observed: dict[str, object] = {}
    original = HinervSubstrate.load_state_dict

    def capture_load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        observed["strict"] = strict
        observed["keys"] = set(state_dict)
        return original(self, state_dict, strict, *args, **kwargs)

    monkeypatch.setattr(
        HinervSubstrate,
        "load_state_dict",
        capture_load_state_dict,
    )

    _, _, rebuilt = build_model_from_archive(blob)

    assert rebuilt.training is False
    assert observed["strict"] is True
    assert set(LATENT_STATE_KEYS).issubset(observed["keys"])
    assert set(decoder_sd).issubset(observed["keys"])


# ENCODE_INFLATE_ROUNDTRIP — Catalog #139 byte-mutation smoke
def test_byte_mutation_changes_inflate_output_no_op_proof():
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = HinervSubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }

    blob_a = pack_archive(
        decoder_sd,
        sd["latents_coarse"].clone(),
        sd["latents_mid"].clone(),
        sd["latents_fine"].clone(),
        _smoke_meta(cfg),
    )

    mutated_fine = sd["latents_fine"].clone()
    mutated_fine[0, 0] = mutated_fine[0, 0] + 1.0
    blob_b = pack_archive(
        decoder_sd,
        sd["latents_coarse"].clone(),
        sd["latents_mid"].clone(),
        mutated_fine,
        _smoke_meta(cfg),
    )

    assert blob_a != blob_b, "no_op_proof: mutating fine latents must change archive bytes"
    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.latents_fine[0, 0], arc_b.latents_fine[0, 0], atol=1e-6)
    # Coarse + mid latents are unchanged
    assert torch.allclose(arc_a.latents_coarse, arc_b.latents_coarse, atol=1e-6)


def test_decoder_bitstream_preparation_and_repack_are_receiver_visible():
    cfg = _smoke_cfg()
    torch.manual_seed(7)
    model = HinervSubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {
        k: v.clone()
        for k, v in sd.items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }

    prepared = prepare_hi_nerv_decoder_bitstream_state(
        decoder_sd,
        pruning_ratio=0.25,
        quant_noise_bits=4,
        quant_noise_scale=0.1,
        quant_noise_seed=123,
    )

    assert prepared.report["shape_preserved"] is True
    assert prepared.report["pruning"]["actual_new_zero_values"] > 0
    assert prepared.report["quant_noise"]["changed_tensor_count"] > 0
    assert any(
        not torch.equal(prepared.state_dict[name], decoder_sd[name])
        for name in decoder_sd
    )

    blob = pack_archive(
        prepared.state_dict,
        sd["latents_coarse"].clone(),
        sd["latents_mid"].clone(),
        sd["latents_fine"].clone(),
        _smoke_meta(cfg),
        decoder_codec="int8_mixed",
    )
    original_sections = split_archive_sections(blob)
    repacked = repack_archive_decoder_codec(blob, decoder_codec="int4_mixed")
    repacked_sections = split_archive_sections(repacked)

    assert repacked != blob
    assert repacked_sections.latents_coarse_blob == original_sections.latents_coarse_blob
    assert repacked_sections.latents_mid_blob == original_sections.latents_mid_blob
    assert repacked_sections.latents_fine_blob == original_sections.latents_fine_blob
    assert repacked_sections.meta["_decoder_state_codec"]["codec"] == "int4_mixed"
    assert parse_archive(repacked).latents_fine.shape == sd["latents_fine"].shape

    roundtrip = measure_hi_nerv_decoder_bitstream_roundtrip(
        decoder_sd,
        decoder_codecs=("int8_mixed", "int4_mixed"),
        pruning_ratio=0.10,
        quant_noise_bits=4,
        quant_noise_scale=0.05,
        quant_noise_seed=321,
    )
    assert roundtrip["score_claim"] is False
    assert len(roundtrip["rows"]) == 2
    assert roundtrip["best_row"]["shape_preserved"] is True
    assert roundtrip["best_row"]["roundtrip_error"]["missing"] == []


def test_forward_pass_produces_unit_interval_rgb():
    """L5 compliance: substrate is a full RGB renderer (not a mask codec)."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = HinervSubstrate(cfg).eval()
    idx = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (1, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (1, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_three_scale_latent_pyramid_is_distinct():
    """Distinctive design check: hi_nerv has 3 separate latent tensors."""
    cfg = _smoke_cfg()
    model = HinervSubstrate(cfg)
    assert hasattr(model, "latents_coarse")
    assert hasattr(model, "latents_mid")
    assert hasattr(model, "latents_fine")
    assert model.latents_coarse.shape == (cfg.num_pairs, cfg.latent_dim_coarse)
    assert model.latents_mid.shape == (cfg.num_pairs, cfg.latent_dim_mid)
    assert model.latents_fine.shape == (cfg.num_pairs, cfg.latent_dim_fine)
    # Three distinct dims (or at least different parameters)
    n_latent_params = (
        model.latents_coarse.numel()
        + model.latents_mid.numel()
        + model.latents_fine.numel()
    )
    assert n_latent_params == cfg.num_pairs * (
        cfg.latent_dim_coarse + cfg.latent_dim_mid + cfg.latent_dim_fine
    )


def test_trilinear_upsample_interpolates_temporal_local_grid() -> None:
    grid = torch.arange(3 * 2 * 2 * 1, dtype=torch.float32).view(3, 2, 2, 1)
    sampled = trilinear_upsample(
        grid,
        torch.tensor([0, 1, 2], dtype=torch.long),
        num_pairs=3,
        target_h=3,
        target_w=4,
        local_scale=2,
    )

    assert sampled.shape == (3, 3, 4, 1)
    assert torch.equal(sampled[0, :, :, 0], torch.tensor([[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1]], dtype=torch.float32))
    assert torch.equal(sampled[2, :, :, 0], torch.tensor([[8, 9, 8, 9], [10, 11, 10, 11], [8, 9, 8, 9]], dtype=torch.float32))
    half_time = trilinear_upsample(
        grid,
        torch.tensor([1], dtype=torch.long),
        num_pairs=5,
        target_h=2,
        target_w=2,
        local_scale=2,
    )
    expected_half = torch.tensor(
        [[[2.0, 3.0], [4.0, 5.0]]],
        dtype=torch.float32,
    )
    assert torch.equal(half_time[:, :, :, 0], expected_half)


def test_official_feature_grid_convnext_mode_is_receiver_visible() -> None:
    cfg = HinervConfig(
        latent_dim_coarse=3,
        latent_dim_mid=4,
        latent_dim_fine=5,
        embed_dim=8,
        initial_grid_h=2,
        initial_grid_w=3,
        decoder_channels=(7, 6),
        sin_frequency=10.0,
        num_upsample_blocks=2,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=4,
        output_height=8,
        output_width=12,
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=2,
        local_grid_channels=3,
        convnext_mlp_ratio=2,
        convnext_kernel_size=3,
    )
    model = HinervSubstrate(cfg).eval()

    assert HINERV_OFFICIAL_FEATURE_GRID_CONVNEXT_PROOF
    assert any(isinstance(module, HierarchicalFeatureGrid) for module in model.modules())
    assert any(isinstance(module, ConvNeXtBlock) for module in model.modules())
    assert "feature_grids.0.grids.0" in model.state_dict()
    assert "convnext_blocks.0.dwconv.weight" in model.state_dict()

    with torch.no_grad():
        rgb_0, rgb_1 = model(torch.tensor([0, 1], dtype=torch.long))
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0 and float(rgb_0.max()) <= 1.0

    base_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    pair_indices = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        baseline_0, baseline_1 = model(pair_indices)
        model.feature_grids[0].proj.bias.add_(0.25)
        grid_mutated_0, grid_mutated_1 = model(pair_indices)
    grid_delta = max(
        float(torch.max(torch.abs(grid_mutated_0 - baseline_0)).item()),
        float(torch.max(torch.abs(grid_mutated_1 - baseline_1)).item()),
    )
    assert grid_delta > 1.0e-7

    model.load_state_dict(base_state, strict=True)
    with torch.no_grad():
        baseline_0, baseline_1 = model(pair_indices)
        first_convnext = model.convnext_blocks[0]
        assert isinstance(first_convnext, ConvNeXtBlock)
        first_convnext.gamma.add_(0.25)
        first_convnext.pwconv2.bias.add_(0.25)
        convnext_mutated_0, convnext_mutated_1 = model(pair_indices)
    convnext_delta = max(
        float(torch.max(torch.abs(convnext_mutated_0 - baseline_0)).item()),
        float(torch.max(torch.abs(convnext_mutated_1 - baseline_1)).item()),
    )
    assert convnext_delta > 1.0e-7


def test_official_feature_grid_convnext_archive_roundtrip_preserves_forward() -> None:
    cfg = HinervConfig(
        latent_dim_coarse=3,
        latent_dim_mid=4,
        latent_dim_fine=5,
        embed_dim=8,
        initial_grid_h=2,
        initial_grid_w=3,
        decoder_channels=(7, 6),
        sin_frequency=10.0,
        num_upsample_blocks=2,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=3,
        output_height=8,
        output_width=12,
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=2,
        local_grid_channels=3,
        convnext_mlp_ratio=2,
        convnext_kernel_size=3,
    )
    torch.manual_seed(41)
    model = HinervSubstrate(cfg).eval()
    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, rgb_1_a = model(idx)

    sd = model.state_dict()
    decoder_sd = {
        k: v
        for k, v in sd.items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }
    blob = pack_archive(
        decoder_sd,
        sd["latents_coarse"].clone(),
        sd["latents_mid"].clone(),
        sd["latents_fine"].clone(),
        _smoke_meta(cfg),
    )
    _, rebuilt_cfg, rebuilt = build_model_from_archive(blob)
    assert rebuilt_cfg.use_hierarchical_feature_grid is True
    assert rebuilt_cfg.use_convnext_blocks is True
    with torch.no_grad():
        rgb_0_b, rgb_1_b = rebuilt(idx)

    assert torch.allclose(rgb_0_a, rgb_0_b, atol=5e-2)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=5e-2)
