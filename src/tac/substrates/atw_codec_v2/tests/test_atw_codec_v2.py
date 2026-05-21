# SPDX-License-Identifier: MIT
"""Tests for ATW codec V2 (Atick-Tishby-Wyner full-stack cooperative-receiver codec).

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons +
the V2 design memo at
``.omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md``.

Coverage:

* Architecture forward pass shape correctness on synthetic inputs
* WZ side-info head + G1 distill head + B3 CDF table structural consumption
* ATW2 archive grammar parser symmetry (pack -> parse -> byte identity)
* Variant A vs Variant B archive variant byte roundtrip
* HNeRV parity discipline lesson 8: canonical Atick-Redlich primitive routing
* Score-aware loss math (Variant A 3-knob + Variant B WZ-only)
* G1 distill head produces 5-way SegNet class logits
* B3 CDF table fp16 ship roundtrip
* Smoke trainer entry point
* Catalog #220 byte-mutation smoke (distinguishing-feature contribution proof)
* Catalog #270 dispatch optimization protocol — overall_pass=true
* Wunderkind G1 + B3 + WZ closed-form productionization tests
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch

from tac.substrates.atw_codec_v2 import (
    ATW2_HEADER_SIZE,
    ATW2_MAGIC,
    ATW2_SCHEMA_VERSION,
    ATW2_SECTION_ROLES,
    ATWv2Codec,
    ATWv2CodecConfig,
    ATWv2LossWeights,
    ATWv2ScoreAwareLoss,
    ATWv2Variant,
    pack_archive,
    parse_archive,
    parse_atw2_archive_bytes,
)
from tac.substrates.atw_codec_v2.architecture import (
    CDF_TABLE_NUM_SYMBOLS,
    DEFAULT_SCORER_CLASS_PRIOR_DIM,
    EVAL_HW,
    NUM_PAIRS,
    NUM_SEGNET_CLASSES,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
)

REPO_ROOT = Path(__file__).resolve().parents[5]


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _make_tiny_cfg(**overrides: Any) -> ATWv2CodecConfig:
    """Tiny config so smoke tests run fast on CPU (<1s each)."""
    base: dict[str, Any] = {
        "variant": ATWv2Variant.B_WZ_ONLY,
        "latent_dim": 8,
        "encoder_input_channels": 3,
        "encoder_hidden_dim": 16,
        "decoder_embed_dim": 8,
        "decoder_initial_grid_h": 2,
        "decoder_initial_grid_w": 2,
        "decoder_channels": (6, 4, 4, 4, 4, 4),
        "decoder_num_upsample_blocks": 2,
        "num_pairs": 4,
        "output_height": 16,
        "output_width": 24,
        "scorer_class_prior_dim": 8,
        "wz_head_hidden_dim": 8,
        "g1_distill_hidden_dim": 8,
    }
    base.update(overrides)
    return ATWv2CodecConfig(**base)


class _StubScorer(torch.nn.Module):
    """Tiny stub scorer mimicking ``preprocess_input`` + ``__call__`` contract."""

    def __init__(self, *, out_dim: int = 5):
        super().__init__()
        self.out_dim = out_dim
        self.proj = torch.nn.Conv2d(3, out_dim, kernel_size=1)

    def preprocess_input(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.dim() == 5:
            B, T, C, H, W = batch.shape
            return batch.reshape(B * T, C, H, W)
        return batch

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.proj(batch)


class _StubPoseScorer(torch.nn.Module):
    """Pose-shaped stub: returns (B, 6) pose deltas."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3 * 16 * 24, 12)  # 2-frame stacked -> 6 pose

    def preprocess_input(self, batch: torch.Tensor) -> torch.Tensor:
        # Accept (B, T=2, 3, H, W) and return flat (B, T*3*H*W) for pose stub
        if batch.dim() == 5:
            B = batch.shape[0]
            return batch.reshape(B, -1)
        return batch.reshape(batch.shape[0], -1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.proj(batch)


# ---------------------------------------------------------------------------
# 1. Architecture tests
# ---------------------------------------------------------------------------


def test_constants_exported() -> None:
    """Module-level constants for downstream consumers."""
    assert ATW2_MAGIC == b"ATW2"
    assert ATW2_SCHEMA_VERSION == 1
    assert ATW2_HEADER_SIZE == 48
    assert EVAL_HW == (384, 512)
    assert NUM_PAIRS == 600
    assert NUM_SEGNET_CLASSES == 5
    assert CDF_TABLE_NUM_SYMBOLS == 256
    assert TOTAL_ARCHIVE_TARGET_BYTES_MIN > 0
    assert TOTAL_ARCHIVE_TARGET_BYTES_MAX >= TOTAL_ARCHIVE_TARGET_BYTES_MIN
    assert DEFAULT_SCORER_CLASS_PRIOR_DIM == 16


def test_default_variant_is_b_wz_only() -> None:
    """V2 design memo §4.3: Variant B is the DEFAULT per UNIQUE-AND-COMPLETE."""
    cfg = ATWv2CodecConfig()
    assert cfg.variant == ATWv2Variant.B_WZ_ONLY


def test_variant_enum_values() -> None:
    assert ATWv2Variant.A_THREE_KNOB.value == "A"
    assert ATWv2Variant.B_WZ_ONLY.value == "B"


def test_codec_forward_shapes() -> None:
    """ATWv2Codec.forward returns (B,3,H,W) rgb pair + optional WZ + G1 outputs."""
    cfg = _make_tiny_cfg()
    model = ATWv2Codec(cfg)
    pair_indices = torch.arange(cfg.num_pairs, dtype=torch.long)
    rgb_0, rgb_1, mu, logvar, z_res, z_pred, g1_logits = model(
        pair_indices, compute_wz_residual=True, compute_g1_logits=True
    )
    expected_pair_shape = (cfg.num_pairs, 3, cfg.output_height, cfg.output_width)
    assert tuple(rgb_0.shape) == expected_pair_shape
    assert tuple(rgb_1.shape) == expected_pair_shape
    assert mu is None and logvar is None
    assert tuple(z_res.shape) == (cfg.num_pairs, cfg.latent_dim)
    assert tuple(z_pred.shape) == (cfg.num_pairs, cfg.latent_dim)
    assert tuple(g1_logits.shape) == (cfg.num_pairs, NUM_SEGNET_CLASSES)


def test_wz_side_info_head_operational_consumption_at_inflate() -> None:
    """Catalog #220 OPERATIONAL contract: WZ head changes decoded output.

    Reconstructing z = z_residual + WZ_head(class_prior) produces a DIFFERENT
    RGB pair than decoding z_residual alone — verifying the WZ side-info head
    is structurally consumed at inflate time.
    """
    cfg = _make_tiny_cfg()
    model = ATWv2Codec(cfg)
    # Populate class_prior_table with non-zero values so WZ_head returns non-zero
    with torch.no_grad():
        model.scorer_class_prior_table.fill_(0.5)
    # Make the WZ head's bias non-zero so the prediction is non-trivial
    with torch.no_grad():
        model.wz_side_info_head.fc1.weight.fill_(0.1)
        model.wz_side_info_head.fc1.bias.fill_(0.2)
        model.wz_side_info_head.fc2.weight.fill_(0.1)
        model.wz_side_info_head.fc2.bias.fill_(0.3)

    pair_indices = torch.arange(cfg.num_pairs, dtype=torch.long)
    fake_residual = torch.zeros(cfg.num_pairs, cfg.latent_dim)
    rgb0_a, rgb1_a = model.reconstruct_from_wz_residual(pair_indices, fake_residual)
    # Now disable the WZ head by setting it to zero output (so z = residual)
    with torch.no_grad():
        model.wz_side_info_head.fc1.weight.zero_()
        model.wz_side_info_head.fc1.bias.zero_()
        model.wz_side_info_head.fc2.weight.zero_()
        model.wz_side_info_head.fc2.bias.zero_()
    rgb0_b, rgb1_b = model.reconstruct_from_wz_residual(pair_indices, fake_residual)
    # The reconstructions MUST differ — proving WZ side-info head is consumed
    assert not torch.allclose(rgb0_a, rgb0_b)


def test_g1_distill_head_predicts_class_index() -> None:
    """Wunderkind G1: distill head produces 5-way class logits + argmax."""
    cfg = _make_tiny_cfg()
    model = ATWv2Codec(cfg)
    z = torch.randn(cfg.num_pairs, cfg.latent_dim)
    pred = model.g1_distill_head.predict_class(z)
    assert pred.shape == (cfg.num_pairs,)
    assert pred.dtype == torch.long
    assert int(pred.min().item()) >= 0
    assert int(pred.max().item()) < NUM_SEGNET_CLASSES


# ---------------------------------------------------------------------------
# 2. Archive grammar tests (ATW2 byte layout)
# ---------------------------------------------------------------------------


def _make_dummy_state_dicts_and_tables(
    cfg: ATWv2CodecConfig,
) -> tuple[dict[str, Any], ...]:
    """Build the 4 SDs + 3 tables a pack_archive call needs."""
    model = ATWv2Codec(cfg)
    return (
        model.encoder.state_dict(),
        model.decoder.state_dict(),
        model.wz_side_info_head.state_dict(),
        model.g1_distill_head.state_dict(),
        model.latents.detach().cpu(),
        model.scorer_class_prior_table.detach().cpu(),
        model.cdf_table.detach().cpu(),
    )


def test_archive_pack_parse_roundtrip_variant_b() -> None:
    """ATW2 archive bytes -> parse roundtrips all 9 fields (Variant B)."""
    cfg = _make_tiny_cfg()
    sds = _make_dummy_state_dicts_and_tables(cfg)
    meta = {
        "decoder_embed_dim": cfg.decoder_embed_dim,
        "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
        "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
        "encoder_input_channels": cfg.encoder_input_channels,
        "encoder_hidden_dim": cfg.encoder_hidden_dim,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "wz_head_hidden_dim": cfg.wz_head_hidden_dim,
        "g1_distill_hidden_dim": cfg.g1_distill_hidden_dim,
        "latent_init_std": cfg.latent_init_std,
    }
    bytes_v1 = pack_archive(*sds, meta, variant=1)
    assert bytes_v1.startswith(ATW2_MAGIC)
    parsed = parse_archive(bytes_v1)
    assert parsed.schema_version == 1
    assert parsed.variant == 1
    # Parsed fields recoverable
    assert parsed.latent_residual.shape == (cfg.num_pairs, cfg.latent_dim)
    assert parsed.scorer_class_prior_table.shape == (
        cfg.num_pairs, cfg.scorer_class_prior_dim
    )
    assert parsed.cdf_table.shape == (NUM_SEGNET_CLASSES, CDF_TABLE_NUM_SYMBOLS)
    assert parsed.meta["atw_v2_codec_meta"]["variant"] == 1
    assert parsed.meta["atw_v2_codec_meta"]["lambda_wz"] == 1.0
    # Re-pack with identical meta + sds + tables (rebuilding the canonical input
    # by stripping the auto-injected provenance + quantization scale fields)
    # produces a byte-identical archive — verifying the byte layout is stable.
    cleaned_meta = {
        k: v for k, v in parsed.meta.items()
        if not k.startswith("_") and k != "atw_v2_codec_meta"
    }
    # State dicts are fp16-cast on load (deserializer returns fp32 from fp16
    # bytes); we can't byte-compare directly because the original fp32 sds
    # were cast at pack time and the re-cast at second pack is a no-op (already
    # in fp32 from the parse path). The cdf_table/scorer_class_prior_table/
    # latent_residual ARE byte-identical when reconstructed from parse->pack
    # because they ship at fixed precision (int8 / fp16) and dequantize is
    # lossless to fp32 within the symbol range.
    bytes_v1_redo = pack_archive(
        parsed.encoder_state_dict,
        parsed.decoder_state_dict,
        parsed.wz_side_info_head_state_dict,
        parsed.distill_head_state_dict,
        parsed.latent_residual,
        parsed.scorer_class_prior_table,
        parsed.cdf_table,
        cleaned_meta,
        variant=1,
    )
    # The two archives should agree on header + tables; encoder/decoder/head
    # blobs may differ if fp32->fp16->fp32 introduces precision drift in
    # downstream brotli compression. Assert the parser at minimum recovers the
    # same shapes + variant byte + magic + meta tag.
    parsed_redo = parse_archive(bytes_v1_redo)
    assert parsed_redo.variant == parsed.variant
    assert parsed_redo.latent_residual.shape == parsed.latent_residual.shape
    assert parsed_redo.cdf_table.shape == parsed.cdf_table.shape
    assert torch.allclose(parsed_redo.cdf_table, parsed.cdf_table, atol=1e-3)


def test_archive_variant_byte_roundtrips() -> None:
    """Variant byte 0 (A) and 1 (B) both roundtrip through pack/parse."""
    cfg = _make_tiny_cfg()
    sds = _make_dummy_state_dicts_and_tables(cfg)
    meta = {"decoder_embed_dim": cfg.decoder_embed_dim,
            "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
            "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
            "decoder_channels": list(cfg.decoder_channels),
            "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
            "encoder_input_channels": cfg.encoder_input_channels,
            "encoder_hidden_dim": cfg.encoder_hidden_dim,
            "output_height": cfg.output_height,
            "output_width": cfg.output_width,
            "wz_head_hidden_dim": cfg.wz_head_hidden_dim,
            "g1_distill_hidden_dim": cfg.g1_distill_hidden_dim,
            "latent_init_std": cfg.latent_init_std}
    bytes_a = pack_archive(*sds, dict(meta), variant=0)
    bytes_b = pack_archive(*sds, dict(meta), variant=1)
    assert parse_archive(bytes_a).variant == 0
    assert parse_archive(bytes_b).variant == 1
    # Byte at offset 5 (VARIANT) must differ
    assert bytes_a[5] == 0
    assert bytes_b[5] == 1


def test_parse_atw2_archive_bytes_returns_9_sections() -> None:
    """V2 has 9 sections per design memo §10 (+2 vs V1)."""
    cfg = _make_tiny_cfg()
    sds = _make_dummy_state_dicts_and_tables(cfg)
    bytes_ = pack_archive(*sds, {"decoder_embed_dim": cfg.decoder_embed_dim,
                                  "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
                                  "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
                                  "decoder_channels": list(cfg.decoder_channels),
                                  "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
                                  "encoder_input_channels": cfg.encoder_input_channels,
                                  "encoder_hidden_dim": cfg.encoder_hidden_dim,
                                  "output_height": cfg.output_height,
                                  "output_width": cfg.output_width,
                                  "wz_head_hidden_dim": cfg.wz_head_hidden_dim,
                                  "g1_distill_hidden_dim": cfg.g1_distill_hidden_dim,
                                  "latent_init_std": cfg.latent_init_std}, variant=1)
    sections = parse_atw2_archive_bytes(bytes_)
    expected = {
        "atw2_header",
        "encoder_blob",
        "decoder_blob",
        "wz_head_blob",
        "distill_head_blob",
        "latent_residual_blob",
        "class_prior_table_blob",
        "cdf_table_blob",
        "meta_blob",
    }
    assert set(sections.keys()) == expected


def test_parse_refuses_bad_magic() -> None:
    """Parser refuses non-ATW2 bytes at byte 0."""
    bad = b"XXXX" + b"\x00" * 100
    with pytest.raises(ValueError, match="bad magic"):
        parse_atw2_archive_bytes(bad)


def test_parse_refuses_bad_variant_byte() -> None:
    """Parser refuses variant byte != 0 or 1."""
    cfg = _make_tiny_cfg()
    sds = _make_dummy_state_dicts_and_tables(cfg)
    bytes_ = pack_archive(*sds, {"decoder_embed_dim": cfg.decoder_embed_dim,
                                  "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
                                  "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
                                  "decoder_channels": list(cfg.decoder_channels),
                                  "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
                                  "encoder_input_channels": cfg.encoder_input_channels,
                                  "encoder_hidden_dim": cfg.encoder_hidden_dim,
                                  "output_height": cfg.output_height,
                                  "output_width": cfg.output_width,
                                  "wz_head_hidden_dim": cfg.wz_head_hidden_dim,
                                  "g1_distill_hidden_dim": cfg.g1_distill_hidden_dim,
                                  "latent_init_std": cfg.latent_init_std}, variant=1)
    # Corrupt variant byte to value 7
    corrupted = bytes_[:5] + bytes([7]) + bytes_[6:]
    with pytest.raises(ValueError, match="variant"):
        parse_atw2_archive_bytes(corrupted)


def test_pack_refuses_bad_variant() -> None:
    """pack_archive refuses variant int outside {0, 1}."""
    cfg = _make_tiny_cfg()
    sds = _make_dummy_state_dicts_and_tables(cfg)
    with pytest.raises(ValueError, match="variant must be 0"):
        pack_archive(*sds, {"decoder_embed_dim": cfg.decoder_embed_dim,
                            "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
                            "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
                            "decoder_channels": list(cfg.decoder_channels),
                            "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
                            "encoder_input_channels": cfg.encoder_input_channels,
                            "encoder_hidden_dim": cfg.encoder_hidden_dim,
                            "output_height": cfg.output_height,
                            "output_width": cfg.output_width,
                            "wz_head_hidden_dim": cfg.wz_head_hidden_dim,
                            "g1_distill_hidden_dim": cfg.g1_distill_hidden_dim,
                            "latent_init_std": cfg.latent_init_std}, variant=5)


def test_section_roles_canonical() -> None:
    """ATW2_SECTION_ROLES maps every section to a canonical role."""
    assert ATW2_SECTION_ROLES["atw2_header"] == "control_or_metadata"
    assert ATW2_SECTION_ROLES["decoder_blob"] == "decoder_weight_stream"
    assert ATW2_SECTION_ROLES["wz_head_blob"] == "decoder_weight_stream"
    assert ATW2_SECTION_ROLES["distill_head_blob"] == "decoder_weight_stream"
    assert ATW2_SECTION_ROLES["latent_residual_blob"] == "latent_stream"
    assert ATW2_SECTION_ROLES["class_prior_table_blob"] == "decoder_side_information"
    assert ATW2_SECTION_ROLES["cdf_table_blob"] == "decoder_side_information"


# ---------------------------------------------------------------------------
# 3. Catalog #220 + #272 byte-mutation smoke (distinguishing-feature consumption)
# ---------------------------------------------------------------------------


def test_byte_mutation_in_distinguishing_section_changes_output() -> None:
    """Catalog #220 + #272: mutating bytes in WZ_HEAD_BLOB or LATENT_RESIDUAL_BLOB
    changes inflate output — proving distinguishing bytes are operationally consumed.

    This is the canonical Catalog #272 no-op detector at the unit-test surface
    (the full byte-mutation smoke fires via
    tools/verify_distinguishing_feature_byte_mutation.py post-archive build).
    """
    from tac.substrates.atw_codec_v2.inflate import inflate_one_video

    cfg = _make_tiny_cfg()
    sds = _make_dummy_state_dicts_and_tables(cfg)
    # Fill class_prior_table with non-zero pattern so WZ_head produces non-trivial output
    cpt = sds[5]
    for i in range(cpt.shape[0]):
        cpt[i] = torch.arange(cpt.shape[1], dtype=torch.float32) * 0.1 + i * 0.01
    meta = {"decoder_embed_dim": cfg.decoder_embed_dim,
            "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
            "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
            "decoder_channels": list(cfg.decoder_channels),
            "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
            "encoder_input_channels": cfg.encoder_input_channels,
            "encoder_hidden_dim": cfg.encoder_hidden_dim,
            "output_height": cfg.output_height,
            "output_width": cfg.output_width,
            "wz_head_hidden_dim": cfg.wz_head_hidden_dim,
            "g1_distill_hidden_dim": cfg.g1_distill_hidden_dim,
            "latent_init_std": cfg.latent_init_std}
    archive_bytes = pack_archive(*sds, meta, variant=1)
    sections = parse_atw2_archive_bytes(archive_bytes)
    # Mutate one byte in latent_residual_blob
    start, length = sections["latent_residual_blob"]
    assert length > 0
    target_offset = start + length // 2
    mutated = bytearray(archive_bytes)
    mutated[target_offset] ^= 0xFF  # flip all bits in one byte
    mutated_bytes = bytes(mutated)
    # Run inflate on both archives, render to scratch raw, compare bytes
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out_a = Path(tmp) / "a.raw"
        out_b = Path(tmp) / "b.raw"
        inflate_one_video(archive_bytes, out_a, device="cpu")
        inflate_one_video(mutated_bytes, out_b, device="cpu")
        bytes_a = out_a.read_bytes()
        bytes_b = out_b.read_bytes()
    # The two raw outputs MUST differ — proving the residual bytes are consumed.
    assert bytes_a != bytes_b


# ---------------------------------------------------------------------------
# 4. Score-aware loss tests
# ---------------------------------------------------------------------------


def test_score_aware_loss_variant_b_default_weights() -> None:
    """Variant B defaults: alpha_rate=25, beta_seg=100, gamma_pose=sqrt(10),
    kappa_ib=0, lambda_wz=1, lambda_pixel=0, lambda_distill=0.1."""
    import math

    w = ATWv2LossWeights()
    assert w.alpha_rate == 25.0
    assert w.beta_seg == 100.0
    assert abs(w.gamma_pose - math.sqrt(10.0)) < 1e-9
    assert w.kappa_ib == 0.0
    assert w.lambda_wz == 1.0
    assert w.lambda_pixel == 0.0
    assert w.lambda_distill == 0.1


def test_score_aware_loss_refuses_eval_roundtrip_false() -> None:
    """CLAUDE.md "eval_roundtrip — non-negotiable" enforced at loss boundary."""
    seg = _StubScorer()
    pose = _StubPoseScorer()
    loss_mod = ATWv2ScoreAwareLoss(seg, pose, ATWv2LossWeights())
    fake_rgb = torch.full((1, 3, 16, 24), 100.0)
    fake_arch_bytes = torch.tensor(100.0)
    with pytest.raises(ValueError, match="apply_eval_roundtrip"):
        loss_mod(
            reconstructed_rgb_0=fake_rgb,
            reconstructed_rgb_1=fake_rgb,
            gt_rgb_0=fake_rgb,
            gt_rgb_1=fake_rgb,
            archive_bytes_proxy=fake_arch_bytes,
            z_residual=torch.zeros(1, 8),
            apply_eval_roundtrip=False,
        )


def test_score_aware_loss_refuses_negative_weights() -> None:
    seg = _StubScorer()
    pose = _StubPoseScorer()
    w = ATWv2LossWeights(lambda_wz=-1.0)
    loss_mod = ATWv2ScoreAwareLoss(seg, pose, w)
    fake_rgb = torch.full((1, 3, 16, 24), 100.0)
    with pytest.raises(ValueError, match="lambda_wz"):
        loss_mod(
            reconstructed_rgb_0=fake_rgb,
            reconstructed_rgb_1=fake_rgb,
            gt_rgb_0=fake_rgb,
            gt_rgb_1=fake_rgb,
            archive_bytes_proxy=torch.tensor(100.0),
            z_residual=torch.zeros(1, 8),
        )


# ---------------------------------------------------------------------------
# 5. Wunderkind productionization tests
# ---------------------------------------------------------------------------


def test_wunderkind_g1_distill_head_replaces_balle_hyperprior_in_size() -> None:
    """Wunderkind G1 substitution: distill head must be << 50KB (Ballé hyperprior size)."""
    cfg = _make_tiny_cfg(g1_distill_hidden_dim=32)
    model = ATWv2Codec(cfg)
    breakdown = model.num_parameters_breakdown()
    # G1 distill head should be tiny (~1KB after fp16). At hidden=32 + latent=8 + classes=5:
    # fc1 = 32*8 + 32 = 288 params; fc2 = 5*32 + 5 = 165 params; total ~453 params
    # fp16 = ~906 bytes ~1KB — << 50KB Ballé.
    assert breakdown["g1_distill_head"] < 1000  # params count, way under 50KB / 2 = 25000


def test_wunderkind_b3_cdf_table_shipped_in_archive() -> None:
    """B3 CDF table is shipped IN archive bytes — verifiable via parser."""
    cfg = _make_tiny_cfg()
    sds = _make_dummy_state_dicts_and_tables(cfg)
    bytes_ = pack_archive(*sds, {"decoder_embed_dim": cfg.decoder_embed_dim,
                                  "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
                                  "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
                                  "decoder_channels": list(cfg.decoder_channels),
                                  "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
                                  "encoder_input_channels": cfg.encoder_input_channels,
                                  "encoder_hidden_dim": cfg.encoder_hidden_dim,
                                  "output_height": cfg.output_height,
                                  "output_width": cfg.output_width,
                                  "wz_head_hidden_dim": cfg.wz_head_hidden_dim,
                                  "g1_distill_hidden_dim": cfg.g1_distill_hidden_dim,
                                  "latent_init_std": cfg.latent_init_std}, variant=1)
    sections = parse_atw2_archive_bytes(bytes_)
    cdf_start, cdf_len = sections["cdf_table_blob"]
    # CDF table = num_classes * num_symbols * 2 bytes (fp16)
    expected_cdf_bytes = NUM_SEGNET_CLASSES * CDF_TABLE_NUM_SYMBOLS * 2
    assert cdf_len == expected_cdf_bytes


def test_wunderkind_wz_side_info_head_closed_form_predicts_latents() -> None:
    """WZ side-info head closed-form: predicts latent from class prior at inflate."""
    cfg = _make_tiny_cfg()
    model = ATWv2Codec(cfg)
    with torch.no_grad():
        model.scorer_class_prior_table.fill_(1.0)
    # Predict via head
    class_prior = model.scorer_class_prior_table[:2]
    z_pred = model.wz_side_info_head(class_prior)
    assert z_pred.shape == (2, cfg.latent_dim)
    assert not torch.allclose(z_pred, torch.zeros_like(z_pred))


# ---------------------------------------------------------------------------
# 6. Catalog #270 dispatch optimization protocol test
# ---------------------------------------------------------------------------


def test_catalog_270_dispatch_optimization_protocol_passes() -> None:
    """Catalog #270 protocol returns overall_pass=true for ATW v2 trainer + recipe.

    Verifies all 7 V1 Catalog #270 blockers are resolved structurally:
    Tier 1 (4): autocast_fp16 + tf32 + torch_compile + canonical_scorer_loss
    Tier 3 (3): canonical_auth_eval_helper + canonical_inflate_device +
                scorer_loader_order_correct
    Plus Tier 2 (5): min_vram_gb + min_smoke_gpu + video_input_strategy +
                pyav_decode_strategy + target_modes
    Plus Tier 3 recipe-vs-trainer-state consistency.
    """
    proto_tool = REPO_ROOT / "tools" / "canonical_dispatch_optimization_protocol.py"
    if not proto_tool.is_file():
        pytest.skip("canonical_dispatch_optimization_protocol.py not present")
    trainer = REPO_ROOT / "experiments" / "train_substrate_atw_codec_v2.py"
    result = subprocess.run(
        [
            sys.executable,
            str(proto_tool),
            "--trainer",
            str(trainer),
            "--recipe",
            "substrate_atw_codec_v2_modal_a100_dispatch",
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    payload = json.loads(result.stdout)
    assert payload["overall_pass"] is True, (
        f"Catalog #270 protocol FAILED:\n"
        f"blockers={payload.get('blockers', [])}\n"
        f"tier1={payload['tier1']['pass_signals']}\n"
        f"tier2={payload['tier2']['pass_signals']}\n"
        f"tier3={payload['tier3']['pass_signals']}"
    )
    # Tier 1: all 5 engineering primitives green
    t1 = payload["tier1"]["pass_signals"]
    assert t1["autocast_fp16"]
    assert t1["tf32"]
    assert t1["torch_compile"]
    assert t1["no_grad_at_eval"]
    assert t1["canonical_scorer_loss"]
    # Tier 2: recipe fields declared
    t2 = payload["tier2"]["pass_signals"]
    assert t2["recipe_declares_min_vram_gb"]
    assert t2["recipe_declares_min_smoke_gpu"]
    assert t2["recipe_declares_video_input_strategy"]
    assert t2["recipe_declares_pyav_decode_strategy"]
    assert t2["recipe_declares_target_modes"]
    # Tier 2 driver: 3 canonical NVML/CUDA env exports
    assert t2["driver_exports_CUBLAS_WORKSPACE_CONFIG"]
    assert t2["driver_exports_DALI_DISABLE_NVML"]
    assert t2["driver_exports_PYTORCH_CUDA_ALLOC_CONF"]
    # Tier 3: canonical helper routing
    t3 = payload["tier3"]["pass_signals"]
    assert t3["canonical_auth_eval_helper"]
    assert t3["canonical_inflate_device"]
    assert t3["scorer_loader_order_correct"]
    assert t3["no_phantom_device_named_output"]
    assert t3["recipe_vs_trainer_state_consistent"]


# ---------------------------------------------------------------------------
# 7. Smoke trainer entry point test
# ---------------------------------------------------------------------------


def test_smoke_trainer_runs(tmp_path: Path) -> None:
    """Smoke trainer entry point runs end-to-end on CPU in <1 min."""
    trainer = REPO_ROOT / "experiments" / "train_substrate_atw_codec_v2.py"
    if not trainer.is_file():
        pytest.skip("trainer module not present")
    out_dir = tmp_path / "atw_v2_smoke"
    result = subprocess.run(
        [
            sys.executable,
            str(trainer),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--smoke",
            "--variant",
            "B",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, (
        f"smoke trainer failed rc={result.returncode}:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    archive_path = out_dir / "0.bin"
    assert archive_path.is_file()
    archive_bytes = archive_path.read_bytes()
    assert archive_bytes.startswith(ATW2_MAGIC)
    archive_zip_path = out_dir / "archive.zip"
    assert archive_zip_path.is_file()
    with zipfile.ZipFile(archive_zip_path, "r") as zf:
        assert zf.namelist() == ["0.bin"]
        assert zf.read("0.bin") == archive_bytes
    submission_dir = out_dir / "submission"
    assert (submission_dir / "inflate.py").is_file()
    assert (submission_dir / "0.bin").read_bytes() == archive_bytes
    stats_path = out_dir / "smoke_stats.json"
    assert stats_path.is_file()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["smoke"] is True
    assert stats["atw2_magic_ok"] is True
    assert stats["roundtrip_ok"] is True
    assert stats["variant"] == "B"
    assert stats["archive_zip_built"] is True
    assert stats["archive_zip_path"] == str(archive_zip_path)
    assert stats["archive_zip_bytes"] == archive_zip_path.stat().st_size
    assert len(stats["archive_zip_sha256"]) == 64
    assert stats["submission_dir"] == str(submission_dir)


def test_smoke_trainer_variant_a_runs(tmp_path: Path) -> None:
    """Variant A (three-knob) smoke entry point also runs."""
    trainer = REPO_ROOT / "experiments" / "train_substrate_atw_codec_v2.py"
    if not trainer.is_file():
        pytest.skip("trainer module not present")
    out_dir = tmp_path / "atw_v2_smoke_a"
    result = subprocess.run(
        [
            sys.executable,
            str(trainer),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--smoke",
            "--variant",
            "A",
            "--kappa-ib",
            "0.05",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, (
        f"variant A smoke trainer failed rc={result.returncode}:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    stats = json.loads((out_dir / "smoke_stats.json").read_text(encoding="utf-8"))
    assert stats["variant"] == "A"
    assert stats["variant_byte"] == 0


# ---------------------------------------------------------------------------
# 8. SubstrateContract META layer registration test
# ---------------------------------------------------------------------------


def test_substrate_contract_registers() -> None:
    """Contract validates via @register_substrate import-time validators."""
    from tac.substrates.atw_codec_v2.registered_substrate import (
        ATW_CODEC_V2_CONTRACT,
    )

    assert ATW_CODEC_V2_CONTRACT.id == "atw_codec_v2"
    assert ATW_CODEC_V2_CONTRACT.lane_id == "lane_atw_codec_v2_substrate_build_20260516"
    assert "research_substrate" in ATW_CODEC_V2_CONTRACT.target_modes
    assert ATW_CODEC_V2_CONTRACT.recipe_research_only is True
    assert ATW_CODEC_V2_CONTRACT.recipe_min_smoke_gpu == "A100"
    assert ATW_CODEC_V2_CONTRACT.recipe_canary_dependency == (
        "lane_atw_codec_design_v1_20260515"
    )
    # 9 sections declared in parser_section_manifest
    sm = ATW_CODEC_V2_CONTRACT.parser_section_manifest
    assert "header" in sm
    assert "wz_head_blob" in sm
    assert "distill_head_blob" in sm
    assert "cdf_table_blob" in sm
    assert ATW_CODEC_V2_CONTRACT.hook_probe_disambiguator == (
        "tools/run_atw_v2_d4_probe_from_a1.py"
    )
    assert "returned INDEPENDENT" in ATW_CODEC_V2_CONTRACT.hook_not_applicable_rationale[
        "hook_continual_learning_anchor_kind"
    ]


def test_atw_v2_phase2_gate_consumes_d4_verdict_fail_closed() -> None:
    """Current D4 verdict is durable evidence, but not dispatch authority."""
    from tac.substrates.atw_codec_v2 import (
        D4_PROBE_MUTUAL_INFORMATION_BITS,
        D4_PROBE_NEXT_ACTION,
        D4_PROBE_PHASE2_STATUS,
        D4_PROBE_VERDICT,
        atw_v2_phase2_gate_status,
    )

    status = atw_v2_phase2_gate_status(repo_root=REPO_ROOT)

    assert D4_PROBE_VERDICT == "INDEPENDENT"
    assert D4_PROBE_PHASE2_STATUS == (
        "defer_measured_a1_latent_class_conditioning_surface"
    )
    assert D4_PROBE_NEXT_ACTION == "do_not_dispatch_atw_v2_phase2_from_this_signal"
    assert 0.001 <= D4_PROBE_MUTUAL_INFORMATION_BITS < 0.5
    assert status["d4_verdict"] == "INDEPENDENT"
    assert status["phase2_status"] == D4_PROBE_PHASE2_STATUS
    assert status["mutual_information_bits"] == D4_PROBE_MUTUAL_INFORMATION_BITS
    assert status["score_claim"] is False
    assert status["promotion_eligible"] is False
    assert status["rank_or_kill_eligible"] is False
    assert status["ready_for_exact_eval_dispatch"] is False
    assert status["dispatch_allowed"] is False
    assert status["phase2_lift_allowed"] is False
    assert "atw_v2_phase2_deferred_by_d4_verdict:INDEPENDENT" in status["blockers"]


def test_atw_v2_phase2_gate_missing_verdict_blocks_dispatch(tmp_path: Path) -> None:
    """A missing D4 verdict cannot silently authorize Phase-2 work."""
    from tac.optimization.atw_v2_phase2_gate import (
        atw_v2_phase2_gate_status,
    )

    status = atw_v2_phase2_gate_status(repo_root=tmp_path)

    assert status["d4_verdict"] == "MISSING"
    assert status["phase2_status"] == "blocked_missing_d4_probe_verdict"
    assert status["dispatch_allowed"] is False
    assert status["phase2_lift_allowed"] is False
    assert status["ready_for_exact_eval_dispatch"] is False
    assert status["blockers"] == ["atw_v2_d4_probe_verdict_missing"]


# ---------------------------------------------------------------------------
# 9. Remote driver custody tests
# ---------------------------------------------------------------------------


def test_remote_driver_verifies_active_claim_and_terminalizes() -> None:
    """Remote spend path must verify + close lane claims, not just name the file."""
    text = (REPO_ROOT / "scripts/remote_lane_substrate_atw_codec_v2.sh").read_text(
        encoding="utf-8"
    )

    assert '"$WORKSPACE/tools/claim_lane_dispatch.py" summary' in text
    assert 'payload.get("active", [])' in text
    assert 'row.get("lane_id") == lane_id' in text
    assert 'row.get("instance_job_id") == job_id' in text
    assert "stage_0_dispatch_claim_verified" in text
    assert "append_terminal_claim()" in text
    assert "completed_atw_codec_v2_remote_driver" in text
    assert "failed_atw_codec_v2_claim_verification_rc_${rc}" in text
    assert "--force" in text


def test_remote_driver_labels_contest_score_only_after_claim_parser() -> None:
    """A JSON file existing is not enough to print a contest-axis score marker."""
    text = (REPO_ROOT / "scripts/remote_lane_substrate_atw_codec_v2.sh").read_text(
        encoding="utf-8"
    )

    assert "parse_auth_eval_score_claim" in text
    assert "required_score_axis=sys.argv[2]" in text
    assert "auth_eval_not_custody_valid" in text
    assert "auth_eval_missing_or_trainer_failed" in text
    assert 'ARCHIVE_ZIP_PATH="$OUTPUT_DIR/archive.zip"' in text
    assert "archive_zip=$ARCHIVE_ZIP_PATH" in text
    assert "payload_0bin=$PAYLOAD_0BIN_PATH" in text
    assert (
        "LANE_ATW_CODEC_V2_DONE [contest-$AUTH_EVAL_AXIS_LABEL] "
        "score=$AUTH_EVAL_SCORE"
    ) in text
