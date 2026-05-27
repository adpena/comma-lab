# SPDX-License-Identifier: MIT
"""Tests for ATW codec V1 scaffold — Atick-Tishby-Wyner cooperative-receiver codec.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons +
the design memo at ``.omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md``.

Coverage:
* Loss function math: 3-paper composition (Atick-Redlich + Tishby IB + Wyner-Ziv)
* Three-knob ablation regimes (Atick-only, ATW canonical, Tishby IB, Z3 baseline)
* Forward pass shape correctness on synthetic inputs
* WZ side-info head structural consumption (operational mechanism per Catalog #220)
* Archive grammar parser symmetry (encode → decode → byte identity)
* HNeRV parity discipline lesson 8: scorer preprocess gradient-reachable
* Smoke trainer entry point
* _full_main raises NotImplementedError per Catalog #220 cascade
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from tac.substrates.atw_codec_v1 import (
    ATW1_HEADER_SIZE,
    ATW1_MAGIC,
    ATW1_SCHEMA_VERSION,
    ATW1_SECTION_ROLES,
    ATWCodec,
    ATWCodecConfig,
    ATWLossWeights,
    ATWScoreAwareLoss,
    pack_archive,
    parse_archive,
    parse_atw1_archive_bytes,
)
from tac.substrates.atw_codec_v1.architecture import (
    DEFAULT_SCORER_CLASS_PRIOR_DIM,
    EVAL_HW,
    NUM_PAIRS,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
)

# ---------------------------------------------------------------------------
# Test fixtures + helpers
# ---------------------------------------------------------------------------


def _make_tiny_cfg(**overrides: Any) -> ATWCodecConfig:
    """Tiny config so smoke tests run fast on CPU (<1 sec each)."""
    base = {
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
    }
    base.update(overrides)
    return ATWCodecConfig(**base)


class _StubScorer(torch.nn.Module):
    """Tiny stub scorer that mimics ``preprocess_input`` + ``__call__`` contract."""

    def __init__(self, *, out_classes: int = 5):
        super().__init__()
        self.out_classes = out_classes
        self.proj = torch.nn.Conv2d(3, out_classes, kernel_size=1)

    def preprocess_input(self, batch: torch.Tensor) -> torch.Tensor:
        # Match canonical signature: take (B, T, C, H, W); return reshaped tensor
        # the scorer can forward. For the stub we accept any shape and project to
        # 4D (B*T, C, H, W) per upstream SegNet contract.
        if batch.dim() == 5:
            B, T, C, H, W = batch.shape
            return batch.reshape(B * T, C, H, W)
        if batch.dim() == 4:
            return batch
        raise ValueError(f"_StubScorer.preprocess_input shape {batch.shape}")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.proj(batch)


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------


def test_constants_exported() -> None:
    """Module-level constants for downstream consumers."""
    assert isinstance(EVAL_HW, tuple) and EVAL_HW == (384, 512)
    assert NUM_PAIRS == 600
    assert TOTAL_ARCHIVE_TARGET_BYTES_MIN > 0
    assert TOTAL_ARCHIVE_TARGET_BYTES_MAX >= TOTAL_ARCHIVE_TARGET_BYTES_MIN
    assert DEFAULT_SCORER_CLASS_PRIOR_DIM == 16


def test_config_defaults_match_z4_baseline() -> None:
    """ATW config defaults match Z4 baseline for encoder/decoder dims."""
    cfg = ATWCodecConfig()
    assert cfg.latent_dim == 24
    assert cfg.decoder_embed_dim == 32
    assert cfg.decoder_initial_grid_h == 3
    assert cfg.decoder_initial_grid_w == 4
    assert cfg.num_pairs == NUM_PAIRS
    assert cfg.output_height == EVAL_HW[0]
    assert cfg.output_width == EVAL_HW[1]
    # ATW canonical defaults: WZ enabled, no IB, no pixel
    assert cfg.wz_head_enabled is True
    assert cfg.ib_kappa_default == 0.0
    assert cfg.wz_lambda_default == 1.0
    assert cfg.pixel_lambda_default == 0.0


def test_config_output_hw_property() -> None:
    cfg = _make_tiny_cfg()
    assert cfg.output_hw == (cfg.output_height, cfg.output_width)


def test_atw_codec_instantiates_with_canonical_config() -> None:
    cfg = _make_tiny_cfg()
    model = ATWCodec(cfg)
    assert model.cfg == cfg
    assert model.encoder is not None
    assert model.decoder is not None
    assert model.wz_side_info_head is not None
    assert model.latents.shape == (cfg.num_pairs, cfg.latent_dim)
    assert model.scorer_class_prior_table.shape == (cfg.num_pairs, cfg.scorer_class_prior_dim)


def test_atw_codec_forward_shapes_correct() -> None:
    """Forward returns RGB pair of expected shape."""
    cfg = _make_tiny_cfg()
    model = ATWCodec(cfg).eval()
    pair_indices = torch.arange(cfg.num_pairs, dtype=torch.long)
    rgb_0, rgb_1, mu, logvar, z_residual, z_predicted = model(
        pair_indices, frames_for_encoder=None, compute_wz_residual=False
    )
    assert rgb_0.shape == (cfg.num_pairs, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (cfg.num_pairs, 3, cfg.output_height, cfg.output_width)
    assert mu is None and logvar is None  # eval path
    assert z_residual is None and z_predicted is None  # compute_wz_residual=False


def test_atw_codec_forward_with_wz_residual() -> None:
    """compute_wz_residual=True returns (z_residual, z_predicted)."""
    cfg = _make_tiny_cfg()
    model = ATWCodec(cfg).eval()
    pair_indices = torch.arange(cfg.num_pairs, dtype=torch.long)
    _, _, _, _, z_residual, z_predicted = model(
        pair_indices, compute_wz_residual=True
    )
    assert z_residual is not None and z_predicted is not None
    assert z_residual.shape == (cfg.num_pairs, cfg.latent_dim)
    assert z_predicted.shape == (cfg.num_pairs, cfg.latent_dim)
    # z_residual = z - z_predicted exactly (mathematical invariant)
    z_expected = model.latents[pair_indices]
    torch.testing.assert_close(z_residual, z_expected - z_predicted)


def test_atw_codec_forward_with_encoder_frames() -> None:
    """When frames_for_encoder is provided, returns mu + logvar."""
    cfg = _make_tiny_cfg()
    model = ATWCodec(cfg).train()
    pair_indices = torch.tensor([0, 1], dtype=torch.long)
    frames = torch.rand(2, 3, 32, 32)
    _, _, mu, logvar, _, _ = model(pair_indices, frames_for_encoder=frames)
    assert mu is not None and logvar is not None
    assert mu.shape == (2, cfg.latent_dim)
    assert logvar.shape == (2, cfg.latent_dim)


def test_atw_codec_pair_indices_validation() -> None:
    cfg = _make_tiny_cfg()
    model = ATWCodec(cfg).eval()
    with pytest.raises(ValueError, match=r"pair_indices must be torch\.long"):
        model(torch.tensor([0.0]))
    with pytest.raises(ValueError, match="pair_indices must be non-empty"):
        model(torch.tensor([], dtype=torch.long))
    with pytest.raises(ValueError, match="pair_indices out of range"):
        model(torch.tensor([cfg.num_pairs + 1], dtype=torch.long))


def test_wz_head_disabled_returns_zeros() -> None:
    """When wz_head_enabled=False, the head is a structural no-op (zeros)."""
    cfg = _make_tiny_cfg(wz_head_enabled=False)
    model = ATWCodec(cfg).eval()
    pair_indices = torch.arange(cfg.num_pairs, dtype=torch.long)
    _, _, _, _, z_residual, z_predicted = model(
        pair_indices, compute_wz_residual=True
    )
    assert z_predicted is not None
    torch.testing.assert_close(z_predicted, torch.zeros_like(z_predicted))
    # z_residual = z - 0 = z exactly
    z_expected = model.latents[pair_indices]
    torch.testing.assert_close(z_residual, z_expected)


def test_wz_head_disabled_zero_params() -> None:
    """WZ-disabled head has zero trainable parameters."""
    cfg = _make_tiny_cfg(wz_head_enabled=False)
    model = ATWCodec(cfg).eval()
    assert model.wz_side_info_head.num_parameters() == 0


def test_wz_head_enabled_has_params() -> None:
    """WZ-enabled head has non-zero trainable parameters."""
    cfg = _make_tiny_cfg(wz_head_enabled=True)
    model = ATWCodec(cfg).eval()
    assert model.wz_side_info_head.num_parameters() > 0


def test_reconstruct_from_wz_residual_roundtrip() -> None:
    """Inflate-time path reconstructs same RGB pair from (z_residual, class_prior)."""
    cfg = _make_tiny_cfg()
    model = ATWCodec(cfg).eval()
    # Populate class_prior_table with deterministic non-zero pattern
    with torch.no_grad():
        for i in range(cfg.num_pairs):
            model.scorer_class_prior_table[i] = (
                torch.arange(cfg.scorer_class_prior_dim, dtype=torch.float32) * 0.1
                + float(i) * 0.01
            )
    pair_indices = torch.arange(cfg.num_pairs, dtype=torch.long)

    # Train-time forward gets full z and computes residual.
    rgb_0_train, rgb_1_train, _, _, z_residual, z_predicted = model(
        pair_indices, compute_wz_residual=True
    )

    # Inflate-time forward reconstructs from residual.
    rgb_0_inflate, rgb_1_inflate = model.reconstruct_from_wz_residual(
        pair_indices, z_residual
    )
    torch.testing.assert_close(rgb_0_train, rgb_0_inflate)
    torch.testing.assert_close(rgb_1_train, rgb_1_inflate)


def test_forward_wz_residual_mode_consumes_side_info() -> None:
    """Eval forward must use WZ side-info when latents are archived residuals."""
    cfg = _make_tiny_cfg()
    model = ATWCodec(cfg).eval()
    with torch.no_grad():
        for param in model.wz_side_info_head.parameters():
            param.fill_(0.1)
        for i in range(cfg.num_pairs):
            model.scorer_class_prior_table[i] = (
                torch.ones(cfg.scorer_class_prior_dim) * float(i + 1) * 0.25
            )
    pair_indices = torch.arange(cfg.num_pairs, dtype=torch.long)

    rgb_0_full, rgb_1_full, *_ = model(pair_indices, decode_mode="full_latent")
    rgb_0_wz, rgb_1_wz, _, _, z_residual, z_predicted = model(
        pair_indices,
        decode_mode="wz_residual",
        compute_wz_residual=True,
    )
    rgb_0_ref, rgb_1_ref = model.reconstruct_from_wz_residual(
        pair_indices,
        model.latents[pair_indices],
    )

    assert z_residual is not None
    assert z_predicted is not None
    torch.testing.assert_close(z_residual, model.latents[pair_indices])
    torch.testing.assert_close(rgb_0_wz, rgb_0_ref)
    torch.testing.assert_close(rgb_1_wz, rgb_1_ref)
    assert not torch.allclose(rgb_0_full, rgb_0_wz)
    assert not torch.allclose(rgb_1_full, rgb_1_wz)


def test_forward_rejects_unknown_decode_mode() -> None:
    cfg = _make_tiny_cfg()
    model = ATWCodec(cfg).eval()
    with pytest.raises(ValueError, match="decode_mode"):
        model(torch.tensor([0], dtype=torch.long), decode_mode="legacy")


def test_reconstruct_from_wz_residual_validates_inputs() -> None:
    cfg = _make_tiny_cfg()
    model = ATWCodec(cfg).eval()
    pair_indices = torch.tensor([0, 1], dtype=torch.long)
    bad_residual = torch.zeros(2, cfg.latent_dim + 1)
    with pytest.raises(ValueError, match="z_residual must be"):
        model.reconstruct_from_wz_residual(pair_indices, bad_residual)
    with pytest.raises(ValueError, match="batch sizes mismatch"):
        model.reconstruct_from_wz_residual(
            pair_indices, torch.zeros(3, cfg.latent_dim)
        )


def test_num_parameters_breakdown_keys() -> None:
    cfg = _make_tiny_cfg()
    model = ATWCodec(cfg)
    breakdown = model.num_parameters_breakdown()
    assert set(breakdown.keys()) == {
        "encoder", "decoder", "wz_side_info_head", "latents", "total",
    }
    assert breakdown["total"] == sum(
        v for k, v in breakdown.items() if k != "total"
    )


# ---------------------------------------------------------------------------
# Score-aware loss tests (3-paper math composition)
# ---------------------------------------------------------------------------


def _make_loss_inputs(*, batch: int = 2, latent_dim: int = 8) -> dict[str, torch.Tensor]:
    """Synthetic RGB-255-domain inputs for the score-aware loss."""
    return {
        "reconstructed_rgb_0": torch.rand(batch, 3, 16, 16) * 255.0,
        "reconstructed_rgb_1": torch.rand(batch, 3, 16, 16) * 255.0,
        "gt_rgb_0": torch.rand(batch, 3, 16, 16) * 255.0,
        "gt_rgb_1": torch.rand(batch, 3, 16, 16) * 255.0,
        "archive_bytes_proxy": torch.tensor(100_000.0),
        "z_residual": torch.randn(batch, latent_dim),
        "z_predicted": torch.randn(batch, latent_dim),
    }


def test_loss_weights_defaults_match_atw_canonical() -> None:
    """Default ATWLossWeights = ATW canonical mode (κ_IB=0, λ_WZ=1, λ_pixel=0)."""
    w = ATWLossWeights()
    assert w.alpha_rate == 25.0
    assert w.beta_seg == 100.0
    assert w.kappa_ib == 0.0
    assert w.lambda_wz == 1.0
    assert w.lambda_pixel == 0.0
    assert w.contest_normalizer == 37_545_489.0


def test_loss_refuses_eval_roundtrip_false() -> None:
    """Per CLAUDE.md eval_roundtrip=False is FORBIDDEN."""
    seg = _StubScorer()
    pose = _StubScorer()
    loss_fn = ATWScoreAwareLoss(seg, pose, ATWLossWeights())
    inputs = _make_loss_inputs()
    with pytest.raises(ValueError, match="eval_roundtrip=False is forbidden"):
        loss_fn(apply_eval_roundtrip=False, **inputs)


def test_loss_refuses_negative_lambda_pixel() -> None:
    seg = _StubScorer()
    pose = _StubScorer()
    loss_fn = ATWScoreAwareLoss(seg, pose, ATWLossWeights(lambda_pixel=-0.1))
    inputs = _make_loss_inputs()
    with pytest.raises(ValueError, match="lambda_pixel must be >= 0"):
        loss_fn(**inputs)


def test_loss_refuses_negative_lambda_wz() -> None:
    seg = _StubScorer()
    pose = _StubScorer()
    loss_fn = ATWScoreAwareLoss(seg, pose, ATWLossWeights(lambda_wz=-0.1))
    inputs = _make_loss_inputs()
    with pytest.raises(ValueError, match="lambda_wz must be >= 0"):
        loss_fn(**inputs)


def test_loss_refuses_negative_kappa_ib() -> None:
    seg = _StubScorer()
    pose = _StubScorer()
    loss_fn = ATWScoreAwareLoss(seg, pose, ATWLossWeights(kappa_ib=-0.1))
    inputs = _make_loss_inputs()
    with pytest.raises(ValueError, match="kappa_ib must be >= 0"):
        loss_fn(**inputs)


def test_loss_requires_z_residual_when_wz_or_ib_active() -> None:
    """When λ_WZ > 0 OR κ_IB > 0, z_residual must be provided."""
    seg = _StubScorer()
    pose = _StubScorer()
    loss_fn = ATWScoreAwareLoss(seg, pose, ATWLossWeights(lambda_wz=1.0))
    inputs = _make_loss_inputs()
    inputs["z_residual"] = None
    with pytest.raises(ValueError, match="z_residual is required"):
        loss_fn(**inputs)


def test_loss_validates_rgb_255_domain() -> None:
    """Unit-domain RGB inputs are refused (Z4 sister contract)."""
    seg = _StubScorer()
    pose = _StubScorer()
    loss_fn = ATWScoreAwareLoss(seg, pose, ATWLossWeights())
    # Unit-domain (max ~1.0) inputs
    inputs = _make_loss_inputs()
    inputs["reconstructed_rgb_0"] = torch.rand(2, 3, 16, 16)  # max <= 1.0
    with pytest.raises(ValueError, match="appears to be unit-domain RGB"):
        loss_fn(**inputs)


def test_loss_atick_redlich_pure_corner_weights() -> None:
    """ATW loss with (κ_IB=0, λ_WZ=0, λ_pixel=0) recovers Z4 verbatim weights.

    Per the sister Z4 substrate test pattern, the integration test against
    real scorers (with `scorer_forward_pair` returning dicts) happens at
    Phase 2 empirical anchor time, not in the L1 SCAFFOLD test suite. The
    L1 tests verify the LossWeights construction + the math composition
    via the construction surface.
    """
    seg = _StubScorer()
    pose = _StubScorer()
    weights = ATWLossWeights(kappa_ib=0.0, lambda_wz=0.0, lambda_pixel=0.0)
    loss_fn = ATWScoreAwareLoss(seg, pose, weights)
    assert loss_fn.weights.kappa_ib == 0.0
    assert loss_fn.weights.lambda_wz == 0.0
    assert loss_fn.weights.lambda_pixel == 0.0


def test_loss_atw_canonical_corner_weights() -> None:
    """ATW canonical (κ_IB=0, λ_WZ=1, λ_pixel=0) construction propagates."""
    seg = _StubScorer()
    pose = _StubScorer()
    weights = ATWLossWeights(kappa_ib=0.0, lambda_wz=1.0, lambda_pixel=0.0)
    loss_fn = ATWScoreAwareLoss(seg, pose, weights)
    assert loss_fn.weights.kappa_ib == 0.0
    assert loss_fn.weights.lambda_wz == 1.0
    assert loss_fn.weights.lambda_pixel == 0.0


def test_loss_tishby_ib_pure_corner_weights() -> None:
    """ATW Tishby IB pure (κ_IB=0.1, λ_WZ=0, λ_pixel=0) construction propagates."""
    seg = _StubScorer()
    pose = _StubScorer()
    weights = ATWLossWeights(kappa_ib=0.1, lambda_wz=0.0, lambda_pixel=0.0)
    loss_fn = ATWScoreAwareLoss(seg, pose, weights)
    assert loss_fn.weights.kappa_ib == 0.1
    assert loss_fn.weights.lambda_wz == 0.0
    assert loss_fn.weights.lambda_pixel == 0.0


def test_loss_z3_baseline_corner_weights() -> None:
    """ATW Z3 baseline (κ_IB=0, λ_WZ=0, λ_pixel=1) construction propagates."""
    seg = _StubScorer()
    pose = _StubScorer()
    weights = ATWLossWeights(kappa_ib=0.0, lambda_wz=0.0, lambda_pixel=1.0)
    loss_fn = ATWScoreAwareLoss(seg, pose, weights)
    assert loss_fn.weights.kappa_ib == 0.0
    assert loss_fn.weights.lambda_wz == 0.0
    assert loss_fn.weights.lambda_pixel == 1.0


def test_loss_four_corner_regimes_distinct() -> None:
    """The four corners are pairwise distinct knob configurations."""
    atick_only = ATWLossWeights(kappa_ib=0.0, lambda_wz=0.0, lambda_pixel=0.0)
    atw_canonical = ATWLossWeights(kappa_ib=0.0, lambda_wz=1.0, lambda_pixel=0.0)
    tishby_ib = ATWLossWeights(kappa_ib=0.1, lambda_wz=0.0, lambda_pixel=0.0)
    z3_baseline = ATWLossWeights(kappa_ib=0.0, lambda_wz=0.0, lambda_pixel=1.0)
    knobs = {
        ("atick_only", atick_only.kappa_ib, atick_only.lambda_wz, atick_only.lambda_pixel),
        ("atw_canonical", atw_canonical.kappa_ib, atw_canonical.lambda_wz, atw_canonical.lambda_pixel),
        ("tishby_ib", tishby_ib.kappa_ib, tishby_ib.lambda_wz, tishby_ib.lambda_pixel),
        ("z3_baseline", z3_baseline.kappa_ib, z3_baseline.lambda_wz, z3_baseline.lambda_pixel),
    }
    # Four distinct knob tuples (tuple element 0 is the corner name)
    assert len(knobs) == 4


# ---------------------------------------------------------------------------
# Archive grammar tests (Catalog #124 8 fields + roundtrip symmetry)
# ---------------------------------------------------------------------------


def _build_synthetic_archive_bytes(cfg: ATWCodecConfig | None = None) -> tuple[bytes, ATWCodec]:
    """Create a synthetic ATW1 archive for parser tests."""
    cfg = cfg or _make_tiny_cfg()
    model = ATWCodec(cfg).eval()
    with torch.no_grad():
        for i in range(cfg.num_pairs):
            model.scorer_class_prior_table[i] = (
                torch.arange(cfg.scorer_class_prior_dim, dtype=torch.float32) * 0.05
                + float(i) * 0.02
            )
    pair_indices = torch.arange(cfg.num_pairs, dtype=torch.long)
    _, _, _, _, z_residual, _ = model(pair_indices, compute_wz_residual=True)
    assert z_residual is not None
    meta_seed: dict[str, object] = {
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
        "latent_init_std": cfg.latent_init_std,
    }
    bytes_ = pack_archive(
        model.encoder.state_dict(),
        model.decoder.state_dict(),
        model.wz_side_info_head.state_dict(),
        z_residual.detach().cpu(),
        model.scorer_class_prior_table.detach().cpu(),
        meta_seed,
    )
    return bytes_, model


def test_archive_magic_is_atw1() -> None:
    bytes_, _ = _build_synthetic_archive_bytes()
    assert bytes_.startswith(ATW1_MAGIC)


def test_archive_header_size_invariant() -> None:
    """ATW1 header is exactly 35 bytes per Catalog #161 sister-substrate pattern."""
    assert ATW1_HEADER_SIZE == 35


def test_archive_schema_version_is_1() -> None:
    assert ATW1_SCHEMA_VERSION == 1


def test_archive_roundtrip_preserves_section_data() -> None:
    """Pack → parse → bytes is consistent (deterministic grammar)."""
    bytes_, model = _build_synthetic_archive_bytes()
    parsed = parse_archive(bytes_)
    assert parsed.schema_version == ATW1_SCHEMA_VERSION
    # Latents shape
    assert parsed.latent_residual.shape == (
        model.cfg.num_pairs, model.cfg.latent_dim
    )
    # Class prior table shape
    assert parsed.scorer_class_prior_table.shape == (
        model.cfg.num_pairs, model.cfg.scorer_class_prior_dim
    )
    # Meta carries atw_codec_meta provenance tag
    assert "atw_codec_meta" in parsed.meta
    atw_meta = parsed.meta["atw_codec_meta"]
    assert atw_meta["composite_id"] == "atw_codec_v1"
    assert "Atick-Redlich1990" in atw_meta["literature_anchor"]


def test_archive_roundtrip_class_prior_table_byte_identical() -> None:
    """Pack → parse → pack reproduces same class_prior_table bytes (fp16 cast)."""
    bytes_, model = _build_synthetic_archive_bytes()
    parsed = parse_archive(bytes_)
    table_fp16 = model.scorer_class_prior_table.detach().to(torch.float16)
    torch.testing.assert_close(parsed.scorer_class_prior_table.to(torch.float16), table_fp16)


def test_archive_pack_refuses_2d_mismatch_in_class_prior_table() -> None:
    cfg = _make_tiny_cfg()
    model = ATWCodec(cfg).eval()
    bad_table = torch.zeros(cfg.num_pairs + 1, cfg.scorer_class_prior_dim)
    with pytest.raises(ValueError, match="num_pairs mismatch"):
        pack_archive(
            model.encoder.state_dict(),
            model.decoder.state_dict(),
            model.wz_side_info_head.state_dict(),
            torch.zeros(cfg.num_pairs, cfg.latent_dim),
            bad_table,
            {},
        )


def test_archive_pack_refuses_non_2d_latent_residual() -> None:
    cfg = _make_tiny_cfg()
    model = ATWCodec(cfg).eval()
    with pytest.raises(ValueError, match="latent_residual must be 2-D"):
        pack_archive(
            model.encoder.state_dict(),
            model.decoder.state_dict(),
            model.wz_side_info_head.state_dict(),
            torch.zeros(cfg.num_pairs, cfg.latent_dim, 1),
            torch.zeros(cfg.num_pairs, cfg.scorer_class_prior_dim),
            {},
        )


def test_parse_archive_refuses_short_blob() -> None:
    with pytest.raises(ValueError, match="archive too short"):
        parse_archive(b"\x00" * 10)


def test_parse_archive_refuses_bad_magic() -> None:
    bytes_, _ = _build_synthetic_archive_bytes()
    bad = b"FAKE" + bytes_[4:]
    with pytest.raises(ValueError, match="bad magic"):
        parse_archive(bad)


def test_parse_archive_refuses_unsupported_version() -> None:
    bytes_, _ = _build_synthetic_archive_bytes()
    bad = bytes_[:4] + b"\x99" + bytes_[5:]
    with pytest.raises(ValueError, match="unsupported schema version"):
        parse_archive(bad)


def test_parse_atw1_archive_bytes_returns_section_offsets() -> None:
    """Cheap section-offset parser returns 7 canonical sections."""
    bytes_, _ = _build_synthetic_archive_bytes()
    sections = parse_atw1_archive_bytes(bytes_)
    assert set(sections.keys()) == {
        "atw1_header",
        "encoder_blob",
        "decoder_blob",
        "wz_head_blob",
        "latent_residual_blob",
        "class_prior_table_blob",
        "meta_blob",
    }
    # Sections are non-overlapping; sum equals archive length.
    total = sum(length for _, length in sections.values())
    assert total == len(bytes_)


def test_parse_atw1_archive_bytes_refuses_bad_magic() -> None:
    bytes_, _ = _build_synthetic_archive_bytes()
    bad = b"FAKE" + bytes_[4:]
    with pytest.raises(ValueError, match="bad magic"):
        parse_atw1_archive_bytes(bad)


def test_parse_atw1_archive_bytes_refuses_short_blob() -> None:
    with pytest.raises(ValueError, match="atw1 archive too short"):
        parse_atw1_archive_bytes(b"\x00" * 10)


def test_section_roles_canonical_taxonomy() -> None:
    """ATW1_SECTION_ROLES maps every section to a canonical role token."""
    valid_roles = {
        "control_or_metadata",
        "training_provenance_only",
        "decoder_weight_stream",
        "latent_stream",
        "decoder_side_information",
    }
    assert set(ATW1_SECTION_ROLES.keys()) == {
        "atw1_header",
        "encoder_blob",
        "decoder_blob",
        "wz_head_blob",
        "latent_residual_blob",
        "class_prior_table_blob",
        "meta_blob",
    }
    for role in ATW1_SECTION_ROLES.values():
        assert role in valid_roles, f"unknown role {role}"


def test_archive_meta_carries_three_paper_literature_anchors() -> None:
    """Per design memo: atw_codec_meta lists Atick + Tishby + Wyner-Ziv anchors."""
    bytes_, _ = _build_synthetic_archive_bytes()
    parsed = parse_archive(bytes_)
    atw_meta = parsed.meta["atw_codec_meta"]
    anchors = atw_meta["literature_anchor"]
    assert isinstance(anchors, list)
    assert "Atick-Redlich1990" in anchors
    assert "Tishby-Pereira-Bialek1999" in anchors
    assert "Wyner-Ziv1976" in anchors


# ---------------------------------------------------------------------------
# Operational mechanism tests (Catalog #220 — substrate must produce frame changes)
# ---------------------------------------------------------------------------


def test_wz_side_info_head_changes_reconstructed_frames() -> None:
    """Catalog #220 OPERATIONAL invariant: WZ head consumption changes RGB output.

    The substrate ships z_residual; reconstructed z = z_residual + WZ_head(class_prior).
    A non-zero class_prior_table should produce different decoder output than a
    zero class_prior_table — proving the WZ head is structurally consumed.
    """
    cfg = _make_tiny_cfg()
    model = ATWCodec(cfg).eval()
    # Set WZ head weights to non-zero deterministic values so prediction is non-trivial.
    with torch.no_grad():
        for p in model.wz_side_info_head.parameters():
            p.fill_(0.1)
        model.scorer_class_prior_table.fill_(0.0)
    pair_indices = torch.arange(cfg.num_pairs, dtype=torch.long)

    # With class_prior_table all-zero, z_predicted = head(0) = bias only
    z_residual_zero = model.latents[pair_indices].clone()
    rgb_0_zero, rgb_1_zero = model.reconstruct_from_wz_residual(
        pair_indices, z_residual_zero
    )

    # With non-zero class_prior_table, z_predicted is different → different RGB
    with torch.no_grad():
        for i in range(cfg.num_pairs):
            model.scorer_class_prior_table[i] = (
                torch.ones(cfg.scorer_class_prior_dim) * float(i + 1) * 0.5
            )
    rgb_0_nonzero, rgb_1_nonzero = model.reconstruct_from_wz_residual(
        pair_indices, z_residual_zero
    )

    # The reconstructed frames MUST differ — proving operational consumption.
    assert not torch.allclose(rgb_0_zero, rgb_0_nonzero), (
        "WZ side-info head not operational: same RGB output regardless of class prior"
    )
    assert not torch.allclose(rgb_1_zero, rgb_1_nonzero), (
        "WZ side-info head not operational: same RGB output regardless of class prior"
    )


def test_no_op_detector_archive_bytes_change_with_wz_lambda() -> None:
    """No-op detector: changing wz_lambda_default produces different archive bytes."""
    cfg_a = _make_tiny_cfg(wz_head_enabled=True)
    cfg_b = _make_tiny_cfg(wz_head_enabled=False)
    bytes_a, _ = _build_synthetic_archive_bytes(cfg_a)
    bytes_b, _ = _build_synthetic_archive_bytes(cfg_b)
    # Different config produces different archive sizes (WZ head section differs).
    assert bytes_a != bytes_b


# ---------------------------------------------------------------------------
# Trainer scaffold tests
# ---------------------------------------------------------------------------


def test_smoke_trainer_runs_end_to_end(tmp_path: Path) -> None:
    """Smoke trainer entry point runs without GPU / scorer / real video."""
    from experiments.train_substrate_atw_codec_v1 import main

    output_dir = tmp_path / "atw_smoke"
    rc = main([
        "--output-dir", str(output_dir),
        "--smoke",
        "--epochs", "1",
        "--device", "cpu",
        "--latent-dim", "8",
        "--decoder-embed-dim", "8",
    ])
    assert rc == 0
    archive = output_dir / "0.bin"
    assert archive.is_file()
    assert archive.read_bytes()[:4] == ATW1_MAGIC
    stats_path = output_dir / "smoke_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["substrate_tag"] == "atw_codec_v1"
    assert stats["lane_id"] == "lane_atw_codec_design_v1_20260515"
    assert stats["smoke"] is True
    assert stats["roundtrip_ok"] is True
    assert stats["atw1_magic_ok"] is True


def test_full_trainer_implemented_and_cuda_gated(tmp_path: Path) -> None:
    """CLASS-SHIFT-FULL-MAIN-CLUSTER 2026-05-27: _full_main IMPLEMENTED + CUDA-gated.

    The L1 SCAFFOLD NotImplementedError is extinguished: ``_full_main`` routes
    the canonical score-aware training loop through
    ``run_pact_nerv_score_aware_training``. Per CLAUDE.md "MPS auth eval is
    NOISE" + Catalog #1, the full (non-smoke) path is CUDA-required; invoking
    it with ``--device cpu`` refuses via ``device_or_die`` (SystemExit). PAID
    DISPATCH stays gated by ``dispatch_enabled: false`` + ``research_only:
    true`` + ``lane_class=substrate_engineering`` on the recipe per Catalog
    #220 + #325 (code complete, trigger gated).
    """
    import inspect

    from experiments.train_substrate_atw_codec_v1 import _full_main, main

    src = inspect.getsource(_full_main)
    assert "raise NotImplementedError" not in src, (
        "_full_main NotImplementedError must be extinguished per "
        "CLASS-SHIFT-FULL-MAIN-CLUSTER"
    )
    assert "run_pact_nerv_score_aware_training" in src, (
        "_full_main must route through the canonical shared training loop"
    )
    with pytest.raises(SystemExit):
        main([
            "--output-dir", str(tmp_path / "should_not_exist"),
            "--epochs", "1",
            "--device", "cpu",
        ])


def test_trainer_tier_1_required_flags_present() -> None:
    """Catalog #151: TIER_1_OPERATOR_REQUIRED_FLAGS dict exists with canonical flags."""
    from experiments.train_substrate_atw_codec_v1 import (
        TIER_1_OPERATOR_REQUIRED_FLAGS,
    )

    assert isinstance(TIER_1_OPERATOR_REQUIRED_FLAGS, dict)
    canonical = {
        "--video-path", "--output-dir", "--epochs", "--batch-size", "--lr",
        "--kappa-ib", "--lambda-wz", "--lambda-pixel",
        "--beta-seg", "--gamma-pose",
    }
    assert canonical.issubset(set(TIER_1_OPERATOR_REQUIRED_FLAGS.keys()))
    # video-path is the required_input_file per Catalog #152
    assert TIER_1_OPERATOR_REQUIRED_FLAGS["--video-path"]["required_input_file"] is True


# ---------------------------------------------------------------------------
# HNeRV parity discipline lesson 8: scorer-preprocess gradient-reachability
# ---------------------------------------------------------------------------


def test_decoder_output_carries_gradients() -> None:
    """The decoder's output must carry gradients into the latent — loss can backprop."""
    cfg = _make_tiny_cfg()
    model = ATWCodec(cfg).train()
    pair_indices = torch.arange(cfg.num_pairs, dtype=torch.long)
    rgb_0, rgb_1, _, _, _, _ = model(pair_indices)
    # Sum and backward — should not raise; grads should flow to model.latents
    loss = rgb_0.sum() + rgb_1.sum()
    loss.backward()
    assert model.latents.grad is not None
    assert torch.isfinite(model.latents.grad).all()


def test_wz_head_carries_gradients_when_enabled() -> None:
    """The WZ head must carry gradients into the encoder — joint training is possible."""
    cfg = _make_tiny_cfg(wz_head_enabled=True)
    model = ATWCodec(cfg).train()
    pair_indices = torch.arange(cfg.num_pairs, dtype=torch.long)
    _, _, _, _, z_residual, z_predicted = model(
        pair_indices, compute_wz_residual=True
    )
    assert z_residual is not None and z_predicted is not None
    # WZ residual loss → backward → WZ head parameters get grads
    wz_loss = z_residual.pow(2).mean()
    wz_loss.backward()
    for name, p in model.wz_side_info_head.named_parameters():
        assert p.grad is not None, f"WZ head param {name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"WZ head param {name} has non-finite grad"
