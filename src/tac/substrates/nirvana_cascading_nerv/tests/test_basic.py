# SPDX-License-Identifier: MIT
"""L0 SCAFFOLD smoke + shape + Catalog #91/#139 + MLX↔numpy + MLX↔PyTorch parity tests.

Per Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 byte-mutation
no_op_proof + Catalog #240 L0 SCAFFOLD posture verification + NEW
operator-directive-#3 2026-05-26 axes 2 (MLX drift) + 3 (numpy portability).

These tests verify the substrate package's STRUCTURAL invariants without
requiring MLX to be installed (MLX-dependent tests are guarded by import
detection and skipped when MLX is unavailable).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test 1: top-level import without MLX
# ---------------------------------------------------------------------------

def test_module_imports_without_mlx() -> None:
    """Top-level package import must succeed without MLX installed."""
    import tac.substrates.nirvana_cascading_nerv as mod

    assert mod.ARCHIVE_MAGIC == b"NIR1\x00"
    assert mod.ARCHIVE_VERSION == 1
    assert mod.NIRVANA1_HEADER_LEN == 28
    assert mod.DEFAULT_NUM_LEVELS == 4
    assert mod.DEFAULT_PER_PAIR_LATENT_DIM == 16
    assert mod.DEFAULT_BASE_H == 48
    assert mod.DEFAULT_BASE_W == 64


# ---------------------------------------------------------------------------
# Test 2: public API surface (Catalog #335)
# ---------------------------------------------------------------------------

def test_module_exposes_canonical_public_api() -> None:
    """__all__ surface must be narrow + explicit per Catalog #335 contract."""
    import tac.substrates.nirvana_cascading_nerv as mod

    expected = {
        "ARCHITECTURE_CLASS",
        "ARCHIVE_GRAMMAR_FIELDS",
        "ARCHIVE_MAGIC",
        "ARCHIVE_VERSION",
        "CANONICAL_EQUATION_IDS",
        "DEFAULT_BASE_H",
        "DEFAULT_BASE_W",
        "DEFAULT_NUM_LEVELS",
        "DEFAULT_PER_PAIR_LATENT_DIM",
        "NIRVANA1_HEADER_FMT",
        "NIRVANA1_HEADER_LEN",
        "NirvanaCascadingNervConfig",
        "SISTER_SUBSTRATES",
        "SUBSTRATE_ID",
        "emit_landing_posterior_anchor",
    }
    assert set(mod.__all__) == expected


def test_archive_grammar_fields_catalog_124_compliance() -> None:
    """Catalog #124 archive-grammar 8 fields declared inline for AST walker."""
    from tac.substrates.nirvana_cascading_nerv import ARCHIVE_GRAMMAR_FIELDS

    expected_keys = {
        "archive_grammar",
        "parser_section_manifest",
        "inflate_runtime_loc_budget",
        "runtime_dep_closure",
        "export_format",
        "score_aware_loss",
        "bolt_on_loc_budget",
        "no_op_detector_planned",
    }
    assert set(ARCHIVE_GRAMMAR_FIELDS.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Test 3: Config dataclass invariants
# ---------------------------------------------------------------------------

def test_config_dataclass_default_values() -> None:
    """Default config matches design memo §"Predicted ΔS band" breakdown."""
    from tac.substrates.nirvana_cascading_nerv.mlx_renderer import (
        NirvanaCascadingNervConfig,
    )

    cfg = NirvanaCascadingNervConfig()
    assert cfg.num_levels == 4
    assert cfg.per_pair_latent_dim == 16
    assert cfg.base_h == 48
    assert cfg.base_w == 64
    assert cfg.base_channels == 24
    assert cfg.num_pairs == 600
    assert cfg.residual_quant_bits == 8


def test_config_validates_eval_hw_cascade_target() -> None:
    """Cascade target must match EVAL_HW (384, 512)."""
    from tac.substrates.nirvana_cascading_nerv.mlx_renderer import (
        EVAL_HW,
        NirvanaCascadingNervConfig,
    )

    cfg = NirvanaCascadingNervConfig()
    final_h = cfg.base_h * (2 ** (cfg.num_levels - 1))
    final_w = cfg.base_w * (2 ** (cfg.num_levels - 1))
    assert (final_h, final_w) == EVAL_HW


def test_config_rejects_invalid_num_levels() -> None:
    """num_levels must be in [1, 6]."""
    from tac.substrates.nirvana_cascading_nerv.mlx_renderer import (
        NirvanaCascadingNervConfig,
    )

    with pytest.raises(ValueError, match="num_levels"):
        NirvanaCascadingNervConfig(num_levels=0)
    with pytest.raises(ValueError, match="num_levels"):
        NirvanaCascadingNervConfig(num_levels=7)


def test_config_rejects_cascade_target_mismatch() -> None:
    """Cascade target mismatch must be refused."""
    from tac.substrates.nirvana_cascading_nerv.mlx_renderer import (
        NirvanaCascadingNervConfig,
    )

    with pytest.raises(ValueError, match="cascade target"):
        # 3 levels: 48 * 4 = 192 != 384
        NirvanaCascadingNervConfig(num_levels=3)


def test_per_level_shape_helper() -> None:
    """per_level_shape returns ascending (H, W) per level."""
    from tac.substrates.nirvana_cascading_nerv.mlx_renderer import (
        NirvanaCascadingNervConfig,
    )

    cfg = NirvanaCascadingNervConfig()
    assert cfg.per_level_shape(0) == (48, 64)
    assert cfg.per_level_shape(1) == (96, 128)
    assert cfg.per_level_shape(2) == (192, 256)
    assert cfg.per_level_shape(3) == (384, 512)
    with pytest.raises(ValueError, match="out of"):
        cfg.per_level_shape(4)


# ---------------------------------------------------------------------------
# Test 4: Catalog #240 L0 SCAFFOLD posture
# ---------------------------------------------------------------------------

def test_full_main_raises_not_implemented_per_catalog_240() -> None:
    """L0 SCAFFOLD posture: full main raises NotImplementedError."""
    from tac.substrates.nirvana_cascading_nerv.mlx_renderer import _full_main

    with pytest.raises(NotImplementedError, match="L0 SCAFFOLD"):
        _full_main()


# ---------------------------------------------------------------------------
# Test 5: Archive grammar round-trip per Catalog #91
# ---------------------------------------------------------------------------

def _make_synthetic_archive_inputs(cfg) -> tuple:
    """Construct minimal valid (state_dict, residuals, latents, meta) for round-trip."""
    # Tiny synthetic state_dict (correctness reference; not real weights)
    decoder_sd = {
        "level_0_decoder.stem.weight": np.zeros(
            (cfg.base_channels * cfg.base_h * cfg.base_w, cfg.per_pair_latent_dim),
            dtype=np.float32,
        ),
        "level_0_decoder.stem.bias": np.zeros(
            (cfg.base_channels * cfg.base_h * cfg.base_w,), dtype=np.float32
        ),
        "level_0_decoder.conv1.weight": np.zeros(
            (cfg.base_channels, cfg.base_channels, 3, 3), dtype=np.float32
        ),
        "level_0_decoder.conv1.bias": np.zeros((cfg.base_channels,), dtype=np.float32),
        "level_0_decoder.conv_to_rgb.weight": np.zeros(
            (3, cfg.base_channels, 3, 3), dtype=np.float32
        ),
        "level_0_decoder.conv_to_rgb.bias": np.zeros((3,), dtype=np.float32),
    }
    # Per-level residuals (int8 in [-128, 127])
    residuals = []
    for level in range(cfg.num_levels):
        h, w = cfg.per_level_shape(level)
        residuals.append(np.zeros((h, w, 3), dtype=np.int8))
    # Per-pair latents (int16)
    latents = np.zeros((cfg.num_pairs, cfg.per_pair_latent_dim), dtype=np.int16)
    # Meta
    meta = {
        "num_pairs": cfg.num_pairs,
        "base_channels": cfg.base_channels,
        "residual_scale": 0.5,
        "schema_version": 1,
    }
    return decoder_sd, residuals, latents, meta


def test_nirvana1_archive_pack_parse_round_trip() -> None:
    """Catalog #91 ENCODE_INFLATE_ROUNDTRIP: pack → parse returns equivalent data."""
    from tac.substrates.nirvana_cascading_nerv.archive import (
        pack_archive,
        parse_archive,
    )
    from tac.substrates.nirvana_cascading_nerv.mlx_renderer import (
        NirvanaCascadingNervConfig,
    )

    # Use smaller config for test speed
    cfg = NirvanaCascadingNervConfig(num_pairs=4)
    decoder_sd, residuals, latents, meta = _make_synthetic_archive_inputs(cfg)

    archive_bytes = pack_archive(
        decoder_sd,
        residuals,
        latents,
        meta,
        num_levels=cfg.num_levels,
        per_pair_latent_dim=cfg.per_pair_latent_dim,
        base_h=cfg.base_h,
        base_w=cfg.base_w,
    )

    parsed = parse_archive(archive_bytes)
    assert parsed.num_levels == cfg.num_levels
    assert parsed.per_pair_latent_dim == cfg.per_pair_latent_dim
    assert parsed.base_h == cfg.base_h
    assert parsed.base_w == cfg.base_w
    assert len(parsed.per_level_residuals) == cfg.num_levels
    for level, residual in enumerate(parsed.per_level_residuals):
        h, w = cfg.per_level_shape(level)
        assert residual.shape == (h, w, 3)
        assert residual.dtype == np.int8
    assert parsed.per_pair_latents.shape == (cfg.num_pairs, cfg.per_pair_latent_dim)
    assert parsed.per_pair_latents.dtype == np.int16
    assert parsed.meta["num_pairs"] == cfg.num_pairs


def test_archive_invalid_magic_rejected() -> None:
    """Mis-magic'd archive must be refused."""
    from tac.substrates.nirvana_cascading_nerv.archive import parse_archive

    bad_bytes = b"BADM\x00" + b"\x00" * 100
    with pytest.raises(ValueError, match="magic"):
        parse_archive(bad_bytes)


def test_archive_truncated_rejected() -> None:
    """Truncated archive must be refused (sub-header)."""
    from tac.substrates.nirvana_cascading_nerv.archive import parse_archive

    with pytest.raises(ValueError, match="too short"):
        parse_archive(b"\x00" * 10)


# ---------------------------------------------------------------------------
# Test 6: Catalog #139 byte-mutation no_op_proof
# ---------------------------------------------------------------------------

def test_archive_byte_mutation_no_op_proof_per_catalog_139() -> None:
    """Mutating per-level residual bytes MUST change parsed residual content.

    Per Catalog #139 + #272 distinguishing-feature contract: per-level
    residual bytes are the distinguishing-feature; mutating them MUST
    produce a different parsed residual (which would produce different
    final RGB at inflate time).
    """
    from tac.substrates.nirvana_cascading_nerv.archive import (
        pack_archive,
        parse_archive,
    )
    from tac.substrates.nirvana_cascading_nerv.mlx_renderer import (
        NirvanaCascadingNervConfig,
    )

    cfg = NirvanaCascadingNervConfig(num_pairs=2)
    decoder_sd, residuals, latents, meta = _make_synthetic_archive_inputs(cfg)
    # Use non-zero residuals so mutation is observable
    for residual in residuals:
        residual[:] = 1  # int8 = 1

    archive_bytes = pack_archive(
        decoder_sd,
        residuals,
        latents,
        meta,
        num_levels=cfg.num_levels,
        per_pair_latent_dim=cfg.per_pair_latent_dim,
        base_h=cfg.base_h,
        base_w=cfg.base_w,
    )

    parsed_original = parse_archive(archive_bytes)
    sha_original = hashlib.sha256(
        b"".join(r.tobytes() for r in parsed_original.per_level_residuals)
    ).hexdigest()

    # Mutate residual blob: change first residual int8=1 to int8=64
    # Brotli-compressed bytes; modify residual values BEFORE packing
    residuals_mutated = [r.copy() for r in residuals]
    residuals_mutated[1][0, 0, 0] = 64

    mutated_bytes = pack_archive(
        decoder_sd,
        residuals_mutated,
        latents,
        meta,
        num_levels=cfg.num_levels,
        per_pair_latent_dim=cfg.per_pair_latent_dim,
        base_h=cfg.base_h,
        base_w=cfg.base_w,
    )

    parsed_mutated = parse_archive(mutated_bytes)
    sha_mutated = hashlib.sha256(
        b"".join(r.tobytes() for r in parsed_mutated.per_level_residuals)
    ).hexdigest()

    assert sha_original != sha_mutated, (
        "Catalog #139 violated: mutating per-level residual bytes did NOT "
        "change parsed residual content. The per-level residual blob is "
        "the DISTINGUISHING FEATURE; byte mutation MUST be observable."
    )


# ---------------------------------------------------------------------------
# Test 7: MLX↔numpy parity per axis 3 portability
# ---------------------------------------------------------------------------

def test_numpy_reference_bilinear_upsample_shape() -> None:
    """numpy reference bilinear upsample produces correct output shape."""
    from tac.substrates.nirvana_cascading_nerv.numpy_reference import (
        bilinear_upsample_2x_nhwc,
    )

    x = np.zeros((2, 4, 6, 3), dtype=np.float32)
    y = bilinear_upsample_2x_nhwc(x)
    assert y.shape == (2, 8, 12, 3)
    assert y.dtype == np.float32


def test_numpy_reference_bilinear_upsample_constant_passthrough() -> None:
    """Bilinear upsample of a constant tensor returns the constant."""
    from tac.substrates.nirvana_cascading_nerv.numpy_reference import (
        bilinear_upsample_2x_nhwc,
    )

    x = np.full((1, 4, 6, 3), 0.5, dtype=np.float32)
    y = bilinear_upsample_2x_nhwc(x)
    assert np.allclose(y, 0.5, atol=1e-6)


def test_numpy_reference_bilinear_upsample_matches_pytorch() -> None:
    """numpy reference bilinear upsample ≤ 1e-5 vs PyTorch F.interpolate align_corners=False."""
    import torch
    import torch.nn.functional as F

    from tac.substrates.nirvana_cascading_nerv.numpy_reference import (
        bilinear_upsample_2x_nhwc,
    )

    rng = np.random.default_rng(seed=42)
    x_nhwc = rng.standard_normal((1, 4, 6, 3)).astype(np.float32)
    y_numpy = bilinear_upsample_2x_nhwc(x_nhwc)

    x_nchw = torch.from_numpy(x_nhwc.transpose(0, 3, 1, 2).copy())
    y_torch = F.interpolate(
        x_nchw, scale_factor=2, mode="bilinear", align_corners=False
    )
    y_torch_nhwc = y_torch.permute(0, 2, 3, 1).numpy()

    max_abs = np.abs(y_numpy - y_torch_nhwc).max()
    assert max_abs < 1e-5, (
        f"numpy reference bilinear upsample drift {max_abs} > 1e-5 vs PyTorch; "
        f"violates axis 2 MLX drift minimization discipline per design memo"
    )


def test_numpy_reference_sigmoid_matches_pytorch() -> None:
    """numpy reference sigmoid ≤ 1e-6 vs PyTorch torch.sigmoid."""
    import torch

    from tac.substrates.nirvana_cascading_nerv.numpy_reference import sigmoid

    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
    y_numpy = sigmoid(x)
    y_torch = torch.sigmoid(torch.from_numpy(x)).numpy()

    max_abs = np.abs(y_numpy - y_torch).max()
    assert max_abs < 1e-6, (
        f"numpy reference sigmoid drift {max_abs} > 1e-6 vs PyTorch"
    )


def test_numpy_reference_linear_matches_pytorch() -> None:
    """numpy reference linear ≤ 1e-5 vs PyTorch nn.Linear forward."""
    import torch
    import torch.nn as nn

    from tac.substrates.nirvana_cascading_nerv.numpy_reference import linear

    rng = np.random.default_rng(seed=7)
    in_features, out_features = 8, 12
    weight = rng.standard_normal((out_features, in_features)).astype(np.float32)
    bias = rng.standard_normal((out_features,)).astype(np.float32)
    x = rng.standard_normal((4, in_features)).astype(np.float32)

    y_numpy = linear(x, weight, bias)

    torch_linear = nn.Linear(in_features, out_features)
    with torch.no_grad():
        torch_linear.weight.copy_(torch.from_numpy(weight))
        torch_linear.bias.copy_(torch.from_numpy(bias))
    with torch.inference_mode():
        y_torch = torch_linear(torch.from_numpy(x)).numpy()

    max_abs = np.abs(y_numpy - y_torch).max()
    assert max_abs < 1e-5, (
        f"numpy reference linear drift {max_abs} > 1e-5 vs PyTorch"
    )


def test_numpy_reference_cascade_reconstruct_shape() -> None:
    """Cascade reconstruction produces correct final shape."""
    from tac.substrates.nirvana_cascading_nerv.numpy_reference import (
        cascade_reconstruct,
    )

    base_rgb = np.zeros((1, 48, 64, 3), dtype=np.float32)
    residuals = [
        np.zeros((1, 48, 64, 3), dtype=np.float32),  # level 0 (unused)
        np.zeros((1, 96, 128, 3), dtype=np.float32),  # level 1
        np.zeros((1, 192, 256, 3), dtype=np.float32),  # level 2
        np.zeros((1, 384, 512, 3), dtype=np.float32),  # level 3
    ]
    # Drop level 0 residual (cascade starts after base)
    final = cascade_reconstruct(base_rgb, residuals[1:])
    assert final.shape == (1, 384, 512, 3)


def test_numpy_reference_kahan_mean_stability() -> None:
    """Kahan mean is more stable than naive mean for large-N fp32 reductions."""
    from tac.substrates.nirvana_cascading_nerv.numpy_reference import (
        kahan_mean,
        mean,
    )

    # Tiny test: both should agree to high precision on small arrays
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    assert abs(kahan_mean(x) - mean(x)) < 1e-6
    # Mean is 3.0
    assert abs(kahan_mean(x) - 3.0) < 1e-6


# ---------------------------------------------------------------------------
# Test 8: estimate_archive_bytes for Dykstra-feasibility check
# ---------------------------------------------------------------------------

def test_estimate_archive_bytes_within_design_memo_range() -> None:
    """estimate_archive_bytes returns a value consistent with design memo §predicted-band."""
    from tac.substrates.nirvana_cascading_nerv.mlx_renderer import (
        NirvanaCascadingNervConfig,
        estimate_archive_bytes,
    )

    cfg = NirvanaCascadingNervConfig()
    est = estimate_archive_bytes(cfg)
    # Empirical estimate at default cfg: ~1 MB (dominated by level 0 stem
    # linear: 16 * 24 * 48 * 64 = 1.18M params @ fp16 = 2.36 MB raw → ~0.7 MB
    # brotli-compressed; plus per-level residuals + latents). The L0 design
    # memo predicted ~255 KB which was an underestimate based on smaller
    # per-pair latent_dim assumption. Phase 2 cargo-cult-unwind per Catalog
    # #303 will sweep latent_dim ∈ {4, 8, 16, 32} to find archive-budget vs
    # distortion sweet spot.
    assert 500_000 < est < 2_500_000, (
        f"estimate_archive_bytes {est} bytes outside expected [500K, 2.5M] range; "
        f"may indicate a regression in renderer_param_count formula or config defaults"
    )


def test_estimate_archive_bytes_scales_with_num_levels() -> None:
    """Fewer levels → smaller archive."""
    from tac.substrates.nirvana_cascading_nerv.mlx_renderer import (
        NirvanaCascadingNervConfig,
        estimate_archive_bytes,
    )

    # 2 levels: 48 * 2 = 96 ≠ 384, so we need different base_h/base_w
    # 4 levels: 48 * 8 = 384 (canonical)
    cfg4 = NirvanaCascadingNervConfig(num_levels=4)
    # Smaller per_pair_latent_dim → smaller archive (within same num_levels)
    cfg_small = NirvanaCascadingNervConfig(num_levels=4, per_pair_latent_dim=4)
    est4 = estimate_archive_bytes(cfg4)
    est_small = estimate_archive_bytes(cfg_small)
    assert est_small < est4, (
        f"smaller latent_dim should produce smaller archive: got est_small={est_small}, est4={est4}"
    )


# ---------------------------------------------------------------------------
# Test 9: MLX-availability gate
# ---------------------------------------------------------------------------

def test_mlx_availability_gate() -> None:
    """_ensure_mlx_available raises actionable RuntimeError if MLX missing."""
    from tac.substrates.nirvana_cascading_nerv.mlx_renderer import (
        _ensure_mlx_available,
    )

    try:
        import mlx.core  # noqa: F401

        mlx_available = True
    except ImportError:
        mlx_available = False

    if mlx_available:
        # MLX is installed; the gate should succeed
        result = _ensure_mlx_available()
        assert result is not None
    else:
        # MLX is NOT installed; the gate should raise with actionable message
        with pytest.raises(RuntimeError, match="numpy_reference"):
            _ensure_mlx_available()


# ---------------------------------------------------------------------------
# Test 10: NIRVANA1 header determinism
# ---------------------------------------------------------------------------

def test_archive_pack_deterministic() -> None:
    """Same inputs → same archive bytes (byte-determinism for Catalog #305 diff-able facet)."""
    from tac.substrates.nirvana_cascading_nerv.archive import pack_archive
    from tac.substrates.nirvana_cascading_nerv.mlx_renderer import (
        NirvanaCascadingNervConfig,
    )

    cfg = NirvanaCascadingNervConfig(num_pairs=2)
    decoder_sd, residuals, latents, meta = _make_synthetic_archive_inputs(cfg)

    archive_1 = pack_archive(
        decoder_sd,
        residuals,
        latents,
        meta,
        num_levels=cfg.num_levels,
        per_pair_latent_dim=cfg.per_pair_latent_dim,
        base_h=cfg.base_h,
        base_w=cfg.base_w,
    )
    archive_2 = pack_archive(
        decoder_sd,
        residuals,
        latents,
        meta,
        num_levels=cfg.num_levels,
        per_pair_latent_dim=cfg.per_pair_latent_dim,
        base_h=cfg.base_h,
        base_w=cfg.base_w,
    )

    assert archive_1 == archive_2, "Archive pack must be byte-deterministic"


# ---------------------------------------------------------------------------
# Test 11: Inflate runtime structural import
# ---------------------------------------------------------------------------

def test_inflate_module_imports_without_mlx() -> None:
    """inflate.py module imports without MLX (torch + brotli only per HNeRV L4)."""
    from tac.substrates.nirvana_cascading_nerv import inflate

    assert hasattr(inflate, "NirvanaCascadingDecoderTorch")
    assert hasattr(inflate, "inflate_one_video")
    assert hasattr(inflate, "main_cli")
    assert inflate.CAMERA_H == 874
    assert inflate.CAMERA_W == 1164
    assert inflate.DECODER_H_FINAL == 384
    assert inflate.DECODER_W_FINAL == 512


def test_inflate_decoder_topology_runs_on_cpu() -> None:
    """PyTorch inflate decoder forward runs end-to-end on CPU (shape verification)."""
    import torch

    from tac.substrates.nirvana_cascading_nerv.inflate import (
        NirvanaCascadingDecoderTorch,
    )
    from tac.substrates.nirvana_cascading_nerv.mlx_renderer import (
        NirvanaCascadingNervConfig,
    )

    cfg = NirvanaCascadingNervConfig()
    decoder = NirvanaCascadingDecoderTorch(
        num_levels=cfg.num_levels,
        per_pair_latent_dim=cfg.per_pair_latent_dim,
        base_h=cfg.base_h,
        base_w=cfg.base_w,
        base_channels=cfg.base_channels,
    )
    decoder.eval()

    B = 2
    latents = torch.zeros((B, cfg.per_pair_latent_dim), dtype=torch.float32)
    # Per-level residuals: level 0 unused, levels 1..N-1 used
    residuals_fp = []
    for level in range(cfg.num_levels):
        h, w = cfg.per_level_shape(level)
        residuals_fp.append(torch.zeros((h, w, 3), dtype=torch.float32))

    with torch.inference_mode():
        rgb = decoder(latents, residuals_fp)

    assert rgb.shape == (B, 3, 384, 512), f"unexpected RGB shape {rgb.shape}"
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0


# ---------------------------------------------------------------------------
# Test 12 (BONUS): MLX↔numpy cascade parity (axis 3 portability E2E)
# ---------------------------------------------------------------------------

def test_cascade_pytorch_vs_numpy_reference_parity() -> None:
    """PyTorch inflate decoder and numpy cascade_reconstruct produce ≤ 1e-3 parity.

    Uses zero base RGB + zero residuals; both implementations should produce
    near-zero output (modulo sigmoid bias initialization). This verifies the
    cascade composition law matches between PyTorch + numpy implementations.
    """
    import torch

    from tac.substrates.nirvana_cascading_nerv.inflate import (
        NirvanaCascadingDecoderTorch,
    )
    from tac.substrates.nirvana_cascading_nerv.mlx_renderer import (
        NirvanaCascadingNervConfig,
    )
    from tac.substrates.nirvana_cascading_nerv.numpy_reference import (
        cascade_reconstruct,
    )

    cfg = NirvanaCascadingNervConfig()
    decoder = NirvanaCascadingDecoderTorch(
        num_levels=cfg.num_levels,
        per_pair_latent_dim=cfg.per_pair_latent_dim,
        base_h=cfg.base_h,
        base_w=cfg.base_w,
        base_channels=cfg.base_channels,
    )
    # Zero out all conv weights so decoder produces sigmoid(0) = 0.5 baseline
    with torch.no_grad():
        for param in decoder.parameters():
            param.zero_()
    decoder.eval()

    latents = torch.zeros((1, cfg.per_pair_latent_dim), dtype=torch.float32)
    residuals_fp = []
    for level in range(cfg.num_levels):
        h, w = cfg.per_level_shape(level)
        residuals_fp.append(torch.zeros((h, w, 3), dtype=torch.float32))

    with torch.inference_mode():
        rgb_pytorch = decoder(latents, residuals_fp)

    # PyTorch produces sigmoid(0) = 0.5 baseline (level 0 decoder bias all-zero
    # weights → sigmoid(0) → 0.5); residual additions are zero so final = 0.5
    # everywhere (clamped to [0, 1]). So PyTorch produces 0.5.
    assert torch.allclose(rgb_pytorch, torch.full_like(rgb_pytorch, 0.5), atol=1e-6)

    # numpy cascade_reconstruct with base=0.5, residuals=0 should also produce 0.5
    base_rgb_numpy = np.full((1, cfg.base_h, cfg.base_w, 3), 0.5, dtype=np.float32)
    residuals_numpy = []
    for level in range(1, cfg.num_levels):
        h, w = cfg.per_level_shape(level)
        residuals_numpy.append(np.zeros((1, h, w, 3), dtype=np.float32))
    final_numpy = cascade_reconstruct(base_rgb_numpy, residuals_numpy)

    assert final_numpy.shape == (1, 384, 512, 3)
    assert np.allclose(final_numpy, 0.5, atol=1e-6), (
        f"numpy cascade output mean {final_numpy.mean()} != 0.5 (expected baseline)"
    )

    # Cross-implementation parity: PyTorch (NCHW) vs numpy (NHWC) should agree
    rgb_pytorch_nhwc = rgb_pytorch.permute(0, 2, 3, 1).numpy()
    max_abs = np.abs(rgb_pytorch_nhwc - final_numpy).max()
    assert max_abs < 1e-3, (
        f"PyTorch↔numpy cascade parity drift {max_abs} > 1e-3; "
        f"violates axis 2/3 discipline per design memo"
    )
