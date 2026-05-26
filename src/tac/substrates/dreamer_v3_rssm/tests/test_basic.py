# SPDX-License-Identifier: MIT
"""Basic L0 SCAFFOLD tests for tac.substrates.dreamer_v3_rssm.

Covers (the minimum operationally faithful surface per CLAUDE.md "Substrate
scaffolds MUST be COMPLETE or RESEARCH-ONLY"):

- import + ARCHIVE_GRAMMAR_FIELDS structural declaration (Catalog #124 surface)
- canonical equation refs registered (Catalog #344)
- MLX module forward-pass shape contract (training + eval)
- archive pack/parse round-trip byte determinism
- MLX↔PyTorch parity at inflate-time decoder boundary (the canonical PR95
  pattern per Catalog #290 ADOPT-CANONICAL-BECAUSE-SERVES decision)
- end-to-end MLX-train → archive → PyTorch inflate → camera-resolution uint8

These tests are MLX-local research-signal per CLAUDE.md "MLX portable-local-
substrate authority"; results carry ``[macOS-MLX research-signal]`` axis tag
and are NOT contest score claims.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:  # pragma: no cover - skip whole module on non-Apple CI
    import mlx.core as mx
    from mlx.utils import tree_flatten
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]


pytestmark = pytest.mark.skipif(
    mx is None,
    reason="DreamerV3 RSSM L0 scaffold requires MLX (macOS Apple Silicon)",
)


def test_module_imports_and_archive_grammar_declared() -> None:
    """Module imports cleanly + Catalog #124 8-field declaration present."""
    from tac.substrates.dreamer_v3_rssm import (
        ARCHIVE_GRAMMAR_FIELDS,
        CANONICAL_EQUATION_IDS,
        DEFAULT_G,
        DEFAULT_K,
        EVAL_HW,
        NUM_PAIRS,
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
        RSSMC1_HEADER_SIZE,
    )

    # Catalog #124 8-field archive-grammar declaration
    required_fields = {
        "archive_grammar",
        "parser_section_manifest",
        "inflate_runtime_loc_budget",
        "runtime_dep_closure",
        "export_format",
        "score_aware_loss",
        "bolt_on_loc_budget",
        "no_op_detector_planned",
    }
    assert required_fields.issubset(ARCHIVE_GRAMMAR_FIELDS.keys()), (
        f"missing fields: {required_fields - set(ARCHIVE_GRAMMAR_FIELDS.keys())}"
    )
    # Each field value MUST be non-empty + non-placeholder per Catalog #287
    for field, value in ARCHIVE_GRAMMAR_FIELDS.items():
        assert isinstance(value, str) and len(value) >= 4, (
            f"{field}: value {value!r} too short or non-string"
        )
        assert not value.lower().startswith("<"), (
            f"{field}: value {value!r} looks like placeholder"
        )

    # Canonical equation registration per Catalog #344
    assert (
        "categorical_posterior_capacity_vs_continuous_gaussian_v1"
        in CANONICAL_EQUATION_IDS
    )
    assert (
        "categorical_blahut_arimoto_rate_distortion_v1" in CANONICAL_EQUATION_IDS
    )

    # Canonical constants exposed
    assert DEFAULT_G == 24
    assert DEFAULT_K == 256
    assert EVAL_HW == (384, 512)
    assert NUM_PAIRS == 600
    assert RSSMC1_HEADER_SIZE == 27


def test_config_invariants() -> None:
    """Canonical config matches symposium + canonical equation registry."""
    from tac.substrates.dreamer_v3_rssm import DreamerV3RSSMConfig

    cfg = DreamerV3RSSMConfig()
    # H(T) = G * log2(K) = 24 * 8 = 192 bits/sample per canonical equation
    assert abs(cfg.categorical_bits_per_sample - 192.0) < 1e-6
    # Latent packing: K=256 fits in u8 → G=24 bytes/pair
    assert cfg.latent_packing_bytes_per_pair == 24

    # Hafner canonical config (G=32, K=32, H=160)
    cfg_hafner = DreamerV3RSSMConfig(num_groups=32, num_categories=32)
    assert abs(cfg_hafner.categorical_bits_per_sample - 160.0) < 1e-6
    assert cfg_hafner.latent_packing_bytes_per_pair == 32


def test_mlx_module_forward_training_shape() -> None:
    """MLX training forward produces correct shapes."""
    from tac.substrates.dreamer_v3_rssm import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
    )

    cfg = DreamerV3RSSMConfig(num_pairs=8)
    mod = DreamerV3RSSMSubstrateMLX(cfg)
    idx = mx.array([0, 1, 2, 3])
    rgb_pair, indices, soft = mod.forward_training(idx)
    assert tuple(int(d) for d in rgb_pair.shape) == (4, 2, 3, 384, 512)
    assert tuple(int(d) for d in indices.shape) == (4, cfg.num_groups)
    assert tuple(int(d) for d in soft.shape) == (4, cfg.num_groups, cfg.num_categories)


def test_mlx_module_forward_eval_shape() -> None:
    """MLX eval forward (from pre-computed indices) produces correct shapes."""
    from tac.substrates.dreamer_v3_rssm import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
    )

    cfg = DreamerV3RSSMConfig(num_pairs=4)
    mod = DreamerV3RSSMSubstrateMLX(cfg)
    # Synthesize indices (any K-valid values)
    indices = mx.zeros((2, cfg.num_groups), dtype=mx.int32)
    rgb_pair = mod.forward_eval_from_indices(indices)
    assert tuple(int(d) for d in rgb_pair.shape) == (2, 2, 3, 384, 512)


def test_archive_round_trip_byte_determinism() -> None:
    """pack_archive → parse_archive is byte-deterministic + structurally lossless."""
    from tac.substrates.dreamer_v3_rssm import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
        pack_archive,
        parse_archive,
    )

    cfg = DreamerV3RSSMConfig(num_pairs=2)
    mod = DreamerV3RSSMSubstrateMLX(cfg)
    flat = dict(tree_flatten(mod.parameters()))
    sd_numpy = {
        k: np.array(v).astype(np.float32)
        for k, v in flat.items()
        if not k.startswith("logits")  # archive stores argmax indices, not logits
    }
    sample_indices = np.random.randint(
        0, cfg.num_categories, size=(cfg.num_pairs, cfg.num_groups), dtype=np.int32
    )
    meta_in = {
        "gumbel_temperature": cfg.gumbel_temperature,
        "use_straight_through": cfg.use_straight_through,
        "lane_id": "lane_dreamer_v3_rssm_mlx_scaffold_20260526",
    }

    archive_bytes_1 = pack_archive(
        sd_numpy, sample_indices, meta_in,
        num_groups=cfg.num_groups,
        num_categories=cfg.num_categories,
        num_pairs=cfg.num_pairs,
        decoder_latent_dim=cfg.decoder_latent_dim,
        base_channels=cfg.base_channels,
    )
    archive_bytes_2 = pack_archive(
        sd_numpy, sample_indices, meta_in,
        num_groups=cfg.num_groups,
        num_categories=cfg.num_categories,
        num_pairs=cfg.num_pairs,
        decoder_latent_dim=cfg.decoder_latent_dim,
        base_channels=cfg.base_channels,
    )
    # Determinism: same input → same bytes
    assert archive_bytes_1 == archive_bytes_2, "pack_archive is non-deterministic"

    parsed = parse_archive(archive_bytes_1)
    assert parsed.num_groups == cfg.num_groups
    assert parsed.num_categories == cfg.num_categories
    assert parsed.num_pairs == cfg.num_pairs
    assert parsed.decoder_latent_dim == cfg.decoder_latent_dim
    assert parsed.base_channels == cfg.base_channels
    assert np.array_equal(parsed.category_indices, sample_indices), "indices drift"
    assert sorted(parsed.decoder_state_dict.keys()) == sorted(sd_numpy.keys()), (
        "state_dict keys drift"
    )
    assert parsed.meta == meta_in, "meta drift"


def test_archive_grammar_section_offsets() -> None:
    """parse_rssmc1_archive_bytes returns correct section offsets per Catalog #139."""
    from tac.substrates.dreamer_v3_rssm import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
        RSSMC1_HEADER_SIZE,
        RSSMC1_MAGIC,
        pack_archive,
        parse_rssmc1_archive_bytes,
    )

    cfg = DreamerV3RSSMConfig(num_pairs=2)
    mod = DreamerV3RSSMSubstrateMLX(cfg)
    flat = dict(tree_flatten(mod.parameters()))
    sd_numpy = {
        k: np.array(v).astype(np.float32)
        for k, v in flat.items()
        if not k.startswith("logits")
    }
    sample_indices = np.zeros(
        (cfg.num_pairs, cfg.num_groups), dtype=np.int32
    )
    archive_bytes = pack_archive(
        sd_numpy, sample_indices, {},
        num_groups=cfg.num_groups,
        num_categories=cfg.num_categories,
        num_pairs=cfg.num_pairs,
        decoder_latent_dim=cfg.decoder_latent_dim,
        base_channels=cfg.base_channels,
    )
    assert archive_bytes.startswith(RSSMC1_MAGIC), "bad magic"
    sections = parse_rssmc1_archive_bytes(archive_bytes)
    expected = {"rssmc1_header", "decoder_blob", "indices_blob", "meta_blob"}
    assert set(sections.keys()) == expected
    # Header is at byte 0, length 27 (RSSMC1_HEADER_SIZE)
    assert sections["rssmc1_header"] == (0, RSSMC1_HEADER_SIZE)
    # Sections cover the entire archive
    total = sum(length for _start, length in sections.values())
    assert total == len(archive_bytes), (
        f"section coverage {total} != archive length {len(archive_bytes)}"
    )


def test_mlx_pytorch_decoder_parity_at_archive_boundary() -> None:
    """Inflate-time decoder forward parity between MLX and PyTorch.

    Catalog #290 ADOPT-CANONICAL-BECAUSE-SERVES decision: the decoder topology
    is borrowed from the canonical PR95 HNeRV (empirically validated medal
    class). This test verifies the MLX module's _decoder_forward and the
    PyTorch DreamerV3RSSMDecoderTorch produce numerically close outputs given
    the same archive bytes — the same parity contract the canonical
    ``src/tac/local_acceleration/pr95_hnerv_mlx.py`` establishes for PR95.

    Target: ``max_abs < 1e-2`` (sub-uint8-rounding-step at the inflate boundary
    per the corrected #1258 empirical anchor 2026-05-26 |S_MLX-S_PT|=0.000011).
    """
    from tac.substrates.dreamer_v3_rssm import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
        pack_archive,
        parse_archive,
    )
    from tac.substrates.dreamer_v3_rssm.inflate import DreamerV3RSSMDecoderTorch
    import torch

    cfg = DreamerV3RSSMConfig(num_pairs=2)
    mlx_mod = DreamerV3RSSMSubstrateMLX(cfg)
    flat = dict(tree_flatten(mlx_mod.parameters()))
    sd_numpy = {
        k: np.array(v).astype(np.float32)
        for k, v in flat.items()
        if not k.startswith("logits")
    }

    # Use deterministic indices for both sides (skip Gumbel-Softmax stochasticity)
    indices_np = np.array(
        [[i % cfg.num_categories for i in range(cfg.num_groups)] for _ in range(2)],
        dtype=np.int32,
    )

    # MLX eval forward
    mlx_indices = mx.array(indices_np.astype(np.int32))
    mlx_out = mlx_mod.forward_eval_from_indices(mlx_indices)
    mlx_np = np.array(mlx_out)  # (B=2, 2, 3, 384, 512)

    # PyTorch decoder
    archive_bytes = pack_archive(
        sd_numpy, indices_np, {},
        num_groups=cfg.num_groups,
        num_categories=cfg.num_categories,
        num_pairs=cfg.num_pairs,
        decoder_latent_dim=cfg.decoder_latent_dim,
        base_channels=cfg.base_channels,
    )
    parsed = parse_archive(archive_bytes)
    torch_decoder = DreamerV3RSSMDecoderTorch(
        num_groups=parsed.num_groups,
        num_categories=parsed.num_categories,
        decoder_latent_dim=parsed.decoder_latent_dim,
        base_channels=parsed.base_channels,
    )
    torch_sd: dict[str, torch.Tensor] = {}
    for key, arr in parsed.decoder_state_dict.items():
        arr_fp32 = arr.astype(np.float32)
        if arr_fp32.ndim == 4:
            arr_fp32 = arr_fp32.transpose(0, 3, 1, 2).copy()
        torch_sd[key] = torch.from_numpy(arr_fp32)
    torch_decoder.load_state_dict(torch_sd, strict=True)
    torch_decoder.eval()
    with torch.inference_mode():
        torch_idx = torch.from_numpy(indices_np).long()
        torch_out = torch_decoder(torch_idx).numpy()  # (B, 2, 3, 384, 512)

    assert mlx_np.shape == torch_out.shape, (
        f"shape mismatch: MLX {mlx_np.shape} vs Torch {torch_out.shape}"
    )
    abs_diff = np.abs(mlx_np - torch_out)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    print(f"MLX↔PyTorch decoder parity: max_abs={max_abs:.4f}, mean_abs={mean_abs:.4f}")
    # FIX-WAVE-R1 A-OP3 threshold tightening (2026-05-26):
    #
    # Pre-FIX-WAVE-R1 baseline (R1 review measurement): max_abs ≈ 24.34 in
    # [0, 255] space (compounded drift from two independent bugs:
    # `_pixel_shuffle_2x_nhwc` channel-LAST convention + `_bilinear_resize_2x_nhwc`
    # mx.repeat instead of align_corners=False bilinear). Test ceiling was 50.0.
    #
    # Post-FIX-WAVE-R1 empirical (this run on 2026-05-26):
    # max_abs=0.0054, mean_abs=0.0007 — ~4500x improvement. Both bugs fixed:
    #   - A-OP1: `_pixel_shuffle_2x_nhwc` now uses canonical channel-FIRST
    #     reshape `(B, H, W, out_C, 2, 2)` + transpose `(0, 1, 4, 2, 5, 3)`
    #     matching sister D=Z6 + canonical PR95 MLX helper.
    #   - A-OP2: `_bilinear_resize_2x_nhwc` now delegates to canonical
    #     `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`
    #     which is empirically PyTorch-byte-stable.
    #
    # Threshold tightened from < 50.0 to < 0.05 (~10x headroom above the
    # post-fix empirical max_abs ≈ 0.0054; well below the R1 review's stated
    # L0→L1 promotion criterion of < 5.0 and approaching the best-case < 1.0).
    # The remaining sub-0.01 drift is fp32 compound-op precision noise across
    # 6 PixelShuffle blocks + sin/sigmoid nonlinearities + final RGB heads;
    # acceptable per CLAUDE.md "Apples-to-apples evidence discipline" because
    # after camera-resolution uint8 quantization the drift is structurally
    # below the per-pixel quantization step (1.0 / 255 ≈ 0.004 in [0, 1] space
    # or equivalently 1.0 in [0, 255] space).
    #
    # Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L9
    # runtime closure: MLX-trained-PyTorch-inflated model now IS the same
    # runtime the MLX trainer observes at convergence.
    assert max_abs < 0.05, (
        f"MLX↔PyTorch decoder parity drift too large: max_abs={max_abs}; "
        "post-FIX-WAVE-R1 expectation is <0.05 in [0,255] space (~1500x "
        "below pre-fix 24.34 ceiling; ~10x headroom above empirical 0.0054)"
    )


def test_end_to_end_mlx_train_archive_pytorch_inflate() -> None:
    """End-to-end: MLX module → archive → PyTorch inflate → camera-resolution uint8.

    Validates the full Path 3 cascade at L0 scaffold level: every byte produced
    by the archive is consumed by the inflate runtime to produce contest-shaped
    raw output (1200 frames × 874 × 1164 × 3 = 3,662,409,600 bytes for full
    600-pair archive; we test 2-pair = 4 frames × 874 × 1164 × 3 = 12,208,032).
    """
    from tac.substrates.dreamer_v3_rssm import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
        pack_archive,
    )
    from tac.substrates.dreamer_v3_rssm.inflate import (
        CAMERA_H,
        CAMERA_W,
        inflate_one_video,
    )

    cfg = DreamerV3RSSMConfig(num_pairs=2)
    mlx_mod = DreamerV3RSSMSubstrateMLX(cfg)
    flat = dict(tree_flatten(mlx_mod.parameters()))
    sd_numpy = {
        k: np.array(v).astype(np.float32)
        for k, v in flat.items()
        if not k.startswith("logits")
    }
    idx = mx.array([0, 1])
    _rgb, indices_mx, _soft = mlx_mod.forward_training(idx)
    sample_indices = np.array(indices_mx).astype(np.int32)

    archive_bytes = pack_archive(
        sd_numpy, sample_indices,
        meta={
            "gumbel_temperature": cfg.gumbel_temperature,
            "use_straight_through": cfg.use_straight_through,
        },
        num_groups=cfg.num_groups,
        num_categories=cfg.num_categories,
        num_pairs=cfg.num_pairs,
        decoder_latent_dim=cfg.decoder_latent_dim,
        base_channels=cfg.base_channels,
    )

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "0.raw"
        n_frames = inflate_one_video(archive_bytes, out_path, device="cpu")
        bytes_written = out_path.stat().st_size

    expected_n_frames = cfg.num_pairs * 2
    expected_bytes = expected_n_frames * CAMERA_H * CAMERA_W * 3
    assert n_frames == expected_n_frames, (
        f"frame count drift: got {n_frames}, expected {expected_n_frames}"
    )
    assert bytes_written == expected_bytes, (
        f"output bytes drift: got {bytes_written}, expected {expected_bytes}"
    )


def test_gumbel_softmax_sample_shapes_and_indices_in_range() -> None:
    """Gumbel-Softmax STE sample produces correct shapes + valid indices."""
    from tac.substrates.dreamer_v3_rssm import gumbel_softmax_sample

    logits = mx.random.normal(shape=(3, 24, 256))
    soft, indices = gumbel_softmax_sample(
        logits, temperature=1.0, use_straight_through=True
    )
    assert tuple(int(d) for d in soft.shape) == (3, 24, 256)
    assert tuple(int(d) for d in indices.shape) == (3, 24)
    indices_np = np.array(indices)
    assert indices_np.min() >= 0
    assert indices_np.max() < 256
    # STE: forward should be one-hot (each (B, G) row has exactly one 1)
    soft_np = np.array(soft)
    # The one-hot rows are NOT exactly bit-equal because of the STE
    # gradient-flow trick (forward = hard + soft - stop_gradient(soft)
    # has forward value = hard, BUT due to MLX dtype intermediate calc
    # may have small numerical perturbation). Test the argmax aligns.
    argmax_from_soft = np.argmax(soft_np, axis=-1)
    assert np.array_equal(argmax_from_soft, indices_np), (
        "STE soft output argmax should match returned indices"
    )


def test_architecture_manifest_includes_canonical_equation_refs() -> None:
    """architecture_manifest() carries canonical equation refs (Catalog #305 + #344)."""
    from tac.substrates.dreamer_v3_rssm import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
    )

    cfg = DreamerV3RSSMConfig(num_pairs=4)
    mod = DreamerV3RSSMSubstrateMLX(cfg)
    manifest = mod.architecture_manifest()
    assert manifest["schema"] == "dreamer_v3_rssm_mlx_architecture_v1"
    assert manifest["num_groups_G"] == cfg.num_groups
    assert manifest["num_categories_K"] == cfg.num_categories
    assert abs(manifest["categorical_bits_per_sample"] - 192.0) < 1e-6
    refs = set(manifest["canonical_equation_refs"])
    assert "categorical_posterior_capacity_vs_continuous_gaussian_v1" in refs
    assert "categorical_blahut_arimoto_rate_distortion_v1" in refs
    # Canonical non-promotable markers per CLAUDE.md "MLX portable-local-
    # substrate authority"
    assert manifest["axis_tag"] == "[macOS-MLX research-signal]"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_rssmc_decoder_param_count_excluding_per_pair_logits() -> None:
    """Decoder param count estimate reasonable (excluding per-pair logits)."""
    from tac.substrates.dreamer_v3_rssm import (
        DreamerV3RSSMConfig,
        rssmc_decoder_param_count,
    )

    cfg = DreamerV3RSSMConfig(num_pairs=600)
    total = rssmc_decoder_param_count(cfg)
    per_pair_logits = cfg.num_pairs * cfg.num_groups * cfg.num_categories
    decoder_only = total - per_pair_logits
    # Decoder + cat_proj + stem + 6 blocks + refine + heads at G=24, K=256, C=24
    # is approximately ~285K params (empirically measured at smoke time).
    # cat_to_continuous dominates: G*K*L = 6144 * 28 = 172K params.
    assert 100_000 < decoder_only < 500_000, (
        f"decoder param count {decoder_only:,} outside reasonable range"
    )
    # Per-pair logits at 600 × 24 × 256 = 3.69M floats (training-only; archive
    # stores argmax indices)
    assert per_pair_logits == 600 * 24 * 256
