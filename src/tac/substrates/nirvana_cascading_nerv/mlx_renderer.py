# SPDX-License-Identifier: MIT
"""nirvana_cascading_nerv.mlx_renderer — MLX hierarchical residual decoder cascade SCAFFOLD (config + helpers; actual renderer class lands Phase 2).

FIX-WAVE-R1' G-OP3 (2026-05-26): module docstring corrected to reflect the
actual L0 SCAFFOLD posture. Per R1' Path 3 G review: this module contains
ONLY the Config dataclass + factory helpers + estimators; the actual MLX
renderer CLASS (the hierarchical residual decoder cascade implementation)
is deferred to Phase 2 per Catalog #325 per-substrate symposium. ZERO
MLX primitives shipped at L0; the design memo's 7 anticipated L1+ MLX
primitives + 3 KNOWN-DRIFT-RISK characterizations are L1+ implementation
guidance, not L0 empirical claims.

L0 SCAFFOLD module: declares the MLX renderer config + factory; defers
heavy MLX-dependent class construction until called. MLX is OPTIONAL at
top-level import time per Catalog #1 device-selection-defaults discipline
+ axis 3 portability.

For the numpy-only / non-Apple-Silicon test path, route through
``tac.substrates.nirvana_cascading_nerv.numpy_reference`` instead.

Per axis 2 MLX drift minimization (operator directive #3 2026-05-26):
- All conv weights kept in MLX-canonical NHWC layout (out_ch, kH, kW, in_ch);
  PyTorch export bridge transposes to NCHW per sister substrate dreamer_v3_rssm pattern
- Bilinear upsample uses canonical `align_corners=False` semantics; AVOID
  mx.repeat substitution which caused sister A=DreamerV3 max_abs=24.34 gap
- Mean reduction uses standard MLX mean for L0; Kahan summation queued for
  L1+ when batch sizes exceed ~1e6 elements

Per Catalog #146 + #205 inflate runtime contract:
- Substrate-engineering LOC budget ≤200 for inflate runtime
- Device selection via canonical `tac.substrates._shared.inflate_runtime.select_inflate_device`
- No scorer load at inflate time
- No /tmp paths in persisted artifacts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Module-level config. MLX is imported lazily inside the factory to keep
# top-level import cheap (sister substrates' canonical pattern).
EVAL_HW: tuple[int, int] = (384, 512)
"""Contest scorer-resolution; final decoder output shape (height, width)."""


@dataclass(frozen=True)
class NirvanaCascadingNervConfig:
    """Substrate configuration; immutable per Catalog #229 PV discipline."""

    num_levels: int = 4
    """Wavelet-pyramid depth (4 = 48×64 → 96×128 → 192×256 → 384×512)."""

    per_pair_latent_dim: int = 16
    """Per-pair latent z dimension (NeRV-family canonical)."""

    base_h: int = 48
    """Level 0 decoder output height (base coarse-level)."""

    base_w: int = 64
    """Level 0 decoder output width (base coarse-level)."""

    base_channels: int = 24
    """Decoder per-level channel count (CARGO-CULTED at L0 per Catalog #303)."""

    num_pairs: int = 600
    """Contest video pair count."""

    residual_quant_bits: int = 8
    """Per-level residual quantization bits (int8 canonical at L0)."""

    decoder_latent_dim: int = 28
    """Intermediate decoder latent dim (NeRV-family canonical sister pattern)."""

    # Per Catalog #229: validate at construction
    def __post_init__(self) -> None:
        if self.num_levels < 1 or self.num_levels > 6:
            raise ValueError(
                f"num_levels={self.num_levels} out of [1, 6] range"
            )
        if self.per_pair_latent_dim < 1:
            raise ValueError(f"per_pair_latent_dim={self.per_pair_latent_dim} must be >= 1")
        if self.base_h < 1 or self.base_w < 1:
            raise ValueError(
                f"base_h={self.base_h} base_w={self.base_w} must be >= 1"
            )
        # Verify the cascade target matches EVAL_HW
        final_h = self.base_h * (2 ** (self.num_levels - 1))
        final_w = self.base_w * (2 ** (self.num_levels - 1))
        if (final_h, final_w) != EVAL_HW:
            raise ValueError(
                f"cascade target ({final_h}, {final_w}) != EVAL_HW {EVAL_HW}; "
                f"adjust base_h/base_w/num_levels"
            )
        if self.residual_quant_bits not in {4, 8, 16}:
            raise ValueError(
                f"residual_quant_bits={self.residual_quant_bits} must be in {{4, 8, 16}}"
            )

    def per_level_shape(self, level: int) -> tuple[int, int]:
        """Return (H, W) for the given decoder level (0-indexed)."""
        if level < 0 or level >= self.num_levels:
            raise ValueError(
                f"level={level} out of [0, {self.num_levels}) range"
            )
        h = self.base_h * (2 ** level)
        w = self.base_w * (2 ** level)
        return (h, w)


def _ensure_mlx_available() -> Any:
    """Lazy import MLX. Raises with actionable message if not installed."""
    try:
        import mlx.core as mx  # noqa: F401

        return mx
    except ImportError as exc:
        raise RuntimeError(
            "MLX is not installed. Install via `uv pip install mlx` "
            "(macOS only). For non-Apple-Silicon iteration, route through "
            "`tac.substrates.nirvana_cascading_nerv.numpy_reference` per axis 3 "
            "portability discipline."
        ) from exc


def renderer_param_count(cfg: NirvanaCascadingNervConfig) -> int:
    """Estimate MLX renderer parameter count (level 0 decoder only).

    Used for cost-band / archive-size prediction WITHOUT instantiating MLX.

    Per inflate.py topology: ONLY level 0 carries a full decoder. Higher
    levels apply per-level int8 residual additions from the archive
    (residuals are STORED bytes, not learned per-pair).

    Level 0 decoder block (canonical NeRV-family pattern):
    - 1 linear (latent → base_channels * base_h * base_w): per_pair_latent_dim
        * base_channels * base_h * base_w
    - 1 conv2d (3x3, base_channels → base_channels): base_channels * 3 * 3 * base_channels
    - 1 conv2d (3x3, base_channels → 3): 3 * 3 * 3 * base_channels
    """
    return (
        cfg.per_pair_latent_dim * cfg.base_channels * cfg.base_h * cfg.base_w  # stem linear
        + cfg.base_channels * 3 * 3 * cfg.base_channels  # conv1
        + 3 * 3 * 3 * cfg.base_channels  # conv_to_rgb
    )


def estimate_archive_bytes(cfg: NirvanaCascadingNervConfig) -> int:
    """Estimate NIRVANA1 archive size in bytes for a given config.

    Used for Dykstra-feasibility check at design time (Catalog #296) without
    instantiating MLX or actually packing.

    Breakdown per design memo §"Predicted ΔS band":
    - Decoder state_dict at fp16, brotli q=9: ~30% of raw fp32 size
    - Per-level residuals at int8: H_i * W_i * 3 channels per level
    - Per-pair latents at int16: num_pairs * per_pair_latent_dim * 2 bytes
    - Header + meta JSON: ~256 bytes
    """
    decoder_fp16_bytes = renderer_param_count(cfg) * 2  # fp16
    decoder_compressed = int(decoder_fp16_bytes * 0.3)  # brotli compression estimate

    residual_bytes = 0
    for level in range(cfg.num_levels):
        h, w = cfg.per_level_shape(level)
        residual_bytes += h * w * 3 * (cfg.residual_quant_bits // 8)
    # Brotli compresses int8 residuals well (~40%)
    residual_compressed = int(residual_bytes * 0.4)

    latents_bytes = cfg.num_pairs * cfg.per_pair_latent_dim * 2  # int16
    latents_compressed = int(latents_bytes * 0.6)  # brotli compression estimate

    header_meta = 256

    return decoder_compressed + residual_compressed + latents_compressed + header_meta


def _full_main(argv: list[str] | None = None) -> int:
    """L0 SCAFFOLD posture per Catalog #240: full main raises NotImplementedError.

    The L0 scaffold ships design + MLX renderer + numpy reference + PyTorch
    inflate + archive grammar + tests + smoke trainer stub. The full MLX
    training path is gated by Phase 2 council symposium per Catalog #325 +
    operator-authorized paid-CUDA dispatch eligibility per Catalog #1265
    MLX-first contest-equivalence gate.
    """
    raise NotImplementedError(
        "nirvana_cascading_nerv full main NOT YET IMPLEMENTED — L0 SCAFFOLD "
        "posture per Catalog #240. Phase 2 council symposium per Catalog "
        "#325 + Catalog #1265 MLX↔PyTorch parity gate REQUIRED before any "
        "paid-CUDA dispatch authorization. See "
        ".omx/research/path_3_g_nirvana_cascading_nerv_substrate_design_20260526.md "
        "for the Phase 2+ roadmap."
    )


__all__ = [
    "EVAL_HW",
    "NirvanaCascadingNervConfig",
    "_ensure_mlx_available",
    "_full_main",
    "estimate_archive_bytes",
    "renderer_param_count",
]
