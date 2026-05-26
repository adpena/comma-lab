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

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised on Apple Silicon with MLX installed.
    import mlx.core as mx
    import mlx.nn as nn
except Exception as exc:  # pragma: no cover - import guard for non-Apple CI.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR: Exception | None = exc
else:
    _MLX_IMPORT_ERROR = None

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
    if mx is None or nn is None:
        raise RuntimeError(
            "MLX is not installed. Install via `uv pip install mlx` "
            "(macOS only). For non-Apple-Silicon iteration, route through "
            "`tac.substrates.nirvana_cascading_nerv.numpy_reference` per axis 3 "
            "portability discipline."
            f" Original import error: {_MLX_IMPORT_ERROR!r}"
        )
    return mx


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


# ---------------------------------------------------------------------------
# L1 RENDERER CLASS LANDED 2026-05-26 per TaskCreate #1338 NIRVANA L1
# EMPIRICAL respawn (predecessor `nirvana-pr110-l1-empirical-mlx-20260526`
# blocker resolution; sister BoostNeRV-PR110 L1 EMPIRICAL pattern at
# `.omx/research/boostnerv_pr110_l1_empirical_landed_20260526.md` is the
# canonical cross-pollination template). Appended per Catalog #110/#113
# APPEND-ONLY HISTORICAL_PROVENANCE: every line above this comment is
# unchanged from the L0 SCAFFOLD posture.
#
# Per the new 2026-05-26 standing directives:
#   - MLX↔CUDA bidirectional drift anticipation: 5 mitigations declared
#     inline (NHWC layout / align_corners=False / fp32 accumulation /
#     deterministic seeding / state_dict transpose at export bridge).
#   - Pushing-the-frontier of optimization algorithms: this is canon-
#     application of the hierarchical-residual-decoder-cascade paradigm
#     (sister BoostNeRV-PR110 ResidualHeadMLX) NOT a frontier-push; the
#     novel surface IS the per-level int8 residual + brotli compose vs
#     BoostNeRV's BPR1-style fp16 sidecar.
# ---------------------------------------------------------------------------


class NirvanaCascadingNervRendererMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX hierarchical residual decoder cascade per L0 scaffold contract.

    L1 EMPIRICAL CLASS landed 2026-05-26 per TaskCreate #1338. Mirrors the
    PyTorch inflate-time topology at
    ``tac.substrates.nirvana_cascading_nerv.inflate.NirvanaCascadingDecoderTorch``
    so the canonical state_dict export bridge (NHWC→NCHW transpose on 4-D
    weight tensors per sister substrate dreamer_v3_rssm pattern) produces
    byte-identical inflate-time decoder reconstruction.

    **DRIFT SURFACE per axis 2 MLX drift minimization standing directive 2026-05-26**:
    - **Mitigation 1 (NHWC layout)**: all conv weights kept in MLX-canonical
      NHWC layout (out_ch, kH, kW, in_ch); export bridge transposes to NCHW
      via `arr.transpose(0, 3, 1, 2)` in inflate.py. The stem vector is first
      reshaped as NCHW and then transposed to NHWC so MLX and PyTorch consume
      identical linear-output ordering.
    - **Mitigation 2 (bilinear align_corners=False)**: uses canonical
      native MLX `bilinear_resize2x_align_corners_false_nhwc`; AVOIDS
      `mx.repeat` substitution and avoids a NumPy round-trip in the hot path.
    - **Mitigation 3 (fp32 accumulation)**: all cascade additions in fp32 per
      `numpy_reference.cascade_reconstruct` line 84 contract; only state_dict
      storage uses fp16 (per archive.py contract; not accumulation).
    - **Mitigation 4 (deterministic seeding)**: callers MUST pass `seed=`
      kwarg to constructor; uses `mx.random.seed(seed)` BEFORE parameter
      allocation for byte-stable parity smoke vs PyTorch with matched seed.
    - **Mitigation 5 (clamp [0, 1] per level)**: matches PyTorch inflate
      line 150 `torch.clamp(rgb, 0.0, 1.0)` per-level after residual add,
      preventing post-sigmoid drift accumulation across the cascade.

    **CANONICAL-VS-FRONTIER-PUSH decision** per new optimization-algorithm-
    research standing directive 2026-05-26: this implementation is CANON-
    APPLICATION of the NeRV-family hierarchical-residual-decoder-cascade
    paradigm (sister BoostNeRV-PR110 ResidualHeadMLX is the closest cross-
    pollination peer; canonical PR95-HNeRV-MLX ships the upsample-block
    primitive pattern this class mirrors). NO novel optimizer / loss /
    architecture term proposed at L1; novelty is RESERVED for L2+ when
    empirical L1 anchors a per-pair-difficulty signal worth a Catalog #344
    canonical equation registration.
    """

    def __init__(
        self,
        cfg: NirvanaCascadingNervConfig,
        *,
        seed: int = 0,
    ) -> None:
        mx_mod = _ensure_mlx_available()
        super().__init__()

        mx_mod.random.seed(int(seed))  # Mitigation 4: deterministic seeding

        self.cfg = cfg
        self.seed = int(seed)
        self._mx = mx_mod

        # Level 0 decoder topology (mirrors inflate.py::_NirvanaLevelDecoder)
        # NHWC conv layout per Mitigation 1.
        self.stem = nn.Linear(  # type: ignore[union-attr]
            cfg.per_pair_latent_dim,
            cfg.base_channels * cfg.base_h * cfg.base_w,
        )
        # MLX Conv2d weights are NHWC (out_ch, kH, kW, in_ch)
        self.conv1 = nn.Conv2d(  # type: ignore[union-attr]
            cfg.base_channels,
            cfg.base_channels,
            3,
            padding=1,
        )
        self.conv_to_rgb = nn.Conv2d(  # type: ignore[union-attr]
            cfg.base_channels,
            3,
            3,
            padding=1,
        )

    def decode_level_0(self, z: Any) -> Any:
        """Level 0 forward: (B, latent_dim) → (B, H0, W0, 3) NHWC RGB in [0, 1].

        Mirrors PyTorch `_NirvanaLevelDecoder.forward`:
            linear(z) → reshape(B, base_channels, base_h, base_w) NCHW
            → transpose to NHWC → sin → conv1 → sin → conv_to_rgb
            → sigmoid → [0, 1]
        """
        mx = self._mx
        cfg = self.cfg
        B = int(z.shape[0])
        x = self.stem(z)  # (B, base_channels * base_h * base_w)
        # PyTorch inflate views the same vector as NCHW. Preserve that memory
        # order, then transpose to MLX's NHWC convolution layout.
        x = mx.reshape(x, (B, cfg.base_channels, cfg.base_h, cfg.base_w))
        x = mx.transpose(x, (0, 2, 3, 1))
        x = mx.sin(x)
        x = self.conv1(x)
        x = mx.sin(x)
        rgb = mx.sigmoid(self.conv_to_rgb(x))  # NHWC; [0, 1]
        return rgb

    def cascade_reconstruct(
        self,
        latents: Any,
        per_level_residuals_fp_nhwc: list[Any],
    ) -> Any:
        """Hierarchical residual cascade per `numpy_reference.cascade_reconstruct`.

        Args:
            latents: (B, latent_dim) MLX array, per-pair latents.
            per_level_residuals_fp_nhwc: list of (H_i, W_i, 3) NHWC residuals
                already dequantized to fp32; index `level` is applied at
                output resolution of level `level` (level 0 unused per
                cascade design; levels 1..num_levels-1 carry residuals).

        Returns:
            (B, H_final, W_final, 3) NHWC RGB in [0, 1].
        """
        mx = self._mx
        from tac.local_acceleration.pr95_hnerv_mlx import (
            bilinear_resize2x_align_corners_false_nhwc,
        )

        cfg = self.cfg
        # Level 0: full per-pair decoder forward
        rgb = self.decode_level_0(latents)  # (B, H0, W0, 3) NHWC

        # Cascade levels 1..N-1: upsample → add residual → clamp
        for level in range(1, cfg.num_levels):
            # Mitigation 2: native MLX bilinear upsample (NHWC, align_corners=False).
            rgb = bilinear_resize2x_align_corners_false_nhwc(rgb)
            # Residual broadcast (1, H, W, 3) → add to (B, H, W, 3)
            residual = per_level_residuals_fp_nhwc[level]
            # Residual stored as (H, W, 3); broadcast across batch dim via reshape
            res_3d = mx.reshape(residual, (1, residual.shape[0], residual.shape[1], 3))
            rgb = rgb + res_3d
            # Mitigation 5: clamp [0, 1] per level
            rgb = mx.clip(rgb, 0.0, 1.0)
        return rgb

    def export_parameter_arrays(self) -> dict[str, Any]:
        """Return the flat parameter map used by the PyTorch inflate bridge."""
        return {
            "stem.weight": self.stem.weight,
            "stem.bias": self.stem.bias,
            "conv1.weight": self.conv1.weight,
            "conv1.bias": self.conv1.bias,
            "conv_to_rgb.weight": self.conv_to_rgb.weight,
            "conv_to_rgb.bias": self.conv_to_rgb.bias,
        }

    def state_dict_for_inflate_export(self) -> dict[str, Any]:
        """Export state_dict in PyTorch NCHW layout for inflate.py compatibility.

        Per Mitigation 1: MLX conv weights are NHWC (out_ch, kH, kW, in_ch);
        PyTorch expects NCHW (out_ch, in_ch, kH, kW). This export uses
        an explicit transpose in the inflate loader for 4-D weight tensors;
        matches `inflate.py::inflate_one_video` line 203 logic.
        """
        sd: dict[str, Any] = {}
        for key, val in self.export_parameter_arrays().items():
            arr = np.asarray(val).astype(np.float32)
            # MLX Linear weight is (out, in); PyTorch Linear weight is (out, in) too;
            # no transpose needed for 2-D. 4-D conv weight needs NHWC→NCHW.
            # Save NHWC layout for the inflate.py NCHW-transpose contract.
            sd[f"level_0_decoder.{key}"] = arr
        return sd

    def architecture_manifest(self) -> dict[str, Any]:
        """Per Catalog #305 observability surface declaration."""
        return {
            "schema": "nirvana_cascading_nerv_mlx_architecture_v1",
            "num_levels": self.cfg.num_levels,
            "per_pair_latent_dim": self.cfg.per_pair_latent_dim,
            "base_h": self.cfg.base_h,
            "base_w": self.cfg.base_w,
            "base_channels": self.cfg.base_channels,
            "num_pairs": self.cfg.num_pairs,
            "residual_quant_bits": self.cfg.residual_quant_bits,
            "internal_layout": "NHWC",
            "decoder_param_count": renderer_param_count(self.cfg),
            "estimate_archive_bytes": estimate_archive_bytes(self.cfg),
            "drift_mitigations": [
                "nhwc_layout_export_transpose_to_nchw",
                "native_mlx_bilinear_upsample_align_corners_false",
                "fp32_accumulation_cascade_additions",
                "deterministic_mx_random_seed_at_init",
                "clamp_0_1_per_cascade_level",
            ],
        }
__all__ = [
    "EVAL_HW",
    "NirvanaCascadingNervConfig",
    "NirvanaCascadingNervRendererMLX",
    "_ensure_mlx_available",
    "_full_main",
    "estimate_archive_bytes",
    "renderer_param_count",
]
