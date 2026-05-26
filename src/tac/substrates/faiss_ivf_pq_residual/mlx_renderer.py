# SPDX-License-Identifier: MIT
"""faiss_ivf_pq_residual.mlx_renderer — MLX-native PQ codebook gather + tile reassemble.

L0 SCAFFOLD module: declares the substrate config + factory + parameter
count estimator + archive bytes estimator. MLX is OPTIONAL at top-level
import time per Catalog #1 device-selection-defaults discipline + axis 3
portability per operator directive #3 2026-05-26.

For the numpy-only / non-Apple-Silicon test path, route through
``tac.substrates.faiss_ivf_pq_residual.numpy_reference`` instead.

Per axis 2 MLX drift minimization (operator directive #3 2026-05-26):
- PQ codebook gather uses MLX `mx.take` / `mx.take_along_axis`; float copy is
  deterministic; expected drift ≤ epsilon-machine vs numpy reference
- Per-tile reassemble uses MLX `mx.reshape` / `mx.concatenate`; deterministic
- Bilinear residual upsample (if used) MUST route through canonical
  `bilinear_resize2x_align_corners_false_nhwc` per sister A=DreamerV3
  forensic anchor (`align_corners=True` anti-pattern caused max_abs=24.34)
- uint8 cast routes canonical Catalog #205 sister rounding

Per Catalog #146 + #205 inflate runtime contract:
- Substrate-engineering LOC budget ≤200 for inflate runtime
- Device selection via canonical `tac.substrates._shared.inflate_runtime.select_inflate_device`
- No scorer load at inflate time
- No /tmp paths in persisted artifacts

Per Catalog #240 L0 SCAFFOLD posture: ``_full_main`` raises NotImplementedError
until PHASE 2 Catalog #325 per-substrate symposium clears for paid dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Module-level config. MLX + Faiss imports are lazy inside factories to keep
# top-level import cheap (sister substrates' canonical pattern per Catalog #1).
EVAL_HW: tuple[int, int] = (384, 512)
"""Contest scorer-resolution; final per-pair RGB residual shape (height, width)."""


@dataclass(frozen=True)
class FaissIVFPQResidualConfig:
    """Substrate configuration; immutable per Catalog #229 PV discipline."""

    m_sub_quantizers: int = 4
    """PQ M parameter — number of sub-quantizers per tile vector.

    Each sub-quantizer carries log2(ksub) bits per tile per sub. Total
    bits-per-tile = m * log2(ksub). M=4 ksub=256 = 32 bits/tile = 4 bytes/tile.
    """

    ksub_codebook_size: int = 256
    """PQ ksub parameter — codebook size per sub-quantizer.

    Per Jégou-Douze-Schmid 2011 canonical: ksub=256 is the SIFT-1M anchor;
    larger ksub = richer codebook but larger codebook bytes (M*ksub*dim*4 raw).
    """

    tile_h: int = 96
    """Spatial tile height. EVAL_HW[0] must be divisible by tile_h."""

    tile_w: int = 128
    """Spatial tile width. EVAL_HW[1] must be divisible by tile_w."""

    num_pairs: int = 600
    """Contest video pair count."""

    residual_quant_bits: int = 16
    """Per-tile residual quantization bits BEFORE PQ encoding (int16 canonical).

    Per-tile vector ∈ [-1, 1] residual range; int16 maps to 32767 levels.
    PQ codebook is trained on int16-quantized residuals.
    """

    residual_scale: float = 0.5
    """Per-pair residual scale factor; residual ∈ [-residual_scale, residual_scale]."""

    def __post_init__(self) -> None:
        if self.m_sub_quantizers < 1 or self.m_sub_quantizers > 16:
            raise ValueError(
                f"m_sub_quantizers={self.m_sub_quantizers} out of [1, 16] range"
            )
        if self.ksub_codebook_size < 2 or self.ksub_codebook_size > 65536:
            raise ValueError(
                f"ksub_codebook_size={self.ksub_codebook_size} out of [2, 65536] range"
            )
        if self.tile_h < 1 or self.tile_w < 1:
            raise ValueError(
                f"tile_h={self.tile_h} tile_w={self.tile_w} must be >= 1"
            )
        if EVAL_HW[0] % self.tile_h != 0:
            raise ValueError(
                f"EVAL_HW[0]={EVAL_HW[0]} must be divisible by tile_h={self.tile_h}"
            )
        if EVAL_HW[1] % self.tile_w != 0:
            raise ValueError(
                f"EVAL_HW[1]={EVAL_HW[1]} must be divisible by tile_w={self.tile_w}"
            )
        tile_dim = self.tile_h * self.tile_w * 3
        if tile_dim % self.m_sub_quantizers != 0:
            raise ValueError(
                f"tile_dim={tile_dim} must be divisible by "
                f"m_sub_quantizers={self.m_sub_quantizers}"
            )
        if self.residual_quant_bits not in {8, 16}:
            raise ValueError(
                f"residual_quant_bits={self.residual_quant_bits} must be in {{8, 16}}"
            )
        if self.residual_scale <= 0.0 or self.residual_scale > 1.0:
            raise ValueError(
                f"residual_scale={self.residual_scale} must be in (0, 1]"
            )

    @property
    def tiles_per_pair(self) -> int:
        """Number of tiles per per-pair RGB residual frame."""
        return (EVAL_HW[0] // self.tile_h) * (EVAL_HW[1] // self.tile_w)

    @property
    def tile_dim(self) -> int:
        """Per-tile vector dimension = tile_h × tile_w × 3."""
        return self.tile_h * self.tile_w * 3

    @property
    def sub_dim(self) -> int:
        """Per-sub-quantizer dimension = tile_dim // m_sub_quantizers."""
        return self.tile_dim // self.m_sub_quantizers

    @property
    def bits_per_tile(self) -> int:
        """Total bits per tile after PQ encoding."""
        # log2(ksub) per sub-quantizer × M sub-quantizers
        import math

        return int(self.m_sub_quantizers * math.ceil(math.log2(self.ksub_codebook_size)))

    @property
    def bytes_per_pair_codeword_stream_raw(self) -> int:
        """Per-pair codeword stream raw bytes (uncompressed)."""
        return (self.tiles_per_pair * self.bits_per_tile + 7) // 8


def _ensure_mlx_available() -> Any:
    """Lazy import MLX. Raises with actionable message if not installed."""
    try:
        import mlx.core as mx  # noqa: F401

        return mx
    except ImportError as exc:
        raise RuntimeError(
            "MLX is not installed. Install via `uv pip install mlx` "
            "(macOS only). For non-Apple-Silicon iteration, route through "
            "`tac.substrates.faiss_ivf_pq_residual.numpy_reference` per axis 3 "
            "portability discipline."
        ) from exc


def estimate_per_pair_codeword_bytes_raw(cfg: FaissIVFPQResidualConfig) -> int:
    """Estimate per-pair codeword stream raw bytes (uncompressed) for a config.

    Used for Dykstra-feasibility check at design time (Catalog #296).
    """
    return cfg.bytes_per_pair_codeword_stream_raw


def estimate_archive_bytes(cfg: FaissIVFPQResidualConfig) -> int:
    """Estimate FAISSPQ1 archive size in bytes for a given config.

    Used for Dykstra-feasibility check at design time (Catalog #296) without
    instantiating MLX or actually packing.

    Breakdown per design memo §"Predicted ΔS band":
    - PQ codebook raw: M × ksub × sub_dim × 4 (float32)
    - PQ codebook brotli-compressed: ~30% of raw (high redundancy)
    - Per-pair codeword stream raw: num_pairs × bytes_per_pair_codeword_stream_raw
    - Per-pair codeword stream brotli-compressed: ~60% of raw (medium entropy)
    - Header + meta JSON: ~256 bytes
    """
    codebook_raw = (
        cfg.m_sub_quantizers * cfg.ksub_codebook_size * cfg.sub_dim * 4
    )  # float32
    codebook_compressed = int(codebook_raw * 0.3)

    codewords_raw = cfg.num_pairs * cfg.bytes_per_pair_codeword_stream_raw
    codewords_compressed = int(codewords_raw * 0.6)

    header_meta = 256

    return codebook_compressed + codewords_compressed + header_meta


def _full_main(argv: list[str] | None = None) -> int:
    """L0 SCAFFOLD posture per Catalog #240: full main raises NotImplementedError.

    The L0 scaffold ships design + MLX-native primitives + numpy reference +
    PyTorch inflate + archive grammar + tests + smoke trainer stub. The full
    MLX training path is gated by Phase 2 council symposium per Catalog
    #325 + operator-authorized paid-CUDA dispatch eligibility per Catalog
    #1265 MLX-first contest-equivalence gate.
    """
    raise NotImplementedError(
        "faiss_ivf_pq_residual full main NOT YET IMPLEMENTED — L0 SCAFFOLD "
        "posture per Catalog #240. Phase 2 council symposium per Catalog "
        "#325 + Catalog #1265 MLX↔PyTorch parity gate REQUIRED before any "
        "paid-CUDA dispatch authorization. See "
        ".omx/research/path_3_i_v1_faiss_ivf_pq_substrate_design_decision_20260526.md "
        "for the Phase 2+ roadmap."
    )


__all__ = [
    "EVAL_HW",
    "FaissIVFPQResidualConfig",
    "_ensure_mlx_available",
    "_full_main",
    "estimate_archive_bytes",
    "estimate_per_pair_codeword_bytes_raw",
]
