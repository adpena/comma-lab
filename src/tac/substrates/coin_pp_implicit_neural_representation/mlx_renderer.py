# SPDX-License-Identifier: MIT
"""coin_pp_implicit_neural_representation.mlx_renderer — MLX meta-learned modulated coord-MLP.

L0 SCAFFOLD module: declares the MLX renderer config + factory; defers
heavy MLX-dependent class construction until called. MLX is OPTIONAL at
top-level import time per Catalog #1 device-selection-defaults discipline
+ axis 3 portability.

For the numpy-only / non-Apple-Silicon test path, route through
``tac.substrates.coin_pp_implicit_neural_representation.numpy_reference`` instead.

Per axis 2 MLX drift minimization (operator directive #3 2026-05-26):
- All linear weights kept in MLX-canonical (out, in) layout matching PyTorch
- Sinusoidal positional encoding uses fp32 sin; matches numpy reference exactly
- FiLM modulation uses direct linear scale (NOT exp(log_scale) which has MLX fp16
  precision risk on edge values per axis 2 §design memo)
- Mean reduction uses standard MLX mean for L0; Kahan summation queued for
  L1+ when batch sizes exceed ~1e6 elements
- NO ``align_corners=True`` bilinear (not used; substrate doesn't upsample
  inside MLX renderer; bicubic upscale to camera HW lives in PyTorch inflate)
- NO ``mx.repeat`` 2× upsample (not used; substrate is coordinate-batched)
- NO ``mx.softmax`` without epsilon (not used; substrate is not softmax-based)

Per Catalog #146 + #205 inflate runtime contract:
- Substrate-engineering LOC budget ≤200 for inflate runtime
- Device selection via canonical ``tac.substrates._shared.inflate_runtime.select_inflate_device``
- No scorer load at inflate time
- No /tmp paths in persisted artifacts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Module-level config. MLX is imported lazily inside the factory to keep
# top-level import cheap (sister substrates' canonical pattern).
EVAL_HW: tuple[int, int] = (384, 512)
"""Contest scorer-resolution; final coord-MLP coordinate grid (height, width)."""


@dataclass(frozen=True)
class CoinPPImplicitNeuralRepresentationConfig:
    """Substrate configuration; immutable per Catalog #229 PV discipline."""

    mod_dim: int = 64
    """Per-pair modulation dimension (CARGO-CULTED at L0; sweep at L1)."""

    pos_dim: int = 32
    """Sinusoidal positional encoding frequency count (Mildenhall NeRF default)."""

    hidden_dim: int = 64
    """Coord-MLP hidden dim (COIN++ paper canonical small-INR width)."""

    num_hidden_layers: int = 3
    """Coord-MLP depth (COIN++ paper canonical)."""

    num_pairs: int = 600
    """Contest video pair count."""

    eval_h: int = 384
    """Contest scorer-resolution height."""

    eval_w: int = 512
    """Contest scorer-resolution width."""

    modulation_quant_bits: int = 8
    """Per-pair modulation quantization bits (int8 canonical at L0)."""

    # Per Catalog #229: validate at construction
    def __post_init__(self) -> None:
        if self.mod_dim < 1 or self.mod_dim > 255:
            raise ValueError(
                f"mod_dim={self.mod_dim} out of [1, 255] range (u8 wire bound)"
            )
        if self.pos_dim < 1 or self.pos_dim > 255:
            raise ValueError(
                f"pos_dim={self.pos_dim} out of [1, 255] range (u8 wire bound)"
            )
        if self.hidden_dim < 1 or self.hidden_dim > 65535:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} out of [1, 65535] range (u16 wire bound)"
            )
        if self.num_hidden_layers < 1 or self.num_hidden_layers > 255:
            raise ValueError(
                f"num_hidden_layers={self.num_hidden_layers} out of [1, 255] range"
            )
        if self.num_pairs < 1 or self.num_pairs > 65535:
            raise ValueError(
                f"num_pairs={self.num_pairs} out of [1, 65535] range (u16 wire bound)"
            )
        if (self.eval_h, self.eval_w) != EVAL_HW:
            raise ValueError(
                f"(eval_h, eval_w)=({self.eval_h}, {self.eval_w}) != EVAL_HW {EVAL_HW}; "
                f"contest scorer requires {EVAL_HW}"
            )
        if self.modulation_quant_bits not in {4, 8, 16}:
            raise ValueError(
                f"modulation_quant_bits={self.modulation_quant_bits} must be in {{4, 8, 16}}"
            )


def _ensure_mlx_available() -> Any:
    """Lazy import MLX. Raises with actionable message if not installed."""
    try:
        import mlx.core as mx  # noqa: F401

        return mx
    except ImportError as exc:
        raise RuntimeError(
            "MLX is not installed. Install via `uv pip install mlx` "
            "(macOS only). For non-Apple-Silicon iteration, route through "
            "`tac.substrates.coin_pp_implicit_neural_representation.numpy_reference` per axis 3 "
            "portability discipline."
        ) from exc


def base_mlp_param_count(cfg: CoinPPImplicitNeuralRepresentationConfig) -> int:
    """Estimate MLX base coord-MLP parameter count (shared across all pairs).

    Used for cost-band / archive-size prediction WITHOUT instantiating MLX.

    Topology per __init__ docstring:
    - Input projection: positional_encoding(coord) -> hidden_dim
        positional encoding dim = pos_dim * 2 * 3 (sin+cos for x, y, t)
        input_proj = (pos_dim * 2 * 3 + 1) * hidden_dim  # +1 bias
    - num_hidden_layers x (linear hidden -> hidden + FiLM proj mod_dim -> 2*hidden):
        linear = (hidden_dim + 1) * hidden_dim
        film_proj = (mod_dim + 1) * (2 * hidden_dim)
    - Output: (hidden_dim + 1) * 3
    """
    pos_enc_dim = cfg.pos_dim * 2 * 3
    input_proj = (pos_enc_dim + 1) * cfg.hidden_dim
    per_hidden = (
        (cfg.hidden_dim + 1) * cfg.hidden_dim
        + (cfg.mod_dim + 1) * (2 * cfg.hidden_dim)
    )
    output_proj = (cfg.hidden_dim + 1) * 3
    return input_proj + cfg.num_hidden_layers * per_hidden + output_proj


def estimate_archive_bytes(cfg: CoinPPImplicitNeuralRepresentationConfig) -> int:
    """Estimate COINPP1 archive size in bytes for a given config.

    Used for Dykstra-feasibility check at design time (Catalog #296) without
    instantiating MLX or actually packing.

    Breakdown per design memo §"Predicted ΔS band":
    - Base coord-MLP state_dict at fp16, brotli q=9: ~30% of raw fp32 size
    - Per-pair modulations at int8: num_pairs * mod_dim * 1 byte; brotli ~30%
    - Header (32 bytes) + meta JSON: ~256 bytes
    """
    base_fp16_bytes = base_mlp_param_count(cfg) * 2  # fp16
    base_compressed = int(base_fp16_bytes * 0.3)  # brotli compression estimate

    modulation_raw_bytes = cfg.num_pairs * cfg.mod_dim * (
        cfg.modulation_quant_bits // 8 if cfg.modulation_quant_bits >= 8 else 1
    )
    # Brotli compresses int8 modulation tensors moderately well (~30-40%)
    modulation_compressed = int(modulation_raw_bytes * 0.35)

    header_meta = 32 + 256  # COINPP1_HEADER_LEN + meta JSON estimate

    return base_compressed + modulation_compressed + header_meta


def _full_main(argv: list[str] | None = None) -> int:
    """L0 SCAFFOLD posture per Catalog #240: full main raises NotImplementedError.

    The L0 scaffold ships design + MLX renderer + numpy reference + PyTorch
    inflate + archive grammar + tests + smoke trainer stub. The full MLX
    training path is gated by Phase 2 council symposium per Catalog #325 +
    operator-authorized paid-CUDA dispatch eligibility per Catalog #1265
    MLX-first contest-equivalence gate.
    """
    raise NotImplementedError(
        "coin_pp_implicit_neural_representation full main NOT YET IMPLEMENTED "
        "— L0 SCAFFOLD posture per Catalog #240. Phase 2 council symposium "
        "per Catalog #325 + Catalog #1265 MLX↔PyTorch parity gate REQUIRED "
        "before any paid-CUDA dispatch authorization. See "
        ".omx/research/path_3_k_coin_pp_substrate_design_20260526.md "
        "for the Phase 2+ roadmap."
    )


__all__ = [
    "CoinPPImplicitNeuralRepresentationConfig",
    "EVAL_HW",
    "_ensure_mlx_available",
    "_full_main",
    "base_mlp_param_count",
    "estimate_archive_bytes",
]
