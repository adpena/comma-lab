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

try:  # pragma: no cover — exercised on Apple Silicon with MLX installed
    import mlx.core as mx
    import mlx.nn as nn
except Exception:  # pragma: no cover — import guard for non-Apple CI
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

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
        import mlx.core as mx

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


def _mlx_sinusoidal_positional_encoding(coords: Any, num_frequencies: int) -> Any:
    """MLX sinusoidal positional encoding (NeRF/COIN++ canonical).

    Byte-stable sister of ``numpy_reference.sinusoidal_positional_encoding``:
    for coord ``c`` in ``[-1, 1]`` and ``L = num_frequencies`` produces
    ``[sin(2^k pi c), cos(2^k pi c)]`` for ``k`` in ``[0, L-1]`` per coordinate
    axis. Explicit fp32 per the axis-2 MLX drift discipline.

    Args:
        coords: ``(..., D)`` MLX array of coordinates in ``[-1, 1]``.
        num_frequencies: ``L``; output dim = ``D * L * 2``.

    Returns:
        ``(..., D * L * 2)`` MLX float32 array.
    """
    import math

    coords32 = coords.astype(mx.float32)
    d = int(coords32.shape[-1])
    bands = (2.0 ** mx.arange(num_frequencies, dtype=mx.float32)) * mx.array(
        math.pi, dtype=mx.float32
    )
    scaled = coords32[..., None] * bands  # (..., D, L)
    stacked = mx.stack([mx.sin(scaled), mx.cos(scaled)], axis=-1)  # (..., D, L, 2)
    out_shape = (*coords32.shape[:-1], d * int(num_frequencies) * 2)
    return mx.reshape(stacked, out_shape)


def _mlx_make_coord_grid_nhwc(height: int, width: int, t: float) -> Any:
    """MLX (H, W, 3) coord grid of (x, y, t) in ``[-1, 1]`` for x/y; t as-is.

    Byte-stable sister of ``numpy_reference.make_coord_grid_nhwc``.
    """
    x = (
        mx.linspace(-1.0, 1.0, width).astype(mx.float32)
        if width > 1
        else mx.zeros((1,), dtype=mx.float32)
    )
    y = (
        mx.linspace(-1.0, 1.0, height).astype(mx.float32)
        if height > 1
        else mx.zeros((1,), dtype=mx.float32)
    )
    yy = y[:, None] * mx.ones((1, width), dtype=mx.float32)
    xx = mx.ones((height, 1), dtype=mx.float32) * x[None, :]
    tt = mx.full((height, width), float(t), dtype=mx.float32)
    return mx.stack([xx, yy, tt], axis=-1)  # (H, W, 3)


class CoinPPRendererMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Trainable COIN++ meta-modulated SIREN coord-MLP — MLX ``nn.Module``.

    MLX-SCORE-AWARE-HARNESS-WAVE 2026-05-27: the prior blocker was that this
    substrate shipped only config + cost estimators — there was NO MLX
    SIREN/COIN++ renderer forward. This class lands the COIN++ base coord-MLP
    (Dupont et al. 2022) + per-pair modulation network as a single trainable
    ``mlx.nn.Module``, mirroring the canonical numpy
    ``coord_mlp_forward`` topology exactly:

    1. sinusoidal positional encoding of (x, y, t) coords (NeRF; pos_dim freqs)
    2. input projection -> hidden_dim
    3. ``num_hidden_layers`` x [Linear hidden->hidden + FiLM(scale+1, shift) +
       sin] where (scale, shift) come from a per-layer FiLM projection of the
       per-pair modulation vector
    4. output projection -> sigmoid -> RGB in [0, 1]

    The per-pair modulation table ``per_pair_modulation`` ``(num_pairs,
    mod_dim)`` is the learnable per-instance code (the COIN++ "modulations");
    the base coord-MLP + FiLM proj weights are shared across all pairs.

    Exposes the harness's ``reconstruct_pair(idx) -> (rgb_0, rgb_1)`` NCHW
    ``[0, 1]`` convention: frame_0 uses ``t=-1``, frame_1 uses ``t=+1``.

    Non-promotable ``[macOS-MLX research-signal]`` per CLAUDE.md "MLX
    portable-local-substrate authority".
    """

    def __init__(self, cfg: CoinPPImplicitNeuralRepresentationConfig) -> None:
        if nn is None:
            raise RuntimeError(
                "MLX is not installed; CoinPPRendererMLX requires MLX "
                "(macOS Apple Silicon). Route through numpy_reference for "
                "non-Apple-Silicon iteration per axis-3 portability."
            )
        super().__init__()
        self.cfg = cfg
        pos_enc_dim = cfg.pos_dim * 2 * 3  # (x, y, t) x L x (sin, cos)
        self.input_proj = nn.Linear(pos_enc_dim, cfg.hidden_dim)
        self.hidden = [
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            for _ in range(cfg.num_hidden_layers)
        ]
        self.film = [
            nn.Linear(cfg.mod_dim, 2 * cfg.hidden_dim)
            for _ in range(cfg.num_hidden_layers)
        ]
        self.output_proj = nn.Linear(cfg.hidden_dim, 3)
        # Learnable per-pair modulation code (the COIN++ "modulations").
        key = mx.random.key(0)
        self.per_pair_modulation = (
            mx.random.normal(shape=(cfg.num_pairs, cfg.mod_dim), key=key) * 0.01
        )

    def _render_frame(self, modulation: Any, t: float) -> Any:
        """Render one frame for a batch of per-pair modulations at coord-time t.

        Args:
            modulation: ``(B, mod_dim)`` per-pair modulation vectors.
            t: coordinate-time (frame_0 -> -1.0, frame_1 -> +1.0).

        Returns:
            ``(B, 3, H, W)`` RGB in ``[0, 1]`` (NCHW; harness convention).
        """
        cfg = self.cfg
        b = int(modulation.shape[0])
        grid = _mlx_make_coord_grid_nhwc(cfg.eval_h, cfg.eval_w, t)  # (H, W, 3)
        coords = mx.broadcast_to(
            grid[None], (b, cfg.eval_h, cfg.eval_w, 3)
        )  # (B, H, W, 3)
        pos_enc = _mlx_sinusoidal_positional_encoding(coords, cfg.pos_dim)
        h = self.input_proj(pos_enc)
        mod_b = modulation[:, None, None, :]  # (B, 1, 1, mod_dim)
        for i in range(cfg.num_hidden_layers):
            h = self.hidden[i](h)
            film_out = self.film[i](mod_b)  # (B, 1, 1, 2*hidden)
            scale = film_out[..., : cfg.hidden_dim] + 1.0
            shift = film_out[..., cfg.hidden_dim :]
            h = mx.sin(h * scale + shift)
        rgb = mx.sigmoid(self.output_proj(h))  # (B, H, W, 3) in [0, 1]
        return mx.transpose(rgb, (0, 3, 1, 2))  # (B, 3, H, W)

    def reconstruct_pair(self, pair_indices: Any) -> tuple[Any, Any]:
        """Harness forward: per-pair modulation -> (rgb_0, rgb_1).

        Args:
            pair_indices: ``(B,)`` int tensor in ``[0, num_pairs)``.

        Returns:
            ``(rgb_0, rgb_1)`` each ``(B, 3, eval_h, eval_w)`` in ``[0, 1]``.
        """
        modulation = self.per_pair_modulation[pair_indices]
        rgb_0 = self._render_frame(modulation, t=-1.0)
        rgb_1 = self._render_frame(modulation, t=1.0)
        return rgb_0, rgb_1

    def __call__(self, pair_indices: Any) -> tuple[Any, Any]:
        """Alias for :meth:`reconstruct_pair` (default forward)."""
        return self.reconstruct_pair(pair_indices)


def _full_main(argv: list[str] | None = None) -> int:
    """L0 SCAFFOLD posture per Catalog #240: full main raises NotImplementedError.

    NOTE: the canonical MLX-first score-aware training entry point is the
    sister trainer
    ``experiments/train_substrate_coin_pp_implicit_neural_representation_mlx.py``
    (``--full``), which routes :class:`CoinPPRendererMLX` through the canonical
    ``tac.substrates._shared.mlx_score_aware`` harness. This module-level
    ``_full_main`` (config + estimators surface) retains the L0 posture; the
    trainable renderer + training path live in the trainer + the harness.
    """
    raise NotImplementedError(
        "coin_pp_implicit_neural_representation.mlx_renderer._full_main is the "
        "config/estimators surface (L0 SCAFFOLD posture per Catalog #240). "
        "The MLX-first score-aware training entry point is "
        "experiments/train_substrate_coin_pp_implicit_neural_representation_mlx.py "
        "--full, which trains CoinPPRendererMLX via the canonical "
        "tac.substrates._shared.mlx_score_aware harness."
    )


__all__ = [
    "EVAL_HW",
    "CoinPPImplicitNeuralRepresentationConfig",
    "CoinPPRendererMLX",
    "_ensure_mlx_available",
    "_full_main",
    "base_mlp_param_count",
    "estimate_archive_bytes",
]
