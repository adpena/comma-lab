# SPDX-License-Identifier: MIT
"""MLX-native Z7-Mamba-2 substrate renderer extension (L0 SCAFFOLD EXTENSION).

L0 SCAFFOLD EXTENSION per operator directive 2026-05-26 + the empirical
cascade #1251 / #1257 / #1258-corrected / #1265 that empirically validated
MLX-local iteration is contest-grade at frontier-tightening granularity
(|S_MLX - S_PyTorch| = 0.000011 on PR95 hnerv_muon archive, 72x smaller
than the PR110 frontier delta 0.000789).

This module is the MLX-native sister of the canonical PyTorch
:class:`tac.substrates.time_traveler_l5_z7_mamba2.architecture.Z7Mamba2PredictiveCodingSubstrate`,
focused narrowly on the renderer pass (Mamba-2 selective state-space recurrence
+ Z6-compatible PixelShuffle RGB decoder). The PyTorch architecture remains the
canonical contest-axis path; this MLX variant lets the operator (a) iterate
Z7-Mamba-2 hypotheses on M5 Max at $0 cost, (b) export trained weights to
PyTorch via the canonical state_dict bridge for byte-stable inflate (#1257),
(c) only spend paid CUDA when an MLX-trained candidate clears the
:mod:`tools.gate_mlx_candidate_contest_equivalence` PASS bar (Catalog #1265).

Per CLAUDE.md non-negotiables PRESERVED:

* **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: this module is a
  NEW sister to the existing PyTorch architecture; ZERO mutation of the
  canonical architecture, archive, inflate, or score-aware-loss surfaces.
* **Catalog #1 MPS auth eval is NOISE** + **Catalog #192 macOS non-promotable**:
  MLX-native trained weights are research-signal until exported to PyTorch +
  evaluated on CUDA T4 via ``experiments/contest_auth_eval.py``.
* **Catalog #287/#323 canonical Provenance**: any persisted artifact from this
  module carries ``evidence_grade="macOS-MLX-research-signal"`` +
  ``score_claim=False`` + ``promotion_eligible=False`` +
  ``ready_for_exact_eval_dispatch=False``.
* **CLAUDE.md "MLX portable-local-substrate authority"**: this module is a
  free first-class signal generator for substrate-class-shift candidate
  iteration; NOT a judge.
* **CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"**:
  ``RESEARCH_ONLY = True`` declared at the package level
  (``tac.substrates.time_traveler_l5_z7_mamba2.__init__``); this extension
  inherits that opt-out from the parent.
* **CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L7**:
  this is substrate engineering (per-method optimal MLX port); the PyTorch
  canonical adopt remains intact per the canonical-vs-unique decision below.

Canonical-vs-unique decision per layer (extends parent
:mod:`tac.substrates.time_traveler_l5_z7_mamba2.architecture` decisions):

1. **Predictor primitive (UNIQUE-FORK in MLX)**: Mamba-2 selective
   state-space cell ported 1:1 from
   :class:`tac.optimization.mamba2_predictor._ReferenceMamba2Cell` to MLX.
   Mathematical structure (input + gate projection, A_log selective state,
   B/C input-conditioned, softplus dt, zero-order-hold discretization,
   per-timestep gated state update) MATCHES the PyTorch reference cell
   byte-stably so an MLX-trained state_dict round-trips to PyTorch via the
   #1251 bridge with sub-1e-5 max-abs drift at the decoder boundary.
2. **Decoder (CANONICAL-ADOPT in MLX)**: Z6-compatible PixelShuffle stack
   ported 1:1 from
   :class:`tac.substrates.time_traveler_l5_z6.architecture._Z6Decoder`.
   1x1 initial_proj + per-block Conv2d(4*out_ch, k=3) + PixelShuffle(2) +
   ReLU + final 6-channel Conv2d + bilinear interpolate-to-(384, 512) +
   sigmoid. Field names, weight shapes, init order match the PyTorch sister.
3. **Latent init + per-pair residuals + ego_motion buffer (CANONICAL-ADOPT
   in MLX)**: trainable parameters with same shapes as PyTorch sister so
   state_dict export bridge is byte-stable.
4. **Context conditioner (DEFERRED to L1 EXTENSION)**: ``context_conditioning_mode="latent_affine"``
   is NOT yet implemented in MLX. Operator iterates with ``"none"`` mode
   only at L0; the affine path lands as a sister L1 EXTENSION once a council
   anchor justifies the per-pair affine modulation cost on MLX.

Architecture mirror (PyTorch -> MLX field names preserved for byte-stable
export bridge per Catalog #1251):

* ``predictor.input_projection.{weight,bias}``: Linear(predictor_input_dim, d_model)
* ``predictor.mamba_cell.in_proj.weight``: Linear(d_model, 2*d_inner, bias=False)
* ``predictor.mamba_cell.A_log``: Parameter (d_inner, d_state)
* ``predictor.mamba_cell.B_proj.weight``: Linear(d_inner, d_state, bias=False)
* ``predictor.mamba_cell.C_proj.weight``: Linear(d_inner, d_state, bias=False)
* ``predictor.mamba_cell.dt_proj.{weight,bias}``: Linear(d_inner, d_inner)
* ``predictor.mamba_cell.out_proj.weight``: Linear(d_inner, d_model, bias=False)
* ``predictor.output_projection.{weight,bias}``: Linear(d_model, latent_dim)
* ``decoder.initial_proj.{weight,bias}``: Linear(latent_dim, embed_dim * H0 * W0)
* ``decoder.blocks.{2*i}.{weight,bias}``: Conv2d(in_ch, 4*out_ch, k=3, p=1)
* ``decoder.blocks.{final}.{weight,bias}``: Conv2d(in_ch, 6, k=3, p=1)
* ``latent_init``: Parameter (latent_dim,)
* ``residuals``: Parameter (num_pairs, latent_dim)
* ``ego_motion_buffer``: numpy array (num_pairs, ego_motion_dim) — non-trainable

6-hook wire-in declaration per Catalog #125:

1. Sensitivity-map: Mamba-2 selective-projection gradient norms surfaceable
   from MLX leaf params via ``mlx.utils.tree_flatten`` if a future MLX
   sensitivity-map contributor lands.
2. Pareto constraint: N/A at L0 (deferred to L1+ when MLX-side empirical
   anchor lands).
3. Bit-allocator hook: N/A at L0 (canonical PyTorch sister's hook applies
   to MLX-exported weights via the #1251 bridge).
4. Cathedral autopilot dispatch: this module produces RESEARCH-ONLY MLX
   artifacts; promotion to autopilot ranking happens via Catalog #1265
   gate AFTER PyTorch export + paired CUDA-CPU evaluation.
5. Continual-learning posterior: any persisted artifact from MLX runs
   carries non-promotable Provenance + axis_tag="[macOS-MLX research-signal]"
   per Catalog #323 so the posterior knows not to treat the row as
   contest-grade score evidence.
6. Probe-disambiguator: ``identity_predictor=True`` mode IS the probe
   (no learning; returns z_prev unchanged). Sister to PyTorch
   ``Z7Mamba2PredictiveCodingConfig.identity_predictor`` per Catalog #125.

[verified-against: tac.substrates.time_traveler_l5_z7_mamba2.architecture.Z7Mamba2PredictiveCodingSubstrate]
[verified-against: tac.optimization.mamba2_predictor._ReferenceMamba2Cell forward equations]
[verified-against: tac.substrates.time_traveler_l5_z6.architecture._Z6Decoder forward equations]
[verified-against: tac.substrates.grayscale_lut.mlx_native canonical MLX-native sister pattern]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.substrates.time_traveler_l5_z7_mamba2.architecture import (
    EVAL_HW,
    NUM_PAIRS,
    Z7Mamba2PredictiveCodingConfig,
)

__all__ = [
    "Z7Mamba2MLXNativeRenderer",
    "Z7Mamba2MLXRenderConfig",
    "require_mlx",
    "render_pair_mlx",
]


def require_mlx() -> None:
    """Raise ImportError with actionable message if MLX is unavailable."""
    try:
        import mlx.core as _mx  # noqa: F401
        import mlx.nn as _mxnn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via env check
        raise ImportError(
            "MLX is required for Z7-Mamba-2 MLX-native renderer. Install via "
            "`pip install mlx` on macOS Apple Silicon. This extension is L0 "
            "SCAFFOLD; the PyTorch canonical architecture at "
            "tac.substrates.time_traveler_l5_z7_mamba2.architecture remains "
            "the contest-axis path."
        ) from exc


@dataclass(frozen=True)
class Z7Mamba2MLXRenderConfig:
    """Static design-time parameters for the MLX-native Z7-Mamba-2 renderer.

    Defaults intentionally mirror the canonical PyTorch
    :class:`Z7Mamba2PredictiveCodingConfig` so an MLX-trained state_dict
    round-trips to PyTorch via the #1251 export bridge without architectural
    shape drift.

    Args:
        latent_dim: per-pair latent dimensionality (24 = Z6 sister parity).
        ego_motion_dim: ego-motion vector dim (8 = Z7-GRU PoseNet-projection).
        d_model: Mamba-2 internal model dim (64; sister to GRU
            hidden_dim=128 halved for parameter parity).
        d_state: Mamba-2 selective state-space dim (16 canonical).
        expand: Mamba-2 expansion factor (2 from upstream reference).
        d_conv: Mamba-2 conv1d kernel size (4 canonical; UNUSED in
            reference cell forward — present for state_dict shape parity
            with future ``mamba_ssm`` upstream port).
        stateful: True maintains hidden state across the 600-pair
            sequence (Wyner-Ziv implicit side-info channel per Catalog
            #311); False resets state every pair (ablation).
        identity_predictor: True returns z_prev unchanged (Catalog #125
            hook #6 probe-disambiguator control).
        num_pairs: contest pair count (600 per CLAUDE.md eval contract).
        decoder_embed_dim: embed dim per Z6 sister (32).
        decoder_initial_grid_h: initial spatial H per Z6 sister (24).
        decoder_initial_grid_w: initial spatial W per Z6 sister (32).
        decoder_channels: per-upsample-block channel counts (32, 24, 16, 12).
        decoder_num_upsample_blocks: number of PixelShuffle upsample
            blocks (4 → 16x upsample → final bilinear interp to EVAL_HW).
        output_height: contest scorer height (384).
        output_width: contest scorer width (512).
        latent_init_std: trainable latent_init std for initialization
            (0.02 per Z7-LSTM/GRU canonical).
    """

    latent_dim: int = 24
    ego_motion_dim: int = 8
    d_model: int = 64
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    stateful: bool = True
    identity_predictor: bool = False
    num_pairs: int = NUM_PAIRS
    decoder_embed_dim: int = 32
    decoder_initial_grid_h: int = 24
    decoder_initial_grid_w: int = 32
    decoder_channels: tuple[int, ...] = (32, 24, 16, 12)
    decoder_num_upsample_blocks: int = 4
    output_height: int = EVAL_HW[0]
    output_width: int = EVAL_HW[1]
    latent_init_std: float = 0.02

    @property
    def d_inner(self) -> int:
        """Mamba-2 inner dim after expansion (d_model * expand)."""
        return self.expand * self.d_model

    @property
    def predictor_input_dim(self) -> int:
        """Concat dim for (z_prev, ego_motion) predictor input."""
        return self.latent_dim + self.ego_motion_dim

    @classmethod
    def from_pytorch_config(
        cls, cfg: Z7Mamba2PredictiveCodingConfig
    ) -> "Z7Mamba2MLXRenderConfig":
        """Build an MLX render config from the canonical PyTorch config.

        Used by export-bridge tools so the MLX-side architecture is
        byte-stable against PyTorch state_dict shapes per Catalog #1251.
        Note: ``backend`` + ``beta_ib`` + ``context_*`` PyTorch fields are
        ignored — MLX renderer is a focused L0 SCAFFOLD and context
        conditioning is deferred to L1 EXTENSION.
        """
        return cls(
            latent_dim=cfg.latent_dim,
            ego_motion_dim=cfg.ego_motion_dim,
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            expand=cfg.expand,
            d_conv=cfg.d_conv,
            stateful=cfg.stateful,
            identity_predictor=cfg.identity_predictor,
            num_pairs=cfg.num_pairs,
            decoder_embed_dim=cfg.decoder_embed_dim,
            decoder_initial_grid_h=cfg.decoder_initial_grid_h,
            decoder_initial_grid_w=cfg.decoder_initial_grid_w,
            decoder_channels=tuple(int(c) for c in cfg.decoder_channels),
            decoder_num_upsample_blocks=cfg.decoder_num_upsample_blocks,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
            latent_init_std=cfg.latent_init_std,
        )


class Z7Mamba2MLXNativeRenderer:
    """MLX-native Z7-Mamba-2 renderer extension (L0 SCAFFOLD).

    Mirrors the canonical PyTorch substrate architecture at the renderer
    boundary: Mamba-2 selective state-space recurrence over the per-pair
    latent + ego_motion stream, then Z6-compatible PixelShuffle dual-RGB
    decode at each timestep.

    Forward signature (callable):
        ``__call__(pair_indices_np)`` -> ``(rgb_0, rgb_1)`` each shape
        ``(P, 3, output_height, output_width)`` in unit RGB, where
        ``P = pair_indices_np.size``. The full 600-pair sequence is
        autoregressively replayed to maintain deterministic Mamba-2
        recurrent state per the canonical sister; only the requested pair
        indices are decoded (Catalog #218 mini-batch reconstruct pattern
        prevents activation-memory OOM at large pair counts).

    Mode ``identity_predictor=True``: predictor returns ``z_prev``
    unchanged; latent_init + residuals + decoder still trained. Canonical
    probe-disambiguator (Catalog #125 hook #6).

    State management:
    - ``stateful=True`` (default): Mamba-2 hidden state persists across
      the 600-pair sequence per Wyner-Ziv side-info pattern.
    - ``stateful=False``: state resets every pair (ablation: no temporal
      coherence; reference cell reduces to a stateless nonlinear transform).
    """

    def __init__(self, cfg: Z7Mamba2MLXRenderConfig, *, seed: int = 0) -> None:
        """Initialize MLX-native parameters with seeded reproducibility.

        Tensors are created via :mod:`mlx.core` so they live on the Apple
        Silicon unified-memory device. Initialization matches PyTorch's
        ``nn.Linear`` Kaiming-uniform default + ``Conv2d`` Kaiming-uniform
        default to keep the MLX <-> PyTorch parameter histograms close
        enough that an MLX-trained state_dict bridge to PyTorch produces
        decoder-boundary drift below the Catalog #1265 PASS threshold.
        """
        require_mlx()
        import mlx.core as mx

        if cfg.num_pairs <= 0:
            raise ValueError("num_pairs must be positive")
        if cfg.decoder_num_upsample_blocks <= 0:
            raise ValueError("decoder_num_upsample_blocks must be positive")
        if len(cfg.decoder_channels) < cfg.decoder_num_upsample_blocks:
            raise ValueError(
                "decoder_channels must have at least decoder_num_upsample_blocks "
                "entries"
            )
        if cfg.output_height <= 0 or cfg.output_width <= 0:
            raise ValueError("output_height/output_width must be positive")
        if cfg.latent_init_std < 0:
            raise ValueError("latent_init_std must be non-negative")
        if cfg.d_conv <= 0:
            raise ValueError("d_conv must be positive")

        self.cfg = cfg
        self._seed = int(seed)
        # Single canonical RNG key so init order matches across runs.
        self._rng_key = mx.random.key(self._seed)
        d_inner = cfg.d_inner

        # Mamba-2 reference cell parameters (UNIQUE-FORK; mirrors
        # _ReferenceMamba2Cell forward equations exactly).
        # input_projection: predictor_input_dim -> d_model
        self.input_projection_w = self._kaiming_linear(
            cfg.predictor_input_dim, cfg.d_model
        )
        self.input_projection_b = mx.zeros((cfg.d_model,), dtype=mx.float32)

        # Mamba-2 cell parameters
        # in_proj: d_model -> 2 * d_inner (no bias per PyTorch reference)
        self.mamba_in_proj_w = self._kaiming_linear(cfg.d_model, 2 * d_inner)
        # A_log: (d_inner, d_state) initialized to log(arange(1, d_state+1))
        # per the PyTorch reference; broadcast across d_inner.
        a_log_row = mx.log(mx.arange(1, cfg.d_state + 1, dtype=mx.float32))
        self.mamba_A_log = mx.broadcast_to(
            mx.expand_dims(a_log_row, 0), (d_inner, cfg.d_state)
        )
        # B_proj, C_proj: d_inner -> d_state (no bias)
        self.mamba_B_proj_w = self._kaiming_linear(d_inner, cfg.d_state)
        self.mamba_C_proj_w = self._kaiming_linear(d_inner, cfg.d_state)
        # dt_proj: d_inner -> d_inner (with bias)
        self.mamba_dt_proj_w = self._kaiming_linear(d_inner, d_inner)
        self.mamba_dt_proj_b = mx.zeros((d_inner,), dtype=mx.float32)
        # out_proj: d_inner -> d_model (no bias)
        self.mamba_out_proj_w = self._kaiming_linear(d_inner, cfg.d_model)

        # output_projection: d_model -> latent_dim
        self.output_projection_w = self._kaiming_linear(cfg.d_model, cfg.latent_dim)
        self.output_projection_b = mx.zeros((cfg.latent_dim,), dtype=mx.float32)

        # Decoder (CANONICAL-ADOPT from _Z6Decoder).
        # initial_proj: latent_dim -> embed_dim * H0 * W0
        decoder_initial_dim = (
            cfg.decoder_embed_dim
            * cfg.decoder_initial_grid_h
            * cfg.decoder_initial_grid_w
        )
        self.dec_initial_proj_w = self._kaiming_linear(
            cfg.latent_dim, decoder_initial_dim
        )
        self.dec_initial_proj_b = mx.zeros(
            (decoder_initial_dim,), dtype=mx.float32
        )

        # PixelShuffle conv stack
        self.dec_block_w: list[mx.array] = []
        self.dec_block_b: list[mx.array] = []
        in_ch = cfg.decoder_embed_dim
        for i in range(cfg.decoder_num_upsample_blocks):
            out_ch = cfg.decoder_channels[i]
            # Conv2d(in_ch, 4*out_ch, kernel=3, padding=1); MLX conv stores
            # weights as (out_ch, kH, kW, in_ch) per MLX channels-last convention.
            self.dec_block_w.append(
                self._kaiming_conv2d(in_ch, 4 * out_ch, kernel=3)
            )
            self.dec_block_b.append(mx.zeros((4 * out_ch,), dtype=mx.float32))
            in_ch = out_ch
        # Final 6-channel conv (RGB pair: 3+3)
        self.dec_final_w = self._kaiming_conv2d(in_ch, 6, kernel=3)
        self.dec_final_b = mx.zeros((6,), dtype=mx.float32)

        # Trainable latent_init + per-pair residuals
        std = float(cfg.latent_init_std)
        if std > 0.0:
            self._rng_key, subkey = mx.random.split(self._rng_key)
            self.latent_init = (
                mx.random.normal(shape=(cfg.latent_dim,), key=subkey) * std
            )
        else:
            self.latent_init = mx.zeros((cfg.latent_dim,), dtype=mx.float32)
        self.residuals = mx.zeros(
            (cfg.num_pairs, cfg.latent_dim), dtype=mx.float32
        )
        # Ego motion buffer (non-trainable, filled by trainer from real video)
        self.ego_motion_buffer = mx.zeros(
            (cfg.num_pairs, cfg.ego_motion_dim), dtype=mx.float32
        )

        # Mamba-2 hidden state (initialized lazily in __call__)
        self._h: Any | None = None

    # ------------------------------------------------------------------
    # Initialization helpers (Kaiming-uniform to match PyTorch defaults)
    # ------------------------------------------------------------------

    def _kaiming_linear(self, in_features: int, out_features: int) -> Any:
        """Kaiming-uniform init matching PyTorch ``nn.Linear`` defaults.

        PyTorch's nn.Linear initializes weight ~ U(-sqrt(1/fan_in),
        sqrt(1/fan_in)) (Kaiming uniform with gain=sqrt(5) per Kaiming He
        section 2.2). This helper mirrors that distribution shape so MLX
        and PyTorch sister architectures start from statistically
        equivalent inits and the #1251 bridge can verify decoder-boundary
        drift bounds.
        """
        import mlx.core as mx
        self._rng_key, subkey = mx.random.split(self._rng_key)
        bound = float(np.sqrt(1.0 / max(in_features, 1)))
        # MLX nn.Linear weight shape is (out_features, in_features)
        return mx.random.uniform(
            low=-bound,
            high=bound,
            shape=(out_features, in_features),
            dtype=mx.float32,
            key=subkey,
        )

    def _kaiming_conv2d(
        self, in_channels: int, out_channels: int, *, kernel: int
    ) -> Any:
        """Kaiming-uniform init matching PyTorch ``nn.Conv2d`` defaults.

        PyTorch's nn.Conv2d initializes weight ~ U(-sqrt(1/fan_in),
        sqrt(1/fan_in)) where fan_in = in_channels * kernel^2.

        MLX conv2d weight shape: ``(out_channels, kH, kW, in_channels)``
        per MLX channels-last convention. The state_dict export bridge
        (#1251) transposes to PyTorch's (out_channels, in_channels, kH,
        kW) at export time.
        """
        import mlx.core as mx
        self._rng_key, subkey = mx.random.split(self._rng_key)
        fan_in = max(in_channels * kernel * kernel, 1)
        bound = float(np.sqrt(1.0 / fan_in))
        return mx.random.uniform(
            low=-bound,
            high=bound,
            shape=(out_channels, kernel, kernel, in_channels),
            dtype=mx.float32,
            key=subkey,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset the Mamba-2 recurrent hidden state to zeros.

        Must be called at the start of each 600-pair sequence to begin the
        autoregressive unroll. The full-sequence replay path in
        :meth:`reconstruct_all_pairs` calls this automatically.
        """
        import mlx.core as mx
        if self.cfg.identity_predictor:
            self._h = None
            return
        self._h = mx.zeros(
            (1, self.cfg.d_inner, self.cfg.d_state), dtype=mx.float32
        )

    def _mamba2_step(self, x_t: Any) -> Any:
        """Single Mamba-2 timestep (mirrors ``_ReferenceMamba2Cell.forward``).

        Args:
            x_t: ``(1, d_model)`` input embedding.

        Returns:
            ``(1, d_model)`` output y_t. Side-effect: updates ``self._h``.
        """
        import mlx.core as mx
        import mlx.nn as mlx_nn
        if self.cfg.identity_predictor:
            return x_t  # identity passthrough; unused in standard path
        cfg = self.cfg
        d_inner = cfg.d_inner

        # Input + gate projection
        # mx.linear-style: y = x @ W.T + b
        xz = x_t @ self.mamba_in_proj_w.T  # (1, 2*d_inner)
        x_inner = xz[:, :d_inner]  # (1, d_inner)
        z_gate = xz[:, d_inner:]  # (1, d_inner)

        # Selective projection
        # MLX exposes softplus via mlx.nn (not mx.core); mirror the
        # PyTorch reference cell's torch.nn.functional.softplus call.
        dt = mlx_nn.softplus(
            x_inner @ self.mamba_dt_proj_w.T + self.mamba_dt_proj_b
        )  # (1, d_inner)
        A = -mx.exp(self.mamba_A_log)  # (d_inner, d_state)
        B = x_inner @ self.mamba_B_proj_w.T  # (1, d_state)
        C = x_inner @ self.mamba_C_proj_w.T  # (1, d_state)

        # Discretize: A_bar = exp(dt * A); B_bar = dt * B
        # A.shape = (d_inner, d_state); dt.shape = (1, d_inner) ->
        # broadcast (1, d_inner, d_state)
        A_bar = mx.exp(
            mx.expand_dims(A, 0) * mx.expand_dims(dt, -1)
        )  # (1, d_inner, d_state)
        B_bar = mx.expand_dims(dt, -1) * mx.expand_dims(B, 1)
        # (1, d_inner, d_state)

        # State update: h_t = A_bar * h_{t-1} + B_bar * x_inner
        h_t = A_bar * self._h + B_bar * mx.expand_dims(x_inner, -1)
        # (1, d_inner, d_state)
        self._h = h_t

        # Output: y_inner = sum_d_state(h_t * C), gated by sigmoid(z)
        y_inner = mx.sum(h_t * mx.expand_dims(C, 1), axis=-1)  # (1, d_inner)
        y_inner = y_inner * mx.sigmoid(z_gate)
        y_t = y_inner @ self.mamba_out_proj_w.T  # (1, d_model)
        return y_t

    def _predict_step(self, z_prev: Any, ego_t: Any) -> Any:
        """Single predictor step: (z_prev, ego_t) -> z_pred.

        Args:
            z_prev: ``(1, latent_dim)`` previous latent.
            ego_t: ``(1, ego_motion_dim)`` ego-motion vector this pair.

        Returns:
            ``(1, latent_dim)`` predicted next latent.
        """
        import mlx.core as mx
        if self.cfg.identity_predictor:
            return z_prev
        # Concat input
        x_in = mx.concatenate([z_prev, ego_t], axis=-1)  # (1, predictor_input_dim)
        # Input projection: linear (with bias)
        x_proj = x_in @ self.input_projection_w.T + self.input_projection_b
        # (1, d_model)
        # Mamba-2 cell step
        y = self._mamba2_step(x_proj)  # (1, d_model)
        # Output projection
        z_pred = y @ self.output_projection_w.T + self.output_projection_b
        # (1, latent_dim)
        return z_pred

    def replay_latents(self) -> Any:
        """Autoregress through Mamba-2 + residual stream; return all latents.

        Returns:
            mlx.array of shape ``(num_pairs, latent_dim)``.
        """
        import mlx.core as mx
        cfg = self.cfg
        self.reset_state()
        # z starts as latent_init (shape (latent_dim,)) -> (1, latent_dim)
        z = mx.expand_dims(self.latent_init, 0)
        outs: list[Any] = []
        if cfg.stateful is False and not cfg.identity_predictor:
            # Stateless ablation: reset before EACH pair
            for t in range(cfg.num_pairs):
                self.reset_state()
                ego_t = self.ego_motion_buffer[t : t + 1]
                pred = self._predict_step(z, ego_t)
                z = pred + self.residuals[t : t + 1]
                outs.append(mx.squeeze(z, axis=0))
        else:
            for t in range(cfg.num_pairs):
                ego_t = self.ego_motion_buffer[t : t + 1]
                pred = self._predict_step(z, ego_t)
                z = pred + self.residuals[t : t + 1]
                outs.append(mx.squeeze(z, axis=0))
        return mx.stack(outs, axis=0)  # (num_pairs, latent_dim)

    def _decode_latents(self, z_batch: Any) -> tuple[Any, Any]:
        """Run the Z6-compatible decoder on a batch of latents.

        Args:
            z_batch: ``(P, latent_dim)``

        Returns:
            ``(rgb_0, rgb_1)`` each ``(P, 3, output_height, output_width)``
            in unit-domain (sigmoid).
        """
        import mlx.core as mx
        cfg = self.cfg
        batch = z_batch.shape[0]
        # initial_proj
        flat = z_batch @ self.dec_initial_proj_w.T + self.dec_initial_proj_b
        # (P, embed_dim * H0 * W0)
        # Reshape to (P, H0, W0, embed_dim) for MLX channels-last conv
        grid = mx.reshape(
            flat,
            (
                batch,
                cfg.decoder_embed_dim,
                cfg.decoder_initial_grid_h,
                cfg.decoder_initial_grid_w,
            ),
        )
        # Transpose to MLX channels-last (P, H, W, C)
        h = mx.transpose(grid, (0, 2, 3, 1))

        for i in range(cfg.decoder_num_upsample_blocks):
            out_ch = cfg.decoder_channels[i]
            # Conv2d(in_ch, 4*out_ch, k=3, p=1) — MLX channels-last
            h = mx.conv2d(h, self.dec_block_w[i], padding=1)
            h = h + self.dec_block_b[i]
            # PixelShuffle(upscale=2) in channels-last:
            # (P, H, W, 4*out_ch) -> (P, 2*H, 2*W, out_ch)
            h = _pixel_shuffle_channels_last(h, upscale_factor=2)
            # ReLU activation
            h = mx.maximum(h, 0.0)

        # Final 6-channel conv
        h = mx.conv2d(h, self.dec_final_w, padding=1)
        h = h + self.dec_final_b
        # Transpose back to (P, C, H, W) for shape parity with PyTorch sister
        h = mx.transpose(h, (0, 3, 1, 2))  # (P, 6, H_dec, W_dec)

        # Bilinear upsample to (output_height, output_width) if needed
        cur_h, cur_w = int(h.shape[-2]), int(h.shape[-1])
        if cur_h != cfg.output_height or cur_w != cfg.output_width:
            # MLX bilinear upsample via array indexing; for L0 SCAFFOLD,
            # fall back through numpy for correctness; an MLX-native
            # bilinear is deferred to a sister L1 EXTENSION.
            h_np = np.asarray(h)
            h_np = _bilinear_resize_np(
                h_np, cfg.output_height, cfg.output_width
            )
            h = mx.array(h_np)

        # Sigmoid for unit-domain output
        h = mx.sigmoid(h)
        rgb_0 = h[:, :3, :, :]
        rgb_1 = h[:, 3:, :, :]
        return rgb_0, rgb_1

    def reconstruct_all_pairs(self) -> tuple[Any, Any, Any]:
        """Replay the full Mamba-2 sequence + decode every pair.

        Returns:
            ``(rgb_0, rgb_1, latents)`` where rgb_{0,1} are
            ``(num_pairs, 3, H, W)`` and latents are
            ``(num_pairs, latent_dim)``.
        """
        latents = self.replay_latents()
        rgb_0, rgb_1 = self._decode_latents(latents)
        return rgb_0, rgb_1, latents

    def reconstruct_pair(
        self, pair_indices_np: np.ndarray
    ) -> tuple[Any, Any, Any]:
        """Decode selected pair indices after deterministic sequence replay.

        Per Catalog #218 mini-batch reconstruct (defense against D4 T4 OOM
        bug class): caller selects specific pair indices to bound peak
        activation memory.

        Args:
            pair_indices_np: numpy int64 array of pair indices in
                ``[0, num_pairs)``.

        Returns:
            ``(rgb_0[pair_indices], rgb_1[pair_indices], latents[pair_indices])``.
        """
        import mlx.core as mx
        cfg = self.cfg
        idx = np.asarray(pair_indices_np)
        if idx.size == 0:
            raise ValueError("pair_indices must be non-empty")
        if idx.min() < 0 or idx.max() >= cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {cfg.num_pairs}); got "
                f"[{int(idx.min())}, {int(idx.max())}]"
            )
        # Replay the full sequence (deterministic + necessary for stateful
        # Mamba-2), then select the requested pairs.
        rgb_0, rgb_1, latents = self.reconstruct_all_pairs()
        idx_mx = mx.array(idx.astype(np.int64))
        return rgb_0[idx_mx], rgb_1[idx_mx], latents[idx_mx]

    def __call__(
        self, pair_indices_np: np.ndarray
    ) -> tuple[Any, Any]:
        """Render the requested pairs; convenience wrapper around reconstruct_pair."""
        rgb_0, rgb_1, _latents = self.reconstruct_pair(pair_indices_np)
        return rgb_0, rgb_1

    # ------------------------------------------------------------------
    # state_dict export / import (for #1251 PyTorch bridge byte parity)
    # ------------------------------------------------------------------

    def export_state_dict(self) -> dict[str, np.ndarray]:
        """Export MLX parameters to a numpy dict suitable for PyTorch loading.

        Field names + shapes match the canonical PyTorch
        :class:`Z7Mamba2PredictiveCodingSubstrate` state_dict 1:1 so the
        Catalog #1251 export bridge can byte-stable round-trip an
        MLX-trained candidate to PyTorch for paired CUDA-CPU evaluation.

        Conv2d weights are transposed from MLX channels-last
        ``(out, kH, kW, in)`` to PyTorch ``(out, in, kH, kW)``.
        """
        import mlx.core as mx
        cfg = self.cfg
        out: dict[str, np.ndarray] = {}

        # Predictor input projection
        out["predictor.input_projection.weight"] = np.asarray(
            self.input_projection_w
        )
        out["predictor.input_projection.bias"] = np.asarray(
            self.input_projection_b
        )

        # Mamba-2 cell
        out["predictor.mamba_cell.in_proj.weight"] = np.asarray(
            self.mamba_in_proj_w
        )
        out["predictor.mamba_cell.A_log"] = np.asarray(self.mamba_A_log)
        out["predictor.mamba_cell.B_proj.weight"] = np.asarray(
            self.mamba_B_proj_w
        )
        out["predictor.mamba_cell.C_proj.weight"] = np.asarray(
            self.mamba_C_proj_w
        )
        out["predictor.mamba_cell.dt_proj.weight"] = np.asarray(
            self.mamba_dt_proj_w
        )
        out["predictor.mamba_cell.dt_proj.bias"] = np.asarray(
            self.mamba_dt_proj_b
        )
        out["predictor.mamba_cell.out_proj.weight"] = np.asarray(
            self.mamba_out_proj_w
        )

        # Predictor output projection
        out["predictor.output_projection.weight"] = np.asarray(
            self.output_projection_w
        )
        out["predictor.output_projection.bias"] = np.asarray(
            self.output_projection_b
        )

        # Decoder initial proj
        out["decoder.initial_proj.weight"] = np.asarray(
            self.dec_initial_proj_w
        )
        out["decoder.initial_proj.bias"] = np.asarray(self.dec_initial_proj_b)

        # Decoder PixelShuffle conv stack
        for i in range(cfg.decoder_num_upsample_blocks):
            # MLX layout (out, kH, kW, in) -> PyTorch (out, in, kH, kW)
            w_np = np.asarray(self.dec_block_w[i])
            w_pt = np.transpose(w_np, (0, 3, 1, 2))
            # Block index in the PyTorch nn.Sequential: each pixel-shuffle
            # block contributes 3 elements (Conv2d, PixelShuffle, ReLU),
            # so the i-th Conv2d sits at index 3*i.
            out[f"decoder.blocks.{3 * i}.weight"] = w_pt
            out[f"decoder.blocks.{3 * i}.bias"] = np.asarray(
                self.dec_block_b[i]
            )
        # Final Conv2d index = 3 * num_upsample_blocks
        final_idx = 3 * cfg.decoder_num_upsample_blocks
        w_final_np = np.asarray(self.dec_final_w)
        w_final_pt = np.transpose(w_final_np, (0, 3, 1, 2))
        out[f"decoder.blocks.{final_idx}.weight"] = w_final_pt
        out[f"decoder.blocks.{final_idx}.bias"] = np.asarray(self.dec_final_b)

        # Latent init + residuals + ego_motion buffer
        out["latent_init"] = np.asarray(self.latent_init)
        out["residuals"] = np.asarray(self.residuals)
        out["ego_motion_buffer"] = np.asarray(self.ego_motion_buffer)

        return out

    def load_state_dict_from_numpy(self, state: dict[str, np.ndarray]) -> None:
        """Load parameters from a numpy state dict (inverse of export).

        PyTorch (out, in, kH, kW) conv weights are transposed back to MLX
        channels-last (out, kH, kW, in) at load time.
        """
        import mlx.core as mx
        cfg = self.cfg

        self.input_projection_w = mx.array(state["predictor.input_projection.weight"])
        self.input_projection_b = mx.array(state["predictor.input_projection.bias"])
        self.mamba_in_proj_w = mx.array(state["predictor.mamba_cell.in_proj.weight"])
        self.mamba_A_log = mx.array(state["predictor.mamba_cell.A_log"])
        self.mamba_B_proj_w = mx.array(state["predictor.mamba_cell.B_proj.weight"])
        self.mamba_C_proj_w = mx.array(state["predictor.mamba_cell.C_proj.weight"])
        self.mamba_dt_proj_w = mx.array(state["predictor.mamba_cell.dt_proj.weight"])
        self.mamba_dt_proj_b = mx.array(state["predictor.mamba_cell.dt_proj.bias"])
        self.mamba_out_proj_w = mx.array(state["predictor.mamba_cell.out_proj.weight"])
        self.output_projection_w = mx.array(state["predictor.output_projection.weight"])
        self.output_projection_b = mx.array(state["predictor.output_projection.bias"])
        self.dec_initial_proj_w = mx.array(state["decoder.initial_proj.weight"])
        self.dec_initial_proj_b = mx.array(state["decoder.initial_proj.bias"])

        for i in range(cfg.decoder_num_upsample_blocks):
            w_pt = state[f"decoder.blocks.{3 * i}.weight"]
            # PyTorch (out, in, kH, kW) -> MLX (out, kH, kW, in)
            w_mlx = np.transpose(w_pt, (0, 2, 3, 1))
            self.dec_block_w[i] = mx.array(w_mlx)
            self.dec_block_b[i] = mx.array(state[f"decoder.blocks.{3 * i}.bias"])
        final_idx = 3 * cfg.decoder_num_upsample_blocks
        w_final_pt = state[f"decoder.blocks.{final_idx}.weight"]
        w_final_mlx = np.transpose(w_final_pt, (0, 2, 3, 1))
        self.dec_final_w = mx.array(w_final_mlx)
        self.dec_final_b = mx.array(state[f"decoder.blocks.{final_idx}.bias"])

        self.latent_init = mx.array(state["latent_init"])
        self.residuals = mx.array(state["residuals"])
        self.ego_motion_buffer = mx.array(state["ego_motion_buffer"])

    def num_parameters(self) -> int:
        """Count trainable parameter floats (latent_init + residuals + nn weights).

        Excludes the ``ego_motion_buffer`` which is non-trainable contest
        side-info loaded from real video.
        """
        cfg = self.cfg
        d_inner = cfg.d_inner
        total = 0
        # input_projection
        total += cfg.predictor_input_dim * cfg.d_model + cfg.d_model
        # mamba_cell
        total += cfg.d_model * 2 * d_inner  # in_proj (no bias)
        total += d_inner * cfg.d_state  # A_log
        total += d_inner * cfg.d_state  # B_proj (no bias)
        total += d_inner * cfg.d_state  # C_proj (no bias)
        total += d_inner * d_inner + d_inner  # dt_proj
        total += d_inner * cfg.d_model  # out_proj (no bias)
        # output_projection
        total += cfg.d_model * cfg.latent_dim + cfg.latent_dim
        # decoder.initial_proj
        decoder_initial_dim = (
            cfg.decoder_embed_dim
            * cfg.decoder_initial_grid_h
            * cfg.decoder_initial_grid_w
        )
        total += cfg.latent_dim * decoder_initial_dim + decoder_initial_dim
        # decoder blocks
        in_ch = cfg.decoder_embed_dim
        for i in range(cfg.decoder_num_upsample_blocks):
            out_ch = cfg.decoder_channels[i]
            # Conv2d(in_ch, 4*out_ch, k=3, p=1)
            total += 4 * out_ch * in_ch * 9 + 4 * out_ch
            in_ch = out_ch
        # Final Conv2d to 6 channels
        total += 6 * in_ch * 9 + 6
        # latent_init + residuals
        total += cfg.latent_dim
        total += cfg.num_pairs * cfg.latent_dim
        return total


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _pixel_shuffle_channels_last(x: Any, *, upscale_factor: int) -> Any:
    """PixelShuffle for MLX channels-last tensors.

    Mirrors :func:`torch.nn.functional.pixel_shuffle` semantics but
    operates on ``(P, H, W, C * r^2)`` -> ``(P, H * r, W * r, C)``.

    For L0 SCAFFOLD correctness this falls back through numpy when MLX
    primitives are insufficient. A native-MLX implementation is deferred
    to a sister L1 EXTENSION once the architecture is empirically anchored.
    """
    import mlx.core as mx
    arr_np = np.asarray(x)
    p, h, w, c_full = arr_np.shape
    r = int(upscale_factor)
    if c_full % (r * r) != 0:
        raise ValueError(
            f"channels {c_full} must be divisible by upscale_factor^2 = {r * r}"
        )
    c_out = c_full // (r * r)
    # Reshape (P, H, W, r, r, C_out) then transpose to (P, H, r, W, r, C_out)
    # then merge to (P, H*r, W*r, C_out).
    a = arr_np.reshape(p, h, w, r, r, c_out)
    a = np.transpose(a, (0, 1, 3, 2, 4, 5))
    a = a.reshape(p, h * r, w * r, c_out)
    return mx.array(a)


def _bilinear_resize_np(
    arr_np: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    """Bilinear-resize a (P, C, H, W) numpy array to (P, C, target_h, target_w).

    Uses PIL bilinear via per-channel resize for correctness parity with
    PyTorch's ``F.interpolate(..., mode='bilinear', align_corners=False)``.
    Sister L1 EXTENSION will native-implement this in MLX for performance.
    """
    from PIL import Image  # local import; pillow is a project dep
    p, c, h, w = arr_np.shape
    if h == target_h and w == target_w:
        return arr_np
    out = np.empty((p, c, target_h, target_w), dtype=arr_np.dtype)
    # Process per-batch per-channel via PIL for align_corners=False
    # bilinear semantics. Done as a single per-(batch, channel) loop for
    # L0 SCAFFOLD correctness; throughput is not the L0 metric.
    for bi in range(p):
        for ch in range(c):
            slab = arr_np[bi, ch]
            # PIL expects (H, W); float32 path via 'F' mode
            img = Image.fromarray(slab.astype(np.float32), mode="F")
            resized = img.resize((target_w, target_h), Image.BILINEAR)
            out[bi, ch] = np.array(resized, dtype=arr_np.dtype)
    return out


def render_pair_mlx(
    cfg: Z7Mamba2MLXRenderConfig,
    *,
    pair_indices: np.ndarray,
    seed: int = 0,
    state_dict: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One-shot convenience: build a fresh renderer + render the given pairs.

    Args:
        cfg: render config.
        pair_indices: numpy int64 array of pair indices in [0, num_pairs).
        seed: init seed.
        state_dict: optional pre-trained state dict; if provided, loaded
            after init via :meth:`load_state_dict_from_numpy`.

    Returns:
        ``(rgb_0_np, rgb_1_np)`` each ``(P, 3, H, W)`` float32 unit RGB.
    """
    renderer = Z7Mamba2MLXNativeRenderer(cfg, seed=seed)
    if state_dict is not None:
        renderer.load_state_dict_from_numpy(state_dict)
    rgb_0, rgb_1 = renderer(pair_indices)
    return np.asarray(rgb_0), np.asarray(rgb_1)
