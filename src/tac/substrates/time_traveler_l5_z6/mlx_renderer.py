# SPDX-License-Identifier: MIT
"""Z6 MLX-native predictive-coding renderer — L0 SCAFFOLD for MLX-local iteration.

Per OVERNIGHT Path 3 candidate #D (operator 2026-05-26 cascade: #1251 MLX→PyTorch
export bridge + #1257 inflate parity closure + #1258 corrected
``|S_MLX − S_PyTorch| = 0.000011`` empirical anchor + #1265 canonical PASS/FAIL
gate) — MLX-local iteration is now CONTEST-GRADE at frontier-tightening
granularity, unblocking the Z6 predictive-coding substrate's previously-paid-Modal-
blocked cascade per #768 / Phase 2 sextet #766 / Phase 3 sextet #839 / Wave 2
BUILD #837 + #847 + #857 driver fix #849.

Purpose
-------

Re-implement the Z6 PyTorch architecture (``architecture.py``) in MLX so the
operator can iterate on the FiLM-conditioned next-frame predictor + encoder +
decoder + autoregressive ``reconstruct_pair`` recurrence at $0 on Apple Silicon
BEFORE any paid CUDA dispatch. The MLX-trained weights export back to PyTorch
state_dict via :mod:`tac.local_acceleration.mlx_to_pytorch_export` (canonical
#1251 bridge); the exported PyTorch substrate then packs the Z6PCWM1 archive via
:mod:`tac.substrates.time_traveler_l5_z6.archive.pack_archive` and the gate
validates contest-equivalence via :mod:`tools.gate_mlx_candidate_contest_equivalence`.

Critical PyTorch-parity invariants honored
------------------------------------------

The MLX renderer mirrors the PyTorch ``Z6PredictiveCodingSubstrate`` EXACTLY so
the exported state_dict loads byte-stably:

- **Layer names match**: ``encoder.stem`` / ``encoder.head_mu`` /
  ``encoder.head_logvar`` / ``decoder.initial_proj`` / ``decoder.blocks.<i>`` /
  ``predictor.film_mlp.<i>`` / ``predictor.input_conv`` / ``predictor.output_conv``.
- **Weight layout matches PyTorch**: Conv2d weights stored as
  ``(out_channels, in_channels, kH, kW)``; Linear weights as
  ``(out_features, in_features)``. MLX internally uses NHWC + HWIO layout but
  this module returns numpy arrays in PyTorch layout via :meth:`export_state_dict`.
- **Forward semantics match**: same nonlinearities (SiLU on FiLM MLP, tanh on
  predictor output, ReLU on decoder blocks, sigmoid on decoder output), same
  bilinear interpolation policy (align_corners=False), same PixelShuffle factor
  (2x), same sigmoid → split-by-3 → (rgb_0, rgb_1) decoder head.

Non-promotable canonical contract
---------------------------------

Per CLAUDE.md "MLX portable-local-substrate authority":

- All MLX outputs are tagged ``[macOS-MLX research-signal]``.
- ``score_claim=False``, ``promotion_eligible=False``,
  ``ready_for_exact_eval_dispatch=False`` for every artifact this module produces.
- Promotion path: MLX state_dict → PyTorch via #1251 → Z6PCWM1 archive via
  canonical :mod:`tac.substrates.time_traveler_l5_z6.archive.pack_archive` →
  #1265 contest-equivalence gate → operator routes paid CUDA dispatch via
  ``tools/operator_authorize.py``.

L0 SCAFFOLD scope (BOUNDED per operator directive)
--------------------------------------------------

- Single-substrate Z6 only (Z7 / Z8 deferred to sister subagents).
- Single-layer FiLM predictor (``Z6PredictiveCodingConfig.predictor_depth=1``);
  multi-layer ``MultiLayerFilmPredictor`` (Wave 2 BUILD pattern) is intentionally
  deferred — the L0 SCAFFOLD validates the simplest-viable predictive-coding
  architecture first per the design memo Section 4.1.
- No identity-predictor ablation in this scaffold (PyTorch architecture
  supports it; MLX scaffold mirrors but doesn't add the disambiguator probe
  yet — operator routes via existing PyTorch trainer for that).
- Bounded ~75K params per the Z6 design memo Section 10 archive byte target.

Catalog #311 ego-motion conditioning
------------------------------------

Per CLAUDE.md "Predictive coding substrate design has ego-motion conditioning":
the MLX predictor accepts an ``ego_motion`` tensor and FiLM-modulates the
predictor conv on it. This satisfies the canonical requirement that Atick-
Redlich cooperative-receiver framing be paired with ego-motion-conditioned
next-frame prediction.

Cross-references
----------------

- Canonical sister PyTorch architecture:
  :mod:`tac.substrates.time_traveler_l5_z6.architecture`
- Canonical MLX HNeRV reference pattern:
  :mod:`tac.local_acceleration.pr95_hnerv_mlx`
- Canonical MLX→PyTorch export bridge (#1251):
  :mod:`tac.local_acceleration.mlx_to_pytorch_export`
- Canonical contest-equivalence gate (#1265):
  ``tools/gate_mlx_candidate_contest_equivalence.py``
- Empirical MLX-contest-grade anchor (#1258 corrected 2026-05-26):
  ``.omx/research/pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526.md``
- Z6 design memo:
  ``.omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.substrates.time_traveler_l5_z6.architecture import (
    EVAL_HW,
    NUM_PAIRS,
    Z6PredictiveCodingConfig,
)

try:  # pragma: no cover — exercised on Apple Silicon with MLX installed.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
except Exception as exc:  # pragma: no cover — import guard for non-Apple CI.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR: Exception | None = exc
else:
    _MLX_IMPORT_ERROR = None


SCHEMA_VERSION = "z6_predictive_coding_mlx_renderer_v1"
"""Canonical schema version for MLX state_dict export manifests."""

LANE_ID = "lane_z6_predictive_coding_mlx_scaffold_20260526"
"""Canonical lane id for #325 sister-symposium chain + Catalog #126 pre-registration."""

EVIDENCE_GRADE = "macOS-MLX research-signal"
"""Per CLAUDE.md 'MLX portable-local-substrate authority' non-promotable tag."""

EVIDENCE_TAG = "[macOS-MLX research-signal]"
"""Per CLAUDE.md axis-tag-format non-promotable label."""


def require_mlx() -> None:
    """Raise an explicit error when MLX is unavailable (e.g. non-Apple CI)."""
    if mx is None or nn is None:
        raise RuntimeError(
            "MLX is not installed; the Z6 MLX-local renderer requires Apple "
            "Silicon + the `mlx` package. Install via `pip install mlx`. "
            f"Original import error: {_MLX_IMPORT_ERROR!r}"
        )


# ---------------------------------------------------------------------------
# MLX building blocks — mirror the PyTorch architecture EXACTLY.
# ---------------------------------------------------------------------------


class _Z6FiLMConditionedNextFramePredictorMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX mirror of ``FilmConditionedNextFramePredictor``.

    Layer naming and forward semantics MUST match PyTorch sister so the exported
    state_dict loads byte-stably via the canonical Z6 ``architecture.py``.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        film_mlp_hidden_dim: int,
        ego_motion_dim: int,
        kernel_size: int = 3,
    ) -> None:
        require_mlx()
        super().__init__()
        if kernel_size not in (1, 3, 5):
            raise ValueError(
                f"predictor_kernel_size must be 1, 3, or 5; got {kernel_size}"
            )
        if kernel_size % 2 == 0:
            raise ValueError(
                f"predictor_kernel_size must be odd for symmetric padding; "
                f"got {kernel_size}"
            )
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.film_mlp_hidden_dim = int(film_mlp_hidden_dim)
        self.ego_motion_dim = int(ego_motion_dim)
        self.kernel_size = int(kernel_size)

        # PyTorch parity: `film_mlp` is nn.Sequential of (Linear, SiLU, Linear).
        # Stored as a list-of-modules so the state_dict export uses dotted
        # names film_mlp.0 and film_mlp.2 matching torch.nn.Sequential indices.
        self.film_mlp_0 = nn.Linear(ego_motion_dim, film_mlp_hidden_dim)  # type: ignore[union-attr]
        # film_mlp.1 is the SiLU activation (no parameters)
        self.film_mlp_2 = nn.Linear(film_mlp_hidden_dim, hidden_dim * 2)  # type: ignore[union-attr]
        self.input_conv = nn.Conv2d(  # type: ignore[union-attr]
            latent_dim, hidden_dim, kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.output_conv = nn.Conv2d(  # type: ignore[union-attr]
            hidden_dim, latent_dim, kernel_size=1, padding=0,
        )

    def __call__(self, z_prev: Any, ego_motion: Any) -> Any:
        """Predict z_t from z_{t-1} + ego_motion. Matches PyTorch forward semantics.

        Args:
            z_prev: ``(B, latent_dim)`` MLX array.
            ego_motion: ``(B, ego_motion_dim)`` MLX array.

        Returns:
            ``(B, latent_dim)`` MLX array predicted z_t.
        """
        if z_prev.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z_prev last dim {z_prev.shape[-1]} != latent_dim {self.latent_dim}"
            )
        if ego_motion.shape[-1] != self.ego_motion_dim:
            raise ValueError(
                f"ego_motion last dim {ego_motion.shape[-1]} != ego_motion_dim "
                f"{self.ego_motion_dim}"
            )
        batch = int(z_prev.shape[0])
        # FiLM MLP: Linear → SiLU → Linear (matches torch.nn.Sequential)
        film_x = self.film_mlp_0(ego_motion)
        film_x = nn.silu(film_x)  # type: ignore[union-attr]
        film_params = self.film_mlp_2(film_x)  # (B, 2*hidden_dim)
        # PyTorch chunk(2, dim=-1) splits into [scale | shift] along last axis.
        scale = film_params[:, : self.hidden_dim]
        shift = film_params[:, self.hidden_dim :]
        scale = mx.reshape(scale, (batch, 1, 1, self.hidden_dim))  # type: ignore[union-attr]
        shift = mx.reshape(shift, (batch, 1, 1, self.hidden_dim))  # type: ignore[union-attr]

        # Project latent (B, latent_dim) -> spatial (B, 1, 1, latent_dim) NHWC
        # PyTorch is (B, latent_dim, 1, 1) NCHW; MLX Conv2d expects NHWC.
        z_prev_spatial = mx.reshape(z_prev, (batch, 1, 1, self.latent_dim))  # type: ignore[union-attr]
        # Single-layer conv (predictor backbone)
        h = self.input_conv(z_prev_spatial)  # (B, 1, 1, hidden_dim)
        # FiLM modulation: h * scale + shift (broadcast)
        h = h * scale + shift
        h = mx.tanh(h)  # type: ignore[union-attr]
        # Project hidden -> latent via 1x1 conv
        z_pred_spatial = self.output_conv(h)  # (B, 1, 1, latent_dim)
        return mx.reshape(z_pred_spatial, (batch, self.latent_dim))  # type: ignore[union-attr]


class _Z6EncoderMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX mirror of `_Z6Encoder` (small CNN -> latent_dim projection)."""

    def __init__(self, *, input_channels: int, hidden_dim: int, latent_dim: int) -> None:
        require_mlx()
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.stem = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)  # type: ignore[union-attr]
        self.head_mu = nn.Linear(hidden_dim, latent_dim)  # type: ignore[union-attr]
        self.head_logvar = nn.Linear(hidden_dim, latent_dim)  # type: ignore[union-attr]

    def __call__(self, frames_nhwc: Any) -> tuple[Any, Any]:
        """Encode frames -> (mu, logvar). Input is NHWC for MLX Conv2d."""
        if len(frames_nhwc.shape) != 4:
            raise ValueError(
                f"encoder expects (B, H, W, C) NHWC; got shape {tuple(frames_nhwc.shape)}"
            )
        feats = self.stem(frames_nhwc)
        # Global average pool over H, W (axes 1, 2 in NHWC)
        pooled = mx.mean(feats, axis=(1, 2))  # type: ignore[union-attr]
        return self.head_mu(pooled), self.head_logvar(pooled)


class _Z6DecoderMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX mirror of `_Z6Decoder` (PixelShuffle NeRV-style with bilinear final resize)."""

    def __init__(
        self,
        *,
        latent_dim: int,
        embed_dim: int,
        initial_grid_h: int,
        initial_grid_w: int,
        decoder_channels: tuple[int, ...],
        num_upsample_blocks: int,
        output_height: int,
        output_width: int,
    ) -> None:
        require_mlx()
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.embed_dim = int(embed_dim)
        self.initial_grid_h = int(initial_grid_h)
        self.initial_grid_w = int(initial_grid_w)
        self.decoder_channels = tuple(int(c) for c in decoder_channels)
        self.num_upsample_blocks = int(num_upsample_blocks)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        if len(self.decoder_channels) < self.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels must have >= num_upsample_blocks entries; "
                f"got {len(self.decoder_channels)} for {self.num_upsample_blocks} blocks"
            )
        # PyTorch parity: initial_proj is Linear(latent_dim, embed_dim*H*W).
        self.initial_proj = nn.Linear(  # type: ignore[union-attr]
            latent_dim, embed_dim * initial_grid_h * initial_grid_w,
        )
        # Build upsample blocks. PyTorch uses Sequential of
        # (Conv2d, PixelShuffle, ReLU, Conv2d, PixelShuffle, ReLU, ...).
        # MLX state_dict export uses dotted names blocks.<3i>, blocks.<3i+2>
        # for the conv weights (PixelShuffle + ReLU have no params).
        # We store as a flat list of conv layers (3*num_upsample_blocks + 1 final).
        self._block_conv_indices: list[int] = []
        in_ch = embed_dim
        block_idx = 0
        self._block_convs: list[Any] = []
        for i in range(self.num_upsample_blocks):
            out_ch = self.decoder_channels[i]
            conv = nn.Conv2d(in_ch, 4 * out_ch, kernel_size=3, padding=1)  # type: ignore[union-attr]
            # Each PyTorch upsample block is 3 modules: Conv2d, PixelShuffle, ReLU.
            # The state_dict prefix is therefore "blocks.<block_idx>." where
            # block_idx = 3*i (the Conv2d slot).
            setattr(self, f"_block_conv_{block_idx}", conv)
            self._block_convs.append(conv)
            self._block_conv_indices.append(block_idx)
            block_idx += 3  # advance past PixelShuffle (block_idx+1) and ReLU (block_idx+2)
            in_ch = out_ch
        # Final conv after the upsample stack: PyTorch's
        # Sequential.append(Conv2d(in_ch, 6, kernel_size=3, padding=1))
        # gets index 3*num_upsample_blocks.
        self._final_conv_index = block_idx  # = 3 * num_upsample_blocks
        self._final_conv = nn.Conv2d(in_ch, 6, kernel_size=3, padding=1)  # type: ignore[union-attr]
        setattr(self, f"_block_conv_{self._final_conv_index}", self._final_conv)

    def __call__(self, z: Any) -> tuple[Any, Any]:
        """Decode latent z -> (rgb_0, rgb_1) NHWC unit-range MLX tensors."""
        if len(z.shape) != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"decoder expects (B, latent_dim={self.latent_dim}); got {tuple(z.shape)}"
            )
        batch = int(z.shape[0])
        flat = self.initial_proj(z)
        # PyTorch reshape: (B, embed_dim, H, W); MLX needs NHWC = (B, H, W, embed_dim).
        # The PyTorch Linear's output is naturally (B, embed_dim * H * W) so we
        # reshape and transpose to NHWC. To preserve PyTorch parity, the
        # initial reshape interprets the flat tensor as (B, embed_dim, H, W)
        # then transposes to NHWC.
        grid_nchw = mx.reshape(  # type: ignore[union-attr]
            flat, (batch, self.embed_dim, self.initial_grid_h, self.initial_grid_w),
        )
        grid_nhwc = mx.transpose(grid_nchw, (0, 2, 3, 1))  # type: ignore[union-attr]
        x = grid_nhwc
        for conv in self._block_convs:
            x = conv(x)
            # PixelShuffle 2x in NHWC
            x = _pixel_shuffle_2x_nhwc(x)
            # ReLU
            x = nn.relu(x)  # type: ignore[union-attr]
        x = self._final_conv(x)
        # If output size differs from target, bilinear interpolate
        if int(x.shape[1]) != self.output_height or int(x.shape[2]) != self.output_width:
            x = _bilinear_resize_nhwc(
                x, target_h=self.output_height, target_w=self.output_width,
            )
        # Sigmoid + split-by-3 into rgb_0, rgb_1
        x = mx.sigmoid(x)  # type: ignore[union-attr]
        rgb_0 = x[:, :, :, :3]
        rgb_1 = x[:, :, :, 3:6]
        return rgb_0, rgb_1


def _pixel_shuffle_2x_nhwc(x: Any) -> Any:
    """PixelShuffle factor 2 for NHWC tensor (matches PyTorch nn.PixelShuffle(2))."""
    require_mlx()
    if len(x.shape) != 4:
        raise ValueError(f"expected NHWC; got shape {x.shape}")
    batch, height, width, channels = (int(d) for d in x.shape)
    if channels % 4 != 0:
        raise ValueError(f"channels must be divisible by 4 for 2x pixel shuffle; got {channels}")
    out_channels = channels // 4
    y = mx.reshape(x, (batch, height, width, out_channels, 2, 2))  # type: ignore[union-attr]
    y = mx.transpose(y, (0, 1, 4, 2, 5, 3))  # type: ignore[union-attr]
    return mx.reshape(y, (batch, height * 2, width * 2, out_channels))  # type: ignore[union-attr]


def _bilinear_resize_nhwc(x: Any, *, target_h: int, target_w: int) -> Any:
    """Simple bilinear resize for NHWC tensor (align_corners=False matches PyTorch)."""
    require_mlx()
    if len(x.shape) != 4:
        raise ValueError(f"expected NHWC; got shape {x.shape}")
    batch, h_in, w_in, c = (int(d) for d in x.shape)
    if h_in == target_h and w_in == target_w:
        return x
    # Compute interpolation indices (align_corners=False)
    h_scale = h_in / target_h
    w_scale = w_in / target_w
    h_idx_f = (mx.arange(target_h, dtype=mx.float32) + 0.5) * h_scale - 0.5  # type: ignore[union-attr]
    w_idx_f = (mx.arange(target_w, dtype=mx.float32) + 0.5) * w_scale - 0.5  # type: ignore[union-attr]
    h_idx_f = mx.clip(h_idx_f, 0.0, h_in - 1.0)  # type: ignore[union-attr]
    w_idx_f = mx.clip(w_idx_f, 0.0, w_in - 1.0)  # type: ignore[union-attr]
    h_lo = mx.floor(h_idx_f).astype(mx.int32)  # type: ignore[union-attr]
    h_hi = mx.minimum(h_lo + 1, h_in - 1)  # type: ignore[union-attr]
    w_lo = mx.floor(w_idx_f).astype(mx.int32)  # type: ignore[union-attr]
    w_hi = mx.minimum(w_lo + 1, w_in - 1)  # type: ignore[union-attr]
    h_frac = h_idx_f - mx.floor(h_idx_f)  # type: ignore[union-attr]
    w_frac = w_idx_f - mx.floor(w_idx_f)  # type: ignore[union-attr]
    # Gather neighbors via fancy indexing
    h_lo_b = h_lo[:, None]  # (Ht, 1)
    h_hi_b = h_hi[:, None]
    w_lo_b = w_lo[None, :]  # (1, Wt)
    w_hi_b = w_hi[None, :]
    tl = x[:, h_lo_b, w_lo_b, :]  # (B, Ht, Wt, C)
    tr = x[:, h_lo_b, w_hi_b, :]
    bl = x[:, h_hi_b, w_lo_b, :]
    br = x[:, h_hi_b, w_hi_b, :]
    h_frac_b = mx.reshape(h_frac, (1, target_h, 1, 1))  # type: ignore[union-attr]
    w_frac_b = mx.reshape(w_frac, (1, 1, target_w, 1))  # type: ignore[union-attr]
    top = tl * (1.0 - w_frac_b) + tr * w_frac_b
    bot = bl * (1.0 - w_frac_b) + br * w_frac_b
    return top * (1.0 - h_frac_b) + bot * h_frac_b


# ---------------------------------------------------------------------------
# Z6 MLX substrate composition + autoregressive recurrence.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Z6MLXRendererStateDictManifest:
    """Manifest describing the MLX state_dict (post-export) for canonical Provenance.

    Per Catalog #287/#323 every score-claim row carries axis+hardware+evidence_grade.
    """

    schema_version: str
    substrate_id: str
    latent_dim: int
    num_pairs: int
    ego_motion_dim: int
    encoder_param_count: int
    decoder_param_count: int
    predictor_param_count: int
    total_param_count: int
    evidence_grade: str
    score_claim: bool
    promotion_eligible: bool
    ready_for_exact_eval_dispatch: bool


class Z6PredictiveCodingMLXRenderer(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Z6 MLX-native predictive-coding renderer.

    Mirrors :class:`tac.substrates.time_traveler_l5_z6.architecture.Z6PredictiveCodingSubstrate`
    exactly so the exported state_dict loads via the canonical PyTorch
    architecture without modification. The exported state_dict + auxiliary
    latent_init / residuals / ego_motion buffers feed
    :func:`tac.substrates.time_traveler_l5_z6.archive.pack_archive` to produce
    a contest Z6PCWM1 archive.

    Composition (PyTorch parity):
    - encoder: small CNN -> (mu, logvar) via mean-pool over spatial dims
    - predictor: FiLM-conditioned single-layer conv block
    - decoder: PixelShuffle NeRV-style with sigmoid + split-by-3 output head
    - latent_init: trainable z_0 (latent_dim,)
    - residuals: trainable per-pair residuals (num_pairs, latent_dim)
    - ego_motion_buffer: fixed per-pair ego-motion (num_pairs, ego_motion_dim)

    Training loop (caller responsibility):
        For pair index t = 1..num_pairs - 1:
            z_t_pred = predictor(z_{t-1}, ego_motion[t])
            z_t = z_t_pred + residuals[t]
            rgb_0, rgb_1 = decoder(z_t)
            loss += score_aware(rgb_0, rgb_1, gt_rgb_0, gt_rgb_1)
                  + lambda_residual * (residuals[t] ** 2).mean()
        # z_0 = latent_init, decoded as rgb_0/rgb_1 at pair t=0

    Inflate-time: same recurrence with state_dict loaded from Z6PCWM1 archive.
    """

    def __init__(self, cfg: Z6PredictiveCodingConfig) -> None:
        require_mlx()
        super().__init__()
        if cfg.predictor_depth > 1:
            raise NotImplementedError(
                "L0 SCAFFOLD: only predictor_depth=1 (FilmConditionedNextFramePredictor) "
                "is implemented in MLX. Multi-layer FiLM (Wave 2 BUILD pattern) is "
                "deferred per the operator's bounded L0 SCAFFOLD scope. Use the "
                "PyTorch sister substrate for depth>=2 iteration."
            )
        if cfg.identity_predictor:
            raise NotImplementedError(
                "L0 SCAFFOLD: identity_predictor=True ablation is not implemented "
                "in MLX. Use the PyTorch sister substrate's "
                "Z6PredictiveCodingSubstrate for the disambiguator probe (Catalog "
                "#125 hook #6) until the MLX renderer is promoted to L1."
            )
        self.cfg = cfg
        self.encoder = _Z6EncoderMLX(
            input_channels=cfg.encoder_input_channels,
            hidden_dim=cfg.encoder_hidden_dim,
            latent_dim=cfg.latent_dim,
        )
        self.decoder = _Z6DecoderMLX(
            latent_dim=cfg.latent_dim,
            embed_dim=cfg.decoder_embed_dim,
            initial_grid_h=cfg.decoder_initial_grid_h,
            initial_grid_w=cfg.decoder_initial_grid_w,
            decoder_channels=cfg.decoder_channels,
            num_upsample_blocks=cfg.decoder_num_upsample_blocks,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
        )
        # Per-pair trainable state. PyTorch uses nn.Parameter; MLX uses
        # plain attribute storage that participates in `tree_flatten`.
        # Initialize with the same std as PyTorch sister so parity tests
        # land near identity with random seed pinned.
        # Note: MLX module attributes that are arrays become parameters
        # automatically when iterated via tree_flatten / .parameters().
        latent_init_arr = (
            np.random.RandomState(0).randn(cfg.latent_dim).astype(np.float32)
            * cfg.latent_init_std
        )
        residuals_arr = (
            np.random.RandomState(1)
            .randn(cfg.num_pairs, cfg.latent_dim)
            .astype(np.float32)
            * cfg.latent_init_std
        )
        # Ego-motion is a fixed buffer (not trainable). Initialize to zeros.
        ego_motion_arr = np.zeros(
            (cfg.num_pairs, cfg.predictor_ego_motion_dim), dtype=np.float32,
        )
        self.latent_init = mx.array(latent_init_arr)  # type: ignore[union-attr]
        self.residuals = mx.array(residuals_arr)  # type: ignore[union-attr]
        self.ego_motion_buffer = mx.array(ego_motion_arr)  # type: ignore[union-attr]
        self.predictor = _Z6FiLMConditionedNextFramePredictorMLX(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.predictor_hidden_dim,
            film_mlp_hidden_dim=cfg.predictor_film_mlp_hidden_dim,
            ego_motion_dim=cfg.predictor_ego_motion_dim,
            kernel_size=cfg.predictor_kernel_size,
        )

    def reconstruct_pair(self, pair_indices: Any) -> tuple[Any, Any, Any]:
        """Autoregressive recurrence + decode for requested pair indices.

        Mirrors PyTorch ``Z6PredictiveCodingSubstrate.reconstruct_pair``: roll
        the predictor from t=0 up to max(pair_indices) then select z_t at the
        requested indices and decode.

        Args:
            pair_indices: ``(B,)`` MLX int32 array in [0, num_pairs).

        Returns:
            (rgb_0, rgb_1, z_at_indices) MLX NHWC + (B, latent_dim).
        """
        require_mlx()
        # Coerce to numpy for range check, then back to mx
        idx_np = np.asarray(pair_indices)
        if idx_np.size == 0:
            raise ValueError("pair_indices must be non-empty")
        if int(idx_np.min()) < 0 or int(idx_np.max()) >= self.cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs}); "
                f"got [{int(idx_np.min())}, {int(idx_np.max())}]"
            )
        max_idx = int(idx_np.max())
        # z_history[t] is (latent_dim,) for each t in 0..max_idx
        z_history: list[Any] = [self.latent_init]
        for t in range(1, max_idx + 1):
            z_prev = mx.reshape(z_history[-1], (1, self.cfg.latent_dim))  # type: ignore[union-attr]
            ego_t = mx.reshape(  # type: ignore[union-attr]
                self.ego_motion_buffer[t], (1, self.cfg.predictor_ego_motion_dim),
            )
            z_pred = self.predictor(z_prev, ego_t)
            z_t = mx.reshape(z_pred, (self.cfg.latent_dim,)) + self.residuals[t]  # type: ignore[union-attr]
            z_history.append(z_t)
        # Stack into (max_idx+1, latent_dim)
        z_stack = mx.stack(z_history, axis=0)  # type: ignore[union-attr]
        # Gather requested rows
        z_at_indices = z_stack[pair_indices]  # (B, latent_dim)
        rgb_0, rgb_1 = self.decoder(z_at_indices)
        return rgb_0, rgb_1, z_at_indices

    def num_parameters_breakdown(self) -> dict[str, int]:
        """Return per-submodule parameter counts (encoder/decoder/predictor/latent_init/residuals)."""
        require_mlx()

        def _count(arr: Any) -> int:
            shape = tuple(int(d) for d in arr.shape)
            total = 1
            for d in shape:
                total *= d
            return int(total)

        def _sum_module(mod: Any) -> int:
            return sum(_count(a) for _, a in tree_flatten(mod.parameters()))  # type: ignore[misc]

        return {
            "encoder": _sum_module(self.encoder),
            "decoder": _sum_module(self.decoder),
            "predictor": _sum_module(self.predictor),
            "latent_init": _count(self.latent_init),
            "residuals": _count(self.residuals),
            "total": (
                _sum_module(self.encoder)
                + _sum_module(self.decoder)
                + _sum_module(self.predictor)
                + _count(self.latent_init)
                + _count(self.residuals)
            ),
        }

    def export_state_dict(self) -> dict[str, np.ndarray]:
        """Export MLX parameters as numpy arrays in PyTorch state_dict layout.

        Returns a dict keyed by PyTorch state_dict parameter name (e.g.
        ``encoder.stem.weight``, ``predictor.film_mlp.0.weight``,
        ``decoder.blocks.0.weight``). Conv2d weights are transposed from MLX's
        HWIO layout back to PyTorch's OIHW layout; Linear weights are in
        PyTorch's (out_features, in_features) order which matches MLX exactly.

        The latent_init / residuals / ego_motion buffers are NOT in the
        state_dict (they are stored separately in the Z6PCWM1 archive via
        :func:`tac.substrates.time_traveler_l5_z6.archive.pack_archive`).
        """
        require_mlx()
        out: dict[str, np.ndarray] = {}

        def _conv_to_pytorch_layout(arr_np: np.ndarray) -> np.ndarray:
            """MLX Conv2d weights are (out_C, kH, kW, in_C) HWIO; PyTorch is (out_C, in_C, kH, kW) OIHW."""
            if arr_np.ndim != 4:
                return arr_np
            return np.transpose(arr_np, (0, 3, 1, 2))

        # Encoder
        out["encoder.stem.weight"] = _conv_to_pytorch_layout(
            np.asarray(self.encoder.stem.weight)
        )
        out["encoder.stem.bias"] = np.asarray(self.encoder.stem.bias)
        out["encoder.head_mu.weight"] = np.asarray(self.encoder.head_mu.weight)
        out["encoder.head_mu.bias"] = np.asarray(self.encoder.head_mu.bias)
        out["encoder.head_logvar.weight"] = np.asarray(self.encoder.head_logvar.weight)
        out["encoder.head_logvar.bias"] = np.asarray(self.encoder.head_logvar.bias)
        # Predictor (PyTorch nn.Sequential indexing: film_mlp.0, film_mlp.2)
        out["predictor.film_mlp.0.weight"] = np.asarray(self.predictor.film_mlp_0.weight)
        out["predictor.film_mlp.0.bias"] = np.asarray(self.predictor.film_mlp_0.bias)
        out["predictor.film_mlp.2.weight"] = np.asarray(self.predictor.film_mlp_2.weight)
        out["predictor.film_mlp.2.bias"] = np.asarray(self.predictor.film_mlp_2.bias)
        out["predictor.input_conv.weight"] = _conv_to_pytorch_layout(
            np.asarray(self.predictor.input_conv.weight)
        )
        out["predictor.input_conv.bias"] = np.asarray(self.predictor.input_conv.bias)
        out["predictor.output_conv.weight"] = _conv_to_pytorch_layout(
            np.asarray(self.predictor.output_conv.weight)
        )
        out["predictor.output_conv.bias"] = np.asarray(self.predictor.output_conv.bias)
        # Decoder
        out["decoder.initial_proj.weight"] = np.asarray(self.decoder.initial_proj.weight)
        out["decoder.initial_proj.bias"] = np.asarray(self.decoder.initial_proj.bias)
        # Decoder blocks: each Conv2d at PyTorch index 3*i, final Conv2d at
        # 3*num_upsample_blocks. The Sequential indexing skips PixelShuffle
        # and ReLU which have no parameters.
        for i in range(self.decoder.num_upsample_blocks):
            pytorch_idx = 3 * i
            mlx_conv = getattr(self.decoder, f"_block_conv_{pytorch_idx}")
            out[f"decoder.blocks.{pytorch_idx}.weight"] = _conv_to_pytorch_layout(
                np.asarray(mlx_conv.weight)
            )
            out[f"decoder.blocks.{pytorch_idx}.bias"] = np.asarray(mlx_conv.bias)
        # Final conv
        final_idx = self.decoder._final_conv_index
        final_conv = getattr(self.decoder, f"_block_conv_{final_idx}")
        out[f"decoder.blocks.{final_idx}.weight"] = _conv_to_pytorch_layout(
            np.asarray(final_conv.weight)
        )
        out[f"decoder.blocks.{final_idx}.bias"] = np.asarray(final_conv.bias)
        return out

    def export_auxiliary_buffers(self) -> dict[str, np.ndarray]:
        """Export latent_init / residuals / ego_motion as numpy arrays.

        These feed :func:`tac.substrates.time_traveler_l5_z6.archive.pack_archive`
        as the per-pair codec state alongside the state_dict.
        """
        require_mlx()
        return {
            "latent_init": np.asarray(self.latent_init),
            "residuals": np.asarray(self.residuals),
            "ego_motion": np.asarray(self.ego_motion_buffer),
        }

    def export_state_dict_manifest(self) -> Z6MLXRendererStateDictManifest:
        """Build canonical Provenance manifest for the exported state_dict.

        Per Catalog #287/#323: every exported state_dict carries the canonical
        (axis_tag, evidence_grade, score_claim, promotion_eligible,
        ready_for_exact_eval_dispatch) triple making the non-promotable nature
        explicit.
        """
        breakdown = self.num_parameters_breakdown()
        return Z6MLXRendererStateDictManifest(
            schema_version=SCHEMA_VERSION,
            substrate_id="time_traveler_l5_z6",
            latent_dim=int(self.cfg.latent_dim),
            num_pairs=int(self.cfg.num_pairs),
            ego_motion_dim=int(self.cfg.predictor_ego_motion_dim),
            encoder_param_count=int(breakdown["encoder"]),
            decoder_param_count=int(breakdown["decoder"]),
            predictor_param_count=int(breakdown["predictor"]),
            total_param_count=int(breakdown["total"]),
            evidence_grade=EVIDENCE_GRADE,
            score_claim=False,
            promotion_eligible=False,
            ready_for_exact_eval_dispatch=False,
        )


__all__ = [
    "EVAL_HW",
    "EVIDENCE_GRADE",
    "EVIDENCE_TAG",
    "LANE_ID",
    "NUM_PAIRS",
    "SCHEMA_VERSION",
    "Z6MLXRendererStateDictManifest",
    "Z6PredictiveCodingConfig",
    "Z6PredictiveCodingMLXRenderer",
    "require_mlx",
]
