# SPDX-License-Identifier: MIT
"""PACT-NeRV-VQ MLX-native renderer — L1 LONG-RUN MLX-LOCAL 2026-05-28.

Per the operator NON-NEGOTIABLE 11th INDIVIDUALLY-FRACTAL standing directive
2026-05-27 + 8th MLX-first standing directive REINFORCED 2026-05-28 ("you can
fire everything and anything on MLX"): this is PACT-NeRV-VQ's OWN canonical
MLX engineering pass. The distinguishing primitive (van den Oord 2017 VQ-VAE
codebook + per-pair discrete index per arXiv:1711.00937 §3.1-3.2) is
implemented substrate-specifically — NOT a shared-helper shortcut from sister
PACT-NeRV cascade variants.

Selection rationale per ULTIMATE STAIRCASE Step 15 PRIORITY 1
=============================================================

Per the SELECTOR-V4 ORTHOGONAL-PIVOT verdict (commit ``f013736de``): the
SELECTOR-PARADIGM cascade (IA3 / V2 / V3 / V4) reached its 32-pair base-decoder
floor at 201-231x convergence ratio + 0.0014-0.0017 final score (stochastic-
seed + AdamW-noise band). PACT-NeRV-VQ provides the ORTHOGONAL paradigm:
DISCRETE TOKENS via VQ-VAE codebook + per-pair index instead of continuous
arithmetic coding selector. Per CLAUDE.md "Portfolio diversification" + Aaron
van den Oord inner council seat alignment.

Canonical PyTorch sister
========================

This module is a 1:1 architectural mirror of
:class:`tac.substrates.pact_nerv_vq.architecture.PactNervVqSubstrate` so the
MLX-trained weights export back to PyTorch state_dict via the canonical
MLX→PyTorch bridge (``tools/export_pact_nerv_vq_mlx_to_pytorch_state_dict.py``)
and the PyTorch substrate packs the PVQ archive via
:func:`tac.substrates.pact_nerv_vq.archive.pack_archive`.

PyTorch-parity invariants honored
---------------------------------

- **Layer names match** (state_dict-compatible): ``latents`` /
  ``quantizer.codebook`` / ``quantizer.ema_cluster_size`` / ``quantizer.ema_w``
  / ``latent_embed`` / ``blocks.<i>.dsc.{depthwise,pointwise}`` /
  ``head_rgb_0`` / ``head_rgb_1``.
- **Weight layout matches PyTorch** at export: Conv2d weights stored as
  ``(out_channels, in_channels, kH, kW)``; Linear weights as
  ``(out_features, in_features)``. MLX internally uses NHWC + HWIO layout
  but :meth:`export_state_dict` returns numpy arrays in PyTorch layout.
- **Forward semantics match**: sin() activation in ``_DsUpBlock``; bilinear
  resize with ``align_corners=False`` for final upscale; per-pair VQ codebook
  lookup + straight-through estimator + commitment loss; sigmoid on RGB heads.
- **VQ-VAE canonical**: nearest-codebook lookup + EMA codebook update (van
  den Oord §3.2) + commitment loss ``||z_e - sg(z_q)||^2`` (van den Oord §3.1).

Non-promotable canonical contract
=================================

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog
#127/#192/#317/#341: all artifacts produced by this MLX renderer are tagged
``[macOS-MLX research-signal]`` with:

- ``score_claim=False``
- ``promotion_eligible=False``
- ``ready_for_exact_eval_dispatch=False``

The canonical MLX harness
(:func:`tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main`)
auto-stamps these markers on the ``TrainingArtifact``.

Promotion path
==============

MLX state_dict → PyTorch via the canonical bridge → PVQ archive via the
canonical pack_archive → contest-equivalence gate via
:mod:`tools.gate_mlx_candidate_contest_equivalence_pact_nerv_vq` → operator
routes paid CUDA dispatch via ``tools/operator_authorize.py``.

INDIVIDUALLY-FRACTAL per UNIQUE-AND-COMPLETE-PER-METHOD
=======================================================

This MLX renderer is PACT-NeRV-VQ's OWN canonical engineering pass per the
11th INDIVIDUALLY-FRACTAL standing directive 2026-05-27. The distinguishing
primitive (van den Oord 2017 VQ-VAE codebook + per-pair discrete index) is
implemented substrate-specifically. The HNeRV-class base decoder topology
is ADOPT-CANONICAL per Catalog #290 + the empirically validated medal-class
topology (per HNeRV parity L7: the architecture distinguishing primitive is
VQ-VAE discrete tokens, NOT the decoder topology).

Cross-references
================

- Canonical sister PyTorch architecture:
  :mod:`tac.substrates.pact_nerv_vq.architecture`
- Canonical MLX HNeRV reference pattern (IA3 sister):
  :mod:`tac.substrates.pact_nerv_ia3.mlx_renderer`
- Canonical MLX→PyTorch export bridge:
  ``tools/export_pact_nerv_vq_mlx_to_pytorch_state_dict.py``
- Canonical MLX score-aware harness:
  :mod:`tac.substrates._shared.mlx_score_aware`
- ULTIMATE design memo:
  ``.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md``
- L0 SCAFFOLD design memo:
  ``.omx/research/pact_nerv_vq_l0_scaffold_design_20260520T211500Z.md``
- This landing memo:
  ``.omx/research/pact_nerv_vq_l1_long_run_mlx_landed_20260528.md``
"""
# AUTOCAST_FP16_WAIVED:MLX_substrate_does_not_use_PyTorch_CUDA_autocast_fp16_per_mlx_first_canonical_doctrine_8th_standing_directive
# TF32_WAIVED:MLX_substrate_does_not_use_PyTorch_CUDA_tf32_per_mlx_first_canonical_doctrine
# TORCH_COMPILE_WAIVED:MLX_substrate_uses_mlx_value_and_grad_not_torch_compile
# NO_GRAD_WAIVED:MLX_substrate_uses_mlx_lazy_eval_not_pytorch_no_grad_per_mlx_first_canonical_doctrine

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tac.substrates.pact_nerv_vq.architecture import PactNervVqConfig

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


SCHEMA_VERSION = "pact_nerv_vq_mlx_renderer_v1"
MLX_EVIDENCE_GRADE = "[macOS-MLX research-signal]"


def _require_mlx() -> None:
    if mx is None:
        raise RuntimeError(
            "MLX is not available on this host; the PACT-NeRV-VQ MLX renderer "
            "requires Apple Silicon with the ``mlx`` package installed. "
            f"Original import error: {_MLX_IMPORT_ERROR!r}"
        )


def _pixel_shuffle_2x_nhwc(x: Any) -> Any:
    """PixelShuffle 2x for NHWC tensors via canonical PR95 helper."""
    _require_mlx()
    from tac.local_acceleration.pr95_hnerv_mlx import (
        pixel_shuffle_2x_nhwc,
    )

    return pixel_shuffle_2x_nhwc(x)


def _bilinear_resize_nhwc(x: Any, target_h: int, target_w: int) -> Any:
    """Bilinear resize NHWC tensor matching PyTorch ``F.interpolate(align_corners=False)``."""
    _require_mlx()
    src_h, src_w = int(x.shape[1]), int(x.shape[2])
    if src_h == target_h and src_w == target_w:
        return x
    from tac.local_acceleration.pr95_hnerv_mlx import (
        bilinear_resize2x_align_corners_false_nhwc,
    )
    if src_h * 2 == target_h and src_w * 2 == target_w:
        return bilinear_resize2x_align_corners_false_nhwc(x)
    raise NotImplementedError(
        f"_bilinear_resize_nhwc generic resize ({src_h}x{src_w} -> "
        f"{target_h}x{target_w}) not implemented; canonical VQ forward "
        "uses integer-ratio 2x PixelShuffle(7) which lands at (384, 512)."
    )


def _siren_uniform_bound(fan_in: int, w: float) -> float:
    return math.sqrt(6.0 / max(fan_in, 1)) / max(w, 1.0)


class _DepthSepConvMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX depth-separable conv (depthwise 3x3 + pointwise 1x1)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        _require_mlx()
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.depthwise: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            padding=1,
            groups=in_ch,
        )
        self.pointwise: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
        )

    def __call__(self, x: Any) -> Any:
        return self.pointwise(self.depthwise(x))


class _DsUpBlockMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX DepthSep -> sin -> PixelShuffle(2) (mirrors PyTorch sister)."""

    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        _require_mlx()
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.w = float(sin_freq)
        self.dsc = _DepthSepConvMLX(in_ch, out_ch * 4)

    def __call__(self, x: Any) -> Any:
        h = self.dsc(x)
        h = mx.sin(self.w * h)  # type: ignore[union-attr]
        return _pixel_shuffle_2x_nhwc(h)


class VectorQuantizerEMAMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX VQ codebook with EMA update + straight-through estimator.

    Mirrors :class:`tac.substrates.pact_nerv_vq.architecture.VectorQuantizerEMA`.

    Per van den Oord 1711.00937 §3.1-3.2:
    - Nearest-codebook lookup (Euclidean distance in latent space)
    - Straight-through estimator: ``z_q_st = z_e + (z_q - z_e).detach()``
    - EMA codebook update with Laplace smoothing (decay=0.99 canonical)
    - Commitment loss ``||z_e - sg(z_q)||^2`` returned for the trainer's
      score-aware Lagrangian.

    MLX-specific notes:
    - MLX does NOT have ``.detach()`` semantics; ``mx.stop_gradient(...)``
      is the canonical equivalent.
    - Codebook + EMA buffers are stored as regular MLX arrays (not nn.Parameter)
      since they are EMA-updated, not gradient-updated. The trainer calls
      ``ema_update`` once per batch in training mode.
    """

    def __init__(
        self,
        *,
        codebook_size: int,
        latent_dim: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ) -> None:
        _require_mlx()
        super().__init__()
        if codebook_size <= 0 or codebook_size > 65535:
            raise ValueError(f"codebook_size {codebook_size} out of uint16 range")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive; got {latent_dim}")
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1); got {decay}")
        self.codebook_size = int(codebook_size)
        self.latent_dim = int(latent_dim)
        self.decay = float(decay)
        self.epsilon = float(epsilon)

        # Codebook + EMA buffers. NOT registered as MLX parameters so they
        # are not touched by value_and_grad — the trainer EMA-updates them
        # explicitly after each forward.
        self._codebook = mx.random.normal(  # type: ignore[union-attr]
            shape=(codebook_size, latent_dim)
        ) * 0.02
        self._ema_cluster_size = mx.zeros((codebook_size,))  # type: ignore[union-attr]
        self._ema_w = mx.array(self._codebook)  # type: ignore[union-attr]

    @property
    def codebook(self) -> Any:
        """Read-only accessor for the current codebook (for tests + export)."""
        return self._codebook

    @property
    def ema_cluster_size(self) -> Any:
        return self._ema_cluster_size

    @property
    def ema_w(self) -> Any:
        return self._ema_w

    def __call__(self, z_e: Any) -> tuple[Any, Any, Any]:
        """Quantize z_e via nearest-codebook lookup.

        Args:
            z_e: (B, latent_dim) MLX array of encoder outputs.

        Returns:
            (z_q_st, indices, commitment_loss):
              - z_q_st: (B, latent_dim) straight-through quantized vectors.
              - indices: (B,) int32 codebook indices.
              - commitment_loss: scalar ``mean((z_e - sg(z_q))^2)``.
        """
        if z_e.ndim != 2 or int(z_e.shape[1]) != self.latent_dim:
            raise ValueError(
                f"z_e must be (B, {self.latent_dim}); got shape {tuple(z_e.shape)}"
            )

        # Distances: (B, codebook_size) via ||z_e||^2 - 2*z_e@cb.T + ||cb||^2
        ze_sq = mx.sum(z_e * z_e, axis=1, keepdims=True)  # type: ignore[union-attr]
        cb_sq = mx.sum(self._codebook * self._codebook, axis=1)  # type: ignore[union-attr]
        dot = z_e @ mx.transpose(self._codebook)  # type: ignore[union-attr]
        dists = ze_sq - 2.0 * dot + mx.expand_dims(cb_sq, axis=0)  # type: ignore[union-attr]
        indices = mx.argmin(dists, axis=1)  # type: ignore[union-attr]

        # Gather codebook entries.
        z_q = mx.take(self._codebook, indices, axis=0)  # type: ignore[union-attr]

        # Commitment loss: ||z_e - sg(z_q)||^2 averaged.
        z_q_sg = mx.stop_gradient(z_q)  # type: ignore[union-attr]
        diff = z_e - z_q_sg
        commitment_loss = mx.mean(diff * diff)  # type: ignore[union-attr]

        # Straight-through estimator: gradient flows around the quantization step.
        z_q_st = z_e + mx.stop_gradient(z_q - z_e)  # type: ignore[union-attr]
        return z_q_st, indices, commitment_loss

    def ema_update(self, z_e: Any, indices: Any) -> None:
        """Update codebook via EMA per van den Oord §3.2.

        Called by the trainer once per batch in training mode (no-op at eval).
        """
        _require_mlx()
        # MLX has no built-in one_hot; build via equality broadcast.
        ar = mx.arange(self.codebook_size)  # type: ignore[union-attr]
        idx_bcast = mx.expand_dims(indices, axis=1)  # type: ignore[union-attr]
        ar_bcast = mx.expand_dims(ar, axis=0)  # type: ignore[union-attr]
        one_hot = (idx_bcast == ar_bcast).astype(z_e.dtype)

        cluster_size = mx.sum(one_hot, axis=0)  # type: ignore[union-attr]
        ema_cluster = (
            self._ema_cluster_size * self.decay + cluster_size * (1.0 - self.decay)
        )
        # Laplace smoothing.
        n = mx.sum(ema_cluster)  # type: ignore[union-attr]
        ema_cluster = (
            (ema_cluster + self.epsilon)
            / (n + float(self.codebook_size) * self.epsilon)
        ) * n
        self._ema_cluster_size = ema_cluster

        # dw = encodings.T @ z_e  shape (codebook_size, latent_dim)
        dw = mx.transpose(one_hot) @ z_e  # type: ignore[union-attr]
        self._ema_w = self._ema_w * self.decay + dw * (1.0 - self.decay)
        # Updated codebook = ema_w / ema_cluster (broadcast over latent_dim).
        denom = mx.expand_dims(ema_cluster, axis=1)  # type: ignore[union-attr]
        # Avoid div-by-zero on entirely-unused codes.
        safe_denom = mx.where(denom > 0, denom, mx.ones_like(denom))  # type: ignore[union-attr]
        new_codebook = self._ema_w / safe_denom
        # Where denom == 0, keep old codebook entry.
        keep_old = (denom == 0).astype(self._codebook.dtype)  # type: ignore[union-attr]
        self._codebook = keep_old * self._codebook + (1.0 - keep_old) * new_codebook


class PactNervVqSubstrateMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX-native PACT-NeRV-VQ substrate (L1 LONG-RUN MLX-LOCAL).

    1:1 architectural mirror of
    :class:`tac.substrates.pact_nerv_vq.architecture.PactNervVqSubstrate`.

    Forward path:

    1. Per-pair latent z_e (from learnable ``self.latents``).
    2. VQ codebook lookup -> z_q (straight-through estimator).
    3. Latent embedding -> initial spatial grid (NHWC).
    4. For each upsample block: DepthSep -> sin -> PixelShuffle(2).
    5. Final 1x1 RGB heads + sigmoid + scale to [0, 255].
    6. Stack to (B, 2, 3, H, W) for the canonical ``call_b2chw_255``
       convention required by :mod:`tac.substrates._shared.mlx_score_aware`.

    Per HNeRV parity L5: outputs RGB at contest camera resolution (384, 512);
    NOT a mask codec.

    The VQ-VAE codebook + per-pair discrete index is the substrate's UNIQUE
    distinguishing primitive vs the canonical HNeRV-class decoder backbone.

    The commitment loss from the most recent forward is exposed via
    :attr:`last_commitment_loss` so the trainer can add it to the
    score-aware Lagrangian per Catalog #6 + canonical score-aware form.
    """

    def __init__(self, cfg: PactNervVqConfig) -> None:
        _require_mlx()
        super().__init__()
        self.cfg = cfg
        num_pairs = int(cfg.num_pairs)
        latent_dim = int(cfg.latent_dim)
        embed_dim = int(cfg.embed_dim)
        initial_grid_h = int(cfg.initial_grid_h)
        initial_grid_w = int(cfg.initial_grid_w)
        num_upsample_blocks = int(cfg.num_upsample_blocks)

        # Per-pair learnable latent (matches PyTorch ``self.latents``).
        self.latents = mx.random.normal(  # type: ignore[union-attr]
            shape=(num_pairs, latent_dim)
        ) * 0.02

        # VQ-VAE quantizer (codebook + EMA + commitment loss).
        self.quantizer = VectorQuantizerEMAMLX(
            codebook_size=int(cfg.codebook_size),
            latent_dim=latent_dim,
            decay=float(cfg.codebook_decay),
        )

        # Latent -> initial spatial grid embedding.
        self.latent_embed: Any = nn.Linear(  # type: ignore[union-attr]
            latent_dim,
            embed_dim * initial_grid_h * initial_grid_w,
        )

        # Per-block upsample (NO IA3 modulation; VQ-VAE is the distinguishing
        # primitive, NOT pose-conditioning).
        channels = [embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at "
                f"least num_upsample_blocks ({num_upsample_blocks}) entries"
            )
        self.blocks: list[Any] = [
            _DsUpBlockMLX(channels[i], channels[i + 1], cfg.sin_frequency)
            for i in range(num_upsample_blocks)
        ]

        final_ch = channels[num_upsample_blocks]
        self.head_rgb_0: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=final_ch, out_channels=3, kernel_size=1
        )
        self.head_rgb_1: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=final_ch, out_channels=3, kernel_size=1
        )

        self._last_commitment_loss = mx.array(0.0)  # type: ignore[union-attr]
        self._last_indices = mx.array([], dtype=mx.int32)  # type: ignore[union-attr]

        self._siren_init()

    def _siren_init(self) -> None:
        """SIREN init for Conv2d + Linear weights (mirrors PyTorch sister)."""
        w = float(self.cfg.sin_frequency)
        fan_in = int(self.cfg.latent_dim)
        bound = _siren_uniform_bound(fan_in, w)
        self.latent_embed.update({
            "weight": mx.random.uniform(  # type: ignore[union-attr]
                low=-bound, high=bound, shape=self.latent_embed.weight.shape
            ),
            "bias": mx.zeros_like(self.latent_embed.bias),  # type: ignore[union-attr]
        })
        for block in self.blocks:
            dsc = block.dsc
            d = dsc.depthwise
            kH, kW = d.weight.shape[1], d.weight.shape[2]
            fan_in_d = int(kH * kW)
            bound_d = _siren_uniform_bound(fan_in_d, w)
            d.update({
                "weight": mx.random.uniform(  # type: ignore[union-attr]
                    low=-bound_d, high=bound_d, shape=d.weight.shape
                ),
                "bias": mx.zeros_like(d.bias),  # type: ignore[union-attr]
            })
            p = dsc.pointwise
            in_ch = p.weight.shape[3]
            fan_in_p = int(in_ch * 1 * 1)
            bound_p = _siren_uniform_bound(fan_in_p, w)
            p.update({
                "weight": mx.random.uniform(  # type: ignore[union-attr]
                    low=-bound_p, high=bound_p, shape=p.weight.shape
                ),
                "bias": mx.zeros_like(p.bias),  # type: ignore[union-attr]
            })
        for head in (self.head_rgb_0, self.head_rgb_1):
            in_ch_h = head.weight.shape[3]
            fan_in_h = int(in_ch_h * 1 * 1)
            bound_h = _siren_uniform_bound(fan_in_h, w)
            head.update({
                "weight": mx.random.uniform(  # type: ignore[union-attr]
                    low=-bound_h, high=bound_h, shape=head.weight.shape
                ),
                "bias": mx.zeros_like(head.bias),  # type: ignore[union-attr]
            })

    def __call__(self, pair_indices: Any, ego_pose: Any = None) -> Any:
        """Forward path returning (B, 2, 3, H, W) in [0, 255] per canonical convention.

        Args:
            pair_indices: (B,) int array of pair indices in [0, num_pairs).
            ego_pose: ignored (VQ has no pose conditioning; accepted for API
                parity with the canonical RendererBundle).

        Returns:
            (B, 2, 3, H, W) MLX float32 array in [0, 255].
        """
        del ego_pose  # VQ has no pose conditioning.

        # Latent gather + VQ quantization.
        z_e = mx.take(self.latents, pair_indices, axis=0)  # type: ignore[union-attr]
        z_q_st, indices, commitment_loss = self.quantizer(z_e)
        # Track for trainer observability per Catalog #305.
        self._last_commitment_loss = commitment_loss
        self._last_indices = indices

        # Latent -> initial spatial grid (NHWC).
        h = self.latent_embed(z_q_st)
        h = mx.reshape(  # type: ignore[union-attr]
            h,
            (
                -1,
                self.cfg.initial_grid_h,
                self.cfg.initial_grid_w,
                self.cfg.embed_dim,
            ),
        )

        # Per-block forward (NO pose modulation).
        for block in self.blocks:
            h = block(h)

        h = _bilinear_resize_nhwc(
            h, int(self.cfg.output_height), int(self.cfg.output_width)
        )

        # 1x1 RGB heads + sigmoid + scale to [0, 255].
        rgb_0_nhwc = mx.sigmoid(self.head_rgb_0(h)) * 255.0  # type: ignore[union-attr]
        rgb_1_nhwc = mx.sigmoid(self.head_rgb_1(h)) * 255.0  # type: ignore[union-attr]
        pair_nhwc = mx.stack([rgb_0_nhwc, rgb_1_nhwc], axis=1)  # type: ignore[union-attr]
        return mx.transpose(pair_nhwc, (0, 1, 4, 2, 3))  # type: ignore[union-attr]

    @property
    def last_commitment_loss(self) -> Any:
        """Commitment loss from the most recent forward (for the trainer)."""
        return self._last_commitment_loss

    @property
    def last_indices(self) -> Any:
        """Codebook indices from the most recent forward."""
        return self._last_indices

    def num_parameters(self) -> int:
        """Total trainable parameter count (excludes EMA buffers).

        Per van den Oord §3.2, the codebook + EMA buffers are EMA-updated
        rather than gradient-updated, so they are not counted in the
        gradient-pass parameter total. The PyTorch sister registers them as
        buffers (not nn.Parameter), so this matches.
        """
        _require_mlx()
        flat = tree_flatten(self.parameters())  # type: ignore[union-attr]
        total = 0
        for _name, arr in flat:
            total += int(np.prod(arr.shape))
        return total

    def export_state_dict(self) -> dict[str, np.ndarray]:
        """Export weights in PyTorch state_dict layout for the canonical bridge.

        Returns:
            dict[str, np.ndarray] keyed by PyTorch sister parameter + buffer names.
        """
        _require_mlx()
        out: dict[str, np.ndarray] = {}
        out["latents"] = np.asarray(self.latents, dtype=np.float32).copy()
        # VQ codebook + EMA buffers (matches PyTorch buffer registrations).
        out["quantizer.codebook"] = np.asarray(
            self.quantizer.codebook, dtype=np.float32
        ).copy()
        out["quantizer.ema_cluster_size"] = np.asarray(
            self.quantizer.ema_cluster_size, dtype=np.float32
        ).copy()
        out["quantizer.ema_w"] = np.asarray(
            self.quantizer.ema_w, dtype=np.float32
        ).copy()
        out["latent_embed.weight"] = np.asarray(
            self.latent_embed.weight, dtype=np.float32
        ).copy()
        out["latent_embed.bias"] = np.asarray(
            self.latent_embed.bias, dtype=np.float32
        ).copy()
        for i, block in enumerate(self.blocks):
            d = block.dsc.depthwise
            p = block.dsc.pointwise
            out[f"blocks.{i}.dsc.depthwise.weight"] = np.transpose(
                np.asarray(d.weight, dtype=np.float32), (0, 3, 1, 2)
            ).copy()
            out[f"blocks.{i}.dsc.depthwise.bias"] = np.asarray(
                d.bias, dtype=np.float32
            ).copy()
            out[f"blocks.{i}.dsc.pointwise.weight"] = np.transpose(
                np.asarray(p.weight, dtype=np.float32), (0, 3, 1, 2)
            ).copy()
            out[f"blocks.{i}.dsc.pointwise.bias"] = np.asarray(
                p.bias, dtype=np.float32
            ).copy()
        for head_name in ("head_rgb_0", "head_rgb_1"):
            head = getattr(self, head_name)
            out[f"{head_name}.weight"] = np.transpose(
                np.asarray(head.weight, dtype=np.float32), (0, 3, 1, 2)
            ).copy()
            out[f"{head_name}.bias"] = np.asarray(
                head.bias, dtype=np.float32
            ).copy()
        return out


__all__ = [
    "MLX_EVIDENCE_GRADE",
    "SCHEMA_VERSION",
    "PactNervVqSubstrateMLX",
    "VectorQuantizerEMAMLX",
]
