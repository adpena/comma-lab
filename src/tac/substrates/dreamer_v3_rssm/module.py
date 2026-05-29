# SPDX-License-Identifier: MIT
"""DreamerV3 RSSM categorical posterior MLX module — L0 SCAFFOLD.

Replaces the C6 IBPS continuous-Gaussian 24-dim per-pair latent with a
categorical posterior at ``G=24`` groups × ``K=256`` categories
(``H(T) = G * log2(K) = 192 bits/sample``) per the canonical equation
``categorical_posterior_capacity_vs_continuous_gaussian_v1``.

L0 SCAFFOLD scope (per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or
RESEARCH-ONLY"):

- MLX-local-iteration ONLY (axis_tag=``[macOS-MLX research-signal]`` per
  CLAUDE.md "MLX portable-local-substrate authority"; no paid CUDA).
- Categorical posterior + Gumbel-Softmax + straight-through estimator
  reparametrization at training; per-pair int32 category indices stored in
  archive for byte-deterministic dequant at inflate.
- Decoder topology borrowed from canonical PR95 HNeRV (6 PixelShuffle blocks;
  ~50K params at ``base_channels=24``) per Catalog #290 canonical-vs-unique
  decision: ADOPT-CANONICAL-BECAUSE-SERVES (the HNeRV decoder is the
  empirically validated PR95/PR101/PR110 medal-class topology; substrate-class
  shift is at the LATENT layer, not the decoder layer).
- NO GRU at L0 (canonical-only ablation per symposium Step 1 assumption #3
  canonical unwind); full RSSM with GRU-deterministic state is L1+ extension
  per Hafner 2024 canonical recipe.
- NO RL reward/value heads (RL-domain-specific; not applicable to dashcam
  per symposium Step 1 assumption #3).
- PyTorch port deferred per Path 3 cascade (#1251 export bridge + #1257
  inflate parity closure + #1265 contest-equivalence gate already in place;
  port mechanics inherit from canonical
  ``src/tac/local_acceleration/pr95_hnerv_mlx.py::load_pytorch_state_dict_into_mlx``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised in environments with MLX installed
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
except Exception as exc:  # pragma: no cover - import guard for non-Apple CI
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR: Exception | None = exc
else:
    _MLX_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Canonical configuration constants per symposium + canonical equation registry.
# -----------------------------------------------------------------------------

EVAL_HW: tuple[int, int] = (384, 512)
"""Contest scorer-resolution (height, width). Matches PR95 HNeRV decoder eval_size."""

NUM_PAIRS: int = 600
"""Contest pair count (1200 frames / 2 per pair)."""

DEFAULT_G: int = 24
"""Default number of categorical groups (matches C6 IBPS 24-dim baseline)."""

DEFAULT_K: int = 256
"""Default categorical alphabet size (8 bits/group → 1 byte int8 packing)."""


def _require_mlx() -> None:
    if mx is None or nn is None:
        raise RuntimeError(
            f"MLX not available in this environment: {_MLX_IMPORT_ERROR!r}. "
            "DreamerV3 RSSM L0 scaffold is MLX-local per CLAUDE.md "
            "'MLX portable-local-substrate authority'."
        )


@dataclass(frozen=True)
class DreamerV3RSSMConfig:
    """Static design-time parameters for the DreamerV3 RSSM substrate.

    Defaults are the C6 Path B2 substrate-adapted canonical configuration per
    the T3 symposium Decision D + canonical equation registry
    (``categorical_posterior_capacity_vs_continuous_gaussian_v1``
    ``canonical_configs.c6_path_b2_substrate_adapted``).
    """

    num_groups: int = DEFAULT_G
    """Number of independent categorical groups (per-pair). G=24 default."""

    num_categories: int = DEFAULT_K
    """Alphabet size per group. K=256 default (8 bits/group)."""

    base_channels: int = 24
    """Decoder base channel count. ~50K params at base_channels=24."""

    decoder_latent_dim: int = 28
    """Decoder input latent dim (after categorical → continuous projection).
    Matches PR95 HNeRV canonical latent_dim=28 for topology re-use."""

    eval_size: tuple[int, int] = EVAL_HW
    """Decoder output (H, W). Bicubic upscale to camera (874, 1164) at inflate."""

    num_pairs: int = NUM_PAIRS
    """Contest pair count."""

    gumbel_temperature: float = 1.0
    """Gumbel-Softmax temperature τ. Annealed during training (1.0 → 0.1)."""

    use_straight_through: bool = True
    """If True, use STE reparametrization (canonical Hafner 2024 recipe)."""

    unimix_alpha: float = 0.01
    """Unimix coefficient α per Hafner 2023 DreamerV3 §3 "Robustness".

    Canonical Hafner 2023 robustness technique: every categorical posterior is
    parameterized as ``p_unimix = (1-α) * softmax(logits) + α * uniform(K)``
    with α = 0.01 (1% uniform mixture). Prevents the posterior from collapsing
    to a hard one-hot (the structural failure mode that breaks gradients
    through the STE) and is the canonical "fixed hyperparameter across diverse
    domains" enabler.

    Wave 3 math-fidelity audit anchor (2026-05-29): the prior L0 SCAFFOLD
    omitted this term; the omission was CARGO-CULTED relative to Hafner 2023
    canonical (the docstring claimed "canonical Hafner 2024 recipe" while the
    implementation skipped the canonical robustness mixture). Restored as a
    HARD-EARNED canonical-1:1 element with α=0.01 default; configurable so
    the K-capacity disambiguator can vary the mixture if needed.

    Sources:
    - Hafner et al. 2023 "Mastering Diverse Domains through World Models"
      arXiv:2301.04104 §3 "Robustness" + Appendix categorical posterior
    - Reference impl https://github.com/danijar/dreamerv3 (jaxutils.cast_to_compute / categorical_kl)
    """

    @property
    def categorical_bits_per_sample(self) -> float:
        """H(T) = G * log2(K) bits/sample per Shannon canonical equation."""
        return float(self.num_groups) * math.log2(float(self.num_categories))

    @property
    def latent_packing_bytes_per_pair(self) -> int:
        """Per-pair archive cost for category indices (assuming K<=256 fits in u8)."""
        if self.num_categories <= 256:
            return self.num_groups  # 1 byte per group
        return self.num_groups * 2  # u16 per group


# -----------------------------------------------------------------------------
# Gumbel-Softmax with straight-through estimator (canonical Hafner 2024 recipe
# per Jang et al. 2016 arXiv:1611.01144 + Maddison et al. 2016 arXiv:1611.00712).
# -----------------------------------------------------------------------------


def apply_unimix_to_logits(logits: Any, *, unimix_alpha: float = 0.01) -> Any:
    """Apply Hafner 2023 1% unimix mixture to categorical logits.

    Canonical Hafner 2023 §3 "Robustness" categorical posterior is
    parameterized as a mixture of the network softmax and a uniform prior:

        p_unimix(y|x) = (1 - α) * softmax(logits) + α * (1 / K)
        logits_unimixed = log(p_unimix)

    The returned logits are equivalent to sampling from p_unimix at any
    temperature τ (the softmax + log invertibility preserves the mixture
    distribution exactly). For α = 0 this is a no-op (returns the input
    logits unchanged up to a constant log Z shift that cancels in softmax).

    Args:
        logits: ``(..., K)`` un-normalized log-probabilities.
        unimix_alpha: mixing coefficient α ∈ [0, 1). 0.01 = Hafner 2023 canonical.

    Returns:
        ``(..., K)`` logits whose softmax equals the unimix mixture.

    Source: Hafner et al. 2023 arXiv:2301.04104 §3 "Robustness" + canonical
    reference https://github.com/danijar/dreamerv3 (categorical posterior
    parameterization).
    """
    _require_mlx()
    if not 0.0 <= float(unimix_alpha) < 1.0:
        raise ValueError(
            f"unimix_alpha must be in [0, 1); got {unimix_alpha}"
        )
    if float(unimix_alpha) == 0.0:
        return logits
    K = int(logits.shape[-1])
    probs = mx.softmax(logits, axis=-1)  # type: ignore[union-attr]
    uniform_probs = mx.full(  # type: ignore[union-attr]
        shape=logits.shape, vals=1.0 / float(K)
    )
    mixed = (1.0 - float(unimix_alpha)) * probs + float(unimix_alpha) * uniform_probs
    # Re-log to recover logits whose softmax equals the mixture.
    return mx.log(mixed)  # type: ignore[union-attr]


def gumbel_softmax_sample(
    logits: Any,
    *,
    temperature: float = 1.0,
    use_straight_through: bool = True,
    unimix_alpha: float = 0.01,
    key: Any = None,
) -> tuple[Any, Any]:
    """Sample categorical from logits via Gumbel-Softmax reparametrization.

    Canonical Hafner 2023 DreamerV3 + Jang 2016 + Maddison 2016 categorical
    reparametrization, with the Hafner 2023 §3 1% unimix robustness mixture
    applied to the logits BEFORE Gumbel perturbation.

    Args:
        logits: ``(..., K)`` un-normalized log-probabilities.
        temperature: Gumbel softmax τ; lower = sharper / more discrete.
        use_straight_through: if True, return one-hot in forward + soft gradient
            in backward (canonical STE).
        unimix_alpha: Hafner 2023 1% unimix coefficient α. Default 0.01 matches
            the canonical Hafner 2023 §3 robustness recipe. Set to 0.0 to
            disable for ablation.
        key: MLX random key. If None, derive from current MLX rng.

    Returns:
        (soft_or_hard_sample, category_indices) — ``soft_or_hard_sample`` is
        ``(..., K)`` simplex (STE if use_straight_through else soft);
        ``category_indices`` is ``(...,)`` int32 of argmax for archive serialization.

    Wave 3 math-fidelity audit 2026-05-29: added unimix_alpha parameter +
    default 0.01 to match Hafner 2023 canonical. Prior L0 SCAFFOLD omitted
    unimix; the omission was CARGO-CULTED relative to Hafner 2023.

    Sources:
    - Hafner et al. 2023 "Mastering Diverse Domains through World Models"
      arXiv:2301.04104 §3 "Robustness"
    - Jang et al. 2016 "Categorical Reparameterization with Gumbel-Softmax"
      arXiv:1611.01144 (Gumbel-Softmax)
    - Maddison et al. 2016 "The Concrete Distribution" arXiv:1611.00712
      (concrete reparametrization)
    """
    _require_mlx()
    if key is None:
        key = mx.random.key(0)  # type: ignore[union-attr]
    # Hafner 2023 §3 unimix: blend logits' softmax with uniform prior at α.
    mixed_logits = apply_unimix_to_logits(logits, unimix_alpha=unimix_alpha)
    # Sample Gumbel(0, 1) noise: g = -log(-log(u)) for u ~ Uniform(0, 1).
    uniform = mx.random.uniform(low=1e-10, high=1.0, shape=logits.shape, key=key)  # type: ignore[union-attr]
    gumbel = -mx.log(-mx.log(uniform))  # type: ignore[union-attr]
    perturbed = (mixed_logits + gumbel) / float(max(temperature, 1e-6))
    soft = mx.softmax(perturbed, axis=-1)  # type: ignore[union-attr]
    indices = mx.argmax(soft, axis=-1)  # type: ignore[union-attr]
    if use_straight_through:
        # Hard one-hot in forward pass, soft gradient via STE trick.
        # MLX equivalent of: hard = onehot; output = hard - soft.detach() + soft
        # which gives gradient = soft and forward value = hard.
        K = int(logits.shape[-1])
        # one_hot via eye + take
        eye = mx.eye(K)  # type: ignore[union-attr]
        hard = mx.take(eye, indices, axis=0)  # type: ignore[union-attr]
        ste_output = hard - mx.stop_gradient(soft) + soft  # type: ignore[union-attr]
        return ste_output, indices
    return soft, indices


# -----------------------------------------------------------------------------
# HNeRV decoder topology (re-used from canonical PR95 per Catalog #290
# ADOPT-CANONICAL-BECAUSE-SERVES decision; decoder layer is empirically
# validated medal-class topology; substrate-class shift is at LATENT layer).
# -----------------------------------------------------------------------------


def _pixel_shuffle_2x_nhwc(x: Any) -> Any:
    """PixelShuffle 2x for NHWC tensors via canonical PR95 helper.

    CONSOLIDATE-OP-1 A-MIGRATION (2026-05-26): delegates to canonical
    ``tac.local_acceleration.pr95_hnerv_mlx::pixel_shuffle_2x_nhwc``, replacing
    the prior local copy that was empirically PyTorch-byte-stable (0.0 drift
    per FIX-WAVE-R1 ``e1b101888``) but DUPLICATED the canonical primitive.
    Sister substrates (D=Z6, F=Z8) also migrate to the canonical helper in
    the same wave so the channel-FIRST reshape convention
    ``(B, H, W, out_C, 2, 2)`` + transpose ``(0, 1, 4, 2, 5, 3)`` is owned
    by exactly one canonical source of truth.

    Historical FIX-WAVE-R1 A-OP1 anchor: the prior channel-LAST convention
    ``(B, H, W, 2, 2, out_C)`` + transpose ``(0, 1, 3, 2, 4, 5)`` produced
    2.40 absolute drift vs PyTorch ``nn.PixelShuffle(2)``; the channel-FIRST
    convention now in the canonical helper is empirically PyTorch-byte-stable
    (0.0 drift per R1 review measurement).

    Catalog #295 self-containment is preserved because the canonical helper
    is imported only at MLX training time in ``module.py``; the substrate's
    inflate runtime at ``inflate.py`` is PyTorch-only and uses native
    ``F.pixel_shuffle(x, upscale_factor=2)``.

    Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L9
    runtime closure: MLX-trained-PyTorch-inflated model MUST be the same
    runtime as the MLX trainer observes at convergence.
    """
    _require_mlx()
    from tac.local_acceleration.pr95_hnerv_mlx import (
        pixel_shuffle_2x_nhwc,
    )

    return pixel_shuffle_2x_nhwc(x)


def _bilinear_resize_2x_nhwc(x: Any) -> Any:
    """Bilinear upsample 2x for NHWC tensors via canonical PR95 helper.

    FIX-WAVE-R1 A-OP2 (2026-05-26): delegates to canonical
    ``tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc``
    which is empirically PyTorch-byte-stable (0.0 drift per R1 review
    measurement) vs the prior ``mx.repeat`` 2x approximation that produced
    0.99 absolute drift vs PyTorch
    ``F.interpolate(scale_factor=2, mode='bilinear', align_corners=False)``.

    Catalog #295 self-containment is preserved because the canonical helper
    is imported only at MLX training time in ``module.py``; the substrate's
    inflate runtime at ``inflate.py`` is PyTorch-only and does NOT import
    MLX (PyTorch uses ``F.interpolate(mode='bilinear', align_corners=False)``
    natively). The Catalog #295 contract scopes
    ``submissions/*/inflate.py`` PYTHONPATH self-containment; this
    substrate's MLX module is at ``src/tac/substrates/dreamer_v3_rssm/``
    which is in-tree by definition.
    """
    _require_mlx()
    from tac.local_acceleration.pr95_hnerv_mlx import (
        bilinear_resize2x_align_corners_false_nhwc,
    )

    return bilinear_resize2x_align_corners_false_nhwc(x)


class _RSSMUpsampleBlock(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Single PixelShuffle upsample block (canonical PR95 HNeRV pattern)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        _require_mlx()
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1)  # type: ignore[union-attr]
        if in_channels != out_channels:
            self.skip_conv: Any = nn.Conv2d(in_channels, out_channels, 1)  # type: ignore[union-attr]
        else:
            self.skip_conv = None

    def __call__(self, x: Any) -> Any:
        identity = _bilinear_resize_2x_nhwc(x)
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
        decoded = _pixel_shuffle_2x_nhwc(self.conv(x))
        return mx.sin(decoded + identity)  # type: ignore[union-attr]


class DreamerV3RSSMSubstrateMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """L0 SCAFFOLD: categorical posterior + HNeRV decoder backbone.

    Architecture (substrate-engineering per HNeRV parity L7 waiver):

    1. Per-pair learned categorical logits: ``self.logits[pair_idx]`` shape
       ``(G, K)``; total stored bytes = ``G * K * 4`` per pair (float32 logits;
       reduced to ``G`` bytes per pair after argmax at inflate).
    2. Gumbel-Softmax sample: ``(G, K)`` simplex via STE at training.
    3. Categorical-to-continuous projection: ``Linear(G*K, decoder_latent_dim)``
       collapses one-hot to continuous embedding for HNeRV decoder input.
    4. HNeRV decoder forward (canonical PR95 topology): 6 PixelShuffle blocks
       from 6×8 → 384×512 with sin() activations and dilated refine.
    5. Per-frame RGB heads: ``rgb_0`` + ``rgb_1`` (sigmoid × 255).

    Forward (training mode, returns soft sample + indices):
        logits = self.logits[pair_indices]  # (B, G, K)
        soft, indices = gumbel_softmax_sample(logits, temp=τ, ste=True)
        embedding = self.cat_to_continuous(soft.reshape(B, G*K))
        rgb_0, rgb_1 = self.decoder(embedding)

    Forward (eval mode, takes pre-computed indices from archive):
        one_hot = eye[indices]  # (B, G, K)
        embedding = self.cat_to_continuous(one_hot.reshape(B, G*K))
        rgb_0, rgb_1 = self.decoder(embedding)
    """

    def __init__(self, cfg: DreamerV3RSSMConfig) -> None:
        _require_mlx()
        super().__init__()
        self.cfg = cfg
        G = int(cfg.num_groups)
        K = int(cfg.num_categories)
        base_h, base_w = 6, 8  # PR95 canonical base grid for 384×512 output
        self.base_h = base_h
        self.base_w = base_w

        # Per-pair learnable categorical logits (G groups × K categories).
        # Initialized near uniform for max-entropy prior.
        key = mx.random.key(0)  # type: ignore[union-attr]
        self.logits = mx.random.normal(  # type: ignore[union-attr]
            shape=(int(cfg.num_pairs), G, K), key=key
        ) * 0.01

        # Categorical-to-continuous projection: (G*K) → decoder_latent_dim
        self.cat_to_continuous = nn.Linear(G * K, int(cfg.decoder_latent_dim))  # type: ignore[union-attr]

        # HNeRV decoder topology (canonical PR95 channel taper per Catalog #290
        # ADOPT-CANONICAL-BECAUSE-SERVES decision).
        C = int(cfg.base_channels)
        channels = [
            C,
            C,
            C,
            int(C * 0.75),
            int(C * 0.58),
            int(C * 0.5),
            int(C * 0.5),
        ]
        if min(channels) < 1:
            raise ValueError(f"base_channels={C} too small for PR95 channel taper")
        self.channels = channels

        self.stem = nn.Linear(  # type: ignore[union-attr]
            int(cfg.decoder_latent_dim), channels[0] * base_h * base_w
        )
        self.blocks = [
            _RSSMUpsampleBlock(channels[i], channels[i + 1]) for i in range(6)
        ]
        final_ch = channels[-1]
        self.refine0 = nn.Conv2d(  # type: ignore[union-attr]
            final_ch, final_ch // 2, 3, padding=2, dilation=2
        )
        self.refine1 = nn.Conv2d(  # type: ignore[union-attr]
            final_ch // 2, final_ch, 3, padding=1
        )
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)  # type: ignore[union-attr]
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)  # type: ignore[union-attr]

    def _decoder_forward(self, embedding: Any) -> Any:
        """Run HNeRV decoder forward on continuous embedding. Returns (B, 2, 3, H, W)."""
        batch = int(embedding.shape[0])
        x = self.stem(embedding)
        x = mx.reshape(  # type: ignore[union-attr]
            x, (batch, self.channels[0], self.base_h, self.base_w)
        )
        x = mx.transpose(x, (0, 2, 3, 1))  # NCHW → NHWC  # type: ignore[union-attr]
        x = mx.sin(x)  # type: ignore[union-attr]
        for block in self.blocks:
            x = block(x)
        refined = self.refine1(self.refine0(x))
        x = x + 0.1 * mx.sin(refined)  # type: ignore[union-attr]
        f0 = mx.sigmoid(self.rgb_0(x)) * 255.0  # type: ignore[union-attr]
        f1 = mx.sigmoid(self.rgb_1(x)) * 255.0  # type: ignore[union-attr]
        # Stack as (B, 2, H, W, C) then transpose to (B, 2, C, H, W) for PR95 parity
        pair_nhwc = mx.stack([f0, f1], axis=1)  # type: ignore[union-attr]
        return mx.transpose(pair_nhwc, (0, 1, 4, 2, 3))  # type: ignore[union-attr]

    def forward_training(
        self,
        pair_indices: Any,
        *,
        gumbel_key: Any = None,
    ) -> tuple[Any, Any, Any]:
        """Training forward: gather logits, Gumbel-Softmax, decoder forward.

        Returns:
            (rgb_pair, category_indices, soft_sample) — rgb_pair is
            ``(B, 2, 3, H, W)`` float in [0, 255]; category_indices is
            ``(B, G)`` int32 for archive serialization;
            soft_sample is ``(B, G, K)`` simplex (STE forward, soft gradient).
        """
        _require_mlx()
        logits = mx.take(self.logits, pair_indices, axis=0)  # type: ignore[union-attr]
        soft, indices = gumbel_softmax_sample(
            logits,
            temperature=float(self.cfg.gumbel_temperature),
            use_straight_through=bool(self.cfg.use_straight_through),
            unimix_alpha=float(self.cfg.unimix_alpha),
            key=gumbel_key,
        )
        B = int(logits.shape[0])
        G = int(self.cfg.num_groups)
        K = int(self.cfg.num_categories)
        flat = mx.reshape(soft, (B, G * K))  # type: ignore[union-attr]
        embedding = self.cat_to_continuous(flat)
        rgb_pair = self._decoder_forward(embedding)
        return rgb_pair, indices, soft

    def forward_eval_from_indices(self, indices_per_pair: Any) -> Any:
        """Eval forward: take pre-stored argmax indices, decode without Gumbel.

        Args:
            indices_per_pair: ``(B, G)`` int32 category indices loaded from archive.

        Returns:
            rgb_pair: ``(B, 2, 3, H, W)`` float in [0, 255].
        """
        _require_mlx()
        B = int(indices_per_pair.shape[0])
        G = int(self.cfg.num_groups)
        K = int(self.cfg.num_categories)
        eye = mx.eye(K)  # type: ignore[union-attr]
        one_hot_flat = mx.take(eye, mx.reshape(indices_per_pair, (B * G,)), axis=0)  # type: ignore[union-attr]
        one_hot = mx.reshape(one_hot_flat, (B, G, K))  # type: ignore[union-attr]
        flat = mx.reshape(one_hot, (B, G * K))  # type: ignore[union-attr]
        embedding = self.cat_to_continuous(flat)
        return self._decoder_forward(embedding)

    def __call__(self, pair_indices: Any) -> Any:
        """Default __call__ is training forward (returns rgb_pair only)."""
        rgb_pair, _indices, _soft = self.forward_training(pair_indices)
        return rgb_pair

    def architecture_manifest(self) -> dict[str, Any]:
        """Canonical observability surface (Catalog #305 facet 5 cite-ability)."""
        return {
            "schema": "dreamer_v3_rssm_mlx_architecture_v1",
            "num_groups_G": self.cfg.num_groups,
            "num_categories_K": self.cfg.num_categories,
            "categorical_bits_per_sample": self.cfg.categorical_bits_per_sample,
            "latent_packing_bytes_per_pair": self.cfg.latent_packing_bytes_per_pair,
            "decoder_latent_dim": self.cfg.decoder_latent_dim,
            "base_channels": self.cfg.base_channels,
            "decoder_channels_taper": list(self.channels),
            "eval_size": list(self.cfg.eval_size),
            "num_pairs": self.cfg.num_pairs,
            "gumbel_temperature": self.cfg.gumbel_temperature,
            "use_straight_through": self.cfg.use_straight_through,
            "unimix_alpha": self.cfg.unimix_alpha,
            "decoder_topology_source": (
                "submissions/hnerv_muon/src/model.py::HNeRVDecoder via "
                "tac.local_acceleration.pr95_hnerv_mlx::HNeRVDecoderMLX"
            ),
            "canonical_equation_refs": [
                "categorical_posterior_capacity_vs_continuous_gaussian_v1",
                "categorical_blahut_arimoto_rate_distortion_v1",
            ],
            "axis_tag": "[macOS-MLX research-signal]",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }


def rssmc_decoder_param_count(cfg: DreamerV3RSSMConfig) -> int:
    """Estimate decoder parameter count for cost-band predictions.

    Returns total params across: per-pair logits + cat_to_continuous projection
    + stem + 6 upsample blocks (+ optional skip_conv) + refine + 2 RGB heads.

    Per-pair logits dominate at G=24, K=256, num_pairs=600:
      24 * 256 * 600 = 3,686,400 floats ≈ 14.7 MB unquantized (reduced to
      ~14.4 KB after argmax + int8 packing in archive).
    """
    G = int(cfg.num_groups)
    K = int(cfg.num_categories)
    C = int(cfg.base_channels)
    L = int(cfg.decoder_latent_dim)

    # Per-pair logits (training-only; archive stores argmax indices)
    logits_params = int(cfg.num_pairs) * G * K
    # Categorical-to-continuous projection
    cat_proj_params = G * K * L + L  # weight + bias
    # Stem (Linear: L → channels[0] * 6 * 8)
    stem_params = L * (C * 48) + (C * 48)
    # 6 upsample blocks
    channels = [C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)]
    blocks_params = 0
    for i in range(6):
        in_ch = channels[i]
        out_ch = channels[i + 1]
        # Conv(in_ch, out_ch * 4, 3x3)
        blocks_params += in_ch * (out_ch * 4) * 9 + (out_ch * 4)
        if in_ch != out_ch:
            blocks_params += in_ch * out_ch * 1 + out_ch
    final_ch = channels[-1]
    # Refine 0: Conv(final_ch, final_ch // 2, 3x3)
    refine0_params = final_ch * (final_ch // 2) * 9 + (final_ch // 2)
    # Refine 1: Conv(final_ch // 2, final_ch, 3x3)
    refine1_params = (final_ch // 2) * final_ch * 9 + final_ch
    # RGB heads
    rgb_params = 2 * (final_ch * 3 * 9 + 3)

    return int(
        logits_params
        + cat_proj_params
        + stem_params
        + blocks_params
        + refine0_params
        + refine1_params
        + rgb_params
    )


__all__ = [
    "DEFAULT_G",
    "DEFAULT_K",
    "DreamerV3RSSMConfig",
    "DreamerV3RSSMSubstrateMLX",
    "EVAL_HW",
    "NUM_PAIRS",
    "apply_unimix_to_logits",
    "gumbel_softmax_sample",
    "rssmc_decoder_param_count",
]
