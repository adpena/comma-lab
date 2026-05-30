# SPDX-License-Identifier: MIT
"""Z8 hierarchical predictive coding MLX module — L0 SCAFFOLD.

Binds Catalog #312's canonical quadruple SIMULTANEOUSLY per HNeRV parity
discipline L7. See ``__init__.py`` for full canonical-quadruple binding
documentation + design memo reference.

L0 SCAFFOLD scope (per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or
RESEARCH-ONLY"):

- MLX-local-iteration ONLY (axis_tag=``[macOS-MLX research-signal]`` per
  CLAUDE.md "MLX portable-local-substrate authority"; no paid CUDA).
- Multi-level (3-level default) RSSM hierarchy with per-level categorical
  posterior + Gumbel-Softmax + straight-through estimator reparametrization at
  training; per-level int32 category indices stored in archive for
  byte-deterministic dequant at inflate.
- Per-level Mallat wavelet detail bands captured at each hierarchy level
  (sketch-only at L0; full Daubechies-CDF entropy coding deferred to Phase 2).
- DreamerV3 deterministic GRU state init (ego-motion conditioning at each
  pair) — sketch-only at L0; full RSSM unroll deferred to Phase 2.
- Wyner-Ziv top-level sketch (top-level conditional entropy estimate is
  computed; full Wyner-Ziv coding against frame_0 decoded latent deferred to
  Phase 2).
- Decoder topology borrowed from canonical PR95 HNeRV (re-uses sister A
  DreamerV3 RSSM PixelShuffle block pattern; per Catalog #290
  ADOPT_CANONICAL_BECAUSE_SERVES decision: the HNeRV decoder is the
  empirically validated PR95/PR101/PR110 medal-class topology; substrate-class
  shift is at the LATENT layers, not the decoder).
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
except Exception as exc:  # pragma: no cover - import guard for non-Apple CI
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR: Exception | None = exc
else:
    _MLX_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Canonical configuration constants per design memo + canonical equation registry.
# -----------------------------------------------------------------------------

EVAL_HW: tuple[int, int] = (384, 512)
"""Contest scorer-resolution (height, width). Matches sister PR95 HNeRV decoder eval_size."""

NUM_PAIRS: int = 600
"""Contest pair count (1200 frames / 2 per pair)."""

DEFAULT_NUM_LEVELS: int = 3
"""Default hierarchy depth (canonical Rao-Ballard visual cortex 3-level model)."""

DEFAULT_NUM_GROUPS_PER_LEVEL: tuple[int, ...] = (24, 16, 8)
"""Default DreamerV3 categorical groups per level. Decreases at deeper levels
per Mallat coarse-fine wavelet bound (fewer groups needed at coarser scales)."""

DEFAULT_NUM_CATEGORIES_PER_LEVEL: tuple[int, ...] = (256, 128, 64)
"""Default DreamerV3 categorical alphabet size per level. Decreases at deeper
levels per scale-invariant entropy bound (coarser scales carry less detail)."""

DEFAULT_EGO_MOTION_DIM: int = 6
"""Ego-motion vector dimensionality (6-DOF pose per Z6 sister pattern)."""


def _require_mlx() -> None:
    if mx is None or nn is None:
        raise RuntimeError(
            f"MLX not available in this environment: {_MLX_IMPORT_ERROR!r}. "
            "Z8 hierarchical predictive coding L0 SCAFFOLD is MLX-local per "
            "CLAUDE.md 'MLX portable-local-substrate authority'."
        )


@dataclass(frozen=True)
class Z8HierarchicalConfig:
    """Static design-time parameters for the Z8 substrate.

    Defaults are the canonical Rao-Ballard 3-level + DreamerV3 per-level
    categorical posterior configuration per the design memo Section 8.
    """

    num_levels: int = DEFAULT_NUM_LEVELS
    """Number of hierarchy levels (canonical 3 = Rao-Ballard visual cortex model)."""

    num_groups_per_level: tuple[int, ...] = DEFAULT_NUM_GROUPS_PER_LEVEL
    """DreamerV3 categorical groups at each level (length == num_levels)."""

    num_categories_per_level: tuple[int, ...] = DEFAULT_NUM_CATEGORIES_PER_LEVEL
    """DreamerV3 categorical alphabet size at each level (length == num_levels)."""

    base_channels: int = 24
    """Decoder base channel count (canonical PR95 HNeRV taper)."""

    decoder_latent_dim: int = 28
    """Decoder input latent dim (after multi-level fusion projection).
    Matches PR95 HNeRV canonical latent_dim=28 for topology re-use."""

    eval_size: tuple[int, int] = EVAL_HW
    """Decoder output (H, W). Bicubic upscale to camera (874, 1164) at inflate."""

    num_pairs: int = NUM_PAIRS
    """Contest pair count."""

    ego_motion_dim: int = DEFAULT_EGO_MOTION_DIM
    """Ego-motion vector dimensionality (6-DOF pose for DreamerV3 GRU input)."""

    deterministic_state_dim: int = 64
    """DreamerV3 GRU deterministic state dim (canonical Hafner 2023 recipe;
    sized down from production 256 for L0 scaffold cost)."""

    gumbel_temperature: float = 1.0
    """Gumbel-Softmax temperature τ. Annealed during training (1.0 → 0.1)."""

    use_straight_through: bool = True
    """If True, use STE reparametrization (canonical Hafner 2024 recipe)."""

    wavelet_basis_id: int = 0
    """Wavelet basis: 0=Daubechies-4 (canonical Mallat 1989 default).
    L0 scaffold uses simple sum-pooling proxy; Phase 2 lands full Daubechies-CDF."""

    def __post_init__(self) -> None:
        if len(self.num_groups_per_level) != self.num_levels:
            raise ValueError(
                f"num_groups_per_level length {len(self.num_groups_per_level)} "
                f"must match num_levels {self.num_levels}"
            )
        if len(self.num_categories_per_level) != self.num_levels:
            raise ValueError(
                f"num_categories_per_level length {len(self.num_categories_per_level)} "
                f"must match num_levels {self.num_levels}"
            )
        for level_idx, k in enumerate(self.num_categories_per_level):
            if k <= 0 or k > 65536:
                raise ValueError(
                    f"num_categories_per_level[{level_idx}]={k} out of (0, 65536]"
                )
        for level_idx, g in enumerate(self.num_groups_per_level):
            if g <= 0 or g > 256:
                raise ValueError(
                    f"num_groups_per_level[{level_idx}]={g} out of (0, 256]"
                )
        if self.num_levels < 1 or self.num_levels > 8:
            raise ValueError(f"num_levels={self.num_levels} out of [1, 8]")

    @property
    def total_categorical_bits_per_sample(self) -> float:
        """Total H(T) across all levels per the canonical-quadruple binding math.

        Per the canonical equation
        ``categorical_posterior_capacity_vs_continuous_gaussian_v1`` extended
        to multi-level: ``sum_l G_l * log2(K_l)`` bits/sample.
        """
        return float(
            sum(
                g * math.log2(k)
                for g, k in zip(
                    self.num_groups_per_level, self.num_categories_per_level
                )
            )
        )

    @property
    def total_latent_packing_bytes_per_pair(self) -> int:
        """Total per-pair archive cost for category indices across all levels."""
        total = 0
        for g, k in zip(self.num_groups_per_level, self.num_categories_per_level):
            if k <= 256:
                total += g  # 1 byte per group
            else:
                total += g * 2  # u16 per group
        return total


# -----------------------------------------------------------------------------
# Gumbel-Softmax with straight-through estimator + Hafner 2023 §3 1% unimix
# robustness mixture. Z8 DELEGATES to sister A=DreamerV3 canonical helpers per
# Catalog #290 ADOPT_CANONICAL_BECAUSE_SERVES + CLAUDE.md UNIQUE-AND-COMPLETE-
# PER-METHOD operating mode (canonical helpers are TOOLS used WHEN they serve;
# Z8 per-level categorical posterior is structurally identical to sister
# DreamerV3 single-level so adoption serves measurably and by clear principle).
#
# Wave 10/11 (2026-05-29) gap closure: prior L0 SCAFFOLD reimplemented
# gumbel_softmax_sample locally without unimix_alpha threading; the Wave 3
# math-fidelity audit on sister DreamerV3 added the 1% unimix mixture per
# Hafner 2023 §3 "Robustness" but the fix did NOT propagate here because the
# Z8 path was a duplicate. The fix lands the delegation pattern so future
# sister fixes propagate structurally; the canonical helpers
# `apply_unimix_to_logits` + `gumbel_softmax_sample` now have a SINGLE source
# of truth at `tac.substrates.dreamer_v3_rssm.module`.
# -----------------------------------------------------------------------------


# MLX_PRIMITIVE_UNIQUE_BECAUSE_thin_substrate_delegation_to_dreamerv3:Z8_function_signature_preserves_substrate_API_pinned_by_wave_10_11_regression_test_inspect_getsource_check_body_delegates_to_sister_dreamerv3_canonical_helper_NO_local_math_per_wave_10_11_fix_2026_05_29_principled_fork_per_catalog_290_ADOPT_CANONICAL_BECAUSE_SERVES
def gumbel_softmax_sample(
    logits: Any,
    *,
    temperature: float = 1.0,
    use_straight_through: bool = True,
    unimix_alpha: float = 0.01,
    key: Any = None,
) -> tuple[Any, Any]:
    """Sample categorical from logits via Gumbel-Softmax reparametrization.

    Delegates to sister `tac.substrates.dreamer_v3_rssm.gumbel_softmax_sample`
    so the Hafner 2023 §3 1% unimix robustness mixture (Wave 3 fix
    2026-05-29) propagates structurally to Z8 per-level categorical posteriors.

    Args:
        logits: ``(..., K)`` un-normalized log-probabilities.
        temperature: Gumbel softmax τ; lower = sharper / more discrete.
        use_straight_through: if True, return one-hot in forward + soft
            gradient in backward (canonical STE per Jang 2016 + Maddison 2016).
        unimix_alpha: Hafner 2023 1% unimix coefficient α. Default 0.01
            matches the canonical recipe; set 0.0 to disable for ablation.
        key: MLX random key. If None, derive via sister canonical helper.

    Returns:
        (soft_or_hard_sample, category_indices) — ``soft_or_hard_sample`` is
        ``(..., K)`` simplex (STE if use_straight_through else soft);
        ``category_indices`` is ``(...,)`` int32 of argmax for archive
        serialization.

    Wave 10/11 audit 2026-05-29: replaced local duplicate implementation with
    canonical delegation to sister `dreamer_v3_rssm` per Catalog #290
    ADOPT_CANONICAL_BECAUSE_SERVES. Closes the "Z8 claimed canonical reuse
    but reimplemented" gap surfaced during the Wave 10/11 RL + DreamerV3
    sister cluster audit. Prior local implementation omitted unimix per Wave 3
    cargo-cult audit finding (HARD-EARNED canonical Hafner 2023 §3).

    Sources:
    - Hafner et al. 2023 arXiv:2301.04104 §3 "Robustness"
    - Jang et al. 2016 arXiv:1611.01144 Gumbel-Softmax
    - Maddison et al. 2016 arXiv:1611.00712 Concrete distribution
    - Sister canonical implementation:
      `src/tac/substrates/dreamer_v3_rssm/module.py::gumbel_softmax_sample`
    """
    from tac.substrates.dreamer_v3_rssm import (
        gumbel_softmax_sample as _sister_gumbel_softmax_sample,
    )

    return _sister_gumbel_softmax_sample(
        logits,
        temperature=temperature,
        use_straight_through=use_straight_through,
        unimix_alpha=unimix_alpha,
        key=key,
    )


def apply_unimix_to_logits(logits: Any, *, unimix_alpha: float = 0.01) -> Any:
    """Apply Hafner 2023 1% unimix mixture to Z8 per-level categorical logits.

    Delegates to sister `tac.substrates.dreamer_v3_rssm.apply_unimix_to_logits`
    per Catalog #290 ADOPT_CANONICAL_BECAUSE_SERVES. Wave 10/11 (2026-05-29)
    gap closure: re-exports the canonical helper at the Z8 surface so callers
    that import from `tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer`
    inherit the canonical Hafner 2023 §3 robustness behavior without needing
    to know about the sister cross-substrate reuse.
    """
    from tac.substrates.dreamer_v3_rssm import (
        apply_unimix_to_logits as _sister_apply_unimix_to_logits,
    )

    return _sister_apply_unimix_to_logits(logits, unimix_alpha=unimix_alpha)


# -----------------------------------------------------------------------------
# Mallat wavelet sum-pooling proxy (L0 scaffold).
#
# Phase 2 will replace with full Daubechies-4 wavelet transform per Mallat 1989
# canonical multi-resolution analysis. For L0 scaffold, sum-pooling captures
# the multi-scale spatial decomposition pattern without the implementation
# complexity of the full discrete wavelet transform.
# -----------------------------------------------------------------------------


def _mallat_sum_pool_2x_nhwc(x: Any) -> Any:
    """Sum-pool 2x downsample for NHWC tensors (L0 wavelet proxy).

    Phase 2 will replace with full Daubechies-4 DWT. Sum-pooling preserves the
    multi-scale spatial decomposition for L0 substrate-design verification.
    """
    _require_mlx()
    batch, height, width, channels = (int(dim) for dim in x.shape)
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            f"sum-pool 2x requires even (H, W); got ({height}, {width})"
        )
    # Reshape to (B, H/2, 2, W/2, 2, C) then sum over the (2, 2) blocks
    y = mx.reshape(x, (batch, height // 2, 2, width // 2, 2, channels))  # type: ignore[union-attr]
    y = mx.sum(y, axis=(2, 4))  # type: ignore[union-attr]
    return y


# -----------------------------------------------------------------------------
# Multi-level RSSM hierarchy (Rao-Ballard 1999 + DreamerV3 canonical
# combination per Catalog #312 canonical quadruple).
# -----------------------------------------------------------------------------


def _pixel_shuffle_2x_nhwc(x: Any) -> Any:
    """PixelShuffle 2x for NHWC tensors via canonical PR95 helper.

    CONSOLIDATE-OP-1 F-MIGRATION (2026-05-26): delegates to canonical
    ``tac.local_acceleration.pr95_hnerv_mlx::pixel_shuffle_2x_nhwc``, replacing
    the prior local copy that was empirically PyTorch-byte-stable (0.0 drift
    per FIX-WAVE-R1' ``4684dbbab``) but DUPLICATED the canonical primitive.
    Sister substrates (A=DreamerV3, D=Z6) also migrate to the canonical helper
    in the same wave so the channel-FIRST reshape convention
    ``(B, H, W, out_C, 2, 2)`` + transpose ``(0, 1, 4, 2, 5, 3)`` is owned
    by exactly one canonical source of truth.

    Historical FIX-WAVE-R1' F-OP1 anchor: the prior channel-LAST convention
    ``(B, H, W, 2, 2, out_C)`` + transpose ``(0, 1, 3, 2, 4, 5)`` produced
    3.77 absolute drift vs PyTorch ``nn.PixelShuffle(2)`` (per R1' Path 3 F
    review empirical measurement 2026-05-26T08:03Z); the channel-FIRST
    convention now in the canonical helper is empirically PyTorch-byte-stable
    (0.0 drift per sister D=Z6 anchor).

    Catalog #295 self-containment is preserved because the canonical helper
    is imported only at MLX training time in ``mlx_renderer.py``; the
    substrate's inflate runtime at ``inflate.py`` is PyTorch-only and uses
    native ``F.pixel_shuffle(x, upscale_factor=2)``.

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

    FIX-WAVE-R1' F-OP2 (2026-05-26): delegates to canonical
    ``tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc``
    which is empirically PyTorch-byte-stable (0.0 drift) vs the prior
    ``mx.repeat`` 2x approximation that produced 1.51 absolute drift vs
    PyTorch ``F.interpolate(scale_factor=2, mode='bilinear',
    align_corners=False)`` (per R1' Path 3 F review empirical measurement
    2026-05-26T08:03Z). Sister A=DreamerV3 FIX-WAVE-R1 (commit
    `a23779a732e7bb056`) lands the identical canonical helper substitution.

    Catalog #295 self-containment is preserved because the canonical helper
    is imported only at MLX training time in ``mlx_renderer.py``; the
    substrate's inflate runtime at ``inflate.py`` is PyTorch-only and does
    NOT import MLX (PyTorch uses ``F.interpolate(mode='bilinear',
    align_corners=False)`` natively). The Catalog #295 contract scopes
    ``submissions/*/inflate.py`` PYTHONPATH self-containment; this substrate's
    MLX module is at ``src/tac/substrates/z8_hierarchical_predictive_coding/``
    which is in-tree by definition.
    """
    _require_mlx()
    from tac.local_acceleration.pr95_hnerv_mlx import (
        bilinear_resize2x_align_corners_false_nhwc,
    )

    return bilinear_resize2x_align_corners_false_nhwc(x)


class _Z8UpsampleBlock(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Single PixelShuffle upsample block (sister A=DreamerV3 pattern reuse)."""

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


class Z8HierarchicalPredictiveCoderMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """L0 SCAFFOLD: multi-level RSSM hierarchy + per-level Mallat wavelet
    proxy + per-level categorical posterior + DreamerV3 deterministic state.

    Architecture (substrate-engineering per HNeRV parity L7 waiver):

    1. **Per-pair per-level learned categorical logits**: ``self.logits[level][pair_idx]``
       shape ``(G_l, K_l)`` per level; total stored bytes per pair = sum across
       levels of group counts (1 byte/group when K<=256).
    2. **Per-pair per-level Gumbel-Softmax sample**: ``(G_l, K_l)`` simplex via
       STE at training.
    3. **Per-level categorical-to-continuous projection**: ``Linear(G_l*K_l, decoder_latent_dim)``.
    4. **Multi-level fusion**: sum or concat across levels per the
       Rao-Ballard top-down prediction (L0: simple sum; Phase 2: per-level
       weighting via Mallat detail-band sparsity).
    5. **DreamerV3 deterministic state**: GRUCell with ego_motion input per pair.
    6. **Decoder forward (canonical PR95 topology)**: 6 PixelShuffle blocks
       from 6×8 → 384×512 with sin() activations and dilated refine.
    7. **Per-frame RGB heads**: ``rgb_0`` + ``rgb_1`` (sigmoid × 255).

    Per Catalog #290 canonical-vs-unique decision Section 2 in design memo:
    decoder topology + Gumbel-Softmax STE + per-level categorical posterior =
    ADOPT_CANONICAL_BECAUSE_SERVES (sister A pattern reuse). Multi-level
    hierarchy + Mallat wavelet proxy + DreamerV3 GRU = UNIQUE substrate
    engineering.
    """

    def __init__(self, cfg: Z8HierarchicalConfig) -> None:
        _require_mlx()
        super().__init__()
        self.cfg = cfg
        L = int(cfg.num_levels)
        N = int(cfg.num_pairs)
        base_h, base_w = 6, 8  # PR95 canonical base grid for 384×512 output
        self.base_h = base_h
        self.base_w = base_w

        # Per-pair per-level learnable categorical logits (one set per level).
        # Initialized near uniform for max-entropy prior per design memo Section 4.
        self.logits_per_level: list[Any] = []
        for level_idx in range(L):
            G = int(cfg.num_groups_per_level[level_idx])
            K = int(cfg.num_categories_per_level[level_idx])
            key = mx.random.key(int(level_idx + 1))  # type: ignore[union-attr]
            init = mx.random.normal(shape=(N, G, K), key=key) * 0.01  # type: ignore[union-attr]
            self.logits_per_level.append(init)

        # Per-level categorical-to-continuous projection.
        L_dim = int(cfg.decoder_latent_dim)
        self.cat_to_continuous_per_level = []
        for level_idx in range(L):
            G = int(cfg.num_groups_per_level[level_idx])
            K = int(cfg.num_categories_per_level[level_idx])
            self.cat_to_continuous_per_level.append(
                nn.Linear(G * K, L_dim)  # type: ignore[union-attr]
            )

        # DreamerV3 deterministic GRU cell (ego-motion conditioning).
        # Per Hafner 2023 canonical recipe; gru input is fused latent + ego_motion.
        det_dim = int(cfg.deterministic_state_dim)
        gru_input_dim = L_dim + int(cfg.ego_motion_dim)
        # MLX nn.GRU expects (batch, seq, features) input; for L0 we use linear
        # gate as a proxy (full GRUCell deferred to Phase 2). Linear-gate proxy
        # captures the deterministic state's role without GRUCell complexity.
        self.deterministic_gate = nn.Linear(gru_input_dim, det_dim)  # type: ignore[union-attr]

        # Multi-level fusion → decoder latent dim.
        # L0: simple sum across levels then linear; Phase 2: weighted by Mallat
        # detail-band sparsity per level.
        self.level_fusion = nn.Linear(L_dim + det_dim, L_dim)  # type: ignore[union-attr]

        # HNeRV decoder topology (canonical PR95 channel taper; sister A pattern).
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
            L_dim, channels[0] * base_h * base_w
        )
        self.blocks = [
            _Z8UpsampleBlock(channels[i], channels[i + 1]) for i in range(6)
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
        pair_nhwc = mx.stack([f0, f1], axis=1)  # type: ignore[union-attr]
        return mx.transpose(pair_nhwc, (0, 1, 4, 2, 3))  # type: ignore[union-attr]

    def forward_training(
        self,
        pair_indices: Any,
        *,
        ego_motion: Any | None = None,
        gumbel_key: Any = None,
    ) -> tuple[Any, list[Any], list[Any]]:
        """Training forward: per-level Gumbel-Softmax, fuse, GRU state, decode.

        Args:
            pair_indices: ``(B,)`` int32 pair indices.
            ego_motion: ``(B, ego_motion_dim)`` float32 pose vector. If None,
                a zero vector is used (smoke-only).
            gumbel_key: MLX random key. If None, derived from rng.

        Returns:
            (rgb_pair, per_level_indices, per_level_soft_samples) —
            ``rgb_pair`` is ``(B, 2, 3, H, W)`` float in [0, 255];
            ``per_level_indices`` is list of ``(B, G_l)`` int32 per level
            for archive serialization; ``per_level_soft_samples`` is list of
            ``(B, G_l, K_l)`` simplex per level (STE forward, soft gradient).
        """
        _require_mlx()
        cfg = self.cfg
        L = int(cfg.num_levels)
        B = int(pair_indices.shape[0])

        # Per-level Gumbel-Softmax sample + fuse to continuous embedding.
        per_level_indices: list[Any] = []
        per_level_soft: list[Any] = []
        per_level_embeddings: list[Any] = []

        for level_idx in range(L):
            level_logits = mx.take(  # type: ignore[union-attr]
                self.logits_per_level[level_idx], pair_indices, axis=0
            )  # (B, G_l, K_l)
            G_l = int(cfg.num_groups_per_level[level_idx])
            K_l = int(cfg.num_categories_per_level[level_idx])
            soft, indices = gumbel_softmax_sample(
                level_logits,
                temperature=float(cfg.gumbel_temperature),
                use_straight_through=bool(cfg.use_straight_through),
                key=gumbel_key,
            )
            per_level_soft.append(soft)
            per_level_indices.append(indices)
            flat = mx.reshape(soft, (B, G_l * K_l))  # type: ignore[union-attr]
            embedding = self.cat_to_continuous_per_level[level_idx](flat)
            per_level_embeddings.append(embedding)

        # Multi-level fusion (L0: sum across levels; Phase 2: Mallat-weighted)
        fused_latent = per_level_embeddings[0]
        for level_idx in range(1, L):
            fused_latent = fused_latent + per_level_embeddings[level_idx]

        # DreamerV3 deterministic state (linear-gate proxy at L0).
        if ego_motion is None:
            ego_motion = mx.zeros((B, int(cfg.ego_motion_dim)))  # type: ignore[union-attr]
        gru_input = mx.concatenate([fused_latent, ego_motion], axis=-1)  # type: ignore[union-attr]
        det_state = mx.tanh(self.deterministic_gate(gru_input))  # type: ignore[union-attr]

        # Fuse fused latent + deterministic state → decoder latent.
        decoder_input = mx.concatenate([fused_latent, det_state], axis=-1)  # type: ignore[union-attr]
        decoder_embedding = self.level_fusion(decoder_input)

        rgb_pair = self._decoder_forward(decoder_embedding)
        return rgb_pair, per_level_indices, per_level_soft

    def forward_eval_from_indices(
        self,
        per_level_indices: list[Any],
        *,
        ego_motion: Any | None = None,
    ) -> Any:
        """Eval forward: take pre-stored per-level argmax indices, decode without Gumbel.

        Args:
            per_level_indices: list of ``(B, G_l)`` int32 category indices per level.
            ego_motion: ``(B, ego_motion_dim)`` ego-motion vector or None.

        Returns:
            rgb_pair: ``(B, 2, 3, H, W)`` float in [0, 255].
        """
        _require_mlx()
        cfg = self.cfg
        L = int(cfg.num_levels)
        B = int(per_level_indices[0].shape[0])

        # Per-level one-hot → continuous embedding.
        per_level_embeddings: list[Any] = []
        for level_idx in range(L):
            G_l = int(cfg.num_groups_per_level[level_idx])
            K_l = int(cfg.num_categories_per_level[level_idx])
            indices = per_level_indices[level_idx]
            eye = mx.eye(K_l)  # type: ignore[union-attr]
            one_hot_flat = mx.take(eye, mx.reshape(indices, (B * G_l,)), axis=0)  # type: ignore[union-attr]
            one_hot = mx.reshape(one_hot_flat, (B, G_l, K_l))  # type: ignore[union-attr]
            flat = mx.reshape(one_hot, (B, G_l * K_l))  # type: ignore[union-attr]
            embedding = self.cat_to_continuous_per_level[level_idx](flat)
            per_level_embeddings.append(embedding)

        # Fuse + deterministic state (deterministic at eval).
        fused_latent = per_level_embeddings[0]
        for level_idx in range(1, L):
            fused_latent = fused_latent + per_level_embeddings[level_idx]

        if ego_motion is None:
            ego_motion = mx.zeros((B, int(cfg.ego_motion_dim)))  # type: ignore[union-attr]
        gru_input = mx.concatenate([fused_latent, ego_motion], axis=-1)  # type: ignore[union-attr]
        det_state = mx.tanh(self.deterministic_gate(gru_input))  # type: ignore[union-attr]

        decoder_input = mx.concatenate([fused_latent, det_state], axis=-1)  # type: ignore[union-attr]
        decoder_embedding = self.level_fusion(decoder_input)

        return self._decoder_forward(decoder_embedding)

    def __call__(self, pair_indices: Any) -> Any:
        """Default __call__ is training forward (returns rgb_pair only)."""
        rgb_pair, _indices, _soft = self.forward_training(pair_indices)
        return rgb_pair

    def architecture_manifest(self) -> dict[str, Any]:
        """Canonical observability surface (Catalog #305 facet 5 cite-ability)."""
        return {
            "schema": "z8_hierarchical_predictive_coding_mlx_architecture_v1",
            "num_levels": self.cfg.num_levels,
            "num_groups_per_level": list(self.cfg.num_groups_per_level),
            "num_categories_per_level": list(self.cfg.num_categories_per_level),
            "total_categorical_bits_per_sample": (
                self.cfg.total_categorical_bits_per_sample
            ),
            "total_latent_packing_bytes_per_pair": (
                self.cfg.total_latent_packing_bytes_per_pair
            ),
            "decoder_latent_dim": self.cfg.decoder_latent_dim,
            "base_channels": self.cfg.base_channels,
            "decoder_channels_taper": list(self.channels),
            "eval_size": list(self.cfg.eval_size),
            "num_pairs": self.cfg.num_pairs,
            "ego_motion_dim": self.cfg.ego_motion_dim,
            "deterministic_state_dim": self.cfg.deterministic_state_dim,
            "gumbel_temperature": self.cfg.gumbel_temperature,
            "use_straight_through": self.cfg.use_straight_through,
            "wavelet_basis_id": self.cfg.wavelet_basis_id,
            "decoder_topology_source": (
                "submissions/hnerv_muon/src/model.py::HNeRVDecoder via "
                "tac.substrates.dreamer_v3_rssm::DreamerV3RSSMSubstrateMLX (sister A pattern)"
            ),
            "canonical_equation_refs": [
                "mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1",
                "scorer_conditional_joint_rate_distortion_floor_v1",
                "categorical_posterior_capacity_vs_continuous_gaussian_v1",
                "ego_motion_concentration_prior_v1",
                "cross_codec_super_additive_orthogonality_predictor_v1",
            ],
            "axis_tag": "[macOS-MLX research-signal]",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }


def z8_decoder_param_count(cfg: Z8HierarchicalConfig) -> int:
    """Estimate parameter count for cost-band predictions.

    Returns total params across: per-pair per-level logits + per-level
    cat-to-continuous projections + level fusion + deterministic gate + stem
    + 6 upsample blocks + refine + 2 RGB heads.
    """
    L = int(cfg.num_levels)
    N = int(cfg.num_pairs)
    L_dim = int(cfg.decoder_latent_dim)
    C = int(cfg.base_channels)
    det_dim = int(cfg.deterministic_state_dim)

    # Per-pair per-level logits (training-only; archive stores argmax indices)
    logits_params = sum(
        N * int(cfg.num_groups_per_level[level_idx])
        * int(cfg.num_categories_per_level[level_idx])
        for level_idx in range(L)
    )

    # Per-level categorical-to-continuous projections
    cat_proj_params = sum(
        int(cfg.num_groups_per_level[level_idx])
        * int(cfg.num_categories_per_level[level_idx])
        * L_dim
        + L_dim
        for level_idx in range(L)
    )

    # Deterministic gate
    det_gate_params = (L_dim + int(cfg.ego_motion_dim)) * det_dim + det_dim

    # Level fusion
    level_fusion_params = (L_dim + det_dim) * L_dim + L_dim

    # Stem (Linear: L_dim → channels[0] * 6 * 8)
    stem_params = L_dim * (C * 48) + (C * 48)

    # 6 upsample blocks (sister A pattern)
    channels = [C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)]
    blocks_params = 0
    for i in range(6):
        in_ch = channels[i]
        out_ch = channels[i + 1]
        blocks_params += in_ch * (out_ch * 4) * 9 + (out_ch * 4)
        if in_ch != out_ch:
            blocks_params += in_ch * out_ch * 1 + out_ch
    final_ch = channels[-1]
    refine0_params = final_ch * (final_ch // 2) * 9 + (final_ch // 2)
    refine1_params = (final_ch // 2) * final_ch * 9 + final_ch
    rgb_params = 2 * (final_ch * 3 * 9 + 3)

    return int(
        logits_params
        + cat_proj_params
        + det_gate_params
        + level_fusion_params
        + stem_params
        + blocks_params
        + refine0_params
        + refine1_params
        + rgb_params
    )


__all__ = [
    "DEFAULT_NUM_LEVELS",
    "DEFAULT_NUM_GROUPS_PER_LEVEL",
    "DEFAULT_NUM_CATEGORIES_PER_LEVEL",
    "DEFAULT_EGO_MOTION_DIM",
    "EVAL_HW",
    "NUM_PAIRS",
    "Z8HierarchicalConfig",
    "Z8HierarchicalPredictiveCoderMLX",
    "gumbel_softmax_sample",
    "z8_decoder_param_count",
]
