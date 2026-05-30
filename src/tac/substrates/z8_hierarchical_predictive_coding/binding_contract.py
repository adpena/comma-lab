# SPDX-License-Identifier: MIT
"""Z8 hierarchical predictive coding per-level binding contract.

Defines the explicit Python Protocol + frozen-dataclass surfaces that every
Phase-2 piece (Mamba-2 SSD deterministic state replacing DreamerV3's GRU;
full Daubechies-4 wavelet replacing the sum-pool proxy; full Wyner-Ziv
top-level coder replacing the sketch; Yousfi-grounded score-aware loss
embedded at each hierarchy level) must satisfy.

Per the operator binding 2026-05-29:

  *"all parts need to be designed with the full stack in mind"*

The PR-95 lesson at deeper resolution: each ingredient isn't bound AFTER it's
built — it's BUILT BOUND. This module is the structural surface that lets the
binding be expressed as code (interfaces) rather than as docstring prose. A
piece that implements the relevant Protocol with matching dataclass shapes
slots into the Z8 forward pass without composition-time surprises.

This file deliberately has NO MLX, PyTorch, NumPy, or sister-substrate
imports. The contract is pure Python typing + frozen dataclasses; implementing
Protocols pull in their own framework. Tests verify the contract is
self-consistent and that the existing `Z8HierarchicalConfig` (in sister
``mlx_renderer.py``) satisfies the per-level shape contract.

Cross-references:

- Sister memory ``z8-hierarchical-predictive-coding-binding-first-active-
  build-target-yousfi-grounded-20260529`` (the build target this binding
  expresses; first commit milestone).
- Sister tracking surface ``build_progress.py`` (the in-source milestone
  tuple that records Phase-2 progress; this file's landing is milestone #1).
- Catalog #312 (canonical quadruple: Rao-Ballard + DreamerV3 + Mallat +
  Wyner-Ziv simultaneously bound).
- Catalog #290 (canonical-vs-unique decision per layer — every Protocol
  below has a documented decision).
- ``mlx_renderer.py`` ``Z8HierarchicalConfig`` (the per-level dimension
  configuration the contract here formalizes).
- HNeRV parity discipline L7 (substrate-engineering exceeds bolt-on size
  budget by design; this contract is substrate-engineering scope).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Canonical constants the contract references. These mirror sister
# ``mlx_renderer.py`` so the contract has a single source of truth for the
# magic numbers (NUM_PAIRS, contest scorer resolution, etc.).
# ---------------------------------------------------------------------------

CONTEST_PAIR_COUNT: int = 600
"""Number of (frame_0, frame_1) pairs per contest video.

Per upstream evaluate.py + canonical 600-pair frame structure; mirrors
``mlx_renderer.NUM_PAIRS``.
"""

CONTEST_SCORER_RESOLUTION: tuple[int, int] = (384, 512)
"""(height, width) at which SegNet + PoseNet consume frames.

Decoder output resolution at training; bicubic upscale to camera (874, 1164)
applied at inflate-time per the contest contract. Mirrors
``mlx_renderer.EVAL_HW``.
"""

CONTEST_PAIR_RGB_SHAPE: tuple[int, int, int, int] = (2, 3, 384, 512)
"""(num_frames_per_pair, channels, H, W) tensor shape at scorer resolution.

The canonical per-pair RGB tensor every level's reconstruction loss is
ultimately compared against (modulo wavelet decomposition at coarser
levels).
"""


# ---------------------------------------------------------------------------
# Per-level dimension contract (frozen dataclass). One instance describes
# the EXPECTED shapes at hierarchy level i. The Z8 substrate constructs
# ``num_levels`` of these from ``Z8HierarchicalConfig`` at runtime; future
# pieces (Mamba-2 SSD, Mallat full DWT, Wyner-Ziv) consume them.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelDimensionContract:
    """Per-level dimension shape contract.

    A single hierarchy level i ∈ {0, 1, ..., num_levels-1} is described by
    one of these. Coarser levels (higher i) have fewer groups + fewer
    categories + smaller wavelet subband resolution per Mallat's
    scale-invariant entropy bound.

    The Z8 binding constructs ``num_levels`` instances of this dataclass
    from ``Z8HierarchicalConfig`` at trainer init; downstream pieces
    (Mamba-2 SSD state-dim selection; Mallat per-level subband sizing;
    Wyner-Ziv top-level entropy budgeting) consume the contract.

    Attributes:
        level_index: Zero-based hierarchy level (0 = finest, num_levels-1 =
            coarsest). Top-level (== num_levels - 1) is where Wyner-Ziv
            coding lives.
        num_categorical_groups: DreamerV3 categorical groups G_l per level.
            Default canonical (24, 16, 8) per ``mlx_renderer``
            ``DEFAULT_NUM_GROUPS_PER_LEVEL``.
        num_categorical_classes: DreamerV3 alphabet size K_l per level.
            Default canonical (256, 128, 64) per
            ``DEFAULT_NUM_CATEGORIES_PER_LEVEL``.
        deterministic_state_dim: Dimension of the recurrent deterministic
            state h_i. THIS is the slot Mamba-2 SSD replaces DreamerV3's
            GRUCell at. Default 64 per ``Z8HierarchicalConfig``.
        wavelet_subband_shape: (H_l, W_l) spatial shape of this level's
            Mallat wavelet detail subband. Coarsens by 2x per level in each
            dimension. At L0 SCAFFOLD this is the sum-pool proxy shape;
            Phase 2 will be the Daubechies-4 detail-subband shape.
        ego_motion_dim: Dimension of the ego-motion vector consumed at this
            level (canonical 6-DOF pose per Z6 sister pattern). Same across
            all levels.
        bit_budget_estimate: Predicted byte budget this level contributes
            to the archive (categorical indices + deterministic state side
            info + wavelet residuals, before Wyner-Ziv at top). The
            allocation is determined by upstream rate-distortion bound;
            this field is the planner's predicted target.

    Invariants enforced in ``__post_init__``:
        - level_index >= 0
        - num_categorical_groups > 0 and <= 256
        - num_categorical_classes in (0, 65536]
        - deterministic_state_dim >= 1
        - wavelet_subband_shape both dims > 0
        - ego_motion_dim >= 1
        - bit_budget_estimate >= 0
    """

    level_index: int
    num_categorical_groups: int
    num_categorical_classes: int
    deterministic_state_dim: int
    wavelet_subband_shape: tuple[int, int]
    ego_motion_dim: int = 6
    bit_budget_estimate: int = 0

    def __post_init__(self) -> None:
        if self.level_index < 0:
            raise ValueError(
                f"level_index must be >= 0; got {self.level_index}"
            )
        if not (0 < self.num_categorical_groups <= 256):
            raise ValueError(
                f"num_categorical_groups must be in (0, 256]; "
                f"got {self.num_categorical_groups}"
            )
        if not (0 < self.num_categorical_classes <= 65536):
            raise ValueError(
                f"num_categorical_classes must be in (0, 65536]; "
                f"got {self.num_categorical_classes}"
            )
        if self.deterministic_state_dim < 1:
            raise ValueError(
                f"deterministic_state_dim must be >= 1; "
                f"got {self.deterministic_state_dim}"
            )
        h, w = self.wavelet_subband_shape
        if h <= 0 or w <= 0:
            raise ValueError(
                f"wavelet_subband_shape dims must be > 0; "
                f"got {self.wavelet_subband_shape}"
            )
        if self.ego_motion_dim < 1:
            raise ValueError(
                f"ego_motion_dim must be >= 1; got {self.ego_motion_dim}"
            )
        if self.bit_budget_estimate < 0:
            raise ValueError(
                f"bit_budget_estimate must be >= 0; "
                f"got {self.bit_budget_estimate}"
            )

    @property
    def categorical_one_hot_size(self) -> int:
        """Flat size of the one-hot per-level categorical tensor.

        Equals ``num_categorical_groups * num_categorical_classes``.
        Downstream projection ``Linear(categorical_one_hot_size,
        decoder_latent_dim)`` consumes this.
        """
        return self.num_categorical_groups * self.num_categorical_classes

    @property
    def categorical_index_bytes_per_pair(self) -> int:
        """Per-pair archive byte cost for this level's category indices.

        u8 per group when K <= 256, else u16. Mirrors the canonical archive
        grammar in ``mlx_renderer.Z8HierarchicalConfig
        .total_latent_packing_bytes_per_pair``.
        """
        if self.num_categorical_classes <= 256:
            return self.num_categorical_groups
        return self.num_categorical_groups * 2


@dataclass(frozen=True)
class HierarchyBindingContract:
    """Full per-Z8-substrate binding contract across all hierarchy levels.

    One instance describes the complete Z8 binding: a tuple of
    LevelDimensionContract per level, plus top-level Wyner-Ziv parameters.

    Constructed from Z8HierarchicalConfig at trainer init; consumed by
    Mamba-2 SSD trainer (per-level state-dim sizing), Mallat full DWT
    (per-level subband shapes), Wyner-Ziv top-level coder (top-level state
    + side-info entropy budget), and the score-aware loss surface (per-level
    sensitivity-weighting).

    Invariants:
        - levels tuple has at least 1 entry
        - level_index values are 0..len(levels)-1 contiguous
        - top-level (last) wavelet_subband_shape (H, W) determines
          wyner_ziv_top_level_side_info_shape
    """

    levels: tuple[LevelDimensionContract, ...]
    wyner_ziv_top_level_side_info_shape: tuple[int, int, int]
    """(C, H, W) shape of the side-info tensor the Wyner-Ziv top-level
    coder consumes (typically frame_0's wavelet-reconstructed latent).
    """

    score_aware_loss_sensitivity_map_shape: tuple[int, int, int]
    """(C, H, W) shape of the empirical scorer-sensitivity map the
    score-aware loss consumes. C is typically 3 (RGB channels). H, W
    typically match ``CONTEST_SCORER_RESOLUTION``.

    This map is the UNIWARD-analog for THIS scorer (SegNet + PoseNet +
    YUV6 + pair-structure) and is the Yousfi-grounding piece of Z8.
    """

    def __post_init__(self) -> None:
        if len(self.levels) < 1:
            raise ValueError("HierarchyBindingContract requires >= 1 level")
        for expected_idx, level in enumerate(self.levels):
            if level.level_index != expected_idx:
                raise ValueError(
                    f"levels[{expected_idx}] has level_index={level.level_index}; "
                    f"must be {expected_idx} (contiguous from 0)"
                )
        c, h, w = self.wyner_ziv_top_level_side_info_shape
        if c <= 0 or h <= 0 or w <= 0:
            raise ValueError(
                f"wyner_ziv_top_level_side_info_shape dims must be > 0; "
                f"got {self.wyner_ziv_top_level_side_info_shape}"
            )
        sc, sh, sw = self.score_aware_loss_sensitivity_map_shape
        if sc <= 0 or sh <= 0 or sw <= 0:
            raise ValueError(
                f"score_aware_loss_sensitivity_map_shape dims must be > 0; "
                f"got {self.score_aware_loss_sensitivity_map_shape}"
            )

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def top_level(self) -> LevelDimensionContract:
        return self.levels[-1]

    @property
    def total_categorical_index_bytes_per_pair(self) -> int:
        return sum(
            level.categorical_index_bytes_per_pair for level in self.levels
        )


# ---------------------------------------------------------------------------
# Behavioral Protocols. Each Phase-2 piece implements the relevant
# Protocol(s) with matching shapes from the LevelDimensionContract +
# HierarchyBindingContract instances. Pieces never import each other;
# they import this module and target its contracts.
# ---------------------------------------------------------------------------


@runtime_checkable
class DeterministicStateUpdate(Protocol):
    """Per-level recurrent state update interface.

    The DreamerV3-style GRUCell currently occupying step 5 of Z8's
    forward pass implements this; Mamba-2 SSD (Dao & Gu 2024) is the
    Phase-2 replacement. The interface is framework-agnostic: MLX
    arrays, PyTorch tensors, or numpy arrays all satisfy `Any`.

    Implementations decide their own framework but MUST match the
    LevelDimensionContract.deterministic_state_dim sizing.

    Mamba-2 SSD specifically REPLACES the GRU because SSD's
    matrix-A-shared-across-channels parameterization produces a
    byte-friendlier state structure that compresses better when the
    top-level Wyner-Ziv coder consumes the state as side info. The
    state-dim is the SAME (matches contract); the parameterization
    is different.

    Per Catalog #290 ADOPT_CANONICAL_BECAUSE_SERVES for the GRU at
    L0 SCAFFOLD; FORK_BECAUSE_PRINCIPLED_MISMATCH at Phase 2 (the
    GRU's matrix-A-per-channel is the suppression; SSD's
    matrix-A-shared is the optimization for our specific
    Wyner-Ziv-compatibility requirement).
    """

    @property
    def state_dim(self) -> int:
        """Must equal contract.deterministic_state_dim for this level."""
        ...

    def initial_state(self, batch_size: int) -> Any:
        """Produce initial state for a batch of size ``batch_size``.

        Shape: (batch_size, state_dim).
        """
        ...

    def step(self, prior_state: Any, input_at_t: Any) -> Any:
        """Single-step recurrence: prior_state @ t-1, input @ t -> state @ t.

        Shape contract:
            prior_state: (batch_size, state_dim)
            input_at_t: (batch_size, input_dim) — implementation-defined
                input_dim
            return: (batch_size, state_dim)
        """
        ...


@runtime_checkable
class WaveletPartition(Protocol):
    """Per-level Mallat wavelet partition interface.

    L0 SCAFFOLD currently uses sum-pool 2x proxy; Phase 2 lands full
    Daubechies-4 DWT per Mallat 1989. The Protocol defines the contract
    both implementations must satisfy: decompose an input tensor into
    one approximation subband + one detail subband at the next coarser
    scale; recompose the inverse for inflate-time.

    Per Catalog #290: ADOPT_CANONICAL sum-pool for L0; FORK at Phase 2
    to full Daubechies-4 because the principled mismatch (sum-pool
    loses high-frequency information that downstream level's residual
    encoder needs to compress) suppresses score.
    """

    def decompose_to_next_level(self, x: Any) -> tuple[Any, Any]:
        """Decompose tensor at scale i into (approximation, detail) at scale i+1.

        Shape contract: input x shape (B, H, W, C); return
        (approximation: (B, H/2, W/2, C), detail: (B, H/2, W/2, C)).

        L0 SCAFFOLD detail is None or zeros; Phase 2 detail is the actual
        Daubechies-4 detail subband.
        """
        ...

    def recompose_from_next_level(
        self,
        approximation: Any,
        detail: Any,
    ) -> Any:
        """Inverse decompose: (approx, detail) at scale i+1 -> tensor at scale i.

        Shape contract: inputs (B, H/2, W/2, C) -> return (B, H, W, C).
        Phase 2 invariant: ``recompose(*decompose(x)) ≈ x`` to wavelet
        reconstruction tolerance (exact for Daubechies-4 with sufficient
        boundary handling).
        """
        ...


@runtime_checkable
class WynerZivTopLevelCoder(Protocol):
    """Top-level conditional coding interface (Wyner-Ziv 1976).

    Only the top level (level_index == num_levels - 1) implements this.
    The top-level latent is coded against side info (typically frame_0's
    wavelet-reconstructed latent at top scale); the conditional entropy
    H(top_state | side_info) is what gets encoded into archive bytes.

    L0 SCAFFOLD sketches the conditional entropy estimate; Phase 2 lands
    full coding with bit-allocation matched to the per-level dimension
    contract's bit_budget_estimate.

    Per Catalog #290: FORK at Phase 2 because no canonical helper
    implements Wyner-Ziv coding against arbitrary side info; sister
    ``tac.codec.wyner_ziv_layer`` is the closest substrate but operates
    at the pipeline-stage surface, not the per-substrate top-level
    surface Z8 needs.
    """

    @property
    def side_info_shape(self) -> tuple[int, int, int]:
        """Must equal contract.wyner_ziv_top_level_side_info_shape."""
        ...

    def encode(self, top_state: Any, side_info: Any) -> bytes:
        """Encode top_state conditioned on side_info into archive bytes.

        Shape contract:
            top_state: (B, deterministic_state_dim)
            side_info: (B, C, H, W) per side_info_shape
            return: variable-length bytes payload
        """
        ...

    def decode(self, payload: bytes, side_info: Any) -> Any:
        """Decode payload back to top_state given side_info.

        Inverse of encode; must round-trip to acceptable distortion (the
        Wyner-Ziv rate-distortion bound is the achievable target).
        """
        ...


@runtime_checkable
class ScoreAwareLevelLoss(Protocol):
    """Per-level Yousfi-grounded score-aware loss interface.

    THIS is where the empirical scorer-sensitivity map (the UNIWARD-analog
    for SegNet + PoseNet + YUV6 + pair-structure) gets embedded at each
    hierarchy level. Per the Yousfi grounding:

        loss_at_level_i = sum_pixel(
            scorer_sensitivity_at_pixel_at_level_i
            * reconstruction_error_at_pixel
        )

    NOT generic L2. The bit budget gets spent where the scorer is
    actually sensitive — exactly Yousfi's "find the detector's blind
    spots and embed there" methodology.

    L0 SCAFFOLD does not implement this (no _full_main yet); Phase 2
    lands the first instance using Slot GGG's per-pixel-roll SegNet-null
    finding as the first empirical anchor in the sensitivity map.

    The map itself comes from `tac.master_gradient` empirical per-pixel
    sensitivity measurement (cross-substrate per Slot AA + sister
    work).

    Per Catalog #290: FORK at Phase 2 because no canonical helper
    embeds per-level scorer-sensitivity weighting at the hierarchy
    surface; sister `tac.composition.*_inverse_steganalysis_*` packages
    operate at the per-archive-bolt-on surface, not the per-level
    integrated surface Z8 needs.
    """

    def per_level_loss(
        self,
        reconstruction: Any,
        target: Any,
        scorer_sensitivity_map: Any,
    ) -> Any:
        """Compute per-level loss weighted by empirical scorer sensitivity.

        Shape contract:
            reconstruction: (B, C, H, W) at this level's resolution
            target: (B, C, H, W) at this level's resolution
            scorer_sensitivity_map: (B, C, H, W) or broadcast-compatible;
                values are non-negative weights (higher = scorer more
                sensitive at this pixel/channel)
            return: scalar tensor (loss).

        Implementations must satisfy: integral over uniform sensitivity
        (sensitivity_map == 1 everywhere) reduces to standard L2/L1
        reconstruction loss. Non-uniform sensitivity reweights the
        per-pixel contribution.
        """
        ...


# ---------------------------------------------------------------------------
# Convenience: build a HierarchyBindingContract from the canonical
# Z8HierarchicalConfig defaults. Sister ``mlx_renderer.py`` provides the
# config; this constructor lives in the contract module so the contract
# can be tested without importing MLX.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Z8ConfigShape:
    """Duck-typed minimal subset of Z8HierarchicalConfig we need.

    Sister ``mlx_renderer.Z8HierarchicalConfig`` has these fields; we
    declare the minimal shape here so the contract can be exercised
    in tests without importing MLX. Production callers pass the real
    Z8HierarchicalConfig.
    """

    num_levels: int
    num_groups_per_level: tuple[int, ...]
    num_categories_per_level: tuple[int, ...]
    deterministic_state_dim: int
    eval_size: tuple[int, int]
    ego_motion_dim: int = 6


def build_canonical_contract_from_config(
    config: Any,
    *,
    wavelet_top_level_shape: tuple[int, int] | None = None,
    side_info_channels: int = 28,
) -> HierarchyBindingContract:
    """Construct a HierarchyBindingContract from a Z8HierarchicalConfig.

    Per-level wavelet subband shapes are derived by halving the eval_size
    once per level (the canonical Mallat pyramid; level 0 gets full
    eval_size, each deeper level halves H + W).

    Args:
        config: A Z8HierarchicalConfig (or duck-typed _Z8ConfigShape).
            Required attributes: num_levels, num_groups_per_level,
            num_categories_per_level, deterministic_state_dim,
            eval_size, ego_motion_dim.
        wavelet_top_level_shape: Optional override for the top-level
            wavelet subband shape. If None, derived from eval_size by
            halving num_levels-1 times.
        side_info_channels: Number of channels in the Wyner-Ziv side-info
            tensor (the wavelet-reconstructed frame_0 latent). Default 28
            matches decoder_latent_dim canonical default.

    Returns:
        HierarchyBindingContract with len(levels) == config.num_levels.

    Raises:
        ValueError if config has missing attributes or invariants fail.
    """
    required_attrs = (
        "num_levels",
        "num_groups_per_level",
        "num_categories_per_level",
        "deterministic_state_dim",
        "eval_size",
        "ego_motion_dim",
    )
    missing = [a for a in required_attrs if not hasattr(config, a)]
    if missing:
        raise ValueError(
            f"config missing required attributes: {missing}; "
            f"expected Z8HierarchicalConfig or compatible shape"
        )

    num_levels = int(config.num_levels)
    base_h, base_w = config.eval_size
    levels: list[LevelDimensionContract] = []
    for i in range(num_levels):
        # Halve H, W i times for the per-level subband shape.
        h_l = max(1, base_h // (2**i))
        w_l = max(1, base_w // (2**i))
        levels.append(
            LevelDimensionContract(
                level_index=i,
                num_categorical_groups=int(config.num_groups_per_level[i]),
                num_categorical_classes=int(config.num_categories_per_level[i]),
                deterministic_state_dim=int(config.deterministic_state_dim),
                wavelet_subband_shape=(h_l, w_l),
                ego_motion_dim=int(config.ego_motion_dim),
            )
        )

    if wavelet_top_level_shape is None:
        wavelet_top_level_shape = levels[-1].wavelet_subband_shape
    top_h, top_w = wavelet_top_level_shape

    return HierarchyBindingContract(
        levels=tuple(levels),
        wyner_ziv_top_level_side_info_shape=(side_info_channels, top_h, top_w),
        score_aware_loss_sensitivity_map_shape=(3, base_h, base_w),
    )


__all__ = [
    "CONTEST_PAIR_COUNT",
    "CONTEST_SCORER_RESOLUTION",
    "CONTEST_PAIR_RGB_SHAPE",
    "LevelDimensionContract",
    "HierarchyBindingContract",
    "DeterministicStateUpdate",
    "WaveletPartition",
    "WynerZivTopLevelCoder",
    "ScoreAwareLevelLoss",
    "build_canonical_contract_from_config",
]
