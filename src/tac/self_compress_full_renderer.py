"""PARADIGM-zeta - full-renderer self-compression NN (Phase 1 scaffold).

This module is the **Phase 1 scaffold** for the zeta paradigm in the PARADIGM-deltaepsilonzeta
blueprint (see ``.omx/research/paradigm_delta_epsilon_zeta_phase1_blueprint_20260507_claude.md``).

Design summary
--------------
The current sibling :mod:`tac.self_compress` applies self-compression to a
~46KB postfilter via :class:`tac.self_compress.SelfCompressingPostFilter`. zeta
extends self-compression to the **full renderer** - the 88K-param
JointFrameGenerator (or DEN/PSD variants) that drives the contest-CUDA score.

Self-compression loop (Csefalvay arXiv 2301.13142):
    1. Train ``R_theta`` normally (typically via delta joint-training).
    2. ``swap_renderer_convs_with_self_compress()`` replaces every eligible
       ``nn.Conv2d`` with :class:`tac.self_compress.SelfCompressingConv2d`,
       which has a learnable per-channel bit-depth ``b_l``.
    3. Fine-tune with rate penalty
       ``L_rate = lambda_sc * sum_l (b_l * params_l)`` - STE allows gradients
       through the discrete bit-depth quantization.
    4. **At least 2000 QAT steps** (Selfcomp revision section 302; 500 leaves bit
       allocation at ~3.5 bpw, far from the 1.5 bpw target).
    5. At export: prune channels with ``b_l < 0.5``; pack remaining at the
       learned bit-depth into ``renderer.bin``.

Wire format
-----------
The exported archive section is identified by magic ``b"ZETA"``:

    bytes 0..3:    magic ``MAGIC_ZETA``
    bytes 4..7:    uint32 little-endian config-header length
    bytes 8..n:    JSON config header (per-layer channel counts +
                   per-layer bit-depths + arch-fingerprint)
    bytes n+1..:   packed weight data (variable bit-depth per channel)

FiLM protection (Hotz/Dykstra clarification section 301)
-------------------------------------------------
FiLM gamma/beta layers are protected at TRAINING TIME ONLY. Self-compression
quantizing the FiLM affines collapses cross-frame conditioning (the lane M
"FiLM modulation = small + scorer-sensitive" finding). At ARCHIVE TIME,
FiLM affines are baked into renderer weights and are not separately stored.

The protection is implemented via name-pattern matching on the qualified
module name. Any layer whose name contains one of
:data:`DEFAULT_FILM_PROTECT_PATTERNS` is skipped during the
``swap_renderer_convs_with_self_compress`` walk.

CLAUDE.md compliance
--------------------
- **Strict-scorer-rule**: no scorer load at inflate time. The export blob is
  pure weights + config; loading it reconstructs the renderer with NO
  scorer dependencies.
- **No /tmp paths**: training artifacts go to
  ``experiments/results/lane_zeta_self_compress_renderer_<timestamp>/``.
- **No silent defaults**: every required field of
  :class:`FullRendererSelfCompressConfig` is explicit; ``protect_patterns``
  always includes the FiLM canonical set.

Implementation status (Phase 1)
-------------------------------
Phase 1 lands:
    - ``MAGIC_ZETA``, ``DEFAULT_FILM_PROTECT_PATTERNS`` constants
    - ``FullRendererSelfCompressConfig`` dataclass + validation
    - ``layer_name_matches_film_protect_pattern`` helper (pure string match;
      fully implemented)
    - ``FullRendererSelfCompress`` ``__init__`` accepts config + renderer
      reference; methods raise NotImplementedError
    - ``train_full_renderer_self_compress`` raises NotImplementedError
    - ``export_full_renderer_self_compress`` /
      ``load_full_renderer_self_compress`` raise NotImplementedError

Phase 2 will land the full QAT loop + export/load roundtrip, gated behind
delta Phase-2 [contest-CUDA] eval (Gate 3).

References
----------
- Blueprint: ``.omx/research/paradigm_delta_epsilon_zeta_phase1_blueprint_20260507_claude.md``
- Postfilter baseline: :mod:`tac.self_compress`
  (``SelfCompressingPostFilter``, ``SC_PROTECTED_NAME_PATTERNS``)
- Csefalvay 2023 arXiv 2301.13142 - original learnable per-channel bit-depth
- Lane S 2026-04-27 conv swap landing: ``swap_renderer_convs_with_self_compress``
- FiLM-protection rationale: Lane M memo (FiLM = small + scorer-sensitive)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "DEFAULT_FILM_PROTECT_PATTERNS",
    "DEFAULT_QAT_STEPS",
    "MAGIC_ZETA",
    "FullRendererSelfCompress",
    "FullRendererSelfCompressConfig",
    "FullRendererSelfCompressError",
    "export_full_renderer_self_compress",
    "layer_name_matches_film_protect_pattern",
    "load_full_renderer_self_compress",
    "train_full_renderer_self_compress",
]


# -- Wire-format / training constants ----------------------------------


# Magic byte identifier for the zeta archive section. 4 ASCII bytes matching
# the project's section-magic convention (``FP4A``, ``ASYM``, ``DPSM``,
# ``LEPR``, ``ZETA``).
MAGIC_ZETA: bytes = b"ZETA"


# Selfcomp revision section 302: 2000-step minimum for QAT. Below this, bit
# allocations sit near initialization (~3.5 bpw) far from the optimal
# 1.5 bpw target. Operators may sweep upward (3000-5000) but never below
# 2000 - Phase 2 enforcement raises.
DEFAULT_QAT_STEPS: int = 2000


# Canonical FiLM-protection pattern set. Matches by case-INSENSITIVE
# substring against qualified module names. Includes both common naming
# conventions (``film``, ``cond``, ``gamma``, ``beta``) and PSD-specific
# names (``scale``, ``shift``). This is intentionally broad - false-positive
# protection (a non-FiLM layer accidentally protected) costs <= a few
# hundred bytes; false-negative protection (missing a real FiLM layer)
# silently collapses cross-frame conditioning.
#
# Cross-ref: :data:`tac.self_compress.SC_PROTECTED_NAME_PATTERNS` lists the
# specific renderer paths used by the postfilter swap; zeta adopts a
# substring-based pattern set so it works across renderer arch variants
# (DEN, PSD, JointFrameGenerator) without hardcoding paths.
DEFAULT_FILM_PROTECT_PATTERNS: tuple[str, ...] = (
    "film",
    "cond",
    "gamma",
    "beta",
    "scale",
    "shift",
)


class FullRendererSelfCompressError(ValueError):
    """Raised when full-renderer self-compression inputs are malformed."""


# -- Pure-string FiLM-name matcher (Phase 1 fully implemented) -----------


def layer_name_matches_film_protect_pattern(
    qualified_name: str,
    *,
    patterns: tuple[str, ...] = DEFAULT_FILM_PROTECT_PATTERNS,
    case_sensitive: bool = False,
) -> bool:
    """Return True iff a layer name should be protected as FiLM-class.

    Substring match (case-insensitive by default). A qualified name like
    ``"renderer.film_decoder.scale"`` matches because it contains
    ``"film"`` and ``"scale"``.

    Args (all required-keyword for the optional ones):
        qualified_name: the dotted module path (typically from
            ``model.named_modules()``).
        patterns: tuple of substring patterns to match against. Default
            :data:`DEFAULT_FILM_PROTECT_PATTERNS`.
        case_sensitive: if False (default), patterns and name are compared
            case-folded.

    Returns:
        True if any pattern is a substring of the qualified name.

    Raises:
        :class:`FullRendererSelfCompressError` on bad inputs.
    """
    if not isinstance(qualified_name, str):
        raise FullRendererSelfCompressError(
            f"qualified_name must be a str; got {type(qualified_name).__name__}"
        )
    if not isinstance(patterns, tuple):
        raise FullRendererSelfCompressError(
            f"patterns must be a tuple of str; got {type(patterns).__name__}"
        )
    if not all(isinstance(p, str) for p in patterns):
        raise FullRendererSelfCompressError(
            f"every pattern must be a str; got {patterns!r}"
        )
    if len(patterns) == 0:
        # Empty pattern set never matches; explicit check so callers cannot
        # accidentally bypass FiLM protection by passing ().
        return False

    if case_sensitive:
        haystack = qualified_name
        needles = patterns
    else:
        haystack = qualified_name.casefold()
        needles = tuple(p.casefold() for p in patterns)

    return any(needle in haystack for needle in needles)


# -- Configuration dataclass --------------------------------------------


@dataclass
class FullRendererSelfCompressConfig:
    """Required-keyword config for :class:`FullRendererSelfCompress`.

    Args:
        target_bits_total: Total renderer-bytes target (informational; the
            actual final size depends on the learned bit allocation +
            arithmetic-coding overhead). Phase 2 may use this as a
            Lagrangian target via ``compute_renderer_rate_penalty``.
        qat_steps: Number of QAT fine-tune steps. MUST be >= 2000
            (Selfcomp section 302). Default ``DEFAULT_QAT_STEPS`` = 2000.
        lambda_rate_sc: Coefficient on the rate penalty
            ``sum_l (b_l * params_l)``. Larger -> smaller archive at cost of
            distortion.
        protect_film_layers: If True (default), apply
            :data:`DEFAULT_FILM_PROTECT_PATTERNS` protection during the
            conv swap. Setting False is FORBIDDEN unless the operator has
            explicitly verified FiLM layers are absent (e.g., DEN profile
            with ``pose_dim=0``). Phase 2 raises if False.
        protect_patterns: Tuple of substring patterns to FiLM-protect.
            Default :data:`DEFAULT_FILM_PROTECT_PATTERNS`.
        film_unprotect_override: required literal override string if
            ``protect_film_layers`` is False. This prevents accidental
            unprotected FiLM compression.
        bit_depth_init: Initial bit-depth value for swapped layers
            (passed through to :class:`tac.self_compress.LearnableBitDepth`).
            Must be in (0, 8].
        prune_threshold: Channels with ``b_l < prune_threshold`` are pruned
            at export. Default 0.5 (Csefalvay convention).
        notes: Free-form provenance string (council ref, dispatch label).
    """

    target_bits_total: int
    """Bytes-budget hint; Phase 2 may enforce as Lagrangian target."""

    qat_steps: int = DEFAULT_QAT_STEPS
    """QAT fine-tune steps; MUST be >= 2000."""

    lambda_rate_sc: float = 1e-7
    """Lagrangian coefficient on the rate penalty. Default tuned for the
    88K-param renderer; sweep range is [1e-8, 1e-5]."""

    protect_film_layers: bool = True
    """If True, FiLM-class layers are skipped during the conv swap. Default
    True; setting False without operator override raises in Phase 2."""

    protect_patterns: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_FILM_PROTECT_PATTERNS
    )
    """Substring patterns for FiLM protection; default
    :data:`DEFAULT_FILM_PROTECT_PATTERNS`."""

    film_unprotect_override: str = ""
    """Required explicit override if ``protect_film_layers`` is False."""

    bit_depth_init: float = 8.0
    """Initial bit-depth for swapped layers; default 8 = full int8."""

    prune_threshold: float = 0.5
    """Channels with ``b_l < prune_threshold`` pruned at export."""

    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.target_bits_total, int) or self.target_bits_total < 1:
            raise FullRendererSelfCompressError(
                f"target_bits_total must be a positive int; got "
                f"{self.target_bits_total!r}"
            )
        if not isinstance(self.qat_steps, int) or self.qat_steps < DEFAULT_QAT_STEPS:
            raise FullRendererSelfCompressError(
                f"qat_steps must be an int >= {DEFAULT_QAT_STEPS} "
                f"(Selfcomp revision section 302 - below this the bit allocation "
                f"sits at ~3.5 bpw far from optimal); got {self.qat_steps!r}"
            )
        if (
            not isinstance(self.lambda_rate_sc, (int, float))
            or self.lambda_rate_sc <= 0
        ):
            raise FullRendererSelfCompressError(
                f"lambda_rate_sc must be a positive number; got "
                f"{self.lambda_rate_sc!r}"
            )
        if not isinstance(self.protect_film_layers, bool):
            raise FullRendererSelfCompressError(
                f"protect_film_layers must be a bool; got "
                f"{type(self.protect_film_layers).__name__}"
            )
        if not isinstance(self.protect_patterns, tuple):
            raise FullRendererSelfCompressError(
                f"protect_patterns must be a tuple of str; got "
                f"{type(self.protect_patterns).__name__}"
            )
        if not all(isinstance(p, str) for p in self.protect_patterns):
            raise FullRendererSelfCompressError(
                f"every protect_pattern must be a str; got "
                f"{self.protect_patterns!r}"
            )
        if self.protect_film_layers:
            if len(self.protect_patterns) == 0:
                raise FullRendererSelfCompressError(
                    "protect_patterns must not be empty when protect_film_layers=True"
                )
            # Custom patterns are additive, never replacements for the default
            # FiLM protection set.
            self.protect_patterns = tuple(
                dict.fromkeys(DEFAULT_FILM_PROTECT_PATTERNS + self.protect_patterns)
            )
        elif self.film_unprotect_override != "ALLOW_UNPROTECTED_FILM_COMPRESSION":
            raise FullRendererSelfCompressError(
                "protect_film_layers=False requires "
                "film_unprotect_override='ALLOW_UNPROTECTED_FILM_COMPRESSION'"
            )
        if (
            not isinstance(self.bit_depth_init, (int, float))
            or not (0 < self.bit_depth_init <= 8)
        ):
            raise FullRendererSelfCompressError(
                f"bit_depth_init must be in (0, 8]; got {self.bit_depth_init!r}"
            )
        if (
            not isinstance(self.prune_threshold, (int, float))
            or not (0 < self.prune_threshold < 1)
        ):
            raise FullRendererSelfCompressError(
                f"prune_threshold must be in (0, 1); got {self.prune_threshold!r}"
            )


# -- Module stub (architecture documented; methods Phase 2 pending) ------


class FullRendererSelfCompress(nn.Module):
    """Full-renderer self-compression wrapper (Phase 2 implementation pending).

    Phase 1 (this module): constructor accepts a renderer reference + a
    validated config; method calls raise NotImplementedError.

    Phase 2 (pending Gate 3): the constructor will:
        1. Run ``swap_renderer_convs_with_self_compress(renderer,
           protected_patterns=cfg.protect_patterns + sc.SC_PROTECTED_NAME_PATTERNS)``
           with FiLM-class names protected.
        2. Initialize :class:`tac.self_compress.LearnableBitDepth` modules
           on every swapped Conv2d.
        3. Hold a reference to the swapped renderer for the QAT loop.

    Args (required-keyword):
        renderer: a built JointFrameGenerator / DEN / PSD renderer module.
        config: validated :class:`FullRendererSelfCompressConfig`.
    """

    def __init__(
        self,
        *,
        renderer: nn.Module,
        config: FullRendererSelfCompressConfig,
    ) -> None:
        super().__init__()
        if not isinstance(renderer, nn.Module):
            raise FullRendererSelfCompressError(
                f"renderer must be an nn.Module; got {type(renderer).__name__}"
            )
        if not isinstance(config, FullRendererSelfCompressConfig):
            raise FullRendererSelfCompressError(
                f"config must be a FullRendererSelfCompressConfig; got "
                f"{type(config).__name__}"
            )
        self.renderer = renderer
        self.config = config

    def swap_renderer_convs_with_self_compress(
        self,
    ) -> dict[str, Any]:  # pragma: no cover
        """Replace eligible Conv2d layers with SelfCompressingConv2d.

        Phase 2 implementation will:
            - Walk ``renderer.named_modules()``.
            - For each ``nn.Conv2d``, check
              :func:`layer_name_matches_film_protect_pattern` - protected
              layers stay FP32.
            - Skip transposed conv + grouped conv per
              :func:`tac.self_compress.swap_renderer_convs_with_self_compress`.
            - Replace surviving Conv2d with SelfCompressingConv2d, copying
              weights + biases.
            - Return diagnostics dict (swapped, protected, skipped lists +
              total swapped param count).

        Raises:
            NotImplementedError: pending Phase 2 (Gate 3).
        """
        raise NotImplementedError(
            "FullRendererSelfCompress.swap_renderer_convs_with_self_compress "
            "is Phase 2 (pending Gate 3: epsilon/zeta Phase 3 dispatch). The "
            "implementation MUST honor cfg.protect_patterns + "
            "tac.self_compress.SC_PROTECTED_NAME_PATTERNS - FiLM-class "
            "layers staying FP32 is what prevents cross-frame conditioning "
            "collapse (Lane M finding). See "
            "tac.self_compress.swap_renderer_convs_with_self_compress for "
            "the canonical postfilter version."
        )

    def forward(
        self, *args: Any, **kwargs: Any
    ) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError(
            "FullRendererSelfCompress.forward is Phase 2 (pending Gate 3). "
            "Will delegate to the swapped renderer's forward, which routes "
            "through SelfCompressingConv2d for non-FiLM layers."
        )


# -- Training / export / load - Phase 2 implementation pending -----------


def train_full_renderer_self_compress(
    *,
    model: FullRendererSelfCompress,
    frames: Any,
    scorers: Any,
    config: FullRendererSelfCompressConfig,
) -> FullRendererSelfCompress:
    """Run the QAT fine-tune loop (Phase 2 implementation pending).

    Phase 2 will:
        - Compute reconstruction loss (delegating to delta's joint loss when
          composed; standalone uses pixel-MSE + SegNet-KL).
        - Compute rate penalty
          ``L_rate = lambda_sc * sum(b_l * params_l)`` over swapped layers.
        - Optimize for >= ``cfg.qat_steps`` steps with EMA(0.997) and
          eval_roundtrip per CLAUDE.md non-negotiables.
        - Periodically check anti-collapse criterion
          (``>= 50% channels with b_l > 1.0``); raise on collapse.
        - Return the trained model.

    Args (required-keyword):
        model: :class:`FullRendererSelfCompress` instance with swapped convs.
        frames: training frame tensor (compress-time data; details Phase 2).
        scorers: SegNet/PoseNet scorer pair (compress-time only;
            strict-scorer-rule).
        config: same config used to build ``model`` (must match).

    Raises:
        NotImplementedError: pending Phase 2 (Gate 3).
    """
    raise NotImplementedError(
        "train_full_renderer_self_compress is Phase 2 (pending Gate 3: epsilon/zeta "
        "Phase 3 dispatch). The implementation MUST: (a) optimize for >= "
        f"cfg.qat_steps (>= {DEFAULT_QAT_STEPS}) - Selfcomp revision section 302, "
        "below this bit allocations sit at ~3.5 bpw far from optimal; (b) "
        "use EMA(0.997) + eval_roundtrip per CLAUDE.md non-negotiables; "
        "(c) check anti-collapse (>=50% channels with b_l > 1.0) before "
        "export; (d) NEVER load scorers into archive (strict-scorer-rule)."
    )


def export_full_renderer_self_compress(
    model: FullRendererSelfCompress,
) -> bytes:  # pragma: no cover
    """Export the trained model as a ``ZETA``-section archive blob.

    Phase 2 will produce the wire format documented at the top of this
    module (magic + uint32 length + JSON header + packed weights). Raises
    if total bytes exceed the operator-set size budget OR if the channel
    pruning threshold leaves a layer with 0 channels (architectural
    collapse).

    Raises:
        NotImplementedError: pending Phase 2 (Gate 3).
    """
    raise NotImplementedError(
        "export_full_renderer_self_compress is Phase 2 (pending Gate 3). "
        "Wire format: MAGIC_ZETA (4) + uint32 header_len (4) + JSON header "
        "(per-layer channel counts + per-layer bit-depths + arch-fingerprint) "
        "+ packed weight data. Channels with b_l < cfg.prune_threshold are "
        "pruned. Raises on collapse (any layer with 0 active channels) "
        "or oversize archive."
    )


def load_full_renderer_self_compress(
    blob: bytes,
    *,
    arch_config: Any,
) -> nn.Module:  # pragma: no cover
    """Reconstruct a renderer from a ``ZETA`` archive blob.

    Phase 2 will validate ``MAGIC_ZETA`` header, parse the JSON header, and
    rebuild the renderer with each layer's channels at their stored
    bit-depth. **No scorer load** anywhere in this path
    (strict-scorer-rule).

    Args (required-keyword for ``arch_config``):
        blob: archive section bytes.
        arch_config: renderer architecture config (base_ch, mid_ch, depth,
            etc.) needed to rebuild the parent :class:`build_renderer`
            module before swapping in the packed weights.

    Raises:
        NotImplementedError: pending Phase 2 (Gate 3).
    """
    raise NotImplementedError(
        "load_full_renderer_self_compress is Phase 2 (pending Gate 3). "
        "Plan: validate MAGIC_ZETA, parse uint32 header_len, parse JSON "
        "config (per-layer channels + bit-depths), rebuild the renderer "
        "from arch_config, swap in SelfCompressingConv2d layers with the "
        "stored bit-depths + packed weights. NO scorer load - pure-CPU "
        "reconstruction (strict-scorer-rule)."
    )
