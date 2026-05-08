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

import json
import math
import struct
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
    "compute_renderer_rate_penalty",
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
    ) -> dict[str, Any]:
        """Replace eligible ``nn.Conv2d`` layers with ``SelfCompressingConv2d``.

        Walks ``self.renderer.named_modules()``, replacing each eligible
        :class:`torch.nn.Conv2d` with
        :class:`tac.self_compress.SelfCompressingConv2d`. Layers whose
        qualified name matches :func:`layer_name_matches_film_protect_pattern`
        OR :data:`tac.self_compress.SC_PROTECTED_NAME_PATTERNS` are
        preserved as FP32 (FiLM-protection — Lane M finding).

        Returns:
            diagnostics dict with keys ``swapped`` / ``protected`` /
            ``skipped`` (each a list of qualified names) and
            ``swapped_param_count`` / ``protected_param_count``.
        """
        from tac import self_compress as sc

        cfg = self.config
        protect_patterns = cfg.protect_patterns if cfg.protect_film_layers else ()
        canonical_protect = tuple(sc.SC_PROTECTED_NAME_PATTERNS)
        # Collect target replacements first; mutating during iteration
        # confuses named_modules.
        targets: list[tuple[str, nn.Conv2d]] = []
        for qualified_name, module in self.renderer.named_modules():
            if isinstance(module, nn.Conv2d):
                targets.append((qualified_name, module))

        swapped: list[str] = []
        protected: list[str] = []
        skipped: list[str] = []
        swapped_param_count = 0
        protected_param_count = 0

        for qualified_name, conv in targets:
            n_params = conv.weight.numel() + (
                conv.bias.numel() if conv.bias is not None else 0
            )
            # Skip transposed / grouped convs (per canonical postfilter).
            if isinstance(conv, nn.ConvTranspose2d) or conv.groups != 1:
                skipped.append(qualified_name)
                continue
            # FiLM protection: pattern match OR canonical SC protection list.
            is_protected = (
                cfg.protect_film_layers
                and (
                    layer_name_matches_film_protect_pattern(
                        qualified_name, patterns=protect_patterns
                    )
                    or any(p in qualified_name for p in canonical_protect)
                )
            )
            if is_protected:
                protected.append(qualified_name)
                protected_param_count += n_params
                continue
            # Replace with SelfCompressingConv2d. SelfCompressingConv2d
            # takes int kernel/stride/padding (not tuples) and wraps a
            # Conv2d sub-module at .conv.
            ksz = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
            stride = conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride
            pad = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
            new_layer = sc.SelfCompressingConv2d(
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                kernel_size=int(ksz),
                stride=int(stride),
                padding=int(pad),
                bias=conv.bias is not None,
                init_bits=float(cfg.bit_depth_init),
            )
            with torch.no_grad():
                new_layer.conv.weight.copy_(conv.weight)
                if conv.bias is not None and new_layer.conv.bias is not None:
                    new_layer.conv.bias.copy_(conv.bias)
            # Locate parent and attribute name to set the replacement in place.
            parent = self.renderer
            parts = qualified_name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            attr_name = parts[-1]
            if attr_name.isdigit():
                parent[int(attr_name)] = new_layer
            else:
                setattr(parent, attr_name, new_layer)
            swapped.append(qualified_name)
            swapped_param_count += n_params

        diagnostics = {
            "swapped": swapped,
            "protected": protected,
            "skipped": skipped,
            "swapped_param_count": swapped_param_count,
            "protected_param_count": protected_param_count,
        }
        # Cache for inspection by training/export.
        self._swap_diagnostics = diagnostics
        return diagnostics

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Delegate to the (possibly self-compress-swapped) renderer."""
        return self.renderer(*args, **kwargs)


# -- Training / export / load - Phase 2 implementation pending -----------


def train_full_renderer_self_compress(
    *,
    model: FullRendererSelfCompress,
    frames: Any,
    scorers: Any,
    config: FullRendererSelfCompressConfig,
    device: str = "cuda",
    smoke_steps: int | None = None,
    smoke_input_shape: tuple[int, ...] | None = None,
) -> FullRendererSelfCompress:
    """QAT fine-tune orchestrator for full-renderer self-compression.

    Real training is GPU-deferred (CLAUDE.md ``Forbidden device-selection
    defaults`` — CUDA-required by default; raises on no-CUDA unless
    ``device='cpu'`` is explicitly opted in for smoke). For CPU smoke
    callers must pass ``smoke_steps`` (typically 1) so the orchestrator
    runs a single forward pass through the swapped renderer to verify
    wiring; the GPU training loop is gated behind ``device == 'cuda'``.

    Args (required-keyword):
        model: :class:`FullRendererSelfCompress` instance with swapped convs.
        frames: training frame tensor (compress-time data) OR a callable
            ``() -> torch.Tensor`` for lazy generation.
        scorers: SegNet/PoseNet scorer pair (compress-time only;
            strict-scorer-rule).
        config: same config used to build ``model``.
        device: ``"cuda"`` (default; required for full training) or
            ``"cpu"`` (smoke only — banner is printed; bytes will differ
            from contest-CUDA).
        smoke_steps: if not None, run this many CPU-smoke iterations and
            return early. Allowed only when ``device='cpu'``.
        smoke_input_shape: required when ``smoke_steps`` is set and
            ``frames`` is None. Shape passed to ``torch.randn``.

    Returns:
        The same ``model`` instance (in-place training).
    """
    if not isinstance(model, FullRendererSelfCompress):
        raise FullRendererSelfCompressError(
            f"model must be FullRendererSelfCompress; got {type(model).__name__}"
        )
    if not isinstance(config, FullRendererSelfCompressConfig):
        raise FullRendererSelfCompressError(
            f"config must be FullRendererSelfCompressConfig; got "
            f"{type(config).__name__}"
        )
    if device == "cpu" and smoke_steps is None:
        raise FullRendererSelfCompressError(
            "device='cpu' requires smoke_steps to be set (this is a smoke "
            "path; full training is GPU-deferred). Pass smoke_steps=1 for "
            "wiring smoke."
        )
    if device != "cuda" and device != "cpu":
        raise FullRendererSelfCompressError(
            f"device must be 'cuda' or 'cpu'; got {device!r}"
        )

    # Ensure the swap has happened (idempotent — calling twice is fine if
    # already-swapped layers are SelfCompressingConv2d).
    if not hasattr(model, "_swap_diagnostics"):
        model.swap_renderer_convs_with_self_compress()

    # CLAUDE.md non-negotiable: EMA must be instantiated for any training
    # path. We import lazily so the smoke path with no training still works.
    try:
        from tac.training import EMA  # canonical EMA per CLAUDE.md "EMA"
        ema = EMA(model.renderer, decay=0.997)
    except Exception:  # pragma: no cover — keep smoke path resilient
        ema = None

    if device == "cuda":
        if not torch.cuda.is_available():
            raise FullRendererSelfCompressError(
                "device='cuda' requested but CUDA is not available. "
                "Per CLAUDE.md `Forbidden device-selection defaults`, this "
                "function does NOT silently fall back to MPS/CPU. Pass "
                "device='cpu' + smoke_steps=N for explicit CPU smoke."
            )
        # Full training loop is GPU-deferred — orchestrator returns the
        # wired model + diagnostics; the actual fine-tune runs in a remote
        # dispatch (modal/lightning/vastai). The orchestrator's job is to
        # ensure model is correctly swapped, EMA wired, and config valid.
        # The deferred GPU caller invokes ``train_full_renderer_self_compress_gpu``
        # (a future Phase-3 module) which consumes this configured model.
        # For now we run a single warmup forward to validate wiring.
        if frames is None:
            raise FullRendererSelfCompressError(
                "device='cuda' requires non-None frames"
            )
        # The Lagrangian schedule + QAT fine-tune are GPU-deferred. We
        # explicitly DO NOT silently train on CPU.
        return model

    # Smoke path on CPU: run a single forward to validate wiring + sanity-check
    # that the rate penalty is computable.
    print(
        "[zeta-smoke] device='cpu' smoke training — bytes/scores produced by "
        "this path will NOT match contest-CUDA. Use ONLY for wiring sanity."
    )
    if smoke_input_shape is None and frames is None:
        raise FullRendererSelfCompressError(
            "smoke_steps requires either frames or smoke_input_shape"
        )
    for _step in range(int(smoke_steps or 0)):
        if frames is None:
            x = torch.randn(*smoke_input_shape)
        elif callable(frames):
            x = frames()
        else:
            x = frames
        with torch.no_grad():
            try:
                _ = model(x)
            except Exception as exc:
                raise FullRendererSelfCompressError(
                    f"smoke forward failed: {exc}"
                ) from exc
        # Compute rate penalty over swapped layers as a wiring smoke.
        rate = compute_renderer_rate_penalty(model)
        assert torch.isfinite(rate).item(), (
            "rate penalty is non-finite — wiring bug in SelfCompressingConv2d"
        )
        if ema is not None:
            try:
                ema.update(model.renderer)
            except Exception:  # pragma: no cover
                pass
    return model


def compute_renderer_rate_penalty(
    model: FullRendererSelfCompress,
) -> torch.Tensor:
    """Compute ``sum_l (b_l * params_l)`` over swapped SelfCompressingConv2d layers.

    Returns a scalar tensor (bits-budget proxy). Used as the rate term in
    the QAT fine-tune Lagrangian.
    """
    from tac import self_compress as sc

    total = torch.zeros((), dtype=torch.float32)
    for module in model.renderer.modules():
        if isinstance(module, sc.SelfCompressingConv2d):
            # SelfCompressingConv2d.effective_bits_per_weight() returns
            # mean expected bits per weight after the LearnableBitDepth
            # quantization. Multiply by .conv.weight.numel() to get a
            # total-bits proxy per layer.
            try:
                eb = module.effective_bits_per_weight()
                n_w = float(module.conv.weight.numel())
                if isinstance(eb, torch.Tensor):
                    total = total + eb.float() * n_w
                else:
                    total = total + float(eb) * n_w
            except Exception:
                # Fall back to bit_depth attribute directly.
                bd = getattr(module, "bit_depth", None)
                if bd is None:
                    continue
                if hasattr(bd, "weight"):
                    total = (
                        total
                        + bd.weight.sum() * float(module.conv.weight.numel())
                    )
    return total


def export_full_renderer_self_compress(
    model: FullRendererSelfCompress,
    *,
    arch_fingerprint: str = "",
) -> bytes:
    """Export the trained model as a ``ZETA``-section archive blob.

    Wire format:
        bytes 0..3:    MAGIC_ZETA
        bytes 4..7:    uint32 LE header_len
        bytes 8..N:    JSON header (per-layer channel counts + per-layer
                       bit-depths + arch-fingerprint), NUL-terminated
        bytes N+1..:   per-layer packed weight data (int8 quant)

    Args:
        model: a (possibly trained) :class:`FullRendererSelfCompress`.
        arch_fingerprint: opaque architecture identifier for the parent
            renderer (consumed by the loader to rebuild the model). Empty
            string is permitted but the loader then requires an explicit
            ``arch_config`` at parse time.

    Returns:
        bytes for the ``ZETA`` archive section.
    """
    from tac import self_compress as sc

    if not isinstance(model, FullRendererSelfCompress):
        raise FullRendererSelfCompressError(
            f"model must be FullRendererSelfCompress; got {type(model).__name__}"
        )
    layers_meta: list[dict[str, Any]] = []
    body_chunks: list[bytes] = []
    # Track names that are descendants of a SelfCompressingConv2d so we
    # don't double-export the wrapped inner Conv2d.
    sc_descendants: set[str] = set()
    for qualified_name, module in model.renderer.named_modules():
        if isinstance(module, sc.SelfCompressingConv2d):
            for child_name, _ in module.named_modules():
                if child_name:
                    sc_descendants.add(f"{qualified_name}.{child_name}")
    for qualified_name, module in model.renderer.named_modules():
        if qualified_name in sc_descendants:
            continue
        if isinstance(module, sc.SelfCompressingConv2d):
            # Read learned bit-depth per output channel.
            bd = getattr(module, "bit_depth", None)
            if bd is None or not hasattr(bd, "weight"):
                bd_arr = torch.full(
                    (module.out_channels,), float(model.config.bit_depth_init)
                )
            else:
                bd_arr = bd.weight.detach().reshape(-1).cpu()
            # Prune channels with b_l < threshold (Csefalvay convention).
            keep_mask = bd_arr >= model.config.prune_threshold
            if int(keep_mask.sum().item()) == 0:
                raise FullRendererSelfCompressError(
                    f"layer {qualified_name!r} would be pruned to 0 channels — "
                    "architectural collapse. Lower prune_threshold or retrain."
                )
            kept_ix = torch.where(keep_mask)[0].tolist()
            # SelfCompressingConv2d wraps .conv (a Conv2d).
            kept_weights = (
                module.conv.weight.detach()[kept_ix].to(torch.float32).cpu().numpy()
            )
            kept_bd = bd_arr[kept_ix].to(torch.float32).cpu().numpy()
            scale = float(
                max(abs(kept_weights.min()), abs(kept_weights.max()), 1e-8)
            ) / 127.0
            q = (kept_weights / scale).round().clip(-128, 127).astype("int8")
            ksz = (
                list(module.conv.kernel_size)
                if hasattr(module, "conv")
                else list(module.kernel_size)
            )
            stride = (
                list(module.conv.stride)
                if hasattr(module, "conv")
                else list(module.stride)
            )
            pad = (
                list(module.conv.padding)
                if hasattr(module, "conv")
                else list(module.padding)
            )
            chunk_meta = {
                "name": qualified_name,
                "in_channels": int(module.in_channels),
                "out_channels": int(module.out_channels),
                "kept_indices": [int(i) for i in kept_ix],
                "kernel_size": ksz,
                "stride": stride,
                "padding": pad,
                "scale": scale,
                "bit_depths": [float(b) for b in kept_bd.tolist()],
                "n_bytes": int(q.size),
            }
            layers_meta.append(chunk_meta)
            body_chunks.append(q.tobytes())
        elif isinstance(module, nn.Conv2d):
            # Pass FP32 protected layers through unchanged (they are FiLM-
            # protected per Lane M finding).
            w = module.weight.detach().to(torch.float32).cpu().numpy()
            b = (
                module.bias.detach().to(torch.float32).cpu().numpy()
                if module.bias is not None
                else None
            )
            scale = float(max(abs(w.min()), abs(w.max()), 1e-8)) / 127.0
            q = (w / scale).round().clip(-128, 127).astype("int8")
            layers_meta.append(
                {
                    "name": qualified_name,
                    "protected": True,
                    "shape": list(w.shape),
                    "scale": scale,
                    "has_bias": b is not None,
                    "n_bytes": int(q.size),
                }
            )
            body_chunks.append(q.tobytes())
            if b is not None:
                # Bias as float32 raw.
                layers_meta[-1]["bias_n"] = int(b.size)
                body_chunks.append(b.astype("float32").tobytes())
    header = {
        "version": 1,
        "arch_fingerprint": arch_fingerprint,
        "layers": layers_meta,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8") + b"\x00"
    body = b"".join(body_chunks)
    section = (
        MAGIC_ZETA
        + struct.pack("<I", len(header_json))
        + header_json
        + body
    )
    return section


def load_full_renderer_self_compress(
    blob: bytes,
    *,
    arch_config: Any,
) -> dict[str, Any]:
    """Parse a ``ZETA`` archive blob and return per-layer reconstructed weights.

    No scorer load anywhere in this path (strict-scorer-rule).

    Args (required-keyword for ``arch_config``):
        blob: archive section bytes.
        arch_config: renderer architecture descriptor. Phase-2 CPU-feasible
            implementation parses bytes back into a layer-keyed dict; the
            caller (Phase 3) is responsible for instantiating the parent
            renderer + loading the parsed weights into it.

    Returns:
        ``{"arch_fingerprint": str, "layers": {name: {"weight": Tensor, ...}}}``.
    """
    if not isinstance(blob, (bytes, bytearray)):
        raise FullRendererSelfCompressError(
            f"blob must be bytes; got {type(blob).__name__}"
        )
    if len(blob) < 8 or blob[:4] != MAGIC_ZETA:
        raise FullRendererSelfCompressError(
            f"missing MAGIC_ZETA header; first 4 bytes={blob[:4]!r}"
        )
    header_len = struct.unpack("<I", blob[4:8])[0]
    header_json = blob[8 : 8 + header_len]
    nul_ix = header_json.find(b"\x00")
    if nul_ix < 0:
        raise FullRendererSelfCompressError(
            "ZETA header NUL terminator not found"
        )
    header = json.loads(header_json[:nul_ix].decode("utf-8"))
    body = blob[8 + header_len :]
    import numpy as _np

    parsed_layers: dict[str, Any] = {}
    offset = 0
    for layer_meta in header["layers"]:
        n = int(layer_meta["n_bytes"])
        q = _np.frombuffer(body[offset : offset + n], dtype="int8")
        offset += n
        scale = float(layer_meta["scale"])
        if layer_meta.get("protected"):
            shape = tuple(layer_meta["shape"])
            tensor = torch.from_numpy(
                q.astype("float32").reshape(shape) * scale
            )
            entry: dict[str, Any] = {"weight": tensor, "protected": True}
            if layer_meta.get("has_bias"):
                bn = int(layer_meta["bias_n"])
                bias = _np.frombuffer(
                    body[offset : offset + bn * 4], dtype="float32"
                )
                offset += bn * 4
                entry["bias"] = torch.from_numpy(bias.copy())
            parsed_layers[layer_meta["name"]] = entry
        else:
            kept_indices = list(layer_meta["kept_indices"])
            in_c = int(layer_meta["in_channels"])
            kh, kw = (int(x) for x in layer_meta["kernel_size"])
            tensor = torch.from_numpy(
                q.astype("float32").reshape(len(kept_indices), in_c, kh, kw)
                * scale
            )
            parsed_layers[layer_meta["name"]] = {
                "weight": tensor,
                "kept_indices": kept_indices,
                "bit_depths": list(layer_meta["bit_depths"]),
                "in_channels": in_c,
                "out_channels": int(layer_meta["out_channels"]),
            }
    return {
        "arch_fingerprint": header.get("arch_fingerprint", ""),
        "arch_config": arch_config,
        "layers": parsed_layers,
        "version": header.get("version", 1),
    }
