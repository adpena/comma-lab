"""PARADIGM-epsilon - MDL/Bayesian learned entropy prior codec (Phase 1 scaffold).

This module is the **Phase 1 scaffold** for the epsilon paradigm in the PARADIGM-deltaepsilonzeta
blueprint (see ``.omx/research/paradigm_delta_epsilon_zeta_phase1_blueprint_20260507_claude.md``).

Design summary
--------------
The current sibling :mod:`tac.mdl_bayesian_codec` is a meta-comparison
framework - it ranks codecs by MDL but does not produce archive bytes. epsilon
extends that scaffold to a **byte-producing codec** that ships a learned
prior in the archive and arithmetic-codes quantized weights under it.

Rate-prior-distortion trade-off (MacKay Ch. 28 + Balle 2018):

    L(D, M) = L(M) + L(D|M)
    L(D|M)  = -sum_i log2 p(y_i | mu_i, sigma_i)

The hyper-encoder/hyper-decoder pair learns ``p(y_i | mu_i, sigma_i)`` per
quantized weight ``y_i``, parameterized through hyper-latents ``z`` shipped
in ``renderer_prior.bin``.

Architectural choices (council-verdicted)
-----------------------------------------
- **HyperEncoder = 1D channel-wise Conv1d** along the channel axis of
  [C_out, C_in, kH, kW] tensors (Balle revision section 304: channel correlations
  dominate spatial in renderer weights). NOT a 2D depthwise-separable conv.
- **HyperDecoder = mixture-Gaussian** by default (Fridrich revision section 298:
  post-zeta weight distributions are multi-modal - pruned channels at 0;
  retained channels at variable bit-depth - and a single Gaussian fits
  poorly).
- **Spike-and-slab prior** (MacKay section 303) is a Phase 3 stretch goal, not
  Phase 1.
- **Magic byte ``b"LEPR"``** identifies the learned-prior section in the
  archive container. Section is OPTIONAL - if absent, callers fall back to
  a static Laplace(0,1) prior. This keeps deltaepsilonzeta archives backward-compatible
  with vanilla ``inflate.sh``.

Wire format (Phase 2 implementation pending)
--------------------------------------------
The ``LEPR`` archive section will contain:

    bytes 0..3:   magic ``LEPR``
    bytes 4..7:   ``uint32`` payload length (little-endian)
    bytes 8..n:   ``HyperDecoderConfig`` JSON header (terminated by NUL)
    bytes n+1..:  int8-quantized + brotli-compressed hyper-decoder weights

Constraint: total `LEPR` section bytes <= ``MAX_LEPR_BYTES`` (5000 by
default; council section 300 / risk #2). This is the ``L(M)`` term in the MDL
accounting.

CLAUDE.md compliance
--------------------
- **Strict-scorer-rule**: this codec ships at INFLATE TIME ONLY - there is
  no scorer call anywhere in the encode/decode path. Scorers are consulted
  only at COMPRESS TIME by the joint-loss training (delta-paradigm).
- **No silent defaults**: every required field of the configs must be set.
- **Pure-CPU encode/decode at inflate time**: arithmetic-coding tables are
  precomputed; inflate is feedforward MLP + table lookup, no torch ops.

Implementation status (Phase 1)
-------------------------------
Phase 1 lands the following:

- ``MAGIC_LEPR``, ``MAX_LEPR_BYTES`` constants (module-level)
- ``HyperEncoderConfig`` dataclass + validation
- ``HyperDecoderConfig`` dataclass + validation (mode literal, mixture count)
- ``HyperEncoder`` / ``HyperDecoder`` ``__init__`` raise NotImplementedError
  (the architectural ``Conv1d``-on-channel-axis is documented but not yet
  built)
- ``LearnableEntropyModel.{rate, encode, decode}`` raise NotImplementedError
- ``LearnableEntropyModelCodec`` archive-builder raises NotImplementedError

Phase 2 will land the full codec gated behind delta Phase-2 [contest-CUDA] eval
showing seg_dist OR pose_dist improvement (Gate 3).

References
----------
- Blueprint: ``.omx/research/paradigm_delta_epsilon_zeta_phase1_blueprint_20260507_claude.md``
- MDL framework scaffold: :mod:`tac.mdl_bayesian_codec`
- Vanilla Balle hyperprior (per-block sigma): :mod:`tac.balle_hyperprior_codec`
- Sensitivity-conditioned variant: :mod:`tac.balle_sensitivity_weighted`
- Balle 2018 ICLR - "Variational image compression with a scale hyperprior"
- MacKay 2003 ITILA - Ch. 28 (MDL), Ch. 41 (mixture models for compression)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

__all__ = [
    "MAGIC_LEPR",
    "MAX_LEPR_BYTES",
    "HyperDecoder",
    "HyperDecoderConfig",
    "HyperEncoder",
    "HyperEncoderConfig",
    "LearnableEntropyModel",
    "LearnableEntropyModelCodec",
    "LearnableEntropyModelError",
]


# -- Wire-format constants ----------------------------------------------


# Magic byte identifier for the learned-entropy-prior archive section.
# 4 ASCII bytes (matches the project's other section-magic convention -
# ``FP4A``, ``ASYM``, ``DPSM``, ``ZETA``, ``LEPR``). Defined as bytes (not
# str) so callers do a direct ``magic == b"LEPR"`` byte comparison.
MAGIC_LEPR: bytes = b"LEPR"

# Hard ceiling on the ``LEPR`` section size. ``L(M)`` must be small enough
# that ``L(M) + L(D|M) < L_baseline(D)`` - i.e. the prior must save more
# bits than it costs (Quantizr's "no 10KB MLP for 5KB savings" rule
# operationalized via :class:`tac.mdl_bayesian_codec.OccamCheck`).
# Default 5000 bytes ~= ``25*5000/37545489 = 0.0033`` rate points budget.
MAX_LEPR_BYTES: int = 5000


class LearnableEntropyModelError(ValueError):
    """Raised when learned-entropy-model inputs are malformed."""


# -- Configuration dataclasses ------------------------------------------


@dataclass
class HyperEncoderConfig:
    """Config for the HyperEncoder (y -> z) network.

    Architecture (Balle revision section 304): the hyper-encoder is a stack of
    **1D channel-wise convolutions** (``nn.Conv1d``) operating along the
    channel axis of flattened-spatial weight tensors. This exploits the
    strong channel correlations in renderer weights (post-FiLM
    conditioning). Spatial correlations are weak; a 2D depthwise-separable
    conv would underperform.

    Args:
        channel_dim: number of input channels (matches renderer weight
            tensor's C_out dimension after flatten). Required.
        hidden_dim: hidden-layer channel count. Required.
        n_layers: number of Conv1d-ReLU blocks. Must be >= 1.
        z_dim: hyper-latent dimensionality (``z`` channel count). Required.
        kernel_size: Conv1d kernel size along the channel axis. Default 3
            captures local channel correlations; 1 = pointwise; 5 = wider.
    """

    channel_dim: int
    hidden_dim: int
    n_layers: int
    z_dim: int
    kernel_size: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.channel_dim, int) or self.channel_dim < 1:
            raise LearnableEntropyModelError(
                f"channel_dim must be a positive int; got {self.channel_dim!r}"
            )
        if not isinstance(self.hidden_dim, int) or self.hidden_dim < 1:
            raise LearnableEntropyModelError(
                f"hidden_dim must be a positive int; got {self.hidden_dim!r}"
            )
        if not isinstance(self.n_layers, int) or self.n_layers < 1:
            raise LearnableEntropyModelError(
                f"n_layers must be a positive int; got {self.n_layers!r}"
            )
        if not isinstance(self.z_dim, int) or self.z_dim < 1:
            raise LearnableEntropyModelError(
                f"z_dim must be a positive int; got {self.z_dim!r}"
            )
        if not isinstance(self.kernel_size, int) or self.kernel_size < 1:
            raise LearnableEntropyModelError(
                f"kernel_size must be a positive int; got {self.kernel_size!r}"
            )
        if self.kernel_size % 2 == 0:
            raise LearnableEntropyModelError(
                f"kernel_size must be odd to allow symmetric padding; got "
                f"{self.kernel_size!r}"
            )


@dataclass
class HyperDecoderConfig:
    """Config for the HyperDecoder (z -> (mu, sigma)) network.

    Args:
        z_dim: hyper-latent dim from the encoder. Required.
        hidden_dim: hidden-layer width. Required.
        n_layers: Conv1d-ReLU block count. Must be >= 1.
        mode: ``"gaussian"`` (single-component) or ``"mixture"``
            (multi-component Gaussian - Fridrich revision section 298).
        n_mixture_components: number of components when ``mode="mixture"``.
            Must equal 1 when ``mode="gaussian"``. Default 1 - overridden
            by operator (e.g. 3 for a typical post-zeta multi-modal weight
            distribution).
        kernel_size: Conv1d kernel size; same convention as encoder.
    """

    z_dim: int
    hidden_dim: int
    n_layers: int
    mode: Literal["gaussian", "mixture"] = "gaussian"
    n_mixture_components: int = 1
    kernel_size: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.z_dim, int) or self.z_dim < 1:
            raise LearnableEntropyModelError(
                f"z_dim must be a positive int; got {self.z_dim!r}"
            )
        if not isinstance(self.hidden_dim, int) or self.hidden_dim < 1:
            raise LearnableEntropyModelError(
                f"hidden_dim must be a positive int; got {self.hidden_dim!r}"
            )
        if not isinstance(self.n_layers, int) or self.n_layers < 1:
            raise LearnableEntropyModelError(
                f"n_layers must be a positive int; got {self.n_layers!r}"
            )
        if self.mode not in ("gaussian", "mixture"):
            raise LearnableEntropyModelError(
                f"mode must be 'gaussian' or 'mixture'; got {self.mode!r}"
            )
        if not isinstance(self.n_mixture_components, int) or self.n_mixture_components < 1:
            raise LearnableEntropyModelError(
                f"n_mixture_components must be a positive int; got "
                f"{self.n_mixture_components!r}"
            )
        if self.mode == "gaussian" and self.n_mixture_components != 1:
            raise LearnableEntropyModelError(
                f"gaussian mode requires n_mixture_components=1; got "
                f"{self.n_mixture_components!r}. Use mode='mixture' for "
                f">1 components."
            )
        if not isinstance(self.kernel_size, int) or self.kernel_size < 1:
            raise LearnableEntropyModelError(
                f"kernel_size must be a positive int; got {self.kernel_size!r}"
            )
        if self.kernel_size % 2 == 0:
            raise LearnableEntropyModelError(
                f"kernel_size must be odd; got {self.kernel_size!r}"
            )


# -- Module stubs (architecture documented; forward Phase 2 pending) ----


class HyperEncoder(nn.Module):
    """1D channel-wise Conv1d hyper-encoder (Phase 2 implementation pending).

    Architecture (Balle section 304):
        - Reshape weight tensor ``y`` of shape [C_out, C_in, kH, kW] to
          ``[1, C_out, C_in*kH*kW]`` (treat C_out as Conv1d's channel axis,
          flatten spatial+input-channel as the length axis).
        - Stack ``cfg.n_layers`` of (Conv1d(channel_dim, hidden_dim, kernel) ->
          ReLU) blocks.
        - Final Conv1d(hidden_dim, z_dim, kernel) emits hyper-latent ``z``.
        - Quantize ``z`` to int8 for storage in ``renderer_prior.bin``.

    Phase 2 will instantiate the layer stack in ``__init__``; Phase 1 raises
    NotImplementedError so callers cannot accidentally use a half-built
    module.

    Args:
        config: validated :class:`HyperEncoderConfig`.
    """

    def __init__(self, *, config: HyperEncoderConfig) -> None:
        super().__init__()
        if not isinstance(config, HyperEncoderConfig):
            raise LearnableEntropyModelError(
                f"config must be a HyperEncoderConfig; got "
                f"{type(config).__name__}"
            )
        self.config = config
        # Phase 2 will build the Conv1d stack here. Phase 1 keeps the
        # constructor pure so test imports + dataclass instantiation work.

    def forward(self, y: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError(
            "HyperEncoder.forward is Phase 2 (pending Gate 3: epsilon/zeta Phase 3 "
            "dispatch). Architecture: reshape weight tensor to "
            "[1, C_out, C_in*kH*kW], apply Conv1d-ReLU stack along channel "
            "axis (NOT 2D conv), emit z hyper-latent. See module docstring "
            "for the full plan."
        )


class HyperDecoder(nn.Module):
    """Hyper-decoder (z -> per-symbol (mu, sigma)) - Phase 2 implementation pending.

    Architecture (Fridrich section 298):
        - Stack ``cfg.n_layers`` of Conv1d-ReLU blocks.
        - Final Conv1d outputs ``2 * n_mixture_components * z_seq_len``
          channels: ``n_components`` Gaussians x (mu, sigma) per symbol.
        - In ``mode="mixture"``, an additional softmax-mixing-weight head
          emits ``n_mixture_components`` logits per symbol.
        - Sigma clamped to [sigma_min, sigma_max] for arithmetic-coder stability.

    Args:
        config: validated :class:`HyperDecoderConfig`.
    """

    def __init__(self, *, config: HyperDecoderConfig) -> None:
        super().__init__()
        if not isinstance(config, HyperDecoderConfig):
            raise LearnableEntropyModelError(
                f"config must be a HyperDecoderConfig; got "
                f"{type(config).__name__}"
            )
        self.config = config

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError(
            "HyperDecoder.forward is Phase 2 (pending Gate 3). Architecture: "
            "Conv1d-ReLU stack on z, emit per-symbol (mu, sigma). For "
            "mode='mixture', also emit softmax-mixing-weight logits. Clamp sigma "
            "to [sigma_min, sigma_max] for arithmetic-coder stability."
        )


# -- Top-level codec ----------------------------------------------------


class LearnableEntropyModel(nn.Module):
    """Learned entropy model bundling HyperEncoder + HyperDecoder.

    The full encode/decode roundtrip will be:
        1. ``z = encode_hyper(y_quantized)`` (hyper-encoder, ship z as side info)
        2. ``(mu, sigma) = decode_hyper(z)`` (per-symbol prior parameters)
        3. ``rate(y) = -log2 p(y | mu, sigma)`` summed over symbols
        4. Arithmetic-code y under ``p(*|mu,sigma)`` for ``encode(y)``
        5. Arithmetic-decode bits under ``p(*|mu,sigma)`` for ``decode(bits)``

    Phase 1 instantiation works (test imports); ``rate`` / ``encode`` /
    ``decode`` raise NotImplementedError.

    Args:
        encoder_config: validated :class:`HyperEncoderConfig`.
        decoder_config: validated :class:`HyperDecoderConfig`. Must have
            ``z_dim`` matching the encoder's z_dim.
    """

    def __init__(
        self,
        *,
        encoder_config: HyperEncoderConfig,
        decoder_config: HyperDecoderConfig,
    ) -> None:
        super().__init__()
        if encoder_config.z_dim != decoder_config.z_dim:
            raise LearnableEntropyModelError(
                f"encoder_config.z_dim ({encoder_config.z_dim}) must match "
                f"decoder_config.z_dim ({decoder_config.z_dim})"
            )
        self.encoder = HyperEncoder(config=encoder_config)
        self.decoder = HyperDecoder(config=decoder_config)

    def rate(self, y: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError(
            "LearnableEntropyModel.rate is Phase 2 (pending Gate 3). Returns "
            "scalar tensor: -log2 p(y | mu, sigma) summed over symbols, where "
            "(mu, sigma) = decoder(encoder(y)). Used as the rate term in delta's "
            "joint loss."
        )

    def encode(self, y: torch.Tensor) -> bytes:  # pragma: no cover
        raise NotImplementedError(
            "LearnableEntropyModel.encode is Phase 2. Returns "
            "arithmetic-coded bytes for y under the learned prior. The "
            "decoder MUST receive z (shipped in renderer_prior.bin) to "
            "reconstruct (mu, sigma) for decoding."
        )

    def decode(
        self, bits: bytes, *, n_symbols: int
    ) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError(
            "LearnableEntropyModel.decode is Phase 2. Reconstructs y from "
            "arithmetic-coded bits using the same (mu, sigma) prior parameters "
            "computed at encode time. n_symbols MUST match the encode-side "
            "symbol count (mismatch is a wire-format error, not a recoverable "
            "decode error)."
        )


# -- Archive-section codec ----------------------------------------------


class LearnableEntropyModelCodec:
    """Builds the ``LEPR`` archive section (Phase 2 implementation pending).

    Wire format (planned):
        bytes 0..3:   magic ``MAGIC_LEPR`` (= b"LEPR")
        bytes 4..7:   uint32 little-endian payload length
        bytes 8..n:   JSON header (HyperDecoderConfig + quantization params,
                      NUL-terminated)
        bytes n+1..:  int8-quantized + brotli-compressed hyper-decoder state

    Total section size enforced <= ``MAX_LEPR_BYTES`` (5000) - Phase 2 raises
    :class:`LearnableEntropyModelError` if the resulting bytes exceed the
    cap. This is the L(M) Occam constraint operationalized.

    Methods:
        - ``build(model: LearnableEntropyModel) -> bytes``: produce the
          ``LEPR`` archive section bytes (Phase 2).
        - ``parse(blob: bytes) -> LearnableEntropyModel``: reconstruct the
          model from archive bytes (Phase 2). Pure-CPU, no torch ops.
    """

    @staticmethod
    def build(model: LearnableEntropyModel) -> bytes:  # pragma: no cover
        raise NotImplementedError(
            "LearnableEntropyModelCodec.build is Phase 2 (pending Gate 3). "
            "Plan: serialize HyperDecoderConfig as JSON header, int8-quantize "
            "+ brotli-compress decoder weights, prepend MAGIC_LEPR + uint32 "
            "payload length, raise if total bytes > MAX_LEPR_BYTES."
        )

    @staticmethod
    def parse(blob: bytes) -> LearnableEntropyModel:  # pragma: no cover
        raise NotImplementedError(
            "LearnableEntropyModelCodec.parse is Phase 2 (pending Gate 3). "
            "Plan: validate MAGIC_LEPR header, extract uint32 length, parse "
            "JSON header, decompress + de-quantize decoder state. Pure-CPU "
            "reconstruction (no torch ops at inflate time per "
            "strict-scorer-rule)."
        )
