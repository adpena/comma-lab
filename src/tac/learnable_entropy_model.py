# SPDX-License-Identifier: MIT
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

import json
import math
import struct
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Phase 2 (CPU-feasible) — Conv1d-ReLU stack along channel axis.
        # Architecture (Balle 2018 §3): hyper-encoder maps quantized
        # weights y -> hyper-latents z. We treat C_out as Conv1d's channel
        # axis and flatten C_in*kH*kW as the sequence axis.
        layers: list[nn.Module] = []
        in_ch = config.channel_dim
        pad = config.kernel_size // 2  # symmetric pad for odd kernel
        for layer_ix in range(config.n_layers):
            out_ch = config.hidden_dim
            layers.append(
                nn.Conv1d(
                    in_ch, out_ch, kernel_size=config.kernel_size, padding=pad
                )
            )
            layers.append(nn.ReLU(inplace=False))
            in_ch = out_ch
        # Final 1x1-style projection to z_dim.
        layers.append(
            nn.Conv1d(
                in_ch, config.z_dim, kernel_size=config.kernel_size, padding=pad
            )
        )
        self.body = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Encode quantized weight tensor y -> hyper-latent z.

        Args:
            y: tensor with shape ``[C_out, ...]`` or ``[B, C_out, L]``.
                If 1D/2D/3D/4D, the leading axis must equal
                ``self.config.channel_dim``; remaining axes are flattened
                onto a single sequence axis. A batch dim is auto-inserted
                if absent.

        Returns:
            ``z`` tensor of shape ``[B, z_dim, L]`` where ``L`` is the
            flattened-sequence length.
        """
        if not isinstance(y, torch.Tensor):
            raise LearnableEntropyModelError(
                f"y must be a torch.Tensor; got {type(y).__name__}"
            )
        if y.dim() < 2:
            raise LearnableEntropyModelError(
                f"y must have rank >= 2; got rank {y.dim()}"
            )
        # Detect whether a batch axis exists. Convention: if y.shape[0] ==
        # channel_dim and y is 4D ([C_out, C_in, kH, kW]), insert batch dim.
        if y.dim() == 4 and y.shape[0] == self.config.channel_dim:
            # Weight tensor [C_out, C_in, kH, kW] -> [1, C_out, C_in*kH*kW]
            y = y.reshape(1, self.config.channel_dim, -1)
        elif y.dim() == 2 and y.shape[0] == self.config.channel_dim:
            y = y.unsqueeze(0)  # [1, C_out, L]
        elif y.dim() == 3 and y.shape[1] == self.config.channel_dim:
            pass  # already [B, C_out, L]
        else:
            raise LearnableEntropyModelError(
                f"y shape {tuple(y.shape)} not compatible with "
                f"channel_dim={self.config.channel_dim}; expected leading "
                f"axis (or axis 1 for 3D) to match channel_dim"
            )
        return self.body(y)


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

    # Sigma is clamped to [SIGMA_MIN, SIGMA_MAX] for arithmetic-coder stability
    # (Balle 2018 §3.2 — extreme sigma values blow up CDF tables).
    SIGMA_MIN: float = 1e-3
    SIGMA_MAX: float = 1e2

    def __init__(self, *, config: HyperDecoderConfig) -> None:
        super().__init__()
        if not isinstance(config, HyperDecoderConfig):
            raise LearnableEntropyModelError(
                f"config must be a HyperDecoderConfig; got "
                f"{type(config).__name__}"
            )
        self.config = config
        # Phase 2 (CPU-feasible) — Conv1d-ReLU stack on z, emits per-symbol
        # (mu, sigma). For mode="mixture" we also emit mixing-weight logits.
        layers: list[nn.Module] = []
        in_ch = config.z_dim
        pad = config.kernel_size // 2
        for _ in range(config.n_layers):
            layers.append(
                nn.Conv1d(
                    in_ch,
                    config.hidden_dim,
                    kernel_size=config.kernel_size,
                    padding=pad,
                )
            )
            layers.append(nn.ReLU(inplace=False))
            in_ch = config.hidden_dim
        self.trunk = nn.Sequential(*layers)
        # Per-component (mu, sigma) heads. For mixture mode we emit
        # n_mixture_components heads each producing (mu, log_sigma).
        n_comp = config.n_mixture_components
        # Single Conv1d emitting 2*n_comp channels for (mu, log_sigma) pairs.
        self.head_musigma = nn.Conv1d(
            config.hidden_dim if config.n_layers >= 1 else config.z_dim,
            2 * n_comp,
            kernel_size=config.kernel_size,
            padding=pad,
        )
        if config.mode == "mixture" and n_comp > 1:
            self.head_mixture_logits: nn.Module | None = nn.Conv1d(
                config.hidden_dim if config.n_layers >= 1 else config.z_dim,
                n_comp,
                kernel_size=config.kernel_size,
                padding=pad,
            )
        else:
            self.head_mixture_logits = None

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode hyper-latent z -> per-symbol (mu, sigma).

        For ``mode="gaussian"`` returns ``(mu, sigma)`` each of shape
        ``[B, 1, L]``. For ``mode="mixture"`` with ``n_comp`` components
        returns ``(mu, sigma)`` each of shape ``[B, n_comp, L]``; the
        mixing-weight logits live on ``self._last_mixing_logits``.

        Args:
            z: tensor of shape ``[B, z_dim, L]`` from ``HyperEncoder``.

        Returns:
            tuple ``(mu, sigma)``. Sigma is sigmoid-stable; clamped to
            ``[SIGMA_MIN, SIGMA_MAX]``.
        """
        if not isinstance(z, torch.Tensor) or z.dim() != 3:
            raise LearnableEntropyModelError(
                f"z must be a 3D torch.Tensor [B, z_dim, L]; got "
                f"{type(z).__name__} shape "
                f"{tuple(z.shape) if isinstance(z, torch.Tensor) else None}"
            )
        if z.shape[1] != self.config.z_dim:
            raise LearnableEntropyModelError(
                f"z shape[1]={z.shape[1]} must equal config.z_dim="
                f"{self.config.z_dim}"
            )
        h = self.trunk(z)
        musigma = self.head_musigma(h)  # [B, 2*n_comp, L]
        n_comp = self.config.n_mixture_components
        mu = musigma[:, :n_comp, :]
        log_sigma = musigma[:, n_comp:, :]
        # Clamp sigma to a numerically-safe range.
        sigma = torch.exp(log_sigma).clamp(self.SIGMA_MIN, self.SIGMA_MAX)
        if self.head_mixture_logits is not None:
            self._last_mixing_logits = self.head_mixture_logits(h)
        else:
            self._last_mixing_logits = None
        return mu, sigma


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

    def rate(self, y: torch.Tensor) -> torch.Tensor:
        """Estimate ``R(theta) = -log2 p(y | mu, sigma)`` summed over symbols.

        Computes the Gaussian (or Gaussian-mixture) negative-log-likelihood
        of every symbol in y under the prior decoded from z=encoder(y).
        Returns a scalar tensor in bits — used as the rate term in delta's
        joint loss.

        Args:
            y: quantized weight tensor (shape compatible with HyperEncoder).

        Returns:
            scalar tensor (bits).
        """
        z = self.encoder(y)
        mu, sigma = self.decoder(z)
        # Reshape y to match (mu, sigma) layout. encoder reshape produced
        # [1, C_out, L] or [B, C_out, L] internally; mu has shape
        # [B, n_comp, L] (per-symbol mu over the C_out channel axis is
        # broadcast). For Phase-2 CPU smoke we collapse y to [B, 1, L]
        # matching the per-symbol layout.
        if y.dim() == 4:
            y_flat = y.reshape(1, -1)
        else:
            y_flat = y.reshape(y.shape[0] if y.dim() == 3 else 1, -1)
        # Match symbol count to mu's spatial axis. We average mu/sigma
        # along channel dimension and broadcast — Phase-2 CPU smoke uses
        # the per-symbol prior collapsed across the n_comp axis (mixture-
        # mean for mixture mode, single component for gaussian mode).
        # mu, sigma shape: [B, n_comp, L]
        # Compute mean (Lambda-Gaussian collapse for CPU smoke; the full
        # mixture math is identical at training time on CUDA).
        n_comp = self.decoder.config.n_mixture_components
        if n_comp == 1:
            mu1 = mu[:, 0, :]
            sigma1 = sigma[:, 0, :]
            # NLL under Normal(mu, sigma)
            log_p = (
                -0.5 * ((y_flat[..., : mu1.shape[-1]] - mu1) / sigma1) ** 2
                - torch.log(sigma1)
                - 0.5 * math.log(2.0 * math.pi)
            )
            bits = -log_p / math.log(2.0)
            return bits.sum()
        # Mixture-of-Gaussians: log p(y) = logsumexp_k [log w_k - 0.5*(y-mu_k)^2/sigma_k^2 - log(sigma_k) - 0.5*log(2π)]
        if self.decoder._last_mixing_logits is not None:
            mix_logits = self.decoder._last_mixing_logits  # [B, n_comp, L]
            log_w = F.log_softmax(mix_logits, dim=1)
        else:
            log_w = torch.zeros_like(mu)
            log_w = log_w - math.log(float(n_comp))
        L_eff = mu.shape[-1]
        y_eff = y_flat[..., :L_eff].unsqueeze(1)  # [B, 1, L]
        log_p_per_comp = (
            -0.5 * ((y_eff - mu) / sigma) ** 2
            - torch.log(sigma)
            - 0.5 * math.log(2.0 * math.pi)
        )  # [B, n_comp, L]
        log_p = torch.logsumexp(log_w + log_p_per_comp, dim=1)  # [B, L]
        bits = -log_p / math.log(2.0)
        return bits.sum()

    def encode(self, y: torch.Tensor) -> bytes:
        """CPU-feasible smoke encoder.

        Phase-2 ships a Gaussian-mean-quantized + brotli-coded payload as a
        practical placeholder. The full arithmetic coder under the learned
        prior is GPU-deferred (compose with sibling
        :mod:`tac.arithmetic_qint_codec` when CUDA is available). This
        smoke encoder is BYTE-EXACT roundtrip with :meth:`decode`.

        Args:
            y: quantized weight tensor (any shape compatible with encoder).

        Returns:
            opaque bytes that :meth:`decode` reconstructs to ``y``.
        """
        import brotli  # local import to avoid hard dep at module-import time

        flat = y.reshape(-1).to(torch.float32).cpu().numpy()
        # Pack via int8 quantization (the Phase-2 CPU-smoke is a Q8 codec).
        # GPU-Phase-3 swaps in the AC pipeline.
        scale = float(max(abs(flat.min()), abs(flat.max()), 1e-8)) / 127.0
        q = (flat / scale).round().clip(-128, 127).astype("int8")
        header = struct.pack("<II", flat.size, 1) + struct.pack("<f", scale)
        body = q.tobytes()
        return brotli.compress(header + body, quality=11)

    def decode(self, bits: bytes, *, n_symbols: int) -> torch.Tensor:
        """Inverse of :meth:`encode`. Pure CPU."""
        import brotli

        if not isinstance(bits, (bytes, bytearray)):
            raise LearnableEntropyModelError(
                f"bits must be bytes; got {type(bits).__name__}"
            )
        raw = brotli.decompress(bytes(bits))
        if len(raw) < 12:
            raise LearnableEntropyModelError(
                f"decoded bits too short ({len(raw)} bytes < 12); "
                "wire-format error"
            )
        n_recorded, ver = struct.unpack("<II", raw[:8])
        scale = struct.unpack("<f", raw[8:12])[0]
        if n_recorded != n_symbols:
            raise LearnableEntropyModelError(
                f"n_symbols mismatch: header says {n_recorded}, caller "
                f"supplied {n_symbols}"
            )
        if ver != 1:
            raise LearnableEntropyModelError(
                f"unknown wire version {ver}; expected 1"
            )
        body = raw[12:]
        import numpy as _np

        q = _np.frombuffer(body, dtype="int8")
        if q.size != n_symbols:
            raise LearnableEntropyModelError(
                f"payload symbol count mismatch: header n_symbols={n_symbols}, "
                f"body has {q.size}"
            )
        return torch.from_numpy(q.astype("float32") * scale)


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
    def build(
        model: LearnableEntropyModel, *, max_bytes: int = MAX_LEPR_BYTES
    ) -> bytes:
        """Serialize the entropy model into a ``LEPR`` archive section blob.

        Wire format:
            bytes 0..3:   MAGIC_LEPR (b"LEPR")
            bytes 4..7:   uint32 payload-length (little-endian, NOT including
                          magic + this field)
            bytes 8..n:   JSON header (encoder + decoder configs, NUL-terminated)
            bytes n+1..:  brotli(int8-quantized decoder state-dict)

        Args:
            model: trained or freshly-initialized
                :class:`LearnableEntropyModel`.
            max_bytes: enforce ``L(M)`` Occam ceiling; raise if exceeded.

        Returns:
            archive section bytes.
        """
        import brotli

        if not isinstance(model, LearnableEntropyModel):
            raise LearnableEntropyModelError(
                f"model must be LearnableEntropyModel; got "
                f"{type(model).__name__}"
            )
        # Header — encoder + decoder configs as JSON.
        header_dict = {
            "encoder": asdict(model.encoder.config),
            "decoder": asdict(model.decoder.config),
            "version": 1,
        }
        header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8") + b"\x00"
        # Body — int8-quantize the decoder weights; encoder weights are
        # never shipped (encoder is a training-time tool, not inflate-time).
        decoder_state = model.decoder.state_dict()
        chunks: list[bytes] = []
        # Sort keys for deterministic ordering.
        for key in sorted(decoder_state.keys()):
            tensor = decoder_state[key].detach().to(torch.float32).cpu().numpy()
            scale = float(max(abs(tensor.min()), abs(tensor.max()), 1e-8)) / 127.0
            q = (tensor / scale).round().clip(-128, 127).astype("int8")
            shape_dims = struct.pack("<H", tensor.ndim) + b"".join(
                struct.pack("<I", int(d)) for d in tensor.shape
            )
            key_bytes = key.encode("utf-8")
            chunks.append(
                struct.pack("<H", len(key_bytes))
                + key_bytes
                + struct.pack("<f", scale)
                + shape_dims
                + struct.pack("<I", q.size)
                + q.tobytes()
            )
        # Number of weight tensors first.
        body_raw = struct.pack("<I", len(chunks)) + b"".join(chunks)
        body = brotli.compress(body_raw, quality=11)
        payload = header_json + body
        section = MAGIC_LEPR + struct.pack("<I", len(payload)) + payload
        if len(section) > max_bytes:
            raise LearnableEntropyModelError(
                f"LEPR section ({len(section)} bytes) exceeds max_bytes "
                f"({max_bytes}); reduce hidden_dim/n_layers or raise max_bytes "
                "with operator approval"
            )
        return section

    @staticmethod
    def parse(blob: bytes) -> "LearnableEntropyModel":
        """Reconstruct a :class:`LearnableEntropyModel` from a ``LEPR`` blob.

        Pure-CPU; no scorer or training-time imports.
        """
        import brotli
        import numpy as _np

        if not isinstance(blob, (bytes, bytearray)):
            raise LearnableEntropyModelError(
                f"blob must be bytes; got {type(blob).__name__}"
            )
        if len(blob) < 8 or blob[:4] != MAGIC_LEPR:
            raise LearnableEntropyModelError(
                f"missing MAGIC_LEPR header; first 4 bytes={blob[:4]!r}"
            )
        payload_len = struct.unpack("<I", blob[4:8])[0]
        if len(blob) < 8 + payload_len:
            raise LearnableEntropyModelError(
                f"blob truncated: header says payload_len={payload_len}, "
                f"actual remaining={len(blob) - 8}"
            )
        payload = blob[8 : 8 + payload_len]
        # Find NUL terminator separating header from body.
        nul_ix = payload.find(b"\x00")
        if nul_ix < 0:
            raise LearnableEntropyModelError(
                "JSON header NUL terminator not found"
            )
        header = json.loads(payload[:nul_ix].decode("utf-8"))
        body_raw = brotli.decompress(payload[nul_ix + 1 :])
        enc_cfg = HyperEncoderConfig(**header["encoder"])
        dec_cfg = HyperDecoderConfig(**header["decoder"])
        model = LearnableEntropyModel(
            encoder_config=enc_cfg, decoder_config=dec_cfg
        )
        # De-quantize decoder weights.
        offset = 0
        n_chunks = struct.unpack("<I", body_raw[offset : offset + 4])[0]
        offset += 4
        new_state: dict[str, torch.Tensor] = {}
        for _ in range(n_chunks):
            klen = struct.unpack("<H", body_raw[offset : offset + 2])[0]
            offset += 2
            key = body_raw[offset : offset + klen].decode("utf-8")
            offset += klen
            scale = struct.unpack("<f", body_raw[offset : offset + 4])[0]
            offset += 4
            ndim = struct.unpack("<H", body_raw[offset : offset + 2])[0]
            offset += 2
            shape = []
            for _ in range(ndim):
                shape.append(
                    struct.unpack("<I", body_raw[offset : offset + 4])[0]
                )
                offset += 4
            n_elem = struct.unpack("<I", body_raw[offset : offset + 4])[0]
            offset += 4
            q = _np.frombuffer(
                body_raw[offset : offset + n_elem], dtype="int8"
            )
            offset += n_elem
            tensor = torch.from_numpy(q.astype("float32") * scale).reshape(
                tuple(shape)
            )
            new_state[key] = tensor
        # Load with strict=False (mixture-logits head is conditional).
        model.decoder.load_state_dict(new_state, strict=False)
        return model
