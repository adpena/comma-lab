# SPDX-License-Identifier: MIT
"""Ballé 2018 ScaleHyperprior wrapper for PARADIGM-δεζ T1.

This module wraps ``compressai.entropy_models.EntropyBottleneck`` +
``compressai.layers.GaussianConditional`` into a thin, joint-trainable
``nn.Module`` that operates on the **per-pair latent table** the frozen A1
encoder learned (see :mod:`tac.paradigm_delta_epsilon_zeta.frozen_a1_encoder`).

The hyperprior plays two roles in T1:

1. **Rate prediction** at training time. The differentiable
   ``-log2 p(y|σ)`` objective gives the joint Lagrangian-ADMM coordinator
   a usable rate gradient.
2. **Side-information emission** at archive build time. The trained
   hyperprior compresses the latent table into the actual inflate-time
   bytes (z_strings + y_strings) consumed by the runtime decoder.

CompressAI dependency
---------------------

CompressAI is the canonical reference implementation of Ballé 2018
(``compressai.models.priors.ScaleHyperprior``). We import it directly rather
than re-implementing because:

- ``EntropyBottleneck`` carries the closed-form factorised prior + the
  range-coding entry points (``compress``, ``decompress``).
- ``GaussianConditional`` carries the conditional-Gaussian rate term + the
  AC entry points (``compress`` with scales, ``decompress``).
- Re-implementing these would introduce wire-format drift vs the published
  reference, which is precisely what we want to AVOID for first-time T1
  scaffolding (per CLAUDE.md "Beauty, simplicity, and developer experience"
  + "Deterministic packet compiler").

The wrapper exposes a narrow interface specific to T1's latent-table shape;
it does NOT re-export the full CompressAI surface.

CLAUDE.md compliance
--------------------

- ``aux_loss()`` (the EntropyBottleneck's quantile-loss) MUST be optimised
  on a separate optimiser per CompressAI's design. The trainer in
  ``experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py``
  wires this up; this module documents the contract.
- ``update()`` (which builds the entropy CDF tables) MUST be called before
  any ``compress()`` / ``decompress()`` invocation. The wrapper raises if
  ``compress()`` is called on a never-updated module.
- The wire format is deterministic given the trained weights + input — any
  drift here would break the T1 archive byte budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BalleHyperpriorConfig:
    """Configuration for the Ballé hyperprior wrapper.

    Attributes
    ----------
    y_channels : int
        Number of "y" channels in the analysis output (i.e. the latent
        dimensionality the GaussianConditional sees). For T1 we operate on
        the A1 latent table reshaped to ``(N_PAIRS, latent_dim)``, so we
        treat ``latent_dim`` as the channel count and ``N_PAIRS`` as the
        spatial extent. Default: 28.
    z_channels : int
        Number of "z" channels in the hyperprior summary. Default: 16.
    hyper_hidden : int
        Hidden width inside the hyper-encoder/decoder MLPs. Default: 32.
    scale_table_min : float
        Smallest scale in the GaussianConditional scale table. Default: 0.11
        (matches CompressAI default).
    scale_table_max : float
        Largest scale in the scale table. Default: 256.
    scale_table_levels : int
        Number of log-spaced scale levels in the table. Default: 64.
    """

    y_channels: int = 28
    z_channels: int = 16
    hyper_hidden: int = 32
    scale_table_min: float = 0.11
    scale_table_max: float = 256.0
    scale_table_levels: int = 64


def _get_default_scale_table(cfg: BalleHyperpriorConfig) -> torch.Tensor:
    """Log-spaced scale table for the GaussianConditional.

    Matches CompressAI's ``get_scale_table()`` shape but exposed here so the
    config is the single source of truth.
    """
    return torch.exp(
        torch.linspace(
            torch.log(torch.tensor(cfg.scale_table_min)),
            torch.log(torch.tensor(cfg.scale_table_max)),
            cfg.scale_table_levels,
        )
    )


class _CompressaiUnavailableError(RuntimeError):
    """Raised if compressai is not importable in the runtime environment."""


def _import_compressai() -> Any:
    try:
        import compressai
        from compressai.entropy_models import EntropyBottleneck, GaussianConditional
    except ImportError as exc:
        raise _CompressaiUnavailableError(
            "compressai is required for tac.paradigm_delta_epsilon_zeta."
            "balle_hyperprior. Install via `uv pip install compressai==1.2.8`."
        ) from exc
    return {
        "compressai": compressai,
        "EntropyBottleneck": EntropyBottleneck,
        "GaussianConditional": GaussianConditional,
    }


class _HyperEncoderMLP(nn.Module):
    """Small MLP that maps y → hyperprior summary z.

    Mirrors Ballé 2018's hyper-encoder but uses an MLP because our "y" is a
    compact per-pair latent (``(N_PAIRS, latent_dim)``) rather than a feature
    map. For the spatial conv variant, callers can subclass and override
    forward.
    """

    def __init__(self, y_channels: int, hidden: int, z_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(y_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, z_channels),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


class _HyperDecoderMLP(nn.Module):
    """Small MLP that maps z → per-element scale σ for the GaussianConditional.

    The output is positive (softplus) so the GaussianConditional sees valid
    scales.
    """

    def __init__(self, z_channels: int, hidden: int, y_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, y_channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.net(z))


class BalleHyperpriorWrapper(nn.Module):
    """Ballé 2018 ScaleHyperprior wrapper specialised for T1's latent table.

    The wrapper exposes:

    - ``forward(y) -> dict`` returning ``y_hat``, ``rate_y_bits``,
      ``rate_z_bits``, ``rate_total_bits``. Use this in the training loop.
    - ``compress(y) -> dict`` returning the actual encoded byte strings.
      Requires ``update()`` to have been called first.
    - ``decompress(strings, shape) -> y_hat`` (round-trip correctness).
    - ``aux_loss() -> Tensor`` exposing the EntropyBottleneck's quantile-loss
      that MUST be optimised on a separate optimiser.
    - ``update()`` building the AC CDF tables. Idempotent on identical state.

    Examples
    --------
    >>> cfg = BalleHyperpriorConfig(y_channels=28, z_channels=16)
    >>> wrap = BalleHyperpriorWrapper(cfg)
    >>> y = torch.randn(600, 28)  # 600 pairs × 28 latent dims
    >>> out = wrap(y)
    >>> sorted(out.keys())
    ['rate_total_bits', 'rate_y_bits', 'rate_z_bits', 'y_hat']
    """

    def __init__(self, config: BalleHyperpriorConfig | None = None):
        super().__init__()
        self.config = config or BalleHyperpriorConfig()
        cai = _import_compressai()
        self._EntropyBottleneck = cai["EntropyBottleneck"]
        self._GaussianConditional = cai["GaussianConditional"]
        self.hyper_encoder = _HyperEncoderMLP(
            self.config.y_channels, self.config.hyper_hidden, self.config.z_channels
        )
        self.hyper_decoder = _HyperDecoderMLP(
            self.config.z_channels, self.config.hyper_hidden, self.config.y_channels
        )
        self.entropy_bottleneck = self._EntropyBottleneck(self.config.z_channels)
        # Initialise GaussianConditional with the scale table (None means
        # "lazy init"; we want the table fixed up-front for determinism).
        scale_table = _get_default_scale_table(self.config)
        self.gaussian_conditional = self._GaussianConditional(scale_table.tolist())
        self._cdf_table_built: bool = False

    @property
    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """Differentiable rate-and-distortion forward pass.

        Parameters
        ----------
        y : torch.Tensor
            ``(N, y_channels)`` (or ``(N, y_channels, *spatial)``) latent.
            Phase 1: T1 uses ``(N_PAIRS, latent_dim)``.

        Returns
        -------
        dict with keys ``y_hat`` (quantised latent), ``rate_y_bits``,
        ``rate_z_bits``, ``rate_total_bits`` (all scalar tensors carrying
        gradient for the joint Lagrangian).
        """
        # Hyperprior pathway: y → z via the hyper-encoder.
        z = self.hyper_encoder(y)
        # CompressAI EntropyBottleneck expects a 4D tensor (B, C, H, W) for
        # spatial inputs. We have a 2D (N, C) tensor for the latent table —
        # add singleton spatial dims, run, strip them off.
        z_4d = z.t().unsqueeze(0).unsqueeze(-1)  # (1, C_z, N, 1)
        z_hat_4d, z_likelihoods_4d = self.entropy_bottleneck(z_4d)
        z_hat = z_hat_4d.squeeze(-1).squeeze(0).t()  # back to (N, C_z)
        z_likelihoods = z_likelihoods_4d.squeeze(-1).squeeze(0).t()
        # Hyper-decoder predicts σ for the y-side GaussianConditional.
        scales = self.hyper_decoder(z_hat)
        # Treat means as zero (Ballé 2018 ScaleHyperprior; not Mean-Scale).
        # GaussianConditional expects scales matching y's shape.
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales)
        # Rate accumulators.
        eps = 1e-12
        rate_y_bits = -torch.log2(y_likelihoods + eps).sum()
        rate_z_bits = -torch.log2(z_likelihoods + eps).sum()
        rate_total_bits = rate_y_bits + rate_z_bits
        return {
            "y_hat": y_hat,
            "rate_y_bits": rate_y_bits,
            "rate_z_bits": rate_z_bits,
            "rate_total_bits": rate_total_bits,
        }

    def aux_loss(self) -> torch.Tensor:
        """EntropyBottleneck quantile-loss (must be on a SEPARATE optimiser).

        Per CompressAI design, this loss optimises the discrete cumulative
        bounds inside the EntropyBottleneck. It must NEVER share parameters
        with the main optimiser or the AC tables drift.
        """
        return self.entropy_bottleneck.loss()

    def update(self, force: bool = False) -> bool:
        """Build the entropy CDF tables. Required before ``compress()``.

        Returns True if a rebuild happened, False if cached.
        """
        # CompressAI's update() returns True if the CDF was rebuilt.
        eb_rebuilt = self.entropy_bottleneck.update(force=force)
        gc_rebuilt = False
        if hasattr(self.gaussian_conditional, "update_scale_table"):
            scale_table = _get_default_scale_table(self.config)
            gc_rebuilt = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        self._cdf_table_built = True
        return bool(eb_rebuilt or gc_rebuilt)

    def compress(self, y: torch.Tensor) -> dict[str, Any]:
        """Encode ``y`` to byte strings (real wire format).

        Returns a dict with ``y_strings`` (list[bytes]), ``z_strings``
        (list[bytes]), and ``z_shape`` for the decompress side.
        """
        if not self._cdf_table_built:
            raise RuntimeError(
                "BalleHyperpriorWrapper.compress() called before update(); "
                "call wrapper.update() once after training to build the AC CDF tables"
            )
        z = self.hyper_encoder(y)
        z_4d = z.t().unsqueeze(0).unsqueeze(-1)
        z_strings = self.entropy_bottleneck.compress(z_4d)
        # decompress to get the discretised z_hat that the receiver will see.
        z_hat_4d = self.entropy_bottleneck.decompress(z_strings, z_4d.shape[-2:])
        z_hat = z_hat_4d.squeeze(-1).squeeze(0).t()
        scales = self.hyper_decoder(z_hat)
        # GaussianConditional needs symbol indices for compress().
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {
            "y_strings": y_strings,
            "z_strings": z_strings,
            "z_shape": list(z_4d.shape),
        }

    def decompress(self, strings: dict[str, Any]) -> torch.Tensor:
        """Reconstruct ``y_hat`` from byte strings."""
        if not self._cdf_table_built:
            raise RuntimeError(
                "BalleHyperpriorWrapper.decompress() called before update()"
            )
        z_strings = strings["z_strings"]
        y_strings = strings["y_strings"]
        z_shape = strings["z_shape"]
        z_hat_4d = self.entropy_bottleneck.decompress(z_strings, z_shape[-2:])
        z_hat = z_hat_4d.squeeze(-1).squeeze(0).t()
        scales = self.hyper_decoder(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes)
        return y_hat


def build_balle_hyperprior(
    config: BalleHyperpriorConfig | None = None,
) -> BalleHyperpriorWrapper:
    """Factory: build a Ballé hyperprior wrapper with optional config."""
    return BalleHyperpriorWrapper(config or BalleHyperpriorConfig())
