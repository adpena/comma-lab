# SPDX-License-Identifier: MIT
"""pact_nerv_bayesian architecture — Pact-NeRV-Bayesian (L0 SKETCH).

HNeRV-class implicit renderer with a Bayesian latent embedding layer (Blundell
et al. 2015 Bayes by Backprop arXiv:1505.05424). Each latent-embedding weight
is a learnable Gaussian (mean, log_sigma); samples flow via the reparameterization
trick at training time; the mean is used at inflate time per Blundell §4.

KL divergence regularization keeps the posterior close to a unit-Gaussian
prior N(0, 1) per Blundell §3.1.

CLAUDE.md compliance: no MPS fallback, no inflate-time scorer load, no /tmp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_CONTEST_H = 384
_CONTEST_W = 512
_PAIRS = 600


@dataclass(frozen=True)
class PactNervBayesianConfig:
    """Static design-time parameters for Pact-NeRV-Bayesian."""

    latent_dim: int = 24
    embed_dim: int = 64
    initial_grid_h: int = 3
    initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = (48, 40, 32, 24, 20, 16, 12)
    sin_frequency: float = 30.0
    num_upsample_blocks: int = 7
    bayesian_prior_sigma: float = 1.0
    """Standard deviation of the Gaussian prior (Blundell canonical: 1.0)."""
    bayesian_log_sigma_init: float = -3.0
    """Initial log-sigma for posterior (small init for early-training stability)."""
    kl_weight: float = 1.0
    """KL divergence regularization weight (Stage 1 sweep target)."""
    num_pairs: int = _PAIRS
    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W


class _SinAct(nn.Module):
    def __init__(self, w: float) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w * x)


class _DepthSepConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _DsUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        super().__init__()
        self.dsc = _DepthSepConv(in_ch, out_ch * 4)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.act(self.dsc(x)))


class BayesianLinearLayer(nn.Module):
    """Bayesian linear layer per Blundell 1505.05424 Bayes by Backprop.

    Weights are learnable Gaussians: each scalar weight w ~ N(mu_w, sigma_w^2)
    with sigma_w = softplus(rho_w) (Blundell §3.2 reparameterization). At
    training time samples flow via the reparameterization trick:
        w = mu_w + sigma_w * epsilon, epsilon ~ N(0, 1)
    At inference (use_mean=True) the mean is used per Blundell §4.

    The KL divergence ||posterior || prior|| with prior N(0, prior_sigma^2)
    is computed and returned for trainer-side regularization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        prior_sigma: float = 1.0,
        log_sigma_init: float = -3.0,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"in/out features must be positive; got {in_features}, {out_features}"
            )
        if prior_sigma <= 0:
            raise ValueError(f"prior_sigma must be positive; got {prior_sigma}")
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = float(prior_sigma)

        # Posterior parameters: mean + log_sigma (via softplus reparam).
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).normal_(std=0.01)
        )
        # rho such that sigma = softplus(rho); init small so sigma is small early.
        self.weight_rho = nn.Parameter(
            torch.full((out_features, in_features), log_sigma_init)
        )
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), log_sigma_init))

        self._last_kl_div = torch.tensor(0.0)

    def forward(self, x: torch.Tensor, *, use_mean: bool = False) -> torch.Tensor:
        if use_mean:
            # Inflate-time: use posterior mean (Blundell §4 canonical).
            weight = self.weight_mu
            bias = self.bias_mu
            self._last_kl_div = torch.tensor(0.0)
        else:
            # Training-time: sample via reparameterization trick.
            weight_sigma = F.softplus(self.weight_rho)
            bias_sigma = F.softplus(self.bias_rho)
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            weight = self.weight_mu + weight_sigma * weight_eps
            bias = self.bias_mu + bias_sigma * bias_eps

            # KL divergence: KL(N(mu, sigma^2) || N(0, prior_sigma^2))
            # = log(prior_sigma / sigma) + (sigma^2 + mu^2) / (2 prior_sigma^2) - 0.5
            prior_var = self.prior_sigma ** 2
            kl_w = (
                torch.log(torch.full_like(weight_sigma, self.prior_sigma) / weight_sigma.clamp(min=1e-8))
                + (weight_sigma.pow(2) + self.weight_mu.pow(2)) / (2 * prior_var)
                - 0.5
            ).sum()
            kl_b = (
                torch.log(torch.full_like(bias_sigma, self.prior_sigma) / bias_sigma.clamp(min=1e-8))
                + (bias_sigma.pow(2) + self.bias_mu.pow(2)) / (2 * prior_var)
                - 0.5
            ).sum()
            self._last_kl_div = kl_w + kl_b

        return F.linear(x, weight, bias)

    @property
    def last_kl_div(self) -> torch.Tensor:
        return self._last_kl_div


class PactNervBayesianSubstrate(nn.Module):
    """Pact-NeRV-Bayesian renderer (L0 SKETCH).

    The distinguishing primitive is the BayesianLinearLayer at the latent
    embedding layer. Decoder weights are deterministic (Bayesian on every
    layer is a Stage 1 expansion path per cargo-cult audit).
    """

    def __init__(self, cfg: PactNervBayesianConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )

        # Bayesian latent embedding (the distinguishing primitive)
        self.bayesian_latent_embed = BayesianLinearLayer(
            in_features=cfg.latent_dim,
            out_features=cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
            prior_sigma=cfg.bayesian_prior_sigma,
            log_sigma_init=cfg.bayesian_log_sigma_init,
        )

        channels = [cfg.embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at "
                f"least num_upsample_blocks ({cfg.num_upsample_blocks}) entries"
            )
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(_DsUpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
        self.blocks = nn.ModuleList(blocks)

        final_ch = channels[cfg.num_upsample_blocks]
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=1)

        # Whether to use posterior mean (inflate-time) vs sample (training).
        self._use_posterior_mean = False

        self._siren_init()

    def _siren_init(self) -> None:
        w = self.cfg.sin_frequency
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    if m.groups > 1:
                        fan_in = m.kernel_size[0] * m.kernel_size[1]
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()

    def use_posterior_mean(self, mean: bool = True) -> None:
        """Switch latent_embed to use posterior mean (inflate-time)."""
        self._use_posterior_mean = mean

    def forward(
        self, pair_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs})"
            )

        z = self.latents[pair_indices]
        h = self.bayesian_latent_embed(z, use_mean=self._use_posterior_mean)
        h = h.view(
            -1,
            self.cfg.embed_dim,
            self.cfg.initial_grid_h,
            self.cfg.initial_grid_w,
        )

        for block in self.blocks:
            h = block(h)

        if h.shape[-2:] != (self.cfg.output_height, self.cfg.output_width):
            h = F.interpolate(
                h,
                size=(self.cfg.output_height, self.cfg.output_width),
                mode="bilinear",
                align_corners=False,
            )

        rgb_0 = torch.sigmoid(self.head_rgb_0(h))
        rgb_1 = torch.sigmoid(self.head_rgb_1(h))
        return rgb_0, rgb_1

    @property
    def last_kl_div(self) -> torch.Tensor:
        """KL divergence from the most recent forward pass (for trainer)."""
        return self.bayesian_latent_embed.last_kl_div

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_bayesian_parameters(self) -> int:
        """Count of Bayesian posterior parameters (mu + rho per weight + bias)."""
        return sum(
            p.numel()
            for p in self.bayesian_latent_embed.parameters()
            if p.requires_grad
        )
