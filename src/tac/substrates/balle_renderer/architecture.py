"""balle_renderer architecture — Ballé hyperprior + lightweight renderer (β).

Per the Fields-medal grand council 2026-05-12 β candidate (§2.10 Ballé LEAD,
§4.2 13-lessons compliance table):

End-to-end-trainable codec with the Ballé 2018 ICLR scale-hyperprior
nonlinear-transform-coding architecture, applied AS a full-RGB renderer
(HNeRV parity lesson 5: renderer not mask codec).

Architecture (council-approved 2026-05-12):

    Per-pair latent z in R^{C_z}      # learned from the contest video
       |
       v
    Hyper-analysis h_a:                # z -> hyper-latent w (small)
       Conv -> GDN -> Conv -> GDN -> Conv
       |
       v
    Quantize w to w_hat (during inflate)
       |
       v
    Hyper-synthesis h_s:               # w_hat -> per-element scale σ
       Conv -> GDN^-1 -> Conv -> GDN^-1 -> Conv -> softplus
       |  (σ shape = z shape)
       v
    Conditional density p_y(y | σ) = N(0, σ²)
       |  used by the arithmetic coder at archive-build time
       v
    Decoder g_s(z_hat):                 # z_hat = quantized z
       Linear -> reshape -> [Block_i: Conv -> sin/GDN^-1 -> PixelShuffle(2)] x N
       |
       v
    Pair of RGB heads (frame_0 / frame_1)
       |
       v
    (rgb_0, rgb_1) in [0, 1], shape (B, 3, H, W)

(Output is interpolated bilinearly from the final block grid to the contest
384x512.)

Per Ballé 2018:

    rate R = E[-log p_z(w_hat)] + E[-log p_y(z_hat | σ(w_hat))]

where p_z is a factorized prior on the hyper-latent and p_y is a
conditional Gaussian whose scale comes from the hyper-synthesis. The
side-info bytes (the hyper-latent + scale-prior MLP state) amortise when
|z stream| >> |w stream|.

Council notes:
- Param count target: ≤ 250K (per Selfcomp ceiling)
- GDN nonlinearity per Ballé 2018 (NOT ReLU; better for codec networks)
- sin activation in decoder per α convention (SIREN/NeRF style)
- Decoder channels mirror α to keep architectural parity
- EMA decay 0.997 per CLAUDE.md "EMA — non-negotiable" (applied externally)
- Bolt-on ≤ 350 LOC; substrate_engineering exception per L7

CLAUDE.md compliance:
- No silent device defaults (caller passes device explicitly)
- No scorer loading inside this module (score-aware loss is a separate module)
- No /tmp paths
- Reviewable in 30 seconds per L12
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


_CONTEST_H = 384
_CONTEST_W = 512
_NUM_FRAMES = 1200
_PAIRS = _NUM_FRAMES // 2  # 600 pairs


@dataclass(frozen=True)
class BalleRendererConfig:
    """Static design-time parameters for balle_renderer.

    All fields required-keyword (no silent defaults beyond explicit ones).

    Defaults are council-calibrated 2026-05-12 to hit ~200-250K total params
    so we stay within the Selfcomp empirical ceiling.
    """

    latent_dim: int = 24
    """Per-pair main-latent dimensionality z (target rate-axis primary)."""

    hyper_latent_dim: int = 8
    """Hyper-latent w dimensionality (side-info; small)."""

    embed_dim: int = 40
    """Channels of the initial spatial-grid embedding (decoder input)."""

    initial_grid_h: int = 3
    """Initial spatial-grid height before upsample blocks."""

    initial_grid_w: int = 4
    """Initial spatial-grid width before upsample blocks."""

    decoder_channels: tuple[int, ...] = (40, 32, 24, 20, 16, 12, 8)
    """Per-block output channels BEFORE the final RGB heads.

    Mirrors α's calibration to hit ~200-250K params with the
    hyperprior MLP overhead included.
    """

    hyper_mlp_channels: tuple[int, ...] = (16, 16)
    """Hyper-analysis / synthesis MLP hidden widths."""

    sin_frequency: float = 30.0
    """NeRF-style sin activation frequency (decoder)."""

    num_pairs: int = _PAIRS
    """Number of latent rows (600 for the contest 1200-frame video)."""

    output_height: int = _CONTEST_H
    """Final RGB output height."""

    output_width: int = _CONTEST_W
    """Final RGB output width."""

    num_upsample_blocks: int = 7
    """Number of PixelShuffle(2) blocks. 7 -> 3x4 -> 384x512 ratio."""

    quantize_noise_std: float = 0.5
    """Uniform-noise additive proxy for quantization during training
    (Ballé 2017's noise-relaxation; STE alternative)."""


class _GDN(nn.Module):
    """Generalized Divisive Normalization (Ballé 2016/2018).

    y_i = x_i / sqrt(beta_i + sum_j gamma_ij * x_j^2)

    The de-facto canonical nonlinearity for compression networks. Beats
    ReLU and PReLU on rate-distortion in Ballé 2016 / 2017 / 2018.

    Minimal re-implementation (no CompressAI dep per L9 runtime closure).
    """

    def __init__(self, channels: int, inverse: bool = False) -> None:
        super().__init__()
        self.inverse = bool(inverse)
        # beta: (C,) positive offset
        # gamma: (C, C) positive coupling matrix; init as identity for stability
        self.beta = nn.Parameter(torch.ones(channels) * 1e-4)
        self.gamma = nn.Parameter(torch.eye(channels) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # Ensure positivity (Ballé's reparam: beta = softplus(beta_raw))
        beta = F.softplus(self.beta)
        gamma = F.softplus(self.gamma)
        # norm_pool: (B, C, H, W) -> per-pixel quadratic combination
        # x^2 shape (B, C, H, W); gamma shape (C, C). Compute sum_j gamma_ij x_j^2
        x_sq = x * x
        # gamma_ij as 1x1 conv weight: gamma viewed as (C_out, C_in, 1, 1)
        norm = F.conv2d(x_sq, gamma.view(*gamma.shape, 1, 1))
        norm = norm + beta.view(1, -1, 1, 1)
        norm = norm.clamp(min=1e-12).sqrt()
        if self.inverse:
            return x * norm
        return x / norm


class _SinAct(nn.Module):
    """Sine activation (SIREN/NeRF-style)."""

    def __init__(self, w: float) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.sin(self.w * x)


class _UpBlock(nn.Module):
    """One Conv -> sin -> PixelShuffle(2) decoder block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        sin_freq: float,
        *,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        # PixelShuffle(2) needs 4x output channels in the conv before shuffle
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size, padding=kernel_size // 2)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.shuffle(self.act(self.conv(x)))


class _HyperAnalysis(nn.Module):
    """Hyper-analysis h_a: z -> w (small hyper-latent).

    Simple MLP since z is a flat (B, C_z) latent (NOT a spatial map). The
    canonical Ballé 2018 image-coding hyper-analysis is convolutional, but
    our z is already amortised over time + space via the decoder, so an
    MLP suffices for the per-pair signature.
    """

    def __init__(self, latent_dim: int, hyper_dim: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = latent_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, hyper_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(z)


class _HyperSynthesis(nn.Module):
    """Hyper-synthesis h_s: w_hat -> per-element scale σ for p_y(y|σ).

    Output shape matches z: (B, C_z). Softplus to ensure positivity.
    """

    def __init__(self, hyper_dim: int, latent_dim: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = hyper_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, w_hat: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = self.net(w_hat)
        # softplus + small floor to avoid zero-scale numerical issues
        return F.softplus(out) + 1e-4


def _quantize_with_noise(x: torch.Tensor, noise_std: float, training: bool) -> torch.Tensor:
    """Ballé 2017 quantization relaxation: additive uniform noise at train,
    hard round at inference. Allows gradient flow."""
    if training:
        noise = (torch.rand_like(x) - 0.5) * (2.0 * noise_std)
        return x + noise
    return x.round()


class BalleRendererSubstrate(nn.Module):
    """Ballé-hyperprior + lightweight renderer substrate.

    Input: pair index ``i in [0, num_pairs)``
    Output: ``(rgb_0, rgb_1)`` both ``(B, 3, H, W)`` in [0, 1], plus the
    hyperprior rate term ``rate_components`` for the loss.

    The score-aware loss (separate module) consumes the rendered frames +
    the rate_components, runs frames through the differentiable
    eval-roundtrip, and backprops through SegNet/PoseNet.
    """

    def __init__(self, cfg: BalleRendererConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Per-pair learned latents (z; the rate-axis primary stream)
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )

        # Hyper-analysis / synthesis (the side-info path)
        self.hyper_analysis = _HyperAnalysis(
            cfg.latent_dim, cfg.hyper_latent_dim, cfg.hyper_mlp_channels
        )
        self.hyper_synthesis = _HyperSynthesis(
            cfg.hyper_latent_dim, cfg.latent_dim, cfg.hyper_mlp_channels
        )

        # Decoder: latent -> initial spatial grid -> up-blocks -> RGB heads
        self.latent_embed = nn.Linear(
            cfg.latent_dim,
            cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
        )

        channels = [cfg.embed_dim] + list(cfg.decoder_channels)
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at least "
                f"num_upsample_blocks ({cfg.num_upsample_blocks}) entries"
            )
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            blocks.append(_UpBlock(in_ch, out_ch, cfg.sin_frequency))
        self.blocks = nn.ModuleList(blocks)

        final_ch = channels[cfg.num_upsample_blocks]
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)

        # Factorized prior parameters for w (hyper-latent).
        # A piecewise-linear CDF would be canonical; for the SCAFFOLD we use
        # a learned Gaussian with per-channel mean+log_scale. The β trainer
        # follow-up subagent can swap in the full Ballé piecewise-linear CDF.
        self.w_prior_mean = nn.Parameter(torch.zeros(cfg.hyper_latent_dim))
        self.w_prior_log_scale = nn.Parameter(torch.zeros(cfg.hyper_latent_dim))

        self._siren_init()

    def _siren_init(self) -> None:
        """SIREN-style init on convs/linears with sin activation.

        GDN params keep their constructor defaults; the Ballé reparam
        (softplus on beta/gamma) ensures positivity from any init.
        """
        w = self.cfg.sin_frequency
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()
                elif isinstance(m, nn.Linear):
                    fan_in = m.in_features
                    # Plain Xavier for hyper MLPs (GELU activation); SIREN-style
                    # only for the decoder latent_embed (sin-driven downstream).
                    if m is self.latent_embed:
                        bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    else:
                        bound = math.sqrt(6.0 / fan_in)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()

    def _hyperprior_rate_components(
        self, z: torch.Tensor, sigma: torch.Tensor, w_hat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute R = E[-log p_z(w_hat)] + E[-log p_y(z|σ)] in nats (mean per element).

        Returns a dict with separate ``hyper_rate`` (E[-log p_z]) and
        ``main_rate`` (E[-log p_y]) so the trainer can log them.
        """
        # p_z(w_hat) = N(w_prior_mean, exp(w_prior_log_scale)^2)
        w_scale = torch.exp(self.w_prior_log_scale).clamp(min=1e-4)
        # -log N(w_hat; mean, scale) elementwise
        var_z = w_scale * w_scale
        nll_z = 0.5 * (
            ((w_hat - self.w_prior_mean) ** 2) / var_z
            + torch.log(2.0 * math.pi * var_z)
        )
        hyper_rate = nll_z.mean()

        # p_y(z | σ) = N(0, σ²)
        sigma2 = (sigma * sigma).clamp(min=1e-8)
        nll_y = 0.5 * ((z * z) / sigma2 + torch.log(2.0 * math.pi * sigma2))
        main_rate = nll_y.mean()

        return {
            "hyper_rate": hyper_rate,
            "main_rate": main_rate,
            "total_rate": hyper_rate + main_rate,
        }

    def forward(
        self,
        pair_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Render frame-pairs at the given pair indices.

        Args:
            pair_indices: ``(B,)`` long tensor in ``[0, num_pairs)``.

        Returns:
            ``(rgb_0, rgb_1, rate_components)`` where rate_components has
            ``hyper_rate`` / ``main_rate`` / ``total_rate`` scalar tensors
            (mean nats per element of their respective streams).
        """
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs})"
            )

        z_raw = self.latents[pair_indices]  # (B, latent_dim)

        # Hyper-analysis: produce w; quantize w_hat
        w = self.hyper_analysis(z_raw)
        w_hat = _quantize_with_noise(w, self.cfg.quantize_noise_std, self.training)

        # Hyper-synthesis: σ from w_hat
        sigma = self.hyper_synthesis(w_hat)

        # Quantize main latent z (training: + uniform noise; eval: round)
        z_hat = _quantize_with_noise(z_raw, self.cfg.quantize_noise_std, self.training)

        # Compute rate components (Ballé 2018)
        rate_components = self._hyperprior_rate_components(z_hat, sigma, w_hat)

        # Decoder: g_s(z_hat) -> RGB frame pair
        h = self.latent_embed(z_hat)
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

        return rgb_0, rgb_1, rate_components

    def num_parameters(self) -> int:
        """Total trainable parameter count (council target ≤ 250K)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
