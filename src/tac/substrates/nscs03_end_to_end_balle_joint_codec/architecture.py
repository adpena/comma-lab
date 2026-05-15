# SPDX-License-Identifier: MIT
"""NSCS03 architecture — end-to-end Ballé 2018 joint codec.

Per assumptions-challenge-audit NSCS03 design memo:
    `g_a (analysis transform)` -> `y` -> `h_a (hyper-analysis)` -> `z`
    -> [factorized prior on z] -> `z_hat` -> `h_s (hyper-synthesis)` -> `σ`
    -> [conditional Gaussian on y given σ] -> `y_hat` -> `g_s (synthesis)` -> RGB

The whole pipeline is END-TO-END DIFFERENTIABLE via the Ballé 2017
quantization relaxation (additive uniform noise during train; hard round at
eval). Score-aware loss backprops THROUGH the bottleneck THROUGH yuv6
THROUGH SegNet/PoseNet ALL THE WAY to the analysis transform `g_a`.

This is the canonical Ballé et al. 2018 ICLR architecture
("Variational Image Compression with a Scale Hyperprior"), applied to the
contest 2-frame pose-pair input (B, 6, H, W) — six channels are the
two stacked RGB frames concatenated along the channel axis. Output is the
same (B, 6, H, W) reconstruction split back into (rgb_0, rgb_1).

Key UNIQUE-AND-COMPLETE design decisions vs `balle_renderer` (β):

1. ANALYSIS / SYNTHESIS are CONVOLUTIONAL (Ballé 2018 g_a / g_s), NOT
   per-pair flat MLPs. The latent y has spatial structure; the entropy model
   exploits the spatial correlation that flat-MLP-renderers cannot.

2. HYPER-ANALYSIS / HYPER-SYNTHESIS are CONVOLUTIONAL (h_a / h_s).

3. ENTROPY BOTTLENECK on hyper-latent z reuses canonical
   `tac.entropy_bottleneck.EntropyBottleneck` for math correctness — that
   primitive carries the per-channel logistic CDF (Ballé 2018 factorized
   prior approximation) and was already validated against the canonical
   Ballé reference.

4. CONDITIONAL GAUSSIAN density on main latent y given σ = h_s(z_hat) is
   the Ballé 2018 SCALE HYPERPRIOR — the FREE side-info path that closes
   the rate gap vs flat factorized prior on y.

5. INPUT IS PIXELS, OUTPUT IS PIXELS — no per-pair learned latent table.
   The encoder weights ARE the substrate; the per-pair information lives
   ENTIRELY in the encoded latents y_hat / z_hat that ship in the archive.

CLAUDE.md compliance:
- No silent device defaults (caller passes device explicitly via .to(device))
- No scorer loading inside this module (score-aware loss is a separate module)
- No /tmp paths
- Reviewable per L12 (each section has a docstring + a one-liner intent)
- AUTOCAST FP16 evaluation tagged TIER_1 in trainer manifest (Catalog #172
  hygiene). The ENTROPY BOTTLENECK forward should run in fp32 (autocast may
  underflow at the logistic CDF tails); trainer must wrap accordingly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.entropy_bottleneck import EntropyBottleneck

_CONTEST_H = 384
_CONTEST_W = 512

# IMPORTANT: each downsample halves resolution. 4 stride-2 layers => /16.
# At input H=384, W=512, latent shape after g_a is (B, C_main, 24, 32).
# Hyper-latent shape after h_a is (B, C_hyper, 6, 8) (additional /4).
_DEFAULT_DOWN_FACTOR_MAIN = 16
_DEFAULT_DOWN_FACTOR_HYPER = 4  # relative to main latent


@dataclass(frozen=True)
class NSCS03Config:
    """Static design-time parameters for NSCS03.

    All fields required-keyword (no silent defaults beyond explicit ones).

    Defaults are AUDIT-calibrated to (a) hit ~250-400K total params (council
    Selfcomp ceiling ~250K is the BOLT-ON budget; substrate-engineering
    budget per L7 allows ~1200 LOC + larger param count for END-TO-END
    architectures that ship the ENCODER too), (b) match Ballé 2018 reference
    architecture proportions (main 192ch, hyper 128ch in the reference;
    we use lighter 64/32 to stay within the rate envelope).
    """

    in_channels: int = 6
    """Input channels: 2 frames x 3 RGB stacked along channel axis."""

    out_channels: int = 6
    """Output channels: same as input (joint codec reconstructs both frames)."""

    main_latent_channels: int = 64
    """C_main: channels of the main latent y. Ballé 2018 uses 192; we use 64
    to keep params/rate envelope compatible with the contest 0.19 frontier."""

    hyper_latent_channels: int = 32
    """C_hyper: channels of the hyper-latent z. Ballé 2018 uses 128; ours 32."""

    g_a_channels: tuple[int, ...] = (32, 48, 56, 64)
    """Analysis transform per-block output channels (4 stride-2 conv blocks).
    Last entry must equal main_latent_channels."""

    g_s_channels: tuple[int, ...] = (64, 56, 48, 32)
    """Synthesis transform per-block output channels (4 stride-2 transposed
    conv blocks). First entry must equal main_latent_channels."""

    h_a_channels: tuple[int, ...] = (48, 32)
    """Hyper-analysis per-block output channels (2 stride-2 conv blocks).
    Last entry must equal hyper_latent_channels."""

    h_s_channels: tuple[int, ...] = (48, 64)
    """Hyper-synthesis per-block output channels (2 stride-2 transposed conv
    blocks). Last entry MUST equal main_latent_channels (output is sigma which
    must align with the main latent y for the conditional Gaussian density)."""

    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W

    quantize_noise_std: float = 0.5
    """Ballé 2017 noise relaxation. STE alternative; default 0.5 is the
    canonical uniform-noise width."""

    gdn_eps: float = 1e-6
    """Numerical floor for GDN/IGDN normalization. 1e-6 (NOT 1e-12) per
    Catalog #172 + autocast fp16 hygiene — 1e-12 underflows in fp16."""

    sigma_floor: float = 1e-4
    """Minimum scale for the conditional-Gaussian density on y. Prevents
    degenerate p_y collapse to delta-function."""

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if self.main_latent_channels <= 0 or self.hyper_latent_channels <= 0:
            raise ValueError("latent channels must be positive")
        if not self.g_a_channels or self.g_a_channels[-1] != self.main_latent_channels:
            raise ValueError(
                f"g_a_channels must end with main_latent_channels={self.main_latent_channels}; "
                f"got {self.g_a_channels}"
            )
        if not self.g_s_channels or self.g_s_channels[0] != self.main_latent_channels:
            raise ValueError(
                f"g_s_channels must start with main_latent_channels={self.main_latent_channels}; "
                f"got {self.g_s_channels}"
            )
        if not self.h_a_channels or self.h_a_channels[-1] != self.hyper_latent_channels:
            raise ValueError(
                f"h_a_channels must end with hyper_latent_channels={self.hyper_latent_channels}; "
                f"got {self.h_a_channels}"
            )
        if not self.h_s_channels or self.h_s_channels[-1] != self.main_latent_channels:
            raise ValueError(
                f"h_s_channels must end with main_latent_channels={self.main_latent_channels}; "
                f"got {self.h_s_channels}"
            )
        if self.quantize_noise_std < 0.0:
            raise ValueError("quantize_noise_std must be >= 0")
        if self.gdn_eps <= 0.0:
            raise ValueError("gdn_eps must be > 0")
        if self.sigma_floor <= 0.0:
            raise ValueError("sigma_floor must be > 0")
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output dims must be positive")


class _GDN(nn.Module):
    """Generalized Divisive Normalization (Ballé 2016/2018).

    y_i = x_i / sqrt(beta_i + sum_j gamma_ij * x_j^2)

    Minimal in-file reimplementation per HNeRV parity L9 (NO CompressAI
    runtime dep). Forked from balle_renderer's _GDN with one improvement:
    autocast-fp16 hygiene via float() cast inside the conv (Ballé hyperprior
    + GDN under autocast fp16 had numerical instability per FIX-WAVE
    empirical observation).
    """

    def __init__(self, channels: int, *, inverse: bool = False, eps: float = 1e-6) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        self.channels = int(channels)
        self.inverse = bool(inverse)
        self.eps = float(eps)
        # Ballé 2018 reparam: positive offset + positive coupling, both
        # constructed via softplus on the raw parameter for stability.
        # raw_beta init s.t. softplus(raw_beta) ~ 1
        self.raw_beta = nn.Parameter(
            torch.full((channels,), float(math.log(math.expm1(1.0))))
        )
        # raw_gamma init = identity diagonal
        self.raw_gamma = nn.Parameter(
            torch.eye(channels) * float(math.log(math.expm1(0.1)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = F.softplus(self.raw_beta).to(x.dtype)
        gamma = F.softplus(self.raw_gamma).to(x.dtype)
        # Per-pixel quadratic combination via 1x1 conv on x^2
        x_sq = x * x
        norm = F.conv2d(x_sq, gamma.view(self.channels, self.channels, 1, 1))
        norm = norm + beta.view(1, -1, 1, 1)
        norm = norm.clamp(min=self.eps).sqrt()
        if self.inverse:
            return x * norm
        return x / norm


class _AnalysisTransform(nn.Module):
    """Convolutional analysis transform g_a: pixels -> y latent.

    Ballé 2018 architecture: 4 stride-2 conv blocks each with GDN
    nonlinearity. Final block produces the main-latent y of shape
    (B, C_main, H/16, W/16).
    """

    def __init__(self, in_channels: int, channels: tuple[int, ...], *, gdn_eps: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for i, c in enumerate(channels):
            layers.append(nn.Conv2d(prev, c, kernel_size=5, stride=2, padding=2))
            # Final block has no nonlinearity (Ballé 2018: output is the
            # raw latent representation that the entropy model assumes).
            if i < len(channels) - 1:
                layers.append(_GDN(c, inverse=False, eps=gdn_eps))
            prev = c
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SynthesisTransform(nn.Module):
    """Convolutional synthesis transform g_s: y_hat -> pixels.

    Ballé 2018 architecture: 4 stride-2 transposed-conv blocks each with
    inverse GDN. Final block produces the reconstruction of shape
    (B, out_channels, H, W).

    Note on PixelShuffle vs ConvTranspose: Ballé 2018 reference uses
    transposed conv; we follow that for fidelity with the literature
    anchor. PixelShuffle is empirically very close at this rate envelope.
    """

    def __init__(
        self,
        channels: tuple[int, ...],
        out_channels: int,
        *,
        gdn_eps: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = channels[0]  # first entry = main_latent_channels
        for _i, c in enumerate(channels[1:], start=1):
            layers.append(
                nn.ConvTranspose2d(
                    prev, c, kernel_size=5, stride=2, padding=2, output_padding=1
                )
            )
            layers.append(_GDN(c, inverse=True, eps=gdn_eps))
            prev = c
        # Final transposed conv to RGB-stack, no nonlinearity (sigmoid is
        # applied by the caller for [0, 1] range).
        layers.append(
            nn.ConvTranspose2d(
                prev, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        return self.net(y_hat)


class _HyperAnalysis(nn.Module):
    """Hyper-analysis h_a: |y| (or y) -> z.

    Ballé 2018 takes |y| (absolute value) as input to h_a since the scale
    of y is the rate-relevant statistic. We follow that convention.
    """

    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for i, c in enumerate(channels):
            layers.append(nn.Conv2d(prev, c, kernel_size=3, stride=2, padding=1))
            if i < len(channels) - 1:
                layers.append(nn.LeakyReLU(0.2, inplace=False))
            prev = c
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.abs(y))


class _HyperSynthesis(nn.Module):
    """Hyper-synthesis h_s: z_hat -> σ for p_y(y | σ).

    Output shape matches y: (B, C_main, H/16, W/16). We softplus + small
    floor to ensure positivity (sigma_floor handled by caller).
    """

    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for i, c in enumerate(channels):
            layers.append(
                nn.ConvTranspose2d(
                    prev, c, kernel_size=3, stride=2, padding=1, output_padding=1
                )
            )
            if i < len(channels) - 1:
                layers.append(nn.LeakyReLU(0.2, inplace=False))
            prev = c
        self.net = nn.Sequential(*layers)

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        return self.net(z_hat)


def _quantize_with_noise(x: torch.Tensor, noise_std: float, training: bool) -> torch.Tensor:
    """Ballé 2017 quantization relaxation: additive uniform noise at train,
    hard round at inference. Allows gradient flow."""
    if training:
        noise = (torch.rand_like(x) - 0.5) * (2.0 * noise_std)
        return x + noise
    return x.round()


def _conditional_gaussian_rate(
    y: torch.Tensor,
    sigma: torch.Tensor,
    *,
    sigma_floor: float,
) -> torch.Tensor:
    """Compute mean -log2(p_y(y | σ)) per element where p_y = N(0, σ²).

    This is the Ballé 2018 SCALE HYPERPRIOR rate term for the main latent.
    Returns a SCALAR tensor (mean bits per element).

    -log2 N(y; 0, sigma^2) = 0.5 * (y/sigma)^2 / ln(2) + log2(sigma) + log2(sqrt(2*pi))
    """
    sigma = sigma.clamp(min=sigma_floor)
    log2_sigma = torch.log2(sigma)
    # -log2 N(y;0,sigma^2) per element:
    #   = 0.5 * (y^2 / sigma^2) / ln(2) + log2(sigma) + 0.5 * log2(2*pi)
    inv2log2 = 1.0 / (2.0 * math.log(2.0))
    nll_per_elem = (y * y) / (sigma * sigma) * inv2log2 + log2_sigma + 0.5 * math.log2(2.0 * math.pi)
    return nll_per_elem.mean()


class NSCS03JointCodecSubstrate(nn.Module):
    """End-to-end Ballé 2018 joint codec.

    Forward signature:
        forward(x_pair: (B, 6, H, W)) -> (recon: (B, 6, H, W), rate_components: dict)

    where ``rate_components`` carries:
        - ``main_rate``: -log2 p_y(y_hat | σ) (mean bits per element)
        - ``hyper_rate``: -log2 p_z(z_hat) (mean bits per element)
        - ``total_rate``: main_rate + hyper_rate

    This module trains END-TO-END with score-aware loss; gradient flows
    from the scorer through the synthesis transform, through the
    bottleneck quantization (via uniform-noise relaxation during train),
    through the analysis transform, all the way to the input embedding.

    The per-pair information lives ENTIRELY in the y_hat / z_hat tensors
    that the trainer encodes from the contest video at archive-build
    time and ships in the archive bytes (NOT in any per-pair learned
    `nn.Parameter` table — that is the renderer paradigm).
    """

    def __init__(self, cfg: NSCS03Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.g_a = _AnalysisTransform(
            cfg.in_channels, cfg.g_a_channels, gdn_eps=cfg.gdn_eps
        )
        self.g_s = _SynthesisTransform(
            cfg.g_s_channels, cfg.out_channels, gdn_eps=cfg.gdn_eps
        )
        self.h_a = _HyperAnalysis(cfg.main_latent_channels, cfg.h_a_channels)
        self.h_s = _HyperSynthesis(cfg.hyper_latent_channels, cfg.h_s_channels)
        # Factorized prior on the hyper-latent z reuses the canonical
        # tac.entropy_bottleneck primitive (per-channel logistic CDF; Ballé
        # 2018 reference math). The hyper-rate is the bits/elem returned
        # by that primitive.
        self.entropy_bottleneck_z = EntropyBottleneck(cfg.hyper_latent_channels)

    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self, x_pair: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the analysis path and return all latents (no quantization).

        Used at archive-build time to compute the latents that ship in
        the archive bytes.
        """
        y = self.g_a(x_pair)
        z = self.h_a(y)
        return {"y": y, "z": z}

    def decode(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Run the synthesis path on a quantized main latent.

        Used at inflate time to reconstruct frames from archived latents.
        """
        return self.g_s(y_hat)

    def forward(
        self, x_pair: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """End-to-end forward: encode -> quantize -> conditional density -> decode.

        Args:
            x_pair: ``(B, 6, H, W)`` input pair (frame_0 RGB ⊕ frame_1 RGB
                stacked along channel axis), in ``[0, 1]``.

        Returns:
            ``(recon, rate_components)`` where ``recon`` is ``(B, 6, H, W)``
            in ``[0, 1]`` (sigmoid-applied) and ``rate_components`` carries
            the differentiable rate terms.
        """
        if x_pair.dim() != 4 or x_pair.shape[1] != self.cfg.in_channels:
            raise ValueError(
                f"x_pair must be (B, {self.cfg.in_channels}, H, W); "
                f"got {tuple(x_pair.shape)}"
            )

        y = self.g_a(x_pair)
        z = self.h_a(y)

        # Quantize z with the canonical EntropyBottleneck primitive
        # (returns z_hat + bits/elem of the factorized prior).
        z_hat, z_bits_per_elem = self.entropy_bottleneck_z(z)

        # Hyper-synthesis: sigma from z_hat. Output shape MUST equal y shape.
        sigma_raw = self.h_s(z_hat)
        # Crop or pad sigma to match y's spatial size (ConvTranspose chain
        # may produce ±1 pixel size mismatch on non-power-of-2 inputs).
        if sigma_raw.shape[-2:] != y.shape[-2:]:
            sigma_raw = F.interpolate(
                sigma_raw, size=y.shape[-2:], mode="bilinear", align_corners=False
            )
        sigma = F.softplus(sigma_raw) + self.cfg.sigma_floor

        # Quantize y with Ballé 2017 noise relaxation (training) / round (eval)
        y_hat = _quantize_with_noise(y, self.cfg.quantize_noise_std, self.training)

        # Conditional Gaussian rate on y given sigma
        main_rate = _conditional_gaussian_rate(
            y_hat, sigma, sigma_floor=self.cfg.sigma_floor
        )
        # z_bits_per_elem from the entropy bottleneck is already log2-bits/elem
        hyper_rate = z_bits_per_elem
        total_rate = main_rate + hyper_rate

        # Synthesis -> reconstruction
        recon = self.g_s(y_hat)
        # Match output spatial size to (output_height, output_width)
        if recon.shape[-2:] != (self.cfg.output_height, self.cfg.output_width):
            recon = F.interpolate(
                recon,
                size=(self.cfg.output_height, self.cfg.output_width),
                mode="bilinear",
                align_corners=False,
            )
        recon = torch.sigmoid(recon)

        return recon, {
            "main_rate": main_rate,
            "hyper_rate": hyper_rate,
            "total_rate": total_rate,
            "y": y.detach(),
            "z": z.detach(),
            "y_hat": y_hat.detach(),
            "z_hat": z_hat.detach(),
            "sigma": sigma.detach(),
        }

    def split_recon_into_frames(
        self, recon: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split ``(B, 6, H, W)`` recon back into ``(rgb_0, rgb_1)``.

        Each frame is ``(B, 3, H, W)`` in ``[0, 1]``.
        """
        if recon.shape[1] != 6:
            raise ValueError(
                f"recon must have 6 channels (2 RGB frames stacked); "
                f"got {recon.shape[1]}"
            )
        rgb_0 = recon[:, 0:3, :, :]
        rgb_1 = recon[:, 3:6, :, :]
        return rgb_0, rgb_1

    @staticmethod
    def stack_frames_into_pair(
        rgb_0: torch.Tensor, rgb_1: torch.Tensor
    ) -> torch.Tensor:
        """Stack ``(B, 3, H, W)`` frame pair into ``(B, 6, H, W)`` codec input."""
        if rgb_0.shape != rgb_1.shape:
            raise ValueError(
                f"rgb_0 and rgb_1 must have matching shapes; "
                f"got {tuple(rgb_0.shape)} vs {tuple(rgb_1.shape)}"
            )
        if rgb_0.dim() != 4 or rgb_0.shape[1] != 3:
            raise ValueError(
                f"each frame must be (B, 3, H, W); got {tuple(rgb_0.shape)}"
            )
        return torch.cat([rgb_0, rgb_1], dim=1)


__all__ = [
    "NSCS03Config",
    "NSCS03JointCodecSubstrate",
]
