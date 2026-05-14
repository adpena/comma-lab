"""IBDecoder — reconstructs frames from low-dim latent z.

Per the C6 across-class shift hypothesis: the decoder is the architectural
counterpart of the encoder bottleneck. Both ``z → frame_0`` and ``z → frame_1``
are produced (the contest scorer slices x[:, -1, ...] for SegNet so frame_1 is
the dominant axis; PoseNet sees both).

Architecture (small NeRV-style PixelShuffle decoder; mirrors sane_hnerv but
much smaller because z is the only conditioning input):

    Input: z (B, latent_dim)
       ↓
    Linear latent_dim → embed_dim · h0 · w0
       ↓
    Reshape (B, embed_dim, h0, w0)
       ↓
    Block 0: Conv → sin → PixelShuffle(2)            → (B, c1, 2h0, 2w0)
    Block 1: Conv → sin → PixelShuffle(2)            → (B, c2, 4h0, 4w0)
    ... (num_upsample_blocks blocks)
    Block N-1: Conv → sin → PixelShuffle(2)
       ↓
    Bilinear interpolate → (B, c_final, H_out, W_out)
       ↓
    Head rgb_0: Conv c_final → 3 → sigmoid           → (B, 3, H_out, W_out)
    Head rgb_1: Conv c_final → 3 → sigmoid           → (B, 3, H_out, W_out)

Per CLAUDE.md FORBIDDEN_PATTERNS:
- NO scorer load in this module.
- NO silent device defaults (caller passes device).
- NO /tmp paths.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SinAct(nn.Module):
    def __init__(self, w: float = 30.0) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.sin(self.w * x)


class _UpBlock(nn.Module):
    """Conv → sin → PixelShuffle(2)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        sin_freq: float,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        # PixelShuffle(2) needs 4x output channels in the conv before shuffle
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size, padding=kernel_size // 2)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.shuffle(self.act(self.conv(x)))


class IBDecoder(nn.Module):
    """Reconstruct ``(rgb_0, rgb_1)`` from a low-dim latent z.

    Args:
        latent_dim: dimensionality of input z (default 24).
        embed_dim: initial spatial-grid channel count (default 32).
        initial_grid_h: initial spatial grid height (default 3).
        initial_grid_w: initial spatial grid width (default 4).
        decoder_channels: per-block output channels (default (24, 20, 16, 12, 8, 6)).
        num_upsample_blocks: number of PixelShuffle(2) blocks (default 6;
            3x4 → 192x256 → bilinear → 384x512).
        output_height: final RGB height (default 384).
        output_width: final RGB width (default 512).
        sin_freq: SIREN activation frequency (default 30.0).
    """

    def __init__(
        self,
        latent_dim: int = 24,
        embed_dim: int = 32,
        initial_grid_h: int = 3,
        initial_grid_w: int = 4,
        decoder_channels: tuple[int, ...] = (24, 20, 16, 12, 8, 6),
        num_upsample_blocks: int = 6,
        output_height: int = 384,
        output_width: int = 512,
        sin_freq: float = 30.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or latent_dim > 256:
            raise ValueError(f"latent_dim must be in (0, 256]; got {latent_dim}")
        if num_upsample_blocks <= 0 or num_upsample_blocks > 10:
            raise ValueError(
                f"num_upsample_blocks must be in (0, 10]; got {num_upsample_blocks}"
            )

        self.latent_dim = int(latent_dim)
        self.embed_dim = int(embed_dim)
        self.initial_grid_h = int(initial_grid_h)
        self.initial_grid_w = int(initial_grid_w)
        self.decoder_channels = tuple(int(c) for c in decoder_channels)
        self.num_upsample_blocks = int(num_upsample_blocks)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.sin_freq = float(sin_freq)

        if len(self.decoder_channels) < self.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(self.decoder_channels)}) must have at least "
                f"num_upsample_blocks ({self.num_upsample_blocks}) entries"
            )

        # latent → initial spatial grid
        self.latent_embed = nn.Linear(
            self.latent_dim,
            self.embed_dim * self.initial_grid_h * self.initial_grid_w,
        )

        channels = [self.embed_dim] + list(self.decoder_channels)
        blocks: list[nn.Module] = []
        for i in range(self.num_upsample_blocks):
            blocks.append(_UpBlock(channels[i], channels[i + 1], self.sin_freq))
        self.blocks = nn.ModuleList(blocks)

        final_ch = channels[self.num_upsample_blocks]
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)

        self._siren_init()

    def _siren_init(self) -> None:
        w = self.sin_freq
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
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode z → (rgb_0, rgb_1).

        Args:
            z: latent ``(B, latent_dim)``.

        Returns:
            (rgb_0, rgb_1): each ``(B, 3, H, W)`` in [0, 1].
        """
        if z.dim() != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"z must be (B, {self.latent_dim}); got {tuple(z.shape)}"
            )

        h = self.latent_embed(z)
        h = h.view(-1, self.embed_dim, self.initial_grid_h, self.initial_grid_w)
        for block in self.blocks:
            h = block(h)
        if h.shape[-2:] != (self.output_height, self.output_width):
            h = F.interpolate(
                h,
                size=(self.output_height, self.output_width),
                mode="bilinear",
                align_corners=False,
            )
        rgb_0 = torch.sigmoid(self.head_rgb_0(h))
        rgb_1 = torch.sigmoid(self.head_rgb_1(h))
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["IBDecoder"]
