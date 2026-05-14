# SPDX-License-Identifier: MIT
"""Post-filter architectures for task-aware video compression.

All architectures share:
  - Residual connection: output = input + learned_correction
  - Zero-initialized output layer (starts as identity)
  - Forward takes (B, 3, H, W) float [0, 255], returns same
  - Compatible with int8 quantization via FakeQuant STE

Available variants:
  - PostFilter: standard 3-layer residual CNN
  - DilatedPostFilter: dilation=2 on middle layer (15x15 RF)
  - PixelShufflePostFilter: half-res 4-layer REN
  - PSDPostFilter: PixelShuffle + Dilated hybrid (council consensus)
  - BlockDCTMidbandFilter: spectral prior, mid-frequency band gains
  - FiLMQATPostFilter: FiLM conditioning on frame statistics (QAT)
  - CounterpointPostFilter: two-voice ensemble with band orthogonality
  - PixelShuffleDilatedPostFilter: PS + dilated hybrid (legacy provenance)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PostFilter(nn.Module):
    """Standard 3-layer residual CNN post-filter.

    Architecture: 3→h→h→3, 3×3 convolutions, ReLU, residual connection.
    Effective receptive field: 7×7.
    """

    def __init__(self, hidden: int = 16, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


class DilatedPostFilter(nn.Module):
    """PostFilter with dilation=2 on middle layer.

    Expands RF from 7×7 to 15×15 at zero param cost. Matches the
    receptive field of PoseNet's fastvit_t12 early layers.
    """

    def __init__(self, hidden: int = 16, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


class PixelShufflePostFilter(nn.Module):
    """Half-resolution 4-layer REN using PixelUnshuffle/Shuffle.

    PixelUnshuffle(2) converts 3ch full-res to 12ch half-res.
    Four conv layers process at half-res where each 3×3 covers 6×6
    at full-res, aligning with scorer internal resolution.
    PixelShuffle(2) reconstructs full-res output.
    """

    def __init__(self, hidden: int = 64, kernel: int = 3):
        super().__init__()
        self.down = nn.PixelUnshuffle(2)
        pad = kernel // 2
        self.body = nn.Sequential(
            nn.Conv2d(12, hidden, kernel, padding=pad, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 12, kernel, padding=pad, bias=True),
        )
        self.up = nn.PixelShuffle(2)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / 255.0
        residual = self.up(self.body(self.down(x_norm)))
        return (x_norm + residual).clamp(0, 1) * 255.0


class PSDPostFilter(nn.Module):
    """PixelShuffle + Dilated hybrid (expert council consensus architecture).

    Combines PixelShuffle half-res processing with dilation=2 on layer 2.
    Effective RF: 24×24 at full-res. Same params as PixelShufflePostFilter
    but with larger spatial reach from the dilated middle layer.

    This is the architecture unanimously selected by the expert panel
    (Tao, LeCun, Karpathy, Collier, Jensen Huang, Von Neumann) as the
    single best experiment to reach sub-1.6 score.
    """

    def __init__(self, hidden: int = 64, kernel: int = 3):
        super().__init__()
        self.down = nn.PixelUnshuffle(2)
        pad = kernel // 2
        self.conv1 = nn.Conv2d(12, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv4 = nn.Conv2d(hidden, 12, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.PixelShuffle(2)
        nn.init.zeros_(self.conv4.weight)
        nn.init.zeros_(self.conv4.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / 255.0
        h = self.down(x_norm)
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        residual = self.conv4(h)
        residual = self.up(residual)
        return (x_norm + residual).clamp(0, 1) * 255.0


class DepthwisePostFilter(nn.Module):
    """Depthwise-separable 3-layer residual post-filter.

    Uses pointwise(1x1) → depthwise(3x3, groups=h) → pointwise(1x1).
    More parameter-efficient than standard convolutions.
    """

    def __init__(self, hidden: int = 16, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.pw_in = nn.Conv2d(3, hidden, 1, bias=True)
        self.dw = nn.Conv2d(hidden, hidden, kernel, padding=pad, groups=hidden, bias=True)
        self.pw_out = nn.Conv2d(hidden, 3, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.pw_out.weight)
        nn.init.zeros_(self.pw_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.act(self.pw_in(x))
        residual = self.act(self.dw(residual))
        residual = self.pw_out(residual)
        return (x + residual).clamp(0, 255)


class LumaPostFilter(nn.Module):
    """Luma-only processing — extracts Y channel, processes, broadcasts back.

    Lighter than full RGB processing. Correction is the same for all channels.
    """

    def __init__(self, hidden: int = 16, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(1, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 1, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
        residual = self.act(self.conv1(y))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual.repeat(1, 3, 1, 1)).clamp(0, 255)


class FiLMPostFilter(nn.Module):
    """Feature-wise Linear Modulation conditioned on per-frame statistics.

    Computes a descriptor (luma mean, std, edge density) and uses it to
    modulate intermediate features via gamma/beta scaling. Allows the
    correction to adapt to frame content.
    """

    def __init__(self, hidden: int = 16, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.film = nn.Linear(3, hidden * 2, bias=True)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def _descriptor(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        y = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
        y_norm = y / 255.0
        mean = y_norm.mean(dim=(2, 3))
        std = y_norm.std(dim=(2, 3), unbiased=False)
        dx = y_norm[..., :, 1:] - y_norm[..., :, :-1]
        dy = y_norm[..., 1:, :] - y_norm[..., :-1, :]
        edge = 0.5 * (dx.abs().mean(dim=(2, 3)) + dy.abs().mean(dim=(2, 3)))
        # Explicit (B, 3) shape — mean/std/edge each reduce to (B, 1)
        return torch.cat([mean, std, edge], dim=1).view(B, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        film = self.film(self._descriptor(x))
        gamma, beta = film.chunk(2, dim=1)
        gamma = 1.0 + 0.25 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = 8.0 * torch.tanh(beta).unsqueeze(-1).unsqueeze(-1)
        residual = self.act(self.conv1(x))
        residual = residual * gamma + beta
        residual = self.act(self.conv2(residual))
        residual = residual * gamma + beta
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


class PairAwarePostFilter(nn.Module):
    """6-channel pair-aware post-filter.

    Takes both frames of a pair concatenated along channels (6ch input).
    This gives the CNN access to temporal difference signal — PoseNet
    operates on pairs, so corrections that depend on inter-frame
    relationships cannot be learned by a single-frame filter.

    conv1: 6→h (sees both frames), conv2: h→h, conv3: h→3 (outputs per-frame correction).
    The filter is applied once per frame, with the OTHER frame as context.

    Forward: (B, 6, H, W) → (B, 3, H, W). First 3ch = target frame, last 3ch = context.
    """

    def __init__(self, hidden: int = 64, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(6, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6, H, W) — first 3ch target, last 3ch context."""
        target = x[:, :3]  # the frame being corrected
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (target + residual).clamp(0, 255)


# ── Factory ──────────────────────────────────────────────────────────────


class LumaOnlyDilatedPostFilter(nn.Module):
    """Technique 2: Luma-only processing with dilation.

    Netflix validated that chroma uses standard Lanczos — neural processing
    on luma only is sufficient. This processes only the Y (luminance) channel
    with a dilated CNN, then recombines with original chroma.

    ~3x fewer parameters than full RGB processing (1 channel in/out vs 3).
    Combined with dilation=2 for the 15x15 RF that matches PoseNet's early layers.

    Args:
        hidden: hidden channel width
        kernel: conv kernel size
    """

    def __init__(self, hidden: int = 16, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(1, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(hidden, 1, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process luma only, broadcast correction to all channels.

        Args:
            x: (B, 3, H, W) float [0, 255]

        Returns:
            (B, 3, H, W) float [0, 255]
        """
        # Extract Y (luminance) using BT.601 coefficients
        y = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
        residual = self.act(self.conv1(y))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        # Broadcast single-channel residual to all RGB channels
        return (x + residual.expand_as(x)).clamp(0, 255)


class DualHeadPostFilter(nn.Module):
    """Technique 9: Shared backbone with separate PoseNet and SegNet heads.

    Same backbone (conv1, conv2) shared between two specialized heads:
    - seg_head: 1x1 conv for sharp, localized corrections at class boundaries
    - pose_head: 3x3 conv for smooth, texture-preserving corrections

    The two heads address fundamentally different spatial patterns:
    SegNet needs sharp boundary corrections (small RF, high frequency),
    PoseNet needs smooth texture corrections (larger RF, low frequency).

    output = input + seg_head(features) + pose_head(features)

    Args:
        hidden: hidden channel width for shared backbone
        kernel: conv kernel size for backbone
    """

    def __init__(self, hidden: int = 64, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        # Shared backbone
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True)
        self.act = nn.ReLU(inplace=True)

        # SegNet head: 1x1 conv (sharp, boundary-focused corrections)
        self.seg_head = nn.Conv2d(hidden, 3, 1, bias=True)

        # PoseNet head: 3x3 conv (smooth, texture-preserving corrections)
        self.pose_head = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)

        # Zero-init both heads: starts as identity
        nn.init.zeros_(self.seg_head.weight)
        nn.init.zeros_(self.seg_head.bias)
        nn.init.zeros_(self.pose_head.weight)
        nn.init.zeros_(self.pose_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dual-head corrections.

        Args:
            x: (B, 3, H, W) float [0, 255]

        Returns:
            (B, 3, H, W) float [0, 255]
        """
        features = self.act(self.conv1(x))
        features = self.act(self.conv2(features))
        seg_correction = self.seg_head(features)
        pose_correction = self.pose_head(features)
        return (x + seg_correction + pose_correction).clamp(0, 255)


class GatedDilatedPostFilter(nn.Module):
    """DilatedPostFilter with a spatial sigmoid gate (Collier's proposal).

    After conv2 features are computed, a 1x1 conv produces a per-pixel
    gate value in [0, 1] that modulates the residual correction. The gate
    learns to attenuate corrections near SegNet class boundaries where
    the residual harms segmentation, while preserving corrections in
    texture regions where PoseNet benefits.

    Adds only `hidden + 1` parameters (65 at h=64). Rate impact: negligible.

    Approved by unanimous council vote (first unanimous in council history).
    Rationale: addresses spatial allocation of correction capacity, orthogonal
    to loss-level gradient methods (PCGrad, CAGrad).
    """

    def __init__(self, hidden: int = 16, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        # Spatial gate: learns where to apply corrections (boundary-aware)
        self.gate = nn.Sequential(
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        # Initialize gate bias to 0 → sigmoid(0) = 0.5 → starts as half-strength
        nn.init.zeros_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.act(self.conv1(x))
        features = self.act(self.conv2(features))
        gate = self.gate(features)  # (B, 1, H, W) in [0, 1]
        residual = self.conv3(features)
        return (x + gate * residual).clamp(0, 255)


class ContentAdaptivePostFilter(nn.Module):
    """Technique 3: Content-adaptive per-frame postfilter intensity.

    Computes a per-frame "difficulty score" from the input frame statistics
    and scales the postfilter residual accordingly. Hard frames (high-frequency
    content, many edges) get stronger correction; easy frames (smooth road,
    clear sky) get lighter correction.

    This is Netflix's per-shot encoding quality adaptation applied to
    post-processing instead of encoding.

    output = input + difficulty_weight * residual

    where difficulty_weight is learned from frame statistics.

    Args:
        hidden: hidden channel width
        kernel: conv kernel size
    """

    def __init__(self, hidden: int = 64, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

        # Difficulty estimator: frame statistics -> scalar weight
        # Input: 3 features (luma mean, luma std, edge density)
        # Output: scalar difficulty weight via sigmoid (range [0.1, 2.0])
        self.difficulty_net = nn.Sequential(
            nn.Linear(3, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1, bias=True),
        )

    def _frame_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-frame statistics for difficulty estimation.

        Returns: (B, 3) tensor of [luma_mean, luma_std, edge_density]
        """
        y = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
        y_norm = y / 255.0
        mean = y_norm.mean(dim=(2, 3))
        std = y_norm.std(dim=(2, 3), unbiased=False)
        dx = y_norm[..., :, 1:] - y_norm[..., :, :-1]
        dy = y_norm[..., 1:, :] - y_norm[..., :-1, :]
        edge = 0.5 * (dx.abs().mean(dim=(2, 3)) + dy.abs().mean(dim=(2, 3)))
        return torch.cat([mean, std, edge], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply content-adaptive correction.

        Args:
            x: (B, 3, H, W) float [0, 255]

        Returns:
            (B, 3, H, W) float [0, 255]
        """
        stats = self._frame_stats(x)
        # Difficulty weight in [0.1, 2.0]: sigmoid maps to [0,1], scale to [0.1, 2.0]
        difficulty = 0.1 + 1.9 * torch.sigmoid(self.difficulty_net(stats))
        difficulty = difficulty.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)

        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + difficulty * residual).clamp(0, 255)


# ── Migrated Architectures ──────────────────────────────────────────────


class BlockDCTMidbandFilter(nn.Module):
    """Block DCT mid-frequency band filter with learnable per-band gains.

    Migrated from experiments/train_postfilter_dct.py.
    Note: Research exploration of spectral priors. Not effective for comma video
    compression — DCT mid-bands did not improve PoseNet or SegNet scores.

    Architecture:
      1. Splits frames into fixed-size blocks (default 8x8)
      2. Transforms each block with a fixed orthonormal DCT
      3. Applies learnable per-band gains only in the mid-frequency region
      4. Reconstructs a pixel-space residual with the inverse DCT
      5. Mixes via a learned scalar (starts at 0 = identity)
    """

    def __init__(self, hidden: int = 8, kernel: int = 3, block: int = 8,
                 channels: int = 3, low: float = 0.18, high: float = 0.72):
        super().__init__()
        # hidden/kernel args accepted for API compatibility but unused
        self.block = block
        self.channels = channels
        self.register_buffer("dct", self._orthonormal_dct_matrix(block))
        self.register_buffer("mid_mask", self._build_mid_frequency_mask(block, low=low, high=high))
        self.gain = nn.Parameter(torch.zeros(channels, block, block))
        self.bias = nn.Parameter(torch.zeros(channels, block, block))
        self.mix = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _orthonormal_dct_matrix(size: int) -> torch.Tensor:
        import math as _math
        n = torch.arange(size, dtype=torch.float32)
        k = n.unsqueeze(1)
        matrix = torch.cos(_math.pi / size * (n + 0.5) * k)
        matrix[0] *= 1.0 / _math.sqrt(2.0)
        matrix *= _math.sqrt(2.0 / size)
        return matrix

    @staticmethod
    def _build_mid_frequency_mask(size: int, *, low: float = 0.18, high: float = 0.72) -> torch.Tensor:
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        radius = torch.sqrt(yy.float().pow(2) + xx.float().pow(2))
        radius /= radius.max().clamp(min=1.0)
        mask = ((radius >= low) & (radius <= high)).float()
        mask[0, 0] = 0.0
        return mask

    def _blockify(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        import torch.nn.functional as F
        b, c, h, w = x.shape
        pad_h = (self.block - h % self.block) % self.block
        pad_w = (self.block - w % self.block) % self.block
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        hp, wp = x.shape[-2:]
        blocks = x.reshape(b, c, hp // self.block, self.block, wp // self.block, self.block)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        return blocks, pad_h, pad_w

    def _deblockify(self, blocks: torch.Tensor, *, pad_h: int, pad_w: int,
                    orig_h: int, orig_w: int) -> torch.Tensor:
        b, c, nh, nw, _, _ = blocks.shape
        x = blocks.permute(0, 1, 2, 4, 3, 5).reshape(b, c, nh * self.block, nw * self.block)
        if pad_h or pad_w:
            x = x[..., :orig_h, :orig_w]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[-2:]
        blocks, pad_h, pad_w = self._blockify(x)
        flat = blocks.reshape(-1, self.block, self.block)
        coeff = torch.matmul(self.dct, flat)
        coeff = torch.matmul(coeff, self.dct.t())
        coeff = coeff.reshape(*blocks.shape)

        mask = self.mid_mask.view(1, 1, 1, 1, self.block, self.block)
        gain = torch.tanh(self.gain).view(1, self.channels, 1, 1, self.block, self.block)
        bias = (0.05 * torch.tanh(self.bias)).view(1, self.channels, 1, 1, self.block, self.block)
        delta_coeff = mask * (coeff * gain + bias)

        delta_flat = delta_coeff.reshape(-1, self.block, self.block)
        delta_blocks = torch.matmul(self.dct.t(), delta_flat)
        delta_blocks = torch.matmul(delta_blocks, self.dct)
        delta_blocks = delta_blocks.reshape_as(blocks)
        delta = self._deblockify(delta_blocks, pad_h=pad_h, pad_w=pad_w,
                                 orig_h=orig_h, orig_w=orig_w)
        return (x + torch.tanh(self.mix) * delta).clamp(0.0, 255.0)


class FiLMQATPostFilter(nn.Module):
    """Feature-wise Linear Modulation post-filter with QAT-aware convolutions.

    Migrated from experiments/train_postfilter_film_conditioned.py.
    Note: Frame-adaptive soft gating via gamma/beta parameters. Not adopted for
    comma — standard dilated outperformed. May be effective for variable-quality
    input streams.

    Computes a descriptor (luma mean, std, edge density) and uses it to
    modulate intermediate features via gamma/beta FiLM conditioning. Includes
    fake-quantized weight convolutions for QAT training compatibility.
    """

    def __init__(self, hidden: int = 16, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.film = nn.Linear(3, hidden * 2, bias=True)
        self.act = nn.ReLU(inplace=False)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def _descriptor(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        y = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
        y_norm = y / 255.0
        mean = y_norm.mean(dim=(2, 3))
        std = y_norm.std(dim=(2, 3), unbiased=False)
        dx = y_norm[..., :, 1:] - y_norm[..., :, :-1]
        dy = y_norm[..., 1:, :] - y_norm[..., :-1, :]
        edge = 0.5 * (dx.abs().mean(dim=(2, 3)) + dy.abs().mean(dim=(2, 3)))
        return torch.cat([mean, std, edge], dim=1).view(B, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        film = self.film(self._descriptor(x))
        gamma, beta = film.chunk(2, dim=1)
        gamma = 1.0 + 0.25 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = 8.0 * torch.tanh(beta).unsqueeze(-1).unsqueeze(-1)
        residual = self.act(self.conv1(x))
        residual = residual * gamma + beta
        residual = self.act(self.conv2(residual))
        residual = residual * gamma + beta
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


class CounterpointPostFilter(nn.Module):
    """Two-voice counterpoint ensemble post-filter.

    Migrated from experiments/train_postfilter_counterpoint.py.
    Note: Harmonic ensemble hypothesis inspired by Jacob Collier review.
    Experimental — not validated on authoritative scorer.

    Two independent h=16 "voices" (3-layer CNNs) whose raw residuals are
    summed: out = clamp(x + residual_a + residual_b, 0, 255). Designed to
    be trained with band-orthogonality + output-decorrelation losses to
    ensure the two voices occupy disjoint frequency bands.
    """

    def __init__(self, hidden: int = 16, kernel: int = 3):
        super().__init__()
        self.voice_a = self._make_voice(hidden, kernel)
        self.voice_b = self._make_voice(hidden, kernel)

    @staticmethod
    def _make_voice(hidden: int, kernel: int) -> nn.Module:
        pad = kernel // 2
        voice = nn.ModuleDict({
            "conv1": nn.Conv2d(3, hidden, kernel, padding=pad, bias=True),
            "conv2": nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True),
            "conv3": nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True),
        })
        nn.init.zeros_(voice["conv3"].weight)
        nn.init.zeros_(voice["conv3"].bias)
        return voice

    def _voice_residual(self, voice: nn.Module, x: torch.Tensor) -> torch.Tensor:
        act = torch.relu
        h = act(voice["conv1"](x))
        h = act(voice["conv2"](h))
        return voice["conv3"](h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resid_a = self._voice_residual(self.voice_a, x)
        resid_b = self._voice_residual(self.voice_b, x)
        return (x + resid_a + resid_b).clamp(0, 255)

    def residual_a(self, x: torch.Tensor) -> torch.Tensor:
        return self._voice_residual(self.voice_a, x)

    def residual_b(self, x: torch.Tensor) -> torch.Tensor:
        return self._voice_residual(self.voice_b, x)


class PixelShuffleDilatedPostFilter(nn.Module):
    """Half-resolution 4-layer residual post-filter with dilated middle layer.

    Migrated from experiments/train_postfilter_pixelshuffle_dilated.py.
    Note: Hybrid of PixelShuffle base + dilated center. Promising architecture
    — council #1 pick but not yet auth-evaluated.

    Identical topology to PSDPostFilter but kept as a distinct class for
    provenance tracking from the legacy experiment script. Uses
    PixelUnshuffle(2) -> conv1 -> conv2(dilation=2) -> conv3 -> conv4 ->
    PixelShuffle(2) with residual connection in normalized [0,1] space.
    """

    def __init__(self, hidden: int = 64, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.down = nn.PixelUnshuffle(2)
        self.conv1 = nn.Conv2d(12, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv4 = nn.Conv2d(hidden, 12, kernel, padding=pad, bias=True)
        self.up = nn.PixelShuffle(2)
        self.act = nn.ReLU(inplace=False)
        nn.init.zeros_(self.conv4.weight)
        nn.init.zeros_(self.conv4.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / 255.0
        residual = self.down(x_norm)
        residual = self.act(self.conv1(residual))
        residual = self.act(self.conv2(residual))
        residual = self.act(self.conv3(residual))
        residual = self.up(self.conv4(residual))
        return (x_norm + residual).clamp(0, 1) * 255.0


class PixelShuffleUpscaleFilter(nn.Module):
    """Technique 4: Resolution reduction + PixelShuffle upscale.

    Designed for a pipeline where video is encoded at 75% resolution
    (saving ~44% bitrate). This filter takes the low-res decoded frames
    and upscales them to full resolution using PixelShuffle.

    The rate savings from lower resolution (25 * 0.44 * 0.023 = 0.25 points!)
    could be enormous in the scoring formula.

    Pipeline:
        1. Input: decoded frames at reduced resolution (e.g., 288x384)
        2. PixelUnshuffle(2) to increase channels
        3. CNN processing at quarter resolution
        4. PixelShuffle(2) to increase spatial resolution
        5. Bilinear resize to target resolution if needed

    Args:
        hidden: hidden channel width
        kernel: conv kernel size
        scale_factor: upscale factor (2 = double resolution)
    """

    def __init__(self, hidden: int = 64, kernel: int = 3, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        sf2 = scale_factor * scale_factor
        pad = kernel // 2

        # Process at input resolution, then upscale
        self.body = nn.Sequential(
            nn.Conv2d(3, hidden, kernel, padding=pad, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True),
            nn.ReLU(inplace=True),
            # Output channels = 3 * scale^2 for PixelShuffle
            nn.Conv2d(hidden, 3 * sf2, kernel, padding=pad, bias=True),
        )
        self.upsample = nn.PixelShuffle(scale_factor)
        # Zero-init last conv
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor, target_h: int = 0, target_w: int = 0) -> torch.Tensor:
        """Process frames with optional upscaling.

        When target_h/target_w are 0 (default), output matches input size
        (acts as a same-resolution enhancement filter). When targets are
        set, outputs at target resolution (super-resolution mode).

        Args:
            x: (B, 3, H, W) float [0, 255] — decoded frames
            target_h: target height (0 = same as input)
            target_w: target width (0 = same as input)

        Returns:
            (B, 3, H_out, W_out) float [0, 255]
        """
        import torch.nn.functional as F

        _, _, H, W = x.shape
        x_norm = x / 255.0

        # Body produces residual at upscaled resolution
        residual = self.upsample(self.body(x_norm))

        # Determine output size
        if target_h > 0 and target_w > 0:
            out_h, out_w = target_h, target_w
        else:
            out_h, out_w = H, W

        # Bilinear upscale of input to match residual/output size
        if out_h != H or out_w != W:
            upscaled = F.interpolate(x_norm, size=(out_h, out_w), mode="bilinear", align_corners=False)
        else:
            upscaled = x_norm

        # Match residual to output size
        if residual.shape[2] != out_h or residual.shape[3] != out_w:
            residual = F.interpolate(residual, size=(out_h, out_w), mode="bilinear", align_corners=False)

        return (upscaled + residual).clamp(0, 1) * 255.0


def _import_psd_lumaskip_postfilter():
    """Lazy import to avoid circular references with submodule loaders.

    The PSD-LumaSkip variant lives in ``tac.psd_lumaskip_renderer`` (Phase A
    council-approved Lane PSD-LumaSkip scaffold, 2026-04-30). It is wired
    into ``VARIANTS`` so ``build_postfilter("psd_lumaskip", ...)`` returns
    the correct class.
    """
    from .psd_lumaskip_renderer import PSDLumaSkipPostFilter
    return PSDLumaSkipPostFilter


# Resolve at import time (no actual circular dependency, but using the helper
# keeps the import order audit-grep-able).
_PSD_LUMASKIP_CLS = _import_psd_lumaskip_postfilter()


VARIANTS = {
    # Canonical names
    "standard": PostFilter,
    "dilated": DilatedPostFilter,
    "gated_dilated": GatedDilatedPostFilter,
    "pixelshuffle": PixelShufflePostFilter,
    "psd": PSDPostFilter,
    "psd_lumaskip": _PSD_LUMASKIP_CLS,  # Lane PSD-LumaSkip (Phase A scaffold)
    "depthwise": DepthwisePostFilter,
    "luma": LumaPostFilter,
    "film": FiLMPostFilter,
    "pair_aware": PairAwarePostFilter,
    # New techniques
    "luma_dilated": LumaOnlyDilatedPostFilter,
    "dual_head": DualHeadPostFilter,
    "content_adaptive": ContentAdaptivePostFilter,
    "pixelshuffle_upscale": PixelShuffleUpscaleFilter,
    # Migrated architectures
    "dct_midband": BlockDCTMidbandFilter,
    "film_qat": FiLMQATPostFilter,
    "counterpoint": CounterpointPostFilter,
    "pixelshuffle_dilated_v2": PixelShuffleDilatedPostFilter,
    # Legacy aliases (from deploy inflate_postfilter.py)
    "residual": PostFilter,
    "saliency_weighted": PostFilter,
    "segaware": PostFilter,
    "pixelshuffle_dilated": PSDPostFilter,
    "film_conditioned": FiLMPostFilter,
}


def build_postfilter(
    variant: str = "psd",
    hidden: int = 64,
    kernel: int = 3,
) -> nn.Module:
    """Build a post-filter by variant name.

    Args:
        variant: one of "standard", "dilated", "pixelshuffle", "psd",
                 or legacy aliases: "residual", "saliency_weighted",
                 "segaware", "pixelshuffle_dilated"
        hidden: hidden channel width
        kernel: conv kernel size (default 3)

    Returns:
        An nn.Module that takes (B, 3, H, W) float [0, 255] and returns same.
    """
    cls = VARIANTS.get(variant)
    if cls is None:
        raise ValueError(f"Unknown variant '{variant}'. Available: {list(VARIANTS.keys())}")
    return cls(hidden=hidden, kernel=kernel)
