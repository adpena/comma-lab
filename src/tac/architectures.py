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
        y = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
        y_norm = y / 255.0
        mean = y_norm.mean(dim=(2, 3))
        std = y_norm.std(dim=(2, 3), unbiased=False)
        dx = y_norm[..., :, 1:] - y_norm[..., :, :-1]
        dy = y_norm[..., 1:, :] - y_norm[..., :-1, :]
        edge = 0.5 * (dx.abs().mean(dim=(2, 3)) + dy.abs().mean(dim=(2, 3)))
        return torch.cat([mean, std, edge], dim=1)

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


VARIANTS = {
    # Canonical names
    "standard": PostFilter,
    "dilated": DilatedPostFilter,
    "gated_dilated": GatedDilatedPostFilter,
    "pixelshuffle": PixelShufflePostFilter,
    "psd": PSDPostFilter,
    "depthwise": DepthwisePostFilter,
    "luma": LumaPostFilter,
    "film": FiLMPostFilter,
    "pair_aware": PairAwarePostFilter,
    # New techniques
    "luma_dilated": LumaOnlyDilatedPostFilter,
    "dual_head": DualHeadPostFilter,
    "content_adaptive": ContentAdaptivePostFilter,
    "pixelshuffle_upscale": PixelShuffleUpscaleFilter,
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
