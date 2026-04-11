"""Neural renderer: segment masks → RGB frames for GPU-lane submission.

Architecture reverse-engineered from mask2mask competitor (score 0.60),
then redesigned for clarity and integration with tac training infrastructure.

Pipeline:
    GT frames → frozen SegNet → 5-class masks → AV1 encode/decode →
    MaskRenderer → RGB frames → scored by frozen PoseNet + SegNet

Key classes:
    - ResBlock: residual convolution block with optional GroupNorm
    - MaskRenderer: U-Net that renders RGB frames from segmentation masks
    - MotionPredictor: predicts optical flow from consecutive mask pairs
    - PairGenerator: combines renderer + motion for PoseNet-compatible pairs

Design target: ~300K parameters (matching competitor's proven size).
Channel widths: 36→60→36 (compact) or 48→80→48 (extended capacity).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ─────────────────────────────────────────────────────


class ResBlock(nn.Module):
    """Pre-activation residual block: GroupNorm → ReLU → Conv → GroupNorm → ReLU → Conv.

    Uses GroupNorm (groups=1 = LayerNorm-like) instead of BatchNorm for
    stability with batch_size=1 during per-pair training.
    """

    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel, padding=pad, bias=False)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel, padding=pad, bias=False)
        self.act = nn.ReLU(inplace=True)
        # Zero-init second conv so residual starts as identity
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


# ── MaskRenderer ────────────────────────────────────────────────────────


class MaskRenderer(nn.Module):
    """U-Net that renders RGB frames from 5-class segmentation masks.

    Architecture:
        mask (H, W) → Embedding(5, embed_dim) → (B, embed_dim, H, W)
        → stem (conv + resblock)
        → down (strided conv + resblock)        [H/2, W/2]
        → bottleneck resblock
        → up (transposed conv + resblock)        [H, W]
        → head (1x1 conv → 3ch RGB)

    The skip connection from stem to decoder is critical — without it
    the upsampled features lose high-frequency spatial detail that PoseNet
    needs for accurate pose estimation.

    Args:
        num_classes: number of segmentation classes (5 for comma SegNet)
        embed_dim: embedding dimension per class (input channels to U-Net)
        base_ch: base channel width (stem and decoder)
        mid_ch: bottleneck channel width (wider for capacity)

    Competitor reference: base_ch=36, mid_ch=60, embed_dim=6 (~308K params in FP4).
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 6,
        base_ch: int = 36,
        mid_ch: int = 60,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Class embedding: each of 5 classes → learned embed_dim-dimensional vector
        self.embedding = nn.Embedding(num_classes, embed_dim)

        # Stem: embed_dim → base_ch at full resolution
        self.stem_conv = nn.Conv2d(embed_dim, base_ch, 3, padding=1, bias=True)
        self.stem_res = ResBlock(base_ch)

        # Downsample: base_ch → mid_ch at half resolution
        self.down_conv = nn.Conv2d(base_ch, mid_ch, 3, stride=2, padding=1, bias=True)
        self.down_res = ResBlock(mid_ch)

        # Bottleneck at half resolution
        self.bottleneck = ResBlock(mid_ch)

        # Upsample: mid_ch → base_ch at full resolution
        self.up_conv = nn.ConvTranspose2d(mid_ch, base_ch, 4, stride=2, padding=1, bias=True)
        self.up_res = ResBlock(base_ch)

        # Fusion after skip: base_ch (skip) + base_ch (upsampled) → base_ch
        self.fuse_conv = nn.Conv2d(base_ch * 2, base_ch, 1, bias=True)

        # Head: base_ch → 3 RGB channels
        self.head = nn.Conv2d(base_ch, 3, 1, bias=True)

        # Initialize head to produce mid-gray (128) — better starting point
        # than zeros for RGB output
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, 128.0)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """Render RGB frames from segmentation masks.

        Args:
            masks: (B, H, W) long tensor with values in [0, num_classes)

        Returns:
            (B, 3, H, W) float tensor in [0, 255]
        """
        # Embed: (B, H, W) → (B, H, W, embed_dim) → (B, embed_dim, H, W)
        x = self.embedding(masks)
        x = x.permute(0, 3, 1, 2).contiguous()

        # Stem (full res)
        stem = self.stem_conv(x)
        stem = self.stem_res(stem)

        # Down (half res)
        down = self.down_conv(stem)
        down = self.down_res(down)

        # Bottleneck
        mid = self.bottleneck(down)

        # Up (back to full res)
        up = self.up_conv(mid)
        # Handle potential size mismatch from odd dimensions
        if up.shape[2:] != stem.shape[2:]:
            up = F.interpolate(up, size=stem.shape[2:], mode="bilinear", align_corners=False)
        up = self.up_res(up)

        # Skip connection: concatenate stem features with upsampled
        fused = torch.cat([stem, up], dim=1)
        fused = self.fuse_conv(fused)

        # Head: produce RGB
        rgb = self.head(fused)
        return rgb.clamp(0.0, 255.0)

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── MotionPredictor ─────────────────────────────────────────────────────


_coord_grid_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def make_coord_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Create normalized coordinate grid for grid_sample (cached).

    Returns: (1, H, W, 2) tensor with values in [-1, 1].
    """
    key = (h, w, device)
    if key not in _coord_grid_cache:
        yy = torch.linspace(-1.0, 1.0, h, device=device)
        xx = torch.linspace(-1.0, 1.0, w, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        _coord_grid_cache[key] = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    return _coord_grid_cache[key]


def warp_with_flow(
    image: torch.Tensor,
    flow: torch.Tensor,
) -> torch.Tensor:
    """Warp image using optical flow via differentiable grid_sample.

    Args:
        image: (B, C, H, W) source image
        flow: (B, 2, H, W) optical flow in normalized [-1, 1] coordinates

    Returns:
        (B, C, H, W) warped image
    """
    B, _, H, W = image.shape
    grid = make_coord_grid(H, W, image.device).expand(B, -1, -1, -1)
    # flow is (B, 2, H, W) → (B, H, W, 2) for grid_sample
    flow_hw = flow.permute(0, 2, 3, 1)
    sample_grid = grid + flow_hw
    return F.grid_sample(image, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)


class MotionPredictor(nn.Module):
    """Predict optical flow from consecutive segmentation mask pairs.

    PoseNet evaluates frame PAIRS to estimate ego-motion. A static renderer
    alone cannot produce the inter-frame differences PoseNet needs. This
    module predicts flow from mask transitions so the renderer's output
    for frame t can be warped to approximate frame t+1.

    Architecture:
        concat(embed(mask_t), embed(mask_t+1)) → conv stack → flow (2ch)

    The flow is in normalized coordinates for grid_sample (range ~[-0.1, 0.1]).

    Args:
        num_classes: segmentation classes
        embed_dim: per-class embedding dimension
        hidden: internal channel width
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 6,
        hidden: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, embed_dim)
        in_ch = embed_dim * 2  # two masks concatenated

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            ResBlock(hidden),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 3, padding=1, bias=True),
        )

        # Zero-init flow output — start with zero motion (identity warp)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        mask_t: torch.Tensor,
        mask_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Predict optical flow from mask pair.

        Args:
            mask_t: (B, H, W) long — mask at time t
            mask_t1: (B, H, W) long — mask at time t+1

        Returns:
            (B, 2, H, W) flow in normalized coordinates
        """
        e_t = self.embedding(mask_t).permute(0, 3, 1, 2)
        e_t1 = self.embedding(mask_t1).permute(0, 3, 1, 2)
        x = torch.cat([e_t, e_t1], dim=1)
        # Scale output to small flow range — typical ego-motion is <5% of frame
        flow = self.net(x) * 0.1
        return flow

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── PairGenerator ───────────────────────────────────────────────────────


class PairGenerator(nn.Module):
    """Generate PoseNet-compatible frame pairs from segmentation masks.

    Combines MaskRenderer and MotionPredictor to produce the (1, 2, H, W, 3)
    HWC pair format that the scorer pipeline expects.

    For a pair of masks (t, t+1):
        1. Render frame_t from mask_t using MaskRenderer
        2. Predict flow from (mask_t, mask_t+1) using MotionPredictor
        3. Warp frame_t with flow to produce frame_t+1_warped
        4. Also render frame_t+1 directly (for blending)
        5. Blend: frame_t+1 = alpha * warped + (1-alpha) * rendered_t+1
        6. Pack into (1, 2, H, W, 3) HWC format

    The blend ratio is learned — it starts at 0.5 and adjusts based on
    whether warping or direct rendering produces better PoseNet scores.

    Args:
        renderer: MaskRenderer instance
        motion: MotionPredictor instance
    """

    def __init__(self, renderer: MaskRenderer, motion: MotionPredictor):
        super().__init__()
        self.renderer = renderer
        self.motion = motion
        # Learned blend weight: sigmoid(raw) gives alpha in [0, 1]
        # Initialize to 0 → sigmoid(0) = 0.5 → equal blend
        self.blend_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        mask_t: torch.Tensor,
        mask_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a scored frame pair from two consecutive masks.

        Args:
            mask_t: (B, H, W) long — mask at time t
            mask_t1: (B, H, W) long — mask at time t+1

        Returns:
            (B, 2, H, W, 3) float tensor in [0, 255] — HWC pair format
        """
        # Render both frames directly
        frame_t = self.renderer(mask_t)        # (B, 3, H, W)
        frame_t1 = self.renderer(mask_t1)      # (B, 3, H, W)

        # Predict flow and warp frame_t → frame_t1_warped
        flow = self.motion(mask_t, mask_t1)    # (B, 2, H, W)
        frame_t1_warped = warp_with_flow(frame_t, flow)

        # Blend warped and directly-rendered frame_t+1
        alpha = torch.sigmoid(self.blend_logit)
        frame_t1_blended = (alpha * frame_t1_warped + (1.0 - alpha) * frame_t1).clamp(0.0, 255.0)

        # Pack to HWC pair format: (B, 2, H, W, 3)
        # CHW → HWC: (B, 3, H, W) → (B, H, W, 3)
        f_t_hwc = frame_t.permute(0, 2, 3, 1)
        f_t1_hwc = frame_t1_blended.permute(0, 2, 3, 1)
        return torch.stack([f_t_hwc, f_t1_hwc], dim=1)

    def param_count(self) -> int:
        """Total trainable parameter count (renderer + motion + blend)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Factory ─────────────────────────────────────────────────────────────


def build_renderer(
    num_classes: int = 5,
    embed_dim: int = 6,
    base_ch: int = 36,
    mid_ch: int = 60,
    motion_hidden: int = 32,
) -> PairGenerator:
    """Build the full mask-to-pair rendering pipeline.

    Default settings match the competitor's proven ~300K param budget.

    Args:
        num_classes: segmentation classes (5 for comma)
        embed_dim: per-class embedding dimension
        base_ch: U-Net base channels (stem + decoder)
        mid_ch: U-Net bottleneck channels
        motion_hidden: MotionPredictor hidden channels

    Returns:
        PairGenerator wrapping MaskRenderer + MotionPredictor
    """
    renderer = MaskRenderer(
        num_classes=num_classes,
        embed_dim=embed_dim,
        base_ch=base_ch,
        mid_ch=mid_ch,
    )
    motion = MotionPredictor(
        num_classes=num_classes,
        embed_dim=embed_dim,
        hidden=motion_hidden,
    )
    pair_gen = PairGenerator(renderer, motion)

    total = pair_gen.param_count()
    r_count = renderer.param_count()
    m_count = motion.param_count()
    print(f"[renderer] Built PairGenerator: {total:,} params "
          f"(renderer={r_count:,}, motion={m_count:,}, blend=1)")

    return pair_gen
