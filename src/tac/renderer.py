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


class CLADENorm(nn.Module):
    """GroupNorm with per-class affine modulation (CLADE, arxiv 2012.04644).

    After GroupNorm normalizes features, per-class gamma/beta are looked up
    from the segmentation mask and applied as spatially-varying affine
    modulation.  With 5 classes this adds only 2 * channels parameters.

    The mask is nearest-neighbor downsampled to match feature resolution.
    """

    def __init__(self, channels: int, num_classes: int = 5):
        super().__init__()
        self.gn = nn.GroupNorm(1, channels)
        self.class_gamma = nn.Embedding(num_classes, channels)
        self.class_beta = nn.Embedding(num_classes, channels)
        # Init: gamma=1, beta=0 → identity modulation at start
        nn.init.ones_(self.class_gamma.weight)
        nn.init.zeros_(self.class_beta.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply class-conditioned normalization.

        Args:
            x: (B, C, H, W) feature tensor
            mask: (B, H_orig, W_orig) long tensor with class indices
        """
        h = self.gn(x)
        # Downsample mask to feature resolution via nearest neighbor
        _, _, fH, fW = x.shape
        if mask.shape[1] != fH or mask.shape[2] != fW:
            mask_ds = F.interpolate(
                mask.unsqueeze(1).float(), size=(fH, fW), mode="nearest"
            ).squeeze(1).long()
        else:
            mask_ds = mask
        # Look up per-class affine: (B, H, W) → (B, H, W, C) → (B, C, H, W)
        gamma = self.class_gamma(mask_ds).permute(0, 3, 1, 2)
        beta = self.class_beta(mask_ds).permute(0, 3, 1, 2)
        return gamma * h + beta


class ResBlock(nn.Module):
    """Pre-activation residual block with optional CLADE per-class conditioning.

    When num_classes > 0, uses CLADENorm (class-adaptive normalization).
    When num_classes == 0, uses plain GroupNorm (backward-compatible).
    """

    def __init__(self, channels: int, kernel: int = 3, num_classes: int = 5):
        super().__init__()
        pad = kernel // 2
        self.use_clade = num_classes > 0
        if self.use_clade:
            self.norm1 = CLADENorm(channels, num_classes)
            self.norm2 = CLADENorm(channels, num_classes)
        else:
            self.norm1 = nn.GroupNorm(1, channels)
            self.norm2 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel, padding=pad, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel, padding=pad, bias=False)
        self.act = nn.ReLU(inplace=True)
        # Zero-init second conv so residual starts as identity
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_clade and mask is not None:
            h = self.act(self.norm1(x, mask))
            h = self.conv1(h)
            h = self.act(self.norm2(h, mask))
        else:
            h = self.act(self.norm1(x) if not self.use_clade else self.norm1.gn(x))
            h = self.conv1(h)
            h = self.act(self.norm2(h) if not self.use_clade else self.norm2.gn(h))
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
        → head (1x1 conv → 3ch RGB, soft sigmoid output)

    CLADE per-class normalization: each ResBlock's GroupNorm is modulated by
    per-class gamma/beta looked up from the segmentation mask, giving the
    network class-specific feature statistics (arxiv 2012.04644).

    Soft sigmoid output: 255 * sigmoid(logits / 50) provides always-flowing
    gradients, replacing hard clamp(0, 255).

    Args:
        num_classes: number of segmentation classes (5 for comma SegNet)
        embed_dim: embedding dimension per class (input channels to U-Net)
        base_ch: base channel width (stem and decoder)
        mid_ch: bottleneck channel width (wider for capacity)
        embedding: optional pre-created Embedding (for weight sharing)

    Competitor reference: base_ch=36, mid_ch=60, embed_dim=6 (~308K params in FP4).
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 6,
        base_ch: int = 36,
        mid_ch: int = 60,
        embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Class embedding: each of 5 classes → learned embed_dim-dimensional vector
        # Accepts external embedding for weight sharing with MotionPredictor
        self.embedding = embedding if embedding is not None else nn.Embedding(num_classes, embed_dim)

        # Stem: embed_dim → base_ch at full resolution
        self.stem_conv = nn.Conv2d(embed_dim, base_ch, 3, padding=1, bias=True)
        self.stem_res = ResBlock(base_ch, num_classes=num_classes)

        # Downsample: base_ch → mid_ch at half resolution
        self.down_conv = nn.Conv2d(base_ch, mid_ch, 3, stride=2, padding=1, bias=True)
        self.down_res = ResBlock(mid_ch, num_classes=num_classes)

        # Bottleneck at half resolution
        self.bottleneck = ResBlock(mid_ch, num_classes=num_classes)

        # Upsample: mid_ch → base_ch at full resolution
        self.up_conv = nn.ConvTranspose2d(mid_ch, base_ch, 4, stride=2, padding=1, bias=True)
        self.up_res = ResBlock(base_ch, num_classes=num_classes)

        # Fusion after skip: base_ch (skip) + base_ch (upsampled) → base_ch
        self.fuse_conv = nn.Conv2d(base_ch * 2, base_ch, 1, bias=True)

        # Head: base_ch → 3 RGB channels (soft sigmoid output, no bias init needed)
        self.head = nn.Conv2d(base_ch, 3, 1, bias=True)
        # Initialize head so sigmoid(head/50) ≈ 0.5 → ~128 at init
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

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

        # Stem (full res) — pass mask for CLADE conditioning
        stem = self.stem_conv(x)
        stem = self.stem_res(stem, masks)

        # Down (half res)
        down = self.down_conv(stem)
        down = self.down_res(down, masks)

        # Bottleneck
        mid = self.bottleneck(down, masks)

        # Up (back to full res)
        up = self.up_conv(mid)
        # Handle potential size mismatch from odd dimensions
        if up.shape[2:] != stem.shape[2:]:
            up = F.interpolate(up, size=stem.shape[2:], mode="bilinear", align_corners=False)
        up = self.up_res(up, masks)

        # Skip connection: concatenate stem features with upsampled
        fused = torch.cat([stem, up], dim=1)
        fused = self.fuse_conv(fused)

        # Head: soft sigmoid output — gradients always flow, no dead zones
        # sigmoid(0/50) = 0.5 → 127.5 at init (mid-gray)
        rgb = 255.0 * torch.sigmoid(self.head(fused) / 50.0)
        return rgb

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
        embedding: optional pre-created Embedding (for weight sharing)
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 6,
        hidden: int = 32,
        embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        # Accepts external embedding for weight sharing with MaskRenderer
        self.embedding = embedding if embedding is not None else nn.Embedding(num_classes, embed_dim)
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

    Features:
        - Shared embedding between renderer and motion predictor
        - CLADE per-class normalization in all ResBlocks
        - Soft sigmoid output for always-flowing gradients

    Args:
        num_classes: segmentation classes (5 for comma)
        embed_dim: per-class embedding dimension
        base_ch: U-Net base channels (stem + decoder)
        mid_ch: U-Net bottleneck channels
        motion_hidden: MotionPredictor hidden channels

    Returns:
        PairGenerator wrapping MaskRenderer + MotionPredictor
    """
    # Shared embedding: renderer and motion predictor learn a single
    # class representation, reducing parameters and improving coherence
    shared_embed = nn.Embedding(num_classes, embed_dim)

    renderer = MaskRenderer(
        num_classes=num_classes,
        embed_dim=embed_dim,
        base_ch=base_ch,
        mid_ch=mid_ch,
        embedding=shared_embed,
    )
    motion = MotionPredictor(
        num_classes=num_classes,
        embed_dim=embed_dim,
        hidden=motion_hidden,
        embedding=shared_embed,
    )
    pair_gen = PairGenerator(renderer, motion)

    # Verify embedding is truly shared
    assert renderer.embedding is motion.embedding, "Embedding sharing failed"

    total = pair_gen.param_count()
    r_count = renderer.param_count()
    m_count = motion.param_count()
    print(f"[renderer] Built PairGenerator: {total:,} params "
          f"(renderer={r_count:,}, motion={m_count:,}, blend=1, "
          f"shared_embed={shared_embed.weight.numel()}, CLADE=on, sigmoid_out=on)")

    return pair_gen
