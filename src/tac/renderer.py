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
            mask_ds = F.interpolate(mask.unsqueeze(1).float(), size=(fH, fW), mode="nearest").squeeze(1).long()
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

    Args (extended):
        depth: number of downscale levels (1 = original single-scale, 2 = two-scale ~450K params)
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 6,
        base_ch: int = 36,
        mid_ch: int = 60,
        embedding: nn.Embedding | None = None,
        depth: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth

        # Class embedding: each of 5 classes → learned embed_dim-dimensional vector
        # Accepts external embedding for weight sharing with MotionPredictor
        self.embedding = embedding if embedding is not None else nn.Embedding(num_classes, embed_dim)

        # Coordinate grid conditioning (council decision: +2 channels for spatial awareness)
        self.use_coord_grid = True
        coord_channels = 2 if self.use_coord_grid else 0

        # Stem: (embed_dim + coord_channels) → base_ch at full resolution
        self.stem_conv = nn.Conv2d(embed_dim + coord_channels, base_ch, 3, padding=1, bias=True)
        self.stem_res = ResBlock(base_ch, num_classes=num_classes)

        # Downsample 1: base_ch → mid_ch at half resolution
        self.down_conv = nn.Conv2d(base_ch, mid_ch, 3, stride=2, padding=1, bias=True)
        self.down_res = ResBlock(mid_ch, num_classes=num_classes)

        if depth >= 2:
            # Downsample 2: mid_ch → mid_ch at quarter resolution
            self.down2_conv = nn.Conv2d(mid_ch, mid_ch, 3, stride=2, padding=1, bias=True)
            self.down2_res = ResBlock(mid_ch, num_classes=num_classes)

        # Bottleneck at lowest resolution
        self.bottleneck = ResBlock(mid_ch, num_classes=num_classes)

        if depth >= 2:
            # Upsample 2: mid_ch → mid_ch back to half resolution
            self.up2_conv = nn.ConvTranspose2d(mid_ch, mid_ch, 4, stride=2, padding=1, bias=True)
            self.up2_res = ResBlock(mid_ch, num_classes=num_classes)
            # Fusion after skip at half resolution: mid_ch (skip) + mid_ch (upsampled) → mid_ch
            self.fuse2_conv = nn.Conv2d(mid_ch * 2, mid_ch, 1, bias=True)

        # Upsample 1: mid_ch → base_ch at full resolution
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

        # Coordinate grid: normalized [-1, 1] spatial coords (Quantizr insight)
        if self.use_coord_grid:
            B, _, H, W = x.shape
            gy = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
            gx = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
            grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
            coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
            x = torch.cat([x, coords], dim=1)  # (B, embed_dim+2, H, W)

        # Stem (full res) — pass mask for CLADE conditioning
        stem = self.stem_conv(x)
        stem = self.stem_res(stem, masks)

        # Down 1 (half res)
        down1 = self.down_conv(stem)
        down1 = self.down_res(down1, masks)

        if self.depth >= 2:
            # Down 2 (quarter res)
            down2 = self.down2_conv(down1)
            down2 = self.down2_res(down2, masks)

            # Bottleneck at quarter res
            mid = self.bottleneck(down2, masks)

            # Up 2 (quarter → half res)
            up2 = self.up2_conv(mid)
            if up2.shape[2:] != down1.shape[2:]:
                up2 = F.interpolate(up2, size=down1.shape[2:], mode="bilinear", align_corners=False)
            up2 = self.up2_res(up2, masks)

            # Skip connection at half res
            fused2 = torch.cat([down1, up2], dim=1)
            half_res = self.fuse2_conv(fused2)
        else:
            # Bottleneck at half res (original behavior)
            half_res = self.bottleneck(down1, masks)

        # Up 1 (half → full res)
        up = self.up_conv(half_res)
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
_COORD_GRID_MAXSIZE = 4  # Bound cache to avoid memory leak with many resolutions


def make_coord_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Create normalized coordinate grid for grid_sample (cached).

    Returns: (1, H, W, 2) tensor with values in [-1, 1].
    Cache is bounded to _COORD_GRID_MAXSIZE entries (FIFO eviction).
    """
    key = (h, w, device)
    if key not in _coord_grid_cache:
        # Evict oldest entry if at capacity
        if len(_coord_grid_cache) >= _COORD_GRID_MAXSIZE:
            oldest_key = next(iter(_coord_grid_cache))
            del _coord_grid_cache[oldest_key]
        yy = torch.linspace(-1.0, 1.0, h, device=device)
        xx = torch.linspace(-1.0, 1.0, w, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        _coord_grid_cache[key] = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    return _coord_grid_cache[key]


def _manual_grid_sample(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """MPS-compatible drop-in replacement for F.grid_sample (bilinear, align_corners=True).

    F.grid_sample backward is not implemented on MPS (aten::grid_sampler_2d_backward).
    This manual implementation uses only ops with full MPS forward+backward support:
    torch.floor, torch.gather, torch.clamp, basic arithmetic.

    Benchmarked: 11.3x faster than CPU fallback on M5 Max (185ms vs 2091ms per iter).
    Max output diff vs F.grid_sample: 3.6e-7. Max gradient diff: 3.1e-5.

    Args:
        image: (B, C, H, W) source image
        grid: (B, H_out, W_out, 2) sampling grid in [-1, 1], align_corners=True

    Returns:
        (B, C, H_out, W_out) bilinearly sampled image
    """
    B, C, H, W = image.shape
    _, H_out, W_out, _ = grid.shape

    # Unnormalize: [-1,1] -> [0, H-1] / [0, W-1] (align_corners=True)
    ix = (grid[..., 0] + 1.0) * (W - 1) / 2.0
    iy = (grid[..., 1] + 1.0) * (H - 1) / 2.0

    # Corner indices (detached — floor has no meaningful gradient)
    ix0 = torch.floor(ix).detach().long()
    iy0 = torch.floor(iy).detach().long()
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    # Bilinear weights (these carry gradients back to the grid)
    wx = ix - ix0.float()
    wy = iy - iy0.float()

    # Border clamp
    ix0 = ix0.clamp(0, W - 1)
    ix1 = ix1.clamp(0, W - 1)
    iy0 = iy0.clamp(0, H - 1)
    iy1 = iy1.clamp(0, H - 1)

    # Flatten spatial dims and gather the 4 corner values
    image_flat = image.reshape(B, C, H * W)
    idx_00 = (iy0 * W + ix0).reshape(B, 1, -1).expand(-1, C, -1)
    idx_01 = (iy0 * W + ix1).reshape(B, 1, -1).expand(-1, C, -1)
    idx_10 = (iy1 * W + ix0).reshape(B, 1, -1).expand(-1, C, -1)
    idx_11 = (iy1 * W + ix1).reshape(B, 1, -1).expand(-1, C, -1)

    v00 = torch.gather(image_flat, 2, idx_00).reshape(B, C, H_out, W_out)
    v01 = torch.gather(image_flat, 2, idx_01).reshape(B, C, H_out, W_out)
    v10 = torch.gather(image_flat, 2, idx_10).reshape(B, C, H_out, W_out)
    v11 = torch.gather(image_flat, 2, idx_11).reshape(B, C, H_out, W_out)

    # Bilinear interpolation
    wx = wx.unsqueeze(1)  # (B, 1, H_out, W_out)
    wy = wy.unsqueeze(1)
    return v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy) + v10 * (1 - wx) * wy + v11 * wx * wy


def warp_with_flow(
    image: torch.Tensor,
    flow: torch.Tensor,
) -> torch.Tensor:
    """Warp image using optical flow via differentiable grid sampling.

    Uses native F.grid_sample on CUDA, manual implementation on MPS
    (where grid_sampler_2d_backward is not implemented).

    Args:
        image: (B, C, H, W) source image
        flow: (B, 2, H, W) optical flow in normalized [-1, 1] coordinates

    Returns:
        (B, C, H, W) warped image
    """
    B, _, H, W = image.shape
    grid = make_coord_grid(H, W, image.device).expand(B, -1, -1, -1)
    flow_hw = flow.permute(0, 2, 3, 1)
    sample_grid = grid + flow_hw

    if image.device.type == "mps":
        return _manual_grid_sample(image, sample_grid)
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
        output_channels: int = 2,
        use_coord_grid: bool = True,
        use_diff_features: bool = True,
        max_flow_px: float = 20.0,
        max_residual: float = 20.0,
        flow_only: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.output_channels = output_channels
        self.use_coord_grid = use_coord_grid
        self.use_diff_features = use_diff_features
        self.max_flow_px = max_flow_px
        self.max_residual = max_residual
        self.flow_only = flow_only
        # Accepts external embedding for weight sharing with MaskRenderer
        self.embedding = embedding if embedding is not None else nn.Embedding(num_classes, embed_dim)

        # Input: e_t + e_t1 + optional |e_t1 - e_t| + optional coords
        in_ch = embed_dim * 2
        if use_diff_features:
            in_ch += embed_dim  # |e_t1 - e_t| (Quantizr insight)
        if use_coord_grid:
            in_ch += 2  # normalized xy coords

        # U-Net-like structure for global receptive field (Quantizr TinyMotionFromMasks)
        # Stem: full resolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=True),
            nn.SiLU(inplace=True),
        )
        # Down: stride-2 → half resolution
        self.down = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, bias=True),
            nn.SiLU(inplace=True),
        )
        # Bottleneck at half resolution
        self.bottleneck = ResBlock(hidden, num_classes=0)
        # Up: upsample back to full resolution
        self.up_conv = nn.Conv2d(hidden, hidden, 3, padding=1, bias=True)
        self.up_act = nn.SiLU(inplace=True)
        # Skip fusion: concat skip + upsampled → hidden
        self.fuse = nn.Conv2d(hidden * 2, hidden, 1, bias=True)
        # Head: predict output channels
        self.head = nn.Conv2d(hidden, output_channels, 3, padding=1, bias=True)

        # Zero-init output — start with zero motion (identity warp)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Gate channel (index 2) init bias to -2.0 → sigmoid(-2)=0.12
        # This makes the model trust the warp from the start, using residual
        # only for correction. Without this, gate=0.5 → half the signal lost.
        if output_channels >= 3:
            with torch.no_grad():
                self.head.bias[2] = -2.0

    def forward(
        self,
        mask_t: torch.Tensor,
        mask_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Predict motion outputs from mask pair.

        Args:
            mask_t: (B, H, W) long — mask at time t
            mask_t1: (B, H, W) long — mask at time t+1

        Returns:
            (B, output_channels, H, W) — channels depend on mode:
              output_channels=2: flow only (backward compat)
              output_channels=6: flow(2) + gate(1) + residual(3)
        """
        e_t = self.embedding(mask_t).permute(0, 3, 1, 2)
        e_t1 = self.embedding(mask_t1).permute(0, 3, 1, 2)

        parts = [e_t, e_t1]
        if self.use_diff_features:
            parts.append((e_t1 - e_t).abs())  # where change happened
        if self.use_coord_grid:
            B, _, H, W = e_t.shape
            gy = torch.linspace(-1, 1, H, device=e_t.device, dtype=e_t.dtype)
            gx = torch.linspace(-1, 1, W, device=e_t.device, dtype=e_t.dtype)
            grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
            coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
            parts.append(coords)

        x = torch.cat(parts, dim=1)

        # U-Net forward: stem → down → bottleneck → up → skip fusion → head
        stem_feat = self.stem(x)                          # (B, hidden, H, W)
        down_feat = self.down(stem_feat)                  # (B, hidden, H/2, W/2)
        bot_feat = self.bottleneck(down_feat)             # (B, hidden, H/2, W/2)
        up_feat = F.interpolate(bot_feat, size=stem_feat.shape[2:], mode="bilinear", align_corners=False)
        up_feat = self.up_act(self.up_conv(up_feat))      # (B, hidden, H, W)
        fused = self.fuse(torch.cat([stem_feat, up_feat], dim=1))  # (B, hidden, H, W)
        raw = self.head(fused)                            # (B, output_channels, H, W)

        if self.output_channels == 2:
            # Legacy: flow only, scaled to small range
            return raw * 0.1
        else:
            # Asymmetric mode: flow(2) + gate(1) + residual(3)
            flow = raw[:, :2].tanh() * (self.max_flow_px / max(mask_t.shape[-2], mask_t.shape[-1]) * 2)
            if self.flow_only:
                # Optimization #15: flow only, no gate/residual (deferred, gated)
                gate = torch.zeros_like(raw[:, 2:3])
                residual = torch.zeros_like(raw[:, 3:6])
            else:
                gate = raw[:, 2:3].sigmoid()
                residual = raw[:, 3:6].tanh() * self.max_residual
            return torch.cat([flow, gate, residual], dim=1)

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

    def __init__(
        self,
        renderer: MaskRenderer,
        motion: MotionPredictor,
        blend_mode: str = "scalar",
        noise_mode: str = "deterministic",
    ):
        super().__init__()
        self.renderer = renderer
        self.motion = motion
        self.blend_mode = blend_mode
        self.noise_mode = noise_mode

        # Blend modes:
        #   "scalar"  — single learned alpha (original behavior)
        #   "spatial" — per-pixel learned blend map via 1x1 conv
        #   "none"    — no blending, use direct rendering only
        if blend_mode == "scalar":
            # Learned blend weight: sigmoid(raw) gives alpha in [0, 1]
            # Initialize to 0 → sigmoid(0) = 0.5 → equal blend
            self.blend_logit = nn.Parameter(torch.tensor(0.0))
            self.blend_conv = None
        elif blend_mode == "spatial":
            self.blend_logit = None
            # 1x1 conv: (B, 6, H, W) [warped + rendered] → (B, 1, H, W) alpha map
            self.blend_conv = nn.Conv2d(6, 1, 1, bias=True)
            nn.init.zeros_(self.blend_conv.weight)
            nn.init.zeros_(self.blend_conv.bias)  # sigmoid(0) = 0.5
        elif blend_mode == "none":
            self.blend_logit = None
            self.blend_conv = None
        else:
            raise ValueError(f"Unknown blend_mode: {blend_mode!r}. Use 'scalar', 'spatial', or 'none'.")

        # Noise modes:
        #   "deterministic" — no noise injection (default)
        #   "shared"        — same noise for both frames (temporal consistency)
        #   "independent"   — independent noise per frame (texture diversity)
        if noise_mode not in ("deterministic", "shared", "independent"):
            raise ValueError(f"Unknown noise_mode: {noise_mode!r}. Use 'deterministic', 'shared', or 'independent'.")
        if noise_mode != "deterministic":
            self.noise_scale = nn.Parameter(torch.tensor(0.0))  # exp(0)=1, scaled down by tanh
        else:
            self.noise_scale = None

    def _inject_noise(self, frame: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Optionally inject learned-scale noise into a rendered frame."""
        if self.noise_scale is None:
            return frame
        if noise is None:
            noise = torch.randn_like(frame)
        # Scale: tanh keeps magnitude bounded, exp(noise_scale) controls amplitude
        scale = torch.tanh(self.noise_scale) * 5.0  # max +-5 pixel noise
        return (frame + scale * noise).clamp(0.0, 255.0)

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
        frame_t = self.renderer(mask_t)  # (B, 3, H, W)
        frame_t1 = self.renderer(mask_t1)  # (B, 3, H, W)

        # Noise injection
        if self.noise_mode == "shared":
            noise = torch.randn_like(frame_t)
            frame_t = self._inject_noise(frame_t, noise)
            frame_t1 = self._inject_noise(frame_t1, noise)
        elif self.noise_mode == "independent":
            frame_t = self._inject_noise(frame_t)
            frame_t1 = self._inject_noise(frame_t1)

        # Blend mode: "none" skips motion entirely
        if self.blend_mode == "none":
            frame_t1_blended = frame_t1
        else:
            # Predict flow and warp frame_t → frame_t1_warped
            flow = self.motion(mask_t, mask_t1)  # (B, 2, H, W)
            frame_t1_warped = warp_with_flow(frame_t, flow)

            if self.blend_mode == "scalar":
                alpha = torch.sigmoid(self.blend_logit)
                frame_t1_blended = (alpha * frame_t1_warped + (1.0 - alpha) * frame_t1).clamp(0.0, 255.0)
            elif self.blend_mode == "spatial":
                alpha_map = torch.sigmoid(self.blend_conv(torch.cat([frame_t1_warped, frame_t1], dim=1)))
                frame_t1_blended = (alpha_map * frame_t1_warped + (1.0 - alpha_map) * frame_t1).clamp(0.0, 255.0)

        # Pack to HWC pair format: (B, 2, H, W, 3)
        # CHW → HWC: (B, 3, H, W) → (B, H, W, 3)
        f_t_hwc = frame_t.permute(0, 2, 3, 1)
        f_t1_hwc = frame_t1_blended.permute(0, 2, 3, 1)
        return torch.stack([f_t_hwc, f_t1_hwc], dim=1)

    def param_count(self) -> int:
        """Total trainable parameter count (renderer + motion + blend)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Technique 6: DoubleTake Hint Mechanism (Niantic) ─────────────────


class HintMLP(nn.Module):
    """Small MLP that processes a previous frame's rendered output as a "hint".

    Niantic's DoubleTake paper showed that feeding previous frame's output
    back into the current frame's renderer improves temporal consistency
    at minimal parameter cost.

    The hint is processed via a lightweight MLP and added to features.

    Args:
        in_ch: input channels (3 for RGB hint)
        hidden: MLP hidden width
        out_ch: output channels (matches feature map channels)
    """

    def __init__(self, in_ch: int = 3, hidden: int = 16, out_ch: int = 36):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=True),
        )
        # Zero-init: hint starts as no-op
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hint: torch.Tensor) -> torch.Tensor:
        """Process hint RGB and return feature-space modulation.

        Args:
            hint: (B, 3, H, W) previous frame's rendered output

        Returns:
            (B, out_ch, H, W) feature modulation to add to current features
        """
        return self.net(hint / 255.0)


class HintedMaskRenderer(nn.Module):
    """MaskRenderer with DoubleTake hint mechanism.

    Wraps MaskRenderer and adds a HintMLP that processes the previous
    frame's rendered output and adds it to the stem features. This
    provides temporal consistency without explicit motion modeling.

    Usage: call forward with both the mask and the previous frame's output.
    For the first frame, pass None as hint (uses zero hint).

    Args:
        renderer: MaskRenderer instance
        hint_hidden: HintMLP hidden width (16-32 recommended)
    """

    def __init__(self, renderer: MaskRenderer, hint_hidden: int = 16):
        super().__init__()
        self.renderer = renderer
        base_ch = renderer.stem_conv.out_channels
        self.hint_mlp = HintMLP(in_ch=3, hidden=hint_hidden, out_ch=base_ch)

    def forward(self, masks: torch.Tensor, hint: torch.Tensor | None = None) -> torch.Tensor:
        """Render RGB with temporal hint from previous frame.

        Args:
            masks: (B, H, W) long tensor with class labels
            hint: (B, 3, H, W) previous frame's rendered RGB, or None

        Returns:
            (B, 3, H, W) float tensor in [0, 255]
        """
        # Standard embedding + coord grid (must match MaskRenderer.forward)
        x = self.renderer.embedding(masks).permute(0, 3, 1, 2).contiguous()
        if self.renderer.use_coord_grid:
            B, _, H, W = x.shape
            gy = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
            gx = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
            grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
            coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
            x = torch.cat([x, coords], dim=1)
        stem = self.renderer.stem_conv(x)
        stem = self.renderer.stem_res(stem, masks)

        # Add hint modulation if available
        if hint is not None:
            hint_feat = self.hint_mlp(hint)
            if hint_feat.shape[2:] != stem.shape[2:]:
                hint_feat = F.interpolate(
                    hint_feat, size=stem.shape[2:], mode="bilinear", align_corners=False
                )
            stem = stem + hint_feat

        # Continue standard MaskRenderer forward from down1 onward
        down1 = self.renderer.down_conv(stem)
        down1 = self.renderer.down_res(down1, masks)

        if self.renderer.depth >= 2:
            down2 = self.renderer.down2_conv(down1)
            down2 = self.renderer.down2_res(down2, masks)
            mid = self.renderer.bottleneck(down2, masks)
            up2 = self.renderer.up2_conv(mid)
            if up2.shape[2:] != down1.shape[2:]:
                up2 = F.interpolate(up2, size=down1.shape[2:], mode="bilinear", align_corners=False)
            up2 = self.renderer.up2_res(up2, masks)
            fused2 = torch.cat([down1, up2], dim=1)
            half_res = self.renderer.fuse2_conv(fused2)
        else:
            half_res = self.renderer.bottleneck(down1, masks)

        up = self.renderer.up_conv(half_res)
        if up.shape[2:] != stem.shape[2:]:
            up = F.interpolate(up, size=stem.shape[2:], mode="bilinear", align_corners=False)
        up = self.renderer.up_res(up, masks)

        fused = torch.cat([stem, up], dim=1)
        fused = self.renderer.fuse_conv(fused)
        rgb = 255.0 * torch.sigmoid(self.renderer.head(fused) / 50.0)
        return rgb

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AsymmetricPairGenerator(nn.Module):
    """Quantizr-inspired warp-based pair generation (council decision 2026-04-12).

    The key insight: frame2 is rendered directly from mask2 (anchor frame).
    Frame1 is derived by warping frame2 with learned optical flow + gated
    residual correction. This makes temporal coherence ARCHITECTURAL, not
    learned through loss signals — PoseNet sees a geometric warp between
    frames, which is exactly what real ego-motion produces.

    Architecture:
        frame2 = renderer(mask2)                        # Direct render
        flow, gate, residual = motion(mask1, mask2)     # Motion from BOTH masks
        frame1 = warp(frame2, flow) + gate * residual   # Geometric + correction

    Args:
        num_classes: segmentation classes (5 for comma)
        embed_dim: class embedding dimension (6)
        base_ch: renderer base channel width (36)
        mid_ch: renderer bottleneck width (60)
        motion_hidden: motion predictor hidden channels (32)
        depth: U-Net depth for renderer (1 or 2)
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 6,
        base_ch: int = 36,
        mid_ch: int = 60,
        motion_hidden: int = 32,
        depth: int = 1,
        max_flow_px: float = 20.0,
        max_residual: float = 20.0,
        flow_only: bool = False,
    ):
        super().__init__()
        # Shared embedding between renderer and motion predictor
        shared_emb = nn.Embedding(num_classes, embed_dim)
        self.renderer = MaskRenderer(
            num_classes=num_classes,
            embed_dim=embed_dim,
            base_ch=base_ch,
            mid_ch=mid_ch,
            embedding=shared_emb,
            depth=depth,
        )
        self.motion = MotionPredictor(
            num_classes=num_classes,
            embed_dim=embed_dim,
            hidden=motion_hidden,
            embedding=shared_emb,
            output_channels=6,  # flow(2) + gate(1) + residual(3)
            use_coord_grid=True,
            use_diff_features=True,
            max_flow_px=max_flow_px,
            max_residual=max_residual,
            flow_only=flow_only,
        )

    def forward(
        self,
        mask_t: torch.Tensor,
        mask_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a frame pair using warp-based asymmetric generation.

        Args:
            mask_t: (B, H, W) long — mask at time t
            mask_t1: (B, H, W) long — mask at time t+1

        Returns:
            (B, 2, H, W, 3) float HWC pair in [0, 255]
        """
        # Render anchor frame (frame_t1) directly from mask
        frame_t1 = self.renderer(mask_t1)  # (B, 3, H, W) float [0, 255]

        # Predict motion from both masks
        motion_out = self.motion(mask_t, mask_t1)  # (B, 6, H, W)
        flow = motion_out[:, :2]       # (B, 2, H, W) normalized flow
        gate = motion_out[:, 2:3]      # (B, 1, H, W) [0, 1]
        residual = motion_out[:, 3:6]  # (B, 3, H, W) [-20, 20]

        # Warp anchor backward to produce frame_t
        warped_t1 = warp_with_flow(frame_t1, flow)
        frame_t = (warped_t1 + gate * residual).clamp(0.0, 255.0)

        # Diagnostic: gate statistics (council Option C — Quantizr adversarial recommendation)
        # If mean gate > 0.7, flow is not contributing and architecture degrades to residual-only
        self._last_gate_mean = gate.mean().item()

        # Pack to HWC: (B, 2, H, W, 3)
        pair = torch.stack([frame_t, frame_t1], dim=1)  # (B, 2, 3, H, W)
        return pair.permute(0, 1, 3, 4, 2).contiguous()  # (B, 2, H, W, 3)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HintedPairGenerator(nn.Module):
    """PairGenerator with DoubleTake hint mechanism.

    For a pair (t, t+1):
    1. Render frame_t from mask_t (no hint for first frame)
    2. Render frame_t+1 from mask_t+1 with frame_t as hint
    3. Predict flow and blend as usual

    Args:
        hinted_renderer: HintedMaskRenderer instance
        motion: MotionPredictor instance
    """

    def __init__(self, hinted_renderer: HintedMaskRenderer, motion: MotionPredictor):
        super().__init__()
        self.hinted_renderer = hinted_renderer
        self.motion = motion
        self.blend_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        mask_t: torch.Tensor,
        mask_t1: torch.Tensor,
        residual_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate pair with temporal hint.

        Args:
            mask_t: (B, H, W) long — mask at time t
            mask_t1: (B, H, W) long — mask at time t+1
            residual_scale: Scale factor for the direct-render (non-warp) pathway.
                0.0 = pure warp (flow warmup), 1.0 = normal blend.
                Used by training to force flow learning before residual develops.

        Returns:
            (B, 2, H, W, 3) float tensor in [0, 255] — HWC pair format
        """
        # Render frame_t without hint (first frame)
        frame_t = self.hinted_renderer(mask_t, hint=None)
        # Render frame_t+1 with frame_t as hint (temporal consistency)
        frame_t1 = self.hinted_renderer(mask_t1, hint=frame_t.detach())

        flow = self.motion(mask_t, mask_t1)
        frame_t1_warped = warp_with_flow(frame_t, flow)

        alpha = torch.sigmoid(self.blend_logit)
        # Flow warmup: residual_scale controls how much the direct renderer contributes.
        # During warmup (residual_scale=0), frame_t1 = pure warp, forcing flow to develop.
        effective_direct = (1.0 - alpha) * residual_scale
        frame_t1_blended = (alpha * frame_t1_warped + effective_direct * frame_t1).clamp(0.0, 255.0)

        f_t_hwc = frame_t.permute(0, 2, 3, 1)
        f_t1_hwc = frame_t1_blended.permute(0, 2, 3, 1)
        return torch.stack([f_t_hwc, f_t1_hwc], dim=1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Factory ─────────────────────────────────────────────────────────────


def build_renderer(
    num_classes: int = 5,
    embed_dim: int = 6,
    base_ch: int = 36,
    mid_ch: int = 60,
    motion_hidden: int = 32,
    depth: int = 1,
    blend_mode: str = "scalar",
    noise_mode: str = "deterministic",
    motion_type: str = "learned_cnn",
    depth_priors: dict[int, float] | None = None,
    focal_length: tuple[float, float] | None = None,
    principal_point: tuple[float, float] | None = None,
    camera_height: float | None = None,
) -> PairGenerator:
    """Build the full mask-to-pair rendering pipeline.

    Default settings match the competitor's proven ~300K param budget.

    Features:
        - Shared embedding between renderer and motion predictor
        - CLADE per-class normalization in all ResBlocks
        - Soft sigmoid output for always-flowing gradients
        - Configurable blend mode, noise injection, and motion predictor type

    Args:
        num_classes: segmentation classes (5 for comma)
        embed_dim: per-class embedding dimension
        base_ch: U-Net base channels (stem + decoder)
        mid_ch: U-Net bottleneck channels
        motion_hidden: MotionPredictor hidden channels
        depth: U-Net downscale levels (1 = single-scale, 2 = two-scale)
        blend_mode: "scalar" (learned alpha), "spatial" (per-pixel), "none"
        noise_mode: "deterministic", "shared", "independent"
        motion_type: "depth_aware", "learned_cnn", "analytical", "none"

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
        depth=depth,
    )

    # Motion predictor type selection
    if motion_type == "learned_cnn":
        motion = MotionPredictor(
            num_classes=num_classes,
            embed_dim=embed_dim,
            hidden=motion_hidden,
            embedding=shared_embed,
        )
    elif motion_type == "analytical":
        motion = AnalyticalMotionPredictor(
            num_classes=num_classes,
            embed_dim=embed_dim,
            embedding=shared_embed,
        )
    elif motion_type == "depth_aware":
        from .depth_motion import DepthAwareMotionPredictor
        motion = DepthAwareMotionPredictor(
            num_classes=num_classes,
            depth_priors=depth_priors,
            focal_length=focal_length,
            principal_point=principal_point,
            camera_height=camera_height,
        )
    elif motion_type == "none":
        # Stub: zero-flow motion predictor (direct rendering only)
        motion = MotionPredictor(
            num_classes=num_classes,
            embed_dim=embed_dim,
            hidden=motion_hidden,
            embedding=shared_embed,
        )
        # Override blend_mode to "none" since no motion is used
        blend_mode = "none"
    else:
        raise ValueError(
            f"Unknown motion_type: {motion_type!r}. "
            "Use 'depth_aware', 'learned_cnn', 'analytical', or 'none'."
        )

    pair_gen = PairGenerator(
        renderer, motion,
        blend_mode=blend_mode,
        noise_mode=noise_mode,
    )

    # Verify embedding is truly shared (for types that use shared embedding)
    if hasattr(motion, "embedding"):
        assert renderer.embedding is motion.embedding, "Embedding sharing failed"

    total = pair_gen.param_count()
    r_count = renderer.param_count()
    m_count = motion.param_count()
    print(
        f"[renderer] Built PairGenerator: {total:,} params "
        f"(renderer={r_count:,}, motion={m_count:,} [{motion_type}], "
        f"blend={blend_mode}, noise={noise_mode}, "
        f"shared_embed={shared_embed.weight.numel()}, CLADE=on, sigmoid_out=on)"
    )

    return pair_gen


# ── Technique 3: Analytical Motion Predictor (SenseTime/KAIST) ─────────


class AnalyticalMotionPredictor(nn.Module):
    """Predict optical flow from mask centroid displacement (analytical).

    Instead of learning flow from scratch with a CNN, this computes rigid
    per-class motion analytically from centroid shifts between consecutive
    masks. A tiny learned refinement network (~5K params) corrects only
    at class boundaries where the rigid assumption breaks down.

    This replaces the 32K-param MotionPredictor with ~5K learned params
    plus zero-cost analytical flow for class interiors.

    Args:
        num_classes: segmentation classes (5 for comma SegNet)
        embed_dim: per-class embedding dimension (unused, kept for API compat)
        refine_hidden: hidden channels for boundary refinement net
        embedding: optional pre-created Embedding (for weight sharing)
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 6,
        refine_hidden: int = 8,
        embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding = embedding if embedding is not None else nn.Embedding(num_classes, embed_dim)

        # Boundary refinement: operates on analytical flow + boundary mask
        # Input: 2 (analytical flow) + 1 (boundary indicator) = 3 channels
        self.refine = nn.Sequential(
            nn.Conv2d(3, refine_hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(refine_hidden, 2, 3, padding=1, bias=True),
        )
        # Zero-init so refinement starts as identity
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def _compute_centroids(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute per-class centroids in normalized [-1, 1] coordinates.

        Args:
            mask: (B, H, W) long tensor

        Returns:
            (B, num_classes, 2) tensor of (cx, cy) in [-1, 1]
        """
        B, H, W = mask.shape
        device = mask.device

        # Create coordinate grids
        yy = torch.linspace(-1.0, 1.0, H, device=device)
        xx = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

        # Vectorised centroid computation using one_hot (replaces per-class loop)
        one_hot = F.one_hot(mask, self.num_classes).float()  # (B, H, W, C)
        count = one_hot.sum(dim=(1, 2)).clamp(min=1.0)  # (B, C)
        # Weighted mean over spatial dims: grid (H, W) broadcast with one_hot (B, H, W, C)
        cx = (grid_x.unsqueeze(0).unsqueeze(-1) * one_hot).sum(dim=(1, 2)) / count  # (B, C)
        cy = (grid_y.unsqueeze(0).unsqueeze(-1) * one_hot).sum(dim=(1, 2)) / count  # (B, C)
        centroids = torch.stack([cx, cy], dim=-1)  # (B, C, 2)

        return centroids

    def _compute_boundary_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Detect class boundary pixels via Laplacian on mask.

        Args:
            mask: (B, H, W) long tensor

        Returns:
            (B, 1, H, W) float tensor, 1.0 at boundaries, 0.0 elsewhere
        """
        m = mask.float().unsqueeze(1)  # (B, 1, H, W)
        # Simple boundary: pixels where any neighbor differs
        padded = F.pad(m, (1, 1, 1, 1), mode="replicate")
        shifts = [
            padded[:, :, 1:-1, 2:],  # right
            padded[:, :, 1:-1, :-2],  # left
            padded[:, :, 2:, 1:-1],  # down
            padded[:, :, :-2, 1:-1],  # up
        ]
        boundary = torch.zeros_like(m)
        for s in shifts:
            boundary = boundary + (m != s).float()
        return (boundary > 0).float()

    def forward(
        self,
        mask_t: torch.Tensor,
        mask_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Predict optical flow from mask pair using analytical centroids.

        Args:
            mask_t: (B, H, W) long
            mask_t1: (B, H, W) long

        Returns:
            (B, 2, H, W) flow in normalized coordinates
        """
        B, H, W = mask_t.shape
        device = mask_t.device

        # 1. Compute per-class centroid displacement
        centroids_t = self._compute_centroids(mask_t)  # (B, C, 2)
        centroids_t1 = self._compute_centroids(mask_t1)  # (B, C, 2)
        displacement = centroids_t1 - centroids_t  # (B, C, 2)

        # 2. Build analytical flow by assigning each pixel its class displacement
        # Use mask_t to determine which class each pixel belongs to
        # displacement: (B, C, 2) → index by mask_t: (B, H, W) → (B, H, W, 2)
        disp_flat = displacement.reshape(B * self.num_classes, 2)
        mask_flat = mask_t.reshape(B, H * W)
        # Add batch offset for gathering
        batch_offset = torch.arange(B, device=device).unsqueeze(1) * self.num_classes
        idx = (mask_flat + batch_offset).reshape(-1)
        analytical_flow = disp_flat[idx].reshape(B, H, W, 2).permute(0, 3, 1, 2)
        # Scale to small flow range
        analytical_flow = analytical_flow * 0.1

        # 3. Boundary refinement: only refine at class boundaries
        boundary = self._compute_boundary_mask(mask_t)  # (B, 1, H, W)
        refine_input = torch.cat([analytical_flow, boundary], dim=1)  # (B, 3, H, W)
        refinement = self.refine(refine_input) * 0.05  # small refinement scale
        # Apply refinement only at boundaries
        flow = analytical_flow + refinement * boundary

        return flow

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Technique 7: Depthwise Cascade Renderer (Samsung) ─────────────────


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise separable convolution block with optional dilation.

    Replaces standard Conv2d with:
        depthwise (groups=channels) → pointwise (1x1)
    3-4x fewer params than standard conv.
    """

    def __init__(self, channels: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel,
            padding=pad,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pw = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pw(self.dw(x)))


class DepthwiseMaskRenderer(nn.Module):
    """Mask renderer using depthwise separable convolutions with dilation cascade.

    Replaces standard Conv2d layers in MaskRenderer with depthwise separable
    convolutions stacked at increasing dilation rates (1, 2, 4, 8). This gives
    a large receptive field with 3-4x fewer parameters and better INT8 behavior.

    Architecture:
        mask → Embedding → stem (1x1 project) → [DWSep d=1, d=2, d=4, d=8] → head → RGB

    Args:
        num_classes: segmentation classes
        embed_dim: per-class embedding dimension
        base_ch: channel width throughout the cascade
        embedding: optional shared Embedding
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 6,
        base_ch: int = 36,
        embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.embedding = embedding if embedding is not None else nn.Embedding(num_classes, embed_dim)

        # Stem: project embedding to working channels
        self.stem = nn.Conv2d(embed_dim, base_ch, 1, bias=True)

        # Cascade of depthwise separable blocks with increasing dilation
        self.cascade = nn.ModuleList(
            [
                DepthwiseSeparableBlock(base_ch, kernel=3, dilation=1),
                DepthwiseSeparableBlock(base_ch, kernel=3, dilation=2),
                DepthwiseSeparableBlock(base_ch, kernel=3, dilation=4),
                DepthwiseSeparableBlock(base_ch, kernel=3, dilation=8),
            ]
        )

        # Residual projections (1x1) for skip connections in cascade
        self.skip_proj = nn.Conv2d(base_ch, base_ch, 1, bias=False)

        # Head: project to RGB
        self.head = nn.Conv2d(base_ch, 3, 1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """Render RGB from masks using depthwise cascade.

        Args:
            masks: (B, H, W) long tensor

        Returns:
            (B, 3, H, W) float in [0, 255]
        """
        x = self.embedding(masks).permute(0, 3, 1, 2).contiguous()
        x = self.stem(x)

        # Cascade with residual connections
        skip = self.skip_proj(x)
        for block in self.cascade:
            x = block(x) + x  # local residual
        x = x + skip  # global skip

        rgb = 255.0 * torch.sigmoid(self.head(x) / 50.0)
        return rgb

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Technique 8: Channel-Recurrent Renderer (Sony) ────────────────────


class ChannelSubNet(nn.Module):
    """Tiny sub-network that predicts one channel conditioned on mask + prior channels.

    Args:
        in_ch: input channels (embed_dim + number of prior channels)
        hidden: hidden width
    """

    def __init__(self, in_ch: int, hidden: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        # Zero-init output for stable start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChannelRecurrentRenderer(nn.Module):
    """Channel-recurrent mask renderer (Sony approach).

    Instead of jointly predicting 3 RGB channels, generates them sequentially:
        1. Y = f_Y(mask_embed)           — luminance from mask only
        2. U = f_U(mask_embed, Y)        — chroma U conditioned on Y
        3. V = f_V(mask_embed, Y, U)     — chroma V conditioned on Y, U

    Each sub-network outputs 1 channel, so each is ~1/3 the capacity needed.
    Total params are 40-60% of joint model because later channels get extra
    conditioning input instead of extra capacity.

    The output is in YUV space, converted to RGB for scoring.

    Args:
        num_classes: segmentation classes
        embed_dim: per-class embedding dimension
        hidden: hidden width per sub-network
        embedding: optional shared Embedding
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 6,
        hidden: int = 24,
        embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.embedding = embedding if embedding is not None else nn.Embedding(num_classes, embed_dim)

        # Y channel: input = embed_dim
        self.y_net = ChannelSubNet(embed_dim, hidden)
        # U channel: input = embed_dim + 1 (Y)
        self.u_net = ChannelSubNet(embed_dim + 1, hidden)
        # V channel: input = embed_dim + 2 (Y, U)
        self.v_net = ChannelSubNet(embed_dim + 2, hidden)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """Render RGB from masks via channel-recurrent YUV prediction.

        Args:
            masks: (B, H, W) long tensor

        Returns:
            (B, 3, H, W) float in [0, 255]
        """
        embed = self.embedding(masks).permute(0, 3, 1, 2).contiguous()

        # Sequential channel generation
        y_raw = self.y_net(embed)  # (B, 1, H, W)
        y_norm = torch.sigmoid(y_raw / 50.0)  # normalized Y

        u_input = torch.cat([embed, y_norm], dim=1)
        u_raw = self.u_net(u_input)  # (B, 1, H, W)
        u_norm = torch.sigmoid(u_raw / 50.0)

        v_input = torch.cat([embed, y_norm, u_norm], dim=1)
        v_raw = self.v_net(v_input)  # (B, 1, H, W)
        v_norm = torch.sigmoid(v_raw / 50.0)

        # YUV → RGB conversion (BT.601)
        # Y in [0,1], U in [0,1] mapped to [-0.5, 0.5], V in [0,1] mapped to [-0.5, 0.5]
        y = y_norm
        u = u_norm - 0.5
        v = v_norm - 0.5

        r = (y + 1.402 * v).clamp(0, 1)
        g = (y - 0.344136 * u - 0.714136 * v).clamp(0, 1)
        b = (y + 1.772 * u).clamp(0, 1)

        rgb = torch.cat([r, g, b], dim=1) * 255.0
        return rgb

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
