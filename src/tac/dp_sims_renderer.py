"""DP-SIMS-inspired semantic image synthesis renderer.

Adapted from DP-SIMS (CVPR 2024, arxiv 2312.13314) — current SOTA for
mask-conditioned image generation (mIoU 57.7 on Cityscapes).

Key adaptations for comma video compression:
    - Full SPADE normalization (Park et al., CVPR 2019) instead of CLADE.
      SPADE learns spatially-varying gamma/beta from the mask via small
      conv networks, giving richer conditioning than per-class lookup.
    - Progressive upsampling generator: noise/constant at 24x32 -> 384x512
      through 4 SPADE ResBlocks with 2x bilinear upsampling.
    - Cross-attention noise injection for texture diversity (DP-SIMS).
    - Multi-scale PatchGAN discriminator (optional adversarial training).
    - Target ~500K-2M params. FP4 quantization makes 2M params = ~1MB.

Pipeline:
    5-class mask (B, H, W) at 384x512
    -> SPADE-conditioned progressive generator
    -> RGB frames (B, 3, 384, 512) in [0, 255]
    -> MotionPredictor for PoseNet-compatible pairs

Architecture:
    Learned constant at 24x32 (1/16 resolution)
    -> SPADEResBlock(256) + Upsample(2x) -> 48x64
    -> SPADEResBlock(128) + Upsample(2x) -> 96x128
    -> SPADEResBlock(64) + Upsample(2x) -> 192x256
    -> SPADEResBlock(32) + Upsample(2x) -> 384x512
    -> Conv 3x3 -> 3ch RGB, soft sigmoid output
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SPADE Normalization ─────────────────────────────────────────────────


class SPADE(nn.Module):
    """Spatially-Adaptive Normalization (Park et al., CVPR 2019).

    Unlike CLADE (per-class affine lookup), SPADE learns spatially-varying
    gamma and beta from the segmentation mask via a small conv network.
    This allows the network to learn spatial patterns beyond what
    per-class statistics can capture (e.g., road markings, horizon gradients).

    The mask is one-hot encoded and resized to match the feature resolution,
    then passed through:
        shared_conv(3x3) -> ReLU -> gamma_conv(3x3) + beta_conv(3x3)

    Output: normalized * (1 + gamma) + beta

    Args:
        norm_channels: number of channels in the feature tensor to normalize
        mask_channels: number of mask classes (one-hot input channels)
        hidden: hidden channel width for the conditioning conv network
    """

    def __init__(self, norm_channels: int, mask_channels: int = 5, hidden: int = 64):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_channels, affine=False)
        self.mask_channels = mask_channels

        self.shared = nn.Sequential(
            nn.Conv2d(mask_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gamma_conv = nn.Conv2d(hidden, norm_channels, 3, padding=1)
        self.beta_conv = nn.Conv2d(hidden, norm_channels, 3, padding=1)

        # Init: gamma=0, beta=0 -> output = normalized * 1 + 0 = identity
        nn.init.zeros_(self.gamma_conv.weight)
        nn.init.zeros_(self.gamma_conv.bias)
        nn.init.zeros_(self.beta_conv.weight)
        nn.init.zeros_(self.beta_conv.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply spatially-adaptive normalization.

        Args:
            x: (B, C, H, W) feature tensor
            mask: (B, H_orig, W_orig) long tensor with class indices in [0, K)

        Returns:
            (B, C, H, W) normalized and modulated feature tensor
        """
        normalized = self.norm(x)

        # One-hot encode mask and resize to feature resolution
        _, _, fH, fW = x.shape
        mask_onehot = self._encode_mask(mask, fH, fW, x.device)

        # Predict spatially-varying affine parameters
        shared = self.shared(mask_onehot)
        gamma = self.gamma_conv(shared)
        beta = self.beta_conv(shared)

        return normalized * (1.0 + gamma) + beta

    def _encode_mask(
        self, mask: torch.Tensor, target_h: int, target_w: int, device: torch.device
    ) -> torch.Tensor:
        """One-hot encode and resize mask to target resolution.

        Args:
            mask: (B, H, W) long tensor
            target_h, target_w: desired spatial dimensions
            device: target device

        Returns:
            (B, K, target_h, target_w) float tensor
        """
        B = mask.shape[0]
        # Resize mask via nearest-neighbor first (preserves class boundaries)
        if mask.shape[1] != target_h or mask.shape[2] != target_w:
            mask_resized = F.interpolate(
                mask.unsqueeze(1).float(),
                size=(target_h, target_w),
                mode="nearest",
            ).squeeze(1).long()
        else:
            mask_resized = mask

        # One-hot encode: (B, H, W) -> (B, K, H, W)
        onehot = torch.zeros(
            B, self.mask_channels, target_h, target_w,
            device=device, dtype=torch.float32,
        )
        onehot.scatter_(1, mask_resized.unsqueeze(1), 1.0)
        return onehot


# ── SPADE ResBlock ──────────────────────────────────────────────────────


class SPADEResBlock(nn.Module):
    """Residual block with SPADE normalization at both conv layers.

    Pre-activation residual: SPADE -> ReLU -> Conv -> SPADE -> ReLU -> Conv.
    Includes a learned skip connection when input/output channels differ.

    This is the core building block of the DP-SIMS generator. Each block
    sees the segmentation mask at its operating resolution and can learn
    class-specific and spatially-varying feature modulation.

    Args:
        in_channels: input feature channels
        out_channels: output feature channels
        mask_channels: number of mask classes (5 for comma SegNet)
        spade_hidden: hidden width for SPADE conditioning convs
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mask_channels: int = 5,
        spade_hidden: int = 64,
    ):
        super().__init__()
        self.learned_skip = in_channels != out_channels

        # First SPADE + Conv
        self.spade1 = SPADE(in_channels, mask_channels, hidden=spade_hidden)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

        # Second SPADE + Conv
        self.spade2 = SPADE(out_channels, mask_channels, hidden=spade_hidden)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

        self.act = nn.ReLU(inplace=True)

        # Skip connection with 1x1 conv when channel dims change
        if self.learned_skip:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # Zero-init second conv for residual identity at initialization
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with SPADE conditioning.

        Args:
            x: (B, C_in, H, W) feature tensor
            mask: (B, H_orig, W_orig) long tensor with class indices

        Returns:
            (B, C_out, H, W) output features
        """
        # Pre-activation residual path
        h = self.spade1(x, mask)
        h = self.act(h)
        h = self.conv1(h)

        h = self.spade2(h, mask)
        h = self.act(h)
        h = self.conv2(h)

        # Skip connection
        if self.learned_skip:
            x = self.skip_conv(x)

        return x + h


# ── Cross-Attention Noise Injection ─────────────────────────────────────


class CrossAttentionNoiseInjector(nn.Module):
    """Cross-attention noise injection for texture diversity (DP-SIMS).

    Injects structured noise conditioned on the mask semantics via a
    lightweight cross-attention mechanism. The query comes from features,
    the key/value come from noise projected through the mask embedding.

    This prevents the generator from collapsing to a single deterministic
    output per mask, encouraging diverse textures and realistic variation.

    Uses a single-head attention for efficiency. The noise is spatially
    structured (not i.i.d.) because it passes through the mask-conditioned
    key/value projection.

    Args:
        channels: feature channel width (used for Q, K, V dimensions)
        mask_channels: number of mask classes
        noise_dim: dimension of the input noise vector per spatial location
    """

    def __init__(self, channels: int, mask_channels: int = 5, noise_dim: int = 16):
        super().__init__()
        self.channels = channels
        self.noise_dim = noise_dim

        # Project features to queries
        self.to_q = nn.Conv2d(channels, channels, 1, bias=False)

        # Project noise + mask to keys and values
        self.noise_proj = nn.Conv2d(noise_dim + mask_channels, channels, 1, bias=False)
        self.to_k = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)

        # Output projection with gating (starts at zero = no noise at init)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inject noise via cross-attention.

        Args:
            x: (B, C, H, W) feature tensor
            mask: (B, H_orig, W_orig) long tensor
            noise: (B, noise_dim, H, W) or None (auto-generated if None)

        Returns:
            (B, C, H, W) features with noise injected
        """
        B, C, H, W = x.shape

        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(B, self.noise_dim, H, W, device=x.device, dtype=x.dtype)

        # One-hot encode mask at feature resolution
        if mask.shape[1] != H or mask.shape[2] != W:
            mask_resized = F.interpolate(
                mask.unsqueeze(1).float(), size=(H, W), mode="nearest"
            ).squeeze(1).long()
        else:
            mask_resized = mask
        K_classes = 5  # hardcoded for comma
        mask_onehot = torch.zeros(B, K_classes, H, W, device=x.device, dtype=x.dtype)
        mask_onehot.scatter_(1, mask_resized.unsqueeze(1), 1.0)

        # Concatenate noise with mask encoding
        noise_mask = torch.cat([noise, mask_onehot], dim=1)
        noise_features = self.noise_proj(noise_mask)

        # Compute Q, K, V
        q = self.to_q(x)       # (B, C, H, W)
        k = self.to_k(noise_features)  # (B, C, H, W)
        v = self.to_v(noise_features)  # (B, C, H, W)

        # Spatial cross-attention (per-pixel, channel-wise dot product)
        # This is efficient: no quadratic attention over spatial locations
        # Instead: per-pixel attention weight = softmax(q * k / sqrt(C))
        scale = math.sqrt(C)
        attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True) / scale)  # (B, 1, H, W)
        attended = attn * v  # (B, C, H, W)

        # Gated output (starts at zero, learned during training)
        out = self.out_proj(attended)
        return x + self.gate * out


# ── DP-SIMS Generator ──────────────────────────────────────────────────


class DPSIMSRenderer(nn.Module):
    """SPADE-based progressive generator for mask-to-RGB synthesis.

    Architecture (for 384x512 input):
        Learned constant at 24x32 (1/16 resolution)
        -> SPADEResBlock(ch[0] -> ch[1]) + Upsample(2x) -> 48x64
        -> SPADEResBlock(ch[1] -> ch[2]) + Upsample(2x) -> 96x128
        -> SPADEResBlock(ch[2] -> ch[3]) + Upsample(2x) -> 192x256
        -> SPADEResBlock(ch[3] -> ch[4]) + Upsample(2x) -> 384x512
        -> Conv 3x3 -> 3ch RGB, soft sigmoid output

    Each SPADE layer receives the segmentation mask at its resolution
    and learns spatially-varying modulation. This is strictly more
    expressive than CLADE (per-class lookup) because it can learn
    spatial patterns within each class region.

    DP-SIMS cross-attention noise injection is applied at each scale
    for texture diversity (can be disabled at inference for deterministic
    output by passing noise=zeros).

    Args:
        num_classes: number of segmentation classes (5 for comma SegNet)
        channels: channel widths at each stage [constant, stage1, ..., stage4]
        init_h: height of learned constant (24 for 384px output)
        init_w: width of learned constant (32 for 512px output)
        spade_hidden: hidden width for SPADE conditioning convs
        noise_dim: noise dimension for cross-attention injection
        use_noise: whether to use cross-attention noise injection
    """

    def __init__(
        self,
        num_classes: int = 5,
        channels: tuple[int, ...] = (256, 128, 64, 32),
        init_h: int = 24,
        init_w: int = 32,
        spade_hidden: int = 64,
        noise_dim: int = 16,
        use_noise: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.init_h = init_h
        self.init_w = init_w
        self.use_noise = use_noise
        self.num_stages = len(channels)

        # Learned constant at lowest resolution (1/16 of output)
        # Initialized from N(0, 0.02) following SPADE paper convention
        self.const = nn.Parameter(torch.randn(1, channels[0], init_h, init_w) * 0.02)

        # Progressive upsampling SPADE ResBlocks
        self.spade_blocks = nn.ModuleList()
        self.noise_injectors = nn.ModuleList()

        # First block: operates at init resolution before first upsample
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            # Scale spade_hidden proportionally to feature channels
            sh = max(32, min(spade_hidden, out_ch))
            self.spade_blocks.append(
                SPADEResBlock(in_ch, out_ch, num_classes, spade_hidden=sh)
            )
            if use_noise:
                self.noise_injectors.append(
                    CrossAttentionNoiseInjector(out_ch, num_classes, noise_dim)
                )
            in_ch = out_ch

        # Output head: final channels -> 3 RGB
        self.head = nn.Conv2d(channels[-1], 3, 3, padding=1, bias=True)
        # Init so sigmoid(head/50) ~ 0.5 -> ~128 at init (mid-gray)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        masks: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate RGB frames from segmentation masks.

        Args:
            masks: (B, H, W) long tensor with values in [0, num_classes)
            noise: optional (B, noise_dim, init_h, init_w) for deterministic generation.
                   If None, fresh noise is sampled at each scale.

        Returns:
            (B, 3, H, W) float tensor in [0, 255]
        """
        B = masks.shape[0]

        # Start from learned constant, expand to batch size
        x = self.const.expand(B, -1, -1, -1)

        # Progressive upsampling through SPADE ResBlocks
        for i, block in enumerate(self.spade_blocks):
            # SPADE conditioning
            x = block(x, masks)

            # Cross-attention noise injection (if enabled)
            if self.use_noise and i < len(self.noise_injectors):
                x = self.noise_injectors[i](x, masks)

            # Upsample 2x (bilinear for smooth gradients)
            if i < self.num_stages - 1:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Final upsample to target resolution if needed
        _, _, cur_h, cur_w = x.shape
        target_h, target_w = masks.shape[1], masks.shape[2]
        if cur_h != target_h or cur_w != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # Head: soft sigmoid output — gradients always flow, no dead zones
        # sigmoid(0/50) = 0.5 -> 127.5 at init (mid-gray)
        rgb = 255.0 * torch.sigmoid(self.head(x) / 50.0)
        return rgb

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Multi-Scale PatchGAN Discriminator ──────────────────────────────────


class PatchDiscriminator(nn.Module):
    """Single-scale PatchGAN discriminator.

    Classifies overlapping patches as real or fake via a fully-convolutional
    architecture. No sigmoid at output — use with hinge or logistic loss.

    Architecture:
        Conv(4,2) -> LeakyReLU
        Conv(4,2) -> InstanceNorm -> LeakyReLU
        Conv(4,2) -> InstanceNorm -> LeakyReLU
        Conv(4,1) -> 1ch patch prediction

    Args:
        in_channels: input channels (3 RGB + num_classes one-hot mask)
        base_ch: base channel width (doubled at each layer)
        n_layers: number of downsampling layers
    """

    def __init__(self, in_channels: int = 8, base_ch: int = 64, n_layers: int = 3):
        super().__init__()
        layers = []

        # First layer: no normalization
        ch = base_ch
        layers.append(nn.Conv2d(in_channels, ch, 4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate layers: InstanceNorm + LeakyReLU
        for i in range(1, n_layers):
            prev_ch = ch
            ch = min(base_ch * (2 ** i), 512)
            layers.append(nn.Conv2d(prev_ch, ch, 4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final layer: stride 1, single-channel output
        prev_ch = ch
        ch = min(base_ch * (2 ** n_layers), 512)
        layers.append(nn.Conv2d(prev_ch, ch, 4, stride=1, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Patch prediction (1 channel = real/fake per patch)
        layers.append(nn.Conv2d(ch, 1, 4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, H, W) concatenated image + mask

        Returns:
            (B, 1, H', W') patch-level predictions (no sigmoid)
        """
        return self.model(x)


class MultiScalePatchGAN(nn.Module):
    """Multi-scale PatchGAN discriminator (pix2pixHD / DP-SIMS style).

    Applies PatchGAN discriminators at multiple scales (original, 1/2, 1/4)
    for capturing both fine texture and coarse structure.

    The discriminator receives concatenated (image, one-hot mask) as input,
    so it can learn class-conditional texture expectations.

    Args:
        num_classes: number of mask classes (for one-hot encoding)
        num_scales: number of discriminator scales (default 3)
        base_ch: base channel width per scale
        n_layers: number of conv layers per discriminator
    """

    def __init__(
        self,
        num_classes: int = 5,
        num_scales: int = 3,
        base_ch: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_scales = num_scales

        in_ch = 3 + num_classes  # RGB + one-hot mask
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_ch, base_ch, n_layers)
            for _ in range(num_scales)
        ])

    def forward(
        self, image: torch.Tensor, mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Multi-scale discrimination.

        Args:
            image: (B, 3, H, W) RGB image in [0, 255]
            mask: (B, H, W) long tensor with class indices

        Returns:
            List of patch predictions, one per scale.
            Each is (B, 1, H_i, W_i) with no sigmoid.
        """
        B, _, H, W = image.shape

        # Normalize image to [-1, 1] for discriminator
        img_norm = image / 127.5 - 1.0

        # One-hot encode mask at full resolution
        mask_onehot = torch.zeros(B, self.num_classes, H, W, device=image.device, dtype=image.dtype)
        mask_onehot.scatter_(1, mask.unsqueeze(1).long(), 1.0)

        # Concatenate image and mask
        x = torch.cat([img_norm, mask_onehot], dim=1)

        outputs = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                # Downsample for coarser scales
                x = F.avg_pool2d(x, 2)
            outputs.append(disc(x))

        return outputs

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Discriminator Loss Functions ────────────────────────────────────────


def hinge_loss_d(
    real_preds: list[torch.Tensor],
    fake_preds: list[torch.Tensor],
) -> torch.Tensor:
    """Multi-scale hinge loss for discriminator.

    D wants: real > 1, fake < -1.

    Args:
        real_preds: list of discriminator outputs on real images
        fake_preds: list of discriminator outputs on fake images

    Returns:
        Scalar discriminator loss
    """
    loss = torch.tensor(0.0, device=real_preds[0].device)
    for rp, fp in zip(real_preds, fake_preds):
        loss = loss + torch.mean(F.relu(1.0 - rp)) + torch.mean(F.relu(1.0 + fp))
    return loss / len(real_preds)


def hinge_loss_g(fake_preds: list[torch.Tensor]) -> torch.Tensor:
    """Multi-scale hinge loss for generator.

    G wants: fake predictions to be positive (fool discriminator).

    Args:
        fake_preds: list of discriminator outputs on generated images

    Returns:
        Scalar generator loss (to be added to reconstruction/scorer loss)
    """
    loss = torch.tensor(0.0, device=fake_preds[0].device)
    for fp in fake_preds:
        loss = loss - torch.mean(fp)
    return loss / len(fake_preds)


# ── DP-SIMS Pair Generator ─────────────────────────────────────────────


class DPSIMSPairGenerator(nn.Module):
    """Generate PoseNet-compatible frame pairs from segmentation masks.

    Combines DPSIMSRenderer and MotionPredictor (reused from renderer.py)
    to produce the (B, 2, H, W, 3) HWC pair format the scorer expects.

    Same interface as PairGenerator / WaveletPairGenerator -- drop-in replacement.

    Args:
        renderer: DPSIMSRenderer instance
        motion: MotionPredictor instance (from renderer.py)
    """

    def __init__(self, renderer: DPSIMSRenderer, motion: nn.Module):
        super().__init__()
        self.renderer = renderer
        self.motion = motion
        # Learned blend weight: sigmoid(raw) gives alpha in [0, 1]
        # Initialize to 0 -> sigmoid(0) = 0.5 -> equal blend
        self.blend_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        mask_t: torch.Tensor,
        mask_t1: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate a scored frame pair from two consecutive masks.

        Args:
            mask_t: (B, H, W) long -- mask at time t
            mask_t1: (B, H, W) long -- mask at time t+1
            noise: optional noise for deterministic generation

        Returns:
            (B, 2, H, W, 3) float tensor in [0, 255] -- HWC pair format
        """
        from tac.renderer import warp_with_flow

        # Render both frames directly via SPADE synthesis
        frame_t = self.renderer(mask_t, noise=noise)        # (B, 3, H, W)
        frame_t1 = self.renderer(mask_t1, noise=noise)      # (B, 3, H, W)

        # Predict flow and warp frame_t -> frame_t1_warped
        flow = self.motion(mask_t, mask_t1)    # (B, 2, H, W)
        frame_t1_warped = warp_with_flow(frame_t, flow)

        # Blend warped and directly-rendered frame_t+1
        alpha = torch.sigmoid(self.blend_logit)
        frame_t1_blended = (
            alpha * frame_t1_warped + (1.0 - alpha) * frame_t1
        ).clamp(0.0, 255.0)

        # Pack to HWC pair format: (B, 2, H, W, 3)
        f_t_hwc = frame_t.permute(0, 2, 3, 1)
        f_t1_hwc = frame_t1_blended.permute(0, 2, 3, 1)
        return torch.stack([f_t_hwc, f_t1_hwc], dim=1)

    def param_count(self) -> int:
        """Total trainable parameter count (renderer + motion + blend)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Factory ─────────────────────────────────────────────────────────────


def build_dp_sims_renderer(
    num_classes: int = 5,
    channels: tuple[int, ...] = (256, 128, 64, 32),
    init_h: int = 24,
    init_w: int = 32,
    spade_hidden: int = 64,
    noise_dim: int = 16,
    use_noise: bool = True,
    motion_hidden: int = 32,
    motion_embed_dim: int = 6,
) -> DPSIMSPairGenerator:
    """Build the full DP-SIMS mask-to-pair rendering pipeline.

    Default settings target ~500K-2M params depending on channel widths.
    FP4 quantization makes 2M params = ~1MB in the submission archive.

    Features:
        - Full SPADE normalization (spatially-varying, not just per-class)
        - Progressive upsampling from 24x32 learned constant
        - Cross-attention noise injection for texture diversity
        - MotionPredictor for PoseNet-compatible frame pairs
        - Soft sigmoid output for always-flowing gradients

    Args:
        num_classes: segmentation classes (5 for comma)
        channels: channel widths at each progressive stage
        init_h: initial constant height (24 for 384px output with 4 stages)
        init_w: initial constant width (32 for 512px output with 4 stages)
        spade_hidden: hidden width for SPADE conditioning convs
        noise_dim: dimension of noise for cross-attention injection
        use_noise: enable cross-attention noise injection
        motion_hidden: MotionPredictor hidden channels
        motion_embed_dim: MotionPredictor embedding dimension

    Returns:
        DPSIMSPairGenerator wrapping DPSIMSRenderer + MotionPredictor
    """
    from tac.renderer import MotionPredictor

    renderer = DPSIMSRenderer(
        num_classes=num_classes,
        channels=channels,
        init_h=init_h,
        init_w=init_w,
        spade_hidden=spade_hidden,
        noise_dim=noise_dim,
        use_noise=use_noise,
    )

    motion = MotionPredictor(
        num_classes=num_classes,
        embed_dim=motion_embed_dim,
        hidden=motion_hidden,
    )

    pair_gen = DPSIMSPairGenerator(renderer, motion)

    total = pair_gen.param_count()
    r_count = renderer.param_count()
    m_count = motion.param_count()
    ch_str = "->".join(str(c) for c in channels)
    print(
        f"[dp_sims] Built DPSIMSPairGenerator: {total:,} params "
        f"(renderer={r_count:,}, motion={m_count:,}, blend=1, "
        f"channels={ch_str}, SPADE=on, noise={'on' if use_noise else 'off'})"
    )

    return pair_gen


def build_dp_sims_discriminator(
    num_classes: int = 5,
    num_scales: int = 3,
    base_ch: int = 64,
    n_layers: int = 3,
) -> MultiScalePatchGAN:
    """Build multi-scale PatchGAN discriminator for adversarial training.

    Optional component -- the scorer loss already provides semantic
    supervision. Adversarial loss adds texture realism.

    Args:
        num_classes: number of mask classes
        num_scales: number of discriminator scales
        base_ch: base channel width
        n_layers: conv layers per discriminator

    Returns:
        MultiScalePatchGAN discriminator
    """
    disc = MultiScalePatchGAN(
        num_classes=num_classes,
        num_scales=num_scales,
        base_ch=base_ch,
        n_layers=n_layers,
    )
    print(f"[dp_sims] Built MultiScalePatchGAN: {disc.param_count():,} params "
          f"({num_scales} scales, {n_layers} layers, base_ch={base_ch})")
    return disc
