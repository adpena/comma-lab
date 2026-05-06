"""Joint Pair Generator — Y-shaped U-Net for mask-conditioned frame pair generation.

Architecture:
  Input: concatenated mask pair (B, 10, H, W) — 5 one-hot classes × 2 frames
  Shared encoder: 4 downsampling blocks with skip connections
  Two decoder heads: each produces one frame (B, 3, H, W)
  Output: (B, 2, H, W, 3) — frame pair in HWC format

Design rationale:
  - Y-shape ensures shared latent captures inter-frame relationship (PoseNet coupling)
  - Skip connections from encoder to BOTH decoders (SegNet boundary preservation)
  - Concatenated mask input: model sees both masks simultaneously
  - No warp/flow — frames generated directly from masks (avoids interpolation artifacts)
  - 500K params target, 300KB FP4 (within rate budget)

Usage:
    model = JointPairGenerator(num_classes=5, base_ch=48)
    frames = model(mask1, mask2)  # (B, 2, H, W, 3)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Depthwise-separable Conv-BN-ReLU block (parameter-efficient)."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        # Depthwise separable: depthwise + pointwise = ~9x fewer params than full conv
        if in_ch == out_ch and in_ch >= 8:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size, padding=kernel_size // 2, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
            )
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    """Downsample: two ConvBlocks + 2x2 max pool."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv2(self.conv1(x))
        return self.pool(feat), feat  # downsampled, skip


class UpBlock(nn.Module):
    """Upsample: bilinear 2x + concat skip + two ConvBlocks."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBlock(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class JointPairGenerator(nn.Module):
    """Y-shaped U-Net: shared encoder → two decoder heads → two frames.

    Architecture:
    - Input: (mask1, mask2) each (B, H, W) long → one-hot → concat → (B, 10, H, W)
    - Shared encoder captures inter-frame relationship
    - Skip connections to BOTH decoders preserve spatial detail
    - Each decoder outputs one RGB frame (B, 3, H, W) in [0, 255]

    Args:
        num_classes: number of segmentation classes (default 5)
        base_ch: base channel count (default 48 for ~500K params)
    """

    def __init__(self, num_classes: int = 5, base_ch: int = 48):
        super().__init__()
        self.num_classes = num_classes
        ch = base_ch

        # Shared encoder — 3 levels (compact for ~500K params)
        self.enc1 = DownBlock(num_classes * 2, ch)         # 10 → ch
        self.enc2 = DownBlock(ch, ch * 2)                   # ch → 2ch
        self.enc3 = DownBlock(ch * 2, ch * 4)               # 2ch → 4ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(ch * 4, ch * 4),
            ConvBlock(ch * 4, ch * 4),
        )

        # Decoder head A (frame 1) — 3 levels with correct skip channel counts
        # dec3: input=bottleneck(ch*4), skip=s3(ch*4) → output ch*2
        # dec2: input=ch*2, skip=s2(ch*2) → output ch
        # dec1: input=ch, skip=s1(ch) → output ch
        self.dec_a3 = UpBlock(ch * 4, ch * 4, ch * 2)
        self.dec_a2 = UpBlock(ch * 2, ch * 2, ch)
        self.dec_a1 = UpBlock(ch, ch, ch)
        self.head_a = nn.Conv2d(ch, 3, 1)

        # Decoder head B (frame 2)
        self.dec_b3 = UpBlock(ch * 4, ch * 4, ch * 2)
        self.dec_b2 = UpBlock(ch * 2, ch * 2, ch)
        self.dec_b1 = UpBlock(ch, ch, ch)
        self.head_b = nn.Conv2d(ch, 3, 1)

    def _masks_to_input(self, mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """Convert mask pair to one-hot concatenated input (B, 10, H, W).

        Note: .contiguous() after permute is required — one_hot produces a
        non-contiguous tensor after permute, and MPS BatchNorm backward
        crashes on non-contiguous inputs (PyTorch MPS .view() bug).
        """
        B, H, W = mask1.shape
        oh1 = F.one_hot(mask1.long(), self.num_classes).permute(0, 3, 1, 2).contiguous().float()
        oh2 = F.one_hot(mask2.long(), self.num_classes).permute(0, 3, 1, 2).contiguous().float()
        return torch.cat([oh1, oh2], dim=1).contiguous()  # (B, 2C, H, W)

    def _decode_3(self, z: torch.Tensor, skips: list[torch.Tensor],
                  dec3, dec2, dec1, head) -> torch.Tensor:
        """Run one decoder head with 3 levels of skip connections.

        ``skips`` is ordered SHALLOWEST-to-DEEPEST (resolution-descending order):

            skips[0] = full-resolution skip (s1)
            skips[1] = half-resolution skip (s2)
            skips[2] = quarter-resolution skip (s3, deepest)

        The decoder consumes them in REVERSE order (deepest first), matching
        the standard U-Net semantics: ``dec3`` consumes ``skips[2]`` (deepest),
        then ``dec2`` consumes ``skips[1]``, then ``dec1`` consumes ``skips[0]``
        as the resolution climbs back to full size before the head 1×1 conv.
        (PARADIGM-γ audit #13 (2026-05-06): docstring clarification — earlier
        revisions did not document the skip ordering, leaving it implicit.)
        """
        x = dec3(z, skips[2])
        x = dec2(x, skips[1])
        x = dec1(x, skips[0])
        return torch.sigmoid(head(x)) * 255.0  # [0, 255] RGB

    def forward(self, mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """Generate frame pair from mask pair.

        Args:
            mask1: (B, H, W) long tensor, class indices for frame 1
            mask2: (B, H, W) long tensor, class indices for frame 2

        Returns:
            (B, 2, H, W, 3) float tensor — frame pair in HWC format
        """
        inp = self._masks_to_input(mask1, mask2)

        # Shared encoder — 3 levels
        x, s1 = self.enc1(inp)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        z = self.bottleneck(x)

        # Two decoder heads — each gets all skip connections
        # skips ordered deepest-first: s3 (1/4 res), s2 (1/2 res), s1 (full res), inp (full res)
        frame1_chw = self._decode_3(z, [s1, s2, s3],
                                     self.dec_a3, self.dec_a2, self.dec_a1, self.head_a)
        frame2_chw = self._decode_3(z, [s1, s2, s3],
                                     self.dec_b3, self.dec_b2, self.dec_b1, self.head_b)

        # Stack and convert CHW → HWC
        pair = torch.stack([frame1_chw, frame2_chw], dim=1)  # (B, 2, 3, H, W)
        return pair.permute(0, 1, 3, 4, 2).contiguous()  # (B, 2, H, W, 3)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Quick architecture verification
    for base_ch in [32, 40, 48, 56, 64]:
        model = JointPairGenerator(num_classes=5, base_ch=base_ch)
        n = count_params(model)
        fp4_kb = n * 4 / 8 / 1024  # 4 bits per param
        print(f"  base_ch={base_ch}: {n:,} params, FP4≈{fp4_kb:.0f}KB")

    # Smoke test
    model = JointPairGenerator(num_classes=5, base_ch=48)
    mask1 = torch.randint(0, 5, (2, 384, 512))
    mask2 = torch.randint(0, 5, (2, 384, 512))
    out = model(mask1, mask2)
    print(f"\n  Input: mask1={mask1.shape}, mask2={mask2.shape}")
    print(f"  Output: {out.shape}, range [{out.min():.1f}, {out.max():.1f}]")
    assert out.shape == (2, 2, 384, 512, 3), f"Wrong shape: {out.shape}"
    print("  PASSED")
