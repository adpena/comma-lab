# SPDX-License-Identifier: MIT
# ROUNDTRIP_NOT_REQUIRED: this is a RENDERER (image-decoder), not an
# encoder/quantizer. The "quantizr" in the name refers to the contestant
# Jimmy "Quantizr" who wrote PR #55; nothing in this module quantizes.
# FP4 quantization happens elsewhere (tac.fp4_quantize) at export time.
"""Lane Q-FAITHFUL: 1:1 reverse-engineering of Quantizr PR #55 architecture.

This module is a faithful, line-for-line port of the JointFrameGenerator from
Quantizr's `submissions/quantizr/inflate.py` (PR #55, head SHA
e0b643b0a7c21f62cc93b5d920bcf3fc0d5a33d9, 0.33 contest score).

Key architectural decisions (verbatim from the audit
.omx/research/quantizr_replica_audit_20260428.md):

1. NO motion module, NO optical flow, NO warp. Quantizr explicitly removed
   these in his PR description: "dropping optical flow and using
   Feature-wise Linear Modulation on pose vectors instead of using both
   masks". Our prior Lane V replica kept the warp; that was the bug.

2. Single mask input. forward(mask2, pose6) -> (frame1, frame2). Both frames
   are derived from the SAME odd-frame mask via two parallel heads sharing a
   trunk. There is no mask_t input.

3. frame2_head is UNCONDITIONAL (Frame2StaticHead). frame1_head is
   FiLM-conditioned on the pose6 vector via a 2-layer MLP that expands
   pose_dim=6 -> cond_dim=48 before feeding into FiLMSepResBlock.

4. DSConv (depthwise-separable conv) trunk. SepConvGNAct = DW + PW + GN +
   SiLU. SepResBlock stacks two of these with a residual.

5. Total params ~88K with c1=56, c2=64, hidden=52, depth_mult=1.

This is the architecture Quantizr trained to score 0.33. Lane V's prior
replica retained our motion module + dual-mask + warp, which was the wrong
architectural family.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "JointFrameGenerator",
    "SharedMaskDecoder",
    "Frame2StaticHead",
    "FiLMFrameHead",
    "FrameHead",  # alias for FiLMFrameHead (matches Quantizr name)
    "SepConvGNAct",
    "SepConv",
    "SepResBlock",
    "FiLMSepResBlock",
    "make_coord_grid",
    "build_quantizr_faithful_renderer",
]


def make_coord_grid(
    batch: int, height: int, width: int, device, dtype=torch.float32
) -> torch.Tensor:
    """Build a (B, 2, H, W) normalized coord grid in [-1, 1] (xx, yy).

    Matches Quantizr's `inflate.py:225-230` exactly. Note this is a different
    layout than `tac.renderer.make_coord_grid` (which returns (1, H, W, 2));
    Quantizr feeds the grid as channels into a conv, so it must be (B, 2, H,
    W). We do NOT reuse the existing helper.
    """
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


class SepConvGNAct(nn.Module):
    """Depthwise-separable conv + GroupNorm + SiLU.

    Matches Quantizr's `inflate.py:83-95` SepConvGNAct.
    Replaces QConv2d (a no-op subclass of nn.Conv2d at inflate time; the
    "quantize_weight" flag affects compress-time behavior only).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        stride: int = 1,
        depth_mult: int = 4,
    ):
        super().__init__()
        pad = k // 2
        mid_ch = in_ch * depth_mult
        self.dw = nn.Conv2d(
            in_ch, mid_ch, k, stride=stride, padding=pad, groups=in_ch, bias=False
        )
        self.pw = nn.Conv2d(mid_ch, out_ch, 1, padding=0, bias=True)
        self.norm = nn.GroupNorm(2, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pw(self.dw(x))))


class SepConv(nn.Module):
    """Depthwise-separable conv (no norm, no act). Matches `inflate.py:97-107`."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        stride: int = 1,
        depth_mult: int = 4,
    ):
        super().__init__()
        pad = k // 2
        mid_ch = in_ch * depth_mult
        self.dw = nn.Conv2d(
            in_ch, mid_ch, k, stride=stride, padding=pad, groups=in_ch, bias=False
        )
        self.pw = nn.Conv2d(mid_ch, out_ch, 1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class SepResBlock(nn.Module):
    """Residual block: SepConvGNAct -> SepConv + GN + SiLU. `inflate.py:109-118`."""

    def __init__(self, ch: int, depth_mult: int = 4):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult)
        self.norm2 = nn.GroupNorm(2, ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.norm2(self.conv2(self.conv1(x))))


class FiLMSepResBlock(nn.Module):
    """FiLM-conditioned residual block. `inflate.py:120-138`.

    The FiLM modulation applies (1 + gamma) * x + beta where gamma, beta come
    from a single Linear projection of the conditioning embedding split in
    half along the channel dim.
    """

    def __init__(self, ch: int, cond_dim: int, depth_mult: int = 4):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult)
        self.norm2 = nn.GroupNorm(2, ch)
        self.film_proj = nn.Linear(cond_dim, ch * 2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm2(self.conv2(self.conv1(x)))
        film = self.film_proj(cond_emb).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = film.chunk(2, dim=1)
        x = x * (1.0 + gamma) + beta
        return self.act(residual + x)


class SharedMaskDecoder(nn.Module):
    """Encoder/decoder trunk that consumes a single mask + coord grid.

    Matches Quantizr's `inflate.py:140-168`. Default params (c1=56, c2=64,
    depth_mult=1) per `JointFrameGenerator.__init__` `inflate.py:198-211`.
    """

    def __init__(
        self,
        num_classes: int = 5,
        emb_dim: int = 6,
        c1: int = 56,
        c2: int = 64,
        depth_mult: int = 1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, emb_dim)

        self.stem_conv = SepConvGNAct(emb_dim + 2, c1, depth_mult=depth_mult)
        self.stem_block = SepResBlock(c1, depth_mult=depth_mult)

        self.down_conv = SepConvGNAct(c1, c2, stride=2, depth_mult=depth_mult)
        self.down_block = SepResBlock(c2, depth_mult=depth_mult)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            SepConvGNAct(c2, c1, depth_mult=depth_mult),
        )

        self.fuse = SepConvGNAct(c1 + c1, c1, depth_mult=depth_mult)
        self.fuse_block = SepResBlock(c1, depth_mult=depth_mult)

    def forward(self, mask2: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        e2 = self.embedding(mask2.long()).permute(0, 3, 1, 2)
        e2_up = F.interpolate(
            e2, size=coords.shape[-2:], mode="bilinear", align_corners=False
        )
        x = torch.cat([e2_up, coords], dim=1)
        s = self.stem_block(self.stem_conv(x))
        z = self.down_block(self.down_conv(s))
        z = self.up(z)
        f = self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))
        return f


class Frame2StaticHead(nn.Module):
    """Unconditional decoder for frame2 (no FiLM). `inflate.py:170-182`.

    Output is sigmoid-clamped and scaled to [0, 255] (RGB pixel range).
    """

    def __init__(self, in_ch: int, hidden: int = 52, depth_mult: int = 1):
        super().__init__()
        self.block1 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = nn.Conv2d(hidden, 3, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.block1(feat)
        x = self.block2(x)
        x = self.pre(x)
        return torch.sigmoid(self.head(x)) * 255.0


class FiLMFrameHead(nn.Module):
    """FiLM-conditioned decoder for frame1. `inflate.py:184-196` (FrameHead)."""

    def __init__(
        self,
        in_ch: int,
        cond_dim: int = 48,
        hidden: int = 52,
        depth_mult: int = 1,
    ):
        super().__init__()
        self.block1 = FiLMSepResBlock(in_ch, cond_dim, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = nn.Conv2d(hidden, 3, 1)

    def forward(self, feat: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        x = self.block1(feat, cond_emb)
        x = self.block2(x)
        x = self.pre(x)
        return torch.sigmoid(self.head(x)) * 255.0


# Alias to keep parity with Quantizr's source naming.
FrameHead = FiLMFrameHead


class JointFrameGenerator(nn.Module):
    """Top-level Quantizr architecture. `inflate.py:198-223`.

    forward(mask2, pose6) -> (frame1, frame2)

    - mask2: (B, H, W) long tensor in [0, num_classes-1]
    - pose6: (B, pose_dim) float tensor (6-DOF egomotion vector)
    - Returns (pred_frame1, pred_frame2) each (B, 3, H_out, W_out) float in
      [0, 255]. H_out=384, W_out=512 (the renderer's native resolution).

    No motion module. No warp. frame1 is FiLM-conditioned on pose;
    frame2 is unconditional. Both are derived from the SAME mask2 input.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pose_dim: int = 6,
        cond_dim: int = 48,
        depth_mult: int = 1,
        out_h: int = 384,
        out_w: int = 512,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pose_dim = pose_dim
        self.cond_dim = cond_dim
        self.depth_mult = depth_mult
        self.out_h = out_h
        self.out_w = out_w

        self.shared_trunk = SharedMaskDecoder(
            num_classes=num_classes, emb_dim=6, c1=56, c2=64, depth_mult=depth_mult
        )
        # Quantizr `compress.py:204-205`: Linear(6,48) -> SiLU -> Linear(48,48).
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.frame1_head = FiLMFrameHead(
            in_ch=56, cond_dim=cond_dim, hidden=52, depth_mult=depth_mult
        )
        self.frame2_head = Frame2StaticHead(
            in_ch=56, hidden=52, depth_mult=depth_mult
        )

    def forward(
        self, mask2: torch.Tensor, pose6: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = mask2.shape[0]
        coords = make_coord_grid(b, self.out_h, self.out_w, mask2.device, torch.float32)

        shared_feat = self.shared_trunk(mask2, coords)
        pred_frame2 = self.frame2_head(shared_feat)

        cond_emb = self.pose_mlp(pose6)
        pred_frame1 = self.frame1_head(shared_feat, cond_emb)

        return pred_frame1, pred_frame2


def build_quantizr_faithful_renderer(
    num_classes: int = 5,
    pose_dim: int = 6,
    cond_dim: int = 48,
    depth_mult: int = 1,
) -> JointFrameGenerator:
    """Builder entrypoint for profiles + train_renderer dispatch.

    Returns a JointFrameGenerator with Quantizr's exact PR-#55 default config.
    """
    return JointFrameGenerator(
        num_classes=num_classes,
        pose_dim=pose_dim,
        cond_dim=cond_dim,
        depth_mult=depth_mult,
    )
