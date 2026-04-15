#!/usr/bin/env python
"""Inflate path using a trained DP-SIMS neural renderer.

The renderer generates RGB frames purely from SegNet masks.  The archive
contains both the renderer weights (~150KB) and pre-extracted masks encoded
as AV1 monochrome video (~30-50KB).  No SegNet loading at inflate time.

Pipeline (contest-compliant, PR #35):
    archive/masks.mkv  ->  AV1 decode  ->  masks (384x512)
    masks              ->  Renderer    ->  frames (384x512)
    frames             ->  bilinear    ->  raw RGB (1164x874)

Fallback (development only, not contest-compliant):
    GT video  ->  SegNet (upstream)  ->  masks (384x512)

Architecture classes (SPADE, SPADEResBlock, DPSIMSRenderer) are inlined
for standalone operation on scorer machines without the tac package.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import av


# ============================================================
# Constants
# ============================================================
OUT_W, OUT_H = 1164, 874
SEG_W, SEG_H = 512, 384
NUM_FRAMES = 1200
NUM_CLASSES = 5
EXPECTED_RAW_BYTES = OUT_W * OUT_H * 3 * NUM_FRAMES  # 3,662,409,600


# ============================================================
# Canonical YUV->RGB (BT.601 limited range, matches frame_utils.py)
# Copied from inflate_postfilter.py — must stay identical.
# ============================================================
def yuv420_to_rgb(frame) -> torch.Tensor:
    H, W = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(H, frame.planes[0].line_size)[:, :W]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(H // 2, frame.planes[1].line_size)[:, :W // 2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(H // 2, frame.planes[2].line_size)[:, :W // 2]

    y_t = torch.from_numpy(y.copy()).float()
    u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)

    u_up = F.interpolate(u_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()

    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)

    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)


# ============================================================
# Inline DPSIMSRenderer (forward-only, no training code)
# Self-contained fallback for scorer machines without tac.
# ============================================================
try:
    from tac.renderer import AsymmetricPairGenerator, MaskRenderer, MotionPredictor, ResBlock, CLADENorm, warp_with_flow, make_coord_grid
    _HAS_TAC_RENDERER = True
except ImportError:
    _HAS_TAC_RENDERER = False

try:
    from tac.dp_sims_renderer import SPADE, SPADEResBlock, CrossAttentionNoiseInjector, DPSIMSRenderer
except ImportError:

    class SPADE(nn.Module):
        """Spatially-Adaptive Normalization (Park et al., CVPR 2019)."""

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
            nn.init.zeros_(self.gamma_conv.weight)
            nn.init.zeros_(self.gamma_conv.bias)
            nn.init.zeros_(self.beta_conv.weight)
            nn.init.zeros_(self.beta_conv.bias)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            normalized = self.norm(x)
            _, _, fH, fW = x.shape
            mask_onehot = self._encode_mask(mask, fH, fW, x.device)
            shared = self.shared(mask_onehot)
            gamma = self.gamma_conv(shared)
            beta = self.beta_conv(shared)
            return normalized * (1.0 + gamma) + beta

        def _encode_mask(self, mask: torch.Tensor, target_h: int, target_w: int, device: torch.device) -> torch.Tensor:
            B = mask.shape[0]
            if mask.shape[1] != target_h or mask.shape[2] != target_w:
                mask_resized = (
                    F.interpolate(mask.unsqueeze(1).float(), size=(target_h, target_w), mode="nearest")
                    .squeeze(1).long()
                )
            else:
                mask_resized = mask
            onehot = torch.zeros(B, self.mask_channels, target_h, target_w, device=device, dtype=torch.float32)
            onehot.scatter_(1, mask_resized.unsqueeze(1), 1.0)
            return onehot

    class SPADEResBlock(nn.Module):
        """Residual block with SPADE normalization."""

        def __init__(self, in_channels: int, out_channels: int, mask_channels: int = 5, spade_hidden: int = 64):
            super().__init__()
            self.learned_skip = in_channels != out_channels
            self.spade1 = SPADE(in_channels, mask_channels, hidden=spade_hidden)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
            self.spade2 = SPADE(out_channels, mask_channels, hidden=spade_hidden)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            self.act = nn.ReLU(inplace=True)
            if self.learned_skip:
                self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            nn.init.zeros_(self.conv2.weight)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            h = self.spade1(x, mask)
            h = self.act(h)
            h = self.conv1(h)
            h = self.spade2(h, mask)
            h = self.act(h)
            h = self.conv2(h)
            if self.learned_skip:
                x = self.skip_conv(x)
            return x + h

    class CrossAttentionNoiseInjector(nn.Module):
        """Cross-attention noise injection for texture diversity."""

        def __init__(self, channels: int, mask_channels: int = 5, noise_dim: int = 16):
            super().__init__()
            self.channels = channels
            self.mask_channels = mask_channels
            self.noise_dim = noise_dim
            self.to_q = nn.Conv2d(channels, channels, 1, bias=False)
            self.noise_proj = nn.Conv2d(noise_dim + mask_channels, channels, 1, bias=False)
            self.to_k = nn.Conv2d(channels, channels, 1, bias=False)
            self.to_v = nn.Conv2d(channels, channels, 1, bias=False)
            self.out_proj = nn.Conv2d(channels, channels, 1, bias=True)
            self.gate = nn.Parameter(torch.zeros(1))
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

        def forward(self, x: torch.Tensor, mask: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
            import math
            B, C, H, W = x.shape
            if noise is None:
                noise = torch.randn(B, self.noise_dim, H, W, device=x.device, dtype=x.dtype)
            if mask.shape[1] != H or mask.shape[2] != W:
                mask_resized = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode="nearest").squeeze(1).long()
            else:
                mask_resized = mask
            mask_onehot = torch.zeros(B, self.mask_channels, H, W, device=x.device, dtype=x.dtype)
            mask_onehot.scatter_(1, mask_resized.unsqueeze(1), 1.0)
            noise_mask = torch.cat([noise, mask_onehot], dim=1)
            noise_features = self.noise_proj(noise_mask)
            q = self.to_q(x)
            k = self.to_k(noise_features)
            v = self.to_v(noise_features)
            scale = math.sqrt(C)
            attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True) / scale)
            attended = attn * v
            out = self.out_proj(attended)
            return x + self.gate * out

    class DPSIMSRenderer(nn.Module):
        """SPADE-based progressive generator for mask-to-RGB synthesis."""

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
            self.const = nn.Parameter(torch.randn(1, channels[0], init_h, init_w) * 0.02)
            self.spade_blocks = nn.ModuleList()
            self.noise_injectors = nn.ModuleList()
            in_ch = channels[0]
            for i, out_ch in enumerate(channels):
                sh = max(32, min(spade_hidden, out_ch))
                self.spade_blocks.append(SPADEResBlock(in_ch, out_ch, num_classes, spade_hidden=sh))
                if use_noise:
                    self.noise_injectors.append(CrossAttentionNoiseInjector(out_ch, num_classes, noise_dim))
                in_ch = out_ch
            self.final_upsample = nn.ConvTranspose2d(channels[-1], channels[-1], 4, stride=2, padding=1, bias=False)
            self.head = nn.Conv2d(channels[-1], 3, 3, padding=1, bias=True)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, masks: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
            B = masks.shape[0]
            x = self.const.expand(B, -1, -1, -1)
            for i, block in enumerate(self.spade_blocks):
                x = block(x, masks)
                if self.use_noise and i < len(self.noise_injectors):
                    x = self.noise_injectors[i](x, masks)
                if i < self.num_stages - 1:
                    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            _, _, cur_h, cur_w = x.shape
            target_h, target_w = masks.shape[1], masks.shape[2]
            if cur_h != target_h or cur_w != target_w:
                x = self.final_upsample(x)
            if x.shape[2] != target_h or x.shape[3] != target_w:
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            rgb = 255.0 * torch.sigmoid(self.head(x) / 50.0)
            return rgb


# ============================================================
# Inline AsymmetricPairGenerator (forward-only, no training code)
# Self-contained fallback for scorer machines without tac.
# ============================================================
if not _HAS_TAC_RENDERER:
    _coord_grid_cache: dict = {}

    def make_coord_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, device)
        if key not in _coord_grid_cache:
            gy = torch.linspace(-1, 1, h, device=device)
            gx = torch.linspace(-1, 1, w, device=device)
            grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
            _coord_grid_cache[key] = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
            if len(_coord_grid_cache) > 4:
                oldest = next(iter(_coord_grid_cache))
                del _coord_grid_cache[oldest]
        return _coord_grid_cache[key]

    def warp_with_flow(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        B, _, H, W = image.shape
        grid = make_coord_grid(H, W, image.device).expand(B, -1, -1, -1)
        flow_hw = flow.permute(0, 2, 3, 1)
        sample_grid = grid + flow_hw
        return F.grid_sample(image, sample_grid, mode="bilinear",
                             padding_mode="border", align_corners=True)

    class CLADENorm(nn.Module):
        def __init__(self, channels: int, num_classes: int = 5):
            super().__init__()
            self.gn = nn.GroupNorm(1, channels)
            self.class_gamma = nn.Embedding(num_classes, channels)
            self.class_beta = nn.Embedding(num_classes, channels)
            nn.init.ones_(self.class_gamma.weight)
            nn.init.zeros_(self.class_beta.weight)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            h = self.gn(x)
            _, _, fH, fW = x.shape
            if mask.shape[1] != fH or mask.shape[2] != fW:
                mask_ds = F.interpolate(mask.unsqueeze(1).float(),
                                        size=(fH, fW), mode="nearest").squeeze(1).long()
            else:
                mask_ds = mask
            gamma = self.class_gamma(mask_ds).permute(0, 3, 1, 2)
            beta = self.class_beta(mask_ds).permute(0, 3, 1, 2)
            return gamma * h + beta

    class ResBlock(nn.Module):
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

    class MaskRenderer(nn.Module):
        def __init__(self, num_classes=5, embed_dim=6, base_ch=36, mid_ch=60,
                     embedding=None, depth=1):
            super().__init__()
            self.num_classes = num_classes
            self.embed_dim = embed_dim
            self.depth = depth
            self.embedding = embedding if embedding is not None else nn.Embedding(num_classes, embed_dim)
            self.use_coord_grid = True
            coord_channels = 2
            self.stem_conv = nn.Conv2d(embed_dim + coord_channels, base_ch, 3, padding=1, bias=True)
            self.stem_res = ResBlock(base_ch, num_classes=num_classes)
            self.down_conv = nn.Conv2d(base_ch, mid_ch, 3, stride=2, padding=1, bias=True)
            self.down_res = ResBlock(mid_ch, num_classes=num_classes)
            if depth >= 2:
                self.down2_conv = nn.Conv2d(mid_ch, mid_ch, 3, stride=2, padding=1, bias=True)
                self.down2_res = ResBlock(mid_ch, num_classes=num_classes)
            self.bottleneck = ResBlock(mid_ch, num_classes=num_classes)
            if depth >= 2:
                self.up2_conv = nn.ConvTranspose2d(mid_ch, mid_ch, 4, stride=2, padding=1, bias=True)
                self.up2_res = ResBlock(mid_ch, num_classes=num_classes)
                self.fuse2_conv = nn.Conv2d(mid_ch * 2, mid_ch, 1, bias=True)
            self.up_conv = nn.ConvTranspose2d(mid_ch, base_ch, 4, stride=2, padding=1, bias=True)
            self.up_res = ResBlock(base_ch, num_classes=num_classes)
            self.fuse_conv = nn.Conv2d(base_ch * 2, base_ch, 1, bias=True)
            self.head = nn.Conv2d(base_ch, 3, 1, bias=True)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, masks: torch.Tensor) -> torch.Tensor:
            x = self.embedding(masks).permute(0, 3, 1, 2).contiguous()
            B, _, H, W = x.shape
            gy = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
            gx = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
            grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
            coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
            x = torch.cat([x, coords], dim=1)
            stem = self.stem_conv(x)
            stem = self.stem_res(stem, masks)
            down1 = self.down_conv(stem)
            down1 = self.down_res(down1, masks)
            if self.depth >= 2:
                down2 = self.down2_conv(down1)
                down2 = self.down2_res(down2, masks)
                mid = self.bottleneck(down2, masks)
                up2 = self.up2_conv(mid)
                if up2.shape[2:] != down1.shape[2:]:
                    up2 = F.interpolate(up2, size=down1.shape[2:], mode="bilinear", align_corners=False)
                up2 = self.up2_res(up2, masks)
                fused2 = torch.cat([down1, up2], dim=1)
                half_res = self.fuse2_conv(fused2)
            else:
                half_res = self.bottleneck(down1, masks)
            up = self.up_conv(half_res)
            if up.shape[2:] != stem.shape[2:]:
                up = F.interpolate(up, size=stem.shape[2:], mode="bilinear", align_corners=False)
            up = self.up_res(up, masks)
            fused = torch.cat([stem, up], dim=1)
            fused = self.fuse_conv(fused)
            rgb = 255.0 * torch.sigmoid(self.head(fused) / 50.0)
            return rgb

    class MotionPredictor(nn.Module):
        def __init__(self, num_classes=5, embed_dim=6, hidden=32, embedding=None,
                     output_channels=2, use_coord_grid=True, use_diff_features=True,
                     max_flow_px=20.0, max_residual=20.0, flow_only=False):
            super().__init__()
            self.num_classes = num_classes
            self.output_channels = output_channels
            self.use_coord_grid = use_coord_grid
            self.use_diff_features = use_diff_features
            self.max_flow_px = max_flow_px
            self.max_residual = max_residual
            self.flow_only = flow_only
            self.embedding = embedding if embedding is not None else nn.Embedding(num_classes, embed_dim)
            in_ch = embed_dim * 2
            if use_diff_features:
                in_ch += embed_dim
            if use_coord_grid:
                in_ch += 2
            # U-Net-like structure for global receptive field (Quantizr TinyMotionFromMasks)
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 3, padding=1, bias=True),
                nn.SiLU(inplace=True),
            )
            self.down = nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, bias=True),
                nn.SiLU(inplace=True),
            )
            self.bottleneck = ResBlock(hidden, num_classes=0)
            self.up_conv = nn.Conv2d(hidden, hidden, 3, padding=1, bias=True)
            self.up_act = nn.SiLU(inplace=True)
            self.fuse = nn.Conv2d(hidden * 2, hidden, 1, bias=True)
            self.head = nn.Conv2d(hidden, output_channels, 3, padding=1, bias=True)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            # Gate channel bias -2.0 → sigmoid(-2)=0.12 (trust warp, not residual)
            if output_channels >= 3:
                with torch.no_grad():
                    self.head.bias[2] = -2.0

        def forward(self, mask_t: torch.Tensor, mask_t1: torch.Tensor) -> torch.Tensor:
            e_t = self.embedding(mask_t).permute(0, 3, 1, 2)
            e_t1 = self.embedding(mask_t1).permute(0, 3, 1, 2)
            parts = [e_t, e_t1]
            if self.use_diff_features:
                parts.append((e_t1 - e_t).abs())
            if self.use_coord_grid:
                B, _, H, W = e_t.shape
                gy = torch.linspace(-1, 1, H, device=e_t.device, dtype=e_t.dtype)
                gx = torch.linspace(-1, 1, W, device=e_t.device, dtype=e_t.dtype)
                grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
                coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
                parts.append(coords)
            x = torch.cat(parts, dim=1)
            # U-Net forward
            stem_feat = self.stem(x)
            down_feat = self.down(stem_feat)
            bot_feat = self.bottleneck(down_feat)
            up_feat = F.interpolate(bot_feat, size=stem_feat.shape[2:], mode="bilinear", align_corners=False)
            up_feat = self.up_act(self.up_conv(up_feat))
            fused = self.fuse(torch.cat([stem_feat, up_feat], dim=1))
            raw = self.head(fused)
            if self.output_channels == 2:
                return raw * 0.1
            else:
                flow = raw[:, :2].tanh() * (self.max_flow_px / max(mask_t.shape[-2], mask_t.shape[-1]) * 2)
                if self.flow_only:
                    gate = torch.zeros_like(raw[:, 2:3])
                    residual = torch.zeros_like(raw[:, 3:6])
                else:
                    gate = raw[:, 2:3].sigmoid()
                    residual = raw[:, 3:6].tanh() * self.max_residual
                return torch.cat([flow, gate, residual], dim=1)

    class AsymmetricPairGenerator(nn.Module):
        def __init__(self, num_classes=5, embed_dim=6, base_ch=36, mid_ch=60,
                     motion_hidden=32, depth=1, max_flow_px=20.0,
                     max_residual=20.0, flow_only=False):
            super().__init__()
            shared_emb = nn.Embedding(num_classes, embed_dim)
            self.renderer = MaskRenderer(
                num_classes=num_classes, embed_dim=embed_dim,
                base_ch=base_ch, mid_ch=mid_ch,
                embedding=shared_emb, depth=depth,
            )
            self.motion = MotionPredictor(
                num_classes=num_classes, embed_dim=embed_dim,
                hidden=motion_hidden, embedding=shared_emb,
                output_channels=6, use_coord_grid=True, use_diff_features=True,
                max_flow_px=max_flow_px, max_residual=max_residual,
                flow_only=flow_only,
            )

        def forward(self, mask_t: torch.Tensor, mask_t1: torch.Tensor) -> torch.Tensor:
            frame_t1 = self.renderer(mask_t1)
            motion_out = self.motion(mask_t, mask_t1)
            flow = motion_out[:, :2]
            gate = motion_out[:, 2:3]
            residual = motion_out[:, 3:6]
            warped_t1 = warp_with_flow(frame_t1, flow)
            frame_t = (warped_t1 + gate * residual).clamp(0.0, 255.0)
            pair = torch.stack([frame_t, frame_t1], dim=1)
            return pair.permute(0, 1, 3, 4, 2).contiguous()


# ============================================================
# Upstream discovery
# ============================================================
def _find_upstream_root(archive_dir: str) -> Path:
    """Locate the upstream directory containing modules.py and models/.

    Search order:
        1. archive_dir/../../  (scorer environment: archive/ is 2 levels deep)
        2. <script_dir>/../../upstream/  (local dev layout)
        3. UPSTREAM_ROOT / TAC_UPSTREAM_DIR / COMMA_CHALLENGE_ROOT env vars
    """
    candidates = []

    # 1. Scorer environment layout
    candidates.append(Path(archive_dir).resolve().parent.parent)

    # 2. Local dev layout
    candidates.append(Path(__file__).resolve().parent.parent.parent / "upstream")

    # 3. Environment variables (check all known conventions)
    for env_var in ("UPSTREAM_ROOT", "TAC_UPSTREAM_DIR", "COMMA_CHALLENGE_ROOT"):
        env_val = os.environ.get(env_var)
        if env_val:
            candidates.append(Path(env_val))

    for candidate in candidates:
        if not candidate.exists():
            continue
        modules_py = candidate / "modules.py"
        models_dir = candidate / "models"
        if modules_py.exists() and models_dir.exists():
            return candidate

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Cannot find upstream root (need modules.py + models/ dir).\n"
        f"Tried:\n  {tried}\n"
        f"Set UPSTREAM_ROOT, TAC_UPSTREAM_DIR, or COMMA_CHALLENGE_ROOT env var."
    )


# ============================================================
# Mask loading from archive (contest-compliant path)
# ============================================================
def _load_masks_from_archive(
    mask_video_path: Path,
    expected_frames: int = NUM_FRAMES,
) -> torch.Tensor:
    """Load pre-extracted masks from AV1 monochrome video in archive.

    This is the contest-compliant path: masks were pre-extracted at compress
    time by compress_masks.py, so no SegNet loading is needed at inflate time.

    The AV1 video uses 5-class grayscale encoding:
        pixel_value = class_label * (255 // 4)
    Decoding inverts this with rounding to handle lossy compression artifacts.

    Args:
        mask_video_path: path to masks.mkv inside archive directory
        expected_frames: expected number of frames (default: 1200)

    Returns:
        (N, SEGNET_H, SEGNET_W) long tensor with values in [0, 4]
    """
    import subprocess

    t0 = time.monotonic()

    if not mask_video_path.exists():
        raise FileNotFoundError(
            f"Pre-extracted mask video not found: {mask_video_path}\n"
            f"Run compress_masks.py at compress time to generate masks.mkv, "
            f"or set INFLATE_MASK_SOURCE=segnet to fall back to SegNet "
            f"(not contest-compliant)."
        )

    # Probe video dimensions
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(mask_video_path),
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {mask_video_path}: {probe.stderr}")

    parts = probe.stdout.strip().split(",")
    W, H = int(parts[0]), int(parts[1])

    # Decode to raw gray frames
    cmd = [
        "ffmpeg",
        "-i", str(mask_video_path),
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-v", "error",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg mask decoding failed:\n"
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )

    # Note: proc.stdout buffers the entire decoded video in memory.
    # At 1200 frames x 48x64 = ~3.5 MB, this is fine.  At full 384x512
    # it would be ~235 MB.  For production at full resolution, consider
    # Popen with streaming reads instead of capture_output.
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_size = H * W
    N = len(raw) // frame_size
    if len(raw) % frame_size != 0:
        raise ValueError(
            f"Decoded data size {len(raw)} not divisible by "
            f"frame size {H}x{W}={frame_size}"
        )

    pixels = raw.reshape(N, H, W)

    # Invert scaling: pixel -> class label
    # Encoding used: class * (255 // 4) -> 0, 63, 127, 191, 255
    scale_factor = 255 // (NUM_CLASSES - 1)
    masks = np.round(pixels.astype(np.float32) / scale_factor).astype(np.int64)
    masks = np.clip(masks, 0, NUM_CLASSES - 1)

    result = torch.from_numpy(masks)

    if expected_frames is not None and N != expected_frames:
        print(
            f"  WARNING: expected {expected_frames} mask frames, got {N}",
            file=sys.stderr,
        )

    elapsed = time.monotonic() - t0
    print(
        f"  Loaded {N} pre-extracted masks ({H}x{W}) from {mask_video_path} "
        f"({elapsed:.1f}s)",
        file=sys.stderr,
    )
    return result


# ============================================================
# SegNet loading (fallback for development, NOT contest-compliant)
# ============================================================
def _load_segnet(upstream_root: Path, device: str) -> nn.Module:
    """Load frozen SegNet from upstream for mask extraction.

    WARNING: This path is NOT contest-compliant. It loads SegNet (~48MB)
    from the upstream models/ directory, which per Yousfi's PR #35 rule
    would need to be included in the archive. Use pre-extracted masks
    (masks.mkv in archive) for contest submissions.
    """
    t0 = time.monotonic()

    # Import SegNet from upstream modules.py
    upstream_str = str(upstream_root)
    sys.path.insert(0, upstream_str)
    try:
        from modules import SegNet
    finally:
        # Remove the exact entry we inserted at position 0
        try:
            sys.path.pop(sys.path.index(upstream_str))
        except ValueError:
            pass  # already removed

    segnet = SegNet()
    segnet_path = upstream_root / "models" / "segnet.safetensors"
    if not segnet_path.exists():
        raise FileNotFoundError(f"SegNet weights not found: {segnet_path}")

    from safetensors.torch import load_file
    sd = load_file(str(segnet_path), device=device)
    segnet.load_state_dict(sd)
    segnet.to(device).eval()

    # Freeze all parameters
    for p in segnet.parameters():
        p.requires_grad = False

    elapsed = time.monotonic() - t0
    print(f"  SegNet loaded from {segnet_path} ({elapsed:.1f}s)", file=sys.stderr)
    return segnet


# ============================================================
# GT video decoding
# ============================================================
def _decode_gt_video(mkv_path: str) -> list[np.ndarray]:
    """Decode ground-truth video via PyAV.

    Returns list of (H, W, 3) uint8ndarrays in RGB order.
    Uses yuv420_to_rgb for BT.601 limited-range decode matching the scorer.
    """
    t0 = time.monotonic()
    container = av.open(mkv_path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        rgb = yuv420_to_rgb(frame)  # (H, W, 3) uint8 tensor
        frames.append(rgb.numpy())
    container.close()
    elapsed = time.monotonic() - t0
    print(f"  Decoded {len(frames)} GT frames from {mkv_path} ({elapsed:.1f}s)", file=sys.stderr)
    return frames


# ============================================================
# Mask extraction
# ============================================================
def _extract_masks(
    frames: list[np.ndarray],
    segnet: nn.Module,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Extract SegNet masks from GT frames.

    Args:
        frames: list of (H, W, 3) uint8 ndarrays
        segnet: frozen SegNet module
        device: torch device string
        batch_size: inference batch size

    Returns:
        (N, 384, 512) long tensor of class indices in [0, 4]
    """
    t0 = time.monotonic()
    N = len(frames)
    masks_list = []

    with torch.inference_mode():
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            # Stack frames -> (B, H, W, 3) uint8 -> (B, 3, H, W) float
            batch_np = np.stack(frames[i:end], axis=0)  # (B, H, W, 3)
            batch_t = torch.from_numpy(batch_np).float().permute(0, 3, 1, 2).to(device)
            # SegNet expects (B, 1, 3, H, W) for preprocess_input
            inp = batch_t.unsqueeze(1)  # (B, 1, 3, H, W)
            seg_in = segnet.preprocess_input(inp)  # (B, 3, 384, 512)
            logits = segnet(seg_in)  # (B, 5, 384, 512)
            mask = logits.argmax(dim=1)  # (B, 384, 512)
            # Store as int8 — values are [0,4], saves ~7x RAM vs int64
            masks_list.append(mask.to(torch.int8).cpu())

            if (i + batch_size) % (batch_size * 10) == 0 or end == N:
                print(f"    Masks: {end}/{N} frames", file=sys.stderr, flush=True)

    masks = torch.cat(masks_list, dim=0)  # (N, 384, 512) int8
    elapsed = time.monotonic() - t0
    print(f"  Extracted {masks.shape[0]} masks ({elapsed:.1f}s)", file=sys.stderr)
    return masks


# ============================================================
# Inline .bin deserializer (Contrarian: standalone on scorer machines)
# ============================================================
def _inline_unpack_values(data, offset, count, bits):
    """Unpack `count` values at `bits` per value from data starting at offset."""
    if bits == 8:
        values = [data[offset + i] for i in range(count)]
        return values, offset + count
    total_bits = count * bits
    total_bytes = (total_bits + 7) // 8
    if count > 10_000_000:
        raise ValueError(f"Implausible value count={count:,} — possible malformed .bin")
    raw = data[offset:offset + total_bytes]
    bit_buffer = int.from_bytes(bytes(raw), byteorder="little")
    mask = (1 << bits) - 1
    values = []
    for _ in range(count):
        values.append(bit_buffer & mask)
        bit_buffer >>= bits
    return values, offset + total_bytes


def _inline_dequantize_values(values, bits, scale):
    """Dequantize unsigned integer values back to float tensor."""
    bits = max(bits, 2)
    n_levels = 2 ** bits
    half = n_levels // 2
    return torch.tensor(
        [(v - half) / max(half - 1, 1) * scale for v in values],
        dtype=torch.float32,
    )


def _inline_load_asym(raw_bytes: bytes, device: str = "cpu") -> nn.Module:
    """Inline ASYM .bin deserializer — no tac dependency required.

    Reads ASYM header → parses JSON config → reconstructs AsymmetricPairGenerator
    → loads quantized weights from blobs.
    """
    import struct

    offset = 0

    # Verify magic
    if raw_bytes[offset:offset + 4] != b"ASYM":
        raise ValueError(f"Not an ASYM binary (got {raw_bytes[:4]!r})")
    offset += 4

    # Read header
    header_len = struct.unpack("<I", raw_bytes[offset:offset + 4])[0]
    offset += 4
    header = json.loads(raw_bytes[offset:offset + header_len].decode("utf-8"))
    offset += header_len

    version = header.get("version", 0)
    if version != 2:
        raise ValueError(f"Unsupported ASYM export version {version} (expected 2)")

    # Build fresh AsymmetricPairGenerator from header config
    model = AsymmetricPairGenerator(
        num_classes=header.get("num_classes", 5),
        embed_dim=header.get("embed_dim", 6),
        base_ch=header.get("base_ch", 36),
        mid_ch=header.get("mid_ch", 60),
        motion_hidden=header.get("motion_hidden", 32),
        depth=header.get("depth", 1),
        max_flow_px=header.get("max_flow_px", 20.0),
        max_residual=header.get("max_residual", 20.0),
        flow_only=header.get("flow_only", False),
    )

    # Build name → module lookups
    embedding_lookup = {}
    conv_lookup = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_lookup[name] = module
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            conv_lookup[name] = module

    # Iterate layers in header order and restore weights
    for layer_meta in header["layers"]:
        name = layer_meta["name"]
        is_embedding = layer_meta.get("is_embedding", False)

        # Read weight blob
        blob_len = struct.unpack("<I", raw_bytes[offset:offset + 4])[0]
        offset += 4
        weight_data = raw_bytes[offset:offset + blob_len]
        offset += blob_len

        if is_embedding:
            shape = layer_meta["shape"]
            bits = layer_meta["bits"]
            count = 1
            for s in shape:
                count *= s
            w_offset = 0
            scale = struct.unpack("<e", weight_data[w_offset:w_offset + 2])[0]
            w_offset += 2
            values, w_offset = _inline_unpack_values(weight_data, w_offset, count, bits)
            emb_tensor = _inline_dequantize_values(values, bits, scale).reshape(shape)
            with torch.no_grad():
                embedding_lookup[name].weight.copy_(emb_tensor)
            continue

        # Read bias blob
        has_bias = layer_meta["has_bias"]
        bias_blob_len = struct.unpack("<I", raw_bytes[offset:offset + 4])[0]
        offset += 4
        bias_data = raw_bytes[offset:offset + bias_blob_len]
        offset += bias_blob_len

        module = conv_lookup[name]
        shape = layer_meta["shape"]
        transposed = layer_meta.get("transposed", False)
        bits = layer_meta["bits"]

        if transposed:
            C_out = shape[1]
            fan_in = shape[0] * shape[2] * shape[3]
            ch_shape = [shape[0]] + shape[2:]
        else:
            C_out = shape[0]
            fan_in = 1
            for s in shape[1:]:
                fan_in *= s
            ch_shape = shape[1:]

        with torch.no_grad():
            module.weight.zero_()
            if module.bias is not None:
                module.bias.zero_()

            w_offset = 0
            for ch_idx in range(C_out):
                scale = struct.unpack("<e", weight_data[w_offset:w_offset + 2])[0]
                w_offset += 2
                values, w_offset = _inline_unpack_values(weight_data, w_offset, fan_in, bits)
                dequant = _inline_dequantize_values(values, bits, scale)
                if transposed:
                    module.weight[:, ch_idx] = dequant.reshape(ch_shape)
                else:
                    module.weight[ch_idx] = dequant.reshape(ch_shape)

            if has_bias and bias_data:
                b_offset = 0
                for ch_idx in range(C_out):
                    scale_b = struct.unpack("<e", bias_data[b_offset:b_offset + 2])[0]
                    b_offset += 2
                    u_val = struct.unpack("<H", bias_data[b_offset:b_offset + 2])[0]
                    b_offset += 2
                    n_levels = 2 ** bits
                    half = n_levels // 2
                    q = u_val - half
                    module.bias[ch_idx] = q / max(half - 1, 1) * scale_b

    # Restore scalar parameters
    scalar_params = header.get("scalar_params", {})
    if scalar_params:
        param_dict = dict(model.named_parameters())
        with torch.no_grad():
            for pname, pval in scalar_params.items():
                if pname in param_dict:
                    param_dict[pname].fill_(pval)

    # Verify all data consumed
    if offset != len(raw_bytes):
        raise ValueError(f"Trailing data: {len(raw_bytes) - offset} bytes unread (expected 0)")

    model = model.to(device)
    model.eval()
    return model


# ============================================================
# Renderer loading
# ============================================================
def _load_renderer(renderer_path: str, device: str) -> nn.Module:
    """Load renderer from a .bin or .pt checkpoint.

    Supports three checkpoint formats:
        1. DPSM binary: DPSIMSRenderer (magic b"DPSM")
        2. ASYM binary: AsymmetricPairGenerator (magic b"ASYM")
        3. PyTorch pickle: state_dict or PairGenerator checkpoint

    Config metadata is read from the checkpoint's header/config key.
    """
    t0 = time.monotonic()
    renderer_path = Path(renderer_path)
    raw_bytes = renderer_path.read_bytes()

    magic = raw_bytes[:4]

    # ── ASYM format: AsymmetricPairGenerator ──
    if magic == b"ASYM":
        try:
            from tac.renderer_export import load_asymmetric_checkpoint
            model = load_asymmetric_checkpoint(raw_bytes, device=device)
        except ImportError:
            model = _inline_load_asym(raw_bytes, device=device)
        elapsed = time.monotonic() - t0
        print(f"  Loaded AsymmetricPairGenerator from .bin ({len(raw_bytes):,} bytes, {elapsed:.1f}s)",
              file=sys.stderr)
        return model

    # ── DPSM format: DPSIMSRenderer ──
    if magic == b"DPSM":
        try:
            from tac.renderer_export import load_renderer_checkpoint
        except ImportError:
            import struct as _struct
            header_len = _struct.unpack("<I", raw_bytes[4:8])[0]
            header = json.loads(raw_bytes[8:8 + header_len].decode("utf-8"))
            raise RuntimeError(
                f"DPSM .bin format detected (version={header.get('version')}), "
                f"but tac.renderer_export is not available. Install the tac package "
                f"or use a .pt checkpoint instead."
            )
        renderer = load_renderer_checkpoint(raw_bytes, device=device)
        elapsed = time.monotonic() - t0
        print(f"  Loaded renderer from .bin format ({len(raw_bytes):,} bytes, {elapsed:.1f}s)",
              file=sys.stderr)
        return renderer

    # PyTorch pickle format (.pt checkpoint from training)
    ckpt = torch.load(renderer_path, map_location=device, weights_only=False)

    # Extract config for architecture reconstruction
    config = ckpt.get("config", {})
    pair_mode = config.get("pair_mode", "dp_sims")

    # Determine which state_dict to use
    if "model_state_dict" in ckpt:
        raw_sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        raw_sd = ckpt["state_dict"]
    else:
        raw_sd = ckpt

    if pair_mode == "asymmetric":
        # Asymmetric warp checkpoint — build AsymmetricPairGenerator
        print(f"  Detected pair_mode=asymmetric in .pt checkpoint", file=sys.stderr)
        renderer = AsymmetricPairGenerator(
            num_classes=config.get("num_classes", 5),
            embed_dim=config.get("embed_dim", 6),
            base_ch=config.get("base_ch", 36),
            mid_ch=config.get("mid_ch", 60),
            motion_hidden=config.get("motion_hidden", 32),
            depth=config.get("renderer_depth", 1),
            max_flow_px=config.get("max_flow_px", 20.0),
            max_residual=config.get("max_residual", 20.0),
            flow_only=config.get("flow_only", False),
        )
        renderer.load_state_dict(raw_sd, strict=True)
        renderer.to(device).eval()
    else:
        # DP-SIMS checkpoint — build DPSIMSRenderer
        num_classes = config.get("num_classes", 5)
        channels = config.get("channels", (256, 128, 64, 32))
        if isinstance(channels, list):
            channels = tuple(channels)
        init_h = config.get("init_h", 24)
        init_w = config.get("init_w", 32)
        spade_hidden = config.get("spade_hidden", 64)
        noise_dim = config.get("noise_dim", 16)
        use_noise = config.get("use_noise", True)

        print(f"  Renderer config: classes={num_classes}, channels={channels}, "
              f"init={init_h}x{init_w}, spade_hidden={spade_hidden}, "
              f"noise={use_noise}", file=sys.stderr)

        renderer = DPSIMSRenderer(
            num_classes=num_classes,
            channels=channels,
            init_h=init_h,
            init_w=init_w,
            spade_hidden=spade_hidden,
            noise_dim=noise_dim,
            use_noise=use_noise,
        )

        # Check if keys are prefixed with "renderer." (from DPSIMSPairGenerator)
        renderer_prefix = "renderer."
        has_prefix = any(k.startswith(renderer_prefix) for k in raw_sd.keys())
        if has_prefix:
            sd = {k[len(renderer_prefix):]: v for k, v in raw_sd.items() if k.startswith(renderer_prefix)}
            print(f"  Extracted {len(sd)} renderer keys from PairGenerator checkpoint", file=sys.stderr)
        else:
            sd = raw_sd

        renderer.load_state_dict(sd, strict=True)
        renderer.to(device).eval()

    # Freeze all parameters
    for p in renderer.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in renderer.parameters())
    elapsed = time.monotonic() - t0
    print(f"  Renderer loaded: {n_params:,} params ({elapsed:.1f}s)", file=sys.stderr)
    return renderer


# ============================================================
# Frame generation + write
# ============================================================
def _is_asymmetric_model(model: nn.Module) -> bool:
    """Detect whether a loaded model is an AsymmetricPairGenerator."""
    return (
        type(model).__name__ == "AsymmetricPairGenerator"
        or (hasattr(model, "renderer") and hasattr(model, "motion")
            and hasattr(model.motion, "output_channels")
            and getattr(model.motion, "output_channels", 2) == 6)
    )


def _generate_and_write(
    masks: torch.Tensor,
    renderer: nn.Module,
    output_path: str,
    device: str,
    batch_size: int,
    out_h: int = OUT_H,
    out_w: int = OUT_W,
) -> int:
    """Generate frames from masks via renderer, upscale, and write raw RGB.

    Supports two model types:
        1. DPSIMSRenderer: independent frame generation (renderer(masks) -> (B,3,H,W))
        2. AsymmetricPairGenerator: pair generation from consecutive mask pairs
           model(mask_t, mask_t1) -> (B, 2, H, W, 3) HWC pair

    For asymmetric mode, masks are processed in consecutive pairs:
        (mask[0], mask[1]) -> (frame[0], frame[1])
        (mask[2], mask[3]) -> (frame[2], frame[3])
        ...

    Args:
        masks: (N, 384, 512) long tensor
        renderer: DPSIMSRenderer or AsymmetricPairGenerator
        output_path: path to output .raw file
        device: torch device string
        batch_size: inference batch size
        out_h: output frame height
        out_w: output frame width

    Returns:
        Number of frames written
    """
    t0 = time.monotonic()
    N = masks.shape[0]
    n_written = 0
    is_asymmetric = _is_asymmetric_model(renderer)

    # Deterministic seed for reproducible output (noise injectors use torch.randn)
    torch.manual_seed(42)

    if is_asymmetric:
        print(f"  Mode: asymmetric pair generation ({N} masks -> {N} frames "
              f"via {N // 2} pairs)", file=sys.stderr)
        if N % 2 != 0:
            print(f"  WARNING: odd number of masks ({N}), last mask will be "
                  f"rendered independently via renderer sub-module", file=sys.stderr)

        with open(output_path, 'wb') as f:
            with torch.inference_mode():
                # Process masks in consecutive pairs
                pair_idx = 0
                while pair_idx < N - 1:
                    # Build a batch of pairs
                    batch_pairs_t = []
                    batch_pairs_t1 = []
                    batch_end = min(pair_idx + batch_size * 2, N - 1)
                    for j in range(pair_idx, batch_end, 2):
                        if j + 1 < N:
                            batch_pairs_t.append(masks[j])
                            batch_pairs_t1.append(masks[j + 1])

                    if not batch_pairs_t:
                        break

                    masks_t = torch.stack(batch_pairs_t).to(device=device, dtype=torch.long)
                    masks_t1 = torch.stack(batch_pairs_t1).to(device=device, dtype=torch.long)

                    # Generate pairs: (B, 2, H, W, 3) HWC in [0, 255]
                    pairs = renderer(masks_t, masks_t1)  # (B, 2, H, W, 3)

                    # Upscale each frame in each pair
                    B_pairs = pairs.shape[0]
                    for p in range(B_pairs):
                        for frame_idx in range(2):  # frame_t then frame_t1
                            frame_hwc = pairs[p, frame_idx]  # (H, W, 3)
                            # Convert HWC -> CHW for interpolation
                            frame_chw = frame_hwc.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
                            frame_up = F.interpolate(
                                frame_chw, size=(out_h, out_w),
                                mode="bilinear", align_corners=False,
                            )  # (1, 3, out_h, out_w)
                            frame_uint8 = frame_up.round().clamp(0, 255).to(torch.uint8)
                            frame_out = frame_uint8.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
                            f.write(frame_out.tobytes())
                            n_written += 1

                    pair_idx += len(batch_pairs_t) * 2

                    if n_written % (batch_size * 10) == 0 or pair_idx >= N - 1:
                        print(f"    Generated: {n_written}/{N} frames",
                              file=sys.stderr, flush=True)

                # Handle odd trailing mask: render independently via the sub-renderer
                if N % 2 != 0:
                    last_mask = masks[N - 1:N].to(device=device, dtype=torch.long)
                    frame = renderer.renderer(last_mask)  # (1, 3, H, W)
                    frame_up = F.interpolate(
                        frame, size=(out_h, out_w),
                        mode="bilinear", align_corners=False,
                    )
                    frame_uint8 = frame_up.round().clamp(0, 255).to(torch.uint8)
                    frame_out = frame_uint8.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
                    f.write(frame_out.tobytes())
                    n_written += 1
                    print(f"    Generated trailing frame: {n_written}/{N}",
                          file=sys.stderr, flush=True)
    else:
        # Standard independent frame generation (DPSIMSRenderer path)
        with open(output_path, 'wb') as f:
            with torch.inference_mode():
                for i in range(0, N, batch_size):
                    end = min(i + batch_size, N)
                    batch_masks = masks[i:end].to(device=device, dtype=torch.long)

                    # Generate frames at SegNet resolution (384x512)
                    frames = renderer(batch_masks)  # (B, 3, 384, 512)

                    # Upscale to output resolution
                    frames_up = F.interpolate(
                        frames, size=(out_h, out_w),
                        mode="bilinear", align_corners=False,
                    )  # (B, 3, out_h, out_w)

                    # Quantize and write as HWC uint8
                    frames_uint8 = frames_up.round().clamp(0, 255).to(torch.uint8)
                    frames_hwc = frames_uint8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                    f.write(frames_hwc.tobytes())
                    n_written += batch_masks.shape[0]

                    if end % (batch_size * 10) == 0 or end == N:
                        print(f"    Generated: {end}/{N} frames",
                              file=sys.stderr, flush=True)

    elapsed = time.monotonic() - t0
    raw_size = os.path.getsize(output_path)
    print(f"  Generated {n_written} frames -> {output_path} "
          f"({raw_size:,} bytes, {elapsed:.1f}s)", file=sys.stderr)
    return n_written


# ============================================================
# Main inflate function
# ============================================================
def _detect_device_and_batch_size() -> tuple[str, int]:
    """Detect best available device and appropriate batch size.

    Returns:
        (device_string, batch_size) tuple.
    """
    if torch.cuda.is_available():
        device = "cuda"
        batch_size = 16
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})", file=sys.stderr)
    else:
        device = "cpu"
        batch_size = 4
        print(f"Device: CPU ({os.cpu_count()} cores)", file=sys.stderr)
    return device, batch_size


def _load_renderer_and_masks(
    archive_dir: str,
    device: str,
    renderer_filename: str = "renderer.bin",
    mask_filename: str = "masks.mkv",
) -> tuple:
    """Load renderer and masks from the archive directory.

    Shared loading logic used by both inflate_renderer() and
    inflate_renderer_with_tto() to avoid code duplication.

    Returns:
        (renderer, masks, mask_video_path) tuple.
    """
    renderer_path = Path(archive_dir) / renderer_filename
    if not renderer_path.exists():
        raise FileNotFoundError(
            f"Renderer not found: {renderer_path}\n"
            f"Expected {renderer_filename} inside archive directory."
        )
    renderer = _load_renderer(str(renderer_path), device)

    mask_video_path = Path(archive_dir) / mask_filename
    if not mask_video_path.exists():
        raise FileNotFoundError(f"Mask video not found: {mask_video_path}")
    masks = _load_masks_from_archive(mask_video_path)

    return renderer, masks, mask_video_path


def inflate_renderer(
    archive_dir: str,
    inflated_dir: str,
    video_names_file: str,
    renderer_filename: str = "renderer.bin",
    mask_filename: str = "masks.mkv",
    out_w: int = OUT_W,
    out_h: int = OUT_H,
) -> None:
    """Full inflate pipeline: archive masks -> renderer -> raw RGB.

    Contest-compliant path (default):
        archive/masks.mkv  ->  AV1 decode  ->  masks  ->  renderer  ->  raw RGB

    Development fallback (INFLATE_MASK_SOURCE=segnet):
        GT video  ->  SegNet (upstream)  ->  masks  ->  renderer  ->  raw RGB

    Args:
        archive_dir: directory containing renderer.bin and masks.mkv
        inflated_dir: output directory for .raw files
        video_names_file: text file listing video names (one per line)
        renderer_filename: renderer checkpoint filename within archive_dir
        mask_filename: mask video filename within archive_dir
        out_w: output frame width
        out_h: output frame height
    """
    t_total_start = time.monotonic()

    # ---- Device detection ----
    if torch.cuda.is_available():
        device = "cuda"
        batch_size = 16
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})", file=sys.stderr)
    else:
        device = "cpu"
        batch_size = 4
        print(f"Device: CPU ({os.cpu_count()} cores)", file=sys.stderr)

    # ---- Determine mask source ----
    mask_source = os.environ.get("INFLATE_MASK_SOURCE", "archive")
    mask_video_path = Path(archive_dir) / mask_filename

    # Auto-detect: if masks.mkv exists in archive, use it; otherwise fall back
    if mask_source == "archive" and not mask_video_path.exists():
        print(
            f"  WARNING: {mask_video_path} not found in archive. "
            f"Falling back to SegNet extraction (NOT contest-compliant).",
            file=sys.stderr,
        )
        mask_source = "segnet"

    use_archive_masks = mask_source == "archive"

    if use_archive_masks:
        print(
            "Mask source: archive (contest-compliant, no SegNet loading)",
            file=sys.stderr,
        )
    else:
        print(
            "Mask source: SegNet extraction (development mode, NOT contest-compliant)",
            file=sys.stderr,
        )

    # ---- Upstream discovery (needed for SegNet fallback and GT video) ----
    segnet = None
    upstream_root = None
    if not use_archive_masks:
        print("Stage 1: Discovering upstream environment ...", file=sys.stderr)
        upstream_root = _find_upstream_root(archive_dir)
        print(f"  Upstream root: {upstream_root}", file=sys.stderr)

        print("Stage 2: Loading SegNet (fallback mode) ...", file=sys.stderr)
        segnet = _load_segnet(upstream_root, device)
    else:
        # Still need upstream_root for GT video discovery in SegNet fallback
        # but for archive path we can try to find it (non-fatal if missing)
        try:
            upstream_root = _find_upstream_root(archive_dir)
        except FileNotFoundError:
            upstream_root = None

    # ---- Load renderer ----
    stage_num = 3 if not use_archive_masks else 1
    print(f"Stage {stage_num}: Loading renderer ...", file=sys.stderr)
    renderer_path = Path(archive_dir) / renderer_filename
    if not renderer_path.exists():
        raise FileNotFoundError(
            f"Renderer not found: {renderer_path}\n"
            f"Expected {renderer_filename} inside archive directory."
        )
    renderer = _load_renderer(str(renderer_path), device)

    # ---- Load masks from archive (contest-compliant path) ----
    masks = None
    if use_archive_masks:
        stage_num += 1
        print(f"Stage {stage_num}: Loading pre-extracted masks ...", file=sys.stderr)
        masks = _load_masks_from_archive(mask_video_path)

        # Verify mask resolution (accept clean downscale factors)
        mask_h, mask_w = masks.shape[1], masks.shape[2]
        if mask_h != SEG_H or mask_w != SEG_W:
            if SEG_H % mask_h == 0 and SEG_W % mask_w == 0:
                scale = SEG_H // mask_h
                print(
                    f"  Archive masks at {mask_h}x{mask_w} "
                    f"(1/{scale} of {SEG_H}x{SEG_W}). "
                    f"Renderer will upsample internally.",
                    file=sys.stderr,
                )
            else:
                raise ValueError(
                    f"Mask resolution {mask_h}x{mask_w} is not a clean "
                    f"downscale factor of {SEG_H}x{SEG_W}. Expected "
                    f"dimensions that evenly divide {SEG_H}x{SEG_W}."
                )

    # ---- Process each video ----
    output_path = Path(inflated_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_names = Path(video_names_file).read_text().splitlines()
    video_names = [v.strip() for v in video_names if v.strip()]

    for idx, rel in enumerate(video_names):
        t_video_start = time.monotonic()
        stem = rel.rsplit(".", 1)[0]
        raw_out = output_path / f"{stem}.raw"
        raw_out.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Video {idx+1}/{len(video_names)}: {rel}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        if use_archive_masks:
            # Contest-compliant path: masks already loaded from archive
            video_masks = masks
        else:
            # Development fallback: extract masks from GT video via SegNet
            gt_candidates = [
                Path(archive_dir).parent / rel,
                Path(archive_dir).parent.parent / "data" / rel,
            ]
            if upstream_root:
                gt_candidates.extend([
                    upstream_root / "data" / rel,
                    upstream_root / "videos" / rel,
                ])
            data_dir = os.environ.get("COMMA_DATA_DIR")
            if data_dir:
                gt_candidates.insert(0, Path(data_dir) / rel)

            gt_path = None
            for candidate in gt_candidates:
                if candidate.exists():
                    gt_path = candidate
                    break

            if gt_path is None:
                tried = "\n  ".join(str(c) for c in gt_candidates)
                raise FileNotFoundError(
                    f"GT video not found for {rel}.\nTried:\n  {tried}\n"
                    f"Set COMMA_DATA_DIR env var to the directory containing GT videos."
                )

            print(f"  GT video: {gt_path}", file=sys.stderr)
            print("  Decoding GT video ...", file=sys.stderr)
            gt_frames = _decode_gt_video(str(gt_path))

            if len(gt_frames) != NUM_FRAMES:
                print(
                    f"  WARNING: expected {NUM_FRAMES} frames, got {len(gt_frames)}",
                    file=sys.stderr,
                )

            print("  Extracting SegNet masks ...", file=sys.stderr)
            video_masks = _extract_masks(gt_frames, segnet, device, batch_size)
            del gt_frames

        # Verify mask resolution (may be downscaled for rate savings)
        mask_h, mask_w = video_masks.shape[1], video_masks.shape[2]
        if mask_h != SEG_H or mask_w != SEG_W:
            # Accept clean downscale factors (e.g. 48x64 = 384/8 x 512/8)
            if SEG_H % mask_h == 0 and SEG_W % mask_w == 0:
                scale = SEG_H // mask_h
                print(
                    f"  Masks at {mask_h}x{mask_w} "
                    f"(1/{scale} of {SEG_H}x{SEG_W}). "
                    f"Renderer will upsample via nearest-neighbor internally.",
                    file=sys.stderr,
                )
            else:
                raise ValueError(
                    f"Mask resolution {mask_h}x{mask_w} is not a clean "
                    f"downscale factor of {SEG_H}x{SEG_W}. This would "
                    f"produce interpolation artifacts. Expected dimensions "
                    f"that evenly divide {SEG_H}x{SEG_W}."
                )

        # Generate and write
        gen_stage = "Stage 3" if use_archive_masks else "Stage 6"
        print(f"{gen_stage}: Generating frames via renderer ...", file=sys.stderr)
        n_written = _generate_and_write(
            video_masks, renderer, str(raw_out), device, batch_size, out_h, out_w
        )

        if not use_archive_masks:
            del video_masks

        # Verify output
        actual_size = os.path.getsize(str(raw_out))
        expected_size = out_w * out_h * 3 * n_written
        if actual_size != expected_size:
            raise RuntimeError(
                f"Output size mismatch: {actual_size:,} != expected {expected_size:,} "
                f"({n_written} frames x {out_h}x{out_w}x3). Corrupt output."
            )

        t_video_elapsed = time.monotonic() - t_video_start
        print(f"  Video complete: {n_written} frames in {t_video_elapsed:.1f}s "
              f"({n_written / max(t_video_elapsed, 0.01):.1f} fps)",
              file=sys.stderr)

    t_total = time.monotonic() - t_total_start
    print(f"\nTotal inflate time: {t_total:.1f}s", file=sys.stderr)


# ============================================================
# Adaptive TTO at inflate time (EXPERIMENTAL, not default)
# ============================================================
# Council vote: 8-1, NOT assumed contest-compliant. Requires
# compliance ruling before use in official submission.
#
# Environment variables:
#   INFLATE_TTO=0          (default) renderer only, fully compliant
#   INFLATE_TTO=1          renderer + adaptive TTO on hardest pairs
#   INFLATE_TTO_BUDGET_SECONDS=1300  time budget for TTO phase
#   INFLATE_TTO_STEPS=100  gradient steps per pair batch
#   INFLATE_TTO_TOP_K=0.3  fraction of pairs to TTO (worst 30%)
#   INFLATE_TTO_LR=0.005   TTO learning rate
#   INFLATE_TTO_BATCH_PAIRS=10  pairs per optimization batch (VRAM)
#   INFLATE_TTO_SEG_WEIGHT=100.0  SegNet loss weight
#   INFLATE_TTO_POSE_WEIGHT=10.0  PoseNet loss weight
# ============================================================

def _compute_per_pair_posenet_distortion(
    renderer_frames: torch.Tensor,
    gt_frames: list[torch.Tensor],
    posenet: nn.Module,
    device: str,
    batch_size: int = 16,
) -> torch.Tensor:
    """Compute PoseNet distortion for each non-overlapping pair.

    PoseNet evaluates consecutive frame pairs: (frame[2k], frame[2k+1]).
    This function returns a (P,) tensor of per-pair distortions, where
    P = N // 2 and higher values indicate harder pairs.

    Args:
        renderer_frames: (N, H, W, 3) float [0, 255] rendered frames.
        gt_frames: list of N (H, W, 3) uint8 tensors (ground truth).
        posenet: frozen PoseNet scorer.
        device: computation device string.
        batch_size: pairs per forward pass.

    Returns:
        (P,) float tensor of per-pair PoseNet distortions.
    """
    N = renderer_frames.shape[0]
    P = N // 2

    # We need to import camera constants for resolution matching
    try:
        from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W
    except ImportError:
        SEGNET_INPUT_H, SEGNET_INPUT_W = 384, 512

    pair_dists = torch.zeros(P)

    with torch.inference_mode():
        for start in range(0, P, batch_size):
            end = min(start + batch_size, P)
            B = end - start

            # Build rendered pairs: (B, 2, H, W, 3) -> posenet input
            rendered_pairs = []
            gt_pairs = []
            for k in range(start, end):
                r0 = renderer_frames[2 * k].float()
                r1 = renderer_frames[2 * k + 1].float()
                rendered_pairs.append(torch.stack([r0, r1], dim=0))

                g0 = torch.as_tensor(gt_frames[2 * k]).float()
                g1 = torch.as_tensor(gt_frames[2 * k + 1]).float()
                gt_pairs.append(torch.stack([g0, g1], dim=0))

            rendered_batch = torch.stack(rendered_pairs).to(device)  # (B, 2, H, W, 3)
            gt_batch = torch.stack(gt_pairs).to(device)

            # Convert HWC -> CHW for PoseNet
            rendered_chw = rendered_batch.permute(0, 1, 4, 2, 3).contiguous()
            gt_chw = gt_batch.permute(0, 1, 4, 2, 3).contiguous()

            # Resize to scorer resolution if needed
            _, _, C, H, W = rendered_chw.shape
            if H != SEGNET_INPUT_H or W != SEGNET_INPUT_W:
                rendered_flat = rendered_chw.reshape(B * 2, C, H, W)
                rendered_flat = F.interpolate(
                    rendered_flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                    mode="bilinear", align_corners=False,
                )
                rendered_chw = rendered_flat.reshape(B, 2, C, SEGNET_INPUT_H, SEGNET_INPUT_W)

                gt_flat = gt_chw.reshape(B * 2, C, H, W)
                gt_flat = F.interpolate(
                    gt_flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                    mode="bilinear", align_corners=False,
                )
                gt_chw = gt_flat.reshape(B, 2, C, SEGNET_INPUT_H, SEGNET_INPUT_W)

            # PoseNet forward
            gt_in = posenet.preprocess_input(gt_chw)
            rendered_in = posenet.preprocess_input(rendered_chw)

            gt_pose = posenet(gt_in)      # (B, 6)
            rendered_pose = posenet(rendered_in)  # (B, 6)

            # Per-pair MSE distortion
            dist = ((gt_pose - rendered_pose) ** 2).mean(dim=1)  # (B,)
            pair_dists[start:end] = dist.cpu()

    return pair_dists


def _adaptive_tto_phase(
    renderer_frames: torch.Tensor,
    masks: torch.Tensor,
    gt_frames: list[torch.Tensor],
    posenet: nn.Module,
    segnet: nn.Module,
    device: str,
    budget_seconds: float = 1300.0,
    tto_steps: int = 100,
    top_k_fraction: float = 0.3,
    tto_lr: float = 0.005,
    batch_pairs: int = 10,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
) -> torch.Tensor:
    """Adaptive TTO: refine the hardest pairs within a time budget.

    Strategy:
        1. Compute per-pair PoseNet distortion (renderer output vs GT).
        2. Sort pairs by distortion (hardest first).
        3. TTO the hardest top_k fraction, stopping when budget exhausted.
        4. Return mixed output: TTO-refined for hard pairs, renderer-only
           for easy pairs.

    COMPLIANCE WARNING: This loads PoseNet+SegNet at inflate time for
    gradient-based optimization. Per council 8-1 vote, this is NOT
    assumed contest-compliant. Requires explicit compliance ruling.

    Args:
        renderer_frames: (N, H, W, 3) float [0, 255] renderer output.
        masks: (N, H, W) long tensor of segmentation masks.
        gt_frames: list of N (H, W, 3) uint8 tensors (ground truth).
        posenet: frozen PoseNet scorer (loaded from upstream).
        segnet: frozen SegNet scorer (loaded from upstream).
        device: computation device string.
        budget_seconds: total wall-clock budget for the TTO phase.
        tto_steps: gradient steps per pair batch.
        top_k_fraction: fraction of pairs to TTO (0.3 = worst 30%).
        tto_lr: Adam learning rate for TTO.
        batch_pairs: pairs per optimization batch (VRAM constrained).
        seg_weight: SegNet loss weight in TTO objective.
        pose_weight: PoseNet loss weight in TTO objective.

    Returns:
        (N, H, W, 3) float tensor of refined frames in [0, 255].
    """
    try:
        from tac.constrained_gen import coupled_trajectory_optimize
        from tac.scorer import extract_gt_pose_targets
    except ImportError as e:
        print(f"  WARNING: tac package not available for TTO ({e}). "
              f"Returning renderer-only output.", file=sys.stderr)
        return renderer_frames

    t_phase_start = time.monotonic()
    N = renderer_frames.shape[0]
    P = N // 2

    # Step 0: Gradient sanity check — verify gradients flow before committing
    # to TTO. If upstream rgb_to_yuv6 or similar has @torch.no_grad, all TTO
    # steps would produce zero PoseNet gradients (the "great gradient bug").
    print("  [TTO] Gradient sanity check...", file=sys.stderr)
    try:
        test_pair = renderer_frames[:2].clone().to(device).requires_grad_(True)
        # SegNet path: (B, 1, C, H, W) -> preprocess -> forward
        seg_in = segnet.preprocess_input(
            test_pair.permute(0, 3, 1, 2).unsqueeze(1).float()
        )
        seg_out = segnet(seg_in)
        seg_loss = seg_out.sum()

        # PoseNet path: (1, 2, C, H, W) -> preprocess -> forward
        # PoseNet expects consecutive frame pairs as the T dimension
        pose_in_chw = test_pair.permute(0, 3, 1, 2).float()  # (2, C, H, W)
        pose_in = pose_in_chw.unsqueeze(0)  # (1, 2, C, H, W)
        pose_preprocessed = posenet.preprocess_input(pose_in)  # (1, 12, H/2, W/2)
        pose_out = posenet(pose_preprocessed)  # (1, 6)
        pose_loss = pose_out.sum()

        total = seg_loss + pose_loss
        total.backward()

        grad_norm = test_pair.grad.norm().item() if test_pair.grad is not None else 0.0
        if grad_norm < 1e-12:
            print(
                f"  [TTO] ERROR: Dead gradients detected (grad_norm={grad_norm:.2e}). "
                f"Skipping TTO, returning renderer-only output.",
                file=sys.stderr,
            )
            return renderer_frames
        print(f"  [TTO] Gradient check PASSED (grad_norm={grad_norm:.4e})",
              file=sys.stderr)
        del test_pair, seg_in, seg_out, pose_in, pose_in_chw, pose_preprocessed, pose_out
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [TTO] ERROR: Gradient check failed ({e}). "
              f"Skipping TTO, returning renderer-only output.",
              file=sys.stderr)
        return renderer_frames

    # Step 1: Compute per-pair PoseNet distortion
    print("  [TTO] Computing per-pair PoseNet distortion...", file=sys.stderr)
    t0 = time.monotonic()
    pair_dists = _compute_per_pair_posenet_distortion(
        renderer_frames, gt_frames, posenet, device,
    )
    t_dist = time.monotonic() - t0
    print(f"  [TTO] Per-pair distortion computed in {t_dist:.1f}s "
          f"(mean={pair_dists.mean():.6f}, max={pair_dists.max():.6f})",
          file=sys.stderr)

    # Step 2: Sort pairs by distortion (hardest first)
    hardest_indices = torch.argsort(pair_dists, descending=True)
    n_to_tto = max(1, int(P * top_k_fraction))
    tto_pairs = hardest_indices[:n_to_tto]
    print(f"  [TTO] Will TTO {n_to_tto} of {P} pairs "
          f"(top {top_k_fraction * 100:.0f}% by distortion)",
          file=sys.stderr)

    # Step 3: Extract GT pose targets for the pairs we will TTO
    print("  [TTO] Extracting GT pose targets...", file=sys.stderr)
    t0 = time.monotonic()
    pose_targets = extract_gt_pose_targets(gt_frames, posenet, torch.device(device))
    t_targets = time.monotonic() - t0
    print(f"  [TTO] GT targets extracted in {t_targets:.1f}s", file=sys.stderr)

    # Step 4: TTO hardest pairs within budget
    refined_frames = renderer_frames.clone()
    n_refined = 0
    t_tto_start = time.monotonic()

    # Process in sub-batches of batch_pairs
    for sub_start in range(0, n_to_tto, batch_pairs):
        # Check time budget
        elapsed_total = time.monotonic() - t_phase_start
        if elapsed_total >= budget_seconds:
            print(f"  [TTO] Budget exhausted at pair {n_refined}/{n_to_tto} "
                  f"({elapsed_total:.1f}s / {budget_seconds:.0f}s budget)",
                  file=sys.stderr)
            break

        sub_end = min(sub_start + batch_pairs, n_to_tto)
        sub_pair_indices = tto_pairs[sub_start:sub_end]

        # Gather frames and masks for these pairs
        frame_indices = []
        for pi in sub_pair_indices:
            frame_indices.extend([2 * pi.item(), 2 * pi.item() + 1])

        sub_frames = renderer_frames[frame_indices].clone()
        sub_masks = masks[frame_indices]
        sub_pose_targets = pose_targets[sub_pair_indices]

        n_sub_pairs = len(sub_pair_indices)
        t0 = time.monotonic()

        try:
            sub_result = coupled_trajectory_optimize(
                masks=sub_masks,
                expected_pose=sub_pose_targets,
                posenet=posenet,
                segnet=segnet,
                num_steps=tto_steps,
                lr=tto_lr,
                seg_weight=seg_weight,
                pose_weight=pose_weight,
                compress_weight=0.5,
                noise_seed=42,
                device=device,
                log_every=max(tto_steps // 3, 1),
                init_frames=sub_frames,
                early_stop_patience=tto_steps + 1,
            )

            # Write back refined frames
            for i, fi in enumerate(frame_indices):
                refined_frames[fi] = sub_result[i].cpu()

            n_refined += n_sub_pairs
            dt = time.monotonic() - t0
            print(f"  [TTO] Batch {sub_start // batch_pairs + 1}: "
                  f"refined {n_sub_pairs} pairs in {dt:.1f}s "
                  f"(total: {n_refined}/{n_to_tto})",
                  file=sys.stderr)

        except Exception as e:
            print(f"  [TTO] WARNING: batch failed ({e}), using renderer output",
                  file=sys.stderr)
        finally:
            # Free GPU memory
            if device == "cuda":
                torch.cuda.empty_cache()

    t_tto_total = time.monotonic() - t_tto_start
    t_phase_total = time.monotonic() - t_phase_start
    print(f"  [TTO] Adaptive TTO complete: refined {n_refined}/{n_to_tto} pairs "
          f"in {t_tto_total:.1f}s (phase total: {t_phase_total:.1f}s)",
          file=sys.stderr)

    return refined_frames


def inflate_renderer_with_tto(
    archive_dir: str,
    inflated_dir: str,
    video_names_file: str,
    renderer_filename: str = "renderer.bin",
    mask_filename: str = "masks.mkv",
    out_w: int = OUT_W,
    out_h: int = OUT_H,
) -> None:
    """Full inflate pipeline with optional adaptive TTO.

    Wraps inflate_renderer() with an additional TTO refinement phase
    controlled by environment variables. Default behavior (INFLATE_TTO=0)
    is identical to inflate_renderer() -- no scorer loading, no TTO.

    COMPLIANCE WARNING: When INFLATE_TTO=1, this loads PoseNet and SegNet
    at inflate time for gradient-based optimization. Per council 8-1 vote,
    this requires an explicit compliance ruling before contest use.

    Environment variables:
        INFLATE_TTO:                0 (off, default) or 1 (enable TTO)
        INFLATE_TTO_BUDGET_SECONDS: seconds for TTO phase (default 1300)
        INFLATE_TTO_STEPS:          gradient steps per batch (default 100)
        INFLATE_TTO_TOP_K:          fraction of pairs to TTO (default 0.3)
        INFLATE_TTO_LR:             learning rate (default 0.005)
        INFLATE_TTO_BATCH_PAIRS:    pairs per batch (default 10)
        INFLATE_TTO_SEG_WEIGHT:     SegNet weight (default 100.0)
        INFLATE_TTO_POSE_WEIGHT:    PoseNet weight (default 10.0)
    """
    inflate_tto = os.environ.get("INFLATE_TTO", "0") == "1"

    if not inflate_tto:
        # Default path: renderer only, fully compliant
        return inflate_renderer(
            archive_dir, inflated_dir, video_names_file,
            renderer_filename=renderer_filename,
            mask_filename=mask_filename,
            out_w=out_w, out_h=out_h,
        )

    # ---- TTO path (EXPERIMENTAL, requires compliance ruling) ----
    print("=" * 60, file=sys.stderr)
    print("INFLATE_TTO=1: Adaptive TTO enabled", file=sys.stderr)
    print("WARNING: Requires compliance ruling for contest use", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    budget_seconds = float(os.environ.get("INFLATE_TTO_BUDGET_SECONDS", "1300"))
    tto_steps = int(os.environ.get("INFLATE_TTO_STEPS", "100"))
    top_k = float(os.environ.get("INFLATE_TTO_TOP_K", "0.3"))
    tto_lr = float(os.environ.get("INFLATE_TTO_LR", "0.005"))
    batch_pairs = int(os.environ.get("INFLATE_TTO_BATCH_PAIRS", "10"))
    seg_weight = float(os.environ.get("INFLATE_TTO_SEG_WEIGHT", "100.0"))
    pose_weight = float(os.environ.get("INFLATE_TTO_POSE_WEIGHT", "10.0"))

    print(f"  TTO config: budget={budget_seconds}s, steps={tto_steps}, "
          f"top_k={top_k}, lr={tto_lr}, batch_pairs={batch_pairs}",
          file=sys.stderr)
    print(f"  TTO weights: seg={seg_weight}, pose={pose_weight}",
          file=sys.stderr)

    t_total_start = time.monotonic()

    # ---- Device detection and loading (shared with inflate_renderer) ----
    device, render_batch_size = _detect_device_and_batch_size()
    renderer, masks, mask_video_path = _load_renderer_and_masks(
        archive_dir, device,
        renderer_filename=renderer_filename,
        mask_filename=mask_filename,
    )

    # ---- Generate renderer frames (at SegNet resolution) ----
    print("Stage 1: Generating renderer frames...", file=sys.stderr)
    t0 = time.monotonic()
    is_asymmetric = _is_asymmetric_model(renderer)
    N = masks.shape[0]

    torch.manual_seed(42)
    all_frames = []
    with torch.inference_mode():
        if is_asymmetric:
            P = N // 2
            for start in range(0, P, render_batch_size):
                end = min(start + render_batch_size, P)
                masks_t = masks[2 * start:2 * end:2].to(device=device, dtype=torch.long)
                masks_t1 = masks[2 * start + 1:2 * end + 1:2].to(device=device, dtype=torch.long)
                pairs = renderer(masks_t, masks_t1)  # (B, 2, H, W, 3)
                B = pairs.shape[0]
                f0 = pairs[:, 0]  # (B, H, W, 3)
                f1 = pairs[:, 1]
                interleaved = torch.stack([f0, f1], dim=1).reshape(2 * B, *f0.shape[1:])
                all_frames.append(interleaved.cpu())
        else:
            for i in range(0, N, render_batch_size):
                end = min(i + render_batch_size, N)
                batch_masks = masks[i:end].to(device=device, dtype=torch.long)
                frames = renderer(batch_masks)  # (B, 3, H, W)
                frames_hwc = frames.permute(0, 2, 3, 1)  # -> (B, H, W, 3)
                all_frames.append(frames_hwc.cpu())

    renderer_frames = torch.cat(all_frames, dim=0).float()  # (N, H, W, 3)
    t_render = time.monotonic() - t0
    print(f"  Generated {renderer_frames.shape[0]} frames in {t_render:.1f}s",
          file=sys.stderr)

    del renderer
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- Load GT frames and scorers for TTO ----
    print("Stage 2: Loading GT frames and scorers for TTO...", file=sys.stderr)
    upstream_root = _find_upstream_root(archive_dir)
    video_names = Path(video_names_file).read_text().splitlines()
    video_names = [v.strip() for v in video_names if v.strip()]

    # Find GT video
    rel = video_names[0]
    gt_candidates = [
        upstream_root / "videos" / rel,
        upstream_root / "data" / rel,
    ]
    data_dir = os.environ.get("COMMA_DATA_DIR")
    if data_dir:
        gt_candidates.insert(0, Path(data_dir) / rel)

    gt_path = None
    for c in gt_candidates:
        if c.exists():
            gt_path = c
            break
    if gt_path is None:
        tried = "\n  ".join(str(c) for c in gt_candidates)
        raise FileNotFoundError(
            f"GT video not found. Tried:\n  {tried}\n"
            f"Set COMMA_DATA_DIR to the directory containing GT videos."
        )

    gt_frames_np = _decode_gt_video(str(gt_path))
    gt_frames_torch = [torch.from_numpy(f) for f in gt_frames_np]

    # Load scorers via tac (differentiable mode)
    try:
        from tac.scorer import load_differentiable_scorers
        posenet, segnet = load_differentiable_scorers(upstream_root, device=device)
    except ImportError:
        print("  FATAL: tac package required for TTO mode", file=sys.stderr)
        raise

    # ---- Run adaptive TTO ----
    print("Stage 3: Running adaptive TTO...", file=sys.stderr)
    refined_frames = _adaptive_tto_phase(
        renderer_frames=renderer_frames,
        masks=masks,
        gt_frames=gt_frames_torch,
        posenet=posenet,
        segnet=segnet,
        device=device,
        budget_seconds=budget_seconds,
        tto_steps=tto_steps,
        top_k_fraction=top_k,
        tto_lr=tto_lr,
        batch_pairs=batch_pairs,
        seg_weight=seg_weight,
        pose_weight=pose_weight,
    )

    # Free scorers
    del posenet, segnet
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- Upscale and write raw output ----
    print("Stage 4: Upscaling and writing raw RGB...", file=sys.stderr)
    output_path = Path(inflated_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, rel in enumerate(video_names):
        stem = rel.rsplit(".", 1)[0]
        raw_out = output_path / f"{stem}.raw"
        raw_out.parent.mkdir(parents=True, exist_ok=True)

        n_written = 0
        with open(str(raw_out), "wb") as f:
            for i in range(0, N, render_batch_size):
                end = min(i + render_batch_size, N)
                batch = refined_frames[i:end]  # (B, H, W, 3)
                # Convert to CHW for interpolation
                batch_chw = batch.permute(0, 3, 1, 2).to(device)  # (B, 3, H, W)
                batch_up = F.interpolate(
                    batch_chw, size=(out_h, out_w),
                    mode="bilinear", align_corners=False,
                )
                batch_uint8 = batch_up.round().clamp(0, 255).to(torch.uint8)
                batch_hwc = batch_uint8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                f.write(batch_hwc.tobytes())
                n_written += batch_hwc.shape[0]

        actual_size = os.path.getsize(str(raw_out))
        expected_size = out_w * out_h * 3 * n_written
        if actual_size != expected_size:
            raise RuntimeError(
                f"Output size mismatch: {actual_size:,} != {expected_size:,}"
            )
        print(f"  Written {n_written} frames to {raw_out} ({actual_size:,} bytes)",
              file=sys.stderr)

    t_total = time.monotonic() - t_total_start
    print(f"\nTotal inflate+TTO time: {t_total:.1f}s", file=sys.stderr)


# ============================================================
# Click CLI (matches inflate_postfilter.py pattern)
# ============================================================
def _cli():
    """Click CLI entry point for inflate_renderer."""
    try:
        import click
    except ImportError:
        # Fallback to plain argparse if click not available
        import argparse
        parser = argparse.ArgumentParser(description="Inflate via neural renderer")
        parser.add_argument("archive_dir", help="Directory containing renderer.bin and masks.mkv")
        parser.add_argument("inflated_dir", help="Output directory for .raw files")
        parser.add_argument("video_names_file", help="Text file listing video names")
        parser.add_argument("--renderer-filename", default="renderer.bin",
                            help="Renderer checkpoint filename")
        parser.add_argument("--mask-filename", default="masks.mkv",
                            help="Pre-extracted mask video filename")
        parser.add_argument("--target-w", type=int, default=OUT_W)
        parser.add_argument("--target-h", type=int, default=OUT_H)
        args = parser.parse_args()
        inflate_renderer_with_tto(
            args.archive_dir, args.inflated_dir, args.video_names_file,
            renderer_filename=args.renderer_filename,
            mask_filename=args.mask_filename,
            out_w=args.target_w, out_h=args.target_h,
        )
        return

    @click.command()
    @click.argument("archive_dir", type=click.Path(exists=True))
    @click.argument("inflated_dir", type=click.Path())
    @click.argument("video_names_file", type=click.Path(exists=True))
    @click.option("--renderer-filename", default="renderer.bin", envvar="RENDERER_FILENAME",
                  help="Renderer checkpoint filename within archive_dir.")
    @click.option("--mask-filename", default="masks.mkv", envvar="MASK_FILENAME",
                  help="Pre-extracted mask video filename within archive_dir.")
    @click.option("--target-w", type=int, envvar="SOURCE_W",
                  default=OUT_W, help="Output frame width.")
    @click.option("--target-h", type=int, envvar="SOURCE_H",
                  default=OUT_H, help="Output frame height.")
    def inflate(archive_dir, inflated_dir, video_names_file,
                renderer_filename, mask_filename, target_w, target_h):
        """Inflate compressed archive using a trained neural renderer.

        \b
        Positional arguments (compatible with inflate.sh dispatch):
          ARCHIVE_DIR       Directory containing renderer.bin + masks.mkv
          INFLATED_DIR      Output directory for .raw files
          VIDEO_NAMES_FILE  Text file listing video names (one per line)

        \b
        Contest-compliant path: reads pre-extracted masks from masks.mkv
        in the archive. No SegNet loading at inflate time.

        \b
        Fallback: set INFLATE_MASK_SOURCE=segnet to extract masks from GT
        video via SegNet (development only, NOT contest-compliant).

        \b
        Adaptive TTO: set INFLATE_TTO=1 to enable test-time optimization
        on the hardest pairs. Requires compliance ruling for contest use.

        \b
        Device is auto-detected (CUDA if available, else CPU).
        Batch size: GPU=16, CPU=4.
        """
        inflate_renderer_with_tto(
            archive_dir, inflated_dir, video_names_file,
            renderer_filename=renderer_filename,
            mask_filename=mask_filename,
            out_w=target_w, out_h=target_h,
        )

    inflate()


if __name__ == "__main__":
    _cli()
