#!/usr/bin/env python
"""Inflate path using a trained DP-SIMS neural renderer.

The renderer generates RGB frames purely from SegNet masks extracted from
the ground-truth video.  No compressed video is stored — only the renderer
weights (~200KB) are in the archive.  This is the ultimate rate-quality
tradeoff: fixed rate regardless of content complexity.

Pipeline:
    GT video  ->  SegNet (upstream)  ->  masks (384x512)
    masks     ->  DPSIMSRenderer     ->  frames (384x512)
    frames    ->  bilinear upscale   ->  raw RGB (1164x874)

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
                     max_flow_px=12.0, max_residual=20.0, flow_only=False):
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
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 3, padding=1, bias=True),
                nn.SiLU(inplace=True),
                ResBlock(hidden, num_classes=0),  # plain GroupNorm — no mask in Sequential
                nn.Conv2d(hidden, hidden, 3, padding=1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, output_channels, 3, padding=1, bias=True),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

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
            raw = self.net(x)
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
                     motion_hidden=32, depth=1, max_flow_px=12.0,
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
        3. UPSTREAM_ROOT env var
    """
    candidates = []

    # 1. Scorer environment layout
    candidates.append(Path(archive_dir).resolve().parent.parent)

    # 2. Local dev layout
    candidates.append(Path(__file__).resolve().parent.parent.parent / "upstream")

    # 3. Environment variable
    env_root = os.environ.get("UPSTREAM_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    # Also check COMMA_CHALLENGE_ROOT (used by inflate_postfilter.py)
    env_root2 = os.environ.get("COMMA_CHALLENGE_ROOT")
    if env_root2:
        candidates.append(Path(env_root2))

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
        f"Set UPSTREAM_ROOT or COMMA_CHALLENGE_ROOT env var."
    )


# ============================================================
# SegNet loading
# ============================================================
def _load_segnet(upstream_root: Path, device: str) -> nn.Module:
    """Load frozen SegNet from upstream for mask extraction."""
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
        except ImportError:
            import struct as _struct
            header_len = _struct.unpack("<I", raw_bytes[4:8])[0]
            header = json.loads(raw_bytes[8:8 + header_len].decode("utf-8"))
            raise RuntimeError(
                f"ASYM .bin format detected (version={header.get('version')}), "
                f"but tac.renderer_export is not available. Install the tac package "
                f"or use a .pt checkpoint instead."
            )
        model = load_asymmetric_checkpoint(raw_bytes, device=device)
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

    # Determine which state_dict to use
    if "model_state_dict" in ckpt:
        raw_sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        raw_sd = ckpt["state_dict"]
    else:
        # Assume the checkpoint IS the state_dict
        raw_sd = ckpt

    # Check if keys are prefixed with "renderer." (from DPSIMSPairGenerator)
    renderer_prefix = "renderer."
    has_prefix = any(k.startswith(renderer_prefix) for k in raw_sd.keys())

    if has_prefix:
        # Extract only renderer.* keys, strip prefix
        sd = {}
        for k, v in raw_sd.items():
            if k.startswith(renderer_prefix):
                sd[k[len(renderer_prefix):]] = v
        print(f"  Extracted {len(sd)} renderer keys from PairGenerator checkpoint", file=sys.stderr)
    else:
        sd = raw_sd

    # Load weights
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
def inflate_renderer(
    archive_dir: str,
    inflated_dir: str,
    video_names_file: str,
    renderer_filename: str = "renderer.bin",
    out_w: int = OUT_W,
    out_h: int = OUT_H,
) -> None:
    """Full inflate pipeline: GT video -> masks -> renderer -> raw RGB.

    Args:
        archive_dir: directory containing renderer.bin
        inflated_dir: output directory for .raw files
        video_names_file: text file listing video names (one per line)
        renderer_filename: renderer checkpoint filename within archive_dir
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

    # ---- Upstream discovery ----
    print("Stage 1: Discovering upstream environment ...", file=sys.stderr)
    upstream_root = _find_upstream_root(archive_dir)
    print(f"  Upstream root: {upstream_root}", file=sys.stderr)

    # ---- Load SegNet ----
    print("Stage 2: Loading SegNet ...", file=sys.stderr)
    segnet = _load_segnet(upstream_root, device)

    # ---- Load renderer ----
    print("Stage 3: Loading renderer ...", file=sys.stderr)
    renderer_path = Path(archive_dir) / renderer_filename
    if not renderer_path.exists():
        raise FileNotFoundError(
            f"Renderer not found: {renderer_path}\n"
            f"Expected {renderer_filename} inside archive directory."
        )
    renderer = _load_renderer(str(renderer_path), device)

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

        # Find GT video: look in the upstream/scorer data directory
        # The scorer provides GT videos alongside the archive for evaluation
        gt_candidates = [
            Path(archive_dir).parent / rel,  # scorer layout: data/<video>.mkv
            Path(archive_dir).parent.parent / "data" / rel,
            upstream_root / "data" / rel,
        ]
        # Also check COMMA_DATA_DIR env var
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

        # Stage 4: Decode GT video
        print("Stage 4: Decoding GT video ...", file=sys.stderr)
        gt_frames = _decode_gt_video(str(gt_path))

        if len(gt_frames) != NUM_FRAMES:
            print(f"  WARNING: expected {NUM_FRAMES} frames, got {len(gt_frames)}", file=sys.stderr)

        # Stage 5: Extract masks
        print("Stage 5: Extracting SegNet masks ...", file=sys.stderr)
        masks = _extract_masks(gt_frames, segnet, device, batch_size)
        del gt_frames  # free memory

        # Verify mask resolution
        assert masks.shape[1] == SEG_H and masks.shape[2] == SEG_W, \
            f"Mask resolution mismatch: {masks.shape} vs expected ({SEG_H}, {SEG_W})"

        # Stage 6: Generate and write
        print("Stage 6: Generating frames via renderer ...", file=sys.stderr)
        n_written = _generate_and_write(masks, renderer, str(raw_out), device, batch_size, out_h, out_w)
        del masks  # free memory

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
        parser.add_argument("archive_dir", help="Directory containing renderer.bin")
        parser.add_argument("inflated_dir", help="Output directory for .raw files")
        parser.add_argument("video_names_file", help="Text file listing video names")
        parser.add_argument("--renderer-filename", default="renderer.bin",
                            help="Renderer checkpoint filename")
        parser.add_argument("--target-w", type=int, default=OUT_W)
        parser.add_argument("--target-h", type=int, default=OUT_H)
        args = parser.parse_args()
        inflate_renderer(
            args.archive_dir, args.inflated_dir, args.video_names_file,
            renderer_filename=args.renderer_filename,
            out_w=args.target_w, out_h=args.target_h,
        )
        return

    @click.command()
    @click.argument("archive_dir", type=click.Path(exists=True))
    @click.argument("inflated_dir", type=click.Path())
    @click.argument("video_names_file", type=click.Path(exists=True))
    @click.option("--renderer-filename", default="renderer.bin", envvar="RENDERER_FILENAME",
                  help="Renderer checkpoint filename within archive_dir.")
    @click.option("--target-w", type=int, envvar="SOURCE_W",
                  default=OUT_W, help="Output frame width.")
    @click.option("--target-h", type=int, envvar="SOURCE_H",
                  default=OUT_H, help="Output frame height.")
    def inflate(archive_dir, inflated_dir, video_names_file,
                renderer_filename, target_w, target_h):
        """Inflate compressed archive using a trained neural renderer.

        \b
        Positional arguments (compatible with inflate.sh dispatch):
          ARCHIVE_DIR       Directory containing renderer.bin
          INFLATED_DIR      Output directory for .raw files
          VIDEO_NAMES_FILE  Text file listing video names (one per line)

        \b
        The renderer generates RGB frames from SegNet masks extracted from
        the ground-truth video. No compressed video is needed in the archive.

        \b
        Device is auto-detected (CUDA if available, else CPU).
        Batch size: GPU=16, CPU=4.
        """
        inflate_renderer(
            archive_dir, inflated_dir, video_names_file,
            renderer_filename=renderer_filename,
            out_w=target_w, out_h=target_h,
        )

    inflate()


if __name__ == "__main__":
    _cli()
