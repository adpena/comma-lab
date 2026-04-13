#!/usr/bin/env python3
"""Kaggle Auth Eval: Asymmetric Warp Renderer

Self-contained Kaggle notebook for authoritative evaluation of the
AsymmetricPairGenerator renderer checkpoints. Third platform in the
auth eval fleet (Modal + Lightning + Kaggle).

Kaggle gives 30h/week free GPU (P100 or T4).

Usage:
    1. Create a new Kaggle notebook with GPU accelerator enabled
    2. Upload your renderer checkpoint (.bin or .pt) as a Kaggle dataset,
       or provide a direct download URL
    3. Paste each cell below into the notebook
    4. Click "Run All"

The script is organized as cells separated by `# %%` markers.
Each cell can be pasted independently into a Kaggle notebook.

Architecture classes (MaskRenderer, MotionPredictor, AsymmetricPairGenerator,
CLADENorm, ResBlock, warp_with_flow, make_coord_grid) are inlined for
standalone operation without the tac package.
"""
from __future__ import annotations

# %% [markdown]
# # Auth Eval: Asymmetric Warp Renderer
#
# Upload a `.bin` or `.pt` checkpoint and run full authoritative evaluation
# against the upstream comma video compression challenge scorer.

# %%
# ============================================================
# Cell 1: Install dependencies
# ============================================================
import subprocess
import sys

DEPS = [
    "av",
    "safetensors",
    "timm",
    "einops",
    "segmentation-models-pytorch",
]

# NOTE: Using pip directly here because uv is not available in the Kaggle base
# image. This is an intentional exception to the uv-only rule in CLAUDE.md.
for dep in DEPS:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", dep],
        stdout=subprocess.DEVNULL,
    )

# Install git-lfs (needed for upstream model weights)
subprocess.run(
    ["bash", "-c", "command -v git-lfs || (apt-get update -qq && apt-get install -y -qq git-lfs)"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

print("Dependencies installed.")

# %%
# ============================================================
# Cell 2: Clone upstream + locate/download checkpoint
# ============================================================
import os
import shutil
import time
from pathlib import Path

# ---------- Configuration ----------
# Option A: Upload checkpoint as Kaggle dataset and set path here
# Option B: Set a direct download URL
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    "/kaggle/input/renderer-checkpoint/renderer.bin",  # default Kaggle dataset path
)
CHECKPOINT_URL = os.environ.get("CHECKPOINT_URL", "")  # set to download URL if needed

# Upstream repo
UPSTREAM_DIR = Path("/kaggle/working/upstream")
WORK_DIR = Path("/kaggle/working/auth_eval")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Clone upstream if not already present
if not UPSTREAM_DIR.exists():
    print("Cloning upstream repo (with LFS assets)...")
    t0 = time.time()
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/commaai/comma_video_compression_challenge.git",
        str(UPSTREAM_DIR),
    ])
    subprocess.check_call(["git", "lfs", "pull"], cwd=str(UPSTREAM_DIR))
    print(f"  Cloned in {time.time() - t0:.1f}s")
else:
    print(f"Upstream already at {UPSTREAM_DIR}")

# Verify upstream assets
for asset in ["models/posenet.safetensors", "models/segnet.safetensors",
              "modules.py", "evaluate.py"]:
    p = UPSTREAM_DIR / asset
    assert p.exists(), f"Missing upstream asset: {p}"
print("Upstream assets verified.")

# Download checkpoint if URL provided
checkpoint_path = Path(CHECKPOINT_PATH)
if CHECKPOINT_URL and not checkpoint_path.exists():
    print(f"Downloading checkpoint from {CHECKPOINT_URL}...")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "wget", "-q", "-O", str(checkpoint_path), CHECKPOINT_URL,
    ])

assert checkpoint_path.exists(), (
    f"Checkpoint not found at {checkpoint_path}.\n"
    f"Upload as Kaggle dataset or set CHECKPOINT_URL."
)
ckpt_size = checkpoint_path.stat().st_size
print(f"Checkpoint: {checkpoint_path} ({ckpt_size:,} bytes)")

# %%
# ============================================================
# Cell 3: Inline architecture (no tac dependency)
# ============================================================
# Canonical architecture classes matching src/tac/renderer.py
# These are inlined so the notebook runs standalone on Kaggle.

import json
import math
import struct
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Constants ---
OUT_W, OUT_H = 1164, 874
SEG_W, SEG_H = 512, 384
NUM_FRAMES = 1200
NUM_CLASSES = 5
EXPECTED_RAW_BYTES = OUT_W * OUT_H * 3 * NUM_FRAMES


# --- Coord grid cache ---
_coord_grid_cache: dict = {}


def make_coord_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    key = (h, w, device)
    if key not in _coord_grid_cache:
        gy = torch.linspace(-1, 1, h, device=device)
        gx = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        _coord_grid_cache[key] = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
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


# --- CLADENorm ---
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


# --- ResBlock ---
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


# --- MaskRenderer ---
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


# --- MotionPredictor (U-Net) ---
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


# --- AsymmetricPairGenerator ---
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


# --- SPADE / DPSIMSRenderer (for .pt checkpoint compat) ---
class SPADE(nn.Module):
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

    def _encode_mask(self, mask, target_h, target_w, device):
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
    def __init__(self, num_classes=5, channels=(256, 128, 64, 32),
                 init_h=24, init_w=32, spade_hidden=64, noise_dim=16, use_noise=True):
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


print("Architecture classes defined.")

# %%
# ============================================================
# Cell 4: Load checkpoint + run auth eval pipeline
# ============================================================
import av

# ---------- YUV -> RGB (BT.601 limited range, matches upstream frame_utils.py) ----------
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


# ---------- ASYM .bin deserializer ----------
def _inline_unpack_values(data, offset, count, bits):
    if bits == 8:
        values = [data[offset + i] for i in range(count)]
        return values, offset + count
    total_bits = count * bits
    total_bytes = (total_bits + 7) // 8
    if count > 10_000_000:
        raise ValueError(f"Implausible value count={count:,}")
    raw = data[offset:offset + total_bytes]
    bit_buffer = int.from_bytes(bytes(raw), byteorder="little")
    mask = (1 << bits) - 1
    values = []
    for _ in range(count):
        values.append(bit_buffer & mask)
        bit_buffer >>= bits
    return values, offset + total_bytes


def _inline_dequantize_values(values, bits, scale):
    bits = max(bits, 2)
    n_levels = 2 ** bits
    half = n_levels // 2
    return torch.tensor(
        [(v - half) / max(half - 1, 1) * scale for v in values],
        dtype=torch.float32,
    )


def load_asym_bin(raw_bytes: bytes, device: str = "cpu") -> nn.Module:
    """Load AsymmetricPairGenerator from ASYM .bin format."""
    offset = 0
    if raw_bytes[offset:offset + 4] != b"ASYM":
        raise ValueError(f"Not an ASYM binary (got {raw_bytes[:4]!r})")
    offset += 4

    header_len = struct.unpack("<I", raw_bytes[offset:offset + 4])[0]
    offset += 4
    header = json.loads(raw_bytes[offset:offset + header_len].decode("utf-8"))
    offset += header_len

    version = header.get("version", 0)
    if version != 2:
        raise ValueError(f"Unsupported ASYM export version {version} (expected 2)")

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

    embedding_lookup = {}
    conv_lookup = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_lookup[name] = module
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            conv_lookup[name] = module

    for layer_meta in header["layers"]:
        name = layer_meta["name"]
        is_embedding = layer_meta.get("is_embedding", False)

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

        has_bias = layer_meta["has_bias"]
        bias_blob_len = struct.unpack("<I", raw_bytes[offset:offset + 4])[0]
        offset += 4
        bias_data = raw_bytes[offset:offset + bias_blob_len]
        offset += bias_blob_len

        conv_module = conv_lookup[name]
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
            conv_module.weight.zero_()
            if conv_module.bias is not None:
                conv_module.bias.zero_()

            w_offset = 0
            for ch_idx in range(C_out):
                scale = struct.unpack("<e", weight_data[w_offset:w_offset + 2])[0]
                w_offset += 2
                values, w_offset = _inline_unpack_values(weight_data, w_offset, fan_in, bits)
                dequant = _inline_dequantize_values(values, bits, scale)
                if transposed:
                    conv_module.weight[:, ch_idx] = dequant.reshape(ch_shape)
                else:
                    conv_module.weight[ch_idx] = dequant.reshape(ch_shape)

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
                    conv_module.bias[ch_idx] = q / max(half - 1, 1) * scale_b

    scalar_params = header.get("scalar_params", {})
    if scalar_params:
        param_dict = dict(model.named_parameters())
        with torch.no_grad():
            for pname, pval in scalar_params.items():
                if pname in param_dict:
                    param_dict[pname].fill_(pval)

    if offset != len(raw_bytes):
        raise ValueError(f"Trailing data: {len(raw_bytes) - offset} bytes unread")

    model = model.to(device)
    model.eval()
    return model


def load_renderer(renderer_path: str, device: str) -> nn.Module:
    """Load renderer from .bin or .pt checkpoint."""
    raw_bytes = Path(renderer_path).read_bytes()
    magic = raw_bytes[:4]

    if magic == b"ASYM":
        model = load_asym_bin(raw_bytes, device=device)
        print(f"Loaded AsymmetricPairGenerator from .bin ({len(raw_bytes):,} bytes)")
        return model

    if magic == b"DPSM":
        header_len = struct.unpack("<I", raw_bytes[4:8])[0]
        header = json.loads(raw_bytes[8:8 + header_len].decode("utf-8"))
        raise RuntimeError(
            f"DPSM .bin format detected (version={header.get('version')}). "
            f"DPSM deserialization requires the tac package. Use ASYM .bin or .pt instead."
        )

    # PyTorch pickle format
    ckpt = torch.load(renderer_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    num_classes = config.get("num_classes", 5)
    channels = config.get("channels", (256, 128, 64, 32))
    if isinstance(channels, list):
        channels = tuple(channels)

    renderer = DPSIMSRenderer(
        num_classes=num_classes,
        channels=channels,
        init_h=config.get("init_h", 24),
        init_w=config.get("init_w", 32),
        spade_hidden=config.get("spade_hidden", 64),
        noise_dim=config.get("noise_dim", 16),
        use_noise=config.get("use_noise", True),
    )

    if "model_state_dict" in ckpt:
        raw_sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        raw_sd = ckpt["state_dict"]
    else:
        raw_sd = ckpt

    prefix = "renderer."
    if any(k.startswith(prefix) for k in raw_sd.keys()):
        sd = {k[len(prefix):]: v for k, v in raw_sd.items() if k.startswith(prefix)}
    else:
        sd = raw_sd

    renderer.load_state_dict(sd, strict=True)
    renderer.to(device).eval()
    for p in renderer.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in renderer.parameters())
    print(f"Loaded DPSIMSRenderer: {n_params:,} params")
    return renderer


def is_asymmetric(model: nn.Module) -> bool:
    return (
        type(model).__name__ == "AsymmetricPairGenerator"
        or (hasattr(model, "renderer") and hasattr(model, "motion")
            and hasattr(model.motion, "output_channels")
            and getattr(model.motion, "output_channels", 2) == 6)
    )


# ---------- SegNet loading ----------
def load_segnet(upstream_root: Path, device: str) -> nn.Module:
    upstream_str = str(upstream_root)
    sys.path.insert(0, upstream_str)
    try:
        from modules import SegNet
    finally:
        try:
            sys.path.pop(sys.path.index(upstream_str))
        except ValueError:
            pass

    segnet = SegNet()
    from safetensors.torch import load_file
    sd = load_file(str(upstream_root / "models" / "segnet.safetensors"), device=device)
    segnet.load_state_dict(sd)
    segnet.to(device).eval()
    for p in segnet.parameters():
        p.requires_grad = False
    print("SegNet loaded.")
    return segnet


# ---------- GT video decode ----------
def decode_gt_video(video_path: str) -> list[np.ndarray]:
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        rgb = yuv420_to_rgb(frame)
        frames.append(rgb.numpy())
    container.close()
    print(f"Decoded {len(frames)} GT frames from {video_path}")
    return frames


# ---------- Mask extraction ----------
def extract_masks(frames: list[np.ndarray], segnet: nn.Module,
                  device: str, batch_size: int = 8) -> torch.Tensor:
    N = len(frames)
    masks_list = []
    with torch.inference_mode():
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch_np = np.stack(frames[i:end], axis=0)
            batch_t = torch.from_numpy(batch_np).float().permute(0, 3, 1, 2).to(device)
            inp = batch_t.unsqueeze(1)
            seg_in = segnet.preprocess_input(inp)
            logits = segnet(seg_in)
            mask = logits.argmax(dim=1)
            masks_list.append(mask.to(torch.int8).cpu())
            if end % (batch_size * 20) == 0 or end == N:
                print(f"  Masks: {end}/{N}")
    masks = torch.cat(masks_list, dim=0)
    print(f"Extracted {masks.shape[0]} masks, shape {masks.shape}")
    return masks


# ---------- Generate frames + write .raw ----------
def generate_and_write(masks: torch.Tensor, renderer: nn.Module,
                       output_path: str, device: str, batch_size: int = 8) -> int:
    t0 = time.time()
    N = masks.shape[0]
    n_written = 0
    asym = is_asymmetric(renderer)
    torch.manual_seed(42)

    with open(output_path, 'wb') as f:
        with torch.inference_mode():
            if asym:
                print(f"Asymmetric pair generation: {N} masks -> {N} frames via {N // 2} pairs")
                pair_idx = 0
                while pair_idx < N - 1:
                    batch_t, batch_t1 = [], []
                    batch_end = min(pair_idx + batch_size * 2, N - 1)
                    for j in range(pair_idx, batch_end, 2):
                        if j + 1 < N:
                            batch_t.append(masks[j])
                            batch_t1.append(masks[j + 1])
                    if not batch_t:
                        break

                    masks_t = torch.stack(batch_t).to(device=device, dtype=torch.long)
                    masks_t1 = torch.stack(batch_t1).to(device=device, dtype=torch.long)
                    pairs = renderer(masks_t, masks_t1)  # (B, 2, H, W, 3)

                    for p in range(pairs.shape[0]):
                        for fi in range(2):
                            frame_hwc = pairs[p, fi]
                            frame_chw = frame_hwc.permute(2, 0, 1).unsqueeze(0)
                            frame_up = F.interpolate(
                                frame_chw, size=(OUT_H, OUT_W),
                                mode="bilinear", align_corners=False,
                            )
                            frame_uint8 = frame_up.round().clamp(0, 255).to(torch.uint8)
                            out = frame_uint8.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
                            f.write(out.tobytes())
                            n_written += 1

                    pair_idx += len(batch_t) * 2
                    if n_written % 100 == 0 or pair_idx >= N - 1:
                        print(f"  Generated: {n_written}/{N} frames")

                # Odd trailing mask
                if N % 2 != 0:
                    last_mask = masks[N - 1:N].to(device=device, dtype=torch.long)
                    frame = renderer.renderer(last_mask)
                    frame_up = F.interpolate(frame, size=(OUT_H, OUT_W),
                                             mode="bilinear", align_corners=False)
                    out = frame_up.round().clamp(0, 255).to(torch.uint8)
                    out = out.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
                    f.write(out.tobytes())
                    n_written += 1
            else:
                print(f"Independent frame generation: {N} masks -> {N} frames")
                for i in range(0, N, batch_size):
                    end = min(i + batch_size, N)
                    batch_masks = masks[i:end].to(device=device, dtype=torch.long)
                    frames_out = renderer(batch_masks)
                    frames_up = F.interpolate(frames_out, size=(OUT_H, OUT_W),
                                              mode="bilinear", align_corners=False)
                    frames_uint8 = frames_up.round().clamp(0, 255).to(torch.uint8)
                    frames_hwc = frames_uint8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                    f.write(frames_hwc.tobytes())
                    n_written += batch_masks.shape[0]
                    if end % 100 == 0 or end == N:
                        print(f"  Generated: {end}/{N} frames")

    elapsed = time.time() - t0
    raw_size = os.path.getsize(output_path)
    print(f"Generated {n_written} frames -> {output_path} ({raw_size:,} bytes, {elapsed:.1f}s)")
    return n_written


# ==================================================================
# MAIN PIPELINE
# ==================================================================
print("=" * 60)
print("AUTH EVAL: ASYMMETRIC WARP RENDERER")
print("=" * 60)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16 if device == "cuda" else 4
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Paths
upstream_root = Path(UPSTREAM_DIR)
# Probe GT video path: videos/0.mkv first (matches all other scripts), then fallbacks
gt_video = upstream_root / "videos" / "0.mkv"
if not gt_video.exists():
    # Try alternate video paths
    videos_dir = upstream_root / "videos"
    if videos_dir.exists():
        candidates = list(videos_dir.glob("*.mkv"))
        if candidates:
            gt_video = candidates[0]
    if not gt_video.exists():
        # Legacy path: data/ directory
        candidates = list((upstream_root / "data").glob("*.mkv")) + list((upstream_root / "data").glob("*.mp4"))
        if candidates:
            gt_video = candidates[0]
assert gt_video.exists(), f"GT video not found. Checked {gt_video}"
print(f"GT video: {gt_video}")

# Stage 1: Load renderer
print("\n--- Stage 1: Load renderer ---")
t_stage = time.time()
renderer = load_renderer(str(checkpoint_path), device)
n_params = sum(p.numel() for p in renderer.parameters())
print(f"  Params: {n_params:,}")
print(f"  Type: {'AsymmetricPairGenerator' if is_asymmetric(renderer) else 'DPSIMSRenderer'}")
print(f"  Loaded in {time.time() - t_stage:.1f}s")

# Stage 2: Load SegNet
print("\n--- Stage 2: Load SegNet ---")
t_stage = time.time()
segnet = load_segnet(upstream_root, device)
print(f"  Loaded in {time.time() - t_stage:.1f}s")

# Stage 3: Decode GT video
print("\n--- Stage 3: Decode GT video ---")
t_stage = time.time()
gt_frames = decode_gt_video(str(gt_video))
n_frames = len(gt_frames)
print(f"  {n_frames} frames, shape {gt_frames[0].shape}")
print(f"  Decoded in {time.time() - t_stage:.1f}s")

# Stage 4: Extract masks
print("\n--- Stage 4: Extract SegNet masks ---")
t_stage = time.time()
masks = extract_masks(gt_frames, segnet, device, batch_size=batch_size)
del gt_frames  # free RAM
print(f"  Extracted in {time.time() - t_stage:.1f}s")

# Stage 5: Generate frames
print("\n--- Stage 5: Generate frames + write .raw ---")
t_stage = time.time()
raw_path = WORK_DIR / "0.raw"
n_written = generate_and_write(masks, renderer, str(raw_path), device, batch_size=batch_size)
del masks  # free RAM
print(f"  Generated in {time.time() - t_stage:.1f}s")

# Verify .raw size
actual = os.path.getsize(str(raw_path))
expected = OUT_W * OUT_H * 3 * n_written
assert actual == expected, f"Size mismatch: {actual:,} != {expected:,}"
print(f"  Verified: {actual:,} bytes ({n_written} frames x {OUT_H}x{OUT_W}x3)")

# Stage 6: Prepare submission dir for upstream evaluate.py
print("\n--- Stage 6: Prepare submission + run upstream evaluate.py ---")
t_stage = time.time()

submission_dir = WORK_DIR / "submission"
inflated_dir = submission_dir / "inflated"
inflated_dir.mkdir(parents=True, exist_ok=True)

# Copy .raw to inflated dir
import shutil
shutil.copy2(str(raw_path), str(inflated_dir / "0.raw"))

# Build archive.zip (contains only the renderer checkpoint)
import zipfile
archive_zip = submission_dir / "archive.zip"
with zipfile.ZipFile(str(archive_zip), "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(str(checkpoint_path), "renderer.bin")
archive_bytes = archive_zip.stat().st_size
print(f"  Archive: {archive_bytes:,} bytes")

# Determine video names file
video_names_file = upstream_root / "public_test_video_names.txt"
if not video_names_file.exists():
    # Create a minimal one
    video_names_file = WORK_DIR / "video_names.txt"
    # Find the actual video name
    video_name = gt_video.name
    video_names_file.write_text(video_name + "\n")
    print(f"  Created video_names.txt with: {video_name}")

# Determine uncompressed videos dir
uncompressed_dir = upstream_root / "videos"
if not uncompressed_dir.exists():
    uncompressed_dir = upstream_root / "data"

# Run upstream evaluate.py
report_path = submission_dir / "report.txt"
eval_cmd = [
    sys.executable,
    str(upstream_root / "evaluate.py"),
    "--submission-dir", str(submission_dir),
    "--uncompressed-dir", str(uncompressed_dir),
    "--report", str(report_path),
    "--video-names-file", str(video_names_file),
    "--device", device,
]
print(f"  Running: {' '.join(eval_cmd)}")
result = subprocess.run(eval_cmd, capture_output=True, text=True, cwd=str(upstream_root))

if result.returncode != 0:
    print(f"  STDERR:\n{result.stderr}")
    print(f"  STDOUT:\n{result.stdout}")
    raise RuntimeError(f"evaluate.py failed with return code {result.returncode}")

eval_time = time.time() - t_stage
print(f"  Evaluation completed in {eval_time:.1f}s")

# %%
# ============================================================
# Cell 5: Print results + save
# ============================================================
import re

print("\n" + "=" * 60)
print("AUTHORITATIVE EVALUATION RESULTS")
print("=" * 60)

# Parse report
if report_path.exists():
    report_text = report_path.read_text()
    print(f"\n--- Raw Report ---\n{report_text}")

    # Extract key metrics
    patterns = {
        "pose_distortion": re.compile(r"pose_distortion:\s*([\d.]+)"),
        "seg_distortion": re.compile(r"seg_distortion:\s*([\d.]+)"),
        "archive_bytes": re.compile(r"archive_bytes:\s*([\d,]+)"),
        "original_bytes": re.compile(r"original_bytes:\s*([\d,]+)"),
        "rate": re.compile(r"rate:\s*([\d.]+)"),
        "score": re.compile(r"score:\s*([\d.]+)"),
    }

    parsed = {}
    for key, pat in patterns.items():
        match = pat.search(report_text)
        if match:
            raw = match.group(1).replace(",", "")
            parsed[key] = float(raw)

    if parsed:
        print("\n--- Parsed Metrics ---")
        if "score" in parsed:
            print(f"  SCORE:              {parsed['score']:.4f}")
        if "pose_distortion" in parsed:
            print(f"  PoseNet distortion: {parsed['pose_distortion']:.6f}")
        if "seg_distortion" in parsed:
            print(f"  SegNet distortion:  {parsed['seg_distortion']:.6f}")
        if "rate" in parsed:
            print(f"  Rate:               {parsed['rate']:.6f}")
        if "archive_bytes" in parsed:
            print(f"  Archive bytes:      {int(parsed['archive_bytes']):,}")

        # Component breakdown
        if all(k in parsed for k in ("pose_distortion", "seg_distortion", "rate")):
            pose_component = math.sqrt(10 * parsed["pose_distortion"])
            seg_component = 100 * parsed["seg_distortion"]
            rate_component = 25 * parsed["rate"]
            print(f"\n--- Score Components ---")
            print(f"  100 * seg_dist  = {seg_component:.4f}")
            print(f"  sqrt(10 * pose) = {pose_component:.4f}")
            print(f"  25 * rate       = {rate_component:.4f}")
            print(f"  Total           = {seg_component + pose_component + rate_component:.4f}")
else:
    print("WARNING: report.txt not found")

# Also show evaluate.py stdout/stderr
if result.stdout.strip():
    print(f"\n--- evaluate.py stdout ---\n{result.stdout}")
if result.stderr.strip():
    print(f"\n--- evaluate.py stderr ---\n{result.stderr}")

# Save results JSON
results_json = {
    "checkpoint": str(checkpoint_path),
    "checkpoint_bytes": ckpt_size,
    "device": device,
    "n_frames": n_written,
    "platform": "kaggle",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
if parsed:
    results_json.update(parsed)

results_path = WORK_DIR / "auth_eval_results.json"
results_path.write_text(json.dumps(results_json, indent=2))
print(f"\nResults saved to {results_path}")

# Copy to Kaggle output for easy download
kaggle_output = Path("/kaggle/working")
if kaggle_output.exists():
    shutil.copy2(str(results_path), str(kaggle_output / "auth_eval_results.json"))
    if report_path.exists():
        shutil.copy2(str(report_path), str(kaggle_output / "report.txt"))
    print(f"Results copied to {kaggle_output} for download")

print("\n" + "=" * 60)
print("AUTH EVAL COMPLETE")
print("=" * 60)
