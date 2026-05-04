#!/usr/bin/env python3
import argparse
import bz2
import io
import os
import pickle
import subprocess
import sys
import math
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import av
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import DistortionNet, SegNet, PoseNet
from safetensors.torch import load_file

try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False

# ─── Constants ────────────────────────────────────────────────────────────────

H_OUT  = 874          # evaluator output
W_OUT  = 1164
H_GEN  = 384          # generator resolution (= segnet_model_input_size)
W_GEN  = 512
H_YUV  = H_GEN // 2  # 192  — native YUV6 resolution (generator output)
W_YUV  = W_GEN // 2  # 256

META_FILE  = "meta.pkl"
MASK_FILE  = "mask.br"
POSE_FILE  = "pose.br"
INTER_POSE_FILE = "inter_pose.br"
MODEL_FILE = "obrdo.br"


# Squeeze excitation block creates self-attention on channel level (recalibrates
# weights for channels (abstract features (road curvature, asphalt texture, reflections)
# from previous layer that are important) get their weight boosted).
# SE asks WHAT is important.
class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.conv(self.avg_pool(x))

# PA focuses on parts of input frame that are important (objects edges, ...).
# PA asks WHERE is important.
class PixelAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(ch, ch // 8, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 8, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.pa(x)

class DSResidualBlockwDilation(nn.Module):
    def __init__(self, ch, dilation=1, cond_dim=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=dilation, dilation=dilation, groups=ch),
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=dilation, dilation=dilation, groups=ch),
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.InstanceNorm2d(ch)               
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.se = SEBlock(ch)
        self.pa = PixelAttention(ch)
        # Incorporated FiLM block - enables Generator to produce frames that are
        # influenced by pose vector (car turns -> pixels in frame will be shifted, ...)
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.film = nn.Linear(cond_dim, ch * 2)
            self.norm = nn.InstanceNorm2d(ch)
            
    def forward(self, x, cond=None):
        h = self.conv(x)
        
        if self.cond_dim is not None and cond is not None:
            gamma, beta = self.film(cond).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
            h = h * (1 + gamma) + beta
            h = self.norm(h)
        
        h = self.act(h)
        h = self.pa(self.se(h))
        return x + h

class GatedFusion(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2 * ch, ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, body_feat, res_feat):
        combined = torch.cat([body_feat, res_feat], dim=1)
        gate = self.gate_conv(combined)
        return gate * body_feat + (1 - gate) * res_feat
       
class MaskEncoder(nn.Module):
    """
    Segmentation mask (5 tříd, 384×512) → feature mapa [B, c1, 384, 512].
    U-Net architecture.
    """
    def __init__(self, num_classes=5, emb_dim=8, c1=48, c2=56):
        super().__init__()
        self.emb  = nn.Embedding(num_classes, emb_dim)
        # stem fuses emb. vectors with [x,y] coords
        self.stem = nn.Sequential(nn.Conv2d(emb_dim + 2, c1, kernel_size=3, padding=1),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  DSResidualBlockwDilation(c1))
        # Conv2d changes data dimensions, in DSResBlock model learns features
        self.down = nn.Sequential(nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
                                  DSResidualBlockwDilation(c2, dilation=2),
                                  DSResidualBlockwDilation(c2, dilation=4))
        self.up   = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                  nn.Conv2d(c2, c1, kernel_size=1),
                                  DSResidualBlockwDilation(c1))
        self.fuse = GatedFusion(c1)
        self.out  = DSResidualBlockwDilation(c1)
        self.out_ch = c1
        
    def forward(self, mask):
        B, Hm, Wm = mask.shape
        e = self.emb(mask).permute(0,3,1,2)
        c = make_coord_grid(B, Hm, Wm, mask.device, e.dtype)
        s = self.stem(torch.cat([e, c], dim=1))
        d = self.down(s)
        u = self.up(d)
        return self.out(self.fuse(u, s))


class FrameHead(nn.Module):
    """
    Feature map [B, ch, H, W] + pose cond → YUV6 [B, 6, H//2, W//2]
    """
    def __init__(self, in_ch, hidden=40, cond_dim=64):
        super().__init__()
        self.b1  = DSResidualBlockwDilation(in_ch, dilation=2, cond_dim=cond_dim) #FiLM
        self.b2  = DSResidualBlockwDilation(in_ch, dilation=4)
        self.b3  = DSResidualBlockwDilation(in_ch, dilation=8)
        # pre-output layer so channel drop gradually 64 -> 40 -> 6 
        # to minimize information loss that would drop from 64 -> 6 cause
        self.pre = nn.Conv2d(in_ch, hidden, kernel_size=1) 
        self.out = nn.Conv2d(hidden, 6, kernel_size=3, stride=2, padding=1)

    def forward(self, feat, cond):
        x = self.b3(self.b2(self.b1(feat, cond))) # b1 uses cond (pose)
        x = self.pre(x)
        x = self.out(F.leaky_relu(x, 0.2))
        return torch.sigmoid(x) * 255.0

class Generator(nn.Module):
    """
    (mask, pose6) → (yuv6_frame1, yuv6_frame2)
    """
    def __init__(self, num_classes=5, pose_dim=6, cond_dim=48):
        super().__init__()
        self.mask_enc = MaskEncoder(num_classes=num_classes)
        self.pose_mlp = nn.Sequential(nn.Linear(pose_dim, cond_dim), nn.SiLU(), 
                                      nn.Linear(cond_dim, cond_dim))
        ch = self.mask_enc.out_ch
        self.head = FrameHead(in_ch=48, hidden=96, cond_dim=cond_dim)
        self.inter_pose_reset = nn.Parameter(torch.full((1, pose_dim), 0.01))

    def forward(self, mask, pose6, inter_pose=None, reset_state=False):
        B = mask.shape[0]
        feat = self.mask_enc(mask) 
        
        if inter_pose is not None and not reset_state:
            cond1 = self.pose_mlp(inter_pose)
        else:
            cond1 = self.pose_mlp(self.inter_pose_reset.expand(B, -1))
        
        yuv6_f1 = checkpoint(self.head, feat, cond1, use_reentrant=False)   # [B, 6, H_YUV, W_YUV]
        
        cond2 = self.pose_mlp(pose6)
        yuv6_f2 = checkpoint(self.head, feat, cond2, use_reentrant=False)
        return yuv6_f1, yuv6_f2

    @staticmethod
    def count_params():
        return sum(p.numel() for p in Generator().parameters())

# Model serialization

def _compress_bytes(data):
    return brotli.compress(data, quality=11, lgwin=24) if HAS_BROTLI else bz2.compress(data, 9)

def _decompress_bytes(data):
    return brotli.decompress(data) if HAS_BROTLI else bz2.decompress(data)

def save_model(model, path):
    packed = []
    for key, t in model.state_dict().items():
        w = t.cpu().numpy().astype(np.float32)
        s = float(np.max(np.abs(w))) / 127.0 + 1e-8
        wq = np.clip(np.round(w/s), -127, 127).astype(np.int8)
        packed.append((key, wq.tobytes(), s, wq.shape))
    path.write_bytes(_compress_bytes(pickle.dumps(packed, protocol=5)))

def load_model(path, device, gen):
    packed = pickle.loads(_decompress_bytes(path.read_bytes()))
    sd = {}
    for key, data, scale, shape in packed:
        wq = np.frombuffer(data, dtype=np.int8).reshape(shape)
        sd[key] = torch.from_numpy(wq.astype(np.float32) * scale)
    blur_key = "mask_enc.fixed_blur.0.weight"
    if blur_key in sd:
        del sd[blur_key]
    if gen:
        m = Generator()
    m.load_state_dict(sd, strict=False)
    return m.to(device)

# YUV6 conversion
def diff_rgb_to_yuv6(rgb_chw: torch.Tensor) -> torch.Tensor:
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., : , :2*H2, :2*W2]
  
    R = rgb[..., 0, :, :]
    G = rgb[..., 1, :, :]
    B = rgb[..., 2, :, :]
  
    kYR, kYG, kYB = 0.299, 0.587, 0.114
    Y = (R * kYR + G * kYG + B * kYB).clamp_(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp_(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp_(0.0, 255.0)
  
    U_sub = (
      U[..., 0::2, 0::2] + U[..., 1::2, 0::2] +
      U[..., 0::2, 1::2] + U[..., 1::2, 1::2]
    ) * 0.25
    V_sub = (
      V[..., 0::2, 0::2] + V[..., 1::2, 0::2] +
      V[..., 0::2, 1::2] + V[..., 1::2, 1::2]
    ) * 0.25
  
    y00 = Y[..., 0::2, 0::2]
    y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]
    y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)

def diff_yuv6_to_rgb(yuv6: torch.Tensor) -> torch.Tensor:
    """
    Inverse function to diff_rgb_to_yuv6.
    yuv6: [B, 6, H2, W2]
    returns: [B, 3, H, W] in range 0-255
    """
    B, C, H2, W2 = yuv6.shape
    H, W = H2 * 2, W2 * 2
    
    Y = torch.zeros((B, H, W), device=yuv6.device, dtype=yuv6.dtype)
    Y[:, 0::2, 0::2] = yuv6[:, 0]
    Y[:, 1::2, 0::2] = yuv6[:, 1]
    Y[:, 0::2, 1::2] = yuv6[:, 2]
    Y[:, 1::2, 1::2] = yuv6[:, 3]
    
    U_sub = yuv6[:, 4:5] # [B, 1, H2, W2]
    V_sub = yuv6[:, 5:6]
    
    U = F.interpolate(U_sub, size=(H, W), mode='nearest').squeeze(1)
    V = F.interpolate(V_sub, size=(H, W), mode='nearest').squeeze(1)
    
    R = Y + 1.402 * (V - 128.0)
    B = Y + 1.772 * (U - 128.0)
    G = (Y - R * 0.299 - B * 0.114) / 0.587
    
    rgb = torch.stack([R, G, B], dim=1)
    return rgb.clamp(0.0, 255.0)

def diff_round(x: torch.Tensor) -> torch.Tensor:
    return (torch.round(x.clamp(0, 255)) - x).detach() + x

def no_upscaler(yuv6_f1: torch.Tensor, yuv6_f2: torch.Tensor):
    hat_rgb_small_f1 = diff_yuv6_to_rgb(yuv6_f1)
    hat_rgb_small_f2 = diff_yuv6_to_rgb(yuv6_f2)
    
    hat_rgb_large_f1 = F.interpolate(hat_rgb_small_f1, size=(874, 1164), mode='bilinear')
    hat_rgb_large_f2 = F.interpolate(hat_rgb_small_f2, size=(874, 1164), mode='bilinear')
    return hat_rgb_large_f1, hat_rgb_large_f2

def make_coord_grid(B, H, W, device, dtype):
    ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
    xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx*2-1, yy*2-1], dim=0).unsqueeze(0).expand(B,-1,-1,-1)

def encode_masks(masks, path, crf=50, fps=20):
    N, H, W = masks.shape
    imgs = (masks.astype(np.uint8) * 42).clip(0, 252)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:0",
           "-c:v", "libaom-av1", "-crf", str(crf), "-b:v", "0",
           "-pix_fmt", "yuv420p", "-cpu-used", "4", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for f in imgs:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"Mask encoding failed")
    
def decode_masks(path):
    c = av.open(str(path))
    frames = []
    for frame in c.decode(video=0):
        img = frame.to_ndarray(format="gray")
        frames.append(np.round(img / 42.0).astype(np.uint8).clip(0, 4))
    c.close()
    return np.stack(frames)

@torch.no_grad()
def extract_masks_and_poses(video_path, segnet, posenet, dn, device):
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"  [extract] {len(frames)} frames")

    masks_list, poses_list, inter_poses_list = [], [], []
    for i in range(0, len(frames) - 1, 2):
        f0t = torch.from_numpy(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)).permute(2,0,1).float().unsqueeze(0).to(device)
        f1t = torch.from_numpy(cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2RGB)).permute(2,0,1).float().unsqueeze(0).to(device)
            
        pair_dn = torch.stack([f0t, f1t], dim=1).permute(0,1,3,4,2).contiguous()  # [1,2,H,W,3]
        posenet_in, segnet_in = dn.preprocess_input(pair_dn)

        seg_logits = segnet(segnet_in)
        mask = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (W_GEN, H_GEN), interpolation=cv2.INTER_NEAREST)
        masks_list.append(mask)

        pose_out = posenet(posenet_in)
        pose6 = pose_out["pose"].squeeze(0)[:6].cpu().numpy()
        poses_list.append(pose6)

        if i + 2 < len(frames):
            f2t = torch.from_numpy(cv2.cvtColor(frames[i+2], cv2.COLOR_BGR2RGB)).permute(2,0,1).float().unsqueeze(0).to(device)
            inter_pair_dn = torch.stack([f1t, f2t], dim=1).permute(0,1,3,4,2).contiguous()  # [1,2,H,W,3]
            inter_posenet_in, _ = dn.preprocess_input(inter_pair_dn)
            inter_pose_out = posenet(inter_posenet_in)
            inter_pose6 = inter_pose_out["pose"].squeeze(0)[:6].cpu().numpy()
            inter_poses_list.append(inter_pose6)
        else:
            inter_poses_list.append(np.zeros(6, dtype=np.float32))

    masks = np.stack(masks_list)
    poses = np.stack(poses_list)
    inter_poses = np.stack(inter_poses_list)
    return masks, poses, inter_poses


# ─── Dataset ──────────────────────────────────────────────────────────────────

class GenDataset(torch.utils.data.Dataset):
    def __init__(self, orig_frames_halfres, orig_frames_fullres, masks, poses, inter_poses):
        n = len(masks)
        assert len(orig_frames_halfres) >= n * 2

        self.masks = torch.from_numpy(masks).long()
        self.poses = torch.from_numpy(poses).float()
        shifted_inter = np.zeros_like(inter_poses)
        shifted_inter[1:] = inter_poses[:-1]
        self.inter_poses = torch.from_numpy(shifted_inter).float()

        self.orig_half = orig_frames_halfres
        self.orig_full = orig_frames_fullres

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        m1, m2 = idx * 2, idx * 2 + 1
        
        p_half = torch.from_numpy(np.stack([self.orig_half[m1], self.orig_half[m2]]))
        p_half = p_half.permute(0, 3, 1, 2).float()
        
        p_full = torch.from_numpy(np.stack([self.orig_full[m1], self.orig_full[m2]]))
        p_full = p_full.permute(0, 3, 1, 2).float()
        return self.masks[idx], self.poses[idx], self.inter_poses[idx], p_half, p_full


def read_rgb_frames_fullres(path):
    import av
    from frame_utils import yuv420_to_rgb
    
    container = av.open(str(path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        rgb = yuv420_to_rgb(frame)
        frames.append(rgb.numpy())
    container.close()
    return frames

def read_rgb_frames(path, H, W):
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

# ─── Loss ─────────────────────────────────────────────────────────────────────
SEG_COLORS = np.array([
    [70,  130, 180],   # 0 Road        — blue
    [255, 255,   0],   # 1 Lane        — yellow
    [180,  20,  20],   # 2 Undrivable  — red
    [255, 165,   0],   # 3 Car         — orange
    [ 50, 205,  50],   # 4 Human       — green
], dtype=np.uint8)

def colorize_mask(mask_hw: np.ndarray) -> np.ndarray:
    rgb = SEG_COLORS[mask_hw.clip(0, 4)]
    return rgb[:, :, ::-1].copy()  # RGB → BGR

def render_bar(value: float, max_value: float, 
               width: int, height: int, 
               label: str, color_bgr=(0, 200, 255)) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    fill_h = int((value / max_value) * (height - 40))
    fill_h = min(fill_h, height - 40)
    cv2.rectangle(canvas,
                  (10, height - 40 - fill_h),
                  (width - 10, height - 40),
                  color_bgr, -1)
    cv2.rectangle(canvas, (10, 10), (width - 10, height - 40), (150, 150, 150), 1)
    cv2.putText(canvas, f"{value:.4f}", (5, height - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(canvas, label, (5, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    return canvas

def draw_trajectory_panel(
    pose_history: list,
    current_idx: int,
    width: int,
    height: int,
    pose_loss_history: list) -> np.ndarray:

    panel = np.zeros((height, width, 3), dtype=np.uint8)

    if len(pose_history) < 2 or current_idx < 1:
        return panel

    poses_so_far = pose_history[:current_idx + 1]
    n = len(poses_so_far)
    losses = pose_loss_history[:current_idx + 1]
    max_loss = max(losses) if losses else 1.0
    min_loss = min(losses) if losses else 0.0
    loss_range = max(max_loss - min_loss, 1e-8)

    traj_w = width // 4
    traj_h = height // 2

    # SPEEDOMETER 
    spd_x0, spd_w = traj_w, width - traj_w*2
    spd = np.zeros((traj_h, spd_w, 3), dtype=np.uint8)
    spd_cx, spd_cy = spd_w // 2, int(traj_h * 0.6)
    spd_r = min(spd_cx, spd_cy) - 15

    speeds = [math.sqrt(float(p[0])**2 + float(p[2])**2) for p in poses_so_far]
    max_speed = max(max(speeds), 1e-6)
    cur_speed = speeds[current_idx]
    speed_ratio = cur_speed / max_speed

    for deg in range(0, 181):
        rad = math.radians(180 - deg)
        px = int(spd_cx + spd_r * math.cos(rad))
        py = int(spd_cy - spd_r * math.sin(rad))
        cv2.circle(spd, (px, py), 2, (60, 60, 60), -1)

    for deg in range(0, int(speed_ratio * 180)):
        rad = math.radians(180 - deg)
        px = int(spd_cx + spd_r * math.cos(rad))
        py = int(spd_cy - spd_r * math.sin(rad))
        hue = int(120 * (1 - speed_ratio))
        color_hsv = np.uint8([[[hue, 255, 220]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
        cv2.circle(spd, (px, py), 3, color_bgr, -1)

    needle_rad = math.radians(180 - speed_ratio * 180)
    nx_n = int(spd_cx + (spd_r - 8) * math.cos(needle_rad))
    ny_n = int(spd_cy - (spd_r - 8) * math.sin(needle_rad))
    cv2.line(spd, (spd_cx, spd_cy), (nx_n, ny_n), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(spd, (spd_cx, spd_cy), 5, (200, 200, 200), -1)
    cv2.putText(spd, f"{cur_speed:.4f}",
                (spd_cx - 25, spd_cy + 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, (220, 220, 220), 1)
    cv2.putText(spd, "SPEED", (spd_cx - 20, spd_cy + 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.3, (150, 150, 150), 1)
    cv2.putText(spd, "SPEEDOMETER", (5, 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, (180, 180, 180), 1)
    panel[:traj_h, spd_x0:spd_x0 + spd_w] = spd

    # Pose loss 
    graph_top    = int(height * 0.75)
    graph_bottom = height - 4
    graph_left   = 6
    graph_right  = width - 6
    ekg_h = graph_bottom - graph_top
    ekg_w = graph_right - graph_left

    if len(losses) > 1:
        max_pl   = max(losses) or 1.0
        min_pl   = min(losses)
        pl_range = max(max_pl - min_pl, 1e-8)

        def loss_to_y(v):
            ratio = (v - min_pl) / pl_range
            return graph_bottom - int(ratio * (ekg_h - 2)) - 1

        def loss_to_color(v):
            ratio = (v - min_pl) / pl_range
            hue = int(120 * (1.0 - ratio))
            hsv = np.uint8([[[hue, 255, 255]]])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()

        pts = []
        for j, v in enumerate(losses):
            px = graph_left + int(j / max(len(losses) - 1, 1) * ekg_w)
            py = loss_to_y(v)
            pts.append((px, py, v))

        cv2.rectangle(panel, (graph_left, graph_top),
                      (graph_right, graph_bottom), (15, 15, 15), -1)

        for px, py, v in pts:
            col = loss_to_color(v)
            cv2.line(panel, (px, py), (px, graph_bottom),
                     [max(0, c - 210) for c in col], 1)

        glow_layer = np.zeros((height, width, 3), dtype=np.uint8)
        for j in range(1, len(pts)):
            x1, y1, v1 = pts[j-1]
            x2, y2, v2 = pts[j]
            col = loss_to_color((v1 + v2) / 2)
            cv2.line(glow_layer, (x1, y1), (x2, y2),
                     [max(0, c - 160) for c in col], 6, cv2.LINE_AA)
        panel = cv2.add(panel, cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=3, sigmaY=3))

        for j in range(1, len(pts)):
            x1, y1, v1 = pts[j-1]
            x2, y2, v2 = pts[j]
            col = loss_to_color((v1 + v2) / 2)
            cv2.line(panel, (x1, y1), (x2, y2),
                     [min(255, c + 40) for c in col], 2, cv2.LINE_AA)
            cv2.line(panel, (x1, y1), (x2, y2),
                     [min(255, c + 120) for c in col], 1, cv2.LINE_AA)

        cx_ekg, cy_ekg, cv_val = pts[-1]
        col = loss_to_color(cv_val)
        bright = [min(255, c + 150) for c in col]
        cv2.circle(panel, (cx_ekg, cy_ekg), 4, bright, -1, cv2.LINE_AA)
        cv2.circle(panel, (cx_ekg, cy_ekg), 7, [c // 2 for c in bright], 1, cv2.LINE_AA)
        cv2.putText(panel, f"pose: {losses[-1]:.5f}",
                    (graph_left + 3, graph_top - 3),
                    cv2.FONT_HERSHEY_DUPLEX, 0.32, bright, 1, cv2.LINE_AA)

    return panel

def save_mask_video(
    seg_logits_orig_list: list,
    seg_logits_hat_list:  list,
    seg_monitor_list:     list,
    pose_list:            list,
    pose_loss_list:       list,
    epoch: int,
    generated_yuv_list:   list = None,   
    out_dir: str = "debug_plots/videos",
    fps: int = 10):

    os.makedirs(out_dir, exist_ok=True)
    if not seg_logits_orig_list:
        return

    first = seg_logits_orig_list[0]
    H, W = first.shape[-2], first.shape[-1]

    panel_w = 420                  
    total_w = W * 2 + panel_w
    bar_h   = 120                  
    total_h = H + bar_h

    max_mon = max(seg_monitor_list) if max(seg_monitor_list) > 0 else 1.0
    out_path = os.path.join(out_dir, f"epoch_{epoch}_diff.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (total_w, total_h))

    flat_poses = []
    for pb in pose_list:
        p = pb[0].cpu().numpy() if isinstance(pb, torch.Tensor) else np.array(pb[0])
        flat_poses.append(p)

    global_step = 0
    for step_i, (lo_batch, lh_batch, mon) in enumerate(
            zip(seg_logits_orig_list, seg_logits_hat_list, seg_monitor_list)):

        B = lo_batch.shape[0]
        gen_batch = generated_yuv_list[step_i] if generated_yuv_list else None

        for b in range(B):
            mask_orig = lo_batch[b].argmax(dim=0).cpu().numpy().astype(np.uint8)
            mask_hat  = lh_batch[b].argmax(dim=0).cpu().numpy().astype(np.uint8)
            orig_bgr  = colorize_mask(mask_orig)

            # Uncertainty heatmap
            prob_hat = torch.softmax(lh_batch[b].float(), dim=0).numpy()
            entropy  = -np.sum(prob_hat * np.log(prob_hat + 1e-8), axis=0)
            entropy_norm = (entropy / np.log(5) * 255).astype(np.uint8)
            uncertainty_bgr = cv2.applyColorMap(entropy_norm, cv2.COLORMAP_INFERNO)

            # Error pixels highlighted with white line
            diff_mask = (mask_orig != mask_hat).astype(np.uint8)
            contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(uncertainty_bgr, contours, -1, (255, 255, 255), 1)

            _put_label(orig_bgr,      "ORIG MASK",    10, 32, 0.75)
            _put_label(uncertainty_bgr,"UNCERTAINTY",  10, 32, 0.75)
            _put_label(uncertainty_bgr,
                       f"step {step_i} | b {b} | err {diff_mask.mean()*100:.1f}%",
                       10, 58, 0.45)

            right = draw_right_panel(
                flat_poses, step_i, panel_w, H,
                pose_loss_list, lo_batch[b], lh_batch[b],
                gen_batch[b] if gen_batch is not None else None)

            bottom = _draw_bottom_bar(mon, max_mon, pose_loss_list, step_i,
                                      total_w, bar_h)

            top_row = np.concatenate([orig_bgr, uncertainty_bgr, right], axis=1)
            frame   = np.concatenate([top_row, bottom], axis=0)
            writer.write(frame)
            global_step += 1

    writer.release()
    print(f"[video-diff] {global_step} frames → {out_path}")


def _put_label(img, text, x, y, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, (x+1, y+1),
                cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_DUPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_bottom_bar(mon, max_mon, pose_loss_list, step_i, total_w, bar_h):
    bottom = np.zeros((bar_h, total_w, 3), dtype=np.uint8)
    bw = 110

    pl = pose_loss_list[step_i] if step_i < len(pose_loss_list) else 0.0
    max_pl = max(pose_loss_list) if pose_loss_list else 1.0

    bottom[:, :bw] = render_bar(mon, max_mon, bw, bar_h, "segnet_loss", (0, 100, 255))
    bottom[:, bw:bw*2] = render_bar(pl, max_pl, bw, bar_h, "posenet_loss", (0, 80, 220))

    for ci, (name, col) in enumerate(zip(
            ["Road", "Lane", "Undrivable", "Car", "Ego-car"], SEG_COLORS)):
        x = bw * 2 + 20 + ci * 100
        cv2.rectangle(bottom, (x, bar_h//2 - 8), (x + 18, bar_h//2 + 8),
                      col[::-1].tolist(), -1)
        _put_label(bottom, name, x + 24, bar_h//2 + 5, 0.42, (220, 220, 220))

    return bottom


def draw_right_panel(pose_history, current_idx, width, height,
                     pose_loss_history, seg_orig_logits, seg_hat_logits,
                     gen_yuv=None):

    panel = np.zeros((height, width, 3), dtype=np.uint8)
    if current_idx < 1:
        return panel
    traj_part = draw_trajectory_panel(
        pose_history, current_idx, width, height, pose_loss_history)
    panel[:height, :] = traj_part[:height, :]
    return panel

def boundary_loss(y_pred, y_true):
    prob = F.softmax(y_pred, dim=1)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3).to(y_pred.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1,1,3,3).to(y_pred.device)
    loss = 0
    
    for c in range(y_pred.shape[1]):
        target_c = (y_true == c).float().unsqueeze(1)
        pred_c = prob[:, c:c+1, :, :]
        
        edge_true = F.conv2d(target_c, sobel_x, padding=1)**2 + F.conv2d(target_c, sobel_y, padding=1)**2
        edge_pred = F.conv2d(pred_c, sobel_x, padding=1)**2 + F.conv2d(pred_c, sobel_y, padding=1)**2
        
        loss += F.mse_loss(edge_pred, edge_true)
        
    return loss / y_pred.shape[1]

def compute_loss(gen, posenet, segnet, dn, mask, pose6, inter_pose6, orig, orig_full,
                 epoch, step,  reset = False):
    """
    orig:  [B, 2, 3, H_GEN, W_GEN] RGB float32
    orig_full:  [B, 2, 3, H_OUT, W_OUT] RGB float32
    mask:  [B, H_GEN, W_GEN] long
    pose6: [B, 6] float32
    """
    B = mask.shape[0]
    
    hat_yuv6_f1, hat_yuv6_f2 = gen(mask, pose6, inter_pose6, reset)   # [B, 6, H_YUV, W_YUV]
    hat_yuv6_f1 = torch.clamp(hat_yuv6_f1, 0, 255)
    hat_yuv6_f2 = torch.clamp(hat_yuv6_f2, 0, 255)
    # Upscale YUV -> Convert to RGB -> downscale
    hat_rgb_large_f1, hat_rgb_large_f2 = no_upscaler(hat_yuv6_f1, hat_yuv6_f2)
        
    # Simulation of evaluation pipeline resizing
    hat_rgb_large_f1 = diff_round(hat_rgb_large_f1)
    hat_rgb_large_f2 = diff_round(hat_rgb_large_f2)
    
    hat_rgb_small_f1 = F.interpolate(hat_rgb_large_f1, size=(384, 512), mode='bilinear')
    hat_rgb_small_f2 = F.interpolate(hat_rgb_large_f2, size=(384, 512), mode='bilinear')
    
    hat_yuv6_for_posenet_f1 = diff_rgb_to_yuv6(hat_rgb_small_f1)
    hat_yuv6_for_posenet_f2 = diff_rgb_to_yuv6(hat_rgb_small_f2)
    
    # ── 3. PoseNet loss —──────────────────
    with torch.no_grad():
        # orig_full: [B, 2, 3, 874, 1164] → permute → [B, 2, 874, 1164, 3]
        orig_full_dn = orig_full.permute(0, 1, 3, 4, 2).contiguous()
        posenet_in_orig, _ = dn.preprocess_input(orig_full_dn)
        # dn.preprocess_input: einops rearrange → F.interpolate bilinear (874→384) → rgb_to_yuv6
        pose_orig = posenet(posenet_in_orig)

    posenet_in_hat = torch.cat([hat_yuv6_for_posenet_f1, hat_yuv6_for_posenet_f2], dim=1)
    pose_hat  = posenet(posenet_in_hat)
    pose_loss = sum(
        (pose_orig[h.name][..., :h.out//2].detach()
         - pose_hat[h.name][..., :h.out//2]).pow(2)
         .mean(dim=tuple(range(1, pose_orig[h.name].ndim)))
        for h in posenet.hydra.heads if h.name == 'pose'
    ).mean()

    # ── 4. SegNet, CE ─────────────────────────────────────────────────────────
    hat_rgb_segnet_f1 = hat_rgb_small_f1
    hat_rgb_segnet_f2 = hat_rgb_small_f2
    hat_pair_dn = torch.stack([hat_rgb_segnet_f1, hat_rgb_segnet_f2], dim=1).permute(0, 1, 3, 4, 2).contiguous()
    _, segnet_in_hat = dn.preprocess_input(hat_pair_dn)

    seg_logits = segnet(segnet_in_hat)   # [B, 5, 384, 512]
    seg_ce = F.cross_entropy(seg_logits, mask) # mask: [B, 384, 512]
    edge_l = boundary_loss(seg_logits, mask)
    seg_ce = seg_ce + edge_l
    
    with torch.no_grad():
        orig_pair_dn = orig.permute(0, 1, 3, 4, 2).contiguous()
        _, segnet_in_orig = dn.preprocess_input(orig_pair_dn)
        seg_orig_logits = segnet(segnet_in_orig)

        seg_monitor = (seg_orig_logits.argmax(dim=1) !=
                       seg_logits.argmax(dim=1)).float().mean().item()
        
    pw = 200
    sw = 10

    loss = (
        pw  * pose_loss +
        sw  * seg_ce    
    )

    return loss, {
        "pose":      pose_loss.item(),
        "seg_ce":    seg_ce.item(),
        "seg":       seg_monitor,
        "est":       100 * seg_monitor + (10 * pose_loss.item()) ** 0.5 + 0.25,
        "seg_logits_orig": seg_orig_logits.cpu(),
        "seg_logits_hat":  seg_logits.detach().cpu(),
        "gen_yuv_f2": hat_yuv6_f2.detach().cpu()
    }

# --- Score check ──────────────────────────────────────────────────────────────

def run_full_evaluation(
    video_names_file: Path,
    device_str: str,
    script_path: Path,
    archive_path:Path = Path("archive2")) -> Optional[float]:
    
    submission_dir = HERE
    inflated_dir = HERE / "inflated"
    inflated_dir.mkdir(parents=True, exist_ok=True)

    with open(video_names_file) as f:
        video_names = [l.strip() for l in f if l.strip()]

    try:
        for name in video_names:
            base = Path(name).stem
            out_raw = inflated_dir / f"{base}.raw"
            decompress(archive_path, out_raw, device=torch.device("cuda"))
    except Exception as e:
        print(f"  [eval] decompress failed: {e}")
        return None

    result = subprocess.run(
        [sys.executable, str(script_path),
         "--submission-dir", str(submission_dir),
         "--video-names-file", str(video_names_file),
         "--device", device_str,
         "--report", str(submission_dir / "report.txt")],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  [eval] evaluate.py failed:\n{result.stderr[-500:]}")
        return None

    posenet = 0
    segnet = 0
    for line in result.stdout.splitlines():
        if "Average PoseNet" in line:
            posenet = float(line.split(":")[-1].strip())
        if "Average SegNet" in line:
            segnet = float(line.split(":")[-1].strip())
        if "Final score:" in line:
            try:
                return posenet, segnet, float(line.split("=")[-1].strip())
            except ValueError:
                pass
    return None

def train(
    video_path:   Path,
    archive_path: Path,
    device:       torch.device,
    epochs:       int   = 300,
    batch_size:   int   = 8,
    lr:           float = 5e-4,
    save_path:    Path  = Path(MODEL_FILE),
    resume_path:  Optional[Path] = None
    ) -> None:
    print(f"[train] Generator: {Generator.count_params()} params")

    mask_ivf = archive_path / "masks.ivf"
    pose_npy = archive_path / "poses_raw.npy"
    inter_pose_npy = archive_path / "inter_poses_raw.npy"
    if not mask_ivf.exists() or not pose_npy.exists() or not inter_pose_npy.exists():
        raise FileNotFoundError(f"Run --mode compress. missing: {mask_ivf}")

    masks = decode_masks(mask_ivf)          # [N, H_GEN, W_GEN]
    poses = np.load(str(pose_npy))          # [N, 6]
    inter_poses = np.load(str(inter_pose_npy))
    
    orig_half  = read_rgb_frames(video_path, H_GEN, W_GEN) # 384x512
    orig_full  = read_rgb_frames_fullres(video_path)       # 874x1164

    dataset = GenDataset(orig_half, orig_full, masks, poses, inter_poses)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=True)

    gen = Generator().to(device)
    if resume_path and Path(resume_path).exists():
        gen = load_model(Path(resume_path), device, gen=True)
        print(f"[train] Resume from {resume_path}")
    else:
        print("[train] Trainging from ground up")

    dn = DistortionNet().to(device)
    dn.load_state_dicts(Path(ROOT / "models/posenet.safetensors"),
                        Path(ROOT / "models/segnet.safetensors"), device)
    dn.eval()
    for p in dn.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(gen.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = None if (resume_path and Path(resume_path).exists()) else \
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

    use_amp  = device.type == "cuda"
    scaler   = torch.amp.GradScaler("cuda", enabled=use_amp)
    best     = float("inf")
    last_change = 0
    s_mean = 0.01
    p_mean = 0.01
    best_eval_score = 0.71
    eval_script = Path("evaluate.py")
    
    for epoch in range(epochs):
        gen.train()
        sums = {k: 0.0 for k in ("loss","pose","seg_ce","seg","est")}
        n = 0
        reset = True
        s_history = []
        pose_history = []
        seg_logits_orig_ep, seg_logits_hat_ep = [], []
        seg_mon_ep = []
        pose_ep = []
        gen_yuv_ep = []
        
        for step, (mask_b, pose_b, inter_pose_b, orig_b, orig_b_full) in enumerate(loader):
            mask_b = mask_b.to(device)
            pose_b = pose_b.to(device)
            inter_pose_b = inter_pose_b.to(device)
            orig_b = orig_b.to(device)
            orig_b_full = orig_b_full.to(device)
            gen.mask_enc.prev_e = None

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, logs = compute_loss(
                    gen, dn.posenet, dn.segnet, dn, mask_b, pose_b, inter_pose_b,
                    orig_b, orig_b_full, epoch, step, reset=reset)
            reset = False
            
            if logs["pose"] >= p_mean or logs["seg_ce"] >= s_mean:
                loss *= 3

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [warn] NaN epoch={epoch} step={step}")
                optimizer.zero_grad(set_to_none=True)
                reset = True
                continue
            else:
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1)
            scaler.step(optimizer)
            scaler.update()
            
            s_history.append(logs["seg_ce"])
            pose_history.append(logs["pose"])
            seg_logits_orig_ep.append(logs["seg_logits_orig"])
            seg_logits_hat_ep.append(logs["seg_logits_hat"])
            seg_mon_ep.append(logs["seg"])
            pose_ep.append(pose_b.detach().cpu())
            gen_yuv_ep.append(logs["gen_yuv_f2"])

            sums["loss"] += loss.item()
            for k in logs:
                sums[k] = sums.get(k, 0.0) + logs[k]
            n += 1
            
        s_mean = torch.tensor(s_history).mean()
        p_mean = torch.tensor(pose_history).mean()
        
        if epoch % 25 == 0:
            save_mask_video(
                seg_logits_orig_ep, seg_logits_hat_ep,
                seg_mon_ep,
                pose_list=pose_ep,
                pose_loss_list=pose_history,
                epoch=epoch,
                generated_yuv_list=gen_yuv_ep,
                out_dir="debug_plots/videos",
                fps=20,
            )

        if scheduler:
            scheduler.step()

        if n == 0:
            continue

        avg = {k: v/n for k, v in sums.items()}
        with open("log.txt", "a", encoding="utf-8") as f:
            print(
                f"Epoch {epoch}: loss={avg['loss']:.3f}  "
                f"pose={avg['pose']:.4f}  seg={avg['seg']:.4f}  s_ce={avg['seg_ce']:.4f}  "
                f"est≈{avg['est']:.3f}",
                file=f
            )

        if avg["est"] < 1.2:
            save_model(gen, Path("submissions/tomasdousek/archive2/obrdo.br"))
            pl, sl, score = run_full_evaluation(
                video_names_file=Path("public_test_video_names.txt"),
                device_str=str(device),
                script_path=eval_script
            )
            with open("log.txt", "a", encoding="utf-8") as f:
                print(f"posenet = {pl:.4f}  segent = {sl:.4f}  score = {score}", file=f)
            if score is not None:
                if score < best_eval_score:
                    best_eval_score = score
                    save_model(gen, Path("submissions/tomasdousek/best_model/obrdo.br"))
                    save_model(gen, save_path)
                    with open("log.txt", "a", encoding="utf-8") as f:
                        print(f"  ✓ → best_model/obrdo.br, score = {score}", file=f)
                        print(f"  ✓ → {save_path}", file=f)
                    last_change = epoch

        if epoch - last_change >= 30:
            pass

    with open("log.txt", "a", encoding="utf-8") as f:
        print("[train] finished.", file=f)


# ─── Komprese ─────────────────────────────────────────────────────────────────

def compress(video_dir, out_path, model_path=Path(MODEL_FILE), mask_crf=50, compress_for_train=False):
    files = sorted(Path(video_dir).glob("*.mkv"))
    if not files:
        raise FileNotFoundError(f"No .mkv in {video_dir}")

    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    segnet  = SegNet().eval().to(device)
    posenet = PoseNet().eval().to(device)
    segnet.load_state_dict(load_file(ROOT / "models/segnet.safetensors",   device=str(device)))
    posenet.load_state_dict(load_file(ROOT / "models/posenet.safetensors", device=str(device)))
    dn = DistortionNet().to(device)
    dn.load_state_dicts(ROOT / "models/posenet.safetensors", ROOT / "models/segnet.safetensors", device)
    dn.eval()
    for p in list(segnet.parameters()) + list(posenet.parameters()):
        p.requires_grad = False

    all_masks, all_poses, inter_all_poses = [], [], []
    for vid in files:
        print(f"[compress] {vid.name}")
        m, p, inter_p = extract_masks_and_poses(vid, segnet, posenet, dn, device)
        all_masks.append(m); all_poses.append(p); inter_all_poses.append(inter_p)

    all_masks = np.concatenate(all_masks)
    all_poses = np.concatenate(all_poses)
    inter_all_poses = np.concatenate(inter_all_poses)
    
    if compress_for_train:
        mask_ivf = out_path / "masks.ivf"
        encode_masks(all_masks, mask_ivf, crf=mask_crf)
        (out_path / MASK_FILE).write_bytes(_compress_bytes(mask_ivf.read_bytes()))

        buf = io.BytesIO()
        np.save(buf, all_poses)
        (out_path / POSE_FILE).write_bytes(_compress_bytes(buf.getvalue()))
        np.save(str(out_path / "poses_raw.npy"), all_poses)

        buf = io.BytesIO()
        np.save(buf, inter_all_poses)
        (out_path / INTER_POSE_FILE).write_bytes(_compress_bytes(buf.getvalue()))
        np.save(str(out_path / "inter_poses_raw.npy"), inter_all_poses)
    else:
        temp_mask_ivf = out_path / "temp_masks.ivf"
        encode_masks(all_masks, temp_mask_ivf, crf=mask_crf)
        (out_path / MASK_FILE).write_bytes(_compress_bytes(temp_mask_ivf.read_bytes()))
        if temp_mask_ivf.exists():
            temp_mask_ivf.unlink()

        buf = io.BytesIO()
        np.save(buf, all_poses)
        (out_path / POSE_FILE).write_bytes(_compress_bytes(buf.getvalue()))

        buf = io.BytesIO()
        np.save(buf, inter_all_poses)
        (out_path / INTER_POSE_FILE).write_bytes(_compress_bytes(buf.getvalue()))
    
    print(f"  mask: {(out_path/MASK_FILE).stat().st_size/1024:.1f} KB  "
          f"pose: {(out_path/POSE_FILE).stat().st_size/1024:.1f} KB  "
          f"inter pose: {(out_path/INTER_POSE_FILE).stat().st_size/1024:.1f} KB")

    if model_path.exists():
        shutil.copy(model_path, out_path / MODEL_FILE)
        print(f"  generator: {(out_path/MODEL_FILE).stat().st_size/1024:.1f} KB")
    else:
        print(f"  [warn] generator not found: {model_path}")

    with open(out_path / META_FILE, "wb") as f:
        pickle.dump({"H_out": H_OUT, "W_out": W_OUT,
                     "H_gen": H_GEN, "W_gen": W_GEN,
                     "H_yuv": H_YUV, "W_yuv": W_YUV,
                     "n_pairs": len(all_masks)}, f)

    total = sum(f.stat().st_size for f in out_path.iterdir()) / 1024
    rate  = sum(f.stat().st_size for f in out_path.iterdir()
                if f.suffix in (".br", ".bz2", ".pkl")) / 37_545_489
    print(f"[compress] {total:.1f} KB | rate={rate:.5f} | rate×25={rate*25:.3f}")


# ─── Dekomprese ───────────────────────────────────────────────────────────────

@torch.no_grad()
def decompress(compressed_path, output_path, device=torch.device("cuda")):
    print("Inflating to raw.")
    with open(compressed_path / META_FILE, "rb") as f:
        meta = pickle.load(f)
    H_out, W_out = meta["H_out"], meta["W_out"]

    mask_raw = _decompress_bytes((compressed_path / MASK_FILE).read_bytes())
    with tempfile.NamedTemporaryFile(suffix=".ivf", delete=False) as tmp:
        tmp.write(mask_raw); tmp_path = tmp.name
    masks = decode_masks(Path(tmp_path))
    os.remove(tmp_path)

    pose_raw = _decompress_bytes((compressed_path / POSE_FILE).read_bytes())
    poses = np.load(io.BytesIO(pose_raw))
    
    inter_pose_raw = _decompress_bytes((compressed_path / INTER_POSE_FILE).read_bytes())
    inter_poses = np.load(io.BytesIO(inter_pose_raw))

    gen = load_model(compressed_path / MODEL_FILE, device, gen=True)
    gen.eval()

    masks_t = torch.from_numpy(masks).long()
    poses_t = torch.from_numpy(poses).float()
    inter_poses_t = torch.from_numpy(inter_poses).float()

    with open(output_path, "wb") as out_f:
        bs = 8
        for i in range(0, len(masks), bs):
            mb = masks_t[i:i+bs].to(device)
            pb = poses_t[i:i+bs].to(device)
            ipb = inter_poses_t[i:i+bs].to(device)
            reset = False
            if i == 0:
                reset = True
            yuv6_f1, yuv6_f2 = gen(mb, pb, ipb, reset)   # [B, 6, H_YUV, W_YUV]
            
            # uspscale to 874x1164 and convert to rgb
            rgb_f1, rgb_f2 = no_upscaler(yuv6_f1, yuv6_f2)
            
            pair = torch.stack([rgb_f1, rgb_f2], dim=1)  # [B, 2, 3, H, W]
            pair = pair.clamp(0,255).round().to(torch.uint8)
            pair = pair.permute(0,1,3,4,2).cpu().numpy()  # [B, 2, H, W, 3]
            for b in range(pair.shape[0]):
                out_f.write(pair[b,0].tobytes())
                out_f.write(pair[b,1].tobytes())

    print(f"[decompress] → {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--mode", required=True, choices=["train","compress","decompress"])
    p.add_argument("--video_path",      type=Path, default=Path("videos/0.mkv"))
    p.add_argument("--video_dir",       type=Path, default=Path(ROOT / "videos"))
    p.add_argument("--compressed_path", type=Path, default=Path(HERE / "archive"))
    p.add_argument("--output_path",     type=Path, default=Path(HERE / "inflated/0.raw"))
    p.add_argument("--model_path",      type=Path, default=Path(HERE / MODEL_FILE))
    p.add_argument("--resume_path",     type=Path, default=None)
    p.add_argument("--device",          type=str,  default="cuda")
    p.add_argument("--epochs",          type=int,  default=50)
    p.add_argument("--batch_size",      type=int,  default=8)
    p.add_argument("--lr",              type=float,default=3e-6)
    p.add_argument("--mask_crf",        type=int,  default=50)
    p.add_argument("--compress_for_train", type=bool,  default=False)
    args   = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "compress":
        compress(args.video_dir, args.compressed_path,
                 args.model_path, args.mask_crf, args.compress_for_train)
    elif args.mode == "train":
        train(args.video_path, args.compressed_path, device,
              args.epochs, args.batch_size, args.lr,
              args.model_path, args.resume_path)
    elif args.mode == "decompress":
        decompress(args.compressed_path, args.output_path, device)

if __name__ == "__main__":
    main()