#!/usr/bin/env python
"""
Fully self-contained Kaggle/Colab training script for PostFilter h=96.

Trains a saliency-weighted 3-layer residual CNN post-filter with:
  - Quantization-Aware Training (QAT) via straight-through estimator
  - Polyak / EMA weight averaging (decay=0.997)
  - Best-checkpoint selection after int8 quantization each epoch
  - Saliency-weighted loss (alpha=20, PoseNet gradient map)

No local imports required. Clones upstream repo, installs deps, decodes video,
loads PoseNet/SegNet, trains, and saves best int8 checkpoint.

Usage on Kaggle/Colab:
  !python cloud_h96_trainer.py --hidden 96 --epochs 2500 --alpha 20

Default output: /kaggle/working/ or /content/drive/MyDrive/postfilter_weights/
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

# ============================================================================
# 0. Environment setup -- clone upstream, install deps
# ============================================================================

def setup_environment():
    """Clone upstream repo and install dependencies if not already present."""
    # Detect environment
    on_kaggle = os.path.exists("/kaggle")
    on_colab = os.path.exists("/content")

    if on_kaggle:
        base_dir = Path("/kaggle/working")
    elif on_colab:
        base_dir = Path("/content")
    else:
        base_dir = Path(tempfile.mkdtemp(prefix="postfilter_"))

    upstream_dir = base_dir / "comma_video_compression_challenge"

    # Clone upstream if not present
    if not upstream_dir.exists():
        print("[setup] Cloning upstream repo...")
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/commaai/comma_video_compression_challenge.git",
            str(upstream_dir),
        ])
        print(f"[setup] Cloned to {upstream_dir}")
    else:
        print(f"[setup] Upstream already at {upstream_dir}")

    # Install dependencies
    # Handle P100 sm_60 by checking GPU arch
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] < 7:
                print(f"[setup] Detected GPU capability sm_{cap[0]}{cap[1]}, "
                      "may need torch 2.2.0+cu118 for P100 support.")
    except ImportError:
        pass

    deps = [
        "av", "safetensors", "timm", "einops",
        "segmentation-models-pytorch", "numpy",
    ]
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            print(f"[setup] Installing {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", dep,
            ])

    return upstream_dir


UPSTREAM_DIR = setup_environment()
MODELS_DIR = UPSTREAM_DIR / "models"
VIDEOS_DIR = UPSTREAM_DIR / "videos"

# Add upstream to sys.path for modules.py imports
sys.path.insert(0, str(UPSTREAM_DIR))

# ============================================================================
# 1. Imports (after setup)
# ============================================================================

import av  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import einops  # noqa: E402

# Import upstream model definitions
from modules import AllNorm, PoseNet, SegNet  # noqa: E402
from frame_utils import camera_size, segnet_model_input_size, seq_len  # noqa: E402

# ============================================================================
# 2. Constants
# ============================================================================

SEQ_LEN = seq_len  # 2
CAMERA_SIZE = camera_size  # (1164, 874)
SEGNET_INPUT_SIZE = segnet_model_input_size  # (512, 384)

# Device selection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"[cloud-trainer] device: {DEVICE}")

# Output directory
if os.path.exists("/kaggle/working"):
    OUTPUT_DIR = Path("/kaggle/working")
elif os.path.exists("/content/drive/MyDrive"):
    OUTPUT_DIR = Path("/content/drive/MyDrive/postfilter_weights")
elif os.path.exists("/content"):
    OUTPUT_DIR = Path("/content/postfilter_weights")
else:
    OUTPUT_DIR = Path("./postfilter_weights")

# ============================================================================
# 3. Monkey-patches for differentiable training
# ============================================================================

def _patched_allnorm_forward(self, x):
    """Make AllNorm work with autograd (no in-place reshape)."""
    return self.bn(x.reshape(-1, 1)).reshape(x.shape)

AllNorm.forward = _patched_allnorm_forward


def rgb_to_yuv6_diff(rgb_chw: torch.Tensor) -> torch.Tensor:
    """Differentiable rgb_to_yuv6 (no @torch.no_grad, no in-place clamp)."""
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, :2 * H2, :2 * W2]
    R = rgb[..., 0, :, :]
    G = rgb[..., 1, :, :]
    B = rgb[..., 2, :, :]
    Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
    U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] +
             U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
    V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] +
             V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
    y00 = Y[..., 0::2, 0::2]
    y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]
    y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)


def _patched_posenet_preprocess(self, x):
    """Differentiable PoseNet preprocessing."""
    batch_size, seq_len_local, *_ = x.shape
    x = einops.rearrange(x, 'b t c h w -> (b t) c h w',
                         b=batch_size, t=seq_len_local, c=3)
    x = F.interpolate(x, size=(SEGNET_INPUT_SIZE[1], SEGNET_INPUT_SIZE[0]),
                       mode='bilinear')
    yuv = rgb_to_yuv6_diff(x)
    return einops.rearrange(yuv, '(b t) c h w -> b (t c) h w',
                            b=batch_size, t=seq_len_local, c=6).contiguous()

PoseNet.preprocess_input = _patched_posenet_preprocess


# ============================================================================
# 4. PostFilter model
# ============================================================================

class PostFilter(nn.Module):
    """
    3-layer residual CNN post-filter.
    Conv: 3 -> hidden -> hidden -> 3 with residual connection and zero-init output.
    """
    def __init__(self, hidden=96, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=False)
        # Zero-init output so initial prediction = input
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        """x: (B, 3, H, W) float32 in [0, 255]. Returns same."""
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


# ============================================================================
# 5. FakeQuantSTE -- straight-through estimator for int8
# ============================================================================

class FakeQuantSTE(torch.autograd.Function):
    """Straight-through estimator for symmetric per-tensor int8 quantization.

    Forward: q = round(clamp(w / s, -128, 127)) * s,  s = max|w| / 127
    Backward: gradient passes through unchanged inside the clamp range,
              zeroed for values that hit the saturation boundary.
    """
    @staticmethod
    def forward(ctx, w):
        with torch.no_grad():
            scale = w.detach().abs().max() / 127.0
            if scale.item() == 0.0:
                ctx.save_for_backward(torch.zeros_like(w, dtype=torch.bool))
                return w
            q = (w / scale).round().clamp(-128.0, 127.0)
            saturated = (q.abs() >= 127.0)
            ctx.save_for_backward(saturated)
            return q * scale

    @staticmethod
    def backward(ctx, grad_out):
        (saturated,) = ctx.saved_tensors
        return grad_out * (~saturated).to(grad_out.dtype)


def fake_quant(t: torch.Tensor) -> torch.Tensor:
    return FakeQuantSTE.apply(t)


class QATPostFilter(nn.Module):
    """PostFilter with fake-quantized weights on every forward pass.

    Mirrors PostFilter architecture so state dict is interchangeable.
    """
    def __init__(self, hidden=96, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=False)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def _qconv(self, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        wq = fake_quant(conv.weight)
        bq = fake_quant(conv.bias) if conv.bias is not None else None
        return F.conv2d(x, wq, bq, padding=conv.padding, stride=conv.stride)

    def forward(self, x):
        residual = self.act(self._qconv(self.conv1, x))
        residual = self.act(self._qconv(self.conv2, residual))
        residual = self._qconv(self.conv3, residual)
        return (x + residual).clamp(0, 255)


# ============================================================================
# 6. EMA weight averaging
# ============================================================================

class EMA:
    """Polyak weight averaging. Stored on the same device as the source."""

    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)


# ============================================================================
# 7. Frame decoding from .mkv via PyAV
# ============================================================================

def yuv420_to_rgb_tensor(frame) -> torch.Tensor:
    """BT.601 limited range YUV420->RGB, returns (H, W, 3) uint8 tensor."""
    H, W = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(
        H, frame.planes[0].line_size)[:, :W]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(
        H // 2, frame.planes[1].line_size)[:, :W // 2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(
        H // 2, frame.planes[2].line_size)[:, :W // 2]
    y_t = torch.from_numpy(y.copy()).float()
    u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)
    u_up = F.interpolate(u_t, size=(H, W), mode="bilinear",
                         align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(H, W), mode="bilinear",
                         align_corners=False).squeeze()
    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)
    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)


def decode_video(path: str, target_h: int = 874,
                 target_w: int = 1164) -> list[torch.Tensor]:
    """Decode a video to list of (H, W, 3) uint8 tensors."""
    container = av.open(path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        t = yuv420_to_rgb_tensor(frame)
        H, W, _ = t.shape
        if H != target_h or W != target_w:
            x = t.permute(2, 0, 1).unsqueeze(0).float()
            x = F.interpolate(x, size=(target_h, target_w),
                              mode="bicubic", align_corners=False)
            t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(
                torch.uint8)
        frames.append(t)
    container.close()
    return frames


# ============================================================================
# 8. Scorer loading from .safetensors
# ============================================================================

def load_scorers(device):
    """Load PoseNet and SegNet from upstream models directory."""
    from safetensors.torch import load_file
    posenet = PoseNet().eval().to(device)
    segnet = SegNet().eval().to(device)
    posenet.load_state_dict(load_file(
        str(MODELS_DIR / "posenet.safetensors"), device=str(device)))
    segnet.load_state_dict(load_file(
        str(MODELS_DIR / "segnet.safetensors"), device=str(device)))
    for p in posenet.parameters():
        p.requires_grad = False
    for p in segnet.parameters():
        p.requires_grad = False
    return posenet, segnet


# ============================================================================
# 9. Saliency map computation (computed from PoseNet gradients)
# ============================================================================

def compute_posenet_saliency(
    frames: list[torch.Tensor],
    posenet: PoseNet,
    device: torch.device,
) -> torch.Tensor:
    """Compute per-pixel PoseNet saliency map from frame pairs.

    For each consecutive pair of frames, compute the gradient of PoseNet
    output w.r.t. the input pixels. The absolute gradient magnitude gives
    a per-pixel saliency map showing which pixels PoseNet is most sensitive to.

    Returns: (n_frames, H, W) float tensor of saliency values.
    """
    n = len(frames)
    H, W = frames[0].shape[0], frames[0].shape[1]
    saliency = torch.zeros(n, H, W)

    print(f"[saliency] Computing PoseNet saliency for {n} frames...")
    posenet.eval()

    for i in range(0, n - 1, SEQ_LEN):
        if i + SEQ_LEN > n:
            break
        # Build pair: (1, 2, H, W, 3) -> (1, 2, 3, H, W)
        pair = torch.stack(frames[i:i + SEQ_LEN]).unsqueeze(0).float().to(device)
        pair.requires_grad_(True)

        # Forward through PoseNet
        pair_chw = pair.permute(0, 1, 4, 2, 3).contiguous()
        posenet_in = posenet.preprocess_input(pair_chw)
        out = posenet(posenet_in)
        pose_vec = out["pose"][..., :6]
        loss = pose_vec.pow(2).sum()
        loss.backward()

        # Gradient magnitude per pixel
        grad = pair.grad.detach().abs()  # (1, 2, H, W, 3)
        for t in range(SEQ_LEN):
            frame_idx = i + t
            # Mean across color channels
            sal = grad[0, t].mean(dim=-1).cpu()  # (H, W)
            # Normalize to [0, 1]
            sal_max = sal.max()
            if sal_max > 0:
                sal = sal / sal_max
            saliency[frame_idx] = sal

    # For the last frame if it wasn't covered
    if n > 1 and saliency[n - 1].sum() == 0:
        saliency[n - 1] = saliency[n - 2]

    print(f"[saliency] Done. Mean={saliency.mean():.4f}, Max={saliency.max():.4f}")
    return saliency


# ============================================================================
# 10. Scoring / loss functions
# ============================================================================

def scorer_forward_pair(pair_btchw, posenet, segnet):
    """Forward both scorers on a (B, T, C, H, W) pair."""
    posenet_in = posenet.preprocess_input(pair_btchw)
    posenet_out = posenet(posenet_in)
    segnet_in = segnet.preprocess_input(pair_btchw)
    segnet_out = segnet(segnet_in)
    return posenet_out, segnet_out


def compute_pair_loss(filtered_pair_hwc, gt_pair_hwc, posenet, segnet):
    """
    Compute differentiable loss for one (filtered, gt) pair.
    Both are (1, 2, H, W, 3) float tensors.
    Returns (loss, pose_dist_scalar, seg_dist_scalar).
    """
    fx = filtered_pair_hwc.permute(0, 1, 4, 2, 3).contiguous()
    gx = gt_pair_hwc.float().permute(0, 1, 4, 2, 3).contiguous()

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)

    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()

    pred_soft = F.softmax(fs_out, dim=1)
    gt_soft = F.softmax(gs_out, dim=1)
    seg_dist = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()

    # Scoring formula: 100*seg + sqrt(10*pose) + 25*rate
    # (rate is constant for post-filter training, so we omit it here)
    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), seg_dist.item()


def compute_saliency_reconstruction_loss(filtered_bchw, original_bchw, weight_hw):
    """
    Saliency-weighted pixel reconstruction loss.
    Penalizes corrections on low-saliency pixels (protecting SegNet).
    """
    residual = filtered_bchw - original_bchw
    inv_weight = 1.0 / weight_hw
    weighted_residual_sq = inv_weight * residual.pow(2)
    return weighted_residual_sq.mean()


def compute_combined_loss(filtered_pair_hwc, gt_pair_hwc, comp_pair_hwc,
                          posenet, segnet, sal_weights_pair, sal_lambda):
    """
    Combined loss:
    1. Standard scorer loss (PoseNet + SegNet)
    2. Saliency-weighted reconstruction penalty (protects SegNet pixels)
    """
    scorer_loss, pose_dist, seg_dist = compute_pair_loss(
        filtered_pair_hwc, gt_pair_hwc, posenet, segnet
    )

    B, T, H, W, C = filtered_pair_hwc.shape
    filtered_bchw = filtered_pair_hwc.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
    comp_bchw = comp_pair_hwc.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2)

    sal_recon_loss = compute_saliency_reconstruction_loss(
        filtered_bchw, comp_bchw, sal_weights_pair
    )

    total_loss = scorer_loss + sal_lambda * sal_recon_loss
    return total_loss, scorer_loss.item(), pose_dist, seg_dist, sal_recon_loss.item()


# ============================================================================
# 11. Apply filter to pair
# ============================================================================

def apply_filter_to_pair(model, pair_uint8, device):
    """Apply model to a (B, T, H, W, C) uint8 pair, return float."""
    B, T, H, W, C = pair_uint8.shape
    x = pair_uint8.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
    if x.device != device:
        x = x.to(device)
    y = model(x)
    return y.permute(0, 2, 3, 1).reshape(B, T, H, W, C)


# ============================================================================
# 12. Int8 save/load utilities
# ============================================================================

def save_model_int8(model, path, *, meta=None, per_channel=False):
    """Save model with symmetric int8 quantization."""
    state = {}
    for name, param in model.state_dict().items():
        p = param.detach().cpu().float()
        if per_channel and p.ndim >= 2 and not name.endswith("bias"):
            flattened = p.reshape(p.shape[0], -1)
            scale = flattened.abs().max(dim=1).values / 127.0
            scale[scale == 0] = 1.0
            shape = [p.shape[0]] + [1] * (p.ndim - 1)
            quantized = (p / scale.view(*shape)).round().clamp(-128, 127).to(
                torch.int8)
            state[name + ".q"] = quantized
            state[name + ".s"] = scale
        elif per_channel and name.endswith("bias"):
            state[name] = p
        else:
            scale = p.abs().max() / 127.0
            if scale == 0:
                scale = torch.tensor(1.0)
            quantized = (p / scale).round().clamp(-128, 127).to(torch.int8)
            state[name + ".q"] = quantized
            state[name + ".s"] = scale
    if meta is not None:
        state["__meta__"] = dict(meta)
    torch.save(state, path)
    return os.path.getsize(path)


def quantize_state_dict_like_saved_int8(
    state_dict: dict[str, torch.Tensor],
    *,
    per_channel: bool = False,
) -> dict[str, torch.Tensor]:
    """Simulate int8 round-trip on a state dict (for eval-time checkpoint selection)."""
    quantized_state: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            quantized_state[name] = tensor.clone()
            continue
        if per_channel and tensor.ndim >= 2 and not name.endswith("bias"):
            flattened = tensor.detach().reshape(tensor.shape[0], -1)
            scale = flattened.abs().max(dim=1).values / 127.0
            scale[scale == 0] = 1.0
            shape = [tensor.shape[0]] + [1] * (tensor.ndim - 1)
            q = torch.clamp(torch.round(tensor / scale.view(*shape)),
                            -128, 127).to(torch.int8)
            quantized_state[name] = (
                q.float() * scale.view(*shape)
            ).to(dtype=tensor.dtype)
            continue
        if per_channel and name.endswith("bias"):
            quantized_state[name] = tensor.clone()
            continue
        scale = tensor.detach().abs().max() / 127.0
        if float(scale) == 0.0:
            quantized_state[name] = tensor.clone()
            continue
        q = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
        quantized_state[name] = (q.float() * scale).to(dtype=tensor.dtype)
    return quantized_state


# ============================================================================
# 13. Utility helpers
# ============================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def normalize_postfilter_meta(hidden, kernel, alpha):
    return {
        "variant": "cloud_qat_ema_saliency",
        "hidden": int(hidden),
        "kernel": int(kernel),
        "alpha": float(alpha),
    }


def build_pair_start_indices(frame_count, pair_len):
    starts = []
    for start in range(0, frame_count - 1, pair_len):
        if start + pair_len > frame_count:
            break
        starts.append(start)
    return starts


def pair_from_frames(frames, start_idx):
    return torch.stack(frames[start_idx:start_idx + SEQ_LEN]).unsqueeze(0)


def saliency_pair_at(base_saliency, *, start_idx, alpha, device):
    """Build saliency weight tensor for a frame pair."""
    slices = []
    last = base_saliency[-1]
    for offset in range(SEQ_LEN):
        frame_idx = start_idx + offset
        if frame_idx < base_saliency.shape[0]:
            sal = base_saliency[frame_idx]
        else:
            sal = last
        slices.append((1.0 + alpha * sal).unsqueeze(0))
    weights = torch.stack(slices, dim=0)
    return weights.to(device)


def autocast_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def save_best_checkpoint(*, model, ema, output_dir, tag, meta, epoch, scorer,
                         shadow_state=None, per_channel_int8=False):
    """Save the best EMA checkpoint in both fp32 and int8 formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / f"postfilter_{tag}_best_fp32.pt"
    int8_path = output_dir / f"postfilter_{tag}_best_int8.pt"
    meta_path = output_dir / f"postfilter_{tag}_best_meta.json"

    source_shadow = shadow_state if shadow_state is not None else ema.shadow
    shadow = {name: tensor.detach().clone()
              for name, tensor in source_shadow.items()}
    torch.save(shadow, fp32_path)

    original_state = {name: tensor.detach().clone()
                      for name, tensor in model.state_dict().items()}
    model.load_state_dict(shadow)
    int8_size = save_model_int8(model, int8_path, meta=meta,
                                per_channel=per_channel_int8)
    model.load_state_dict(original_state)

    payload = {
        "epoch": epoch,
        "scorer": scorer,
        "fp32_path": str(fp32_path),
        "int8_path": str(int8_path),
        "int8_size": int8_size,
        "meta": meta,
    }
    meta_path.write_text(json.dumps(payload, indent=2))
    print(f"[checkpoint] epoch={epoch} scorer={scorer:.4f} "
          f"int8={int8_size} bytes -> {int8_path}")
    return payload


# ============================================================================
# 14. Main training loop
# ============================================================================

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Cloud PostFilter trainer (QAT + EMA + saliency)")
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--alpha", type=float, default=20.0,
                   help="Saliency emphasis: weight = 1 + alpha * saliency")
    p.add_argument("--sal-lambda", type=float, default=0.1,
                   help="Weight for saliency reconstruction penalty")
    p.add_argument("--train-subsample", type=int, default=4,
                   help="Train on 1/N of pairs per epoch")
    p.add_argument("--eval-subsample", type=int, default=4,
                   help="Evaluate on every Nth pair")
    p.add_argument("--accum-steps", type=int, default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--checkpoint-eval-every", type=int, default=10,
                   help="Run int8 eval checkpoint selection every N epochs")
    p.add_argument("--per-channel-int8", action="store_true",
                   help="Use per-channel int8 quantization for conv weights")
    p.add_argument("--cuda-autocast", action="store_true",
                   help="Use fp16 autocast on CUDA for scorer forwards")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--video", type=str, default=None,
                   help="Path to ground-truth video (default: upstream 0.mkv)")
    p.add_argument("--compressed-video", type=str, default=None,
                   help="Path to compressed video to train on. "
                        "If not provided, trains on GT (identity baseline).")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    alpha = args.alpha
    tag = args.tag or f"cloud_qat_h{args.hidden}_a{int(alpha)}"
    meta = normalize_postfilter_meta(args.hidden, args.kernel, alpha)

    print(f"\n{'=' * 70}")
    print(f"Cloud PostFilter Trainer")
    print(f"{'=' * 70}")
    print(f"  hidden={args.hidden}  kernel={args.kernel}  alpha={alpha}")
    print(f"  epochs={args.epochs}  ema={args.ema_decay}  lr={args.lr}")
    print(f"  grad_clip={args.grad_clip}  accum={args.accum_steps}")
    print(f"  device={DEVICE}  tag={tag}")
    print(f"  output_dir={OUTPUT_DIR}")
    print(f"{'=' * 70}\n")

    if args.epochs == 0:
        print("[cloud-trainer] --epochs 0: import check only, exiting.")
        return {"tag": tag, "status": "import_check_ok"}

    # ---- Load scorers ----
    print("[cloud-trainer] Loading PoseNet + SegNet...")
    posenet, segnet = load_scorers(DEVICE)
    print(f"[cloud-trainer] Scorers loaded on {DEVICE}")

    # ---- Decode ground truth video ----
    gt_video_path = args.video or str(VIDEOS_DIR / "0.mkv")
    print(f"[cloud-trainer] Decoding ground truth: {gt_video_path}")
    gt_frames = decode_video(gt_video_path)
    print(f"[cloud-trainer] GT frames: {len(gt_frames)}")

    # ---- Decode compressed video (or use GT as identity baseline) ----
    if args.compressed_video:
        print(f"[cloud-trainer] Decoding compressed: {args.compressed_video}")
        comp_frames = decode_video(args.compressed_video)
        print(f"[cloud-trainer] Compressed frames: {len(comp_frames)}")
    else:
        # No compressed video provided -- use GT as both
        # This means the post-filter learns identity, which is useful for
        # testing the pipeline. In production you'd provide a compressed video.
        print("[cloud-trainer] No --compressed-video provided.")
        print("[cloud-trainer] Using GT as compressed input (identity baseline).")
        print("[cloud-trainer] For real training, encode first then pass "
              "--compressed-video path/to/compressed.mkv")
        comp_frames = gt_frames

    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]

    # ---- Compute saliency map ----
    print("[cloud-trainer] Computing PoseNet saliency map...")
    sal_base = compute_posenet_saliency(gt_frames, posenet, DEVICE)
    print(f"[cloud-trainer] Saliency: mean={sal_base.mean():.4f}, "
          f"max={sal_base.max():.4f}")

    # ---- Build frame pairs ----
    pair_starts = build_pair_start_indices(n, SEQ_LEN)
    n_pairs = len(pair_starts)
    print(f"[cloud-trainer] {n_pairs} frame pairs from {n} frames")

    # ---- Init model ----
    model = QATPostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    eval_model = PostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    param_count = count_params(model)
    print(f"[cloud-trainer] QATPostFilter: {param_count} params "
          f"(~{param_count} bytes int8)")

    ema = EMA(model, decay=args.ema_decay)

    # ---- Baseline evaluation ----
    eval_indices = list(range(0, n_pairs, args.eval_subsample))
    n_eval = len(eval_indices)
    print(f"\n[cloud-trainer] Baseline (no filter) on "
          f"{n_eval}/{n_pairs} pairs...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            start = pair_starts[idx]
            comp_pair = pair_from_frames(comp_frames, start).to(DEVICE)
            gt_pair = pair_from_frames(gt_frames, start).to(DEVICE)
            with autocast_context(DEVICE, args.cuda_autocast):
                _, pd, sd = compute_pair_loss(
                    comp_pair.float(), gt_pair, posenet, segnet)
            total_pose += pd
            total_seg += sd
    baseline_pose = total_pose / n_eval
    baseline_seg = total_seg / n_eval
    baseline_loss = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)
    print(f"[cloud-trainer] Baseline: loss={baseline_loss:.4f}  "
          f"pose={baseline_pose:.6f}  seg={baseline_seg:.6f}")

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def lr_at(epoch_idx):
        if epoch_idx < args.warmup_epochs:
            return (epoch_idx + 1) / max(1, args.warmup_epochs)
        progress = (epoch_idx - args.warmup_epochs) / max(
            1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress)) * (1 - 0.02) + 0.02

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    train_size = max(1, n_pairs // args.train_subsample)
    print(f"\n[cloud-trainer] Training: {args.epochs} epochs, "
          f"{train_size} pairs/epoch")
    print(f"{'epoch':>6} {'total':>10} {'scorer':>10} {'pose':>12} "
          f"{'seg':>12} {'sal_rcon':>10} {'lr':>10} {'best':>10}")
    print("-" * 86)

    best_scorer = float("inf")
    best_shadow_state = None

    # ---- Training loop ----
    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(n_pairs)[:train_size].tolist()
        ep_loss = ep_scorer = ep_pose = ep_seg = ep_sal = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(indices):
            start = pair_starts[idx]
            comp_pair = pair_from_frames(comp_frames, start).to(DEVICE)
            gt_pair = pair_from_frames(gt_frames, start).to(DEVICE)
            sal_pair = saliency_pair_at(
                sal_base, start_idx=start, alpha=alpha, device=DEVICE)

            with autocast_context(DEVICE, args.cuda_autocast):
                filtered = apply_filter_to_pair(model, comp_pair, DEVICE)
                total_loss, scorer_loss, pd, sd, sal_recon = (
                    compute_combined_loss(
                        filtered, gt_pair, comp_pair,
                        posenet, segnet, sal_pair, args.sal_lambda))

            (total_loss / args.accum_steps).backward()
            ep_loss += total_loss.item()
            ep_scorer += scorer_loss
            ep_pose += pd
            ep_seg += sd
            ep_sal += sal_recon

            if ((step_i + 1) % args.accum_steps == 0 or
                    (step_i + 1) == len(indices)):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

        scheduler.step()

        # ---- Checkpoint selection ----
        avg_scorer = ep_scorer / len(indices)
        score_for_checkpoint = avg_scorer

        # Periodically evaluate EMA after int8 quantization
        if ((epoch + 1) % args.checkpoint_eval_every == 0 or
                epoch == 0 or (epoch + 1) == args.epochs):
            eval_state = quantize_state_dict_like_saved_int8(
                ema.shadow, per_channel=args.per_channel_int8)
            eval_model.load_state_dict(eval_state)
            eval_model.eval()
            total_pose_eval, total_seg_eval = 0.0, 0.0
            with torch.no_grad():
                for idx in eval_indices:
                    start = pair_starts[idx]
                    comp_pair = pair_from_frames(
                        comp_frames, start).to(DEVICE)
                    gt_pair = pair_from_frames(gt_frames, start).to(DEVICE)
                    with autocast_context(DEVICE, args.cuda_autocast):
                        filtered = apply_filter_to_pair(
                            eval_model, comp_pair, DEVICE)
                        _, pd, sd = compute_pair_loss(
                            filtered, gt_pair, posenet, segnet)
                    total_pose_eval += pd
                    total_seg_eval += sd
            pose_eval = total_pose_eval / n_eval
            seg_eval = total_seg_eval / n_eval
            score_for_checkpoint = (
                100.0 * seg_eval + math.sqrt(10.0 * pose_eval))

        if epoch == 0 or score_for_checkpoint < best_scorer:
            best_scorer = score_for_checkpoint
            payload = save_best_checkpoint(
                model=model, ema=ema, output_dir=OUTPUT_DIR, tag=tag,
                meta=meta, epoch=epoch + 1, scorer=score_for_checkpoint,
                per_channel_int8=args.per_channel_int8)
            best_shadow_state = {
                name: tensor.detach().clone()
                for name, tensor in ema.shadow.items()
            }

        # ---- Logging ----
        if (epoch + 1) % 50 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
            n_steps = len(indices)
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"{epoch + 1:>6} {ep_loss / n_steps:>10.4f} "
                f"{ep_scorer / n_steps:>10.4f} "
                f"{ep_pose / n_steps:>12.6f} "
                f"{ep_seg / n_steps:>12.6f} "
                f"{ep_sal / n_steps:>10.4f} "
                f"{lr:>10.6f} "
                f"{best_scorer:>10.4f}",
                flush=True,
            )

    # ---- Final evaluation ----
    if best_shadow_state is not None:
        eval_model.load_state_dict(best_shadow_state)
    else:
        ema.copy_to(eval_model)
    eval_model.eval()

    print(f"\n[cloud-trainer] Final eval on best EMA weights "
          f"({n_eval} pairs)...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            start = pair_starts[idx]
            comp_pair = pair_from_frames(comp_frames, start).to(DEVICE)
            gt_pair = pair_from_frames(gt_frames, start).to(DEVICE)
            with autocast_context(DEVICE, args.cuda_autocast):
                filtered = apply_filter_to_pair(eval_model, comp_pair, DEVICE)
                _, pd, sd = compute_pair_loss(
                    filtered, gt_pair, posenet, segnet)
            total_pose += pd
            total_seg += sd
    final_pose = total_pose / n_eval
    final_seg = total_seg / n_eval
    final_loss = 100.0 * final_seg + math.sqrt(10.0 * final_pose)

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {tag}")
    print(f"{'=' * 70}")
    print(f"Baseline: loss={baseline_loss:.4f}  "
          f"pose={baseline_pose:.6f}  seg={baseline_seg:.6f}")
    print(f"Filtered: loss={final_loss:.4f}  "
          f"pose={final_pose:.6f}  seg={final_seg:.6f}")
    delta = final_loss - baseline_loss
    print(f"Delta:    {delta:+.4f}  "
          f"pose={final_pose - baseline_pose:+.6f}  "
          f"seg={final_seg - baseline_seg:+.6f}")
    if delta < 0:
        print(f"*** IMPROVEMENT: {-delta:.4f} points ***")
    else:
        print("*** NO IMPROVEMENT ***")

    # Save final checkpoint alongside best
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_fp32 = OUTPUT_DIR / f"postfilter_{tag}_final_fp32.pt"
    final_int8 = OUTPUT_DIR / f"postfilter_{tag}_final_int8.pt"
    torch.save(eval_model.state_dict(), final_fp32)
    final_int8_size = save_model_int8(eval_model, final_int8, meta=meta,
                                       per_channel=args.per_channel_int8)
    print(f"\nSaved final fp32: {final_fp32}")
    print(f"Saved final int8: {final_int8} ({final_int8_size} bytes)")
    print(f"Best checkpoint:  {OUTPUT_DIR / f'postfilter_{tag}_best_int8.pt'}")

    return {
        "tag": tag,
        "baseline_loss": baseline_loss,
        "final_loss": final_loss,
        "delta": delta,
        "best_scorer": best_scorer,
    }


if __name__ == "__main__":
    main()
