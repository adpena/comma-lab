#!/usr/bin/env python
# ============================================================================
# LEGACY — This script predates the tac library and is superseded by:
#   python experiments/train_tac.py --profile proven_baseline
# Unique logic has been migrated to src/tac/. Kept for git history reference.
# ============================================================================
"""
Train a tiny learned post-filter for the comma video compression challenge.

The filter is a small CNN applied after bicubic upscale in the inflate path.
It is trained directly against the scorer's loss function (PoseNet + SegNet).

Usage:
  uv run --with torch --with safetensors --with timm --with einops \
         --with segmentation-models-pytorch --with av \
         experiments/train_postfilter.py
"""

import gc
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
UPSTREAM = PROJECT / "workspace" / "upstream" / "comma_video_compression_challenge"
MODELS_DIR = UPSTREAM / "models"
VIDEOS_DIR = UPSTREAM / "videos"
ARCHIVE_ZIP = PROJECT / "submissions" / "robust_current" / "archive.zip"
OUTPUT_DIR = PROJECT / "experiments" / "postfilter_weights"

# Add upstream to path so modules.py can import frame_utils
sys.path.insert(0, str(UPSTREAM))

import einops
from frame_utils import camera_size, segnet_model_input_size, seq_len
from modules import AllNorm, PoseNet, SegNet

# ── Monkey-patches for differentiable training ────────────────────────────

# AllNorm uses .view() which fails on non-contiguous tensors in backward pass
def _patched_allnorm_forward(self, x):
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


# PoseNet.preprocess_input uses @torch.no_grad rgb_to_yuv6, replace it
def _patched_posenet_preprocess(self, x):
    batch_size, seq_len_local, *_ = x.shape
    x = einops.rearrange(x, 'b t c h w -> (b t) c h w', b=batch_size, t=seq_len_local, c=3)
    x = F.interpolate(x, size=(segnet_model_input_size[1], segnet_model_input_size[0]), mode='bilinear')
    yuv = rgb_to_yuv6_diff(x)
    return einops.rearrange(yuv, '(b t) c h w -> b (t c) h w',
                            b=batch_size, t=seq_len_local, c=6).contiguous()
PoseNet.preprocess_input = _patched_posenet_preprocess


# ── device ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"[postfilter] device: {DEVICE}")


# ── tiny post-filter model ────────────────────────────────────────────────
class PostFilter(nn.Module):
    """
    Tiny residual CNN post-filter.
    3 conv layers: 3->16->16->3 with residual connection.
    ~3.2K params -> ~3.2KB int8.
    """
    def __init__(self, hidden=16, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=False)  # no inplace for autograd safety

        # Init conv3 near zero so initial output ~ input
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        """x: (B, 3, H, W) float32 in [0, 255]. Returns same."""
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


# ── frame decoding ────────────────────────────────────────────────────────
def yuv420_to_rgb_tensor(frame) -> torch.Tensor:
    """BT.601 limited range YUV420->RGB, returns (H, W, 3) uint8 tensor."""
    H, W = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(H, frame.planes[0].line_size)[:, :W]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(H // 2, frame.planes[1].line_size)[:, :W // 2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(H // 2, frame.planes[2].line_size)[:, :W // 2]
    y_t = torch.from_numpy(y.copy()).float()
    u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)
    u_up = F.interpolate(u_t, size=(H, W), mode="bilinear", align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(H, W), mode="bilinear", align_corners=False).squeeze()
    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)
    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)


def decode_video(path: str, target_h: int = 874, target_w: int = 1164) -> list[torch.Tensor]:
    """Decode a video to list of (H, W, 3) uint8 tensors, bicubic upscale if needed."""
    container = av.open(path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        t = yuv420_to_rgb_tensor(frame)
        H, W, _ = t.shape
        if H != target_h or W != target_w:
            x = t.permute(2, 0, 1).unsqueeze(0).float()
            x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
            t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
        frames.append(t)
    container.close()
    return frames


def decode_archive(archive_path: str) -> list[torch.Tensor]:
    """Extract .mkv from archive and decode it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(tmpdir)
        mkv = list(Path(tmpdir).glob("*.mkv"))[0]
        return decode_video(str(mkv))


# ── scorer loading ────────────────────────────────────────────────────────
def load_scorers(device):
    from safetensors.torch import load_file
    posenet = PoseNet().eval().to(device)
    segnet = SegNet().eval().to(device)
    posenet.load_state_dict(load_file(str(MODELS_DIR / "posenet.safetensors"), device=str(device)))
    segnet.load_state_dict(load_file(str(MODELS_DIR / "segnet.safetensors"), device=str(device)))
    for p in posenet.parameters():
        p.requires_grad = False
    for p in segnet.parameters():
        p.requires_grad = False
    return posenet, segnet


# ── build frame pairs ─────────────────────────────────────────────────────
def build_pairs(frames_hwc: list[torch.Tensor]) -> list[torch.Tensor]:
    """Build consecutive frame pairs as (1, 2, H, W, 3) uint8 tensors."""
    pairs = []
    for i in range(0, len(frames_hwc) - 1, seq_len):
        if i + seq_len > len(frames_hwc):
            break
        pair = torch.stack(frames_hwc[i:i + seq_len]).unsqueeze(0)  # (1, 2, H, W, 3)
        pairs.append(pair)
    return pairs


# ── scorer forward (single pair) ──────────────────────────────────────────
def scorer_forward_pair(pair_btchw, posenet, segnet):
    """
    Run one pair through both scorers.
    pair_btchw: (1, 2, C, H, W) float, requires_grad OK.
    Returns (posenet_output_dict, segnet_output_tensor).
    """
    posenet_in = posenet.preprocess_input(pair_btchw)
    posenet_out = posenet(posenet_in)
    segnet_in = segnet.preprocess_input(pair_btchw)
    segnet_out = segnet(segnet_in)
    return posenet_out, segnet_out


def compute_pair_loss(filtered_pair_hwc, gt_pair_hwc, posenet, segnet):
    """
    Compute differentiable loss for one (filtered, gt) pair.
    Both are (1, 2, H, W, 3) float tensors. filtered has grad, gt does not.
    Returns (loss, pose_dist_scalar, seg_dist_scalar).
    """
    # (B, T, H, W, C) -> (B, T, C, H, W)
    fx = filtered_pair_hwc.permute(0, 1, 4, 2, 3).contiguous()
    gx = gt_pair_hwc.float().permute(0, 1, 4, 2, 3).contiguous()

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)

    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

    # PoseNet distortion: MSE of first 6 pose outputs
    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()

    # SegNet distortion: differentiable surrogate
    # Actual: (argmax(pred) != argmax(gt)).float().mean()
    # Surrogate: 1 - sum(softmax(pred) * softmax(gt)) per pixel, averaged
    pred_soft = F.softmax(fs_out, dim=1)
    gt_soft = F.softmax(gs_out, dim=1)
    seg_dist = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()

    # Composite loss matching the actual scoring formula
    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), seg_dist.item()


# ── apply filter ──────────────────────────────────────────────────────────
def apply_filter_to_pair(model, pair_uint8, device):
    """
    Apply post-filter to a single pair.
    pair_uint8: (1, 2, H, W, 3) uint8 on device.
    Returns (1, 2, H, W, 3) float with grad.
    """
    B, T, H, W, C = pair_uint8.shape
    x = pair_uint8.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()  # (2, 3, H, W)
    y = model(x)  # (2, 3, H, W)
    return y.permute(0, 2, 3, 1).reshape(B, T, H, W, C)


# ── utils ─────────────────────────────────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def save_model_int8(model, path):
    """Save model weights quantized to int8."""
    state = {}
    for name, param in model.state_dict().items():
        p = param.detach().cpu().float()
        scale = p.abs().max() / 127.0
        if scale == 0:
            scale = torch.tensor(1.0)
        quantized = (p / scale).round().clamp(-128, 127).to(torch.int8)
        state[name + ".q"] = quantized
        state[name + ".s"] = scale
    torch.save(state, path)
    return os.path.getsize(path)


# ── main ──────────────────────────────────────────────────────────────────
def main():
    N_EPOCHS = 100
    # Subsample: train on every Nth pair to keep epochs fast on CPU/MPS
    TRAIN_SUBSAMPLE = 8  # use 1/8 of pairs per epoch (~75 pairs)
    ACCUM_STEPS = 4      # gradient accumulation steps

    print(f"[postfilter] Loading scorer models...")
    posenet, segnet = load_scorers(DEVICE)
    print(f"[postfilter] Scorers loaded on {DEVICE}")

    # Decode frames
    print(f"[postfilter] Decoding compressed archive...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    print(f"[postfilter] Compressed frames: {len(comp_frames)}")

    print(f"[postfilter] Decoding ground truth...")
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    print(f"[postfilter] GT frames: {len(gt_frames)}")

    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]

    # Build pairs and move to device
    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)
    n_pairs = len(comp_pairs)
    print(f"[postfilter] {n_pairs} frame pairs")

    # Free frame lists
    del comp_frames, gt_frames
    gc.collect()

    # Move pairs to device
    comp_pairs = [p.to(DEVICE) for p in comp_pairs]
    gt_pairs = [p.to(DEVICE) for p in gt_pairs]

    # Initialize model
    model = PostFilter(hidden=16, kernel=3).to(DEVICE)
    param_count = count_params(model)
    print(f"[postfilter] Model: {param_count} params, ~{param_count} bytes int8")
    assert param_count < 50_000, f"Model too large: {param_count} params"

    # Compute baseline score (no filter)
    print(f"\n[postfilter] Computing baseline score (no filter)...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for cp, gp in zip(comp_pairs, gt_pairs):
            _, pd, sd = compute_pair_loss(cp.float(), gp, posenet, segnet)
            total_pose += pd
            total_seg += sd
    baseline_pose = total_pose / n_pairs
    baseline_seg = total_seg / n_pairs
    baseline_loss = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)
    print(f"[postfilter] Baseline: loss={baseline_loss:.4f}, pose={baseline_pose:.6f}, seg={baseline_seg:.6f}")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)

    train_size = max(1, n_pairs // TRAIN_SUBSAMPLE)
    print(f"\n[postfilter] Training: {N_EPOCHS} epochs, {train_size} pairs/epoch, accum={ACCUM_STEPS}")
    print(f"{'epoch':>5} {'loss':>10} {'pose':>12} {'seg':>12} {'lr':>10}")
    print("-" * 55)

    best_loss = float("inf")
    best_state = None

    for epoch in range(N_EPOCHS):
        model.train()
        # Sample a subset of pairs for this epoch
        indices = torch.randperm(n_pairs)[:train_size].tolist()

        epoch_loss = 0.0
        epoch_pose = 0.0
        epoch_seg = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(indices):
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            loss, pd, sd = compute_pair_loss(filtered, gt_pairs[idx], posenet, segnet)

            # Scale loss for gradient accumulation
            (loss / ACCUM_STEPS).backward()

            epoch_loss += loss.item()
            epoch_pose += pd
            epoch_seg += sd

            if (step_i + 1) % ACCUM_STEPS == 0 or (step_i + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        avg_loss = epoch_loss / len(indices)
        avg_pose = epoch_pose / len(indices)
        avg_seg = epoch_seg / len(indices)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch + 1:>5} {avg_loss:>10.4f} {avg_pose:>12.6f} {avg_seg:>12.6f} {lr:>10.6f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final full evaluation
    print(f"\n[postfilter] Final evaluation on ALL {n_pairs} pairs...")
    model.eval()
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for cp, gp in zip(comp_pairs, gt_pairs):
            filtered = apply_filter_to_pair(model, cp, DEVICE)
            _, pd, sd = compute_pair_loss(filtered, gp, posenet, segnet)
            total_pose += pd
            total_seg += sd
    final_pose = total_pose / n_pairs
    final_seg = total_seg / n_pairs
    final_loss = 100.0 * final_seg + math.sqrt(10.0 * final_pose)

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Baseline: loss={baseline_loss:.4f}  pose={baseline_pose:.6f}  seg={baseline_seg:.6f}")
    print(f"Filtered: loss={final_loss:.4f}  pose={final_pose:.6f}  seg={final_seg:.6f}")
    delta = final_loss - baseline_loss
    print(f"Delta:    {delta:+.4f}")
    if delta < 0:
        print(f"*** IMPROVEMENT: {-delta:.4f} points ***")
    else:
        print(f"*** NO IMPROVEMENT ***")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fp32_path = OUTPUT_DIR / "postfilter_fp32.pt"
    torch.save(model.state_dict(), fp32_path)
    print(f"\nSaved fp32:  {fp32_path} ({os.path.getsize(fp32_path)} bytes)")

    int8_path = OUTPUT_DIR / "postfilter_int8.pt"
    int8_size = save_model_int8(model, int8_path)
    print(f"Saved int8:  {int8_path} ({int8_size} bytes)")

    # Save standalone loader module
    loader_path = OUTPUT_DIR / "postfilter_module.py"
    with open(loader_path, "w") as f:
        f.write('''#!/usr/bin/env python
"""Standalone post-filter for integration into inflate.py."""
import torch
import torch.nn as nn


class PostFilter(nn.Module):
    def __init__(self, hidden=16, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


def load_postfilter_int8(path, device="cpu"):
    """Load int8-quantized post-filter weights."""
    state = torch.load(path, map_location=device, weights_only=True)
    float_state = {}
    keys = set(k.rsplit(".", 1)[0] for k in state.keys())
    for key in keys:
        q = state[key + ".q"].float()
        s = state[key + ".s"]
        float_state[key] = q * s
    model = PostFilter()
    model.load_state_dict(float_state)
    return model.eval().to(device)
''')
    print(f"Saved loader: {loader_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
