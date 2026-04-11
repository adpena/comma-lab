#!/usr/bin/env python
# ============================================================================
# LEGACY — This script predates the tac library and is superseded by:
#   python experiments/train_tac.py --profile proven_baseline
# Unique logic has been migrated to src/tac/. Kept for git history reference.
# ============================================================================
"""
Experiment 3: Post-filter with dilated conv2 (hidden=16, dilation=2).
Based on train_postfilter_canonical.py.
Change: conv2 uses dilation=2 for 11x11 receptive field at zero param cost.
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
ARCHIVE_ZIP = PROJECT / "reports" / "raw" / "2026-04-06-av1-roi-experiments" / "decode_base_archive.zip"
OUTPUT_DIR = PROJECT / "experiments" / "postfilter_weights"
INT8_OUTPUT_NAME = "postfilter_dilated_int8.pt"
FP32_OUTPUT_NAME = "postfilter_dilated_fp32.pt"

sys.path.insert(0, str(UPSTREAM))

import einops
from frame_utils import camera_size, segnet_model_input_size, seq_len
from modules import AllNorm, PoseNet, SegNet

def _patched_allnorm_forward(self, x):
    return self.bn(x.reshape(-1, 1)).reshape(x.shape)
AllNorm.forward = _patched_allnorm_forward

def rgb_to_yuv6_diff(rgb_chw: torch.Tensor) -> torch.Tensor:
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
    batch_size, seq_len_local, *_ = x.shape
    x = einops.rearrange(x, 'b t c h w -> (b t) c h w', b=batch_size, t=seq_len_local, c=3)
    x = F.interpolate(x, size=(segnet_model_input_size[1], segnet_model_input_size[0]), mode='bilinear')
    yuv = rgb_to_yuv6_diff(x)
    return einops.rearrange(yuv, '(b t) c h w -> b (t c) h w',
                            b=batch_size, t=seq_len_local, c=6).contiguous()
PoseNet.preprocess_input = _patched_posenet_preprocess

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"[postfilter-dilated] device: {DEVICE}")


class PostFilter(nn.Module):
    """hidden=16, conv2 with dilation=2 for wider receptive field."""
    def __init__(self, hidden=16, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        # Dilated middle layer: dilation=2, padding=2 to maintain spatial dims
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=False)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


def yuv420_to_rgb_tensor(frame) -> torch.Tensor:
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


def decode_video(path, target_h=874, target_w=1164):
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


def decode_archive(archive_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(tmpdir)
        mkv = list(Path(tmpdir).glob("*.mkv"))[0]
        return decode_video(str(mkv))


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


def build_pairs(frames_hwc):
    pairs = []
    for i in range(0, len(frames_hwc) - 1, seq_len):
        if i + seq_len > len(frames_hwc):
            break
        pair = torch.stack(frames_hwc[i:i + seq_len]).unsqueeze(0)
        pairs.append(pair)
    return pairs


def scorer_forward_pair(pair_btchw, posenet, segnet):
    posenet_in = posenet.preprocess_input(pair_btchw)
    posenet_out = posenet(posenet_in)
    segnet_in = segnet.preprocess_input(pair_btchw)
    segnet_out = segnet(segnet_in)
    return posenet_out, segnet_out


def compute_pair_loss(filtered_pair_hwc, gt_pair_hwc, posenet, segnet):
    fx = filtered_pair_hwc.permute(0, 1, 4, 2, 3).contiguous()
    gx = gt_pair_hwc.float().permute(0, 1, 4, 2, 3).contiguous()
    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)
    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)
    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()
    pred_soft = F.softmax(fs_out, dim=1)
    gt_soft = F.softmax(gs_out, dim=1)
    seg_dist = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()
    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), seg_dist.item()


def apply_filter_to_pair(model, pair_uint8, device):
    B, T, H, W, C = pair_uint8.shape
    x = pair_uint8.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
    y = model(x)
    return y.permute(0, 2, 3, 1).reshape(B, T, H, W, C)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def save_model_int8(model, path):
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


def main():
    N_EPOCHS = 100
    TRAIN_SUBSAMPLE = 8
    ACCUM_STEPS = 4

    print(f"[postfilter-dilated] Loading scorer models...")
    posenet, segnet = load_scorers(DEVICE)

    print(f"[postfilter-dilated] Decoding compressed archive...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    print(f"[postfilter-dilated] Compressed frames: {len(comp_frames)}")

    print(f"[postfilter-dilated] Decoding ground truth...")
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    print(f"[postfilter-dilated] GT frames: {len(gt_frames)}")

    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]

    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)
    n_pairs = len(comp_pairs)
    print(f"[postfilter-dilated] {n_pairs} frame pairs")

    del comp_frames, gt_frames
    gc.collect()

    # Keep pairs on CPU, move to device on-the-fly to avoid MPS OOM
    # comp_pairs and gt_pairs stay on CPU

    model = PostFilter(hidden=16, kernel=3).to(DEVICE)
    param_count = count_params(model)
    print(f"[postfilter-dilated] Model: {param_count} params, ~{param_count} bytes int8")
    assert param_count < 50_000, f"Model too large: {param_count} params"

    # Baseline
    print(f"\n[postfilter-dilated] Computing baseline score (no filter)...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for cp, gp in zip(comp_pairs, gt_pairs):
            _, pd, sd = compute_pair_loss(cp.float().to(DEVICE), gp.to(DEVICE), posenet, segnet)
            total_pose += pd
            total_seg += sd
    baseline_pose = total_pose / n_pairs
    baseline_seg = total_seg / n_pairs
    baseline_loss = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)
    print(f"[postfilter-dilated] Baseline: loss={baseline_loss:.4f}, pose={baseline_pose:.6f}, seg={baseline_seg:.6f}")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)

    train_size = max(1, n_pairs // TRAIN_SUBSAMPLE)
    print(f"\n[postfilter-dilated] Training: {N_EPOCHS} epochs, {train_size} pairs/epoch, accum={ACCUM_STEPS}")
    print(f"{'epoch':>5} {'loss':>10} {'pose':>12} {'seg':>12} {'lr':>10}")
    print("-" * 55)

    best_loss = float("inf")
    best_state = None

    for epoch in range(N_EPOCHS):
        model.train()
        indices = torch.randperm(n_pairs)[:train_size].tolist()
        epoch_loss = 0.0
        epoch_pose = 0.0
        epoch_seg = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(indices):
            filtered = apply_filter_to_pair(model, comp_pairs[idx].to(DEVICE), DEVICE)
            loss, pd, sd = compute_pair_loss(filtered, gt_pairs[idx].to(DEVICE), posenet, segnet)
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

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval
    print(f"\n[postfilter-dilated] Final evaluation on ALL {n_pairs} pairs...")
    model.eval()
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for cp, gp in zip(comp_pairs, gt_pairs):
            filtered = apply_filter_to_pair(model, cp.to(DEVICE), DEVICE)
            _, pd, sd = compute_pair_loss(filtered, gp.to(DEVICE), posenet, segnet)
            total_pose += pd
            total_seg += sd
    final_pose = total_pose / n_pairs
    final_seg = total_seg / n_pairs
    final_loss = 100.0 * final_seg + math.sqrt(10.0 * final_pose)

    print(f"\n{'=' * 60}")
    print(f"RESULTS (dilated, hidden=16, dilation=2)")
    print(f"{'=' * 60}")
    print(f"Baseline: loss={baseline_loss:.4f}  pose={baseline_pose:.6f}  seg={baseline_seg:.6f}")
    print(f"Filtered: loss={final_loss:.4f}  pose={final_pose:.6f}  seg={final_seg:.6f}")
    delta = final_loss - baseline_loss
    print(f"Delta:    {delta:+.4f}")
    if delta < 0:
        print(f"*** IMPROVEMENT: {-delta:.4f} points ***")
    else:
        print(f"*** NO IMPROVEMENT ***")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32_path = OUTPUT_DIR / FP32_OUTPUT_NAME
    torch.save(model.state_dict(), fp32_path)
    print(f"\nSaved fp32:  {fp32_path} ({os.path.getsize(fp32_path)} bytes)")
    int8_path = OUTPUT_DIR / INT8_OUTPUT_NAME
    int8_size = save_model_int8(model, int8_path)
    print(f"Saved int8:  {int8_path} ({int8_size} bytes)")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
