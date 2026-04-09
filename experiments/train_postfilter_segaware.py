#!/usr/bin/env python
"""
Train a SegNet-aware learned post-filter for the comma video compression challenge.

Key difference from train_postfilter_canonical.py:
  - Pre-computes SegNet class probability maps on ground truth
  - Uses KL-divergence between filtered/GT SegNet class probabilities
  - This directly preserves SegNet semantic boundaries instead of the
    softmax-dot-product surrogate that doesn't correlate well with the
    real argmax-based SegNet metric

Usage:
  uv run --with torch --with safetensors --with timm --with einops \
         --with segmentation-models-pytorch --with av \
         experiments/train_postfilter_segaware.py --hidden 16

  uv run --with torch --with safetensors --with timm --with einops \
         --with segmentation-models-pytorch --with av \
         experiments/train_postfilter_segaware.py --hidden 32
"""

import argparse
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

DEFAULT_HIDDEN = 16
DEFAULT_KERNEL = 3

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
print(f"[segaware] device: {DEVICE}")


# ── tiny post-filter model ────────────────────────────────────────────────
class PostFilter(nn.Module):
    """
    Tiny residual CNN post-filter.
    3 conv layers: 3->h->h->3 with residual connection.
    """
    def __init__(self, hidden=16, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=False)

        # Init conv3 near zero so initial output ~ input
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        """x: (B, 3, H, W) float32 in [0, 255]. Returns same."""
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


def normalize_postfilter_meta(hidden: int, kernel: int) -> dict[str, int | str]:
    return {
        "variant": "segaware",
        "hidden": int(hidden),
        "kernel": int(kernel),
    }


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


# ── pre-compute GT SegNet probability maps ────────────────────────────────
def precompute_gt_segnet_probs(gt_pairs, segnet, device):
    """
    Run SegNet on all GT pairs and cache the log-probability maps.
    Returns a list of tensors, each (1, 5, H_seg, W_seg) log-softmax.
    """
    print(f"[segaware] Pre-computing GT SegNet probability maps...")
    gt_seg_logprobs = []
    with torch.no_grad():
        for gp in gt_pairs:
            # (1, 2, H, W, 3) -> (1, 2, C, H, W)
            gx = gp.float().permute(0, 1, 4, 2, 3).contiguous()
            seg_in = segnet.preprocess_input(gx)
            seg_out = segnet(seg_in)  # (1, 5, H_seg, W_seg) logits
            # Store log-probabilities for KL-div (target)
            gt_seg_logprobs.append(F.log_softmax(seg_out, dim=1).cpu())
    print(f"[segaware] Cached {len(gt_seg_logprobs)} GT SegNet log-prob maps")
    return gt_seg_logprobs


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


def compute_pair_loss_segaware(filtered_pair_hwc, gt_pair_hwc, gt_seg_logprob,
                                posenet, segnet, seg_weight=50.0):
    """
    Compute differentiable loss using direct SegNet KL-divergence.

    - pose_loss: MSE between filtered/GT PoseNet outputs (same as before)
    - seg_loss: KL-div between filtered SegNet class probs and GT SegNet class probs
      This directly measures how much the filter corrupts SegNet's semantic map.

    Both pair inputs are (1, 2, H, W, 3) float tensors.
    gt_seg_logprob is (1, 5, H_seg, W_seg) pre-computed log-softmax from GT.
    Returns (loss, pose_dist_scalar, seg_dist_scalar).
    """
    # (B, T, H, W, C) -> (B, T, C, H, W)
    fx = filtered_pair_hwc.permute(0, 1, 4, 2, 3).contiguous()
    gx = gt_pair_hwc.float().permute(0, 1, 4, 2, 3).contiguous()

    # Forward through both scorers on filtered frames
    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)

    # PoseNet: compare against GT
    with torch.no_grad():
        gp_out = posenet(posenet.preprocess_input(gx))

    # PoseNet distortion: MSE of first 6 pose outputs
    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()

    # SegNet distortion: KL divergence between filtered probs and GT probs
    # F.kl_div expects input=log_probs, target=probs (with log_target=False)
    # or input=log_probs, target=log_probs (with log_target=True)
    filtered_log_probs = F.log_softmax(fs_out, dim=1)
    gt_log_probs_dev = gt_seg_logprob.to(fx.device)

    # KL(GT || filtered) = sum(GT_prob * (log(GT_prob) - log(filtered_prob)))
    # Using log_target=True since both are log-probabilities
    seg_kl = F.kl_div(filtered_log_probs, gt_log_probs_dev,
                      reduction='batchmean', log_target=True)

    # Also compute the actual SegNet argmax mismatch for monitoring
    with torch.no_grad():
        gt_seg_out_dev = gt_log_probs_dev.exp()  # convert back to probs for argmax
        seg_argmax_dist = (fs_out.argmax(dim=1) != gt_seg_out_dev.argmax(dim=1)).float().mean().item()

    # Combined loss:
    # - pose_loss scaled by 1.0 (critical, dominates the score)
    # - seg_loss scaled by seg_weight (direct KL preservation)
    pose_loss = 1.0 * torch.sqrt(10.0 * pose_dist + 1e-8)
    seg_loss = seg_weight * seg_kl

    total_loss = pose_loss + seg_loss

    return total_loss, pose_dist.item(), seg_argmax_dist, seg_kl.item()


def compute_pair_eval(filtered_pair_hwc, gt_pair_hwc, posenet, segnet):
    """
    Compute the REAL scorer metrics for evaluation (not training).
    Returns (pose_dist, seg_dist) matching the actual scoring formula.
    """
    fx = filtered_pair_hwc.permute(0, 1, 4, 2, 3).contiguous()
    gx = gt_pair_hwc.float().permute(0, 1, 4, 2, 3).contiguous()

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)

    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()
    seg_dist = (fs_out.argmax(dim=1) != gs_out.argmax(dim=1)).float().mean()

    return pose_dist.item(), seg_dist.item()


# ── apply filter ──────────────────────────────────────────────────────────
def apply_filter_to_pair(model, pair_uint8, device):
    """
    Apply post-filter to a single pair.
    pair_uint8: (1, 2, H, W, 3) uint8 on device.
    Returns (1, 2, H, W, 3) float with grad.
    """
    B, T, H, W, C = pair_uint8.shape
    x = pair_uint8.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
    y = model(x)
    return y.permute(0, 2, 3, 1).reshape(B, T, H, W, C)


# ── utils ─────────────────────────────────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def save_model_int8(model, path, *, meta=None):
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
    if meta is not None:
        state["__meta__"] = dict(meta)
    torch.save(state, path)
    return os.path.getsize(path)


# ── main ──────────────────────────────────────────────────────────────────
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SegNet-aware learned post-filter.")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN,
                        help="Hidden channel count (default: 16)")
    parser.add_argument("--kernel", type=int, default=DEFAULT_KERNEL,
                        help="Kernel size (default: 3)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs.")
    parser.add_argument("--seg-weight", type=float, default=50.0,
                        help="Weight for SegNet KL-divergence loss (default: 50.0)")
    parser.add_argument("--train-subsample", type=int, default=8,
                        help="Train on every Nth pair per epoch.")
    parser.add_argument("--accum-steps", type=int, default=4,
                        help="Gradient accumulation steps.")
    return parser


def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)
    meta = normalize_postfilter_meta(args.hidden, args.kernel)
    n_epochs = args.epochs
    train_subsample = args.train_subsample
    accum_steps = args.accum_steps
    seg_weight = args.seg_weight

    tag = f"h{args.hidden}"
    int8_output = f"postfilter_segaware_{tag}_int8.pt"
    fp32_output = f"postfilter_segaware_{tag}_fp32.pt"

    print(f"[segaware] Loading scorer models...")
    posenet, segnet = load_scorers(DEVICE)
    print(f"[segaware] Scorers loaded on {DEVICE}")

    # Decode frames
    print(f"[segaware] Decoding compressed archive...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    print(f"[segaware] Compressed frames: {len(comp_frames)}")

    print(f"[segaware] Decoding ground truth...")
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    print(f"[segaware] GT frames: {len(gt_frames)}")

    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]

    # Build pairs and move to device
    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)
    n_pairs = len(comp_pairs)
    print(f"[segaware] {n_pairs} frame pairs")

    del comp_frames, gt_frames
    gc.collect()

    comp_pairs = [p.to(DEVICE) for p in comp_pairs]
    gt_pairs = [p.to(DEVICE) for p in gt_pairs]

    # Pre-compute GT SegNet probability maps (cached on CPU to save GPU memory)
    gt_seg_logprobs = precompute_gt_segnet_probs(gt_pairs, segnet, DEVICE)

    # Initialize model
    model = PostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    param_count = count_params(model)
    print(f"[segaware] Model: {param_count} params, ~{param_count} bytes int8")
    assert param_count < 50_000, f"Model too large: {param_count} params"

    # Compute baseline score (no filter) using REAL metrics
    print(f"\n[segaware] Computing baseline score (no filter)...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for cp, gp in zip(comp_pairs, gt_pairs):
            pd, sd = compute_pair_eval(cp.float(), gp, posenet, segnet)
            total_pose += pd
            total_seg += sd
    baseline_pose = total_pose / n_pairs
    baseline_seg = total_seg / n_pairs
    baseline_score = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)
    print(f"[segaware] Baseline: score={baseline_score:.4f}, pose={baseline_pose:.6f}, seg={baseline_seg:.6f}")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    train_size = max(1, n_pairs // train_subsample)
    print(
        f"\n[segaware] Training: {n_epochs} epochs, {train_size} pairs/epoch, accum={accum_steps}, "
        f"hidden={args.hidden}, kernel={args.kernel}, seg_weight={seg_weight}"
    )
    print(f"{'epoch':>5} {'loss':>10} {'pose':>12} {'seg_argmax':>12} {'seg_kl':>12} {'lr':>10}")
    print("-" * 67)

    best_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(n_pairs)[:train_size].tolist()

        epoch_loss = 0.0
        epoch_pose = 0.0
        epoch_seg_argmax = 0.0
        epoch_seg_kl = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(indices):
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            loss, pd, seg_am, seg_kl = compute_pair_loss_segaware(
                filtered, gt_pairs[idx], gt_seg_logprobs[idx],
                posenet, segnet, seg_weight=seg_weight
            )

            (loss / accum_steps).backward()

            epoch_loss += loss.item()
            epoch_pose += pd
            epoch_seg_argmax += seg_am
            epoch_seg_kl += seg_kl

            if (step_i + 1) % accum_steps == 0 or (step_i + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        avg_loss = epoch_loss / len(indices)
        avg_pose = epoch_pose / len(indices)
        avg_seg_am = epoch_seg_argmax / len(indices)
        avg_seg_kl = epoch_seg_kl / len(indices)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch + 1:>5} {avg_loss:>10.4f} {avg_pose:>12.6f} {avg_seg_am:>12.6f} {avg_seg_kl:>12.6f} {lr:>10.6f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final full evaluation using REAL metrics (argmax-based SegNet, MSE PoseNet)
    print(f"\n[segaware] Final evaluation on ALL {n_pairs} pairs...")
    model.eval()
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for cp, gp in zip(comp_pairs, gt_pairs):
            filtered = apply_filter_to_pair(model, cp, DEVICE)
            pd, sd = compute_pair_eval(filtered, gp, posenet, segnet)
            total_pose += pd
            total_seg += sd
    final_pose = total_pose / n_pairs
    final_seg = total_seg / n_pairs
    final_score = 100.0 * final_seg + math.sqrt(10.0 * final_pose)

    print(f"\n{'=' * 70}")
    print(f"RESULTS (h={args.hidden}, seg_weight={seg_weight})")
    print(f"{'=' * 70}")
    print(f"Baseline: score={baseline_score:.4f}  pose={baseline_pose:.6f}  seg={baseline_seg:.6f}")
    print(f"Filtered: score={final_score:.4f}  pose={final_pose:.6f}  seg={final_seg:.6f}")
    pose_delta = final_pose - baseline_pose
    seg_delta = final_seg - baseline_seg
    score_delta = final_score - baseline_score
    print(f"Delta:    score={score_delta:+.4f}  pose={pose_delta:+.6f}  seg={seg_delta:+.6f}")
    if score_delta < 0:
        print(f"*** IMPROVEMENT: {-score_delta:.4f} points ***")
    else:
        print(f"*** NO IMPROVEMENT ***")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fp32_path = OUTPUT_DIR / fp32_output
    torch.save(model.state_dict(), fp32_path)
    print(f"\nSaved fp32:  {fp32_path} ({os.path.getsize(fp32_path)} bytes)")

    int8_path = OUTPUT_DIR / int8_output
    int8_size = save_model_int8(model, int8_path, meta=meta)
    print(f"Saved int8:  {int8_path} ({int8_size} bytes)")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
