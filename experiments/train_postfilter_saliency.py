#!/usr/bin/env python
# ============================================================================
# LEGACY — This script predates the tac library and is superseded by:
#   python experiments/train_tac.py --profile proven_baseline
# Unique logic has been migrated to src/tac/. Kept for git history reference.
# ============================================================================
"""
Train a saliency-weighted post-filter for the comma video compression challenge.

Key change vs canonical: The reconstruction loss is weighted per-pixel using a
PoseNet saliency map, so the model learns to correct PoseNet-critical pixels
while leaving SegNet-critical pixels alone.

weight = 1.0 + alpha * saliency

Usage:
  uv run --with torch --with safetensors --with timm --with einops \
         --with segmentation-models-pytorch --with av --with numpy \
         experiments/train_postfilter_saliency.py --alpha 10 --hidden 16
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
SALIENCY_PATH = PROJECT / "experiments" / "masks" / "posenet_saliency.npy"
DEFAULT_HIDDEN = 16
DEFAULT_KERNEL = 3

# Add upstream to path so modules.py can import frame_utils
sys.path.insert(0, str(UPSTREAM))

import einops
from frame_utils import camera_size, segnet_model_input_size, seq_len
from modules import AllNorm, PoseNet, SegNet

# ── Monkey-patches for differentiable training ────────────────────────────

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
print(f"[postfilter-saliency] device: {DEVICE}")


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


def normalize_postfilter_meta(hidden: int, kernel: int, alpha: float) -> dict:
    return {
        "variant": "saliency_weighted",
        "hidden": int(hidden),
        "kernel": int(kernel),
        "alpha": float(alpha),
    }


def clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in state_dict.items()}


def init_ema_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return clone_state_dict(model.state_dict())


def update_ema_state(
    ema_state: dict[str, torch.Tensor],
    model: nn.Module,
    decay: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 < decay < 1.0:
        raise ValueError(f"EMA decay must be between 0 and 1, got {decay}")
    model_state = model.state_dict()
    for name, tensor in model_state.items():
        ema_state[name].mul_(decay).add_(tensor.detach(), alpha=1.0 - decay)
    return ema_state


def fake_quantize_weight(weight: torch.Tensor) -> torch.Tensor:
    scale = weight.detach().abs().max() / 127.0
    if float(scale) == 0.0:
        return weight
    quantized = torch.clamp(torch.round(weight / scale), -128, 127) * scale
    return weight + (quantized - weight).detach()


class QATPostFilter(PostFilter):
    """Residual CNN with weight-only fake quantization to better match saved int8 weights."""

    def _conv_qat(self, x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        weight = fake_quantize_weight(conv.weight)
        return F.conv2d(x, weight, conv.bias, stride=conv.stride, padding=conv.padding)

    def forward(self, x):
        residual = self.act(self._conv_qat(x, self.conv1))
        residual = self.act(self._conv_qat(residual, self.conv2))
        residual = self._conv_qat(residual, self.conv3)
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
        pair = torch.stack(frames_hwc[i:i + seq_len]).unsqueeze(0)
        pairs.append(pair)
    return pairs


# ── saliency weight map ──────────────────────────────────────────────────
def load_saliency_weights(alpha: float, n_frames: int, device) -> torch.Tensor:
    """
    Load PoseNet saliency map and build per-pixel weight tensor.
    Returns (n_frames, 1, H, W) weight tensor on device.
    weight = 1.0 + alpha * saliency
    """
    sal = np.load(str(SALIENCY_PATH))  # (30, 874, 1164)
    sal_t = torch.from_numpy(sal).float()  # (30, H, W)
    # Trim or pad to match frame count
    if sal_t.shape[0] < n_frames:
        # Repeat last frame saliency
        pad = sal_t[-1:].expand(n_frames - sal_t.shape[0], -1, -1)
        sal_t = torch.cat([sal_t, pad], dim=0)
    sal_t = sal_t[:n_frames]
    weights = 1.0 + alpha * sal_t  # (n_frames, H, W)
    weights = weights.unsqueeze(1).to(device)  # (n_frames, 1, H, W)
    return weights


# ── scorer forward (single pair) ──────────────────────────────────────────
def scorer_forward_pair(pair_btchw, posenet, segnet):
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

    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), seg_dist.item()


def compute_saliency_reconstruction_loss(filtered_bchw, original_bchw, weight_hw):
    """
    Saliency-weighted pixel reconstruction loss.
    This is an AUXILIARY loss that penalizes pixel changes weighted by
    inverse saliency -- i.e., on LOW-saliency pixels the model should
    make minimal corrections (protecting SegNet), while on HIGH-saliency
    pixels larger corrections are tolerated.

    filtered_bchw, original_bchw: (B, 3, H, W) float
    weight_hw: (B, 1, H, W) per-pixel weight (1 + alpha * saliency)

    Returns weighted MSE of the correction residual, encouraging small
    corrections everywhere but allowing larger ones where saliency is high.
    """
    residual = filtered_bchw - original_bchw  # the correction
    # On high-saliency pixels, weight > 1, so the model is *allowed* to correct more
    # (the reconstruction penalty is focused on getting these right).
    # On low-saliency pixels, weight ~ 1, so overcorrection is penalized equally.
    #
    # But we want the OPPOSITE for the protection term: penalize corrections on
    # low-saliency pixels MORE. So we use inverse weighting for the protection loss.
    inv_weight = 1.0 / weight_hw  # high saliency -> low penalty for correction
    weighted_residual_sq = inv_weight * residual.pow(2)
    return weighted_residual_sq.mean()


def compute_combined_loss(filtered_pair_hwc, gt_pair_hwc, comp_pair_hwc,
                          posenet, segnet, sal_weights_pair, sal_lambda):
    """
    Combined loss:
    1. Standard scorer loss (PoseNet + SegNet) -- same as canonical
    2. Saliency-weighted reconstruction penalty -- protects SegNet pixels

    sal_weights_pair: (2, 1, H, W) saliency weights for this pair's frames
    sal_lambda: weight for the saliency reconstruction term
    """
    # Standard scorer loss
    scorer_loss, pose_dist, seg_dist = compute_pair_loss(
        filtered_pair_hwc, gt_pair_hwc, posenet, segnet
    )

    # Saliency-weighted reconstruction penalty on frame 1 only
    # SegNet only uses the last frame (index 1), so only protect that one
    B, T, H, W, C = filtered_pair_hwc.shape
    filtered_bchw = filtered_pair_hwc[:, 1].permute(0, 3, 1, 2)  # frame 1 only
    comp_bchw = comp_pair_hwc[:, 1].float().permute(0, 3, 1, 2)

    sal_recon_loss = compute_saliency_reconstruction_loss(
        filtered_bchw, comp_bchw, sal_weights_pair[1:2]  # frame 1 saliency only
    )

    total_loss = scorer_loss + sal_lambda * sal_recon_loss
    return total_loss, scorer_loss.item(), pose_dist, seg_dist, sal_recon_loss.item()


# ── apply filter ──────────────────────────────────────────────────────────
def apply_filter_to_pair(model, pair_uint8, device):
    B, T, H, W, C = pair_uint8.shape
    x = pair_uint8.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
    y = model(x)
    return y.permute(0, 2, 3, 1).reshape(B, T, H, W, C)


# ── utils ─────────────────────────────────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def save_model_int8(model, path, *, meta=None, per_channel: bool = False):
    state = {}
    for name, param in model.state_dict().items():
        p = param.detach().cpu().float()
        if per_channel and p.ndim >= 2 and not name.endswith("bias"):
            flattened = p.reshape(p.shape[0], -1)
            scale = flattened.abs().max(dim=1).values / 127.0
            scale[scale == 0] = 1.0
            shape = [p.shape[0]] + [1] * (p.ndim - 1)
            quantized = (p / scale.view(*shape)).round().clamp(-128, 127).to(torch.int8)
            state[name + ".q"] = quantized
            state[name + ".s"] = scale
        elif per_channel and name.endswith("bias"):
            # Biases are tiny; keep them in fp32 rather than quantizing away useful signal.
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


# ── main ──────────────────────────────────────────────────────────────────
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train saliency-weighted post-filter."
    )
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Saliency emphasis: weight = 1 + alpha * saliency")
    parser.add_argument("--sal-lambda", type=float, default=0.1,
                        help="Weight for the saliency reconstruction penalty")
    parser.add_argument("--train-subsample", type=int, default=8)
    parser.add_argument("--eval-subsample", type=int, default=4,
                        help="Evaluate on every Nth pair (1=all, 4=25%%)")
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--step-log-every", type=int, default=0,
                        help="Emit progress every N training steps inside an epoch (0 disables).")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="Enable EMA checkpointing when set between 0 and 1.")
    parser.add_argument("--qat-weight-only", action="store_true",
                        help="Apply weight-only fake quantization during training to better match deployed int8 weights.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Output tag. Default: saliency_alpha{N}")
    return parser


def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)
    alpha = args.alpha
    sal_lambda = args.sal_lambda
    tag = args.tag or f"saliency_alpha{int(alpha)}"
    if args.hidden != DEFAULT_HIDDEN:
        tag += f"_h{args.hidden}"
    meta = normalize_postfilter_meta(args.hidden, args.kernel, alpha)
    n_epochs = args.epochs
    train_subsample = args.train_subsample
    eval_subsample = args.eval_subsample
    accum_steps = args.accum_steps
    step_log_every = args.step_log_every
    ema_decay = args.ema_decay

    int8_output = f"postfilter_{tag}_int8.pt"
    fp32_output = f"postfilter_{tag}_fp32.pt"

    print(f"[postfilter-saliency] alpha={alpha}, sal_lambda={sal_lambda}, "
          f"hidden={args.hidden}, tag={tag}")
    print(f"[postfilter-saliency] Output: {int8_output}")

    print(f"[postfilter-saliency] Loading scorer models...")
    posenet, segnet = load_scorers(DEVICE)
    print(f"[postfilter-saliency] Scorers loaded on {DEVICE}")

    # Decode frames
    print(f"[postfilter-saliency] Decoding compressed archive...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    print(f"[postfilter-saliency] Compressed frames: {len(comp_frames)}")

    print(f"[postfilter-saliency] Decoding ground truth...")
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    print(f"[postfilter-saliency] GT frames: {len(gt_frames)}")

    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]

    # Load saliency weights
    print(f"[postfilter-saliency] Loading saliency map (alpha={alpha})...")
    sal_all = load_saliency_weights(alpha, n, DEVICE)  # (n_frames, 1, H, W)
    sal_stats = sal_all.mean().item()
    sal_max = sal_all.max().item()
    print(f"[postfilter-saliency] Saliency weights: mean={sal_stats:.3f}, max={sal_max:.1f}")

    # Build saliency weight pairs matching frame pairs
    # Each pair uses frames [i, i+1], so saliency pair uses same indices
    sal_pairs = []
    for i in range(0, n - 1, seq_len):
        if i + seq_len > n:
            break
        sp = sal_all[i:i + seq_len]  # (2, 1, H, W)
        sal_pairs.append(sp)

    # Build pairs and move to device
    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)
    n_pairs = len(comp_pairs)
    print(f"[postfilter-saliency] {n_pairs} frame pairs")

    del comp_frames, gt_frames, sal_all
    gc.collect()

    comp_pairs = [p.to(DEVICE) for p in comp_pairs]
    gt_pairs = [p.to(DEVICE) for p in gt_pairs]

    # Initialize model
    model_cls = QATPostFilter if args.qat_weight_only else PostFilter
    model = model_cls(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    param_count = count_params(model)
    print(f"[postfilter-saliency] Model: {param_count} params, ~{param_count} bytes int8")
    assert param_count < 50_000, f"Model too large: {param_count} params"

    # Compute baseline score (no filter) -- subsampled for speed
    eval_indices = list(range(0, n_pairs, eval_subsample))
    n_eval = len(eval_indices)
    print(f"\n[postfilter-saliency] Computing baseline score (no filter) on {n_eval}/{n_pairs} pairs...")
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            _, pd, sd = compute_pair_loss(comp_pairs[idx].float(), gt_pairs[idx], posenet, segnet)
            total_pose += pd
            total_seg += sd
    baseline_pose = total_pose / n_eval
    baseline_seg = total_seg / n_eval
    baseline_loss = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)
    print(f"[postfilter-saliency] Baseline: loss={baseline_loss:.4f}, "
          f"pose={baseline_pose:.6f}, seg={baseline_seg:.6f}")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5
    )
    ema_state = init_ema_state(model) if 0.0 < ema_decay < 1.0 else None

    train_size = max(1, n_pairs // train_subsample)
    print(
        f"\n[postfilter-saliency] Training: {n_epochs} epochs, {train_size} pairs/epoch, "
        f"accum={accum_steps}, hidden={args.hidden}, alpha={alpha}, sal_lambda={sal_lambda}, "
        f"ema={ema_decay if ema_state is not None else 'off'}, "
        f"qat={'on' if args.qat_weight_only else 'off'}"
    )
    print(f"{'epoch':>5} {'total':>10} {'scorer':>10} {'pose':>12} {'seg':>12} "
          f"{'sal_recon':>10} {'lr':>10}")
    print("-" * 75)

    best_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(n_pairs)[:train_size].tolist()

        epoch_loss = 0.0
        epoch_scorer = 0.0
        epoch_pose = 0.0
        epoch_seg = 0.0
        epoch_sal_recon = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(indices):
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            total_loss, scorer_loss, pd, sd, sal_recon = compute_combined_loss(
                filtered, gt_pairs[idx], comp_pairs[idx],
                posenet, segnet, sal_pairs[idx], sal_lambda
            )

            (total_loss / accum_steps).backward()

            epoch_loss += total_loss.item()
            epoch_scorer += scorer_loss
            epoch_pose += pd
            epoch_seg += sd
            epoch_sal_recon += sal_recon

            if (step_i + 1) % accum_steps == 0 or (step_i + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if ema_state is not None:
                    update_ema_state(ema_state, model, ema_decay)
                optimizer.zero_grad()

            if step_log_every > 0 and ((step_i + 1) % step_log_every == 0 or (step_i + 1) == len(indices)):
                print(
                    f"[postfilter-saliency] epoch {epoch + 1}/{n_epochs} step {step_i + 1}/{len(indices)} "
                    f"total={total_loss.item():.4f} scorer={scorer_loss:.4f} pose={pd:.6f} seg={sd:.6f}",
                    flush=True,
                )

        scheduler.step()

        avg_loss = epoch_loss / len(indices)
        avg_scorer = epoch_scorer / len(indices)
        avg_pose = epoch_pose / len(indices)
        avg_seg = epoch_seg / len(indices)
        avg_sal_recon = epoch_sal_recon / len(indices)

        # Track best by SCORER loss (not total), since that's what matters
        if avg_scorer < best_loss:
            best_loss = avg_scorer
            best_state = clone_state_dict(ema_state) if ema_state is not None else clone_state_dict(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch + 1:>5} {avg_loss:>10.4f} {avg_scorer:>10.4f} "
                  f"{avg_pose:>12.6f} {avg_seg:>12.6f} {avg_sal_recon:>10.4f} {lr:>10.6f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on eval subset
    print(f"\n[postfilter-saliency] Final evaluation on {n_eval}/{n_pairs} pairs...")
    model.eval()
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for idx in eval_indices:
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            _, pd, sd = compute_pair_loss(filtered, gt_pairs[idx], posenet, segnet)
            total_pose += pd
            total_seg += sd
    final_pose = total_pose / n_eval
    final_seg = total_seg / n_eval
    final_loss = 100.0 * final_seg + math.sqrt(10.0 * final_pose)

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {tag}")
    print(f"{'=' * 70}")
    print(f"Baseline: loss={baseline_loss:.4f}  pose={baseline_pose:.6f}  seg={baseline_seg:.6f}")
    print(f"Filtered: loss={final_loss:.4f}  pose={final_pose:.6f}  seg={final_seg:.6f}")
    pose_delta = final_pose - baseline_pose
    seg_delta = final_seg - baseline_seg
    delta = final_loss - baseline_loss
    print(f"Delta:    total={delta:+.4f}  pose={pose_delta:+.6f}  seg={seg_delta:+.6f}")
    if delta < 0:
        print(f"*** IMPROVEMENT: {-delta:.4f} points ***")
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

    print(f"\nDone: {tag}")
    return {
        "tag": tag,
        "baseline_loss": baseline_loss,
        "final_loss": final_loss,
        "delta": delta,
        "pose": final_pose,
        "seg": final_seg,
        "pose_delta": pose_delta,
        "seg_delta": seg_delta,
    }


if __name__ == "__main__":
    main()
