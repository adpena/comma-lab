#!/usr/bin/env python
"""Self-contained Kaggle/Colab trainer for the fixed h32 SegNet attack lane.

This is a standalone variant of the cloud trainer pattern:
  - clones the pinned upstream challenge repo if needed
  - installs runtime deps when they are missing
  - decodes the archive and GT video locally
  - patches the upstream scorers for differentiable training
  - uses the SegNet hard-argmax disagreement loss family
  - keeps the hidden width fixed at 32
  - emits durable best and final metadata JSON artifacts

The module itself stays import-safe for smoke tests. Heavy setup is deferred
until ``main`` runs.
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

import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name == "experiments":
    PROJECT_ROOT = SCRIPT_PATH.parents[1]
else:
    PROJECT_ROOT = SCRIPT_PATH.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.entrypoints import (
    build_postfilter_meta,
    make_fixed_h32_segnet_tag,
    normalize_archive_source_path,
    resolve_cloud_asset_bundle,
    resolve_cloud_base_dir,
    resolve_cloud_output_dir,
    save_best_checkpoint as save_best_checkpoint_shared,
    save_final_artifacts as save_final_artifacts_shared,
)

FIXED_HIDDEN = 32
DEFAULT_KERNEL = 3
DEFAULT_ALPHA = 20.0
SEQ_LEN = 2
CAMERA_SIZE = (1164, 874)
SEGNET_INPUT_SIZE = (512, 384)
UPSTREAM_REPO_URL = "https://github.com/commaai/comma_video_compression_challenge.git"
DEFAULT_ARCHIVE = PROJECT_ROOT / "reports" / "raw" / "2026-04-06-av1-roi-experiments" / "decode_base_archive.zip"
ASSET_BUNDLE = resolve_cloud_asset_bundle(
    PROJECT_ROOT,
    SCRIPT_PATH,
    archive_relative_path="reports/raw/2026-04-06-av1-roi-experiments/decode_base_archive.zip",
    saliency_relative_path="experiments/masks/posenet_saliency.npy",
)
DEFAULT_ARCHIVE = ASSET_BUNDLE["archive_path"]


def select_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability(0)
        except Exception:
            capability = None
        if capability is None or capability[0] >= 7:
            return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = select_device()


def resolve_output_dir() -> Path:
    return resolve_cloud_output_dir(PROJECT_ROOT)


def resolve_base_dir() -> Path:
    base = resolve_cloud_base_dir()
    if base == Path.cwd():
        return Path(tempfile.mkdtemp(prefix="postfilter_"))
    return base
def setup_environment() -> Path:
    """Clone upstream and install runtime deps when necessary."""
    base_dir = resolve_base_dir()
    upstream_dir = base_dir / "comma_video_compression_challenge"

    if not upstream_dir.exists():
        print("[setup] Cloning upstream repo...")
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                UPSTREAM_REPO_URL,
                str(upstream_dir),
            ]
        )
        print(f"[setup] Cloned to {upstream_dir}")
    else:
        print(f"[setup] Using existing upstream at {upstream_dir}")

    deps = [
        "av",
        "einops",
        "numpy",
        "safetensors",
        "segmentation-models-pytorch",
        "timm",
    ]
    for dep in deps:
        module_name = dep.replace("-", "_")
        try:
            __import__(module_name)
        except ImportError:
            print(f"[setup] Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])

    return upstream_dir


def _patch_upstream_scorers(upstream_dir: Path):
    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))

    import einops
    from modules import AllNorm, PoseNet, SegNet

    def _patched_allnorm_forward(self, x):
        return self.bn(x.reshape(-1, 1)).reshape(x.shape)

    AllNorm.forward = _patched_allnorm_forward

    def _patched_posenet_preprocess(self, x):
        batch_size, seq_len_local, *_ = x.shape
        x = einops.rearrange(
            x,
            "b t c h w -> (b t) c h w",
            b=batch_size,
            t=seq_len_local,
            c=3,
        )
        x = F.interpolate(
            x,
            size=(SEGNET_INPUT_SIZE[1], SEGNET_INPUT_SIZE[0]),
            mode="bilinear",
            align_corners=False,
        )
        yuv = rgb_to_yuv6_diff(x)
        return einops.rearrange(
            yuv,
            "(b t) c h w -> b (t c) h w",
            b=batch_size,
            t=seq_len_local,
            c=6,
        ).contiguous()

    PoseNet.preprocess_input = _patched_posenet_preprocess
    return PoseNet, SegNet


def rgb_to_yuv6_diff(rgb_chw: torch.Tensor) -> torch.Tensor:
    """Differentiable RGB -> YUV6 conversion for PoseNet preprocessing."""
    h, w = rgb_chw.shape[-2], rgb_chw.shape[-1]
    h2, w2 = h // 2, w // 2
    rgb = rgb_chw[..., :, : 2 * h2, : 2 * w2]
    r = rgb[..., 0, :, :]
    g = rgb[..., 1, :, :]
    b = rgb[..., 2, :, :]
    y = (r * 0.299 + g * 0.587 + b * 0.114).clamp(0.0, 255.0)
    u = ((b - y) / 1.772 + 128.0).clamp(0.0, 255.0)
    v = ((r - y) / 1.402 + 128.0).clamp(0.0, 255.0)
    u_sub = (u[..., 0::2, 0::2] + u[..., 1::2, 0::2] +
             u[..., 0::2, 1::2] + u[..., 1::2, 1::2]) * 0.25
    v_sub = (v[..., 0::2, 0::2] + v[..., 1::2, 0::2] +
             v[..., 0::2, 1::2] + v[..., 1::2, 1::2]) * 0.25
    y00 = y[..., 0::2, 0::2]
    y10 = y[..., 1::2, 0::2]
    y01 = y[..., 0::2, 1::2]
    y11 = y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, u_sub, v_sub], dim=-3)


class PostFilter(nn.Module):
    """Fixed h32 residual post-filter."""

    def __init__(self, kernel: int = DEFAULT_KERNEL):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, FIXED_HIDDEN, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(FIXED_HIDDEN, FIXED_HIDDEN, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(FIXED_HIDDEN, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=False)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


class FakeQuantSTE(torch.autograd.Function):
    """Straight-through estimator for symmetric per-tensor int8 quantization."""

    @staticmethod
    def forward(ctx, w):
        with torch.no_grad():
            scale = w.detach().abs().max() / 127.0
            if scale.item() == 0.0:
                ctx.save_for_backward(torch.zeros_like(w, dtype=torch.bool))
                return w
            q = (w / scale).round().clamp(-128.0, 127.0)
            saturated = q.abs() >= 127.0
            ctx.save_for_backward(saturated)
            return q * scale

    @staticmethod
    def backward(ctx, grad_out):
        (saturated,) = ctx.saved_tensors
        return grad_out * (~saturated).to(grad_out.dtype)


def fake_quant(t: torch.Tensor) -> torch.Tensor:
    return FakeQuantSTE.apply(t)


class QATPostFilter(nn.Module):
    """Fixed h32 post-filter with fake-quantized weights on each forward."""

    def __init__(self, kernel: int = DEFAULT_KERNEL):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, FIXED_HIDDEN, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(FIXED_HIDDEN, FIXED_HIDDEN, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(FIXED_HIDDEN, 3, kernel, padding=pad, bias=True)
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


class EMA:
    """Polyak weight averaging."""

    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

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


def yuv420_to_rgb_tensor(frame) -> torch.Tensor:
    """BT.601 limited range YUV420->RGB, returns (H, W, 3) uint8 tensor."""
    import numpy as np

    h, w = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(h, frame.planes[0].line_size)[:, :w]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(h // 2, frame.planes[1].line_size)[:, : w // 2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(h // 2, frame.planes[2].line_size)[:, : w // 2]
    y_t = torch.from_numpy(y.copy()).float()
    u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)
    u_up = F.interpolate(u_t, size=(h, w), mode="bilinear", align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(h, w), mode="bilinear", align_corners=False).squeeze()
    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)
    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)


def decode_video(path: str, target_h: int = CAMERA_SIZE[1], target_w: int = CAMERA_SIZE[0]) -> list[torch.Tensor]:
    import av

    container = av.open(path)
    stream = container.streams.video[0]
    frames: list[torch.Tensor] = []
    for frame in container.decode(stream):
        t = yuv420_to_rgb_tensor(frame)
        h, w, _ = t.shape
        if h != target_h or w != target_w:
            x = t.permute(2, 0, 1).unsqueeze(0).float()
            x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
            t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
        frames.append(t)
    container.close()
    return frames


def decode_archive(archive_path: str) -> list[torch.Tensor]:
    import zipfile

    source = normalize_archive_source_path(archive_path)
    if source.suffix.lower() == ".mkv":
        return decode_video(str(source))
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(source) as zf:
            zf.extractall(tmpdir)
        mkv = list(Path(tmpdir).glob("*.mkv"))[0]
        return decode_video(str(mkv))


def load_scorers(device: torch.device, upstream_dir: Path):
    from safetensors.torch import load_file

    PoseNet, SegNet = _patch_upstream_scorers(upstream_dir)
    models_dir = upstream_dir / "models"
    posenet = PoseNet().eval().to(device)
    segnet = SegNet().eval().to(device)
    posenet.load_state_dict(load_file(str(models_dir / "posenet.safetensors"), device=str(device)))
    segnet.load_state_dict(load_file(str(models_dir / "segnet.safetensors"), device=str(device)))
    for param in posenet.parameters():
        param.requires_grad = False
    for param in segnet.parameters():
        param.requires_grad = False
    return posenet, segnet


def build_pairs(frames_hwc: list[torch.Tensor]) -> list[torch.Tensor]:
    pairs = []
    for i in range(0, len(frames_hwc) - 1, SEQ_LEN):
        if i + SEQ_LEN > len(frames_hwc):
            break
        pairs.append(torch.stack(frames_hwc[i:i + SEQ_LEN]).unsqueeze(0))
    return pairs


def scorer_forward_pair(pair_btchw, posenet, segnet):
    posenet_in = posenet.preprocess_input(pair_btchw)
    posenet_out = posenet(posenet_in)
    segnet_in = segnet.preprocess_input(pair_btchw)
    segnet_out = segnet(segnet_in)
    return posenet_out, segnet_out


def compute_pair_loss_segnet_attack(filtered_pair_hwc, gt_pair_hwc, posenet, segnet):
    fx = filtered_pair_hwc.permute(0, 1, 4, 2, 3).contiguous()
    gx = gt_pair_hwc.float().permute(0, 1, 4, 2, 3).contiguous()

    fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)
    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()

    with torch.no_grad():
        gt_labels = gs_out.argmax(dim=1)
        pred_labels = fs_out.argmax(dim=1)
        hard_disagree = (pred_labels != gt_labels).float().mean()

    flat_labels = gt_labels.reshape(-1)
    flat_logits = fs_out.permute(0, 2, 3, 1).reshape(-1, fs_out.shape[1])
    soft_ce = F.cross_entropy(flat_logits, flat_labels, reduction="mean")
    seg_dist = soft_ce + (hard_disagree - soft_ce).detach()

    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    return loss, pose_dist.item(), hard_disagree.item()


def compute_saliency_reconstruction_loss(filtered_bchw, original_bchw, weight_hw):
    residual = filtered_bchw - original_bchw
    inv_weight = 1.0 / weight_hw
    return (inv_weight * residual.pow(2)).mean()


def saliency_pair_at(base_saliency: torch.Tensor, *, start_idx: int, alpha: float, device: torch.device) -> torch.Tensor:
    slices = []
    last = base_saliency[-1]
    for offset in range(SEQ_LEN):
        frame_idx = start_idx + offset
        sal = base_saliency[frame_idx] if frame_idx < base_saliency.shape[0] else last
        slices.append((1.0 + alpha * sal).unsqueeze(0))
    return torch.stack(slices, dim=0).to(device)


def compute_posenet_saliency(frames: list[torch.Tensor], posenet, device: torch.device) -> torch.Tensor:
    n = len(frames)
    h, w = frames[0].shape[:2]
    saliency = torch.zeros(n, h, w)

    posenet.eval()
    for start in range(0, n - 1, SEQ_LEN):
        if start + SEQ_LEN > n:
            break
        pair = torch.stack(frames[start:start + SEQ_LEN]).unsqueeze(0).float().to(device)
        pair.requires_grad_(True)
        pair_chw = pair.permute(0, 1, 4, 2, 3).contiguous()
        posenet_in = posenet.preprocess_input(pair_chw)
        out = posenet(posenet_in)
        pose_vec = out["pose"][..., :6]
        loss = pose_vec.pow(2).sum()
        loss.backward()

        grad = pair.grad.detach().abs()
        for t in range(SEQ_LEN):
            frame_idx = start + t
            sal = grad[0, t].mean(dim=-1).cpu()
            sal_max = sal.max()
            if sal_max > 0:
                sal = sal / sal_max
            saliency[frame_idx] = sal

    if n > 1 and saliency[n - 1].sum() == 0:
        saliency[n - 1] = saliency[n - 2]
    return saliency


def load_saliency_weights(alpha: float, frames: list[torch.Tensor], posenet, device: torch.device) -> torch.Tensor:
    saliency = compute_posenet_saliency(frames, posenet, device)
    return 1.0 + alpha * saliency


def compute_combined_loss_segnet_attack(
    filtered_pair_hwc,
    gt_pair_hwc,
    comp_pair_hwc,
    posenet,
    segnet,
    sal_weights_pair,
    sal_lambda,
):
    scorer_loss, pose_dist, seg_dist = compute_pair_loss_segnet_attack(
        filtered_pair_hwc, gt_pair_hwc, posenet, segnet
    )

    b, t, h, w, c = filtered_pair_hwc.shape
    filtered_bchw = filtered_pair_hwc.reshape(b * t, h, w, c).permute(0, 3, 1, 2)
    comp_bchw = comp_pair_hwc.float().reshape(b * t, h, w, c).permute(0, 3, 1, 2)
    sal_recon = compute_saliency_reconstruction_loss(
        filtered_bchw, comp_bchw, sal_weights_pair
    )

    total = scorer_loss + sal_lambda * sal_recon
    return total, scorer_loss.item(), pose_dist, seg_dist, sal_recon.item()


def apply_filter_to_pair(model, pair_uint8, device):
    b, t, h, w, c = pair_uint8.shape
    x = pair_uint8.float().reshape(b * t, h, w, c).permute(0, 3, 1, 2).contiguous()
    if x.device != device:
        x = x.to(device)
    y = model(x)
    return y.permute(0, 2, 3, 1).reshape(b, t, h, w, c)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def quantize_state_dict_like_saved_int8(state_dict: dict[str, torch.Tensor], *, per_channel: bool = False):
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
            q = torch.clamp(torch.round(tensor / scale.view(*shape)), -128, 127).to(torch.int8)
            quantized_state[name] = (q.float() * scale.view(*shape)).to(dtype=tensor.dtype)
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


def normalize_postfilter_meta(alpha: float, kernel: int = DEFAULT_KERNEL) -> dict[str, int | float | str]:
    return build_postfilter_meta(
        variant="cloud_segnet_attack_h32",
        hidden=FIXED_HIDDEN,
        kernel=kernel,
        alpha=alpha,
    )


def build_pair_start_indices(frame_count: int, pair_len: int = SEQ_LEN) -> list[int]:
    starts = []
    for start in range(0, frame_count - 1, pair_len):
        if start + pair_len > frame_count:
            break
        starts.append(start)
    return starts


def derive_default_tag(alpha: float) -> str:
    return make_fixed_h32_segnet_tag(alpha)


def save_best_checkpoint(
    *,
    model: nn.Module,
    ema: EMA,
    output_dir: Path,
    tag: str,
    meta: dict[str, object],
    epoch: int,
    scorer: float,
    shadow_state: dict[str, torch.Tensor] | None = None,
    per_channel_int8: bool = False,
) -> dict[str, object]:
    source_shadow = shadow_state if shadow_state is not None else ema.shadow
    return save_best_checkpoint_shared(
        model=model,
        shadow_state=source_shadow,
        output_dir=output_dir,
        tag=tag,
        meta=meta,
        epoch=epoch,
        scorer=scorer,
        per_channel_int8=per_channel_int8,
    )


def save_final_artifacts(
    *,
    model: nn.Module,
    output_dir: Path,
    tag: str,
    meta: dict[str, object],
    baseline_loss: float,
    final_loss: float,
    final_pose: float,
    final_seg: float,
    best_eval_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    return save_final_artifacts_shared(
        model=model,
        output_dir=output_dir,
        tag=tag,
        meta=meta,
        final_metrics={
            "baseline_loss": baseline_loss,
            "final_loss": final_loss,
            "final_pose": final_pose,
            "final_seg": final_seg,
        },
        best_eval_payload=best_eval_payload,
    )


def evaluate_ema_model(
    *,
    model: nn.Module,
    ema: EMA,
    eval_indices: list[int],
    comp_pairs: list[torch.Tensor],
    gt_pairs: list[torch.Tensor],
    posenet: nn.Module,
    segnet: nn.Module,
) -> tuple[float, float, float]:
    original_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    ema.copy_to(model)
    quantized_state = quantize_state_dict_like_saved_int8(model.state_dict())
    model.load_state_dict(quantized_state, strict=False)
    model.eval()

    total_pose = 0.0
    total_seg = 0.0
    with torch.no_grad():
        for idx in eval_indices:
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            _, pd, sd = compute_pair_loss_segnet_attack(filtered, gt_pairs[idx], posenet, segnet)
            total_pose += pd
            total_seg += sd

    model.load_state_dict(original_state)
    avg_pose = total_pose / len(eval_indices)
    avg_seg = total_seg / len(eval_indices)
    score = 100.0 * avg_seg + math.sqrt(10.0 * avg_pose)
    return score, avg_pose, avg_seg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed h32 SegNet-attack post-filter trainer")
    parser.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--sal-lambda", type=float, default=0.1)
    parser.add_argument("--train-subsample", type=int, default=8)
    parser.add_argument("--eval-subsample", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to the ground-truth video (defaults to upstream 0.mkv)",
    )
    parser.add_argument(
        "--archive",
        type=str,
        default=None,
        help="Path to the compressed archive/video. If omitted, uses GT as the input baseline.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_arg_parser().parse_args(argv)
    alpha = args.alpha
    tag = args.tag or derive_default_tag(alpha)
    meta = normalize_postfilter_meta(alpha=alpha, kernel=args.kernel)
    upstream_dir = setup_environment()
    output_dir = resolve_output_dir()

    print("[cloud-segnet-h32] device:", DEVICE)
    print(f"[cloud-segnet-h32] hidden={FIXED_HIDDEN} kernel={args.kernel} alpha={alpha}")
    print(f"[cloud-segnet-h32] epochs={args.epochs} ema={args.ema_decay} lr={args.lr}")
    print(f"[cloud-segnet-h32] tag={tag}")
    print(f"[cloud-segnet-h32] output_dir={output_dir}")

    if args.epochs == 0:
        print("[cloud-segnet-h32] --epochs 0: import/setup check only, exiting.")
        return {"tag": tag, "status": "import_check_ok"}

    posenet, segnet = load_scorers(DEVICE, upstream_dir)
    videos_dir = upstream_dir / "videos"
    gt_video_path = args.video or str(videos_dir / "0.mkv")

    print("[cloud-segnet-h32] Decoding archive + GT...")
    gt_frames = decode_video(gt_video_path)
    default_archive = DEFAULT_ARCHIVE
    if args.archive:
        archive_path = Path(args.archive)
        if archive_path.suffix.lower() == ".zip":
            comp_frames = decode_archive(str(archive_path))
        else:
            comp_frames = decode_video(str(archive_path))
    elif default_archive.exists():
        print(f"[cloud-segnet-h32] Using default archive {default_archive}")
        comp_frames = decode_archive(str(default_archive))
    else:
        print("[cloud-segnet-h32] No --archive provided; using GT as the compressed baseline.")
        comp_frames = gt_frames
    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]
    print(f"[cloud-segnet-h32] frames={n}")

    sal_all = load_saliency_weights(alpha, gt_frames, posenet, DEVICE)
    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)
    sal_pairs = [saliency_pair_at(sal_all, start_idx=i, alpha=alpha, device=DEVICE) for i in build_pair_start_indices(n)]
    n_pairs = min(len(comp_pairs), len(gt_pairs), len(sal_pairs))
    comp_pairs = comp_pairs[:n_pairs]
    gt_pairs = gt_pairs[:n_pairs]
    sal_pairs = sal_pairs[:n_pairs]

    del comp_frames, gt_frames, sal_all
    gc.collect()

    comp_pairs = [pair.to(DEVICE) for pair in comp_pairs]
    gt_pairs = [pair.to(DEVICE) for pair in gt_pairs]

    model = QATPostFilter(kernel=args.kernel).to(DEVICE)
    ema = EMA(model, decay=args.ema_decay)
    eval_indices = list(range(0, n_pairs, args.eval_subsample))
    n_eval = len(eval_indices)

    print(f"[cloud-segnet-h32] model params={count_params(model)}")
    print(f"[cloud-segnet-h32] evaluating baseline on {n_eval}/{n_pairs} pairs...")
    total_pose = 0.0
    total_seg = 0.0
    with torch.no_grad():
        for idx in eval_indices:
            _, pd, sd = compute_pair_loss_segnet_attack(comp_pairs[idx].float(), gt_pairs[idx], posenet, segnet)
            total_pose += pd
            total_seg += sd
    baseline_pose = total_pose / n_eval
    baseline_seg = total_seg / n_eval
    baseline_loss = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)
    print(
        f"[cloud-segnet-h32] baseline loss={baseline_loss:.4f} "
        f"pose={baseline_pose:.6f} seg={baseline_seg:.6f}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def lr_at(epoch_idx: int) -> float:
        if epoch_idx < args.warmup_epochs:
            return (epoch_idx + 1) / max(1, args.warmup_epochs)
        progress = (epoch_idx - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress)) * (1 - 0.02) + 0.02

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    train_size = max(1, n_pairs // args.train_subsample)
    print(f"[cloud-segnet-h32] training for {args.epochs} epochs, {train_size} pairs/epoch")
    best_eval_score = math.inf
    best_eval_payload: dict[str, object] | None = None

    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(n_pairs)[:train_size].tolist()
        ep_loss = ep_scorer = ep_pose = ep_seg = ep_sal = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(indices):
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            total, scorer, pd, sd, sal_recon = compute_combined_loss_segnet_attack(
                filtered,
                gt_pairs[idx],
                comp_pairs[idx],
                posenet,
                segnet,
                sal_pairs[idx],
                args.sal_lambda,
            )
            (total / args.accum_steps).backward()
            ep_loss += total.item()
            ep_scorer += scorer
            ep_pose += pd
            ep_seg += sd
            ep_sal += sal_recon

            if (step_i + 1) % args.accum_steps == 0 or (step_i + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            ns = len(indices)
            print(
                f"[cloud-segnet-h32] epoch {epoch + 1:>4} total={ep_loss / ns:>10.4f} "
                f"scorer={ep_scorer / ns:>10.4f} pose={ep_pose / ns:>12.6f} "
                f"seg={ep_seg / ns:>12.6f} sal={ep_sal / ns:>10.4f} lr={lr:>10.6f}"
            )
            eval_score, eval_pose, eval_seg = evaluate_ema_model(
                model=model,
                ema=ema,
                eval_indices=eval_indices,
                comp_pairs=comp_pairs,
                gt_pairs=gt_pairs,
                posenet=posenet,
                segnet=segnet,
            )
            print(
                f"[cloud-segnet-h32] eval score={eval_score:>10.4f} "
                f"pose={eval_pose:>12.6f} seg={eval_seg:>12.6f}"
            )
            if eval_score < best_eval_score:
                best_eval_score = eval_score
                best_eval_payload = save_best_checkpoint(
                    model=model,
                    ema=ema,
                    output_dir=output_dir,
                    tag=tag,
                    meta=meta,
                    epoch=epoch + 1,
                    scorer=eval_score,
                )
                print(
                    f"[cloud-segnet-h32] best checkpoint -> epoch {epoch + 1} "
                    f"score={eval_score:.4f} int8={best_eval_payload['int8_size']} bytes"
                )

    ema.copy_to(model)
    model.eval()

    print(f"[cloud-segnet-h32] final eval on EMA weights ({n_eval} pairs)...")
    total_pose = 0.0
    total_seg = 0.0
    with torch.no_grad():
        for idx in eval_indices:
            filtered = apply_filter_to_pair(model, comp_pairs[idx], DEVICE)
            _, pd, sd = compute_pair_loss_segnet_attack(filtered, gt_pairs[idx], posenet, segnet)
            total_pose += pd
            total_seg += sd
    final_pose = total_pose / n_eval
    final_seg = total_seg / n_eval
    final_loss = 100.0 * final_seg + math.sqrt(10.0 * final_pose)

    print("=" * 70)
    print(f"RESULTS: {tag}")
    print("=" * 70)
    print(
        f"Baseline: loss={baseline_loss:.4f}  pose={baseline_pose:.6f}  seg={baseline_seg:.6f}"
    )
    print(f"Filtered: loss={final_loss:.4f}  pose={final_pose:.6f}  seg={final_seg:.6f}")
    delta = final_loss - baseline_loss
    print(f"Delta:    {delta:+.4f}")
    if delta < 0:
        print(f"*** IMPROVEMENT: {-delta:.4f} points ***")
    if best_eval_payload is not None:
        print(
            f"Best EMA checkpoint: epoch {best_eval_payload['epoch']} "
            f"score={best_eval_payload['scorer']:.4f} "
            f"int8={best_eval_payload['int8_size']} bytes"
        )

    final_payload = save_final_artifacts(
        model=model,
        output_dir=output_dir,
        tag=tag,
        meta=meta,
        baseline_loss=baseline_loss,
        final_loss=final_loss,
        final_pose=final_pose,
        final_seg=final_seg,
        best_eval_payload=best_eval_payload,
    )
    print(f"Saved fp32: {final_payload['fp32_path']}")
    print(f"Saved int8: {final_payload['int8_path']} ({final_payload['int8_size']} bytes)")
    print(f"Saved final meta: {final_payload['final_meta_path']}")
    if best_eval_payload is not None:
        print(f"Ensured best meta: {final_payload['best_meta_path']}")
    return {"tag": tag, "baseline_loss": baseline_loss, "final_loss": final_loss}


if __name__ == "__main__":
    main()
