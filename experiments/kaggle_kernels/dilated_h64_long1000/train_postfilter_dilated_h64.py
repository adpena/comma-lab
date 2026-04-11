def tac_has_required_entrypoints(module: object) -> bool:
    required = {'build_postfilter_meta', 'make_dilated_default_tag', 'resolve_cloud_asset_bundle', 'resolve_cloud_output_dir', 'save_best_checkpoint', 'save_final_artifacts', 'normalize_archive_source_path'}
    return all(hasattr(module, name) for name in required)


def find_tac_wheel_candidates(*, input_root: Path = Path("/kaggle/input"), script_dir: Path = SCRIPT_PATH.parent) -> list[Path]:
    candidates = [*sorted(script_dir.glob("comma_video_lab_ball_pack-*.whl"))]
    exact_root = input_root / "comma-lab-private-assets"
    candidates.extend(sorted(exact_root.glob("comma_video_lab_ball_pack-*.whl")))
    if input_root.exists():
        candidates.extend(sorted(input_root.rglob("comma_video_lab_ball_pack-*.whl")))
    return candidates


def ensure_tac_importable() -> None:
    try:
        import tac  # noqa: F401
        from tac import entrypoints as tac_entrypoints  # type: ignore
        if tac_has_required_entrypoints(tac_entrypoints):
            return
    except ImportError:
        pass

    wheel_candidates = find_tac_wheel_candidates()
    if not wheel_candidates:
        input_root = Path("/kaggle/input")
        visible = sorted(str(path) for path in input_root.glob("*")) if input_root.exists() else []
        raise ImportError(
            f"tac is not importable and no bundled wheel was found for Kaggle bootstrap; "
            f"visible input roots={visible}"
        )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(wheel_candidates[0])])

#!/usr/bin/env python
"""Self-contained Kaggle/Colab trainer for the deploy-correct dilated h64 lane.

This script vendors the full training path so it can run without importing any
local helper modules. It keeps the QAT + EMA recipe, saliency-weighted loss,
and durable best-checkpoint metadata emission, while swapping the post-filter
architecture to the deploy-correct dilated h64 variant.

Usage on Kaggle/Colab:
  !python train_postfilter_dilated_h64.py --epochs 2500 --alpha 20
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
import zipfile
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
    make_dilated_default_tag,
    normalize_archive_source_path,
    resolve_cloud_asset_bundle,
    resolve_cloud_output_dir,
    save_best_checkpoint as save_best_checkpoint_shared,
    save_final_artifacts as save_final_artifacts_shared,
)

DEFAULT_HIDDEN = 64
DEFAULT_KERNEL = 3
SEQ_LEN = 2
DEFAULT_SEGNET_INPUT_SIZE = (512, 384)
DEFAULT_CAMERA_SIZE = (1164, 874)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability(0)
        except Exception:
            capability = None
        if capability is None or capability[0] >= 7:
            return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = detect_device()
OUTPUT_DIR = None
UPSTREAM_DIR = None
MODELS_DIR = None
VIDEOS_DIR = None
SEGNET_INPUT_SIZE = DEFAULT_SEGNET_INPUT_SIZE


def resolve_output_dir() -> Path:
    return resolve_cloud_output_dir(PROJECT_ROOT)


OUTPUT_DIR = resolve_output_dir()
ASSET_BUNDLE = resolve_cloud_asset_bundle(
    PROJECT_ROOT,
    SCRIPT_PATH,
    archive_relative_path="reports/raw/2026-04-06-av1-roi-experiments/decode_base_archive.zip",
    saliency_relative_path="experiments/masks/posenet_saliency.npy",
)
ARCHIVE_ZIP = ASSET_BUNDLE["archive_path"]
SALIENCY_PATH = ASSET_BUNDLE["saliency_path"]


def setup_environment() -> Path:
    """Clone upstream and install runtime dependencies when needed."""
    base_dir: Path
    if Path("/kaggle").exists():
        base_dir = Path("/kaggle/working")
    elif Path("/content").exists():
        base_dir = Path("/content")
    else:
        base_dir = Path(tempfile.mkdtemp(prefix="comma_lab_"))

    upstream_dir = base_dir / "comma_video_compression_challenge"
    if not upstream_dir.exists():
        print("[setup] Cloning upstream repo...")
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/commaai/comma_video_compression_challenge.git",
            str(upstream_dir),
        ])
    else:
        print(f"[setup] Upstream already present at {upstream_dir}")

    deps = [
        "av",
        "numpy",
        "safetensors",
        "timm",
        "segmentation-models-pytorch",
    ]
    for dep in deps:
        module_name = dep.replace("-", "_")
        try:
            __import__(module_name)
        except ImportError:
            print(f"[setup] Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])

    return upstream_dir


def import_upstream_modules() -> None:
    """Import upstream scorer modules after the upstream repo is on sys.path."""
    global SEGNET_INPUT_SIZE
    sys.path.insert(0, str(UPSTREAM_DIR))
    from frame_utils import camera_size, segnet_model_input_size, seq_len  # type: ignore
    from modules import AllNorm, PoseNet, SegNet  # type: ignore

    globals()["AllNorm"] = AllNorm
    globals()["PoseNet"] = PoseNet
    globals()["SegNet"] = SegNet
    globals()["camera_size"] = camera_size
    globals()["segnet_model_input_size"] = segnet_model_input_size
    globals()["seq_len"] = seq_len
    SEGNET_INPUT_SIZE = tuple(segnet_model_input_size)


def _patched_allnorm_forward(self, x):
    return self.bn(x.reshape(-1, 1)).reshape(x.shape)


def rgb_to_yuv6_diff(rgb_chw: torch.Tensor) -> torch.Tensor:
    """Differentiable rgb_to_yuv6 without in-place ops."""
    h, w = rgb_chw.shape[-2], rgb_chw.shape[-1]
    h2, w2 = h // 2, w // 2
    rgb = rgb_chw[..., : 2 * h2, : 2 * w2]
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    y = (r * 0.299 + g * 0.587 + b * 0.114).clamp(0.0, 255.0)
    u = ((b - y) / 1.772 + 128.0).clamp(0.0, 255.0)
    v = ((r - y) / 1.402 + 128.0).clamp(0.0, 255.0)
    u_sub = (u[..., 0::2, 0::2] + u[..., 1::2, 0::2] + u[..., 0::2, 1::2] + u[..., 1::2, 1::2]) * 0.25
    v_sub = (v[..., 0::2, 0::2] + v[..., 1::2, 0::2] + v[..., 0::2, 1::2] + v[..., 1::2, 1::2]) * 0.25
    y00 = y[..., 0::2, 0::2]
    y10 = y[..., 1::2, 0::2]
    y01 = y[..., 0::2, 1::2]
    y11 = y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, u_sub, v_sub], dim=1)


def _patched_posenet_preprocess(self, x):
    batch_size, seq_len_local, channels, h, w = x.shape
    x = x.reshape(batch_size * seq_len_local, channels, h, w)
    x = F.interpolate(
        x,
        size=(SEGNET_INPUT_SIZE[1], SEGNET_INPUT_SIZE[0]),
        mode="bilinear",
        align_corners=False,
    )
    yuv = rgb_to_yuv6_diff(x)
    return yuv.reshape(batch_size, seq_len_local * 6, yuv.shape[-2], yuv.shape[-1]).contiguous()


class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        with torch.no_grad():
            scale = w.detach().abs().max() / 127.0
            if float(scale) == 0.0:
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


class DilatedPostFilter(nn.Module):
    def __init__(self, hidden: int = DEFAULT_HIDDEN, kernel: int = DEFAULT_KERNEL):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=False)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


class QATDilatedPostFilter(DilatedPostFilter):
    def _qconv(self, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        weight = fake_quant(conv.weight)
        bias = fake_quant(conv.bias) if conv.bias is not None else None
        return F.conv2d(x, weight, bias, stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

    def forward(self, x):
        residual = self.act(self._qconv(self.conv1, x))
        residual = self.act(self._qconv(self.conv2, residual))
        residual = self._qconv(self.conv3, residual)
        return (x + residual).clamp(0, 255)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def maybe_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor if tensor.device == device else tensor.to(device)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def yuv420_to_rgb_tensor(frame) -> torch.Tensor:
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


def decode_video(path: str, target_h: int = DEFAULT_CAMERA_SIZE[1], target_w: int = DEFAULT_CAMERA_SIZE[0]) -> list[torch.Tensor]:
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
    source = normalize_archive_source_path(archive_path)
    if source.suffix.lower() == ".mkv":
        return decode_video(str(source))
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(source) as zf:
            zf.extractall(tmpdir)
        mkv_files = sorted(Path(tmpdir).glob("*.mkv"))
        if not mkv_files:
            raise FileNotFoundError(f"No .mkv found inside archive: {archive_path}")
        return decode_video(str(mkv_files[0]))


def load_scorers(device: torch.device):
    from safetensors.torch import load_file

    assert MODELS_DIR is not None, "Environment not initialized"
    posenet = PoseNet().eval().to(device)
    segnet = SegNet().eval().to(device)
    posenet.load_state_dict(load_file(str(MODELS_DIR / "posenet.safetensors"), device=str(device)))
    segnet.load_state_dict(load_file(str(MODELS_DIR / "segnet.safetensors"), device=str(device)))
    for p in posenet.parameters():
        p.requires_grad = False
    for p in segnet.parameters():
        p.requires_grad = False
    return posenet, segnet


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


def load_saliency_weights(alpha: float, n_frames: int, device: torch.device, *, gt_frames=None, posenet=None) -> torch.Tensor:
    import numpy as np

    if SALIENCY_PATH.exists():
        sal = np.load(str(SALIENCY_PATH))
        sal_t = torch.from_numpy(sal).float()
    elif gt_frames is not None and posenet is not None:
        sal_t = compute_posenet_saliency(gt_frames, posenet, device)
    else:
        raise FileNotFoundError(f"Missing saliency map: {SALIENCY_PATH}")
    if sal_t.shape[0] < n_frames:
        pad = sal_t[-1:].expand(n_frames - sal_t.shape[0], -1, -1)
        sal_t = torch.cat([sal_t, pad], dim=0)
    sal_t = sal_t[:n_frames]
    weights = 1.0 + alpha * sal_t
    return weights.unsqueeze(1).to(device)


def build_pair_start_indices(frame_count: int, pair_len: int) -> list[int]:
    starts: list[int] = []
    for start in range(0, frame_count - 1, pair_len):
        if start + pair_len > frame_count:
            break
        starts.append(start)
    return starts


def pair_from_frames(frames: list[torch.Tensor], start_idx: int) -> torch.Tensor:
    return torch.stack(frames[start_idx:start_idx + SEQ_LEN]).unsqueeze(0)


def saliency_pair_at(base_saliency: torch.Tensor, *, start_idx: int, alpha: float, device: torch.device) -> torch.Tensor:
    slices = []
    last = base_saliency[-1]
    for offset in range(SEQ_LEN):
        frame_idx = start_idx + offset
        sal = base_saliency[frame_idx] if frame_idx < base_saliency.shape[0] else last
        slices.append((1.0 + alpha * sal).unsqueeze(0))
    return torch.stack(slices, dim=0).to(device)


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


def compute_saliency_reconstruction_loss(filtered_bchw, original_bchw, weight_hw):
    residual = filtered_bchw - original_bchw
    inv_weight = 1.0 / weight_hw
    return (inv_weight * residual.pow(2)).mean()


def compute_combined_loss(filtered_pair_hwc, gt_pair_hwc, comp_pair_hwc, posenet, segnet, sal_weights_pair, sal_lambda):
    scorer_loss, pose_dist, seg_dist = compute_pair_loss(filtered_pair_hwc, gt_pair_hwc, posenet, segnet)
    filtered_bchw = filtered_pair_hwc[:, 1].permute(0, 3, 1, 2)
    comp_bchw = comp_pair_hwc[:, 1].float().permute(0, 3, 1, 2)
    sal_recon_loss = compute_saliency_reconstruction_loss(filtered_bchw, comp_bchw, sal_weights_pair[1:2])
    total_loss = scorer_loss + sal_lambda * sal_recon_loss
    return total_loss, scorer_loss.item(), pose_dist, seg_dist, sal_recon_loss.item()


def apply_filter_to_pair(model, pair_uint8, device):
    b, t, h, w, c = pair_uint8.shape
    x = pair_uint8.float().reshape(b * t, h, w, c).permute(0, 3, 1, 2).contiguous()
    x = x.to(device) if x.device != device else x
    y = model(x)
    return y.permute(0, 2, 3, 1).reshape(b, t, h, w, c)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def normalize_postfilter_meta(hidden: int, kernel: int, alpha: float) -> dict:
    return build_postfilter_meta(variant="dilated", hidden=hidden, kernel=kernel, alpha=alpha)


def make_default_tag(hidden: int, alpha: float) -> str:
    return make_dilated_default_tag(hidden, alpha)


def quantize_state_dict_like_saved_int8(state_dict: dict[str, torch.Tensor], *, per_channel: bool = False) -> dict[str, torch.Tensor]:
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


def save_best_checkpoint(
    *,
    model: nn.Module,
    ema: EMA,
    output_dir: Path,
    tag: str,
    meta: dict,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Self-contained dilated h64 QAT+EMA trainer")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--alpha", type=float, default=20.0, help="Saliency emphasis: weight = 1 + alpha * saliency")
    parser.add_argument("--sal-lambda", type=float, default=0.1)
    parser.add_argument("--train-subsample", type=int, default=4)
    parser.add_argument("--eval-subsample", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--checkpoint-eval-every", type=int, default=10)
    parser.add_argument("--checkpoint-select-int8", action="store_true")
    parser.add_argument("--per-channel-int8", action="store_true")
    parser.add_argument("--cuda-autocast", action="store_true")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--compressed-video", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    return parser


def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)
    alpha = args.alpha
    tag = args.tag or make_default_tag(args.hidden, alpha)
    meta = normalize_postfilter_meta(args.hidden, args.kernel, alpha)

    if args.epochs == 0:
        return {"tag": tag, "status": "import_check_ok", "meta": meta}

    global UPSTREAM_DIR, MODELS_DIR, VIDEOS_DIR, OUTPUT_DIR
    UPSTREAM_DIR = setup_environment()
    MODELS_DIR = UPSTREAM_DIR / "models"
    VIDEOS_DIR = UPSTREAM_DIR / "videos"
    OUTPUT_DIR = resolve_output_dir()
    import_upstream_modules()

    from frame_utils import seq_len as upstream_seq_len  # type: ignore
    globals()["seq_len"] = upstream_seq_len
    global SEQ_LEN
    SEQ_LEN = int(upstream_seq_len)

    globals()["AllNorm"].forward = _patched_allnorm_forward
    globals()["PoseNet"].preprocess_input = _patched_posenet_preprocess

    print(f"[dilated-h64] device={DEVICE} alpha={alpha} hidden={args.hidden} ema={args.ema_decay} tag={tag}")

    posenet, segnet = load_scorers(DEVICE)

    gt_video_path = args.video or str(VIDEOS_DIR / "0.mkv")
    comp_video_path = args.compressed_video

    gt_frames = decode_video(gt_video_path)
    if comp_video_path:
        comp_frames = decode_video(comp_video_path)
    else:
        comp_frames = decode_archive(str(ARCHIVE_ZIP))

    n = min(len(comp_frames), len(gt_frames))
    comp_frames = comp_frames[:n]
    gt_frames = gt_frames[:n]
    pair_starts = build_pair_start_indices(n, SEQ_LEN)
    n_pairs = len(pair_starts)

    sal_base = load_saliency_weights(alpha, n, DEVICE, gt_frames=gt_frames, posenet=posenet)
    sal_base = sal_base.squeeze(1).detach().cpu()

    model = QATDilatedPostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    eval_model = DilatedPostFilter(hidden=args.hidden, kernel=args.kernel).to(DEVICE)
    ema = EMA(model, decay=args.ema_decay)

    eval_indices = list(range(0, n_pairs, args.eval_subsample))
    n_eval = len(eval_indices)

    print(f"[dilated-h64] evaluating baseline on {n_eval}/{n_pairs} pairs")
    total_pose = total_seg = 0.0
    with torch.no_grad():
        for idx in eval_indices:
            start = pair_starts[idx]
            comp_pair = maybe_to_device(pair_from_frames(comp_frames, start), DEVICE)
            gt_pair = maybe_to_device(pair_from_frames(gt_frames, start), DEVICE)
            with autocast_context(DEVICE, args.cuda_autocast):
                _, pd, sd = compute_pair_loss(comp_pair.float(), gt_pair, posenet, segnet)
            total_pose += pd
            total_seg += sd
    baseline_pose = total_pose / n_eval
    baseline_seg = total_seg / n_eval
    baseline_loss = 100.0 * baseline_seg + math.sqrt(10.0 * baseline_pose)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.warmup_epochs > 0:
        def lr_at(epoch_idx: int) -> float:
            if epoch_idx < args.warmup_epochs:
                return float(epoch_idx + 1) / max(1, args.warmup_epochs)
            progress = (epoch_idx - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress)) * (1.0 - 0.02) + 0.02
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-5)

    train_size = max(1, n_pairs // args.train_subsample)
    best_scorer = float("inf")
    best_shadow_state: dict[str, torch.Tensor] | None = None

    print(f"[dilated-h64] training epochs={args.epochs} train_size={train_size} eval={n_eval}")
    print(f"{'epoch':>5} {'total':>10} {'scorer':>10} {'pose':>12} {'seg':>12} {'sal_recon':>10} {'lr':>10}")
    print("-" * 75)

    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(n_pairs)[:train_size].tolist()
        ep_loss = ep_scorer = ep_pose = ep_seg = ep_sal = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(indices):
            start = pair_starts[idx]
            comp_pair = maybe_to_device(pair_from_frames(comp_frames, start), DEVICE)
            gt_pair = maybe_to_device(pair_from_frames(gt_frames, start), DEVICE)
            sal_pair = saliency_pair_at(sal_base, start_idx=start, alpha=alpha, device=DEVICE)
            with autocast_context(DEVICE, args.cuda_autocast):
                filtered = apply_filter_to_pair(model, comp_pair, DEVICE)
                total_loss, scorer_loss, pd, sd, sal_recon = compute_combined_loss(
                    filtered, gt_pair, comp_pair, posenet, segnet, sal_pair, args.sal_lambda
                )
            (total_loss / args.accum_steps).backward()
            ep_loss += total_loss.item()
            ep_scorer += scorer_loss
            ep_pose += pd
            ep_seg += sd
            ep_sal += sal_recon

            if (step_i + 1) % args.accum_steps == 0 or (step_i + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

        scheduler.step()

        avg_scorer = ep_scorer / len(indices)
        score_for_checkpoint = avg_scorer
        if args.checkpoint_select_int8 and ((epoch + 1) % args.checkpoint_eval_every == 0 or epoch == 0 or (epoch + 1) == args.epochs):
            eval_state = quantize_state_dict_like_saved_int8(ema.shadow, per_channel=args.per_channel_int8)
            eval_model.load_state_dict(eval_state)
            eval_model.eval()
            total_pose_eval = total_seg_eval = 0.0
            with torch.no_grad():
                for idx in eval_indices:
                    start = pair_starts[idx]
                    comp_pair = maybe_to_device(pair_from_frames(comp_frames, start), DEVICE)
                    gt_pair = maybe_to_device(pair_from_frames(gt_frames, start), DEVICE)
                    with autocast_context(DEVICE, args.cuda_autocast):
                        filtered = apply_filter_to_pair(eval_model, comp_pair, DEVICE)
                        _, pd, sd = compute_pair_loss(filtered, gt_pair, posenet, segnet)
                    total_pose_eval += pd
                    total_seg_eval += sd
            score_for_checkpoint = 100.0 * (total_seg_eval / n_eval) + math.sqrt(10.0 * (total_pose_eval / n_eval))

        if epoch == 0 or score_for_checkpoint < best_scorer:
            best_scorer = score_for_checkpoint
            best_payload = save_best_checkpoint(
                model=model,
                ema=ema,
                output_dir=OUTPUT_DIR,
                tag=tag,
                meta=meta,
                epoch=epoch + 1,
                scorer=score_for_checkpoint,
                per_channel_int8=args.per_channel_int8,
            )
            best_shadow_state = {name: tensor.detach().cpu().clone() for name, tensor in ema.shadow.items()}
        else:
            best_payload = None

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"{epoch + 1:>5} {ep_loss / len(indices):>10.4f} {ep_scorer / len(indices):>10.4f} "
                f"{ep_pose / len(indices):>12.6f} {ep_seg / len(indices):>12.6f} "
                f"{ep_sal / len(indices):>10.4f} {lr:>10.6f}"
            )

    if best_shadow_state is not None:
        eval_model.load_state_dict(best_shadow_state)
    else:
        ema.copy_to(eval_model)
    eval_model.eval()

    print(f"[dilated-h64] final eval on {n_eval}/{n_pairs} pairs")
    total_pose = total_seg = 0.0
    with torch.no_grad():
        for idx in eval_indices:
            start = pair_starts[idx]
            comp_pair = maybe_to_device(pair_from_frames(comp_frames, start), DEVICE)
            gt_pair = maybe_to_device(pair_from_frames(gt_frames, start), DEVICE)
            with autocast_context(DEVICE, args.cuda_autocast):
                filtered = apply_filter_to_pair(eval_model, comp_pair, DEVICE)
                _, pd, sd = compute_pair_loss(filtered, gt_pair, posenet, segnet)
            total_pose += pd
            total_seg += sd
    final_pose = total_pose / n_eval
    final_seg = total_seg / n_eval
    final_loss = 100.0 * final_seg + math.sqrt(10.0 * final_pose)
    delta = final_loss - baseline_loss

    final_payload = save_final_artifacts_shared(
        model=eval_model,
        output_dir=OUTPUT_DIR,
        tag=f"{tag}_final",
        meta=meta,
        final_metrics={
            "baseline_loss": baseline_loss,
            "final_loss": final_loss,
            "delta": delta,
            "final_pose": final_pose,
            "final_seg": final_seg,
        },
        best_eval_payload=best_payload if "best_payload" in locals() else None,
        per_channel_int8=args.per_channel_int8,
    )

    print(f"RESULTS: {tag}")
    print(f"Baseline: loss={baseline_loss:.4f} pose={baseline_pose:.6f} seg={baseline_seg:.6f}")
    print(f"Filtered: loss={final_loss:.4f} pose={final_pose:.6f} seg={final_seg:.6f}")
    print(f"Delta:    {delta:+.4f}")
    print(f"Saved final fp32: {final_payload['fp32_path']}")
    print(f"Saved final int8: {final_payload['int8_path']} ({final_payload['int8_size']} bytes)")

    return {
        "tag": tag,
        "baseline_loss": baseline_loss,
        "final_loss": final_loss,
        "delta": delta,
        "best_scorer": best_scorer,
        "final_artifacts": final_payload,
        "best_checkpoint": best_payload if "best_payload" in locals() else None,
    }


if __name__ == "__main__":
    main()
