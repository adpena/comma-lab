#!/usr/bin/env python
"""Inflate path with learned post-filter applied after bicubic upscale.

The post-filter is a tiny CNN (3,203 params, 7.5KB int8) trained directly
against the scorer's loss function via backprop. It learns to correct the
decoded video to maximize PoseNet+SegNet scores.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import av


DEFAULT_POSTFILTER_META = {
    "variant": "residual",
    "hidden": 16,
    "kernel": 3,
}


# ============================================================
# Post-filter model (matches training architecture)
# ============================================================
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


class DepthwisePostFilter(nn.Module):
    def __init__(self, hidden=16, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.pw_in = nn.Conv2d(3, hidden, 1, bias=True)
        self.dw = nn.Conv2d(hidden, hidden, kernel, padding=pad, groups=hidden, bias=True)
        self.pw_out = nn.Conv2d(hidden, 3, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

        nn.init.zeros_(self.pw_out.weight)
        nn.init.zeros_(self.pw_out.bias)

    def forward(self, x):
        residual = self.act(self.pw_in(x))
        residual = self.act(self.dw(residual))
        residual = self.pw_out(residual)
        return (x + residual).clamp(0, 255)


class LumaPostFilter(nn.Module):
    def __init__(self, hidden=16, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(1, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 1, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)

        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        y = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
        residual = self.act(self.conv1(y))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual.repeat(1, 3, 1, 1)).clamp(0, 255)


class PixelShufflePostFilter(nn.Module):
    """REN-style post-filter using PixelUnshuffle/Shuffle.

    REN-style post-filter using PixelUnshuffle/Shuffle for half-resolution
    processing. PixelUnshuffle(2) converts 3-channel full-res to 12-channel
    half-res; 4 conv layers process; PixelShuffle(2) reconstructs full-res.

    Key advantage: corrections are naturally aligned with what both PoseNet
    and SegNet see after their bilinear downsample to 512x384. Each 3x3
    conv at half-res covers 6x6 at full-res, matching the scorer's RF.
    """
    def __init__(self, hidden=64, kernel=3):
        super().__init__()
        self.down = nn.PixelUnshuffle(2)
        pad = kernel // 2
        self.body = nn.Sequential(
            nn.Conv2d(12, hidden, kernel, padding=pad, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 12, kernel, padding=pad, bias=True),
        )
        self.up = nn.PixelShuffle(2)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        x_norm = x / 255.0
        residual = self.up(self.body(self.down(x_norm)))
        return (x_norm + residual).clamp(0, 1) * 255.0


class PixelShuffleDilatedPostFilter(nn.Module):
    """Half-resolution PixelShuffle path with an explicit dilated middle layer."""

    def __init__(self, hidden=64, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.down = nn.PixelUnshuffle(2)
        self.conv1 = nn.Conv2d(12, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv4 = nn.Conv2d(hidden, 12, kernel, padding=pad, bias=True)
        self.up = nn.PixelShuffle(2)
        self.act = nn.ReLU(inplace=True)

        nn.init.zeros_(self.conv4.weight)
        nn.init.zeros_(self.conv4.bias)

    def forward(self, x):
        x_norm = x / 255.0
        residual = self.down(x_norm)
        residual = self.act(self.conv1(residual))
        residual = self.act(self.conv2(residual))
        residual = self.act(self.conv3(residual))
        residual = self.up(self.conv4(residual))
        return (x_norm + residual).clamp(0, 1) * 255.0


class DilatedPostFilter(nn.Module):
    """PostFilter with dilation=2 on the middle conv layer.

    Expands receptive field from 7x7 to 15x15 at zero param cost.
    LeCun + Karpathy consensus: this matches fastvit_t12's early-layer
    receptive field and resolves the mid-frequency bottleneck.
    """
    def __init__(self, hidden=16, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        # Dilation=2 on middle layer: effective kernel is 5x5, RF grows to 15x15
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)

        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


class FiLMPostFilter(nn.Module):
    def __init__(self, hidden=16, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.film = nn.Linear(3, hidden * 2, bias=True)
        self.act = nn.ReLU(inplace=True)

    def _descriptor(self, x: torch.Tensor) -> torch.Tensor:
        # Stateless per-frame descriptor: luma mean, luma std, edge density.
        y = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
        y_norm = y / 255.0
        mean = y_norm.mean(dim=(2, 3))
        std = y_norm.std(dim=(2, 3), unbiased=False)
        dx = y_norm[..., :, 1:] - y_norm[..., :, :-1]
        dy = y_norm[..., 1:, :] - y_norm[..., :-1, :]
        edge = 0.5 * (dx.abs().mean(dim=(2, 3)) + dy.abs().mean(dim=(2, 3)))
        return torch.cat([mean, std, edge], dim=1)

    def _film_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        film = self.film(self._descriptor(x))
        gamma, beta = film.chunk(2, dim=1)
        gamma = 1.0 + 0.25 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = 8.0 * torch.tanh(beta).unsqueeze(-1).unsqueeze(-1)
        return gamma, beta

    def forward(self, x):
        gamma, beta = self._film_params(x)
        residual = self.act(self.conv1(x))
        residual = residual * gamma + beta
        residual = self.act(self.conv2(residual))
        residual = residual * gamma + beta
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


def normalize_postfilter_meta(meta: object | None) -> dict[str, int | str]:
    normalized = dict(DEFAULT_POSTFILTER_META)
    if isinstance(meta, dict):
        if "variant" in meta:
            normalized["variant"] = str(meta["variant"])
        if "hidden" in meta:
            normalized["hidden"] = int(meta["hidden"])
        if "kernel" in meta:
            normalized["kernel"] = int(meta["kernel"])
    return normalized


def build_postfilter(meta: object | None = None) -> PostFilter:
    normalized = normalize_postfilter_meta(meta)
    variant = normalized["variant"]
    hidden = int(normalized["hidden"])
    kernel = int(normalized["kernel"])
    if variant in {"standard", "residual", "saliency_weighted", "segaware"}:
        return PostFilter(hidden=hidden, kernel=kernel)
    if variant == "depthwise":
        return DepthwisePostFilter(hidden=hidden, kernel=kernel)
    if variant == "luma":
        return LumaPostFilter(hidden=hidden, kernel=kernel)
    if variant == "pixelshuffle":
        return PixelShufflePostFilter(hidden=hidden, kernel=kernel)
    if variant == "pixelshuffle_dilated":
        return PixelShuffleDilatedPostFilter(hidden=hidden, kernel=kernel)
    if variant == "dilated":
        return DilatedPostFilter(hidden=hidden, kernel=kernel)
    if variant == "film_conditioned":
        return FiLMPostFilter(hidden=hidden, kernel=kernel)
    raise ValueError(f"Unsupported post-filter variant: {variant}")


def load_postfilter_int8(path: str, device: str = "cpu") -> PostFilter:
    """Load int8-quantized post-filter weights.

    Supports three on-disk formats (backward compatible):
      * ``key.q`` int8 + scalar ``key.s`` -> legacy per-tensor symmetric
      * ``key.q`` int8 + vector ``key.s`` (shape [C]) -> per-channel symmetric
        broadcasted across the first weight dimension
      * ``key`` float tensor (no .q/.s suffix) -> uncompressed fp32 fallback,
        used when biases are stored in full precision to keep a tiny tensor
        from losing fidelity for the sake of a few bytes.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    float_state: dict[str, torch.Tensor] = {}
    seen = set()
    for raw_key in state.keys():
        if raw_key == "__meta__":
            continue
        if raw_key.endswith(".q") or raw_key.endswith(".s"):
            base = raw_key[:-2]
            if base in seen:
                continue
            seen.add(base)
            q = state[base + ".q"].float()
            s = state[base + ".s"]
            if s.ndim == 0:
                float_state[base] = q * s
            else:
                # per-channel: reshape s to (C, 1, 1, ...) matching q's rank
                shape = [s.shape[0]] + [1] * (q.ndim - 1)
                float_state[base] = q * s.view(*shape)
        else:
            # uncompressed fp32 tensor (e.g., bias stored in full precision)
            float_state[raw_key] = state[raw_key].float()
            seen.add(raw_key)
    meta = normalize_postfilter_meta(state.get("__meta__"))
    if (
        meta.get("variant") in {"standard", "residual", "saliency_weighted", "segaware"}
        and "conv4.weight" in float_state
        and "conv1.weight" in float_state
        and float_state["conv1.weight"].ndim == 4
        and int(float_state["conv1.weight"].shape[1]) == 12
    ):
        meta["variant"] = "pixelshuffle_dilated"
    model = build_postfilter(meta)
    model.load_state_dict(float_state)
    return model.eval().to(device)


# ============================================================
# Canonical YUV→RGB (BT.601 limited range, matches frame_utils.py)
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


def inflate_with_postfilter(
    video_path: str, dst: str, postfilter_path: str,
    target_w: int = 1164, target_h: int = 874, device: str = "cpu"
) -> int:
    """Decode, upscale, apply learned post-filter, write raw RGB."""
    print(f"  Loading post-filter from {postfilter_path}", file=sys.stderr)
    model = load_postfilter_int8(postfilter_path, device=device)

    container = av.open(video_path)
    stream = container.streams.video[0]
    n = 0

    with open(dst, 'wb') as f:
        for frame in container.decode(stream):
            t = yuv420_to_rgb(frame)  # (H, W, 3) uint8
            H, W, _ = t.shape

            # Bicubic upscale (canonical inflate)
            if H != target_h or W != target_w:
                x = t.permute(2, 0, 1).unsqueeze(0).float()
                x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
                x = x.clamp(0, 255)
            else:
                x = t.permute(2, 0, 1).unsqueeze(0).float()

            # Apply learned post-filter
            with torch.no_grad():
                x = model(x.to(device))

            # Convert back to uint8 (H, W, 3)
            t = x.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu()
            f.write(t.contiguous().numpy().tobytes())
            n += 1

            if n % 300 == 0:
                print(f"  Processed {n} frames ...", file=sys.stderr, flush=True)

    container.close()
    print(f"Inflated {n} frames with post-filter -> {dst}", file=sys.stderr)
    return n


if __name__ == "__main__":
    archive_dir = sys.argv[1]
    inflated_dir = sys.argv[2]
    video_names_file = sys.argv[3]
    # Look for weights in archive dir first (bundled in archive.zip), then submission dir
    script_dir = Path(__file__).resolve().parent
    default_paths = [
        Path(sys.argv[1]) / "postfilter_int8.pt",  # inside archive dir (bundled in zip)
        script_dir / "postfilter_int8.pt",           # alongside inflate script
    ]
    postfilter_path = sys.argv[4] if len(sys.argv) > 4 else None
    if postfilter_path is None:
        for p in default_paths:
            if p.exists():
                postfilter_path = str(p)
                break
        else:
            print("ERROR: postfilter_int8.pt not found", file=sys.stderr)
            sys.exit(1)

    inflated_dir = Path(inflated_dir)
    inflated_dir.mkdir(parents=True, exist_ok=True)

    for line in Path(video_names_file).read_text().splitlines():
        rel = line.strip()
        if not rel:
            continue
        stem = rel.rsplit(".", 1)[0]
        mkv_path = Path(archive_dir) / f"{stem}.mkv"
        out_path = inflated_dir / f"{stem}.raw"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Inflating {mkv_path} -> {out_path} (post-filter)", file=sys.stderr)
        inflate_with_postfilter(str(mkv_path), str(out_path), postfilter_path)
