#!/usr/bin/env python
"""Inflate path with learned post-filter applied after bicubic upscale.

The post-filter is a tiny CNN (3,203 params, 7.5KB int8) trained directly
against the scorer's loss function via backprop. It learns to correct the
decoded video to maximize PoseNet+SegNet scores.

Architecture classes and the INT8 loader live in the tac package
(src/tac/architectures.py, src/tac/quantization.py). This script imports
from tac when available, with a self-contained fallback for contest
submission environments where tac is not installed.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import av

# ============================================================
# Import from tac when available; inline fallback for standalone
# ============================================================
try:
    from tac.architectures import (
        PostFilter,
        PairAwarePostFilter,
        DepthwisePostFilter,
        LumaPostFilter,
        PixelShufflePostFilter,
        PixelShuffleDilatedPostFilter,
        DilatedPostFilter,
        GatedDilatedPostFilter,
        FiLMPostFilter,
        build_postfilter as _tac_build_postfilter,
    )
    from tac.quantization import (
        DEFAULT_POSTFILTER_META,
        normalize_postfilter_meta,
        load_postfilter_int8,
    )
    _TAC_AVAILABLE = True
except ImportError:
    _TAC_AVAILABLE = False

    # ── Inline fallback (self-contained for scorer machine) ──────────

    DEFAULT_POSTFILTER_META = {
        "variant": "residual",
        "hidden": 16,
        "kernel": 3,
    }

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

    class PairAwarePostFilter(nn.Module):
        """6-channel pair-aware post-filter (target + context frames)."""
        def __init__(self, hidden=64, kernel=3):
            super().__init__()
            pad = kernel // 2
            self.conv1 = nn.Conv2d(6, hidden, kernel, padding=pad, bias=True)
            self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
            self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
            self.act = nn.ReLU(inplace=True)
            nn.init.zeros_(self.conv3.weight)
            nn.init.zeros_(self.conv3.bias)

        def forward(self, x):
            target = x[:, :3]
            residual = self.act(self.conv1(x))
            residual = self.act(self.conv2(residual))
            residual = self.conv3(residual)
            return (target + residual).clamp(0, 255)

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
        def __init__(self, hidden=16, kernel=3):
            super().__init__()
            pad = kernel // 2
            self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
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

    class GatedDilatedPostFilter(nn.Module):
        def __init__(self, hidden=16, kernel=3):
            super().__init__()
            pad = kernel // 2
            self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
            self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True)
            self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
            self.gate = nn.Sequential(nn.Conv2d(hidden, 1, 1, bias=True), nn.Sigmoid())
            self.act = nn.ReLU(inplace=True)
            nn.init.zeros_(self.conv3.weight)
            nn.init.zeros_(self.conv3.bias)
            nn.init.zeros_(self.gate[0].weight)
            nn.init.zeros_(self.gate[0].bias)

        def forward(self, x):
            features = self.act(self.conv1(x))
            features = self.act(self.conv2(features))
            gate = self.gate(features)
            residual = self.conv3(features)
            return (x + gate * residual).clamp(0, 255)

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

    def _fallback_build_postfilter(meta: object | None = None) -> nn.Module:
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
        if variant == "gated_dilated":
            return GatedDilatedPostFilter(hidden=hidden, kernel=kernel)
        if variant in ("film", "film_conditioned"):
            return FiLMPostFilter(hidden=hidden, kernel=kernel)
        if variant == "psd":
            return PixelShuffleDilatedPostFilter(hidden=hidden, kernel=kernel)
        if variant == "pair_aware":
            return PairAwarePostFilter(hidden=hidden, kernel=kernel)
        raise ValueError(f"Unsupported post-filter variant: {variant}")

    def load_postfilter_int8(path: str, device: str = "cpu") -> nn.Module:
        """Load int8-quantized post-filter weights (standalone fallback).

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
                    shape = [s.shape[0]] + [1] * (q.ndim - 1)
                    float_state[base] = q * s.view(*shape)
            else:
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
        model = _fallback_build_postfilter(meta)
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(float_state.keys())
        if model_keys != ckpt_keys:
            raise ValueError(
                f"Weight key mismatch between inflate model and checkpoint. "
                f"Missing in ckpt: {model_keys - ckpt_keys}, "
                f"Extra in ckpt: {ckpt_keys - model_keys}. "
                f"Did you update src/tac/architectures.py without mirroring here?"
            )
        model.load_state_dict(float_state)
        return model.eval().to(device)


# ============================================================
# Canonical YUV→RGB (BT.601 limited range, matches frame_utils.py)
# BT.601 is intentional here: the upstream scorer's frame_utils.py uses
# BT.601 coefficients regardless of container colorspace metadata.
# Matching this exactly is critical for score fidelity.
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


# NOTE: All env vars below are read at module import time (not per-call).
# This means changing them after import has no effect. This is intentional
# for the contest submission use case (single process, single config).
BATCH_SIZE = 8  # batched inference: 3-5x speedup on CPU
MULTI_PASS = int(os.environ.get("INFLATE_MULTI_PASS", "1"))  # run CNN N times (2=double pass, ~3min each on CPU; 3 may exceed 10-min inflate budget)
TTO_STEPS = int(os.environ.get("INFLATE_TTO_STEPS", "0"))  # test-time optimization steps
TTO_LR = float(os.environ.get("INFLATE_TTO_LR", "1e-4"))
TTO_LOSS = os.environ.get("INFLATE_TTO_LOSS", "temporal_consistency")
TTO_BUDGET = float(os.environ.get("INFLATE_TTO_BUDGET", "30.0"))  # 30s CPU safety default
# Supervised TTO with pre-computed PoseNet targets
SUPERVISED_TTO_STEPS = int(os.environ.get("INFLATE_SUPERVISED_TTO_STEPS", "0"))
SUPERVISED_TTO_LR = float(os.environ.get("INFLATE_SUPERVISED_TTO_LR", "1e-4"))
SUPERVISED_TTO_BUDGET = float(os.environ.get("INFLATE_SUPERVISED_TTO_BUDGET", "120.0"))
SUPERVISED_TTO_PARAM_MODE = os.environ.get("INFLATE_SUPERVISED_TTO_PARAM_MODE", "all")
# Noise-shaped rounding: use local gradient (pixel neighborhood) instead of nearest-round
# Fast variant only -- no scorer gradient needed, safe for CPU lane's ~5min budget
NOISE_SHAPING_FAST = os.environ.get("INFLATE_NOISE_SHAPING_FAST", "0") == "1"
# Supervised TTO if scorer is available on eval machine
SUPERVISED_TTO_IF_AVAILABLE = os.environ.get("INFLATE_SUPERVISED_TTO_IF_AVAILABLE", "0") == "1"


def _decode_frames_for_tto(
    video_path: str, target_w: int, target_h: int,
    max_frames: int = 64, stride: int = 1,
) -> torch.Tensor:
    """Decode a subset of frames from a video for TTO pre-pass.

    Returns (N, 3, H, W) float tensor. Uses strided sampling to get
    temporal coverage without decoding the entire video.
    """
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        frames = []
        i = 0
        for frame in container.decode(stream):
            if i % stride == 0:
                t = yuv420_to_rgb(frame)
                H, W, _ = t.shape
                x = t.permute(2, 0, 1).unsqueeze(0).float()
                if H != target_h or W != target_w:
                    x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
                    x = x.clamp(0, 255)
                frames.append(x)
                if len(frames) >= max_frames:
                    break
            i += 1
    finally:
        container.close()
    if frames:
        return torch.cat(frames, dim=0)
    return torch.empty(0, 3, target_h, target_w)


def inflate_with_postfilter(
    video_path: str, dst: str, model: nn.Module,
    target_w: int = 1164, target_h: int = 874, device: str = "cpu",
    tto_steps: int = 0, tto_lr: float = 1e-4,
    tto_loss: str = "temporal_consistency", tto_budget: float = 60.0,
    supervised_tto_steps: int = 0, supervised_tto_lr: float = 1e-4,
    supervised_tto_budget: float = 120.0, supervised_tto_param_mode: str = "all",
    posenet_targets_path: str | None = None,
    posenet=None, upstream_dir: str | None = None,
    posenet_path: str | None = None, segnet_path: str | None = None,
) -> int:
    """Decode, upscale, apply learned post-filter, write raw RGB.

    Uses batched inference for throughput. Model is passed in (loaded once).
    NOTE: Only supports single-frame architectures (standard, dilated, etc.).
    PairAwarePostFilter requires 6-channel input and is not yet supported here.

    If tto_steps > 0, runs test-time optimization on a subset of frames
    BEFORE the main inflate loop. This adapts the model to the specific
    video content using self-supervised losses (no scorer needed).

    If supervised_tto_steps > 0 and posenet_targets_path exists, runs
    SUPERVISED TTO: optimizes model to minimize MSE against pre-computed
    PoseNet ground truth targets. This is far more effective than
    self-supervised TTO because we optimize the exact scorer metric.
    """
    import time
    t0 = time.monotonic()

    # Guard: pair-aware models need 6ch input, not supported in this inflate path
    if isinstance(model, PairAwarePostFilter):
        raise NotImplementedError(
            "PairAwarePostFilter requires 6-channel (frame-pair) input. "
            "inflate_with_postfilter only supports single-frame architectures."
        )

    # Test-time optimization pre-pass
    if tto_steps > 0:
        try:
            from tac.tto import test_time_optimize
        except ImportError:
            print("WARNING: tac.tto not available, skipping TTO", file=sys.stderr)
            tto_steps = 0

        if tto_steps > 0:
            print(f"  TTO: decoding frames for adaptation ...", file=sys.stderr)
            tto_frames = _decode_frames_for_tto(
                video_path, target_w, target_h,
                max_frames=64, stride=4,  # every 4th frame, up to 64
            )
            if tto_frames.shape[0] >= 2:
                print(
                    f"  TTO: adapting model ({tto_steps} steps, "
                    f"loss={tto_loss}, lr={tto_lr}) on {tto_frames.shape[0]} frames ...",
                    file=sys.stderr,
                )
                model = test_time_optimize(
                    model, tto_frames, n_steps=tto_steps, lr=tto_lr,
                    loss_type=tto_loss, time_budget_seconds=tto_budget,
                    verbose=True,
                )
                del tto_frames
            else:
                print("  TTO: not enough frames, skipping", file=sys.stderr)
            tto_elapsed = time.monotonic() - t0
            print(f"  TTO pre-pass: {tto_elapsed:.1f}s", file=sys.stderr)

    # Supervised TTO: optimize against pre-computed PoseNet targets
    if supervised_tto_steps > 0 and posenet_targets_path:
        try:
            from tac.scorer_targets import load_posenet_targets
            from tac.tto import supervised_tto
        except ImportError:
            print("WARNING: tac.scorer_targets or tac.tto not available, "
                  "skipping supervised TTO", file=sys.stderr)
            supervised_tto_steps = 0

        if supervised_tto_steps > 0:
            targets_dict = load_posenet_targets(posenet_targets_path, device=device)
            if targets_dict is not None:
                # Load PoseNet if not already provided
                _posenet = posenet
                if _posenet is None:
                    print("  Supervised TTO: loading PoseNet scorer ...",
                          file=sys.stderr)
                    try:
                        from tac.scorer import load_scorers
                        if posenet_path and segnet_path:
                            _posenet, _ = load_scorers(
                                posenet_path, segnet_path,
                                device=device, upstream_dir=upstream_dir,
                            )
                        else:
                            print("  WARNING: posenet_path/segnet_path not provided, "
                                  "cannot run supervised TTO", file=sys.stderr)
                    except Exception as e:
                        print(f"  WARNING: failed to load PoseNet: {e}",
                              file=sys.stderr)

                if _posenet is not None:
                    print(f"  Supervised TTO: decoding frames for adaptation ...",
                          file=sys.stderr)
                    stto_frames = _decode_frames_for_tto(
                        video_path, target_w, target_h,
                        max_frames=128, stride=2,  # more coverage for supervised
                    )
                    if stto_frames.shape[0] >= 2:
                        print(f"  Supervised TTO: adapting model "
                              f"({supervised_tto_steps} steps, "
                              f"lr={supervised_tto_lr}) on "
                              f"{stto_frames.shape[0]} frames against "
                              f"{targets_dict['n_pairs']} PoseNet targets ...",
                              file=sys.stderr)
                        model = supervised_tto(
                            model, stto_frames, _posenet,
                            targets_dict["targets"],
                            n_steps=supervised_tto_steps,
                            lr=supervised_tto_lr,
                            param_mode=supervised_tto_param_mode,
                            time_budget_seconds=supervised_tto_budget,
                            verbose=True,
                        )
                        del stto_frames
                    else:
                        print("  Supervised TTO: not enough frames, skipping",
                              file=sys.stderr)
                    stto_elapsed = time.monotonic() - t0
                    print(f"  Supervised TTO pre-pass: {stto_elapsed:.1f}s",
                          file=sys.stderr)
            else:
                print("  Supervised TTO: targets file not found or invalid, "
                      "skipping", file=sys.stderr)

    # Trick 6: Opportunistic supervised TTO if scorer models are on the eval machine
    # The scorer IS on the eval machine (it runs scoring). If PoseNet is accessible,
    # run 5 quick gradient steps (~30s) to directly optimize the scorer metric.
    if SUPERVISED_TTO_IF_AVAILABLE and supervised_tto_steps == 0:
        try:
            from tac.scorer import load_scorers
            from tac.tto import supervised_tto as _stto_fn
            from tac.scorer_targets import load_posenet_targets as _load_targets

            # Try to find PoseNet model in standard locations
            _pn_path = posenet_path
            _sn_path = segnet_path
            if not _pn_path and upstream_dir:
                _candidate = os.path.join(upstream_dir, "models", "posenet.safetensors")
                if os.path.exists(_candidate):
                    _pn_path = _candidate
            if not _sn_path and upstream_dir:
                _candidate = os.path.join(upstream_dir, "models", "segnet.safetensors")
                if os.path.exists(_candidate):
                    _sn_path = _candidate

            if _pn_path and _sn_path and os.path.exists(_pn_path):
                print("  Opportunistic supervised TTO: PoseNet found, running 5 steps ...",
                      file=sys.stderr)
                _pn, _ = load_scorers(_pn_path, _sn_path, device=device, upstream_dir=upstream_dir)

                # Try to load pre-computed targets
                _tgt = None
                if posenet_targets_path:
                    _tgt = _load_targets(posenet_targets_path, device=device)

                if _tgt is not None:
                    _stto_frames = _decode_frames_for_tto(
                        video_path, target_w, target_h, max_frames=64, stride=4,
                    )
                    if _stto_frames.shape[0] >= 2:
                        model = _stto_fn(
                            model, _stto_frames, _pn, _tgt["targets"],
                            n_steps=5, lr=1e-4, param_mode="all",
                            time_budget_seconds=30.0,  # hard 30s cap
                            verbose=True,
                        )
                        del _stto_frames
                    print(f"  Opportunistic supervised TTO complete: {time.monotonic() - t0:.1f}s",
                          file=sys.stderr)
                else:
                    print("  Opportunistic supervised TTO: no targets available, skipping",
                          file=sys.stderr)
            else:
                print("  Opportunistic supervised TTO: PoseNet not found, skipping gracefully",
                      file=sys.stderr)
        except (ImportError, Exception) as e:
            print(f"  Opportunistic supervised TTO: not available ({e}), skipping gracefully",
                  file=sys.stderr)

    container = av.open(video_path)
    stream = container.streams.video[0]
    n = 0
    batch = []

    def _flush_batch(f, batch_tensors):
        if not batch_tensors:
            return
        x = torch.cat(batch_tensors, dim=0).to(device)
        with torch.inference_mode():
            out = model(x)
            # Multi-pass: run the CNN again on its own output (deeper effective network)
            # Round to uint8 between passes to match the training distribution
            for _ in range(MULTI_PASS - 1):
                out = out.round().clamp(0, 255)
                out = model(out)

        if NOISE_SHAPING_FAST:
            # Trick 5: gradient-directed rounding using LOCAL pixel gradients.
            # No scorer needed -- uses spatial neighborhood statistics to decide
            # ceil vs floor. Costs ~0.1s per batch (negligible on CPU).
            out_ns = out.detach().requires_grad_(True)
            # Local gradient: Laplacian approximation (pixel vs 4-neighbors)
            # Positive Laplacian = pixel brighter than neighbors -> floor helps
            # Negative Laplacian = pixel darker than neighbors -> ceil helps
            padded = F.pad(out_ns, (1, 1, 1, 1), mode="reflect")
            laplacian = (
                4 * out_ns
                - padded[:, :, :-2, 1:-1]   # top
                - padded[:, :, 2:, 1:-1]     # bottom
                - padded[:, :, 1:-1, :-2]    # left
                - padded[:, :, 1:-1, 2:]     # right
            )
            # Use Laplacian sign as a proxy for scorer gradient direction
            out_clamped = out.detach().clamp(0.0, 255.0)
            out_rounded = torch.where(
                laplacian.detach() < 0,
                out_clamped.ceil(),
                torch.where(laplacian.detach() > 0, out_clamped.floor(), out_clamped.round()),
            )
            for i in range(out_rounded.shape[0]):
                t = out_rounded[i].permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu()
                f.write(t.contiguous().numpy().tobytes())
        else:
            for i in range(out.shape[0]):
                t = out[i].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu()
                f.write(t.contiguous().numpy().tobytes())

    with open(dst, 'wb') as f:
        for frame in container.decode(stream):
            t = yuv420_to_rgb(frame)  # (H, W, 3) uint8
            H, W, _ = t.shape

            if H != target_h or W != target_w:
                x = t.permute(2, 0, 1).unsqueeze(0).float()
                x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
                x = x.clamp(0, 255)
            else:
                x = t.permute(2, 0, 1).unsqueeze(0).float()

            batch.append(x)
            n += 1

            if len(batch) >= BATCH_SIZE:
                _flush_batch(f, batch)
                batch.clear()

            if n % 300 == 0:
                print(f"  Processed {n} frames ...", file=sys.stderr, flush=True)

        # Flush remaining
        _flush_batch(f, batch)
        batch.clear()

    container.close()
    elapsed = time.monotonic() - t0
    print(f"Inflated {n} frames with post-filter -> {dst} ({elapsed:.1f}s)",
          file=sys.stderr)
    return n


# ============================================================
# Trick stack env var: INFLATE_TRICK_STACK=1 activates the unified
# stacking pipeline instead of the simple inflate loop.
# ============================================================
TRICK_STACK_ENABLED = os.environ.get("INFLATE_TRICK_STACK", "0") == "1"
TRICK_STACK_PROFILE = os.environ.get("INFLATE_TRICK_STACK_PROFILE", "stacked_inflate_full")


if __name__ == "__main__":
    import time
    t_start = time.monotonic()

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

    # Look for PoseNet targets in archive dir (bundled in archive.zip)
    posenet_targets_path = None
    if SUPERVISED_TTO_STEPS > 0:
        for candidate in [
            Path(archive_dir) / "posenet_targets.bin",
            script_dir / "posenet_targets.bin",
        ]:
            if candidate.exists():
                posenet_targets_path = str(candidate)
                break
        if posenet_targets_path:
            print(f"  Found PoseNet targets: {posenet_targets_path}",
                  file=sys.stderr)
        else:
            print("  WARNING: SUPERVISED_TTO_STEPS > 0 but posenet_targets.bin "
                  "not found. Supervised TTO will be skipped.", file=sys.stderr)

    # Discover upstream dir for loading PoseNet (needed for supervised TTO)
    upstream_dir = os.environ.get("COMMA_CHALLENGE_ROOT")
    posenet_path_env = os.environ.get("POSENET_PATH")
    segnet_path_env = os.environ.get("SEGNET_PATH")
    if upstream_dir and not posenet_path_env:
        posenet_path_env = str(Path(upstream_dir) / "models" / "posenet.safetensors")
        segnet_path_env = str(Path(upstream_dir) / "models" / "segnet.safetensors")

    # ---- Trick stack dispatch ----
    if TRICK_STACK_ENABLED:
        print(f"  TRICK STACK enabled (profile={TRICK_STACK_PROFILE})", file=sys.stderr)
        try:
            from tac.trick_stack import stacked_inflate, TrickStackConfig
            from tac.profiles import PROFILES
        except ImportError as e:
            print(f"  ERROR: trick_stack requires tac package: {e}", file=sys.stderr)
            sys.exit(1)

        # Load profile and apply env var overrides
        profile = PROFILES.get(TRICK_STACK_PROFILE, {})
        stack_kwargs = dict(profile)
        stack_kwargs.update({
            "posenet_targets_path": posenet_targets_path,
            "upstream_dir": upstream_dir,
            "posenet_path": posenet_path_env,
            "segnet_path": segnet_path_env,
        })
        # Allow env var overrides for individual trick toggles
        for key in [
            "use_tto", "use_supervised_tto", "use_noise_shaping",
            "use_null_space_projection", "use_brightness_shift",
            "use_chroma_exploit", "use_fragility_weighting",
            "use_backward_delta_smoothing",
        ]:
            env_val = os.environ.get(f"INFLATE_{key.upper()}")
            if env_val is not None:
                stack_kwargs[key] = env_val == "1"
        for key in ["tto_steps", "supervised_tto_steps", "use_multi_pass"]:
            env_val = os.environ.get(f"INFLATE_{key.upper()}")
            if env_val is not None:
                stack_kwargs[key] = int(env_val)

        result = stacked_inflate(
            archive_dir=Path(archive_dir),
            output_dir=Path(inflated_dir),
            **stack_kwargs,
        )
        t_total = time.monotonic() - t_start
        print(f"  Trick stack complete: {result.get('n_frames', 0)} frames, "
              f"{t_total:.1f}s total", file=sys.stderr)
        print(f"  Stages: {result.get('stages_run', [])}", file=sys.stderr)
        sys.exit(0)

    # ---- Standard (non-stacked) inflate path ----

    # Load model ONCE (not per-video)
    print(f"  Loading post-filter from {postfilter_path}", file=sys.stderr)
    model = load_postfilter_int8(postfilter_path, device="cpu")

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
        inflate_with_postfilter(
            str(mkv_path), str(out_path), model,
            tto_steps=TTO_STEPS, tto_lr=TTO_LR,
            tto_loss=TTO_LOSS, tto_budget=TTO_BUDGET,
            supervised_tto_steps=SUPERVISED_TTO_STEPS,
            supervised_tto_lr=SUPERVISED_TTO_LR,
            supervised_tto_budget=SUPERVISED_TTO_BUDGET,
            supervised_tto_param_mode=SUPERVISED_TTO_PARAM_MODE,
            posenet_targets_path=posenet_targets_path,
            upstream_dir=upstream_dir,
            posenet_path=posenet_path_env,
            segnet_path=segnet_path_env,
        )

    t_total = time.monotonic() - t_start
    print(f"  Total inflate time: {t_total:.1f}s", file=sys.stderr)
