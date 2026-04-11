#!/usr/bin/env python
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


class DepthwisePostFilter(nn.Module):
    def __init__(self, hidden=16, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.pw_in = nn.Conv2d(3, hidden, 1, bias=True)
        self.dw = nn.Conv2d(hidden, hidden, kernel, padding=pad, groups=hidden, bias=True)
        self.pw_out = nn.Conv2d(hidden, 3, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

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

    def forward(self, x):
        y = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
        residual = self.act(self.conv1(y))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual.repeat(1, 3, 1, 1)).clamp(0, 255)


def load_postfilter_int8(path, device="cpu"):
    """Load int8-quantized post-filter weights."""
    state = torch.load(path, map_location=device, weights_only=True)
    meta = state.get("__meta__", {"variant": "residual", "hidden": 16, "kernel": 3})
    float_state = {}
    keys = set(k.rsplit(".", 1)[0] for k in state.keys() if k != "__meta__")
    for key in keys:
        q = state[key + ".q"].float()
        s = state[key + ".s"]
        # Handle per-channel quantization: s may be (out_channels,) not scalar
        if s.ndim > 0 and q.ndim > 1:
            s = s.view(s.shape[0], *([1] * (q.ndim - 1)))
        float_state[key] = q * s
    variant = meta.get("variant", "residual")
    hidden = int(meta.get("hidden", 16))
    kernel = int(meta.get("kernel", 3))
    if variant == "depthwise":
        model = DepthwisePostFilter(hidden=hidden, kernel=kernel)
    elif variant == "luma":
        model = LumaPostFilter(hidden=hidden, kernel=kernel)
    else:
        model = PostFilter(hidden=hidden, kernel=kernel)
    model.load_state_dict(float_state)
    return model.eval().to(device)
