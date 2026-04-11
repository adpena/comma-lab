#!/usr/bin/env python
# ============================================================================
# LEGACY — This script predates the tac library and is superseded by:
#   python experiments/train_tac.py --profile pixelshuffle_dilated_smoke
# Unique logic has been migrated to src/tac/. Kept for git history reference.
# ============================================================================
"""Train a hybrid PixelShuffle + dilated post-filter on top of QAT+EMA.

This wrapper keeps the current saliency-weighted QAT+EMA recipe, but swaps
the residual CNN for a half-resolution PixelUnshuffle/PixelShuffle path with
a dilated middle layer and an h64 default width.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train_postfilter_qat_ema as _base  # type: ignore
from train_postfilter_qat_ema import EMA, FakeQuantSTE, save_best_checkpoint  # type: ignore


DEFAULT_HIDDEN = 64
DEFAULT_KERNEL = 3


def fake_quant(t: torch.Tensor) -> torch.Tensor:
    return FakeQuantSTE.apply(t)


class PixelShuffleDilatedPostFilter(nn.Module):
    """Half-resolution residual post-filter with a dilated middle layer."""

    def __init__(self, hidden: int = DEFAULT_HIDDEN, kernel: int = DEFAULT_KERNEL):
        super().__init__()
        pad = kernel // 2
        self.down = nn.PixelUnshuffle(2)
        self.conv1 = nn.Conv2d(12, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv4 = nn.Conv2d(hidden, 12, kernel, padding=pad, bias=True)
        self.up = nn.PixelShuffle(2)
        self.act = nn.ReLU(inplace=False)

        nn.init.zeros_(self.conv4.weight)
        nn.init.zeros_(self.conv4.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / 255.0
        residual = self.down(x_norm)
        residual = self.act(self.conv1(residual))
        residual = self.act(self.conv2(residual))
        residual = self.act(self.conv3(residual))
        residual = self.up(self.conv4(residual))
        return (x_norm + residual).clamp(0, 1) * 255.0


class QATPixelShuffleDilatedPostFilter(PixelShuffleDilatedPostFilter):
    """Weight-only QAT wrapper for the hybrid PixelShuffle+dilated model."""

    def _qconv(self, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        wq = fake_quant(conv.weight)
        bq = fake_quant(conv.bias) if conv.bias is not None else None
        return F.conv2d(
            x,
            wq,
            bq,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / 255.0
        residual = self.down(x_norm)
        residual = self.act(self._qconv(self.conv1, residual))
        residual = self.act(self._qconv(self.conv2, residual))
        residual = self.act(self._qconv(self.conv3, residual))
        residual = self.up(self._qconv(self.conv4, residual))
        return (x_norm + residual).clamp(0, 1) * 255.0


# Compatibility aliases for the existing QAT+EMA trainer internals.
PostFilter = PixelShuffleDilatedPostFilter
QATPostFilter = QATPixelShuffleDilatedPostFilter


def normalize_postfilter_meta(hidden: int, kernel: int, alpha: float) -> dict[str, int | float | str]:
    return {
        "variant": "pixelshuffle_dilated",
        "hidden": int(hidden),
        "kernel": int(kernel),
        "alpha": float(alpha),
        "scale": 2,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hybrid PixelShuffle + dilated QAT+EMA post-filter")
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=20.0,
                   help="Saliency emphasis: weight = 1 + alpha * saliency")
    p.add_argument("--sal-lambda", type=float, default=0.1)
    p.add_argument("--train-subsample", type=int, default=8)
    p.add_argument("--eval-subsample", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--eager-pair-transfer", action="store_true",
                   help="Move all decoded pairs to DEVICE up front. Default keeps pairs on CPU and transfers per step.")
    p.add_argument("--cuda-autocast", action="store_true",
                   help="Use fp16 autocast for scorer/model forwards on CUDA to reduce memory pressure.")
    p.add_argument("--restart-t0", type=int, default=0,
                   help="Enable CosineAnnealingWarmRestarts with this initial cycle length when > 0.")
    p.add_argument("--restart-tmult", type=int, default=2)
    p.add_argument("--restart-eta-min", type=float, default=1e-5)
    p.add_argument("--swa-start-epoch", type=int, default=0,
                   help="If > 0, start averaging EMA shadows into an SWA shadow from this epoch onward.")
    p.add_argument("--swa-every", type=int, default=10,
                   help="Average one EMA shadow into SWA every N epochs once SWA starts.")
    p.add_argument("--checkpoint-eval-every", type=int, default=10,
                   help="Run eval-based checkpoint selection every N epochs.")
    p.add_argument("--checkpoint-select-int8", action="store_true",
                   help="Select best checkpoints after quantizing the EMA shadow like the saved int8 payload.")
    p.add_argument("--per-channel-int8", action="store_true",
                   help="Save int8 checkpoints with per-channel conv scales and fp32 biases.")
    p.add_argument("--tag", type=str, default="psd_h64_long1000")
    return p


def main(argv: list[str] | None = None) -> dict[str, object]:
    old_build_arg_parser = _base.build_arg_parser
    old_postfilter = _base.PostFilter
    old_qat_postfilter = _base.QATPostFilter
    old_meta = _base.normalize_postfilter_meta
    try:
        _base.build_arg_parser = build_arg_parser
        _base.PostFilter = PixelShuffleDilatedPostFilter
        _base.QATPostFilter = QATPixelShuffleDilatedPostFilter
        _base.normalize_postfilter_meta = normalize_postfilter_meta
        return _base.main(argv)
    finally:
        _base.build_arg_parser = old_build_arg_parser
        _base.PostFilter = old_postfilter
        _base.QATPostFilter = old_qat_postfilter
        _base.normalize_postfilter_meta = old_meta


if __name__ == "__main__":
    main()
