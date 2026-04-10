#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "submissions" / "robust_current"))

from inflate_postfilter import DilatedPostFilter  # type: ignore
import train_postfilter_qat_ema as trainer  # type: ignore


class QATDilatedPostFilter(nn.Module):
    def __init__(self, hidden: int = 64, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=False)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def _qconv(self, conv: nn.Conv2d, x):
        from train_postfilter_qat_ema import fake_quant  # type: ignore

        wq = fake_quant(conv.weight)
        bq = fake_quant(conv.bias) if conv.bias is not None else None
        return F.conv2d(
            x,
            wq,
            bq,
            padding=conv.padding,
            stride=conv.stride,
            dilation=conv.dilation,
        )

    def forward(self, x):
        residual = self.act(self._qconv(self.conv1, x))
        residual = self.act(self._qconv(self.conv2, residual))
        residual = self._qconv(self.conv3, residual)
        return (x + residual).clamp(0, 255)


def normalize_postfilter_meta(hidden: int, kernel: int, alpha: float) -> dict:
    return {
        "variant": "dilated",
        "hidden": int(hidden),
        "kernel": int(kernel),
        "alpha": float(alpha),
    }


trainer.QATPostFilter = QATDilatedPostFilter
trainer.PostFilter = DilatedPostFilter
trainer.normalize_postfilter_meta = normalize_postfilter_meta


def main(argv: list[str] | None = None):
    argv = list(argv or sys.argv[1:])
    if "--tag" not in argv:
        argv.extend(["--tag", "dilated_h64_long1000"])
    return trainer.main(argv)


if __name__ == "__main__":
    main()
