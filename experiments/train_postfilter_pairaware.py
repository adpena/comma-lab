#!/usr/bin/env python
"""Pair-aware 6-channel post-filter scaffold.

This is a minimal experiment lane for a post-filter that reads both frames
in a pair as a single 6-channel tensor and emits a corrected version of the
current frame only.

The scaffold is intentionally small:
- no training loop
- no dataset plumbing
- no launch/orchestration system

It is still runnable as a tiny dry-run CLI so the architecture can be smoke
checked before being wired into a larger experiment.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn


class PairAwarePostFilter(nn.Module):
    """Residual CNN that reads a 6-channel pair and corrects the current frame."""

    def __init__(self, hidden: int = 16, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(6, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=False)

        # Start as an identity map on the current frame.
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, pair_6ch: torch.Tensor) -> torch.Tensor:
        """pair_6ch: (B, 6, H, W) float tensor in [0, 255]."""
        if pair_6ch.ndim != 4 or pair_6ch.shape[1] != 6:
            raise ValueError(f"expected (B, 6, H, W), got {tuple(pair_6ch.shape)}")

        current = pair_6ch[:, 3:6]
        residual = self.act(self.conv1(pair_6ch))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (current + residual).clamp(0, 255)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pair-aware 6-channel post-filter dry-run")
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--height", type=int, default=8)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    return p


def pair_bhwc_to_6ch(pair_b2hwc: torch.Tensor) -> torch.Tensor:
    """Convert (B, 2, H, W, 3) pair tensors into (B, 6, H, W)."""
    if pair_b2hwc.ndim != 5 or pair_b2hwc.shape[1] != 2 or pair_b2hwc.shape[-1] != 3:
        raise ValueError(f"expected (B, 2, H, W, 3), got {tuple(pair_b2hwc.shape)}")
    b, _, h, w, _ = pair_b2hwc.shape
    return pair_b2hwc.float().permute(0, 1, 4, 2, 3).reshape(b, 6, h, w).contiguous()


def current_frame_bchw_from_pair(pair_b2hwc: torch.Tensor) -> torch.Tensor:
    """Extract the current frame, which is the second frame in the pair."""
    if pair_b2hwc.ndim != 5 or pair_b2hwc.shape[1] != 2 or pair_b2hwc.shape[-1] != 3:
        raise ValueError(f"expected (B, 2, H, W, 3), got {tuple(pair_b2hwc.shape)}")
    return pair_b2hwc[:, 1].float().permute(0, 3, 1, 2).contiguous()


def apply_pairaware_postfilter(model: PairAwarePostFilter, pair_b2hwc: torch.Tensor) -> torch.Tensor:
    """Run the model and return the corrected current frame as (B, 3, H, W)."""
    return model(pair_bhwc_to_6ch(pair_b2hwc))


def make_dummy_pair(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Create a deterministic dummy pair for the CLI dry-run."""
    prev = torch.full((batch_size, height, width, 3), 17.0, device=device)
    curr = torch.full((batch_size, height, width, 3), 231.0, device=device)
    return torch.stack([prev, curr], dim=1)


def dry_run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    model = PairAwarePostFilter(hidden=args.hidden, kernel=args.kernel).to(device)
    model.eval()

    pair = make_dummy_pair(args.batch_size, args.height, args.width, device)
    corrected = apply_pairaware_postfilter(model, pair)
    current = current_frame_bchw_from_pair(pair)

    delta = corrected - current
    summary = {
        "device": str(device),
        "model": "PairAwarePostFilter",
        "hidden": int(args.hidden),
        "kernel": int(args.kernel),
        "input_shape": list(pair.shape),
        "pair_6ch_shape": list(pair_bhwc_to_6ch(pair).shape),
        "output_shape": list(corrected.shape),
        "max_abs_delta": float(delta.abs().max().item()),
        "mean_abs_delta": float(delta.abs().mean().item()),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_arg_parser().parse_args(argv)
    return dry_run(args)


if __name__ == "__main__":
    main()
