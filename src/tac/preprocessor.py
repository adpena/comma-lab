# SPDX-License-Identifier: MIT
"""Netflix-inspired learned pre-processor before codec encoding.

Technique 1: A small CNN that adjusts pixel values BEFORE AV1 encoding.
The pre-processor biases pixels toward what PoseNet/SegNet prefer, so that
codec compression artifacts align with scorer sensitivities rather than
fighting them.

Architecture: same 3-layer residual as the postfilter, but applied BEFORE
encoding. Can be trained jointly with the postfilter through a differentiable
codec proxy, or separately against scorer gradients on GT frames.

Integration: insert before ffmpeg in compress.sh pipeline:
    python preprocessor_apply.py input.mkv preprocessed.mkv
    ffmpeg -i preprocessed.mkv ... (standard AV1 encode)

The pre-processor output is NOT part of the archive (it modifies the video
before encoding, not after decoding), so it has zero rate cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedPreProcessor(nn.Module):
    """Pre-codec CNN that biases pixels toward scorer preferences.

    Architecture: 3-layer residual CNN (same pattern as PostFilter).
    Applied to GT frames BEFORE AV1 encoding. The idea is that small
    pixel adjustments before compression can steer codec artifacts into
    regions the scorers are less sensitive to.

    The correction is small (scaled by `strength` parameter) to avoid
    visible artifacts in the encoded video.

    Args:
        hidden: hidden channel width (16-32 recommended, smaller than postfilter)
        kernel: conv kernel size
        strength: max correction magnitude as fraction of 255 (default 0.02 = ~5 pixel values)
    """

    # hidden=16: small RF (7x7), less capacity needed
    def __init__(self, hidden: int = 16, kernel: int = 3, strength: float = 0.02):
        super().__init__()
        pad = kernel // 2
        self.strength = strength
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        # Zero-init output: starts as identity
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-process frames before codec encoding.

        Args:
            x: (B, 3, H, W) float [0, 255] — GT frames

        Returns:
            (B, 3, H, W) float [0, 255] — adjusted frames for encoding
        """
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        # tanh squashes to [-1, 1], then scale by strength * 255
        residual = torch.tanh(self.conv3(residual)) * (self.strength * 255.0)
        return (x + residual).clamp(0, 255)


class DilatedPreProcessor(nn.Module):
    """Pre-codec CNN with dilation=2 for larger receptive field.

    Same idea as LearnedPreProcessor but with dilated middle layer
    (matching DilatedPostFilter). The larger RF lets the pre-processor
    see more context when deciding how to adjust pixels.

    Args:
        hidden: hidden channel width
        kernel: conv kernel size
        strength: max correction magnitude as fraction of 255
    """

    # hidden=32: dilated middle layer gives 15x15 RF, needs more capacity
    # to leverage the larger receptive field effectively
    def __init__(self, hidden: int = 32, kernel: int = 3, strength: float = 0.02):
        super().__init__()
        pad = kernel // 2
        self.strength = strength
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = torch.tanh(self.conv3(residual)) * (self.strength * 255.0)
        return (x + residual).clamp(0, 255)


class JointPrePostFilter(nn.Module):
    """Joint pre-processor + post-filter with differentiable codec proxy.

    For end-to-end training, we need a differentiable approximation of the
    codec. This module wraps pre-processor and post-filter with a simple
    additive noise + quantization proxy for the AV1 codec.

    Pipeline: GT -> PreProcessor -> CodecProxy -> PostFilter -> Scored

    The codec proxy adds:
        1. Uniform quantization noise (simulates compression)
        2. Gaussian blur (simulates block-boundary smoothing)
        3. Optional downscale/upscale (simulates resolution reduction)

    Args:
        preprocessor: LearnedPreProcessor or DilatedPreProcessor
        postfilter: any postfilter from architectures.py
        noise_std: codec noise standard deviation (higher = more aggressive codec)
    """

    def __init__(
        self,
        preprocessor: nn.Module,
        postfilter: nn.Module,
        noise_std: float = 3.0,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.postfilter = postfilter
        self.noise_std = noise_std

    def _codec_proxy(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable codec proxy: quantization noise + STE rounding."""
        if self.training:
            # Add quantization-like noise
            noise = torch.randn_like(x) * self.noise_std
            # STE rounding: forward rounds to integers, backward passes through
            rounded = x + noise
            rounded = rounded + (rounded.round() - rounded).detach()
            return rounded.clamp(0, 255)
        else:
            # Eval mode: deterministic rounding. While this creates a slight
            # train/eval distribution gap (training uses noise-based STE),
            # eval must be deterministic for reproducible scores.
            return x.round().clamp(0, 255)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end: preprocess -> codec proxy -> postfilter.

        Args:
            x: (B, 3, H, W) float [0, 255] — GT frames

        Returns:
            (B, 3, H, W) float [0, 255] — final output
        """
        preprocessed = self.preprocessor(x)
        compressed = self._codec_proxy(preprocessed)
        return self.postfilter(compressed)


PREPROCESSOR_VARIANTS = {
    "standard": LearnedPreProcessor,
    "dilated": DilatedPreProcessor,
}


def build_preprocessor(
    variant: str = "standard",
    hidden: int = 16,
    kernel: int = 3,
    strength: float = 0.02,
) -> nn.Module:
    """Build a learned pre-processor by variant name.

    Args:
        variant: "standard" or "dilated"
        hidden: hidden channel width (16-32 recommended)
        kernel: conv kernel size
        strength: max correction magnitude as fraction of 255

    Returns:
        An nn.Module that takes (B, 3, H, W) float [0, 255] and returns same.
    """
    cls = PREPROCESSOR_VARIANTS.get(variant)
    if cls is None:
        raise ValueError(f"Unknown preprocessor variant '{variant}'. Available: {list(PREPROCESSOR_VARIANTS.keys())}")
    return cls(hidden=hidden, kernel=kernel, strength=strength)
