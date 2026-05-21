# SPDX-License-Identifier: MIT
"""Portable SegNet/EfficientNet-B2-UNet building blocks.

ARCH-4a scope: expose the highest-leverage SegNet primitives without claiming
full upstream scorer parity.  The upstream scorer still owns authority:
``SegNet = smp.Unet('tu-efficientnet_b2', classes=5, activation=None)``.

This file gives MLX and PyTorch substrate trainers a shared implementation for
the pieces we need next: squeeze-excite / MBConv-style encoder blocks and the
SMP-shaped UNet decoder.  NumPy remains the weight interchange format, so MLX
local training can export to PyTorch/CUDA without inventing a separate state
authority surface.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from tac.portable_primitives.backend import Backend, resolve_backend
from tac.portable_primitives.nn import PortableConv2d, relu, sigmoid
from tac.portable_primitives.nn_extended import (
    PortableBatchNorm2d,
    PortableDepthwiseConv2d,
    silu,
)

__all__ = [
    "SEGNET_CLASSES",
    "SEGNET_DECODER_CHANNELS",
    "SEGNET_ENCODER_CHANNELS",
    "SEGNET_MODEL_INPUT_SIZE",
    "SEGNET_MODEL_INPUT_HW",
    "PortableConvBnAct2d",
    "PortableSqueezeExcite",
    "PortableMBConvBlock",
    "PortableUnetDecoderBlock",
    "PortableUnetDecoder",
    "PortableSegmentationHead",
]

SEGNET_CLASSES = 5
SEGNET_MODEL_INPUT_SIZE = (512, 384)  # upstream frame_utils.py: (W, H)
SEGNET_MODEL_INPUT_HW = (384, 512)
SEGNET_ENCODER_CHANNELS = (3, 16, 24, 48, 120, 352)
SEGNET_DECODER_CHANNELS = (256, 128, 64, 32, 16)


class PortableConvBnAct2d:
    """Conv2d + frozen BatchNorm2d + activation in portable NCHW layout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        backend: Backend | str,
        kernel_size: int = 3,
        activation: str = "relu",
        seed: int | None = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.activation = activation
        self.backend = resolve_backend(backend)
        self.conv = PortableConv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            backend=self.backend,
            seed=seed,
        )
        self.bn = PortableBatchNorm2d(self.out_channels, backend=self.backend)
        self.bn.eval()

    def __call__(self, x: Any) -> Any:
        y = self.bn(self.conv(x))
        if self.activation == "relu":
            return relu(y, backend=self.backend)
        if self.activation == "silu":
            return silu(y, backend=self.backend)
        if self.activation in {"none", None}:
            return y
        raise ValueError(f"unsupported activation: {self.activation!r}")

    def load_weights(
        self,
        conv_weight: np.ndarray,
        conv_bias: np.ndarray | None = None,
        *,
        bn_weight: np.ndarray | None = None,
        bn_bias: np.ndarray | None = None,
        bn_running_mean: np.ndarray | None = None,
        bn_running_var: np.ndarray | None = None,
    ) -> None:
        """Load canonical PyTorch-layout conv + BN weights."""

        if conv_bias is None:
            conv_bias = np.zeros((self.out_channels,), dtype=np.float32)
        self.conv.load_weights(conv_weight, conv_bias)
        self.bn.load_weights(
            weight_np=bn_weight,
            bias_np=bn_bias,
            running_mean_np=bn_running_mean,
            running_var_np=bn_running_var,
        )
        self.bn.eval()

    def export_weights(self) -> dict[str, np.ndarray]:
        conv_w, conv_b = self.conv.export_weights()
        exported = {"conv.weight": conv_w, "conv.bias": conv_b}
        for key, value in self.bn.export_weights().items():
            exported[f"bn.{key}"] = value
        return exported


class PortableSqueezeExcite:
    """EfficientNet-style squeeze-excite block with NumPy weight interchange."""

    def __init__(
        self,
        in_channels: int,
        *,
        backend: Backend | str,
        squeeze_channels: int | None = None,
        squeeze_ratio: float = 0.25,
        seed: int | None = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.squeeze_channels = (
            int(squeeze_channels)
            if squeeze_channels is not None
            else max(1, int(round(self.in_channels * float(squeeze_ratio))))
        )
        self.backend = resolve_backend(backend)
        base_seed = 0 if seed is None else int(seed)
        self.reduce = PortableConv2d(
            self.in_channels,
            self.squeeze_channels,
            kernel_size=1,
            backend=self.backend,
            seed=base_seed,
        )
        self.expand = PortableConv2d(
            self.squeeze_channels,
            self.in_channels,
            kernel_size=1,
            backend=self.backend,
            seed=base_seed + 1,
        )

    def __call__(self, x: Any) -> Any:
        pooled = _global_mean_2d(x, self.backend)
        gate = sigmoid(self.expand(silu(self.reduce(pooled), backend=self.backend)), backend=self.backend)
        return x * gate

    def load_weights(
        self,
        reduce_weight: np.ndarray,
        reduce_bias: np.ndarray,
        expand_weight: np.ndarray,
        expand_bias: np.ndarray,
    ) -> None:
        self.reduce.load_weights(reduce_weight, reduce_bias)
        self.expand.load_weights(expand_weight, expand_bias)

    def export_weights(self) -> dict[str, np.ndarray]:
        reduce_w, reduce_b = self.reduce.export_weights()
        expand_w, expand_b = self.expand.export_weights()
        return {
            "reduce.weight": reduce_w,
            "reduce.bias": reduce_b,
            "expand.weight": expand_w,
            "expand.bias": expand_b,
        }


class PortableMBConvBlock:
    """Small MBConv scaffold for EfficientNet-B2 encoder parity work."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        backend: Backend | str,
        expand_ratio: int = 6,
        kernel_size: int = 3,
        stride: int = 1,
        squeeze_ratio: float = 0.25,
        seed: int | None = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.expand_ratio = int(expand_ratio)
        self.mid_channels = self.in_channels * self.expand_ratio
        self.stride = int(stride)
        self.backend = resolve_backend(backend)
        base_seed = 0 if seed is None else int(seed)
        self.use_residual = self.stride == 1 and self.in_channels == self.out_channels

        self.expand = (
            PortableConvBnAct2d(
                self.in_channels,
                self.mid_channels,
                backend=self.backend,
                kernel_size=1,
                activation="silu",
                seed=base_seed,
            )
            if self.expand_ratio != 1
            else None
        )
        self.depthwise = PortableDepthwiseConv2d(
            self.mid_channels,
            kernel_size=kernel_size,
            backend=self.backend,
            stride=self.stride,
            padding=kernel_size // 2,
            bias=False,
            seed=base_seed + 1,
        )
        self.depthwise_bn = PortableBatchNorm2d(self.mid_channels, backend=self.backend)
        self.depthwise_bn.eval()
        self.se = PortableSqueezeExcite(
            self.mid_channels,
            backend=self.backend,
            squeeze_ratio=squeeze_ratio,
            seed=base_seed + 2,
        )
        self.project = PortableConvBnAct2d(
            self.mid_channels,
            self.out_channels,
            backend=self.backend,
            kernel_size=1,
            activation="none",
            seed=base_seed + 4,
        )

    def __call__(self, x: Any) -> Any:
        y = x if self.expand is None else self.expand(x)
        y = silu(self.depthwise_bn(self.depthwise(y)), backend=self.backend)
        y = self.se(y)
        y = self.project(y)
        return x + y if self.use_residual else y


class PortableUnetDecoderBlock:
    """SMP Unet decoder block: upsample, concat skip, conv, conv."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        backend: Backend | str,
        seed: int | None = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.skip_channels = int(skip_channels)
        self.out_channels = int(out_channels)
        self.backend = resolve_backend(backend)
        base_seed = 0 if seed is None else int(seed)
        self.conv1 = PortableConvBnAct2d(
            self.in_channels + self.skip_channels,
            self.out_channels,
            backend=self.backend,
            kernel_size=3,
            activation="relu",
            seed=base_seed,
        )
        self.conv2 = PortableConvBnAct2d(
            self.out_channels,
            self.out_channels,
            backend=self.backend,
            kernel_size=3,
            activation="relu",
            seed=base_seed + 1,
        )

    def __call__(self, x: Any, skip: Any | None = None) -> Any:
        if skip is None:
            h, w = _spatial_shape(x)
            target_hw = (h * 2, w * 2)
        else:
            target_hw = _spatial_shape(skip)
        x = _nearest_upsample_to(x, target_hw, self.backend)
        if skip is not None:
            x = _concat_channels((x, skip), self.backend)
        return self.conv2(self.conv1(x))


class PortableUnetDecoder:
    """SMP-shaped decoder for encoder channels ``[3, 16, 24, 48, 120, 352]``."""

    def __init__(
        self,
        *,
        backend: Backend | str,
        encoder_channels: Iterable[int] = SEGNET_ENCODER_CHANNELS,
        decoder_channels: Iterable[int] = SEGNET_DECODER_CHANNELS,
        seed: int | None = None,
    ) -> None:
        self.encoder_channels = tuple(int(v) for v in encoder_channels)
        self.decoder_channels = tuple(int(v) for v in decoder_channels)
        if len(self.encoder_channels) != 6:
            raise ValueError(f"expected 6 encoder channel entries, got {self.encoder_channels}")
        if len(self.decoder_channels) != 5:
            raise ValueError(f"expected 5 decoder channel entries, got {self.decoder_channels}")
        self.backend = resolve_backend(backend)
        base_seed = 0 if seed is None else int(seed)

        skip_channels = tuple(reversed(self.encoder_channels[1:-1])) + (0,)
        in_channels = (self.encoder_channels[-1],) + self.decoder_channels[:-1]
        self.blocks = [
            PortableUnetDecoderBlock(
                in_ch,
                skip_ch,
                out_ch,
                backend=self.backend,
                seed=base_seed + idx * 2,
            )
            for idx, (in_ch, skip_ch, out_ch) in enumerate(
                zip(in_channels, skip_channels, self.decoder_channels, strict=True)
            )
        ]

    def __call__(self, features: list[Any] | tuple[Any, ...]) -> Any:
        if len(features) != len(self.encoder_channels):
            raise ValueError(f"expected {len(self.encoder_channels)} feature tensors, got {len(features)}")
        x = features[-1]
        skips = list(reversed(features[1:-1]))
        for idx, block in enumerate(self.blocks):
            skip = skips[idx] if idx < len(skips) else None
            x = block(x, skip)
        return x


class PortableSegmentationHead:
    """Segmentation head matching SMP's 3x3 conv to five classes."""

    def __init__(
        self,
        in_channels: int = SEGNET_DECODER_CHANNELS[-1],
        classes: int = SEGNET_CLASSES,
        *,
        backend: Backend | str,
        seed: int | None = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.classes = int(classes)
        self.backend = resolve_backend(backend)
        self.conv = PortableConv2d(
            self.in_channels,
            self.classes,
            kernel_size=3,
            backend=self.backend,
            seed=seed,
        )

    def __call__(self, x: Any) -> Any:
        return self.conv(x)

    def load_weights(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.conv.load_weights(weight, bias)

    def export_weights(self) -> tuple[np.ndarray, np.ndarray]:
        return self.conv.export_weights()


def _global_mean_2d(x: Any, backend: Backend) -> Any:
    if backend is Backend.MLX:
        import mlx.core as mx

        return mx.mean(x, axis=(2, 3), keepdims=True)
    import torch

    return torch.mean(x, dim=(2, 3), keepdim=True)


def _spatial_shape(x: Any) -> tuple[int, int]:
    shape = tuple(x.shape)
    if len(shape) != 4:
        raise ValueError(f"expected NCHW tensor rank 4, got shape={shape}")
    return int(shape[2]), int(shape[3])


def _nearest_upsample_to(x: Any, target_hw: tuple[int, int], backend: Backend) -> Any:
    h, w = _spatial_shape(x)
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if (h, w) == (target_h, target_w):
        return x
    if backend is Backend.MLX:
        import mlx.core as mx

        repeat_h = max(1, int(math_ceil_div(target_h, h)))
        repeat_w = max(1, int(math_ceil_div(target_w, w)))
        y = mx.repeat(x, repeats=repeat_h, axis=2)
        y = mx.repeat(y, repeats=repeat_w, axis=3)
        return y[:, :, :target_h, :target_w]
    import torch.nn.functional as F

    return F.interpolate(x, size=(target_h, target_w), mode="nearest")


def _concat_channels(xs: tuple[Any, ...], backend: Backend) -> Any:
    if backend is Backend.MLX:
        import mlx.core as mx

        return mx.concatenate(xs, axis=1)
    import torch

    return torch.cat(xs, dim=1)


def math_ceil_div(num: int, den: int) -> int:
    return -(-int(num) // int(den))
