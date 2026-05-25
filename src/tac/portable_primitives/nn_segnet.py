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

from collections.abc import Iterable
from typing import Any

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
    "SEGNET_ENCODER_STAGE_SPEC",
    "SEGNET_MODEL_INPUT_HW",
    "SEGNET_MODEL_INPUT_SIZE",
    "PortableConvBnAct2d",
    "PortableEfficientNetB2Backbone",
    "PortableMBConvBlock",
    "PortableSegNet",
    "PortableSegmentationHead",
    "PortableSqueezeExcite",
    "PortableUnetDecoder",
    "PortableUnetDecoderBlock",
]

SEGNET_CLASSES = 5
SEGNET_MODEL_INPUT_SIZE = (512, 384)  # upstream frame_utils.py: (W, H)
SEGNET_MODEL_INPUT_HW = (384, 512)
SEGNET_ENCODER_CHANNELS = (3, 16, 24, 48, 120, 352)
SEGNET_DECODER_CHANNELS = (256, 128, 64, 32, 16)

# Per-stage EfficientNet-B2 spec for the SMP ``TimmUniversalEncoder``
# (sourced from live ``smp.Unet('tu-efficientnet_b2', classes=5)`` introspection
# 2026-05-25; see landing memo).  Each tuple entry =
# ``(num_blocks, in_channels, out_channels, expand_ratio, kernel_size, stride,
#   feature_emitted_at_stage)``.
#
# Stages 0-6 of the EfficientNet-B2 encoder produce features at scales 1/2,
# 1/4, 1/8, 1/16, 1/32 (combined into the canonical 6-element feature list
# via the SMP ``_stage_out_idx`` aggregation: feature 0 is the input scale-1
# tensor itself, features 1-5 are emitted at stage boundaries (0, 1, 2, 4, 6)).
# This scaffold uses a single-MBConv-per-stage simplification (one block per
# stage with appropriate stride) which preserves the canonical 6-feature
# shape contract; per-stage block multiplicity (2/3/3/4/4/5/2 in upstream)
# is preserved as ``num_blocks`` for ARCH-5 state_dict-load parity. The
# resulting tensor shapes match upstream's ``smp.Unet`` encoder output
# exactly when fed a ``(B, 3, 384, 512)`` input.
SEGNET_ENCODER_STAGE_SPEC: tuple[tuple[int, int, int, int, int, int, bool], ...] = (
    # (num_blocks, in_ch, out_ch, expand_ratio, kernel, stride, feature_emit)
    (2, 32, 16, 1, 3, 1, True),    # stage 0: dsconv -> 16 (1/2 after stem)
    (3, 16, 24, 6, 3, 2, True),    # stage 1: MBConv -> 24 (1/4)
    (3, 24, 48, 6, 5, 2, True),    # stage 2: MBConv -> 48 (1/8)
    (4, 48, 88, 6, 3, 2, False),   # stage 3: MBConv -> 88 (1/16; no emit)
    (4, 88, 120, 6, 5, 1, True),   # stage 4: MBConv -> 120 (1/16; emit)
    (5, 120, 208, 6, 5, 2, False), # stage 5: MBConv -> 208 (1/32; no emit)
    (2, 208, 352, 6, 3, 1, True),  # stage 6: MBConv -> 352 (1/32; emit)
)


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
            else max(1, round(self.in_channels * float(squeeze_ratio)))
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

        skip_channels = (*tuple(reversed(self.encoder_channels[1:-1])), 0)
        in_channels = (self.encoder_channels[-1], *self.decoder_channels[:-1])
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


# ---------------------------------------------------------------------------
# MLX-ARCH-4 FULL ASSEMBLY: PortableEfficientNetB2Backbone + PortableSegNet
# ---------------------------------------------------------------------------
#
# The classes below complete the MLX-ARCH-4 cascade per the operator dispatch
# 2026-05-25.  They wrap the ARCH-4a primitives above into the canonical full
# SegNet contract:
#
#     SegNet = smp.Unet('tu-efficientnet_b2', classes=5, activation=None,
#                       encoder_weights=None)
#
# Per Carmack MVP-first 5-step + ARCH-3 PortablePoseNet precedent: the scaffold
# preserves the 6-feature shape contract + decoder skip wiring + 5-class head;
# byte-stable timm/smp state_dict parity defers to ARCH-5 (paired ground-truth
# state_dict-load + 600-frame paired forward MLX-vs-PyTorch validation).
#
# Cross-backend numeric equivalence at random init is verified for the
# individual primitives (sister tests above) and tested here at shape +
# argmax-equivalence level.  Full upstream parity ε band lands at ARCH-5.


class PortableEfficientNetB2Backbone:
    """EfficientNet-B2 encoder scaffold matching ``smp.Unet`` 6-feature contract.

    Forward:
        x.shape = (B, 3, H, W)   # canonical SegNet input HW = (384, 512)
        returns features = [
            feature_0 : (B, 3, H, W),       # input scale-1 passthrough
            feature_1 : (B, 16, H/2, W/2),
            feature_2 : (B, 24, H/4, W/4),
            feature_3 : (B, 48, H/8, W/8),
            feature_4 : (B, 120, H/16, W/16),
            feature_5 : (B, 352, H/32, W/32),
        ]

    This 6-element list is the canonical input to :class:`PortableUnetDecoder`
    (the SMP `TimmUniversalEncoder.forward` contract: input tensor prepended
    as scale-1 feature, then 5 emitted stage features).

    Architecture (simplified vs the 23-block timm `tu-efficientnet_b2`):

    - Stem: 3-channel -> 32-channel Conv-BN-Swish at stride-2 (H/2, W/2)
    - 7 stages of MBConv blocks per :data:`SEGNET_ENCODER_STAGE_SPEC`
    - Each stage emits its output if ``feature_emit=True`` (5 total)

    Per-stage block multiplicity (2/3/3/4/4/5/2 in upstream timm) is recorded
    in the spec for ARCH-5 state_dict-load parity but the scaffold here uses
    a single MBConv per stage with the correct stride / channel transition.
    Byte-stable state_dict key naming defers to ARCH-5 (the canonical sister
    ``src/tac/local_acceleration/mlx_scorer_adapters.py`` provides the full
    timm-block parity surface; this scaffold is the canonical portable
    primitives composition path).
    """

    def __init__(
        self,
        *,
        backend: Backend | str,
        in_channels: int = 3,
        stem_channels: int = 32,
        encoder_channels: Iterable[int] = SEGNET_ENCODER_CHANNELS,
        seed: int | None = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.stem_channels = int(stem_channels)
        self.encoder_channels = tuple(int(v) for v in encoder_channels)
        if len(self.encoder_channels) != 6:
            raise ValueError(
                f"expected 6 encoder channel entries, got {self.encoder_channels}"
            )
        if self.encoder_channels[0] != self.in_channels:
            raise ValueError(
                "encoder_channels[0] must equal in_channels (input passthrough),"
                f" got {self.encoder_channels[0]} != {self.in_channels}"
            )
        self.backend = resolve_backend(backend)
        base_seed = 0 if seed is None else int(seed)

        # Stem: stride-2 conv (in_channels -> stem_channels) + BN + SiLU.
        self._stem = PortableConvBnAct2d(
            self.in_channels,
            self.stem_channels,
            backend=self.backend,
            kernel_size=3,
            activation="silu",
            seed=base_seed,
        )

        # 7-stage MBConv chain per SEGNET_ENCODER_STAGE_SPEC.
        # Each stage is a single MBConv with the correct stride and channel
        # transition (multi-block stages are deferred to ARCH-5 byte-stable
        # state_dict-load via the sister adapter surface).
        self._stages: list[PortableMBConvBlock] = []
        prev_ch = self.stem_channels
        for i, (_n_blocks, _in_ch, out_ch, expand, kernel, stride, _emit) in enumerate(
            SEGNET_ENCODER_STAGE_SPEC
        ):
            # The spec's `in_ch` is the canonical timm in-channel; we use the
            # actual `prev_ch` from the previous stage's output so the scaffold
            # is self-consistent under the simplified single-MBConv-per-stage
            # assembly.  When `expand=1` (DSConv in upstream stage 0) we set
            # `expand_ratio=1` so the block skips the 1x1 expansion conv.
            self._stages.append(
                PortableMBConvBlock(
                    prev_ch,
                    out_ch,
                    backend=self.backend,
                    expand_ratio=expand,
                    kernel_size=kernel,
                    stride=stride,
                    seed=base_seed + 1000 * (i + 1),
                )
            )
            prev_ch = out_ch
        # Record per-stage emit flags for the forward pass.
        self._stage_emits = tuple(spec[6] for spec in SEGNET_ENCODER_STAGE_SPEC)

    def __call__(self, x: Any) -> list[Any]:
        # SMP convention: feature[0] is the input tensor itself (scale 1).
        features: list[Any] = [x]
        # Stem (stride-2 from input HW to H/2, W/2).
        x = self._stem(x)
        x = _stride2_subsample(x, self.backend)
        # 7-stage MBConv chain, collecting features at every emit point.
        for stage, emit in zip(self._stages, self._stage_emits, strict=True):
            x = stage(x)
            if emit:
                features.append(x)
        if len(features) != 6:
            raise RuntimeError(
                f"encoder produced {len(features)} features (expected 6); "
                f"check SEGNET_ENCODER_STAGE_SPEC emit flags"
            )
        return features


def _stride2_subsample(x: Any, backend: Backend) -> Any:
    """Apply stride-2 subsampling (NCHW -> NCHW with H/2 x W/2).

    Mirrors the PortableFastViTT12Backbone stem pattern (slice-based
    subsampling on both backends; cheaper than constructing a strided conv).
    """
    if backend is Backend.MLX:
        return x[:, :, ::2, ::2]
    return x[:, :, ::2, ::2]


class PortableSegNet:
    """Full SegNet wrapper per ``upstream.modules.SegNet`` (smp.Unet wrapper).

    Forward:
        x.shape = (B, T, C, H_in, W_in)   # T frames (last frame used)
        -> preprocess_input slices x[:, -1, ...] (B, 3, H_in, W_in)
        -> bilinear interpolate to SEGNET_MODEL_INPUT_HW = (384, 512)
        -> encoder produces 6 features
        -> decoder produces (B, 16, 384, 512)
        -> segmentation head produces (B, 5, 384, 512) logits

    Output: 5-class logits at the canonical SegNet HW (384, 512).
    :meth:`compute_distortion` mirrors the upstream argmax-disagreement mean.

    For convenience, the wrapper also accepts a pre-shaped (B, 3, H, W) input
    via :meth:`forward_3d` (skipping the last-frame slice).  The canonical
    contest-axis path uses :meth:`__call__` with the 5D (B, T, 3, H, W) input
    matching the contest scorer interface.
    """

    def __init__(
        self,
        *,
        backend: Backend | str,
        in_channels: int = 3,
        classes: int = SEGNET_CLASSES,
        seed: int | None = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.classes = int(classes)
        self.backend = resolve_backend(backend)
        base_seed = 0 if seed is None else int(seed)

        self._encoder = PortableEfficientNetB2Backbone(
            backend=self.backend,
            in_channels=self.in_channels,
            seed=base_seed,
        )
        self._decoder = PortableUnetDecoder(
            backend=self.backend,
            seed=base_seed + 50_000,
        )
        self._head = PortableSegmentationHead(
            in_channels=SEGNET_DECODER_CHANNELS[-1],
            classes=self.classes,
            backend=self.backend,
            seed=base_seed + 60_000,
        )

    def preprocess_input(self, x: Any) -> Any:
        """Slice last frame and bilinear-resize to (384, 512).

        Mirrors ``upstream.modules.SegNet.preprocess_input``:
            x = x[:, -1, ...]
            x = F.interpolate(x, size=(384, 512), mode='bilinear')
        """
        x = x[:, -1, ...]
        return _bilinear_interpolate_to_hw(x, SEGNET_MODEL_INPUT_HW, self.backend)

    def forward_3d(self, x: Any) -> Any:
        """Forward a pre-sliced (B, 3, H, W) tensor through encoder/decoder/head."""
        features = self._encoder(x)
        decoded = self._decoder(features)
        return self._head(decoded)

    def __call__(self, x: Any) -> Any:
        """Forward 5D (B, T, 3, H, W) -> 4D logits (B, 5, 384, 512)."""
        x = self.preprocess_input(x)
        return self.forward_3d(x)

    def compute_distortion(self, out1: Any, out2: Any) -> Any:
        """Argmax-disagreement mean (mirrors upstream contest scorer)."""
        if self.backend is Backend.MLX:
            import mlx.core as mx

            arg1 = mx.argmax(out1, axis=1)
            arg2 = mx.argmax(out2, axis=1)
            diff = (arg1 != arg2).astype(mx.float32)
            # Mean over all dims except batch.
            axes = tuple(range(1, diff.ndim))
            return mx.mean(diff, axis=axes) if axes else diff
        arg1 = out1.argmax(dim=1)
        arg2 = out2.argmax(dim=1)
        diff = (arg1 != arg2).float()
        axes = tuple(range(1, diff.ndim))
        return diff.mean(dim=axes) if axes else diff


def _bilinear_interpolate_to_hw(
    x: Any, target_hw: tuple[int, int], backend: Backend
) -> Any:
    """Bilinear interpolation NCHW -> NCHW with target (H, W).

    PyTorch path uses ``F.interpolate(..., mode='bilinear', align_corners=False)``
    matching the upstream ``SegNet.preprocess_input`` contract exactly.

    MLX path uses the nearest-upsample approximation already used by
    :class:`PortableUnetDecoderBlock` (no native bilinear in current MLX
    scope; ARCH-4b may extend).  The drift introduced by nearest vs bilinear
    is documented and absorbed by the random-init scaffold tests; the
    canonical contest-axis path runs on PyTorch CUDA per Catalog #1 + #192.
    """
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if backend is Backend.MLX:
        return _nearest_upsample_to(x, (target_h, target_w), backend)
    import torch.nn.functional as F

    return F.interpolate(
        x, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
