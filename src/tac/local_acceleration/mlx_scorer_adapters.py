# SPDX-License-Identifier: MIT
"""Parity-tested PyTorch-to-MLX scorer adapter primitives.

These helpers are intentionally small and local to scorer-port work.  They
convert individual upstream PyTorch layers into MLX layers and expose NCHW
wrapper functions so tests can compare against upstream PyTorch tensors before
larger FastViT/EfficientNet blocks are ported.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "mlx_reference_conv2d_nhwc",
    "nchw_to_nhwc",
    "nhwc_to_nchw",
    "run_mlx_batchnorm2d_nchw",
    "run_mlx_conv2d_nchw",
    "run_mlx_conv2d_relu_nchw",
    "run_mlx_distortion_scorer_nchw",
    "run_mlx_efficientnet_block_nchw",
    "run_mlx_efficientnet_features_nchw",
    "run_mlx_efficientnet_stage_nchw",
    "run_mlx_efficientnet_stem_nchw",
    "run_mlx_fastvit_stage_nchw",
    "run_mlx_fastvit_vision_nchw",
    "run_mlx_linear",
    "run_mlx_mobileone_block_nchw",
    "run_mlx_mobileone_stem_nchw",
    "run_mlx_patch_embed_nchw",
    "run_mlx_posenet_nchw",
    "run_mlx_reference_conv2d_nchw",
    "run_mlx_repmixer_block_nchw",
    "run_mlx_segmentation_head_nchw",
    "run_mlx_segnet_nchw",
    "run_mlx_timm_universal_encoder_nchw",
    "run_mlx_unet_decoder_block_nchw",
    "run_mlx_unet_decoder_nchw",
    "scorer_distortion_components_numpy",
    "temporary_mlx_device",
    "torch_batchnorm2d_to_mlx",
    "torch_conv2d_relu_to_mlx",
    "torch_conv2d_to_mlx",
    "torch_conv2d_to_mlx_reference",
    "torch_conv_mlp_to_mlx",
    "torch_distortion_net_to_mlx",
    "torch_efficientnet_block_to_mlx",
    "torch_efficientnet_features_to_mlx",
    "torch_efficientnet_stage_to_mlx",
    "torch_efficientnet_stem_to_mlx",
    "torch_fastvit_stage_to_mlx",
    "torch_fastvit_vision_to_mlx",
    "torch_linear_to_mlx",
    "torch_mobileone_block_to_mlx",
    "torch_mobileone_stem_to_mlx",
    "torch_patch_embed_to_mlx",
    "torch_posenet_to_mlx",
    "torch_repmixer_block_to_mlx",
    "torch_segmentation_head_to_mlx",
    "torch_segnet_to_mlx",
    "torch_timm_universal_encoder_to_mlx",
    "torch_unet_decoder_block_to_mlx",
    "torch_unet_decoder_to_mlx",
]

VALID_MLX_REFERENCE_CONV2D_ACCUMULATION_MODES = frozenset(
    {"fixed_fp32", "kahan_fp32", "fixed_fp64"}
)


class MLXConv2dReLUAdapter:
    """MLX adapter for SMP ``Conv2dReLU`` blocks."""

    def __init__(self, torch_block: Any):
        children = list(torch_block)
        if len(children) != 3:
            raise NotImplementedError(f"Conv2dReLU expected 3 children, got {len(children)}")
        if _class_path(children[0]) != "torch.nn.modules.conv.Conv2d":
            raise NotImplementedError(f"unsupported Conv2dReLU conv: {_class_path(children[0])}")
        if _class_path(children[1]) != "torch.nn.modules.batchnorm.BatchNorm2d":
            raise NotImplementedError(f"unsupported Conv2dReLU norm: {_class_path(children[1])}")
        if _class_path(children[2]) != "torch.nn.modules.activation.ReLU":
            raise NotImplementedError(f"unsupported Conv2dReLU activation: {_class_path(children[2])}")
        self.conv = torch_conv2d_to_mlx(children[0])
        self.bn = torch_batchnorm2d_to_mlx(children[1])

    def __call__(self, x_nhwc: Any) -> Any:
        return mlx_relu(self.bn(self.conv(x_nhwc)))


class MLXConvNormAct2dAdapter:
    """MLX adapter for timm ``ConvNormAct``."""

    def __init__(self, torch_module: Any):
        if getattr(torch_module, "aa", None) is not None:
            raise NotImplementedError("ConvNormAct anti-alias branch is not covered")
        conv = getattr(torch_module, "conv", None)
        bn = getattr(torch_module, "bn", None)
        if conv is None or bn is None:
            raise TypeError("ConvNormAct adapter requires .conv and .bn children")
        self.conv = torch_conv2d_to_mlx(conv)
        self.bn = torch_batchnorm2d_to_mlx(bn)
        self.activation_path = _class_path(getattr(bn, "act", None))

    def __call__(self, x_nhwc: Any) -> Any:
        return _apply_2d_activation(self.bn(self.conv(x_nhwc)), self.activation_path)


class MLXBatchNorm2dEvalAffineAdapter:
    """Eval-mode BatchNorm2d affine transform on NHWC tensors."""

    def __init__(self, torch_bn: Any):
        import mlx.core as mx

        if getattr(torch_bn, "training", False):
            raise NotImplementedError("training-mode BatchNorm2d is not covered")
        if not bool(getattr(torch_bn, "track_running_stats", False)):
            raise NotImplementedError("BatchNorm2d without running stats is not covered")
        running_mean = getattr(torch_bn, "running_mean", None)
        running_var = getattr(torch_bn, "running_var", None)
        if running_mean is None or running_var is None:
            raise TypeError("BatchNorm2d eval affine adapter requires running stats")

        channels = int(torch_bn.num_features)
        mean = _torch_tensor_to_numpy(running_mean).reshape(channels)
        var = _torch_tensor_to_numpy(running_var).reshape(channels)
        if bool(getattr(torch_bn, "affine", False)):
            weight = _torch_tensor_to_numpy(torch_bn.weight).reshape(channels)
            bias = _torch_tensor_to_numpy(torch_bn.bias).reshape(channels)
        else:
            weight = np.ones(channels, dtype=np.float32)
            bias = np.zeros(channels, dtype=np.float32)

        inv_std = np.float32(1.0) / np.sqrt(
            var.astype(np.float32, copy=False) + np.float32(float(torch_bn.eps)),
            dtype=np.float32,
        )
        scale = weight.astype(np.float32, copy=False) * inv_std
        offset = bias.astype(np.float32, copy=False) - mean.astype(np.float32, copy=False) * scale
        self.scale = mx.array(np.ascontiguousarray(scale.reshape(1, 1, 1, channels)))
        self.offset = mx.array(np.ascontiguousarray(offset.reshape(1, 1, 1, channels)))

    def __call__(self, x_nhwc: Any) -> Any:
        return x_nhwc * self.scale + self.offset


class MLXBatchNormAct2dAdapter:
    """MLX adapter for timm ``BatchNormAct2d``."""

    def __init__(self, torch_bn: Any):
        if _class_path(getattr(torch_bn, "drop", None)) != "torch.nn.modules.linear.Identity":
            raise NotImplementedError("BatchNormAct2d non-identity drop is not covered")
        self.bn = torch_batchnorm2d_to_mlx(torch_bn)
        self.activation_path = _class_path(getattr(torch_bn, "act", None))

    def __call__(self, x_nhwc: Any) -> Any:
        return _apply_2d_activation(self.bn(x_nhwc), self.activation_path)


class MLXSEModuleAdapter:
    """MLX adapter for timm ``SEModule`` on NHWC tensors."""

    def __init__(self, torch_se: Any):
        if bool(getattr(torch_se, "add_maxpool", False)):
            raise NotImplementedError("SEModule add_maxpool branch is not covered")
        if _class_path(getattr(torch_se, "bn", None)) != "torch.nn.modules.linear.Identity":
            raise NotImplementedError("SEModule non-identity BN is not covered")
        if _class_path(getattr(torch_se, "act", None)) != "torch.nn.modules.activation.ReLU":
            raise NotImplementedError(f"unsupported SEModule activation: {_class_path(torch_se.act)}")
        gate_path = _class_path(getattr(torch_se, "gate", None))
        if gate_path != "timm.layers.activations.Sigmoid":
            raise NotImplementedError(f"unsupported SEModule gate: {gate_path}")
        self.fc1 = MLXExplicitSpatialConv2dAdapter(torch_se.fc1)
        self.fc2 = MLXExplicitSpatialConv2dAdapter(torch_se.fc2)

    def __call__(self, x_nhwc: Any) -> Any:
        import mlx.core as mx

        x_se = mx.mean(mx.mean(x_nhwc, axis=2, keepdims=True), axis=1, keepdims=True)
        x_se = mlx_relu(self.fc1(x_se))
        x_se = self.fc2(x_se)
        return x_nhwc * mlx_sigmoid(x_se)


class MLXEfficientNetSqueezeExciteAdapter:
    """MLX adapter for timm EfficientNet ``SqueezeExcite``."""

    def __init__(self, torch_se: Any):
        gate_path = _class_path(getattr(torch_se, "gate", None))
        if gate_path != "torch.nn.modules.activation.Sigmoid":
            raise NotImplementedError(f"unsupported SqueezeExcite gate: {gate_path}")
        if _class_path(getattr(torch_se, "act1", None)) != "torch.nn.modules.activation.SiLU":
            raise NotImplementedError(f"unsupported SqueezeExcite act1: {_class_path(torch_se.act1)}")
        self.conv_reduce = torch_conv2d_to_mlx(torch_se.conv_reduce)
        self.conv_expand = torch_conv2d_to_mlx(torch_se.conv_expand)

    def __call__(self, x_nhwc: Any) -> Any:
        import mlx.core as mx

        x_se = mx.mean(mx.mean(x_nhwc, axis=2, keepdims=True), axis=1, keepdims=True)
        x_se = mlx_silu(self.conv_reduce(x_se))
        x_se = self.conv_expand(x_se)
        return x_nhwc * mlx_sigmoid(x_se)


class MLXDepthwiseSeparableConvAdapter:
    """MLX adapter for timm EfficientNet ``DepthwiseSeparableConv``."""

    def __init__(self, torch_block: Any):
        if getattr(torch_block, "conv_s2d", None) is not None:
            raise NotImplementedError("DepthwiseSeparableConv conv_s2d is not covered")
        if _class_path(getattr(torch_block, "aa", None)) != "torch.nn.modules.linear.Identity":
            raise NotImplementedError("DepthwiseSeparableConv anti-alias branch is not covered")
        if _class_path(getattr(torch_block, "drop_path", None)) != "torch.nn.modules.linear.Identity":
            raise NotImplementedError("DepthwiseSeparableConv non-identity DropPath is not covered")
        self.has_skip = bool(getattr(torch_block, "has_skip", False))
        self.conv_dw = torch_conv2d_to_mlx(torch_block.conv_dw)
        self.bn1 = MLXBatchNormAct2dAdapter(torch_block.bn1)
        self.se = MLXEfficientNetSqueezeExciteAdapter(torch_block.se)
        self.conv_pw = torch_conv2d_to_mlx(torch_block.conv_pw)
        self.bn2 = MLXBatchNormAct2dAdapter(torch_block.bn2)

    def __call__(self, x_nhwc: Any) -> Any:
        shortcut = x_nhwc
        out = self.bn1(self.conv_dw(x_nhwc))
        out = self.se(out)
        out = self.bn2(self.conv_pw(out))
        return out + shortcut if self.has_skip else out


class MLXInvertedResidualAdapter:
    """MLX adapter for timm EfficientNet ``InvertedResidual``."""

    def __init__(self, torch_block: Any):
        if getattr(torch_block, "conv_s2d", None) is not None:
            raise NotImplementedError("InvertedResidual conv_s2d is not covered")
        if _class_path(getattr(torch_block, "aa", None)) != "torch.nn.modules.linear.Identity":
            raise NotImplementedError("InvertedResidual anti-alias branch is not covered")
        if _class_path(getattr(torch_block, "drop_path", None)) != "torch.nn.modules.linear.Identity":
            raise NotImplementedError("InvertedResidual non-identity DropPath is not covered")
        self.has_skip = bool(getattr(torch_block, "has_skip", False))
        self.conv_pw = torch_conv2d_to_mlx(torch_block.conv_pw)
        self.bn1 = MLXBatchNormAct2dAdapter(torch_block.bn1)
        self.conv_dw = torch_conv2d_to_mlx(torch_block.conv_dw)
        self.bn2 = MLXBatchNormAct2dAdapter(torch_block.bn2)
        self.se = MLXEfficientNetSqueezeExciteAdapter(torch_block.se)
        self.conv_pwl = torch_conv2d_to_mlx(torch_block.conv_pwl)
        self.bn3 = MLXBatchNormAct2dAdapter(torch_block.bn3)

    def __call__(self, x_nhwc: Any) -> Any:
        shortcut = x_nhwc
        out = self.bn1(self.conv_pw(x_nhwc))
        out = self.bn2(self.conv_dw(out))
        out = self.se(out)
        out = self.bn3(self.conv_pwl(out))
        return out + shortcut if self.has_skip else out


class MLXEfficientNetStemAdapter:
    """MLX adapter for EfficientNet conv stem plus BatchNormAct2d."""

    def __init__(self, torch_efficientnet: Any):
        self.conv_stem = torch_conv2d_to_mlx(torch_efficientnet.conv_stem)
        self.bn1 = MLXBatchNormAct2dAdapter(torch_efficientnet.bn1)

    def __call__(self, x_nhwc: Any) -> Any:
        return self.bn1(self.conv_stem(x_nhwc))


class MLXEfficientNetStageAdapter:
    """Sequential MLX adapter for one EfficientNet block stage."""

    def __init__(self, torch_stage: Any):
        self.blocks = [torch_efficientnet_block_to_mlx(block) for block in torch_stage]

    def __call__(self, x_nhwc: Any) -> Any:
        out = x_nhwc
        for block in self.blocks:
            out = block(out)
        return out


class MLXEfficientNetFeaturesAdapter:
    """MLX adapter for timm ``EfficientNetFeatures`` feature-list output."""

    def __init__(self, torch_model: Any):
        if getattr(torch_model, "feature_hooks", None) is not None:
            raise NotImplementedError("EfficientNet feature_hooks path is not covered")
        self.stage_out_idx = set(getattr(torch_model, "_stage_out_idx", {}))
        self.stem = torch_efficientnet_stem_to_mlx(torch_model)
        self.stages = [torch_efficientnet_stage_to_mlx(stage) for stage in torch_model.blocks]

    def __call__(self, x_nhwc: Any) -> list[Any]:
        out = self.stem(x_nhwc)
        features = []
        if 0 in self.stage_out_idx:
            features.append(out)
        for index, stage in enumerate(self.stages):
            out = stage(out)
            if index + 1 in self.stage_out_idx:
                features.append(out)
        return features


class MLXTimmUniversalEncoderAdapter:
    """MLX adapter for SMP ``TimmUniversalEncoder`` around EfficientNetFeatures."""

    def __init__(self, torch_encoder: Any):
        if bool(getattr(torch_encoder, "_is_channel_last", False)):
            raise NotImplementedError("channel-last TimmUniversalEncoder path is not covered")
        if bool(getattr(torch_encoder, "_is_transformer_style", False)):
            raise NotImplementedError("transformer-style TimmUniversalEncoder path is not covered")
        self.prepend_input = not bool(getattr(torch_encoder, "_is_vgg_style", False))
        self.model = torch_efficientnet_features_to_mlx(torch_encoder.model)

    def __call__(self, x_nhwc: Any) -> list[Any]:
        features = self.model(x_nhwc)
        return [x_nhwc, *features] if self.prepend_input else features


class MLXUnetDecoderBlockAdapter:
    """MLX adapter for SMP ``UnetDecoderBlock``."""

    def __init__(self, torch_block: Any):
        interpolation_mode = getattr(torch_block, "interpolation_mode", None)
        if interpolation_mode != "nearest":
            raise NotImplementedError(f"unsupported UnetDecoderBlock interpolation: {interpolation_mode!r}")
        if _class_path(getattr(torch_block.attention1, "attention", None)) != (
            "torch.nn.modules.linear.Identity"
        ):
            raise NotImplementedError("UnetDecoderBlock attention1 is not identity")
        if _class_path(getattr(torch_block.attention2, "attention", None)) != (
            "torch.nn.modules.linear.Identity"
        ):
            raise NotImplementedError("UnetDecoderBlock attention2 is not identity")
        self.conv1 = torch_conv2d_relu_to_mlx(torch_block.conv1)
        self.conv2 = torch_conv2d_relu_to_mlx(torch_block.conv2)

    def __call__(
        self,
        feature_map_nhwc: Any,
        target_height: int,
        target_width: int,
        skip_connection_nhwc: Any | None = None,
    ) -> Any:
        import mlx.core as mx

        out = mlx_interpolate_nearest_size(feature_map_nhwc, target_height, target_width)
        if skip_connection_nhwc is not None:
            out = mx.concatenate([out, skip_connection_nhwc], axis=-1)
        out = self.conv1(out)
        return self.conv2(out)


class MLXUnetDecoderAdapter:
    """MLX adapter for SMP ``UnetDecoder`` feature-list contract."""

    def __init__(self, torch_decoder: Any):
        if _class_path(getattr(torch_decoder, "center", None)) != "torch.nn.modules.linear.Identity":
            raise NotImplementedError("UnetDecoder non-identity center is not covered")
        self.blocks = [torch_unet_decoder_block_to_mlx(block) for block in torch_decoder.blocks]

    def __call__(self, features_nhwc: list[Any]) -> Any:
        spatial_shapes = [(int(feature.shape[1]), int(feature.shape[2])) for feature in features_nhwc]
        spatial_shapes = spatial_shapes[::-1]
        features = features_nhwc[1:]
        features = features[::-1]
        out = features[0]
        skip_connections = features[1:]
        for index, decoder_block in enumerate(self.blocks):
            target_height, target_width = spatial_shapes[index + 1]
            skip_connection = (
                skip_connections[index] if index < len(skip_connections) else None
            )
            out = decoder_block(out, target_height, target_width, skip_connection)
        return out


class MLXSegmentationHeadAdapter:
    """MLX adapter for SMP ``SegmentationHead`` without activation."""

    def __init__(self, torch_head: Any):
        children = list(torch_head)
        if len(children) != 3:
            raise NotImplementedError(f"SegmentationHead expected 3 children, got {len(children)}")
        if _class_path(children[0]) != "torch.nn.modules.conv.Conv2d":
            raise NotImplementedError(f"unsupported SegmentationHead conv: {_class_path(children[0])}")
        if _class_path(children[1]) != "torch.nn.modules.linear.Identity":
            raise NotImplementedError("SegmentationHead upsampling is not identity")
        activation = getattr(children[2], "activation", None)
        if _class_path(activation) != "torch.nn.modules.linear.Identity":
            raise NotImplementedError("SegmentationHead activation is not identity")
        self.conv = MLXExplicitSpatialConv2dAdapter(children[0])

    def __call__(self, x_nhwc: Any) -> Any:
        return self.conv(x_nhwc)


class MLXExplicitSpatialConv2dAdapter:
    """Small explicit Conv2d for numerically sensitive SegNet head logits."""

    def __init__(self, torch_conv: Any):
        import mlx.core as mx

        if int(torch_conv.groups) != 1:
            raise NotImplementedError("explicit spatial Conv2d requires groups=1")
        if _pair(torch_conv.stride) != (1, 1):
            raise NotImplementedError("explicit spatial Conv2d requires stride=1")
        if _pair(torch_conv.dilation) != (1, 1):
            raise NotImplementedError("explicit spatial Conv2d requires dilation=1")
        weight = _torch_tensor_to_numpy(torch_conv.weight)
        if weight.ndim != 4:
            raise ValueError(f"Conv2d weight must be rank-4 OIHW, got {weight.shape}")
        self.padding = _pair(torch_conv.padding)
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        self.out_channels = int(out_channels)
        self.terms = [
            (
                int(kh),
                int(kw),
                int(channel),
                mx.array(np.ascontiguousarray(weight[:, channel, kh, kw].reshape(1, 1, 1, -1))),
            )
            for kh in range(kernel_h)
            for kw in range(kernel_w)
            for channel in range(in_channels)
        ]
        self.bias = (
            None
            if torch_conv.bias is None
            else mx.array(
                np.ascontiguousarray(
                    _torch_tensor_to_numpy(torch_conv.bias).reshape(1, 1, 1, self.out_channels)
                )
            )
        )

    def __call__(self, x_nhwc: Any) -> Any:
        import mlx.core as mx

        pad_h, pad_w = self.padding
        x_pad = mx.pad(x_nhwc, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
        batch, height, width, _ = x_nhwc.shape
        out = mx.zeros((batch, height, width, self.out_channels), dtype=x_nhwc.dtype)
        for kh, kw, channel, weight in self.terms:
            out = out + x_pad[:, kh : kh + height, kw : kw + width, channel : channel + 1] * weight
        return out if self.bias is None else out + self.bias


def mlx_reference_conv2d_nhwc(
    x_nhwc: Any,
    weight_ohwi: Any,
    bias: Any | None = None,
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    accumulation_mode: str = "fixed_fp32",
) -> Any:
    """Fixed-order NHWC Conv2d reference used by MLX drift probes.

    This is the shared, explicit accumulation implementation for parity and
    mitigation work. Fast production paths should keep using native
    ``mx.conv2d`` unless a measured probe proves this slower fixed-order path is
    worth the throughput cost.
    """

    import mlx.core as mx

    if accumulation_mode not in VALID_MLX_REFERENCE_CONV2D_ACCUMULATION_MODES:
        raise ValueError(
            "accumulation_mode must be one of "
            f"{sorted(VALID_MLX_REFERENCE_CONV2D_ACCUMULATION_MODES)}, got "
            f"{accumulation_mode!r}"
        )
    if int(groups) < 1:
        raise ValueError(f"groups must be >= 1, got {groups}")
    if len(x_nhwc.shape) != 4:
        raise ValueError(f"x_nhwc must be NHWC rank-4, got {x_nhwc.shape}")
    if len(weight_ohwi.shape) != 4:
        raise ValueError(
            f"weight_ohwi must have shape (O,kH,kW,I/group), got {weight_ohwi.shape}"
        )

    original_dtype = x_nhwc.dtype
    accum_dtype = mx.float64 if accumulation_mode == "fixed_fp64" else mx.float32
    try:
        x_work = x_nhwc.astype(accum_dtype)
        weight_work = weight_ohwi.astype(accum_dtype)
        bias_work = None if bias is None else bias.astype(accum_dtype)
    except ValueError as exc:
        raise ValueError(
            f"{accumulation_mode} Conv2d accumulation is unsupported on the "
            "current MLX device; use fixed_fp32/kahan_fp32 on Metal or fixed_fp64 "
            "on MLX CPU"
        ) from exc
    pad_h, pad_w = _pair(padding)
    stride_h, stride_w = _pair(stride)
    dilation_h, dilation_w = _pair(dilation)
    groups_i = int(groups)
    batch, height_in, width_in, channels_in = x_nhwc.shape
    out_channels, kernel_h, kernel_w, in_channels_per_group = weight_ohwi.shape
    if int(channels_in) != groups_i * int(in_channels_per_group):
        raise ValueError(
            f"input channels {channels_in} do not match grouped weight channels "
            f"{groups_i * int(in_channels_per_group)}"
        )
    if int(out_channels) % groups_i != 0:
        raise ValueError(f"out_channels {out_channels} not divisible by groups {groups_i}")
    if bias_work is not None:
        if int(bias_work.size) != int(out_channels):
            raise ValueError(
                f"bias must contain {out_channels} values, got {bias_work.shape}"
            )
        bias_work = mx.reshape(bias_work, (int(out_channels),))

    height_out = (
        int(height_in) + 2 * pad_h - dilation_h * (int(kernel_h) - 1) - 1
    ) // stride_h + 1
    width_out = (
        int(width_in) + 2 * pad_w - dilation_w * (int(kernel_w) - 1) - 1
    ) // stride_w + 1
    if height_out < 1 or width_out < 1:
        raise ValueError(
            f"invalid Conv2d output shape ({height_out}, {width_out}) for input "
            f"{(height_in, width_in)}"
        )

    x_pad = mx.pad(x_work, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    out_channels_per_group = int(out_channels) // groups_i
    group_outputs = []
    for group_index in range(groups_i):
        group_out = mx.zeros(
            (batch, height_out, width_out, out_channels_per_group),
            dtype=accum_dtype,
        )
        compensation = (
            mx.zeros_like(group_out) if accumulation_mode == "kahan_fp32" else None
        )
        for kh in range(int(kernel_h)):
            h0 = kh * dilation_h
            h1 = h0 + stride_h * height_out
            for kw in range(int(kernel_w)):
                w0 = kw * dilation_w
                w1 = w0 + stride_w * width_out
                for local_channel in range(int(in_channels_per_group)):
                    channel = group_index * int(in_channels_per_group) + local_channel
                    weight = weight_work[
                        group_index
                        * out_channels_per_group : (group_index + 1)
                        * out_channels_per_group,
                        kh,
                        kw,
                        local_channel,
                    ]
                    term = (
                        x_pad[
                            :,
                            h0:h1:stride_h,
                            w0:w1:stride_w,
                            channel : channel + 1,
                        ]
                        * mx.reshape(weight, (1, 1, 1, out_channels_per_group))
                    )
                    if compensation is None:
                        group_out = group_out + term
                    else:
                        y = term - compensation
                        t = group_out + y
                        compensation = (t - group_out) - y
                        group_out = t
        group_outputs.append(group_out)

    out = (
        group_outputs[0]
        if len(group_outputs) == 1
        else mx.concatenate(group_outputs, axis=-1)
    )
    if bias_work is not None:
        out = out + mx.reshape(bias_work, (1, 1, 1, int(out_channels)))
    return out.astype(original_dtype)


class MLXReferenceConv2dAdapter:
    """Fixed-order Conv2d reference path for MLX numerical drift probes.

    This is intentionally not the fast scorer path. It exists to make MLX-side
    accumulation order explicit when debugging PyTorch/MLX drift. It supports
    grouped Conv2d, stride, padding, dilation, optional Kahan compensation, and
    optional fp64 intermediate accumulation.
    """

    def __init__(
        self,
        torch_conv: Any,
        *,
        accumulation_mode: str = "fixed_fp32",
    ):
        import mlx.core as mx

        if accumulation_mode not in VALID_MLX_REFERENCE_CONV2D_ACCUMULATION_MODES:
            raise ValueError(
                "accumulation_mode must be one of fixed_fp32/kahan_fp32/fixed_fp64, "
                f"got {accumulation_mode!r}"
            )
        self.accumulation_mode = accumulation_mode
        self.stride = _pair(torch_conv.stride)
        self.padding = _pair(torch_conv.padding)
        self.dilation = _pair(torch_conv.dilation)
        self.groups = int(torch_conv.groups)
        weight = _torch_tensor_to_numpy(torch_conv.weight)
        if weight.ndim != 4:
            raise ValueError(f"Conv2d weight must be rank-4 OIHW, got {weight.shape}")
        out_channels, in_channels_per_group, kernel_h, kernel_w = weight.shape
        if out_channels % self.groups != 0:
            raise ValueError(
                f"out_channels {out_channels} not divisible by groups {self.groups}"
            )
        self.out_channels = int(out_channels)
        self.in_channels_per_group = int(in_channels_per_group)
        self.kernel_size = (int(kernel_h), int(kernel_w))
        self.out_channels_per_group = int(out_channels // self.groups)
        dtype = mx.float64 if accumulation_mode == "fixed_fp64" else mx.float32
        self.weight = mx.array(
            np.ascontiguousarray(weight.transpose(0, 2, 3, 1)),
            dtype=dtype,
        )
        self.bias = (
            None
            if torch_conv.bias is None
            else mx.array(
                np.ascontiguousarray(
                    _torch_tensor_to_numpy(torch_conv.bias).reshape(
                        1, 1, 1, self.out_channels
                    )
                ),
                dtype=dtype,
            )
        )

    def __call__(self, x_nhwc: Any) -> Any:
        return mlx_reference_conv2d_nhwc(
            x_nhwc,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            accumulation_mode=self.accumulation_mode,
        )


class MLXSegNetAdapter:
    """MLX adapter for upstream ``SegNet`` through raw segmentation logits."""

    def __init__(self, torch_segnet: Any):
        self.encoder = torch_timm_universal_encoder_to_mlx(torch_segnet.encoder)
        self.decoder = torch_unet_decoder_to_mlx(torch_segnet.decoder)
        self.segmentation_head = torch_segmentation_head_to_mlx(torch_segnet.segmentation_head)

    def __call__(self, x_nhwc: Any) -> Any:
        features = self.encoder(x_nhwc)
        decoder_output = self.decoder(features)
        return self.segmentation_head(decoder_output)


class MLXDistortionScorerAdapter:
    """MLX adapter for upstream ``DistortionNet`` on fixed scorer-input tensors."""

    def __init__(self, torch_distortion_net: Any):
        self.posenet = torch_posenet_to_mlx(torch_distortion_net.posenet)
        self.segnet = torch_segnet_to_mlx(torch_distortion_net.segnet)

    def __call__(self, posenet_yuv6_pair_nhwc: Any, segnet_last_rgb_nhwc: Any) -> dict[str, Any]:
        return {
            "posenet": self.posenet(posenet_yuv6_pair_nhwc),
            "segnet": self.segnet(segnet_last_rgb_nhwc),
        }


class MLXMobileOneBlockAdapter:
    """MLX adapter for upstream PoseNet stem ``MobileOneBlock`` in eval mode."""

    def __init__(self, torch_block: Any):
        if getattr(torch_block, "reparam_conv", None) is not None:
            raise NotImplementedError("reparameterized MobileOneBlock is not covered yet")
        se = getattr(torch_block, "se", None)
        se_path = _class_path(se)
        if se_path == "torch.nn.modules.linear.Identity":
            self.se = None
        elif se_path == "timm.layers.squeeze_excite.SEModule":
            self.se = MLXSEModuleAdapter(se)
        elif se is None:
            self.se = None
        else:
            raise NotImplementedError(f"unsupported MobileOne SE branch: {se_path}")
        self.identity = (
            torch_batchnorm2d_to_mlx(torch_block.identity)
            if getattr(torch_block, "identity", None) is not None
            else None
        )
        self.conv_scale = (
            MLXConvNormAct2dAdapter(torch_block.conv_scale)
            if getattr(torch_block, "conv_scale", None) is not None
            else None
        )
        self.conv_kxk = [
            MLXConvNormAct2dAdapter(branch)
            for branch in (getattr(torch_block, "conv_kxk", None) or [])
        ]
        if (
            self.identity is None
            and self.conv_scale is None
            and not self.conv_kxk
        ):
            raise NotImplementedError(
                "unsupported MobileOneBlock with no active identity, scale, or kxk branches"
            )
        self.use_gelu_tanh = _class_path(getattr(torch_block, "act", None)) == (
            "timm.layers.activations.GELUTanh"
        )
        if not self.use_gelu_tanh and _class_path(getattr(torch_block, "act", None)) != (
            "torch.nn.modules.linear.Identity"
        ):
            raise NotImplementedError(f"unsupported MobileOne activation: {_class_path(torch_block.act)}")

    def __call__(self, x_nhwc: Any) -> Any:
        out = None
        if self.identity is not None:
            out = self.identity(x_nhwc)
        if self.conv_scale is not None:
            branch = self.conv_scale(x_nhwc)
            out = branch if out is None else out + branch
        for branch_adapter in self.conv_kxk:
            branch = branch_adapter(x_nhwc)
            out = branch if out is None else out + branch
        if out is None:
            raise RuntimeError("MobileOneBlock adapter has no active branches")
        if self.se is not None:
            out = self.se(out)
        return mlx_gelu_tanh(out) if self.use_gelu_tanh else out


class MLXMobileOneStemAdapter:
    """Sequential adapter for the upstream PoseNet FastViT stem."""

    def __init__(self, torch_stem: Any):
        self.blocks = [torch_mobileone_block_to_mlx(block) for block in torch_stem]

    def __call__(self, x_nhwc: Any) -> Any:
        out = x_nhwc
        for block in self.blocks:
            out = block(out)
        return out


class MLXLayerScale2dAdapter:
    """MLX adapter for timm ``LayerScale2d`` on NHWC tensors."""

    def __init__(self, torch_layer_scale: Any):
        import mlx.core as mx

        gamma = getattr(torch_layer_scale, "gamma", None)
        if gamma is None:
            raise TypeError("LayerScale2d adapter requires .gamma")
        gamma_np = _torch_tensor_to_numpy(gamma).reshape(-1)
        self.gamma = mx.array(np.ascontiguousarray(gamma_np.reshape(1, 1, 1, -1)))

    def __call__(self, x_nhwc: Any) -> Any:
        return x_nhwc * self.gamma


class MLXConvMlpAdapter:
    """MLX adapter for eval-mode FastViT ``ConvMlp``."""

    def __init__(self, torch_mlp: Any):
        drop = getattr(torch_mlp, "drop", None)
        if drop is not None and getattr(drop, "training", False):
            raise NotImplementedError("training-mode ConvMlp dropout is not covered")
        if _class_path(getattr(torch_mlp, "act", None)) != "timm.layers.activations.GELUTanh":
            raise NotImplementedError(f"unsupported ConvMlp activation: {_class_path(torch_mlp.act)}")
        self.conv = MLXConvNormAct2dAdapter(torch_mlp.conv)
        self.fc1 = torch_conv2d_to_mlx(torch_mlp.fc1)
        self.fc2 = torch_conv2d_to_mlx(torch_mlp.fc2)

    def __call__(self, x_nhwc: Any) -> Any:
        out = self.conv(x_nhwc)
        out = self.fc1(out)
        out = mlx_gelu_tanh(out)
        return self.fc2(out)


class MLXRepMixerAdapter:
    """MLX adapter for eval-mode FastViT ``RepMixer``."""

    def __init__(self, torch_repmixer: Any):
        if getattr(torch_repmixer, "reparam_conv", None) is not None:
            raise NotImplementedError("reparameterized RepMixer is not covered yet")
        self.norm = torch_mobileone_block_to_mlx(torch_repmixer.norm)
        self.mixer = torch_mobileone_block_to_mlx(torch_repmixer.mixer)
        self.layer_scale = MLXLayerScale2dAdapter(torch_repmixer.layer_scale)

    def __call__(self, x_nhwc: Any) -> Any:
        return x_nhwc + self.layer_scale(self.mixer(x_nhwc) - self.norm(x_nhwc))


class MLXRepMixerBlockAdapter:
    """MLX adapter for the first PoseNet FastViT ``RepMixerBlock`` family."""

    def __init__(self, torch_block: Any):
        if _class_path(getattr(torch_block, "drop_path", None)) != (
            "torch.nn.modules.linear.Identity"
        ):
            raise NotImplementedError("non-identity DropPath is not covered")
        self.token_mixer = MLXRepMixerAdapter(torch_block.token_mixer)
        self.mlp = torch_conv_mlp_to_mlx(torch_block.mlp)
        self.layer_scale = MLXLayerScale2dAdapter(torch_block.layer_scale)

    def __call__(self, x_nhwc: Any) -> Any:
        out = self.token_mixer(x_nhwc)
        return out + self.layer_scale(self.mlp(out))


class MLXReparamLargeKernelConvAdapter:
    """MLX adapter for FastViT ``ReparamLargeKernelConv`` before reparam folding."""

    def __init__(self, torch_module: Any):
        if getattr(torch_module, "reparam_conv", None) is not None:
            raise NotImplementedError("reparameterized large-kernel conv is not covered yet")
        if _class_path(getattr(torch_module, "se", None)) != "torch.nn.modules.linear.Identity":
            raise NotImplementedError("ReparamLargeKernelConv SE branch is not covered")
        act_path = _class_path(getattr(torch_module, "act", None))
        if act_path not in {
            "torch.nn.modules.linear.Identity",
            "timm.layers.activations.GELUTanh",
        }:
            raise NotImplementedError(f"unsupported ReparamLargeKernelConv activation: {act_path}")
        self.large_conv = MLXConvNormAct2dAdapter(torch_module.large_conv)
        self.small_conv = (
            MLXConvNormAct2dAdapter(torch_module.small_conv)
            if getattr(torch_module, "small_conv", None) is not None
            else None
        )
        self.use_gelu_tanh = act_path == "timm.layers.activations.GELUTanh"

    def __call__(self, x_nhwc: Any) -> Any:
        out = self.large_conv(x_nhwc)
        if self.small_conv is not None:
            out = out + self.small_conv(x_nhwc)
        return mlx_gelu_tanh(out) if self.use_gelu_tanh else out


class MLXPatchEmbedAdapter:
    """Sequential MLX adapter for FastViT ``PatchEmbed``."""

    def __init__(self, torch_patch_embed: Any):
        self.proj = [_torch_fastvit_child_to_mlx(child) for child in torch_patch_embed.proj]

    def __call__(self, x_nhwc: Any) -> Any:
        out = x_nhwc
        for adapter in self.proj:
            out = adapter(out)
        return out


class MLXFastVitStageAdapter:
    """MLX adapter for eval-mode FastViT stages used by upstream PoseNet."""

    def __init__(self, torch_stage: Any):
        downsample_path = _class_path(getattr(torch_stage, "downsample", None))
        if downsample_path == "torch.nn.modules.linear.Identity":
            self.downsample = None
        elif downsample_path == "timm.models.fastvit.PatchEmbed":
            self.downsample = torch_patch_embed_to_mlx(torch_stage.downsample)
        else:
            raise NotImplementedError(f"unsupported FastVitStage downsample: {downsample_path}")
        if _class_path(getattr(torch_stage, "pos_emb", None)) != "torch.nn.modules.linear.Identity":
            raise NotImplementedError("non-identity FastVitStage positional embedding is not covered")
        self.blocks = [torch_repmixer_block_to_mlx(block) for block in torch_stage.blocks]

    def __call__(self, x_nhwc: Any) -> Any:
        out = x_nhwc if self.downsample is None else self.downsample(x_nhwc)
        for block in self.blocks:
            out = block(out)
        return out


class MLXClassifierHeadAdapter:
    """MLX adapter for FastViT ``ClassifierHead`` with average pooling."""

    def __init__(self, torch_head: Any):
        global_pool = getattr(torch_head, "global_pool", None)
        pool_type = getattr(global_pool, "pool_type", None)
        if pool_type != "avg":
            raise NotImplementedError(f"unsupported classifier pool type: {pool_type!r}")
        drop = getattr(torch_head, "drop", None)
        if drop is not None and getattr(drop, "training", False):
            raise NotImplementedError("training-mode classifier dropout is not covered")
        if _class_path(getattr(torch_head, "flatten", None)) != "torch.nn.modules.linear.Identity":
            raise NotImplementedError("non-identity classifier flatten is not covered")
        self.fc = torch_linear_to_mlx(torch_head.fc)

    def __call__(self, x_nhwc: Any) -> Any:
        import mlx.core as mx

        pooled = mx.mean(x_nhwc, axis=(1, 2))
        return self.fc(pooled)


class MLXFastVitVisionAdapter:
    """MLX adapter for upstream PoseNet's FastViT vision trunk."""

    def __init__(self, torch_vision: Any):
        if bool(getattr(torch_vision, "fork_feat", False)):
            raise NotImplementedError("FastViT fork_feat path is not covered")
        self.stem = torch_mobileone_stem_to_mlx(torch_vision.stem)
        self.stages = [torch_fastvit_stage_to_mlx(stage) for stage in torch_vision.stages]
        self.final_conv = torch_mobileone_block_to_mlx(torch_vision.final_conv)
        self.head = MLXClassifierHeadAdapter(torch_vision.head)

    def __call__(self, x_nhwc: Any) -> Any:
        out = self.stem(x_nhwc)
        for stage in self.stages:
            out = stage(out)
        out = self.final_conv(out)
        return self.head(out)


class MLXAllNormAdapter:
    """MLX adapter for upstream ``AllNorm`` eval-mode BatchNorm1d scalar affine."""

    def __init__(self, torch_allnorm: Any):
        import mlx.core as mx

        bn = getattr(torch_allnorm, "bn", None)
        if bn is None:
            raise TypeError("AllNorm adapter requires .bn")
        if getattr(bn, "training", False):
            raise NotImplementedError("training-mode AllNorm is not covered")
        weight = _torch_tensor_to_numpy(bn.weight).reshape(())
        bias = _torch_tensor_to_numpy(bn.bias).reshape(())
        running_mean = _torch_tensor_to_numpy(bn.running_mean).reshape(())
        running_var = _torch_tensor_to_numpy(bn.running_var).reshape(())
        scale = weight / np.sqrt(running_var + float(bn.eps))
        offset = bias - running_mean * scale
        self.scale = mx.array(np.float32(scale))
        self.offset = mx.array(np.float32(offset))

    def __call__(self, x: Any) -> Any:
        return x * self.scale + self.offset


class MLXSequential1dAdapter:
    """Small eval-mode adapter for upstream dense ``nn.Sequential`` heads."""

    def __init__(self, torch_seq: Any):
        self.layers = [_torch_dense_child_to_mlx(child) for child in torch_seq]

    def __call__(self, x: Any) -> Any:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class MLXResBlockAdapter:
    """MLX adapter for upstream dense ``ResBlock``."""

    def __init__(self, torch_resblock: Any):
        self.block_a = MLXSequential1dAdapter(torch_resblock.block_a)
        block_b_layers = list(torch_resblock.block_b)
        self.block_b_starts_inplace_relu = (
            bool(block_b_layers)
            and _class_path(block_b_layers[0]) == "torch.nn.modules.activation.ReLU"
            and bool(getattr(block_b_layers[0], "inplace", False))
        )
        if self.block_b_starts_inplace_relu:
            block_b_layers = block_b_layers[1:]
        self.block_b = MLXSequential1dAdapter(block_b_layers)

    def __call__(self, x: Any) -> Any:
        a_out = x + self.block_a(x)
        if self.block_b_starts_inplace_relu:
            block_b_input = mlx_relu(a_out)
            return mlx_relu(block_b_input + self.block_b(block_b_input))
        return mlx_relu(a_out + self.block_b(a_out))


class MLXSummarizerAdapter:
    """MLX adapter for PoseNet ``summarizer``."""

    def __init__(self, torch_summarizer: Any):
        self.layers = [_torch_dense_child_to_mlx(child) for child in torch_summarizer]

    def __call__(self, x: Any) -> Any:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class MLXHydraAdapter:
    """MLX adapter for upstream PoseNet single-head ``Hydra``."""

    def __init__(self, torch_hydra: Any):
        self.resblock = MLXResBlockAdapter(torch_hydra.resblock)
        self.in_layer = {
            name: torch_linear_to_mlx(layer)
            for name, layer in torch_hydra.in_layer.items()
        }
        self.res_layer = {
            name: MLXSequential1dAdapter(layer)
            for name, layer in torch_hydra.res_layer.items()
        }
        self.final_layer = {
            name: torch_linear_to_mlx(layer)
            for name, layer in torch_hydra.final_layer.items()
        }

    def __call__(self, x: Any) -> dict[str, Any]:
        out = self.resblock(x)
        in_layer = {
            name: mlx_relu(layer(out))
            for name, layer in self.in_layer.items()
        }
        res_layer = {
            name: mlx_relu(in_layer[name] + self.res_layer[name](in_layer[name]))
            for name in in_layer
        }
        return {
            name: self.final_layer[name](res_layer[name])
            for name in res_layer
        }


class MLXPoseNetAdapter:
    """MLX adapter for upstream PoseNet through the ``pose`` output head."""

    def __init__(self, torch_posenet: Any):
        import mlx.core as mx

        mean = _torch_tensor_to_numpy(torch_posenet._mean).reshape(12)
        std = _torch_tensor_to_numpy(torch_posenet._std).reshape(12)
        self.mean = mx.array(np.ascontiguousarray(mean.reshape(1, 1, 1, 12)))
        self.std = mx.array(np.ascontiguousarray(std.reshape(1, 1, 1, 12)))
        self.vision = torch_fastvit_vision_to_mlx(torch_posenet.vision)
        self.summarizer = MLXSummarizerAdapter(torch_posenet.summarizer)
        self.hydra = MLXHydraAdapter(torch_posenet.hydra)

    def __call__(self, x_nhwc: Any) -> dict[str, Any]:
        vision_out = self.vision((x_nhwc - self.mean) / self.std)
        summary = self.summarizer(vision_out)
        return self.hydra(summary)


def torch_conv2d_to_mlx(torch_conv: Any) -> Any:
    """Convert a PyTorch ``nn.Conv2d`` layer to MLX ``nn.Conv2d``."""

    import mlx.core as mx
    import mlx.nn as nn

    conv = nn.Conv2d(
        int(torch_conv.in_channels),
        int(torch_conv.out_channels),
        _pair(torch_conv.kernel_size),
        stride=_pair(torch_conv.stride),
        padding=_pair(torch_conv.padding),
        dilation=_pair(torch_conv.dilation),
        groups=int(torch_conv.groups),
        bias=torch_conv.bias is not None,
    )
    weight = _torch_tensor_to_numpy(torch_conv.weight)
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be rank-4 OIHW, got {weight.shape}")
    params: dict[str, Any] = {
        "weight": mx.array(np.ascontiguousarray(weight.transpose(0, 2, 3, 1))),
    }
    if torch_conv.bias is not None:
        params["bias"] = mx.array(_torch_tensor_to_numpy(torch_conv.bias))
    conv.update(params)
    return conv


def torch_conv2d_to_mlx_reference(
    torch_conv: Any,
    *,
    accumulation_mode: str = "fixed_fp32",
) -> MLXReferenceConv2dAdapter:
    """Convert PyTorch ``nn.Conv2d`` to a fixed-order MLX reference adapter."""

    return MLXReferenceConv2dAdapter(
        torch_conv,
        accumulation_mode=accumulation_mode,
    )


def torch_batchnorm2d_to_mlx(torch_bn: Any) -> Any:
    """Convert PyTorch ``nn.BatchNorm2d`` to an MLX-compatible adapter."""

    import mlx.core as mx
    import mlx.nn as nn

    if (
        not bool(getattr(torch_bn, "training", False))
        and bool(getattr(torch_bn, "track_running_stats", False))
        and getattr(torch_bn, "running_mean", None) is not None
        and getattr(torch_bn, "running_var", None) is not None
    ):
        return MLXBatchNorm2dEvalAffineAdapter(torch_bn)

    bn = nn.BatchNorm(
        int(torch_bn.num_features),
        eps=float(torch_bn.eps),
        momentum=float(torch_bn.momentum),
        affine=bool(torch_bn.affine),
        track_running_stats=bool(torch_bn.track_running_stats),
    )
    params: dict[str, Any] = {}
    for name in ("weight", "bias", "running_mean", "running_var"):
        value = getattr(torch_bn, name, None)
        if value is not None:
            params[name] = mx.array(_torch_tensor_to_numpy(value))
    bn.update(params, strict=False)
    if not torch_bn.training:
        bn.eval()
    return bn


def torch_conv2d_relu_to_mlx(torch_block: Any) -> MLXConv2dReLUAdapter:
    """Convert an SMP ``Conv2dReLU`` block to MLX."""

    return MLXConv2dReLUAdapter(torch_block)


def torch_linear_to_mlx(torch_linear: Any) -> Any:
    """Convert PyTorch ``nn.Linear`` to MLX ``nn.Linear``."""

    import mlx.core as mx
    import mlx.nn as nn

    linear = nn.Linear(
        int(torch_linear.in_features),
        int(torch_linear.out_features),
        bias=torch_linear.bias is not None,
    )
    params: dict[str, Any] = {"weight": mx.array(_torch_tensor_to_numpy(torch_linear.weight))}
    if torch_linear.bias is not None:
        params["bias"] = mx.array(_torch_tensor_to_numpy(torch_linear.bias))
    linear.update(params)
    return linear


def torch_mobileone_block_to_mlx(torch_block: Any) -> MLXMobileOneBlockAdapter:
    """Convert a timm FastViT ``MobileOneBlock`` to a parity-tested MLX adapter."""

    return MLXMobileOneBlockAdapter(torch_block)


def torch_mobileone_stem_to_mlx(torch_stem: Any) -> MLXMobileOneStemAdapter:
    """Convert the upstream PoseNet FastViT stem to a sequential MLX adapter."""

    return MLXMobileOneStemAdapter(torch_stem)


def torch_conv_mlp_to_mlx(torch_mlp: Any) -> MLXConvMlpAdapter:
    """Convert a timm FastViT ``ConvMlp`` to a parity-tested MLX adapter."""

    return MLXConvMlpAdapter(torch_mlp)


def torch_efficientnet_block_to_mlx(torch_block: Any) -> Any:
    """Convert a timm EfficientNet block to a parity-tested MLX adapter."""

    class_path = _class_path(torch_block)
    if class_path == "timm.models._efficientnet_blocks.DepthwiseSeparableConv":
        return MLXDepthwiseSeparableConvAdapter(torch_block)
    if class_path == "timm.models._efficientnet_blocks.InvertedResidual":
        return MLXInvertedResidualAdapter(torch_block)
    raise NotImplementedError(f"unsupported EfficientNet block: {class_path}")


def torch_efficientnet_stem_to_mlx(torch_efficientnet: Any) -> MLXEfficientNetStemAdapter:
    """Convert an EfficientNet feature model stem to MLX."""

    return MLXEfficientNetStemAdapter(torch_efficientnet)


def torch_efficientnet_stage_to_mlx(torch_stage: Any) -> MLXEfficientNetStageAdapter:
    """Convert an EfficientNet block stage to MLX."""

    return MLXEfficientNetStageAdapter(torch_stage)


def torch_efficientnet_features_to_mlx(torch_model: Any) -> MLXEfficientNetFeaturesAdapter:
    """Convert a timm ``EfficientNetFeatures`` model to MLX."""

    return MLXEfficientNetFeaturesAdapter(torch_model)


def torch_timm_universal_encoder_to_mlx(torch_encoder: Any) -> MLXTimmUniversalEncoderAdapter:
    """Convert SMP ``TimmUniversalEncoder`` to MLX."""

    return MLXTimmUniversalEncoderAdapter(torch_encoder)


def torch_unet_decoder_block_to_mlx(torch_block: Any) -> MLXUnetDecoderBlockAdapter:
    """Convert an SMP ``UnetDecoderBlock`` to MLX."""

    return MLXUnetDecoderBlockAdapter(torch_block)


def torch_unet_decoder_to_mlx(torch_decoder: Any) -> MLXUnetDecoderAdapter:
    """Convert an SMP ``UnetDecoder`` to MLX."""

    return MLXUnetDecoderAdapter(torch_decoder)


def torch_segmentation_head_to_mlx(torch_head: Any) -> MLXSegmentationHeadAdapter:
    """Convert an SMP ``SegmentationHead`` to MLX."""

    return MLXSegmentationHeadAdapter(torch_head)


def torch_segnet_to_mlx(torch_segnet: Any) -> MLXSegNetAdapter:
    """Convert upstream ``SegNet`` to an eval-mode MLX adapter."""

    return MLXSegNetAdapter(torch_segnet)


def torch_distortion_net_to_mlx(torch_distortion_net: Any) -> MLXDistortionScorerAdapter:
    """Convert upstream ``DistortionNet`` to MLX for fixed scorer-input tensors."""

    return MLXDistortionScorerAdapter(torch_distortion_net)


def torch_repmixer_block_to_mlx(torch_block: Any) -> MLXRepMixerBlockAdapter:
    """Convert a timm FastViT ``RepMixerBlock`` to a parity-tested MLX adapter."""

    return MLXRepMixerBlockAdapter(torch_block)


def torch_patch_embed_to_mlx(torch_patch_embed: Any) -> MLXPatchEmbedAdapter:
    """Convert a timm FastViT ``PatchEmbed`` to a parity-tested MLX adapter."""

    return MLXPatchEmbedAdapter(torch_patch_embed)


def torch_fastvit_stage_to_mlx(torch_stage: Any) -> MLXFastVitStageAdapter:
    """Convert a timm FastViT stage to a parity-tested MLX adapter."""

    return MLXFastVitStageAdapter(torch_stage)


def torch_fastvit_vision_to_mlx(torch_vision: Any) -> MLXFastVitVisionAdapter:
    """Convert upstream PoseNet's FastViT vision trunk to MLX."""

    return MLXFastVitVisionAdapter(torch_vision)


def torch_posenet_to_mlx(torch_posenet: Any) -> MLXPoseNetAdapter:
    """Convert upstream PoseNet to an eval-mode MLX adapter."""

    return MLXPoseNetAdapter(torch_posenet)


def run_mlx_conv2d_nchw(mlx_conv: Any, x_nchw: np.ndarray) -> np.ndarray:
    """Run an MLX Conv2d on NCHW input and return NCHW output."""

    import mlx.core as mx

    out = mlx_conv(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_reference_conv2d_nchw(
    adapter: MLXReferenceConv2dAdapter,
    x_nchw: np.ndarray,
) -> np.ndarray:
    """Run a fixed-order MLX Conv2d reference on NCHW input."""

    import mlx.core as mx

    out = adapter(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_conv2d_relu_nchw(adapter: MLXConv2dReLUAdapter, x_nchw: np.ndarray) -> np.ndarray:
    """Run an SMP Conv2dReLU adapter on NCHW input and return NCHW output."""

    import mlx.core as mx

    out = adapter(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_efficientnet_block_nchw(adapter: Any, x_nchw: np.ndarray) -> np.ndarray:
    """Run an EfficientNet block adapter on NCHW input and return NCHW output."""

    import mlx.core as mx

    out = adapter(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_efficientnet_stem_nchw(
    adapter: MLXEfficientNetStemAdapter,
    x_nchw: np.ndarray,
) -> np.ndarray:
    """Run an EfficientNet stem adapter on NCHW input and return NCHW output."""

    import mlx.core as mx

    out = adapter(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_efficientnet_features_nchw(
    adapter: MLXEfficientNetFeaturesAdapter,
    x_nchw: np.ndarray,
) -> list[np.ndarray]:
    """Run an EfficientNetFeatures adapter on NCHW input and return NCHW features."""

    import mlx.core as mx

    features = adapter(mx.array(nchw_to_nhwc(x_nchw)))
    return [nhwc_to_nchw(_mlx_array_to_numpy(feature)) for feature in features]


def run_mlx_timm_universal_encoder_nchw(
    adapter: MLXTimmUniversalEncoderAdapter,
    x_nchw: np.ndarray,
) -> list[np.ndarray]:
    """Run a TimmUniversalEncoder adapter on NCHW input and return NCHW features."""

    import mlx.core as mx

    features = adapter(mx.array(nchw_to_nhwc(x_nchw)))
    return [nhwc_to_nchw(_mlx_array_to_numpy(feature)) for feature in features]


def run_mlx_unet_decoder_block_nchw(
    adapter: MLXUnetDecoderBlockAdapter,
    feature_map_nchw: np.ndarray,
    target_height: int,
    target_width: int,
    skip_connection_nchw: np.ndarray | None = None,
) -> np.ndarray:
    """Run a UnetDecoderBlock adapter on NCHW input and return NCHW output."""

    import mlx.core as mx

    skip = (
        mx.array(nchw_to_nhwc(skip_connection_nchw))
        if skip_connection_nchw is not None
        else None
    )
    out = adapter(
        mx.array(nchw_to_nhwc(feature_map_nchw)),
        target_height,
        target_width,
        skip,
    )
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_unet_decoder_nchw(
    adapter: MLXUnetDecoderAdapter,
    features_nchw: list[np.ndarray],
) -> np.ndarray:
    """Run a UnetDecoder adapter on NCHW features and return NCHW output."""

    import mlx.core as mx

    features = [mx.array(nchw_to_nhwc(feature)) for feature in features_nchw]
    out = adapter(features)
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_segmentation_head_nchw(
    adapter: MLXSegmentationHeadAdapter,
    x_nchw: np.ndarray,
) -> np.ndarray:
    """Run a SegmentationHead adapter on NCHW input and return NCHW logits."""

    import mlx.core as mx

    out = adapter(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_segnet_nchw(adapter: MLXSegNetAdapter, x_nchw: np.ndarray) -> np.ndarray:
    """Run a SegNet adapter on NCHW input and return NCHW logits."""

    import mlx.core as mx

    out = adapter(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_distortion_scorer_nchw(
    adapter: MLXDistortionScorerAdapter,
    posenet_yuv6_pair_nchw: np.ndarray,
    segnet_last_rgb_nchw: np.ndarray,
) -> dict[str, Any]:
    """Run PoseNet+SegNet MLX responses on fixed scorer-input tensors."""

    import mlx.core as mx

    outputs = adapter(
        mx.array(nchw_to_nhwc(posenet_yuv6_pair_nchw)),
        mx.array(nchw_to_nhwc(segnet_last_rgb_nchw)),
    )
    return {
        "posenet": {
            name: _mlx_array_to_numpy(value)
            for name, value in outputs["posenet"].items()
        },
        "segnet": nhwc_to_nchw(_mlx_array_to_numpy(outputs["segnet"])),
    }


def run_mlx_efficientnet_stage_nchw(
    adapter: MLXEfficientNetStageAdapter,
    x_nchw: np.ndarray,
) -> np.ndarray:
    """Run an EfficientNet stage adapter on NCHW input and return NCHW output."""

    import mlx.core as mx

    out = adapter(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_batchnorm2d_nchw(mlx_bn: Any, x_nchw: np.ndarray) -> np.ndarray:
    """Run an MLX BatchNorm on NCHW input and return NCHW output."""

    import mlx.core as mx

    out = mlx_bn(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_linear(mlx_linear: Any, x: np.ndarray) -> np.ndarray:
    """Run an MLX Linear layer and return a NumPy array."""

    import mlx.core as mx

    return _mlx_array_to_numpy(mlx_linear(mx.array(np.ascontiguousarray(x))))


def run_mlx_mobileone_block_nchw(adapter: MLXMobileOneBlockAdapter, x_nchw: np.ndarray) -> np.ndarray:
    """Run a MobileOneBlock adapter on NCHW input and return NCHW output."""

    import mlx.core as mx

    out = adapter(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_mobileone_stem_nchw(adapter: MLXMobileOneStemAdapter, x_nchw: np.ndarray) -> np.ndarray:
    """Run a MobileOne stem adapter on NCHW input and return NCHW output."""

    import mlx.core as mx

    out = adapter(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_patch_embed_nchw(adapter: MLXPatchEmbedAdapter, x_nchw: np.ndarray) -> np.ndarray:
    """Run a FastViT PatchEmbed adapter on NCHW input and return NCHW output."""

    import mlx.core as mx

    out = adapter(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_repmixer_block_nchw(adapter: MLXRepMixerBlockAdapter, x_nchw: np.ndarray) -> np.ndarray:
    """Run a RepMixerBlock adapter on NCHW input and return NCHW output."""

    import mlx.core as mx

    out = adapter(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_fastvit_stage_nchw(adapter: MLXFastVitStageAdapter, x_nchw: np.ndarray) -> np.ndarray:
    """Run a FastViT stage adapter on NCHW input and return NCHW output."""

    import mlx.core as mx

    out = adapter(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def run_mlx_fastvit_vision_nchw(adapter: MLXFastVitVisionAdapter, x_nchw: np.ndarray) -> np.ndarray:
    """Run a FastViT vision adapter on NCHW input and return a NumPy array."""

    import mlx.core as mx

    return _mlx_array_to_numpy(adapter(mx.array(nchw_to_nhwc(x_nchw))))


def run_mlx_posenet_nchw(adapter: MLXPoseNetAdapter, x_nchw: np.ndarray) -> dict[str, np.ndarray]:
    """Run a PoseNet adapter on NCHW input and return NumPy output arrays."""

    import mlx.core as mx

    outputs = adapter(mx.array(nchw_to_nhwc(x_nchw)))
    return {name: _mlx_array_to_numpy(value) for name, value in outputs.items()}


def mlx_gelu_tanh(x: Any) -> Any:
    """MLX implementation of ``torch.nn.functional.gelu(..., approximate='tanh')``."""

    import mlx.core as mx

    return 0.5 * x * (
        1.0 + mx.tanh(0.7978845608028654 * (x + 0.044715 * mx.power(x, 3)))
    )


def mlx_relu(x: Any) -> Any:
    import mlx.core as mx

    return mx.maximum(x, 0.0)


def mlx_silu(x: Any) -> Any:
    return x * mlx_sigmoid(x)


def mlx_sigmoid(x: Any) -> Any:
    import mlx.core as mx

    return 1.0 / (1.0 + mx.exp(-x))


def scorer_distortion_components_numpy(
    reference_outputs: dict[str, Any],
    candidate_outputs: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Compute upstream PoseNet and SegNet distortion components from NumPy outputs."""

    reference_pose = np.asarray(reference_outputs["posenet"]["pose"], dtype=np.float32)
    candidate_pose = np.asarray(candidate_outputs["posenet"]["pose"], dtype=np.float32)
    pose_diff = reference_pose[..., :6] - candidate_pose[..., :6]
    pose_axes = tuple(range(1, pose_diff.ndim))
    pose_distortion = np.mean(np.square(pose_diff), axis=pose_axes, dtype=np.float32)

    reference_seg = np.asarray(reference_outputs["segnet"], dtype=np.float32)
    candidate_seg = np.asarray(candidate_outputs["segnet"], dtype=np.float32)
    seg_diff = np.argmax(reference_seg, axis=1) != np.argmax(candidate_seg, axis=1)
    seg_axes = tuple(range(1, seg_diff.ndim))
    seg_distortion = np.mean(seg_diff.astype(np.float32), axis=seg_axes, dtype=np.float32)

    return {
        "posenet": np.asarray(pose_distortion, dtype=np.float32),
        "segnet": np.asarray(seg_distortion, dtype=np.float32),
    }


def mlx_interpolate_nearest_size(x_nhwc: Any, target_height: int, target_width: int) -> Any:
    """Match ``torch.nn.functional.interpolate(..., size=..., mode='nearest')``."""

    import mlx.core as mx

    in_height = int(x_nhwc.shape[1])
    in_width = int(x_nhwc.shape[2])
    if in_height == target_height and in_width == target_width:
        return x_nhwc
    if in_height <= 0 or in_width <= 0 or target_height <= 0 or target_width <= 0:
        raise ValueError(
            "nearest interpolation requires positive input and target dimensions"
        )
    y_idx = mx.floor(mx.arange(target_height, dtype=mx.float32) * in_height / target_height).astype(mx.int32)
    x_idx = mx.floor(mx.arange(target_width, dtype=mx.float32) * in_width / target_width).astype(mx.int32)
    out = mx.take(x_nhwc, y_idx, axis=1)
    return mx.take(out, x_idx, axis=2)


def nchw_to_nhwc(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 4:
        raise ValueError(f"expected NCHW rank-4 input, got {arr.shape}")
    return np.ascontiguousarray(arr.transpose(0, 2, 3, 1))


def nhwc_to_nchw(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 4:
        raise ValueError(f"expected NHWC rank-4 input, got {arr.shape}")
    return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))


def _torch_tensor_to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _mlx_array_to_numpy(array: Any) -> np.ndarray:
    import mlx.core as mx

    mx.eval(array)
    try:
        mx.synchronize()
    except AttributeError:
        pass
    return np.asarray(array)


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"expected pair, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _class_path(obj: Any) -> str:
    if obj is None:
        return ""
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _apply_2d_activation(x: Any, activation_path: str) -> Any:
    if activation_path in {"", "torch.nn.modules.linear.Identity"}:
        return x
    if activation_path == "torch.nn.modules.activation.SiLU":
        return mlx_silu(x)
    raise NotImplementedError(f"unsupported 2D activation: {activation_path}")


def _torch_fastvit_child_to_mlx(child: Any) -> Any:
    class_path = _class_path(child)
    if class_path == "timm.models.fastvit.ReparamLargeKernelConv":
        return MLXReparamLargeKernelConvAdapter(child)
    if class_path == "timm.models.fastvit.MobileOneBlock":
        return torch_mobileone_block_to_mlx(child)
    raise NotImplementedError(f"unsupported FastViT child module: {class_path}")


def _torch_dense_child_to_mlx(child: Any) -> Any:
    class_path = _class_path(child)
    if class_path == "torch.nn.modules.linear.Linear":
        return torch_linear_to_mlx(child)
    if class_path == "torch.nn.modules.activation.ReLU":
        return mlx_relu
    if class_path == "modules.AllNorm":
        return MLXAllNormAdapter(child)
    if class_path == "modules.ResBlock":
        return MLXResBlockAdapter(child)
    raise NotImplementedError(f"unsupported dense child module: {class_path}")


class temporary_mlx_device:
    """Temporarily set the MLX default device to ``cpu`` or ``gpu``."""

    def __init__(self, device_type: str):
        if device_type not in {"cpu", "gpu"}:
            raise ValueError(f"device_type must be 'cpu' or 'gpu', got {device_type!r}")
        self._device_type = device_type
        self._old_device: Any | None = None

    def __enter__(self) -> None:
        import mlx.core as mx

        try:
            mx.synchronize()
        except AttributeError:
            pass
        try:
            mx.clear_cache()
        except AttributeError:
            pass
        self._old_device = mx.default_device()
        kind = mx.cpu if self._device_type == "cpu" else mx.gpu
        mx.set_default_device(mx.Device(kind))

    def __exit__(self, *_exc: object) -> None:
        if self._old_device is not None:
            import mlx.core as mx

            try:
                mx.synchronize()
            except AttributeError:
                pass
            mx.set_default_device(self._old_device)
