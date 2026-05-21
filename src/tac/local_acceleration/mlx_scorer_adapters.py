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
    "nchw_to_nhwc",
    "nhwc_to_nchw",
    "run_mlx_batchnorm2d_nchw",
    "run_mlx_conv2d_nchw",
    "run_mlx_linear",
    "run_mlx_patch_embed_nchw",
    "run_mlx_mobileone_block_nchw",
    "run_mlx_mobileone_stem_nchw",
    "run_mlx_repmixer_block_nchw",
    "run_mlx_fastvit_stage_nchw",
    "run_mlx_fastvit_vision_nchw",
    "torch_batchnorm2d_to_mlx",
    "torch_conv2d_to_mlx",
    "torch_conv_mlp_to_mlx",
    "torch_fastvit_stage_to_mlx",
    "torch_fastvit_vision_to_mlx",
    "torch_linear_to_mlx",
    "torch_mobileone_block_to_mlx",
    "torch_mobileone_stem_to_mlx",
    "torch_patch_embed_to_mlx",
    "run_mlx_posenet_nchw",
    "torch_posenet_to_mlx",
    "torch_repmixer_block_to_mlx",
    "temporary_mlx_device",
]


class MLXConvNormAct2dAdapter:
    """MLX adapter for timm ``ConvNormAct`` with activation omitted."""

    def __init__(self, torch_module: Any):
        if getattr(torch_module, "aa", None) is not None:
            raise NotImplementedError("ConvNormAct anti-alias branch is not covered")
        conv = getattr(torch_module, "conv", None)
        bn = getattr(torch_module, "bn", None)
        if conv is None or bn is None:
            raise TypeError("ConvNormAct adapter requires .conv and .bn children")
        self.conv = torch_conv2d_to_mlx(conv)
        self.bn = torch_batchnorm2d_to_mlx(bn)

    def __call__(self, x_nhwc: Any) -> Any:
        return self.bn(self.conv(x_nhwc))


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
        self.fc1 = torch_conv2d_to_mlx(torch_se.fc1)
        self.fc2 = torch_conv2d_to_mlx(torch_se.fc2)

    def __call__(self, x_nhwc: Any) -> Any:
        import mlx.core as mx

        x_se = mx.mean(x_nhwc, axis=(1, 2), keepdims=True)
        x_se = mlx_relu(self.fc1(x_se))
        x_se = self.fc2(x_se)
        return x_nhwc * mlx_sigmoid(x_se)


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
        self.use_gelu_tanh = _class_path(getattr(torch_block, "act", None)) == (
            "timm.layers.activations.GELUTanh"
        )
        if not self.use_gelu_tanh and _class_path(getattr(torch_block, "act", None)) != (
            "torch.nn.modules.linear.Identity"
        ):
            raise NotImplementedError(f"unsupported MobileOne activation: {_class_path(torch_block.act)}")

    def __call__(self, x_nhwc: Any) -> Any:
        import mlx.core as mx

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
            out = mx.zeros_like(x_nhwc)
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


def torch_batchnorm2d_to_mlx(torch_bn: Any) -> Any:
    """Convert eval-mode PyTorch ``nn.BatchNorm2d`` to MLX ``nn.BatchNorm``."""

    import mlx.core as mx
    import mlx.nn as nn

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


def mlx_sigmoid(x: Any) -> Any:
    import mlx.core as mx

    return 1.0 / (1.0 + mx.exp(-x))


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
