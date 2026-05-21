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
    "torch_batchnorm2d_to_mlx",
    "torch_conv2d_to_mlx",
    "torch_linear_to_mlx",
    "temporary_mlx_device",
]


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


def run_mlx_conv2d_nchw(mlx_conv: Any, x_nchw: np.ndarray) -> np.ndarray:
    """Run an MLX Conv2d on NCHW input and return NCHW output."""

    import mlx.core as mx

    out = mlx_conv(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(np.asarray(out))


def run_mlx_batchnorm2d_nchw(mlx_bn: Any, x_nchw: np.ndarray) -> np.ndarray:
    """Run an MLX BatchNorm on NCHW input and return NCHW output."""

    import mlx.core as mx

    out = mlx_bn(mx.array(nchw_to_nhwc(x_nchw)))
    return nhwc_to_nchw(np.asarray(out))


def run_mlx_linear(mlx_linear: Any, x: np.ndarray) -> np.ndarray:
    """Run an MLX Linear layer and return a NumPy array."""

    import mlx.core as mx

    return np.asarray(mlx_linear(mx.array(np.ascontiguousarray(x))))


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


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"expected pair, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class temporary_mlx_device:
    """Temporarily set the MLX default device to ``cpu`` or ``gpu``."""

    def __init__(self, device_type: str):
        if device_type not in {"cpu", "gpu"}:
            raise ValueError(f"device_type must be 'cpu' or 'gpu', got {device_type!r}")
        self._device_type = device_type
        self._old_device: Any | None = None

    def __enter__(self) -> None:
        import mlx.core as mx

        self._old_device = mx.default_device()
        kind = mx.cpu if self._device_type == "cpu" else mx.gpu
        mx.set_default_device(mx.Device(kind))

    def __exit__(self, *_exc: object) -> None:
        if self._old_device is not None:
            import mlx.core as mx

            mx.set_default_device(self._old_device)
