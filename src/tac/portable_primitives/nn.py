# SPDX-License-Identifier: MIT
"""Canonical neural-network primitives with MLX + PyTorch sister implementations.

Per OVERNIGHT-WW: the primitives below cover the surface needed by Selfcomp
grayscale_lut + STC sidecar + sister substrate-class trainers. Each primitive
exposes a uniform constructor signature ``Primitive(backend=...)`` and a
forward call ``primitive(x)`` that operates on the appropriate backend's
native tensors.

Numerical equivalence MLX-vs-PyTorch is pinned by sister tests in
``src/tac/portable_primitives/tests/`` within ε ≤ 1e-5 fp32 OR ε ≤ 1e-3 bf16.

Per the canonical contract: callers must initialize primitives with the
SAME weight initialization seeded externally if they want cross-backend
numerical match — the primitives don't seed themselves so the canonical
weight-export pipeline (:mod:`tac.local_acceleration.mlx_to_pytorch_export`)
remains the source of truth for converting trained MLX weights to PyTorch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tac.portable_primitives.backend import Backend, resolve_backend

__all__ = [
    "PortableLinear",
    "PortableConv2d",
    "PortableLayerNorm",
    "gelu",
    "relu",
    "sigmoid",
    "softmax",
    "bilinear_upsample",
    "matmul",
]


class PortableLinear:
    """Canonical Linear (Dense) layer.

    Constructor: ``PortableLinear(in_features, out_features, backend=...)``
    Forward: ``out = layer(x)`` where ``x.shape = (..., in_features)``.

    Weights initialized via ``np.random.RandomState(seed)`` for determinism
    if seed is supplied; otherwise backend-default init (which differs
    between MLX and PyTorch, intentionally — caller should seed externally
    or use :meth:`load_weights` to inject canonical weights).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        backend: Backend | str,
        seed: int | None = None,
    ) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.backend = resolve_backend(backend)

        # Canonical init: Xavier uniform with seed for reproducibility.
        rng = np.random.RandomState(seed if seed is not None else 0)
        bound = (6.0 / (self.in_features + self.out_features)) ** 0.5
        w_np = rng.uniform(-bound, bound, size=(self.out_features, self.in_features)).astype(np.float32)
        b_np = np.zeros(self.out_features, dtype=np.float32)

        if self.backend is Backend.MLX:
            import mlx.core as mx

            self._w = mx.array(w_np)
            self._b = mx.array(b_np)
        else:
            import torch

            self._w = torch.from_numpy(w_np)
            self._b = torch.from_numpy(b_np)

    def __call__(self, x: Any) -> Any:
        if self.backend is Backend.MLX:
            import mlx.core as mx

            return mx.matmul(x, self._w.T) + self._b
        import torch
        import torch.nn.functional as F

        return F.linear(x, self._w, self._b)

    def load_weights(self, w_np: np.ndarray, b_np: np.ndarray) -> None:
        """Inject canonical weights (used by export pipeline)."""
        if w_np.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"weight shape mismatch: expected {(self.out_features, self.in_features)}, got {w_np.shape}"
            )
        if b_np.shape != (self.out_features,):
            raise ValueError(
                f"bias shape mismatch: expected {(self.out_features,)}, got {b_np.shape}"
            )
        if self.backend is Backend.MLX:
            import mlx.core as mx

            self._w = mx.array(w_np.astype(np.float32))
            self._b = mx.array(b_np.astype(np.float32))
        else:
            import torch

            self._w = torch.from_numpy(w_np.astype(np.float32).copy())
            self._b = torch.from_numpy(b_np.astype(np.float32).copy())

    def export_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Export weights as numpy arrays (canonical export pipeline format)."""
        if self.backend is Backend.MLX:
            import mlx.core as mx

            mx.eval(self._w, self._b)
            return np.array(self._w), np.array(self._b)
        return self._w.detach().cpu().numpy(), self._b.detach().cpu().numpy()


class PortableConv2d:
    """Canonical Conv2d layer (kernel_size square, stride 1, padding via 'same').

    Constructor: ``PortableConv2d(in_channels, out_channels, kernel_size, backend=...)``
    Forward: ``out = layer(x)`` where ``x.shape = (B, in_channels, H, W)``.
    Output: ``(B, out_channels, H, W)`` with same-padding.

    For Selfcomp grayscale_lut we need kernel_size=3, stride=1, padding=1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        backend: Backend | str,
        seed: int | None = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.backend = resolve_backend(backend)

        # Canonical init: He uniform per PyTorch default.
        rng = np.random.RandomState(seed if seed is not None else 0)
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound = (1.0 / fan_in) ** 0.5
        # PyTorch Conv2d weight shape: (out_channels, in_channels, kH, kW)
        w_np = rng.uniform(
            -bound, bound,
            size=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        ).astype(np.float32)
        b_np = np.zeros(self.out_channels, dtype=np.float32)

        if self.backend is Backend.MLX:
            import mlx.core as mx

            # MLX Conv2d expects weight shape (out_channels, kH, kW, in_channels) (NHWC layout).
            # We store in the canonical PyTorch layout and transpose at forward time.
            self._w_pt_layout = mx.array(w_np)
            self._w_mlx_layout = mx.array(np.transpose(w_np, (0, 2, 3, 1)).copy())
            self._b = mx.array(b_np)
        else:
            import torch

            self._w_pt_layout = torch.from_numpy(w_np)
            self._b = torch.from_numpy(b_np)

    def __call__(self, x: Any) -> Any:
        if self.backend is Backend.MLX:
            import mlx.core as mx
            import mlx.nn as nn

            # MLX conv2d expects NHWC input layout; PyTorch convention is NCHW.
            # Convert at the boundary.
            x_nhwc = mx.transpose(x, (0, 2, 3, 1))
            out_nhwc = mx.conv2d(x_nhwc, self._w_mlx_layout, stride=1, padding=self.padding)
            # Add bias (broadcast across spatial dims). Output shape (B, H, W, C_out).
            out_nhwc = out_nhwc + self._b
            # Convert back to NCHW.
            return mx.transpose(out_nhwc, (0, 3, 1, 2))
        import torch
        import torch.nn.functional as F

        return F.conv2d(x, self._w_pt_layout, self._b, stride=1, padding=self.padding)

    def load_weights(self, w_pt_layout_np: np.ndarray, b_np: np.ndarray) -> None:
        """Inject canonical weights in PyTorch layout (out_channels, in_channels, kH, kW)."""
        expected = (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        if w_pt_layout_np.shape != expected:
            raise ValueError(f"weight shape mismatch: expected {expected}, got {w_pt_layout_np.shape}")
        if b_np.shape != (self.out_channels,):
            raise ValueError(f"bias shape mismatch: expected {(self.out_channels,)}, got {b_np.shape}")
        w_pt = w_pt_layout_np.astype(np.float32)
        if self.backend is Backend.MLX:
            import mlx.core as mx

            self._w_pt_layout = mx.array(w_pt)
            self._w_mlx_layout = mx.array(np.transpose(w_pt, (0, 2, 3, 1)).copy())
            self._b = mx.array(b_np.astype(np.float32))
        else:
            import torch

            self._w_pt_layout = torch.from_numpy(w_pt.copy())
            self._b = torch.from_numpy(b_np.astype(np.float32).copy())

    def export_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Export weights as numpy arrays in PyTorch layout (canonical)."""
        if self.backend is Backend.MLX:
            import mlx.core as mx

            mx.eval(self._w_pt_layout, self._b)
            return np.array(self._w_pt_layout), np.array(self._b)
        return self._w_pt_layout.detach().cpu().numpy(), self._b.detach().cpu().numpy()


class PortableLayerNorm:
    """Canonical LayerNorm over the last dim."""

    def __init__(
        self,
        normalized_shape: int,
        *,
        backend: Backend | str,
        eps: float = 1e-5,
    ) -> None:
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.backend = resolve_backend(backend)

        if self.backend is Backend.MLX:
            import mlx.core as mx

            self._gamma = mx.ones((self.normalized_shape,))
            self._beta = mx.zeros((self.normalized_shape,))
        else:
            import torch

            self._gamma = torch.ones(self.normalized_shape)
            self._beta = torch.zeros(self.normalized_shape)

    def __call__(self, x: Any) -> Any:
        if self.backend is Backend.MLX:
            import mlx.core as mx

            mean = mx.mean(x, axis=-1, keepdims=True)
            var = mx.var(x, axis=-1, keepdims=True)
            x_norm = (x - mean) * mx.rsqrt(var + self.eps)
            return x_norm * self._gamma + self._beta
        import torch
        import torch.nn.functional as F

        return F.layer_norm(x, (self.normalized_shape,), self._gamma, self._beta, self.eps)


def gelu(x: Any, *, backend: Backend | str) -> Any:
    """GELU activation (exact, not tanh approximation)."""
    kind = resolve_backend(backend)
    if kind is Backend.MLX:
        import mlx.nn as nn

        return nn.gelu(x)
    import torch.nn.functional as F

    return F.gelu(x)


def relu(x: Any, *, backend: Backend | str) -> Any:
    """ReLU activation."""
    kind = resolve_backend(backend)
    if kind is Backend.MLX:
        import mlx.core as mx

        return mx.maximum(x, 0.0)
    import torch.nn.functional as F

    return F.relu(x)


def sigmoid(x: Any, *, backend: Backend | str) -> Any:
    """Sigmoid activation."""
    kind = resolve_backend(backend)
    if kind is Backend.MLX:
        import mlx.core as mx

        return mx.sigmoid(x)
    import torch

    return torch.sigmoid(x)


def softmax(x: Any, *, axis: int = -1, backend: Backend | str) -> Any:
    """Softmax along the given axis."""
    kind = resolve_backend(backend)
    if kind is Backend.MLX:
        import mlx.core as mx

        return mx.softmax(x, axis=axis)
    import torch
    import torch.nn.functional as F

    return F.softmax(x, dim=axis)


def matmul(a: Any, b: Any, *, backend: Backend | str) -> Any:
    """Matrix multiplication."""
    kind = resolve_backend(backend)
    if kind is Backend.MLX:
        import mlx.core as mx

        return mx.matmul(a, b)
    import torch

    return torch.matmul(a, b)


def bilinear_upsample(
    x: Any,
    *,
    size: tuple[int, int],
    backend: Backend | str,
    align_corners: bool = False,
) -> Any:
    """Bilinear upsample (B, C, H, W) -> (B, C, *size).

    align_corners=False matches PyTorch's default (and Selfcomp's usage in
    architecture.py via F.interpolate). MLX does not have native
    interpolate; we implement via vectorized indexing for the NN-grid case
    or fall back to numpy round-trip for MLX. For Selfcomp the size jump is
    96->384 (4x), 128->512 (4x).
    """
    kind = resolve_backend(backend)
    if kind is Backend.PYTORCH:
        import torch.nn.functional as F

        return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)

    # MLX implementation: numpy round-trip is acceptable for ε equivalence
    # because PyTorch's bilinear interpolate is itself fp32 numerical; we
    # use the reference numpy implementation for byte-stable behavior.
    import mlx.core as mx
    import numpy as np
    import torch
    import torch.nn.functional as F

    # Materialize, route through PyTorch's reference impl, hand back to MLX.
    # This is intentional: MLX 0.x doesn't ship a 1:1-faithful bilinear
    # interpolate primitive yet; reaching through PyTorch CPU keeps the
    # numerics exact while preserving the portable API.
    mx.eval(x)
    x_np = np.array(x)
    x_t = torch.from_numpy(x_np)
    y_t = F.interpolate(x_t, size=size, mode="bilinear", align_corners=align_corners)
    return mx.array(y_t.numpy())
