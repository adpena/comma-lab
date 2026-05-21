# SPDX-License-Identifier: MIT
"""Extended portable neural-network primitives for FastViT-T12 + EfficientNet-B2-UNet.

MLX-ARCH-1 (sister-1 of 5-stage MLX architecture port cascade per operator
directive 2026-05-21 + Carmack MVP-first 5-step per CLAUDE.md `be125b878`).

Extends OVERNIGHT-WW canonical primitives in :mod:`tac.portable_primitives.nn`
with 5 foundational ops required by FastViT (PoseNet backbone) +
EfficientNet-B2-UNet (SegNet backbone) architecture-level port:

- :class:`PortableBatchNorm2d` — 2D BatchNorm with running stats; eval-mode
  per PR101 contract + EfficientNet's frozen BN at inference
- :class:`PortableDepthwiseConv2d` — Conv2d groups=in_channels variant per
  FastViT RepMixer + EfficientNet MBConv depthwise stages
- :class:`PortableMaxPool2d` — spatial max pooling (FastViT stem alternative)
- :class:`PortableAvgPool2d` — spatial + global avg pooling (EfficientNet
  squeeze-excite + global pool)
- :func:`silu` — Swish/SiLU activation ``x * sigmoid(x)`` per EfficientNet

Per the canonical contract from :mod:`tac.portable_primitives.nn`: every
primitive exposes a uniform constructor signature
``Primitive(..., backend=...)`` and a forward call ``primitive(x)`` that
operates on backend-native tensors. Numerical equivalence MLX-vs-PyTorch
is pinned by sister tests in
``src/tac/portable_primitives/tests/test_portable_primitives_extended.py``
within ε ≤ 5e-3 fp32 per Phase 1 PV (Metal FMA reordering).

Per CLAUDE.md non-negotiables PRESERVED:
- **MPS auth eval is NOISE** (Catalog #1): MLX-backend scores remain
  non-promotable; this layer is PRIMITIVE-LEVEL only per ZZ scope.
- **QAT pipeline**: BatchNorm2d freeze (eval-mode + frozen running stats)
  per CLAUDE.md "QAT pipeline — non-negotiable for FP4 deployment" step 2.
- **Beauty, simplicity, and developer experience**: thin adapters; sister
  to MLX/PyTorch frameworks' native ops without re-implementing the math.

Sister of:
- :mod:`tac.portable_primitives.nn` (canonical base primitives)
- :mod:`tac.local_acceleration.mlx_to_pytorch_export` (weight export pipeline
  the architecture port depends on for MLX-train -> PyTorch-eval round-trip)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tac.portable_primitives.backend import Backend, resolve_backend

__all__ = [
    "PortableBatchNorm2d",
    "PortableDepthwiseConv2d",
    "PortableMaxPool2d",
    "PortableAvgPool2d",
    "silu",
]


class PortableBatchNorm2d:
    """Canonical BatchNorm2d with running stats.

    Constructor: ``PortableBatchNorm2d(num_features, eps, momentum, affine,
    track_running_stats, backend=...)``
    Forward: ``out = bn(x)`` where ``x.shape = (B, num_features, H, W)``.

    Per CLAUDE.md "QAT pipeline" step 2: BatchNorm stats are frozen during
    QAT fine-tune (eval mode). Use :meth:`train` / :meth:`eval` to toggle.

    The PyTorch backend stores params + buffers in PyTorch's canonical
    (B, C, H, W) NCHW layout. The MLX backend internally re-routes through
    NHWC at forward time (matching :class:`tac.portable_primitives.nn.PortableConv2d`
    convention), but the running_mean / running_var / weight / bias buffers
    remain shape ``(num_features,)`` which is layout-agnostic.
    """

    def __init__(
        self,
        num_features: int,
        *,
        backend: Backend | str,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)
        self.backend = resolve_backend(backend)
        self._training = True  # default to train mode; flip with .eval()

        if self.backend is Backend.MLX:
            import mlx.core as mx

            self._weight = mx.ones((self.num_features,)) if self.affine else None
            self._bias = mx.zeros((self.num_features,)) if self.affine else None
            if self.track_running_stats:
                self._running_mean = mx.zeros((self.num_features,))
                self._running_var = mx.ones((self.num_features,))
            else:
                self._running_mean = None
                self._running_var = None
        else:
            import torch

            self._weight = torch.ones(self.num_features) if self.affine else None
            self._bias = torch.zeros(self.num_features) if self.affine else None
            if self.track_running_stats:
                self._running_mean = torch.zeros(self.num_features)
                self._running_var = torch.ones(self.num_features)
            else:
                self._running_mean = None
                self._running_var = None

    def train(self) -> None:
        """Set training mode (use batch stats; update running stats)."""
        self._training = True

    def eval(self) -> None:
        """Set evaluation mode (use running stats; freeze updates).

        Per CLAUDE.md "QAT pipeline" step 2: BatchNorm MUST be frozen
        (``eval()``) before QAT fake-quant insertion.
        """
        self._training = False

    def __call__(self, x: Any) -> Any:
        if self.backend is Backend.MLX:
            return self._mlx_forward(x)
        return self._pytorch_forward(x)

    def _mlx_forward(self, x: Any) -> Any:
        """MLX forward: input is (B, C, H, W) NCHW; route via NHWC."""
        import mlx.core as mx

        # Convert NCHW -> NHWC for MLX-style ops; we compute manually here
        # to maintain numerical match with PyTorch's batch_norm. The MLX
        # nn.BatchNorm layer is also NHWC but has its own running-stats
        # state machinery; doing it manually keeps the weight/buffer
        # contract layout-agnostic.
        x_nhwc = mx.transpose(x, (0, 2, 3, 1))  # (B, H, W, C)

        if self._training or not self.track_running_stats:
            # Compute batch stats along (B, H, W) axes -> (C,).
            mean = mx.mean(x_nhwc, axis=(0, 1, 2))
            var = mx.var(x_nhwc, axis=(0, 1, 2))
            if self.track_running_stats and self._running_mean is not None:
                # Update running stats (in-place via reassignment).
                self._running_mean = (
                    (1.0 - self.momentum) * self._running_mean + self.momentum * mean
                )
                self._running_var = (
                    (1.0 - self.momentum) * self._running_var + self.momentum * var
                )
        else:
            mean = self._running_mean
            var = self._running_var

        x_norm = (x_nhwc - mean) * mx.rsqrt(var + self.eps)
        if self.affine:
            x_norm = x_norm * self._weight + self._bias

        # Back to NCHW.
        return mx.transpose(x_norm, (0, 3, 1, 2))

    def _pytorch_forward(self, x: Any) -> Any:
        """PyTorch forward: native F.batch_norm on (B, C, H, W)."""
        import torch.nn.functional as F

        return F.batch_norm(
            x,
            running_mean=self._running_mean if self.track_running_stats else None,
            running_var=self._running_var if self.track_running_stats else None,
            weight=self._weight if self.affine else None,
            bias=self._bias if self.affine else None,
            training=self._training or not self.track_running_stats,
            momentum=self.momentum,
            eps=self.eps,
        )

    def load_weights(
        self,
        weight_np: np.ndarray | None = None,
        bias_np: np.ndarray | None = None,
        running_mean_np: np.ndarray | None = None,
        running_var_np: np.ndarray | None = None,
    ) -> None:
        """Inject canonical weights + running stats."""
        for arr, name, expected in (
            (weight_np, "weight", (self.num_features,)),
            (bias_np, "bias", (self.num_features,)),
            (running_mean_np, "running_mean", (self.num_features,)),
            (running_var_np, "running_var", (self.num_features,)),
        ):
            if arr is not None and arr.shape != expected:
                raise ValueError(
                    f"{name} shape mismatch: expected {expected}, got {arr.shape}"
                )

        if self.backend is Backend.MLX:
            import mlx.core as mx

            if weight_np is not None and self.affine:
                self._weight = mx.array(weight_np.astype(np.float32))
            if bias_np is not None and self.affine:
                self._bias = mx.array(bias_np.astype(np.float32))
            if running_mean_np is not None and self.track_running_stats:
                self._running_mean = mx.array(running_mean_np.astype(np.float32))
            if running_var_np is not None and self.track_running_stats:
                self._running_var = mx.array(running_var_np.astype(np.float32))
        else:
            import torch

            if weight_np is not None and self.affine:
                self._weight = torch.from_numpy(weight_np.astype(np.float32).copy())
            if bias_np is not None and self.affine:
                self._bias = torch.from_numpy(bias_np.astype(np.float32).copy())
            if running_mean_np is not None and self.track_running_stats:
                self._running_mean = torch.from_numpy(
                    running_mean_np.astype(np.float32).copy()
                )
            if running_var_np is not None and self.track_running_stats:
                self._running_var = torch.from_numpy(
                    running_var_np.astype(np.float32).copy()
                )

    def export_weights(self) -> dict[str, np.ndarray]:
        """Export weights + running stats as numpy arrays.

        Returns dict keyed by ``weight`` / ``bias`` / ``running_mean`` /
        ``running_var`` (omitting entries where the field is None per
        constructor flags).
        """
        out: dict[str, np.ndarray] = {}
        if self.backend is Backend.MLX:
            import mlx.core as mx

            if self.affine:
                mx.eval(self._weight, self._bias)
                out["weight"] = np.array(self._weight)
                out["bias"] = np.array(self._bias)
            if self.track_running_stats:
                mx.eval(self._running_mean, self._running_var)
                out["running_mean"] = np.array(self._running_mean)
                out["running_var"] = np.array(self._running_var)
        else:
            if self.affine:
                out["weight"] = self._weight.detach().cpu().numpy()
                out["bias"] = self._bias.detach().cpu().numpy()
            if self.track_running_stats:
                out["running_mean"] = self._running_mean.detach().cpu().numpy()
                out["running_var"] = self._running_var.detach().cpu().numpy()
        return out


class PortableDepthwiseConv2d:
    """Canonical Depthwise Conv2d (groups=in_channels variant).

    Constructor: ``PortableDepthwiseConv2d(in_channels, kernel_size, stride,
    padding, bias, backend=...)``
    Forward: ``out = layer(x)`` where ``x.shape = (B, in_channels, H, W)``.
    Output: ``(B, in_channels, H_out, W_out)`` (one filter per input channel).

    Per FastViT RepMixer + EfficientNet MBConv: depthwise convolution is the
    canonical compute-light spatial mixer. PyTorch convention exposes this
    via ``nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels)``.
    MLX exposes the same via ``mlx.nn.Conv2d(... groups=in_channels)``.

    Weight shape (PyTorch layout, canonical for export):
    ``(in_channels, 1, kernel_size, kernel_size)``
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        *,
        backend: Backend | str,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = True,
        seed: int | None = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        # Default to same-padding for stride=1 if not specified.
        self.padding = (
            self.kernel_size // 2 if padding is None else int(padding)
        )
        self.bias = bool(bias)
        self.backend = resolve_backend(backend)

        # Canonical init: He uniform per PyTorch default.
        rng = np.random.RandomState(seed if seed is not None else 0)
        fan_in = self.kernel_size * self.kernel_size  # groups=in_channels => fan_in per channel
        bound = (1.0 / fan_in) ** 0.5
        # PyTorch depthwise weight shape: (in_channels, 1, kH, kW).
        w_np = rng.uniform(
            -bound,
            bound,
            size=(self.in_channels, 1, self.kernel_size, self.kernel_size),
        ).astype(np.float32)
        b_np = np.zeros(self.in_channels, dtype=np.float32) if self.bias else None

        if self.backend is Backend.MLX:
            import mlx.core as mx

            self._w_pt_layout = mx.array(w_np)
            # MLX Conv2d expects (out_channels, kH, kW, in_channels_per_group)
            # For depthwise: out_channels=in_channels, in_channels_per_group=1.
            # Transpose (C, 1, kH, kW) -> (C, kH, kW, 1).
            self._w_mlx_layout = mx.array(
                np.transpose(w_np, (0, 2, 3, 1)).copy()
            )
            self._b = mx.array(b_np) if self.bias else None
        else:
            import torch

            self._w_pt_layout = torch.from_numpy(w_np)
            self._b = torch.from_numpy(b_np) if self.bias else None

    def __call__(self, x: Any) -> Any:
        if self.backend is Backend.MLX:
            import mlx.core as mx

            # NCHW -> NHWC for MLX.
            x_nhwc = mx.transpose(x, (0, 2, 3, 1))
            out_nhwc = mx.conv2d(
                x_nhwc,
                self._w_mlx_layout,
                stride=self.stride,
                padding=self.padding,
                groups=self.in_channels,
            )
            if self.bias:
                out_nhwc = out_nhwc + self._b
            return mx.transpose(out_nhwc, (0, 3, 1, 2))

        import torch.nn.functional as F

        return F.conv2d(
            x,
            self._w_pt_layout,
            self._b,
            stride=self.stride,
            padding=self.padding,
            groups=self.in_channels,
        )

    def load_weights(
        self,
        w_pt_layout_np: np.ndarray,
        b_np: np.ndarray | None = None,
    ) -> None:
        """Inject canonical weights in PyTorch layout (in_channels, 1, kH, kW)."""
        expected = (self.in_channels, 1, self.kernel_size, self.kernel_size)
        if w_pt_layout_np.shape != expected:
            raise ValueError(
                f"weight shape mismatch: expected {expected}, got {w_pt_layout_np.shape}"
            )
        if self.bias:
            if b_np is None:
                raise ValueError("bias=True requires b_np")
            if b_np.shape != (self.in_channels,):
                raise ValueError(
                    f"bias shape mismatch: expected {(self.in_channels,)}, got {b_np.shape}"
                )
        w_pt = w_pt_layout_np.astype(np.float32)
        if self.backend is Backend.MLX:
            import mlx.core as mx

            self._w_pt_layout = mx.array(w_pt)
            self._w_mlx_layout = mx.array(np.transpose(w_pt, (0, 2, 3, 1)).copy())
            if self.bias and b_np is not None:
                self._b = mx.array(b_np.astype(np.float32))
        else:
            import torch

            self._w_pt_layout = torch.from_numpy(w_pt.copy())
            if self.bias and b_np is not None:
                self._b = torch.from_numpy(b_np.astype(np.float32).copy())

    def export_weights(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Export weights as numpy arrays in PyTorch layout (canonical)."""
        if self.backend is Backend.MLX:
            import mlx.core as mx

            mx.eval(self._w_pt_layout)
            w = np.array(self._w_pt_layout)
            if self.bias:
                mx.eval(self._b)
                return w, np.array(self._b)
            return w, None
        w = self._w_pt_layout.detach().cpu().numpy()
        if self.bias:
            return w, self._b.detach().cpu().numpy()
        return w, None


class PortableMaxPool2d:
    """Canonical 2D max pooling.

    Constructor: ``PortableMaxPool2d(kernel_size, stride, padding, backend=...)``
    Forward: ``out = pool(x)`` where ``x.shape = (B, C, H, W)``.

    Default ``stride=None`` matches PyTorch convention (defaults to
    ``kernel_size``). Per FastViT stem: typically kernel_size=3, stride=2,
    padding=1.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        *,
        backend: Backend | str,
        stride: int | None = None,
        padding: int = 0,
    ) -> None:
        self.kernel_size = int(kernel_size)
        self.stride = self.kernel_size if stride is None else int(stride)
        self.padding = int(padding)
        self.backend = resolve_backend(backend)

    def __call__(self, x: Any) -> Any:
        if self.backend is Backend.MLX:
            import mlx.core as mx
            import mlx.nn as mlxnn

            # MLX MaxPool2d operates on NHWC.
            x_nhwc = mx.transpose(x, (0, 2, 3, 1))
            pool = mlxnn.MaxPool2d(
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
            out_nhwc = pool(x_nhwc)
            return mx.transpose(out_nhwc, (0, 3, 1, 2))

        import torch.nn.functional as F

        return F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )


class PortableAvgPool2d:
    """Canonical 2D average pooling.

    Constructor: ``PortableAvgPool2d(kernel_size, stride, padding, backend=...)``
    Forward: ``out = pool(x)`` where ``x.shape = (B, C, H, W)``.

    Supports global average pooling by setting ``kernel_size = (H, W)`` to
    match input spatial dims. Per EfficientNet squeeze-excite + final
    global pool: typically ``kernel_size`` matches feature-map spatial size.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int] = 2,
        *,
        backend: Backend | str,
        stride: int | tuple[int, int] | None = None,
        padding: int = 0,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = int(padding)
        self.backend = resolve_backend(backend)

    def __call__(self, x: Any) -> Any:
        if self.backend is Backend.MLX:
            import mlx.core as mx
            import mlx.nn as mlxnn

            x_nhwc = mx.transpose(x, (0, 2, 3, 1))
            pool = mlxnn.AvgPool2d(
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
            out_nhwc = pool(x_nhwc)
            return mx.transpose(out_nhwc, (0, 3, 1, 2))

        import torch.nn.functional as F

        return F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )


def silu(x: Any, *, backend: Backend | str) -> Any:
    """Swish / SiLU activation: ``x * sigmoid(x)``.

    Canonical EfficientNet activation (per Tan & Le 2019). MLX exposes
    via :func:`mlx.nn.silu`; PyTorch via :func:`torch.nn.functional.silu`.

    Both backends compute the exact form (no tanh approximation).
    """
    kind = resolve_backend(backend)
    if kind is Backend.MLX:
        import mlx.nn as mlxnn

        return mlxnn.silu(x)
    import torch.nn.functional as F

    return F.silu(x)
