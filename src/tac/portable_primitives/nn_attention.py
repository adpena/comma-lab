# SPDX-License-Identifier: MIT
"""Attention primitives for FastViT-T12 PoseNet backbone port.

MLX-ARCH-2 (sister-2 of 5-stage MLX architecture port cascade per operator
directive 2026-05-21 + Carmack MVP-first 5-step per CLAUDE.md `be125b878`).

Extends OVERNIGHT-WW canonical base primitives (:mod:`tac.portable_primitives.nn`)
+ MLX-ARCH-1 foundational primitives (:mod:`tac.portable_primitives.nn_extended`)
with 4 attention-specific primitives required by FastViT-T12 (PoseNet
backbone) architecture-level port:

- :class:`PortableLayerScale` — per-channel learnable scale applied to
  residual branch output before skip connection (FastViT init ``γ=1e-5``)
- :class:`PortableMHSA` — Multi-Head Self-Attention block; Q/K/V via
  fused linear projection; scaled dot-product attention; output projection
  (FastViT stages 3+4; reference timm ``fastvit_t12.MHSA``)
- :class:`PortableTokenMixer` — channel-mixing MLP per FastViT (alternates
  with spatial mixer ``PortableRepMixer`` per stage); 2-layer MLP with
  GELU activation
- :class:`PortableRepMixer` — re-parameterized depthwise conv block per
  FastViT paper (training: 3 branches = 3x3 DW + 1x1 + skip; inference:
  fused single 3x3 DW); used in FastViT stages 1+2 as the spatial mixer

Per the canonical contract from :mod:`tac.portable_primitives.nn`: every
primitive exposes a uniform constructor signature
``Primitive(..., backend=...)`` and a forward call ``primitive(x)`` that
operates on backend-native tensors. Numerical equivalence MLX-vs-PyTorch
is pinned by sister tests in
``src/tac/portable_primitives/tests/test_portable_primitives_attention.py``
within ε ≤ 5e-3 fp32 per ARCH-1 Phase 1 PV (Metal FMA reordering).

**Architecture reference (FastViT-T12)**: 4 stages with progressive
downsampling (96→192→384→768 dimensions); stages 1+2 use RepMixer
spatial mixer; stages 3+4 use MHSA; each stage has LayerScale per
residual branch; token mixer alternates per stage; Hydra head:
vision(2048) → summary(512) → ResBlock → 12-dim pose → first 6 used.

Per CLAUDE.md non-negotiables PRESERVED:
- **MPS auth eval is NOISE** (Catalog #1): MLX-backend scores remain
  non-promotable; this layer is PRIMITIVE-LEVEL only per ZZ scope; ARCH-5
  paired-eval is where contest-axis promotion gates apply.
- **Beauty, simplicity, and developer experience**: thin adapters; sister
  to MLX/PyTorch frameworks' native ops without re-implementing the math.

Sister of:
- :mod:`tac.portable_primitives.nn` (canonical base 9 primitives)
- :mod:`tac.portable_primitives.nn_extended` (ARCH-1 5 foundational ops)
- :mod:`tac.local_acceleration.mlx_to_pytorch_export` (weight export pipeline
  the architecture port depends on for MLX-train -> PyTorch-eval round-trip)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tac.portable_primitives.backend import Backend, resolve_backend
from tac.portable_primitives.nn import (
    PortableConv2d,
    PortableLayerNorm,
    PortableLinear,
    gelu,
)
from tac.portable_primitives.nn_extended import PortableDepthwiseConv2d

__all__ = [
    "PortableLayerScale",
    "PortableMHSA",
    "PortableTokenMixer",
    "PortableRepMixer",
]


class PortableLayerScale:
    """Canonical LayerScale primitive (per-channel learnable scale).

    Constructor: ``PortableLayerScale(num_channels, init_value, backend=...)``
    Forward: ``out = scale(x)`` where ``x.shape = (B, num_channels, ...)``
    OR ``(B, ..., num_channels)`` (last-dim form for transformer tokens).

    Per FastViT paper (Vasu et al. 2023) + CaiT (Touvron et al. 2021):
    ``LayerScale(x) = γ * x`` where ``γ`` is a learnable per-channel
    parameter initialized to a small value (default ``1e-5``). Applied
    to residual branch output BEFORE skip connection so the residual
    starts as effectively identity.

    The primitive auto-detects layout: if ``x.shape[-1] == num_channels``
    (token form, e.g. transformer ``(B, N, C)``) the scale is broadcast
    along the channel axis; if ``x.shape[1] == num_channels`` (NCHW
    conv form) the scale is reshaped for broadcasting along H, W.
    Callers can pass ``channels_last=True`` to force the token-form
    layout when ``num_channels`` matches both ``shape[1]`` and ``shape[-1]``
    (ambiguous cases).
    """

    def __init__(
        self,
        num_channels: int,
        *,
        backend: Backend | str,
        init_value: float = 1e-5,
        channels_last: bool | None = None,
    ) -> None:
        self.num_channels = int(num_channels)
        self.init_value = float(init_value)
        self.backend = resolve_backend(backend)
        self.channels_last = channels_last  # None => auto-detect

        gamma_np = np.full((self.num_channels,), self.init_value, dtype=np.float32)

        if self.backend is Backend.MLX:
            import mlx.core as mx

            self._gamma = mx.array(gamma_np)
        else:
            import torch

            self._gamma = torch.from_numpy(gamma_np)

    def __call__(self, x: Any) -> Any:
        # Resolve layout: token-form (channels at last dim) vs NCHW (channels at dim 1).
        if self.channels_last is not None:
            channels_last = self.channels_last
        else:
            # Auto-detect: if last-dim matches num_channels and dim 1 doesn't, token form;
            # otherwise NCHW form.
            shape = tuple(x.shape)
            channels_last = (
                len(shape) >= 2
                and shape[-1] == self.num_channels
                and (len(shape) == 2 or shape[1] != self.num_channels)
            )

        if self.backend is Backend.MLX:
            import mlx.core as mx

            if channels_last:
                # x.shape (..., C); broadcast gamma along last dim.
                return x * self._gamma
            # NCHW form: reshape gamma to (1, C, 1, 1, ...) for broadcasting.
            ndim = len(x.shape)
            reshape_shape = [1] * ndim
            reshape_shape[1] = self.num_channels
            gamma_reshape = mx.reshape(self._gamma, tuple(reshape_shape))
            return x * gamma_reshape

        import torch

        if channels_last:
            return x * self._gamma
        ndim = x.dim()
        reshape_shape = [1] * ndim
        reshape_shape[1] = self.num_channels
        gamma_reshape = self._gamma.reshape(reshape_shape)
        return x * gamma_reshape

    def load_weights(self, gamma_np: np.ndarray) -> None:
        """Inject canonical learned gamma."""
        if gamma_np.shape != (self.num_channels,):
            raise ValueError(
                f"gamma shape mismatch: expected {(self.num_channels,)}, got {gamma_np.shape}"
            )
        if self.backend is Backend.MLX:
            import mlx.core as mx

            self._gamma = mx.array(gamma_np.astype(np.float32))
        else:
            import torch

            self._gamma = torch.from_numpy(gamma_np.astype(np.float32).copy())

    def export_weights(self) -> np.ndarray:
        """Export gamma as numpy array."""
        if self.backend is Backend.MLX:
            import mlx.core as mx

            mx.eval(self._gamma)
            return np.array(self._gamma)
        return self._gamma.detach().cpu().numpy()


class PortableMHSA:
    """Canonical Multi-Head Self-Attention (MHSA) block.

    Constructor: ``PortableMHSA(dim, num_heads, qkv_bias, backend=...)``
    Forward: ``out = mhsa(x)`` where ``x.shape = (B, N, dim)`` (token form;
    N = number of tokens; dim = embed dim).

    Implements the canonical scaled dot-product self-attention from
    Vaswani et al. 2017 + reference timm ``Attention`` module:

        Q, K, V = split(Linear(x))   # (B, N, 3*dim) -> 3x (B, N, dim)
        Q, K, V = reshape_heads(Q, K, V)  # (B, num_heads, N, head_dim)
        attn = softmax(Q @ K^T / sqrt(head_dim))  # (B, num_heads, N, N)
        out = attn @ V  # (B, num_heads, N, head_dim)
        out = reshape_combine(out)  # (B, N, dim)
        return Linear(out)

    Per FastViT stages 3+4: ``num_heads=8`` is the canonical reference.
    ``dim`` MUST be divisible by ``num_heads``.

    Note: this primitive does NOT include relative positional encodings,
    attention bias, or attention dropout (those are typically per-stage
    sister wrappers, e.g. FastViT stages 3+4 add relpos). Sister-3
    (FastViT stage assembly) layers them on top.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        *,
        backend: Backend | str,
        qkv_bias: bool = True,
        seed: int | None = None,
    ) -> None:
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        # Canonical scale = 1/sqrt(head_dim).
        self.scale = float(self.head_dim) ** -0.5
        self.qkv_bias = bool(qkv_bias)
        self.backend = resolve_backend(backend)

        # Fused QKV projection (canonical timm convention; 3x faster than 3 separate
        # linears because of single matmul call).
        self._qkv = PortableLinear(
            self.dim, self.dim * 3, backend=self.backend, seed=seed
        )
        # Zero out QKV bias if disabled (PortableLinear always allocates bias).
        if not self.qkv_bias:
            zero_bias = np.zeros((self.dim * 3,), dtype=np.float32)
            w_existing, _ = self._qkv.export_weights()
            self._qkv.load_weights(w_existing, zero_bias)

        # Output projection.
        self._proj = PortableLinear(
            self.dim, self.dim, backend=self.backend, seed=(seed + 1 if seed is not None else None)
        )

    def __call__(self, x: Any) -> Any:
        # Input shape: (B, N, dim). Output shape: (B, N, dim).
        if self.backend is Backend.MLX:
            import mlx.core as mx

            B, N, C = x.shape
            # Fused QKV projection: (B, N, 3*dim).
            qkv = self._qkv(x)
            # Reshape and split: (B, N, 3, num_heads, head_dim).
            qkv = mx.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
            # Transpose to (3, B, num_heads, N, head_dim).
            qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
            # Scaled dot-product: (B, num_heads, N, N).
            attn = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale
            attn = mx.softmax(attn, axis=-1)
            # Attention output: (B, num_heads, N, head_dim).
            out = mx.matmul(attn, v)
            # Combine heads: (B, N, num_heads, head_dim) -> (B, N, dim).
            out = mx.transpose(out, (0, 2, 1, 3))
            out = mx.reshape(out, (B, N, C))
            # Output projection.
            return self._proj(out)

        import torch
        import torch.nn.functional as F

        B, N, C = x.shape
        qkv = self._qkv(x)  # (B, N, 3*dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Manual scaled dot-product (not torch SDPA fused kernel) so MLX + PyTorch
        # match within Phase 1 ε band; SDPA may use different reduction order.
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, N, C)
        return self._proj(out)

    def load_weights(
        self,
        qkv_w_np: np.ndarray,
        qkv_b_np: np.ndarray | None,
        proj_w_np: np.ndarray,
        proj_b_np: np.ndarray,
    ) -> None:
        """Inject canonical learned weights.

        Shapes:
        - ``qkv_w_np``: ``(3*dim, dim)`` (PyTorch Linear weight layout)
        - ``qkv_b_np``: ``(3*dim,)`` or None (None = use zero bias when ``qkv_bias=False``)
        - ``proj_w_np``: ``(dim, dim)``
        - ``proj_b_np``: ``(dim,)``
        """
        zero_qkv_b = np.zeros((self.dim * 3,), dtype=np.float32)
        self._qkv.load_weights(qkv_w_np, qkv_b_np if qkv_b_np is not None else zero_qkv_b)
        self._proj.load_weights(proj_w_np, proj_b_np)

    def export_weights(self) -> dict[str, np.ndarray]:
        """Export weights as numpy dict (canonical for sister Wave 4 export pipeline)."""
        qkv_w, qkv_b = self._qkv.export_weights()
        proj_w, proj_b = self._proj.export_weights()
        return {
            "qkv_weight": qkv_w,
            "qkv_bias": qkv_b,
            "proj_weight": proj_w,
            "proj_bias": proj_b,
        }


class PortableTokenMixer:
    """Canonical 2-layer MLP token-mixer (channel-mixer variant).

    Constructor: ``PortableTokenMixer(dim, hidden_dim, backend=...)``
    Forward: ``out = mixer(x)`` where ``x.shape = (B, N, dim)`` (token form)
    OR ``(B, dim, H, W)`` (NCHW conv form).

    Per FastViT paper: the token mixer alternates between spatial mixing
    (RepMixer per stage 1+2) and channel mixing (this MLP per stage 3+4).
    The channel-mixer is the canonical transformer MLP:

        out = Linear(GELU(Linear(x)))

    For NCHW input the primitive transparently routes through (B, N, C)
    by treating spatial positions as tokens; the FastViT token-mixer
    pattern uses (B, N, C) at all stages above stem so the auto-detect
    path is rarely exercised — but supporting NCHW keeps the primitive
    drop-in compatible with sister conv-stack consumers.

    Default ``hidden_dim`` follows transformer convention (4x dim).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        *,
        backend: Backend | str,
        seed: int | None = None,
    ) -> None:
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else 4 * self.dim
        self.backend = resolve_backend(backend)

        self._fc1 = PortableLinear(
            self.dim, self.hidden_dim, backend=self.backend, seed=seed
        )
        self._fc2 = PortableLinear(
            self.hidden_dim,
            self.dim,
            backend=self.backend,
            seed=(seed + 1 if seed is not None else None),
        )

    def __call__(self, x: Any) -> Any:
        # Auto-detect layout. Token form: 3D (B, N, C). NCHW: 4D (B, C, H, W).
        shape = tuple(x.shape)
        if len(shape) == 4:
            # NCHW -> permute to (B, H, W, C) -> reshape (B, H*W, C).
            B, C, H, W = shape
            if self.backend is Backend.MLX:
                import mlx.core as mx

                x_tok = mx.transpose(x, (0, 2, 3, 1))
                x_tok = mx.reshape(x_tok, (B, H * W, C))
                out = self._fc2(gelu(self._fc1(x_tok), backend=self.backend))
                out = mx.reshape(out, (B, H, W, C))
                return mx.transpose(out, (0, 3, 1, 2))

            import torch

            x_tok = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            out = self._fc2(gelu(self._fc1(x_tok), backend=self.backend))
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            return out

        # Token form (3D or higher with channels last): direct path.
        return self._fc2(gelu(self._fc1(x), backend=self.backend))

    def load_weights(
        self,
        fc1_w_np: np.ndarray,
        fc1_b_np: np.ndarray,
        fc2_w_np: np.ndarray,
        fc2_b_np: np.ndarray,
    ) -> None:
        """Inject canonical learned weights."""
        self._fc1.load_weights(fc1_w_np, fc1_b_np)
        self._fc2.load_weights(fc2_w_np, fc2_b_np)

    def export_weights(self) -> dict[str, np.ndarray]:
        """Export weights as numpy dict."""
        fc1_w, fc1_b = self._fc1.export_weights()
        fc2_w, fc2_b = self._fc2.export_weights()
        return {
            "fc1_weight": fc1_w,
            "fc1_bias": fc1_b,
            "fc2_weight": fc2_w,
            "fc2_bias": fc2_b,
        }


class PortableRepMixer:
    """Canonical Re-Parameterized Mixer block (FastViT stages 1+2 spatial mixer).

    Constructor: ``PortableRepMixer(dim, kernel_size, backend=...)``
    Forward: ``out = mixer(x)`` where ``x.shape = (B, dim, H, W)`` (NCHW).
    Output: same shape.

    Per FastViT paper (Vasu et al. 2023): the RepMixer is the spatial
    mixer used in stages 1+2. It uses **structural re-parameterization**:

    **Training mode** (3 branches; default):
        out = DW3x3(x) + DW1x1(x) + x  # 3 parallel branches summed

    **Inference mode** (1 branch; after re-parameterization):
        out = DW3x3_fused(x)  # weights of 3 branches absorbed into single 3x3 kernel

    The re-parameterization is mathematically equivalent: a 1x1 kernel can
    be zero-padded to 3x3, and the identity branch can be expressed as
    3x3 kernel that is identity in the center. After fusing, the inference
    path is a single 3x3 depthwise conv with NO performance penalty.

    Use :meth:`reparameterize` to fuse the 3 training branches into the
    inference path. Toggle with :meth:`train` / :meth:`eval`. In MLX
    contexts the eval form is faster (single conv); in training contexts
    keep the 3-branch form so gradients flow through each branch.

    Per CLAUDE.md HNeRV parity discipline L4 (≤100 LOC inflate budget):
    the RepMixer's inference path is a single DW conv per layer — that's
    the surface this primitive optimizes for.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        *,
        backend: Backend | str,
        seed: int | None = None,
    ) -> None:
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.backend = resolve_backend(backend)
        self._training = True

        # Branch 1: 3x3 depthwise conv.
        self._dw3 = PortableDepthwiseConv2d(
            self.dim, self.kernel_size, backend=self.backend, seed=seed, bias=True
        )
        # Branch 2: 1x1 depthwise conv.
        self._dw1 = PortableDepthwiseConv2d(
            self.dim, 1, backend=self.backend, seed=(seed + 1 if seed is not None else None), bias=True
        )
        # Branch 3: identity skip (no params).

        # Re-parameterized (fused) kernel; populated by reparameterize().
        self._fused_w: Any = None
        self._fused_b: Any = None
        self._is_fused = False

    def train(self) -> None:
        """Set training mode (use 3 branches; gradients flow through each)."""
        self._training = True

    def eval(self) -> None:
        """Set evaluation mode (use fused branch if available, else 3 branches)."""
        self._training = False

    def reparameterize(self) -> None:
        """Fuse 3 training branches into a single 3x3 depthwise conv for inference.

        Mathematically equivalent: ``DW3x3(x) + DW1x1(x) + x`` becomes
        ``DW3x3_fused(x)`` where the fused 3x3 kernel is:

            W_fused = W_3x3 + pad_to_3x3(W_1x1) + identity_3x3
            b_fused = b_3x3 + b_1x1

        After fusing, ``eval()`` uses the single fused conv (inference fast path).
        """
        w_3x3, b_3x3 = self._dw3.export_weights()  # (dim, 1, 3, 3), (dim,)
        w_1x1, b_1x1 = self._dw1.export_weights()  # (dim, 1, 1, 1), (dim,)

        # Pad 1x1 kernel to 3x3 (zero-pad on the border).
        pad = self.kernel_size // 2
        w_1x1_padded = np.zeros(
            (self.dim, 1, self.kernel_size, self.kernel_size), dtype=np.float32
        )
        w_1x1_padded[:, :, pad, pad] = w_1x1[:, :, 0, 0]

        # Identity 3x3 kernel (1 at center, 0 elsewhere; per-channel).
        w_identity = np.zeros(
            (self.dim, 1, self.kernel_size, self.kernel_size), dtype=np.float32
        )
        w_identity[:, :, pad, pad] = 1.0

        # Fuse all 3 branches.
        w_fused = w_3x3 + w_1x1_padded + w_identity
        b_fused = b_3x3.copy()
        if b_1x1 is not None:
            b_fused = b_fused + b_1x1

        # Replace the 3x3 conv weights with the fused weights; mark fused.
        self._dw3.load_weights(w_fused, b_fused)
        self._fused_w = w_fused
        self._fused_b = b_fused
        self._is_fused = True

    def __call__(self, x: Any) -> Any:
        if self._is_fused or not self._training:
            # Inference path: single fused DW conv (if reparameterize() was called)
            # OR 3 branches summed (if not).
            if self._is_fused:
                return self._dw3(x)
            # Eval-mode but not reparameterized: still use 3 branches.
            return self._forward_3_branch(x)
        # Training path: always 3 branches.
        return self._forward_3_branch(x)

    def _forward_3_branch(self, x: Any) -> Any:
        """3-branch sum: DW3x3(x) + DW1x1(x) + x."""
        return self._dw3(x) + self._dw1(x) + x

    def load_weights(
        self,
        dw3_w_np: np.ndarray,
        dw3_b_np: np.ndarray,
        dw1_w_np: np.ndarray,
        dw1_b_np: np.ndarray,
    ) -> None:
        """Inject canonical learned weights for training-mode 3 branches."""
        self._dw3.load_weights(dw3_w_np, dw3_b_np)
        self._dw1.load_weights(dw1_w_np, dw1_b_np)
        # Clear any cached fused weights (need to re-call reparameterize()).
        self._fused_w = None
        self._fused_b = None
        self._is_fused = False

    def export_weights(self) -> dict[str, np.ndarray]:
        """Export weights as numpy dict.

        If reparameterized, returns ``fused_weight`` + ``fused_bias`` only.
        Otherwise returns all 4 keys (``dw3_weight`` / ``dw3_bias`` /
        ``dw1_weight`` / ``dw1_bias``).
        """
        if self._is_fused:
            return {
                "fused_weight": self._fused_w,
                "fused_bias": self._fused_b,
            }
        dw3_w, dw3_b = self._dw3.export_weights()
        dw1_w, dw1_b = self._dw1.export_weights()
        return {
            "dw3_weight": dw3_w,
            "dw3_bias": dw3_b,
            "dw1_weight": dw1_w,
            "dw1_bias": dw1_b,
        }
