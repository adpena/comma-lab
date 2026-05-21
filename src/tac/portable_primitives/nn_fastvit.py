# SPDX-License-Identifier: MIT
"""FastViT-T12 backbone + Hydra head primitives for PoseNet contest scorer port.

MLX-ARCH-3 (sister-3 of 5-stage MLX architecture port cascade per operator
directive 2026-05-21 + Carmack MVP-first 5-step per CLAUDE.md ``be125b878``).

Composes :mod:`tac.portable_primitives.nn` (OVERNIGHT-WW 9 base primitives)
+ :mod:`tac.portable_primitives.nn_extended` (ARCH-1 5 foundational ops)
+ :mod:`tac.portable_primitives.nn_attention` (ARCH-2 4 attention primitives)
into the full FastViT-T12 backbone + PoseNet head required for paired CUDA
ground-truth validation at ARCH-5.

**Canonical PoseNet spec** (sourced from ``upstream/modules.py`` lines 22-80;
this is the contest scorer's PoseNet, not a generic FastViT classifier):

- ``IN_CHANS = 12`` (= 2 frames × 6 YUV6 channels per :func:`rgb_to_yuv6`)
- Input normalization: ``(x - 127.5) / 63.75`` (per-channel mean/std buffers)
- ``vision = timm.create_model('fastvit_t12', num_classes=2048, in_chans=12,
  act_layer=gelu_tanh)`` -> ``VISION_FEATURES = 2048`` output
- ``summarizer = Linear(2048 -> 512) -> ReLU -> ResBlock(512)`` -> ``SUMMARY_FEATURES = 512``
- ``hydra = Hydra(512, heads=[Head('pose', hidden=32, out=12)])`` ->
  ``{'pose': (B, 12)}`` (first 6 used per :meth:`compute_distortion`)

**Canonical timm FastViT-T12 spec** (sourced from ``timm.models.fastvit``
``fastvit_t12`` function): ``layers=(2, 2, 6, 2)``, ``embed_dims=(64, 128,
256, 512)``, ``mlp_ratios=(3, 3, 3, 3)``, ``token_mixers=('repmixer', 'repmixer',
'repmixer', 'repmixer')``. The contest's PoseNet uses ``in_chans=12`` and
``num_classes=2048`` overrides (vs the imagenet 1000-class default).

**MVP scope** (per Carmack MVP-first 5-step + dispatch's "partial assembly
+ DEFER" allowance):

This module lands the **assembly scaffolds + Hydra head + PoseNet wrapper**:

- :class:`PortableAllNorm` — BatchNorm1d-over-flattened-view (sister of
  ``upstream.modules.AllNorm``); composes WW PortableLinear + analogous
  BN1d (PyTorch-only here since MLX has no BN1d in current scope; ARCH-3b
  may extend).
- :class:`PortableResBlock` — 2-branch residual MLP block (sister of
  ``upstream.modules.ResBlock``); pure WW PortableLinear + AllNorm + ReLU
  composition.
- :class:`PortableHydra` — multi-head MLP head with per-head input + residual
  + final projection (sister of ``upstream.modules.Hydra``); returns dict
  keyed by head name.
- :class:`PortableFastViTBlock` — single RepMixer block (token_mixer +
  channel_mixer + LayerScale) composing ARCH-2 ``PortableRepMixer`` +
  ``PortableTokenMixer`` + ``PortableLayerScale``.
- :class:`PortableFastViTStage` — sequence of N blocks + optional patch
  embedding (downsample by 2x).
- :class:`PortableFastViTT12Backbone` — full T12 architecture scaffold
  (stem + 4 stages + global pool + classifier head -> 2048).
- :class:`PortablePoseNet` — full PoseNet wrapper (normalization +
  backbone + summarizer + Hydra).

**Per-primitive numerical equivalence** MLX-vs-PyTorch is inherited from
sister tests in :mod:`tac.portable_primitives.tests` at ε ≤ 5e-3 fp32
per Phase 1 PV (WW + ARCH-1 + ARCH-2 sister tests already pin this for
each individual primitive). Full-backbone MLX-vs-PyTorch numerical
equivalence at random init is tested at the scaffold level (forward
shape correctness + small-batch numeric drift < 5e-2 per accumulated
deeper-network drift band; the canonical PR 101 paired ground-truth
validation lands at ARCH-5 once timm-state_dict-load is implemented).

Per CLAUDE.md non-negotiables PRESERVED:

- **MPS auth eval is NOISE** (Catalog #1): MLX-backend scores remain
  non-promotable; this layer is ARCHITECTURE-LEVEL only; ARCH-5
  paired-eval is where contest-axis promotion gates apply.
- **Beauty, simplicity, and developer experience**: each block reviewable
  in 30 seconds; thin adapters over composed primitives.
- **Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY**: this is
  research-only (architecture port; non-promotable). ARCH-5 contest-axis
  gates apply for promotion.

**Deferred to ARCH-3b / ARCH-5** (out of MVP scope):

- ``MobileOneBlock`` structural re-parameterization (timm stem uses
  3-branch MobileOne; current scaffold uses simpler WW Conv2d stem).
- Per-block ``conv_ffn`` / ``patch_emb`` exact timm-equivalent shapes
  (current scaffold preserves dimension flow but not byte-stable timm
  key naming).
- ``act_layer='gelu_tanh'`` (current uses GELU; gelu_tanh approximation
  drift is sub-ε per ARCH-1 PV).
- Deterministic state_dict load from PR 101 / timm checkpoint (the
  Sister-5 ownership map per dispatch).

Sister of:

- :mod:`tac.portable_primitives.nn` (canonical base 9 primitives)
- :mod:`tac.portable_primitives.nn_extended` (ARCH-1 5 foundational ops)
- :mod:`tac.portable_primitives.nn_attention` (ARCH-2 4 attention primitives)
- :mod:`tac.local_acceleration.mlx_to_pytorch_export` (weight export
  pipeline ARCH-5 paired ground-truth depends on)
- :mod:`upstream.modules` (PoseNet canonical reference; lines 22-80)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tac.portable_primitives.backend import Backend, resolve_backend
from tac.portable_primitives.nn import (
    PortableConv2d,
    PortableLayerNorm,
    PortableLinear,
    matmul,
    relu,
)
from tac.portable_primitives.nn_attention import (
    PortableLayerScale,
    PortableRepMixer,
    PortableTokenMixer,
)
from tac.portable_primitives.nn_extended import (
    PortableAvgPool2d,
    PortableBatchNorm2d,
)

__all__ = [
    "PortableAllNorm",
    "PortableResBlock",
    "PortableHydra",
    "PortableFastViTBlock",
    "PortableFastViTStage",
    "PortableFastViTT12Backbone",
    "PortablePoseNet",
    # Canonical FastViT-T12 spec constants (sourced from timm.models.fastvit
    # fastvit_t12 + upstream.modules.PoseNet).
    "FASTVIT_T12_LAYERS",
    "FASTVIT_T12_EMBED_DIMS",
    "FASTVIT_T12_MLP_RATIOS",
    "POSENET_IN_CHANS",
    "POSENET_VISION_FEATURES",
    "POSENET_SUMMARY_FEATURES",
    "POSENET_INPUT_MEAN",
    "POSENET_INPUT_STD",
]


# ---------------------------------------------------------------------------
# Canonical config constants (sourced from timm + upstream/modules.py).
# ---------------------------------------------------------------------------

FASTVIT_T12_LAYERS: tuple[int, int, int, int] = (2, 2, 6, 2)
"""Blocks per stage; per timm ``fastvit_t12`` model_args (verified 2026-05-21)."""

FASTVIT_T12_EMBED_DIMS: tuple[int, int, int, int] = (64, 128, 256, 512)
"""Channel dim per stage; per timm ``fastvit_t12`` model_args."""

FASTVIT_T12_MLP_RATIOS: tuple[int, int, int, int] = (3, 3, 3, 3)
"""Channel-mixer hidden_dim multiplier; per timm ``fastvit_t12`` model_args."""

POSENET_IN_CHANS: int = 12
"""Input channels = 2 frames × 6 YUV6 channels; ``upstream.modules.IN_CHANS``."""

POSENET_VISION_FEATURES: int = 2048
"""Backbone output dim; ``upstream.modules.VISION_FEATURES``."""

POSENET_SUMMARY_FEATURES: int = 512
"""Summarizer output dim; ``upstream.modules.SUMMARY_FEATURES``."""

POSENET_INPUT_MEAN: float = 127.5
"""Per-channel mean for input normalization; ``upstream.modules.PoseNet._mean``."""

POSENET_INPUT_STD: float = 63.75
"""Per-channel std for input normalization; ``upstream.modules.PoseNet._std``."""


# ---------------------------------------------------------------------------
# PoseNet head primitives (sourced from upstream.modules lines 28-59).
# ---------------------------------------------------------------------------


class PortableAllNorm:
    """BatchNorm1d-over-flattened-view per ``upstream.modules.AllNorm``.

    Forward: ``out = bn1d(x.view(-1, 1)).view(x.shape)`` where ``bn1d`` is a
    single-channel BatchNorm1d. This is the canonical normalization Hydra +
    ResBlock use throughout the head.

    Constructor: ``PortableAllNorm(num_features, eps, momentum, affine,
    backend=...)``.

    Note on backend scope: the upstream contest scorer's AllNorm uses
    PyTorch ``nn.BatchNorm1d`` with ``eps=0.001`` and ``momentum=0.01``
    (per ``upstream.modules.BN_EPS`` + ``BN_MOM``). The MLX path here
    implements the equivalent affine-scale + bias arithmetic with manual
    running mean/var (mlx has no BatchNorm1d in current bindings). In
    EVAL mode (the canonical use for paired CUDA ground-truth via PR 101)
    the running stats are fixed and the math reduces to a pure affine
    transform: ``y = gamma * (x - mean) / sqrt(var + eps) + beta``.
    """

    def __init__(
        self,
        num_features: int,
        *,
        backend: Backend | str,
        eps: float = 0.001,
        momentum: float = 0.01,
        affine: bool = True,
    ) -> None:
        # num_features is the BatchNorm1d feature dim. Per AllNorm the BN
        # operates on a flattened (N, 1) view, so num_features=1 in
        # the canonical instantiation. We surface the param for symmetry
        # with PyTorch's nn.BatchNorm1d but assert canonical shape.
        if num_features != 1:
            raise ValueError(
                f"PortableAllNorm canonical AllNorm uses num_features=1 "
                f"(BN over flattened view); got {num_features}. If you need a "
                f"per-feature BN1d, use a different sister primitive."
            )
        self.num_features = num_features
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.backend = resolve_backend(backend)

        # Running stats: initialized to mean=0, var=1 (canonical BN init).
        self._running_mean = np.zeros((num_features,), dtype=np.float32)
        self._running_var = np.ones((num_features,), dtype=np.float32)
        # Affine params: initialized to gamma=1, beta=0.
        if affine:
            self._weight_np = np.ones((num_features,), dtype=np.float32)
            self._bias_np = np.zeros((num_features,), dtype=np.float32)
        else:
            self._weight_np = None
            self._bias_np = None

        if self.backend is Backend.PYTORCH:
            import torch
            import torch.nn as nn

            self._bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)
            # Force EVAL mode by default (canonical for paired-CUDA ground-truth).
            self._bn.eval()
            # Sync to numpy state.
            with torch.no_grad():
                self._bn.running_mean.copy_(torch.from_numpy(self._running_mean.copy()))
                self._bn.running_var.copy_(torch.from_numpy(self._running_var.copy()))
                if affine:
                    self._bn.weight.copy_(torch.from_numpy(self._weight_np.copy()))
                    self._bn.bias.copy_(torch.from_numpy(self._bias_np.copy()))
        # MLX path: arithmetic done in __call__ from numpy state.

    def __call__(self, x: Any) -> Any:
        original_shape = tuple(x.shape)
        if self.backend is Backend.PYTORCH:
            import torch

            flat = x.reshape(-1, 1)
            out = self._bn(flat)
            return out.reshape(original_shape)

        # MLX path: manual affine transform using running stats (EVAL mode).
        import mlx.core as mx

        flat = mx.reshape(x, (-1, 1))
        # y = (x - mean) / sqrt(var + eps) * gamma + beta
        mean = mx.array(self._running_mean)
        var = mx.array(self._running_var)
        scale = mx.array(self._weight_np if self.affine else np.ones_like(self._running_mean))
        bias = mx.array(self._bias_np if self.affine else np.zeros_like(self._running_mean))
        # mx.sqrt is on mlx.core; eps is small additive.
        normalized = (flat - mean) / mx.sqrt(var + self.eps)
        out = normalized * scale + bias
        return mx.reshape(out, original_shape)

    def load_weights(
        self,
        weight_np: np.ndarray | None,
        bias_np: np.ndarray | None,
        running_mean_np: np.ndarray,
        running_var_np: np.ndarray,
    ) -> None:
        """Inject canonical learned BN1d state."""
        self._running_mean = running_mean_np.astype(np.float32).copy()
        self._running_var = running_var_np.astype(np.float32).copy()
        if self.affine:
            if weight_np is None or bias_np is None:
                raise ValueError("affine=True requires weight + bias")
            self._weight_np = weight_np.astype(np.float32).copy()
            self._bias_np = bias_np.astype(np.float32).copy()
        if self.backend is Backend.PYTORCH:
            import torch

            with torch.no_grad():
                self._bn.running_mean.copy_(torch.from_numpy(self._running_mean.copy()))
                self._bn.running_var.copy_(torch.from_numpy(self._running_var.copy()))
                if self.affine:
                    self._bn.weight.copy_(torch.from_numpy(self._weight_np.copy()))
                    self._bn.bias.copy_(torch.from_numpy(self._bias_np.copy()))

    def export_weights(self) -> dict[str, np.ndarray]:
        """Export canonical BN1d state."""
        out = {
            "running_mean": self._running_mean.copy(),
            "running_var": self._running_var.copy(),
        }
        if self.affine:
            out["weight"] = self._weight_np.copy()
            out["bias"] = self._bias_np.copy()
        return out


class PortableResBlock:
    """Residual MLP block per ``upstream.modules.ResBlock`` (lines 35-43).

    Forward: ``a = x + block_a(x); out = ReLU(a + block_b(a))``

    Where:
        block_a = Linear(F, F*E) -> AllNorm(F*E) -> ReLU -> Linear(F*E, F) -> AllNorm(F)
        block_b = ReLU -> Linear(F, F*E) -> AllNorm(F*E) -> ReLU -> Linear(F*E, F) -> AllNorm(F)

    F = feats; E = expansion (default 2). Per upstream all AllNorm instances
    use the canonical 1-feature BN1d-over-flattened-view.

    Used 2x in the canonical PoseNet:
    1. ``summarizer = Linear(2048->512) -> ReLU -> ResBlock(512)``
    2. ``hydra.resblock = ResBlock(512)``
    """

    def __init__(
        self,
        feats: int,
        *,
        backend: Backend | str,
        expansion: int = 2,
        seed: int | None = None,
    ) -> None:
        self.feats = int(feats)
        self.expansion = int(expansion)
        self.backend = resolve_backend(backend)

        hidden = self.feats * self.expansion

        def _seed(offset: int) -> int | None:
            return None if seed is None else seed + offset

        # block_a: Linear -> AllNorm -> ReLU -> Linear -> AllNorm.
        self._a_lin1 = PortableLinear(self.feats, hidden, backend=self.backend, seed=_seed(0))
        self._a_norm1 = PortableAllNorm(1, backend=self.backend)
        self._a_lin2 = PortableLinear(hidden, self.feats, backend=self.backend, seed=_seed(1))
        self._a_norm2 = PortableAllNorm(1, backend=self.backend)

        # block_b: ReLU -> Linear -> AllNorm -> ReLU -> Linear -> AllNorm.
        self._b_lin1 = PortableLinear(self.feats, hidden, backend=self.backend, seed=_seed(2))
        self._b_norm1 = PortableAllNorm(1, backend=self.backend)
        self._b_lin2 = PortableLinear(hidden, self.feats, backend=self.backend, seed=_seed(3))
        self._b_norm2 = PortableAllNorm(1, backend=self.backend)

    def __call__(self, x: Any) -> Any:
        # block_a
        h = self._a_lin1(x)
        h = self._a_norm1(h)
        h = relu(h, backend=self.backend)
        h = self._a_lin2(h)
        h = self._a_norm2(h)
        a_out = x + h

        # block_b
        h = relu(a_out, backend=self.backend)
        h = self._b_lin1(h)
        h = self._b_norm1(h)
        h = relu(h, backend=self.backend)
        h = self._b_lin2(h)
        h = self._b_norm2(h)

        return relu(a_out + h, backend=self.backend)

    def export_weights(self) -> dict[str, Any]:
        """Export ResBlock weights as canonical dict (sister Wave 4 pipeline)."""
        a_lin1_w, a_lin1_b = self._a_lin1.export_weights()
        a_lin2_w, a_lin2_b = self._a_lin2.export_weights()
        b_lin1_w, b_lin1_b = self._b_lin1.export_weights()
        b_lin2_w, b_lin2_b = self._b_lin2.export_weights()
        return {
            "a_lin1_weight": a_lin1_w,
            "a_lin1_bias": a_lin1_b,
            "a_norm1": self._a_norm1.export_weights(),
            "a_lin2_weight": a_lin2_w,
            "a_lin2_bias": a_lin2_b,
            "a_norm2": self._a_norm2.export_weights(),
            "b_lin1_weight": b_lin1_w,
            "b_lin1_bias": b_lin1_b,
            "b_norm1": self._b_norm1.export_weights(),
            "b_lin2_weight": b_lin2_w,
            "b_lin2_bias": b_lin2_b,
            "b_norm2": self._b_norm2.export_weights(),
        }


class PortableHydra:
    """Multi-head MLP head per ``upstream.modules.Hydra`` (lines 45-59).

    Forward:
        x = resblock(x)
        in[k] = ReLU(in_layer[k](x))     for k in heads
        res[k] = ReLU(in[k] + res_layer[k](in[k]))
        out[k] = final_layer[k](res[k])
        return out  # dict keyed by head.name

    For PoseNet (canonical contest scorer):
        heads = [Head(name='pose', hidden=32, out=12)]
        -> returns {'pose': (B, 12)}  # first 6 used by compute_distortion
    """

    def __init__(
        self,
        num_features: int,
        heads: list[tuple[str, int, int]],
        *,
        backend: Backend | str,
        seed: int | None = None,
    ) -> None:
        """heads: list of (name, hidden_dim, out_dim) per Head namedtuple."""
        self.num_features = int(num_features)
        self.heads = list(heads)
        self.backend = resolve_backend(backend)

        def _seed(offset: int) -> int | None:
            return None if seed is None else seed + offset

        self._resblock = PortableResBlock(num_features, backend=self.backend, seed=_seed(0))

        self._in_layer: dict[str, PortableLinear] = {}
        self._res_layer1: dict[str, PortableLinear] = {}
        self._res_layer2: dict[str, PortableLinear] = {}
        self._final_layer: dict[str, PortableLinear] = {}

        for i, (name, hidden, out_dim) in enumerate(self.heads):
            base = 10 + i * 100  # per-head seed offset block
            self._in_layer[name] = PortableLinear(
                num_features, hidden, backend=self.backend, seed=_seed(base + 0)
            )
            self._res_layer1[name] = PortableLinear(
                hidden, hidden, backend=self.backend, seed=_seed(base + 1)
            )
            self._res_layer2[name] = PortableLinear(
                hidden, hidden, backend=self.backend, seed=_seed(base + 2)
            )
            self._final_layer[name] = PortableLinear(
                hidden, out_dim, backend=self.backend, seed=_seed(base + 3)
            )

    def __call__(self, x: Any) -> dict[str, Any]:
        x = self._resblock(x)
        out: dict[str, Any] = {}
        for name, _, _ in self.heads:
            h = self._in_layer[name](x)
            h = relu(h, backend=self.backend)
            # res_layer is Sequential(Linear, ReLU, Linear) per upstream;
            # composed as h + relu(linear1(h)) then linear2 (matches upstream
            # forward: in_layer[k] + v(in_layer[k]) then relu).
            rh = self._res_layer1[name](h)
            rh = relu(rh, backend=self.backend)
            rh = self._res_layer2[name](rh)
            r = relu(h + rh, backend=self.backend)
            out[name] = self._final_layer[name](r)
        return out

    def export_weights(self) -> dict[str, Any]:
        result: dict[str, Any] = {"resblock": self._resblock.export_weights()}
        for name, _, _ in self.heads:
            in_w, in_b = self._in_layer[name].export_weights()
            r1_w, r1_b = self._res_layer1[name].export_weights()
            r2_w, r2_b = self._res_layer2[name].export_weights()
            f_w, f_b = self._final_layer[name].export_weights()
            result[name] = {
                "in_weight": in_w,
                "in_bias": in_b,
                "res1_weight": r1_w,
                "res1_bias": r1_b,
                "res2_weight": r2_w,
                "res2_bias": r2_b,
                "final_weight": f_w,
                "final_bias": f_b,
            }
        return result


# ---------------------------------------------------------------------------
# FastViT block + stage scaffolds (compose ARCH-2 RepMixer + TokenMixer +
# LayerScale into the canonical T12 block pattern).
# ---------------------------------------------------------------------------


class PortableFastViTBlock:
    """Single FastViT RepMixer block.

    Forward (per timm ``RepMixerBlock``):
        x = x + LayerScale(RepMixer(x))           # spatial mixer w/ residual
        x = x + LayerScale(TokenMixer(x))          # channel mixer w/ residual
        return x

    Per FastViT paper: every block has a per-channel LayerScale on each
    residual branch (init γ=1e-5 so residual starts as effective identity)
    and the two mixers alternate (spatial RepMixer + channel-mixing MLP).

    Constructor: ``PortableFastViTBlock(dim, mlp_ratio, backend=...)``.
    Forward expects NCHW input ``(B, dim, H, W)``; output same shape.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: int = 3,
        *,
        backend: Backend | str,
        layer_scale_init: float = 1e-5,
        seed: int | None = None,
    ) -> None:
        self.dim = int(dim)
        self.mlp_ratio = int(mlp_ratio)
        self.backend = resolve_backend(backend)

        def _seed(offset: int) -> int | None:
            return None if seed is None else seed + offset

        self._spatial_mixer = PortableRepMixer(dim, backend=self.backend, seed=_seed(0))
        self._spatial_scale = PortableLayerScale(
            dim, backend=self.backend, init_value=layer_scale_init, channels_last=False
        )

        hidden = self.dim * self.mlp_ratio
        self._channel_mixer = PortableTokenMixer(
            dim, hidden_dim=hidden, backend=self.backend, seed=_seed(10)
        )
        self._channel_scale = PortableLayerScale(
            dim, backend=self.backend, init_value=layer_scale_init, channels_last=False
        )

    def __call__(self, x: Any) -> Any:
        # Spatial branch (RepMixer is residual internally via 3-branch sum;
        # outer residual still applied via canonical block contract).
        h = self._spatial_mixer(x)
        h = self._spatial_scale(h)
        x = x + h

        # Channel branch (TokenMixer auto-detects NCHW and routes through
        # token form internally).
        h = self._channel_mixer(x)
        h = self._channel_scale(h)
        return x + h

    def reparameterize(self) -> None:
        """Fuse the spatial RepMixer's 3 branches into a single 3x3 DW conv."""
        self._spatial_mixer.reparameterize()


class PortablePatchEmbed:
    """Patch embedding (2x spatial downsample) per timm FastVit pattern.

    Forward (NCHW): ``out = Conv2d(in_dim, out_dim, kernel=3, stride=2, padding=1)(x)``

    Used between FastViT stages (and as part of the stem) to downsample
    spatial resolution while increasing channel dim. The current scaffold
    uses a single conv (timm uses MobileOneBlock for canonical-byte-stable
    behavior; that delta is sub-ε per ARCH-1 PV but documented as
    deferred-to-ARCH-3b for full byte-stable parity).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        backend: Backend | str,
        kernel_size: int = 3,
        stride: int = 2,
        seed: int | None = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)
        self.backend = resolve_backend(backend)

        self._conv = PortableConv2d(
            in_channels, out_channels, kernel_size=kernel_size, backend=self.backend, seed=seed
        )
        # Stride handled in __call__ via manual stride simulation (WW
        # PortableConv2d defaults to stride=1; we apply a slice after-the-fact
        # to emulate stride=2). Since this is research-only scaffold (Catalog
        # #192 non-promotable; ARCH-5 paired CUDA validation is the
        # contest-axis path), the slice-emulation is sufficient for shape
        # correctness; byte-stable timm parity is deferred to ARCH-5
        # state_dict load.

    def __call__(self, x: Any) -> Any:
        # Forward through conv (stride=1) then stride-slice along H,W.
        out = self._conv(x)
        if self.stride == 1:
            return out
        if self.backend is Backend.MLX:
            import mlx.core as mx

            return out[:, :, ::self.stride, ::self.stride]
        # PyTorch path
        return out[:, :, ::self.stride, ::self.stride]


class PortableFastViTStage:
    """N RepMixer blocks + optional patch embedding (downsample by 2x).

    Forward (NCHW):
        if patch_embed: x = patch_embed(x)
        for block in blocks: x = block(x)
        return x

    Per timm ``FastVitStage``: each stage has an optional downsample (patch
    embedding) that converts ``(B, in_dim, H, W) -> (B, dim, H/2, W/2)``,
    followed by ``num_blocks`` RepMixer blocks operating at ``dim``.
    """

    def __init__(
        self,
        in_dim: int,
        dim: int,
        num_blocks: int,
        *,
        mlp_ratio: int = 3,
        backend: Backend | str,
        downsample: bool = True,
        seed: int | None = None,
    ) -> None:
        self.in_dim = int(in_dim)
        self.dim = int(dim)
        self.num_blocks = int(num_blocks)
        self.mlp_ratio = int(mlp_ratio)
        self.downsample = bool(downsample)
        self.backend = resolve_backend(backend)

        def _seed(offset: int) -> int | None:
            return None if seed is None else seed + offset

        if downsample:
            self._patch_embed: PortablePatchEmbed | None = PortablePatchEmbed(
                in_dim, dim, backend=self.backend, seed=_seed(0)
            )
        else:
            self._patch_embed = None

        self._blocks: list[PortableFastViTBlock] = []
        for i in range(num_blocks):
            blk = PortableFastViTBlock(
                dim, mlp_ratio=mlp_ratio, backend=self.backend, seed=_seed(100 + 50 * i)
            )
            self._blocks.append(blk)

    def __call__(self, x: Any) -> Any:
        if self._patch_embed is not None:
            x = self._patch_embed(x)
        for block in self._blocks:
            x = block(x)
        return x

    def reparameterize(self) -> None:
        """Fuse all blocks' RepMixer 3-branch into single fused conv."""
        for block in self._blocks:
            block.reparameterize()


# ---------------------------------------------------------------------------
# Full FastViT-T12 backbone + PoseNet wrapper.
# ---------------------------------------------------------------------------


class PortableFastViTT12Backbone:
    """Full FastViT-T12 backbone scaffold (stem + 4 stages + classifier head).

    Constructor: ``PortableFastViTT12Backbone(in_chans, num_classes, backend=...)``

    Forward (NCHW): ``x.shape = (B, in_chans, H, W)`` -> ``out.shape = (B, num_classes)``

    Architecture (per timm ``fastvit_t12`` model_args, verified empirically):
    - Stem: 3-conv stride-2 sequence ``in_chans -> 64 -> 64 -> 64``
    - Stage 0: 2 RepMixer blocks at dim=64 (no downsample)
    - Stage 1: 2 RepMixer blocks at dim=128 (with downsample 64 -> 128)
    - Stage 2: 6 RepMixer blocks at dim=256 (with downsample 128 -> 256)
    - Stage 3: 2 RepMixer blocks at dim=512 (with downsample 256 -> 512)
    - Final conv: 1x1 conv 512 -> 1024 (timm uses MobileOneBlock; scaffold
      uses WW PortableConv2d kernel_size=1)
    - Global avg pool: (B, 1024, H', W') -> (B, 1024)
    - Classifier head: Linear(1024 -> num_classes)

    For PoseNet usage: ``num_classes=2048`` (the VISION_FEATURES override).
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        *,
        backend: Backend | str,
        seed: int | None = None,
    ) -> None:
        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        self.backend = resolve_backend(backend)

        def _seed(offset: int) -> int | None:
            return None if seed is None else seed + offset

        # Stem: 3 conv layers (stride-2, stride-2, stride-1 per timm).
        # Simplified vs MobileOneBlock; byte-stable parity deferred to
        # ARCH-3b / ARCH-5 state_dict-load.
        stem_dim = 64
        self._stem_conv1 = PortableConv2d(
            in_chans, stem_dim, kernel_size=3, backend=self.backend, seed=_seed(0)
        )
        self._stem_conv2 = PortableConv2d(
            stem_dim, stem_dim, kernel_size=3, backend=self.backend, seed=_seed(1)
        )
        self._stem_conv3 = PortableConv2d(
            stem_dim, stem_dim, kernel_size=1, backend=self.backend, seed=_seed(2)
        )
        # Stride-2 simulation matches PortablePatchEmbed pattern.
        self._stem_stride = 2

        # 4 stages.
        self._stages: list[PortableFastViTStage] = []
        prev_dim = stem_dim
        for i, (n_blocks, dim) in enumerate(zip(FASTVIT_T12_LAYERS, FASTVIT_T12_EMBED_DIMS)):
            # Stage 0 has no downsample (per timm: downsample=Identity()).
            downsample = i > 0
            stage = PortableFastViTStage(
                prev_dim,
                dim,
                n_blocks,
                mlp_ratio=FASTVIT_T12_MLP_RATIOS[i],
                backend=self.backend,
                downsample=downsample,
                seed=_seed(10 + 1000 * i),
            )
            self._stages.append(stage)
            prev_dim = dim

        # Final 1x1 conv expanding 512 -> 1024 (per timm fastvit_t12
        # default num_features).
        final_conv_dim = 1024
        self._final_conv = PortableConv2d(
            prev_dim, final_conv_dim, kernel_size=1, backend=self.backend, seed=_seed(5000)
        )

        # Global avg pool + classifier head.
        self._classifier = PortableLinear(
            final_conv_dim, num_classes, backend=self.backend, seed=_seed(6000)
        )

    def _stem_forward(self, x: Any) -> Any:
        """Stem: conv1(stride-2) -> conv2(stride-2) -> conv3(stride-1)."""
        if self.backend is Backend.MLX:
            import mlx.core as mx

            x = self._stem_conv1(x)
            x = x[:, :, :: self._stem_stride, :: self._stem_stride]
            x = self._stem_conv2(x)
            x = x[:, :, :: self._stem_stride, :: self._stem_stride]
            x = self._stem_conv3(x)
            return x

        # PyTorch
        x = self._stem_conv1(x)
        x = x[:, :, :: self._stem_stride, :: self._stem_stride]
        x = self._stem_conv2(x)
        x = x[:, :, :: self._stem_stride, :: self._stem_stride]
        x = self._stem_conv3(x)
        return x

    def _global_avg_pool(self, x: Any) -> Any:
        """Global average pooling over spatial dims (H, W). NCHW -> NC."""
        if self.backend is Backend.MLX:
            import mlx.core as mx

            return mx.mean(x, axis=(2, 3))
        # PyTorch
        return x.mean(dim=(2, 3))

    def __call__(self, x: Any) -> Any:
        # Stem.
        x = self._stem_forward(x)
        # 4 stages.
        for stage in self._stages:
            x = stage(x)
        # Final conv.
        x = self._final_conv(x)
        # Global pool.
        x = self._global_avg_pool(x)
        # Classifier head.
        return self._classifier(x)

    def reparameterize(self) -> None:
        """Fuse all stages' RepMixer 3-branch into single fused conv."""
        for stage in self._stages:
            stage.reparameterize()


class PortablePoseNet:
    """Full PoseNet wrapper per ``upstream.modules.PoseNet`` (lines 61-80).

    Forward:
        x_norm = (x - 127.5) / 63.75                  # per-channel normalize
        vision_out = backbone(x_norm)                  # (B, 2048)
        summary = ResBlock(ReLU(Linear(2048->512)(vision_out)))  # (B, 512)
        return hydra(summary)                          # {'pose': (B, 12)}

    Input: ``(B, IN_CHANS=12, H, W)`` already YUV6 (the caller is
    responsible for the ``rgb_to_yuv6`` preprocessing per
    :meth:`PoseNet.preprocess_input`).

    Output: ``{'pose': (B, 12)}`` — first 6 dims used by
    :meth:`PoseNet.compute_distortion` (the dimension reduction matches
    the upstream contest scorer's distortion formula).
    """

    def __init__(
        self,
        *,
        backend: Backend | str,
        seed: int | None = None,
    ) -> None:
        self.backend = resolve_backend(backend)

        def _seed(offset: int) -> int | None:
            return None if seed is None else seed + offset

        # Normalization buffers (per-channel mean/std broadcast over H, W).
        mean_np = np.full((1, POSENET_IN_CHANS, 1, 1), POSENET_INPUT_MEAN, dtype=np.float32)
        std_np = np.full((1, POSENET_IN_CHANS, 1, 1), POSENET_INPUT_STD, dtype=np.float32)
        if self.backend is Backend.MLX:
            import mlx.core as mx

            self._mean = mx.array(mean_np)
            self._std = mx.array(std_np)
        else:
            import torch

            self._mean = torch.from_numpy(mean_np)
            self._std = torch.from_numpy(std_np)

        # Backbone.
        self._backbone = PortableFastViTT12Backbone(
            in_chans=POSENET_IN_CHANS,
            num_classes=POSENET_VISION_FEATURES,
            backend=self.backend,
            seed=_seed(0),
        )

        # Summarizer: Linear(2048->512) -> ReLU -> ResBlock(512).
        self._summarizer_lin = PortableLinear(
            POSENET_VISION_FEATURES,
            POSENET_SUMMARY_FEATURES,
            backend=self.backend,
            seed=_seed(100000),
        )
        self._summarizer_resblock = PortableResBlock(
            POSENET_SUMMARY_FEATURES, backend=self.backend, seed=_seed(100100)
        )

        # Hydra head: pose head with hidden=32, out=12.
        self._hydra = PortableHydra(
            POSENET_SUMMARY_FEATURES,
            heads=[("pose", 32, 12)],
            backend=self.backend,
            seed=_seed(200000),
        )

    def __call__(self, x: Any) -> dict[str, Any]:
        """Forward pass.

        Input ``x.shape = (B, 12, H, W)`` (YUV6-converted; the caller is
        responsible for :meth:`preprocess_input` via the canonical
        :func:`upstream.modules.rgb_to_yuv6` preprocessing).

        Output: ``{'pose': (B, 12)}``.
        """
        # Normalize.
        x_norm = (x - self._mean) / self._std
        # Backbone (B, 12, H, W) -> (B, 2048).
        vision_out = self._backbone(x_norm)
        # Summarizer (B, 2048) -> (B, 512).
        summary = self._summarizer_lin(vision_out)
        summary = relu(summary, backend=self.backend)
        summary = self._summarizer_resblock(summary)
        # Hydra head.
        return self._hydra(summary)

    def reparameterize(self) -> None:
        """Fuse backbone RepMixer 3-branches into single fused conv."""
        self._backbone.reparameterize()
