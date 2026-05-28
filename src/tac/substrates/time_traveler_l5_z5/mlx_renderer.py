# SPDX-License-Identifier: MIT
"""Z5 Rao-Ballard MLX renderer — L1 LONG-RUN MLX-LOCAL 2026-05-28.

Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" + the
8th MLX-FIRST standing directive REINFORCED 2026-05-27 ("always prefer MLX
first always") + T4 SYMPOSIUM Wave N+13 verdict ``f5d3c6835`` op-routable #1
(Z5-first among Z4/Z5/Z6/Z7/Z8 class-shift queue).

This module is the MLX-native sister of
:class:`tac.substrates.time_traveler_l5_z5.architecture.Z5RaoBallardSubstrate`.
It mirrors the canonical Z6-v2 MLX renderer pattern at
:mod:`tac.substrates.z6_v2_cargo_cult_unwind.mlx_renderer` and the Z7-Mamba-2
mlx_module pattern at
:mod:`tac.substrates.time_traveler_l5_z7_mamba2.mlx_module`.

Z5 distinguishing primitive (per Catalog #272)
----------------------------------------------

Rao-Ballard 1999 EXPLICIT 2-level hierarchical predictive coding:

1. **Level-0 (low-level)**: per-pair latent ``z_low_t`` is the direct decoder
   input (Z6-style FiLM-free PixelShuffle decoder per architecture.py).
2. **Level-1 (high-level)**: meta-latent ``z_high_t`` + ``ego_motion_t``
   feed the canonical 2-layer predictor that maps
   ``(z_high_t, ego_motion_t) -> z_low_t_pred``.
3. **Hierarchical Bayesian inference**: top-down prediction
   ``z_low_t_pred`` competes with bottom-up encoding; the residual
   ``r_t = z_low_t - z_low_t_pred`` is what the score-aware Lagrangian
   penalizes (Catalog #311 cooperative-receiver gradient binding).

Distinct from sister Z6/Z7:

- Z6 (single-level FiLM ego-motion conditioning) — no level-1 predictor.
- Z7-Mamba-2 (state-space recurrence) — no explicit hierarchical boundary.
- Z5 (THIS) — EXPLICIT 2-level Rao-Ballard hierarchy with separate
  ``z_low`` + ``z_high`` per-pair latents + predictor.

PyTorch-parity invariants (state_dict-compatible bridge):

- ``low_latents`` / ``high_latents`` / ``ego_vecs`` — per-pair learnable tensors.
- ``predictor.high_to_hidden.{weight,bias}`` — high latent -> hidden.
- ``predictor.ego_to_hidden.{weight,bias}`` — ego motion -> hidden.
- ``predictor.hidden_layers.{i}.{weight,bias}`` — Sequential Linear+GELU stack.
- ``predictor.hidden_to_low.{weight,bias}`` — hidden -> low latent prediction.
- ``decoder.initial_proj.{weight,bias}`` — latent -> initial spatial grid.
- ``decoder.blocks.{i}.{weight,bias}`` — Conv2d / PixelShuffle / GELU stack.

Weight layout: Conv2d (out, in, kH, kW) (PyTorch); MLX NHWC Conv2d is
(out, kH, kW, in) — export bridge transposes (0, 3, 1, 2).

INDIVIDUALLY-FRACTAL per UNIQUE-AND-COMPLETE-PER-METHOD (Catalog #290):
the predictor + 2-level latent split + sigmoid-clamped decoder output is
Z5's substrate-distinguishing primitive per Catalog #272 — NOT shared-helper
shortcut from Z6/Z7 sisters.

[verified-against: src/tac/substrates/z6_v2_cargo_cult_unwind/mlx_renderer.py]
[verified-against: src/tac/substrates/time_traveler_l5_z5/architecture.py]
[verified-against: Rao+Ballard 1999 hierarchical predictive coding]
"""
# AUTOCAST_FP16_WAIVED:MLX_substrate_does_not_use_PyTorch_CUDA_autocast_fp16_per_mlx_first_canonical_doctrine_8th_standing_directive
# TF32_WAIVED:MLX_substrate_does_not_use_PyTorch_CUDA_tf32_per_mlx_first_canonical_doctrine
# TORCH_COMPILE_WAIVED:MLX_substrate_uses_mlx_value_and_grad_not_torch_compile
# NO_GRAD_WAIVED:MLX_substrate_uses_mlx_lazy_eval_not_pytorch_no_grad_per_mlx_first_canonical_doctrine

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tac.substrates.time_traveler_l5_z5.architecture import Z5RaoBallardConfig

try:  # pragma: no cover — exercised on Apple Silicon with MLX installed.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
except Exception as exc:  # pragma: no cover — import guard for non-Apple CI.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR: Exception | None = exc
else:
    _MLX_IMPORT_ERROR = None


SCHEMA_VERSION = "z5_rao_ballard_mlx_renderer_v1_20260528"
MLX_EVIDENCE_GRADE = "[macOS-MLX research-signal]"


def _require_mlx() -> None:
    if mx is None:
        raise RuntimeError(
            "MLX is not available on this host; the Z5 MLX renderer requires "
            "Apple Silicon with the ``mlx`` package installed. Original import "
            f"error: {_MLX_IMPORT_ERROR!r}"
        )


def _bilinear_resize_nhwc(x: Any, target_h: int, target_w: int) -> Any:
    """Bilinear resize NHWC tensor to (target_h, target_w).

    Mirrors PyTorch
    ``F.interpolate(out, size=(H,W), mode="bilinear", align_corners=False)``.
    Falls back to a simple gather-based bilinear that matches PyTorch's
    pixel-corner convention.
    """
    _require_mlx()
    src_h, src_w = int(x.shape[1]), int(x.shape[2])
    if src_h == target_h and src_w == target_w:
        return x
    # Compute the source coordinates per PyTorch bilinear (align_corners=False).
    scale_y = src_h / float(target_h)
    scale_x = src_w / float(target_w)
    y = (mx.arange(target_h, dtype=mx.float32) + 0.5) * scale_y - 0.5  # type: ignore[union-attr]
    x_idx = (mx.arange(target_w, dtype=mx.float32) + 0.5) * scale_x - 0.5  # type: ignore[union-attr]
    y0 = mx.clip(mx.floor(y), 0, src_h - 1).astype(mx.int32)  # type: ignore[union-attr]
    y1 = mx.clip(y0 + 1, 0, src_h - 1)  # type: ignore[union-attr]
    x0 = mx.clip(mx.floor(x_idx), 0, src_w - 1).astype(mx.int32)  # type: ignore[union-attr]
    x1 = mx.clip(x0 + 1, 0, src_w - 1)  # type: ignore[union-attr]
    wy = (y - y0.astype(mx.float32)).reshape((target_h, 1))  # type: ignore[union-attr]
    wx = (x_idx - x0.astype(mx.float32)).reshape((1, target_w))  # type: ignore[union-attr]
    # Gather four corners; broadcasting (B, target_h, target_w, C).
    g00 = x[:, y0[:, None], x0[None, :], :]  # type: ignore[index]
    g01 = x[:, y0[:, None], x1[None, :], :]  # type: ignore[index]
    g10 = x[:, y1[:, None], x0[None, :], :]  # type: ignore[index]
    g11 = x[:, y1[:, None], x1[None, :], :]  # type: ignore[index]
    wy_b = wy[None, :, :, None]  # type: ignore[index]
    wx_b = wx[None, :, :, None]  # type: ignore[index]
    out = (
        g00 * (1.0 - wy_b) * (1.0 - wx_b)
        + g01 * (1.0 - wy_b) * wx_b
        + g10 * wy_b * (1.0 - wx_b)
        + g11 * wy_b * wx_b
    )
    return out


class _Z5HierarchicalPredictorMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX-native 2-level Rao-Ballard predictor.

    Mirrors :class:`tac.substrates.time_traveler_l5_z5.architecture._Z5HierarchicalPredictor`
    forward semantics::

        h = tanh(W_high @ z_high + W_ego @ ego_motion)
        h = GELU(W_hidden @ h)  [num_layers-1 times]
        z_low_pred = W_out @ h
    """

    def __init__(
        self,
        *,
        high_latent_dim: int,
        low_latent_dim: int,
        ego_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        _require_mlx()
        super().__init__()
        if num_layers not in (2, 3):
            raise ValueError(
                f"predictor_num_layers must be 2 or 3; got {num_layers}"
            )
        self.high_latent_dim = int(high_latent_dim)
        self.low_latent_dim = int(low_latent_dim)
        self.ego_dim = int(ego_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.high_to_hidden = nn.Linear(  # type: ignore[union-attr]
            int(high_latent_dim), int(hidden_dim)
        )
        self.ego_to_hidden = nn.Linear(  # type: ignore[union-attr]
            int(ego_dim), int(hidden_dim)
        )
        # PyTorch sister uses Sequential(Linear, GELU, ...); MLX names
        # the linear layers ``hidden_layers.{i}`` where i goes 0, 2, 4 ...
        # (since GELU has no params, it does not occupy a slot). To preserve
        # PyTorch state_dict naming we use the same numbering convention:
        # hidden_layers.0 / hidden_layers.2 / ... map to the Linear params.
        self.hidden_layers: list[Any] = []
        for _ in range(num_layers - 1):
            layer = nn.Linear(int(hidden_dim), int(hidden_dim))  # type: ignore[union-attr]
            self.hidden_layers.append(layer)
        self.hidden_to_low = nn.Linear(  # type: ignore[union-attr]
            int(hidden_dim), int(low_latent_dim)
        )

    def __call__(self, z_high: Any, ego_motion: Any) -> Any:
        _require_mlx()
        h = mx.tanh(  # type: ignore[union-attr]
            self.high_to_hidden(z_high) + self.ego_to_hidden(ego_motion)
        )
        for layer in self.hidden_layers:
            h = nn.gelu(layer(h))  # type: ignore[union-attr]
        return self.hidden_to_low(h)


class _Z5DecoderMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX-native Z6-style PixelShuffle decoder.

    Mirrors :class:`tac.substrates.time_traveler_l5_z5.architecture._Z5Decoder`.
    Forward: ``z_low -> initial_proj -> reshape NHWC -> blocks ->
    bilinear_resize (if needed) -> sigmoid -> split rgb_0/rgb_1``.
    """

    def __init__(self, cfg: "Z5RaoBallardConfig") -> None:
        _require_mlx()
        super().__init__()
        self.cfg = cfg
        self.initial_proj = nn.Linear(  # type: ignore[union-attr]
            int(cfg.low_latent_dim),
            int(cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w),
        )
        in_ch = int(cfg.embed_dim)
        # PyTorch sister uses Sequential(Conv2d, PixelShuffle, GELU)*N + Conv2d.
        # We mirror the state_dict layout (blocks.0 = Conv2d, blocks.1 =
        # PixelShuffle (no params), blocks.2 = GELU (no params), blocks.3 =
        # Conv2d, ...). So ``blocks.{3*i}`` is the i-th Conv2d.
        self.blocks: list[Any] = []
        for i in range(int(cfg.num_upsample_blocks)):
            out_ch = int(cfg.decoder_channels[i])
            conv = nn.Conv2d(  # type: ignore[union-attr]
                in_channels=int(in_ch),
                out_channels=int(4 * out_ch),
                kernel_size=3,
                padding=1,
            )
            self.blocks.append(conv)
            in_ch = out_ch
        # Final 1x1 conv producing 6 channels (rgb_0 concat rgb_1).
        self.head_rgb = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=int(in_ch),
            out_channels=6,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, z_low: Any) -> tuple[Any, Any]:
        _require_mlx()
        if z_low.ndim != 2 or int(z_low.shape[1]) != int(self.cfg.low_latent_dim):
            raise ValueError(
                f"decoder expects (B, low_latent_dim={self.cfg.low_latent_dim}); "
                f"got {tuple(z_low.shape)}"
            )
        batch = int(z_low.shape[0])
        flat = self.initial_proj(z_low)
        # MLX uses NHWC; PyTorch uses NCHW. Reshape to NHWC here.
        grid = mx.reshape(  # type: ignore[union-attr]
            flat,
            (
                batch,
                int(self.cfg.initial_grid_h),
                int(self.cfg.initial_grid_w),
                int(self.cfg.embed_dim),
            ),
        )
        h = grid
        for conv in self.blocks:
            h = conv(h)
            # PixelShuffle 2x for NHWC via canonical helper.
            from tac.local_acceleration.pr95_hnerv_mlx import pixel_shuffle_2x_nhwc

            h = pixel_shuffle_2x_nhwc(h)
            h = nn.gelu(h)  # type: ignore[union-attr]
        # Final 1x1 conv -> NHWC (B, H, W, 6).
        out = self.head_rgb(h)
        # Resize if needed to (output_height, output_width).
        out = _bilinear_resize_nhwc(
            out, int(self.cfg.output_height), int(self.cfg.output_width)
        )
        out = mx.sigmoid(out)  # type: ignore[union-attr]
        # Split last dim into rgb_0 / rgb_1, transpose to NCHW (B, 3, H, W).
        rgb_0_nhwc = out[..., :3]
        rgb_1_nhwc = out[..., 3:]
        rgb_0 = mx.transpose(rgb_0_nhwc, (0, 3, 1, 2))  # type: ignore[union-attr]
        rgb_1 = mx.transpose(rgb_1_nhwc, (0, 3, 1, 2))  # type: ignore[union-attr]
        return rgb_0, rgb_1


class Z5RaoBallardSubstrateMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX-native Z5 substrate (L1 LONG-RUN MLX-LOCAL).

    1:1 architectural mirror of
    :class:`tac.substrates.time_traveler_l5_z5.architecture.Z5RaoBallardSubstrate`.

    Forward path::

        pair_indices ->
            z_low      = low_latents[indices]
            z_high     = high_latents[indices]
            ego_motion = ego_vecs[indices]
            z_low_pred = predictor(z_high, ego_motion)
            residual   = z_low - z_low_pred   (stored for residual-savings)
            rgb_0, rgb_1 = decoder(z_low)
        -> output (B, 2, 3, H, W) per canonical ``call_b2chw_255`` convention.

    Output values in [0, 255] (sigmoid * 255) so the canonical mlx_score_aware
    harness's PyTorch teacher cache + reconstruction MSE see them in the same
    range as the PyTorch sister.

    Per HNeRV parity L5: outputs RGB at contest camera resolution (384, 512);
    NOT a mask codec. Per CLAUDE.md "MLX portable-local-substrate authority":
    every artifact is non-promotable ``[macOS-MLX research-signal]``.
    """

    def __init__(self, cfg: "Z5RaoBallardConfig", *, seed: int = 0) -> None:
        _require_mlx()
        super().__init__()
        self.cfg = cfg
        num_pairs = int(cfg.num_pairs)
        low_latent_dim = int(cfg.low_latent_dim)
        high_latent_dim = int(cfg.high_latent_dim)
        ego_dim = int(cfg.ego_dim)
        # Deterministic init per seed for reproducibility.
        mx.random.seed(int(seed))  # type: ignore[union-attr]
        self.low_latents = (
            mx.random.normal(shape=(num_pairs, low_latent_dim)) * 0.02  # type: ignore[union-attr]
        )
        self.high_latents = (
            mx.random.normal(shape=(num_pairs, high_latent_dim)) * 0.02  # type: ignore[union-attr]
        )
        self.ego_vecs = (
            mx.random.normal(shape=(num_pairs, ego_dim)) * 0.02  # type: ignore[union-attr]
        )
        self.predictor = _Z5HierarchicalPredictorMLX(
            high_latent_dim=high_latent_dim,
            low_latent_dim=low_latent_dim,
            ego_dim=ego_dim,
            hidden_dim=int(cfg.predictor_hidden_dim),
            num_layers=int(cfg.predictor_num_layers),
        )
        self.decoder = _Z5DecoderMLX(cfg)

    def reconstruct_pair(
        self, pair_indices_np: np.ndarray | Any
    ) -> tuple[Any, Any, Any]:
        """Return (rgb_0_b3hw, rgb_1_b3hw, latents) per harness contract.

        ``forward_convention="reconstruct_pair_nchw01"`` per the canonical
        mlx_score_aware harness: output is (B, 3, H, W) in [0, 1] (we divide
        by 255 here since we apply sigmoid * 255 in the decoder, which is
        the contract the PyTorch sister uses).

        ``latents`` is a 3-tuple-ish container for the harness's optional
        cooperative-receiver residual computation; we return the per-pair
        residual ``z_low - z_low_pred`` so the score-aware Lagrangian can
        penalize it.
        """
        _require_mlx()
        # The harness calls reconstruct_pair with either an mx.array or np.ndarray.
        if not isinstance(pair_indices_np, mx.array.__class__) and not hasattr(  # type: ignore[union-attr]
            pair_indices_np, "shape"
        ):
            pair_indices_np = np.asarray(pair_indices_np, dtype=np.int32)
        # Convert to mx.array if needed.
        if hasattr(pair_indices_np, "dtype") and not str(
            type(pair_indices_np)
        ).startswith("<class 'mlx"):
            indices = mx.array(  # type: ignore[union-attr]
                np.asarray(pair_indices_np, dtype=np.int32)
            )
        else:
            indices = pair_indices_np
        z_low = mx.take(self.low_latents, indices, axis=0)  # type: ignore[union-attr]
        z_high = mx.take(self.high_latents, indices, axis=0)  # type: ignore[union-attr]
        ego = mx.take(self.ego_vecs, indices, axis=0)  # type: ignore[union-attr]
        z_low_pred = self.predictor(z_high, ego)
        residual = z_low - z_low_pred
        rgb_0, rgb_1 = self.decoder(z_low)
        # Output in [0, 1] per harness contract (sigmoid output, no x255 here
        # since we use reconstruct_pair_nchw01 NOT call_b2chw_255).
        return rgb_0, rgb_1, residual

    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        _require_mlx()
        flat = tree_flatten(self.parameters())  # type: ignore[union-attr]
        total = 0
        for _name, arr in flat:
            total += int(np.prod(arr.shape))
        return total

    def export_state_dict(self) -> dict[str, np.ndarray]:
        """Export weights in PyTorch state_dict layout for the Z5RB1 bridge.

        Mirrors sister Z6-v2 ``Z6V2SubstrateMLX.export_state_dict``:

        - Conv2d weights: PyTorch ``(out, in, kH, kW)``. MLX NHWC stores
          weights as ``(out, kH, kW, in)`` — transpose ``(0, 3, 1, 2)``.
        - Linear weights: ``(out, in)`` — MLX matches PyTorch directly.
        - Per-pair latents / ego_vecs: ``(num_pairs, D)`` — MLX matches.
        """
        _require_mlx()
        out: dict[str, np.ndarray] = {}
        out["low_latents"] = np.asarray(self.low_latents, dtype=np.float32).copy()
        out["high_latents"] = np.asarray(
            self.high_latents, dtype=np.float32
        ).copy()
        out["ego_vecs"] = np.asarray(self.ego_vecs, dtype=np.float32).copy()

        # Predictor.
        out["predictor.high_to_hidden.weight"] = np.asarray(
            self.predictor.high_to_hidden.weight, dtype=np.float32
        ).copy()
        out["predictor.high_to_hidden.bias"] = np.asarray(
            self.predictor.high_to_hidden.bias, dtype=np.float32
        ).copy()
        out["predictor.ego_to_hidden.weight"] = np.asarray(
            self.predictor.ego_to_hidden.weight, dtype=np.float32
        ).copy()
        out["predictor.ego_to_hidden.bias"] = np.asarray(
            self.predictor.ego_to_hidden.bias, dtype=np.float32
        ).copy()
        # PyTorch sister uses Sequential(Linear, GELU, Linear, GELU, ...);
        # the Linear params live at hidden_layers.0, .2, .4 ... (positions
        # where the GELU has no params). MLX list-based naming uses .0, .1,
        # ... so to bridge to PyTorch state_dict we expand by 2x.
        for i, layer in enumerate(self.predictor.hidden_layers):
            torch_idx = 2 * i  # Sequential slot for Linear
            out[f"predictor.hidden_layers.{torch_idx}.weight"] = np.asarray(
                layer.weight, dtype=np.float32
            ).copy()
            out[f"predictor.hidden_layers.{torch_idx}.bias"] = np.asarray(
                layer.bias, dtype=np.float32
            ).copy()
        out["predictor.hidden_to_low.weight"] = np.asarray(
            self.predictor.hidden_to_low.weight, dtype=np.float32
        ).copy()
        out["predictor.hidden_to_low.bias"] = np.asarray(
            self.predictor.hidden_to_low.bias, dtype=np.float32
        ).copy()

        # Decoder.
        out["decoder.initial_proj.weight"] = np.asarray(
            self.decoder.initial_proj.weight, dtype=np.float32
        ).copy()
        out["decoder.initial_proj.bias"] = np.asarray(
            self.decoder.initial_proj.bias, dtype=np.float32
        ).copy()
        # PyTorch sister: Sequential(Conv2d, PixelShuffle, GELU)*N + Conv2d.
        # Linear/conv slots are 0, 3, 6 ... and the head sits at 3*N.
        for i, conv in enumerate(self.decoder.blocks):
            torch_idx = 3 * i
            # MLX NHWC weight: (out, kH, kW, in) -> PyTorch NCHW (out, in, kH, kW)
            w = np.asarray(conv.weight, dtype=np.float32)
            w_pyt = np.transpose(w, (0, 3, 1, 2)).copy()
            out[f"decoder.blocks.{torch_idx}.weight"] = w_pyt
            out[f"decoder.blocks.{torch_idx}.bias"] = np.asarray(
                conv.bias, dtype=np.float32
            ).copy()
        # Final head conv (last slot in PyTorch Sequential).
        head_idx = 3 * int(self.cfg.num_upsample_blocks)
        w_head = np.asarray(self.decoder.head_rgb.weight, dtype=np.float32)
        out[f"decoder.blocks.{head_idx}.weight"] = np.transpose(
            w_head, (0, 3, 1, 2)
        ).copy()
        out[f"decoder.blocks.{head_idx}.bias"] = np.asarray(
            self.decoder.head_rgb.bias, dtype=np.float32
        ).copy()
        return out


__all__ = [
    "MLX_EVIDENCE_GRADE",
    "SCHEMA_VERSION",
    "Z5RaoBallardSubstrateMLX",
]
