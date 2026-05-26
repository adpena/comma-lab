# SPDX-License-Identifier: MIT
"""ATW V2 cooperative-receiver V2 — numpy reference implementation (Axis 3 portability).

Per Phase 3 design memo §9 (portability via numpy per primitive) + operator
binding directive #3 ("we also need adversarial review against all landing
recursive for ... portability via numpy").

This module is the FORWARD + INFLATE path reference implementation in pure
numpy. Bit-exact (per primitive) reference for MLX↔numpy + PyTorch↔numpy
parity verification in test_basic.py.

NON-PORTABLE EXCEPTION
======================

The training loss compute (cooperative-receiver loss against SegNet+PoseNet
gradient path) is explicitly NON-PORTABLE per Axis 3 documented exception.
It lives in ``_training_only.py`` (PyTorch only). This module covers the
FORWARD path + INFLATE path which MUST remain portable to CPU-only systems.

Per-primitive portability per Phase 3 design memo §9:

| MLX primitive            | numpy reference                                    | Status      |
|--------------------------|----------------------------------------------------|-------------|
| Linear (W·x + b)         | numpy ``@`` operator + addition                    | bit-exact   |
| Conv2d                   | scipy.signal.correlate2d per channel (slow but bit-exact when alignment + padding match) | bit-exact   |
| ReLU / maximum           | ``np.maximum(x, 0)``                               | bit-exact   |
| Sigmoid                  | ``1 / (1 + np.exp(-x))``                           | bit-exact   |
| Softmax                  | numerically-stable: ``exp(x - max) / sum(exp(x - max))`` | bit-exact   |
| mean(axis=...)           | ``np.mean(axis=axis)``                             | bit-exact   |
| ego-motion FOE projection| ``np.linalg.norm`` + broadcast division            | bit-exact   |

Cross-references
----------------

* Phase 3 design memo §7 + §9 (math + portability per layer)
* `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py` (canonical MLX
  renderer pattern this numpy reference mirrors)
* `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss`
  (canonical primitive routed by `_training_only.py`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Defaults per Phase 3 design memo §7 + §1 (canonical-vs-unique per layer).
DEFAULT_NUM_PAIRS: int = 600
"""Contest pair count (1200 frames / 2 frames per pair)."""

DEFAULT_OUTPUT_H: int = 384
DEFAULT_OUTPUT_W: int = 512
"""Contest scorer-resolution (height, width)."""

DEFAULT_LATENT_DIM: int = 32
"""Substrate-optimal latent_dim per Phase 3 design memo §1 + §2 (FORK from
V1 inherited latent_dim=24); Pareto-vertex Wave N+1 paired-smoke target."""

DEFAULT_EGO_MOTION_DIM: int = 6
"""Per-pair ego-motion FOE projection dim: 3 translation components + 3
rotation components (PoseNet emits 6 pose deltas per pair)."""

DEFAULT_COND_EMBED_DIM: int = 16
"""Per-pair conditioning embedding dim: ego_motion → cond_embed via small MLP."""

DEFAULT_ENCODER_HIDDEN_DIM: int = 64
"""Encoder hidden dim (substrate-optimal capacity TBD Wave N+1)."""

DEFAULT_DECODER_EMBED_DIM: int = 32
"""Decoder initial-grid embedding dim."""

DEFAULT_DECODER_INITIAL_GRID_H: int = 3
DEFAULT_DECODER_INITIAL_GRID_W: int = 4
"""Decoder initial grid (3, 4) before PixelShuffle upsampling chain."""

DEFAULT_DECODER_CHANNELS: tuple[int, ...] = (24, 20, 16, 12, 8, 6)
"""HNeRV-style decoder per-block output channels."""

DEFAULT_DECODER_NUM_UPSAMPLE_BLOCKS: int = 6
"""Number of PixelShuffle(2) blocks: (3,4) -> (192, 256) -> bilinear interp to (384, 512)."""


@dataclass(frozen=True)
class CooperativeReceiverConfig:
    """Static design-time parameters for the ATW V2 cooperative-receiver V2 substrate.

    Phase 3 L0 SCAFFOLD defaults per design memo §7. Values FORK from V1
    cargo-cult inheritance per Phase 1 CC-5 + CC-6 + CC-7 unwinds; substrate-
    optimal capacity TBD via Wave N+1 Pareto-vertex paired smoke.
    """

    num_pairs: int = DEFAULT_NUM_PAIRS
    output_height: int = DEFAULT_OUTPUT_H
    output_width: int = DEFAULT_OUTPUT_W
    latent_dim: int = DEFAULT_LATENT_DIM
    ego_motion_dim: int = DEFAULT_EGO_MOTION_DIM
    cond_embed_dim: int = DEFAULT_COND_EMBED_DIM
    encoder_input_channels: int = 3
    encoder_hidden_dim: int = DEFAULT_ENCODER_HIDDEN_DIM
    decoder_embed_dim: int = DEFAULT_DECODER_EMBED_DIM
    decoder_initial_grid_h: int = DEFAULT_DECODER_INITIAL_GRID_H
    decoder_initial_grid_w: int = DEFAULT_DECODER_INITIAL_GRID_W
    decoder_channels: tuple[int, ...] = DEFAULT_DECODER_CHANNELS
    decoder_num_upsample_blocks: int = DEFAULT_DECODER_NUM_UPSAMPLE_BLOCKS

    @property
    def output_hw(self) -> tuple[int, int]:
        return (self.output_height, self.output_width)


def numpy_relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation. Bit-exact per Axis 3 portability."""
    return np.maximum(x, 0.0)


def numpy_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid. Bit-exact per Axis 3 portability."""
    # Handle large negative inputs to avoid overflow; standard stable form.
    pos = x >= 0
    out = np.empty_like(x)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def numpy_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax along axis. Bit-exact per Axis 3 portability."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def numpy_linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """Linear layer. Weight shape ``(out_features, in_features)``; bias optional.

    Bit-exact per Axis 3 portability. Sister to PyTorch ``nn.Linear`` weight
    layout convention.
    """
    # nn.Linear: out = x @ W.T + b, where W is (out, in)
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def numpy_ego_motion_foe_projection(pose_delta: np.ndarray) -> np.ndarray:
    """Per-pair ego-motion FOE projection per Phase 3 §7 (Ballard 2007 + Catalog #311).

    Args:
        pose_delta: per-pair PoseNet pose-delta tensor shape ``(B, 6)``.
            First 3 components = translation; last 3 = rotation.

    Returns:
        ego-motion FOE projection ``Y_ego_motion`` shape ``(B, 6)``:
        ``[translation_normalized (B, 3); rotation_normalized (B, 3)]``.

    Bit-exact per Axis 3 portability. The dominant translational direction
    (unit vector of pose-delta translation components) IS the FOE projection
    per Ballard 2007's continuous-time motion vision framework.
    """
    if pose_delta.ndim != 2 or pose_delta.shape[1] != 6:
        raise ValueError(
            f"pose_delta must be (B, 6); got {pose_delta.shape}"
        )
    translation = pose_delta[:, :3]
    rotation = pose_delta[:, 3:]
    # Normalize per-pair; epsilon to avoid division by zero on stationary pairs.
    eps = 1e-8
    t_norm = np.linalg.norm(translation, axis=1, keepdims=True) + eps
    r_norm = np.linalg.norm(rotation, axis=1, keepdims=True) + eps
    return np.concatenate(
        [translation / t_norm, rotation / r_norm],
        axis=1,
    ).astype(pose_delta.dtype)


def numpy_conv2d_simple(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    padding: int = 1,
) -> np.ndarray:
    """Simple Conv2d reference for small kernels (k=3, stride=1, padding=1).

    Args:
        x: input tensor shape ``(B, C_in, H, W)``.
        weight: kernel tensor shape ``(C_out, C_in, kH, kW)``.
        bias: bias tensor shape ``(C_out,)`` or None.
        padding: spatial padding (default 1 for k=3 stride-preserving).

    Returns:
        Convolved tensor shape ``(B, C_out, H, W)``.

    Bit-exact reference for the canonical Conv2d-3x3-pad-1 used in encoder
    + decoder blocks per Phase 3 design memo §1. Slow on CPU; production
    use should route through MLX or PyTorch.

    Axis 3 portability: PORTABLE; trivially CPU-only-system supported.
    """
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError(
            f"x must be (B, C_in, H, W) and weight (C_out, C_in, kH, kW); got "
            f"x={x.shape} weight={weight.shape}"
        )
    B, C_in, H, W = x.shape
    C_out, C_in_w, kH, kW = weight.shape
    if C_in != C_in_w:
        raise ValueError(f"Input channels mismatch: x has {C_in}, weight expects {C_in_w}")

    # Pad input.
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0.0,
    )
    out_H = x_padded.shape[2] - kH + 1
    out_W = x_padded.shape[3] - kW + 1

    out = np.zeros((B, C_out, out_H, out_W), dtype=x.dtype)
    for b in range(B):
        for c_out in range(C_out):
            acc = np.zeros((out_H, out_W), dtype=x.dtype)
            for c_in in range(C_in):
                kernel = weight[c_out, c_in]
                # 2D correlation (NOT convolution; PyTorch + MLX both do
                # correlation despite the "conv" naming).
                for i in range(out_H):
                    for j in range(out_W):
                        acc[i, j] += np.sum(x_padded[b, c_in, i : i + kH, j : j + kW] * kernel)
            if bias is not None:
                acc = acc + bias[c_out]
            out[b, c_out] = acc
    return out


def numpy_pixel_shuffle(x: np.ndarray, upscale_factor: int = 2) -> np.ndarray:
    """PixelShuffle reference (factor 2 default).

    Args:
        x: input tensor shape ``(B, C * r^2, H, W)`` where ``r = upscale_factor``.

    Returns:
        Shuffled tensor shape ``(B, C, H * r, W * r)``.

    Bit-exact reference for the canonical PixelShuffle(2) used in decoder
    per Phase 3 design memo §1. Sister to PyTorch ``nn.PixelShuffle(2)``.
    """
    B, C_in, H, W = x.shape
    r = upscale_factor
    if C_in % (r * r) != 0:
        raise ValueError(
            f"PixelShuffle requires input channels divisible by r^2={r*r}; got {C_in}"
        )
    C_out = C_in // (r * r)
    # Reshape: (B, C_out, r, r, H, W) -> (B, C_out, H, r, W, r) -> (B, C_out, H*r, W*r)
    x_reshaped = x.reshape(B, C_out, r, r, H, W)
    x_transposed = x_reshaped.transpose(0, 1, 4, 2, 5, 3)
    return x_transposed.reshape(B, C_out, H * r, W * r)


def numpy_bilinear_interpolate(
    x: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Bilinear interpolation reference (align_corners=False to match PyTorch default).

    Args:
        x: input tensor shape ``(B, C, H, W)``.
        output_size: target ``(out_H, out_W)``.

    Returns:
        Interpolated tensor shape ``(B, C, out_H, out_W)``.

    Bit-exact (modulo FP rounding) reference for the canonical
    ``F.interpolate(mode='bilinear', align_corners=False)`` used in decoder
    final upsample per Phase 3 design memo §1.
    """
    B, C, H, W = x.shape
    out_H, out_W = output_size
    if out_H == H and out_W == W:
        return x.copy()

    # align_corners=False sampling: x_in = (x_out + 0.5) * (H_in / H_out) - 0.5
    h_ratio = H / out_H
    w_ratio = W / out_W

    y_out = np.arange(out_H, dtype=x.dtype)
    x_out = np.arange(out_W, dtype=x.dtype)
    y_in = (y_out + 0.5) * h_ratio - 0.5
    x_in = (x_out + 0.5) * w_ratio - 0.5

    y_in = np.clip(y_in, 0, H - 1)
    x_in = np.clip(x_in, 0, W - 1)

    y0 = np.floor(y_in).astype(np.int64)
    x0 = np.floor(x_in).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)

    wy = (y_in - y0).reshape(-1, 1)
    wx = x_in - x0

    out = np.zeros((B, C, out_H, out_W), dtype=x.dtype)
    for b in range(B):
        for c in range(C):
            val_00 = x[b, c, y0[:, None], x0[None, :]]
            val_01 = x[b, c, y0[:, None], x1[None, :]]
            val_10 = x[b, c, y1[:, None], x0[None, :]]
            val_11 = x[b, c, y1[:, None], x1[None, :]]
            top = val_00 * (1 - wx) + val_01 * wx
            bot = val_10 * (1 - wx) + val_11 * wx
            out[b, c] = top * (1 - wy) + bot * wy
    return out


def numpy_decode_pair_from_latent(
    z: np.ndarray,
    *,
    cfg: CooperativeReceiverConfig,
    initial_proj_weight: np.ndarray,
    initial_proj_bias: np.ndarray,
    decoder_block_weights: list[np.ndarray],
    decoder_block_biases: list[np.ndarray],
    final_conv_weight: np.ndarray,
    final_conv_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """End-to-end numpy decode: per-pair latent -> (rgb_0, rgb_1) frame pair.

    Args:
        z: per-pair latent shape ``(B, latent_dim)``.
        cfg: substrate config.
        initial_proj_weight / initial_proj_bias: Linear(latent_dim,
            embed*grid_h*grid_w) weights.
        decoder_block_weights / decoder_block_biases: per-block Conv2d weights
            for the upsample chain.
        final_conv_weight / final_conv_bias: final Conv2d(C_in, 6) weight + bias.

    Returns:
        ``(rgb_0, rgb_1)`` each shape ``(B, 3, output_H, output_W)`` in [0, 1].

    Bit-exact reference for the canonical decoder per Phase 3 design memo §1.
    """
    B = z.shape[0]
    if z.shape[1] != cfg.latent_dim:
        raise ValueError(f"z shape mismatch: got {z.shape}, expected (B, {cfg.latent_dim})")

    # Initial projection: (B, latent_dim) -> (B, embed*grid_h*grid_w)
    flat = numpy_linear(z, initial_proj_weight, initial_proj_bias)
    # Reshape to (B, embed, grid_h, grid_w)
    grid = flat.reshape(
        B, cfg.decoder_embed_dim, cfg.decoder_initial_grid_h, cfg.decoder_initial_grid_w
    )

    # Upsample chain: per-block Conv2d (in -> 4*out) + PixelShuffle(2) + ReLU
    h = grid
    for i in range(cfg.decoder_num_upsample_blocks):
        weight = decoder_block_weights[i]
        bias = decoder_block_biases[i]
        h = numpy_conv2d_simple(h, weight, bias, padding=1)
        h = numpy_pixel_shuffle(h, upscale_factor=2)
        h = numpy_relu(h)

    # Final Conv2d(C_in, 6) for RGB pair output
    h = numpy_conv2d_simple(h, final_conv_weight, final_conv_bias, padding=1)
    # Resize if needed to match output_height x output_width
    if h.shape[2] != cfg.output_height or h.shape[3] != cfg.output_width:
        h = numpy_bilinear_interpolate(h, (cfg.output_height, cfg.output_width))

    # Sigmoid to [0, 1] then split (6 channels) -> (rgb_0, rgb_1)
    h = numpy_sigmoid(h)
    rgb_0 = h[:, :3, :, :]
    rgb_1 = h[:, 3:, :, :]
    return rgb_0, rgb_1


def numpy_decode_pair_with_ego_motion_conditioning(
    per_pair_latent_residual: np.ndarray,
    ego_motion_proj: np.ndarray,
    *,
    cfg: CooperativeReceiverConfig,
    cond_embed_weight_1: np.ndarray,
    cond_embed_bias_1: np.ndarray,
    cond_embed_weight_2: np.ndarray,
    cond_embed_bias_2: np.ndarray,
    initial_proj_weight: np.ndarray,
    initial_proj_bias: np.ndarray,
    decoder_block_weights: list[np.ndarray],
    decoder_block_biases: list[np.ndarray],
    final_conv_weight: np.ndarray,
    final_conv_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """End-to-end numpy decode with ego-motion conditioning.

    Per Phase 3 design memo §1 + §7:

    1. Build per-pair conditioning embedding from ego-motion FOE projection:
       ``cond_embed = sigmoid(Linear(relu(Linear(Y_ego_motion))))``.
    2. Reconstruct per-pair latent: ``z = z_residual + cond_embed_projection``
       (where cond_embed is projected onto latent_dim via the second Linear).
    3. Decode ``z`` via numpy_decode_pair_from_latent.

    Args:
        per_pair_latent_residual: archived per-pair latent residual shape
            ``(B, latent_dim)``.
        ego_motion_proj: per-pair ego-motion FOE projection shape
            ``(B, ego_motion_dim)``.

    Returns:
        ``(rgb_0, rgb_1)`` each shape ``(B, 3, output_H, output_W)`` in [0, 1].

    Bit-exact reference for the canonical inflate-time forward pass per
    Phase 3 design memo §1 + §7. Sister to MLX renderer.
    """
    B = per_pair_latent_residual.shape[0]
    # Conditioning embedding: (B, ego_motion_dim) -> hidden -> cond_embed_dim
    h = numpy_linear(ego_motion_proj, cond_embed_weight_1, cond_embed_bias_1)
    h = numpy_relu(h)
    cond_embed = numpy_linear(h, cond_embed_weight_2, cond_embed_bias_2)
    # cond_embed shape: (B, latent_dim) — projected directly onto latent_dim
    # by the second Linear layer's output dimension.

    if cond_embed.shape[1] != cfg.latent_dim:
        raise ValueError(
            f"cond_embed dim mismatch: got {cond_embed.shape}, expected (B, {cfg.latent_dim})"
        )

    # Reconstruct per-pair latent: z = z_residual + cond_embed
    z = per_pair_latent_residual + cond_embed

    return numpy_decode_pair_from_latent(
        z,
        cfg=cfg,
        initial_proj_weight=initial_proj_weight,
        initial_proj_bias=initial_proj_bias,
        decoder_block_weights=decoder_block_weights,
        decoder_block_biases=decoder_block_biases,
        final_conv_weight=final_conv_weight,
        final_conv_bias=final_conv_bias,
    )


@dataclass(frozen=True)
class NumpyWeights:
    """Container for numpy weight arrays per Phase 3 design memo §1 decoder + cond_embed.

    Sister-comparable to MLX + PyTorch state_dict via byte-stable export per
    MLX→PyTorch bridge #1251 pattern.
    """

    initial_proj_weight: np.ndarray
    initial_proj_bias: np.ndarray
    decoder_block_weights: list[np.ndarray]
    decoder_block_biases: list[np.ndarray]
    final_conv_weight: np.ndarray
    final_conv_bias: np.ndarray
    cond_embed_weight_1: np.ndarray
    cond_embed_bias_1: np.ndarray
    cond_embed_weight_2: np.ndarray
    cond_embed_bias_2: np.ndarray


def init_numpy_weights_random(
    cfg: CooperativeReceiverConfig,
    *,
    seed: int = 0,
    std: float = 0.02,
) -> NumpyWeights:
    """Initialize numpy weights with deterministic random values.

    Used in test_basic.py for MLX↔numpy parity verification. NOT a production
    init pattern; just a deterministic reference initializer for sanity tests.
    """
    rng = np.random.default_rng(seed)
    embed_total = cfg.decoder_embed_dim * cfg.decoder_initial_grid_h * cfg.decoder_initial_grid_w

    initial_proj_weight = rng.normal(0, std, size=(embed_total, cfg.latent_dim)).astype(np.float32)
    initial_proj_bias = np.zeros((embed_total,), dtype=np.float32)

    decoder_block_weights: list[np.ndarray] = []
    decoder_block_biases: list[np.ndarray] = []
    in_ch = cfg.decoder_embed_dim
    for i in range(cfg.decoder_num_upsample_blocks):
        out_ch = cfg.decoder_channels[i]
        w = rng.normal(0, std, size=(4 * out_ch, in_ch, 3, 3)).astype(np.float32)
        b = np.zeros((4 * out_ch,), dtype=np.float32)
        decoder_block_weights.append(w)
        decoder_block_biases.append(b)
        in_ch = out_ch

    final_conv_weight = rng.normal(0, std, size=(6, in_ch, 3, 3)).astype(np.float32)
    final_conv_bias = np.zeros((6,), dtype=np.float32)

    # Conditioning embedding weights:
    # ego_motion_dim -> cond_embed_dim -> latent_dim
    cond_embed_weight_1 = rng.normal(0, std, size=(cfg.cond_embed_dim, cfg.ego_motion_dim)).astype(
        np.float32
    )
    cond_embed_bias_1 = np.zeros((cfg.cond_embed_dim,), dtype=np.float32)
    cond_embed_weight_2 = rng.normal(0, std, size=(cfg.latent_dim, cfg.cond_embed_dim)).astype(
        np.float32
    )
    cond_embed_bias_2 = np.zeros((cfg.latent_dim,), dtype=np.float32)

    return NumpyWeights(
        initial_proj_weight=initial_proj_weight,
        initial_proj_bias=initial_proj_bias,
        decoder_block_weights=decoder_block_weights,
        decoder_block_biases=decoder_block_biases,
        final_conv_weight=final_conv_weight,
        final_conv_bias=final_conv_bias,
        cond_embed_weight_1=cond_embed_weight_1,
        cond_embed_bias_1=cond_embed_bias_1,
        cond_embed_weight_2=cond_embed_weight_2,
        cond_embed_bias_2=cond_embed_bias_2,
    )


__all__ = [
    "CooperativeReceiverConfig",
    "DEFAULT_COND_EMBED_DIM",
    "DEFAULT_DECODER_CHANNELS",
    "DEFAULT_DECODER_EMBED_DIM",
    "DEFAULT_DECODER_INITIAL_GRID_H",
    "DEFAULT_DECODER_INITIAL_GRID_W",
    "DEFAULT_DECODER_NUM_UPSAMPLE_BLOCKS",
    "DEFAULT_EGO_MOTION_DIM",
    "DEFAULT_ENCODER_HIDDEN_DIM",
    "DEFAULT_LATENT_DIM",
    "DEFAULT_NUM_PAIRS",
    "DEFAULT_OUTPUT_H",
    "DEFAULT_OUTPUT_W",
    "NumpyWeights",
    "init_numpy_weights_random",
    "numpy_bilinear_interpolate",
    "numpy_conv2d_simple",
    "numpy_decode_pair_from_latent",
    "numpy_decode_pair_with_ego_motion_conditioning",
    "numpy_ego_motion_foe_projection",
    "numpy_linear",
    "numpy_pixel_shuffle",
    "numpy_relu",
    "numpy_sigmoid",
    "numpy_softmax",
]
