# SPDX-License-Identifier: MIT
"""Pure-numpy reference port of the capstone VQ-NeRV decoder forward (Task #78).

This is the **portability contract** half of the operator's MLX-FIRST law
("MLX fast path -> numpy reference = portability contract -> PyTorch"). It
reproduces ``CapstoneVqNervBundle._decode_with_film`` (the per-frame-FiLM crux
fix, 2026-06-10) in pure numpy so the contest ``inflate.py`` runtime can decode
the archive on ANY host (CPU/CUDA-agnostic, no MLX, no torch) and write the
exact frames the MLX bundle renders.

Parameterized over ``base_channels`` (reads the channel taper + weights from the
parsed archive), so it works for the 16/20/24/36 class without code change.

Op-by-op fidelity to the MLX forward (``HNeRVDecoderMLX`` + ``_PoseFiLM``):

  stem (Linear: ``x @ W.T + b``)
    -> reshape (B, channels[0], base_h, base_w)
    -> transpose to NHWC
    -> ``sin``
    -> 6 upsample blocks, each:
         identity = bilinear_2x(x, align_corners=False)   [+ 1x1 skip_conv if ch change]
         decoded  = pixel_shuffle_2x(conv3x3(x, pad=1))    [channel-FIRST layout]
         x = sin(decoded + identity)
    -> refined = refine1(refine0(x))   (refine0: 3x3 pad=2 dil=2; refine1: 3x3 pad=1)
    -> feat = x + 0.1 * sin(refined)
    -> PER-FRAME FiLM (separate film0/film1 modulating feat DIFFERENTLY):
         g_k = 1 + tanh(fc2_k(sin(fc1_k(norm_pose)))[:, :C])
         b_k = fc2_k(...)[:, C:]
         feat_k = g_k[:,None,None,:] * feat + b_k[:,None,None,:]
    -> f_k = sigmoid(rgb_k(feat_k)) * 255      (rgb_k: 3x3 pad=1)
    -> stack -> (B, 2, 3, H, W) N2CHW.

Authority: this is the NUMERIC REFERENCE. The MLX path may drift from it by the
small fp32 accumulation order delta of ``mx.conv2d`` / MLX matmul; the parity
test measures that residual and asserts it is NOT score-affecting (argmax/pose
invariant). No scorer is loaded here (Strict scorer rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

POSE_DIM = 6


@dataclass(frozen=True)
class CapstoneDecodeConfig:
    """Numeric config the numpy decode needs (derived from the bundle/archive).

    ``base_channels`` reconstructs the PR95 channel taper; ``latent_dim`` /
    ``codebook_size`` are carried for completeness. ``film_enabled`` mirrors the
    bundle flag; ``pose_normalize`` + ``pose_mean`` / ``pose_std`` reproduce the
    FiLM input standardization.
    """

    base_channels: int = 36
    latent_dim: int = 28
    codebook_size: int = 256
    base_h: int = 6
    base_w: int = 8
    film_enabled: bool = True
    pose_normalize: bool = True
    pose_mean: tuple[float, ...] = (0.0,) * POSE_DIM
    pose_std: tuple[float, ...] = (1.0,) * POSE_DIM

    def channels(self) -> list[int]:
        """The PR95 channel taper (matches ``HNeRVDecoderMLX.channels``)."""
        bc = int(self.base_channels)
        ch = [bc, bc, bc, int(bc * 0.75), int(bc * 0.58), int(bc * 0.5), int(bc * 0.5)]
        if min(ch) < 1:
            raise ValueError("base_channels too small for PR95 channel taper")
        return ch


# --------------------------------------------------------------------------
# numpy op primitives (op-for-op with the MLX canonical helpers)
# --------------------------------------------------------------------------


def _sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable logistic (matches mx.sigmoid to fp32 eps).
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    """MLX ``nn.Linear``: ``x @ W.T + b`` with W shaped (out, in)."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = x.astype(np.float32) @ weight.astype(np.float32).T
    if bias is not None:
        out = out + bias.astype(np.float32)
    return out.astype(np.float32)


def conv2d_nhwc(
    x_nhwc: np.ndarray,
    weight_ohwi: np.ndarray,
    bias: np.ndarray | None,
    *,
    padding: int = 0,
    dilation: int = 1,
) -> np.ndarray:
    """NHWC conv matching ``mx.conv2d`` (groups=1, stride=1).

    Weight layout ``(O, kH, kW, I)`` (MLX NHWC convention). Implemented as an
    explicit im2col GEMM (fp32) — the same arithmetic the MLX kernel computes,
    up to accumulation order.
    """
    x = x_nhwc.astype(np.float32)
    w = weight_ohwi.astype(np.float32)
    B, H, W, Cin = x.shape
    out_ch, kH, kW, Wi = w.shape
    if Wi != Cin:
        raise ValueError(f"conv channel mismatch: input {Cin} vs weight {Wi}")
    pad = int(padding)
    dil = int(dilation)
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")
    Hp, Wp = x.shape[1], x.shape[2]
    Hout = Hp - dil * (kH - 1) - 1 + 1
    Wout = Wp - dil * (kW - 1) - 1 + 1
    # im2col: (B, Hout, Wout, kH*kW*Cin)
    cols = np.empty((B, Hout, Wout, kH * kW * Cin), dtype=np.float32)
    c = 0
    for kh in range(kH):
        for kw in range(kW):
            patch = x[:, kh * dil : kh * dil + Hout, kw * dil : kw * dil + Wout, :]
            cols[..., c * Cin : (c + 1) * Cin] = patch
            c += 1
    # weight reordered to (kH*kW*Cin, out_ch) matching the im2col channel order.
    w_re = np.transpose(w, (1, 2, 3, 0)).reshape(kH * kW * Cin, out_ch)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = cols.reshape(B * Hout * Wout, kH * kW * Cin) @ w_re
    out = out.reshape(B, Hout, Wout, out_ch)
    if bias is not None:
        out = out + bias.astype(np.float32)
    return out.astype(np.float32)


def pixel_shuffle_2x_nhwc(x_nhwc: np.ndarray) -> np.ndarray:
    """Channel-FIRST PixelShuffle(2) for NHWC (matches the canonical MLX helper).

    Reshape ``(B,H,W,out_C,2,2)`` then transpose ``(0,1,4,2,5,3)``. This is the
    PyTorch-byte-stable convention (0.0 drift per the PR95 canonical helper).
    """
    B, H, W, C = x_nhwc.shape
    if C % 4:
        raise ValueError(f"channels {C} must be divisible by 4 for 2x pixel shuffle")
    out_c = C // 4
    y = x_nhwc.reshape(B, H, W, out_c, 2, 2)
    y = np.transpose(y, (0, 1, 4, 2, 5, 3))
    return np.ascontiguousarray(y.reshape(B, H * 2, W * 2, out_c)).astype(np.float32)


def bilinear_resize2x_nhwc(x_nhwc: np.ndarray) -> np.ndarray:
    """2x bilinear (align_corners=False) for NHWC — matches the canonical MLX helper.

    The MLX helper exploits the closed-form scale=2 align_corners=False weights
    ``0.75, 0.25``: width pass (even/odd interleaved), then height pass. This
    numpy port reproduces that EXACT sequence so the skip identity matches.
    """
    x = x_nhwc.astype(np.float32)
    left = np.concatenate([x[:, :, :1, :], x[:, :, :-1, :]], axis=2)
    right = np.concatenate([x[:, :, 1:, :], x[:, :, -1:, :]], axis=2)
    even_w = x * 0.75 + left * 0.25
    odd_w = x * 0.75 + right * 0.25
    B, H, W, C = x.shape
    width_up = np.stack([even_w, odd_w], axis=3).reshape(B, H, W * 2, C)
    top = np.concatenate([width_up[:, :1, :, :], width_up[:, :-1, :, :]], axis=1)
    bottom = np.concatenate([width_up[:, 1:, :, :], width_up[:, -1:, :, :]], axis=1)
    even_h = width_up * 0.75 + top * 0.25
    odd_h = width_up * 0.75 + bottom * 0.25
    return np.stack([even_h, odd_h], axis=2).reshape(B, H * 2, W * 2, C).astype(np.float32)


def bilinear_resize_to_nhwc(
    x_nhwc: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    """General bilinear resize (align_corners=False) — matches PyTorch F.interpolate.

    Used by the inflate runtime to map the 384x512 render to the camera
    1164x874 (W,H) frame the evaluator reads. Mirrors PyTorch's
    ``F.interpolate(size=..., mode='bilinear', align_corners=False)`` so the
    inflated frames match a torch reference inflate byte-for-byte after the
    round/clamp/uint8 cast.
    """
    x = x_nhwc.astype(np.float32)
    B, H, W, C = x.shape
    if target_h == H and target_w == W:
        return x
    h_scale = H / target_h
    w_scale = W / target_w
    hy = (np.arange(target_h, dtype=np.float32) + 0.5) * h_scale - 0.5
    wx = (np.arange(target_w, dtype=np.float32) + 0.5) * w_scale - 0.5
    hy = np.clip(hy, 0.0, H - 1.0)
    wx = np.clip(wx, 0.0, W - 1.0)
    h_lo = np.floor(hy).astype(np.int64)
    h_hi = np.minimum(h_lo + 1, H - 1)
    w_lo = np.floor(wx).astype(np.int64)
    w_hi = np.minimum(w_lo + 1, W - 1)
    h_frac = (hy - np.floor(hy)).astype(np.float32)
    w_frac = (wx - np.floor(wx)).astype(np.float32)
    tl = x[:, h_lo[:, None], w_lo[None, :], :]
    tr = x[:, h_lo[:, None], w_hi[None, :], :]
    bl = x[:, h_hi[:, None], w_lo[None, :], :]
    br = x[:, h_hi[:, None], w_hi[None, :], :]
    wf = w_frac[None, None, :, None]
    hf = h_frac[None, :, None, None]
    top = tl * (1.0 - wf) + tr * wf
    bot = bl * (1.0 - wf) + br * wf
    return (top * (1.0 - hf) + bot * hf).astype(np.float32)


# --------------------------------------------------------------------------
# the decoder forward (op-for-op with _decode_with_film)
# --------------------------------------------------------------------------


def _norm_pose(
    pose6: np.ndarray, cfg: CapstoneDecodeConfig
) -> np.ndarray:
    if not cfg.pose_normalize:
        return pose6.astype(np.float32)
    mean = np.asarray(cfg.pose_mean, dtype=np.float32)
    std = np.asarray(cfg.pose_std, dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return ((pose6.astype(np.float32) - mean) / std).astype(np.float32)


def _pose_film(
    weights: dict[str, np.ndarray], prefix: str, norm_pose: np.ndarray, channels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce ``_PoseFiLM.__call__``: (gamma, beta) per channel.

    h = sin(fc1(pose)); gb = fc2(h); gamma = 1 + tanh(gb[:, :C]); beta = gb[:, C:].
    """
    h = _sin(linear(norm_pose, weights[f"{prefix}.fc1.weight"], weights[f"{prefix}.fc1.bias"]))
    gb = linear(h, weights[f"{prefix}.fc2.weight"], weights[f"{prefix}.fc2.bias"])
    gamma_pre = gb[:, :channels]
    beta = gb[:, channels:]
    gamma = 1.0 + np.tanh(gamma_pre)
    return gamma.astype(np.float32), beta.astype(np.float32)


def _features_nhwc(
    z_q: np.ndarray, weights: dict[str, np.ndarray], cfg: CapstoneDecodeConfig
) -> np.ndarray:
    """The shared decoder feature (pre-FiLM): exact ``HNeRVDecoderMLX.features_nhwc``."""
    ch = cfg.channels()
    B = z_q.shape[0]
    # stem (Linear) -> (B, channels[0]*base_h*base_w)
    x = linear(z_q, weights["stem.weight"], weights["stem.bias"])
    x = x.reshape(B, ch[0], cfg.base_h, cfg.base_w)
    x = np.transpose(x, (0, 2, 3, 1))  # NHWC
    x = _sin(x)
    # 6 upsample blocks
    for i in range(6):
        in_c = int(x.shape[-1])
        identity = bilinear_resize2x_nhwc(x)
        skip_w = weights.get(f"blocks.{i}.skip_conv.weight")
        if skip_w is not None:
            identity = conv2d_nhwc(
                identity, skip_w, weights[f"blocks.{i}.skip_conv.bias"], padding=0
            )
        decoded = pixel_shuffle_2x_nhwc(
            conv2d_nhwc(
                x, weights[f"blocks.{i}.conv.weight"], weights[f"blocks.{i}.conv.bias"],
                padding=1,
            )
        )
        x = _sin(decoded + identity)
        del in_c
    # refine
    refined = conv2d_nhwc(x, weights["refine0.weight"], weights["refine0.bias"], padding=2, dilation=2)
    refined = conv2d_nhwc(refined, weights["refine1.weight"], weights["refine1.bias"], padding=1)
    feat = x + 0.1 * _sin(refined)
    return feat.astype(np.float32)


def numpy_decode_pair(
    z_q: np.ndarray,
    pose6: np.ndarray | None,
    weights: dict[str, np.ndarray],
    cfg: CapstoneDecodeConfig,
) -> np.ndarray:
    """Render ONE batch of pairs from per-pair quantized latents ``z_q``.

    Args:
        z_q: ``(B, latent_dim)`` quantized latent per pair (= ``codebook[index]``).
        pose6: ``(B, 6)`` stored GT pose for the FiLM (or None for identity FiLM).
        weights: name->fp32 array of decoder + FiLM params (the FULL render basis;
            see :func:`weights_from_archive_dict`).
        cfg: :class:`CapstoneDecodeConfig`.

    Returns:
        ``(B, 2, 3, 384, 512)`` float32 N2CHW render in ``[0, 255]`` — exactly
        what ``CapstoneVqNervBundle._decode_with_film`` produces.
    """
    feat = _features_nhwc(np.asarray(z_q, dtype=np.float32), weights, cfg)
    fc = cfg.channels()[-1]
    if cfg.film_enabled and pose6 is not None and "pose_film0.fc1.weight" in weights:
        pn = _norm_pose(np.asarray(pose6, dtype=np.float32), cfg)
        g0, b0 = _pose_film(weights, "pose_film0", pn, fc)
        g1, b1 = _pose_film(weights, "pose_film1", pn, fc)
        feat0 = g0[:, None, None, :] * feat + b0[:, None, None, :]
        feat1 = g1[:, None, None, :] * feat + b1[:, None, None, :]
    else:
        feat0 = feat
        feat1 = feat
    f0 = _sigmoid(conv2d_nhwc(feat0, weights["rgb_0.weight"], weights["rgb_0.bias"], padding=1)) * 255.0
    f1 = _sigmoid(conv2d_nhwc(feat1, weights["rgb_1.weight"], weights["rgb_1.bias"], padding=1)) * 255.0
    # (B, H, W, 3) each -> stack to (B, 2, H, W, 3) -> N2CHW
    pair_hwc = np.stack([f0, f1], axis=1)  # (B, 2, H, W, 3)
    pair = np.transpose(pair_hwc, (0, 1, 4, 2, 3))  # (B, 2, 3, H, W)
    return pair.astype(np.float32)


# --------------------------------------------------------------------------
# weight extraction helpers (bundle <-> numpy dict)
# --------------------------------------------------------------------------


def full_render_weights_from_bundle(bundle: Any) -> dict[str, np.ndarray]:
    """Extract the FULL render-basis weight dict (decoder + FiLM) from a bundle.

    The decoder params keep their ``decoder.parameters()`` names (``stem.weight``,
    ``blocks.i.conv.weight``, ``refine0.weight``, ``rgb_0.weight`` ...). The FiLM
    params are prefixed ``pose_film0.`` / ``pose_film1.`` so the numpy decode
    can find them. THIS is the complete set the archive must carry for a
    FiLM-enabled bundle (the decoder-only export is insufficient — it would drop
    the per-frame FiLM the render depends on).
    """
    from mlx.utils import tree_flatten  # local import: MLX only on the train host

    out: dict[str, np.ndarray] = {}
    for k, v in tree_flatten(bundle.decoder.parameters()):
        out[k] = np.asarray(v, dtype=np.float32)
    if getattr(bundle, "film_enabled", False) and hasattr(bundle, "pose_film0"):
        for prefix in ("pose_film0", "pose_film1"):
            film = getattr(bundle, prefix)
            for k, v in tree_flatten(film.parameters()):
                out[f"{prefix}.{k}"] = np.asarray(v, dtype=np.float32)
    return out


def decode_config_from_bundle(bundle: Any) -> CapstoneDecodeConfig:
    """Build the numpy :class:`CapstoneDecodeConfig` from a live MLX bundle."""
    cfg = bundle.cfg
    return CapstoneDecodeConfig(
        base_channels=int(cfg.base_channels),
        latent_dim=int(cfg.latent_dim),
        codebook_size=int(cfg.codebook_size),
        base_h=int(bundle.decoder.base_h),
        base_w=int(bundle.decoder.base_w),
        film_enabled=bool(getattr(bundle, "film_enabled", False)),
        pose_normalize=bool(cfg.pose_normalize),
        pose_mean=tuple(float(v) for v in np.asarray(bundle._pose_mean)),
        pose_std=tuple(float(v) for v in np.asarray(bundle._pose_std)),
    )


__all__ = [
    "POSE_DIM",
    "CapstoneDecodeConfig",
    "bilinear_resize2x_nhwc",
    "bilinear_resize_to_nhwc",
    "conv2d_nhwc",
    "decode_config_from_bundle",
    "full_render_weights_from_bundle",
    "linear",
    "numpy_decode_pair",
    "pixel_shuffle_2x_nhwc",
]
