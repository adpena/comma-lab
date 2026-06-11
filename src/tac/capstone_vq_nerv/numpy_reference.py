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

    ``hinerv_grid_pe`` (default ``False``, byte-identical-when-off): the HiNeRV
    delta over HNeRV. When enabled, a DETERMINISTIC multi-frequency sinusoidal
    coordinate grid (computed at inflate from coordinates — ~0 stored bytes) is
    projected by a tiny learned linear (``grid_pe_proj.{weight,bias}``, the only
    new stored params; ``channels[0] x pe_dim``) and ADDED to the stem feature
    BEFORE the ``sin`` activation. This injects the spatial inductive bias the
    pure latent->Linear stem lacks (the ~72.3% BD-rate HiNeRV lever's grid-PE
    half; the bilinear-skip half is ALREADY structurally present in every
    upsample block). ``grid_pe_num_freqs`` controls the encoding bandwidth
    (``pe_dim = 4 * num_freqs`` = sin/cos x {x, y} x num_freqs).
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
    # HiNeRV grid positional-encoding (opt-in; default-off = byte-identical).
    hinerv_grid_pe: bool = False
    grid_pe_num_freqs: int = 4
    # L1 weight-tie depth (opt-in; ``<=1`` = no tie = byte-identical). When ``>=2``
    # the leading ``tie_depth`` upsample blocks (all ``base_ch->base_ch``) share the
    # ONE stored ``tied_conv.{weight,bias}`` + per-stage ``tied_stage_film.{i}.*``
    # symmetry-breakers, instead of their own ``blocks.{i}.conv.*``. The inflate
    # reads ``tie_depth`` from the config sidecar and dispatches the leading stages
    # to the shared conv (op-for-op with the MLX ``_tied_block_forward``).
    tie_depth: int = 0

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


def _cubic_conv_weights(t: np.ndarray, a: float = -0.75) -> np.ndarray:
    """PyTorch bicubic cubic-convolution kernel weights for fractional offsets ``t``.

    Matches ``aten::upsample_bicubic2d`` (Keys cubic convolution, a=-0.75). For each
    output sample the 4 input taps at offsets ``(t+1, t, 1-t, 2-t)`` get weights from
    the piecewise cubic. Returns ``(N, 4)`` weights summing to 1, tap order
    ``[x-1, x, x+1, x+2]`` (i.e. offsets ``1+t, t, 1-t, 2-t``).
    """
    t = np.asarray(t, dtype=np.float64)

    def w(d: np.ndarray) -> np.ndarray:
        d = np.abs(d)
        out = np.zeros_like(d)
        m1 = d <= 1.0
        m2 = (d > 1.0) & (d < 2.0)
        out[m1] = ((a + 2.0) * d[m1] - (a + 3.0)) * d[m1] * d[m1] + 1.0
        out[m2] = (((d[m2] - 5.0) * d[m2] + 8.0) * d[m2] - 4.0) * a
        return out

    w0 = w(t + 1.0)
    w1 = w(t)
    w2 = w(1.0 - t)
    w3 = w(2.0 - t)
    return np.stack([w0, w1, w2, w3], axis=-1)


def bicubic_resize_to_nhwc(
    x_nhwc: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    """Bicubic resize (align_corners=False) — matches PyTorch ``F.interpolate(mode='bicubic')``.

    [A3] This is the CAMERA upsample PR95 uses (``score.py::_decoded_to_camera`` +
    ``stages/common.py`` both ``F.interpolate(..., mode='bicubic', align_corners=False)``).
    The inflate runtime previously used BILINEAR for the 384x512 -> camera upscale,
    which diverges from the eval roundtrip — so the advisory could not predict the
    real ``inflate.sh -> evaluate.py``. This is the byte-stable bicubic port.

    Op-for-op with ``aten::upsample_bicubic2d``: align_corners=False source mapping
    ``src = (dst + 0.5) * (in/out) - 0.5``, Keys cubic convolution kernel a=-0.75,
    4x4 separable taps with edge-clamped indices. Matches torch to fp32 eps.
    """
    x = x_nhwc.astype(np.float64)
    B, H, W, C = x.shape
    if target_h == H and target_w == W:
        return x.astype(np.float32)
    h_scale = H / target_h
    w_scale = W / target_w
    # align_corners=False source coordinates.
    hy = (np.arange(target_h, dtype=np.float64) + 0.5) * h_scale - 0.5
    wx = (np.arange(target_w, dtype=np.float64) + 0.5) * w_scale - 0.5
    h_floor = np.floor(hy)
    w_floor = np.floor(wx)
    h_t = hy - h_floor
    w_t = wx - w_floor
    h_w = _cubic_conv_weights(h_t)  # (target_h, 4)
    w_w = _cubic_conv_weights(w_t)  # (target_w, 4)
    h_idx0 = h_floor.astype(np.int64)
    w_idx0 = w_floor.astype(np.int64)
    # 4 tap indices per output, edge-clamped to [0, H-1] / [0, W-1].
    h_taps = np.clip(h_idx0[:, None] + np.array([-1, 0, 1, 2]), 0, H - 1)  # (target_h, 4)
    w_taps = np.clip(w_idx0[:, None] + np.array([-1, 0, 1, 2]), 0, W - 1)  # (target_w, 4)
    # Horizontal pass: gather W taps, weight, sum -> (B, H, target_w, C).
    # x_w[b, h, j, k, c] = x[b, h, w_taps[j, k], c]
    gathered_w = x[:, :, w_taps, :]  # (B, H, target_w, 4, C)
    horiz = np.einsum("bhjkc,jk->bhjc", gathered_w, w_w)  # (B, H, target_w, C)
    # Vertical pass: gather H taps, weight, sum -> (B, target_h, target_w, C).
    gathered_h = horiz[:, h_taps, :, :]  # (B, target_h, 4, target_w, C)
    out = np.einsum("bikjc,ik->bijc", gathered_h, h_w)  # (B, target_h, target_w, C)
    return out.astype(np.float32)


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


def grid_positional_encoding(
    base_h: int, base_w: int, num_freqs: int
) -> np.ndarray:
    """DETERMINISTIC multi-frequency sinusoidal coordinate grid (HiNeRV grid-PE).

    Returns ``(base_h * base_w, pe_dim)`` fp32 where ``pe_dim = 4 * num_freqs``
    (``sin`` and ``cos`` of ``{x, y}`` over ``num_freqs`` geometric frequencies).

    The grid is computed PURELY from coordinates (no stored values — the inflate
    runtime regenerates it from ``base_h``/``base_w``/``num_freqs``, ~0 archive
    bytes), so it is identical on every host and between the MLX and numpy paths.

    Coordinate convention (matches the stem reshape ``(C, base_h, base_w)`` then
    NHWC transpose -> rows iterate ``base_h`` first, then ``base_w``):

        y in linspace(0, 1, base_h), x in linspace(0, 1, base_w), row-major flat.

    Frequencies are ``2**k * pi`` for ``k in [0, num_freqs)`` (NeRF-style). The
    feature order is ``[sin(f0 x), cos(f0 x), sin(f0 y), cos(f0 y), sin(f1 x),
    ...]`` so MLX + numpy build the SAME column layout op-for-op.
    """
    bh, bw, nf = int(base_h), int(base_w), int(num_freqs)
    if bh < 1 or bw < 1:
        raise ValueError(f"base grid must be positive; got ({bh}, {bw})")
    if nf < 1:
        raise ValueError(f"grid_pe_num_freqs must be >= 1; got {nf}")
    ys = np.linspace(0.0, 1.0, bh, dtype=np.float32)
    xs = np.linspace(0.0, 1.0, bw, dtype=np.float32)
    # row-major (y outer, x inner) to match the NHWC stem layout.
    yy, xx = np.meshgrid(ys, xs, indexing="ij")  # (bh, bw) each
    y_flat = yy.reshape(-1).astype(np.float32)  # (bh*bw,)
    x_flat = xx.reshape(-1).astype(np.float32)
    cols: list[np.ndarray] = []
    for k in range(nf):
        freq = np.float32((2.0**k) * np.pi)
        cols.append(np.sin(freq * x_flat))
        cols.append(np.cos(freq * x_flat))
        cols.append(np.sin(freq * y_flat))
        cols.append(np.cos(freq * y_flat))
    return np.stack(cols, axis=1).astype(np.float32)  # (bh*bw, 4*nf)


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


def _grid_pe_weight_keys(
    weights: dict[str, np.ndarray]
) -> tuple[str | None, str | None]:
    """Resolve the grid-PE projection (weight, bias) keys in the render dict.

    The MLX export emits ``grid_pe_proj.proj.weight`` / ``grid_pe_proj.proj.bias``
    (the ``_GridPE`` module nests the projection in ``self.proj = nn.Linear``,
    so ``tree_flatten`` keeps the ``.proj.`` segment, exactly as FiLM keeps
    ``fc1``/``fc2``). The flat ``grid_pe_proj.weight`` form is accepted for
    forward-compatibility. Returns ``(None, None)`` when no grid-PE proj exists.
    """
    if "grid_pe_proj.proj.weight" in weights:
        return "grid_pe_proj.proj.weight", "grid_pe_proj.proj.bias"
    if "grid_pe_proj.weight" in weights:
        return "grid_pe_proj.weight", "grid_pe_proj.bias"
    return None, None


def _features_nhwc(
    z_q: np.ndarray, weights: dict[str, np.ndarray], cfg: CapstoneDecodeConfig
) -> np.ndarray:
    """The shared decoder feature (pre-FiLM): exact ``HNeRVDecoderMLX.features_nhwc``."""
    ch = cfg.channels()
    B = z_q.shape[0]
    # stem (Linear) -> (B, channels[0]*base_h*base_w)
    x = linear(z_q, weights["stem.weight"], weights["stem.bias"])
    x = x.reshape(B, ch[0], cfg.base_h, cfg.base_w)
    x = np.transpose(x, (0, 2, 3, 1))  # NHWC -> (B, base_h, base_w, channels[0])
    # HiNeRV grid-PE: add a DETERMINISTIC coordinate grid (projected by the small
    # learned grid_pe_proj) to the stem feature BEFORE sin. The grid is regenerated
    # from coords here (0 stored bytes); only the projection weights are stored.
    # Op-for-op with the MLX bundle's _decode_with_film grid-PE injection.
    #
    # The exported key is ``grid_pe_proj.proj.weight`` (the MLX ``_GridPE`` wraps the
    # projection in ``self.proj = nn.Linear``, so ``tree_flatten`` emits the nested
    # ``.proj.`` name, mirroring how FiLM emits ``pose_film0.fc1.weight``). The
    # legacy flat ``grid_pe_proj.weight`` is also accepted for forward-compat.
    pe_w_key, pe_b_key = _grid_pe_weight_keys(weights)
    if cfg.hinerv_grid_pe and pe_w_key is not None:
        pe = grid_positional_encoding(
            cfg.base_h, cfg.base_w, cfg.grid_pe_num_freqs
        )  # (base_h*base_w, pe_dim)
        pe_proj = linear(
            pe, weights[pe_w_key], weights.get(pe_b_key) if pe_b_key else None
        )  # (base_h*base_w, channels[0])
        pe_proj = pe_proj.reshape(cfg.base_h, cfg.base_w, ch[0])[None, ...]
        x = x + pe_proj
    x = _sin(x)
    # 6 upsample blocks. With the L1 weight-tie (tie_depth>=2), the leading
    # ``tie_depth`` blocks use the SHARED ``tied_conv`` (+ per-stage FiLM) instead
    # of their own ``blocks.{i}.conv``; the rest use their own convs. Op-for-op with
    # the MLX ``CapstoneVqNervBundle._tied_block_forward`` / per-block forward.
    tie_depth = int(cfg.tie_depth)
    has_tie = tie_depth >= 2 and "tied_conv.weight" in weights
    for i in range(6):
        identity = bilinear_resize2x_nhwc(x)
        if has_tie and i < tie_depth:
            # Tied leading stage: shared conv (no skip_conv since base_ch->base_ch),
            # then the per-stage FiLM symmetry-breaker on the pre-sin sum (stage 0
            # has no FiLM = identity).
            decoded = pixel_shuffle_2x_nhwc(
                conv2d_nhwc(
                    x, weights["tied_conv.weight"], weights["tied_conv.bias"],
                    padding=1,
                )
            )
            y = decoded + identity
            if i >= 1:
                gamma_delta = weights[f"tied_stage_films.{i - 1}.gamma_delta"]
                beta = weights[f"tied_stage_films.{i - 1}.beta"]
                y = (1.0 + gamma_delta) * y + beta
            x = _sin(y)
            continue
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
    tie_depth = int(getattr(bundle, "tie_depth", 0) or 0)
    has_tie = tie_depth >= 2 and hasattr(bundle, "tied_conv")
    for k, v in tree_flatten(bundle.decoder.parameters()):
        # L1 weight-tie: the leading ``tie_depth`` blocks' own per-block conv weights
        # are REPLACED by the shared tied_conv at decode, so they are dead bytes —
        # DROP them from the export (the whole rate point). The decoder still holds
        # them (it is the shared backbone), but the archive must not carry them.
        if has_tie:
            dropped = False
            for i in range(tie_depth):
                if k.startswith(f"blocks.{i}.conv."):
                    dropped = True
                    break
            if dropped:
                continue
        out[k] = np.asarray(v, dtype=np.float32)
    # L1 weight-tie: export the ONE shared conv + the per-stage FiLM symmetry-breakers.
    if has_tie:
        for k, v in tree_flatten(bundle.tied_conv.parameters()):
            out[f"tied_conv.{k}"] = np.asarray(v, dtype=np.float32)
        for stage_idx, film in enumerate(bundle.tied_stage_films):
            for k, v in tree_flatten(film.parameters()):
                out[f"tied_stage_films.{stage_idx}.{k}"] = np.asarray(v, dtype=np.float32)
    if getattr(bundle, "film_enabled", False) and hasattr(bundle, "pose_film0"):
        for prefix in ("pose_film0", "pose_film1"):
            film = getattr(bundle, prefix)
            for k, v in tree_flatten(film.parameters()):
                out[f"{prefix}.{k}"] = np.asarray(v, dtype=np.float32)
    # HiNeRV grid-PE projection (the only new stored params; tiny: channels[0] x pe_dim).
    if getattr(bundle, "hinerv_grid_pe", False) and hasattr(bundle, "grid_pe_proj"):
        for k, v in tree_flatten(bundle.grid_pe_proj.parameters()):
            out[f"grid_pe_proj.{k}"] = np.asarray(v, dtype=np.float32)
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
        hinerv_grid_pe=bool(getattr(cfg, "hinerv_grid_pe", False)),
        grid_pe_num_freqs=int(getattr(cfg, "grid_pe_num_freqs", 4)),
        tie_depth=int(getattr(bundle, "tie_depth", 0) or 0),
    )


__all__ = [
    "POSE_DIM",
    "CapstoneDecodeConfig",
    "bicubic_resize_to_nhwc",
    "bilinear_resize2x_nhwc",
    "bilinear_resize_to_nhwc",
    "conv2d_nhwc",
    "decode_config_from_bundle",
    "full_render_weights_from_bundle",
    "grid_positional_encoding",
    "linear",
    "numpy_decode_pair",
    "pixel_shuffle_2x_nhwc",
]
