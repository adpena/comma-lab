# SPDX-License-Identifier: MIT
"""Lever-C: small per-pair-latent CONVOLUTIONAL frame decoder (HNeRV-class) — task #62.

THE #57/#61 WALL this addresses: the amortized COORDINATE-INR frame1 carrier cannot hold the SegNet
argmax (d_seg) AND the PoseNet pose luma (d_pose) JOINTLY at score-native byte budgets — the two
objectives are ANTAGONISTIC at coordinate-MLP capacity (pose-trained INR → d_seg 0.733; seg-trained
palette → d_pose 12.14; cheapest joint frame1 > 400 KB). The diagnosed cause: a smooth coordinate-MLP
cannot represent the SHARP, high-frequency SegNet argmax boundaries.

THE LEVER-C BUILD (the structurally-expressive carrier #57/#61 named): a per-pair latent ``z_p`` →
small spatial seed → N conv-blocks (``Conv(in, out*4, 3x3) + PixelShuffle(2) + bilinear-skip + sin``,
the PR95 L18 verified leaderboard decoder block) → camera-res RGB. The decoder weights are SHARED
across pairs (amortized); the per-pair latents carry pair identity (PR95 L19: ~94% weights / ~6%
latents). This is the HNeRV-class block family — it CAN represent sharp boundaries the coordinate-MLP
cannot, so it is the honest test of whether the dual constraint is jointly satisfiable.

THE SCORER FACTS (upstream/modules.py, verified):
  * SegNet reads ONLY the last frame (``x[:, -1, ...]`` = frame1) → frame1 carries d_seg.
  * PoseNet reads BOTH frames → frame1 ALSO carries pose; frame0 is SegNet-invisible (pose-only).
  * So frame1 must JOINTLY satisfy d_seg (its SegNet argmax == GT) AND d_pose (its luma gives PoseNet
    the right 6-dim pose). The lever-C decoder generates frame1 (and optionally frame0 for pose).

COMPUTE-SUBSTRATE LAW (CLAUDE.md): the numpy forward here is the PORTABILITY reference (the inflate-time
decoder is pure numpy, scorer-free, deterministic). MLX/torch is the training fast path. NO MPS.
d_seg/d_pose read on the exact frozen CPU-torch scorer; GT via ``frame_utils.yuv420_to_rgb`` ONLY.
Evidence ``[macOS-MLX research-signal]`` for the decoder forward; ``[local CPU-torch advisory]`` once
scored. Non-promotable.

NO-FAKE (class 2): the numpy forward ACTUALLY runs the conv stack (not a stored per-pair frame table);
the reconstructed frame ACTUALLY depends on (pair latent, x, y); the byte cost is the brotli of the
ACTUAL quantized weights+latents (not a constant). A stub returning a constant frame FAILS the tests
(which assert the output varies across pairs and pixels, and that a constant decoder cannot reduce
either d_seg or d_pose on the real scorers).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")

# Camera-native resolution the inflate-time decoder must emit (the scored frame).
CAMERA_H, CAMERA_W = 874, 1164


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


@dataclass(frozen=True)
class ConvDecoderConfig:
    """Architecture hyperparameters that fully determine the decoder shape + byte cost.

    The decoder maps a per-pair latent ``z_p`` (``latent_dim``) → a spatial seed
    (``seed_ch`` × ``seed_h`` × ``seed_w``) → ``len(stage_channels)`` PixelShuffle(2) conv-blocks →
    3-channel RGB at the final resolution. The capacity knobs are ``latent_dim`` / ``seed_ch`` /
    ``stage_channels``. The number of stages must lift the seed to >= camera resolution; the final
    decode bilinearly resizes the block output to exactly (CAMERA_H, CAMERA_W).

    ``num_pairs`` is the data shape. Byte cost = int8+brotli of (shared decoder weights + per-pair
    latents) + fp16 per-tensor dequant scales.
    """

    num_pairs: int
    latent_dim: int = 24
    seed_ch: int = 32
    seed_h: int = 6
    seed_w: int = 8
    stage_channels: tuple[int, ...] = (32, 24, 16, 12)  # per PixelShuffle(2) stage; 4 stages = 16x up
    n_channels: int = 3
    quant_bits: int = 8

    def to_dict(self) -> dict:
        return {
            "num_pairs": self.num_pairs,
            "latent_dim": self.latent_dim,
            "seed_ch": self.seed_ch,
            "seed_h": self.seed_h,
            "seed_w": self.seed_w,
            "stage_channels": list(self.stage_channels),
            "n_channels": self.n_channels,
            "quant_bits": self.quant_bits,
        }

    def final_hw(self) -> tuple[int, int]:
        """Block-stack output resolution before the final bilinear resize to camera res."""

        return self.seed_h * (2 ** len(self.stage_channels)), self.seed_w * (2 ** len(self.stage_channels))


# ---------------------------------------------------------------------------
# Pure-numpy primitives (the portable inflate-time reference; scorer-free).
# ---------------------------------------------------------------------------
def _sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SAME-padding 3x3 conv. x:(Cin,H,W), w:(Cout,Cin,3,3), b:(Cout,). Returns (Cout,H,W).

    Pure numpy (im2col), float64 — the portable reference. Deterministic; mirrors torch conv2d
    with padding=1, stride=1.
    """

    cin, h, w_dim = x.shape
    cout = w.shape[0]
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="constant")
    # im2col: (Cin*9, H*W)
    cols = np.empty((cin * 9, h * w_dim), dtype=np.float64)
    idx = 0
    for ky in range(3):
        for kx in range(3):
            patch = xp[:, ky : ky + h, kx : kx + w_dim]  # (Cin,H,W)
            cols[idx * cin : (idx + 1) * cin, :] = patch.reshape(cin, -1)
            idx += 1
    # weight reorder to match the (ky,kx,cin) col order: w is (Cout,Cin,3,3)
    wflat = np.empty((cout, cin * 9), dtype=np.float64)
    idx = 0
    for ky in range(3):
        for kx in range(3):
            wflat[:, idx * cin : (idx + 1) * cin] = w[:, :, ky, kx]
            idx += 1
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = np.ascontiguousarray(wflat) @ np.ascontiguousarray(cols) + b[:, None]
    return out.reshape(cout, h, w_dim)


def _pixel_shuffle(x: np.ndarray, r: int) -> np.ndarray:
    """PixelShuffle upscale by r. x:(C*r*r, H, W) -> (C, H*r, W*r). Mirrors torch.nn.PixelShuffle."""

    c_rr, h, w = x.shape
    c = c_rr // (r * r)
    x = x.reshape(c, r, r, h, w)
    x = x.transpose(0, 3, 1, 4, 2)  # (c, h, r, w, r)
    return x.reshape(c, h * r, w * r)


def _bilinear_resize(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize (C,H,W) -> (C,out_h,out_w), align_corners=False (matches torch default)."""

    c, h, w = x.shape
    if (h, w) == (out_h, out_w):
        return x
    # align_corners=False sampling grid
    ys = (np.arange(out_h, dtype=np.float64) + 0.5) * (h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float64) + 0.5) * (w / out_w) - 0.5
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    out = (
        x[:, y0][:, :, x0] * (1 - wy)[None] * (1 - wx)[None]
        + x[:, y0][:, :, x1] * (1 - wy)[None] * wx[None]
        + x[:, y1][:, :, x0] * wy[None] * (1 - wx)[None]
        + x[:, y1][:, :, x1] * wy[None] * wx[None]
    )
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def numpy_reference_forward(
    params: dict[str, np.ndarray],
    cfg: ConvDecoderConfig,
    latent: np.ndarray,
) -> np.ndarray:
    """Pure-numpy mirror of ``TorchConvPairDecoder.forward`` for one pair latent.

    Returns (n_channels, final_h, final_w) RGB in [0,255] at the block-stack resolution (BEFORE the
    final camera-res resize — :func:`decoder_frame` does the resize). float64 accumulation.

    Params (the shared decoder weights, NOT the per-pair latents):
      * ``seed.weight`` (seed_ch*seed_h*seed_w, latent_dim), ``seed.bias``
      * ``stage{i}.weight`` (out_ch*4, in_ch, 3, 3), ``stage{i}.bias`` for each PixelShuffle stage
      * ``stage{i}.skip`` (out_ch, in_ch, 1, 1), ``stage{i}.skip_bias`` — 1x1 conv for the
        bilinear-skip path (projects channels so the upsampled skip can be added)
      * ``out.weight`` (n_channels, last_ch, 3, 3), ``out.bias``
    """

    p = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}
    z = np.asarray(latent, dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        seed = p["seed.weight"] @ z + p["seed.bias"]  # (seed_ch*seed_h*seed_w,)
        h = seed.reshape(cfg.seed_ch, cfg.seed_h, cfg.seed_w)

        for i, _out_ch in enumerate(cfg.stage_channels):
            conv = _conv3x3(h, p[f"stage{i}.weight"], p[f"stage{i}.bias"])  # (out_ch*4, H, W)
            up = _pixel_shuffle(conv, 2)  # (out_ch, H*2, W*2)
            # bilinear-skip: upsample input h to the new size + 1x1 channel projection, then add.
            skip_in = _bilinear_resize(h, up.shape[1], up.shape[2])  # (in_ch, H*2, W*2)
            sk_w = p[f"stage{i}.skip"]  # (out_ch, in_ch, 1, 1)
            skip = np.einsum("oixy,ihw->ohw", sk_w, skip_in) + p[f"stage{i}.skip_bias"][:, None, None]
            h = _sin(up + skip)  # sin activation (PR95 L18)

        rgb01 = _sigmoid(_conv3x3(h, p["out.weight"], p["out.bias"]))
    return (rgb01 * 255.0)  # (n_channels, final_h, final_w)


def decoder_frame(
    params: dict[str, np.ndarray],
    cfg: ConvDecoderConfig,
    latents: np.ndarray,
    pair_idx: int,
    out_h: int = CAMERA_H,
    out_w: int = CAMERA_W,
) -> np.ndarray:
    """Per-pair decoder RGB frame (H, W, 3) uint8 — the numpy-portable inflate-time decode.

    ``latents`` is (num_pairs, latent_dim). Runs the conv stack for ``pair_idx`` then bilinearly
    resizes the block output to the camera resolution (the scored quantity is the camera-res frame).
    SCORER-FREE.
    """

    z = np.asarray(latents)[pair_idx]
    rgb_chw = numpy_reference_forward(params, cfg, z)  # (3, fh, fw)
    rgb_chw = _bilinear_resize(rgb_chw, out_h, out_w)  # (3, H, W)
    rgb_hwc = np.transpose(rgb_chw, (1, 2, 0))
    return np.clip(np.round(rgb_hwc), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Byte accounting — quantized shared weights + per-pair latents (brotli q11).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConvDecoderByteAccount:
    weight_bytes: int   # quantized shared decoder weights, brotli
    latent_bytes: int   # quantized per-pair latents, brotli
    scale_bytes: int    # fp16 per-tensor dequant scales
    total_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "weight_bytes": self.weight_bytes,
            "latent_bytes": self.latent_bytes,
            "scale_bytes": self.scale_bytes,
            "total_bytes": self.total_bytes,
        }


def _quantize_symmetric(arr: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    a = np.asarray(arr, dtype=np.float64)
    qmax = float(2 ** (bits - 1) - 1)
    amax = float(np.max(np.abs(a))) if a.size else 0.0
    scale = (amax / qmax) if amax > 0 else 1.0
    codes = np.clip(np.round(a / scale), -qmax, qmax).astype(np.int16)
    return codes, scale


def quantize_params(params: dict[str, np.ndarray], bits: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    codes: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    for k, v in params.items():
        c, s = _quantize_symmetric(v, bits)
        codes[k] = c
        scales[k] = s
    return codes, scales


def dequantize_params(codes: dict[str, np.ndarray], scales: dict[str, float]) -> dict[str, np.ndarray]:
    return {k: (codes[k].astype(np.float32) * np.float32(scales[k])) for k in codes}


def measure_decoder_bytes(
    weights: dict[str, np.ndarray],
    latents: np.ndarray,
    cfg: ConvDecoderConfig,
) -> ConvDecoderByteAccount:
    """Honest brotli-q11 byte cost: quantized shared weights + per-pair latents + fp16 scales.

    The shared decoder weights are quantized + brotli'd as ONE blob; the per-pair latents as another.
    This mirrors the PR95 L19/L20 split (decoder weights dominate; latents are the small per-pair tail).
    """

    import brotli

    wcodes, wscales = quantize_params(weights, cfg.quant_bits)
    lcodes, lscale = _quantize_symmetric(latents, cfg.quant_bits)

    wchunks = [
        (c.astype(np.int8) if cfg.quant_bits <= 8 else c.astype(np.int16)).tobytes()
        for c in wcodes.values()
    ]
    weight_blob = brotli.compress(b"".join(wchunks), quality=11) if wchunks else b""
    lblob = brotli.compress(
        (lcodes.astype(np.int8) if cfg.quant_bits <= 8 else lcodes.astype(np.int16)).tobytes(),
        quality=11,
    )
    scale_vals = [*list(wscales.values()), lscale]
    scale_bytes = len(np.asarray(scale_vals, dtype=np.float16).tobytes())
    return ConvDecoderByteAccount(
        weight_bytes=len(weight_blob),
        latent_bytes=len(lblob),
        scale_bytes=scale_bytes,
        total_bytes=len(weight_blob) + len(lblob) + scale_bytes,
    )


def decoder_param_count(cfg: ConvDecoderConfig) -> int:
    """Total trainable param count (shared decoder weights + per-pair latents)."""

    n = 0
    # seed
    seed_out = cfg.seed_ch * cfg.seed_h * cfg.seed_w
    n += seed_out * cfg.latent_dim + seed_out
    in_ch = cfg.seed_ch
    for out_ch in cfg.stage_channels:
        n += (out_ch * 4) * in_ch * 9 + (out_ch * 4)  # 3x3 conv to out*4
        n += out_ch * in_ch * 1 + out_ch              # 1x1 skip
        in_ch = out_ch
    n += cfg.n_channels * in_ch * 9 + cfg.n_channels  # out 3x3
    n += cfg.num_pairs * cfg.latent_dim               # per-pair latents
    return int(n)


# ---------------------------------------------------------------------------
# Checkpoint I/O — portable .npz (NO /tmp).
# ---------------------------------------------------------------------------
def save_decoder_npz(
    path: Path,
    weights: dict[str, np.ndarray],
    latents: np.ndarray,
    cfg: ConvDecoderConfig,
) -> Path:
    path = Path(path)
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"w::{k}": np.asarray(v).astype(np.float32) for k, v in weights.items()}
    flat["latents"] = np.asarray(latents).astype(np.float32)
    for k, v in cfg.to_dict().items():
        flat[f"cfg::{k}"] = np.asarray(v)
    np.savez(path, **flat)
    return path


def load_decoder_npz(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, ConvDecoderConfig]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        weights = {k[len("w::"):]: z[k] for k in z.files if k.startswith("w::")}
        latents = z["latents"]
        cfg_kv = {k[len("cfg::"):]: z[k] for k in z.files if k.startswith("cfg::")}
    cfg = ConvDecoderConfig(
        num_pairs=int(cfg_kv["num_pairs"]),
        latent_dim=int(cfg_kv["latent_dim"]),
        seed_ch=int(cfg_kv["seed_ch"]),
        seed_h=int(cfg_kv["seed_h"]),
        seed_w=int(cfg_kv["seed_w"]),
        stage_channels=tuple(int(x) for x in cfg_kv["stage_channels"]),
        n_channels=int(cfg_kv["n_channels"]),
        quant_bits=int(cfg_kv["quant_bits"]),
    )
    return weights, latents, cfg


__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "ConvDecoderByteAccount",
    "ConvDecoderConfig",
    "decoder_frame",
    "decoder_param_count",
    "dequantize_params",
    "load_decoder_npz",
    "measure_decoder_bytes",
    "numpy_reference_forward",
    "quantize_params",
    "save_decoder_npz",
]
