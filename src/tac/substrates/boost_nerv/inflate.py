# SPDX-License-Identifier: MIT
"""boost_nerv inflate runtime — numpy-portable; BSV1-v2 boosting-chain consumer.

Per the 8th MLX-first standing directive (2026-05-26): TRAINING is MLX/PyTorch
but INFLATE is numpy-portable — ``torch`` / ``mlx`` are FORBIDDEN at decode time.
This runtime reconstructs the BoostNeRV decode forward (base decoder + iterative
residual boosting chain) in pure numpy from the REAL trained weights shipped in
the BSV1-v2 archive (Catalog #369: consumes the real base decoder + boosting
heads + latents, NOT a synthetic frame base).

Forward path (pure numpy, via the canonical numpy-portable inflate bridge):

1. ``parse_archive_numpy(bytes)`` -> torch-free ``BoostnervArchiveNumpy`` (decoder
   state_dict as fp32 ndarrays + int16-dequant latents + num_boosting_rounds).
2. Per pair i: base decode (latent_embed -> grid -> N×(depthwise + pointwise +
   sin + PixelShuffle(2)) -> bilinear resize -> 1×1 RGB heads -> sigmoid), then
   run the boosting chain: per round residual = tanh(conv2(relu(conv1([rgb;
   z_grid])))) clamped to [-gain, gain]; rgb = clamp(rgb + residual, 0, 1) (the
   BoostNeRV distinguishing mechanism — Catalog #220 operational consumption of
   the boosting heads).
3. Write ``output_dir/{2*i}.png`` and ``{2*i+1}.png``.

Runtime tree: numpy + brotli (archive decompress) + PIL (PNG write) — within
HNeRV parity L4. No ``select_inflate_device`` device fork because numpy is
device-free (Catalog #205; MPS structurally impossible). Per Catalog #295 the
archive parser + bridge are vendored into the submission tree so the inflate
path is PYTHONPATH self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tac.substrates._shared.numpy_portable_inflate import (
    bilinear_resize_nhwc,
    conv2d_numpy,
    linear,
    pixel_shuffle_2x_nhwc,
    relu,
    sigmoid,
    tanh,
    to_float32,
)
from tac.substrates.boost_nerv.archive import parse_archive_numpy


def _depthwise_conv3x3_nhwc(x: np.ndarray, w_nchw: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Depthwise 3×3 conv (groups=C) pad=1 in NHWC. ``w_nchw`` is (C, 1, 3, 3)."""
    x32 = to_float32(x)
    w32 = to_float32(w_nchw)
    n, h, w, c = x32.shape
    kh, kw = int(w32.shape[2]), int(w32.shape[3])
    pad_h, pad_w = kh // 2, kw // 2
    xp = np.pad(x32, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    out = np.zeros((n, h, w, c), dtype=np.float32)
    for di in range(kh):
        for dj in range(kw):
            tap = w32[:, 0, di, dj]
            out += xp[:, di : di + h, dj : dj + w, :] * tap[None, None, None, :]
    return out + to_float32(bias)[None, None, None, :]


def _pointwise_conv1x1_nhwc(x: np.ndarray, w_nchw: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Pointwise 1×1 conv in NHWC. ``w_nchw`` is (C_out, C_in, 1, 1)."""
    return linear(x, to_float32(w_nchw)[:, :, 0, 0], bias)


def _conv3x3_nhwc(x: np.ndarray, w_nchw: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """General cross-channel 3×3 conv pad=1 in NHWC. ``w_nchw`` is (C_out, C_in, 3, 3)."""
    w_nhwc = np.transpose(to_float32(w_nchw), (0, 2, 3, 1))
    return conv2d_numpy(x, w_nhwc, to_float32(bias), padding=1)


def _boost_round(rgb: np.ndarray, z: np.ndarray, head: dict[str, np.ndarray], gain: float) -> np.ndarray:
    """One boosting round: rgb_new = clamp(rgb + clamp(tanh(conv2(relu(conv1([rgb;z_grid])))), -gain, gain), 0, 1).

    ``rgb`` is NHWC (1, H, W, 3); ``z`` is (1, latent_dim).
    """
    z_emb = linear(z, head["z_proj.weight"], head["z_proj.bias"])  # (1, hidden)
    _, hh, ww, _ = rgb.shape
    z_grid = np.broadcast_to(z_emb[:, None, None, :], (1, hh, ww, z_emb.shape[1]))
    h = np.concatenate([rgb, z_grid], axis=-1)  # (1, H, W, 3+hidden)
    h = relu(_conv3x3_nhwc(h, head["conv1.weight"], head["conv1.bias"]))
    residual = tanh(_pointwise_conv1x1_nhwc(h, head["conv2.weight"], head["conv2.bias"]))
    residual = np.clip(residual, -gain, gain)
    return np.clip(rgb + residual, 0.0, 1.0)


def inflate_one_video(archive_bytes: bytes, output_dir: Path) -> None:
    """Inflate one BSV1-v2 archive into per-frame PNGs in output_dir (numpy-only)."""
    arc = parse_archive_numpy(archive_bytes)
    meta = arc.meta
    sd = arc.decoder_state_dict

    num_pairs = int(arc.latents.shape[0])
    embed_dim = int(meta["embed_dim"])
    init_gh = int(meta["initial_grid_h"])
    init_gw = int(meta["initial_grid_w"])
    num_blocks = int(meta["num_upsample_blocks"])
    sin_freq = float(meta["sin_frequency"])
    out_h = int(meta["output_height"])
    out_w = int(meta["output_width"])
    num_rounds = int(arc.num_boosting_rounds)
    gain = float(meta["boosting_gain_clamp"])

    output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image  # type: ignore[import-not-found]

    for pair_idx in range(num_pairs):
        z = arc.latents[pair_idx : pair_idx + 1]  # (1, latent_dim)
        flat = linear(z, sd["latent_embed.weight"], sd["latent_embed.bias"])
        grid_nchw = flat.reshape(1, embed_dim, init_gh, init_gw)
        h = np.transpose(grid_nchw, (0, 2, 3, 1))  # (1, gh0, gw0, embed)
        for b in range(num_blocks):
            p = f"blocks.{b}.dsc."
            h = _depthwise_conv3x3_nhwc(h, sd[p + "depthwise.weight"], sd[p + "depthwise.bias"])
            h = _pointwise_conv1x1_nhwc(h, sd[p + "pointwise.weight"], sd[p + "pointwise.bias"])
            h = np.sin(sin_freq * h)
            h = pixel_shuffle_2x_nhwc(h)
        if h.shape[1] != out_h or h.shape[2] != out_w:
            h = bilinear_resize_nhwc(h, target_h=out_h, target_w=out_w, align_corners=False)

        rgb_0 = sigmoid(_pointwise_conv1x1_nhwc(h, sd["head_rgb_0.weight"], sd["head_rgb_0.bias"]))
        rgb_1 = sigmoid(_pointwise_conv1x1_nhwc(h, sd["head_rgb_1.weight"], sd["head_rgb_1.bias"]))

        for r in range(num_rounds):
            hp = f"boosting_heads.{r}."
            head = {k[len(hp):]: v for k, v in sd.items() if k.startswith(hp)}
            rgb_0 = _boost_round(rgb_0, z, head, gain)
            rgb_1 = _boost_round(rgb_1, z, head, gain)

        for off, rgb in ((0, rgb_0), (1, rgb_1)):
            arr = (np.clip(rgb[0], 0.0, 1.0) * 255.0).round().clip(0, 255).astype(np.uint8)
            frame_idx = 2 * pair_idx + off
            Image.fromarray(arr).save(output_dir / f"{frame_idx}.png")


def main_cli() -> int:
    """CLI: inflate.py <archive_dir> <output_dir> <file_list> per Catalog #146."""
    if len(sys.argv) < 4:
        print("usage: inflate.py <archive_dir> <output_dir> <file_list>", file=sys.stderr)
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])

    file_list = file_list_path.read_text(encoding="utf-8").strip().splitlines()
    archive_bytes = (archive_dir / "0.bin").read_bytes()
    for fname in file_list:
        base = Path(fname).stem
        inflate_one_video(archive_bytes, output_dir / base)
    return 0


__all__ = ["inflate_one_video", "main_cli"]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
