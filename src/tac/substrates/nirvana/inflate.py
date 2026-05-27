# SPDX-License-Identifier: MIT
"""nirvana inflate runtime — numpy-portable; NRV1-v2 patch-decode consumer.

Per the 8th MLX-first standing directive (2026-05-26): TRAINING is MLX/PyTorch
but INFLATE is numpy-portable — ``torch`` / ``mlx`` are FORBIDDEN at decode time.
This runtime reconstructs the NIRVANA patch-decode + stitch forward in pure
numpy from the REAL trained weights shipped in the NRV1-v2 archive (Catalog
#369: consumes the real shared per-patch decoder + patch embeddings + latents,
NOT a synthetic frame base).

Forward path (pure numpy, via the canonical numpy-portable inflate bridge):

1. ``parse_archive_numpy(bytes)`` -> torch-free ``NirvanaArchiveNumpy`` (decoder
   state_dict as fp32 ndarrays + int16-dequant latents + patch grid).
2. Per pair i: decode every patch from ``[z_i ; patch_embedding[p]]`` through
   the shared decoder (combined_embed -> N×(depthwise + pointwise + sin +
   PixelShuffle(2)) -> 1×1 RGB heads -> sigmoid), then stitch the per-patch RGBs
   into the full ``(H, W)`` frame (the NIRVANA distinguishing mechanism —
   Catalog #220 operational consumption of per-patch embeddings).
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
    linear,
    pixel_shuffle_2x_nhwc,
    sigmoid,
    to_float32,
)
from tac.substrates.nirvana.archive import parse_archive_numpy


def _depthwise_conv3x3_nhwc(
    x: np.ndarray, w_nchw: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """Depthwise 3×3 conv (groups=C) pad=1 in NHWC. ``w_nchw`` is (C, 1, 3, 3)."""
    x32 = to_float32(x)
    w32 = to_float32(w_nchw)  # (C, 1, kH, kW)
    n, h, w, c = x32.shape
    kh, kw = int(w32.shape[2]), int(w32.shape[3])
    pad_h, pad_w = kh // 2, kw // 2
    xp = np.pad(x32, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    out = np.zeros((n, h, w, c), dtype=np.float32)
    for di in range(kh):
        for dj in range(kw):
            # per-channel kernel weight at tap (di, dj): shape (C,)
            tap = w32[:, 0, di, dj]
            out += xp[:, di : di + h, dj : dj + w, :] * tap[None, None, None, :]
    return out + to_float32(bias)[None, None, None, :]


def _pointwise_conv1x1_nhwc(
    x: np.ndarray, w_nchw: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """Pointwise 1×1 conv in NHWC: y = x @ W[:, :, 0, 0].T + b. ``w_nchw`` (C_out, C_in, 1, 1)."""
    w_lin = to_float32(w_nchw)[:, :, 0, 0]  # (C_out, C_in)
    return linear(x, w_lin, bias)


def _decode_patch_frame(
    h: np.ndarray, head_w: np.ndarray, head_b: np.ndarray
) -> np.ndarray:
    """1×1 RGB head + sigmoid -> NHWC (BP, h, w, 3)."""
    return sigmoid(_pointwise_conv1x1_nhwc(h, head_w, head_b))


def inflate_one_video(archive_bytes: bytes, output_dir: Path) -> None:
    """Inflate one NRV1-v2 archive into per-frame PNGs in output_dir (numpy-only)."""
    arc = parse_archive_numpy(archive_bytes)
    meta = arc.meta
    sd = arc.decoder_state_dict

    latent_dim = int(arc.latents.shape[1])
    num_pairs = int(arc.latents.shape[0])
    gh = int(arc.patch_grid_h)
    gw = int(arc.patch_grid_w)
    num_patches = gh * gw
    embed_dim = int(meta["embed_dim"])
    init_gh = int(meta["initial_patch_grid_h"])
    init_gw = int(meta["initial_patch_grid_w"])
    num_blocks = int(meta["num_upsample_blocks"])
    sin_freq = float(meta["sin_frequency"])
    out_h = int(meta["output_height"])
    out_w = int(meta["output_width"])
    patch_h = out_h // gh
    patch_w = out_w // gw

    patch_emb = sd["patch_embeddings"]  # (num_patches, patch_embed_dim)

    output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image  # type: ignore[import-not-found]

    for pair_idx in range(num_pairs):
        z = arc.latents[pair_idx : pair_idx + 1]  # (1, latent_dim)
        # combined input per patch: [z ; patch_embedding[p]] -> (num_patches, L+E)
        z_exp = np.broadcast_to(z, (num_patches, latent_dim))
        combined = np.concatenate([z_exp, patch_emb], axis=-1)  # (P, L+E)
        flat = linear(combined, sd["combined_embed.weight"], sd["combined_embed.bias"])
        grid_nchw = flat.reshape(num_patches, embed_dim, init_gh, init_gw)
        h = np.transpose(grid_nchw, (0, 2, 3, 1))  # (P, gh0, gw0, embed)
        for b in range(num_blocks):
            p = f"blocks.{b}.dsc."
            h = _depthwise_conv3x3_nhwc(h, sd[p + "depthwise.weight"], sd[p + "depthwise.bias"])
            h = _pointwise_conv1x1_nhwc(h, sd[p + "pointwise.weight"], sd[p + "pointwise.bias"])
            h = np.sin(sin_freq * h)
            h = pixel_shuffle_2x_nhwc(h)  # (P, 2h, 2w, out_ch)

        rgb0_p = _decode_patch_frame(h, sd["head_rgb_0.weight"], sd["head_rgb_0.bias"])
        rgb1_p = _decode_patch_frame(h, sd["head_rgb_1.weight"], sd["head_rgb_1.bias"])

        for off, rgb_p in ((0, rgb0_p), (1, rgb1_p)):
            full = _stitch_patches(rgb_p, gh, gw, patch_h, patch_w)  # (out_h, out_w, 3)
            arr = (np.clip(full, 0.0, 1.0) * 255.0).round().clip(0, 255).astype(np.uint8)
            frame_idx = 2 * pair_idx + off
            Image.fromarray(arr).save(output_dir / f"{frame_idx}.png")


def _stitch_patches(
    patch_rgb: np.ndarray, gh: int, gw: int, patch_h: int, patch_w: int
) -> np.ndarray:
    """Stitch (num_patches, h_native, w_native, 3) -> (out_h, out_w, 3) NHWC.

    Mirrors ``NirvanaSubstrate._stitch_patches``: resize each patch to
    (patch_h, patch_w), then place patch p at row-major (gh, gw) grid position.
    """
    if patch_rgb.shape[1] != patch_h or patch_rgb.shape[2] != patch_w:
        patch_rgb = bilinear_resize_nhwc(
            patch_rgb, target_h=patch_h, target_w=patch_w, align_corners=False
        )
    # (P, ph, pw, 3) -> (gh, gw, ph, pw, 3) -> (gh, ph, gw, pw, 3) -> (out_h, out_w, 3)
    grid = patch_rgb.reshape(gh, gw, patch_h, patch_w, 3)
    grid = np.transpose(grid, (0, 2, 1, 3, 4))
    return grid.reshape(gh * patch_h, gw * patch_w, 3)


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
