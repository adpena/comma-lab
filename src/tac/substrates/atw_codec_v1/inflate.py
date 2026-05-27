# SPDX-License-Identifier: MIT
"""ATW codec V1 inflate runtime — numpy-portable; ATW1 raw-output consumer.

Per the 8th MLX-first standing directive (2026-05-26): TRAINING is MLX/PyTorch
but INFLATE is numpy-portable — ``torch`` / ``mlx`` are FORBIDDEN at decode time.
This runtime reconstructs the ATW codec decode forward in pure numpy from the
REAL trained weights shipped in the ATW1 archive (Catalog #369: consumes the
real decoder + WZ side-info head + scorer_class_prior_table, NOT a synthetic
frame base).

Forward path (pure numpy, via the canonical numpy-portable inflate bridge):

1. ``parse_archive_numpy(bytes)`` -> torch-free ``ATWCodecArchiveNumpy`` with
   decoder + WZ-head state_dicts as fp32 ndarrays, z_residual, class-prior table.
2. Per pair i: reconstruct ``z = z_residual[i] + wz_head(class_prior_table[i])``
   (the Wyner-Ziv side-info mechanism — Catalog #220 operational consumption).
3. NeRV-style decode (NHWC): ``initial_proj`` -> reshape grid -> N×(conv +
   PixelShuffle(2) + ReLU) -> final conv -> bilinear resize -> sigmoid ->
   split RGB pair (channels 0:3 = frame_0, 3:6 = frame_1).
4. Write a single contest ``.raw`` file (1200 frames of 874×1164 RGB) via the
   bridge's ``write_rgb_pair_to_raw_numpy`` (Catalog #367 byte-count assert).

Runtime tree: numpy + brotli (archive decompress) — within HNeRV parity L4
(≤200 LOC, CUDA-or-CPU agnostic via numpy, reviewable in 30s). No
``select_inflate_device`` device fork because numpy is device-free (Catalog
#205; MPS structurally impossible). Per Catalog #295 the archive parser + bridge
are vendored into the submission tree so the inflate path is PYTHONPATH
self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tac.substrates._shared.numpy_portable_inflate import (
    CONTEST_RAW_BYTES_PER_VIDEO,
    bilinear_resize_nhwc,
    conv2d_numpy,
    linear,
    pixel_shuffle_2x_nhwc,
    relu,
    sigmoid,
    write_rgb_pair_to_raw_numpy,
)
from tac.substrates.atw_codec_v1.archive import parse_archive_numpy


def _conv_nhwc(x: np.ndarray, w_nchw: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Run a torch Conv2d(pad=1) in NHWC. ``w_nchw`` is (C_out, C_in, kH, kW)."""
    w_nhwc = np.transpose(np.asarray(w_nchw, dtype=np.float32), (0, 2, 3, 1))
    return conv2d_numpy(x, w_nhwc, np.asarray(bias, dtype=np.float32), padding=1)


def _wz_predict(
    class_prior: np.ndarray, sd: dict[str, np.ndarray]
) -> np.ndarray:
    """WZ side-info head: fc2(relu(fc1(class_prior))). Returns zeros if disabled."""
    if "fc1.weight" not in sd or "fc2.weight" not in sd:
        return np.zeros((class_prior.shape[0], 0), dtype=np.float32)
    h = relu(linear(class_prior, sd["fc1.weight"], sd["fc1.bias"]))
    return linear(h, sd["fc2.weight"], sd["fc2.bias"])


def _decode_pair(
    z: np.ndarray,
    dec: dict[str, np.ndarray],
    *,
    embed_dim: int,
    grid_h: int,
    grid_w: int,
    num_blocks: int,
    out_h: int,
    out_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """NeRV-style decode of latent ``z`` (1, latent_dim) -> NHWC RGB pair (1,H,W,3)."""
    flat = linear(z, dec["initial_proj.weight"], dec["initial_proj.bias"])  # (1, embed*gh*gw)
    # torch reshapes to NCHW (1, embed, gh, gw); convert to NHWC for the bridge.
    grid_nchw = flat.reshape(1, embed_dim, grid_h, grid_w)
    h = np.transpose(grid_nchw, (0, 2, 3, 1))  # (1, gh, gw, embed)
    for i in range(num_blocks):
        p = f"blocks.{3 * i}."
        h = _conv_nhwc(h, dec[p + "weight"], dec[p + "bias"])  # (1, gh, gw, 4*out_ch)
        h = pixel_shuffle_2x_nhwc(h)  # (1, 2*gh, 2*gw, out_ch)
        h = relu(h)
    final_p = f"blocks.{3 * num_blocks}."
    out = _conv_nhwc(h, dec[final_p + "weight"], dec[final_p + "bias"])  # (1, H', W', 6)
    if out.shape[1] != out_h or out.shape[2] != out_w:
        out = bilinear_resize_nhwc(out, target_h=out_h, target_w=out_w, align_corners=False)
    out = sigmoid(out)  # (1, out_h, out_w, 6)
    return out[..., 0:3], out[..., 3:6]


def inflate_one_video(archive_bytes: bytes, output_raw_path: Path) -> int:
    """Inflate one ATW1 archive's bytes into one contest ``.raw`` file (numpy-only)."""
    arc = parse_archive_numpy(archive_bytes)
    meta = arc.meta
    dec = arc.decoder_state_dict
    wz = arc.wz_side_info_head_state_dict

    embed_dim = int(meta["decoder_embed_dim"])
    grid_h = int(meta["decoder_initial_grid_h"])
    grid_w = int(meta["decoder_initial_grid_w"])
    num_blocks = int(meta["decoder_num_upsample_blocks"])
    out_h = int(meta.get("output_height", 384))
    out_w = int(meta.get("output_width", 512))
    num_pairs = int(arc.latent_residual.shape[0])

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with output_raw_path.open("wb") as fh:
        for i in range(num_pairs):
            z_residual = arc.latent_residual[i : i + 1]  # (1, latent_dim)
            class_prior = arc.scorer_class_prior_table[i : i + 1]  # (1, prior_dim)
            z_pred = _wz_predict(class_prior, wz)
            z = z_residual + z_pred if z_pred.shape[1] else z_residual
            rgb_0, rgb_1 = _decode_pair(
                z, dec, embed_dim=embed_dim, grid_h=grid_h, grid_w=grid_w,
                num_blocks=num_blocks, out_h=out_h, out_w=out_w,
            )
            frames_written += write_rgb_pair_to_raw_numpy(
                fh, rgb_0, rgb_1, input_range="unit"
            )
    written_bytes = output_raw_path.stat().st_size
    if written_bytes != CONTEST_RAW_BYTES_PER_VIDEO and num_pairs == 600:
        raise RuntimeError(
            f"ATW1 inflate wrote {written_bytes} bytes != contest "
            f"{CONTEST_RAW_BYTES_PER_VIDEO} (Catalog #367 raw-byte fail-closed)"
        )
    return frames_written


def _raw_output_path_numpy(output_dir: Path, video_name: str) -> Path:
    """Torch-free copy of the contest-safe raw output path helper."""
    raw = str(video_name).replace("\\", "/").strip()
    rel = Path(raw)
    if (
        not raw
        or "//" in raw
        or rel.is_absolute()
        or any(part in {"", ".."} for part in rel.parts)
    ):
        raise ValueError(f"unsafe file_list video name for raw output: {video_name!r}")
    root = output_dir.resolve(strict=False)
    target = (output_dir / rel.with_suffix(".raw")).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"file_list video name escapes output directory: {video_name!r}"
        ) from exc
    return target


def _read_single_member_archive_bytes(archive_dir: Path) -> bytes:
    """Read the single contest archive member, failing on ambiguity."""
    zero_bin = archive_dir / "0.bin"
    x_member = archive_dir / "x"
    present = [path for path in (zero_bin, x_member) if path.is_file()]
    if len(present) != 1:
        if not present:
            raise FileNotFoundError(
                f"expected exactly one archive member at {zero_bin} or {x_member}"
            )
        raise ValueError(
            f"ambiguous archive members present: {zero_bin} and {x_member}"
        )
    return present[0].read_bytes()


def main_cli() -> int:
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>`` (Catalog #146)."""
    if len(sys.argv) < 4:
        print(
            "usage: inflate.py <archive_dir> <output_dir> <file_list>",
            file=sys.stderr,
        )
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])

    file_list = file_list_path.read_text(encoding="utf-8").strip().splitlines()
    archive_bytes = _read_single_member_archive_bytes(archive_dir)
    for fname in file_list:
        name = fname.strip()
        if not name:
            continue
        inflate_one_video(archive_bytes, _raw_output_path_numpy(output_dir, name))
    return 0


__all__ = ["_read_single_member_archive_bytes", "inflate_one_video", "main_cli"]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
