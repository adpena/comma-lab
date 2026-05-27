# SPDX-License-Identifier: MIT
"""Z5 predictive-coding world-model inflate runtime — numpy-portable raw output.

Per the 8th MLX-first standing directive (2026-05-26): TRAINING is MLX/PyTorch
but INFLATE is numpy-portable — ``torch`` / ``mlx`` are FORBIDDEN at decode time.
This runtime reconstructs the Z5 predictive-coding decode forward (autoregressive
hierarchical predictor + NeRV decoder) in pure numpy from the REAL trained
weights shipped in the Z5PCWM1 archive (Catalog #369: consumes the real
predictor + decoder + latent_init + residuals + ego_motion, NOT a synthetic
frame base).

Forward path (pure numpy, via the canonical numpy-portable inflate bridge):

1. ``parse_archive_numpy(bytes)`` -> torch-free ``PredictiveCodingArchiveNumpy``
   (predictor + decoder state_dicts as fp32 ndarrays + int8-dequant latent_init,
   residuals, ego_motion).
2. Autoregressive rollout (Rao-Ballard predictive coding — Catalog #220
   operational consumption of the predictor): ``z_0 = latent_init``; for t in
   1..T: ``z_t = predictor(z_{t-1}, ego_motion[t]) + residuals[t]``. The
   predictor is an MLP: ``tanh(z_to_hidden(z) + ego_to_hidden(ego))`` ->
   GELU fused layers -> ``hidden_to_z``.
3. Per pair: NeRV decode (initial_proj -> grid -> N×(conv + PixelShuffle(2) +
   ReLU) -> final conv -> bilinear resize -> sigmoid -> RGB pair).
4. Write a single contest ``.raw`` file (1200 frames of 874×1164 RGB) via the
   bridge's ``write_rgb_pair_to_raw_numpy`` (Catalog #367 byte-count assert).

Runtime tree: numpy + brotli (archive decompress) — within HNeRV parity L4. No
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
    gelu,
    linear,
    pixel_shuffle_2x_nhwc,
    relu,
    sigmoid,
    tanh,
    to_float32,
    write_rgb_pair_to_raw_numpy,
)
from tac.substrates.z5_predictive_coding_world_model.archive import parse_archive_numpy


def _conv_nhwc(x: np.ndarray, w_nchw: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Run a torch Conv2d(pad=1) in NHWC. ``w_nchw`` is (C_out, C_in, kH, kW)."""
    w_nhwc = np.transpose(to_float32(w_nchw), (0, 2, 3, 1))
    return conv2d_numpy(x, w_nhwc, to_float32(bias), padding=1)


def _predict_next_z(
    z_prev: np.ndarray,
    ego: np.ndarray,
    pred: dict[str, np.ndarray],
    *,
    num_layers: int,
    identity: bool,
) -> np.ndarray:
    """Hierarchical MLP predictor: predict z_t from z_{t-1} + ego_motion.

    Mirrors ``HierarchicalPredictor.forward``: identity returns z_prev; else
    ``tanh(z_to_hidden(z) + ego_to_hidden(ego))`` -> (num_layers-1)×(Linear+GELU)
    -> ``hidden_to_z``. ``z_prev``/``ego`` are (1, dim).
    """
    if identity or "z_to_hidden.weight" not in pred:
        return z_prev
    z_h = linear(z_prev, pred["z_to_hidden.weight"], pred["z_to_hidden.bias"])
    ego_h = linear(ego, pred["ego_to_hidden.weight"], pred["ego_to_hidden.bias"])
    fused = tanh(z_h + ego_h)
    for k in range(num_layers - 1):
        lk = f"fused_layers.{2 * k}."  # Linear at index 2k; GELU at 2k+1
        fused = gelu(linear(fused, pred[lk + "weight"], pred[lk + "bias"]))
    return linear(fused, pred["hidden_to_z.weight"], pred["hidden_to_z.bias"])


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
    """NeRV decode of latent ``z`` (1, latent_dim) -> NHWC RGB pair (1,H,W,3)."""
    flat = linear(z, dec["initial_proj.weight"], dec["initial_proj.bias"])
    grid_nchw = flat.reshape(1, embed_dim, grid_h, grid_w)
    h = np.transpose(grid_nchw, (0, 2, 3, 1))  # (1, gh, gw, embed)
    for i in range(num_blocks):
        p = f"blocks.{3 * i}."
        h = _conv_nhwc(h, dec[p + "weight"], dec[p + "bias"])
        h = pixel_shuffle_2x_nhwc(h)
        h = relu(h)
    final_p = f"blocks.{3 * num_blocks}."
    out = _conv_nhwc(h, dec[final_p + "weight"], dec[final_p + "bias"])
    if out.shape[1] != out_h or out.shape[2] != out_w:
        out = bilinear_resize_nhwc(out, target_h=out_h, target_w=out_w, align_corners=False)
    out = sigmoid(out)
    return out[..., 0:3], out[..., 3:6]


def _rollout_latents(arc, *, num_pairs: int, num_layers: int, identity: bool) -> np.ndarray:
    """Autoregressive latent rollout: z_0 = latent_init; z_t = pred(z_{t-1}, ego_t) + r_t.

    Returns ``(num_pairs, latent_dim)`` fp32.
    """
    pred = arc.predictor_state_dict
    z = arc.latent_init.reshape(1, -1).astype(np.float32)  # (1, latent_dim)
    out = np.zeros((num_pairs, z.shape[1]), dtype=np.float32)
    out[0] = z[0]
    for t in range(1, num_pairs):
        ego_t = arc.ego_motion[t : t + 1]  # (1, ego_motion_dim)
        z_pred = _predict_next_z(z, ego_t, pred, num_layers=num_layers, identity=identity)
        z = z_pred + arc.residuals[t : t + 1]
        out[t] = z[0]
    return out


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one Z5PCWM1 archive's bytes into one contest ``.raw`` file (numpy-only).

    ``device`` is accepted for backward API compatibility with the prior
    torch-based runtime but is IGNORED: numpy is device-free per Catalog #205
    (MPS structurally impossible; CPU/CUDA agnostic by construction).
    """
    _ = device  # numpy is device-free; kept only for API compatibility
    arc = parse_archive_numpy(archive_bytes)
    meta = arc.meta
    dec = arc.decoder_state_dict

    embed_dim = int(meta["decoder_embed_dim"])
    grid_h = int(meta["decoder_initial_grid_h"])
    grid_w = int(meta["decoder_initial_grid_w"])
    num_blocks = int(meta["decoder_num_upsample_blocks"])
    out_h = int(meta.get("output_height", 384))
    out_w = int(meta.get("output_width", 512))
    num_pairs = int(arc.residuals.shape[0])
    pc_meta = meta.get("predictive_coding_world_model_meta", {})
    num_layers = int(pc_meta.get("predictor_num_layers", 2))
    identity = bool(pc_meta.get("identity_predictor", False))

    z_all = _rollout_latents(arc, num_pairs=num_pairs, num_layers=num_layers, identity=identity)

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with output_raw_path.open("wb") as fh:
        for i in range(num_pairs):
            rgb_0, rgb_1 = _decode_pair(
                z_all[i : i + 1], dec, embed_dim=embed_dim, grid_h=grid_h,
                grid_w=grid_w, num_blocks=num_blocks, out_h=out_h, out_w=out_w,
            )
            frames_written += write_rgb_pair_to_raw_numpy(
                fh, rgb_0, rgb_1, input_range="unit"
            )
    written_bytes = output_raw_path.stat().st_size
    if written_bytes != CONTEST_RAW_BYTES_PER_VIDEO and num_pairs == 600:
        raise RuntimeError(
            f"Z5PCWM1 inflate wrote {written_bytes} bytes != contest "
            f"{CONTEST_RAW_BYTES_PER_VIDEO} (Catalog #367 raw-byte fail-closed)"
        )
    return frames_written


def _raw_output_path_numpy(output_dir: Path, video_name: str) -> Path:
    """Torch-free contest-safe raw output path for one relative file-list entry."""
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
    """Read the single contest archive member, failing on missing/ambiguous input."""
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


__all__ = [
    "_read_single_member_archive_bytes",
    "inflate_one_video",
    "main_cli",
]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
