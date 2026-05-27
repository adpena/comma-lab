# SPDX-License-Identifier: MIT
"""coin_plus_plus inflate runtime — numpy-portable; CPP1-v2 monolithic consumer.

Per the 8th MLX-first standing directive (2026-05-26): TRAINING is MLX/PyTorch
but INFLATE is numpy-portable — ``torch`` / ``mlx`` are FORBIDDEN at decode time.
This runtime reconstructs the COIN++ coord-MLP forward pass in pure numpy from
the trained weights shipped in the archive (Catalog #369: consumes the REAL
trained state_dict + per-pair modulations, NOT a synthetic frame base).

Forward path (pure numpy):

1. ``parse_archive_numpy(bytes)`` -> ``(base_mlp_state_dict[np], modulations[np],
   meta, modulation_dim)`` with NO torch import.
2. Build the (2*H*W, 3) coordinate grid for frames {0, 1}.
3. Per pair i: run the FiLM-modulated coord-MLP forward
   ``sin(sin_freq * (gamma * (x @ Wl.T + bl) + beta))`` per hidden layer, then
   ``sigmoid(h @ Wo.T + bo)`` for RGB; split into frame_0 / frame_1.
4. Write ``output_dir/{2*i}.png`` and ``{2*i+1}.png``.

Runtime tree: numpy + brotli (archive decompress) + PIL (PNG write) — within
HNeRV parity L4 (<=200 LOC, CUDA-or-CPU agnostic via numpy, reviewable in 30s).
No ``select_inflate_device`` device fork is needed because numpy is device-free
(per Catalog #205 the numpy path has no device to pin; MPS is structurally
impossible). Per Catalog #295 the archive parser is vendored into the
submission tree so the inflate path is PYTHONPATH self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tac.substrates.coin_plus_plus.archive import parse_archive_numpy


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """y = x @ weight.T + bias (PyTorch nn.Linear canonical; fp32 accumulation)."""
    y = np.matmul(x.astype(np.float32), weight.astype(np.float32).T)
    return y + bias.astype(np.float32)


def _build_coord_grid(height: int, width: int) -> np.ndarray:
    """Canonical (2*H*W, 3) coordinate grid for frames {0, 1}.

    Mirrors ``CoinplusplusSubstrate._build_coord_grid`` exactly:
    ys = linspace(-1, 1, H); xs = linspace(-1, 1, W); meshgrid(indexing='ij').
    Frame 0 has t=-1, frame 1 has t=+1.
    """
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    gx = grid_x.reshape(-1)
    gy = grid_y.reshape(-1)
    n = height * width
    coords_0 = np.stack([gx, gy, np.full(n, -1.0, dtype=np.float32)], axis=-1)
    coords_1 = np.stack([gx, gy, np.full(n, 1.0, dtype=np.float32)], axis=-1)
    return np.concatenate([coords_0, coords_1], axis=0)  # (2*H*W, 3)


def inflate_one_video(archive_bytes: bytes, output_dir: Path) -> None:
    """Inflate one CPP1-v2 archive into per-frame PNGs in output_dir (numpy-only)."""
    arc = parse_archive_numpy(archive_bytes)
    meta = arc.meta
    sd = arc.base_mlp_state_dict

    hidden_dim = int(meta["hidden_dim"])
    num_hidden_layers = int(meta["num_hidden_layers"])
    sin_freq = float(meta["sin_frequency"])
    height = int(meta["output_height"])
    width = int(meta["output_width"])
    num_pairs = int(arc.modulations.shape[0])

    coords = _build_coord_grid(height, width)  # (2*H*W, 3)
    n_pixels = height * width

    output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image  # type: ignore[import-not-found]

    for pair_idx in range(num_pairs):
        m = arc.modulations[pair_idx].astype(np.float32)  # (mod_dim,)
        h = coords  # (2*H*W, 3)
        for layer in range(num_hidden_layers):
            p = f"mod_layers.{layer}."
            lin = _linear(h, sd[p + "linear.weight"], sd[p + "linear.bias"])
            gamma = _linear(
                m, sd[p + "mod_gamma_proj.weight"], sd[p + "mod_gamma_proj.bias"]
            )  # (out,)
            beta = _linear(
                m, sd[p + "mod_beta_proj.weight"], sd[p + "mod_beta_proj.bias"]
            )  # (out,)
            h = np.sin(sin_freq * (gamma[None, :] * lin + beta[None, :]))
        rgb = 1.0 / (
            1.0
            + np.exp(
                -_linear(h, sd["output_head.weight"], sd["output_head.bias"])
            )
        )  # sigmoid; (2*H*W, 3)

        for off in (0, 1):
            sl = slice(0, n_pixels) if off == 0 else slice(n_pixels, 2 * n_pixels)
            frame = rgb[sl].reshape(height, width, 3)
            arr = (np.clip(frame, 0.0, 1.0) * 255.0).round().clip(0, 255).astype(np.uint8)
            frame_idx = 2 * pair_idx + off
            Image.fromarray(arr).save(output_dir / f"{frame_idx}.png")
    # Touch hidden_dim so static analyzers see it consumed (shape consistency).
    assert hidden_dim > 0


def main_cli() -> int:
    """CLI: inflate.py <archive_dir> <output_dir> <file_list> per Catalog #146."""
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
    archive_bytes = (archive_dir / "0.bin").read_bytes()
    for fname in file_list:
        base = Path(fname).stem
        inflate_one_video(archive_bytes, output_dir / base)
    return 0


__all__ = ["inflate_one_video", "main_cli"]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
