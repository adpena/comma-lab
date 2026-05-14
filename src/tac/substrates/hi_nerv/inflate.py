# SPDX-License-Identifier: MIT
"""hi_nerv inflate runtime — <= 100 LOC; HIV1 monolithic-archive consumer.

Forward path:

1. Read archive bytes.
2. ``parse_archive(bytes)`` -> ``(decoder_state_dict, latents_c/m/f, meta)``.
3. Build the substrate from ``meta`` (deterministic).
4. ``model.load_state_dict(decoder_state_dict)``; copy 3-scale latents.
5. For each pair index, render (rgb_0, rgb_1); write per-frame PNGs.

L4 budget: <= 100 LOC, <= 2 external deps (torch + brotli).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .architecture import HinervConfig, HinervSubstrate
from .archive import parse_archive


def inflate_one_video(
    archive_bytes: bytes,
    output_dir: Path,
    *,
    device: str = "cpu",
) -> None:
    """Inflate one archive's bytes into per-frame PNGs in output_dir."""
    arc = parse_archive(archive_bytes)
    meta = arc.meta

    cfg = HinervConfig(
        latent_dim_coarse=int(arc.latents_coarse.shape[1]),
        latent_dim_mid=int(arc.latents_mid.shape[1]),
        latent_dim_fine=int(arc.latents_fine.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        mid_injection_block_index=int(meta["mid_injection_block_index"]),
        fine_injection_block_index=int(meta["fine_injection_block_index"]),
        num_pairs=int(arc.latents_coarse.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
    )

    model = HinervSubstrate(cfg).to(device).eval()
    model.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        model.latents_coarse.copy_(
            arc.latents_coarse.to(device=device, dtype=model.latents_coarse.dtype)
        )
        model.latents_mid.copy_(
            arc.latents_mid.to(device=device, dtype=model.latents_mid.dtype)
        )
        model.latents_fine.copy_(
            arc.latents_fine.to(device=device, dtype=model.latents_fine.dtype)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image  # type: ignore[import-not-found]

    with torch.no_grad():
        for pair_idx in range(cfg.num_pairs):
            idx_tensor = torch.tensor([pair_idx], device=device, dtype=torch.long)
            rgb_0, rgb_1 = model(idx_tensor)
            for off, rgb in ((0, rgb_0), (1, rgb_1)):
                frame_idx = 2 * pair_idx + off
                arr = (rgb[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0)
                arr = arr.round().clip(0, 255).astype("uint8")
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
    for fname in file_list:
        base = Path(fname).stem
        archive_bytes = (archive_dir / "0.bin").read_bytes()
        inflate_one_video(archive_bytes, output_dir / base, device="cpu")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
