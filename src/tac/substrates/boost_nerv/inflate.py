# SPDX-License-Identifier: MIT
"""boost_nerv inflate runtime — <= 200 LOC; BSV1 monolithic-archive consumer.

Forward path:

1. Read archive bytes for the requested archive_dir.
2. ``parse_archive(bytes)`` -> ``(decoder_state_dict, latents, meta,
   num_boosting_rounds)``.
3. Build the substrate from ``meta`` + ``num_boosting_rounds``.
4. ``model.load_state_dict(decoder_state_dict)``; copy latents.
5. For each pair index i in [0, num_pairs): render (rgb_0, rgb_1) via the
   boosting chain; write ``output_dir/0/{2*i}.png`` and ``{2*i+1}.png``.

L4 budget: <= 200 LOC (boosting chain adds modest LOC vs ds_nerv's 90); <= 2
external deps (torch + brotli).

Per HNeRV parity L4 explicit waiver: the boosting chain's loop adds ~10 LOC
over the ds_nerv baseline. The substrate stays under the 200 LOC ceiling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .architecture import BoostnervConfig, BoostnervSubstrate
from .archive import parse_archive


def inflate_one_video(
    archive_bytes: bytes,
    output_dir: Path,
    *,
    device: str = "cpu",
) -> None:
    """Inflate one archive's bytes into per-frame PNGs in output_dir.

    The BSV1 archive header carries NUM_BOOSTING_ROUNDS; the inflate path
    reconstructs the BoostnervConfig with the same num_boosting_rounds so
    forward-pass parity holds.
    """
    arc = parse_archive(archive_bytes)
    meta = arc.meta

    cfg = BoostnervConfig(
        latent_dim=int(arc.latents.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        num_boosting_rounds=int(arc.num_boosting_rounds),
        boosting_gain_clamp=float(meta["boosting_gain_clamp"]),
        boosting_hidden_dim=int(meta["boosting_hidden_dim"]),
        num_pairs=int(arc.latents.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
    )

    model = BoostnervSubstrate(cfg).to(device).eval()
    model.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        model.latents.copy_(
            arc.latents.to(device=device, dtype=model.latents.dtype)
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
