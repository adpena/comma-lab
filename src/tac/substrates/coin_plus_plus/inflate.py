# SPDX-License-Identifier: MIT
"""coin_plus_plus inflate runtime — <= 200 LOC; CPP1 monolithic-archive consumer.

Forward path:

1. Read archive bytes for the requested archive_dir.
2. ``parse_archive(bytes)`` -> ``(base_mlp_state_dict, modulations, meta,
   modulation_dim)``.
3. Build the substrate from ``meta`` + modulation_dim.
4. ``model.load_state_dict(base_mlp_state_dict)``; copy modulations.
5. For each pair index i in [0, num_pairs): render (rgb_0, rgb_1) via the
   coord-MLP forward path; write ``output_dir/0/{2*i}.png`` and
   ``{2*i+1}.png``.

L4 budget: <= 200 LOC (coord-MLP forward is light); <= 2 external deps
(torch + brotli).

Performance note: the coord-MLP forward pass is O(H*W) per pixel; this is
inherent to coordinate-based INRs. Per-pair cost is higher than spatial-
grid PixelShuffle decoders but the per-pair LATENT (modulation) is much
smaller.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .architecture import CoinplusplusConfig, CoinplusplusSubstrate
from .archive import parse_archive


def inflate_one_video(
    archive_bytes: bytes,
    output_dir: Path,
    *,
    device: str = "cpu",
) -> None:
    """Inflate one archive's bytes into per-frame PNGs in output_dir.

    The CPP1 archive header carries MODULATION_DIM; the inflate path
    reconstructs the CoinplusplusConfig with the same modulation_dim so
    forward-pass parity holds.
    """
    arc = parse_archive(archive_bytes)
    meta = arc.meta

    cfg = CoinplusplusConfig(
        modulation_dim=int(arc.modulation_dim),
        hidden_dim=int(meta["hidden_dim"]),
        num_hidden_layers=int(meta["num_hidden_layers"]),
        sin_frequency=float(meta["sin_frequency"]),
        coord_input_dim=int(meta["coord_input_dim"]),
        output_channels=int(meta["output_channels"]),
        num_pairs=int(arc.modulations.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
    )

    model = CoinplusplusSubstrate(cfg).to(device).eval()
    model.load_state_dict(arc.base_mlp_state_dict, strict=False)
    with torch.no_grad():
        model.modulations.copy_(
            arc.modulations.to(device=device, dtype=model.modulations.dtype)
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
